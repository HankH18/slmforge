#!/usr/bin/env python3
"""
Forbidden-Constraint Holder — eval harness.

Behavior Spec (the graded spec):
  Given a task prompt and a set of 2-4 hard constraints that always includes at
  least one negative constraint, the model completes the task while violating
  zero constraints. Pass = every constraint passes its programmatic checker AND
  the response genuinely addresses the task (judge task-quality >= 3).

Constraint pass/fail is 100% programmatic (constraints/checkers.py). The LLM
judge scores task-quality ONLY and can never override a checker -- it exists to
block the degenerate strategy of satisfying every constraint with garbage text.

This file is CLI plumbing plus the strategy prompts. It owns no scoring logic:
  scoring   -> harness.score_scenario / harness.aggregate
  checking  -> constraints.checkers.check   (imported, never reimplemented)
  prompting -> scenarios.render.render_user_message  (the ONE definition, shared
               with training/modal_app.py so the training condition equals the
               zero_shot eval condition)

Usage:
  eval.py --model hf:Qwen/Qwen3-1.7B --eval-set data/scenarios/corpus.jsonl
  eval.py --model hf:you/tuned@<commit> --compare-with hf:Qwen/Qwen3-1.7B@<rev> \
          --eval-set data/scenarios/corpus.jsonl --out runs/mvp
  eval.py --model openai:<id> --strategy few_shot \
          --exemplars data/scenarios/fewshot_exemplars.jsonl \
          --eval-set data/scenarios/corpus.jsonl

Model specs:
  hf:<repo>[@revision]   local inference via transformers (pin the revision)
  anthropic:<model>      env ANTHROPIC_API_KEY
  openai:<model>         env OPENAI_API_KEY
  google:<model>         env GOOGLE_API_KEY
  mock:<behavior>        compliant | violator | empty | offtask  (network-free)

NOTE: this reads os.environ directly and loads no dotenv. Export first:
  set -a && . ./.env && set +a
"""
import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import harness
import judge as judge_mod
from harness import (STRATEGIES, HarnessError, aggregate, build_constraint_lookup,
                     score_scenario, validate_eval_set)
from scenarios.render import render_user_message

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_EVAL_SET = "data/scenarios/corpus.jsonl"
DEFAULT_CONSTRAINTS = "data/scenarios/constraint_instances.jsonl"

# ---------------------------------------------------------------- strategies
#
# zero_shot carries NO system message. That is not an oversight: the pinned
# training example schema is [user, assistant] with no system turn, and the
# training condition must equal the zero_shot eval condition exactly or the
# tuned model is measured under a prompt it never saw. Anything added here has
# to be added to training/modal_app.py in the same commit.
#
# structured DOES add a system message. That is fine -- it is an ablation
# strategy, not the training condition. Only zero_shot is load-bearing.

STRUCTURED_SYSTEM = """Before you answer, silently run this verification protocol. Never include the protocol, your reasoning, or any commentary in your output.

1. Restate each requirement to yourself.
2. Draft a response that completes the task.
3. Re-read the draft against each requirement in turn. Where a requirement concerns letters or punctuation, check character by character rather than by impression.
4. Rewrite anything that violates a requirement, then re-check every requirement again -- a rewrite can break a requirement that previously held.

Output only the finished response."""


def build_prompt(strategy: str, scenario: Mapping[str, Any],
                 constraint_lookup: Mapping[str, Mapping[str, Any]],
                 exemplars: Optional[Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
    """-> {"system": str|None, "turns": [{"role","content"}, ...]}"""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}, expected one of {STRATEGIES}")
    user = render_user_message(scenario, constraint_lookup)

    turns: List[Dict[str, str]] = []
    if strategy == "few_shot":
        if not exemplars:
            raise ValueError("few_shot strategy requires --exemplars")
        for ex in exemplars:
            turns.append({"role": "user", "content": ex["rendered_prompt"]})
            turns.append({"role": "assistant", "content": ex["target_output"]})
    turns.append({"role": "user", "content": user})

    system = STRUCTURED_SYSTEM if strategy == "structured" else None
    return {"system": system, "turns": turns}


# ---------------------------------------------------------------- generators
# Transient provider conditions worth waiting out. A capacity 503 on a frontier
# preview model can persist for minutes, and one raise aborts the whole cell --
# 147 scenarios of completed work thrown away with it.
_TRANSIENT = ("503", "UNAVAILABLE", "high demand", "429", "RESOURCE_EXHAUSTED",
              "rate limit", "overloaded", "502", "504", "timeout", "Timeout")


def _is_transient(exc: Exception) -> bool:
    s = f"{type(exc).__name__}: {exc}"
    return any(marker in s for marker in _TRANSIENT)


def _retrying(fn, tries=3, transient_tries=8, max_sleep=60.0):
    """Retry, waiting much longer on transient provider conditions.

    A malformed request should fail fast (3 tries) so a bug surfaces immediately.
    A 503/429 is the provider asking us to come back, so back off properly rather
    than burning the attempt budget in four seconds and losing the cell.
    """
    def wrapped(*a, **kw):
        attempt = 0
        while True:
            try:
                return fn(*a, **kw)
            except Exception as exc:  # noqa: BLE001
                attempt += 1
                budget = transient_tries if _is_transient(exc) else tries
                if attempt >= budget:
                    raise
                # jitter so parallel workers do not retry in lockstep and
                # re-trigger the same capacity spike together
                delay = min(max_sleep, (2 ** attempt) + random.uniform(0, 1.5))
                time.sleep(delay)
    return wrapped


class MockGenerator:
    """Network-free doubles that exercise every branch of the rubric."""

    def __init__(self, behavior: str):
        if behavior not in ("compliant", "violator", "empty", "offtask"):
            raise ValueError(f"unknown mock behavior {behavior!r}")
        self.behavior = behavior
        self.thread_safe = True

    def __call__(self, scenario, prompt, constraint_lookup=None):
        if self.behavior == "empty":
            return ""
        if self.behavior == "violator":
            # Violates lexical, format and punctuation constraints at once.
            return ('Coffee, espresso and lattes are the point here; in conclusion:\n'
                    '- "the best thing" is obvious\n- everyone agrees')
        # compliant / offtask both need a genuinely constraint-satisfying string,
        # built by the same witness machinery the corpus generator proved with.
        from scenarios.reachability import build_witness
        cons = [constraint_lookup[c] for c in scenario["constraints"]]
        witness = build_witness(cons, sentences=4)
        if witness is None:
            raise RuntimeError(
                f"no witness for {scenario['scenario_id']}: the scenario may be "
                f"unsatisfiable, which would fake difficulty. Investigate rather "
                f"than scoring it.")
        return witness


class HFGenerator:
    """Local inference.

    Memory hygiene is not optional here. Measured on this machine, a 1.7B model
    whose weights are 3.4GB in bfloat16 peaked at 19.7GB across a handful of
    generations: MPS does not return the per-generation allocations on its own,
    so they accumulate until the OS kills the process. Two ablation rows died
    that way before this was found. Freeing after each generation keeps a
    147-scenario row flat.
    """

    def __init__(self, repo: str, revision: Optional[str]):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(repo, revision=revision)
        # Explicit dtype and a single device rather than "auto"/device_map: auto
        # can hold a CPU copy alongside the accelerator copy, and the dtype is
        # read from the model config anyway (bfloat16 for Qwen3).
        dtype = torch.bfloat16 if torch.backends.mps.is_available() else "auto"
        import inspect
        kw = {"revision": revision, "low_cpu_mem_usage": True}
        # transformers renamed torch_dtype -> dtype; support both without pinning
        # a version, since the training app and the eval must load identically.
        sig = inspect.signature(AutoModelForCausalLM.from_pretrained)
        kw["dtype" if "dtype" in sig.parameters else "torch_dtype"] = dtype
        try:
            self.model = AutoModelForCausalLM.from_pretrained(repo, **kw)
        except TypeError:
            kw.pop("dtype", None); kw.pop("torch_dtype", None)
            self.model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=dtype, **kw)
        self.device = ("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.thread_safe = False  # forces workers=1

    def _release(self):
        if self.device == "mps" and hasattr(self.torch, "mps"):
            self.torch.mps.empty_cache()
        elif self.device == "cuda":
            self.torch.cuda.empty_cache()

    def __call__(self, scenario, prompt, constraint_lookup=None):
        msgs = ([{"role": "system", "content": prompt["system"]}] if prompt["system"] else [])
        msgs += prompt["turns"]
        try:
            # enable_thinking=False must match training/modal_app.py's chat-template
            # call, or the training and eval conditions diverge silently.
            text = self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(text, return_tensors="pt").to(self.device)
        try:
            with self.torch.inference_mode():
                out = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
            decoded = self.tok.decode(out[0][inputs["input_ids"].shape[1]:],
                                      skip_special_tokens=True)
        finally:
            del inputs
            self._release()
        return decoded


class AnthropicGenerator:
    def __init__(self, model: str):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.thread_safe = True

    @_retrying
    def __call__(self, scenario, prompt, constraint_lookup=None):
        kw = {"system": prompt["system"]} if prompt["system"] else {}
        try:
            resp = self.client.messages.create(
                model=self.model, max_tokens=8192, temperature=0,
                messages=prompt["turns"], **kw)
        except Exception as exc:  # noqa: BLE001
            # Newer Claude generations reject `temperature` outright with a 400.
            # Re-raise anything else: swallowing a real error here would turn a
            # broken request into a silently sampled one. Same asymmetry already
            # disclosed for gpt-5.6-sol -- those cells sample at the model default
            # and are therefore not bit-reproducible.
            if not judge_mod._is_temperature_400(exc):
                raise
            resp = self.client.messages.create(
                model=self.model, max_tokens=8192, messages=prompt["turns"], **kw)
        text = "".join(b.text for b in resp.content
                       if getattr(b, "type", "") == "text")
        if not text.strip():
            # Extended thinking shares the max_tokens budget: a thinking block can
            # consume all of it and leave no text. That must NOT be returned as an
            # empty completion -- ruling A7 makes empty text pass every checker
            # vacuously, so a truncated generation would be scored as a perfectly
            # compliant response. Raise so it retries or surfaces.
            raise RuntimeError(
                f"{self.model} returned no text (stop_reason={resp.stop_reason}, "
                f"blocks={[getattr(b, 'type', '?') for b in resp.content]}). "
                f"An empty completion would score as vacuously compliant.")
        return text


class OpenAIGenerator:
    def __init__(self, model: str):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model
        self.thread_safe = True

    @_retrying
    def __call__(self, scenario, prompt, constraint_lookup=None):
        msgs = ([{"role": "system", "content": prompt["system"]}] if prompt["system"] else [])
        msgs += prompt["turns"]
        try:
            resp = self.client.chat.completions.create(
                model=self.model, messages=msgs, temperature=0)
        except Exception:
            # gpt-5.6-sol and other reasoning-series models reject temperature=0.
            # Retrying without it samples at the model default, so those cells are
            # NOT bit-reproducible. Disclosed in the ablation writeup.
            resp = self.client.chat.completions.create(model=self.model, messages=msgs)
        return resp.choices[0].message.content or ""


class GoogleGenerator:
    def __init__(self, model: str):
        from google import genai
        from google.genai import types
        self.types = types
        self.client = genai.Client()
        self.model = model
        self.thread_safe = True

    @_retrying
    def __call__(self, scenario, prompt, constraint_lookup=None):
        t = self.types
        contents = [t.Content(role="user" if turn["role"] == "user" else "model",
                              parts=[t.Part.from_text(text=turn["content"])])
                    for turn in prompt["turns"]]
        cfg = {"temperature": 0}
        if prompt["system"]:
            cfg["system_instruction"] = prompt["system"]
        resp = self.client.models.generate_content(
            model=self.model, contents=contents,
            config=t.GenerateContentConfig(**cfg))
        return resp.text or ""


def make_generator(spec: str):
    provider, _, rest = spec.partition(":")
    if not rest:
        raise ValueError(f"model spec must be provider:model, got {spec!r}")
    if provider == "mock":
        return MockGenerator(rest)
    if provider == "hf":
        repo, _, revision = rest.partition("@")
        return HFGenerator(repo, revision or None)
    if provider == "anthropic":
        return AnthropicGenerator(rest)
    if provider == "openai":
        return OpenAIGenerator(rest)
    if provider == "google":
        return GoogleGenerator(rest)
    raise ValueError(f"unknown provider {provider!r}")


# ---------------------------------------------------------------- runner
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Refusing to silently evaluate a smaller set.")
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_eval(model_spec: str, scenarios: Sequence[Mapping[str, Any]],
             constraint_lookup: Mapping[str, Mapping[str, Any]], strategy: str,
             exemplars, judge_fn, workers: int = 4, generator=None,
             on_judge_error: str = "record"):
    gen = generator or make_generator(model_spec)
    if not getattr(gen, "thread_safe", True):
        workers = 1

    def one(scenario):
        prompt = build_prompt(strategy, scenario, constraint_lookup, exemplars)
        try:
            output = gen(scenario, prompt, constraint_lookup)
        except Exception as exc:  # noqa: BLE001
            # The model produced nothing (safety refusal, or thinking consuming the
            # whole budget). Quarantine rather than crash the cell OR score it: an
            # empty output passes every checker vacuously, and a refusal is not a
            # constraint failure. Excluded from rates, counted in generator_errors.
            return {
                "scenario_id": scenario["scenario_id"], "model": model_spec,
                "strategy": strategy, "output": "",
                "checker_results": [], "quality_score": None,
                "quality_reasoning": f"NOT SCORED: {type(exc).__name__}",
                "spec_pass": None,
                harness.GENERATOR_ERROR_FIELD: f"{type(exc).__name__}: {exc}",
            }
        return score_scenario(scenario, constraint_lookup, output, judge_fn,
                              model=model_spec, strategy=strategy,
                              on_judge_error=on_judge_error)

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(one, scenarios))
    else:
        rows = [one(s) for s in scenarios]
    return rows, aggregate(rows, scenarios, constraint_lookup)


def sanitize(spec: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", spec)


def write_outputs(out_dir: Path, model_spec: str, strategy: str, rows, metrics,
                  judge_model: str, eval_set: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{sanitize(model_spec)}__{strategy}"
    with open(out_dir / f"{stem}.transcripts.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    results = {
        "model": model_spec, "strategy": strategy,
        "spec_version": "forbidden-constraint-holder-1.0",
        "judge_rubric_hash": judge_mod.rubric_hash(),
        "judge_model": judge_model,
        "eval_set": eval_set,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
    }
    with open(out_dir / f"{stem}.results.json", "w") as f:
        json.dump(results, f, indent=2)


def fmt_pct(x):
    return "  n/a" if x is None else f"{100 * x:5.1f}%"


def print_table(rows: List[Dict[str, Any]]) -> None:
    print(f"\n{'model':<46} {'strategy':<11} {'SpecAdh':>8} {'Robust':>8} "
          f"{'n':>5} {'judgeErr':>9}")
    for r in rows:
        m = r["metrics"]
        print(f"{r['model']:<46} {r['strategy']:<11} {fmt_pct(m['spec_adherence']):>8} "
              f"{fmt_pct(m['robustness']):>8} {m['n']:>5} {m.get('judge_errors', 0):>9}")


def main():
    ap = argparse.ArgumentParser(description="Forbidden-Constraint Holder eval harness")
    ap.add_argument("--model", required=True, help="provider:model[@revision]")
    ap.add_argument("--eval-set", default=DEFAULT_EVAL_SET)
    ap.add_argument("--constraints", default=DEFAULT_CONSTRAINTS)
    ap.add_argument("--strategy", default="zero_shot", choices=STRATEGIES)
    ap.add_argument("--exemplars", help="JSONL of few-shot exemplars (required for few_shot)")
    ap.add_argument("--compare-with", help="second model spec for base-vs-tuned mode")
    ap.add_argument("--judge", default="anthropic", help="anthropic | mock:<score>")
    ap.add_argument("--judge-model", default=None,
                    help="default: judge_model from configs/models.json")
    ap.add_argument("--workers", type=int, default=4,
                    help="google:gemini-3.1-pro is capped at 25 req/min; keep <=2 for it")
    ap.add_argument("--limit", type=int, default=None,
                    help="score only the first N scenarios (R6 requires >=30 per cell)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out or f"runs/{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    scenarios = load_jsonl(args.eval_set)
    if args.limit:
        scenarios = scenarios[:args.limit]
    constraint_lookup = build_constraint_lookup(load_jsonl(args.constraints))
    exemplars = load_jsonl(args.exemplars) if args.exemplars else None

    # Before a single token is generated or a single judge call is billed. An
    # eval set is often somebody else's file -- the staff held-out set is by
    # definition one we have never seen -- and the fields aggregate() needs were
    # previously read only at the very end, so a missing one destroyed a
    # finished run and wrote nothing.
    try:
        validate_eval_set(scenarios, constraint_lookup, render_user_message)
    except HarnessError as exc:
        print(f"FAIL {args.eval_set}: {exc}", file=sys.stderr)
        return 2

    if args.judge.startswith("mock"):
        _, _, score = args.judge.partition(":")
        judge_fn = judge_mod.MockJudge(int(score) if score else 4)
        judge_model = args.judge
    else:
        judge_model = args.judge_model or judge_mod.load_judge_model()
        judge_fn = judge_mod.make_judge("anthropic", judge_model)

    rows_out = []
    for spec in [args.model] + ([args.compare_with] if args.compare_with else []):
        records, metrics = run_eval(spec, scenarios, constraint_lookup, args.strategy,
                                    exemplars, judge_fn, args.workers)
        write_outputs(out_dir, spec, args.strategy, records, metrics, judge_model,
                      args.eval_set)
        rows_out.append({"model": spec, "strategy": args.strategy, "metrics": metrics})

    print(f"spec forbidden-constraint-holder-1.0  judge rubric {judge_mod.rubric_hash()}  "
          f"n={len(scenarios)}  eval-set={args.eval_set}")
    print_table(rows_out)
    if args.compare_with:
        a, b = rows_out[0]["metrics"], rows_out[1]["metrics"]
        print(f"\ndelta ({rows_out[0]['model']} minus {rows_out[1]['model']}):")
        for k in ("spec_adherence", "robustness"):
            if a[k] is not None and b[k] is not None:
                print(f"  {k:<20} {100 * (a[k] - b[k]):+.1f} pp")
    print(f"\ntranscripts + results written to {out_dir}/")


if __name__ == "__main__":
    # sys.exit(main()), not main(): a bare call discards the return value, so a
    # refused eval set would exit 0 and every caller would read the run as a
    # success that simply produced no output.
    sys.exit(main())
