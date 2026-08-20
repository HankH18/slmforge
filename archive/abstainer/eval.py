#!/usr/bin/env python3
"""
Grounded Abstainer — eval harness (rubric v1.0)

Behavior Spec:
  Given a question and a context passage, the model answers only if the
  passage contains the answer, outputting exactly one `Answer: <answer>`
  line followed by one or more `Citation: "<span>"` lines where every
  cited span is a verbatim substring of the passage and the spans jointly
  entail the answer; otherwise its entire output is exactly
  `Not answerable from the provided context.`

Pass rule (per scenario, given the gold label):
  gold=answerable   -> format parses AND every cited span is a verbatim
                       substring of the passage (programmatic) AND the
                       spans entail the answer (LLM judge, single binary
                       question, invoked ONLY when the programmatic checks
                       pass -- the judge can flip a would-be pass to fail,
                       never rescue a programmatic fail)
  gold=unanswerable -> output equals the refusal string (programmatic)

Normalizations applied before checking (the ONLY two; everything else is
exact-match):
  1. strip leading/trailing whitespace from the whole output
  2. drop blank lines

Metrics:
  spec_adherence    pass rate over all scenarios
  robustness        pass rate over scenarios marked adversarial=true
  false_answer_rate  P(answered | gold unanswerable)
  false_refusal_rate P(refused  | gold answerable)
  plus a per-failure-mode breakdown and a gold x predicted confusion matrix

Usage:
  eval.py --model hf:Qwen/Qwen3-1.7B --eval-set scenarios_v1.jsonl
  eval.py --model hf:you/tuned-repo@<commit> --compare-with hf:Qwen/Qwen3-1.7B \
          --eval-set scenarios_v1.jsonl --strategy zero_shot --out runs/mvp
  eval.py --model anthropic:<model> --strategy few_shot \
          --exemplars fewshot_exemplars.jsonl --eval-set scenarios_v1.jsonl

Model specs:
  hf:<repo>[@revision]   local inference via transformers (pin the revision!)
  anthropic:<model>      Anthropic API   (env ANTHROPIC_API_KEY)
  openai:<model>         OpenAI API      (env OPENAI_API_KEY)
  google:<model>         Google GenAI    (env GOOGLE_API_KEY / GEMINI_API_KEY)
  mock:<behavior>        test doubles: oracle | always_refuse |
                         nonverbatim | format_breaker | bad_entail

Judge: --judge anthropic (default; model via --judge-model or env
JUDGE_MODEL) or --judge mock:yes / mock:no for pipeline tests.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

RUBRIC_VERSION = "1.0"
REFUSAL_STRING = "Not answerable from the provided context."
DEFAULT_JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "claude-sonnet-4-6")

ANSWER_RE = re.compile(r'^Answer: (.+)$')
CITATION_RE = re.compile(r'^Citation: "(.*)"$')

# ---------------------------------------------------------------- prompts
SYSTEM_CORE = """You answer questions using ONLY the provided context passage. Never use outside knowledge.

Output format -- follow it exactly, with no other text before or after:
- If the passage fully supports an answer, output exactly one line `Answer: <your answer>` followed by one or more lines `Citation: "<span>"`, where each span is copied verbatim, character for character, from the passage and supports the answer.
- If the passage does not fully support an answer, output exactly this single line and nothing else:
Not answerable from the provided context."""

STRUCTURED_ADDENDUM = """

Before producing your output, silently run this verification protocol (never include the protocol or any reasoning in your output):
1. Locate the candidate span or spans in the passage that would answer the question.
2. Confirm each candidate span appears verbatim, character for character, in the passage.
3. Confirm the spans, taken alone and without any outside knowledge, entail the answer to this exact question -- not a similar question about a different attribute, entity, or time period.
If any step fails, output the refusal line exactly."""

JUDGE_PROMPT = """Question: {q}
Answer given: {a}
Cited span(s):
{spans}

Do the cited span(s), taken alone and without outside knowledge, entail that this answer correctly answers the question? Reply with YES or NO on the first line, then one sentence of reasoning."""

STRATEGIES = ("zero_shot", "few_shot", "structured")


def rubric_hash() -> str:
    """Pin of everything that defines scoring: prompts + judge prompt + version."""
    blob = "\x1e".join([RUBRIC_VERSION, REFUSAL_STRING, SYSTEM_CORE,
                        STRUCTURED_ADDENDUM, JUDGE_PROMPT])
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def user_msg(passage: str, question: str) -> str:
    return f"Context passage:\n{passage}\n\nQuestion: {question}"


def build_prompt(strategy: str, scenario: dict, exemplars: list | None) -> dict:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}")
    system = SYSTEM_CORE + (STRUCTURED_ADDENDUM if strategy == "structured" else "")
    turns = []
    if strategy == "few_shot":
        if not exemplars:
            raise ValueError("few_shot strategy requires --exemplars")
        for ex in exemplars:
            turns.append({"role": "user", "content": user_msg(ex["passage"], ex["question"])})
            turns.append({"role": "assistant", "content": ex["target_output"]})
    turns.append({"role": "user", "content": user_msg(scenario["passage"], scenario["question"])})
    return {"system": system, "turns": turns}


# ---------------------------------------------------------------- parsing / checks
def parse_output(text: str) -> dict:
    t = text.strip()                                   # normalization 1
    if t == REFUSAL_STRING:
        return {"class": "refusal"}
    lines = [ln for ln in t.split("\n") if ln.strip()]  # normalization 2
    if not lines:
        return {"class": "malformed", "why": "empty output"}
    m = ANSWER_RE.match(lines[0])
    if not m:
        return {"class": "malformed", "why": f"first line is not an Answer line: {lines[0][:60]!r}"}
    if len(lines) < 2:
        return {"class": "malformed", "why": "Answer line without any Citation line"}
    spans = []
    for ln in lines[1:]:
        c = CITATION_RE.match(ln)
        if not c:
            return {"class": "malformed", "why": f"non-Citation line after Answer: {ln[:60]!r}"}
        spans.append(c.group(1))
    return {"class": "answer", "answer": m.group(1), "spans": spans}


def score_example(scenario: dict, raw_output: str, judge_fn) -> dict:
    """The rubric. Used identically for the prompt-ceiling ablation and every
    base-vs-tuned comparison."""
    parsed = parse_output(raw_output)
    rec = {
        "scenario_id": scenario["id"],
        "type": scenario.get("type"),
        "adversarial": bool(scenario.get("adversarial")),
        "gold_label": scenario["gold_label"],
        "raw_output": raw_output,
        "pred_class": parsed["class"],
        "judge_verdict": None,
        "judge_reasoning": None,
    }
    if scenario["gold_label"] == "unanswerable":
        if parsed["class"] == "refusal":
            rec.update(passed=True, failure=None)
        elif parsed["class"] == "answer":
            rec.update(passed=False, failure="false_answer")
        else:
            rec.update(passed=False, failure="malformed")
        return rec

    # gold answerable
    if parsed["class"] == "refusal":
        rec.update(passed=False, failure="false_refusal")
        return rec
    if parsed["class"] == "malformed":
        rec.update(passed=False, failure="malformed")
        return rec
    nonverbatim = [s for s in parsed["spans"] if s not in scenario["passage"]]
    if nonverbatim:
        rec.update(passed=False, failure="nonverbatim_citation",
                   nonverbatim_spans=nonverbatim)
        return rec
    verdict, reasoning = judge_fn(scenario["question"], parsed["answer"], parsed["spans"])
    rec["judge_verdict"] = verdict
    rec["judge_reasoning"] = reasoning
    rec.update(passed=bool(verdict), failure=None if verdict else "entailment_fail")
    return rec


def aggregate(records: list[dict]) -> dict:
    n = len(records)
    passes = sum(r["passed"] for r in records)
    adv = [r for r in records if r["adversarial"]]
    unans = [r for r in records if r["gold_label"] == "unanswerable"]
    ans = [r for r in records if r["gold_label"] == "answerable"]
    confusion = {}
    for r in records:
        key = f"gold_{r['gold_label']}__pred_{r['pred_class']}"
        confusion[key] = confusion.get(key, 0) + 1
    failures = {}
    for r in records:
        if r["failure"]:
            failures[r["failure"]] = failures.get(r["failure"], 0) + 1
    return {
        "n": n,
        "spec_adherence": passes / n if n else None,
        "robustness": (sum(r["passed"] for r in adv) / len(adv)) if adv else None,
        "false_answer_rate": (sum(r["pred_class"] == "answer" for r in unans) / len(unans)) if unans else None,
        "false_refusal_rate": (sum(r["pred_class"] == "refusal" for r in ans) / len(ans)) if ans else None,
        "failure_breakdown": failures,
        "confusion": confusion,
    }


# ---------------------------------------------------------------- judges
class MockJudge:
    def __init__(self, verdict: bool):
        self.verdict = verdict

    def __call__(self, question, answer, spans):
        return self.verdict, f"mock judge: auto-{'YES' if self.verdict else 'NO'}"


class AnthropicJudge:
    """Single binary entailment question. temp 0, pinned model + prompt.
    Judge parse/API failure after retries scores conservative FAIL (documented)."""

    def __init__(self, model: str = DEFAULT_JUDGE_MODEL):
        import anthropic  # lazy: not needed for mock runs
        self.client = anthropic.Anthropic()
        self.model = model

    def __call__(self, question, answer, spans):
        prompt = JUDGE_PROMPT.format(
            q=question, a=answer, spans="\n".join(f"- {s}" for s in spans))
        last_err = "unparseable"
        for attempt in range(3):
            try:
                resp = self.client.messages.create(
                    model=self.model, max_tokens=250, temperature=0,
                    messages=[{"role": "user", "content": prompt}])
                text = "".join(b.text for b in resp.content
                               if getattr(b, "type", "") == "text").strip()
                first = text.split("\n", 1)[0].strip().upper()
                if first.startswith("YES"):
                    return True, text
                if first.startswith("NO"):
                    return False, text
                last_err = f"unparseable verdict: {text[:80]!r}"
            except Exception as e:  # noqa: BLE001
                last_err = str(e)
                time.sleep(2 ** attempt)
        return False, f"judge_error_conservative_fail: {last_err}"


def make_judge(spec: str, judge_model: str):
    if spec == "anthropic":
        return AnthropicJudge(judge_model)
    if spec == "mock:yes":
        return MockJudge(True)
    if spec == "mock:no":
        return MockJudge(False)
    raise ValueError(f"unknown judge spec {spec!r}")


# ---------------------------------------------------------------- generators
def _retrying(fn, tries=3):
    def wrapped(*a, **kw):
        for attempt in range(tries):
            try:
                return fn(*a, **kw)
            except Exception:  # noqa: BLE001
                if attempt == tries - 1:
                    raise
                time.sleep(2 ** attempt)
    return wrapped


class MockGenerator:
    """Test doubles that exercise every branch of the rubric."""

    def __init__(self, behavior: str):
        self.behavior = behavior

    def __call__(self, scenario, prompt):
        b = self.behavior
        if b == "always_refuse":
            return REFUSAL_STRING
        if b == "format_breaker":
            return "Sure! Based on the passage, I think the answer is probably 42."
        if b == "nonverbatim":
            return 'Answer: a guess.\nCitation: "THIS SPAN IS NOT IN THE PASSAGE"'
        if b == "oracle":
            if scenario["gold_label"] == "unanswerable":
                return REFUSAL_STRING
            return (f"Answer: {scenario['gold_answer']}\n"
                    f'Citation: "{scenario["gold_spans"][0]}"')
        if b == "bad_entail":
            if scenario["gold_label"] == "unanswerable":
                return REFUSAL_STRING
            span = " ".join(scenario["passage"].split()[:8])
            return f'Answer: a deliberately wrong answer.\nCitation: "{span}"'
        raise ValueError(f"unknown mock behavior {b!r}")


class HFGenerator:
    def __init__(self, repo: str, revision: str | None):
        from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy
        self.tok = AutoTokenizer.from_pretrained(repo, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            repo, revision=revision, torch_dtype="auto", device_map="auto")
        self.thread_safe = False  # force workers=1

    def __call__(self, scenario, prompt):
        msgs = [{"role": "system", "content": prompt["system"]}] + prompt["turns"]
        try:  # Qwen3 chat templates accept enable_thinking; disable it — thinking
            #   prose would (correctly) fail the format check but we want to
            #   measure abstention, not a reasoning-channel artifact.
            text = self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(text, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
        return self.tok.decode(out[0][inputs["input_ids"].shape[1]:],
                               skip_special_tokens=True)


class AnthropicGenerator:
    def __init__(self, model: str):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.thread_safe = True

    @_retrying
    def __call__(self, scenario, prompt):
        resp = self.client.messages.create(
            model=self.model, max_tokens=512, temperature=0,
            system=prompt["system"], messages=prompt["turns"])
        return "".join(b.text for b in resp.content
                       if getattr(b, "type", "") == "text")


class OpenAIGenerator:
    def __init__(self, model: str):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model
        self.thread_safe = True

    @_retrying
    def __call__(self, scenario, prompt):
        msgs = [{"role": "system", "content": prompt["system"]}] + prompt["turns"]
        try:
            resp = self.client.chat.completions.create(
                model=self.model, messages=msgs, temperature=0)
        except Exception:
            # some reasoning-series models reject temperature; retry without
            resp = self.client.chat.completions.create(model=self.model, messages=msgs)
        return resp.choices[0].message.content or ""


class GoogleGenerator:
    def __init__(self, model: str):
        from google import genai
        from google.genai import types
        self.genai, self.types = genai, types
        self.client = genai.Client()
        self.model = model
        self.thread_safe = True

    @_retrying
    def __call__(self, scenario, prompt):
        t = self.types
        contents = [t.Content(role="user" if turn["role"] == "user" else "model",
                              parts=[t.Part.from_text(text=turn["content"])])
                    for turn in prompt["turns"]]
        resp = self.client.models.generate_content(
            model=self.model, contents=contents,
            config=t.GenerateContentConfig(temperature=0,
                                           system_instruction=prompt["system"]))
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
def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_eval(model_spec: str, eval_set: list[dict], strategy: str,
             exemplars: list | None, judge_fn, workers: int = 8,
             generator=None) -> tuple[list[dict], dict]:
    gen = generator or make_generator(model_spec)
    if not getattr(gen, "thread_safe", True):
        workers = 1

    def one(scenario):
        prompt = build_prompt(strategy, scenario, exemplars)
        raw = gen(scenario, prompt)
        rec = score_example(scenario, raw, judge_fn)
        rec.update(model=model_spec, strategy=strategy)
        return rec

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            records = list(pool.map(one, eval_set))
    else:
        records = [one(s) for s in eval_set]
    return records, aggregate(records)


def sanitize(spec: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", spec)


def write_outputs(out_dir: Path, model_spec: str, strategy: str,
                  records: list[dict], metrics: dict, judge_model: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{sanitize(model_spec)}__{strategy}"
    with open(out_dir / f"{stem}.transcripts.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    results = {
        "model": model_spec, "strategy": strategy,
        "rubric_version": RUBRIC_VERSION, "rubric_hash": rubric_hash(),
        "judge_model": judge_model,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
    }
    with open(out_dir / f"{stem}.results.json", "w") as f:
        json.dump(results, f, indent=2)


def fmt_pct(x):
    return "  n/a" if x is None else f"{100 * x:5.1f}%"


def print_table(rows: list[dict]) -> None:
    print(f"\n{'model':<44} {'strategy':<11} {'SpecAdh':>8} {'Robust':>8} "
          f"{'FalseAns':>9} {'FalseRef':>9}  failures")
    for r in rows:
        m = r["metrics"]
        fails = ", ".join(f"{k}={v}" for k, v in sorted(m["failure_breakdown"].items())) or "-"
        print(f"{r['model']:<44} {r['strategy']:<11} {fmt_pct(m['spec_adherence']):>8} "
              f"{fmt_pct(m['robustness']):>8} {fmt_pct(m['false_answer_rate']):>9} "
              f"{fmt_pct(m['false_refusal_rate']):>9}  {fails}")


def main():
    ap = argparse.ArgumentParser(description="Grounded Abstainer eval harness")
    ap.add_argument("--model", required=True, help="provider:model[@revision]")
    ap.add_argument("--eval-set", required=True)
    ap.add_argument("--strategy", default="zero_shot", choices=STRATEGIES)
    ap.add_argument("--exemplars", help="JSONL of few-shot exemplars (required for few_shot)")
    ap.add_argument("--compare-with", help="second model spec for base-vs-tuned mode")
    ap.add_argument("--judge", default="anthropic", help="anthropic | mock:yes | mock:no")
    ap.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None, help="output dir (default runs/<utc timestamp>)")
    args = ap.parse_args()

    out_dir = Path(args.out or f"runs/{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    eval_set = load_jsonl(args.eval_set)
    exemplars = load_jsonl(args.exemplars) if args.exemplars else None
    judge_fn = make_judge(args.judge, args.judge_model)

    rows = []
    for spec in [args.model] + ([args.compare_with] if args.compare_with else []):
        records, metrics = run_eval(spec, eval_set, args.strategy, exemplars,
                                    judge_fn, args.workers)
        write_outputs(out_dir, spec, args.strategy, records, metrics, args.judge_model)
        rows.append({"model": spec, "strategy": args.strategy, "metrics": metrics})

    print(f"rubric v{RUBRIC_VERSION} (hash {rubric_hash()})  n={len(eval_set)}  "
          f"eval-set={args.eval_set}")
    print_table(rows)
    if args.compare_with:
        a, b = rows[0]["metrics"], rows[1]["metrics"]
        print(f"\ndelta ({rows[0]['model']} minus {rows[1]['model']}):")
        for k in ("spec_adherence", "robustness", "false_answer_rate", "false_refusal_rate"):
            if a[k] is not None and b[k] is not None:
                print(f"  {k:<20} {100 * (a[k] - b[k]):+.1f} pp")
    print(f"\ntranscripts + results written to {out_dir}/")


if __name__ == "__main__":
    main()
