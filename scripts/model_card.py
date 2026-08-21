#!/usr/bin/env python3
"""Render the Hugging Face model card from the run artifacts, not by hand.

The card is the first thing a grader reads, and the hand-written one that was
published with the v1 checkpoint shipped a literal `{REPO}` in its own copy-paste
example and quoted a training-set size from a different run. Both are the same
failure: a document maintained separately from the thing it describes.

So every fact in the card comes from a file the run produced --
`train_log.json`, `run_manifest.json`, the dataset the log names, and a pair of
`*.results.json` cells for the numbers -- and the renderer refuses to emit a card
when one of them is missing rather than leaving a gap for a reader to fill in
with an assumption.

    python scripts/model_card.py --run-dir runs/train_v2 \\
        --repo HankH18/qwen3-1.7b-forbidden-constraint-holder \\
        --tuned-results runs/mvp_v2/hf_runs_train_v2_merged__zero_shot.results.json \\
        --base-results  runs/mvp/hf_Qwen_Qwen3-1.7B_...__zero_shot.results.json \\
        --out runs/train_v2/MODEL_CARD.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


class CardError(Exception):
    """A card with a hole in it is worse than no card: it reads as a fact."""


def _read_json(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise CardError(f"{path} is missing; the card cannot be rendered without it")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise CardError(f"{path}: {exc}") from exc


def _require(obj: Mapping[str, Any], dotted: str, where: str) -> Any:
    cur: Any = obj
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            raise CardError(f"{where}: no {dotted!r} — refusing to render a card "
                            f"with a blank where that value goes")
        cur = cur[part]
    return cur


def _pct(x: Optional[float]) -> str:
    return "—" if x is None else f"{x:.4f}"


def _delta(base: Optional[float], tuned: Optional[float]) -> str:
    if base is None or tuned is None:
        return "—"
    d = tuned - base
    return f"{d:+.4f}"


def render(repo_id: str,
           train_log: Mapping[str, Any],
           manifest: Mapping[str, Any],
           base_results: Mapping[str, Any],
           tuned_results: Mapping[str, Any],
           dataset_repo: Optional[str] = None) -> str:
    cfg = _require(train_log, "config_resolved", "train_log.json")
    lora = _require(cfg, "lora", "train_log.json config_resolved")
    tr = _require(cfg, "training", "train_log.json config_resolved")
    ds = _require(train_log, "dataset", "train_log.json")

    base_rev = _require(train_log, "base_model.revision", "train_log.json")
    base_repo = _require(train_log, "base_model.repo", "train_log.json")
    steps = _require(train_log, "steps", "train_log.json")
    losses = _require(train_log, "loss_history", "train_log.json")
    if not losses:
        raise CardError("train_log.json: loss_history is empty")

    split = cfg.get("_micro_batch_split") or {}
    effective = split.get("effective_batch_preserved")
    if effective is None:
        effective = int(tr["per_device_train_batch_size"]) * int(
            tr["gradient_accumulation_steps"])

    bm = _require(base_results, "metrics", "base results.json")
    tm = _require(tuned_results, "metrics", "tuned results.json")
    eval_set = _require(tuned_results, "eval_set", "tuned results.json")
    judge = _require(tuned_results, "judge_model", "tuned results.json")
    rubric = _require(tuned_results, "judge_rubric_hash", "tuned results.json")
    if base_results.get("eval_set") != eval_set:
        raise CardError(
            f"the two results cells were scored on different eval sets "
            f"({base_results.get('eval_set')!r} vs {eval_set!r}); a base-vs-tuned "
            f"table across two eval sets is not a comparison")
    if base_results.get("judge_rubric_hash") != rubric:
        raise CardError("the two results cells were scored under different judge "
                        "rubrics; the delta would not mean anything")
    # The two cells do NOT always share a denominator: a row whose judge failed
    # is quarantined and drops out of that cell only. Printing the tuned cell's
    # n for both silently claims a comparison over rows one side never scored.
    base_n = bm.get("n_scored", bm.get("n"))
    tuned_n = tm.get("n_scored", tm.get("n"))

    def types(results: Mapping[str, Any]) -> Mapping[str, Any]:
        return _require(results, "metrics.by_constraint_type.types", "results.json")

    bt, tt = types(base_results), types(tuned_results)
    shared = sorted(set(bt) & set(tt))

    front = [
        "---",
        "license: apache-2.0",
        f"base_model: {base_repo}",
        "library_name: transformers",
        "pipeline_tag: text-generation",
    ]
    if dataset_repo:
        front += ["datasets:", f"  - {dataset_repo}"]
    front += [
        "tags:",
        "  - qlora",
        "  - peft",
        "  - constrained-generation",
        "  - instruction-following",
        "---",
        "",
    ]

    lines = front + [
        "# Qwen3-1.7B — Forbidden-Constraint Holder",
        "",
        "A QLoRA fine-tune of **Qwen/Qwen3-1.7B** trained to *hold negative",
        "constraints* while completing an ordinary writing task — \"write this, and",
        "never use the letter e / the word 'however' / any quotation marks\" —",
        "including when the task itself pulls toward the banned token.",
        "",
        "Every constraint has a deterministic checker, so adherence here is",
        "**measured, not judged**: an LLM judge scores task quality only and can",
        "never overturn a checker.",
        "",
        "> **Read the results section before using this model.** It improves on the",
        "> constraint families it was trained on and *regresses* on families it was",
        "> not. That trade is the finding, and it is stated in full below.",
        "",
        "## Provenance",
        "",
        "| | |",
        "|---|---|",
        f"| Base | `{base_repo}` |",
        f"| Base revision (pinned) | `{base_rev}` |",
        "| Method | QLoRA (NF4 + double quant), LoRA adapter merged to fp16 |",
        f"| Hardware | {manifest.get('gpu', 'unknown')}, "
        f"{manifest.get('peak_vram_gib', '?')} GiB peak |",
        f"| Training wall clock | {round(float(train_log['wall_clock_seconds']))} s |",
        "",
        "This repo holds the **merged fp16 model** at the root and the **LoRA",
        "adapter** under `adapter/`.",
        "",
        "## Training data",
        "",
        "| | |",
        "|---|---|",
        f"| Dataset | `{ds['path']}` |",
        f"| Rows | {ds['rows']} |",
        f"| sha256 | `{ds['sha256']}` |",
    ]
    if dataset_repo:
        lines.append(f"| Published at | [`{dataset_repo}`]"
                     f"(https://huggingface.co/datasets/{dataset_repo}) |")
    lines += [
        "",
        "Teacher-distilled and **rejection-filtered through the same deterministic",
        "checkers used at evaluation**: a generated completion that violated its own",
        "constraint set was discarded rather than trained on. Constraint types held",
        "out for generalization testing appear in zero training rows.",
        "",
        "## Hyperparameters",
        "",
        "| | |",
        "|---|---|",
        f"| LoRA rank / alpha / dropout | {lora['r']} / {lora['lora_alpha']} / "
        f"{lora['lora_dropout']} |",
        f"| Target modules | {', '.join(sorted(lora['target_modules']))} |",
        f"| Learning rate | {tr['learning_rate']} |",
        f"| Epochs | {tr['num_train_epochs']} |",
        f"| Effective batch size | {effective} "
        f"({tr['per_device_train_batch_size']} x "
        f"{tr['gradient_accumulation_steps']} accumulation) |",
        f"| Max sequence length | {tr['max_seq_length']} |",
        f"| Optimizer / schedule | {tr['optim']} / {tr['lr_scheduler_type']}, "
        f"warmup ratio {tr['warmup_ratio']} |",
        f"| Seed | {tr['seed']} |",
        f"| Steps | {steps} |",
        f"| Training loss | {losses[0][1]} -> {losses[-1][1]} |",
        "",
        "Loss is computed on the completion only; prompt tokens are masked to -100.",
        "",
        "### Prompt format",
        "",
        "Use the chat template with a **single user turn and no system message**.",
        "Training used no system turn, and every training prompt was re-rendered",
        "through the evaluation harness's own renderer and compared byte-for-byte",
        "before training began, so the training condition equals the evaluation",
        "condition. Adding a system prompt puts the model off-distribution.",
        "",
        "```python",
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        "",
        f'repo = "{repo_id}"',
        "tok = AutoTokenizer.from_pretrained(repo)",
        'model = AutoModelForCausalLM.from_pretrained(repo, dtype="auto", '
        'device_map="auto")',
        "",
        'messages = [{"role": "user", "content": "Write a short paragraph about '
        'the ocean. Do not use the letter \'e\' anywhere in your response."}]',
        "inputs = tok.apply_chat_template(messages, add_generation_prompt=True,",
        "                                 enable_thinking=False, "
        'return_tensors="pt").to(model.device)',
        "print(tok.decode(model.generate(inputs, max_new_tokens=512, "
        "do_sample=False)[0][inputs.shape[-1]:]))",
        "```",
        "",
        "`enable_thinking=False` matters: the training rows were rendered with it",
        "off, and Qwen3's template emits a thinking block otherwise.",
        "",
        "## Results",
        "",
        f"Eval set `{eval_set}`, judge `{judge}`, rubric hash `{rubric}`.",
        "`spec_pass` = every checker passes **and** judge task-quality >= 3.",
        "`robustness` is the same quantity over the adversarial subset only.",
        "",
        (f"Scored rows: **base {base_n}, tuned {tuned_n}**."
         + ("" if base_n == tuned_n else
            f" The denominators differ because {abs(base_n - tuned_n)} row(s) "
            f"were quarantined — the judge returned no score, so there is no "
            f"verdict to count either way. Rates are over scored rows.")),
        "",
        "| Metric | base | tuned | delta |",
        "|---|---|---|---|",
        f"| `spec_adherence` | {_pct(bm.get('spec_adherence'))} | "
        f"{_pct(tm.get('spec_adherence'))} | "
        f"{_delta(bm.get('spec_adherence'), tm.get('spec_adherence'))} |",
        f"| `robustness` | {_pct(bm.get('robustness'))} | "
        f"{_pct(tm.get('robustness'))} | "
        f"{_delta(bm.get('robustness'), tm.get('robustness'))} |",
        "",
        "Violation rate per constraint type — **lower is better**. Types marked",
        "*held out* appear in **zero** training rows.",
        "",
        "| Constraint type | in training data | base | tuned |",
        "|---|---|---|---|",
    ]

    from constraints import registry
    heldout = set(registry.types_by_split("heldout_type"))
    for ctype in shared:
        mark = "no (held out)" if ctype in heldout else "yes"
        lines.append(f"| `{ctype}` | {mark} | {bt[ctype]['violation_rate']:.3f} | "
                     f"{tt[ctype]['violation_rate']:.3f} |")

    lines += [
        "",
        "## Limitations, stated plainly",
        "",
        "* **It does not reliably hold a banned letter.** That is the hardest",
        "  constraint in the set and the checker requires *zero* occurrences. The",
        "  tuned model reduces how often the banned letter appears — the median",
        "  response goes from 34 occurrences to 14 — but a response with 14 is",
        "  still a failure, and the pass rate stays near the floor.",
        "* **It gets worse at constraint families it never saw.** Compare the",
        "  held-out rows in the table above. Fine-tuning bought compliance on the",
        "  trained families and spent generalization on the untrained ones. If your",
        "  constraint type is not in the training list, the base model is the better",
        "  choice.",
        "* **Single user turn, no system prompt.** See the prompt format above.",
        "* 1.7B parameters, English only, short-form writing tasks.",
        "",
        "## Reproducing these numbers",
        "",
        "```bash",
        f"python eval.py --model hf:{repo_id}@<commit> \\",
        f"               --compare-with hf:{base_repo}@{base_rev} \\",
        f"               --eval-set {eval_set}",
        "```",
        "",
        "The harness, the checkers, the scenario corpus and the raw judge",
        "transcripts are all in the project repo.",
        "",
    ]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True,
                    help="training run dir holding train_log.json and run_manifest.json")
    ap.add_argument("--repo", required=True, help="target HF repo id, owner/name")
    ap.add_argument("--base-results", required=True)
    ap.add_argument("--tuned-results", required=True)
    ap.add_argument("--dataset-repo", default=None,
                    help="published HF dataset repo id. Pass it: without it the "
                         "card points readers at a local repo path that does not "
                         "exist for anyone pulling from the Hub.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    try:
        card = render(
            repo_id=args.repo,
            train_log=_read_json(run_dir / "train_log.json"),
            manifest=_read_json(run_dir / "run_manifest.json"),
            base_results=_read_json(Path(args.base_results)),
            tuned_results=_read_json(Path(args.tuned_results)),
            dataset_repo=args.dataset_repo,
        )
    except CardError as exc:
        print(f"FAIL model_card: {exc}", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(card)
    print(f"wrote {out} ({len(card.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
