#!/usr/bin/env python3
"""
Prompt-Ceiling Ablation runner.

Sweeps every model x strategy cell over ONE common eval set, scored by the
exact same rubric as eval.py (imported, not copied). Emits:
  - per-cell transcripts JSONL + results JSON (via eval.write_outputs)
  - ablation_table.md  (mean Spec-adherence / Robustness per model x strategy)
  - ablation_summary.json

Usage:
  run_ablation.py \
    --models "anthropic:<model>,openai:<model>,hf:Qwen/Qwen3-1.7B" \
    --strategies zero_shot,few_shot,structured \
    --eval-set scenarios_v1.jsonl --exemplars fewshot_exemplars.jsonl \
    --out runs/ablation

Notes:
  - Include the prompted small base model (hf:...) as a third row: it
    evidences the doc's "a well-prompted base model can't already do it
    reliably" test and doubles as the MVP base-vs-tuned baseline.
  - Each generator is constructed once and reused across its strategies
    (matters for hf: models, which load weights).
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from eval import (RUBRIC_VERSION, load_jsonl, make_generator, make_judge,
                  rubric_hash, run_eval, write_outputs, print_table,
                  DEFAULT_JUDGE_MODEL)


def main():
    ap = argparse.ArgumentParser(description="Prompt-Ceiling Ablation")
    ap.add_argument("--models", required=True,
                    help="comma-separated model specs (provider:model[@rev])")
    ap.add_argument("--strategies", default="zero_shot,few_shot,structured")
    ap.add_argument("--eval-set", required=True)
    ap.add_argument("--exemplars", required=True)
    ap.add_argument("--judge", default="anthropic")
    ap.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="runs/ablation")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    eval_set = load_jsonl(args.eval_set)
    exemplars = load_jsonl(args.exemplars)
    judge_fn = make_judge(args.judge, args.judge_model)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, cells = [], {}
    for spec in models:
        gen = make_generator(spec)  # construct once per model
        for strat in strategies:
            records, metrics = run_eval(spec, eval_set, strat, exemplars,
                                        judge_fn, args.workers, generator=gen)
            write_outputs(out_dir, spec, strat, records, metrics, args.judge_model)
            rows.append({"model": spec, "strategy": strat, "metrics": metrics})
            cells[(spec, strat)] = metrics
            m = metrics
            print(f"[cell done] {spec} x {strat}: "
                  f"SA={m['spec_adherence']:.3f} R={m['robustness']:.3f}")

    # markdown table for the Architecture Defense
    lines = [
        f"# Prompt-Ceiling Ablation — rubric v{RUBRIC_VERSION} "
        f"(hash `{rubric_hash()}`), n={len(eval_set)} common scenarios, "
        f"judge={args.judge_model}",
        "",
        "Cell = mean Spec-adherence / Robustness (adversarial subset).",
        "",
        "| model | " + " | ".join(strategies) + " |",
        "|---|" + "---|" * len(strategies),
    ]
    for spec in models:
        cell_strs = []
        for strat in strategies:
            m = cells[(spec, strat)]
            cell_strs.append(f"{100 * m['spec_adherence']:.1f}% / "
                             f"{100 * m['robustness']:.1f}%")
        lines.append(f"| `{spec}` | " + " | ".join(cell_strs) + " |")
    lines += ["", "## Failure mode that survives the best prompting attempt",
              "", "_TODO: written after inspecting transcripts — name the specific",
              "failure mode, with scenario IDs as evidence._", ""]
    (out_dir / "ablation_table.md").write_text("\n".join(lines))

    with open(out_dir / "ablation_summary.json", "w") as f:
        json.dump({"rubric_version": RUBRIC_VERSION, "rubric_hash": rubric_hash(),
                   "judge_model": args.judge_model, "n_scenarios": len(eval_set),
                   "eval_set": args.eval_set,
                   "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                   "cells": rows}, f, indent=2)

    print_table(rows)
    print(f"\nablation_table.md + ablation_summary.json written to {out_dir}/")


if __name__ == "__main__":
    main()
