#!/usr/bin/env python3
"""
G1 Prompt-Ceiling Ablation runner.

Sweeps model x strategy over ONE common scenario corpus, scored by the exact same
harness every later base-vs-tuned comparison uses (imported, not copied). Emits
per-cell transcripts and results plus a run manifest, then scripts/gate_check.py
reads those and returns PROCEED / ESCALATE / HARD_STOP.

  python run_ablation.py --config configs/g1.json
  python run_ablation.py --config configs/g1.json --only frontier
  python run_ablation.py --config configs/g1.json --rung k3_adversarial_60

Two things this deliberately does NOT do:

* It does not decide the gate. gate_check.py does, from the artifacts on disk,
  so the verdict is re-derivable by anyone without re-billing the providers.
* It does not touch thresholds. The escalation ladder changes the SCENARIOS
  (raise k, then raise the adversarial share); it never changes the bar.

Export the environment first -- the harness reads os.environ and loads no dotenv:
  set -a && . ./.env && set +a
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import eval as E
import judge as judge_mod
from harness import build_constraint_lookup

REPO_ROOT = Path(__file__).resolve().parent


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def workers_for(spec: str, cfg: Dict[str, Any]) -> int:
    provider = spec.split(":", 1)[0]
    w = cfg.get("workers", {})
    return int(w.get("by_provider", {}).get(provider, w.get("default", 4)))


def _cells_on_disk(out_dir: Path, frontier: List[str], base: Any) -> List[Dict[str, Any]]:
    """Every cell present in ``out_dir``, re-read from its results JSON.

    The manifest is provenance for the whole run, but the frontier grid and the
    local base row are run as separate invocations (the base row is hours of local
    inference and must not gate the frontier cells). Reporting only the cells this
    process happened to compute would make a complete run look partial.
    """
    found: List[Dict[str, Any]] = []
    for rp in sorted(out_dir.glob("*.results.json")):
        try:
            data = json.loads(rp.read_text())
        except json.JSONDecodeError:
            continue                      # mid-write; the gate re-reads these anyway
        model, strategy = data.get("model"), data.get("strategy")
        metrics = data.get("metrics", {})
        stem = rp.name[: -len(".results.json")]
        found.append({
            "model": model, "strategy": strategy,
            "classification": "frontier" if model in frontier else (
                "local_base" if model == base else "unknown"),
            "results_json": rp.name,
            "transcripts_jsonl": f"{stem}.transcripts.jsonl",
            "spec_adherence": metrics.get("spec_adherence"),
            "robustness": metrics.get("robustness"),
            "n": metrics.get("n"),
            "judge_errors": metrics.get("judge_errors", 0),
        })
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description="G1 Prompt-Ceiling Ablation")
    ap.add_argument("--config", default="configs/g1.json")
    ap.add_argument("--only", choices=("frontier", "base", "all"), default="all",
                    help="frontier = the 6 cells the gate reads; base = the local "
                         "base-model row, which is evidence but not part of the ceiling")
    ap.add_argument("--rung", default=None,
                    help="escalation ladder rung name; defaults to config ladder.current")
    ap.add_argument("--limit", type=int, default=None,
                    help="score only the first N scenarios (R6 floor is 30 per cell)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--models", default=None,
                    help="comma-separated model specs to run, overriding --only. Lets a "
                         "killed row be resumed without re-billing cells already on disk; "
                         "they must still appear in the config, so this cannot smuggle in "
                         "a model the gate did not sanction.")
    ap.add_argument("--manifest-only", action="store_true",
                    help="rebuild run_manifest.json from the cells already on "
                         "disk and exit; calls no provider and re-bills nothing")
    args = ap.parse_args()

    cfg = json.loads((REPO_ROOT / args.config).read_text())
    out_dir = Path(args.out or cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_set_path = REPO_ROOT / cfg["eval_set"]
    constraints_path = REPO_ROOT / cfg.get(
        "constraints", "data/scenarios/constraint_instances.jsonl")
    exemplars_path = REPO_ROOT / cfg["exemplars"]
    for p in (eval_set_path, constraints_path, exemplars_path):
        if not p.exists():
            print(f"ERROR: {p} does not exist. Refusing to run a partial ablation.",
                  file=sys.stderr)
            return 2

    scenarios = E.load_jsonl(str(eval_set_path))
    if args.limit:
        scenarios = scenarios[:args.limit]
    if len(scenarios) < 30:
        print(f"ERROR: {len(scenarios)} scenarios; R6 requires >= 30 per cell.",
              file=sys.stderr)
        return 2

    constraint_lookup = build_constraint_lookup(E.load_jsonl(str(constraints_path)))
    exemplars = E.load_jsonl(str(exemplars_path))

    judge_model = cfg["judge"]["model"]
    judge_fn = judge_mod.make_judge(cfg["judge"].get("provider", "anthropic"), judge_model)

    frontier: List[str] = list(cfg["models"]["frontier"])
    base = cfg["models"].get("local_base")
    if args.only == "frontier":
        specs = frontier
    elif args.only == "base":
        specs = [base] if base else []
    else:
        specs = frontier + ([base] if base else [])

    if args.models:
        wanted = [m.strip() for m in args.models.split(",") if m.strip()]
        sanctioned = set(frontier) | ({base} if base else set())
        unknown = [m for m in wanted if m not in sanctioned]
        if unknown:
            print(f"ERROR: {unknown} are not in configs/g1.json's model list. The gate is "
                  f"defined over the sanctioned grid; running anything else would produce "
                  f"cells it must reject.", file=sys.stderr)
            return 2
        specs = wanted

    strategies = list(cfg["strategies"])
    rung = args.rung or (cfg.get("ladder", {}) or {}).get("current")
    if not rung:
        print("ERROR: no ladder rung. Pass --rung or set ladder.current in the config; "
              "the gate needs to know which rung produced this run.", file=sys.stderr)
        return 2

    print(f"G1 ablation  rung={rung}  n={len(scenarios)}  "
          f"cells={len(specs)}x{len(strategies)}  eval_set={cfg['eval_set']}")

    cells: List[Dict[str, Any]] = []
    if args.manifest_only:
        specs = []          # fall through to the manifest write, calling nothing
    for spec in specs:
        generator = E.make_generator(spec)     # constructed once per model
        w = workers_for(spec, cfg)
        for strategy in strategies:
            rows, metrics = E.run_eval(
                spec, scenarios, constraint_lookup, strategy, exemplars, judge_fn,
                workers=w, generator=generator, on_judge_error="record")
            E.write_outputs(out_dir, spec, strategy, rows, metrics, judge_model,
                            cfg["eval_set"])
            stem = f"{E.sanitize(spec)}__{strategy}"
            cells.append({
                "model": spec, "strategy": strategy,
                "classification": "frontier" if spec in frontier else "local_base",
                "results_json": f"{stem}.results.json",
                "transcripts_jsonl": f"{stem}.transcripts.jsonl",
                "spec_adherence": metrics["spec_adherence"],
                "robustness": metrics["robustness"],
                "n": metrics["n"],
                "judge_errors": metrics.get("judge_errors", 0),
            })
            # A cell whose rows were all quarantined has SA/robustness of None.
            # Formatting None raised and killed the whole run AFTER the cell's data
            # was already written -- a reporting bug destroying a paid run twice.
            def _p(x):
                return "  none" if x is None else f"{x:.3f}"
            sa, rb = metrics["spec_adherence"], metrics["robustness"]
            print(f"  [cell] {spec} x {strategy}: SA={_p(sa)} R={_p(rb)} "
                  f"fail={_p(None if sa is None else 1 - sa)} "
                  f"scored={metrics.get('n_scored')} "
                  f"judgeErr={metrics.get('judge_errors', 0)} "
                  f"genErr={metrics.get('generator_errors', 0)}")
            if sa is None:
                print(f"         ^ every row quarantined; the cell measured nothing. "
                      f"Investigate before trusting this run.")

    # Re-derive the cell list from disk so a manifest written by either invocation
    # describes the WHOLE run. The frontier and base rows are run separately.
    cells = _cells_on_disk(out_dir, frontier, base) or cells

    manifest = {
        "run_id": cfg.get("run_id", "g1"),
        "ladder_rung": rung,
        "eval_set": cfg["eval_set"],
        "eval_set_sha256": sha256_of(eval_set_path),
        "constraints": cfg.get("constraints", "data/scenarios/constraint_instances.jsonl"),
        "constraints_sha256": sha256_of(constraints_path),
        "exemplars": cfg["exemplars"],
        "n_scenarios": len(scenarios),
        "models": {"frontier": frontier, "local_base": base},
        "strategies": strategies,
        "judge": {"provider": cfg["judge"].get("provider", "anthropic"),
                  "model": judge_model, "rubric_hash": judge_mod.rubric_hash()},
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cells": cells,
        "_note": ("Provenance only. scripts/gate_check.py re-derives every metric from "
                  "the transcripts and re-runs the checkers; nothing here is trusted "
                  "as a measurement."),
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    frontier_cells = [c for c in cells if c["classification"] == "frontier"
                      and c["spec_adherence"] is not None]
    if frontier_cells:
        best = max(frontier_cells, key=lambda c: c["spec_adherence"])
        print(f"\nbest frontier cell: {best['model']} x {best['strategy']} "
              f"SA={best['spec_adherence']:.4f}  ->  fail_rate="
              f"{1 - best['spec_adherence']:.4f}   (R6 bar is >= 0.20)")
    print(f"\nartifacts in {out_dir}/  — now run:\n"
          f"  python scripts/gate_check.py --config {args.config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
