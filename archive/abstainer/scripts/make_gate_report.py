#!/usr/bin/env python3
"""
P2 GATE G1 — compute runs/ablation/gate_report.json from the committed ablation.

    python scripts/make_gate_report.py runs/ablation
    python scripts/make_gate_report.py <fixture_dir> --out <fixture_dir>/gate_report.json

The frozen `run_ablation.py` writes the per-cell results/transcripts plus
ablation_table.md and ablation_summary.json — but not the gate report that
BUILD_PLAN P2 requires and `check_ablation.py` re-checks. This script closes that
gap, and only that gap: it derives, it does not decide.

    gate_fail_rate    = 1 - max(spec_adherence over the 6 FRONTIER cells)
    escalation_needed = gate_fail_rate < 0.20

"Frontier" means a hosted frontier family (openai / google / anthropic). The
local prompted base model (hf:...) is the ablation's third row — it is evidence
that a small base model cannot already do the task, NOT a ceiling measurement —
so it is excluded from the max and recorded separately. Every cell's
classification is written into the report, so the split a grader is asked to
trust is auditable instead of implicit.

Nothing is invented here: `check_ablation.py` independently recomputes
gate_fail_rate from the same results JSONs and fails if this file disagrees, so
hand-editing the output is pointless. This script refuses to write at all if the
grid is not the 3x3 the gate is defined over, or if the cells disagree about the
rubric — a gate report over an invalid ablation would be worse than none.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import eval as E  # noqa: E402  (frozen harness: imported, never reimplemented)

PINNED_RUBRIC_HASH = "c0adb876c3e6c4c9"
ESCALATION_THRESHOLD = 0.20       # BUILD_PLAN P2
FRONTIER_PROVIDERS = {"openai", "google", "anthropic"}
LOCAL_PROVIDERS = {"hf"}
EXPECTED_FRONTIER_CELLS = 6       # 2 frontier families x 3 strategies
EXPECTED_STRATEGIES = {"zero_shot", "few_shot", "structured"}


def die(msg):
    print(f"FAIL {msg}", file=sys.stderr)
    return 1


def provider_of(spec):
    return spec.split(":", 1)[0] if ":" in spec else ""


def load_cells(out_dir):
    """[(path, results_dict), ...] for every *.results.json in the ablation dir."""
    cells = []
    for rp in sorted(out_dir.glob("*.results.json")):
        with open(rp) as f:
            cells.append((rp, json.load(f)))
    return cells


def classify(cells):
    """Split cells into frontier / base / unknown by their model spec's provider."""
    frontier, base, unknown = [], [], []
    for rp, res in cells:
        model = res.get("model")
        strategy = res.get("strategy")
        entry = {
            "model": model,
            "strategy": strategy,
            "provider": provider_of(model or ""),
            "spec_adherence": (res.get("metrics") or {}).get("spec_adherence"),
            "results_json": rp.name,
        }
        if entry["provider"] in FRONTIER_PROVIDERS:
            frontier.append(entry)
        elif entry["provider"] in LOCAL_PROVIDERS:
            base.append(entry)
        else:
            unknown.append(entry)
    return frontier, base, unknown


def build_report(out_dir):
    """Returns (report_dict, None) or (None, error_message)."""
    cells = load_cells(out_dir)
    if not cells:
        return None, f"no *.results.json in {out_dir} — run run_ablation.py first"

    for rp, res in cells:
        if not isinstance(res.get("model"), str) or not isinstance(res.get("strategy"), str):
            return None, f"{rp.name}: missing 'model'/'strategy' string fields"

    hashes = {res.get("rubric_hash") for _, res in cells}
    if hashes != {PINNED_RUBRIC_HASH}:
        return None, (f"the cells were scored under rubric hash(es) "
                      f"{sorted(h for h in hashes if h)}, not a single "
                      f"{PINNED_RUBRIC_HASH!r} — a mixed-rubric ablation has no "
                      f"gate to report")
    if E.rubric_hash() != PINNED_RUBRIC_HASH:
        return None, (f"eval.rubric_hash() == {E.rubric_hash()!r} but this script "
                      f"pins {PINNED_RUBRIC_HASH!r} — the rubric drifted; STOP")

    frontier, base, unknown = classify(cells)
    if unknown:
        return None, (f"cell(s) from a provider that is neither a frontier family "
                      f"{sorted(FRONTIER_PROVIDERS)} nor the local base "
                      f"{sorted(LOCAL_PROVIDERS)}: "
                      f"{[(c['model'], c['strategy']) for c in unknown]}")
    if len(frontier) != EXPECTED_FRONTIER_CELLS:
        return None, (f"expected {EXPECTED_FRONTIER_CELLS} frontier cells "
                      f"(2 families x 3 strategies), found {len(frontier)}: "
                      f"{[(c['model'], c['strategy']) for c in frontier]}")
    families = sorted({c["provider"] for c in frontier})
    if len(families) < 2:
        return None, (f"the gate is defined over >= 2 DIFFERENT frontier families; "
                      f"found only {families}")
    if not base:
        return None, ("no local base-model (hf:...) cells — BUILD_PLAN P2 requires "
                      "the prompted base model as the third row")
    strategies = {c["strategy"] for c in frontier}
    if strategies != EXPECTED_STRATEGIES:
        return None, (f"frontier strategies are {sorted(strategies)}, expected "
                      f"{sorted(EXPECTED_STRATEGIES)}")
    missing_sa = [(c["model"], c["strategy"]) for c in frontier
                  if not isinstance(c["spec_adherence"], (int, float))]
    if missing_sa:
        return None, f"frontier cell(s) with no numeric spec_adherence: {missing_sa}"

    best = max(frontier, key=lambda c: c["spec_adherence"])
    gate_fail_rate = 1.0 - float(best["spec_adherence"])
    escalation_needed = gate_fail_rate < ESCALATION_THRESHOLD

    return {
        "rubric_hash": PINNED_RUBRIC_HASH,
        "rubric_version": E.RUBRIC_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "ablation_dir": str(out_dir),
        "definition": ("gate_fail_rate = 1 - max(spec_adherence over the frontier "
                       "cells); escalation_needed = gate_fail_rate < "
                       f"{ESCALATION_THRESHOLD}"),
        "escalation_threshold": ESCALATION_THRESHOLD,
        "gate_fail_rate": gate_fail_rate,
        "escalation_needed": escalation_needed,
        "max_spec_adherence": float(best["spec_adherence"]),
        "best_frontier_cell": {"model": best["model"], "strategy": best["strategy"]},
        "frontier_providers": families,
        "n_frontier_cells": len(frontier),
        "n_base_cells": len(base),
        # the classification a grader is asked to trust, written down
        "frontier_cells": sorted(frontier, key=lambda c: (c["model"], c["strategy"])),
        "base_cells": sorted(base, key=lambda c: (c["model"], c["strategy"])),
    }, None


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Compute the P2 / GATE G1 gate_report.json from a committed "
                    "ablation directory")
    ap.add_argument("ablation_dir", nargs="?", default="runs/ablation",
                    help="directory holding the ablation's *.results.json "
                         "(default: runs/ablation)")
    ap.add_argument("--out", default=None,
                    help="where to write the gate report (default: "
                         "<ablation_dir>/gate_report.json). A directory is "
                         "accepted and gate_report.json is written inside it.")
    args = ap.parse_args(argv)

    out_dir = Path(args.ablation_dir)
    if not out_dir.is_dir():
        return die(f"{out_dir} does not exist or is not a directory")

    report, err = build_report(out_dir)
    if err:
        return die(err)

    dest = Path(args.out) if args.out else out_dir / "gate_report.json"
    if dest.is_dir():
        dest = dest / "gate_report.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    best = report["best_frontier_cell"]
    print(f"  ok   {report['n_frontier_cells']} frontier cells "
          f"({', '.join(report['frontier_providers'])}), "
          f"{report['n_base_cells']} local base cells excluded from the max")
    print(f"  ok   best frontier cell: {best['model']} x {best['strategy']} at "
          f"spec_adherence {report['max_spec_adherence']:.4f}")
    print(f"\nPASS gate_fail_rate {report['gate_fail_rate']:.4f}, "
          f"escalation_needed={report['escalation_needed']} "
          f"(threshold {ESCALATION_THRESHOLD}) -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
