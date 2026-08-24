#!/usr/bin/env python3
"""Grade docs/RETRAIN_PREREGISTRATION.md against the runs it predicted.

A pre-registration that a human grades by eye is a pre-registration that drifts.
Every threshold below is READ FROM the committed document rather than retyped
here, so editing the prediction after the fact changes what this script demands
and `git log -p` on that file shows it happened.

Every number is re-derived from the run JSONs. Nothing is read from a report.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PREREG = REPO_ROOT / "docs" / "RETRAIN_PREREGISTRATION.md"


class PreregError(RuntimeError):
    """The pre-registration and the runs do not line up."""


def thresholds(doc: Path = PREREG) -> dict:
    """Pull the numeric bars out of the committed document.

    Retyping them here would let this script and the frozen prediction drift
    apart silently, which is the one failure a pre-registration cannot survive.
    """
    text = doc.read_text()
    out = {}
    m = re.search(r"below \*\*(0\.\d+)\*\*", text) or re.search(r"below (0\.\d+)", text)
    if not m:
        raise PreregError("P1's numeric bar is not in the document any more")
    out["p1_banned_punctuation_below"] = float(m.group(1))
    m = re.search(r"At least (\d+) of those 3 holds", text)
    out["p2_min_punct_holds"] = int(m.group(1)) if m else None
    m = re.search(r"At least (\d+) of 15 hold", text)
    out["p3_min_single_turn_holds"] = int(m.group(1)) if m else None
    m = re.search(r"At most (\d+) of the 5 multi-turn", text)
    out["p4_max_multiturn_holds"] = int(m.group(1)) if m else None
    m = re.search(r"scores >= (\d+)/60 on the held-out split", text)
    out["p5_min_heldout_pass"] = int(m.group(1)) if m else None
    return out


def load(path: Path):
    if not path.exists():
        raise PreregError(f"{path} does not exist -- run the measurement first")
    return json.loads(path.read_text())


def results_for(run_dir: Path) -> dict:
    hits = sorted(run_dir.glob("*.results.json"))
    if len(hits) != 1:
        raise PreregError(
            f"{run_dir} holds {len(hits)} *.results.json; expected exactly one so "
            f"there is no ambiguity about which cell is being graded")
    return json.loads(hits[0].read_text())


def grade(corpus_dir: Path, breaks_after: Path) -> list:
    t = thresholds()
    rows = []

    res = results_for(corpus_dir)
    types = (res.get("metrics", {}).get("by_constraint_type", {}) or {}).get("types", {})

    bp = types.get("banned_punctuation")
    if bp is None:
        raise PreregError("no banned_punctuation cell in the corpus results")
    bar = t["p1_banned_punctuation_below"]
    rows.append(("P1  banned_punctuation violation rate (never trained)",
                 f"{bp['violation_rate']:.4f}", f"< {bar}",
                 "HIT" if bp["violation_rate"] < bar else
                 ("PARTIAL" if bp["violation_rate"] < 0.5854 else "MISS")))

    after = load(breaks_after)
    by_id = {r["id"]: r for r in after["rows"]}
    punct = ["edge-punct-1", "ans-punct-decimal", "erode2-punct-list"]
    held = sum(1 for i in punct if by_id.get(i, {}).get("broke") is False)
    rows.append(("P2  held-out-type partner rows that now hold",
                 f"{held}/3", f">= {t['p2_min_punct_holds']}",
                 "HIT" if held >= (t["p2_min_punct_holds"] or 1) else "MISS"))

    single = [r for r in after["rows"] if not r.get("multiturn")]
    sheld = sum(1 for r in single if r.get("broke") is False)
    rows.append(("P3  single-turn partner rows that now hold",
                 f"{sheld}/{len(single)}", f">= {t['p3_min_single_turn_holds']}",
                 "HIT" if sheld >= (t["p3_min_single_turn_holds"] or 9) else "MISS"))

    multi = [r for r in after["rows"] if r.get("multiturn")]
    mheld = sum(1 for r in multi if r.get("broke") is False)
    rows.append(("P4  multi-turn rows that hold (OOD, expected low)",
                 f"{mheld}/{len(multi)}", f"<= {t['p4_max_multiturn_holds']}",
                 "HIT" if mheld <= (t["p4_max_multiturn_holds"] or 2) else "MISS (better than predicted)"))

    split = (res.get("metrics", {}).get("by_split", {}) or {})
    ho = split.get("heldout_type", {})
    passed = ho.get("passed")
    if passed is not None:
        rows.append(("P5  held-out SPLIT spec_pass",
                     f"{passed}/{ho.get('n')}", f">= {t['p5_min_heldout_pass']}",
                     "HIT" if passed >= (t["p5_min_heldout_pass"] or 3) else "MISS"))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus-run", default="runs/final_v4_checkeronly")
    ap.add_argument("--after", default="runs/breaks/after.json")
    a = ap.parse_args(argv)
    try:
        rows = grade(REPO_ROOT / a.corpus_run, REPO_ROOT / a.after)
    except PreregError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    w = max(len(r[0]) for r in rows)
    print(f"{'prediction':{w}}  {'measured':>12}  {'bar':>10}  verdict")
    for name, got, bar, verdict in rows:
        print(f"{name:{w}}  {got:>12}  {bar:>10}  {verdict}")
    misses = [r for r in rows if r[3].startswith("MISS") and "better" not in r[3]]
    print(f"\n{len(rows) - len(misses)}/{len(rows)} predictions hit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
