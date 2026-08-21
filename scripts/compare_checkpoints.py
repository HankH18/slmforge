#!/usr/bin/env python3
"""Paired comparison of two committed eval cells, on the same scenarios.

Two checkpoints scored on the same eval set are not two independent samples;
they are one sample measured twice. Comparing their headline rates by eye throws
away the pairing and invites the reading this project keeps running into: "v1
scored 7/69 on robustness and v2 scored 6/69, so v2 is ten times worse on the
delta." Those are the same scenario set differing by one row.

So this reports two things, both paired:

* **spec_pass** — McNemar's exact test over the discordant pairs. Only the rows
  where the two checkpoints disagree carry information; the rest cancel.
* **banned-letter occurrences** — a sign test and a bootstrap CI on the median
  difference, counted with `constraints.checkers._fold`, the same normalisation
  the checker applies.

The pair of tests is the point. A cliff metric can be blind to a difference the
continuous measure resolves overwhelmingly, and saying so requires running both
on the same rows.

    python scripts/compare_checkpoints.py \\
        --a runs/mvp/hf_HankH18_..._b97eb115..__zero_shot.transcripts.jsonl \\
        --b runs/mvp_v2/hf_runs_train_v2_merged__zero_shot.transcripts.jsonl \\
        --label-a v1 --label-b v2
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from math import comb
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from constraints import checkers  # noqa: E402

BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 7


class CompareError(Exception):
    pass


def read_jsonl(path: Path) -> List[Mapping[str, Any]]:
    if not path.exists():
        raise CompareError(f"{path} does not exist")
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def two_sided_sign_p(neg: int, pos: int) -> float:
    """Exact two-sided binomial p on the discordant/non-tied pairs."""
    n = neg + pos
    if n == 0:
        return 1.0
    k = min(neg, pos)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def banned_letter_occurrences(output: str,
                              scenario: Mapping[str, Any],
                              instances: Mapping[str, Mapping[str, Any]]) -> Optional[int]:
    """Total occurrences of this scenario's banned letter(s), or None if it has none."""
    letters = [instances[c]["params"]["letter"]
               for c in scenario["constraints"]
               if instances[c]["type"] == "banned_letter"]
    if not letters:
        return None
    folded, _ = checkers._fold(output)
    return sum(folded.count(letter.casefold()) for letter in letters)


def compare(rows_a: Sequence[Mapping[str, Any]],
            rows_b: Sequence[Mapping[str, Any]],
            eval_index: Mapping[str, Mapping[str, Any]],
            instances: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    a = {r["scenario_id"]: r for r in rows_a}
    b = {r["scenario_id"]: r for r in rows_b}
    shared = sorted(set(a) & set(b) & set(eval_index))
    if not shared:
        raise CompareError("the two cells share no scenario -- there is nothing "
                           "to pair")

    def subset(pressure: Optional[str]) -> Tuple[int, int, int]:
        """(a-only-pass, b-only-pass, paired) over rows both cells scored."""
        a_only = b_only = paired = 0
        for sid in shared:
            if pressure and eval_index[sid].get("pressure") != pressure:
                continue
            pa, pb = a[sid].get("spec_pass"), b[sid].get("spec_pass")
            if pa is None or pb is None:
                continue
            paired += 1
            if pa and not pb:
                a_only += 1
            elif pb and not pa:
                b_only += 1
        return a_only, b_only, paired

    spec = {}
    for name, pressure in (("overall", None), ("adversarial", "adversarial")):
        a_only, b_only, paired = subset(pressure)
        spec[name] = {
            "paired_rows": paired,
            "a_only_pass": a_only,
            "b_only_pass": b_only,
            "discordant": a_only + b_only,
            "mcnemar_exact_p": round(two_sided_sign_p(a_only, b_only), 4),
        }

    pairs: List[Tuple[int, int]] = []
    for sid in shared:
        oa, ob = a[sid].get("output"), b[sid].get("output")
        if not isinstance(oa, str) or not isinstance(ob, str):
            continue
        ca = banned_letter_occurrences(oa, eval_index[sid], instances)
        cb = banned_letter_occurrences(ob, eval_index[sid], instances)
        if ca is None or cb is None:
            continue
        pairs.append((ca, cb))

    occ: Dict[str, Any] = {"paired_rows": len(pairs)}
    if pairs:
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        diffs = [y - x for x, y in pairs]
        lower = sum(1 for d in diffs if d < 0)
        higher = sum(1 for d in diffs if d > 0)
        rng = random.Random(BOOTSTRAP_SEED)
        boots = []
        for _ in range(BOOTSTRAP_RESAMPLES):
            idx = [rng.randrange(len(pairs)) for _ in pairs]
            boots.append(statistics.median([ys[i] for i in idx])
                         - statistics.median([xs[i] for i in idx]))
        boots.sort()
        lo = boots[int(0.025 * BOOTSTRAP_RESAMPLES)]
        hi = boots[int(0.975 * BOOTSTRAP_RESAMPLES) - 1]
        occ.update({
            "a_median": statistics.median(xs),
            "b_median": statistics.median(ys),
            "b_lower_on": lower,
            "b_higher_on": higher,
            "tied": len(diffs) - lower - higher,
            "sign_test_p": two_sided_sign_p(lower, higher),
            "median_difference_ci95": [lo, hi],
        })
    return {"shared_scenarios": len(shared), "spec_pass": spec,
            "banned_letter_occurrences": occ}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--a", required=True, help="transcripts JSONL of checkpoint A")
    ap.add_argument("--b", required=True, help="transcripts JSONL of checkpoint B")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--eval-set", default="data/scenarios/corpus.jsonl")
    ap.add_argument("--instances", default="data/scenarios/constraint_instances.jsonl")
    ap.add_argument("--json", action="store_true", help="emit the result as JSON")
    args = ap.parse_args(argv)

    try:
        eval_index = {s["scenario_id"]: s for s in read_jsonl(Path(args.eval_set))}
        instances = {c["id"]: c for c in read_jsonl(Path(args.instances))}
        got = compare(read_jsonl(Path(args.a)), read_jsonl(Path(args.b)),
                      eval_index, instances)
    except CompareError as exc:
        print(f"FAIL compare_checkpoints: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(got, indent=2))
        return 0

    A, B = args.label_a, args.label_b
    print(f"paired on {got['shared_scenarios']} shared scenarios\n")
    print("spec_pass (the graded cliff metric), McNemar exact:")
    for name, s in got["spec_pass"].items():
        print(f"  {name:<12} {A}-only-pass {s['a_only_pass']}, {B}-only-pass "
              f"{s['b_only_pass']}, of {s['paired_rows']} paired   p = "
              f"{s['mcnemar_exact_p']:.3f}")
    occ = got["banned_letter_occurrences"]
    print("\nbanned-letter occurrences (the continuous measure):")
    if occ["paired_rows"]:
        lo, hi = occ["median_difference_ci95"]
        print(f"  median {A} {occ['a_median']} -> {B} {occ['b_median']} on "
              f"{occ['paired_rows']} paired rows")
        print(f"  {B} lower on {occ['b_lower_on']}, higher on "
              f"{occ['b_higher_on']}, tied on {occ['tied']}")
        print(f"  sign test p = {occ['sign_test_p']:.2e}")
        print(f"  bootstrap 95% CI on median({B}) - median({A}): "
              f"[{lo:.1f}, {hi:.1f}]")
    else:
        print("  no scenario in common carries a banned_letter constraint")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
