#!/usr/bin/env python3
"""Does holding a constraint get harder when other constraints share the prompt?

Stretch Ladder rung 3 asks for "composed behavior -- instill a second,
potentially competing constraint and show the model holds both at once". This
script answers the half of that question the committed runs can answer, and it
is written so the half they CANNOT answer is impossible to mistake for the half
they can. See docs/COMPOSITION.md for the reading of the requirement; this
docstring covers the measurement.

WHAT IS MEASURED. Every scenario in data/scenarios/corpus.jsonl carries k = 2,
3 or 4 simultaneous constraints, so a per-constraint hold rate can be cut by k.
If holding a single constraint were unaffected by its neighbours, that cut would
be flat. It is not flat, and the whole difficulty is that k is NOT randomly
assigned: adversarial pressure rises from 17% of k=2 scenarios to 61% of k=4
scenarios, so a naive "hold rate falls with k" reading is partly a restatement
of "hold rate falls under adversarial pressure". Every trend number here is
therefore reported three ways -- unstratified, stratified by pressure, and
stratified by pressure x split -- from ONE test implementation.

THE TEST, chosen once and used everywhere: a **stratified exact permutation
test of the Cochran-Armitage linear trend statistic**

    T = sum over strata s, units i in s of (k_i - mean_k(s)) * hold_i

with the hold outcomes permuted WITHIN each stratum. Two-sided Monte-Carlo p is
(1 + #{|T*| >= |T_obs|}) / (1 + B), which is a valid p-value at any B.

Why this one:

* Cochran-Armitage is the standard test for a trend in a binary outcome across
  ORDERED groups, which is exactly the k = 2, 3, 4 shape. A 3x2 chi-square would
  throw away the ordering; three pairwise Fisher tests would spend the power on
  multiplicity.
* The PERMUTATION version rather than the usual normal approximation, because
  several cells here are tiny (base `banned_words` at k=2 is 2 scenarios) and the
  asymptotic variance is not trustworthy there. Permutation is exact under the
  null of exchangeability at any n.
* STRATIFIED, because the confound is real and large. Permuting within a stratum
  holds the stratum's pressure/split mix fixed, so the stratified p asks "does k
  still predict holding once pressure is held constant?" -- which is the question
  the confound forces.
* The unstratified call is the same function with one stratum, so there is one
  implementation of the test and no chance of two drifting apart.

Effect size is reported separately from the test, as the k=4 minus k<=3
difference in hold rate with a **Newcombe hybrid-score 95% CI**, which is built
out of `curve_report.wilson_interval` -- the interval this repo already uses --
rather than a second interval implementation. The CI is unstratified; the
per-stratum rates are printed beside it so a reader can see the adjustment by
eye.

THE UNIT OF ANALYSIS, and why it is not the check. 27 scenarios carry two
`banned_words` constraints and one carries three, so per-CHECK counts are
clustered: those scenarios contribute 2-3 correlated Bernoullis from a single
response. A test run over checks would treat them as independent and overstate
its own significance. So every TEST here runs on the SCENARIO as the unit -- a
scenario "holds type T" iff every type-T constraint in it passes -- while the
per-check table is still printed, because it is the table the finding was first
stated in and dropping it would hide the difference. Both are labelled.

QUARANTINE. A row is scored here if it has a usable `output`. Unlike
`scripts/curve_report.py`, a row whose JUDGE failed is NOT dropped: this script
measures programmatic checker verdicts only, and the judge takes no part in
them. On the committed runs that is exactly one row -- base `s-0103`, whose
judge failed three times and whose real 336-character output passes all four of
its checkers. `--exclude-quarantined` reproduces curve_report's convention
instead; the two conventions differ by that row and nothing else, which is
asserted rather than asserted-in-prose -- `quarantine_contrast` refuses to
return if the excluded rows are not exactly the quarantined ones.

NOTHING STORED IS TRUSTED, AND NOTHING IS TRUSTED TO BE STORED. Every verdict
is re-derived from (constraint, output); the transcript's own `checker_results`
entry is then required to AGREE with the re-derivation, and the number of
entries is required to equal the number of checks re-run. Agreement alone
catches a flipped verdict but is satisfied vacuously by a transcript whose
verdict block was deleted -- the tables come out identical while nothing holds
the transcript to them -- so the count is asserted too, and printed in the
default summary rather than only in `--json`.

WHAT `--check` BINDS. The generated block in section 4 of docs/COMPOSITION.md
is regenerated and compared. The figures the PROSE states around it are bound
as well: each numeral outside the block is matched by a pattern in
`PROSE_FACTS`, recomputed from this same report, and compared at the precision
the sentence itself writes. A numeral no pattern claims fails the check exactly
as loudly as a wrong one. This exists because it did not: an audit replaced the
headline permutation p (0.0515) with 0.9999 in the prose and `--check` exited 0,
and two of the document's wrong numbers lived in that same unbound region.

WHAT IS NOT MEASURED, stated here so it cannot be read in: nothing in this
script instils a NEW constraint type. Both cells are checkpoints that already
exist. The k cut measures whether composition degrades holding; it does not and
cannot show that a second constraint type was successfully TAUGHT.

Usage:

    python scripts/composition_report.py                  # summary to stdout
    python scripts/composition_report.py --markdown        # the report tables
    python scripts/composition_report.py --json            # machine-readable
    python scripts/composition_report.py --check docs/COMPOSITION.md
    python scripts/composition_report.py --check docs/COMPOSITION.md --show-facts
    python scripts/composition_report.py --selftest        # verify the statistics
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from itertools import permutations
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import gate_check as G                      # noqa: E402  (sha256_of, GateError)
from compare_checkpoints import two_sided_sign_p   # noqa: E402  (McNemar exact)
from curve_report import PLACES, wilson_interval   # noqa: E402  (the repo's CI)
from constraints import checkers            # noqa: E402
from scenarios import reachability as RB    # noqa: E402
import eval as E                            # noqa: E402  (sanitize: the run naming)

#: Monte-Carlo resamples for the permutation test, and the seed. The seed is the
#: one `compare_checkpoints.py` already uses for its bootstrap, so the two tools
#: are reproducible under the same number.
RESAMPLES = 20000
SEED = 7

#: Which configs/models.json keys name the two cells, and what to call them.
#: Reading the pins rather than the filenames means a moved pin FAILS here
#: instead of silently reporting a different checkpoint under the same label.
CELL_PINS: Tuple[Tuple[str, str], ...] = (("base", "base"), ("tuned", "tuned"))

#: A (model, type) cell with fewer scenarios than this is not tabulated at all;
#: a cell WITH variance but a floored or ceilinged outcome IS tabulated and then
#: explicitly refused by the degeneracy branch of the test, because "the test
#: could not run" and "the test found nothing" are different statements and this
#: project has confused them before.
MIN_CELL_SCENARIOS = 2

#: The stratifications the trend test is run under, in report order.
TREND_VARIANTS: Tuple[str, ...] = ("unstratified", "by_pressure",
                                   "by_pressure_split")

#: The significance level the multiplicity arithmetic in the document quotes.
ALPHA = 0.05

#: The confidence level every interval here is built at (curve_report's Wilson
#: interval, and the Newcombe difference built out of it).
CI_LEVEL = 95

#: The cut the tension summaries count against ("how many ratios fall below").
BELOW_THRESHOLD = 0.9

MARKER_BEGIN = "<!-- BEGIN composition_report -->"
MARKER_END = "<!-- END composition_report -->"


class CompositionError(Exception):
    """A number that cannot be re-derived is not a number."""


# ------------------------------------------------------------------ loading
def read_jsonl(path: Path) -> List[Mapping[str, Any]]:
    if not path.exists():
        raise CompositionError(f"{path} does not exist")
    rows = []
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise CompositionError(f"{path}:{lineno}: {exc}") from exc
    return rows


def resolve_cells(run_dir: Path, models_json: Path,
                  strategy: str) -> List[Dict[str, Any]]:
    """The (label, model spec, transcripts path) triples, FROM THE PINS.

    Deliberately not "the two files in the directory": the label `base` has to
    mean the model `configs/models.json` calls base, and a pin that moves has to
    break this script loudly rather than relabel a different checkpoint.
    """
    if not models_json.exists():
        raise CompositionError(f"{models_json} is missing; the cell labels come "
                               f"from its pins, not from the filenames")
    cfg = json.loads(models_json.read_text())
    cells = []
    for key, label in CELL_PINS:
        spec = cfg.get(key)
        if not isinstance(spec, str) or not spec:
            raise CompositionError(
                f"configs/models.json[{key!r}] == {spec!r}; expected a model spec")
        full = spec if ":" in spec.split("@")[0] else f"hf:{spec}"
        stem = f"{E.sanitize(full)}__{strategy}"
        path = run_dir / f"{stem}.transcripts.jsonl"
        if not path.exists():
            raise CompositionError(
                f"{path} is missing. configs/models.json[{key!r}] pins {full!r} "
                f"and the harness names its transcripts after the model, so "
                f"either {run_dir} does not hold that cell or the pin moved "
                f"without the run being redone.")
        cells.append({"key": key, "label": label, "model": full, "path": path})
    return cells


# --------------------------------------------------------------- statistics
def rate(num: int, den: int) -> Optional[float]:
    return round(float(Fraction(num, den)), PLACES) if den else None


def newcombe_difference_ci(s1: int, n1: int, s2: int, n2: int
                           ) -> Optional[List[float]]:
    """95% CI for p2 - p1 by Newcombe's hybrid-score (method 10) interval.

    Built ON TOP of `curve_report.wilson_interval` on purpose. The Wald interval
    for a difference is useless at the rates in this report -- several cells are
    at 0/17 or 25/25, where Wald has zero width and is centred on a boundary.
    Newcombe's method takes each arm's Wilson interval and combines them, so it
    behaves at the boundary and it inherits, rather than duplicates, the
    interval this repo already committed to.
    """
    if n1 <= 0 or n2 <= 0:
        return None
    p1, p2 = s1 / n1, s2 / n2
    l1, u1 = wilson_interval(s1, n1)
    l2, u2 = wilson_interval(s2, n2)
    lower = (p2 - p1) - ((p2 - l2) ** 2 + (u1 - p1) ** 2) ** 0.5
    upper = (p2 - p1) + ((u2 - p2) ** 2 + (p1 - l1) ** 2) ** 0.5
    return [round(max(-1.0, lower), PLACES), round(min(1.0, upper), PLACES)]


#: How many of the sampled permutations get their stratum invariant checked.
#: Permuting ACROSS strata instead of within them silently un-does the confound
#: adjustment -- the numbers still print, and they are the unadjusted numbers
#: wearing the adjusted column's label. That is the one defect in this file that
#: would be invisible in the output, so it is caught at runtime rather than left
#: to a test. 100 draws is enough: a leak has to preserve every stratum's hold
#: count 100 times running to escape.
INVARIANT_RESAMPLES = 100


def _strata_preserved(units: Sequence[Tuple[Any, int, int]],
                      labels: Sequence[int]) -> bool:
    """Does `labels` hold each stratum's outcomes exactly, only reordered?"""
    want: Dict[Any, Counter] = defaultdict(Counter)
    got: Dict[Any, Counter] = defaultdict(Counter)
    for (stratum, _, y), label in zip(units, labels):
        want[stratum][y] += 1
        got[stratum][label] += 1
    return want == got


def _trend_statistic(units: Sequence[Tuple[Any, int, int]]) -> float:
    """Cochran-Armitage T = sum_s sum_i (k_i - mean_k(s)) * y_i."""
    by_stratum: Dict[Any, List[Tuple[int, int]]] = defaultdict(list)
    for stratum, k, y in units:
        by_stratum[stratum].append((k, y))
    total = 0.0
    for members in by_stratum.values():
        mean_k = sum(k for k, _ in members) / len(members)
        total += sum((k - mean_k) * y for k, y in members)
    return total


def permutation_trend_test(units: Sequence[Tuple[Any, int, int]],
                           resamples: int = RESAMPLES,
                           seed: int = SEED,
                           exact_below: int = 0) -> Dict[str, Any]:
    """Stratified exact permutation test for a trend in `y` across ordered `k`.

    `units` is a sequence of (stratum, k, hold) with hold in {0, 1}. The hold
    labels are permuted within each stratum; the two-sided p is

        (1 + #{|T*| >= |T_obs|}) / (1 + B)

    which never returns 0 and is valid for any B. DEGENERACY IS REPORTED, NOT
    HIDDEN: if every hold is identical, or every stratum has a single k, T is
    identically 0 under every permutation and no trend is testable. Such a cell
    comes back with `degenerate: true` and p = 1.0, and callers must print the
    reason rather than the number.

    With `exact_below` set and n at or under it, every within-stratum
    permutation is enumerated instead of sampled, which makes the p exact rather
    than Monte-Carlo. `--selftest` uses that to check the sampler.
    """
    n = len(units)
    ys = [y for _, _, y in units]
    ks = [k for _, k, _ in units]
    strata = sorted({s for s, _, _ in units}, key=repr)
    varying = any(len({k for s2, k, _ in units if s2 == s}) > 1 for s in strata)
    if n == 0 or len(set(ys)) < 2 or not varying:
        return {"n": n, "statistic": 0.0, "p": 1.0, "degenerate": True,
                "resamples": 0, "exact": False, "strata": len(strata),
                "reason": ("no scenario" if n == 0 else
                           "every outcome identical" if len(set(ys)) < 2 else
                           "no stratum contains more than one k")}

    observed = _trend_statistic(units)
    index_by_stratum: Dict[Any, List[int]] = defaultdict(list)
    for i, (s, _, _) in enumerate(units):
        index_by_stratum[s].append(i)

    def statistic_from(labels: Sequence[int]) -> float:
        return _trend_statistic([(units[i][0], ks[i], labels[i])
                                 for i in range(n)])

    if exact_below and n <= exact_below:
        # Complete enumeration of within-stratum relabelings. Only ever used by
        # --selftest, on deliberately tiny inputs.
        groups = [sorted(idx) for _, idx in sorted(index_by_stratum.items(),
                                                   key=lambda kv: repr(kv[0]))]
        arrangements = [set(permutations([ys[i] for i in g])) for g in groups]
        count = total = 0

        def walk(depth: int, chosen: List[Tuple[int, ...]]) -> None:
            nonlocal count, total
            if depth == len(groups):
                labels = [0] * n
                for g, assign in zip(groups, chosen):
                    for i, v in zip(g, assign):
                        labels[i] = v
                total += 1
                if abs(statistic_from(labels)) >= abs(observed) - 1e-12:
                    count += 1
                return
            for assign in sorted(arrangements[depth]):
                walk(depth + 1, chosen + [assign])

        walk(0, [])
        return {"n": n, "statistic": round(observed, 6), "p": round(count / total, 6),
                "degenerate": False, "resamples": total, "exact": True,
                "strata": len(strata)}

    rng = random.Random(seed)
    hits = 0
    labels = list(ys)
    for draw in range(resamples):
        for idx in index_by_stratum.values():
            pool = [labels[i] for i in idx]
            rng.shuffle(pool)
            for i, v in zip(idx, pool):
                labels[i] = v
        if draw < INVARIANT_RESAMPLES and not _strata_preserved(units, labels):
            raise CompositionError(
                "a permutation moved outcomes BETWEEN strata. The stratified "
                "p-value is only a confound adjustment because the relabeling "
                "stays inside a stratum; a draw that crosses one produces the "
                "unstratified null under the stratified column's name.")
        if abs(statistic_from(labels)) >= abs(observed) - 1e-12:
            hits += 1
    return {"n": n, "statistic": round(observed, 6),
            "p": round((1 + hits) / (1 + resamples), 6), "degenerate": False,
            "resamples": resamples, "exact": False, "strata": len(strata)}


def holm_adjust(pvalues: Sequence[float]) -> List[float]:
    """Holm-Bonferroni step-down adjusted p-values, in the input order.

    Eight trend tests are run (two models x four types with any variance). At
    alpha = 0.05 the family-wise error of eight independent tests is 34%, which
    is exactly the machinery that turns noise into a finding. Holm rather than
    Bonferroni because it is uniformly more powerful and needs no independence
    assumption beyond what Bonferroni already needs.
    """
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    m = len(pvalues)
    out = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        adjusted = min(1.0, (m - rank) * pvalues[i])
        running = max(running, adjusted)
        out[i] = round(running, 6)
    return out


# ------------------------------------------------------------- the tabulation
def require_transcript_coverage(name: str, agreeing: int, total: int) -> int:
    """Every re-derived check must have had a stored verdict to agree WITH.

    Requiring agreement catches a FLIPPED verdict. It does not catch a DELETED
    one: a transcript whose `checker_results` block is stripped agrees with the
    re-derivation vacuously, the tables come out identical, and the cross-check
    silently degrades to zero coverage. So the count is asserted, not just the
    verdicts: `checks_agreeing_with_transcript` must equal the number of checks
    re-run, and the count is printed in the default summary rather than hiding
    in `--json`.
    """
    if agreeing != total:
        raise CompositionError(
            f"{name}: {agreeing} of {total} re-derived checks have a stored "
            f"verdict to agree with; {total - agreeing} have none. A missing "
            f"`checker_results` entry is not a passing cross-check, it is an "
            f"absent one -- the tables would come out unchanged while nothing "
            f"held the transcript to them, so this stops instead.")
    return total


def score_cells(cells: Sequence[Mapping[str, Any]],
                corpus: Mapping[str, Mapping[str, Any]],
                instances: Mapping[str, Mapping[str, Any]],
                exclude_quarantined: bool) -> Dict[str, Any]:
    """Re-run every checker from (constraint, output). Nothing stored is trusted.

    The row's own `checker_results` block is deliberately ignored: it is the
    thing being re-derived. `constraints.checkers.check` is the same function the
    harness and `gate_check` call, so a verdict here cannot drift from the
    verdict the headline was computed with.
    """
    out: Dict[str, Any] = {}
    for cell in cells:
        rows = read_jsonl(cell["path"])
        ids = {r["scenario_id"] for r in rows}
        if ids != set(corpus):
            missing = sorted(set(corpus) - ids)[:5]
            extra = sorted(ids - set(corpus))[:5]
            raise CompositionError(
                f"{cell['path'].name} does not score the eval set: "
                f"{len(ids)} rows against {len(corpus)} scenarios "
                f"(missing {missing}, unexpected {extra}). Every cut below "
                f"assumes the two cells cover the same scenarios.")

        checks: Dict[Tuple[str, int], List[int]] = defaultdict(lambda: [0, 0])
        scen_hold: Dict[str, Dict[str, bool]] = defaultdict(dict)
        quarantined: List[str] = []
        quarantined_rows: List[Dict[str, Any]] = []
        skipped: List[str] = []
        verified_against_transcript = 0
        total_checks = 0
        for row in rows:
            sid = row["scenario_id"]
            output = row.get("output")
            quarantine = (row.get("judge_error") is not None
                          or row.get("generator_error") is not None)
            if quarantine:
                quarantined.append(sid)
            if not isinstance(output, str) or not output.strip():
                skipped.append(sid)
                continue
            if quarantine and exclude_quarantined:
                skipped.append(sid)
                continue
            per_type: Dict[str, List[bool]] = defaultdict(list)
            scenario = corpus[sid]
            stored = {c["constraint_id"]: bool(c["pass"])
                      for c in (row.get("checker_results") or [])
                      if isinstance(c, dict) and "constraint_id" in c}
            for cid in scenario["constraints"]:
                constraint = instances[cid]
                passed = bool(checkers.check(constraint, output)["pass"])
                # Bidirectional: the verdict is RE-DERIVED, and the stored one is
                # then required to agree. Re-deriving without checking would let a
                # changed checker rewrite history silently; trusting the stored
                # verdict would not be a re-derivation at all.
                if cid in stored and stored[cid] != passed:
                    raise CompositionError(
                        f"{cell['path'].name} {sid} {cid}: the transcript "
                        f"records pass={stored[cid]} and re-running "
                        f"constraints.checkers.check on the same output gives "
                        f"pass={passed}. Either the checker changed since the "
                        f"run or the transcript was edited; both make every "
                        f"number below unsafe, so this stops rather than "
                        f"picking a side.")
                verified_against_transcript += int(cid in stored)
                total_checks += 1
                per_type[constraint["type"]].append(passed)
                cell_key = (constraint["type"], scenario["k"])
                checks[cell_key][0] += 1
                checks[cell_key][1] += int(passed)
            for ctype, verdicts in per_type.items():
                scen_hold[sid][ctype] = all(verdicts)
            if quarantine:
                quarantined_rows.append({
                    "scenario_id": sid,
                    "output_chars": len(output),
                    "checks": len(scenario["constraints"]),
                    "checks_held": sum(sum(v) for v in per_type.values()),
                    "judge_error": row.get("judge_error") is not None,
                    "generator_error": row.get("generator_error") is not None,
                })
        require_transcript_coverage(cell["path"].name, verified_against_transcript,
                                    total_checks)
        out[cell["label"]] = {
            "model": cell["model"],
            "transcripts": cell["path"].name,
            "transcripts_sha256": G.sha256_of(cell["path"]),
            "n_rows": len(rows),
            "n_scored": len(rows) - len(skipped),
            "checks_total": total_checks,
            "checks_agreeing_with_transcript": verified_against_transcript,
            "quarantined": sorted(quarantined),
            "quarantined_rows": sorted(quarantined_rows,
                                       key=lambda r: r["scenario_id"]),
            "skipped": sorted(skipped),
            "checks": {f"{t}|{k}": {"checks": v[0], "held": v[1]}
                       for (t, k), v in sorted(checks.items())},
            "_scenario_hold": dict(scen_hold),
        }
    return out


def type_k_table(scored: Mapping[str, Any], corpus: Mapping[str, Any],
                 types: Sequence[str], ks: Sequence[int], unit: str
                 ) -> Dict[str, Any]:
    """{label: {type: {k: {n, held, rate, ci95}}}} at check or scenario level."""
    table: Dict[str, Any] = {}
    for label, cell in scored.items():
        per_type: Dict[str, Any] = {}
        for ctype in types:
            per_k: Dict[str, Any] = {}
            for k in ks:
                if unit == "check":
                    entry = cell["checks"].get(f"{ctype}|{k}")
                    n, held = (entry["checks"], entry["held"]) if entry else (0, 0)
                else:
                    sids = [sid for sid, h in cell["_scenario_hold"].items()
                            if ctype in h and corpus[sid]["k"] == k]
                    n = len(sids)
                    held = sum(cell["_scenario_hold"][sid][ctype] for sid in sids)
                if n == 0:
                    continue
                per_k[str(k)] = {"n": n, "held": held, "rate": rate(held, n),
                                 "ci95": wilson_interval(held, n)}
            if per_k:
                per_type[ctype] = per_k
        table[label] = per_type
    return table


def scenario_units(cell: Mapping[str, Any], corpus: Mapping[str, Any],
                   ctype: str) -> List[Dict[str, Any]]:
    units = []
    for sid, holds in cell["_scenario_hold"].items():
        if ctype not in holds:
            continue
        scenario = corpus[sid]
        units.append({"scenario_id": sid, "k": scenario["k"],
                      "pressure": scenario["pressure"], "split": scenario["split"],
                      "hold": int(holds[ctype])})
    return sorted(units, key=lambda u: u["scenario_id"])


def trend_tests(scored: Mapping[str, Any], corpus: Mapping[str, Any],
                types: Sequence[str], resamples: int,
                variants: Sequence[str] = TREND_VARIANTS) -> List[Dict[str, Any]]:
    """`variants` selects which stratifications to run; the default runs all
    three. `("unstratified",)` is the cheap call used to re-derive the
    `--exclude-quarantined` contrast quoted in the document -- same function,
    same seed, one third of the permutation work."""
    unknown = [v for v in variants if v not in TREND_VARIANTS]
    if unknown:
        raise CompositionError(f"unknown trend variant(s) {unknown}")
    results: List[Dict[str, Any]] = []
    for label, cell in scored.items():
        for ctype in types:
            units = scenario_units(cell, corpus, ctype)
            if len(units) < MIN_CELL_SCENARIOS:
                continue
            low = [u for u in units if u["k"] <= 3]
            high = [u for u in units if u["k"] == 4]
            s1, n1 = sum(u["hold"] for u in low), len(low)
            s2, n2 = sum(u["hold"] for u in high), len(high)
            keys = {"unstratified": lambda u: 0,
                    "by_pressure": lambda u: u["pressure"],
                    "by_pressure_split": lambda u: (u["pressure"], u["split"])}
            tests = {}
            for name in variants:
                key = keys[name]
                tests[name] = permutation_trend_test(
                    [(key(u), u["k"], u["hold"]) for u in units], resamples=resamples)
            strata_rates = {}
            for pressure in ("clean", "adversarial"):
                lo = [u for u in units if u["pressure"] == pressure and u["k"] <= 3]
                hi = [u for u in units if u["pressure"] == pressure and u["k"] == 4]
                strata_rates[pressure] = {
                    "k_le_3": {"n": len(lo), "held": sum(u["hold"] for u in lo),
                               "rate": rate(sum(u["hold"] for u in lo), len(lo))},
                    "k_eq_4": {"n": len(hi), "held": sum(u["hold"] for u in hi),
                               "rate": rate(sum(u["hold"] for u in hi), len(hi))},
                }
            results.append({
                "model": label, "type": ctype, "unit": "scenario",
                "n_scenarios": len(units),
                "k_le_3": {"n": n1, "held": s1, "rate": rate(s1, n1)},
                "k_eq_4": {"n": n2, "held": s2, "rate": rate(s2, n2)},
                "difference": (round(s2 / n2 - s1 / n1, PLACES)
                               if n1 and n2 else None),
                "difference_ci95": newcombe_difference_ci(s1, n1, s2, n2),
                "by_pressure": strata_rates,
                "tests": tests,
            })
    for name, field in (("unstratified", "holm_p_unstratified"),
                        ("by_pressure", "holm_p_by_pressure")):
        if name not in variants:
            continue
        live = [r for r in results if not r["tests"][name]["degenerate"]]
        for r, p in zip(live, holm_adjust([r["tests"][name]["p"] for r in live])):
            r[field] = p
    return results


def trend_multiplicity(trend_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """The family the Holm adjustment is actually taken over, and its FWER.

    Stated here rather than counted by hand in prose: the family size is
    whatever the degeneracy branch left with variance to test, and the
    family-wise error at alpha is 1 - (1 - alpha)^m on that same m.
    """
    live = [r for r in trend_rows
            if not r["tests"]["unstratified"]["degenerate"]]
    m = len(live)
    return {
        "cells_tabulated": len(trend_rows),
        "tests_with_variance": m,
        "not_testable": len(trend_rows) - m,
        "alpha": ALPHA,
        "family_wise_error": round(1.0 - (1.0 - ALPHA) ** m, 4),
    }


def interaction_tests(scored: Mapping[str, Any], corpus: Mapping[str, Any],
                      types: Sequence[str], resamples: int) -> Dict[str, Any]:
    """Does the tuned-minus-base advantage GROW with k? The reading, tested.

    "Base pays a composition cost and the tuned model does not" is an
    INTERACTION claim, and comparing two separate trend tests is not a test of
    it -- two p-values on opposite sides of 0.05 are not a difference. So the
    outcome here is the paired per-scenario contrast

        d = tuned holds the type - base holds the type    (in {-1, 0, +1})

    and the same Cochran-Armitage permutation statistic is run on it. The
    statistic is a weighted sum of the outcomes and never assumed the outcome
    was binary, so no second implementation appears. Pairing removes every
    scenario-level nuisance the two cells share -- topic, letter, pressure,
    split -- because both models answered the same prompt.
    """
    labels = list(scored)
    if len(labels) != 2:
        raise CompositionError(f"expected two cells to pair, got {labels}")
    a, b = scored[labels[0]], scored[labels[1]]
    out: Dict[str, Any] = {"a": labels[0], "b": labels[1], "types": {}}
    for ctype in types:
        units = []
        for sid, holds in a["_scenario_hold"].items():
            if ctype not in holds or ctype not in b["_scenario_hold"].get(sid, {}):
                continue
            scenario = corpus[sid]
            units.append((scenario, int(b["_scenario_hold"][sid][ctype])
                          - int(holds[ctype])))
        if len(units) < MIN_CELL_SCENARIOS:
            continue
        low = [d for s, d in units if s["k"] <= 3]
        high = [d for s, d in units if s["k"] == 4]
        out["types"][ctype] = {
            "n_scenarios": len(units),
            "mean_advantage_k_le_3": (round(sum(low) / len(low), PLACES)
                                      if low else None),
            "mean_advantage_k_eq_4": (round(sum(high) / len(high), PLACES)
                                      if high else None),
            "tests": {
                "unstratified": permutation_trend_test(
                    [(0, s["k"], d) for s, d in units], resamples=resamples),
                "by_pressure": permutation_trend_test(
                    [(s["pressure"], s["k"], d) for s, d in units],
                    resamples=resamples),
            },
        }
    for name in ("unstratified", "by_pressure"):
        live = [c for c in out["types"].values()
                if not c["tests"][name]["degenerate"]]
        for c, adjusted in zip(live, holm_adjust([c["tests"][name]["p"]
                                                  for c in live])):
            c[f"holm_p_{name}"] = adjusted
    return out


# ----------------------------------------------------------------- confounds
def confound_report(corpus: Mapping[str, Any], instances: Mapping[str, Any],
                    resamples: int) -> Dict[str, Any]:
    """Is k assigned independently of everything else? Measured, not assumed."""
    scenarios = list(corpus.values())
    ks = sorted({s["k"] for s in scenarios})

    def cross(field: str, value: str) -> Dict[str, Any]:
        rows = {str(k): {"n": 0, "hits": 0} for k in ks}
        for s in scenarios:
            rows[str(s["k"])]["n"] += 1
            rows[str(s["k"])]["hits"] += int(s[field] == value)
        for r in rows.values():
            r["rate"] = rate(r["hits"], r["n"])
        # The SAME permutation engine, with the covariate standing in for hold.
        test = permutation_trend_test(
            [(0, s["k"], int(s[field] == value)) for s in scenarios],
            resamples=resamples)
        return {"field": field, "value": value, "by_k": rows, "test": test}

    co_occurrence: Dict[str, Dict[str, Dict[str, int]]] = {}
    for ctype in sorted({instances[c]["type"] for s in scenarios
                         for c in s["constraints"]}):
        per_k: Dict[str, Dict[str, int]] = {}
        for k in ks:
            counter: Counter = Counter()
            for s in scenarios:
                if s["k"] != k:
                    continue
                present = {instances[c]["type"] for c in s["constraints"]}
                if ctype not in present:
                    continue
                counter["_scenarios"] += 1
                for other in present - {ctype}:
                    counter[other] += 1
            if counter:
                per_k[str(k)] = dict(sorted(counter.items()))
        co_occurrence[ctype] = per_k

    letters: Dict[str, Dict[str, Dict[str, int]]] = {}
    for ctype in ("banned_letter", "sentence_initial_letter"):
        per_k = {}
        for k in ks:
            counter = Counter(instances[c]["params"]["letter"]
                              for s in scenarios if s["k"] == k
                              for c in s["constraints"]
                              if instances[c]["type"] == ctype)
            if counter:
                per_k[str(k)] = dict(sorted(counter.items()))
        letters[ctype] = per_k

    clustering: Dict[str, Any] = {}
    for ctype in sorted({instances[c]["type"] for s in scenarios
                         for c in s["constraints"]}):
        counts = [sum(1 for c in s["constraints"]
                      if instances[c]["type"] == ctype) for s in scenarios]
        counts = [c for c in counts if c]
        clustering[ctype] = {
            "scenarios": len(counts),
            "checks": sum(counts),
            "scenarios_with_more_than_one": sum(1 for c in counts if c > 1),
            "max_per_scenario": max(counts) if counts else 0,
        }

    return {
        "pressure_vs_k": cross("pressure", "adversarial"),
        "split_vs_k": cross("split", "heldout_type"),
        "co_occurring_types_by_k": co_occurrence,
        "letters_by_k": letters,
        "checks_per_scenario": clustering,
    }


def pressure_effect_at_k4(scored: Mapping[str, Any], corpus: Mapping[str, Any],
                          types: Sequence[str]) -> Dict[str, Any]:
    """Clean vs adversarial at FIXED k=4 -- the competition the corpus does pose.

    Holding k at 4 removes the composition question entirely, so whatever is left
    is the effect of the embedded instruction that tells the model to break the
    rule. That is constraint-vs-INSTRUCTION tension, which is the only kind this
    corpus contains in quantity (see the tension section).
    """
    out: Dict[str, Any] = {}
    for label, cell in scored.items():
        per_type = {}
        for ctype in types:
            arms = {}
            for pressure in ("clean", "adversarial"):
                sids = [sid for sid, h in cell["_scenario_hold"].items()
                        if ctype in h and corpus[sid]["k"] == 4
                        and corpus[sid]["pressure"] == pressure]
                held = sum(cell["_scenario_hold"][sid][ctype] for sid in sids)
                arms[pressure] = {"n": len(sids), "held": held,
                                  "rate": rate(held, len(sids))}
            if arms["clean"]["n"] and arms["adversarial"]["n"]:
                arms["difference_ci95"] = newcombe_difference_ci(
                    arms["clean"]["held"], arms["clean"]["n"],
                    arms["adversarial"]["held"], arms["adversarial"]["n"])
                per_type[ctype] = arms
        out[label] = per_type
    return out


def paired_base_vs_tuned(scored: Mapping[str, Any], corpus: Mapping[str, Any],
                         types: Sequence[str]) -> Dict[str, Any]:
    """McNemar exact, base against tuned, on the SAME scenarios, split by k.

    Reuses `compare_checkpoints.two_sided_sign_p` rather than re-deriving the
    exact binomial: the two cells are one scenario set measured twice, and this
    repo already owns the right test for that shape.
    """
    labels = list(scored)
    if len(labels) != 2:
        raise CompositionError(f"expected two cells to pair, got {labels}")
    a, b = scored[labels[0]], scored[labels[1]]
    out: Dict[str, Any] = {"a": labels[0], "b": labels[1], "cells": {}}
    for ctype in types:
        for group, name in (((2, 3), "k<=3"), ((4,), "k=4"), ((2, 3, 4), "all k")):
            sids = sorted(sid for sid in a["_scenario_hold"]
                          if ctype in a["_scenario_hold"][sid]
                          and sid in b["_scenario_hold"]
                          and ctype in b["_scenario_hold"][sid]
                          and corpus[sid]["k"] in group)
            a_only = sum(1 for sid in sids
                         if a["_scenario_hold"][sid][ctype]
                         and not b["_scenario_hold"][sid][ctype])
            b_only = sum(1 for sid in sids
                         if b["_scenario_hold"][sid][ctype]
                         and not a["_scenario_hold"][sid][ctype])
            if not sids:
                continue
            out["cells"][f"{ctype}|{name}"] = {
                "paired": len(sids), "a_only_hold": a_only, "b_only_hold": b_only,
                "discordant": a_only + b_only,
                "mcnemar_exact_p": round(two_sided_sign_p(a_only, b_only), 4),
                "mcnemar_exact_p_full": two_sided_sign_p(a_only, b_only),
            }
    # Bonferroni over the family this table actually is, computed here so the
    # adjusted number quoted in prose is DERIVED rather than transcribed from a
    # neighbouring cell. Multiplied from the unrounded exact p: rounding first
    # and multiplying second moves the answer.
    family = len(out["cells"])
    out["family_size"] = family
    for c in out["cells"].values():
        c["bonferroni_p"] = round(
            min(1.0, c["mcnemar_exact_p_full"] * family), 4)
    return out


# ------------------------------------------------------------------- tension
def _pool() -> Tuple[str, ...]:
    """The reachability word pool, deduplicated and order-preserved."""
    pool = tuple(dict.fromkeys(
        RB.NOUNS + RB.VERBS + RB.ADJECTIVES + RB.ADVERBS + RB.PREPOSITIONS))
    if len(pool) < 100:
        raise CompositionError(
            f"scenarios.reachability's word pool is {len(pool)} words; the "
            f"tension ratios below are meaningless on a pool that small, so "
            f"this is a hard failure rather than a quiet number")
    return pool


def tension_report(corpus: Mapping[str, Any], instances: Mapping[str, Any],
                   witness: bool = True) -> Dict[str, Any]:
    """How much genuine constraint-vs-constraint tension does the corpus hold?

    Four measurements, weakest claim first, all decided by `scenarios.
    reachability` -- the module the corpus was BUILT against, so a scenario
    cannot be tense here and clean there.

    1. `find_conflicts` -- analytic impossibility. Expected 0: the generator
       refuses such a set.
    2. `build_witness` -- a real paragraph the ground-truth checkers pass against
       every constraint at once. A witness for all 147 IS the rung-3 "both at
       once" claim, at the level of the DATA rather than the model.
    3. LEXICAL INTERACTION. Words surviving all constraints jointly, against the
       product of each constraint's own survival rate. Ratio 1.0 means the
       constraints cut independent slices of the vocabulary -- they compose but
       do not compete. Below 1.0 means they pull against each other.
    4. POSITIONAL TENSION, the sharpest case the corpus can contain: a
       `sentence_initial_letter` demanding openers in L while a co-occurring
       constraint bans letters elsewhere. Ratio = (survival among words starting
       with L) / (survival across the whole pool). Below 1.0 means the ban bites
       the openers harder than the language at large, which is genuine
       competition; above 1.0 means the two rules happen to accommodate.
    """
    pool = _pool()
    scenarios = list(corpus.values())
    conflicts: List[Dict[str, Any]] = []
    witness_failures: List[str] = []
    lexical: List[Dict[str, Any]] = []
    positional: List[Dict[str, Any]] = []
    task_collisions: List[Dict[str, Any]] = []
    redundant_pairs = 0
    total_pairs = 0

    for scenario in scenarios:
        cons = [instances[c] for c in scenario["constraints"]]
        found = RB.find_conflicts(cons)
        if found:
            conflicts.append({"scenario_id": scenario["scenario_id"],
                              "conflicts": found})
        if witness and RB.build_witness(cons) is None:
            witness_failures.append(scenario["scenario_id"])

        joint = len(RB._usable(pool, RB._summarise(cons))) / len(pool)
        product = 1.0
        for con in cons:
            product *= len(RB._usable(pool, RB._summarise([con]))) / len(pool)
        lexical.append({"scenario_id": scenario["scenario_id"], "k": scenario["k"],
                        "joint_survival": round(joint, 4),
                        "independent_expectation": round(product, 4),
                        "ratio": round(joint / product, 4) if product else None})

        sil = [c for c in cons if c["type"] == "sentence_initial_letter"]
        if sil:
            letter = sil[0]["params"]["letter"].casefold()
            rest = [c for c in cons if c["type"] != "sentence_initial_letter"]
            openers = RB._starts_with(pool, letter)
            if openers and rest:
                info = RB._summarise(rest)
                whole = len(RB._usable(pool, info)) / len(pool)
                open_rate = len(RB._usable(openers, info)) / len(openers)
                positional.append({
                    "scenario_id": scenario["scenario_id"], "k": scenario["k"],
                    "initial_letter": letter, "openers": len(openers),
                    "openers_surviving": len(RB._usable(openers, info)),
                    "co_banned_letters": sorted(
                        {c["params"]["letter"].casefold() for c in rest
                         if c["type"] == "banned_letter"}),
                    "pool_survival": round(whole, 4),
                    "opener_survival": round(open_rate, 4),
                    "ratio": round(open_rate / whole, 4) if whole else None})

        # Redundancy is the OPPOSITE of tension and is worth counting so the two
        # are not confused: a banned word that already contains a banned letter
        # is a constraint whose work another constraint has done.
        letters = {c["params"]["letter"].casefold()
                   for c in cons if c["type"] == "banned_letter"}
        for con in cons:
            if con["type"] != "banned_words":
                continue
            for word in con["params"]["words"]:
                total_pairs += 1
                if any(l in word.casefold() for l in letters):
                    redundant_pairs += 1

        # Task-vs-constraint collisions, on nudge-free tasks only. An adversarial
        # row's stored task carries the appended nudge, and reachability's own
        # docstring says task_conflicts must see the BASE task; handing it the
        # rendered one would flag every format nudge in the corpus as a
        # contradiction. Clean rows carry no nudge, so they are the honest
        # subset, and saying so is better than a number computed on the wrong
        # input.
        if scenario["pressure"] == "clean":
            found = RB.task_conflicts(scenario["task"], cons)
            if found:
                task_collisions.append({"scenario_id": scenario["scenario_id"],
                                        "conflicts": found})

    def summarise(values: Sequence[float]) -> Optional[Dict[str, float]]:
        if not values:
            return None
        return {"n": len(values), "min": round(min(values), 4),
                "median": round(statistics.median(values), 4),
                "max": round(max(values), 4),
                "below_0_9": sum(1 for v in values if v < BELOW_THRESHOLD)}

    return {
        "scenarios": len(scenarios),
        "pool_words": len(pool),
        "analytic_conflicts": conflicts,
        "witness_checked": witness,
        "witness_constructed": (len(scenarios) - len(witness_failures)
                                if witness else None),
        "witness_failures": witness_failures,
        "lexical_interaction": {
            "all": summarise([r["ratio"] for r in lexical if r["ratio"] is not None]),
            "by_k": {str(k): summarise([r["ratio"] for r in lexical
                                        if r["k"] == k and r["ratio"] is not None])
                     for k in sorted({s["k"] for s in scenarios})},
        },
        "positional_tension": {
            "all": summarise([r["ratio"] for r in positional if r["ratio"] is not None]),
            "rows": sorted(positional, key=lambda r: (r["ratio"] is None, r["ratio"])),
        },
        "banned_word_redundant_with_banned_letter": {
            "banned_words_entries": total_pairs, "already_excluded": redundant_pairs,
            "share": rate(redundant_pairs, total_pairs)},
        "task_constraint_collisions_clean_rows": task_collisions,
        "clean_scenarios_checked": sum(1 for s in scenarios
                                       if s["pressure"] == "clean"),
        "sentence_initial_with_banned_letter": sum(
            1 for s in scenarios
            if {instances[c]["type"] for c in s["constraints"]}
            >= {"sentence_initial_letter", "banned_letter"}),
    }


# ------------------------------------------------------------------- assembly
def build_report(run_dir: Path, corpus_path: Path, instances_path: Path,
                 models_json: Path, strategy: str, exclude_quarantined: bool,
                 resamples: int, witness: bool) -> Dict[str, Any]:
    corpus = {s["scenario_id"]: s for s in read_jsonl(corpus_path)}
    instances = {c["id"]: c for c in read_jsonl(instances_path)}
    if not corpus:
        raise CompositionError(f"{corpus_path}: no scenarios")
    missing = {c for s in corpus.values() for c in s["constraints"]} - set(instances)
    if missing:
        raise CompositionError(
            f"{sorted(missing)[:5]} are referenced by the corpus and absent from "
            f"{instances_path.name}; every per-type cut below would silently "
            f"lose them")

    cells = resolve_cells(run_dir, models_json, strategy)
    scored = score_cells(cells, corpus, instances, exclude_quarantined)
    types = sorted({instances[c]["type"] for s in corpus.values()
                    for c in s["constraints"]})
    ks = sorted({s["k"] for s in corpus.values()})

    trends = trend_tests(scored, corpus, types, resamples)
    report = {
        "_note": ("Every number here is re-derived from the committed transcripts "
                  "by scripts/composition_report.py. Do not hand-edit a number "
                  "into docs/COMPOSITION.md: `python scripts/composition_report.py "
                  "--check docs/COMPOSITION.md` regenerates the tables AND "
                  "re-derives every figure the prose states, and fails if the "
                  "document and the transcripts disagree anywhere."),
        "run_dir": str(run_dir.relative_to(REPO_ROOT)),
        "strategy": strategy,
        "eval_set": str(corpus_path.relative_to(REPO_ROOT)),
        "eval_set_sha256": G.sha256_of(corpus_path),
        "instances_sha256": G.sha256_of(instances_path),
        "scenarios": len(corpus),
        "constraint_instances": len(instances),
        "k_values": ks,
        "exclude_quarantined": exclude_quarantined,
        "resamples": resamples,
        "seed": SEED,
        "cells": {label: {kk: vv for kk, vv in cell.items()
                          if not kk.startswith("_")}
                  for label, cell in scored.items()},
        "by_check": type_k_table(scored, corpus, types, ks, "check"),
        "by_scenario": type_k_table(scored, corpus, types, ks, "scenario"),
        "trend_tests": trends,
        "trend_multiplicity": trend_multiplicity(trends),
        "interaction_tests": interaction_tests(scored, corpus, types, resamples),
        "confounds": confound_report(corpus, instances, resamples),
        "pressure_at_k4": pressure_effect_at_k4(scored, corpus, types),
        "paired_base_vs_tuned": paired_base_vs_tuned(scored, corpus, types),
        "tension": tension_report(corpus, instances, witness=witness),
    }
    return report


def quarantine_contrast(run_dir: Path, corpus_path: Path, instances_path: Path,
                        models_json: Path, strategy: str, resamples: int
                        ) -> Dict[str, Any]:
    """The `--exclude-quarantined` answer, re-derived so section 2 is bound too.

    The document states what dropping the judge-failed row does to the largest
    effect -- its size, its raw p and its Holm-adjusted p. Those three numbers
    are not in the default report, so `--check` re-derives them here rather
    than trusting the sentence. Only the unstratified variant is run, because
    that is all the sentence quotes; Holm still needs the whole family, so the
    family is re-run, at one third of the permutation work of a second report.
    """
    corpus = {s["scenario_id"]: s for s in read_jsonl(corpus_path)}
    instances = {c["id"]: c for c in read_jsonl(instances_path)}
    cells = resolve_cells(run_dir, models_json, strategy)
    scored = score_cells(cells, corpus, instances, exclude_quarantined=True)
    types = sorted({instances[c]["type"] for s in corpus.values()
                    for c in s["constraints"]})
    for label, cell in scored.items():
        dropped = cell["n_rows"] - cell["n_scored"]
        if dropped != len(cell["quarantined"]):
            raise CompositionError(
                f"{label}: the two conventions are supposed to differ by the "
                f"quarantined rows and nothing else, and this one drops "
                f"{dropped} rows against {len(cell['quarantined'])} "
                f"quarantined ({cell['quarantined']}). The document's "
                f"statement of the difference would be wrong, so this stops.")
    return {
        "exclude_quarantined": True,
        "trend_tests": trend_tests(scored, corpus, types, resamples,
                                   variants=("unstratified",)),
        "cells": {label: {"n_scored": cell["n_scored"],
                          "n_rows": cell["n_rows"],
                          "quarantined": cell["quarantined"],
                          "checks_total": cell["checks_total"]}
                  for label, cell in scored.items()},
    }


# --------------------------------------------------------------- presentation
def _pct(entry: Optional[Mapping[str, Any]]) -> str:
    if not entry or not entry.get("n"):
        return "—"
    return f"{entry['held']}/{entry['n']} ({100 * entry['held'] / entry['n']:.0f}%)"


def _cell(entry: Optional[Mapping[str, Any]]) -> str:
    if not entry:
        return "—"
    lo, hi = entry["ci95"]
    return (f"{entry['held']}/{entry['n']} ({100 * entry['held'] / entry['n']:.0f}%) "
            f"[{100 * lo:.0f}–{100 * hi:.0f}]")


def markdown_tables(rep: Mapping[str, Any]) -> str:
    ks = [str(k) for k in rep["k_values"]]
    out: List[str] = []

    out.append("### Hold rate by constraint type and k")
    out.append("")
    out.append("Unit: one **check**. Each cell is held/checks (rate) [95% Wilson "
               "interval, percentage points].")
    out.append("")
    out.append("| model | constraint type | " + " | ".join(f"k={k}" for k in ks) + " |")
    out.append("|---|---|" + "---|" * len(ks))
    for label in rep["by_check"]:
        for ctype, per_k in rep["by_check"][label].items():
            cells = " | ".join(_cell(per_k.get(k)) for k in ks)
            out.append(f"| {label} | `{ctype}` | {cells} |")
    out.append("")

    out.append("Unit: one **scenario** (a scenario holds a type iff every "
               "constraint of that type in it passes). This is the unit every "
               "test below uses; 27 scenarios carry two `banned_words` "
               "constraints and one carries three, so checks are clustered and "
               "the table above cannot be tested as if its rows were independent.")
    out.append("")
    out.append("| model | constraint type | " + " | ".join(f"k={k}" for k in ks) + " |")
    out.append("|---|---|" + "---|" * len(ks))
    for label in rep["by_scenario"]:
        for ctype, per_k in rep["by_scenario"][label].items():
            cells = " | ".join(_cell(per_k.get(k)) for k in ks)
            out.append(f"| {label} | `{ctype}` | {cells} |")
    out.append("")

    out.append("### Does hold rate decline with k?")
    out.append("")
    out.append("Stratified permutation test of the Cochran-Armitage trend "
               "statistic, scenario-level, "
               f"{rep['resamples']} resamples, seed {rep['seed']}. `p (raw)` is "
               "unstratified; `p (pressure)` permutes only within clean and "
               "within adversarial, so it asks whether k still predicts holding "
               "with pressure held fixed. `p (Holm)` adjusts the unstratified p "
               "across the family of tests actually run.")
    out.append("")
    out.append("| model | type | k≤3 | k=4 | Δ (pp) | 95% CI on Δ (pp) | p (raw) | "
               "p (Holm) | p (pressure) | p (pressure×split) |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for t in rep["trend_tests"]:
        tests = t["tests"]
        if tests["unstratified"]["degenerate"]:
            reason = tests["unstratified"]["reason"]
            out.append(f"| {t['model']} | `{t['type']}` | {_pct(t['k_le_3'])} | "
                       f"{_pct(t['k_eq_4'])} | — | — | not testable "
                       f"({reason}) | — | — | — |")
            continue
        ci = t["difference_ci95"]
        out.append(
            f"| {t['model']} | `{t['type']}` | {_pct(t['k_le_3'])} | "
            f"{_pct(t['k_eq_4'])} | {100 * t['difference']:+.1f} | "
            f"[{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}] | "
            f"{tests['unstratified']['p']:.4f} | "
            f"{t.get('holm_p_unstratified', float('nan')):.4f} | "
            f"{tests['by_pressure']['p']:.4f} | "
            f"{tests['by_pressure_split']['p']:.4f} |")
    out.append("")

    out.append("Same cells, split by pressure, so the confound is visible "
               "without a test.")
    out.append("")
    out.append("| model | type | clean k≤3 | clean k=4 | adversarial k≤3 | "
               "adversarial k=4 |")
    out.append("|---|---|---|---|---|---|")
    for t in rep["trend_tests"]:
        p = t["by_pressure"]
        out.append(f"| {t['model']} | `{t['type']}` | "
                   f"{_pct(p['clean']['k_le_3'])} | {_pct(p['clean']['k_eq_4'])} | "
                   f"{_pct(p['adversarial']['k_le_3'])} | "
                   f"{_pct(p['adversarial']['k_eq_4'])} |")
    out.append("")

    out.append("### Does the tuned model pay a SMALLER composition cost?")
    out.append("")
    inter = rep["interaction_tests"]
    out.append(f"The reading being tested is an interaction, so it is tested as "
               f"one: the outcome is the paired per-scenario contrast "
               f"d = ({inter['b']} holds) − ({inter['a']} holds), and the same "
               f"trend statistic asks whether d grows with k. Both models "
               f"answered the same prompts, so pairing removes topic, letter, "
               f"pressure and split.")
    out.append("")
    out.append("| type | scenarios | mean d, k≤3 | mean d, k=4 | p (raw) | "
               "p (Holm) | p (pressure) | p (pressure, Holm) |")
    out.append("|---|---|---|---|---|---|---|---|")
    for ctype, c in inter["types"].items():
        tests = c["tests"]
        raw = ("not testable" if tests["unstratified"]["degenerate"]
               else f"{tests['unstratified']['p']:.4f}")
        strat = ("not testable" if tests["by_pressure"]["degenerate"]
                 else f"{tests['by_pressure']['p']:.4f}")
        holm_raw = (f"{c['holm_p_unstratified']:.4f}"
                    if "holm_p_unstratified" in c else "—")
        holm_strat = (f"{c['holm_p_by_pressure']:.4f}"
                      if "holm_p_by_pressure" in c else "—")
        out.append(f"| `{ctype}` | {c['n_scenarios']} | "
                   f"{c['mean_advantage_k_le_3']:+.4f} | "
                   f"{c['mean_advantage_k_eq_4']:+.4f} | {raw} | {holm_raw} | "
                   f"{strat} | {holm_strat} |")
    out.append("")

    out.append("### The confound: k is not randomly assigned")
    out.append("")
    conf = rep["confounds"]
    for key, title in (("pressure_vs_k", "adversarial share"),
                       ("split_vs_k", "held-out-type share")):
        block = conf[key]
        cells = " | ".join(
            f"{block['by_k'][k]['hits']}/{block['by_k'][k]['n']} "
            f"({100 * block['by_k'][k]['rate']:.0f}%)" for k in ks)
        out.append(f"| {title} | " + " | ".join(f"k={k}" for k in ks) + " | trend p |")
        out.append("|---|" + "---|" * (len(ks) + 1))
        out.append(f"| scenarios | {cells} | {block['test']['p']:.4f} |")
        out.append("")

    out.append("Which types share the prompt, by k (scenario counts; "
               "`_scenarios` is the cell's own denominator):")
    out.append("")
    out.append("| type | " + " | ".join(f"k={k}" for k in ks) + " |")
    out.append("|---|" + "---|" * len(ks))
    for ctype, per_k in conf["co_occurring_types_by_k"].items():
        cells = []
        for k in ks:
            entry = per_k.get(k)
            if not entry:
                cells.append("—")
                continue
            n = entry.get("_scenarios", 0)
            others = ", ".join(f"{t} {v}" for t, v in entry.items()
                               if t != "_scenarios") or "none"
            cells.append(f"{n} scen: {others}")
        out.append(f"| `{ctype}` | " + " | ".join(cells) + " |")
    out.append("")

    out.append("Letters drawn, by k (letter difficulty is not constant, so a "
               "shifting letter mix would masquerade as a k effect):")
    out.append("")
    out.append("| type | " + " | ".join(f"k={k}" for k in ks) + " |")
    out.append("|---|" + "---|" * len(ks))
    for ctype, per_k in conf["letters_by_k"].items():
        cells = " | ".join(
            (", ".join(f"{l}×{n}" for l, n in per_k[k].items()) if k in per_k else "—")
            for k in ks)
        out.append(f"| `{ctype}` | {cells} |")
    out.append("")

    out.append("Clustering of checks within a scenario (why the tests use the "
               "scenario, not the check):")
    out.append("")
    out.append("| type | scenarios | checks | scenarios with >1 | max in one "
               "scenario |")
    out.append("|---|---|---|---|---|")
    for ctype, c in conf["checks_per_scenario"].items():
        out.append(f"| `{ctype}` | {c['scenarios']} | {c['checks']} | "
                   f"{c['scenarios_with_more_than_one']} | "
                   f"{c['max_per_scenario']} |")
    out.append("")

    out.append("### Pressure at fixed k = 4")
    out.append("")
    out.append("k is held at 4, so nothing here is a composition effect. What is "
               "left is the embedded instruction telling the model to break the "
               "rule — the only kind of competition this corpus contains in "
               "quantity.")
    out.append("")
    out.append("| model | type | clean | adversarial | 95% CI on Δ (pp) |")
    out.append("|---|---|---|---|---|")
    for label, per_type in rep["pressure_at_k4"].items():
        for ctype, arms in per_type.items():
            ci = arms.get("difference_ci95")
            ci_s = f"[{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]" if ci else "—"
            out.append(f"| {label} | `{ctype}` | {_pct(arms['clean'])} | "
                       f"{_pct(arms['adversarial'])} | {ci_s} |")
    out.append("")

    out.append("### Base against tuned, paired on the same scenarios")
    out.append("")
    pair = rep["paired_base_vs_tuned"]
    out.append(f"McNemar exact. `{pair['a']}-only` counts scenarios where "
               f"{pair['a']} held the type and {pair['b']} did not.")
    out.append("")
    out.append(f"| type / k | paired | {pair['a']}-only | {pair['b']}-only | "
               f"exact p | Bonferroni over {pair['family_size']} tests |")
    out.append("|---|---|---|---|---|---|")
    for key, c in pair["cells"].items():
        ctype, group = key.split("|")
        out.append(f"| `{ctype}` {group} | {c['paired']} | {c['a_only_hold']} | "
                   f"{c['b_only_hold']} | {c['mcnemar_exact_p']:.4f} | "
                   f"{c['bonferroni_p']:.4f} |")
    out.append("")

    out.append("### How much genuine constraint-vs-constraint tension exists?")
    out.append("")
    ten = rep["tension"]
    out.append(f"* Scenarios: {ten['scenarios']}.")
    out.append(f"* `reachability.find_conflicts` fires on "
               f"{len(ten['analytic_conflicts'])} of them.")
    if ten["witness_checked"]:
        built = ten["scenarios"] - len(ten["witness_failures"])
        out.append(f"* `reachability.build_witness` constructs a paragraph that "
                   f"the ground-truth checkers pass against every constraint at "
                   f"once for {built}/{ten['scenarios']}.")
    out.append(f"* `reachability.task_conflicts` fires on "
               f"{len(ten['task_constraint_collisions_clean_rows'])} of the "
               f"nudge-free (clean) tasks.")
    red = ten["banned_word_redundant_with_banned_letter"]
    out.append(f"* {red['already_excluded']}/{red['banned_words_entries']} "
               f"banned-word entries are already excluded by a co-occurring "
               f"banned letter — redundancy, not tension.")
    out.append("")
    lex = ten["lexical_interaction"]["all"]
    out.append(f"**Lexical interaction.** Joint vocabulary survival against the "
               f"independence expectation, over {lex['n']} scenarios: median "
               f"{lex['median']}, range [{lex['min']}, {lex['max']}]. "
               f"{lex['below_0_9']} scenarios fall below 0.9.")
    out.append("")
    out.append("| k | n | min | median | max | below 0.9 |")
    out.append("|---|---|---|---|---|---|")
    for k, s in ten["lexical_interaction"]["by_k"].items():
        if s:
            out.append(f"| {k} | {s['n']} | {s['min']} | {s['median']} | "
                       f"{s['max']} | {s['below_0_9']} |")
    out.append("")
    pos = ten["positional_tension"]["all"]
    if pos:
        out.append(f"**Positional tension**, the sharpest case the corpus can "
                   f"hold: a `sentence_initial_letter` meeting a letter ban. "
                   f"Ratio below 1 = the ban bites the required openers harder "
                   f"than the language at large. Over {pos['n']} scenarios: "
                   f"median {pos['median']}, range [{pos['min']}, {pos['max']}], "
                   f"{pos['below_0_9']} below 0.9.")
        out.append("")
        out.append("| scenario | k | initial | openers surviving | pool survival "
                   "| opener survival | ratio |")
        out.append("|---|---|---|---|---|---|---|")
        for r in ten["positional_tension"]["rows"]:
            out.append(f"| `{r['scenario_id']}` | {r['k']} | `{r['initial_letter']}` "
                       f"| {r['openers_surviving']}/{r['openers']} | "
                       f"{r['pool_survival']} | {r['opener_survival']} | "
                       f"{r['ratio']} |")
    return "\n".join(out)


def plain_summary(rep: Mapping[str, Any]) -> str:
    out = [f"{rep['scenarios']} scenarios, k in {rep['k_values']}, "
           f"strategy {rep['strategy']}, run {rep['run_dir']}"]
    for label, cell in rep["cells"].items():
        out.append(f"  {label:<6} {cell['model']}  scored {cell['n_scored']}/"
                   f"{cell['n_rows']}  quarantined {cell['quarantined'] or 'none'}")
        out.append(f"         checks {cell['checks_total']}, "
                   f"{cell['checks_agreeing_with_transcript']} of them with a "
                   f"stored verdict that agrees (a missing verdict fails, so "
                   f"these two counts are always equal here)")
    out.append("")
    mult = rep["trend_multiplicity"]
    out.append(f"trend family: {mult['tests_with_variance']} of "
               f"{mult['cells_tabulated']} cells have variance to test "
               f"({mult['not_testable']} not testable); family-wise error at "
               f"alpha={mult['alpha']} is {mult['family_wise_error']:.4f}")
    out.append(f"McNemar family: {rep['paired_base_vs_tuned']['family_size']} "
               f"tests, Bonferroni factor applied in the table")
    out.append("")
    out.append("trend in k (scenario unit, stratified permutation CA test):")
    for t in rep["trend_tests"]:
        tests = t["tests"]
        if tests["unstratified"]["degenerate"]:
            out.append(f"  {t['model']:<6} {t['type']:<24} NOT TESTABLE "
                       f"({tests['unstratified']['reason']})")
            continue
        ci = t["difference_ci95"]
        out.append(
            f"  {t['model']:<6} {t['type']:<24} k<=3 {_pct(t['k_le_3']):<14} "
            f"k=4 {_pct(t['k_eq_4']):<14} d={100 * t['difference']:+6.1f}pp "
            f"CI [{100 * ci[0]:+.1f},{100 * ci[1]:+.1f}]  p={tests['unstratified']['p']:.4f} "
            f"holm={t.get('holm_p_unstratified', 1.0):.4f} "
            f"p|pressure={tests['by_pressure']['p']:.4f}")
    out.append("")
    out.append("interaction (does tuned-minus-base grow with k? paired contrast):")
    for ctype, c in rep["interaction_tests"]["types"].items():
        tests = c["tests"]
        raw = ("not testable" if tests["unstratified"]["degenerate"]
               else f"{tests['unstratified']['p']:.4f}")
        strat = ("not testable" if tests["by_pressure"]["degenerate"]
                 else f"{tests['by_pressure']['p']:.4f}")
        holm_raw = (f"{c['holm_p_unstratified']:.4f}"
                    if "holm_p_unstratified" in c else "—")
        out.append(f"  {ctype:<24} mean d k<=3 {c['mean_advantage_k_le_3']:+.4f}  "
                   f"k=4 {c['mean_advantage_k_eq_4']:+.4f}  p={raw:<12} "
                   f"holm={holm_raw:<8} p|pressure={strat}")
    out.append("")
    conf = rep["confounds"]
    for key, title in (("pressure_vs_k", "adversarial"), ("split_vs_k", "heldout_type")):
        block = conf[key]
        cells = "  ".join(f"k={k} {v['hits']}/{v['n']}"
                          for k, v in block["by_k"].items())
        out.append(f"confound {title:<12} {cells}   trend p={block['test']['p']:.4f}")
    ten = rep["tension"]
    lex = ten["lexical_interaction"]["all"]
    pos = ten["positional_tension"]["all"]
    out.append("")
    out.append(f"tension: analytic conflicts {len(ten['analytic_conflicts'])}, "
               f"witness failures {len(ten['witness_failures'])}, "
               f"clean-task collisions "
               f"{len(ten['task_constraint_collisions_clean_rows'])}")
    out.append(f"  lexical interaction ratio median {lex['median']} "
               f"[{lex['min']}, {lex['max']}], {lex['below_0_9']} below 0.9")
    if pos:
        out.append(f"  positional tension ratio median {pos['median']} "
                   f"[{pos['min']}, {pos['max']}], {pos['below_0_9']} below 0.9")
    return "\n".join(out)


# ------------------------------------------------- binding the prose numbers
# `--check` used to bind ONLY the generated block in section 4. Everything
# outside it -- the headline, the confound reading, the conclusions -- was
# bound to nothing, and an audit demonstrated the hole by replacing the
# headline permutation p (0.0515) with 0.9999 in the prose and watching
# `--check` exit 0. Two of the numbers that were wrong in that document lived
# in exactly that unbound region.
#
# So the prose is bound too. Each entry below names a figure the document
# states, a pattern that finds every sentence stating it, and a derivation that
# recomputes it from the same report the tables come from. The comparison is
# made at the precision the sentence itself writes: "0.0515" is compared to
# four decimals, "about 37 %" to none, "2.7 x 10^-4" to two significant
# figures. A numeral that no pattern claims fails the check exactly as loudly
# as a wrong one -- otherwise "gated" would quietly mean "gated except in
# sentences nobody thought of".

NUMBER_WORDS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
                "five": "5", "six": "6", "seven": "7", "eight": "8",
                "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
                "thirteen": "13", "fourteen": "14", "fifteen": "15",
                "sixteen": "16", "seventeen": "17", "eighteen": "18",
                "nineteen": "19", "twenty": "20", "twenty-one": "21"}

SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷"
                             "⁸⁹⁻", "0123456789-")


def _norm_number_token(token: str) -> str:
    """"1.5 x 10^-5", "20 000", "nine", "-16.0" -> a float-parseable string."""
    word = NUMBER_WORDS.get(token.strip().casefold())
    if word is not None:
        return word
    s = token.replace("−", "-").replace(" ", "").replace(" ", "")
    s = s.translate(SUPERSCRIPTS).replace("×10", "e")
    return s.lstrip("+")


def _number_agrees(token: str, derived: float, mode: str) -> bool:
    """Compare at the precision the DOCUMENT states, not at full precision.

    A sentence that says "about 37 %" is checked to the nearest per cent; one
    that says 0.0515 is checked to four decimals. Rounding the derivation to
    the document's own precision is what makes "0.9999" fail and "0.0515" pass
    without inventing a tolerance.
    """
    s = _norm_number_token(token)
    try:
        stated = float(s)
    except ValueError:
        return False
    if mode == "le":
        return derived <= stated + 1e-12
    if mode == "ge":
        return derived >= stated - 1e-12
    if "e" in s.casefold():
        mantissa = s.casefold().split("e")[0].lstrip("-")
        digits = len(mantissa.split(".")[1]) if "." in mantissa else 0
        return ("%%.%de" % digits) % derived == ("%%.%de" % digits) % stated
    decimals = len(s.split(".")[1]) if "." in s else 0
    return ("%%.%df" % decimals) % derived == ("%%.%df" % decimals) % stated


def EQ(value: Any) -> Tuple[str, Any]:
    return ("eq", float(value))


def PCT(rate_value: float) -> Tuple[str, Any]:
    return ("eq", 100.0 * float(rate_value))


def LE(value: Any) -> Tuple[str, Any]:
    return ("le", float(value))


def TXT(value: Any) -> Tuple[str, Any]:
    return ("text", str(value))


def PRE(value: str) -> Tuple[str, Any]:
    return ("prefix", str(value))


def SUF(value: str) -> Tuple[str, Any]:
    return ("suffix", str(value))


def _expectation_agrees(mode: str, want: Any, token: str) -> bool:
    token = token.strip()
    if mode == "text":
        return token == want
    if mode == "prefix":
        return bool(token) and want.startswith(token)
    if mode == "suffix":
        return bool(token) and want.endswith(token)
    return _number_agrees(token, want, mode)


def _show(mode: str, want: Any) -> str:
    if mode in ("prefix", "suffix"):
        return f"{mode} of {want!r}"
    if mode == "text":
        return repr(want)
    return f"{want!r}" if mode == "eq" else f"{mode} {want!r}"


# ------------------------------------------------------ derivation shortcuts
def _trend_row(ctx: Mapping[str, Any], model: str, ctype: str,
               key: str = "rep") -> Mapping[str, Any]:
    for row in ctx[key]["trend_tests"]:
        if row["model"] == model and row["type"] == ctype:
            return row
    raise CompositionError(f"no trend test for {model}/{ctype}")


def _mcn(ctx: Mapping[str, Any], ctype: str, group: str) -> Mapping[str, Any]:
    return ctx["rep"]["paired_base_vs_tuned"]["cells"][f"{ctype}|{group}"]


def _ix(ctx: Mapping[str, Any], ctype: str) -> Mapping[str, Any]:
    return ctx["rep"]["interaction_tests"]["types"][ctype]


def _scen(ctx: Mapping[str, Any], model: str, ctype: str,
          ks: Sequence[str] = ("2", "3", "4")) -> Tuple[int, int]:
    per_k = ctx["rep"]["by_scenario"][model][ctype]
    return (sum(per_k[k]["held"] for k in ks if k in per_k),
            sum(per_k[k]["n"] for k in ks if k in per_k))


def _cells_agree(ctx: Mapping[str, Any], field: str) -> Any:
    values = {label: cell[field] for label, cell in ctx["rep"]["cells"].items()}
    if len(set(values.values())) != 1:
        raise CompositionError(
            f"the document states one {field} for both cells and they differ: "
            f"{values}")
    return next(iter(values.values()))


def _lexical_deviation(ctx: Mapping[str, Any]) -> float:
    lex = ctx["rep"]["tension"]["lexical_interaction"]["all"]
    return 100.0 * max(1.0 - lex["min"], lex["max"] - 1.0)


def _positional_rows(ctx: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    return ctx["rep"]["tension"]["positional_tension"]["rows"]


#: (key, pattern, derivation). The derivation returns one expectation per
#: capture group, in order.
PROSE_FACTS: Tuple[Tuple[str, str, Any], ...] = (
    ("headline: scenarios",
     r"the corpus proves (\d+) constraint sets are held simultaneously",
     lambda c: [EQ(c["rep"]["scenarios"])]),
    ("headline: largest effect",
     r"base `banned_words`,? (?:at )?([−-]?\d+\.\d) pp from k≤3 to k=4",
     lambda c: [EQ(100 * _trend_row(c, "base", "banned_words")["difference"])]),
    ("headline: raw permutation p",
     r"raw (?:permutation )?p (?:=|is) (\d\.\d+)",
     lambda c: [EQ(_trend_row(c, "base", "banned_words")
                   ["tests"]["unstratified"]["p"])]),
    ("headline: Holm-adjusted p",
     r"Holm-adjusted p (?:=|is) (\d\.\d+)",
     lambda c: [EQ(_trend_row(c, "base", "banned_words")
                   ["holm_p_unstratified"])]),
    ("headline: pressure-stratified p",
     r"pressure-stratified p (?:=|is) (\d\.\d+)",
     lambda c: [EQ(_trend_row(c, "base", "banned_words")
                   ["tests"]["by_pressure"]["p"])]),
    ("headline: k and pressure entangled",
     r"pressure are entangled at trend p = (\d\.\d+)",
     lambda c: [EQ(c["rep"]["confounds"]["pressure_vs_k"]["test"]["p"])]),
    ("limits: k confounded with pressure",
     r"confounded with pressure at trend p = (\d\.\d+)",
     lambda c: [EQ(c["rep"]["confounds"]["pressure_vs_k"]["test"]["p"])]),
    ("headline: lexical independence",
     r"Over all (\d+) scenarios the vocabulary surviving every constraint "
     r"jointly is within (\d\.\d+) % of what independence predicts \(ratio "
     r"range \[(\d\.\d+), (\d\.\d+)\]\)",
     lambda c: [EQ(c["rep"]["tension"]["lexical_interaction"]["all"]["n"]),
                EQ(_lexical_deviation(c)),
                EQ(c["rep"]["tension"]["lexical_interaction"]["all"]["min"]),
                EQ(c["rep"]["tension"]["lexical_interaction"]["all"]["max"])]),
    ("headline: nothing below the tension cut",
     r"not one scenario falls below (\d\.\d)",
     lambda c: [EQ(BELOW_THRESHOLD)]),
    ("section 1: analytic conflicts",
     r"(\d+)/(\d+) carry an analytic conflict",
     lambda c: [EQ(len(c["rep"]["tension"]["analytic_conflicts"])),
                EQ(c["rep"]["tension"]["scenarios"])]),
    ("setup: eval set",
     r"`data/scenarios/corpus\.jsonl`, (\d+) scenarios, sha256 "
     r"`([0-9a-f]+)…([0-9a-f]+)`",
     lambda c: [EQ(c["rep"]["scenarios"]), PRE(c["rep"]["eval_set_sha256"]),
                SUF(c["rep"]["eval_set_sha256"])]),
    ("setup: constraint instances",
     r"`data/scenarios/constraint_instances\.jsonl`, (\d+) instances, sha256 "
     r"`([0-9a-f]+)…([0-9a-f]+)`",
     lambda c: [EQ(c["rep"]["constraint_instances"]),
                PRE(c["rep"]["instances_sha256"]),
                SUF(c["rep"]["instances_sha256"])]),
    ("setup: base cell",
     r"\| Base cell \| `([^`…]+)…([0-9a-f]+)`, transcripts sha256 "
     r"`([0-9a-f]+)…([0-9a-f]+)` \|",
     lambda c: [PRE(c["rep"]["cells"]["base"]["model"]),
                SUF(c["rep"]["cells"]["base"]["model"]),
                PRE(c["rep"]["cells"]["base"]["transcripts_sha256"]),
                SUF(c["rep"]["cells"]["base"]["transcripts_sha256"])]),
    ("setup: tuned cell",
     r"\| Tuned cell \| `([^`…]+)…([0-9a-f]+)`, transcripts sha256 "
     r"`([0-9a-f]+)…([0-9a-f]+)` \|",
     lambda c: [PRE(c["rep"]["cells"]["tuned"]["model"]),
                SUF(c["rep"]["cells"]["tuned"]["model"]),
                PRE(c["rep"]["cells"]["tuned"]["transcripts_sha256"]),
                SUF(c["rep"]["cells"]["tuned"]["transcripts_sha256"])]),
    ("setup: k distribution",
     r"k=2: (\d+) scenarios, k=3: (\d+), k=4: (\d+)",
     lambda c: [EQ(c["rep"]["confounds"]["pressure_vs_k"]["by_k"][k]["n"])
                for k in ("2", "3", "4")]),
    ("setup: transcript cross-check coverage",
     r"All (\d+) constraint checks per cell agree, and all (\d+) are required "
     r"to have a stored verdict to agree with",
     lambda c: [EQ(_cells_agree(c, "checks_agreeing_with_transcript")),
                EQ(_cells_agree(c, "checks_total"))]),
    ("setup: the quarantined row",
     r"exactly one row: base `(s-\d+)`, whose judge failed three attempts and "
     r"whose real (\d+)-character output passes all (four|\d+) of its checkers",
     lambda c: [TXT(c["rep"]["cells"]["base"]["quarantined_rows"][0]
                    ["scenario_id"]),
                EQ(c["rep"]["cells"]["base"]["quarantined_rows"][0]
                   ["output_chars"]),
                EQ(c["rep"]["cells"]["base"]["quarantined_rows"][0]["checks"])]),
    ("setup: what the quarantine convention moves",
     r"from ([−-]?\d+\.\d) pp \(p = (\d\.\d+), Holm (\d\.\d+)\) to "
     r"([−-]?\d+\.\d) pp \(p = (\d\.\d+), Holm (\d\.\d+)\)",
     lambda c: [EQ(100 * _trend_row(c, "base", "banned_words")["difference"]),
                EQ(_trend_row(c, "base", "banned_words")
                   ["tests"]["unstratified"]["p"]),
                EQ(_trend_row(c, "base", "banned_words")["holm_p_unstratified"]),
                EQ(100 * _trend_row(c, "base", "banned_words",
                                    key="exq")["difference"]),
                EQ(_trend_row(c, "base", "banned_words", key="exq")
                   ["tests"]["unstratified"]["p"]),
                EQ(_trend_row(c, "base", "banned_words", key="exq")
                   ["holm_p_unstratified"])]),
    ("section 3: clustered checks",
     r"(\d+) scenarios carry two `banned_words` constraints and one carries "
     r"three",
     lambda c: [EQ(c["rep"]["confounds"]["checks_per_scenario"]["banned_words"]
                   ["scenarios_with_more_than_one"])]),
    ("section 3: cluster size",
     r"contribute (\d)–(\d) correlated Bernoullis",
     lambda c: [TXT("2"),
                EQ(c["rep"]["confounds"]["checks_per_scenario"]["banned_words"]
                   ["max_per_scenario"])]),
    ("section 3: resamples and seed",
     r"(\d[\d ]*\d) draws, seed (\d+)",
     lambda c: [EQ(c["rep"]["resamples"]), EQ(c["rep"]["seed"])]),
    ("section 3: the tiny cell",
     r"base `banned_words`(?: at k=2 is | k=2 cell is )(\d+) scenarios",
     lambda c: [EQ(c["rep"]["by_scenario"]["base"]["banned_words"]["2"]["n"])]),
    ("section 3: interval level",
     r"(\d+) % CI",
     lambda c: [EQ(CI_LEVEL)]),
    ("section 3: cells where Wald fails",
     r"several cells are (\d+)/(\d+) or (\d+)/(\d+)",
     lambda c: [EQ(c["rep"]["by_check"]["base"]["banned_letter"]["2"]["held"]),
                EQ(c["rep"]["by_check"]["base"]["banned_letter"]["2"]["n"]),
                EQ(c["rep"]["pressure_at_k4"]["tuned"]["banned_words"]["clean"]
                   ["held"]),
                EQ(c["rep"]["pressure_at_k4"]["tuned"]["banned_words"]["clean"]
                   ["n"])]),
    ("section 3: the multiplicity family",
     r"([A-Za-z]+|\d+) trend tests have any variance to test",
     lambda c: [EQ(c["rep"]["trend_multiplicity"]["tests_with_variance"])]),
    ("section 3: family-wise error",
     r"At α = (\d\.\d+) the family-wise error over ([a-z]+|\d+) tests is "
     r"about (\d+) %",
     lambda c: [EQ(c["rep"]["trend_multiplicity"]["alpha"]),
                EQ(c["rep"]["trend_multiplicity"]["tests_with_variance"]),
                PCT(c["rep"]["trend_multiplicity"]["family_wise_error"])]),
    ("section 5: the same family, again",
     r"with ([a-z]+|\d+) live trend tests it would not survive one",
     lambda c: [EQ(c["rep"]["trend_multiplicity"]["tests_with_variance"])]),
    ("section 3: the floored cell",
     r"`banned_letter` for base is (\d+)/(\d+)",
     lambda c: [EQ(_scen(c, "base", "banned_letter")[0]),
                EQ(_scen(c, "base", "banned_letter")[1])]),
    ("section 3: the ceilinged cells",
     r"`banned_phrases` and `no_quotation_marks` are (\d+) % everywhere in "
     r"both cells",
     lambda c: [PCT(min(per_k["rate"]
                        for label in c["rep"]["by_scenario"]
                        for t in ("banned_phrases", "no_quotation_marks")
                        for per_k in c["rep"]["by_scenario"][label][t].values()))]),
    ("section 5: the pressure confound",
     r"adversarial share of scenarios runs (\d+)/(\d+) \((\d+) %\) at k=2, "
     r"(\d+)/(\d+) \((\d+) %\) at k=3, (\d+)/(\d+) \((\d+) %\) at k=4 "
     r"— trend p = (\d\.\d+)",
     lambda c: [e for k in ("2", "3", "4")
                for e in (EQ(c["rep"]["confounds"]["pressure_vs_k"]["by_k"][k]
                             ["hits"]),
                          EQ(c["rep"]["confounds"]["pressure_vs_k"]["by_k"][k]
                             ["n"]),
                          PCT(c["rep"]["confounds"]["pressure_vs_k"]["by_k"][k]
                              ["rate"]))]
     + [EQ(c["rep"]["confounds"]["pressure_vs_k"]["test"]["p"])]),
    ("section 5: the split confound",
     r"held-out-type share is (\d+)/(\d+), (\d+)/(\d+), (\d+)/(\d+) — "
     r"non-monotone and trend p = (\d\.\d+)",
     lambda c: [e for k in ("2", "3", "4")
                for e in (EQ(c["rep"]["confounds"]["split_vs_k"]["by_k"][k]
                             ["hits"]),
                          EQ(c["rep"]["confounds"]["split_vs_k"]["by_k"][k]
                             ["n"]))]
     + [EQ(c["rep"]["confounds"]["split_vs_k"]["test"]["p"])]),
    ("section 5: what shares the k=4 prompt",
     r"The k=4 cell is (\d+) scenarios that additionally carry `banned_letter` "
     r"\((\d+)\), `no_lists` \((\d+)\), `banned_phrases` \((\d+)\) and "
     r"`banned_punctuation` \((\d+)\), (\d+) of them adversarial",
     lambda c: [EQ(c["rep"]["confounds"]["co_occurring_types_by_k"]
                   ["banned_words"]["4"]["_scenarios"])]
     + [EQ(c["rep"]["confounds"]["co_occurring_types_by_k"]["banned_words"]["4"]
           [t]) for t in ("banned_letter", "no_lists", "banned_phrases",
                          "banned_punctuation")]
     + [EQ(_trend_row(c, "base", "banned_words")["by_pressure"]["adversarial"]
           ["k_eq_4"]["n"])]),
    ("section 5: the k=2 against k=4 quote",
     r"(\d+)/(\d+) at k=2 against (\d+)/(\d+) at k=4",
     lambda c: [EQ(c["rep"]["by_scenario"]["base"]["banned_words"]["2"]["held"]),
                EQ(c["rep"]["by_scenario"]["base"]["banned_words"]["2"]["n"]),
                EQ(_trend_row(c, "base", "banned_words")["k_eq_4"]["held"]),
                EQ(_trend_row(c, "base", "banned_words")["k_eq_4"]["n"])]),
    ("section 5: the letter mix is thin",
     r"on (\d+) / (\d+) / (\d+) instances",
     lambda c: [EQ(sum(c["rep"]["confounds"]["letters_by_k"]
                       ["sentence_initial_letter"][k].values()))
                for k in ("2", "3", "4")]),
    ("section 5: the headline cell",
     r"(\d+)/(\d+) \((\d+) %\) at k≤3 against (\d+)/(\d+) \((\d+) %\) at "
     r"k=4, Δ = ([−-]?\d+\.\d) pp, 95 % CI "
     r"\[([−-]?\d+\.\d), ([−-]?\d+\.\d)\]",
     lambda c: [EQ(_trend_row(c, "base", "banned_words")["k_le_3"]["held"]),
                EQ(_trend_row(c, "base", "banned_words")["k_le_3"]["n"]),
                PCT(_trend_row(c, "base", "banned_words")["k_le_3"]["rate"]),
                EQ(_trend_row(c, "base", "banned_words")["k_eq_4"]["held"]),
                EQ(_trend_row(c, "base", "banned_words")["k_eq_4"]["n"]),
                PCT(_trend_row(c, "base", "banned_words")["k_eq_4"]["rate"]),
                EQ(100 * _trend_row(c, "base", "banned_words")["difference"]),
                EQ(100 * _trend_row(c, "base", "banned_words")
                   ["difference_ci95"][0]),
                EQ(100 * _trend_row(c, "base", "banned_words")
                   ["difference_ci95"][1])]),
    ("section 5: the headline cell by pressure",
     r"(\d+)/(\d+) → (\d+)/(\d+) within clean and (\d+)/(\d+) → "
     r"(\d+)/(\d+) within adversarial",
     lambda c: [EQ(_trend_row(c, "base", "banned_words")["by_pressure"][arm][k]
                   [f] )
                for arm in ("clean", "adversarial")
                for k in ("k_le_3", "k_eq_4")
                for f in ("held", "n")]),
    ("section 5: the pressure x split column",
     r"pressure×split column reports (\d\.\d+)",
     lambda c: [EQ(_trend_row(c, "base", "banned_words")["tests"]
                   ["by_pressure_split"]["p"])]),
    ("section 5: the smallest Holm-adjusted p",
     r"smallest Holm-adjusted p in the trend table is (\d\.\d+)",
     lambda c: [EQ(min(r["holm_p_unstratified"] for r in c["rep"]["trend_tests"]
                       if "holm_p_unstratified" in r))]),
    ("section 6: the paired family",
     r"the (\d+) tests in that table",
     lambda c: [EQ(c["rep"]["paired_base_vs_tuned"]["family_size"])]),
    ("section 6: same prompts",
     r"answered the \*\*same\*\* (\d+) prompts",
     lambda c: [EQ(c["rep"]["scenarios"])]),
    ("section 6: sentence_initial_letter, all k",
     r"all k: (\d+) scenarios base held and tuned did not, (\d+) the other "
     r"way, exact p = ([\d.]+ × 10⁻[⁰-⁹])",
     lambda c: [EQ(_mcn(c, "sentence_initial_letter", "all k")["a_only_hold"]),
                EQ(_mcn(c, "sentence_initial_letter", "all k")["b_only_hold"]),
                EQ(_mcn(c, "sentence_initial_letter", "all k")
                   ["mcnemar_exact_p_full"])]),
    ("section 6: sentence_initial_letter rates",
     r"Base holds (\d+)/(\d+) at k≤3 and (\d+)/(\d+) at k=4; tuned holds "
     r"(\d+)/(\d+) in total",
     lambda c: [EQ(v) for v in
                (_scen(c, "base", "sentence_initial_letter", ("2", "3"))[0],
                 _scen(c, "base", "sentence_initial_letter", ("2", "3"))[1],
                 _scen(c, "base", "sentence_initial_letter", ("4",))[0],
                 _scen(c, "base", "sentence_initial_letter", ("4",))[1],
                 _scen(c, "tuned", "sentence_initial_letter")[0],
                 _scen(c, "tuned", "sentence_initial_letter")[1])]),
    ("section 6: banned_punctuation, all k",
     r"all k: (\d+) base-only against (\d+) tuned-only, exact p = "
     r"([\d.]+ × 10⁻[⁰-⁹])\*\* \(Bonferroni over (\d+) "
     r"tests: (\d\.\d+)\)\. Base (\d+)/(\d+), tuned (\d+)/(\d+)",
     lambda c: [EQ(_mcn(c, "banned_punctuation", "all k")["a_only_hold"]),
                EQ(_mcn(c, "banned_punctuation", "all k")["b_only_hold"]),
                EQ(_mcn(c, "banned_punctuation", "all k")
                   ["mcnemar_exact_p_full"]),
                EQ(c["rep"]["paired_base_vs_tuned"]["family_size"]),
                EQ(_mcn(c, "banned_punctuation", "all k")["bonferroni_p"]),
                EQ(_scen(c, "base", "banned_punctuation")[0]),
                EQ(_scen(c, "base", "banned_punctuation")[1]),
                EQ(_scen(c, "tuned", "banned_punctuation")[0]),
                EQ(_scen(c, "tuned", "banned_punctuation")[1])]),
    ("section 6: banned_words at k=4",
     r"`banned_words` at k=4 is (\d+) base-only against (\d+) tuned-only, "
     r"exact p = (\d\.\d+)",
     lambda c: [EQ(_mcn(c, "banned_words", "k=4")["a_only_hold"]),
                EQ(_mcn(c, "banned_words", "k=4")["b_only_hold"]),
                EQ(_mcn(c, "banned_words", "k=4")["mcnemar_exact_p"])]),
    ("section 6: banned_words at k<=3",
     r"at k≤3 it is (\d+) against (\d+), p = (\d\.\d+)",
     lambda c: [EQ(_mcn(c, "banned_words", "k<=3")["a_only_hold"]),
                EQ(_mcn(c, "banned_words", "k<=3")["b_only_hold"]),
                EQ(_mcn(c, "banned_words", "k<=3")["mcnemar_exact_p"])]),
    ("section 6: the interaction, tested",
     r"p = (\d\.\d+) raw, Holm-adjusted (\d\.\d+), pressure-stratified "
     r"(\d\.\d+)",
     lambda c: [EQ(_ix(c, "banned_words")["tests"]["unstratified"]["p"]),
                EQ(_ix(c, "banned_words")["holm_p_unstratified"]),
                EQ(_ix(c, "banned_words")["tests"]["by_pressure"]["p"])]),
    ("section 8: the interaction, restated",
     r"p = (\d\.\d+) raw, (\d\.\d+) Holm-adjusted",
     lambda c: [EQ(_ix(c, "banned_words")["tests"]["unstratified"]["p"]),
                EQ(_ix(c, "banned_words")["holm_p_unstratified"])]),
    ("section 6: banned_words k=4 against multiplicity",
     r"(\d\.\d+) does not survive multiplicity across the (\d+) McNemar tests "
     r"either \(Bonferroni: (\d\.\d+)\)",
     lambda c: [EQ(_mcn(c, "banned_words", "k=4")["mcnemar_exact_p"]),
                EQ(c["rep"]["paired_base_vs_tuned"]["family_size"]),
                EQ(_mcn(c, "banned_words", "k=4")["bonferroni_p"])]),
    ("section 6: the mean advantage",
     r"d = ([+−-]?\d\.\d+) at k≤3 and ([+−-]?\d\.\d+) at k=4",
     lambda c: [EQ(_ix(c, "banned_words")["mean_advantage_k_le_3"]),
                EQ(_ix(c, "banned_words")["mean_advantage_k_eq_4"])]),
    ("section 6: the no_lists interaction",
     r"interaction is p = (\d\.\d+) pressure-stratified \(Holm (\d\.\d+)\)",
     lambda c: [EQ(_ix(c, "no_lists")["tests"]["by_pressure"]["p"]),
                EQ(_ix(c, "no_lists")["holm_p_by_pressure"])]),
    ("section 6: the no_lists discordants",
     r"(four|\d+) discordant scenarios out of (\d+)\*\*: (\d+) where base held "
     r"and tuned did not at k≤3, and (\d+) the other way at k=4",
     lambda c: [EQ(_mcn(c, "no_lists", "all k")["discordant"]),
                EQ(_mcn(c, "no_lists", "all k")["paired"]),
                EQ(_mcn(c, "no_lists", "k<=3")["a_only_hold"]),
                EQ(_mcn(c, "no_lists", "k=4")["b_only_hold"])]),
    ("section 7: analytic conflicts fire on none",
     r"`find_conflicts` fires on (\d+) of (\d+) scenarios",
     lambda c: [EQ(len(c["rep"]["tension"]["analytic_conflicts"])),
                EQ(c["rep"]["tension"]["scenarios"])]),
    ("section 7: the constructive witness",
     r"constructs, for \*\*(\d+)/(\d+)\*\*",
     lambda c: [EQ(c["rep"]["tension"]["witness_constructed"]),
                EQ(c["rep"]["tension"]["scenarios"])]),
    ("section 8: the witness, restated",
     r"All (\d+) constraint sets are jointly satisfiable",
     lambda c: [EQ(c["rep"]["tension"]["witness_constructed"])]),
    ("section 7: lexical interaction",
     r"median (\d\.\d+) across all (\d+), range \*\*\[(\d\.\d+), (\d\.\d+)\]\*\*"
     r", and \*\*(\d+) scenarios below (\d\.\d)\*\*",
     lambda c: [EQ(c["rep"]["tension"]["lexical_interaction"]["all"]["median"]),
                EQ(c["rep"]["tension"]["lexical_interaction"]["all"]["n"]),
                EQ(c["rep"]["tension"]["lexical_interaction"]["all"]["min"]),
                EQ(c["rep"]["tension"]["lexical_interaction"]["all"]["max"]),
                EQ(c["rep"]["tension"]["lexical_interaction"]["all"]
                   ["below_0_9"]),
                EQ(BELOW_THRESHOLD)]),
    ("section 7: positional pairs",
     r"(\d+) scenarios pair a `sentence_initial_letter` with a `banned_letter`",
     lambda c: [EQ(c["rep"]["tension"]
                   ["sentence_initial_with_banned_letter"])]),
    ("section 7: positional tension",
     r"Observed over (\d+) scenarios: median (\d\.\d+), range "
     r"\*\*\[(\d\.\d+), (\d\.\d+)\]\*\*, \*\*(\d+) below (\d\.\d)\*\*",
     lambda c: [EQ(c["rep"]["tension"]["positional_tension"]["all"]["n"]),
                EQ(c["rep"]["tension"]["positional_tension"]["all"]["median"]),
                EQ(c["rep"]["tension"]["positional_tension"]["all"]["min"]),
                EQ(c["rep"]["tension"]["positional_tension"]["all"]["max"]),
                EQ(c["rep"]["tension"]["positional_tension"]["all"]
                   ["below_0_9"]),
                EQ(BELOW_THRESHOLD)]),
    ("section 7: the most tense scenario",
     r"is `(s-\d+)` — open every sentence with `([a-z])`, ban the letter "
     r"`([a-z])` — at (\d\.\d+)",
     lambda c: [TXT(_positional_rows(c)[0]["scenario_id"]),
                TXT(_positional_rows(c)[0]["initial_letter"]),
                TXT(", ".join(_positional_rows(c)[0]["co_banned_letters"])),
                EQ(_positional_rows(c)[0]["ratio"])]),
    ("section 7: openers that survive better than the pool",
     r"(\d+) of the (\d+) sit \*\*above\*\* 1\.0",
     lambda c: [EQ(sum(1 for r in _positional_rows(c) if r["ratio"] > 1.0)),
                EQ(c["rep"]["tension"]["positional_tension"]["all"]["n"])]),
    ("section 7: redundancy",
     r"(\d+) of (\d+) banned-word entries \((\d+) %\)",
     lambda c: [EQ(c["rep"]["tension"]
                   ["banned_word_redundant_with_banned_letter"]
                   ["already_excluded"]),
                EQ(c["rep"]["tension"]
                   ["banned_word_redundant_with_banned_letter"]
                   ["banned_words_entries"]),
                PCT(c["rep"]["tension"]
                    ["banned_word_redundant_with_banned_letter"]["share"])]),
    ("section 7: task collisions",
     r"collisions: (\d+)\*\* on the (\d+) nudge-free \(clean\) tasks",
     lambda c: [EQ(len(c["rep"]["tension"]
                       ["task_constraint_collisions_clean_rows"])),
                EQ(c["rep"]["tension"]["clean_scenarios_checked"])]),
    ("section 7: the word pool",
     r"(\d+)-word synthetic pool",
     lambda c: [EQ(c["rep"]["tension"]["pool_words"])]),
    ("section 7: pressure at fixed k=4",
     r"base `banned_words` runs (\d+)/(\d+) clean against (\d+)/(\d+) "
     r"adversarial and tuned runs (\d+)/(\d+) against (\d+)/(\d+)",
     lambda c: [EQ(c["rep"]["pressure_at_k4"][label][ "banned_words"][arm][f])
                for label in ("base", "tuned")
                for arm in ("clean", "adversarial")
                for f in ("held", "n")]),
    ("section 5: the weight of the tiny cell",
     r"with a weight proportional to its (\d+) scenarios",
     lambda c: [EQ(c["rep"]["by_scenario"]["base"]["banned_words"]["2"]["n"])]),
    ("section 7: the eval's n",
     r"not established at n = (\d+)",
     lambda c: [EQ(c["rep"]["scenarios"])]),
    ("sections 5 and 8: the interval on the largest effect",
     r"95 % CI \[([−-]?\d+\.\d), ([−-]?\d+\.\d)\]",
     lambda c: [EQ(100 * _trend_row(c, "base", "banned_words")
                   ["difference_ci95"][0]),
                EQ(100 * _trend_row(c, "base", "banned_words")
                   ["difference_ci95"][1])]),
    ("section 8: the McNemar result that motivates the reading",
     r"McNemar result \(p = (\d\.\d+)\)",
     lambda c: [EQ(_mcn(c, "banned_words", "k=4")["mcnemar_exact_p"])]),
    ("section 8: base point estimates",
     r"negative for base on all (four|\d+) testable types",
     lambda c: [EQ(sum(1 for r in c["rep"]["trend_tests"]
                       if r["model"] == "base"
                       and not r["tests"]["unstratified"]["degenerate"]))]),
    ("section 8: lexical interaction, restated",
     r"within (\d\.\d+) % of independence for all (\d+) scenarios; (\d+) below "
     r"(\d\.\d); the most tense scenario in the corpus is at (\d\.\d+)",
     lambda c: [EQ(_lexical_deviation(c)),
                EQ(c["rep"]["tension"]["lexical_interaction"]["all"]["n"]),
                EQ(c["rep"]["tension"]["lexical_interaction"]["all"]
                   ["below_0_9"]),
                EQ(BELOW_THRESHOLD),
                EQ(c["rep"]["tension"]["positional_tension"]["all"]["min"])]),
    ("section 8: the two held-out results",
     r"`sentence_initial_letter` (\d+) base-only against (\d+) \(p = "
     r"([\d.]+ × 10⁻[⁰-⁹])\) and `banned_punctuation` "
     r"(\d+) against (\d+) \(p = ([\d.]+ × 10⁻[⁰-⁹])\)",
     lambda c: [EQ(_mcn(c, "sentence_initial_letter", "all k")["a_only_hold"]),
                EQ(_mcn(c, "sentence_initial_letter", "all k")["b_only_hold"]),
                EQ(_mcn(c, "sentence_initial_letter", "all k")
                   ["mcnemar_exact_p_full"]),
                EQ(_mcn(c, "banned_punctuation", "all k")["a_only_hold"]),
                EQ(_mcn(c, "banned_punctuation", "all k")["b_only_hold"]),
                EQ(_mcn(c, "banned_punctuation", "all k")
                   ["mcnemar_exact_p_full"])]),
    ("section 8: limits",
     r"(\d+) scenarios and (\d+) constraint instances",
     lambda c: [EQ(c["rep"]["scenarios"]),
                EQ(c["rep"]["constraint_instances"])]),
    ("section 8: programmatic share",
     r"Constraint checking is (\d+) % programmatic",
     lambda c: [PCT(_cells_agree(c, "checks_total")
                    / _cells_agree(c, "checks_total"))]),
)

#: Numerals in the prose that this report cannot derive, each with the reason.
#: They are printed by `--check --show-facts` so the list stays short and
#: visible rather than becoming a silent exemption.
PROSE_UNBOUND: Tuple[Tuple[str, str], ...] = (
    (r"McNemar p = 1\.000",
     "quoted from an earlier analysis in this repo, not from these runs"),
)

#: Numerals that are labels rather than measurements: section numbering and
#: cross-references, the spec's page number, the k levels themselves, an exit
#: status, the model's parameter count, a ratio's reference value of 1.
STRUCTURAL_NUMERALS: Tuple[str, ...] = (
    r"\[§\d+\]\(#[^)]*\)",
    r"§\d+",
    r"(?m)^#{1,6} \d+\.",
    r"(?m)^\d+\. ",
    r"rung[- ]3",
    r"page 4",
    r"clause 1",
    r"k ?[=≤] ?\d(?:, ?\d)*(?:,? (?:or|and) \d)?",
    r"3×2 chi-square",
    r"exits 1",
    r"1\.7B",
    r"ratio of exactly 1",
    r"below 1 means",
    r"Ratio below 1",
)


def _masked_prose(text: str) -> str:
    """The document with the generated block blanked out, line numbers intact.

    Blanking rather than removing keeps every offset -- and so every reported
    line number -- equal to the line number in the file on disk.
    """
    if MARKER_BEGIN not in text or MARKER_END not in text:
        raise CompositionError(
            f"the document has no {MARKER_BEGIN} / {MARKER_END} block")
    start = text.index(MARKER_BEGIN)
    end = text.index(MARKER_END) + len(MARKER_END)
    blanked = "".join(ch if ch == "\n" else " " for ch in text[start:end])
    return text[:start] + blanked + text[end:]


def _line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


_NUMERAL_RE = re.compile(r"\d+(?:\.\d+)?")
_CODE_RE = re.compile(r"```.*?```|`[^`\n]*`", re.S)


def check_prose_numbers(text: str, ctx: Mapping[str, Any],
                        facts: Sequence[Tuple[str, str, Any]] = PROSE_FACTS,
                        unbound: Sequence[Tuple[str, str]] = PROSE_UNBOUND,
                        structural: Sequence[str] = STRUCTURAL_NUMERALS
                        ) -> Tuple[List[str], List[Tuple[str, int, str]],
                                   List[Tuple[int, str]]]:
    """Re-derive every figure the prose states. Returns (failures, bound, notes).

    A failure is either a stated number the transcripts do not reproduce, a
    registered figure whose sentence has vanished, or a numeral in the prose
    that no pattern claims -- the last one is what keeps the guarantee from
    decaying as sentences are added.
    """
    prose = _masked_prose(text)
    failures: List[str] = []
    bound: List[Tuple[str, int, str]] = []
    notes: List[Tuple[int, str]] = []
    covered: List[Tuple[int, int]] = []

    for key, pattern, derive in facts:
        expectations = list(derive(ctx))
        matches = list(re.finditer(pattern, prose))
        if not matches:
            failures.append(
                f"{key}: no sentence matches the pattern that binds it. The "
                f"figure was reworded or deleted, and a reworded figure is an "
                f"unbound one; update PROSE_FACTS in the same commit.")
            continue
        for m in matches:
            line = _line_of(prose, m.start())
            if len(m.groups()) != len(expectations):
                failures.append(
                    f"line {line}: {key} captures {len(m.groups())} figures "
                    f"and {len(expectations)} are derived")
                continue
            for i, (mode, want) in enumerate(expectations, start=1):
                token = m.group(i)
                if not _expectation_agrees(mode, want, token):
                    failures.append(
                        f"line {line}: {key} states {token.strip()!r}; the "
                        f"transcripts derive {_show(mode, want)}")
            covered.append((m.start(), m.end()))
            bound.append((key, line, m.group(0)))

    for pattern, reason in unbound:
        for m in re.finditer(pattern, prose):
            covered.append((m.start(), m.end()))
            notes.append((_line_of(prose, m.start()),
                          f"{m.group(0)} -- {reason}"))
    for pattern in structural:
        for m in re.finditer(pattern, prose):
            covered.append((m.start(), m.end()))
    for m in _CODE_RE.finditer(prose):
        covered.append((m.start(), m.end()))

    for m in _NUMERAL_RE.finditer(prose):
        if any(a <= m.start() and m.end() <= b for a, b in covered):
            continue
        context = prose[max(0, m.start() - 40):m.end() + 40].replace("\n", " ")
        failures.append(
            f"line {_line_of(prose, m.start())}: the numeral "
            f"{m.group(0)!r} in \u2026{context}\u2026 is bound to nothing. "
            f"Every figure in this "
            f"document has to be re-derivable from the transcripts, so either "
            f"add it to PROSE_FACTS or state it outside the prose.")
    return failures, bound, notes


# -------------------------------------------------------------------- selftest
def selftest() -> int:
    """Known-answer checks for every statistic this script invents.

    The tables above are only as good as these four functions, and this script
    owns them, so they are checked against closed forms and against complete
    enumeration rather than against themselves.
    """
    failures: List[str] = []

    def check(name: str, got: Any, want: Any, tol: float = 0.0) -> None:
        ok = (abs(got - want) <= tol) if isinstance(want, float) else (got == want)
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: got {got!r}, want {want!r}")
        if not ok:
            failures.append(name)

    print("permutation trend test against complete enumeration:")
    # A tiny, fully enumerable case: 4 units, one stratum, perfect trend.
    # 4!/(2!2!) = 6 distinct label arrangements; only the observed one and its
    # mirror reach |T| = 2.
    units = [(0, 2, 0), (0, 2, 0), (0, 4, 1), (0, 4, 1)]
    exact = permutation_trend_test(units, exact_below=8)
    check("exact p on a perfect 2x2 trend", exact["p"], 2 / 6, 1e-5)
    check("exact enumeration ran", exact["exact"], True)
    sampled = permutation_trend_test(units, resamples=20000)
    check("Monte-Carlo p tracks the exact p", sampled["p"], 2 / 6, 0.02)

    # A hand-countable case with an unbalanced statistic: 4 units at k=2 holding
    # 3, 4 units at k=4 holding 1, so T = -2. Of the C(8,4) = 70 arrangements,
    # |T*| >= 2 for the 34 that put 0, 1, 3 or 4 of the holds at k=4.
    lopsided = [(0, 2, 1), (0, 2, 1), (0, 2, 1), (0, 2, 0),
                (0, 4, 1), (0, 4, 0), (0, 4, 0), (0, 4, 0)]
    check("exact p on a hand-counted 8-unit case",
          permutation_trend_test(lopsided, exact_below=8)["p"], 34 / 70, 1e-5)

    # Stratifying must be able to DESTROY a trend that lives entirely between
    # strata. Here every stratum holds one value of k, so no within-stratum
    # relabeling can move the statistic and the test correctly refuses.
    between = [("a", 2, 1), ("a", 2, 1), ("a", 2, 1), ("a", 2, 0),
               ("b", 4, 1), ("b", 4, 0), ("b", 4, 0), ("b", 4, 0)]
    got = permutation_trend_test(between, resamples=2000)
    check("a trend that lives entirely between strata is not testable within them",
          got["degenerate"], True)
    check("...and it is testable once the strata are pooled",
          permutation_trend_test([(0, k, y) for _, k, y in between],
                                 exact_below=8)["p"], 34 / 70, 1e-5)

    print("within-stratum relabeling (the confound adjustment itself):")
    # Two strata whose hold counts differ, both spanning both k values. A draw
    # that crossed a stratum boundary would almost surely change a stratum's
    # hold count, and the runtime invariant would raise rather than return.
    crossable = [("a", 2, 1), ("a", 2, 1), ("a", 4, 1), ("a", 4, 0),
                 ("b", 2, 0), ("b", 2, 0), ("b", 4, 0), ("b", 4, 0)]
    try:
        permutation_trend_test(crossable, resamples=500)
        crossed = False
    except CompositionError:
        crossed = True
    check("500 draws all stay inside their stratum", crossed, False)
    check("a relabeling that swaps two strata's outcomes is rejected",
          _strata_preserved(crossable, [0, 0, 0, 0, 1, 1, 1, 0]), False)
    check("a relabeling that reorders within a stratum is accepted",
          _strata_preserved(crossable, [1, 1, 0, 1, 0, 0, 0, 0]), True)

    print("degeneracy:")
    check("all-hold cell", permutation_trend_test([(0, 2, 1), (0, 4, 1)])["degenerate"],
          True)
    check("all-drop cell", permutation_trend_test([(0, 2, 0), (0, 4, 0)])["degenerate"],
          True)
    check("empty cell", permutation_trend_test([])["degenerate"], True)

    print("Newcombe difference interval:")
    # Newcombe 1998, method 10, worked example: 56/70 vs 48/80.
    ci = newcombe_difference_ci(48, 80, 56, 70)
    check("published example lower", ci[0], 0.0524, 0.001)
    check("published example upper", ci[1], 0.3339, 0.001)
    boundary = newcombe_difference_ci(0, 17, 0, 45)
    check("a 0/n vs 0/n difference is exactly zero", boundary[0] <= 0.0 <= boundary[1],
          True)
    check("a 0/n vs 0/n interval still has width", boundary[1] > 0.0, True)

    print("Holm adjustment:")
    # 0.01*3 = 0.03; 0.03*2 = 0.06; 0.04*1 = 0.04, raised to 0.06 by the
    # step-down monotonicity rule. Reported in the INPUT order.
    check("Holm on [0.01, 0.04, 0.03]", holm_adjust([0.01, 0.04, 0.03]),
          [0.03, 0.06, 0.06])
    check("Holm is monotone in the sorted order",
          holm_adjust([0.5, 0.5]), [1.0, 1.0])

    print("Wilson interval, imported from curve_report:")
    check("Wilson 0/17 lower bound is 0", wilson_interval(0, 17)[0], 0.0)
    check("Wilson 0/17 upper bound", wilson_interval(0, 17)[1], 0.1846, 0.001)

    print("transcript cross-check coverage (a DELETED verdict, not a flipped one):")
    check("493 of 493 checks carry a stored verdict",
          require_transcript_coverage("cell", 493, 493), 493)
    try:
        require_transcript_coverage("cell", 492, 493)
        missed = False
    except CompositionError:
        missed = True
    check("one check without a stored verdict stops the script", missed, True)
    try:
        require_transcript_coverage("cell", 0, 493)
        stripped = False
    except CompositionError:
        stripped = True
    check("a transcript with its verdicts stripped stops the script",
          stripped, True)

    print("multiplicity arithmetic:")
    fake = [{"tests": {"unstratified": {"degenerate": d}}}
            for d in (False,) * 9 + (True,) * 5]
    mult = trend_multiplicity(fake)
    check("the family is the cells with variance, not the cells tabulated",
          (mult["tests_with_variance"], mult["cells_tabulated"]), (9, 14))
    check("family-wise error over 9 tests at alpha=0.05",
          mult["family_wise_error"], 0.3698, 1e-9)
    check("...and over 8 it would be a different number",
          trend_multiplicity(fake[:8] + fake[9:])["family_wise_error"],
          0.3366, 1e-4)

    print("Bonferroni over the McNemar family:")
    exact_p = two_sided_sign_p(16, 1)
    check("16 against 1 discordant, unrounded",
          exact_p, 0.000274658203125, 1e-15)
    check("Bonferroni over 21 tests, from the unrounded p",
          round(min(1.0, exact_p * 21), 4), 0.0058, 1e-9)
    check("rounding the p FIRST would give a different number",
          round(min(1.0, round(exact_p, 4) * 21), 4), 0.0063, 1e-9)

    print("binding the prose numbers:")
    mini = (f"The rate is 0.0515 and 3 of 4 held.\n{MARKER_BEGIN}\n"
            f"a generated 999 nobody types by hand\n{MARKER_END}\n")
    mini_facts = (("rate", r"rate is (\d\.\d+)", lambda c: [EQ(c["p"])]),
                  ("held", r"(\d+) of (\d+) held",
                   lambda c: [EQ(c["held"]), EQ(c["n"])]))
    mini_ctx = {"p": 0.051497, "held": 3, "n": 4}
    fails, bound, _ = check_prose_numbers(mini, mini_ctx, mini_facts, (), ())
    check("a document whose prose re-derives passes", fails, [])
    check("...with every figure bound", len(bound), 2)
    fails, _, _ = check_prose_numbers(mini.replace("0.0515", "0.9999"),
                                      mini_ctx, mini_facts, (), ())
    check("the audit's edit (0.0515 -> 0.9999) is caught", len(fails), 1)
    fails, _, _ = check_prose_numbers(mini.replace("held.", "held, plus 7 more."),
                                      mini_ctx, mini_facts, (), ())
    check("a numeral no pattern claims is caught", len(fails), 1)
    fails, _, _ = check_prose_numbers(mini.replace("The rate is 0.0515 and ", ""),
                                      mini_ctx, mini_facts, (), ())
    check("a deleted sentence is caught", len(fails), 1)

    print("precision: the document is checked at the precision it states:")
    check("0.0515 accepts 0.051497", _number_agrees("0.0515", 0.051497, "eq"), True)
    check("0.0515 rejects 0.0516", _number_agrees("0.0515", 0.0516, "eq"), False)
    check("about 37 % accepts 36.98", _number_agrees("37", 36.98, "eq"), True)
    check("about 37 % rejects 33.66", _number_agrees("37", 33.66, "eq"), False)
    check("2.7 x 10^-4 accepts the exact p",
          _number_agrees("2.7 × 10⁻⁴", 0.000274658203125, "eq"), True)
    check("2.7 x 10^-4 rejects the neighbouring cell's 0.0063",
          _number_agrees("2.7 × 10⁻⁴", 0.00634765625, "eq"), False)
    check("a family size written as a word", _number_agrees("nine", 9, "eq"), True)
    check("a thousands separator written as a space",
          _number_agrees("20 000", 20000, "eq"), True)
    check("a minus sign is not a hyphen, and both parse",
          _number_agrees("−16.0", -15.955, "eq"), True)

    print("McNemar, imported from compare_checkpoints:")
    check("no discordant pairs", two_sided_sign_p(0, 0), 1.0)
    check("0 vs 6 discordant", two_sided_sign_p(0, 6), 2 / 64, 1e-12)

    if failures:
        print(f"\nFAIL selftest: {len(failures)} check(s) failed: {failures}",
              file=sys.stderr)
        return 1
    print("\nok   selftest: every statistic matches its closed form")
    return 0


# ------------------------------------------------------------------------ cli
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", default=str(REPO_ROOT / "runs" / "final"))
    ap.add_argument("--eval-set",
                    default=str(REPO_ROOT / "data" / "scenarios" / "corpus.jsonl"))
    ap.add_argument("--instances",
                    default=str(REPO_ROOT / "data" / "scenarios" /
                                "constraint_instances.jsonl"))
    ap.add_argument("--models-json", default=str(REPO_ROOT / "configs" / "models.json"))
    ap.add_argument("--strategy", default="zero_shot")
    ap.add_argument("--exclude-quarantined", action="store_true",
                    help="drop rows whose judge or generator failed, reproducing "
                         "scripts/curve_report.py's denominator. The checkers do "
                         "not need the judge, so the default keeps them.")
    ap.add_argument("--resamples", type=int, default=RESAMPLES)
    ap.add_argument("--no-witness", action="store_true",
                    help="skip the constructive witness pass (the slow part)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--check", metavar="DOC", nargs="?",
                    const=str(REPO_ROOT / "docs" / "COMPOSITION.md"),
                    help="verify that DOC's generated block AND every figure "
                         "its prose states match what the transcripts re-derive "
                         "to; exit 1 on any difference, including a numeral "
                         "bound to nothing. DOC defaults to "
                         "docs/COMPOSITION.md, the document this script owns.")
    ap.add_argument("--show-facts", action="store_true",
                    help="with --check, print every prose figure the document "
                         "states, the line it is on, and what it was compared "
                         "against")
    ap.add_argument("--selftest", action="store_true",
                    help="known-answer checks for the statistics, no data needed")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()

    try:
        rep = build_report(Path(args.run_dir), Path(args.eval_set),
                           Path(args.instances), Path(args.models_json),
                           args.strategy, args.exclude_quarantined,
                           args.resamples, witness=not args.no_witness)
    except (CompositionError, G.GateError) as exc:
        print(f"FAIL composition_report: {exc}", file=sys.stderr)
        return 1

    if args.check:
        doc = Path(args.check)
        if not doc.exists():
            print(f"FAIL {doc} does not exist", file=sys.stderr)
            return 1
        text = doc.read_text()
        if MARKER_BEGIN not in text or MARKER_END not in text:
            print(f"FAIL {doc} has no {MARKER_BEGIN} / {MARKER_END} block, so its "
                  f"numbers are not bound to anything", file=sys.stderr)
            return 1
        block = text.split(MARKER_BEGIN, 1)[1].split(MARKER_END, 1)[0].strip()
        want = markdown_tables(rep).strip()
        if block != want:
            print(f"FAIL {doc}: the generated block is not what the transcripts "
                  f"re-derive to. Re-run `python scripts/composition_report.py "
                  f"--markdown` and paste the result between the markers, or "
                  f"explain why the rows changed.", file=sys.stderr)
            for i, (a, b) in enumerate(zip(block.splitlines(), want.splitlines())):
                if a != b:
                    print(f"  first difference, line {i + 1}:\n    doc:  {a}\n"
                          f"    derived: {b}", file=sys.stderr)
                    break
            else:
                print(f"  the block is {len(block.splitlines())} lines and the "
                      f"derivation is {len(want.splitlines())}", file=sys.stderr)
            return 1
        if args.exclude_quarantined:
            print("FAIL --check is defined against the default (keep the "
                  "judge-failed row) convention; drop --exclude-quarantined",
                  file=sys.stderr)
            return 1
        ctx = {"rep": rep,
               "exq": quarantine_contrast(Path(args.run_dir),
                                          Path(args.eval_set),
                                          Path(args.instances),
                                          Path(args.models_json),
                                          args.strategy, args.resamples)}
        failures, bound, notes = check_prose_numbers(text, ctx)
        if args.show_facts:
            for key, line, snippet in bound:
                print(f"  line {line:>4}  {key}: {snippet}")
            for line, note in notes:
                print(f"  line {line:>4}  not derived here: {note}")
        if failures:
            print(f"FAIL {doc}: the prose states figures the transcripts do "
                  f"not re-derive.", file=sys.stderr)
            for failure in failures:
                print(f"  {failure}", file=sys.stderr)
            return 1
        print(f"ok   {doc.name} matches the tables re-derived from "
              f"{args.run_dir}, and all {len(bound)} figures its prose states "
              f"re-derive to the same numbers")
        return 0

    if args.json:
        print(json.dumps(rep, indent=2, ensure_ascii=False))
        return 0

    print(plain_summary(rep))
    if args.markdown:
        print()
        print(markdown_tables(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
