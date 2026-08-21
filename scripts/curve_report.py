#!/usr/bin/env python3
"""Re-derive the data-efficiency curve from the committed transcripts.

The curve is a graded ablation, so every number in it has to be reproducible by
someone who did not run it. Before this script, `runs/efficiency/curve.json` was
assembled by hand and the column the report actually argues from -- how many
times the banned letter appears in the response -- was, in the words of
`docs/EFFICIENCY_CURVE.md`, "not stored anywhere and computed from the
transcripts directly". A number computed once, by hand, in a shell that is gone
is not evidence; it is a claim.

So: `runs/efficiency/points.json` DECLARES which runs are curve points and
nothing else -- no metric may be written there. This script reads that
declaration, re-derives every metric from the raw rows, and writes
`curve.json`. `--check` re-derives and compares against the committed
`curve.json`, so a hand edit to the curve fails the gate.

What "re-derived" means here, precisely:

* `spec_adherence` / `robustness` come from `gate_check.score_cell`, the same
  function `check_mvp.py` uses. It re-runs every checker from (constraint,
  output), recomputes `spec_pass` from those verdicts plus the row's
  quality_score, refuses a row set that is not exactly the eval set, and
  cross-checks the stored aggregate. Nothing programmatic in a row is trusted.
* the per-split and per-constraint-type breakdowns are recomputed HERE from the
  same re-run checker verdicts, not read from `metrics` in the results JSON.

  These deliberately do NOT always equal the `by_constraint_type` block the
  harness stores. `harness.aggregate` counts a constraint check for every row,
  including a QUARANTINED one -- a row with no `spec_pass`, because the judge or
  the generator produced no usable result. This script counts only the rows
  `gate_check.score_cell` scores, which is the same denominator
  `spec_adherence` uses: a row that does not count toward the headline should
  not count toward the breakdown printed beside it.

  On the committed runs this touches exactly one cell. The base run's scenario
  `s-0103` was quarantined because the JUDGE failed all three attempts, so no
  quality score exists and `spec_pass` is None -- but its output is real (336
  characters) and it PASSES all four of its checkers. So the four checks this
  script drops are four checks the base model passed, and excluding them makes
  base's per-type violation rates slightly HIGHER here than in the stored block:
  `sentence_initial_letter` is 7/24 = 0.2917 here and 7/25 = 0.280 there. The
  difference runs against this report's own claim that base beats the tuned
  model on the held-out families, not for it. Every fine-tuned point quarantined
  nothing and the two agree exactly.
* the banned-letter occurrence counts are computed from the response text with
  `constraints.checkers._fold`, the same normalisation the checker applies, so
  a response cannot lower its count with a zero-width space or a fullwidth
  character.

Usage:

    python scripts/curve_report.py                  # rewrite runs/efficiency/curve.json
    python scripts/curve_report.py --check          # verify it matches; exit 1 if not
    python scripts/curve_report.py --markdown       # emit the report tables
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import gate_check as G  # noqa: E402  (the pinned gate: imported, never reimplemented)
from constraints import checkers  # noqa: E402
from constraints import registry  # noqa: E402

DEFAULT_POINTS = REPO_ROOT / "runs" / "efficiency" / "points.json"
DEFAULT_CURVE = REPO_ROOT / "runs" / "efficiency" / "curve.json"

#: Rounding for every derived rate. Four places is what the reports quote; doing
#: it in one place means the committed curve and the printed table cannot drift.
PLACES = 4


class CurveError(Exception):
    """A curve that cannot be re-derived is not a curve."""


# ------------------------------------------------------------------ loading
def read_jsonl(path: Path) -> List[Mapping[str, Any]]:
    rows = []
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise CurveError(f"{path}:{lineno}: {exc}") from exc
    return rows


def load_declaration(path: Path) -> Mapping[str, Any]:
    decl = json.loads(path.read_text())
    points = decl.get("points")
    if not isinstance(points, list) or not points:
        raise CurveError(f"{path}: 'points' must be a non-empty list")

    # A metric in the DECLARATION would defeat the point of having one.
    forbidden = {"spec_adherence", "robustness", "median_banned_letter_occurrences",
                 "all_checkers", "n_scored", "metrics"}
    for i, pt in enumerate(points):
        stray = forbidden.intersection(pt)
        if stray:
            raise CurveError(
                f"{path}: points[{i}] carries {sorted(stray)}. points.json declares "
                f"WHICH runs are curve points; every metric is re-derived from the "
                f"transcripts. A number written here would never be checked against "
                f"anything.")
        for key in ("n", "label", "model", "run_dir"):
            if key not in pt:
                raise CurveError(f"{path}: points[{i}] has no {key!r}")

    ns = [pt["n"] for pt in points]
    if ns != sorted(ns):
        raise CurveError(f"{path}: points must be in ascending n, got {ns}")
    if len(set(ns)) != len(ns):
        raise CurveError(f"{path}: duplicate n in {ns}")
    return decl


def locate_cell(run_dir: Path, model: str, strategy: str) -> Dict[str, Path]:
    """The results/transcripts pair for this model, by the harness's own naming.

    A run dir may hold several cells (runs/mvp holds base AND tuned), so the
    file is addressed by the model spec rather than by "the only one in there".
    """
    stem = f"{G.sanitize(model) if hasattr(G, 'sanitize') else _sanitize(model)}__{strategy}"
    results = run_dir / f"{stem}.results.json"
    transcripts = run_dir / f"{stem}.transcripts.jsonl"
    if not results.exists():
        raise CurveError(
            f"{results} is missing. The curve point declares model {model!r} in "
            f"{run_dir}; the harness names its output after the model, so either "
            f"the run is not there or the declaration names the wrong model.")
    if not transcripts.exists():
        raise CurveError(
            f"{transcripts} is missing -- the aggregate cannot be re-derived "
            f"without the rows it came from")
    return {"results": results, "transcripts": transcripts}


def _sanitize(spec: str) -> str:
    """Fallback mirror of the harness's filename sanitiser."""
    import eval as E
    return E.sanitize(spec)


# ------------------------------------------------------- derived breakdowns
def wilson_interval(successes: int, n: int, z: float = 1.96) -> Optional[List[float]]:
    """95% Wilson score interval for a proportion.

    The curve's own stated rule for a minimum viable N -- "the smallest point
    whose spec_adherence is indistinguishable from the largest point's" -- is
    unusable without this. Every point on this curve is indistinguishable from
    every other, so applied naively the rule names the SMALLEST N on the sweep,
    which is nonsense. The interval turns that into a statement anyone can
    check: it says how large a gap 147 scenarios can resolve at all, and the
    whole observed range of spec_adherence is narrower than a single point's
    interval.

    Wilson rather than the normal approximation because these proportions sit
    near 0.08 on n=147, where the normal interval reaches below zero.
    """
    if n <= 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return [round(max(0.0, centre - half), PLACES),
            round(min(1.0, centre + half), PLACES)]


def banned_letter_of(constraint: Mapping[str, Any]) -> str:
    letter = constraint.get("params", {}).get("letter")
    if not isinstance(letter, str) or len(letter) != 1:
        raise CurveError(f"banned_letter instance {constraint.get('id')!r} has "
                         f"params.letter {letter!r}, not a single character")
    return letter


def count_occurrences(text: str, letter: str) -> int:
    """Occurrences of ``letter`` in ``text`` under the CHECKER's normalisation.

    Using ``checkers._fold`` rather than ``text.lower().count(letter)`` is the
    whole point: the checker folds invisible characters, diacritics and
    homoglyphs before it looks, so counting the raw text would let a response
    that the checker fails report a lower count than the checker sees.
    """
    folded, _ = checkers._fold(text)
    return folded.count(letter.casefold())


def derive_breakdowns(rows: Sequence[Mapping[str, Any]],
                      eval_index: Mapping[str, Mapping[str, Any]],
                      instances: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Per-split and per-type rates, re-run from (constraint, output).

    Only rows that `score_cell` would score are counted: a quarantined row
    (generator or judge produced nothing) has no verdict to re-derive, and
    counting it either way would be a fabrication.
    """
    split_total: Counter = Counter()
    split_pass: Counter = Counter()
    type_checks: Counter = Counter()
    type_violations: Counter = Counter()
    letter_occurrences: List[int] = []
    letter_per_1000: List[float] = []
    letter_lengths: List[int] = []
    e_ban_occurrences: List[int] = []
    n_all_checkers_pass = 0
    n_scored = 0

    for row in rows:
        if row.get("judge_error") is not None or row.get("generator_error") is not None:
            continue
        sid = row["scenario_id"]
        scen = eval_index[sid]
        output = row["output"]
        n_scored += 1

        verdicts = {}
        for cid in scen["constraints"]:
            con = instances[cid]
            verdict = checkers.check(con, output)
            passed = bool(verdict["pass"] if isinstance(verdict, dict) else verdict[0])
            verdicts[cid] = passed
            type_checks[con["type"]] += 1
            if not passed:
                type_violations[con["type"]] += 1

            if con["type"] == "banned_letter":
                letter = banned_letter_of(con)
                occ = count_occurrences(output, letter)
                letter_occurrences.append(occ)
                letter_lengths.append(len(output))
                # Density needs a length to divide by. Skipping only THIS list on
                # an empty output would compute the per-1,000 column over a
                # different row set than the two columns printed beside it, and
                # the report presents all three as cuts of the same rows. An
                # empty scored output has no occurrences either, so 0 is the
                # honest density and it keeps the row sets identical.
                letter_per_1000.append(1000.0 * occ / len(output) if output else 0.0)
                if letter == "e":
                    e_ban_occurrences.append(occ)

        all_pass = all(verdicts.values())
        if all_pass:
            n_all_checkers_pass += 1

        split = scen["split"]
        split_total[split] += 1
        if all_pass and int(row["quality_score"]) >= G.QUALITY_SCORE_MIN:
            split_pass[split] += 1

    def rate(num: int, den: int) -> Optional[float]:
        return round(float(Fraction(num, den)), PLACES) if den else None

    return {
        "n_scored": n_scored,
        "all_checkers_pass": n_all_checkers_pass,
        "by_split": {
            split: {
                "n": split_total[split],
                "passed": split_pass[split],
                "spec_adherence": rate(split_pass[split], split_total[split]),
            }
            for split in sorted(split_total)
        },
        "by_constraint_type": {
            ctype: {
                "checks": type_checks[ctype],
                "violations": type_violations[ctype],
                "violation_rate": rate(type_violations[ctype], type_checks[ctype]),
            }
            for ctype in sorted(type_checks)
        },
        "banned_letter": {
            "rows": len(letter_occurrences),
            "median_occurrences": (round(statistics.median(letter_occurrences), 1)
                                   if letter_occurrences else None),
            "mean_occurrences": (round(statistics.mean(letter_occurrences), 2)
                                 if letter_occurrences else None),
            "median_per_1000_chars": (round(statistics.median(letter_per_1000), 1)
                                      if letter_per_1000 else None),
            "median_response_chars": (int(statistics.median(letter_lengths))
                                      if letter_lengths else None),
            "median_occurrences_e_ban_rows": (
                round(statistics.median(e_ban_occurrences), 1)
                if e_ban_occurrences else None),
            "clean_responses": sum(1 for o in letter_occurrences if o == 0),
        },
    }


# -------------------------------------------------------------- the curve
def score_point(pt: Mapping[str, Any], decl: Mapping[str, Any],
                eval_index: Mapping[str, Mapping[str, Any]],
                instances: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    strategy = decl.get("strategy", "zero_shot")
    run_dir = (REPO_ROOT / pt["run_dir"]).resolve()
    paths = locate_cell(run_dir, pt["model"], strategy)

    errors: List[str] = []
    stats = G.score_cell(paths["results"], paths["transcripts"], pt["model"],
                         strategy, eval_index, instances, errors)
    if stats is None:
        raise CurveError(
            f"curve point n={pt['n']} ({pt['label']}) could not be re-derived from "
            f"{paths['transcripts'].name}:\n  - " + "\n  - ".join(errors))
    if errors:
        raise CurveError(
            f"curve point n={pt['n']} ({pt['label']}) scored with complaints:\n  - "
            + "\n  - ".join(errors))

    extra = derive_breakdowns(read_jsonl(paths["transcripts"]), eval_index, instances)
    if extra["n_scored"] != stats["n"]:
        raise CurveError(
            f"curve point n={pt['n']}: this script scored {extra['n_scored']} rows "
            f"but gate_check.score_cell scored {stats['n']} -- the two disagree "
            f"about which rows are scorable")

    return {
        "n": pt["n"],
        "label": pt["label"],
        "model": pt["model"],
        "dataset": pt.get("dataset"),
        "run_dir": pt["run_dir"],
        "n_scored": stats["n"],
        "spec_adherence": round(stats["spec_adherence"], PLACES),
        "spec_adherence_ci95": wilson_interval(stats["n_spec_pass"], stats["n"]),
        "robustness": (round(stats["robustness"], PLACES)
                       if stats["robustness"] is not None else None),
        "robustness_ci95": (wilson_interval(
            round(stats["robustness"] * stats["n_adversarial"]),
            stats["n_adversarial"]) if stats["robustness"] is not None else None),
        "passed": stats["n_spec_pass"],
        "n_adversarial": stats["n_adversarial"],
        "all_checkers": extra["all_checkers_pass"],
        "median_banned_letter_occurrences": extra["banned_letter"]["median_occurrences"],
        "by_split": extra["by_split"],
        "by_constraint_type": extra["by_constraint_type"],
        "banned_letter": extra["banned_letter"],
        "transcripts_sha256": stats["transcripts_jsonl_sha256"],
    }


def build_curve(points_path: Path) -> Dict[str, Any]:
    decl = load_declaration(points_path)
    eval_path = REPO_ROOT / decl.get("eval_set", "data/scenarios/corpus.jsonl")
    inst_path = REPO_ROOT / decl.get("instances", "data/scenarios/constraint_instances.jsonl")

    eval_index = {s["scenario_id"]: s for s in read_jsonl(eval_path)}
    instances = {c["id"]: c for c in read_jsonl(inst_path)}
    if not eval_index:
        raise CurveError(f"{eval_path}: no scenarios")

    heldout = set(registry.types_by_split("heldout_type"))
    scored = [score_point(pt, decl, eval_index, instances) for pt in decl["points"]]
    return {
        "_note": ("Re-derived from the committed transcripts by "
                  "scripts/curve_report.py. Do not hand-edit: `python "
                  "scripts/curve_report.py --check` fails if this file and the "
                  "transcripts disagree."),
        "eval_set": decl.get("eval_set"),
        "eval_set_sha256": G.sha256_of(eval_path),
        "strategy": decl.get("strategy", "zero_shot"),
        "heldout_types": sorted(heldout),
        "points": scored,
    }


# ------------------------------------------------------------- presentation
def markdown_tables(curve: Mapping[str, Any]) -> str:
    pts = curve["points"]
    out: List[str] = []

    out.append("| N | `spec_adherence` | 95% CI | `robustness` | passes | "
               "all checkers pass | median banned-letter occurrences |")
    out.append("|---|---|---|---|---|---|---|")
    for p in pts:
        lo, hi = p["spec_adherence_ci95"]
        out.append(f"| {p['n']} | {p['spec_adherence']:.4f} | [{lo:.3f}, {hi:.3f}] | "
                   f"{p['robustness']:.4f} | {p['passed']}/{p['n_scored']} | "
                   f"{p['all_checkers']}/{p['n_scored']} | "
                   f"{p['median_banned_letter_occurrences']} |")
    out.append("")

    out.append("| N | `spec_adherence`, train-split types | `spec_adherence`, "
               "held-out types |")
    out.append("|---|---|---|")
    for p in pts:
        tr = p["by_split"].get("train", {})
        ho = p["by_split"].get("heldout_type", {})
        out.append(f"| {p['n']} | {tr.get('spec_adherence')} "
                   f"({tr.get('passed')}/{tr.get('n')}) | {ho.get('spec_adherence')} "
                   f"({ho.get('passed')}/{ho.get('n')}) |")
    out.append("")

    types = sorted({t for p in pts for t in p["by_constraint_type"]})
    out.append("Violation rate per constraint type (lower is better). "
               "**Bold** = a type the training data never contained.")
    out.append("")
    out.append("| N | " + " | ".join(
        f"**{t}**" if t in curve["heldout_types"] else f"`{t}`" for t in types) + " |")
    out.append("|---|" + "---|" * len(types))
    for p in pts:
        cells = []
        for t in types:
            entry = p["by_constraint_type"].get(t)
            cells.append("—" if entry is None else f"{entry['violation_rate']:.3f}")
        out.append(f"| {p['n']} | " + " | ".join(cells) + " |")
    out.append("")

    out.append("| N | median occurrences | per 1,000 chars | median response chars | "
               "median occurrences, `e`-ban rows |")
    out.append("|---|---|---|---|---|")
    for p in pts:
        b = p["banned_letter"]
        out.append(f"| {p['n']} | {b['median_occurrences']} | "
                   f"{b['median_per_1000_chars']} | {b['median_response_chars']} | "
                   f"{b['median_occurrences_e_ban_rows']} |")
    return "\n".join(out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--points", default=str(DEFAULT_POINTS))
    ap.add_argument("--out", default=str(DEFAULT_CURVE))
    ap.add_argument("--check", action="store_true",
                    help="re-derive and compare against --out; exit 1 on any "
                         "difference instead of rewriting it")
    ap.add_argument("--markdown", action="store_true",
                    help="print the report tables to stdout")
    args = ap.parse_args(argv)

    try:
        curve = build_curve(Path(args.points))
    except (CurveError, G.GateError) as exc:
        print(f"FAIL curve_report: {exc}", file=sys.stderr)
        return 1

    text = json.dumps(curve, indent=2, ensure_ascii=False) + "\n"
    out = Path(args.out)

    if args.check:
        if not out.exists():
            print(f"FAIL {out} does not exist; run without --check to write it",
                  file=sys.stderr)
            return 1
        if out.read_text() != text:
            print(f"FAIL {out} is not what the transcripts re-derive to. "
                  f"Re-run `python scripts/curve_report.py` and commit the result, "
                  f"or explain why the rows changed.", file=sys.stderr)
            return 1
        print(f"ok   {out.name} matches the {len(curve['points'])} curve points "
              f"re-derived from their transcripts")
    else:
        out.write_text(text)
        print(f"wrote {out} ({len(curve['points'])} points re-derived from transcripts)")

    if args.markdown:
        print()
        print(markdown_tables(curve))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
