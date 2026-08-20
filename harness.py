#!/usr/bin/env python3
"""Scoring layer for the Forbidden-Constraint Holder behavior.

This module turns ``(scenario, model output)`` into the pinned transcript row
and turns a pile of rows into the pinned metrics. It is the single place where
``spec_pass`` is decided, and it owns nothing else: it does not generate, it
does not call a model to produce ``output``, and it does not reimplement a
single checker.

Binding contracts (docs/PIVOT_DESIGN.md, "Interfaces" / "Metrics")::

    row  = {"scenario_id", "model", "strategy", "output", "checker_results",
            "quality_score", "quality_reasoning", "spec_pass"}
    checker_results[i] = {"constraint_id", "pass", "evidence"}
    spec_pass          = ALL checkers pass AND quality_score >= 3
    Spec-adherence     = mean(spec_pass) over a split
    Robustness         = mean(spec_pass) over the pressure=adversarial subset

R4 -- THE JUDGE MUST NEVER OVERRIDE A CHECKER
---------------------------------------------
Enforced by construction, in three independent ways:

1. ``checker_results`` is computed *before* the judge is called, from
   ``constraints.checkers.check`` alone. The judge's return value is never
   written into a checker result, and there is no code path that can.
2. The judge signature (:mod:`judge`) is ``judge_fn(task, output)``. It is not
   given the constraints, so it has nothing to override. See judge.py.
3. The verdict is one pure three-line function, :func:`compute_spec_pass`,
   which ANDs an all-checkers-pass boolean with a threshold comparison. A
   quality_score of 5 contributes ``True`` to one side of an ``and`` whose
   other side is already ``False``. There is no branch in which quality
   rescues a violation, and tests/test_harness.py pins that in both
   directions.

Decisions taken here (stated, not silent)
-----------------------------------------
* **The judge IS called even when the checkers already failed** --
  :data:`JUDGE_ON_CHECKER_FAILURE`. It costs money and it cannot change the
  verdict, so the tradeoff is real. Called anyway because (a) a null
  ``quality_score`` on ~40% of rows makes the transcript schema
  non-uniform and forces every downstream reader to special-case it, and
  (b) the surviving-failure-mode paragraph T-4 owes Hank needs to separate
  "the model wrote a good answer and slipped one constraint" from "the model
  degenerated into garbage" -- and those two look identical if quality is
  unmeasured on failing rows. Flip the constant to skip the call; the flag is
  greppable and one test pins the current choice.
* **A judge failure quarantines the row, it does not score it.** Default
  ``on_judge_error="raise"``: the run stops. With ``on_judge_error="record"``
  the row carries ``quality_score=None``, ``spec_pass=None`` and an extra
  ``judge_error`` key, and :func:`aggregate` excludes it from every rate and
  reports the count. Nothing anywhere invents a number.
* **:func:`aggregate` takes the scenarios and the constraint lookup, not just
  the rows.** The pinned transcript row carries neither ``pressure`` nor ``k``
  nor the constraint *type*, so Robustness, the per-k breakdown and the
  per-type breakdown are literally not computable from rows alone. Widening
  the row would break the pinned schema; joining on ``scenario_id`` does not.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from constraints.checkers import check

__all__ = [
    "QUALITY_THRESHOLD",
    "JUDGE_ON_CHECKER_FAILURE",
    "TRANSCRIPT_FIELDS",
    "CHECKER_RESULT_FIELDS",
    "STRATEGIES",
    "PRESSURES",
    "HarnessError",
    "build_constraint_lookup",
    "run_checkers",
    "compute_spec_pass",
    "score_scenario",
    "validate_transcript_row",
    "aggregate",
]

#: quality_score must be >= this for spec_pass. Pinned by Design §Interfaces.
QUALITY_THRESHOLD = 3

#: See module docstring. True = always pay for the judge call.
JUDGE_ON_CHECKER_FAILURE = True

#: The pinned transcript row keys, in order. A successful row has EXACTLY these.
TRANSCRIPT_FIELDS: Tuple[str, ...] = (
    "scenario_id", "model", "strategy", "output",
    "checker_results", "quality_score", "quality_reasoning", "spec_pass",
)

#: The pinned checker_results entry keys. Exactly these -- adding "type" here
#: to make aggregate() easier would break the contract, which is why
#: aggregate() takes a constraint_lookup instead.
CHECKER_RESULT_FIELDS: Tuple[str, ...] = ("constraint_id", "pass", "evidence")

STRATEGIES: Tuple[str, ...] = ("zero_shot", "few_shot", "structured")
PRESSURES: Tuple[str, ...] = ("clean", "adversarial")

#: Extra key present ONLY on quarantined rows. Grep it to find broken scoring.
JUDGE_ERROR_FIELD = "judge_error"
# A generator can decline to produce output at all -- a safety-classifier refusal
# returns stop_reason="refusal" with zero content blocks. That is NOT an empty
# completion: empty text passes every checker vacuously (ruling A7), so scoring a
# refusal as a response would record it as perfectly compliant. It is also not a
# constraint failure -- the model never attempted the task. Measured at ~4% on
# claude-opus-5, on innocuous prompts. Quarantine it the way judge errors are
# quarantined: recorded, excluded from every rate, and COUNTED so the exclusion is
# visible rather than silent.
GENERATOR_ERROR_FIELD = "generator_error"


class HarnessError(RuntimeError):
    """Structural problem in the inputs to scoring -- never a measurement."""


# ==========================================================================
# Inputs
# ==========================================================================

def build_constraint_lookup(constraints: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """``[constraint, ...]`` -> ``{id: constraint}``, rejecting duplicate ids.

    A duplicate id means two different constraints answer to the same name and
    every row referencing it is ambiguous, so it raises rather than picking one.
    """
    lookup: Dict[str, Dict[str, Any]] = {}
    for c in constraints:
        if not isinstance(c, Mapping):
            raise HarnessError(f"constraint must be an object, got {type(c).__name__}")
        cid = c.get("id")
        if not isinstance(cid, str) or not cid:
            raise HarnessError(f"constraint has no usable 'id': {c!r}")
        if cid in lookup:
            raise HarnessError(f"duplicate constraint id {cid!r}")
        lookup[cid] = dict(c)
    return lookup


def _resolve(constraint_lookup: Mapping[str, Mapping[str, Any]], cid: Any,
             scenario_id: Any) -> Mapping[str, Any]:
    if not isinstance(cid, str) or not cid:
        raise HarnessError(
            f"scenario {scenario_id!r}: constraint id must be a non-empty string, got {cid!r}")
    try:
        return constraint_lookup[cid]
    except KeyError:
        raise HarnessError(
            f"scenario {scenario_id!r} references constraint {cid!r}, which is not in the "
            f"lookup. Refusing to score a scenario whose constraints cannot all be "
            f"checked -- a skipped checker silently inflates spec_pass."
        ) from None


def _require(scenario: Mapping[str, Any], key: str) -> Any:
    if key not in scenario:
        raise HarnessError(f"scenario is missing required key {key!r}: {scenario!r}")
    return scenario[key]


# ==========================================================================
# The verdict
# ==========================================================================

def run_checkers(scenario: Mapping[str, Any],
                 constraint_lookup: Mapping[str, Mapping[str, Any]],
                 output: str) -> List[Dict[str, Any]]:
    """Every constraint of the scenario, in scenario order, via the pinned
    ``constraints.checkers.check``. Pure: no judge, no network, no model."""
    scenario_id = scenario.get("scenario_id")
    ids = _require(scenario, "constraints")
    if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes)):
        raise HarnessError(
            f"scenario {scenario_id!r}: 'constraints' must be a list of ids, got {ids!r}")
    if len(ids) == 0:
        raise HarnessError(
            f"scenario {scenario_id!r} has zero constraints; the behavior spec requires "
            f"2-4. Scoring it would report a vacuous pass.")
    results = []
    for cid in ids:
        verdict = check(_resolve(constraint_lookup, cid, scenario_id), output)
        results.append({
            "constraint_id": cid,
            "pass": bool(verdict["pass"]),
            "evidence": str(verdict["evidence"]),
        })
    return results


def compute_spec_pass(checker_results: Sequence[Mapping[str, Any]],
                      quality_score: Optional[int]) -> Optional[bool]:
    """THE single source of truth for spec_pass.

    ``all checkers pass AND quality_score >= QUALITY_THRESHOLD``.

    The two halves are computed from disjoint inputs and combined with ``and``.
    A quality_score of 5 cannot make the left half True, and there is no
    ordering, no override and no special case in which it does. Returns None
    when the quality score is missing (judge failure), because "unknown" is not
    "failed" and must not be counted as either.
    """
    checkers_pass = all(bool(r["pass"]) for r in checker_results)
    if quality_score is None:
        return None
    return checkers_pass and (quality_score >= QUALITY_THRESHOLD)


def score_scenario(scenario: Mapping[str, Any],
                   constraint_lookup: Mapping[str, Mapping[str, Any]],
                   output: str,
                   judge_fn: Callable[[str, str], Tuple[int, str]],
                   *,
                   model: str,
                   strategy: str,
                   on_judge_error: str = "raise") -> Dict[str, Any]:
    """Score one (scenario x model x strategy) into a pinned transcript row.

    ``model`` and ``strategy`` are required keyword-only: they are pinned
    fields of the transcript row and defaulting them would stamp fabricated
    provenance onto every row.

    ``judge_fn`` is called as ``judge_fn(task, output) -> (score, reasoning)``.
    It is handed the task and the output and NOTHING else -- in particular not
    the constraints, not the constraint ids and not the checker results.

    ``on_judge_error``:
      * ``"raise"`` (default) -- a judge failure propagates. The run stops.
      * ``"record"`` -- the row is quarantined: ``quality_score=None``,
        ``spec_pass=None``, plus a ``judge_error`` key. :func:`aggregate`
        excludes such rows from every rate and reports the count.
    """
    if on_judge_error not in ("raise", "record"):
        raise ValueError(
            f"on_judge_error must be 'raise' or 'record', got {on_judge_error!r}")
    if not isinstance(model, str) or not model:
        raise HarnessError(f"model must be a non-empty string, got {model!r}")
    if strategy not in STRATEGIES:
        raise HarnessError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
    if output is None:
        raise HarnessError(
            "output is None. An absent generation is not an empty generation; "
            "scoring it would report a vacuous checker pass.")
    if not isinstance(output, str):
        raise HarnessError(f"output must be a string, got {type(output).__name__}")

    scenario_id = _require(scenario, "scenario_id")
    task = _require(scenario, "task")

    # (1) Ground truth first, from the checkers alone.
    checker_results = run_checkers(scenario, constraint_lookup, output)

    # (2) Quality second, from a judge that cannot see (1) or the constraints.
    quality_score: Optional[int]
    judge_error: Optional[str] = None
    if checker_results and not all(r["pass"] for r in checker_results) \
            and not JUDGE_ON_CHECKER_FAILURE:
        quality_score, quality_reasoning = None, "judge not called: checkers already failed"
    else:
        try:
            quality_score, quality_reasoning = _call_judge(judge_fn, task, output)
        except Exception as exc:  # noqa: BLE001 -- re-raised below unless quarantining
            if on_judge_error == "raise":
                raise
            judge_error = f"{type(exc).__name__}: {exc}"
            quality_score = None
            quality_reasoning = f"JUDGE_ERROR (row NOT scored): {judge_error}"

    # (3) AND them. Order of the operands is irrelevant; that is the point.
    spec_pass = compute_spec_pass(checker_results, quality_score)

    row: Dict[str, Any] = {
        "scenario_id": scenario_id,
        "model": model,
        "strategy": strategy,
        "output": output,
        "checker_results": checker_results,
        "quality_score": quality_score,
        "quality_reasoning": quality_reasoning,
        "spec_pass": spec_pass,
    }
    if judge_error is not None:
        row[JUDGE_ERROR_FIELD] = judge_error
    return row


def _call_judge(judge_fn: Callable[[str, str], Tuple[int, str]],
                task: str, output: str) -> Tuple[int, str]:
    """Call the judge and validate its return. A judge that returns nonsense is
    a broken judge, not a low score."""
    verdict = judge_fn(task, output)
    try:
        score, reasoning = verdict
    except (TypeError, ValueError) as exc:
        raise HarnessError(
            f"judge must return (score, reasoning), got {verdict!r}") from exc
    if isinstance(score, bool) or not isinstance(score, int):
        raise HarnessError(f"judge returned a non-integer score {score!r}")
    if not (1 <= score <= 5):
        raise HarnessError(f"judge returned score {score}, outside the 1-5 scale")
    return score, str(reasoning)


def validate_transcript_row(row: Mapping[str, Any]) -> None:
    """Raise unless ``row`` matches the pinned schema exactly, field for field."""
    if not isinstance(row, Mapping):
        raise HarnessError(f"transcript row must be an object, got {type(row).__name__}")
    allowed = set(TRANSCRIPT_FIELDS) | {JUDGE_ERROR_FIELD, GENERATOR_ERROR_FIELD}
    missing = [k for k in TRANSCRIPT_FIELDS if k not in row]
    if missing:
        raise HarnessError(f"transcript row is missing {missing}")
    extra = sorted(set(row) - allowed)
    if extra:
        raise HarnessError(f"transcript row has unpinned key(s) {extra}")
    quarantined = JUDGE_ERROR_FIELD in row or GENERATOR_ERROR_FIELD in row

    for key in ("scenario_id", "model", "strategy", "output", "quality_reasoning"):
        if not isinstance(row[key], str):
            raise HarnessError(f"transcript row {key!r} must be a string, got {row[key]!r}")
    if row["strategy"] not in STRATEGIES:
        raise HarnessError(f"transcript row strategy {row['strategy']!r} not in {STRATEGIES}")

    results = row["checker_results"]
    if not isinstance(results, list) or not results:
        # A QUARANTINED row is the one legitimate exception: the generator produced
        # nothing, so there was no text to check. Every scored row must still carry
        # results -- an empty list on a scored row would mean "no constraints were
        # checked", which compute_spec_pass would read as vacuously passing.
        if not quarantined:
            raise HarnessError(
                "transcript row checker_results must be a non-empty list")
    for entry in results:
        if not isinstance(entry, Mapping):
            raise HarnessError(f"checker_results entry must be an object, got {entry!r}")
        if set(entry) != set(CHECKER_RESULT_FIELDS):
            raise HarnessError(
                f"checker_results entry keys {sorted(entry)} != {list(CHECKER_RESULT_FIELDS)}")
        if not isinstance(entry["constraint_id"], str) or not entry["constraint_id"]:
            raise HarnessError(f"checker_results constraint_id unusable: {entry!r}")
        if not isinstance(entry["pass"], bool):
            raise HarnessError(f"checker_results 'pass' must be a bool: {entry!r}")
        if not isinstance(entry["evidence"], str) or not entry["evidence"]:
            raise HarnessError(f"checker_results 'evidence' must be a non-empty string: {entry!r}")

    score = row["quality_score"]
    if quarantined:
        if score is not None or row["spec_pass"] is not None:
            raise HarnessError(
                "a row carrying judge_error must have quality_score=None and "
                "spec_pass=None; anything else is a fabricated measurement")
        return
    if isinstance(score, bool) or not isinstance(score, int) or not (1 <= score <= 5):
        raise HarnessError(f"transcript row quality_score must be an int 1-5, got {score!r}")
    if not isinstance(row["spec_pass"], bool):
        raise HarnessError(f"transcript row spec_pass must be a bool, got {row['spec_pass']!r}")
    expected = compute_spec_pass(results, score)
    if row["spec_pass"] != expected:
        raise HarnessError(
            f"transcript row spec_pass={row['spec_pass']} disagrees with "
            f"compute_spec_pass(...)={expected}; the row was not scored by this harness")


# ==========================================================================
# Metrics
# ==========================================================================

def _rate(passes: int, n: int) -> Optional[float]:
    """None, never 0.0, on an empty denominator. A fabricated 0% reads as a
    measured catastrophe."""
    return (passes / n) if n else None


def _cell(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    passes = sum(1 for r in rows if r["spec_pass"] is True)
    return {"n": n, "passed": passes, "spec_adherence": _rate(passes, n)}


def _scenario_map(scenarios: Any) -> Dict[str, Mapping[str, Any]]:
    if isinstance(scenarios, Mapping):
        return dict(scenarios)
    out: Dict[str, Mapping[str, Any]] = {}
    for s in scenarios:
        sid = s.get("scenario_id")
        if not isinstance(sid, str) or not sid:
            raise HarnessError(f"scenario has no usable 'scenario_id': {s!r}")
        out[sid] = s
    return out


def aggregate(rows: Sequence[Mapping[str, Any]],
              scenarios: Any,
              constraint_lookup: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Rows -> the pinned metrics dict.

    ``scenarios`` may be a list of scenario objects or an ``{id: scenario}``
    map; ``constraint_lookup`` is ``{id: constraint}``. Both are needed because
    the pinned transcript row carries neither ``pressure``/``k`` (needed for
    Robustness and the per-k breakdown) nor the constraint ``type`` (needed for
    the per-type breakdown), and widening the row would break its schema.

    Rows quarantined by a judge failure are excluded from every rate and
    counted under ``judge_errors`` (and, together with any other row whose
    ``spec_pass`` is None, under ``n_unscored``). The per-constraint-type breakdown uses ALL
    rows including quarantined ones, because checker verdicts are produced
    without the judge and are valid regardless of it; ``by_constraint_type.rows``
    reports that denominator separately so the two are never confused.

    Every rate is ``None`` -- never ``0.0`` -- when its subset is empty.
    """
    rows = list(rows)
    smap = _scenario_map(scenarios)

    judge_error_rows = [r for r in rows if r.get(JUDGE_ERROR_FIELD) is not None]
    scored = [r for r in rows if r.get("spec_pass") is not None
              and r.get(JUDGE_ERROR_FIELD) is None
              and r.get(GENERATOR_ERROR_FIELD) is None]
    unscored = len(rows) - len(scored)

    def scenario_of(row: Mapping[str, Any]) -> Mapping[str, Any]:
        sid = row["scenario_id"]
        if sid not in smap:
            raise HarnessError(
                f"transcript row references scenario {sid!r} which is not in the "
                f"scenario set; refusing to aggregate a broken join.")
        return smap[sid]

    def field_of(row: Mapping[str, Any], key: str) -> Any:
        scenario = scenario_of(row)
        if key not in scenario:
            raise HarnessError(
                f"scenario {row['scenario_id']!r} has no {key!r}; the {key} breakdown "
                f"is real evidence and cannot be faked from a missing field.")
        return scenario[key]

    def k_of(row: Mapping[str, Any]) -> Any:
        return field_of(row, "k")

    # -- headline ----------------------------------------------------------
    overall = _cell(scored)
    adversarial = [r for r in scored if field_of(r, "pressure") == "adversarial"]

    # -- breakdowns keyed by scenario metadata ----------------------------
    by_pressure: Dict[str, Any] = {}
    for pressure in PRESSURES:
        by_pressure[pressure] = _cell(
            [r for r in scored if field_of(r, "pressure") == pressure])

    by_split: Dict[str, Any] = {}
    for split in sorted({str(field_of(r, "split")) for r in scored}):
        by_split[split] = _cell([r for r in scored if str(field_of(r, "split")) == split])

    by_k: Dict[str, Any] = {}
    for k in sorted({k_of(r) for r in scored}):
        by_k[str(k)] = _cell([r for r in scored if k_of(r) == k])

    # -- per constraint TYPE: which types actually get violated -----------
    type_stats: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        for entry in row["checker_results"]:
            cid = entry["constraint_id"]
            constraint = _resolve(constraint_lookup, cid, row["scenario_id"])
            ctype = constraint.get("type")
            if not isinstance(ctype, str) or not ctype:
                raise HarnessError(f"constraint {cid!r} has no 'type'; cannot break down by type")
            stat = type_stats.setdefault(ctype, {"checks": 0, "violations": 0})
            stat["checks"] += 1
            if not entry["pass"]:
                stat["violations"] += 1
    for stat in type_stats.values():
        stat["violation_rate"] = _rate(stat["violations"], stat["checks"])

    return {
        "n": len(rows),
        "n_scored": len(scored),
        "n_unscored": unscored,
        "judge_errors": len(judge_error_rows),
        "generator_errors": len([r for r in rows
                                 if r.get(GENERATOR_ERROR_FIELD) is not None]),
        "passed": overall["passed"],
        "spec_adherence": overall["spec_adherence"],
        "robustness": _cell(adversarial)["spec_adherence"],
        "by_pressure": by_pressure,
        "by_split": by_split,
        "by_k": by_k,
        "by_constraint_type": {
            "rows": len(rows),
            "types": dict(sorted(type_stats.items())),
        },
    }
