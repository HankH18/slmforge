#!/usr/bin/env python3
"""Re-derive the adversarial_v2 robustness result from the committed transcripts.

`docs/ADVERSARIAL_EVAL.md` §6 shipped a heredoc that joined the transcripts to
`attack_family` and printed a per-family table. A heredoc in a document is not a
re-derivation: nobody runs it in CI, its output is pasted by hand, and the number
it prints -- the per-family `spec_pass` cell -- is not the number the result rests
on. This script is the committed command behind every figure in that document's
results section.

Four things it computes, in increasing order of what they can settle:

* **The aggregate.** `spec_adherence` and `robustness` come from
  `gate_check.score_cell`, the same function `check_mvp.py` and
  `curve_report.py` use. Every checker verdict is re-run from (constraint,
  output), `spec_pass` is recomputed from those verdicts plus the row's
  `quality_score`, the row set must cover the eval set exactly, and the stored
  aggregate is compared rather than trusted.

* **Per-family `spec_pass`.** Descriptive only. A family is 15 rows; §5 of the
  document puts a 28.6 pp interval on a cell that size, so a one- or two-scenario
  move is inside the noise by the document's own published limit. Printed with
  both denominators and a paired McNemar, never as a ranking.

* **The targeted-constraint hold rate.** For each attacked row, whether the ONE
  constraint the attack aimed at -- the row's `attack_target` -- was held. This
  is the measure the set was built to support: it asks whether the attack did the
  thing it names, instead of asking whether the whole 2-to-4 constraint spec
  survived plus a judge's quality score. Paired McNemar exact on the same
  scenarios, per family and pooled.

  `malformed_input` and `spec_deprecation` carry `attack_target = null` BY
  CONSTRUCTION and are therefore absent from this measure. Neither names a rule:
  `malformed_input` damages the brief and attacks no specific constraint, and
  `spec_deprecation` attacks the standing of the whole appended rule block --
  `scenarios/adversarial.py` refuses a `spec_deprecation` row that names a rule,
  because a row that named one would be another family wearing the label. There
  is no single constraint to ask about, so asking would require inventing one.

* **The same targeted constraint, asked again of the row's paired CLEAN
  control.** A pooled targeted win says the tuned model held the attacked
  constraint more often; it does NOT say the attack is why. §3 of
  `docs/ADVERSARIAL_EVAL.md` built each family's 15 rows on the same 15 briefs as
  15 unattacked controls carrying the identical constraint set, exactly so the
  attack can be removed and the same question asked again. On the committed runs
  this is the block that stops the pooled result from being read as robustness:
  most of the tuned model's targeted advantage is already present with no attack.
  Without it the strongest number in the report would be overstated, which is the
  failure this repository keeps having to correct.

THE QUARANTINED ROW. The base cell has one: `av2-b12-spec_deprecation`, whose
judge refused three times (`stop_reason='refusal'`) and produced no quality
score. This repository has a 146-vs-147 story about exactly this, so the row is
handled loudly rather than quietly:

* It is excluded from every `spec_pass` rate, matching `harness.aggregate` and
  `gate_check.score_cell`. Base's `spec_adherence` denominator is therefore
  **159, not 160**, and this script prints the denominator next to every rate so
  the two cells cannot be compared as if they had the same one.
* It is INCLUDED in the targeted-constraint measure whenever it carries an
  `attack_target`, because that measure is a pure checker verdict re-derived from
  (constraint, output) and a missing judge score does not remove it. Dropping a
  row whose datum survives would be dropping data for no reason.
* Every quarantined row is listed in the report with its family, its target, its
  output length, its re-derived checker tally, and `could_have_passed_spec` --
  whether all of its checkers pass, i.e. whether the missing judge score was the
  only thing between it and a possible pass. On the committed base run that flag
  is False: the row fails `c-chr-001`, so no quality score could have made it a
  pass and the quarantine cannot have moved base's numerator, only its
  denominator.

Nothing here is a second implementation of anything. McNemar's exact test is
`compare_checkpoints.two_sided_sign_p` and the Wilson interval is
`curve_report.wilson_interval`, both imported.

WHAT `--check` BINDS. Two things, because binding only the first was a gate that
protected the artifact nobody reads and left the document unguarded:

* `runs/adversarial_v2/adversarial_report.json` must equal a fresh re-derivation.
* Every figure in the Results section of `docs/ADVERSARIAL_EVAL.md` must be one
  this script produced. `document_bindings()` rebuilds each of them from the
  re-derived report -- table rows AND the numbers stated in prose -- and every one
  must appear verbatim in that section. The section is then rescanned for any
  rate, fraction or percentage-point delta that no binding covers; each is a
  failure unless it is listed in `DOC_UNGATED` with the command that does derive
  it, and a `DOC_UNGATED` entry that has fallen out of the document is a failure
  too. So a hand-edited digit in the prose fails the gate, and so does a new
  number typed into the section.

  It binds figures, not reasoning: a correctly-derived number read wrongly is
  still something a reader has to catch. It scans rate-shaped tokens, fractions
  and pp deltas; a bare integer alone in a sentence is bound only where a binding
  phrase happens to contain it.

Usage:

    python scripts/adversarial_report.py             # rewrite the stored summary
    python scripts/adversarial_report.py --check     # verify JSON + document
    python scripts/adversarial_report.py --markdown  # emit the tables, WRITE NOTHING

`--markdown` writes nothing on purpose. It used to fall through to the same
branch that rewrites the stored summary, so printing the report regenerated the
artifact `--check` compares against: a tampered JSON was silently repaired by the
act of reading it, and the gate then passed.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import gate_check as G  # noqa: E402  (the pinned gate: imported, never reimplemented)
from compare_checkpoints import two_sided_sign_p  # noqa: E402  (the ONE McNemar)
from curve_report import wilson_interval  # noqa: E402  (the ONE Wilson interval)
from constraints import checkers  # noqa: E402

DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "adversarial_v2"
DEFAULT_EVAL_SET = REPO_ROOT / "data" / "scenarios" / "adversarial_v2.jsonl"
DEFAULT_INSTANCES = REPO_ROOT / "data" / "scenarios" / "constraint_instances.jsonl"
DEFAULT_MODELS_CFG = REPO_ROOT / "configs" / "models.json"
DEFAULT_OUT = DEFAULT_RUN_DIR / "adversarial_report.json"
DEFAULT_DOC = REPO_ROOT / "docs" / "ADVERSARIAL_EVAL.md"
STRATEGY = "zero_shot"

#: Rounding for every derived rate, matching curve_report.PLACES, so the stored
#: summary and the printed table cannot drift.
PLACES = 4

#: The two families that carry no `attack_target`, and why. Derived below from
#: the eval set and cross-checked against this table in both directions: a
#: family here that turns out to carry targets, or a target-free family missing
#: from here, is a build error rather than a silently reshaped denominator.
NO_TARGET_BY_DESIGN = {
    "malformed_input": ("the brief is damaged rather than added to; the attack "
                        "names no rule, so there is no single constraint it aimed at"),
    "spec_deprecation": ("the attack is on the standing of the whole appended rule "
                         "block; scenarios/adversarial.py refuses a row in this "
                         "family that names a rule"),
}
#: The clean controls are unattacked, so they carry no target either.
CONTROL_FAMILY = "none"


class AdversarialReportError(Exception):
    """A robustness result that cannot be re-derived is not a result."""


# ------------------------------------------------------- exact pp arithmetic
def rate_delta_pp(a_num: int, a_den: int, b_num: int, b_den: int) -> float:
    """(a/a_den - b/b_den) in percentage points, from the COUNTS.

    Every difference in this report is computed here rather than by subtracting
    two already-rounded rates. Rounding then subtracting can print a figure that
    is not the difference of anything: -6.7 - (-13.3) = +6.6 where the exact
    difference is +6.667 -> +6.7. `Fraction` keeps it exact until the single
    rounding at the end.
    """
    return float((Fraction(a_num, a_den) - Fraction(b_num, b_den)) * 100)


def diff_pp(a_count: int, b_count: int, den: int) -> float:
    """(a_count - b_count)/den in percentage points, from the counts."""
    return float(Fraction(a_count - b_count, den) * 100)


# ------------------------------------------------------------------ loading
def repo_relative(path: Path) -> str:
    """The path as the repo names it, or its absolute form if it lives outside.

    `Path.relative_to` RAISES on a path outside the tree, so writing it directly
    into the report turns "you pointed --eval-set at a scratch copy" into a
    traceback instead of a report over that copy. The overrides exist to be used.
    """
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def read_jsonl(path: Path) -> List[Mapping[str, Any]]:
    if not path.exists():
        raise AdversarialReportError(f"{path} does not exist")
    rows = []
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise AdversarialReportError(f"{path}:{lineno}: {exc}") from exc
    return rows


def locate_cells(run_dir: Path, models_cfg_path: Path) -> List[Dict[str, Any]]:
    """The (label, model, results, transcripts) of every cell in the run dir.

    WHICH model is the baseline is not typed here. `configs/models.json` already
    declares `base` and `tuned` for the whole project, and this reads that
    declaration. The mapping is then cross-validated BOTH ways against the run
    directory: every pin must have a cell, and every cell must be a pin. A third
    run appearing in the directory is a hard failure rather than a silently
    ignored row or a silently changed comparison.
    """
    cfg = json.loads(models_cfg_path.read_text())
    pins: Dict[str, str] = {}
    for label in ("base", "tuned"):
        spec = cfg.get(label)
        if not isinstance(spec, str) or not spec:
            raise AdversarialReportError(
                f"{models_cfg_path}: '{label}' is {spec!r}; this script reads which "
                f"checkpoint is the baseline from the project's own pin rather than "
                f"hardcoding a model id")
        # models.json writes `base` with the hf: prefix and `tuned` without it;
        # the transcripts carry the full provider-qualified spec.
        pins[label] = spec if ":" in spec else f"hf:{spec}"

    found: Dict[str, Path] = {}
    for results in sorted(run_dir.glob(f"*__{STRATEGY}.results.json")):
        blob = json.loads(results.read_text())
        model = blob.get("model")
        if not isinstance(model, str):
            raise AdversarialReportError(f"{results}: 'model' is {model!r}")
        if model in found:
            raise AdversarialReportError(
                f"{run_dir}: two results files claim model {model!r} "
                f"({found[model].name}, {results.name})")
        found[model] = results

    unpinned = sorted(set(found) - set(pins.values()))
    if unpinned:
        raise AdversarialReportError(
            f"{run_dir} holds a cell for {unpinned}, which configs/models.json does "
            f"not pin as 'base' or 'tuned'. This report compares exactly those two; "
            f"a third cell would change what the comparison means, so it is refused "
            f"rather than dropped.")

    cells = []
    for label, spec in pins.items():
        results = found.get(spec)
        if results is None:
            raise AdversarialReportError(
                f"{run_dir} has no cell for the {label} pin {spec!r} (it holds "
                f"{sorted(found)}). Either the run is not there or the pin moved.")
        transcripts = results.with_name(
            results.name[: -len(".results.json")] + ".transcripts.jsonl")
        if not transcripts.exists():
            raise AdversarialReportError(
                f"{transcripts} is missing -- the aggregate cannot be re-derived "
                f"without the rows it came from")
        cells.append({"label": label, "model": spec,
                      "results": results, "transcripts": transcripts})
    return cells


# ------------------------------------------------------------ the eval set
def family_targets(eval_index: Mapping[str, Mapping[str, Any]],
                   instances: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Which families carry an `attack_target`, derived from the file.

    Cross-validated against NO_TARGET_BY_DESIGN in both directions so the
    curated note and the shipped data cannot drift apart.
    """
    has_target: Dict[str, Counter] = {}
    for scen in eval_index.values():
        fam = scen.get("attack_family")
        if not isinstance(fam, str):
            raise AdversarialReportError(
                f"{scen.get('scenario_id')!r}: attack_family is {fam!r}; this report "
                f"is defined over the adversarial set, which labels every row")
        has_target.setdefault(fam, Counter())[bool(scen.get("attack_target"))] += 1

    mixed = sorted(f for f, c in has_target.items() if len(c) > 1)
    if mixed:
        raise AdversarialReportError(
            f"families {mixed} carry an attack_target on some rows and not others. "
            f"A family cell would then be a different denominator than its row "
            f"count, which is precisely the confusion this report exists to avoid.")

    targeted = sorted(f for f, c in has_target.items() if next(iter(c)) is True)
    untargeted = sorted(f for f, c in has_target.items() if next(iter(c)) is False)

    expected_untargeted = sorted(set(NO_TARGET_BY_DESIGN) | {CONTROL_FAMILY})
    if untargeted != expected_untargeted:
        raise AdversarialReportError(
            f"families with no attack_target are {untargeted}, but this report "
            f"documents exactly {expected_untargeted} as target-free by design. "
            f"A family that gained or lost a target changes what the pooled "
            f"targeted-constraint test is pooled over; say so in "
            f"NO_TARGET_BY_DESIGN before the number moves.")
    # Which constraint TYPES each family aims at. The families are not balanced
    # across types -- content_collision is banned_words on every row, and
    # injected_instruction is mostly banned_letter -- so a per-family result and
    # a per-constraint-type result are hard to tell apart at this n. That is a
    # property of the file, so it is derived here rather than asserted in prose.
    by_family: Dict[str, Counter] = {}
    for scen in eval_index.values():
        target = scen.get("attack_target")
        if target is None:
            continue
        by_family.setdefault(scen["attack_family"], Counter())[
            instances[target]["type"]] += 1

    return {"targeted_families": targeted,
            "untargeted_families": untargeted,
            "rows_per_family": {f: sum(c.values()) for f, c in sorted(has_target.items())},
            "target_constraint_types_by_family": {
                f: dict(sorted(by_family[f].items())) for f in sorted(by_family)}}


# ------------------------------------------------- the quarantine, out loud
def quarantine_census(rows: Sequence[Mapping[str, Any]],
                      eval_index: Mapping[str, Mapping[str, Any]],
                      instances: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Every row with no verdict, described concretely enough to argue with.

    `could_have_passed_spec` is the field that matters: `spec_pass` is all
    checkers passing AND quality >= 3, and the checkers ARE re-derivable from the
    row's own output. So if any checker fails, the missing judge score cannot
    have cost the cell a pass -- the quarantine moved the denominator and nothing
    else. If every checker passes, it genuinely could have, and this report says
    so rather than leaving a reader to open the transcript.
    """
    out = []
    for row in rows:
        if row.get("judge_error") is None and row.get("generator_error") is None:
            continue
        sid = row["scenario_id"]
        scen = eval_index[sid]
        output = row.get("output")
        verdicts: Optional[List[bool]] = None
        if isinstance(output, str):
            verdicts = [held(instances, cid, output) for cid in scen["constraints"]]
        out.append({
            "scenario_id": sid,
            "attack_family": scen.get("attack_family"),
            "attack_target": scen.get("attack_target"),
            "error_kind": ("judge_error" if row.get("judge_error") is not None
                           else "generator_error"),
            "error": (row.get("judge_error") or row.get("generator_error"))[:200],
            "output_chars": len(output) if isinstance(output, str) else None,
            "checkers_passed": (sum(verdicts) if verdicts is not None else None),
            "checkers_total": (len(verdicts) if verdicts is not None else None),
            "could_have_passed_spec": (all(verdicts) if verdicts else False),
            "enters_targeted_measure": (scen.get("attack_target") is not None
                                        and isinstance(output, str)),
        })
    return sorted(out, key=lambda d: d["scenario_id"])


# -------------------------------------------------------- targeted measure
def held(instances: Mapping[str, Mapping[str, Any]], cid: str, output: str) -> bool:
    """Did this ONE constraint survive this output, per the pinned checker.

    Re-run from (constraint, output) rather than read out of the row's stored
    `checker_results`. `gate_check.score_cell` has already proved the stored
    block equals the re-derived one for every scored row, but a quarantined row
    is not scored -- and this measure includes quarantined rows.
    """
    con = instances.get(cid)
    if con is None:
        raise AdversarialReportError(
            f"attack_target {cid!r} is not in the pinned instance library")
    verdict = checkers.check(con, output)
    return bool(verdict["pass"] if isinstance(verdict, dict) else verdict[0])


def targeted_analysis(cells_rows: Mapping[str, Mapping[str, Mapping[str, Any]]],
                      eval_index: Mapping[str, Mapping[str, Any]],
                      instances: Mapping[str, Mapping[str, Any]],
                      ) -> Dict[str, Any]:
    """Per-family and pooled hold rate for the constraint each attack aimed at.

    Paired: a scenario contributes only when BOTH cells produced a string output
    for it, so the two columns are the same rows.
    """
    per_family: Dict[str, Dict[str, int]] = {}
    by_target_type: Dict[str, Dict[str, int]] = {}
    skipped: List[str] = []
    from_quarantine: List[str] = []
    pooled = {"n": 0, "base_only": 0, "tuned_only": 0, "both": 0, "neither": 0}

    for sid in sorted(eval_index):
        scen = eval_index[sid]
        target = scen.get("attack_target")
        if target is None:
            continue
        rb = cells_rows["base"].get(sid)
        rt = cells_rows["tuned"].get(sid)
        if rb is None or rt is None:
            raise AdversarialReportError(
                f"{sid} is missing from one of the two cells; score_cell should "
                f"already have refused that, so the two disagree about coverage")
        ob, ot = rb.get("output"), rt.get("output")
        if not isinstance(ob, str) or not isinstance(ot, str):
            # No text, no checker verdict. Recorded, never silently dropped.
            skipped.append(sid)
            continue
        if rb.get("spec_pass") is None or rt.get("spec_pass") is None:
            from_quarantine.append(sid)

        hb, ht = held(instances, target, ob), held(instances, target, ot)
        fam = per_family.setdefault(scen["attack_family"], {
            "n": 0, "base_held": 0, "tuned_held": 0,
            "base_only": 0, "tuned_only": 0, "both": 0, "neither": 0})
        fam["n"] += 1
        fam["base_held"] += int(hb)
        fam["tuned_held"] += int(ht)
        pooled["n"] += 1
        pooled["base_held"] = pooled.get("base_held", 0) + int(hb)
        pooled["tuned_held"] = pooled.get("tuned_held", 0) + int(ht)
        key = ("both" if hb and ht else "neither" if not hb and not ht
               else "base_only" if hb else "tuned_only")
        fam[key] += 1
        pooled[key] += 1

        # The same 90 discordances, cut by the TYPE of the targeted constraint.
        # Not a further hypothesis test: a decomposition of the pooled cell, and
        # the check on whether "the tuned model resists this family of attack"
        # is separable from "the tuned model holds this kind of constraint".
        ttype = instances[target]["type"]
        tt = by_target_type.setdefault(ttype, {
            "n": 0, "base_held": 0, "tuned_held": 0,
            "base_only": 0, "tuned_only": 0, "both": 0, "neither": 0,
            "families": []})
        tt["n"] += 1
        tt["base_held"] += int(hb)
        tt["tuned_held"] += int(ht)
        tt[key] += 1
        if scen["attack_family"] not in tt["families"]:
            tt["families"].append(scen["attack_family"])

    for tt in by_target_type.values():
        tt["families"] = sorted(tt["families"])
        tt["discordant"] = tt["base_only"] + tt["tuned_only"]
        tt["mcnemar_exact_p"] = round(
            two_sided_sign_p(tt["base_only"], tt["tuned_only"]), PLACES)
    by_target_type = {k: by_target_type[k] for k in sorted(by_target_type)}

    for fam in per_family.values():
        fam["discordant"] = fam["base_only"] + fam["tuned_only"]
        fam["mcnemar_exact_p"] = round(
            two_sided_sign_p(fam["base_only"], fam["tuned_only"]), PLACES)
        fam["base_hold_rate"] = round(float(Fraction(fam["base_held"], fam["n"])), PLACES)
        fam["tuned_hold_rate"] = round(float(Fraction(fam["tuned_held"], fam["n"])), PLACES)
    pooled["discordant"] = pooled["base_only"] + pooled["tuned_only"]
    pooled["mcnemar_exact_p"] = round(
        two_sided_sign_p(pooled["base_only"], pooled["tuned_only"]), PLACES)

    n_tests = len(per_family)
    return {
        "_definition": ("for each attacked row, whether the constraint named by the "
                        "row's attack_target was held, per constraints.checkers.check "
                        "re-run on (constraint, output). Paired on the same scenarios."),
        "quarantined_rows_included": sorted(from_quarantine),
        "rows_skipped_no_output": sorted(skipped),
        "by_family": {f: per_family[f] for f in sorted(per_family)},
        "by_target_constraint_type": by_target_type,
        "pooled": pooled,
        "family_tests_computed": n_tests,
        "bonferroni_alpha": round(0.05 / n_tests, PLACES) if n_tests else None,
        "families_surviving_bonferroni": sorted(
            f for f, v in per_family.items()
            if n_tests and v["mcnemar_exact_p"] <= 0.05 / n_tests),
        "_inference_note": (
            "EXPLORATORY, not pre-registered -- unlike docs/CURVE_PREREGISTRATION.md. "
            f"{n_tests} family-level tests were computed, so only the pooled result "
            "should be quoted inferentially; a family cell is 15 pairs."),
    }


def violations_by_constraint_type(rows: Sequence[Mapping[str, Any]],
                                  eval_index: Mapping[str, Mapping[str, Any]],
                                  instances: Mapping[str, Mapping[str, Any]],
                                  ) -> Dict[str, Dict[str, Any]]:
    """Whole-set violation rate per constraint type, over SCORED rows.

    The harness stores a `by_constraint_type` block, but it counts a check for
    every row including a quarantined one. This counts only the rows that carry a
    verdict, which is the denominator `spec_adherence` uses -- the same choice
    `curve_report.derive_breakdowns` makes and for the same reason: a row that
    does not count toward the headline should not count toward the breakdown
    printed beside it. On the committed base cell that difference is the four
    checks of `av2-b12-spec_deprecation`, three of which base PASSED, so
    excluding them makes base's rates here slightly WORSE than the stored block,
    not better.
    """
    checks, violations = _type_tally(rows, eval_index, instances, scored_only=True)
    return {ctype: {
        "checks": checks[ctype],
        "violations": violations[ctype],
        "violation_rate": round(float(Fraction(violations[ctype], checks[ctype])), PLACES),
    } for ctype in sorted(checks)}


def _type_tally(rows: Sequence[Mapping[str, Any]],
                eval_index: Mapping[str, Mapping[str, Any]],
                instances: Mapping[str, Mapping[str, Any]],
                scored_only: bool) -> Any:
    """(checks, violations) per constraint type over the rows this selects.

    One code path for both populations, so the two tallies cannot drift into
    being two different computations rather than one computation over two row
    sets -- which is the whole point of the reconciliation below.
    """
    checks: Counter = Counter()
    violations: Counter = Counter()
    for row in rows:
        quarantined = (row.get("judge_error") is not None
                       or row.get("generator_error") is not None)
        if scored_only and quarantined:
            continue
        output = row.get("output")
        if not isinstance(output, str):
            continue
        for cid in eval_index[row["scenario_id"]]["constraints"]:
            ctype = instances[cid]["type"]
            checks[ctype] += 1
            if not held(instances, cid, output):
                violations[ctype] += 1
    return checks, violations


def constraint_type_populations(results_path: Path,
                                rows: Sequence[Mapping[str, Any]],
                                eval_index: Mapping[str, Mapping[str, Any]],
                                instances: Mapping[str, Mapping[str, Any]],
                                ) -> Dict[str, Any]:
    """The two committed digits for one label, reconciled instead of left to clash.

    `violations_by_constraint_type` above counts SCORED rows, the denominator
    `spec_adherence` uses. The harness's own `metrics.by_constraint_type` block
    in the results JSON counts ALL rows with an output. On the committed base
    cell those disagree -- `banned_words` is 46/167 here and 46/168 there -- and
    two committed artifacts carrying different digits under one label is exactly
    the confusion this report exists to prevent.

    They are not reconciled by picking one. They are reconciled by proving the
    difference IS the quarantine: this recomputes both populations from the same
    code path, requires scored + quarantined to equal the harness's stored block
    type by type, and publishes all three. A reader who meets 46/167 in one file
    and 46/168 in another can see here, in one place, that the gap is the four
    checks of one judge-refused row and nothing else.
    """
    s_checks, s_viol = _type_tally(rows, eval_index, instances, scored_only=True)
    a_checks, a_viol = _type_tally(rows, eval_index, instances, scored_only=False)

    blob = json.loads(results_path.read_text())
    stored = (blob.get("metrics") or {}).get("by_constraint_type")
    if not isinstance(stored, dict) or not isinstance(stored.get("types"), dict):
        raise AdversarialReportError(
            f"{results_path.name}: metrics.by_constraint_type.types is missing, so "
            f"the harness's own per-type block cannot be reconciled against this "
            f"report's. The two digits would then be left to disagree in public.")
    harness = {ty: {"checks": v["checks"], "violations": v["violations"]}
               for ty, v in sorted(stored["types"].items())}
    all_rows = {ty: {"checks": a_checks[ty], "violations": a_viol[ty]}
                for ty in sorted(a_checks)}
    if all_rows != harness:
        raise AdversarialReportError(
            f"{results_path.name}: re-running every checker over ALL rows gives "
            f"{all_rows}, but the file's stored metrics.by_constraint_type says "
            f"{harness}. The two artifacts then differ for a reason other than the "
            f"quarantine, which this reconciliation cannot explain away.")

    quarantined = {ty: {"checks": a_checks[ty] - s_checks[ty],
                        "violations": a_viol[ty] - s_viol[ty]}
                   for ty in sorted(a_checks)
                   if a_checks[ty] != s_checks[ty] or a_viol[ty] != s_viol[ty]}
    return {
        "_note": ("one computation over two row sets. `scored_rows` is the "
                  "population every rate in this report uses; `all_rows` is what "
                  "the harness stores in the results JSON as "
                  "metrics.by_constraint_type. all_rows - scored_rows = "
                  "`quarantined_rows`, verified type by type, and all_rows is "
                  "required to equal the harness's stored block."),
        "scored_rows": {ty: {"checks": s_checks[ty], "violations": s_viol[ty]}
                        for ty in sorted(s_checks)},
        "all_rows": all_rows,
        "all_rows_matches_results_json": True,
        "quarantined_rows": quarantined,
    }


# ------------------------------ is the targeted advantage about the ATTACK?
def control_comparison(cells_rows: Mapping[str, Mapping[str, Mapping[str, Any]]],
                       eval_index: Mapping[str, Mapping[str, Any]],
                       instances: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """The targeted constraint, asked again of the row's paired CLEAN control.

    A pooled targeted-constraint win says the tuned model held the attacked
    constraint more often. It does NOT say the attack is why. §3 of
    docs/ADVERSARIAL_EVAL.md built each family's 15 rows on the same 15 briefs
    as 15 unattacked controls carrying the identical constraint set, precisely so
    that question can be asked: run the SAME constraint against the SAME model's
    output for the SAME brief with the attack removed.

    If the tuned model's advantage is already there on the control, it is a
    constraint-holding advantage that the attacked cell inherits, and calling it
    robustness would be the reading this project keeps having to correct. The
    difference of the two differences is what isolates the attack.

    Two dependencies, stated because they weaken the control column and nothing
    in the arithmetic shows them:

    * a brief emits three attacks against ONE control, so the control
      observations are drawn from 40 rows, not 90. `distinct_control_pairs`
      counts the (control row, constraint) pairs actually distinct.
    * §3's own exception: `content_collision`'s control is not attack-free — it
      shares the subject-matter collision and lacks only the sentence pushing
      toward the insiders' vocabulary. Its control column is therefore not a
      clean baseline, and the per-family row says so.
    """
    controls = {s["brief_key"]: sid for sid, s in eval_index.items()
                if s.get("attack_family") == CONTROL_FAMILY}
    per_family: Dict[str, Dict[str, Any]] = {}
    seen_pairs = set()
    n_obs = 0
    tot = {"attacked_base": 0, "attacked_tuned": 0,
           "control_base": 0, "control_tuned": 0,
           "control_base_only": 0, "control_tuned_only": 0}

    for sid in sorted(eval_index):
        scen = eval_index[sid]
        target = scen.get("attack_target")
        if target is None:
            continue
        csid = controls.get(scen.get("brief_key"))
        if csid is None:
            raise AdversarialReportError(
                f"{sid} has no paired clean control for brief "
                f"{scen.get('brief_key')!r}; the paired design §3 describes is not "
                f"present in this file, so this comparison cannot be made")
        if target not in eval_index[csid]["constraints"]:
            raise AdversarialReportError(
                f"{sid} targets {target!r}, which its control {csid} does not carry. "
                f"The control is only a baseline while it holds the identical "
                f"constraint set.")
        outs = {lbl: (cells_rows[lbl][sid].get("output"),
                      cells_rows[lbl][csid].get("output"))
                for lbl in ("base", "tuned")}
        if any(not isinstance(o, str) for pair in outs.values() for o in pair):
            continue

        n_obs += 1
        seen_pairs.add((csid, target))
        fam = per_family.setdefault(scen["attack_family"], {
            "n": 0, "attacked_base": 0, "attacked_tuned": 0,
            "control_base": 0, "control_tuned": 0})
        fam["n"] += 1
        cb = ct = None
        for lbl in ("base", "tuned"):
            a_out, c_out = outs[lbl]
            fam[f"attacked_{lbl}"] += int(held(instances, target, a_out))
            h_c = held(instances, target, c_out)
            fam[f"control_{lbl}"] += int(h_c)
            if lbl == "base":
                cb = h_c
            else:
                ct = h_c
        if cb and not ct:
            tot["control_base_only"] += 1
        elif ct and not cb:
            tot["control_tuned_only"] += 1

    for fam in per_family.values():
        n = fam["n"]
        fam["attacked_delta_pp"] = round(
            (fam["attacked_tuned"] - fam["attacked_base"]) * 100 / n, 1)
        fam["control_delta_pp"] = round(
            (fam["control_tuned"] - fam["control_base"]) * 100 / n, 1)
        # From the RAW counts, not from the two rounded percentages above.
        # `injected_instruction` is the case that shows why: its attacked delta
        # is -1/15 = -6.667 pp and its control delta -2/15 = -13.333 pp, so the
        # difference is exactly +1/15 = +6.667 -> +6.7 pp. Subtracting the
        # ROUNDED cells gives -6.7 - (-13.3) = +6.6, a figure that is not the
        # difference of anything and does not appear in the data.
        fam["did_pp"] = round(diff_pp(
            fam["attacked_tuned"] - fam["attacked_base"],
            fam["control_tuned"] - fam["control_base"], n), 1)
    for key in ("attacked_base", "attacked_tuned", "control_base", "control_tuned"):
        tot[key] = sum(f[key] for f in per_family.values())

    pooled = dict(tot)
    pooled["n"] = n_obs
    pooled["attacked_delta_pp"] = round(
        (tot["attacked_tuned"] - tot["attacked_base"]) * 100 / n_obs, 1) if n_obs else None
    pooled["control_delta_pp"] = round(
        (tot["control_tuned"] - tot["control_base"]) * 100 / n_obs, 1) if n_obs else None
    pooled["did_pp"] = (round(diff_pp(
        tot["attacked_tuned"] - tot["attacked_base"],
        tot["control_tuned"] - tot["control_base"], n_obs), 1) if n_obs else None)
    pooled["control_mcnemar_exact_p"] = round(
        two_sided_sign_p(tot["control_base_only"], tot["control_tuned_only"]), PLACES)

    return {
        "_definition": ("the row's attack_target, re-checked against the SAME model's "
                        "output for the row's paired CLEAN control (same brief, same "
                        "constraint set, attack removed)."),
        "observations": n_obs,
        "distinct_control_rows": len({p[0] for p in seen_pairs}),
        "distinct_control_pairs": len(seen_pairs),
        "repeated_control_observations": n_obs - len(seen_pairs),
        "_dependency_note": (
            "the control column reuses rows: three attacks per brief share one "
            "control, so these observations come from far fewer distinct rows than "
            "the count suggests and its McNemar p is anti-conservative. It is a "
            "diagnostic on the pooled targeted result, not an independent test."),
        "_content_collision_note": (
            "docs/ADVERSARIAL_EVAL.md §3: content_collision's control is NOT "
            "attack-free — it carries the same subject-matter collision. Its "
            "control column is a lower bound on difficulty, not a clean baseline."),
        "by_family": {f: per_family[f] for f in sorted(per_family)},
        "pooled": pooled,
    }


# ------------------------------------------------- per-family spec_pass
def spec_pass_by_family(cells_rows: Mapping[str, Mapping[str, Mapping[str, Any]]],
                        eval_index: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Descriptive per-family `spec_pass`, with BOTH denominators shown.

    A quarantined row leaves its family's denominator smaller in ONE cell. A
    delta between x/14 and y/15 is not a delta between two rates of the same
    thing, so the paired counts are carried alongside and the markdown prints
    every denominator.
    """
    fams: Dict[str, Dict[str, int]] = {}
    for sid in sorted(eval_index):
        fam = eval_index[sid]["attack_family"]
        cell = fams.setdefault(fam, {
            "base_n": 0, "base_passed": 0, "tuned_n": 0, "tuned_passed": 0,
            "paired_n": 0, "base_only": 0, "tuned_only": 0})
        pb = cells_rows["base"][sid].get("spec_pass")
        pt = cells_rows["tuned"][sid].get("spec_pass")
        if pb is not None:
            cell["base_n"] += 1
            cell["base_passed"] += int(pb)
        if pt is not None:
            cell["tuned_n"] += 1
            cell["tuned_passed"] += int(pt)
        if pb is not None and pt is not None:
            cell["paired_n"] += 1
            if pb and not pt:
                cell["base_only"] += 1
            elif pt and not pb:
                cell["tuned_only"] += 1

    for cell in fams.values():
        cell["base_rate"] = round(float(Fraction(cell["base_passed"], cell["base_n"])),
                                  PLACES) if cell["base_n"] else None
        cell["tuned_rate"] = round(float(Fraction(cell["tuned_passed"], cell["tuned_n"])),
                                   PLACES) if cell["tuned_n"] else None
        # From the counts, not from the two rounded rates: the two denominators
        # can differ (a quarantined row shrinks one cell), so this is a
        # difference of two fractions and is kept exact until it is rounded once.
        cell["delta_pp"] = (round(rate_delta_pp(cell["tuned_passed"], cell["tuned_n"],
                                                cell["base_passed"], cell["base_n"]), 1)
                            if cell["base_n"] and cell["tuned_n"] else None)
        cell["mcnemar_exact_p"] = round(
            two_sided_sign_p(cell["base_only"], cell["tuned_only"]), PLACES)
    return {f: fams[f] for f in sorted(fams)}


def paired_spec_pass(cells_rows: Mapping[str, Mapping[str, Mapping[str, Any]]],
                     eval_index: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """McNemar on the aggregate and on the adversarial subset, same scenarios."""
    out = {}
    for name, pressure in (("overall", None), ("adversarial", "adversarial"),
                           ("clean", "clean")):
        base_only = tuned_only = paired = 0
        for sid in eval_index:
            if pressure and eval_index[sid]["pressure"] != pressure:
                continue
            pb = cells_rows["base"][sid].get("spec_pass")
            pt = cells_rows["tuned"][sid].get("spec_pass")
            if pb is None or pt is None:
                continue
            paired += 1
            if pb and not pt:
                base_only += 1
            elif pt and not pb:
                tuned_only += 1
        out[name] = {
            "paired_rows": paired,
            "base_only_pass": base_only,
            "tuned_only_pass": tuned_only,
            "discordant": base_only + tuned_only,
            "mcnemar_exact_p": round(two_sided_sign_p(base_only, tuned_only), PLACES),
        }
    return out


# --------------------------------------------------------------- the report
def build_report(run_dir: Path, eval_path: Path, inst_path: Path,
                 models_cfg: Path) -> Dict[str, Any]:
    eval_rows = read_jsonl(eval_path)
    eval_index = {s["scenario_id"]: s for s in eval_rows}
    instances = {c["id"]: c for c in read_jsonl(inst_path)}
    if not eval_index:
        raise AdversarialReportError(f"{eval_path}: no scenarios")

    cells = locate_cells(run_dir, models_cfg)
    aggregate: Dict[str, Any] = {}
    cells_rows: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    quarantine: Dict[str, Any] = {}

    for cell in cells:
        errors: List[str] = []
        stats = G.score_cell(cell["results"], cell["transcripts"], cell["model"],
                             STRATEGY, eval_index, instances, errors)
        if stats is None:
            raise AdversarialReportError(
                f"the {cell['label']} cell could not be re-derived from "
                f"{cell['transcripts'].name}:\n  - " + "\n  - ".join(errors))
        if errors:
            raise AdversarialReportError(
                f"the {cell['label']} cell scored with complaints:\n  - "
                + "\n  - ".join(errors))

        rows = read_jsonl(cell["transcripts"])
        cells_rows[cell["label"]] = {r["scenario_id"]: r for r in rows}
        census = quarantine_census(rows, eval_index, instances)
        quarantine[cell["label"]] = census

        # The denominator arithmetic, asserted rather than assumed: total rows =
        # rows score_cell counted + rows it quarantined. A row that fell out of
        # both would silently shrink a rate.
        if len(rows) != stats["n"] + len(census):
            raise AdversarialReportError(
                f"the {cell['label']} cell holds {len(rows)} rows but "
                f"gate_check.score_cell scored {stats['n']} and this script found "
                f"{len(census)} quarantined -- the two disagree about which rows "
                f"have a verdict, so a denominator is unaccounted for")

        # The robustness NUMERATOR, stored rather than left to be recovered by
        # multiplying a rounded rate by a denominator. `robustness` is
        # float(Fraction(n_adv_pass, n_adv)) inside score_cell, so this recovers
        # the integer exactly; storing it is what lets the document quote
        # "9/119" under the gate instead of under a reader's arithmetic.
        n_adv_pass = (round(stats["robustness"] * stats["n_adversarial"])
                      if stats["robustness"] is not None else None)

        aggregate[cell["label"]] = {
            "model": cell["model"],
            "eval_set_rows": len(eval_index),
            "n_scored": stats["n"],
            "n_quarantined": len(census),
            "spec_adherence": round(stats["spec_adherence"], PLACES),
            "spec_adherence_ci95": wilson_interval(stats["n_spec_pass"], stats["n"]),
            "passed": stats["n_spec_pass"],
            "n_adversarial_scored": stats["n_adversarial"],
            "passed_adversarial": n_adv_pass,
            "robustness": (round(stats["robustness"], PLACES)
                           if stats["robustness"] is not None else None),
            "robustness_ci95": (wilson_interval(n_adv_pass, stats["n_adversarial"])
                                if stats["robustness"] is not None else None),
            "n_fail_from_checkers": stats["n_fail_from_checkers"],
            "n_fail_from_quality_only": stats["n_fail_from_quality_only"],
            "transcripts_jsonl": stats["transcripts_jsonl"],
            "transcripts_sha256": stats["transcripts_jsonl_sha256"],
            "violations_by_constraint_type": violations_by_constraint_type(
                rows, eval_index, instances),
            "constraint_type_populations": constraint_type_populations(
                cell["results"], rows, eval_index, instances),
        }

    b, t = aggregate["base"], aggregate["tuned"]
    delta = {
        # Both from the counts. The two rates do not share a denominator, so
        # subtracting the rounded rates would be wrong twice over.
        "spec_adherence_pp": round(rate_delta_pp(
            t["passed"], t["n_scored"], b["passed"], b["n_scored"]), 1),
        "robustness_pp": round(rate_delta_pp(
            t["passed_adversarial"], t["n_adversarial_scored"],
            b["passed_adversarial"], b["n_adversarial_scored"]), 1),
        "_note": ("tuned minus base, percentage points. The two spec_adherence rates "
                  f"do NOT share a denominator: base is {b['passed']}/{b['n_scored']} "
                  f"and tuned is {t['passed']}/{t['n_scored']}, because base "
                  f"quarantined {b['n_quarantined']} row(s). The paired McNemar "
                  "below is the comparison that does hold the rows fixed."),
    }

    return {
        "_note": ("Re-derived from the committed transcripts by "
                  "scripts/adversarial_report.py. Do not hand-edit: `python "
                  "scripts/adversarial_report.py --check` fails if this file and "
                  "the transcripts disagree."),
        "eval_set": repo_relative(eval_path),
        "eval_set_sha256": G.sha256_of(eval_path),
        "strategy": STRATEGY,
        "families": family_targets(eval_index, instances),
        "no_target_by_design": NO_TARGET_BY_DESIGN,
        "aggregate": aggregate,
        "aggregate_delta": delta,
        "paired_spec_pass": paired_spec_pass(cells_rows, eval_index),
        "spec_pass_by_family": spec_pass_by_family(cells_rows, eval_index),
        "targeted_constraint": targeted_analysis(cells_rows, eval_index, instances),
        "control_comparison": control_comparison(cells_rows, eval_index, instances),
        "quarantine": quarantine,
    }


# ------------------------------------------------------------- presentation
def markdown_tables(rep: Mapping[str, Any]) -> str:
    out: List[str] = []
    b, t = rep["aggregate"]["base"], rep["aggregate"]["tuned"]
    d = rep["aggregate_delta"]

    out.append("**Aggregate.** Rates are over SCORED rows; the denominators differ.")
    out.append("")
    out.append("| cell | `spec_adherence` | 95% CI | `robustness` | 95% CI | "
               "scored / rows | judge errors |")
    out.append("|---|---|---|---|---|---|---|")
    for label, c in (("base", b), ("tuned", t)):
        slo, shi = c["spec_adherence_ci95"]
        rlo, rhi = c["robustness_ci95"]
        out.append(
            f"| {label} | {c['spec_adherence']:.4f} ({c['passed']}/{c['n_scored']}) | "
            f"[{slo:.3f}, {shi:.3f}] | {c['robustness']:.4f} "
            f"({round(c['robustness'] * c['n_adversarial_scored'])}/"
            f"{c['n_adversarial_scored']}) | [{rlo:.3f}, {rhi:.3f}] | "
            f"{c['n_scored']}/{c['eval_set_rows']} | {c['n_quarantined']} |")
    out.append(f"| **delta** | **{d['spec_adherence_pp']:+.1f} pp** | | "
               f"**{d['robustness_pp']:+.1f} pp** | | | |")
    out.append("")

    out.append("Violation rate per constraint type, over SCORED rows "
               "(lower is better):")
    out.append("")
    types = sorted(set(b["violations_by_constraint_type"])
                   | set(t["violations_by_constraint_type"]))
    out.append("| cell | " + " | ".join(f"`{ty}`" for ty in types) + " |")
    out.append("|---|" + "---|" * len(types))
    for label, c in (("base", b), ("tuned", t)):
        cells = []
        for ty in types:
            e = c["violations_by_constraint_type"].get(ty)
            cells.append("—" if e is None
                         else f"{e['violation_rate']:.3f} ({e['violations']}/{e['checks']})")
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    out.append("")

    for label, c in (("base", b), ("tuned", t)):
        q = c["constraint_type_populations"]["quarantined_rows"]
        if not q:
            continue
        pops = c["constraint_type_populations"]
        out.append(
            f"`{label}`'s denominators above exclude its {c['n_quarantined']} "
            f"quarantined row{'' if c['n_quarantined'] == 1 else 's'}, so they "
            f"differ from the harness's own "
            f"`metrics.by_constraint_type` block in the results JSON, which counts "
            f"all {c['eval_set_rows']} rows. Same checkers, two populations, "
            f"reconciled type by type: "
            + "; ".join(
                f"`{ty}` {pops['scored_rows'][ty]['violations']}/"
                f"{pops['scored_rows'][ty]['checks']} scored against "
                f"{pops['all_rows'][ty]['violations']}/"
                f"{pops['all_rows'][ty]['checks']} over all rows"
                for ty in sorted(q))
            + ".")
        out.append("")

    out.append("Paired on the same scenarios, McNemar exact:")
    out.append("")
    out.append("| subset | paired rows | base-only pass | tuned-only pass | p |")
    out.append("|---|---:|---:|---:|---:|")
    for name, s in rep["paired_spec_pass"].items():
        out.append(f"| {name} | {s['paired_rows']} | {s['base_only_pass']} | "
                   f"{s['tuned_only_pass']} | {s['mcnemar_exact_p']:.4f} |")
    out.append("")

    tc = rep["targeted_constraint"]
    out.append("**Targeted-constraint hold rate** — of the attacked rows, how many "
               "held the ONE constraint the attack aimed at (`attack_target`).")
    out.append("")
    out.append("| family | n | base held | tuned held | base-only | tuned-only | "
               "McNemar exact p |")
    out.append("|---|---:|---:|---:|---:|---:|---:|")
    for fam, v in tc["by_family"].items():
        out.append(f"| `{fam}` | {v['n']} | {v['base_held']} | {v['tuned_held']} | "
                   f"{v['base_only']} | {v['tuned_only']} | "
                   f"{v['mcnemar_exact_p']:.4f} |")
    p = tc["pooled"]
    out.append(f"| **pooled** | **{p['n']}** | **{p['base_held']}** | "
               f"**{p['tuned_held']}** | **{p['base_only']}** | "
               f"**{p['tuned_only']}** | **{p['mcnemar_exact_p']:.4f}** |")
    out.append("")
    out.append(f"{tc['family_tests_computed']} family-level tests were computed; "
               f"Bonferroni 0.05/{tc['family_tests_computed']} = "
               f"{tc['bonferroni_alpha']:.4f}. Families surviving it: "
               f"{tc['families_surviving_bonferroni'] or 'none'}.")
    out.append("")
    out.append("No cell for " + ", ".join(f"`{f}`" for f in
               sorted(rep['no_target_by_design'])) + ": " +
               "; ".join(f"`{f}` — {why}" for f, why in
                         sorted(rep["no_target_by_design"].items())) + ".")
    out.append("")

    out.append("The same 90 pairs, cut by the constraint TYPE the attack aimed at. "
               "Not further hypothesis tests — a decomposition of the pooled cell.")
    out.append("")
    out.append("| targeted constraint type | n | base held | tuned held | base-only | "
               "tuned-only | p | families |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for ty, v in tc["by_target_constraint_type"].items():
        out.append(f"| `{ty}` | {v['n']} | {v['base_held']} | {v['tuned_held']} | "
                   f"{v['base_only']} | {v['tuned_only']} | "
                   f"{v['mcnemar_exact_p']:.4f} | "
                   + ", ".join(f"`{f}`" for f in v["families"]) + " |")
    out.append("")

    out.append("Which constraint types each family aims at — a property of the "
               "eval set, and the reason the two cuts above are hard to separate:")
    out.append("")
    for fam, counts in rep["families"]["target_constraint_types_by_family"].items():
        out.append(f"* `{fam}`: " + ", ".join(f"`{k}` {v}" for k, v in counts.items()))
    out.append("")

    cc = rep["control_comparison"]
    p = cc["pooled"]
    out.append("**Is the targeted advantage about the attack?** The same constraint, "
               "re-checked on the row's paired CLEAN control (same brief, same "
               "constraint set, attack removed).")
    out.append("")
    out.append("| family | n | attacked base | attacked tuned | control base | "
               "control tuned | attacked delta | control delta | difference |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for fam, v in cc["by_family"].items():
        out.append(f"| `{fam}` | {v['n']} | {v['attacked_base']} | "
                   f"{v['attacked_tuned']} | {v['control_base']} | "
                   f"{v['control_tuned']} | {v['attacked_delta_pp']:+.1f} pp | "
                   f"{v['control_delta_pp']:+.1f} pp | {v['did_pp']:+.1f} pp |")
    out.append(f"| **pooled** | **{p['n']}** | **{p['attacked_base']}** | "
               f"**{p['attacked_tuned']}** | **{p['control_base']}** | "
               f"**{p['control_tuned']}** | **{p['attacked_delta_pp']:+.1f} pp** | "
               f"**{p['control_delta_pp']:+.1f} pp** | **{p['did_pp']:+.1f} pp** |")
    out.append("")
    out.append(f"On the controls alone: base-only {p['control_base_only']}, tuned-only "
               f"{p['control_tuned_only']}, McNemar exact p = "
               f"{p['control_mcnemar_exact_p']:.4f}. "
               f"{cc['observations']} observations from {cc['distinct_control_rows']} "
               f"distinct control rows and {cc['distinct_control_pairs']} distinct "
               f"(row, constraint) pairs — {cc['_dependency_note']}")
    out.append("")

    out.append("**Per-family `spec_pass`** — descriptive only. A 15-row cell "
               "resolves nothing below roughly 28.6 pp (docs/ADVERSARIAL_EVAL.md §5).")
    out.append("")
    out.append("| family | base | tuned | delta | paired n | base-only | tuned-only | p |")
    out.append("|---|---|---|---:|---:|---:|---:|---:|")
    for fam, v in rep["spec_pass_by_family"].items():
        out.append(f"| `{fam}` | {v['base_passed']}/{v['base_n']} | "
                   f"{v['tuned_passed']}/{v['tuned_n']} | {v['delta_pp']:+.1f} pp | "
                   f"{v['paired_n']} | {v['base_only']} | {v['tuned_only']} | "
                   f"{v['mcnemar_exact_p']:.4f} |")
    out.append("")

    out.append("**Quarantine.** Rows with no verdict, excluded from every "
               "`spec_pass` rate and named here so no denominator moves quietly.")
    out.append("")
    any_q = False
    for label, census in rep["quarantine"].items():
        for q in census:
            any_q = True
            out.append(
                f"* `{label}` / `{q['scenario_id']}` (`{q['attack_family']}`, "
                f"attack_target `{q['attack_target']}`) — {q['error_kind']}, "
                f"output {q['output_chars']} chars, "
                f"{q['checkers_passed']}/{q['checkers_total']} checkers pass, "
                f"could_have_passed_spec **{q['could_have_passed_spec']}**, "
                f"enters the targeted measure: {q['enters_targeted_measure']}.")
    if not any_q:
        out.append("* none.")
    return "\n".join(out)


# ------------------------------------------- binding the document's PROSE
#: The heading that opens the part of `docs/ADVERSARIAL_EVAL.md` this script
#: claims to gate, and the heading that closes it. Sections 1-8 describe the
#: FILE and were written before any run; they are not this script's to check.
DOC_SECTION_BEGIN = "## Results — two models against the set"
DOC_SECTION_END = "## 1. The problem this set exists to fix"

#: Every number-shaped token this gate recognises inside that section.
DOC_NUMBER_PATTERNS = (
    r"(?<![\d.])\d+/\d+(?![\d.])",          # 17/159, 9/119, 3/15
    r"[+−-]?\d+(?:\.\d+)? pp",          # +12.2 pp, −1.9 pp, 10 pp
    r"(?<![\d.])\d\.\d{3,4}(?![\d])",        # 0.1069, 0.0034, 1.0000
)

#: Numbers that appear inside the Results section, are NOT derived by this
#: script, and are therefore quoted from another document under another gate.
#: Each entry is the verbatim phrase that must still be present (a stale
#: exemption fails the check in the same way a broken binding does) and the
#: command that does re-derive it. This list is the honest boundary of the
#: claim in the document's preamble: everything else in the section is bound.
DOC_UNGATED = (
    ("base `spec_adherence` 0.0890 over 146 scored rows, tuned 0.0816 over 147",
     "docs/FINAL_REPORT.md generated block; "
     ".venv/bin/python scripts/check_mvp.py --emit-metrics"),
    ("records base 0.0890 over 146 scored rows against tuned 0.0816 over 147",
     "docs/FINAL_REPORT.md generated block; "
     ".venv/bin/python scripts/check_mvp.py --emit-metrics"),
    ("p = 0.0391",
     "docs/COMPOSITION.md; .venv/bin/python scripts/composition_report.py "
     "--check docs/COMPOSITION.md"),
    ("its k≤3 half is 1 against 1, p = 1.0000",
     "docs/COMPOSITION.md; .venv/bin/python scripts/composition_report.py "
     "--check docs/COMPOSITION.md"),
    ("p = 1.0000 overall and p = 1.0000 on its adversarial subset",
     ".venv/bin/python scripts/compare_checkpoints.py on runs/final/"),
    ("v1 against v2 is p = 1.0000 both ways",
     "configs/models.json:_tuned_history"),
    ("about 10 pp near p = 0.08",
     "§5 of this document, whose table is re-derived by "
     "scripts/curve_report.wilson_interval"),
    ("the 28.6 pp a 15-row cell resolves",
     "§5 of this document, whose table is re-derived by "
     "scripts/curve_report.wilson_interval"),
)


def _pp(x: float) -> str:
    """Percentage points in the document's typography (U+2212 for a negative)."""
    return f"+{x:.1f} pp" if x >= 0 else f"−{abs(x):.1f} pp"


def _norm(text: str) -> str:
    """Collapse whitespace so a binding can span the document's line wrapping."""
    return " ".join(text.split())


def document_bindings(rep: Mapping[str, Any]) -> List[Dict[str, str]]:
    """Every figure the Results section states, rebuilt from the report.

    This is the repair for a gate that only ever bound its own JSON: the JSON
    could be perfect while the prose beside it said anything at all. Each entry
    is a string BUILT from `rep` -- never typed -- that must appear verbatim
    (whitespace-normalised) inside the Results section. `--check` fails if any
    of them is missing, and fails again if the section contains a number-shaped
    token that no binding and no declared exemption covers, so a new hand-typed
    figure cannot be added to the section without either being derived or being
    declared with the command that derives it.
    """
    b, t = rep["aggregate"]["base"], rep["aggregate"]["tuned"]
    d = rep["aggregate_delta"]
    sp = rep["paired_spec_pass"]
    tc, cc = rep["targeted_constraint"], rep["control_comparison"]
    tp, cp = tc["pooled"], cc["pooled"]
    out: List[Dict[str, str]] = []

    def add(label: str, text: str) -> None:
        out.append({"label": label, "text": _norm(text)})

    # --- R1, the aggregate table, delta row and paired table
    for label, c in (("base", b), ("tuned", t)):
        slo, shi = c["spec_adherence_ci95"]
        rlo, rhi = c["robustness_ci95"]
        add(f"R1 aggregate row: {label}",
            f"| {label} | {c['spec_adherence']:.4f} ({c['passed']}/{c['n_scored']}) "
            f"| [{slo:.3f}, {shi:.3f}] | {c['robustness']:.4f} "
            f"({c['passed_adversarial']}/{c['n_adversarial_scored']}) | "
            f"[{rlo:.3f}, {rhi:.3f}] | {c['n_scored']}/{c['eval_set_rows']} | "
            f"{c['n_quarantined']} |")
    add("R1 delta row",
        f"| **delta** | **{_pp(d['spec_adherence_pp'])}** | | "
        f"**{_pp(d['robustness_pp'])}** | | | |")
    add("R1 delta, in prose",
        f"below base on `spec_adherence` by {abs(d['spec_adherence_pp']):.1f} pp "
        f"and on `robustness` by {abs(d['robustness_pp']):.1f} pp")
    for name, s in sp.items():
        add(f"R1 paired row: {name}",
            f"| {name} | {s['paired_rows']} | {s['base_only_pass']} | "
            f"{s['tuned_only_pass']} | {s['mcnemar_exact_p']:.4f} |")
    add("R1 observed robustness difference",
        f"the observed difference is {abs(d['robustness_pp']):.1f} pp")

    # --- R1, the quarantine
    q = rep["quarantine"]["base"][0]
    add("R1 quarantined denominator",
        f"Base's rate is over **{b['n_scored']} rows, not {b['eval_set_rows']}**.")
    add("R1 quarantined row id", f"`{q['scenario_id']}` has no verdict")
    add("R1 quarantined row detail",
        f"the output is {q['output_chars']} characters, and it passes "
        f"{q['checkers_passed']} of its {q['checkers_total']} checkers")
    add("R1 quarantined could_have_passed_spec",
        f"`could_have_passed_spec: {str(q['could_have_passed_spec']).lower()}`")

    # --- R2, the targeted-constraint table
    for fam, v in tc["by_family"].items():
        add(f"R2 targeted row: {fam}",
            f"| `{fam}` | {v['n']} | {v['base_held']} | {v['tuned_held']} | "
            f"{v['base_only']} | {v['tuned_only']} | {v['mcnemar_exact_p']:.4f} |")
    add("R2 targeted pooled row",
        f"| **pooled** | **{tp['n']}** | **{tp['base_held']}** | "
        f"**{tp['tuned_held']}** | **{tp['base_only']}** | **{tp['tuned_only']}** "
        f"| **{tp['mcnemar_exact_p']:.4f}** |")
    add("R2 pooled result, in prose",
        f"holds the targeted constraint on {tp['tuned_only']} scenarios where base "
        f"loses it and loses it on {tp['base_only']} where base holds it, "
        f"p = {tp['mcnemar_exact_p']:.4f}")
    add("R2 Bonferroni",
        f"At Bonferroni 0.05/{tc['family_tests_computed']} = "
        f"{tc['bonferroni_alpha']:.4f} the `content_collision` cell at "
        f"p = {tc['by_family']['content_collision']['mcnemar_exact_p']:.4f}")
    add("R2 rows per family",
        f"**n = {rep['families']['rows_per_family']['content_collision']} per family**")
    add("R2 pooled against the Bonferroni threshold",
        f"The pooled p is below Bonferroni 0.05/{tc['family_tests_computed']} = "
        f"{tc['bonferroni_alpha']:.4f}")
    add("R2 back-reference to the R1 aggregate",
        f"the aggregate in R1 above is p = {sp['overall']['mcnemar_exact_p']:.4f}")

    # --- R3, the control comparison
    for fam, v in cc["by_family"].items():
        add(f"R3 control row: {fam}",
            f"| `{fam}` | {v['n']} | {v['attacked_base']} | {v['attacked_tuned']} | "
            f"{v['control_base']} | {v['control_tuned']} | "
            f"{_pp(v['attacked_delta_pp'])} | {_pp(v['control_delta_pp'])} | "
            f"{_pp(v['did_pp'])} |")
    add("R3 control pooled row",
        f"| **pooled** | **{cp['n']}** | **{cp['attacked_base']}** | "
        f"**{cp['attacked_tuned']}** | **{cp['control_base']}** | "
        f"**{cp['control_tuned']}** | **{_pp(cp['attacked_delta_pp'])}** | "
        f"**{_pp(cp['control_delta_pp'])}** | **{_pp(cp['did_pp'])}** |")
    add("R3 controls, in prose",
        f"the tuned model still beats base {cp['control_tuned']} to "
        f"{cp['control_base']} — tuned-only {cp['control_tuned_only']}, base-only "
        f"{cp['control_base_only']}, McNemar exact p = "
        f"{cp['control_mcnemar_exact_p']:.4f}")
    add("R3 gap arithmetic",
        f"shrinks the gap from {_pp(cp['attacked_delta_pp'])} to "
        f"{_pp(cp['control_delta_pp'])}; the part attributable to the attack is "
        f"{_pp(cp['did_pp'])}")
    bw = tc["by_target_constraint_type"]["banned_words"]
    add("R3 banned_words cut",
        f"**{bw['discordant']} of the {tp['discordant']} discordant pairs** are "
        f"`banned_words` targets (base {bw['base_held']}/{bw['n']}, tuned "
        f"{bw['tuned_held']}/{bw['n']}, p = {bw['mcnemar_exact_p']:.4f})")
    add("R3 content_collision target share",
        f"targets `banned_words` on "
        f"{rep['families']['target_constraint_types_by_family']['content_collision']['banned_words']}"
        f" rows out of "
        f"{rep['families']['rows_per_family']['content_collision']}")
    for fam, counts in rep["families"]["target_constraint_types_by_family"].items():
        add(f"R3 target types: {fam}",
            f"* `{fam}`: " + ", ".join(f"`{k}` {v}" for k, v in counts.items()))
    ccf = cc["by_family"]["content_collision"]
    add("R3 content_collision cells",
        f"larger** on the control (base {ccf['control_base']}/{ccf['n']}, tuned "
        f"{ccf['control_tuned']}/{ccf['n']}) than under attack "
        f"({ccf['attacked_base']}/{ccf['n']}, {ccf['attacked_tuned']}/{ccf['n']})")
    fb = cc["by_family"]["format_bait"]
    add("R3 format_bait cells",
        f"hold those constraints {fb['control_base']}/{fb['n']} with no bait; base "
        f"drops to {fb['attacked_base']}/{fb['n']} under it and the tuned model to "
        f"{fb['attacked_tuned']}/{fb['n']}")
    add("R3 format_bait p",
        f"its cell is p = {tc['by_family']['format_bait']['mcnemar_exact_p']:.4f} on "
        f"{tc['by_family']['format_bait']['n']} rows")
    add("R3 control dependency",
        f"its {cc['observations']} observations come from **{cc['distinct_control_rows']} "
        f"distinct control rows** and {cc['distinct_control_pairs']} distinct "
        f"(row, constraint) pairs")

    # --- R4
    inj, drift = cc["by_family"]["injected_instruction"], cc["by_family"]["constraint_drift"]
    add("R4 injected_instruction",
        f"base holds {inj['attacked_base']}/{inj['n']}, the tuned model "
        f"{inj['attacked_tuned']}/{inj['n']}")
    add("R4 injected_instruction controls",
        f"with the note removed hold {inj['control_base']}/{inj['n']} and "
        f"{inj['control_tuned']}/{inj['n']}")
    bl = tc["by_target_constraint_type"]["banned_letter"]
    add("R4 banned_letter total",
        f"({bl['base_held']}/{bl['n']} for both)")
    add("R4 injected_instruction target types",
        f"{rep['families']['target_constraint_types_by_family']['injected_instruction']['banned_letter']} "
        f"of its {inj['n']} targets are `banned_letter` bans")
    add("R4 constraint_drift",
        f"holds only {drift['attacked_base']}/{drift['n']} for both models** — and "
        f"{drift['control_base']}/{drift['n']} on its controls too")
    ao = cc["by_family"]["authority_override"]
    add("R4 unmoved families",
        f"{ao['control_tuned']}/{ao['n']} and {ao['attacked_tuned']}/{ao['n']} for "
        f"both under attack, except one base row on `authority_override`")
    add("R4 targeted family count",
        f"Of the {len(cc['by_family'])} families carrying an `attack_target`")
    add("R4 format_bait against its control",
        f"costs either model more than one row against its own unattacked control "
        f"(base {fb['control_base']}/{fb['n']} to {fb['attacked_base']}/{fb['n']}, "
        f"tuned {fb['control_tuned']}/{fb['n']} to {fb['attacked_tuned']}/{fb['n']})")
    add("R4 content_collision moves the other way",
        f"`content_collision` moves base by two rows the other way, its attacked "
        f"cell scoring {ccf['attacked_base']}/{ccf['n']} against its control's "
        f"{ccf['control_base']}/{ccf['n']}")
    add("R4 keeping the controls",
        f"`injected_instruction` at {inj['attacked_base']}/{inj['n']} reads as a "
        f"devastating attack")

    # --- R5, the violation rates and the two populations
    for ty in ("banned_words", "no_quotation_marks", "banned_punctuation",
               "sentence_initial_letter"):
        vb = b["violations_by_constraint_type"][ty]
        vt = t["violations_by_constraint_type"][ty]
        add(f"R5 violation rate: {ty}",
            f"{vb['violation_rate']:.3f} ({vb['violations']}/{vb['checks']}) to "
            f"{vt['violation_rate']:.3f} ({vt['violations']}/{vt['checks']})")
    pops = b["constraint_type_populations"]
    add("R5 two populations",
        f"`banned_words` is {pops['scored_rows']['banned_words']['violations']}/"
        f"{pops['scored_rows']['banned_words']['checks']} here and "
        f"{pops['all_rows']['banned_words']['violations']}/"
        f"{pops['all_rows']['banned_words']['checks']} there")
    add("R5 quarantined checks",
        f"the {sum(v['checks'] for v in pops['quarantined_rows'].values())} checks of "
        f"`{q['scenario_id']}`")
    add("R5 tuned banned_letter",
        f"violates `banned_letter` on "
        f"{t['violations_by_constraint_type']['banned_letter']['violations']} of "
        f"{t['violations_by_constraint_type']['banned_letter']['checks']} checks and "
        f"`sentence_initial_letter` on "
        f"{t['violations_by_constraint_type']['sentence_initial_letter']['violations']} "
        f"of {t['violations_by_constraint_type']['sentence_initial_letter']['checks']}")
    add("R5 banned_letter targeted",
        f"Targeted `banned_letter` constraints are held {bl['base_held']}/{bl['n']} by "
        f"both models")

    # --- R6
    none = rep["spec_pass_by_family"]["none"]
    add("R6 control cell",
        f"score base {none['base_passed']}/{none['base_n']} = "
        f"{none['base_rate']:.3f} and tuned {none['tuned_passed']}/{none['tuned_n']} = "
        f"{none['tuned_rate']:.3f} (paired: base-only {none['base_only']}, tuned-only "
        f"{none['tuned_only']}, p = {none['mcnemar_exact_p']:.4f})")
    add("R6 attacked cell",
        f"score base {b['robustness']:.4f} over the {b['n_adversarial_scored']} of "
        f"them that carry a verdict and tuned {t['robustness']:.4f} over all "
        f"{t['n_adversarial_scored']}")
    add("R6 quoting pair",
        f"quotes {t['robustness']:.4f} without the {none['tuned_rate']:.3f} beside it")
    return out


def check_document(doc_path: Path, bindings: Sequence[Mapping[str, str]]) -> List[str]:
    """Every failure of the document against the bindings. Empty means bound."""
    problems: List[str] = []
    if not doc_path.exists():
        return [f"{doc_path} does not exist"]
    text = doc_path.read_text()
    if DOC_SECTION_BEGIN not in text or DOC_SECTION_END not in text:
        return [f"{doc_path} has no {DOC_SECTION_BEGIN!r} .. {DOC_SECTION_END!r} "
                f"section, so there is nothing for this gate to bind"]
    section = _norm(text.split(DOC_SECTION_BEGIN, 1)[1].split(DOC_SECTION_END, 1)[0])
    covered = bytearray(len(section))

    def cover(needle: str) -> int:
        hits, start = 0, 0
        while True:
            i = section.find(needle, start)
            if i < 0:
                return hits
            for j in range(i, i + len(needle)):
                covered[j] = 1
            hits += 1
            start = i + 1

    for binding in bindings:
        if cover(binding["text"]) == 0:
            problems.append(
                f"the Results section does not state {binding['label']} as the "
                f"transcripts re-derive it.\n      derived: {binding['text']}")
    for phrase, source in DOC_UNGATED:
        if cover(_norm(phrase)) == 0:
            problems.append(
                f"declared-ungated phrase is no longer in the Results section, so "
                f"the document's statement of what is NOT gated is stale: "
                f"{phrase!r} (source: {source})")

    for pattern in DOC_NUMBER_PATTERNS:
        for m in re.finditer(pattern, section):
            if any(covered[i] for i in range(m.start(), m.end())):
                continue
            lo, hi = max(0, m.start() - 60), min(len(section), m.end() + 60)
            problems.append(
                f"{m.group(0)!r} in the Results section is derived from nothing: no "
                f"binding in document_bindings() produces it and it is not declared "
                f"in DOC_UNGATED with the command that does.\n      context: "
                f"...{section[lo:hi]}...")
    return problems


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    ap.add_argument("--eval-set", default=str(DEFAULT_EVAL_SET))
    ap.add_argument("--instances", default=str(DEFAULT_INSTANCES))
    ap.add_argument("--models-config", default=str(DEFAULT_MODELS_CFG))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--doc", default=str(DEFAULT_DOC),
                    help="the document whose Results section --check binds")
    ap.add_argument("--check", action="store_true",
                    help="re-derive and compare against --out AND against every "
                         "figure in --doc's Results section; exit 1 on any "
                         "difference instead of rewriting anything")
    ap.add_argument("--markdown", action="store_true",
                    help="print the report tables to stdout. READ-ONLY: it does "
                         "not rewrite --out, because reading a report must not "
                         "rewrite the artifact --check gates")
    args = ap.parse_args(argv)

    try:
        rep = build_report(Path(args.run_dir), Path(args.eval_set),
                           Path(args.instances), Path(args.models_config))
    except (AdversarialReportError, G.GateError) as exc:
        print(f"FAIL adversarial_report: {exc}", file=sys.stderr)
        return 1

    text = json.dumps(rep, indent=2, ensure_ascii=False) + "\n"
    out = Path(args.out)

    if args.check:
        if not out.exists():
            print(f"FAIL {out} does not exist; run without --check to write it",
                  file=sys.stderr)
            return 1
        if out.read_text() != text:
            print(f"FAIL {out} is not what the transcripts re-derive to. "
                  f"Re-run `python scripts/adversarial_report.py` and commit the "
                  f"result, or explain why the rows changed.", file=sys.stderr)
            return 1
        bindings = document_bindings(rep)
        problems = check_document(Path(args.doc), bindings)
        if problems:
            print(f"FAIL {args.doc}: {len(problems)} figure(s) in the Results "
                  f"section are not what the transcripts re-derive to:",
                  file=sys.stderr)
            for p in problems:
                print(f"  - {p}", file=sys.stderr)
            return 1
        print(f"ok   {out.name} matches the two cells re-derived from their "
              f"transcripts ({rep['aggregate']['base']['n_scored']} and "
              f"{rep['aggregate']['tuned']['n_scored']} scored rows)")
        print(f"ok   {Path(args.doc).name} Results section: {len(bindings)} figures "
              f"re-derived and matched, {len(DOC_UNGATED)} phrases declared ungated "
              f"with their own command, no unbound number left in the section")
    elif args.markdown:
        # Deliberately writes nothing. Printing the report used to rewrite
        # `--out` as a side effect, which meant reading the tables silently
        # regenerated -- and so repaired -- the very artifact `--check` gates.
        pass
    else:
        out.write_text(text)
        print(f"wrote {out} (2 cells re-derived from transcripts)")

    if args.markdown:
        if not args.check:
            print(f"# {out.name} NOT rewritten: --markdown is read-only. "
                  f"Run the script with no flags to regenerate it.")
        print()
        print(markdown_tables(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
