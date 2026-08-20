#!/usr/bin/env python3
"""
GATE G1 — the Prompt-Ceiling gate (docs/PIVOT_SPEC.md R6 / S1, T-4).

    python scripts/gate_check.py runs/g1
    python scripts/gate_check.py runs/g1/g1_table.json      # a file resolves to its dir

R6, verbatim:
    Proceed only if the best model x strategy cell shows fail-rate >= 0.20.
    Escalation ladder on a clean row: raise k (2->3->4), then raise adversarial
    pressure share, rerun. Hard stop + report to Hank if best-cell adherence
    >= 0.90 after the full ladder.

The ceiling claim is about FRONTIER models, so the ceiling is computed over the
frontier cells only; the prompted local base row (hf:...) is evidence that a
small model cannot already do this, not a ceiling measurement, and is classified
and reported separately. Every cell's classification is written into the report
so a reader can audit which cells fed the gate instead of inferring it.

WHY THIS IS A SCRIPT AND NOT A JUDGMENT CALL (Design: Verification strategy).
The previous target's pipeline was built downstream of an unproven ceiling. So:

  * spec_adherence is RECOMPUTED from the raw transcript rows. A stored
    aggregate is only ever compared, never trusted; disagreement is a failure.
  * checker_results are themselves RE-DERIVED from (constraint, output) by
    calling the pinned constraints.checkers.check on the row's own output, with
    the constraint ids resolved against the pinned instance library. Re-deriving
    spec_pass from a stored checker_results block only proves the row is
    self-consistent: a run whose outputs genuinely satisfy every constraint
    could fabricate its own failures and manufacture a passing gate. The stored
    verdict is COMPARED to the re-derived one, and any disagreement — pass flag
    or evidence — makes the run unratable.
  * spec_pass is recomputed from the RE-DERIVED checker verdicts and the row's
    quality_score (the pinned definition: all checkers pass AND quality >= 3).
    A row that stores spec_pass=true while carrying a failing checker fails.
    quality_score is the one programmatic input the gate cannot re-derive (it
    is a judge call that cannot be replayed offline), so it is trusted as
    stored — and the report therefore attributes every counted failure to the
    checkers or to quality alone, so a reader can see how much of a fail rate
    rests on the un-recomputable half.
  * the rows must COVER the eval set exactly - one row per scenario, no
    duplicates, no extras, no subset. Authentic, correctly-scored rows can still
    be selectively dropped to re-weight a cell's mean, so coverage is checked
    even when every surviving row is perfect.
  * the CONFIG pins the corpus. The run under test does not get to choose what
    it is graded against: a run manifest that names a different eval set (or a
    different constraint instance library) is an ERROR, not an override, and
    the content hash of the corpus actually scored is written into the report
    so the reading stays auditable after the fact.
  * every path the gate READS is resolved, and the resolved path must lie inside
    the tree it claims to come from: a run's transcripts, results JSON and
    manifest inside the resolved run directory, a repo-relative pinned corpus
    inside the repo. A lexical check is not containment — `is_absolute()` and
    `".." in parts` are both clean for a symlink, for a hop through a symlinked
    directory, and for a hard link, and all three let a cell be scored against a
    donor run's rows while the report records an innocuous in-run filename. The
    RESOLVED path and the sha256 of every file scored go into the report, so the
    audit trail describes the bytes rather than the name they were given.
    HONEST LIMIT: this cannot detect a COPY of another run's transcripts, and
    nothing available to this gate can — a copy is byte-identical to a file the
    run wrote itself. What is provable about the rows is proved (checkers re-run,
    coverage exact, hashes recorded); "these are the rows the model produced" is
    not among the things a static verifier can establish.
  * booleans are not numbers and strings are not numbers. Every stored scalar is
    type-checked before it is compared (`isinstance(True, int)` is True in
    Python, and that bypass has defeated guards in this repo).
  * the ladder rung a run claims is verified against the eval set that run used
    (max k actually reached, adversarial share actually met, and -- where the
    rung declares one -- the character-level constraint share actually
    reached), because HARD_STOP is only legitimate on the last rung and because
    a rung field nothing derives from the corpus is a claim, not a constraint.

There is deliberately NO flag that softens the threshold, and none may be added.
The only verdict that exits 0 is PROCEED.

Exit codes:  0 PROCEED  |  1 invalid/inconsistent run  |  2 ESCALATE  |  3 HARD_STOP

This module does NOT import eval.py. eval.py is mid-pivot (T-3) and its
pre-pivot aggregate() reads gold_label/pred_class fields that no longer exist in
the transcript contract; importing it here would couple the gate to a moving
file and would not compute the right number anyway. It does not import
harness.py either: harness.run_checkers is the loop that PRODUCED the stored
checker_results, and a verifier that reuses the producer's loop cannot see a bug
in it. What it does import — and never reimplements — is
constraints.checkers.check, the pinned ground-truth checker itself. The one
definition this file restates is the pinned Design contract `spec_pass = all
checkers pass AND quality_score >= 3`, which is the thing the gate exists to
independently verify.
"""
import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    # scripts/ is sys.path[0] when this file is run directly; the ground-truth
    # checker lives at the repo root and is imported, never reimplemented.
    sys.path.insert(0, str(REPO_ROOT))

from constraints.checkers import check as run_checker          # noqa: E402
from constraints.registry import (ConstraintError, family_of,  # noqa: E402
                                  validate_constraint)

# ---------------------------------------------------------------- thresholds
# R6. Exact rationals, not floats: 24/120 as a float is 0.19999999999999996,
# which would flip a legitimate PROCEED at the 0.20 boundary into ESCALATE.
PROCEED_FAIL_RATE_MIN = Fraction("0.20")   # fail_rate >= 0.20  -> PROCEED
HARD_STOP_ADHERENCE_MIN = Fraction("0.90")  # adherence >= 0.90 after the full ladder
QUALITY_SCORE_MIN = 3                      # Design: spec_pass needs quality >= 3
QUALITY_SCORE_RANGE = (1, 5)
MIN_SCENARIOS_PER_CELL = 30                # R6: ">= 30 scenarios per cell"
# The constraint FAMILY the r6-char65 rung is about. Taken from the registry
# (constraints.registry.FAMILIES) rather than a list of type names typed out
# here, so a character type added later counts automatically instead of
# silently deflating the share this gate verifies.
CHARACTER_FAMILY = "character"
# A row the generator or judge could not produce a verdict for is excluded from
# the rate. Beyond this share the cell stops being evidence about the model.
MAX_QUARANTINE_SHARE = 0.10

EXPECTED_STRATEGIES = ("zero_shot", "few_shot", "structured")
FRONTIER_PROVIDERS = frozenset({"openai", "google", "anthropic"})
LOCAL_BASE_PROVIDERS = frozenset({"hf"})
PRESSURES = frozenset({"clean", "adversarial"})

VERDICT_PROCEED = "PROCEED"
VERDICT_ESCALATE = "ESCALATE"
VERDICT_HARD_STOP = "HARD_STOP"

EXIT_PROCEED = 0
EXIT_INVALID = 1
EXIT_ESCALATE = 2
EXIT_HARD_STOP = 3

EXIT_FOR_VERDICT = {
    VERDICT_PROCEED: EXIT_PROCEED,
    VERDICT_ESCALATE: EXIT_ESCALATE,
    VERDICT_HARD_STOP: EXIT_HARD_STOP,
}

MANIFEST_NAMES = ("g1_table.json", "ablation_summary.json", "run_manifest.json")

#: The pinned constraint instance library. The gate re-runs every checker, so it
#: needs the constraint OBJECTS the scenario ids refer to. A config may pin a
#: different library with "constraint_instances"; this is the repo default and
#: the same file scenarios.paths.INSTANCES_PATH names. It is deliberately NOT a
#: CLI flag: swapping in a laxer library would move the measurement.
DEFAULT_INSTANCES_PATH = "data/scenarios/constraint_instances.jsonl"

ROW_KEYS = ("scenario_id", "model", "strategy", "output", "checker_results",
            "quality_score", "quality_reasoning", "spec_pass")


class GateError(Exception):
    """An inconsistency that makes the run unratable. Never softened."""


# ------------------------------------------------------------- type guards
def is_real_number(x):
    """int/float only. bool is NOT a number here: isinstance(True, int) is True."""
    return (isinstance(x, (int, float)) and not isinstance(x, bool)
            and math.isfinite(x))


def is_real_bool(x):
    """A JSON true/false. Not 1, not 0, not "true"."""
    return isinstance(x, bool)


def is_real_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def is_nonempty_str(x):
    return isinstance(x, str) and x.strip() != ""


def describe(x):
    return f"{x!r} (type {type(x).__name__})"


def require_number(errors, where, field, value):
    """Type-check a stored scalar BEFORE any numeric comparison touches it."""
    if not is_real_number(value):
        errors.append(f"{where}: {field} is {describe(value)}, not a finite "
                      f"int/float — a stored aggregate of this type cannot be "
                      f"compared to the recomputed value")
        return False
    return True


# ------------------------------------------------------------------- io
def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"{path}: unreadable ({exc})") from exc


def read_jsonl(path):
    rows = []
    try:
        with open(path) as f:
            for lineno, line in enumerate(f, 1):
                if line.strip():
                    rows.append((lineno, json.loads(line)))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"{path}: unreadable ({exc})") from exc
    return rows


def sha256_of(path):
    """Content hash of the corpus actually scored, so the report is auditable."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    except OSError as exc:
        raise GateError(f"{path}: unreadable ({exc})") from exc
    return h.hexdigest()


def is_within(real, base_real):
    """Is `real` the base directory itself or something underneath it?

    Both arguments must already be RESOLVED. Compared component-wise, never as
    strings: "/runs/g1-donor".startswith("/runs/g1") is True and would let a
    sibling directory read as contained.
    """
    return real == base_real or base_real in real.parents


def contained_path(cand, base_real, where, label, base_label):
    """Resolve `cand` and require the REAL file to live under `base_real`.

    A lexical check is NOT containment. `p.is_absolute()` is False and `".."
    not in p.parts` is True for `donor.jsonl` when `donor.jsonl` is a SYMLINK,
    and for `sub/rows.jsonl` when `sub` is a symlinked directory — both resolve
    anywhere on the filesystem. Only resolving the path and testing the result
    against the resolved base makes the containment claim true.

    Returns the resolved path; raises GateError when it escapes.
    """
    real = Path(cand).resolve()
    if not is_within(real, base_real):
        raise GateError(
            f"{where}: {label} resolves to {real}, which escapes "
            f"the {base_label} {base_real} (a symlink, or a symlinked parent "
            f"directory, leaves the tree while the name stays innocuous). The "
            f"gate scores the files shipped with a run, never a file from "
            f"somewhere else on disk, and it will not record an in-tree name for "
            f"an out-of-tree file")
    return real


def require_sole_named_file(real, where, label):
    """A run artifact must be a file with exactly ONE name on disk.

    Containment stops a symlink from leaving the run tree; it cannot see a HARD
    LINK, which puts a second name on a donor run's bytes INSIDE the run
    directory and resolves to itself. eval.write_outputs creates its files with
    open(..., "w"), so every honest run artifact has st_nlink == 1 (verified
    against every file in runs/ before this check was added); more than one name
    means the bytes the gate is about to score are also somebody else's file.

    HONEST LIMIT, stated so nobody reads more into a PROCEED than is there: this
    does NOT detect a byte-for-byte COPY of another run's transcripts, and
    nothing available to this gate can. A copy is indistinguishable from a file
    the run produced itself. What the gate can prove about the rows it scores is
    what it does prove — the checkers are re-run over every output, coverage is
    exact, and the resolved path and sha256 of every file are in the report.
    """
    if not real.is_file():
        return real
    nlink = real.stat().st_nlink
    if nlink != 1:
        raise GateError(
            f"{where}: {label} is a hard link — the same bytes answer to {nlink} "
            f"names on disk, at least one of which the gate cannot see. A run "
            f"writes its own artifacts (eval.write_outputs opens a new file per "
            f"cell, st_nlink == 1), so a second name means these rows are shared "
            f"with a file somewhere else and containment cannot tell you where. "
            f"The gate refuses to rate it rather than record an in-run name for "
            f"bytes it cannot show are this run's")
    return real


def resolve_pinned_path(raw, label):
    """Resolve a CONFIG-pinned data path. Repo-relative or absolute, nothing else.

    The old resolver SEARCHED cwd, then the run dir, then the repo root, so the
    file that answered to `data/eval/scenarios.jsonl` depended on where the gate
    was invoked from and on what the run under test happened to ship. A pinned
    corpus resolves to exactly one place: the repo root (config paths are
    repo-relative by contract) or the absolute path given.

    The returned path is RESOLVED, so the report names the file that was really
    read. A repo-relative pin must also STAY in the repo once resolved: "pinned
    by the config, repo-relative by contract" is not true of a checked-in
    symlink that leaves the tree. An absolute pin is the config author naming a
    location outright — that is a deliberate, reviewable choice, so it is
    allowed and recorded (resolved, with its sha256) rather than second-guessed.
    """
    p = Path(raw)
    cand = p if p.is_absolute() else REPO_ROOT / p
    if not cand.exists():
        raise GateError(
            f"{label} {raw!r} not found at {cand}. The gate cannot verify a run "
            f"against a corpus it cannot read, so it fails rather than rating a "
            f"slice (config paths are repo-relative to {REPO_ROOT})")
    if p.is_absolute():
        return cand.resolve()
    return contained_path(cand, REPO_ROOT.resolve(), label,
                          f"the repo-relative pinned path {raw!r}", "repo root")


def resolve_transcripts_path(raw, run_dir_real, where):
    """A results JSON may name its transcripts, but only INSIDE its own run dir.

    eval.write_outputs never emits this key (it uses the stem convention), so a
    declared path is pure attack surface: an absolute path, a `..` hop, a
    symlink, or a hop through a symlinked directory all let a cell be scored
    against a transcript file from some other run while the report records the
    innocuous in-run name it was given.

    Returns the RESOLVED path, or None when nothing is there.
    """
    p = Path(raw)
    if p.is_absolute() or ".." in p.parts:
        raise GateError(
            f"{where}: declares transcripts {raw!r}, which escapes the run "
            f"directory. A cell is scored against transcripts shipped with it, "
            f"never against a file from somewhere else on disk")
    cand = run_dir_real / p
    if not cand.exists():
        return None
    real = contained_path(cand, run_dir_real, where, f"declared transcripts {raw!r}",
                          "run directory")
    return require_sole_named_file(real, where, f"declared transcripts {raw!r}")


def provider_of(spec):
    return spec.split(":", 1)[0] if isinstance(spec, str) and ":" in spec else ""


def classify_provider(spec):
    prov = provider_of(spec)
    if prov in FRONTIER_PROVIDERS:
        return "frontier"
    if prov in LOCAL_BASE_PROVIDERS:
        return "local_base"
    return "unknown"


# --------------------------------------------------------------- config
def load_config(path):
    cfg = read_json(path)
    if not isinstance(cfg, dict):
        raise GateError(f"{path}: config is not a JSON object")

    models = cfg.get("models")
    if not isinstance(models, dict):
        raise GateError(f"{path}: 'models' is missing or not an object")
    frontier = models.get("frontier")
    if (not isinstance(frontier, list) or len(frontier) < 2
            or not all(is_nonempty_str(m) for m in frontier)):
        raise GateError(f"{path}: models.frontier must be a list of >= 2 "
                        f"'provider:model' strings, got {describe(frontier)}")
    families = {provider_of(m) for m in frontier}
    if len(families) < 2:
        raise GateError(f"{path}: R6 requires >= 2 frontier models from DIFFERENT "
                        f"families; models.frontier spans only {sorted(families)}")
    bad = [m for m in frontier if classify_provider(m) != "frontier"]
    if bad:
        raise GateError(f"{path}: models.frontier lists non-frontier spec(s) {bad}; "
                        f"frontier providers are {sorted(FRONTIER_PROVIDERS)}")
    base = models.get("local_base")
    if not is_nonempty_str(base) or classify_provider(base) != "local_base":
        raise GateError(f"{path}: models.local_base must be a local spec "
                        f"({'/'.join(sorted(LOCAL_BASE_PROVIDERS))}:...), got "
                        f"{describe(base)}")

    strategies = cfg.get("strategies")
    if list(strategies or []) != list(EXPECTED_STRATEGIES):
        raise GateError(f"{path}: strategies must be exactly "
                        f"{list(EXPECTED_STRATEGIES)}, got {describe(strategies)}")

    ladder = cfg.get("ladder")
    if not isinstance(ladder, dict):
        raise GateError(f"{path}: 'ladder' is missing or not an object")
    rungs = ladder.get("rungs")
    if not isinstance(rungs, list) or not rungs:
        raise GateError(f"{path}: ladder.rungs must be a non-empty list")
    seen = set()
    for i, rung in enumerate(rungs):
        if not isinstance(rung, dict):
            raise GateError(f"{path}: ladder.rungs[{i}] is not an object")
        name = rung.get("name")
        if not is_nonempty_str(name):
            raise GateError(f"{path}: ladder.rungs[{i}] has no 'name'")
        if name in seen:
            raise GateError(f"{path}: duplicate ladder rung name {name!r}")
        seen.add(name)
        ks = rung.get("k_values")
        if (not isinstance(ks, list) or not ks
                or not all(is_real_int(k) and k >= 2 for k in ks)):
            raise GateError(f"{path}: ladder rung {name!r} k_values must be a "
                            f"non-empty list of ints >= 2 (the design forbids "
                            f"k=1), got {describe(ks)}")
        share = rung.get("adversarial_share")
        if not is_real_number(share) or not 0 <= share <= 1:
            raise GateError(f"{path}: ladder rung {name!r} adversarial_share must "
                            f"be a number in [0, 1], got {describe(share)}")
        # OPTIONAL, and only meaningful because verify_rung_against_eval_set
        # recomputes it from the shipped corpus. A rung field nothing derives
        # is a claim, not a constraint, so the shape is pinned here and the
        # number is checked there.
        char_share = rung.get("min_character_constraint_share")
        if char_share is not None and (not is_real_number(char_share)
                                       or not 0 <= char_share <= 1):
            raise GateError(f"{path}: ladder rung {name!r} "
                            f"min_character_constraint_share must be a number in "
                            f"[0, 1], got {describe(char_share)}")
    current = ladder.get("current")
    if current is not None and current not in seen:
        raise GateError(f"{path}: ladder.current {current!r} is not one of the "
                        f"declared rungs {sorted(seen)}")
    return cfg


def expected_cells(cfg):
    models = [cfg["models"]["local_base"]] + list(cfg["models"]["frontier"])
    return [(m, s) for m in models for s in EXPECTED_STRATEGIES]


# ------------------------------------------------ constraint instances
def load_constraint_instances(path):
    """{constraint_id: constraint}. The objects the gate re-runs checkers with.

    Every instance is validated by the pinned registry, so a library row that
    has drifted from the schema (a params key the checker ignores, a family that
    disagrees with the type) is a loud failure here rather than a checker that
    quietly answers a question nobody asked.
    """
    rows = read_jsonl(path)
    if not rows:
        raise GateError(f"{path}: constraint instance library is empty")
    lib = {}
    for lineno, c in rows:
        try:
            validate_constraint(c)
        except ConstraintError as exc:
            raise GateError(f"{path}:{lineno}: invalid constraint instance "
                            f"({exc})") from exc
        cid = c["id"]
        if cid in lib:
            raise GateError(f"{path}:{lineno}: duplicate constraint id {cid!r} — "
                            f"two objects answer to the same name and every "
                            f"scenario referencing it is ambiguous")
        lib[cid] = c
    return lib


def check_constraints_resolve(eval_index, instances, eval_path, instances_path):
    """Every constraint id the corpus names must exist in the pinned library.

    An unresolvable id would mean the gate could not re-run that checker, and a
    skipped checker silently inflates spec_pass.
    """
    unknown = sorted({cid for s in eval_index.values() for cid in s["constraints"]
                      if cid not in instances})
    if unknown:
        raise GateError(
            f"{eval_path.name}: {len(unknown)} constraint id(s) are not in the "
            f"pinned instance library {instances_path} (e.g. {unknown[:5]}) — the "
            f"gate re-runs every checker from (constraint, output) and refuses to "
            f"rate a scenario whose constraints it cannot resolve")


# ------------------------------------------------------------- eval set
def load_eval_set(path):
    """{scenario_id: scenario}. The gate needs the real corpus, not a stand-in."""
    rows = read_jsonl(path)
    if not rows:
        raise GateError(f"{path}: eval set is empty")
    index = {}
    for lineno, s in rows:
        if not isinstance(s, dict):
            raise GateError(f"{path}:{lineno}: scenario is not an object")
        sid = s.get("scenario_id")
        if not is_nonempty_str(sid):
            raise GateError(f"{path}:{lineno}: scenario_id is {describe(sid)}")
        if sid in index:
            raise GateError(f"{path}:{lineno}: duplicate scenario_id {sid!r} in "
                            f"the eval set")
        cons = s.get("constraints")
        if (not isinstance(cons, list) or not cons
                or not all(is_nonempty_str(c) for c in cons)):
            raise GateError(f"{path}:{lineno}: scenario {sid!r} constraints is "
                            f"{describe(cons)}, expected a non-empty list of "
                            f"constraint ids")
        if len(set(cons)) != len(cons):
            raise GateError(f"{path}:{lineno}: scenario {sid!r} repeats a "
                            f"constraint id: {cons}")
        k = s.get("k")
        if not is_real_int(k) or k != len(cons):
            raise GateError(f"{path}:{lineno}: scenario {sid!r} k is {describe(k)} "
                            f"but it carries {len(cons)} constraint(s)")
        pressure = s.get("pressure")
        if pressure not in PRESSURES:
            raise GateError(f"{path}:{lineno}: scenario {sid!r} pressure is "
                            f"{describe(pressure)}, expected one of "
                            f"{sorted(PRESSURES)}")
        index[sid] = s
    if len(index) < MIN_SCENARIOS_PER_CELL:
        raise GateError(f"{path}: {len(index)} scenarios; R6 requires >= "
                        f"{MIN_SCENARIOS_PER_CELL} per model x strategy cell")
    return index


# ------------------------------------------------------------- transcripts
def rederive_checker_results(scenario, instances, output, where, errors):
    """Re-run the PINNED checker over (constraint, output) for this row.

    Returns {constraint_id: (pass, evidence)} or None if a constraint could not
    be checked. This is the half of the row the gate CAN reproduce, and the
    whole reason it can: a run whose outputs genuinely comply could otherwise
    write itself a set of failures and manufacture a PROCEED.
    """
    derived = {}
    for cid in scenario["constraints"]:
        constraint = instances.get(cid)
        if constraint is None:      # pre-checked by check_constraints_resolve
            errors.append(f"{where}: constraint {cid!r} is not in the pinned "
                          f"instance library, so its checker cannot be re-run")
            return None
        try:
            verdict = run_checker(constraint, output)
        except ConstraintError as exc:
            errors.append(f"{where}: re-running the checker for {cid!r} raised "
                          f"{exc} — a constraint the ground-truth layer refuses "
                          f"cannot have produced a trustworthy verdict")
            return None
        derived[cid] = (bool(verdict["pass"]), str(verdict["evidence"]))
    return derived


def compare_checker_results(stored, derived, where, sid, errors):
    """Stored verdicts vs the ones the pinned checker produces. Any gap fails."""
    ok = True
    for cr in stored:
        cid = cr["constraint_id"]
        want_pass, want_evidence = derived[cid]
        if cr["pass"] != want_pass:
            errors.append(
                f"{where} (scenario {sid}): checker_results for {cid} stores "
                f"pass={cr['pass']!r}, but re-running constraints.checkers.check "
                f"on this row's own output gives pass={want_pass!r} "
                f"(evidence: {want_evidence!r}) — the stored verdict is fabricated")
            ok = False
        elif cr["evidence"] != want_evidence:
            errors.append(
                f"{where} (scenario {sid}): checker_results for {cid} stores "
                f"evidence {cr['evidence']!r} but the pinned checker produces "
                f"{want_evidence!r} for this output — the verdict was not produced "
                f"by the checker it claims")
            ok = False
    return ok


def score_cell(results_path, transcripts_path, model, strategy, eval_index,
               instances, errors, declared_transcripts=None):
    """Recompute this cell from its raw rows. Returns a stats dict or None.

    Nothing programmatic in the row is trusted: the checker verdicts are re-run
    from (constraint, output), spec_pass is re-derived from THOSE verdicts plus
    the row's quality_score, and the row set must cover the eval set exactly.
    """
    numbered = read_jsonl(transcripts_path)
    tag = transcripts_path.name

    if not numbered:
        errors.append(f"{tag}: no transcript rows")
        return None

    seen = {}
    quarantined_rows: List[str] = []
    n_pass = 0
    n_adv = 0
    n_adv_pass = 0
    n_checker_fail = 0
    n_quality_only_fail = 0
    ok = True

    for lineno, row in numbered:
        where = f"{tag}:{lineno}"
        if not isinstance(row, dict):
            errors.append(f"{where}: row is not an object")
            ok = False
            continue
        missing = [k for k in ROW_KEYS if k not in row]
        if missing:
            errors.append(f"{where}: row missing required field(s) {missing}")
            ok = False
            continue

        sid = row["scenario_id"]
        if not is_nonempty_str(sid):
            errors.append(f"{where}: scenario_id is {describe(sid)}")
            ok = False
            continue
        scen = eval_index.get(sid)
        if scen is None:
            errors.append(f"{where}: scenario_id {sid!r} is not in the eval set — "
                          f"this row inflates or dilutes the cell with a scenario "
                          f"the gate is not defined over")
            ok = False
            continue
        if sid in seen:
            errors.append(f"{where}: duplicate row for scenario {sid!r} (first at "
                          f"line {seen[sid]}) — a repeated row re-weights the mean")
            ok = False
            continue
        seen[sid] = lineno

        if row["model"] != model:
            errors.append(f"{where}: row model {describe(row['model'])} != this "
                          f"cell's model {model!r}")
            ok = False
        if row["strategy"] != strategy:
            errors.append(f"{where}: row strategy {describe(row['strategy'])} != "
                          f"this cell's strategy {strategy!r}")
            ok = False
        # --- QUARANTINED rows: the generator or the judge produced nothing, so
        # there is no verdict to re-derive. The harness records these with
        # spec_pass=None and a *_error field, excludes them from every rate and
        # counts them. The gate must do the same: demanding a boolean spec_pass
        # here would reject an honest run, and silently accepting the row as a
        # PASS would score a refusal as perfect compliance (empty text satisfies
        # every checker vacuously). Excluded from re-derivation, counted below,
        # and capped -- a cell that quarantines much of the eval set is not a
        # measurement, and quarantining is otherwise a way to drop rows you fail.
        if row.get("judge_error") is not None or row.get("generator_error") is not None:
            if row.get("spec_pass") is not None:
                errors.append(f"{where}: carries an error marker but spec_pass is "
                              f"{describe(row['spec_pass'])} — a quarantined row "
                              f"must not also claim a verdict")
                ok = False
            quarantined_rows.append(sid)
            continue

        row_ok = True
        if not isinstance(row["output"], str):
            # row_ok, not just ok: the checker is re-run over this value below,
            # and constraints.checkers.check raises TypeError on a non-string
            # text. A row this broken must be REPORTED, not turned into a
            # traceback that buries the other 39 problems in the run.
            errors.append(f"{where}: output is {describe(row['output'])}, not a "
                          f"string — the gate cannot re-run the checkers over it")
            ok = row_ok = False
        if not is_nonempty_str(row["quality_reasoning"]):
            errors.append(f"{where}: quality_reasoning is "
                          f"{describe(row['quality_reasoning'])} — an unexplained "
                          f"quality score is not a judged score")
            ok = False

        # --- checker results must cover exactly this scenario's constraints
        crs = row["checker_results"]
        # NB: row_ok is NOT reset here — a row whose output is unusable is
        # already disqualified from re-derivation above.
        if not isinstance(crs, list) or not crs:
            errors.append(f"{where}: checker_results is {describe(crs)}, expected a "
                          f"non-empty list")
            ok = row_ok = False
        else:
            ids = []
            for j, cr in enumerate(crs):
                if not isinstance(cr, dict):
                    errors.append(f"{where}: checker_results[{j}] is not an object")
                    ok = row_ok = False
                    continue
                cid = cr.get("constraint_id")
                if not is_nonempty_str(cid):
                    errors.append(f"{where}: checker_results[{j}].constraint_id is "
                                  f"{describe(cid)}")
                    ok = row_ok = False
                    continue
                ids.append(cid)
                if not is_real_bool(cr.get("pass")):
                    errors.append(f"{where}: checker_results[{j}] ({cid}) pass is "
                                  f"{describe(cr.get('pass'))}, not a JSON boolean")
                    ok = row_ok = False
                if not is_nonempty_str(cr.get("evidence")):
                    errors.append(f"{where}: checker_results[{j}] ({cid}) evidence is "
                                  f"{describe(cr.get('evidence'))}; the contract "
                                  f"requires the violating substring or \"clean\"")
                    ok = row_ok = False
            expected_ids = list(scen["constraints"])
            if sorted(ids) != sorted(expected_ids):
                errors.append(
                    f"{where}: checker_results cover {sorted(ids)} but scenario "
                    f"{sid!r} declares constraints {sorted(expected_ids)} — a "
                    f"dropped or invented checker result silently changes spec_pass")
                ok = row_ok = False

        # --- quality score
        q = row["quality_score"]
        lo, hi = QUALITY_SCORE_RANGE
        if not is_real_int(q) or not lo <= q <= hi:
            errors.append(f"{where}: quality_score is {describe(q)}, expected an "
                          f"integer in [{lo}, {hi}]")
            ok = row_ok = False

        # --- spec_pass, recomputed from this row's own fields
        stored = row["spec_pass"]
        if not is_real_bool(stored):
            errors.append(f"{where}: spec_pass is {describe(stored)}, not a JSON "
                          f"boolean")
            ok = row_ok = False
        if not row_ok:
            continue

        # --- the checker verdicts, RE-DERIVED from (constraint, output)
        # Everything above only proved the row is self-consistent. This is where
        # the row is compared to what the pinned checker actually says about the
        # output it shipped.
        derived = rederive_checker_results(scen, instances, row["output"], where,
                                           errors)
        if derived is None:
            ok = False
            continue
        if not compare_checker_results(crs, derived, where, sid, errors):
            ok = False
            continue

        # Deliberate redundancy: the derived verdicts are the authority even
        # though compare_checker_results has just proved the stored block equals
        # them. If that comparison is ever downgraded, spec_pass still comes
        # from the checker rather than from the transcript.
        checkers_pass = all(p for p, _ in derived.values())
        recomputed = checkers_pass and q >= QUALITY_SCORE_MIN
        if recomputed != stored:
            errors.append(
                f"{where} (scenario {sid}): spec_pass == {stored!r} but the pinned "
                f"definition (all checkers pass AND quality_score >= "
                f"{QUALITY_SCORE_MIN}) gives {recomputed!r} from the re-derived "
                f"checker verdicts and this row's quality_score")
            ok = False
            continue

        if recomputed:
            n_pass += 1
        elif not checkers_pass:
            n_checker_fail += 1
        else:
            # A failure the gate cannot reproduce: the checkers all pass and the
            # judge's score is the only thing holding it down. Counted in the
            # metric (it is the pinned definition) and surfaced in the report.
            n_quality_only_fail += 1
        if scen["pressure"] == "adversarial":
            n_adv += 1
            if recomputed:
                n_adv_pass += 1

    # --- coverage: the rows must BE the eval set, not a flattering slice of it.
    # A quarantined scenario HAS a row -- it is covered, it just has no verdict --
    # so it counts here and is excluded from the rate denominator below.
    # (quarantined ids are already in `seen`: a quarantined scenario HAS a row,
    # it just has no verdict, so it is covered but not scored.)
    missing_ids = sorted(set(eval_index) - set(seen))
    if missing_ids:
        errors.append(
            f"{tag}: {len(missing_ids)} of {len(eval_index)} eval-set scenarios have "
            f"no row (e.g. {missing_ids[:5]}) — a strict subset re-weights "
            f"spec_adherence even when every surviving row is correctly scored")
        ok = False

    # --- quarantine cap. Excluding unscorable rows is honest; excluding a large
    # share is not a measurement, and is indistinguishable from dropping the rows
    # a model failed. Cap it and say the number out loud either way.
    n_quarantined = len(quarantined_rows)
    if n_quarantined:
        share = n_quarantined / max(len(eval_index), 1)
        if share > MAX_QUARANTINE_SHARE:
            errors.append(
                f"{tag}: {n_quarantined} of {len(eval_index)} rows quarantined "
                f"({share:.1%}) exceeds the {MAX_QUARANTINE_SHARE:.0%} cap — with "
                f"this many rows unscored the cell is not a measurement of the "
                f"model, and dropping failures would look identical")
            ok = False

    if not ok:
        return None

    # seen holds EVERY row (coverage and duplicate detection need that). The rate
    # denominator must exclude quarantined rows, matching harness.aggregate, which
    # divides by scored rows only. Counting them would silently deflate the cell.
    n = len(seen) - n_quarantined
    if n < MIN_SCENARIOS_PER_CELL:
        errors.append(f"{tag}: {n} scored rows; R6 requires >= "
                      f"{MIN_SCENARIOS_PER_CELL} scenarios per cell")
        return None

    adherence = Fraction(n_pass, n)
    fail_rate = Fraction(n - n_pass, n)
    robustness = Fraction(n_adv_pass, n_adv) if n_adv else None

    # --- the stored aggregate is compared, never trusted
    results = read_json(results_path)
    if not isinstance(results, dict):
        errors.append(f"{results_path.name}: results is not a JSON object")
        return None
    metrics = results.get("metrics")
    if not isinstance(metrics, dict):
        errors.append(f"{results_path.name}: 'metrics' is missing or not an object")
        return None
    where = results_path.name
    stored_ok = True
    if "spec_adherence" not in metrics:
        errors.append(f"{where}: metrics.spec_adherence is missing")
        stored_ok = False
    elif not require_number(errors, where, "metrics.spec_adherence",
                            metrics["spec_adherence"]):
        stored_ok = False
    elif abs(float(metrics["spec_adherence"]) - float(adherence)) > 1e-9:
        errors.append(
            f"{where}: metrics.spec_adherence == {metrics['spec_adherence']!r} but "
            f"recomputing from {tag} gives {float(adherence)!r} ({n_pass}/{n} rows "
            f"with spec_pass) — the stored aggregate disagrees with its own rows")
        stored_ok = False
    for optional, actual in (("fail_rate", fail_rate),
                             ("robustness", robustness),
                             ("n", n)):
        if optional not in metrics:
            continue
        if optional == "n":
            # harness.aggregate reports n = TOTAL rows, with n_scored/n_unscored
            # alongside; the gate's `n` is the rate denominator, i.e. scored rows.
            # Check both meanings against the right field rather than conflating
            # them -- comparing total to scored fails every honest run that
            # quarantined anything.
            if not is_real_int(metrics["n"]):
                errors.append(f"{where}: metrics.n is {describe(metrics['n'])}, "
                              f"not an integer")
                stored_ok = False
            elif metrics["n"] != n + n_quarantined:
                errors.append(f"{where}: metrics.n == {metrics['n']!r} but {tag} "
                              f"holds {n + n_quarantined} rows "
                              f"({n} scored + {n_quarantined} quarantined)")
                stored_ok = False
            elif "n_scored" in metrics and is_real_int(metrics["n_scored"]) \
                    and metrics["n_scored"] != n:
                errors.append(f"{where}: metrics.n_scored == {metrics['n_scored']!r} "
                              f"but {tag} holds {n} scored rows")
                stored_ok = False
            continue
        if actual is None:
            continue
        if not require_number(errors, where, f"metrics.{optional}",
                              metrics[optional]):
            stored_ok = False
        elif abs(float(metrics[optional]) - float(actual)) > 1e-9:
            errors.append(f"{where}: metrics.{optional} == {metrics[optional]!r} "
                          f"but the rows give {float(actual)!r}")
            stored_ok = False
    if not stored_ok:
        return None

    return {
        "model": model,
        "strategy": strategy,
        "provider": provider_of(model),
        "classification": classify_provider(model),
        "n": n,
        "n_spec_pass": n_pass,
        "n_spec_fail": n - n_pass,
        "spec_adherence": float(adherence),
        "fail_rate": float(fail_rate),
        "n_adversarial": n_adv,
        "robustness": float(robustness) if robustness is not None else None,
        "n_fail_from_checkers": n_checker_fail,
        "n_fail_from_quality_only": n_quality_only_fail,
        # The audit trail describes the file that was actually scored: the
        # RESOLVED path plus a content hash, not the name the run gave it. A
        # reader can re-hash both files and get the same numbers back.
        "results_json": results_path.name,
        "results_json_path": str(results_path),
        "results_json_sha256": sha256_of(results_path),
        "transcripts_jsonl": transcripts_path.name,
        "transcripts_jsonl_path": str(transcripts_path),
        "transcripts_jsonl_sha256": sha256_of(transcripts_path),
        "transcripts_declared_as": declared_transcripts,
        "_adherence_exact": adherence,
        "_fail_rate_exact": fail_rate,
    }


def load_cells(run_dir_real, cfg, eval_index, instances, errors):
    """Every *.results.json in the run dir, recomputed. {(model, strategy): stats}.

    `run_dir_real` is already resolved, and EVERY path read below is resolved and
    required to stay inside it — the results JSON itself as much as the
    transcripts it points at. One escapable resolution site is a reason to
    distrust its siblings, not a reason to fix only the one that was demoed.
    """
    paths = sorted(run_dir_real.glob("*.results.json"))
    if not paths:
        errors.append(f"{run_dir_real}: no *.results.json — nothing to gate on. Run "
                      f"run_ablation.py --config configs/g1.json first")
        return {}

    allowed_models = set(cfg["models"]["frontier"]) | {cfg["models"]["local_base"]}
    cells = {}
    scored = {}          # resolved transcripts path -> the cell that already used it
    for listed in paths:
        try:
            _load_one_cell(listed, run_dir_real, allowed_models, eval_index,
                           instances, cells, scored, errors)
        except GateError as exc:
            # A cell whose paths do not resolve inside the run, or whose files
            # cannot be read, is ONE reported problem — not a traceback that
            # hides the other eight cells' problems.
            errors.append(str(exc))
    return cells


def _load_one_cell(listed, run_dir_real, allowed_models, eval_index, instances,
                   cells, scored, errors):
    """One *.results.json -> one entry in `cells` (None if it failed to score)."""
    tag = listed.name
    rp = contained_path(listed, run_dir_real, tag, f"results JSON {tag!r}",
                        "run directory")
    require_sole_named_file(rp, tag, f"results JSON {tag!r}")
    results = read_json(rp)
    if not isinstance(results, dict):
        errors.append(f"{tag}: results is not a JSON object")
        return
    model = results.get("model")
    strategy = results.get("strategy")
    if not is_nonempty_str(model) or not is_nonempty_str(strategy):
        errors.append(f"{tag}: model/strategy must be non-empty strings, "
                      f"got model={describe(model)} strategy={describe(strategy)}")
        return
    # classification first: an unclassifiable provider gets told so, rather
    # than being reported as merely "unpinned"
    if classify_provider(model) == "unknown":
        errors.append(f"{tag}: model {model!r} is neither a frontier family "
                      f"{sorted(FRONTIER_PROVIDERS)} nor the local base "
                      f"{sorted(LOCAL_BASE_PROVIDERS)}; it cannot be classified")
        return
    if model not in allowed_models:
        errors.append(f"{tag}: model {model!r} is not one of the models "
                      f"configs/g1.json pins ({sorted(allowed_models)}) — a cell "
                      f"from an unpinned model cannot feed the gate")
        return
    if strategy not in EXPECTED_STRATEGIES:
        errors.append(f"{tag}: strategy {strategy!r} is not one of "
                      f"{list(EXPECTED_STRATEGIES)}")
        return
    key = (model, strategy)
    if key in cells:
        errors.append(f"{tag}: duplicate cell for {model} x {strategy}")
        return

    raw_tp = results.get("transcripts")
    if is_nonempty_str(raw_tp):
        tp = resolve_transcripts_path(raw_tp, run_dir_real, tag)
        if tp is None:
            errors.append(f"{tag}: declares transcripts {raw_tp!r}, which "
                          f"does not exist")
            return
    else:
        # The stem convention pairs by the name the RUN DIRECTORY lists, not by
        # whatever a symlinked results JSON happens to resolve to.
        stem_tp = run_dir_real / (tag[:-len(".results.json")] + ".transcripts.jsonl")
        if not stem_tp.exists():
            errors.append(f"{stem_tp.name}: missing — the gate re-derives every "
                          f"aggregate from raw transcript rows and refuses to "
                          f"rate a cell that ships only its summary")
            return
        tp = contained_path(stem_tp, run_dir_real, tag, "transcripts "
                            f"{stem_tp.name!r}", "run directory")
        require_sole_named_file(tp, tag, f"transcripts {stem_tp.name!r}")

    # Two cells that resolve to the SAME rows are one measurement counted twice,
    # however differently the two files are named.
    if tp in scored:
        m, s = scored[tp]
        errors.append(f"{tag}: transcripts resolve to {tp}, already scored as the "
                      f"{m} x {s} cell — one row set cannot be two cells")
        return
    scored[tp] = key

    cells[key] = score_cell(rp, tp, model, strategy, eval_index, instances, errors,
                            declared_transcripts=raw_tp if is_nonempty_str(raw_tp)
                            else None)


def check_grid(cells, cfg, errors):
    """Every pinned model x strategy present exactly once.

    There is deliberately no "extra cell" branch: load_cells() already refuses a
    results JSON whose model is not pinned, whose strategy is not one of the
    three, or that duplicates a cell, so `cells` can only ever be a SUBSET of the
    pinned grid. A branch here for the impossible case would read like protection
    that is not there — a mutation pass proved it unreachable.
    """
    for model, strategy in expected_cells(cfg):
        if (model, strategy) not in cells:
            errors.append(f"grid incomplete: no results JSON for {model} x "
                          f"{strategy} — R6 is defined over the full "
                          f"{len(cfg['models']['frontier']) + 1} x "
                          f"{len(EXPECTED_STRATEGIES)} grid")


# ---------------------------------------------------------------- ladder
def resolve_rung(cfg, manifest, cli_rung, errors):
    """Which rung produced THIS run — from the run itself where possible."""
    rungs = cfg["ladder"]["rungs"]
    names = [r["name"] for r in rungs]

    name, source = None, None
    if is_nonempty_str(manifest.get("ladder_rung") if manifest else None):
        name, source = manifest["ladder_rung"], "run_manifest"
    elif manifest and "ladder_rung" in manifest:
        errors.append(f"run manifest records ladder_rung "
                      f"{describe(manifest['ladder_rung'])}, which is not a name")
        return None
    elif is_nonempty_str(cli_rung):
        name, source = cli_rung, "cli"

    if name is None:
        errors.append(
            "cannot determine which escalation-ladder rung produced this run: no "
            "'ladder_rung' in the run manifest and no --rung given. HARD_STOP is "
            "only legitimate on the last rung, so the gate refuses to guess")
        return None
    if name not in names:
        errors.append(f"ladder_rung {name!r} (from {source}) is not a rung declared "
                      f"in the config ladder {names}")
        return None

    idx = names.index(name)
    return {
        "rung": name,
        "rung_source": source,
        "rung_index": idx + 1,
        "n_rungs": len(names),
        "exhausted": idx == len(names) - 1,
        "next_rung": names[idx + 1] if idx + 1 < len(names) else None,
        "definition": rungs[idx],
    }


def character_constraint_share(eval_index, instances):
    """``(n_scenarios carrying a CHARACTER-family constraint, n_scenarios)``.

    Derived from the shipped corpus and the pinned instance library, never from
    a manifest field: the number the r6-char65 rung rests on has to be
    recomputable by someone who does not trust the label. Family comes from
    :mod:`constraints.registry`, so ``banned_letter`` and ``banned_punctuation``
    are counted because the registry says they are character-level, not because
    they are named here.
    """
    n_char = 0
    for scenario in eval_index.values():
        for cid in scenario["constraints"]:
            constraint = instances.get(cid)
            if constraint is None:
                continue          # check_constraints_resolve already refused this run
            if family_of(constraint["type"]) == CHARACTER_FAMILY:
                n_char += 1
                break
    return n_char, len(eval_index)


def verify_rung_against_eval_set(rung, eval_index, instances, eval_path, errors):
    """The rung a run CLAIMS must match the corpus it actually used."""
    defn = rung["definition"]
    observed_k = sorted({s["k"] for s in eval_index.values()})
    allowed_k = sorted(set(defn["k_values"]))
    stray = [k for k in observed_k if k not in allowed_k]
    if stray:
        errors.append(f"{eval_path.name}: scenarios with k={stray} but rung "
                      f"{rung['rung']!r} declares k_values {allowed_k}")
    if observed_k and max(observed_k) != max(allowed_k):
        errors.append(f"{eval_path.name}: highest k present is {max(observed_k)} but "
                      f"rung {rung['rung']!r} claims the ladder was climbed to k="
                      f"{max(allowed_k)} — the rung label overstates this run")

    n = len(eval_index)
    n_adv = sum(1 for s in eval_index.values() if s["pressure"] == "adversarial")
    required = Fraction(str(defn["adversarial_share"]))
    if Fraction(n_adv, n) < required:
        errors.append(f"{eval_path.name}: adversarial share is {n_adv}/{n} = "
                      f"{float(Fraction(n_adv, n)):.4f}, below the "
                      f"{float(required):.4f} that rung {rung['rung']!r} declares")
    observed = {"k_values": observed_k, "n_adversarial": n_adv, "n_scenarios": n,
                "adversarial_share": float(Fraction(n_adv, n))}

    # The rung that the whole escalation rests on: >= 65% of scenarios must
    # carry a character-level constraint. Absent on the earlier rungs, which
    # never made the claim, so its absence is not a failure -- but where it IS
    # declared it is derived from the corpus, exactly like k and the
    # adversarial share, rather than taken on the label's word.
    required_char = defn.get("min_character_constraint_share")
    if required_char is not None:
        n_char, _ = character_constraint_share(eval_index, instances)
        observed["n_character"] = n_char
        observed["character_constraint_share"] = float(Fraction(n_char, n))
        observed["min_character_constraint_share"] = float(required_char)
        if Fraction(n_char, n) < Fraction(str(required_char)):
            errors.append(
                f"{eval_path.name}: {n_char}/{n} = "
                f"{float(Fraction(n_char, n)):.4f} of scenarios carry a "
                f"character-level constraint, below the "
                f"{float(required_char):.4f} that rung {rung['rung']!r} declares "
                f"— the corpus is not the one that rung names")
    return observed


# --------------------------------------------------------------- verdict
def decide(fail_rate, adherence, ladder_exhausted):
    """R6, and nothing but R6. The ONLY place a verdict is chosen.

    fail_rate / adherence are exact Fractions so the 0.20 and 0.90 boundaries
    fall where R6 says they fall (both are inclusive: '>=').
    """
    if fail_rate >= PROCEED_FAIL_RATE_MIN:
        return VERDICT_PROCEED, (
            f"best frontier cell fail_rate {float(fail_rate):.4f} >= "
            f"{float(PROCEED_FAIL_RATE_MIN):.2f} (R6)")
    if not ladder_exhausted:
        return VERDICT_ESCALATE, (
            f"best frontier cell fail_rate {float(fail_rate):.4f} < "
            f"{float(PROCEED_FAIL_RATE_MIN):.2f}; the escalation ladder has rungs "
            f"left — raise k / the adversarial share and rerun")
    if adherence >= HARD_STOP_ADHERENCE_MIN:
        return VERDICT_HARD_STOP, (
            f"best frontier cell adherence {float(adherence):.4f} >= "
            f"{float(HARD_STOP_ADHERENCE_MIN):.2f} after the full ladder — R6 hard "
            f"stop: report to Hank, do not start T-5")
    return VERDICT_HARD_STOP, (
        f"the escalation ladder is exhausted and the best frontier cell fail_rate "
        f"{float(fail_rate):.4f} is still < {float(PROCEED_FAIL_RATE_MIN):.2f}; R6 "
        f"permits PROCEED only at >= {float(PROCEED_FAIL_RATE_MIN):.2f} — stop and "
        f"report to Hank")


# ----------------------------------------------------------------- report
def pinned_corpus_path(cfg, config_path, manifest, manifest_name, key, label,
                       default=None):
    """The CONFIG pins the corpus; the run does not get to choose.

    The manifest may RECORD what the run used — that is useful provenance — but
    it is checked against the config, never allowed to outrank it. A run that
    scored a different (softer) corpus than the one the config pins is a run the
    gate cannot rate, not a run the gate re-points at.
    """
    raw = cfg.get(key, default)
    if not is_nonempty_str(raw):
        raise GateError(
            f"{config_path}: '{key}' is {describe(cfg.get(key))} — the {label} is "
            f"pinned by the config and by nothing else. There is no flag and no "
            f"manifest key that supplies one")
    path = resolve_pinned_path(raw, f"{label} (pinned by {config_path}:{key})")

    recorded = (manifest or {}).get(key)
    if recorded is not None:
        if not is_nonempty_str(recorded):
            raise GateError(f"{manifest_name}: records {key} {describe(recorded)}, "
                            f"which is not a path")
        rec_path = Path(recorded)
        rec_path = rec_path if rec_path.is_absolute() else REPO_ROOT / rec_path
        same = rec_path.exists() and path.exists() and rec_path.samefile(path)
        if not same:
            raise GateError(
                f"{manifest_name}: the run records {key} {recorded!r} but "
                f"{config_path} pins {raw!r} ({path}). The config pins the {label}; "
                f"a run that scored a different corpus than the pinned one is "
                f"unratable — rerun it against the pinned {label} or change the "
                f"config deliberately. A manifest is provenance, not an override")
    return path


def build_report(run_dir, config_path, cli_rung=None):
    """(report, errors). A report is returned only when the run is fully consistent."""
    errors = []
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return None, [f"{run_dir} does not exist or is not a directory"]
    # Every path the gate reads is measured against the RESOLVED run directory,
    # so a symlinked run dir is fine and a symlink that LEAVES it is not.
    run_dir_real = run_dir.resolve()

    cfg = load_config(config_path)

    manifest, manifest_name = None, None
    for name in MANIFEST_NAMES:
        p = run_dir_real / name
        if p.exists():
            # The manifest sets the ladder rung, and the rung decides whether a
            # sub-threshold run ESCALATEs or HARD_STOPs. It is as much an input
            # as the transcripts are, so it is contained the same way.
            manifest_path = contained_path(p, run_dir_real, name,
                                           f"run manifest {name!r}", "run directory")
            require_sole_named_file(manifest_path, name, f"run manifest {name!r}")
            manifest, manifest_name = read_json(manifest_path), name
            break
    if manifest is not None and not isinstance(manifest, dict):
        return None, [f"{manifest_name}: run manifest is not a JSON object"]

    eval_path = pinned_corpus_path(cfg, config_path, manifest, manifest_name,
                                   "eval_set", "eval set")
    instances_path = pinned_corpus_path(cfg, config_path, manifest, manifest_name,
                                        "constraint_instances",
                                        "constraint instance library",
                                        default=DEFAULT_INSTANCES_PATH)
    eval_sha256 = sha256_of(eval_path)
    instances_sha256 = sha256_of(instances_path)

    eval_index = load_eval_set(eval_path)
    instances = load_constraint_instances(instances_path)
    check_constraints_resolve(eval_index, instances, eval_path, instances_path)

    rung = resolve_rung(cfg, manifest, cli_rung, errors)
    observed = None
    if rung is not None:
        observed = verify_rung_against_eval_set(rung, eval_index, instances,
                                                eval_path, errors)

    cells = load_cells(run_dir_real, cfg, eval_index, instances, errors)
    check_grid(cells, cfg, errors)

    good = {k: v for k, v in cells.items() if v is not None}
    frontier = sorted((v for v in good.values() if v["classification"] == "frontier"),
                      key=lambda c: (c["model"], c["strategy"]))
    local_base = sorted((v for v in good.values() if v["classification"] == "local_base"),
                        key=lambda c: (c["model"], c["strategy"]))

    n_expected_frontier = len(cfg["models"]["frontier"]) * len(EXPECTED_STRATEGIES)
    if len(frontier) != n_expected_frontier:
        errors.append(f"expected {n_expected_frontier} frontier cells "
                      f"({len(cfg['models']['frontier'])} families x "
                      f"{len(EXPECTED_STRATEGIES)} strategies), got {len(frontier)}")
    families = sorted({c["provider"] for c in frontier})
    if len(families) < 2:
        errors.append(f"R6 requires >= 2 frontier families in the gate; the usable "
                      f"frontier cells span only {families}")
    if not local_base:
        errors.append("no local base-model cell — the ablation's third row is "
                      "required (it is excluded from the ceiling, but its absence "
                      "means the run is not the G1 ablation)")

    if errors:
        return None, errors

    best = max(frontier, key=lambda c: (c["_adherence_exact"], c["model"], c["strategy"]))
    adherence = best["_adherence_exact"]
    fail_rate = best["_fail_rate_exact"]
    verdict, reason = decide(fail_rate, adherence, rung["exhausted"])

    all_cells = sorted(good.values(), key=lambda c: (c["classification"], c["model"],
                                                     c["strategy"]))
    report = {
        "gate": "G1",
        "spec_refs": ["docs/PIVOT_SPEC.md R6", "docs/PIVOT_SPEC.md S1"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir_real),
        "config": str(config_path),
        "eval_set": str(eval_path),
        "eval_set_sha256": eval_sha256,
        "eval_set_source": f"pinned by {config_path}:eval_set",
        "constraint_instances": str(instances_path),
        "constraint_instances_sha256": instances_sha256,
        "run_manifest": manifest_name,
        "n_scenarios": len(eval_index),
        "definition": (
            "spec_pass = all checkers pass AND quality_score >= "
            f"{QUALITY_SCORE_MIN}; spec_adherence = mean(spec_pass) over the eval "
            "set; fail_rate = 1 - spec_adherence; the gate reads the BEST frontier "
            "cell only (the local base row is excluded from the ceiling claim). "
            "Every number here was recomputed from raw transcript rows, and every "
            "checker verdict was re-derived by running constraints.checkers.check "
            "on each row's own output; the stored verdicts were compared to those, "
            "never trusted. quality_score is the one input that cannot be replayed "
            "offline, so n_fail_from_quality_only records how many counted "
            "failures rest on the judge alone."),
        "thresholds": {
            "proceed_fail_rate_min": float(PROCEED_FAIL_RATE_MIN),
            "proceed_boundary": "inclusive: fail_rate == 0.20 PROCEEDs",
            "hard_stop_adherence_min": float(HARD_STOP_ADHERENCE_MIN),
            "quality_score_min": QUALITY_SCORE_MIN,
            "min_scenarios_per_cell": MIN_SCENARIOS_PER_CELL,
        },
        "ladder": {
            "rung": rung["rung"],
            "rung_source": rung["rung_source"],
            "rung_index": rung["rung_index"],
            "n_rungs": rung["n_rungs"],
            "exhausted": rung["exhausted"],
            "next_rung": rung["next_rung"],
            "rung_definition": rung["definition"],
            "observed_in_eval_set": observed,
        },
        "classification": {
            "frontier_providers": families,
            "local_base_providers": sorted({c["provider"] for c in local_base}),
            "n_frontier_cells": len(frontier),
            "n_local_base_cells": len(local_base),
            "rule": (f"provider in {sorted(FRONTIER_PROVIDERS)} -> frontier (counted "
                     f"in the gate); provider in {sorted(LOCAL_BASE_PROVIDERS)} -> "
                     f"local_base (recorded, excluded from the ceiling)"),
        },
        "cells": [{k: v for k, v in c.items() if not k.startswith("_")}
                  | {"counted_in_gate": c["classification"] == "frontier"}
                  for c in all_cells],
        "best_frontier_cell": {
            "model": best["model"],
            "strategy": best["strategy"],
            "spec_adherence": best["spec_adherence"],
            "fail_rate": best["fail_rate"],
            "n": best["n"],
            "n_spec_pass": best["n_spec_pass"],
            "n_fail_from_checkers": best["n_fail_from_checkers"],
            "n_fail_from_quality_only": best["n_fail_from_quality_only"],
        },
        "gate_fail_rate": float(fail_rate),
        "gate_spec_adherence": float(adherence),
        "verdict": verdict,
        "verdict_reason": reason,
    }
    report["verdict_line"] = (
        f"{verdict}: best frontier cell {best['model']} x {best['strategy']} "
        f"adherence {float(adherence):.4f} / fail_rate {float(fail_rate):.4f} "
        f"vs threshold >= {float(PROCEED_FAIL_RATE_MIN):.2f} "
        f"(n={best['n']}, rung {rung['rung']} {rung['rung_index']}/{rung['n_rungs']}"
        f"{', ladder exhausted' if rung['exhausted'] else ''}) — {reason}")
    return report, []


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="GATE G1 — the Prompt-Ceiling gate (R6/S1). Exits 0 only on "
                    "PROCEED. There is no flag that softens the threshold.")
    ap.add_argument("run_dir", nargs="?", default="runs/g1",
                    help="the G1 run directory (a file inside it also works)")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "g1.json"))
    # There is deliberately no --eval-set. The config pins the corpus; letting
    # the invocation (or the run manifest) name a different one is how a run
    # picks the exam it sits, and the report's eval_set_sha256 would then be
    # auditing whatever the caller chose.
    ap.add_argument("--rung", default=None,
                    help="ladder rung that produced this run, when the run manifest "
                         "does not record one (recorded in the report as cli)")
    ap.add_argument("--out", default=None,
                    help="where to write gate_report.json "
                         "(default: <run_dir>/gate_report.json)")
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    if run_dir.is_file():
        run_dir = run_dir.parent

    try:
        report, errors = build_report(run_dir, Path(args.config), args.rung)
    except GateError as exc:
        errors, report = [str(exc)], None

    if report is None:
        print(f"FAIL GATE G1 is not ratable on {run_dir} — {len(errors)} problem(s):",
              file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        stale = Path(args.out) if args.out else run_dir / "gate_report.json"
        if stale.is_dir():
            stale = stale / "gate_report.json"
        if stale.exists():
            print(f"  ! {stale} is STALE: it does not describe this run and was not "
                  f"overwritten", file=sys.stderr)
        print("HARD_STOP: the run is inconsistent; no verdict was issued.",
              file=sys.stderr)
        return EXIT_INVALID

    dest = Path(args.out) if args.out else run_dir / "gate_report.json"
    if dest.is_dir():
        dest = dest / "gate_report.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    cls = report["classification"]
    print(f"  ok   {cls['n_frontier_cells']} frontier cells "
          f"({', '.join(cls['frontier_providers'])}) fed the gate; "
          f"{cls['n_local_base_cells']} local base cell(s) "
          f"({', '.join(cls['local_base_providers'])}) recorded and excluded")
    print(f"  ok   every cell recomputed from raw transcript rows over all "
          f"{report['n_scenarios']} eval-set scenarios")
    print(f"  ok   every checker verdict re-derived from (constraint, output) "
          f"against {report['constraint_instances']}")
    print(f"  ok   eval set {report['eval_set']} "
          f"sha256:{report['eval_set_sha256'][:16]}… (pinned by the config)")
    print(f"  ok   report -> {dest}")
    print(report["verdict_line"])
    return EXIT_FOR_VERDICT[report["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
