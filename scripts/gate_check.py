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
  * spec_pass is itself recomputed from each row's own checker_results and
    quality_score (the pinned definition: all checkers pass AND quality >= 3).
    A row that stores spec_pass=true while carrying a failing checker fails.
  * the rows must COVER the eval set exactly - one row per scenario, no
    duplicates, no extras, no subset. Authentic, correctly-scored rows can still
    be selectively dropped to re-weight a cell's mean, so coverage is checked
    even when every surviving row is perfect.
  * booleans are not numbers and strings are not numbers. Every stored scalar is
    type-checked before it is compared (`isinstance(True, int)` is True in
    Python, and that bypass has defeated guards in this repo).
  * the ladder rung a run claims is verified against the eval set that run used
    (max k actually reached, adversarial share actually met), because HARD_STOP
    is only legitimate on the last rung.

There is deliberately NO flag that softens the threshold, and none may be added.
The only verdict that exits 0 is PROCEED.

Exit codes:  0 PROCEED  |  1 invalid/inconsistent run  |  2 ESCALATE  |  3 HARD_STOP

This module does NOT import eval.py. eval.py is mid-pivot (T-3) and its
pre-pivot aggregate() reads gold_label/pred_class fields that no longer exist in
the transcript contract; importing it here would couple the gate to a moving
file and would not compute the right number anyway. The one definition this file
restates is the pinned Design contract `spec_pass = all checkers pass AND
quality_score >= 3`, which is the thing the gate exists to independently verify.
"""
import argparse
import json
import math
import sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------- thresholds
# R6. Exact rationals, not floats: 24/120 as a float is 0.19999999999999996,
# which would flip a legitimate PROCEED at the 0.20 boundary into ESCALATE.
PROCEED_FAIL_RATE_MIN = Fraction("0.20")   # fail_rate >= 0.20  -> PROCEED
HARD_STOP_ADHERENCE_MIN = Fraction("0.90")  # adherence >= 0.90 after the full ladder
QUALITY_SCORE_MIN = 3                      # Design: spec_pass needs quality >= 3
QUALITY_SCORE_RANGE = (1, 5)
MIN_SCENARIOS_PER_CELL = 30                # R6: ">= 30 scenarios per cell"

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


def resolve_path(raw, run_dir):
    """Resolve a config/manifest path. Deterministic order, all candidates named."""
    p = Path(raw)
    if p.is_absolute():
        return p if p.exists() else None
    for base in (Path.cwd(), Path(run_dir), REPO_ROOT):
        cand = base / p
        if cand.exists():
            return cand
    return None


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
    current = ladder.get("current")
    if current is not None and current not in seen:
        raise GateError(f"{path}: ladder.current {current!r} is not one of the "
                        f"declared rungs {sorted(seen)}")
    return cfg


def expected_cells(cfg):
    models = [cfg["models"]["local_base"]] + list(cfg["models"]["frontier"])
    return [(m, s) for m in models for s in EXPECTED_STRATEGIES]


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
def score_cell(results_path, transcripts_path, model, strategy, eval_index, errors):
    """Recompute this cell from its raw rows. Returns a stats dict or None.

    Nothing stored in the row is trusted: spec_pass is re-derived from the row's
    own checker_results + quality_score, and the row set must cover the eval set
    exactly.
    """
    numbered = read_jsonl(transcripts_path)
    tag = transcripts_path.name

    if not numbered:
        errors.append(f"{tag}: no transcript rows")
        return None

    seen = {}
    n_pass = 0
    n_adv = 0
    n_adv_pass = 0
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
        if not isinstance(row["output"], str):
            errors.append(f"{where}: output is {describe(row['output'])}, not a string")
            ok = False
        if not is_nonempty_str(row["quality_reasoning"]):
            errors.append(f"{where}: quality_reasoning is "
                          f"{describe(row['quality_reasoning'])} — an unexplained "
                          f"quality score is not a judged score")
            ok = False

        # --- checker results must cover exactly this scenario's constraints
        crs = row["checker_results"]
        row_ok = True
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
        recomputed = all(cr["pass"] for cr in crs) and q >= QUALITY_SCORE_MIN
        if recomputed != stored:
            errors.append(
                f"{where} (scenario {sid}): spec_pass == {stored!r} but the pinned "
                f"definition (all checkers pass AND quality_score >= "
                f"{QUALITY_SCORE_MIN}) gives {recomputed!r} from this row's own "
                f"checker_results/quality_score")
            ok = False
            continue

        if recomputed:
            n_pass += 1
        if scen["pressure"] == "adversarial":
            n_adv += 1
            if recomputed:
                n_adv_pass += 1

    # --- coverage: the rows must BE the eval set, not a flattering slice of it
    missing_ids = sorted(set(eval_index) - set(seen))
    if missing_ids:
        errors.append(
            f"{tag}: {len(missing_ids)} of {len(eval_index)} eval-set scenarios have "
            f"no row (e.g. {missing_ids[:5]}) — a strict subset re-weights "
            f"spec_adherence even when every surviving row is correctly scored")
        ok = False

    if not ok:
        return None

    n = len(seen)
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
            if not is_real_int(metrics["n"]):
                errors.append(f"{where}: metrics.n is {describe(metrics['n'])}, "
                              f"not an integer")
                stored_ok = False
            elif metrics["n"] != n:
                errors.append(f"{where}: metrics.n == {metrics['n']!r} but {tag} "
                              f"holds {n} scored rows")
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
        "results_json": results_path.name,
        "transcripts_jsonl": transcripts_path.name,
        "_adherence_exact": adherence,
        "_fail_rate_exact": fail_rate,
    }


def load_cells(run_dir, cfg, eval_index, errors):
    """Every *.results.json in run_dir, recomputed. Returns {(model, strategy): stats}."""
    paths = sorted(run_dir.glob("*.results.json"))
    if not paths:
        errors.append(f"{run_dir}: no *.results.json — nothing to gate on. Run "
                      f"run_ablation.py --config configs/g1.json first")
        return {}

    allowed_models = set(cfg["models"]["frontier"]) | {cfg["models"]["local_base"]}
    cells = {}
    for rp in paths:
        results = read_json(rp)
        if not isinstance(results, dict):
            errors.append(f"{rp.name}: results is not a JSON object")
            continue
        model = results.get("model")
        strategy = results.get("strategy")
        if not is_nonempty_str(model) or not is_nonempty_str(strategy):
            errors.append(f"{rp.name}: model/strategy must be non-empty strings, "
                          f"got model={describe(model)} strategy={describe(strategy)}")
            continue
        # classification first: an unclassifiable provider gets told so, rather
        # than being reported as merely "unpinned"
        if classify_provider(model) == "unknown":
            errors.append(f"{rp.name}: model {model!r} is neither a frontier family "
                          f"{sorted(FRONTIER_PROVIDERS)} nor the local base "
                          f"{sorted(LOCAL_BASE_PROVIDERS)}; it cannot be classified")
            continue
        if model not in allowed_models:
            errors.append(f"{rp.name}: model {model!r} is not one of the models "
                          f"configs/g1.json pins ({sorted(allowed_models)}) — a cell "
                          f"from an unpinned model cannot feed the gate")
            continue
        if strategy not in EXPECTED_STRATEGIES:
            errors.append(f"{rp.name}: strategy {strategy!r} is not one of "
                          f"{list(EXPECTED_STRATEGIES)}")
            continue
        key = (model, strategy)
        if key in cells:
            errors.append(f"{rp.name}: duplicate cell for {model} x {strategy}")
            continue

        raw_tp = results.get("transcripts")
        if is_nonempty_str(raw_tp):
            tp = resolve_path(raw_tp, run_dir)
            if tp is None:
                errors.append(f"{rp.name}: declares transcripts {raw_tp!r}, which "
                              f"does not exist")
                continue
        else:
            tp = rp.parent / (rp.name[:-len(".results.json")] + ".transcripts.jsonl")
            if not tp.exists():
                errors.append(f"{tp.name}: missing — the gate re-derives every "
                              f"aggregate from raw transcript rows and refuses to "
                              f"rate a cell that ships only its summary")
                continue

        stats = score_cell(rp, tp, model, strategy, eval_index, errors)
        cells[key] = stats  # None marks a cell that failed; grid check still sees it
    return cells


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


def verify_rung_against_eval_set(rung, eval_index, eval_path, errors):
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
    return {"k_values": observed_k, "n_adversarial": n_adv, "n_scenarios": n,
            "adversarial_share": float(Fraction(n_adv, n))}


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
def build_report(run_dir, config_path, eval_set_arg=None, cli_rung=None):
    """(report, errors). A report is returned only when the run is fully consistent."""
    errors = []
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return None, [f"{run_dir} does not exist or is not a directory"]

    cfg = load_config(config_path)

    manifest, manifest_name = None, None
    for name in MANIFEST_NAMES:
        p = run_dir / name
        if p.exists():
            manifest, manifest_name = read_json(p), name
            break
    if manifest is not None and not isinstance(manifest, dict):
        return None, [f"{manifest_name}: run manifest is not a JSON object"]

    raw_eval = (eval_set_arg
                or (manifest or {}).get("eval_set")
                or cfg.get("eval_set"))
    if not is_nonempty_str(raw_eval):
        return None, ["no eval set path: not given with --eval-set, not recorded in "
                      "the run manifest, not pinned in the config"]
    eval_path = resolve_path(raw_eval, run_dir)
    if eval_path is None:
        return None, [
            f"eval set {raw_eval!r} not found (looked in {Path.cwd()}, {run_dir}, "
            f"{REPO_ROOT}). The gate cannot check that the transcripts cover the "
            f"eval set without it, so it fails rather than rating a slice"]
    eval_index = load_eval_set(eval_path)

    rung = resolve_rung(cfg, manifest, cli_rung, errors)
    observed = None
    if rung is not None:
        observed = verify_rung_against_eval_set(rung, eval_index, eval_path, errors)

    cells = load_cells(run_dir, cfg, eval_index, errors)
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
        "run_dir": str(run_dir),
        "config": str(config_path),
        "eval_set": str(eval_path),
        "run_manifest": manifest_name,
        "n_scenarios": len(eval_index),
        "definition": (
            "spec_pass = all checkers pass AND quality_score >= "
            f"{QUALITY_SCORE_MIN}; spec_adherence = mean(spec_pass) over the eval "
            "set; fail_rate = 1 - spec_adherence; the gate reads the BEST frontier "
            "cell only (the local base row is excluded from the ceiling claim). "
            "Every number here was recomputed from raw transcript rows."),
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
    ap.add_argument("--eval-set", default=None,
                    help="override the eval set path recorded in the run manifest")
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
        report, errors = build_report(run_dir, Path(args.config), args.eval_set,
                                      args.rung)
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
    print(f"  ok   report -> {dest}")
    print(report["verdict_line"])
    return EXIT_FOR_VERDICT[report["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
