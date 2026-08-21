"""The shipped training data is audited by the suite, not by memory.

An adversarial verifier found datagen/audit.py was a manual CLI that nothing
called -- not check_loop, not modal_app, not any test -- so the only gate on the
dataset the project trains and reports on was a human remembering the command.
It also found the auditor certified verbatim eval-corpus prose as clean, because
it re-derived only (task, constraint-set) pairs while datagen's own documented
standard is task-level. Both are bound here.
"""
import json
from pathlib import Path

import pytest

from datagen.audit import audit
from scenarios.paths import load_corpus, load_instances, load_smoke

REPO_ROOT = Path(__file__).resolve().parent.parent
# Training datasets only. The sidecars -- <name>.rejected.jsonl and
# <name>.dpo_negatives.jsonl -- deliberately hold VIOLATING rows (that is their
# purpose: candidate DPO negatives), so auditing them for compliance is a category
# error. A training file has exactly one dot-segment before .jsonl.
SHIPPED = sorted(p for p in (REPO_ROOT / "data" / "train").glob("*.jsonl")
                 if p.name.count(".") == 1)
SIDECARS = sorted(p for p in (REPO_ROOT / "data" / "train").glob("*.jsonl")
                  if p.name.count(".") > 1)


@pytest.mark.parametrize("path", SHIPPED, ids=lambda p: p.name)
def test_every_shipped_training_file_passes_its_own_audit(path):
    """If this fails, the dataset is contaminated or non-compliant -- and nothing
    else in the project would notice."""
    result = audit(str(path), min_rows=1)
    violations, detail = result["violations"], result["detail"]
    assert not violations, f"{path.name} fails its audit: {violations}\n{detail}"


def test_there_is_at_least_one_shipped_training_file():
    """Guards the parametrization above from silently covering nothing."""
    assert SHIPPED, "no data/train/*.jsonl; the audit parametrization covers nothing"


def _row_with_task(task_text, corpus_row, instances):
    """A schema-valid, genuinely compliant row carrying the given task text."""
    from harness import build_constraint_lookup
    from scenarios.render import render_user_message
    from scenarios.reachability import build_witness
    lookup = build_constraint_lookup(instances.values())
    cids = [c for c in corpus_row["constraints"]
            if instances[c]["split"] == "train"] or corpus_row["constraints"][:1]
    scen = {"scenario_id": "planted", "task": task_text, "constraints": cids,
            "k": len(cids), "pressure": "clean", "split": "train"}
    witness = build_witness([instances[c] for c in cids], sentences=4)
    if witness is None:
        pytest.skip("no witness available for this constraint set")
    return {"messages": [{"role": "user", "content": render_user_message(scen, lookup)},
                         {"role": "assistant", "content": witness}],
            "constraints": cids,
            "provenance": {"teacher": "test", "attempt": 1},
            "scenario": scen}


def test_audit_rejects_verbatim_eval_prose_under_a_different_constraint_set(tmp_path):
    """FINDING B. Reusing an eval task's exact prose under other constraints was
    certified 'zero eval overlap'. A model that has seen the exact text it will be
    graded on is contaminated regardless of which constraints accompanied it."""
    instances = load_instances()
    corpus = list(load_corpus())
    victim = corpus[0]
    row = _row_with_task(victim["task"], victim, instances)
    f = tmp_path / "planted.jsonl"
    f.write_text(json.dumps(row) + "\n")
    result = audit(str(f), min_rows=1)
    violations = result["violations"]
    assert "eval_task_overlap" in violations, (
        f"verbatim eval prose was certified clean; violations={violations}")


def test_audit_still_rejects_the_exact_eval_pair(tmp_path):
    """The pair-level check must survive the task-level addition."""
    instances = load_instances()
    victim = list(load_corpus())[0]
    row = _row_with_task(victim["task"], victim, instances)
    row["constraints"] = victim["constraints"]
    row["scenario"]["constraints"] = victim["constraints"]
    row["scenario"]["k"] = len(victim["constraints"])
    f = tmp_path / "pair.jsonl"
    f.write_text(json.dumps(row) + "\n")
    result = audit(str(f), min_rows=1)
    assert result["violations"], "the exact eval pair was certified clean"


def test_audit_is_not_vacuous_on_an_empty_file(tmp_path):
    f = tmp_path / "empty.jsonl"
    f.write_text("")
    result = audit(str(f), min_rows=1)
    assert result["violations"], "an empty dataset passed the audit"


@pytest.mark.parametrize("path", SIDECARS, ids=lambda p: p.name)
def test_sidecars_are_not_mistaken_for_training_data(path):
    """The rejected/DPO-negative sidecars hold violating rows on purpose. This
    pins that they are NOT in the audited set, so a future rename cannot quietly
    slip violating rows into training."""
    assert path not in SHIPPED
    assert path.name.endswith((".rejected.jsonl", ".dpo_negatives.jsonl")), (
        f"{path.name} is neither a training file nor a recognised sidecar -- it is "
        f"unaudited and unclassified")


# ==========================================================================
# The nested family the data-efficiency curve rests on
# ==========================================================================

def _row_key(row):
    """Content identity, ignoring the wall-clock stamp a rerun necessarily moves."""
    import copy
    row = copy.deepcopy(row)
    row.get("provenance", {}).pop("generated_utc", None)
    return json.dumps(row, sort_keys=True)


def test_v3_contains_every_v2_row():
    """The upward extension of the curve is only interpretable if the larger
    corpus CONTAINS the smaller one -- otherwise the points differ by which rows
    they drew as well as by how many, and the difference is unattributable.

    Set inclusion, deliberately not prefix equality. The v2 run abandoned rows on
    a provider error, a failed call is never cached, so those requests were
    re-issued during the v3 run and some succeeded; the two files agree for a
    few hundred rows and then interleave. Asserting prefix equality here would
    fail on a correct dataset.
    """
    v2 = REPO_ROOT / "data" / "train" / "v2.jsonl"
    v3 = REPO_ROOT / "data" / "train" / "v3.jsonl"
    if not v3.exists():
        pytest.skip("v3 has not been generated")
    keys2 = {_row_key(json.loads(l)) for l in v2.read_text().splitlines() if l.strip()}
    keys3 = {_row_key(json.loads(l)) for l in v3.read_text().splitlines() if l.strip()}
    missing = keys2 - keys3
    assert not missing, (
        f"{len(missing)} of {len(keys2)} v2 rows are absent from v3; the curve's "
        f"nested family is broken and the points are no longer comparable")
    assert len(keys3) > len(keys2), "v3 must be strictly larger than v2"
