"""Adapting a foreign eval set must change what it is CALLED, never what it MEANS.

The harness pins constraint ids to a regex and pins each type's family and split,
so a staff-supplied constraint library dies at scoring time. Adapting it is the
difference between grading the held-out set and not grading it — but an adapter
that quietly alters semantics produces numbers for a different eval set than the
one supplied, which is worse than refusing.

Every assertion here binds one of those two failures: a set that still will not
score, or a set that scores differently than it should.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from constraints import registry  # noqa: E402
from constraints.checkers import check  # noqa: E402

SCRIPT = REPO_ROOT / "scripts" / "adapt_evalset.py"
CORPUS = REPO_ROOT / "data" / "scenarios" / "corpus.jsonl"
INSTANCES = REPO_ROOT / "data" / "scenarios" / "constraint_instances.jsonl"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("adapt_evalset", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["adapt_evalset"] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def instances():
    return [json.loads(l) for l in INSTANCES.read_text().splitlines() if l.strip()]


@pytest.fixture(scope="module")
def corpus():
    return [json.loads(l) for l in CORPUS.read_text().splitlines() if l.strip()]


def _foreign(instances):
    """The shipped library with every id renamed to a scheme the registry rejects."""
    return [dict(c, id=f"staff-{i:04d}") for i, c in enumerate(instances)]


# ------------------------------------------------------- semantics preserved
def test_type_and_params_survive_byte_for_byte(mod, instances):
    """THE ONE THAT MATTERS. type + params are the whole behavioural content of
    a constraint. If either changes, the adapted set measures something else."""
    got = mod.adapt_constraints(_foreign(instances))
    assert len(got["constraints"]) == len(instances)
    for original, adapted in zip(instances, got["constraints"]):
        assert adapted["type"] == original["type"]
        assert adapted["params"] == original["params"]


def test_adapted_constraints_give_identical_checker_verdicts(mod, instances):
    """The behavioural proof, not the structural one: run the real checker over
    the same text with the original and the adapted constraint and require the
    same verdict and the same evidence."""
    probes = [
        "The library reopens on Monday; however, the annexe stays shut.",
        "A plain sentence with no forbidden material in it at all.",
        '- a list item\n- another one, with "quotes" and, commas',
        "Every sentence starts simply. Some do not. Endings vary.",
    ]
    adapted = mod.adapt_constraints(_foreign(instances))["constraints"]
    compared = 0
    for original, new in zip(instances, adapted):
        for text in probes:
            before, after = check(original, text), check(new, text)
            assert before["pass"] == after["pass"], (original["id"], text)
            assert before["evidence"] == after["evidence"], (original["id"], text)
            compared += 1
    assert compared == len(instances) * len(probes)


def test_scenarios_keep_every_field_except_the_constraint_ids(mod, corpus, instances):
    foreign = _foreign(instances)
    mapping = mod.adapt_constraints(foreign)["mapping"]
    rename = {c["id"]: f"staff-{i:04d}" for i, c in enumerate(instances)}
    staff_scen = [dict(s, constraints=[rename[c] for c in s["constraints"]])
                  for s in corpus[:10]]
    out = mod.adapt_scenarios(staff_scen, mapping)["scenarios"]
    for before, after in zip(staff_scen, out):
        assert set(before) == set(after)
        for key in before:
            if key != "constraints":
                assert before[key] == after[key], key
        assert len(after["constraints"]) == len(before["constraints"])


def test_new_ids_satisfy_the_registry_regex(mod, instances):
    for adapted in mod.adapt_constraints(_foreign(instances))["constraints"]:
        assert registry.ID_RE.match(adapted["id"]), adapted["id"]
        assert adapted["family"] == registry.family_of(adapted["type"])
        assert adapted["split"] == registry.split_of(adapted["type"])


def test_the_mapping_is_a_bijection(mod, instances):
    got = mod.adapt_constraints(_foreign(instances))
    assert len(set(got["mapping"].values())) == len(got["mapping"])
    assert set(got["mapping"].values()) == {c["id"] for c in got["constraints"]}


# --------------------------------------------------------------- refusals
def test_an_unimplemented_constraint_type_is_refused_not_dropped(mod, instances):
    """Dropping it and scoring the rest reports a number for a different eval
    set. The adapter has to say so instead."""
    rows = _foreign(instances) + [{"id": "staff-9999", "type": "max_words",
                                   "params": {"n": 50}}]
    with pytest.raises(mod.AdaptError, match="max_words"):
        mod.adapt_constraints(rows)


def test_two_constraints_that_would_merge_are_refused(mod):
    """Identical once the non-registry keys are dropped: adapting them would
    collapse two distinct constraints into one and re-weight every rate."""
    rows = [
        {"id": "a", "type": "banned_words", "params": {"words": ["x"]}, "note": "A"},
        {"id": "b", "type": "banned_words", "params": {"words": ["x"]}, "note": "B"},
    ]
    with pytest.raises(mod.AdaptError, match="merge"):
        mod.adapt_constraints(rows)


def test_a_duplicate_source_id_is_refused(mod):
    rows = [{"id": "a", "type": "banned_words", "params": {"words": ["x"]}},
            {"id": "a", "type": "banned_words", "params": {"words": ["y"]}}]
    with pytest.raises(mod.AdaptError, match="duplicate"):
        mod.adapt_constraints(rows)


def test_a_scenario_referencing_an_unknown_constraint_is_refused(mod, instances):
    mapping = mod.adapt_constraints(_foreign(instances))["mapping"]
    with pytest.raises(mod.AdaptError, match="not in the supplied constraints"):
        mod.adapt_scenarios([{"scenario_id": "s-1", "task": "t",
                              "constraints": ["nope"]}], mapping)


def test_dropped_keys_are_recorded_not_silently_discarded(mod):
    rows = [{"id": "a", "type": "banned_words", "params": {"words": ["x"]},
             "difficulty": "hard", "author": "staff"}]
    got = mod.adapt_constraints(rows)
    assert got["dropped_keys"]["a"] == ["author", "difficulty"]


def test_missing_scenario_fields_are_reported(mod, instances):
    """The adapter must not invent pressure/split/k. It reports them so the
    preflight failure is expected rather than surprising."""
    mapping = mod.adapt_constraints(_foreign(instances))["mapping"]
    row = {"scenario_id": "s-1", "task": "t",
           "constraints": [next(iter(mapping))]}
    got = mod.adapt_scenarios([row], mapping)
    assert got["missing_fields"] == {"pressure": 1, "split": 1, "k": 1}


# ------------------------------------------------------------------ the CLI
def test_cli_output_passes_preflight(tmp_path, corpus, instances):
    """End to end: a set the harness refuses goes in, a set it accepts comes
    out, and the transformation is written down."""
    rename = {c["id"]: f"staff-{i:04d}" for i, c in enumerate(instances)}
    (tmp_path / "cons.jsonl").write_text(
        "".join(json.dumps(dict(c, id=rename[c["id"]])) + "\n" for c in instances))
    (tmp_path / "scen.jsonl").write_text("".join(
        json.dumps(dict(s, constraints=[rename[c] for c in s["constraints"]])) + "\n"
        for s in corpus[:8]))

    out = tmp_path / "adapted"
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--scenarios", str(tmp_path / "scen.jsonl"),
         "--constraints", str(tmp_path / "cons.jsonl"), "--out-dir", str(out)],
        cwd=REPO_ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr

    record = json.loads((out / "adaptation.json").read_text())
    assert len(record["id_mapping"]) == len(instances)

    pre = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "preflight_evalset.py"),
         str(out / "scenarios.jsonl"), str(out / "constraint_instances.jsonl")],
        cwd=REPO_ROOT, capture_output=True, text=True)
    assert pre.returncode == 0, pre.stdout + pre.stderr
    assert "PREFLIGHT PASS" in pre.stdout
