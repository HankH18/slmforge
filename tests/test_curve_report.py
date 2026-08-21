"""The curve is a graded ablation, so it has to be re-derivable, not readable.

Every assertion here binds a way the reported curve could drift away from the
rows it claims to summarise: a metric hand-written into the declaration, a
number hand-edited into `curve.json`, a denominator quietly inflated by rows
that were never scored, or an occurrence count lowered by a character the
checker folds away but a naive `str.count` does not.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "curve_report.py"
POINTS = REPO_ROOT / "runs" / "efficiency" / "points.json"
CURVE = REPO_ROOT / "runs" / "efficiency" / "curve.json"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("curve_report", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["curve_report"] = m
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------------- the declaration
def _decl(tmp_path, points, **top):
    body = {"eval_set": "data/scenarios/corpus.jsonl",
            "instances": "data/scenarios/constraint_instances.jsonl",
            "strategy": "zero_shot", "points": points}
    body.update(top)
    p = tmp_path / "points.json"
    p.write_text(json.dumps(body))
    return p


def _point(n, label="p", model="hf:x", run_dir="runs/mvp", **extra):
    out = {"n": n, "label": label, "model": model, "run_dir": run_dir}
    out.update(extra)
    return out


def test_declaration_refuses_a_metric_written_into_it(tmp_path, mod):
    """THE ONE THAT MATTERS. points.json says WHICH runs are curve points. A
    metric written there would be reported without ever being checked against a
    transcript, which is exactly the hand-assembled curve this script replaces."""
    path = _decl(tmp_path, [_point(1, spec_adherence=0.99)])
    with pytest.raises(mod.CurveError, match="spec_adherence"):
        mod.load_declaration(path)


def test_declaration_refuses_out_of_order_points(tmp_path, mod):
    path = _decl(tmp_path, [_point(846), _point(106)])
    with pytest.raises(mod.CurveError, match="ascending"):
        mod.load_declaration(path)


def test_declaration_refuses_duplicate_n(tmp_path, mod):
    path = _decl(tmp_path, [_point(106), _point(106, label="q")])
    with pytest.raises(mod.CurveError, match="[Dd]uplicate"):
        mod.load_declaration(path)


def test_declaration_requires_the_addressing_fields(tmp_path, mod):
    path = tmp_path / "points.json"
    path.write_text(json.dumps({"points": [{"n": 1, "label": "p"}]}))
    with pytest.raises(mod.CurveError, match="model"):
        mod.load_declaration(path)


def test_committed_declaration_loads(mod):
    decl = mod.load_declaration(POINTS)
    assert [p["n"] for p in decl["points"]] == [0, 106, 211, 423, 846]


# --------------------------------------------------- the occurrence count
def test_occurrences_use_the_checkers_fold_not_str_count(mod):
    """A zero-width space between two 'e's renders as nothing, so the checker
    folds it away and still sees both. Counting the raw string would too, but
    a FULLWIDTH 'ｅ' would not: `str.count('e')` misses it and reports a
    cleaner response than the checker fails."""
    assert mod.count_occurrences("s​ee", "e") == 2
    assert "ｅ".count("e") == 0          # what a naive count would say
    assert mod.count_occurrences("ｅ", "e") == 1
    assert mod.count_occurrences("Eé", "e") == 2   # case and diacritic fold


def test_occurrences_of_an_absent_letter_are_zero(mod):
    assert mod.count_occurrences("this holds no such thing", "z") == 0


def test_banned_letter_of_rejects_a_malformed_instance(mod):
    with pytest.raises(mod.CurveError, match="params.letter"):
        mod.banned_letter_of({"id": "c-chr-x", "params": {"letter": "ee"}})


# ------------------------------------------------------- the derivation
def test_quarantined_rows_are_excluded_from_every_denominator(mod):
    """An empty output satisfies every checker vacuously. A quarantined row
    counted as a check donates a free pass to each of its constraint types and
    lowers the violation rate of a model that never produced anything."""
    scen = {"scenario_id": "s-1", "constraints": ["c-lex-001"], "split": "train",
            "pressure": "clean", "k": 2}
    inst = {"c-lex-001": {"id": "c-lex-001", "type": "banned_words",
                          "params": {"words": ["latte"]}, "family": "lexical",
                          "split": "train"}}
    rows = [{"scenario_id": "s-1", "output": "", "quality_score": 5,
             "generator_error": "empty completion", "spec_pass": None}]
    got = mod.derive_breakdowns(rows, {"s-1": scen}, inst)
    assert got["n_scored"] == 0
    assert got["by_constraint_type"] == {}
    assert got["by_split"] == {}


def test_a_scored_row_counts_once_per_constraint(mod):
    scen = {"scenario_id": "s-1", "constraints": ["c-lex-001", "c-chr-001"],
            "split": "train", "pressure": "clean", "k": 2}
    inst = {"c-lex-001": {"id": "c-lex-001", "type": "banned_words",
                          "params": {"words": ["latte"]}, "family": "lexical",
                          "split": "train"},
            "c-chr-001": {"id": "c-chr-001", "type": "banned_letter",
                          "params": {"letter": "e"}, "family": "character",
                          "split": "train"}}
    rows = [{"scenario_id": "s-1", "output": "a latte, and three e's",
             "quality_score": 5, "spec_pass": False}]
    got = mod.derive_breakdowns(rows, {"s-1": scen}, inst)
    assert got["n_scored"] == 1
    assert got["by_constraint_type"]["banned_words"] == {
        "checks": 1, "violations": 1, "violation_rate": 1.0}
    assert got["banned_letter"]["rows"] == 1
    # the row violates a constraint, so it cannot be a split pass however the
    # judge scored it
    assert got["by_split"]["train"]["passed"] == 0


def test_quality_score_below_the_threshold_blocks_a_split_pass(mod):
    """spec_pass is `all checkers pass AND quality >= 3`. A clean response the
    judge called degenerate is not a pass, and the split breakdown has to use
    the same definition as the headline or the two tables contradict."""
    scen = {"scenario_id": "s-1", "constraints": ["c-lex-001"], "split": "train",
            "pressure": "clean", "k": 2}
    inst = {"c-lex-001": {"id": "c-lex-001", "type": "banned_words",
                          "params": {"words": ["latte"]}, "family": "lexical",
                          "split": "train"}}
    clean = "a plain flat white, nothing forbidden here"
    assert mod.derive_breakdowns(
        [{"scenario_id": "s-1", "output": clean, "quality_score": 3,
          "spec_pass": True}], {"s-1": scen}, inst)["by_split"]["train"]["passed"] == 1
    assert mod.derive_breakdowns(
        [{"scenario_id": "s-1", "output": clean, "quality_score": 2,
          "spec_pass": False}], {"s-1": scen}, inst)["by_split"]["train"]["passed"] == 0


# ------------------------------------------------- the committed artifact
def test_committed_curve_matches_its_transcripts():
    """`--check` is the gate: it re-derives every point and refuses to agree
    with a curve.json that the rows do not produce."""
    proc = subprocess.run([sys.executable, str(SCRIPT), "--check"],
                          cwd=REPO_ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_check_rejects_a_hand_edited_curve(tmp_path):
    """The failure mode this whole script exists for: a number typed into the
    report by hand. Copy the real curve, move one digit, and the gate must
    refuse it."""
    curve = json.loads(CURVE.read_text())
    curve["points"][-1]["spec_adherence"] = 0.99
    tampered = tmp_path / "curve.json"
    tampered.write_text(json.dumps(curve, indent=2, ensure_ascii=False) + "\n")
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--check", "--out", str(tampered)],
        cwd=REPO_ROOT, capture_output=True, text=True)
    assert proc.returncode == 1
    assert "not what the transcripts re-derive to" in proc.stderr


def test_every_committed_point_reports_both_splits():
    """The split decomposition is the curve's actual finding. A point missing
    one of the two splits would silently drop half the evidence."""
    curve = json.loads(CURVE.read_text())
    for point in curve["points"]:
        assert set(point["by_split"]) == {"train", "heldout_type"}, point["label"]
        assert point["by_split"]["train"]["n"] + point["by_split"]["heldout_type"]["n"] \
            == point["n_scored"], point["label"]
