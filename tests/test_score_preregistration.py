"""Bind scripts/score_preregistration.py -- it produces the graded numbers.

Network-free, model-free, GPU-free. Every fixture is built by hand so the
expected verdict is known by construction rather than read off a real run.
"""
import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def sp():
    spec = importlib.util.spec_from_file_location(
        "score_prereg", REPO_ROOT / "scripts" / "score_preregistration.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------- the thresholds
def test_thresholds_come_from_the_committed_document_not_from_code(sp):
    """THE ONE THAT MATTERS. If these are ever restated in the script, the
    frozen prediction and its grader can drift, which is the single failure a
    pre-registration cannot survive."""
    src = (REPO_ROOT / "scripts" / "score_preregistration.py").read_text()
    body = src.split("def thresholds", 1)[1].split("def load", 1)[0]
    assert "read_text()" in body, "thresholds() no longer reads the document"
    t = sp.thresholds()
    assert t["p1_banned_punctuation_below"] == 0.5
    assert t["p3_min_single_turn_holds"] == 9


def test_a_document_without_p1s_bar_is_refused_rather_than_defaulted(sp, tmp_path):
    doc = tmp_path / "prereg.md"
    doc.write_text("# no numeric bar anywhere in here\n")
    with pytest.raises(sp.PreregError, match="P1's numeric bar"):
        sp.thresholds(doc)


def test_an_edited_bar_changes_what_the_grader_demands(sp, tmp_path):
    """Editing the prediction after the fact must move the grader with it --
    visibly -- rather than leaving a stale constant in code."""
    doc = tmp_path / "prereg.md"
    doc.write_text("falls to below **0.900** for the v4 checkpoint\n")
    assert sp.thresholds(doc)["p1_banned_punctuation_below"] == 0.9


# ---------------------------------------------------------- the grading
def _corpus_run(tmp_path, punct_rate, heldout_passed=5, n=60):
    d = tmp_path / "corpus"; d.mkdir()
    (d / "cell.results.json").write_text(json.dumps({
        "metrics": {
            "by_constraint_type": {"types": {
                "banned_punctuation": {"checks": 41, "violations": 1,
                                       "violation_rate": punct_rate}}},
            "by_split": {"heldout_type": {"n": n, "passed": heldout_passed}},
        }}))
    return d


def _after(tmp_path, punct_hold=(), single_held=0, multi_held=0):
    rows = []
    punct = ["edge-punct-1", "ans-punct-decimal", "erode2-punct-list"]
    for i, pid in enumerate(punct):
        rows.append({"id": pid, "multiturn": pid.startswith("erode"),
                     "broke": pid not in punct_hold})
    for i in range(13):  # 13 more single-turn -> 15 single total (2 punct are single)
        rows.append({"id": f"s-{i}", "multiturn": False, "broke": i >= single_held})
    for i in range(4):   # 4 more multi-turn -> 5 total
        rows.append({"id": f"m-{i}", "multiturn": True, "broke": i >= multi_held})
    p = tmp_path / "after.json"
    p.write_text(json.dumps({"rows": rows}))
    return p


def test_p1_hits_below_the_bar(sp, tmp_path):
    rows = sp.grade(_corpus_run(tmp_path, 0.42), _after(tmp_path))
    assert rows[0][3] == "HIT"


def test_p1_is_partial_when_it_moved_but_did_not_clear_the_bar(sp, tmp_path):
    """0.5-0.5854 is a real finding, not a failure. Collapsing it into MISS
    would discard evidence that the mechanism moved a never-trained type."""
    rows = sp.grade(_corpus_run(tmp_path, 0.55), _after(tmp_path))
    assert rows[0][3] == "PARTIAL"


def test_p1_misses_when_it_did_not_move_off_the_published_rate(sp, tmp_path):
    rows = sp.grade(_corpus_run(tmp_path, 0.60), _after(tmp_path))
    assert rows[0][3] == "MISS"


def test_p4_reports_beating_the_prediction_rather_than_failing_it(sp, tmp_path):
    """P4 predicts multi-turn stays broken. If it does better, that is a
    finding, and calling it MISS with no qualifier would read as a regression."""
    rows = sp.grade(_corpus_run(tmp_path, 0.42), _after(tmp_path, multi_held=5))
    p4 = next(r for r in rows if r[0].startswith("P4"))
    assert "better than predicted" in p4[3]


def test_a_run_dir_with_two_result_cells_is_refused_not_guessed(sp, tmp_path):
    d = _corpus_run(tmp_path, 0.42)
    (d / "second.results.json").write_text("{}")
    with pytest.raises(sp.PreregError, match="expected exactly one"):
        sp.grade(d, _after(tmp_path))


def test_a_missing_after_run_is_named_rather_than_scored_as_zero(sp, tmp_path):
    with pytest.raises(sp.PreregError, match="does not exist"):
        sp.grade(_corpus_run(tmp_path, 0.42), tmp_path / "nope.json")


def test_a_corpus_without_the_banned_punctuation_cell_is_refused(sp, tmp_path):
    d = tmp_path / "c"; d.mkdir()
    (d / "cell.results.json").write_text(json.dumps(
        {"metrics": {"by_constraint_type": {"types": {}}, "by_split": {}}}))
    with pytest.raises(sp.PreregError, match="banned_punctuation"):
        sp.grade(d, _after(tmp_path))
