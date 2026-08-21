"""Two checkpoints on one eval set are one sample measured twice.

The reading this guards against is real and was made about this project's own
numbers: "v1 scored 7/69 and v2 scored 6/69, so v2 is ten times worse on the
delta". Those are the same 69 scenarios differing by one row. Every assertion
here binds a way the paired comparison could go back to being an unpaired one.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "compare_checkpoints.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("compare_checkpoints", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["compare_checkpoints"] = m
    spec.loader.exec_module(m)
    return m


LETTER = {"id": "c-chr-001", "type": "banned_letter", "params": {"letter": "e"},
          "family": "character", "split": "train"}
WORDS = {"id": "c-lex-001", "type": "banned_words", "params": {"words": ["latte"]},
         "family": "lexical", "split": "train"}


def _scen(sid, constraints, pressure="clean"):
    return {"scenario_id": sid, "task": "t", "constraints": constraints, "k": 2,
            "pressure": pressure, "split": "train"}


def _row(sid, output, spec_pass):
    return {"scenario_id": sid, "output": output, "spec_pass": spec_pass,
            "quality_score": 4}


def test_only_discordant_pairs_carry_information(mod):
    """Rows where both checkpoints agree cancel. Ten shared passes and ten
    shared failures must not move the p-value away from 1.0."""
    index, a, b = {}, [], []
    for i in range(20):
        sid = f"s-{i}"
        index[sid] = _scen(sid, ["c-lex-001"])
        verdict = i < 10
        a.append(_row(sid, "clean", verdict))
        b.append(_row(sid, "clean", verdict))
    got = mod.compare(a, b, index, {"c-lex-001": WORDS})
    assert got["spec_pass"]["overall"]["discordant"] == 0
    assert got["spec_pass"]["overall"]["mcnemar_exact_p"] == 1.0


def test_a_one_row_difference_is_not_significant(mod):
    """THE ONE THAT MATTERS. The exact case that was misread: the two cells
    differ by a single scenario, and the test must say so."""
    index, a, b = {}, [], []
    for i in range(69):
        sid = f"s-{i}"
        index[sid] = _scen(sid, ["c-lex-001"], pressure="adversarial")
        a.append(_row(sid, "clean", i < 7))
        b.append(_row(sid, "clean", i < 6))
    got = mod.compare(a, b, index, {"c-lex-001": WORDS})["spec_pass"]["adversarial"]
    assert (got["a_only_pass"], got["b_only_pass"]) == (1, 0)
    assert got["mcnemar_exact_p"] == 1.0


def test_a_consistent_one_sided_difference_is_significant(mod):
    """The other half: when one checkpoint wins on many rows and loses on none,
    the same test must resolve it."""
    index, a, b = {}, [], []
    for i in range(30):
        sid = f"s-{i}"
        index[sid] = _scen(sid, ["c-lex-001"])
        a.append(_row(sid, "clean", False))
        b.append(_row(sid, "clean", True))
    got = mod.compare(a, b, index, {"c-lex-001": WORDS})["spec_pass"]["overall"]
    assert got["b_only_pass"] == 30
    assert got["mcnemar_exact_p"] < 1e-6


def test_unscored_rows_are_never_paired(mod):
    """A quarantined row has no verdict. Pairing it against a real one would
    invent a disagreement."""
    index = {"s-1": _scen("s-1", ["c-lex-001"])}
    a = [{"scenario_id": "s-1", "output": None, "spec_pass": None,
          "generator_error": "empty"}]
    b = [_row("s-1", "clean", True)]
    got = mod.compare(a, b, index, {"c-lex-001": WORDS})
    assert got["spec_pass"]["overall"]["paired_rows"] == 0
    assert got["banned_letter_occurrences"]["paired_rows"] == 0


def test_occurrences_counted_only_where_the_constraint_exists(mod):
    index = {"s-1": _scen("s-1", ["c-lex-001"]),
             "s-2": _scen("s-2", ["c-chr-001"])}
    inst = {"c-lex-001": WORDS, "c-chr-001": LETTER}
    a = [_row("s-1", "no letters banned here", False), _row("s-2", "eee", False)]
    b = [_row("s-1", "no letters banned here", False), _row("s-2", "e", False)]
    got = mod.compare(a, b, index, inst)["banned_letter_occurrences"]
    assert got["paired_rows"] == 1          # s-1 has no banned_letter constraint
    assert (got["a_median"], got["b_median"]) == (3, 1)


def test_occurrences_use_the_checkers_fold(mod):
    """A fullwidth 'e' is an occurrence to the checker; str.count would miss it
    and report a cleaner response than the checker fails."""
    index = {"s-1": _scen("s-1", ["c-chr-001"])}
    inst = {"c-chr-001": LETTER}
    assert mod.banned_letter_occurrences("ｅｅ", index["s-1"], inst) == 2
    assert "ｅｅ".count("e") == 0


def test_no_shared_scenarios_is_an_error_not_an_empty_result(mod):
    """Comparing two cells that share nothing must fail loudly. Reporting
    'p = 1.0, no difference' over an empty intersection is the worst outcome."""
    with pytest.raises(mod.CompareError, match="share no scenario"):
        mod.compare([_row("s-1", "x", True)], [_row("s-2", "x", True)],
                    {"s-1": _scen("s-1", ["c-lex-001"]),
                     "s-2": _scen("s-2", ["c-lex-001"])},
                    {"c-lex-001": WORDS})


def test_rows_are_joined_by_scenario_id_not_by_position(mod):
    """THE PROPERTY THE DOCSTRING CLAIMS. Every other test in this file builds
    rows_a and rows_b in the same order, so a positional join would pass all of
    them. Shuffle one side and the answer must not move."""
    index, a, b = {}, [], []
    for i in range(12):
        sid = f"s-{i:02d}"
        index[sid] = _scen(sid, ["c-chr-001"])
        a.append(_row(sid, "e" * 10, i < 6))
        b.append(_row(sid, "e" * 2, i < 6))
    inst = {"c-chr-001": LETTER}

    ordered = mod.compare(a, b, index, inst)
    reversed_b = mod.compare(a, list(reversed(b)), index, inst)
    assert ordered == reversed_b

    # And a genuine per-row difference must survive reordering: give ONE
    # scenario a different value on the b side and check it lands on that
    # scenario, not on whichever row happens to sit opposite it.
    b_odd = [dict(r, output="e" * 40) if r["scenario_id"] == "s-00" else r for r in b]
    straight = mod.compare(a, b_odd, index, inst)
    shuffled = mod.compare(a, list(reversed(b_odd)), index, inst)
    assert straight == shuffled
    assert straight["banned_letter_occurrences"]["b_higher_on"] == 1


def test_a_negative_count_raises_instead_of_reporting_certainty(mod):
    """range(k+1) is empty for k < 0, so the sum is 0 and the function would
    answer p = 0.0 -- a perfect p-value -- for an impossible input."""
    with pytest.raises(mod.CompareError, match="counts"):
        mod.two_sided_sign_p(-1, 5)


def test_sign_p_is_symmetric_and_bounded(mod):
    assert mod.two_sided_sign_p(0, 0) == 1.0
    assert mod.two_sided_sign_p(3, 4) == mod.two_sided_sign_p(4, 3)
    assert 0.0 < mod.two_sided_sign_p(10, 0) < 0.01


def test_the_bootstrap_is_deterministic(mod):
    """A confidence interval that changes between runs cannot be quoted in a
    report the gate re-derives."""
    index, a, b = {}, [], []
    for i in range(40):
        sid = f"s-{i}"
        index[sid] = _scen(sid, ["c-chr-001"])
        a.append(_row(sid, "e" * (10 + i % 5), False))
        b.append(_row(sid, "e" * (4 + i % 3), False))
    inst = {"c-chr-001": LETTER}
    first = mod.compare(a, b, index, inst)["banned_letter_occurrences"]
    second = mod.compare(a, b, index, inst)["banned_letter_occurrences"]
    assert first["median_difference_ci95"] == second["median_difference_ci95"]
    assert first["median_difference_ci95"][1] < 0    # b is genuinely lower
