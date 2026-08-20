"""Prompt construction per strategy -- the ablation's independent variable."""
import eval as E
import pytest

from conftest import run_mock


def test_zero_shot_is_system_plus_one_user_turn(scenarios):
    p = E.build_prompt("zero_shot", scenarios[0], None)
    assert p["system"] == E.SYSTEM_CORE
    assert len(p["turns"]) == 1
    assert p["turns"][0]["role"] == "user"
    assert scenarios[0]["passage"] in p["turns"][0]["content"]


def test_structured_appends_the_addendum_only(scenarios):
    p = E.build_prompt("structured", scenarios[0], None)
    assert p["system"] == E.SYSTEM_CORE + E.STRUCTURED_ADDENDUM
    assert len(p["turns"]) == 1


def test_few_shot_interleaves_exemplars_then_the_query(scenarios, exemplars):
    p = E.build_prompt("few_shot", scenarios[0], exemplars)
    assert p["system"] == E.SYSTEM_CORE
    assert len(p["turns"]) == 2 * len(exemplars) + 1
    assert [t["role"] for t in p["turns"][:4]] == ["user", "assistant", "user", "assistant"]
    assert p["turns"][-1]["role"] == "user"
    assert scenarios[0]["question"] in p["turns"][-1]["content"]


def test_few_shot_without_exemplars_is_an_error(scenarios):
    with pytest.raises(ValueError, match="requires --exemplars"):
        E.build_prompt("few_shot", scenarios[0], None)


def test_unknown_strategy_rejected(scenarios):
    with pytest.raises(ValueError, match="unknown strategy"):
        E.build_prompt("chain_of_thought", scenarios[0], None)


def test_all_three_strategies_run_end_to_end(scenarios, exemplars):
    """The doc requires >=3 strategies per model; each must score the full set."""
    assert set(E.STRATEGIES) == {"zero_shot", "few_shot", "structured"}
    for strategy in E.STRATEGIES:
        _, m = run_mock(scenarios, "oracle", strategy=strategy, exemplars=exemplars)
        assert m["n"] == 32
        assert m["spec_adherence"] == 1.0


def test_training_condition_matches_zero_shot_eval_condition(scenarios):
    """Contract: training examples use SYSTEM_CORE verbatim + user_msg(). If this
    drifts, the tuned model is evaluated under a prompt it never saw."""
    s = scenarios[0]
    p = E.build_prompt("zero_shot", s, None)
    assert p["system"] == E.SYSTEM_CORE
    assert p["turns"][0]["content"] == E.user_msg(s["passage"], s["question"])
