"""The five test doubles pin every branch of the rubric to a known number on the
real 32-scenario eval set. These are the regression gate: if a refactor moves any
of these, scoring changed."""
import eval as E
import pytest

from conftest import run_mock


def test_oracle_scores_perfect(scenarios):
    _, m = run_mock(scenarios, "oracle", judge="mock:yes")
    assert m["n"] == 32
    assert m["spec_adherence"] == pytest.approx(1.000, abs=5e-4)
    assert m["robustness"] == pytest.approx(1.000, abs=5e-4)
    assert m["false_answer_rate"] == 0.0
    assert m["false_refusal_rate"] == 0.0
    assert m["failure_breakdown"] == {}


def test_always_refuse_is_the_abstention_floor(scenarios):
    """Refusing everything scores 19/32 -- the score a model gets for free by
    never answering. Any tuned model must beat this, not just beat zero."""
    _, m = run_mock(scenarios, "always_refuse")
    assert m["spec_adherence"] == pytest.approx(0.594, abs=5e-4)
    assert m["false_refusal_rate"] == 1.0
    assert m["false_answer_rate"] == 0.0
    assert m["failure_breakdown"] == {"false_refusal": 13}


def test_nonverbatim_citations_score_zero(scenarios):
    _, m = run_mock(scenarios, "nonverbatim")
    assert m["spec_adherence"] == 0.0
    assert m["false_answer_rate"] == 1.0
    assert m["failure_breakdown"] == {"nonverbatim_citation": 13, "false_answer": 19}


def test_format_breaker_is_malformed_everywhere(scenarios):
    _, m = run_mock(scenarios, "format_breaker")
    assert m["spec_adherence"] == 0.0
    assert m["failure_breakdown"] == {"malformed": 32}


def test_bad_entail_isolates_the_judge(scenarios):
    """Verbatim spans that do not entail the answer: passes every programmatic
    check, caught only by the judge."""
    _, m = run_mock(scenarios, "bad_entail", judge="mock:no")
    assert m["failure_breakdown"] == {"entailment_fail": 13}
    assert m["false_answer_rate"] == 0.0
    assert m["false_refusal_rate"] == 0.0


def test_confusion_matrix_covers_every_scenario(scenarios):
    for behavior in ("oracle", "always_refuse", "nonverbatim", "format_breaker"):
        _, m = run_mock(scenarios, behavior)
        assert sum(m["confusion"].values()) == 32, behavior


def test_transcripts_carry_judge_reasoning(scenarios):
    """The project doc requires per-example judge score + reasoning, not just
    aggregates."""
    recs, _ = run_mock(scenarios, "oracle", judge="mock:yes")
    judged = [r for r in recs if r["gold_label"] == "answerable"]
    assert len(judged) == 13
    for r in judged:
        assert r["judge_verdict"] is True
        assert r["judge_reasoning"]
    for r in recs:
        assert {"scenario_id", "raw_output", "pred_class", "passed", "failure"} <= set(r)


def test_unknown_mock_behavior_raises():
    with pytest.raises(ValueError, match="unknown mock behavior"):
        E.MockGenerator("does_not_exist")({"gold_label": "answerable"}, {})
