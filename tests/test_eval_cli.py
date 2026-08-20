"""The graded one-command entry point.

The project spec's Verification Requirements say: "`eval.py --model <hf-repo-id>
--eval-set <path>` regenerates your full results table from nothing. If it takes
more than one command, it isn't verified." These tests bind that, and bind the
strategy prompts that decide what the ablation actually measures.

Network-free throughout: mock generators, mock judge.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

import eval as E
from harness import build_constraint_lookup, validate_transcript_row
from scenarios.paths import load_instances

REPO_ROOT = Path(__file__).resolve().parent.parent
SMOKE = "data/scenarios/smoke.jsonl"


@pytest.fixture(scope="module")
def lookup():
    return build_constraint_lookup(load_instances().values())


@pytest.fixture(scope="module")
def scenario():
    return E.load_jsonl(str(REPO_ROOT / SMOKE))[0]


def run_cli(*args, cwd=None):
    return subprocess.run([sys.executable, str(REPO_ROOT / "eval.py"), *args],
                          cwd=str(cwd or REPO_ROOT), capture_output=True, text=True)


# ------------------------------------------------- the training/eval contract
def test_zero_shot_carries_no_system_message(scenario, lookup):
    """THE load-bearing test. The pinned training example schema is
    [user, assistant] with no system turn. If zero_shot grows a system message
    and training does not, the tuned model is scored under a prompt it never saw
    and every MVP number is meaningless."""
    p = E.build_prompt("zero_shot", scenario, lookup, None)
    assert p["system"] is None
    assert len(p["turns"]) == 1
    assert p["turns"][0]["role"] == "user"


def test_zero_shot_user_message_is_exactly_the_shared_renderer(scenario, lookup):
    from scenarios.render import render_user_message
    p = E.build_prompt("zero_shot", scenario, lookup, None)
    assert p["turns"][0]["content"] == render_user_message(scenario, lookup)


def test_structured_adds_a_system_message_and_zero_shot_does_not(scenario, lookup):
    """structured is an ablation strategy, not the training condition, so it may
    differ -- but only structured."""
    assert E.build_prompt("structured", scenario, lookup, None)["system"] == E.STRUCTURED_SYSTEM
    assert E.build_prompt("zero_shot", scenario, lookup, None)["system"] is None
    assert E.build_prompt("few_shot", scenario, lookup,
                          [{"rendered_prompt": "p", "target_output": "o"}])["system"] is None


def test_few_shot_interleaves_exemplars_then_the_query(scenario, lookup):
    ex = [{"rendered_prompt": "P1", "target_output": "O1"},
          {"rendered_prompt": "P2", "target_output": "O2"}]
    p = E.build_prompt("few_shot", scenario, lookup, ex)
    assert [t["role"] for t in p["turns"]] == ["user", "assistant", "user", "assistant", "user"]
    assert [t["content"] for t in p["turns"][:4]] == ["P1", "O1", "P2", "O2"]


def test_few_shot_without_exemplars_is_an_error(scenario, lookup):
    with pytest.raises(ValueError, match="requires --exemplars"):
        E.build_prompt("few_shot", scenario, lookup, None)


def test_unknown_strategy_rejected(scenario, lookup):
    with pytest.raises(ValueError, match="unknown strategy"):
        E.build_prompt("chain_of_thought", scenario, lookup, None)


def test_all_three_strategies_exist():
    """The spec requires >=3 prompting strategies per model."""
    assert set(E.STRATEGIES) == {"zero_shot", "few_shot", "structured"}


# ---------------------------------------------------------------- the CLI
def test_one_command_writes_transcripts_and_results(tmp_path):
    r = run_cli("--model", "mock:compliant", "--eval-set", SMOKE,
                "--judge", "mock:4", "--out", str(tmp_path / "run"))
    assert r.returncode == 0, r.stderr
    out = tmp_path / "run"
    transcripts = out / "mock_compliant__zero_shot.transcripts.jsonl"
    results = out / "mock_compliant__zero_shot.results.json"
    assert transcripts.exists() and results.exists()

    rows = [json.loads(ln) for ln in transcripts.read_text().splitlines() if ln.strip()]
    assert len(rows) == 5
    for row in rows:
        validate_transcript_row(row)          # the pinned schema, not a lookalike

    data = json.loads(results.read_text())
    assert data["metrics"]["spec_adherence"] == 1.0
    assert data["model"] == "mock:compliant" and data["strategy"] == "zero_shot"
    assert data["judge_rubric_hash"] and data["timestamp_utc"] and data["eval_set"]


def test_transcripts_carry_per_example_judge_output(tmp_path):
    """The spec requires full per-example judge score + reasoning as JSONL, not
    just the aggregate table."""
    r = run_cli("--model", "mock:compliant", "--eval-set", SMOKE,
                "--judge", "mock:4", "--out", str(tmp_path / "run"))
    assert r.returncode == 0, r.stderr
    rows = [json.loads(ln) for ln in
            (tmp_path / "run" / "mock_compliant__zero_shot.transcripts.jsonl")
            .read_text().splitlines() if ln.strip()]
    for row in rows:
        assert row["quality_score"] == 4
        assert row["quality_reasoning"]
        assert row["checker_results"]


def test_compare_with_emits_both_models_and_a_delta(tmp_path):
    """The base-vs-tuned mechanism the MVP is graded on."""
    r = run_cli("--model", "mock:compliant", "--compare-with", "mock:violator",
                "--eval-set", SMOKE, "--judge", "mock:4", "--out", str(tmp_path / "run"))
    assert r.returncode == 0, r.stderr
    assert "delta (mock:compliant minus mock:violator)" in r.stdout
    assert "spec_adherence" in r.stdout and "pp" in r.stdout
    for stem in ("mock_compliant__zero_shot", "mock_violator__zero_shot"):
        assert (tmp_path / "run" / f"{stem}.transcripts.jsonl").exists()
        assert (tmp_path / "run" / f"{stem}.results.json").exists()


def test_a_missing_eval_set_fails_loudly(tmp_path):
    """Silently scoring a smaller set would quietly change every rate."""
    r = run_cli("--model", "mock:compliant", "--eval-set", "data/scenarios/nope.jsonl",
                "--judge", "mock:4", "--out", str(tmp_path / "run"))
    assert r.returncode != 0
    assert "does not exist" in (r.stderr + r.stdout)


def test_limit_reduces_the_scored_set(tmp_path):
    r = run_cli("--model", "mock:compliant", "--eval-set", SMOKE, "--limit", "2",
                "--judge", "mock:4", "--out", str(tmp_path / "run"))
    assert r.returncode == 0, r.stderr
    data = json.loads((tmp_path / "run" / "mock_compliant__zero_shot.results.json").read_text())
    assert data["metrics"]["n"] == 2


# ---------------------------------------------------------------- mock doubles
@pytest.mark.parametrize("behavior,judge,expected", [
    ("compliant", "mock:4", 1.0),   # obeys everything, decent answer
    ("violator", "mock:4", 0.0),    # breaks constraints outright
    ("empty", "mock:1", 0.0),       # says nothing: checkers vacuously pass, judge kills it
    ("offtask", "mock:1", 0.0),     # obeys everything, ignores the task
])
def test_mock_behaviors_score_as_expected(tmp_path, behavior, judge, expected):
    r = run_cli("--model", f"mock:{behavior}", "--eval-set", SMOKE,
                "--judge", judge, "--out", str(tmp_path / behavior))
    assert r.returncode == 0, r.stderr
    data = json.loads((tmp_path / behavior /
                       f"mock_{behavior}__zero_shot.results.json").read_text())
    assert data["metrics"]["spec_adherence"] == expected


def test_the_judge_cannot_rescue_a_failed_checker(tmp_path):
    """Spec R4: the judge scores task-quality only and never overrides a checker.
    A perfect 5 on output that violates constraints must still fail."""
    r = run_cli("--model", "mock:violator", "--eval-set", SMOKE,
                "--judge", "mock:5", "--out", str(tmp_path / "run"))
    assert r.returncode == 0, r.stderr
    data = json.loads((tmp_path / "run" / "mock_violator__zero_shot.results.json").read_text())
    assert data["metrics"]["spec_adherence"] == 0.0


def test_empty_output_is_caught_by_quality_not_by_the_checkers(tmp_path):
    """Ruling A7: empty text vacuously passes every checker. That is deliberate --
    the quality floor is what rejects it. If the judge scored it >=3 it would
    pass, which is exactly why the judge exists."""
    r = run_cli("--model", "mock:empty", "--eval-set", SMOKE,
                "--judge", "mock:5", "--out", str(tmp_path / "run"))
    assert r.returncode == 0, r.stderr
    rows = [json.loads(ln) for ln in
            (tmp_path / "run" / "mock_empty__zero_shot.transcripts.jsonl")
            .read_text().splitlines() if ln.strip()]
    for row in rows:
        assert all(c["pass"] for c in row["checker_results"]), "checkers pass on empty"
        assert row["spec_pass"] is True, "with a 5 from the judge, empty passes"


def test_unknown_mock_behavior_raises():
    with pytest.raises(ValueError, match="unknown mock behavior"):
        E.make_generator("mock:does_not_exist")


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="unknown provider"):
        E.make_generator("bedrock:some-model")


def test_model_spec_without_a_colon_raises():
    with pytest.raises(ValueError, match="provider:model"):
        E.make_generator("gpt-5.6-sol")
