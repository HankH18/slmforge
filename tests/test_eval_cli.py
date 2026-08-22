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


def test_model_spec_that_is_neither_provider_qualified_nor_a_repo_id_raises():
    """Renamed only -- the assertion below is unchanged and still binds.

    The old name ("without a colon raises") stated a general rule that the spec
    contradicts: <hf-repo-id> has no colon and must now resolve. The INPUT here
    is unaffected: 'gpt-5.6-sol' has no slash either, so it is not a Hub repo id
    under any reading, and it must still fail locally and immediately."""
    with pytest.raises(ValueError, match="provider:model"):
        E.make_generator("gpt-5.6-sol")


# ------------------------------------------- the spec's documented model form
# project_spec.pdf p3, Verification Requirements, verbatim:
#   "eval.py --model <hf-repo-id> --eval-set <path> regenerates your full
#    results table from nothing."
# The bare repo id is therefore a requirement, not a convenience. These bind
# both halves of it: the bare form resolves, and a typo still dies here.

class _RecordingHF:
    """Stands in for HFGenerator so the parse can be asserted without loading
    multi-GB weights. Only make_generator's routing is under test."""

    def __init__(self, repo, revision):
        self.repo, self.revision = repo, revision


@pytest.fixture
def hf_spy(monkeypatch):
    monkeypatch.setattr(E, "HFGenerator", _RecordingHF)
    return _RecordingHF


def test_bare_hf_repo_id_resolves_to_the_hf_provider(hf_spy):
    g = E.make_generator("HankH18/qwen3-1.7b-forbidden-constraint-holder")
    assert isinstance(g, _RecordingHF)
    assert g.repo == "HankH18/qwen3-1.7b-forbidden-constraint-holder"
    assert g.revision is None


def test_bare_hf_repo_id_carries_its_pinned_revision(hf_spy):
    g = E.make_generator("Qwen/Qwen3-1.7B@70d244cc0e1c4b6b5b4b4b")
    assert (g.repo, g.revision) == ("Qwen/Qwen3-1.7B", "70d244cc0e1c4b6b5b4b4b")


def test_bare_and_prefixed_forms_of_the_same_repo_agree(hf_spy):
    bare = E.make_generator("Qwen/Qwen3-1.7B@abc123")
    pref = E.make_generator("hf:Qwen/Qwen3-1.7B@abc123")
    assert (bare.repo, bare.revision) == (pref.repo, pref.revision)


@pytest.mark.parametrize("spec,cls,rest", [
    ("hf:Qwen/Qwen3-1.7B", "HFGenerator", "Qwen/Qwen3-1.7B"),
    ("hf:runs/efficiency/n106/merged", "HFGenerator", "runs/efficiency/n106/merged"),
    ("mock:compliant", "MockGenerator", "compliant"),
    ("anthropic:claude-sonnet-4-5", "AnthropicGenerator", "claude-sonnet-4-5"),
    ("openai:gpt-5.6-sol", "OpenAIGenerator", "gpt-5.6-sol"),
    ("google:gemini-3-pro", "GoogleGenerator", "gemini-3-pro"),
])
def test_every_preexisting_spec_form_still_routes_unchanged(monkeypatch, spec, cls, rest):
    """Regression fence around the bare-repo-id branch: adding it must not have
    moved any provider-qualified spec. Every constructor is stubbed, so this
    asserts routing only and needs no API key and no weights."""
    seen = {}

    def recorder(name):
        def _make(*args):
            seen["cls"], seen["args"] = name, args
            return object()
        return _make

    for name in ("HFGenerator", "MockGenerator", "AnthropicGenerator",
                 "OpenAIGenerator", "GoogleGenerator"):
        monkeypatch.setattr(E, name, recorder(name))
    E.make_generator(spec)
    assert seen["cls"] == cls
    # hf splits repo@revision; every other provider passes its rest through whole
    assert seen["args"][0] == rest


def test_a_local_weights_directory_resolves_bare_and_prefixed(hf_spy, tmp_path):
    """runs/efficiency/n106/merged is a real local merged-weights dir in this
    repo and is referenced as hf:<path>. Both spellings must load from disk."""
    d = tmp_path / "merged"
    d.mkdir()
    assert E.make_generator(f"hf:{d}").repo == str(d)
    assert E.make_generator(str(d)).repo == str(d)


@pytest.mark.parametrize("spec", [
    "openai;gpt-5",          # semicolon typo: no colon AND no slash
    "HankH18",               # owner only, no repo half
    "anthropc:x",            # misspelled provider -- must stay an unknown provider
    "C:\\models\\qwen",  # Windows path: the colon rules, provider 'C' is unknown
    "a/b/c",                 # two slashes is not a Hub repo id
    "owner/",                # empty repo half
    "Qwen/Qwen3-1.7B@",      # empty revision
])
def test_malformed_specs_still_fail_locally_and_loudly(hf_spy, spec):
    """A typo must die here -- free, instant, obvious -- and never be routed to
    the Hub, where it would surface as a slow 404 or an auth prompt."""
    with pytest.raises(ValueError):
        E.make_generator(spec)


def test_help_documents_the_bare_repo_id_form():
    r = run_cli("--help")
    assert r.returncode == 0, r.stderr
    assert "owner/name" in r.stdout, r.stdout


def test_bare_repo_id_is_canonicalized_to_the_hf_prefix():
    """The bare form must not fork the artifact namespace. sanitize() turns the
    spec into the output filename and the "model" field of every transcript row,
    and run_ablation / check_mvp both do spec.split(":") on those -- so a bare
    spec is rewritten to hf:<repo> at the boundary rather than carried through."""
    assert E.normalize_model_spec("Qwen/Qwen3-1.7B") == "hf:Qwen/Qwen3-1.7B"
    assert (E.normalize_model_spec("HankH18/tuned@abc123")
            == "hf:HankH18/tuned@abc123")
    assert E.sanitize(E.normalize_model_spec("Qwen/Qwen3-1.7B")) == \
        E.sanitize("hf:Qwen/Qwen3-1.7B")


@pytest.mark.parametrize("spec", [
    "hf:Qwen/Qwen3-1.7B", "hf:runs/efficiency/n106/merged", "mock:compliant",
    "anthropic:claude-sonnet-4-5", "openai:gpt-5.6-sol", "google:gemini-3-pro",
    "bedrock:whatever",   # unknown providers stay untouched here, caught on dispatch
])
def test_provider_qualified_specs_are_returned_byte_identical(spec):
    assert E.normalize_model_spec(spec) == spec


def test_cli_rejects_a_malformed_model_spec_before_any_judge_call(tmp_path):
    """The typo must cost nothing: no judge, no weights, no output directory."""
    out = tmp_path / "run"
    r = run_cli("--model", "openai;gpt-5", "--eval-set", SMOKE,
                "--judge", "mock:4", "--out", str(out))
    assert r.returncode == 2, (r.returncode, r.stdout, r.stderr)
    assert "openai;gpt-5" in r.stderr
    assert not out.exists(), "a rejected spec must not create a run directory"
