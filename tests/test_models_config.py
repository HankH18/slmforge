"""P1 contract tests over configs/models.json. Network-free: nothing here calls
a provider. Skips cleanly when the artifact has not been generated yet, so a
fresh clone still has a green suite.

These assertions are written independently of scripts/ping_backends.py's own
regexes so that a bug in the checker cannot make the tests vacuously pass.
"""
import json
import re
from datetime import datetime, timezone

import pytest
from conftest import REPO_ROOT

MODELS_JSON = REPO_ROOT / "configs" / "models.json"
CONTRACT_KEYS = ("frontier", "judge_model", "base", "verified_utc", "sources")


@pytest.fixture(scope="module")
def cfg():
    if not MODELS_JSON.exists():
        pytest.skip("configs/models.json not generated yet "
                    "(run scripts/ping_backends.py)")
    return json.loads(MODELS_JSON.read_text())


def test_top_level_is_an_object_with_every_contract_key(cfg):
    assert isinstance(cfg, dict)
    for key in CONTRACT_KEYS:
        assert key in cfg, f"contract key {key!r} missing from configs/models.json"


def test_frontier_is_exactly_two_provider_model_specs(cfg):
    frontier = cfg["frontier"]
    assert isinstance(frontier, list)
    assert len(frontier) == 2, f"expected 2 frontier models, got {len(frontier)}"
    for spec in frontier:
        assert isinstance(spec, str)
        provider, sep, model = spec.partition(":")
        assert sep == ":", f"{spec!r} is not provider:model"
        assert provider and model.strip(), f"{spec!r} has an empty half"


def test_frontier_models_come_from_two_different_families(cfg):
    families = [spec.split(":", 1)[0] for spec in cfg["frontier"]]
    assert len(set(families)) == 2, (
        f"the ablation needs two distinct provider families, got {families}")


def test_frontier_providers_are_ones_the_frozen_harness_dispatches(cfg):
    """No network: read the provider literals eval.make_generator branches on."""
    import inspect
    import eval as E
    src = inspect.getsource(E.make_generator)
    dispatched = set(re.findall(r'provider == "([a-z]+)"', src))
    assert dispatched, "could not read the provider dispatch out of eval.make_generator"
    for spec in cfg["frontier"]:
        provider = spec.split(":", 1)[0]
        assert provider != "mock", f"{spec!r} is a mock, not a real frontier backend"
        assert provider in dispatched, (
            f"eval.make_generator has no branch for {provider!r}; it dispatches {sorted(dispatched)}")
    base_provider = cfg["base"].split(":", 1)[0]
    assert base_provider in dispatched


def test_judge_is_an_anthropic_id(cfg):
    judge = cfg["judge_model"]
    assert isinstance(judge, str) and judge.strip()
    assert ":" not in judge, "judge_model is a bare Anthropic id, not provider:model"
    assert re.fullmatch(r"claude-[A-Za-z0-9._\-]+", judge), (
        f"{judge!r} is not an Anthropic model id")


def test_judge_is_a_third_family_and_never_scores_itself(cfg):
    """BUILD_PLAN decision 3: OpenAI + Google ablate, Anthropic judges."""
    families = {spec.split(":", 1)[0] for spec in cfg["frontier"]}
    assert "anthropic" not in families, (
        "the judge family appears in the ablation -- a model would score itself")
    frontier_models = {spec.split(":", 1)[1] for spec in cfg["frontier"]}
    assert cfg["judge_model"] not in frontier_models
    assert cfg["judge_model"] not in set(cfg["frontier"])


def test_judge_model_accepts_the_temperature_the_frozen_judge_sends(cfg):
    """eval.AnthropicJudge always sends temperature=0; models that reject it
    turn every judge call into a silent conservative FAIL. Static guard only --
    no API call."""
    assert cfg["judge_model"] not in {
        "claude-opus-5", "claude-sonnet-5", "claude-fable-5", "claude-mythos-5",
        "claude-opus-4-8", "claude-opus-4-7",
    }, ("this Claude generation rejects `temperature`, which eval.AnthropicJudge "
        "sends unconditionally; every scenario would score a conservative FAIL")


def test_base_is_an_hf_repo_pinned_to_a_commit_sha(cfg):
    base = cfg["base"]
    assert isinstance(base, str)
    assert base.startswith("hf:"), f"{base!r} must be an hf: spec"
    rest = base[len("hf:"):]
    repo, sep, revision = rest.partition("@")
    assert sep == "@", f"{base!r} has no pinned revision"
    assert "/" in repo, f"{repo!r} is not <owner>/<name>"
    assert re.fullmatch(r"[0-9a-f]{40}", revision), (
        f"revision {revision!r} is not a 40-hex commit sha -- a branch or tag "
        "would drift under the ablation")


def test_base_is_the_qwen3_1_7b_decided_in_the_build_plan(cfg):
    repo = cfg["base"][len("hf:"):].split("@", 1)[0]
    assert repo == "Qwen/Qwen3-1.7B", (
        f"BUILD_PLAN decision 1 pins Qwen3-1.7B, got {repo!r}")


def test_verified_utc_parses_as_iso8601_utc(cfg):
    stamp = cfg["verified_utc"]
    assert isinstance(stamp, str) and stamp
    parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None, "verified_utc must carry a timezone offset"
    assert parsed.utcoffset() == timezone.utc.utcoffset(None), (
        "verified_utc must be UTC")


def test_sources_are_non_empty_provider_doc_urls(cfg):
    sources = cfg["sources"]
    assert isinstance(sources, list)
    assert sources, "sources must record the doc pages the ids were verified against"
    for url in sources:
        assert isinstance(url, str) and url.startswith("http"), f"{url!r}"


def test_tuned_when_present_is_repo_at_commit(cfg):
    """P5 appends `tuned`; absent before then."""
    if "tuned" not in cfg:
        pytest.skip("tuned is added in P5")
    tuned = cfg["tuned"]
    assert isinstance(tuned, str)
    repo, sep, commit = tuned.partition("@")
    assert sep == "@" and repo and commit, f"{tuned!r} is not <repo>@<commit>"


def test_no_secret_material_leaked_into_the_artifact(cfg):
    blob = json.dumps(cfg)
    for marker in ("sk-", "hf_", "AIza", "sk-ant-", "API_KEY"):
        assert marker not in blob, f"configs/models.json contains {marker!r}"


def test_writer_preserves_a_p5_tuned_key(tmp_path):
    """A P1 rerun after P5 must not clobber the published model pin."""
    import importlib.util  # noqa: PLC0415
    _pb = REPO_ROOT / "archive" / "abstainer" / "scripts" / "ping_backends.py"
    if not _pb.exists():
        raise AssertionError("archived ping_backends.py is missing; it wrote configs/models.json")
    _s = importlib.util.spec_from_file_location("pb", _pb)
    pb = importlib.util.module_from_spec(_s); _s.loader.exec_module(pb)

    target = tmp_path / "models.json"
    target.write_text(json.dumps({
        "frontier": ["openai:old", "google:old"],
        "judge_model": "claude-old",
        "base": "hf:Qwen/Qwen3-1.7B@" + "0" * 40,
        "verified_utc": "2020-01-01T00:00:00+00:00",
        "sources": ["https://old.example"],
        "tuned": "HankH18/grounded-abstainer@" + "a" * 40,
    }))

    payload = pb.write_models_json(
        target,
        ["openai:new", "google:new"],
        "claude-new",
        "hf:Qwen/Qwen3-1.7B@" + "1" * 40,
        ["https://new.example"],
    )

    assert payload["tuned"] == "HankH18/grounded-abstainer@" + "a" * 40
    assert payload["frontier"] == ["openai:new", "google:new"]
    assert json.loads(target.read_text())["tuned"] == payload["tuned"]
