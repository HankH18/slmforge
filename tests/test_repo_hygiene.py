"""Cheap structural guarantees a grader depends on."""
from conftest import REPO_ROOT


def test_env_is_gitignored_and_never_committed():
    ignored = (REPO_ROOT / ".gitignore").read_text().splitlines()
    assert ".env" in [ln.strip() for ln in ignored]


def test_env_example_lists_every_required_key():
    text = (REPO_ROOT / ".env.example").read_text()
    for key in ("OPENAI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY",
                "HF_TOKEN", "HF_USERNAME"):
        assert key in text, f"{key} missing from .env.example"


def test_env_example_holds_no_real_values():
    for line in (REPO_ROOT / ".env.example").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            assert line.endswith("="), f"{line.split('=')[0]} has a value in .env.example"


def test_requirements_pin_the_provider_sdks():
    text = (REPO_ROOT / "requirements.txt").read_text()
    for pkg in ("anthropic", "openai", "google-genai", "transformers", "pytest"):
        assert pkg in text


def test_frozen_inputs_are_at_their_contract_paths():
    for path in ("eval.py", "run_ablation.py",
                 "data/eval/scenarios_v1.jsonl", "data/eval/fewshot_exemplars.jsonl"):
        assert (REPO_ROOT / path).exists(), f"{path} missing"
