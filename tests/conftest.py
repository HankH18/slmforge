"""Shared fixtures. Every test here is network-free: mock generators, mock judges."""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import eval as E  # noqa: E402  (needs the path insert above)

SCENARIOS_PATH = REPO_ROOT / "data" / "eval" / "scenarios_v1.jsonl"
EXEMPLARS_PATH = REPO_ROOT / "data" / "eval" / "fewshot_exemplars.jsonl"


@pytest.fixture(scope="session")
def repo_root():
    return REPO_ROOT


@pytest.fixture(scope="session")
def scenarios():
    return E.load_jsonl(str(SCENARIOS_PATH))


@pytest.fixture(scope="session")
def exemplars():
    return E.load_jsonl(str(EXEMPLARS_PATH))


def run_mock(scenarios, behavior, judge="mock:yes", strategy="zero_shot", exemplars=None):
    """One mock model x one strategy over the full eval set -> (records, metrics)."""
    return E.run_eval(f"mock:{behavior}", scenarios, strategy, exemplars,
                      E.make_judge(judge, "unused"), workers=1)
