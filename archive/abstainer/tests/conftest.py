"""Point the archived abstainer tests at the archived abstainer harness.

The live root eval.py now scores the Forbidden-Constraint Holder behavior. These
tests pin the abstainer rubric (hash c0adb876c3e6c4c9) that produced the P2
ablation numbers, so they must import archive/abstainer/eval.py, not the root one.
Run them with:  python -m pytest archive/abstainer/tests -q
"""
import importlib.util
import sys
from pathlib import Path

ABSTAINER_ROOT = Path(__file__).resolve().parent.parent   # archive/abstainer/
REPO_ROOT = ABSTAINER_ROOT.parent.parent

# Load archive/abstainer/eval.py under the name `eval` BEFORE any test imports it.
if "eval" not in sys.modules:
    spec = importlib.util.spec_from_file_location("eval", ABSTAINER_ROOT / "eval.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval"] = module
    spec.loader.exec_module(module)

sys.path.insert(0, str(ABSTAINER_ROOT))
sys.path.insert(0, str(REPO_ROOT))

import eval as E  # noqa: E402

SCENARIOS_PATH = ABSTAINER_ROOT / "scenarios_v1.jsonl"
EXEMPLARS_PATH = ABSTAINER_ROOT / "fewshot_exemplars.jsonl"

import pytest  # noqa: E402


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
    return E.run_eval(f"mock:{behavior}", scenarios, strategy, exemplars,
                      E.make_judge(judge, "unused"), workers=1)
