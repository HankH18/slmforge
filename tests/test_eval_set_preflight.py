"""The staff held-out set is a file this project has never seen.

Every assertion here binds a way an unfamiliar scenario file could take a run
down. The one that motivated this module: `pressure`, `split` and `k` were read
only inside `aggregate()`, which runs after every generation and every judge
call, so a file missing one of them burned the whole run and wrote nothing --
the failure is unrecoverable and arrives at the end, which is the worst place
for it.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from harness import HarnessError, build_constraint_lookup, validate_eval_set  # noqa: E402
from scenarios.render import render_user_message  # noqa: E402

SCRIPT = REPO_ROOT / "scripts" / "preflight_evalset.py"
CORPUS = REPO_ROOT / "data" / "scenarios" / "corpus.jsonl"
INSTANCES = REPO_ROOT / "data" / "scenarios" / "constraint_instances.jsonl"


@pytest.fixture(scope="module")
def lookup():
    return build_constraint_lookup(
        [json.loads(l) for l in INSTANCES.read_text().splitlines() if l.strip()])


@pytest.fixture(scope="module")
def corpus():
    return [json.loads(l) for l in CORPUS.read_text().splitlines() if l.strip()]


def test_the_shipped_corpus_validates(corpus, lookup):
    validate_eval_set(corpus, lookup, render_user_message)


@pytest.mark.parametrize("field", ["pressure", "split", "k"])
def test_a_field_only_aggregate_reads_is_caught_up_front(corpus, lookup, field):
    """THE ONE THAT MATTERS. These three are read only at the very end. Catching
    them here is the difference between a free refusal and a destroyed run."""
    rows = [{k: v for k, v in s.items() if k != field} for s in corpus[:4]]
    with pytest.raises(HarnessError) as exc:
        validate_eval_set(rows, lookup, render_user_message)
    assert field in str(exc.value)
    assert "paid for" in str(exc.value)


def test_every_problem_is_reported_not_just_the_first(corpus, lookup):
    """Handed a file with several mistakes, naming one buys a round trip per
    mistake."""
    rows = [dict(s) for s in corpus[:3]]
    del rows[0]["pressure"]
    del rows[1]["split"]
    rows[2] = {k: v for k, v in rows[2].items() if k != "k"}
    msg = str(pytest.raises(HarnessError, validate_eval_set, rows, lookup).value)
    assert "pressure" in msg and "split" in msg and "'k'" in msg


def test_an_unknown_constraint_id_is_caught(corpus, lookup):
    rows = [dict(corpus[0], constraints=["c-lex-999"])]
    with pytest.raises(HarnessError, match="unknown constraint id"):
        validate_eval_set(rows, lookup, render_user_message)


def test_an_inline_constraint_object_gets_a_diagnostic_not_a_typeerror(corpus, lookup):
    """A self-contained eval set is a plausible staff format, and it used to die
    with a bare `TypeError: unhashable type: 'dict'`."""
    inline = {"id": "c-lex-001", "type": "banned_words", "params": {"words": ["x"]}}
    rows = [dict(corpus[0], constraints=[inline])]
    with pytest.raises(HarnessError, match="not an id string"):
        validate_eval_set(rows, lookup, render_user_message)


def test_an_empty_constraint_list_is_refused(corpus, lookup):
    """Zero constraints passes every checker vacuously -- a free 100%."""
    rows = [dict(corpus[0], constraints=[])]
    with pytest.raises(HarnessError, match="vacuous"):
        validate_eval_set(rows, lookup, render_user_message)


def test_a_duplicate_scenario_id_is_refused(corpus, lookup):
    rows = [dict(corpus[0]), dict(corpus[0])]
    with pytest.raises(HarnessError, match="duplicate scenario_id"):
        validate_eval_set(rows, lookup, render_user_message)


def test_a_missing_task_is_refused(corpus, lookup):
    rows = [{k: v for k, v in corpus[0].items() if k != "task"}]
    with pytest.raises(HarnessError, match="'task'"):
        validate_eval_set(rows, lookup, render_user_message)


def test_an_unknown_pressure_value_is_refused(corpus, lookup):
    rows = [dict(corpus[0], pressure="medium")]
    with pytest.raises(HarnessError, match="pressure"):
        validate_eval_set(rows, lookup, render_user_message)


def test_an_empty_eval_set_is_refused(lookup):
    with pytest.raises(HarnessError, match="empty"):
        validate_eval_set([], lookup, render_user_message)


# ---------------------------------------------------------------- the CLI
def _run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *args],
                          cwd=REPO_ROOT, capture_output=True, text=True)


def test_preflight_passes_the_shipped_corpus():
    r = _run(str(CORPUS))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PREFLIGHT PASS" in r.stdout


def test_preflight_rejects_a_foreign_constraint_id(tmp_path, corpus):
    """The trap a staff-supplied constraint library walks into: ids are pinned
    to `^c-(?:lex|chr|str|fmt)-\\d+$`, and a foreign one raises at SCORING time,
    which is after the money. Preflight has to find it for free."""
    cons = [json.loads(l) for l in INSTANCES.read_text().splitlines() if l.strip()]
    rename = {c["id"]: f"staff-{i:03d}" for i, c in enumerate(cons)}
    (tmp_path / "cons.jsonl").write_text(
        "".join(json.dumps(dict(c, id=rename[c["id"]])) + "\n" for c in cons))
    (tmp_path / "eval.jsonl").write_text("".join(
        json.dumps(dict(s, constraints=[rename[c] for c in s["constraints"]])) + "\n"
        for s in corpus[:5]))
    r = _run(str(tmp_path / "eval.jsonl"), str(tmp_path / "cons.jsonl"))
    assert r.returncode == 1
    assert "does not match" in r.stderr
    assert "PREFLIGHT FAIL" in r.stderr


def test_preflight_reports_what_the_set_contains():
    """A grader deciding whether to trust a number needs the shape of the set
    that produced it, not just a pass/fail."""
    r = _run(str(CORPUS))
    for expected in ("by split", "by pressure", "by k", "adversarial subset"):
        assert expected in r.stdout


def test_preflight_on_a_missing_file_is_an_error():
    r = _run("data/scenarios/does_not_exist.jsonl")
    assert r.returncode == 1
    assert "does not exist" in r.stderr
