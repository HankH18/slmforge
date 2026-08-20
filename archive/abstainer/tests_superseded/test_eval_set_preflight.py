"""Preflight validator for held-out eval sets (scripts/check_eval_set.py).

The graded staff run points the frozen harness at a scenario file nobody here
has seen. `eval.py` validates nothing, so a plausible schema variant either
crashes mid-run or -- worse -- is scored silently and wrongly. These tests pin
both halves:

  * the validator passes on the real data/eval/scenarios_v1.jsonl, and
  * for each defect class it fails on a synthetic file, AND (for the silent
    class) the harness really does misscore that same file, so the test is
    bound to the defect rather than to the validator's own opinion.
"""
import importlib.util
import json

import eval as E
import pytest

from conftest import REPO_ROOT, ABSTAINER_ROOT

CHECKER_PATH = ABSTAINER_ROOT / "scripts" / "check_eval_set.py"

_spec = importlib.util.spec_from_file_location("check_eval_set", CHECKER_PATH)
check_eval_set = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_eval_set)


# ------------------------------------------------------------------ helpers
def base_rows():
    """A minimal, valid two-row set: one answerable, one unanswerable."""
    passage_a = ("Brightwater Mill processed 12,400 tonnes of ore in the third quarter. "
                 "The site ran two shifts through September.")
    passage_u = ("Brightwater Mill ran two shifts through September. "
                 "Quarterly throughput was not disclosed.")
    return [
        {"id": "A1", "type": "clean_answerable", "adversarial": False,
         "gold_label": "answerable",
         "question": "How much ore did Brightwater Mill process in the third quarter?",
         "passage": passage_a,
         "gold_answer": "12,400 tonnes.",
         "gold_spans": ["processed 12,400 tonnes of ore in the third quarter"]},
        {"id": "U1", "type": "support_deleted", "adversarial": True,
         "gold_label": "unanswerable",
         "question": "How much ore did Brightwater Mill process in the third quarter?",
         "passage": passage_u,
         "gold_answer": None, "gold_spans": None},
    ]


def write_rows(tmp_path, rows, name="heldout.jsonl"):
    path = tmp_path / name
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def run_checker(path, *extra):
    return check_eval_set.main([str(path), *extra])


def checks_reported(path):
    """The set of ERROR check-names the validator raises for a file."""
    rep = check_eval_set.Report()
    rows = check_eval_set.load_rows(path, rep)
    check_eval_set.check_rows(rows, rep)
    return {name for name, _ in rep.errors}


# ------------------------------------------------------------------ the real set
def test_real_eval_set_passes(capsys):
    """The committed 32-scenario set must be preflight-clean, or the validator
    is calibrated wrong and every FAIL below is meaningless."""
    path = REPO_ROOT / "data" / "eval" / "scenarios_v1.jsonl"
    assert run_checker(path) == 0
    out = capsys.readouterr().out
    assert "PASS" in out
    assert "scenarios          32" in out


def test_minimal_synthetic_set_passes(tmp_path):
    assert run_checker(write_rows(tmp_path, base_rows())) == 0


# ------------------------------------------------------------------ the silent misscore
@pytest.mark.parametrize("bad_label", ["not_answerable", "Unanswerable", "no_answer", False])
def test_gold_label_outside_the_two_values_is_caught(tmp_path, bad_label):
    rows = base_rows()
    rows[1]["gold_label"] = bad_label
    path = write_rows(tmp_path, rows)
    assert run_checker(path) == 1
    assert "bad_gold_label" in checks_reported(path)


def test_the_harness_really_does_misscore_a_relabelled_set():
    """Bind the check to the defect. `score_example` branches on
    `gold_label == "unanswerable"` and has no else-validation, so any other
    spelling falls through to the answerable branch: a perfect abstainer is
    scored 0% with failure=false_refusal, and false_answer_rate goes to None
    because the unanswerable bucket is empty. No exception, no warning."""
    good = [r for r in base_rows() if r["gold_label"] == "unanswerable"]
    bad = [dict(r, gold_label="not_answerable") for r in good]
    judge = E.make_judge("mock:yes", "unused")

    _, m_good = E.run_eval("mock:always_refuse", good, "zero_shot", None, judge, workers=1)
    _, m_bad = E.run_eval("mock:always_refuse", bad, "zero_shot", None, judge, workers=1)

    assert m_good["spec_adherence"] == 1.0
    assert m_good["false_answer_rate"] == 0.0
    # same model, same outputs, one relabelled word -> inverted result
    assert m_bad["spec_adherence"] == 0.0
    assert m_bad["failure_breakdown"] == {"false_refusal": len(bad)}
    assert m_bad["false_answer_rate"] is None


# ------------------------------------------------------------------ crash classes
@pytest.mark.parametrize("key", ["id", "gold_label", "question", "passage"])
def test_missing_hard_required_key_is_caught(tmp_path, key):
    rows = base_rows()
    rows[0].pop(key)
    path = write_rows(tmp_path, rows)
    assert run_checker(path) == 1
    assert "missing_required_key" in checks_reported(path)


@pytest.mark.parametrize("key", ["id", "gold_label", "question", "passage"])
def test_missing_hard_required_key_really_breaks_the_harness(key):
    row = base_rows()[1]
    row.pop(key)
    with pytest.raises(KeyError, match=key):
        # exactly what run_eval.one() does per scenario (eval.py:414-419)
        prompt = E.build_prompt("zero_shot", row, None)
        E.score_example(row, E.REFUSAL_STRING, E.make_judge("mock:yes", "unused"))
        assert prompt  # unreachable for question/passage, which die above


def test_missing_gold_spans_on_answerable_row_is_caught(tmp_path):
    rows = base_rows()
    rows[0].pop("gold_spans")
    path = write_rows(tmp_path, rows)
    assert run_checker(path) == 1
    assert "missing_gold_field" in checks_reported(path)


def test_empty_gold_spans_on_answerable_row_is_caught(tmp_path):
    rows = base_rows()
    rows[0]["gold_spans"] = []
    path = write_rows(tmp_path, rows)
    assert run_checker(path) == 1
    assert "empty_gold_spans" in checks_reported(path)


def test_missing_gold_spans_really_breaks_the_oracle_double():
    row = base_rows()[0]
    row.pop("gold_spans")
    with pytest.raises(KeyError):
        E.MockGenerator("oracle")(row, {})
    row["gold_spans"] = []
    with pytest.raises(IndexError):
        E.MockGenerator("oracle")(row, {})


# ------------------------------------------------------------------ quiet-wrong classes
def test_nonverbatim_gold_span_is_caught(tmp_path):
    rows = base_rows()
    rows[0]["gold_spans"] = ["Processed 12400 tonnes of ore"]  # case + comma differ
    path = write_rows(tmp_path, rows)
    assert run_checker(path) == 1
    assert "nonverbatim_gold_span" in checks_reported(path)


def test_nonverbatim_gold_span_really_makes_the_row_unpassable():
    """A model that cites the gold span character-for-character still fails."""
    row = base_rows()[0]
    row["gold_spans"] = ["Processed 12400 tonnes of ore"]
    raw = E.MockGenerator("oracle")(row, {})
    rec = E.score_example(row, raw, E.make_judge("mock:yes", "unused"))
    assert rec["passed"] is False
    assert rec["failure"] == "nonverbatim_citation"


def test_double_quote_in_passage_is_caught(tmp_path):
    rows = base_rows()
    rows[0]["passage"] = ('The board called the quarter a "structural inflection" '
                          'and processed 12,400 tonnes of ore in the third quarter.')
    rows[0]["gold_spans"] = ["processed 12,400 tonnes of ore in the third quarter"]
    path = write_rows(tmp_path, rows)
    assert run_checker(path) == 1
    assert "quote_in_passage" in checks_reported(path)


def test_double_quote_span_scores_on_quoting_convention_not_behavior():
    """Only the raw-quote rendering survives; every other faithful rendering of
    the same span is scored nonverbatim_citation."""
    passage = 'The board called the quarter a "structural inflection" in its filing.'
    row = {"id": "Q1", "adversarial": False, "gold_label": "answerable",
           "question": "What did the board call the quarter?", "passage": passage,
           "gold_answer": "A structural inflection.",
           "gold_spans": ['called the quarter a "structural inflection"']}
    judge = E.make_judge("mock:yes", "unused")
    raw_quotes = 'Answer: A structural inflection.\nCitation: "called the quarter a "structural inflection""'
    escaped = 'Answer: A structural inflection.\nCitation: "called the quarter a \\"structural inflection\\""'
    curly = 'Answer: A structural inflection.\nCitation: "called the quarter a “structural inflection”"'
    assert E.score_example(row, raw_quotes, judge)["passed"] is True
    for variant in (escaped, curly):
        rec = E.score_example(row, variant, judge)
        assert rec["pred_class"] == "answer"          # it parses...
        assert rec["failure"] == "nonverbatim_citation"  # ...and then fails


def test_missing_adversarial_key_is_caught(tmp_path):
    rows = [{k: v for k, v in r.items() if k != "adversarial"} for r in base_rows()]
    path = write_rows(tmp_path, rows)
    assert run_checker(path) == 1
    assert "missing_adversarial" in checks_reported(path)


def test_missing_adversarial_really_empties_the_robustness_metric():
    rows = [{k: v for k, v in r.items() if k != "adversarial"} for r in base_rows()]
    _, m = E.run_eval("mock:oracle", rows, "zero_shot", None,
                      E.make_judge("mock:yes", "unused"), workers=1)
    assert m["spec_adherence"] == 1.0
    assert m["robustness"] is None  # prints as "n/a" -- the metric vanished


@pytest.mark.parametrize("value", ["true", "false", 1, 0, "no"])
def test_non_boolean_adversarial_is_caught(tmp_path, value):
    rows = base_rows()
    rows[1]["adversarial"] = value
    path = write_rows(tmp_path, rows)
    assert run_checker(path) == 1
    assert "non_boolean_adversarial" in checks_reported(path)


def test_string_false_adversarial_really_inflates_the_denominator():
    """bool("false") is True: a stringly-typed 'false' silently joins the
    adversarial subset."""
    row = dict(base_rows()[1], adversarial="false")
    rec = E.score_example(row, E.REFUSAL_STRING, E.make_judge("mock:yes", "unused"))
    assert rec["adversarial"] is True


def test_duplicate_ids_are_caught(tmp_path):
    rows = base_rows()
    rows[1]["id"] = rows[0]["id"]
    path = write_rows(tmp_path, rows)
    assert run_checker(path) == 1
    assert "duplicate_id" in checks_reported(path)


def test_malformed_json_line_is_caught(tmp_path):
    path = tmp_path / "broken.jsonl"
    body = "".join(json.dumps(r) + "\n" for r in base_rows())
    path.write_text(body + '{"id": "X1", "gold_label": \n', encoding="utf-8")
    assert run_checker(path) == 1
    assert "bad_json" in checks_reported(path)


def test_empty_file_is_caught(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    assert run_checker(path) == 1


def test_missing_file_exits_two(tmp_path):
    assert run_checker(tmp_path / "nope.jsonl") == 2


# ------------------------------------------------------------------ contract
def test_validator_is_read_only(tmp_path):
    """It must not rewrite the eval set it is handed, and must not touch the
    frozen harness."""
    path = write_rows(tmp_path, base_rows())
    before_set = path.read_bytes()
    eval_py = REPO_ROOT / "eval.py"
    before_eval = eval_py.read_bytes()

    bad = write_rows(tmp_path, [dict(base_rows()[1], gold_label="nope")], name="bad.jsonl")
    before_bad = bad.read_bytes()

    assert run_checker(path) == 0
    assert run_checker(bad) == 1

    assert path.read_bytes() == before_set
    assert bad.read_bytes() == before_bad
    assert eval_py.read_bytes() == before_eval
    assert E.rubric_hash() == "c0adb876c3e6c4c9"


def test_json_output_is_machine_readable(tmp_path, capsys):
    rows = base_rows()
    rows[1]["gold_label"] = "not_answerable"
    path = write_rows(tmp_path, rows)
    assert run_checker(path, "--json") == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["stats"]["n"] == 2
    assert any(e["check"] == "bad_gold_label" for e in payload["errors"])


def readme_text():
    return (REPO_ROOT / "README.md").read_text(encoding="utf-8")


def test_readme_documents_the_preflight():
    """The command is only useful to a grader if it is written down, under a
    heading they can find."""
    lines = readme_text().splitlines()
    assert "## Held-out eval sets" in lines, "README lost the held-out heading"
    invocations = [ln.strip() for ln in lines
                   if ln.strip().startswith("python scripts/check_eval_set.py")]
    assert invocations, "README no longer shows a runnable preflight invocation"
    # and the eval command for an unseen set, gated on the preflight
    assert any(ln.strip().endswith("\\") or "&&" in ln for ln in invocations), (
        "the preflight is documented but never chained to the eval run")
    assert 'eval.py --model hf:<tuned-repo>@<commit>' in readme_text()


def test_readme_records_the_two_environment_facts():
    """Both trip a grader before a single scenario is scored: the interpreter
    floor, and the provider prefix the project spec omits."""
    readme = readme_text()
    assert "Python 3.11+ is required" in readme
    assert "unsupported operand type(s) for |" in readme  # the 3.9 failure text
    assert "model spec must be provider:model" in readme  # the bare-repo-id failure
    assert "hf:Qwen/Qwen3-1.7B@" in readme                # the working form, revision pinned


def test_readme_only_claims_paths_that_exist():
    """DEFECT: the README used to advertise repo contents that were not there.
    Every path the layout table marks 'present' must actually be present; the
    rest must be attributed to the phase that produces them."""
    import re

    rows = [ln for ln in readme_text().splitlines()
            if ln.startswith("|") and ln.count("|") >= 4]
    checked = 0
    for row in rows:
        cells = [c.strip() for c in row.strip("|").split("|")]
        path_cell, status = cells[0], cells[-1]
        if not status.startswith("present"):
            continue
        for path in re.findall(r"`([^`]+)`", path_cell):
            assert (REPO_ROOT / path).exists(), f"README claims {path!r} is present; it is not"
            checked += 1
    assert checked >= 8, "layout table stopped listing concrete paths"


def test_readme_attributes_unproduced_artifacts_to_a_phase():
    readme = readme_text()
    for path, phase in (("data/train/", "P3"), ("docs/MVP_REPORT.md", "P6")):
        row = next((ln for ln in readme.splitlines()
                    if ln.startswith("|") and f"`{path}`" in ln), None)
        assert row is not None, f"{path} vanished from the layout table"
        assert "produced by" in row and phase in row, (
            f"{path} is not in the repo; the README must attribute it to {phase}, got: {row}")
