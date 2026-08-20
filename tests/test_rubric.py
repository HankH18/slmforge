"""The rubric is the contract. If anything here fails, scoring drifted and every
previously reported number is invalid."""
import eval as E
import pytest

from conftest import EXEMPLARS_PATH, SCENARIOS_PATH


def test_rubric_hash_pinned():
    """Pins prompts + judge prompt + version together. A change here silently
    invalidates cross-run comparisons, so it is asserted, not logged."""
    assert E.rubric_hash() == "c0adb876c3e6c4c9"


def test_rubric_version_and_refusal_string():
    assert E.RUBRIC_VERSION == "1.0"
    assert E.REFUSAL_STRING == "Not answerable from the provided context."


def test_eval_set_shape():
    rows = E.load_jsonl(str(SCENARIOS_PATH))
    assert len(rows) == 32, "project doc requires >= 30 scenarios per model x strategy cell"
    assert sum(r["gold_label"] == "unanswerable" for r in rows) == 19
    assert sum(r["gold_label"] == "answerable" for r in rows) == 13
    assert sum(bool(r["adversarial"]) for r in rows) == 16
    assert len({r["id"] for r in rows}) == 32, "scenario ids must be unique"
    for r in rows:
        assert r["gold_label"] in ("answerable", "unanswerable")
        assert '"' not in r["passage"], f"{r['id']}: quote in passage breaks Citation parsing"
        if r["gold_label"] == "answerable":
            assert r["gold_spans"], f"{r['id']}: answerable scenario needs gold spans"
            for s in r["gold_spans"]:
                assert s in r["passage"], f"{r['id']}: gold span is not verbatim"


def test_exemplars_are_on_spec():
    """Few-shot exemplars teach the format; if they are off-spec the few_shot
    ablation cell measures the exemplars, not the model."""
    rows = E.load_jsonl(str(EXEMPLARS_PATH))
    assert len(rows) == 6
    for ex in rows:
        parsed = E.parse_output(ex["target_output"])
        assert parsed["class"] in ("answer", "refusal"), f"{ex['id']}: {parsed}"
        if parsed["class"] == "answer":
            for span in parsed["spans"]:
                assert span in ex["passage"], f"{ex['id']}: exemplar cites a non-verbatim span"


# ---------------------------------------------------------------- parse_output
def test_parse_exact_refusal():
    assert E.parse_output(E.REFUSAL_STRING) == {"class": "refusal"}


def test_parse_refusal_tolerates_only_the_two_normalizations():
    # 1: surrounding whitespace, 2: blank lines
    assert E.parse_output(f"\n\n  {E.REFUSAL_STRING}  \n\n")["class"] == "refusal"
    # anything else is not the refusal
    assert E.parse_output(E.REFUSAL_STRING + " Sorry!")["class"] == "malformed"
    assert E.parse_output(E.REFUSAL_STRING.lower())["class"] == "malformed"


def test_parse_wellformed_answer():
    out = 'Answer: 11 percent.\nCitation: "cover rose to 11 percent"'
    p = E.parse_output(out)
    assert p == {"class": "answer", "answer": "11 percent.",
                 "spans": ["cover rose to 11 percent"]}


def test_parse_multiple_citations():
    p = E.parse_output('Answer: x\nCitation: "a"\nCitation: "b"')
    assert p["spans"] == ["a", "b"]


@pytest.mark.parametrize("bad,why", [
    ("", "empty output"),
    ("Sure! The answer is 42.", "first line is not an Answer line"),
    ("Answer: x", "Answer line without any Citation line"),
    ('Answer: x\nCitation: "a"\nHope that helps!', "non-Citation line after Answer"),
    ('Citation: "a"\nAnswer: x', "first line is not an Answer line"),
    ("Answer: x\nCitation: no quotes", "non-Citation line after Answer"),
])
def test_parse_malformed_cases(bad, why):
    p = E.parse_output(bad)
    assert p["class"] == "malformed"
    assert why in p["why"]


# ---------------------------------------------------------------- scoring branches
def _scn(**kw):
    base = {"id": "T1", "type": "t", "adversarial": False, "passage": "alpha beta gamma delta.",
            "question": "q?", "gold_label": "answerable", "gold_answer": "beta",
            "gold_spans": ["alpha beta"]}
    base.update(kw)
    return base


def test_unanswerable_requires_exact_refusal():
    s = _scn(gold_label="unanswerable", gold_answer=None, gold_spans=None)
    yes = E.make_judge("mock:yes", "u")
    assert E.score_example(s, E.REFUSAL_STRING, yes)["passed"] is True
    r = E.score_example(s, 'Answer: beta\nCitation: "alpha beta"', yes)
    assert (r["passed"], r["failure"]) == (False, "false_answer")
    r = E.score_example(s, "I don't know.", yes)
    assert (r["passed"], r["failure"]) == (False, "malformed")


def test_answerable_failure_modes():
    s, yes = _scn(), E.make_judge("mock:yes", "u")
    r = E.score_example(s, E.REFUSAL_STRING, yes)
    assert (r["passed"], r["failure"]) == (False, "false_refusal")
    r = E.score_example(s, 'Answer: beta\nCitation: "NOT IN PASSAGE"', yes)
    assert (r["passed"], r["failure"]) == (False, "nonverbatim_citation")
    assert r["nonverbatim_spans"] == ["NOT IN PASSAGE"]


def test_judge_can_fail_but_never_rescues():
    """Judge runs only after the programmatic checks pass."""
    s = _scn()
    good = 'Answer: beta\nCitation: "alpha beta"'
    assert E.score_example(s, good, E.make_judge("mock:yes", "u"))["passed"] is True
    r = E.score_example(s, good, E.make_judge("mock:no", "u"))
    assert (r["passed"], r["failure"]) == (False, "entailment_fail")

    # a nonverbatim citation fails programmatically, so a YES judge cannot rescue it
    bad = 'Answer: beta\nCitation: "NOT IN PASSAGE"'
    r = E.score_example(s, bad, E.make_judge("mock:yes", "u"))
    assert r["passed"] is False
    assert r["judge_verdict"] is None, "judge must not be consulted after a programmatic fail"
