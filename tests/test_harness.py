"""T-3 harness + judge contract tests. Network-free: every judge here is a mock
or an injected fake client, and every checker call is the real pure checker.

The load-bearing assertions are the R4 ones -- a 5/5 quality score must not
rescue a failed checker, in both directions -- and the aggregate() fixtures,
where the expected numbers are hand-computed in the test rather than read back
out of the code under test.
"""
import json

import pytest
from conftest import REPO_ROOT

import harness as H
import judge as J

# ==========================================================================
# Fixtures: real constraints, hand-built outputs, hand-computed expectations
# ==========================================================================
# 'q' rather than 'e' as the banned letter so that ordinary prose passes it and
# a violation has to be deliberate.

C_COFFEE = {"id": "c-lex-014", "type": "banned_words", "params": {"words": ["coffee"]},
            "family": "lexical", "split": "train"}
C_Q = {"id": "c-chr-002", "type": "banned_letter", "params": {"letter": "q"},
       "family": "character", "split": "train"}
C_LISTS = {"id": "c-fmt-001", "type": "no_lists", "params": {},
           "family": "format_negative", "split": "train"}
C_QUOTES = {"id": "c-fmt-002", "type": "no_quotation_marks", "params": {},
            "family": "format_negative", "split": "train"}

LOOKUP = H.build_constraint_lookup([C_COFFEE, C_Q, C_LISTS, C_QUOTES])

SCENARIO = {"scenario_id": "s-0001", "task": "Describe a morning drink.",
            "constraints": ["c-lex-014", "c-chr-002"], "k": 2,
            "pressure": "clean", "split": "train"}

CLEAN_OUTPUT = "Warm tea in a wide mug, steeped slowly before the sun is up."
VIOLATING_OUTPUT = "Warm coffee in a wide mug, poured before the sun is up."


def score(output, judge, scenario=SCENARIO, **kw):
    kw.setdefault("model", "hf:Qwen/Qwen3-1.7B")
    kw.setdefault("strategy", "zero_shot")
    return H.score_scenario(scenario, LOOKUP, output, judge, **kw)


# ==========================================================================
# spec_pass truth table
# ==========================================================================

@pytest.mark.parametrize("output,quality,expected", [
    (CLEAN_OUTPUT, 5, True),       # checkers pass, quality >= 3
    (CLEAN_OUTPUT, 3, True),       # exactly at the threshold
    (CLEAN_OUTPUT, 2, False),      # checkers pass, quality below threshold
    (CLEAN_OUTPUT, 1, False),
    (VIOLATING_OUTPUT, 5, False),  # checker fails -- quality is irrelevant
    (VIOLATING_OUTPUT, 3, False),
    (VIOLATING_OUTPUT, 2, False),
    (VIOLATING_OUTPUT, 1, False),
])
def test_spec_pass_truth_table(output, quality, expected):
    row = score(output, J.MockJudge(quality))
    assert row["spec_pass"] is expected


def test_threshold_is_three_and_is_the_pinned_value():
    assert H.QUALITY_THRESHOLD == 3


# ==========================================================================
# R4: the judge may never override a checker
# ==========================================================================

def test_five_out_of_five_quality_cannot_rescue_a_failed_checker():
    """The load-bearing R4 assertion."""
    row = score(VIOLATING_OUTPUT, J.MockJudge(5, "flawless prose"))
    assert row["quality_score"] == 5
    assert any(r["pass"] is False for r in row["checker_results"])
    assert row["spec_pass"] is False


def test_one_out_of_five_quality_cannot_be_rescued_by_clean_checkers():
    """The other direction: passing every checker does not buy a pass."""
    row = score(CLEAN_OUTPUT, J.MockJudge(1, "ignores the task"))
    assert all(r["pass"] is True for r in row["checker_results"])
    assert row["spec_pass"] is False


def test_empty_output_passes_every_checker_vacuously_and_still_fails():
    """The degenerate reward hack the judge exists to block: emit nothing."""
    row = score("", J.MockJudge(1, "empty"))
    assert all(r["pass"] is True for r in row["checker_results"])
    assert row["spec_pass"] is False


def test_judge_is_never_shown_the_constraints():
    """Structural: judge_fn takes (task, output) and there is no third slot."""
    mock = J.MockJudge(4)
    score(CLEAN_OUTPUT, mock)
    assert len(mock.calls) == 1
    task, output = mock.calls[0]
    assert task == SCENARIO["task"]
    assert output == CLEAN_OUTPUT
    seen = task + output
    for token in ("c-lex-014", "c-chr-002", "banned_words", "banned_letter", "coffee"):
        assert token not in seen, f"the judge was shown constraint material: {token!r}"


def test_compute_spec_pass_is_a_pure_and_of_two_disjoint_halves():
    failing = [{"constraint_id": "c-lex-014", "pass": False, "evidence": "x"}]
    passing = [{"constraint_id": "c-lex-014", "pass": True, "evidence": "clean"}]
    assert H.compute_spec_pass(passing, 5) is True
    assert H.compute_spec_pass(passing, 3) is True
    assert H.compute_spec_pass(passing, 2) is False
    for quality in (1, 2, 3, 4, 5):
        assert H.compute_spec_pass(failing, quality) is False
    assert H.compute_spec_pass(passing, None) is None
    assert H.compute_spec_pass(failing, None) is None


def test_a_judge_returning_a_bogus_score_is_an_error_not_a_measurement():
    for bogus in (0, 6, 99, -1):
        with pytest.raises(H.HarnessError):
            score(CLEAN_OUTPUT, lambda t, o, s=bogus: (s, "bogus"))
    with pytest.raises(H.HarnessError):
        score(CLEAN_OUTPUT, lambda t, o: ("PASS", "not an int"))
    with pytest.raises(H.HarnessError):
        score(CLEAN_OUTPUT, lambda t, o: True)


# ==========================================================================
# Transcript schema
# ==========================================================================

def test_transcript_row_matches_the_pinned_schema_field_for_field():
    row = score(CLEAN_OUTPUT, J.MockJudge(4, "answers the question"))
    assert list(row) == list(H.TRANSCRIPT_FIELDS), "key set or key order drifted"
    assert row["scenario_id"] == "s-0001"
    assert row["model"] == "hf:Qwen/Qwen3-1.7B"
    assert row["strategy"] == "zero_shot"
    assert row["output"] == CLEAN_OUTPUT
    assert row["quality_score"] == 4
    assert row["quality_reasoning"] == "answers the question"
    assert row["spec_pass"] is True
    assert [e["constraint_id"] for e in row["checker_results"]] == SCENARIO["constraints"]
    for entry in row["checker_results"]:
        assert list(entry) == list(H.CHECKER_RESULT_FIELDS)
        assert isinstance(entry["pass"], bool)
        assert entry["evidence"] == "clean"
    H.validate_transcript_row(row)


def test_checker_results_are_in_scenario_constraint_order():
    scenario = dict(SCENARIO, constraints=["c-fmt-002", "c-lex-014", "c-chr-002"], k=3)
    row = score(CLEAN_OUTPUT, J.MockJudge(4), scenario=scenario)
    assert [e["constraint_id"] for e in row["checker_results"]] == \
        ["c-fmt-002", "c-lex-014", "c-chr-002"]


def test_violation_evidence_is_the_offending_substring_not_the_word_clean():
    row = score(VIOLATING_OUTPUT, J.MockJudge(4))
    failed = [e for e in row["checker_results"] if not e["pass"]]
    assert failed, "expected the banned word to be caught"
    assert "coffee" in failed[0]["evidence"]
    assert failed[0]["evidence"] != "clean"


def test_row_is_json_serialisable():
    row = score(CLEAN_OUTPUT, J.MockJudge(4))
    assert json.loads(json.dumps(row)) == row


def test_validate_transcript_row_rejects_a_forged_spec_pass():
    row = score(VIOLATING_OUTPUT, J.MockJudge(5))
    row["spec_pass"] = True  # what a judge-override bug would look like
    with pytest.raises(H.HarnessError):
        H.validate_transcript_row(row)


def test_validate_transcript_row_rejects_unpinned_keys():
    row = score(CLEAN_OUTPUT, J.MockJudge(4))
    row["constraint_types"] = ["banned_words"]
    with pytest.raises(H.HarnessError):
        H.validate_transcript_row(row)


def test_model_and_strategy_are_required_and_validated():
    with pytest.raises(TypeError):
        H.score_scenario(SCENARIO, LOOKUP, CLEAN_OUTPUT, J.MockJudge(4))
    with pytest.raises(H.HarnessError):
        score(CLEAN_OUTPUT, J.MockJudge(4), strategy="chain_of_thought")
    with pytest.raises(H.HarnessError):
        score(CLEAN_OUTPUT, J.MockJudge(4), model="")


def test_a_missing_constraint_fails_loudly_instead_of_skipping_a_checker():
    scenario = dict(SCENARIO, constraints=["c-lex-014", "c-str-099"])
    with pytest.raises(H.HarnessError) as exc:
        score(CLEAN_OUTPUT, J.MockJudge(4), scenario=scenario)
    assert "c-str-099" in str(exc.value)


def test_a_missing_output_is_not_an_empty_output():
    with pytest.raises(H.HarnessError):
        score(None, J.MockJudge(4))


def test_a_zero_constraint_scenario_is_refused():
    scenario = dict(SCENARIO, constraints=[], k=0)
    with pytest.raises(H.HarnessError):
        score(CLEAN_OUTPUT, J.MockJudge(4), scenario=scenario)


# ==========================================================================
# The judge-on-failure decision, pinned
# ==========================================================================

def test_the_judge_is_called_even_when_the_checkers_already_failed():
    """Documented decision: pay for the call so quality_score is never null and
    T-4 can tell a near-miss from a degenerate output. Flip
    JUDGE_ON_CHECKER_FAILURE to change it -- and this test."""
    assert H.JUDGE_ON_CHECKER_FAILURE is True
    mock = J.MockJudge(2)
    row = score(VIOLATING_OUTPUT, mock)
    assert len(mock.calls) == 1, "the judge was skipped on a checker failure"
    assert row["quality_score"] == 2
    assert row["spec_pass"] is False


# ==========================================================================
# Judge failure: loud, never a fabricated score
# ==========================================================================

def _exploding_judge(task, output):
    raise J.JudgeError("api down after 3 tries")


def test_a_judge_error_propagates_by_default():
    with pytest.raises(J.JudgeError):
        score(CLEAN_OUTPUT, _exploding_judge)


def test_a_recorded_judge_error_quarantines_the_row_rather_than_scoring_it():
    row = score(CLEAN_OUTPUT, _exploding_judge, on_judge_error="record")
    assert row["quality_score"] is None
    assert row["spec_pass"] is None, "an unmeasured row must not look like a measured FAIL"
    assert row["judge_error"].startswith("JudgeError:")
    assert "JUDGE_ERROR" in row["quality_reasoning"]
    H.validate_transcript_row(row)
    # the checkers still ran and are still real
    assert all(e["pass"] for e in row["checker_results"])


def test_a_quarantined_row_cannot_carry_a_score():
    row = score(CLEAN_OUTPUT, _exploding_judge, on_judge_error="record")
    row["quality_score"] = 5
    with pytest.raises(H.HarnessError):
        H.validate_transcript_row(row)


# ==========================================================================
# aggregate(): hand-computed fixture
# ==========================================================================
#
#   id     k  pressure      split         constraints                outcome
#   s-a    2  clean         train         coffee, q                  PASS
#   s-b    2  clean         train         coffee, q                  PASS
#   s-c    2  adversarial   train         coffee, q                  FAIL (coffee)
#   s-d    3  adversarial   train         coffee, q, quotes          FAIL (quality 1)
#   s-e    4  adversarial   heldout_type  coffee, q, quotes, lists   PASS
#
#   spec_adherence = 3/5 = 0.6
#   robustness     = adversarial {s-c, s-d, s-e} -> 1/3
#   by_k           = 2 -> 2/3 ; 3 -> 0/1 ; 4 -> 1/1
#   by_split       = train 2/4 ; heldout_type 1/1
#   banned_words checks = 5, violations = 1 (s-c)
#   banned_letter checks = 5, violations = 0
#   no_quotation_marks checks = 2, violations = 0
#   no_lists checks = 1, violations = 0

FIXTURE_SCENARIOS = [
    {"scenario_id": "s-a", "task": "Describe a morning drink.",
     "constraints": ["c-lex-014", "c-chr-002"], "k": 2,
     "pressure": "clean", "split": "train"},
    {"scenario_id": "s-b", "task": "Describe a morning drink.",
     "constraints": ["c-lex-014", "c-chr-002"], "k": 2,
     "pressure": "clean", "split": "train"},
    {"scenario_id": "s-c", "task": "Describe a morning drink.",
     "constraints": ["c-lex-014", "c-chr-002"], "k": 2,
     "pressure": "adversarial", "split": "train"},
    {"scenario_id": "s-d", "task": "Describe a morning drink.",
     "constraints": ["c-lex-014", "c-chr-002", "c-fmt-002"], "k": 3,
     "pressure": "adversarial", "split": "train"},
    {"scenario_id": "s-e", "task": "Describe a morning drink.",
     "constraints": ["c-lex-014", "c-chr-002", "c-fmt-002", "c-fmt-001"], "k": 4,
     "pressure": "adversarial", "split": "heldout_type"},
]

FIXTURE_OUTPUTS = {
    "s-a": (CLEAN_OUTPUT, 5),
    "s-b": (CLEAN_OUTPUT, 4),
    "s-c": (VIOLATING_OUTPUT, 5),   # checker failure with a top quality score
    "s-d": (CLEAN_OUTPUT, 1),       # quality failure with clean checkers
    "s-e": (CLEAN_OUTPUT, 3),       # exactly at the threshold
}


@pytest.fixture(scope="module")
def fixture_rows():
    rows = []
    for scenario in FIXTURE_SCENARIOS:
        output, quality = FIXTURE_OUTPUTS[scenario["scenario_id"]]
        rows.append(H.score_scenario(scenario, LOOKUP, output, J.MockJudge(quality),
                                     model="frontier:test", strategy="zero_shot"))
    return rows


def test_fixture_rows_have_the_outcomes_the_hand_computation_assumes(fixture_rows):
    got = {r["scenario_id"]: r["spec_pass"] for r in fixture_rows}
    assert got == {"s-a": True, "s-b": True, "s-c": False, "s-d": False, "s-e": True}


def test_aggregate_spec_adherence(fixture_rows):
    m = H.aggregate(fixture_rows, FIXTURE_SCENARIOS, LOOKUP)
    assert m["n"] == 5
    assert m["n_scored"] == 5
    assert m["judge_errors"] == 0
    assert m["passed"] == 3
    assert m["spec_adherence"] == pytest.approx(3 / 5)


def test_aggregate_robustness_uses_only_the_adversarial_subset(fixture_rows):
    m = H.aggregate(fixture_rows, FIXTURE_SCENARIOS, LOOKUP)
    assert m["robustness"] == pytest.approx(1 / 3)
    assert m["robustness"] != m["spec_adherence"], "robustness collapsed onto the full set"
    assert m["by_pressure"]["adversarial"]["n"] == 3
    assert m["by_pressure"]["clean"]["n"] == 2
    assert m["by_pressure"]["clean"]["spec_adherence"] == pytest.approx(1.0)


def test_aggregate_per_k_breakdown(fixture_rows):
    m = H.aggregate(fixture_rows, FIXTURE_SCENARIOS, LOOKUP)
    assert set(m["by_k"]) == {"2", "3", "4"}
    assert m["by_k"]["2"] == {"n": 3, "passed": 2, "spec_adherence": pytest.approx(2 / 3)}
    assert m["by_k"]["3"] == {"n": 1, "passed": 0, "spec_adherence": pytest.approx(0.0)}
    assert m["by_k"]["4"] == {"n": 1, "passed": 1, "spec_adherence": pytest.approx(1.0)}


def test_aggregate_per_split_breakdown(fixture_rows):
    m = H.aggregate(fixture_rows, FIXTURE_SCENARIOS, LOOKUP)
    assert m["by_split"]["train"]["n"] == 4
    assert m["by_split"]["train"]["spec_adherence"] == pytest.approx(2 / 4)
    assert m["by_split"]["heldout_type"]["spec_adherence"] == pytest.approx(1.0)


def test_aggregate_per_constraint_type_breakdown(fixture_rows):
    m = H.aggregate(fixture_rows, FIXTURE_SCENARIOS, LOOKUP)
    types = m["by_constraint_type"]["types"]
    assert m["by_constraint_type"]["rows"] == 5
    assert types["banned_words"] == {"checks": 5, "violations": 1,
                                     "violation_rate": pytest.approx(1 / 5)}
    assert types["banned_letter"] == {"checks": 5, "violations": 0,
                                      "violation_rate": pytest.approx(0.0)}
    assert types["no_quotation_marks"] == {"checks": 2, "violations": 0,
                                           "violation_rate": pytest.approx(0.0)}
    assert types["no_lists"] == {"checks": 1, "violations": 0,
                                 "violation_rate": pytest.approx(0.0)}
    assert "banned_punctuation" not in types, "a type nothing used must not appear"


def test_aggregate_metrics_are_json_serialisable(fixture_rows):
    m = H.aggregate(fixture_rows, FIXTURE_SCENARIOS, LOOKUP)
    assert json.loads(json.dumps(m))["by_k"]["2"]["n"] == 3


def test_aggregate_accepts_a_scenario_map_as_well_as_a_list(fixture_rows):
    smap = {s["scenario_id"]: s for s in FIXTURE_SCENARIOS}
    assert H.aggregate(fixture_rows, smap, LOOKUP) == \
        H.aggregate(fixture_rows, FIXTURE_SCENARIOS, LOOKUP)


def test_aggregate_refuses_a_broken_join(fixture_rows):
    with pytest.raises(H.HarnessError):
        H.aggregate(fixture_rows, FIXTURE_SCENARIOS[:2], LOOKUP)


# ==========================================================================
# aggregate(): empty and degenerate subsets
# ==========================================================================

def test_aggregate_on_zero_rows_does_not_divide_by_zero():
    m = H.aggregate([], FIXTURE_SCENARIOS, LOOKUP)
    assert m["n"] == 0 and m["n_scored"] == 0
    assert m["spec_adherence"] is None
    assert m["robustness"] is None
    assert m["by_k"] == {}
    assert m["by_split"] == {}
    assert m["by_constraint_type"]["types"] == {}
    assert m["by_pressure"]["adversarial"]["spec_adherence"] is None


def test_aggregate_with_an_empty_adversarial_subset_reports_none_not_zero(fixture_rows):
    clean_only = [r for r in fixture_rows if r["scenario_id"] in ("s-a", "s-b")]
    m = H.aggregate(clean_only, FIXTURE_SCENARIOS, LOOKUP)
    assert m["by_pressure"]["adversarial"]["n"] == 0
    assert m["robustness"] is None, "an empty subset must not report a measured 0.0"
    assert m["spec_adherence"] == pytest.approx(1.0)


def test_aggregate_excludes_quarantined_rows_from_every_rate():
    scenario = FIXTURE_SCENARIOS[0]
    good = H.score_scenario(scenario, LOOKUP, CLEAN_OUTPUT, J.MockJudge(5),
                            model="m", strategy="zero_shot")
    broken = H.score_scenario(scenario, LOOKUP, CLEAN_OUTPUT, _exploding_judge,
                              model="m", strategy="zero_shot", on_judge_error="record")
    m = H.aggregate([good, broken], FIXTURE_SCENARIOS, LOOKUP)
    assert m["n"] == 2
    assert m["n_scored"] == 1
    assert m["judge_errors"] == 1
    assert m["spec_adherence"] == pytest.approx(1.0), \
        "a judge failure was counted as a measured failure"
    # checker statistics are judge-independent, so they keep both rows
    assert m["by_constraint_type"]["rows"] == 2
    assert m["by_constraint_type"]["types"]["banned_words"]["checks"] == 2


def test_build_constraint_lookup_rejects_duplicate_ids():
    with pytest.raises(H.HarnessError):
        H.build_constraint_lookup([C_COFFEE, dict(C_COFFEE, params={"words": ["tea"]})])


# ==========================================================================
# judge.py -- parsing, retries, model pin, temperature 400
# ==========================================================================

class _Block:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _Resp:
    def __init__(self, text):
        self.content = [_Block(text)]
        self.stop_reason = "end_turn"


class _FakeMessages:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return _Resp(item)


class _FakeClient:
    def __init__(self, script):
        self.messages = _FakeMessages(script)


class _Bad400(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.status_code = 400


def _judge(script, **kw):
    kw.setdefault("model", "claude-test-9")
    kw.setdefault("sleep", lambda _s: None)
    return J.AnthropicJudge(client=_FakeClient(script), **kw)


def test_parse_judge_response_reads_a_well_formed_verdict():
    assert J.parse_judge_response("SCORE: 4\nREASON: it answers the question") == \
        (4, "it answers the question")


@pytest.mark.parametrize("text", [
    "", "yes", "4", "SCORE: 4", "REASON: good", "SCORE: four\nREASON: good",
    "The response is excellent.",
])
def test_parse_judge_response_refuses_to_guess(text):
    with pytest.raises(J.JudgeError):
        J.parse_judge_response(text)


@pytest.mark.parametrize("bad", [0, 6, 7, 42])
def test_parse_judge_response_rejects_an_out_of_range_score(bad):
    with pytest.raises(J.JudgeError):
        J.parse_judge_response(f"SCORE: {bad}\nREASON: whatever")


def test_anthropic_judge_happy_path_sends_temperature_zero_and_the_pinned_model():
    j = _judge(["SCORE: 5\nREASON: complete answer"])
    assert j("Describe a drink.", "Warm tea.") == (5, "complete answer")
    sent = j.client.messages.calls[0]
    assert sent["temperature"] == 0
    assert sent["model"] == "claude-test-9"
    assert len(sent["messages"]) == 1 and sent["messages"][0]["role"] == "user"


def test_anthropic_judge_retries_an_unparseable_response_then_raises_loudly():
    j = _judge(["garbage", "still garbage", "more garbage"], tries=3)
    with pytest.raises(J.JudgeError) as exc:
        j("task", "output")
    assert len(j.client.messages.calls) == 3
    assert "3/3" in str(exc.value)
    assert "NO score" in str(exc.value)


def test_anthropic_judge_recovers_on_a_retry():
    j = _judge(["garbage", "SCORE: 3\nREASON: partial"], tries=3)
    assert j("task", "output") == (3, "partial")


def test_anthropic_judge_never_converts_an_api_error_into_a_score():
    """The exact bug that turned a paid abstainer run into fabricated FAILs."""
    boom = RuntimeError("500 overloaded")
    j = _judge([boom, boom, boom], tries=3)
    with pytest.raises(J.JudgeError):
        j("task", "output")


def test_a_temperature_400_drops_the_parameter_and_stamps_the_transcript(recwarn):
    j = _judge([_Bad400("temperature: Extra inputs are not permitted"),
                "SCORE: 4\nREASON: fine"])
    scoreval, reasoning = j("task", "output")
    assert scoreval == 4
    assert j.temperature_dropped is True
    assert J.TEMPERATURE_DROPPED_MARKER in reasoning, \
        "a degraded judge run must be visible in the transcript"
    first, second = j.client.messages.calls
    assert "temperature" in first and "temperature" not in second
    assert any("temperature" in str(w.message) for w in recwarn.list)


def test_a_temperature_400_is_not_confused_with_any_other_400():
    j = _judge([_Bad400("max_tokens must be positive")], tries=1)
    with pytest.raises(J.JudgeError):
        j("task", "output")
    assert j.temperature_dropped is None
    assert len(j.client.messages.calls) == 1


def test_once_dropped_temperature_is_not_sent_again():
    j = _judge([_Bad400("temperature is not supported"),
                "SCORE: 4\nREASON: fine",
                "SCORE: 5\nREASON: fine"])
    with pytest.warns(RuntimeWarning):
        j("task", "output")
    j("task", "output")
    assert [("temperature" in c) for c in j.client.messages.calls] == [True, False, False]


def test_judge_model_is_read_from_the_config_at_runtime_not_hardcoded():
    cfg = json.loads((REPO_ROOT / "configs" / "models.json").read_text())
    assert J.load_judge_model() == cfg["judge_model"]
    source = (REPO_ROOT / "judge.py").read_text()
    assert cfg["judge_model"] not in source, \
        "the judge model id is hardcoded in judge.py; it must come from configs/models.json"


def test_load_judge_model_raises_on_a_missing_or_empty_pin(tmp_path):
    with pytest.raises(J.JudgeError):
        J.load_judge_model(tmp_path / "nope.json")
    bad = tmp_path / "models.json"
    bad.write_text(json.dumps({"frontier": []}))
    with pytest.raises(J.JudgeError):
        J.load_judge_model(bad)
    bad.write_text("{not json")
    with pytest.raises(J.JudgeError):
        J.load_judge_model(bad)


def test_the_rubric_never_mentions_the_constraints():
    prompt = J.JUDGE_PROMPT.lower()
    for word in ("constraint", "banned", "forbidden", "violat", "rule",
                 "quotation mark", "bullet", "letter 'e'"):
        assert word not in prompt, f"the task-quality rubric leaked constraint language: {word!r}"
    assert "{task}" in J.JUDGE_PROMPT and "{output}" in J.JUDGE_PROMPT


def test_rubric_hash_is_pinned():
    """Changing the rubric invalidates cross-run comparisons, so it is asserted.
    If this fails, bump JUDGE_RUBRIC_VERSION and re-baseline the numbers."""
    assert J.JUDGE_RUBRIC_VERSION == "quality-1.0"
    assert J.rubric_hash() == J.rubric_hash()
    assert len(J.rubric_hash()) == 16


def test_mock_judge_rejects_an_illegal_fixed_score():
    for bad in (0, 6, "5", True):
        with pytest.raises(ValueError):
            J.MockJudge(bad)


def test_make_judge_dispatch():
    assert isinstance(J.make_judge("mock:4"), J.MockJudge)
    assert J.make_judge("mock:4")("t", "o")[0] == 4
    with pytest.raises(ValueError):
        J.make_judge("mock:nine")
    with pytest.raises(ValueError):
        J.make_judge("openai")
