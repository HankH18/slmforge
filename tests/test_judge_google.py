"""GoogleJudge: same rubric, same contract, same failure policy as AnthropicJudge.

Why this file exists at all
---------------------------
On 2026-08-20 the judge and a graded frontier model ended up on ONE Anthropic
account. The account hit its monthly spend cap mid-run and both died together,
quarantining 147/147 rows of a cell. A judge that shares a failure domain with a
model it grades is one billing event away from ending a run, so the judge needs a
second provider in a third family.

A second provider is only worth having if it is the SAME instrument. Two things
make that true and both are asserted here rather than assumed:

* the rubric BYTES each judge sends are identical -- not "both read the same
  constant", which a future refactor could quietly break by wrapping one
  provider's prompt in a system preamble or a JSON envelope. If the prompts ever
  differ, ``rubric_hash()`` is a lie and every cross-provider number is void.
* the FAILURE POLICY is identical, because both inherit it from one class. A
  provider that returned a score where the other raised would silently convert
  Google's outages into fabricated data -- the exact bug judge.py was written to
  kill.

Everything here is network-free. The live calibration that picked the model is
reported in the ticket, not run in the suite: a test that calls a paid API is a
test that fails when someone else's quota runs out.
"""
import json
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import judge as J  # noqa: E402
from conftest import REPO_ROOT  # noqa: E402


# ==========================================================================
# Fakes -- one per provider, deliberately shaped like the real SDK responses
# ==========================================================================

class _FakeGoogleModels:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0) if self.script else RuntimeError("script exhausted")
        if isinstance(item, Exception):
            raise item
        return item


class FakeGoogleClient:
    def __init__(self, script):
        self.models = _FakeGoogleModels(script)


class _FakeAnthropicMessages:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0) if self.script else RuntimeError("script exhausted")
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=item)], stop_reason="end_turn")


class FakeAnthropicClient:
    def __init__(self, script):
        self.messages = _FakeAnthropicMessages(script)


class Bad400(Exception):
    """google-genai's ClientError carries `code`, NOT `status_code` (measured
    against google-genai 2.19.0 on 2026-08-20)."""

    def __init__(self, message, attr="code"):
        super().__init__(message)
        setattr(self, attr, 400)


def gresp(text=None, finish="STOP", parts=None, candidates=True, block_reason=None):
    """Build a fake GenerateContentResponse.

    ``parts`` lets a test supply thought parts; otherwise a single text part is
    built from ``text``. ``finish`` is passed as an enum-like object with .name
    for the realistic case, since that is what the SDK actually returns.
    """
    if parts is None:
        parts = [SimpleNamespace(text=text, thought=None)] if text is not None else []
    feedback = SimpleNamespace(block_reason=block_reason) if block_reason else None
    if not candidates:
        return SimpleNamespace(candidates=[], prompt_feedback=feedback)
    finish_obj = SimpleNamespace(name=finish) if isinstance(finish, str) else finish
    cand = SimpleNamespace(finish_reason=finish_obj,
                           content=SimpleNamespace(parts=parts))
    return SimpleNamespace(candidates=[cand], prompt_feedback=feedback)


def gjudge(script, **kw):
    kw.setdefault("model", "gemini-test-9")
    kw.setdefault("sleep", lambda _s: None)
    return J.GoogleJudge(client=FakeGoogleClient(script), **kw)


def ajudge(script, **kw):
    kw.setdefault("model", "claude-test-9")
    kw.setdefault("sleep", lambda _s: None)
    return J.AnthropicJudge(client=FakeAnthropicClient(script), **kw)


TASK = "Describe a drink in two sentences."
OUTPUT = "Warm tea, brewed strong. It sits in a chipped blue mug."


# ==========================================================================
# 1. The interchangeability claim: identical rubric BYTES on the wire
# ==========================================================================

def _anthropic_sent_prompt(judge):
    sent = judge.client.messages.calls[0]
    assert len(sent["messages"]) == 1 and sent["messages"][0]["role"] == "user"
    return sent["messages"][0]["content"]


def _google_sent_prompt(judge):
    sent = judge.client.models.calls[0]
    contents = sent["contents"]
    assert len(contents) == 1 and contents[0]["role"] == "user"
    parts = contents[0]["parts"]
    assert len(parts) == 1
    return parts[0]["text"]


def test_both_judges_send_byte_identical_rubric_text():
    """THE interchangeability test.

    Not "both modules reference JUDGE_PROMPT" -- that stays true while one
    provider wraps it in a system preamble, a JSON schema envelope or a
    provider-specific instruction, any of which is a different rubric and makes
    every cross-provider comparison void.
    """
    a = ajudge(["SCORE: 4\nREASON: fine"])
    g = gjudge([gresp("SCORE: 4\nREASON: fine")])
    a(TASK, OUTPUT)
    g(TASK, OUTPUT)

    a_prompt = _anthropic_sent_prompt(a)
    g_prompt = _google_sent_prompt(g)
    assert a_prompt.encode("utf-8") == g_prompt.encode("utf-8"), (
        "the two judges sent different rubric bytes; they are not the same "
        "instrument and rubric_hash() no longer identifies what was sent")
    assert a_prompt == J.JUDGE_PROMPT.format(task=TASK, output=OUTPUT)


def test_neither_judge_sends_anything_besides_the_rubric():
    """No system prompt, no extra turn, no envelope on either side."""
    a = ajudge(["SCORE: 4\nREASON: fine"])
    g = gjudge([gresp("SCORE: 4\nREASON: fine")])
    a(TASK, OUTPUT)
    g(TASK, OUTPUT)
    assert "system" not in a.client.messages.calls[0]
    gcall = g.client.models.calls[0]
    assert set(gcall) == {"model", "contents", "config"}
    assert "system_instruction" not in gcall["config"]


def test_the_rubric_hash_is_a_property_of_the_rubric_not_the_provider():
    """One rubric hash across providers is the whole basis for comparing cells
    scored by different judges."""
    before = J.rubric_hash()
    gjudge([gresp("SCORE: 5\nREASON: ok")])(TASK, OUTPUT)
    ajudge(["SCORE: 5\nREASON: ok"])(TASK, OUTPUT)
    assert J.rubric_hash() == before


def test_the_google_judge_cannot_be_shown_the_constraints():
    """Structural, like the Anthropic one: the signature has nowhere to put them."""
    import inspect
    sig = inspect.signature(J.GoogleJudge.__call__)
    assert list(sig.parameters) == ["self", "task", "output"]


# ==========================================================================
# 2. Same failure policy, by construction and by behaviour
# ==========================================================================

def test_both_judges_inherit_one_failure_policy():
    """If a provider ever overrides __call__ it can grow its own quiet fallback."""
    assert issubclass(J.GoogleJudge, J._RetryingJudge)
    assert issubclass(J.AnthropicJudge, J._RetryingJudge)
    assert J.GoogleJudge.__call__ is J._RetryingJudge.__call__
    assert J.AnthropicJudge.__call__ is J._RetryingJudge.__call__


def test_google_judge_happy_path_sends_temperature_zero_and_the_pinned_model():
    j = gjudge([gresp("SCORE: 5\nREASON: complete answer")])
    assert j(TASK, OUTPUT) == (5, "complete answer")
    sent = j.client.models.calls[0]
    assert sent["model"] == "gemini-test-9"
    assert sent["config"]["temperature"] == 0
    assert sent["config"]["max_output_tokens"] == j.max_tokens


def test_google_judge_retries_an_unparseable_response_then_raises_loudly():
    j = gjudge([gresp("garbage"), gresp("still garbage"), gresp("more garbage")], tries=3)
    with pytest.raises(J.JudgeError) as exc:
        j(TASK, OUTPUT)
    assert len(j.client.models.calls) == 3
    assert "3/3" in str(exc.value) and "NO score" in str(exc.value)


def test_google_judge_recovers_on_a_retry():
    j = gjudge([gresp("garbage"), gresp("SCORE: 3\nREASON: partial")], tries=3)
    assert j(TASK, OUTPUT) == (3, "partial")


def test_google_judge_never_converts_an_api_error_into_a_score():
    boom = RuntimeError("503 UNAVAILABLE: model is experiencing high demand")
    j = gjudge([boom, boom, boom], tries=3)
    with pytest.raises(J.JudgeError):
        j(TASK, OUTPUT)


def test_google_judge_rejects_an_out_of_range_score():
    j = gjudge([gresp("SCORE: 9\nREASON: off the scale")], tries=1)
    with pytest.raises(J.JudgeError):
        j(TASK, OUTPUT)


# ==========================================================================
# 3. Google's real failure modes
# ==========================================================================

def test_a_temperature_400_drops_the_parameter_and_stamps_the_transcript(recwarn):
    j = gjudge([Bad400("GenerateContentRequest.generation_config.temperature: "
                       "temperature is not supported by this model"),
                gresp("SCORE: 4\nREASON: fine")])
    score, reasoning = j(TASK, OUTPUT)
    assert score == 4
    assert j.temperature_dropped is True
    assert J.TEMPERATURE_DROPPED_MARKER in reasoning, \
        "a degraded judge run must be visible in the transcript"
    first, second = (c["config"] for c in j.client.models.calls)
    assert "temperature" in first and "temperature" not in second
    assert any("temperature" in str(w.message) for w in recwarn.list)


def test_a_temperature_400_is_not_confused_with_any_other_400():
    """A 400 about anything else is a real error, not a licence to change the
    sampling parameters and try again."""
    j = gjudge([Bad400("max_output_tokens must be positive")], tries=1)
    with pytest.raises(J.JudgeError):
        j(TASK, OUTPUT)
    assert j.temperature_dropped is None
    assert len(j.client.models.calls) == 1


def test_once_dropped_temperature_is_not_sent_again():
    j = gjudge([Bad400("temperature is not supported"),
                gresp("SCORE: 4\nREASON: fine"),
                gresp("SCORE: 5\nREASON: fine")])
    with pytest.warns(RuntimeWarning):
        j(TASK, OUTPUT)
    j(TASK, OUTPUT)
    assert [("temperature" in c["config"]) for c in j.client.models.calls] == \
        [True, False, False]


def test_a_google_400_is_recognised_through_code_not_status_code():
    """google-genai sets `code`; anthropic sets `status_code`. Reading only one
    of them silently disables the fallback for the other provider."""
    assert J._status_of(Bad400("temperature", attr="code")) == 400
    assert J._status_of(Bad400("temperature", attr="status_code")) == 400
    assert J._is_temperature_400(Bad400("temperature unsupported", attr="code"))
    assert not J._is_temperature_400(Bad400("bad max_output_tokens", attr="code"))


@pytest.mark.parametrize("finish", ["SAFETY", "RECITATION", "PROHIBITED_CONTENT",
                                    "BLOCKLIST", "MALFORMED_FUNCTION_CALL"])
def test_a_refusal_raises_instead_of_scoring(finish):
    """A safety stop is NOT a 1. The judge never read the response, so there is
    no verdict; a 1 recorded here is a fabricated measurement, and the whole
    module exists to refuse those."""
    j = gjudge([gresp(finish=finish)], tries=1)
    with pytest.raises(J.JudgeError) as exc:
        j(TASK, OUTPUT)
    assert finish in str(exc.value)


def test_a_prompt_level_safety_block_raises():
    j = gjudge([gresp(candidates=False, block_reason="SAFETY")], tries=1)
    with pytest.raises(J.JudgeError) as exc:
        j(TASK, OUTPUT)
    assert "blocked before generation" in str(exc.value)


def test_an_empty_candidate_list_raises():
    j = gjudge([gresp(candidates=False)], tries=1)
    with pytest.raises(J.JudgeError) as exc:
        j(TASK, OUTPUT)
    assert "no candidates" in str(exc.value)


def test_an_empty_response_raises():
    j = gjudge([gresp(text="")], tries=1)
    with pytest.raises(J.JudgeError):
        j(TASK, OUTPUT)


def test_a_truncated_verdict_raises_even_though_its_SCORE_line_parses():
    """MEASURED 2026-08-20 on gemini-3-flash-preview: 982 thinking tokens out of
    a 1024 budget left 42 for the verdict, and the body came back as a COMPLETE
    'SCORE: 2' line with the REASON cut off mid-word, finish_reason=MAX_TOKENS.
    That parses. Trusting it would write a truncated reason into the transcript
    and, worse, would make the score depend on how long the model happened to
    think. finish_reason is checked BEFORE the text is parsed for exactly this.
    """
    truncated = ("SCORE: 2\nREASON: The response fails to explain why the service "
                 "is replaced, provide any information about the revised")
    j = gjudge([gresp(truncated, finish="MAX_TOKENS")], tries=1)
    with pytest.raises(J.JudgeError) as exc:
        j(TASK, OUTPUT)
    assert "MAX_TOKENS" in str(exc.value)
    assert J.parse_judge_response(truncated)[0] == 2, \
        "the guard must be the finish_reason, not the parser -- this text parses fine"


def test_thought_parts_are_not_mistaken_for_the_verdict():
    """A thinking model returns its reasoning as a part flagged thought=True. It
    often contains the string 'SCORE: n' as the model talks itself through the
    scale; concatenating it can score the row off the model's scratchpad."""
    parts = [SimpleNamespace(text="Maybe SCORE: 5? No. REASON: hmm", thought=True),
             SimpleNamespace(text="SCORE: 2\nREASON: barely responsive", thought=None)]
    j = gjudge([gresp(parts=parts)])
    assert j(TASK, OUTPUT) == (2, "barely responsive")


def test_a_finish_reason_that_is_a_real_enum_is_read_by_name():
    """The SDK returns FinishReason.STOP, whose str() is 'FinishReason.STOP'."""
    import enum

    class FinishReason(enum.Enum):
        STOP = "STOP"
        SAFETY = "SAFETY"

    ok = gjudge([gresp("SCORE: 4\nREASON: fine", finish=FinishReason.STOP)])
    assert ok(TASK, OUTPUT)[0] == 4
    bad = gjudge([gresp("SCORE: 4\nREASON: fine", finish=FinishReason.SAFETY)], tries=1)
    with pytest.raises(J.JudgeError):
        bad(TASK, OUTPUT)


def test_the_default_token_budget_covers_measured_thinking():
    """Regression guard on the number, not the code.

    MEASURED 2026-08-20 over real run outputs: gemini-3.6-flash spends ~140-155
    thinking tokens on a judge call, but gemini-3-flash-preview was seen at 982
    and truncated at a 1024 budget. The default has to clear the worst observed
    thinker with room, because the failure mode is a quarantined row, and
    quarantined rows are capped at 10% before the whole run is void.
    """
    assert J.GoogleJudge(client=FakeGoogleClient([]), model="m").max_tokens >= 2048


# ==========================================================================
# 4. Provider is configuration, end to end
# ==========================================================================

def test_make_judge_dispatches_google():
    j = J.make_judge("google", "gemini-test-9", client=FakeGoogleClient(
        [gresp("SCORE: 5\nREASON: ok")]))
    assert isinstance(j, J.GoogleJudge)
    assert j(TASK, OUTPUT) == (5, "ok")


def test_make_judge_still_dispatches_anthropic_and_mock():
    a = J.make_judge("anthropic", "claude-test-9",
                     client=FakeAnthropicClient(["SCORE: 5\nREASON: ok"]))
    assert isinstance(a, J.AnthropicJudge)
    assert isinstance(J.make_judge("mock:4"), J.MockJudge)


def test_make_judge_still_refuses_a_provider_it_cannot_dispatch():
    with pytest.raises(ValueError) as exc:
        J.make_judge("openai")
    assert "openai" in str(exc.value)
    for spec in ("gemini", "Google", "google:gemini-3.6-flash", ""):
        with pytest.raises(ValueError):
            J.make_judge(spec)


def test_load_judge_provider_defaults_to_anthropic_when_the_key_is_absent(tmp_path):
    """Every models.json written before the second provider existed lacks the
    key, and those runs were scored by Anthropic. Defaulting anywhere else would
    silently re-score old work with a different instrument."""
    p = tmp_path / "models.json"
    p.write_text(json.dumps({"judge_model": "claude-old"}))
    assert J.load_judge_provider(p) == "anthropic"
    assert J.DEFAULT_JUDGE_PROVIDER == "anthropic"


def test_load_judge_provider_rejects_a_provider_that_cannot_be_dispatched(tmp_path):
    p = tmp_path / "models.json"
    for bad in ("openai", "", None, 5, "Google"):
        p.write_text(json.dumps({"judge_model": "m", "judge_provider": bad}))
        with pytest.raises(J.JudgeError):
            J.load_judge_provider(p)


def test_every_dispatchable_provider_is_a_declared_provider():
    assert set(J.JUDGE_PROVIDERS) == set(J._JUDGE_CLASSES)
    for name in J.JUDGE_PROVIDERS:
        assert issubclass(J._JUDGE_CLASSES[name], J._RetryingJudge)


def test_a_judge_refuses_a_model_pinned_for_a_different_provider(tmp_path):
    """Both providers read ONE judge_model pin. Without this check, flipping
    judge_provider to google and forgetting judge_model asks Gemini for a
    claude-* id -- which fails deep in the SDK with a message about model names
    rather than about configuration, mid-run, after the bill has started."""
    p = tmp_path / "models.json"
    p.write_text(json.dumps({"judge_model": "claude-old", "judge_provider": "anthropic"}))
    assert J.load_judge_model(p, provider="anthropic") == "claude-old"
    with pytest.raises(J.JudgeError) as exc:
        J.load_judge_model(p, provider="google")
    assert "judge_provider" in str(exc.value)


def test_load_judge_model_is_unchanged_when_no_provider_is_named(tmp_path):
    """The frozen call sites (eval.py) pass no provider; they must behave exactly
    as before."""
    p = tmp_path / "models.json"
    p.write_text(json.dumps({"judge_model": "claude-old", "judge_provider": "google"}))
    assert J.load_judge_model(p) == "claude-old"


# ==========================================================================
# 5. The shipped configuration -- and the guard against flipping it by accident
# ==========================================================================

MODELS = json.loads((REPO_ROOT / "configs" / "models.json").read_text())
G1 = json.loads((REPO_ROOT / "configs" / "g1.json").read_text())


def test_models_json_declares_its_judge_provider():
    assert MODELS["judge_provider"] in J.JUDGE_PROVIDERS


def test_g1_and_models_json_agree_on_the_provider_as_well_as_the_model():
    """They already had to agree on the model. A provider that disagreed would
    run a Gemini id through the Anthropic SDK, or the reverse."""
    assert G1["judge"]["provider"] == MODELS["judge_provider"]
    assert G1["judge"]["model"] == MODELS["judge_model"]


def test_the_configured_judge_is_still_anthropic():
    """A G1 ablation was IN FLIGHT when the Google judge was added. Adding a
    provider is safe mid-run; switching the live default is not -- half a run
    scored by one instrument and half by another is not one experiment.

    This is the guard that says so out loud. Flipping the judge is a deliberate
    edit to configs/models.json + configs/g1.json AND to this assertion, never a
    side effect of some other change.
    """
    assert MODELS["judge_provider"] == "anthropic"
    assert G1["judge"]["provider"] == "anthropic"


def test_the_alternate_judge_is_recorded_so_the_flip_is_not_a_guess():
    """The vetted-but-not-live model is written down. Discovering mid-outage
    that nobody knows which Gemini id has quota is how the last outage went."""
    alt = MODELS["judge_alternates"]
    assert alt["google"] and isinstance(alt["google"], str)
    assert ":" not in alt["google"], "a bare model id, like judge_model"
    assert alt["google"].startswith("gemini-")


def test_the_alternate_is_never_also_an_ablated_model():
    """The reason the Google judge exists is that nothing should score itself or
    share a failure domain with what it grades."""
    frontier = set(MODELS["frontier"])
    frontier_ids = {s.split(":", 1)[1] for s in frontier}
    for provider, model in MODELS["judge_alternates"].items():
        assert model not in frontier_ids
        assert f"{provider}:{model}" not in frontier


def test_flipping_the_provider_would_pick_up_a_matching_model(tmp_path):
    """Proves the flip is mechanical rather than a hope: writing the alternate's
    provider AND model into a models.json produces a consistent config that the
    cross-check accepts."""
    flipped = dict(MODELS)
    flipped["judge_provider"] = "google"
    flipped["judge_model"] = MODELS["judge_alternates"]["google"]
    p = tmp_path / "models.json"
    p.write_text(json.dumps(flipped))
    assert J.load_judge_provider(p) == "google"
    assert J.load_judge_model(p, provider="google") == MODELS["judge_alternates"]["google"]
    with pytest.raises(J.JudgeError):
        J.load_judge_model(p, provider="anthropic")
