#!/usr/bin/env python3
"""Task-quality judge -- the anti-reward-hacking gate, and NOTHING else.

Binding contract (docs/PIVOT_DESIGN.md, "Decisions & rationale")::

    Judge scores task-quality only; never constraint pass/fail.

That rule is enforced *structurally* here, not by good intentions: a judge is
any callable with the signature ::

    judge_fn(task: str, output: str) -> (score: int, reasoning: str)

There is no third parameter. A judge cannot be handed the constraint list, the
constraint ids, the checker verdicts or the rendered constraint text, because
the call site has nowhere to put them. If somebody later "improves" the judge
into a constraint scorer they have to widen this signature first, and every
caller and every test in tests/test_harness.py breaks loudly.

Why the judge exists at all
---------------------------
The checkers are programmatic and un-gameable, but they are all *negative*
constraints: an empty string, a single space, or a paragraph about a completely
different subject passes every one of them vacuously (constraints.checkers,
ruling A7). Without a task-quality gate the degenerate winning strategy is to
emit nothing. The judge closes exactly that hole and no other.

Failure policy, learned the hard way
------------------------------------
The abstainer harness (eval.py:AnthropicJudge) converts any API error or
unparseable verdict into ``return False`` -- a "conservative fail". That looks
identical in the transcript to a real measured failure, and on the night a
newer Claude id started rejecting ``temperature=0`` with a 400 it silently
turned an entire paid run into a wall of fabricated FAILs.

This module never does that. Every error path raises :class:`JudgeError`.
Nothing here invents a score. Two specific hazards are handled explicitly:

* **temperature 400.** ``temperature`` is a valid parameter on Claude Opus 4.6
  and Sonnet 4.6 but was REMOVED on Fable 5 / Opus 5 / Opus 4.7 / Opus 4.8 /
  Sonnet 5, which reject it with a 400. Since ``judge_model`` is read from
  configs/models.json at runtime and can be re-pinned to any of those, the
  judge probes once, drops the parameter if the model rejects it, records that
  it happened on :attr:`AnthropicJudge.temperature_dropped`, warns on stderr,
  and stamps :data:`TEMPERATURE_DROPPED_MARKER` into the reasoning of every
  affected row so the transcript itself shows the degraded determinism.
* **unparseable verdict.** Retried, then raised. Never scored.

Two providers, one rubric
-------------------------
:class:`AnthropicJudge` and :class:`GoogleJudge` are interchangeable by
construction: both render the same :data:`JUDGE_PROMPT` bytes and both inherit
their entire failure policy from :class:`_RetryingJudge`, so ``rubric_hash()``
cannot differ between them and neither can what happens when a call fails. Only
the transport and the provider-specific hazards differ. If the prompts ever
diverge the two judges are no longer the same instrument and every
cross-provider comparison is void, which is why tests/test_judge_google.py
asserts the bytes each one SENDS are identical rather than trusting that both
read the same constant.

The second provider exists because on 2026-08-20 the judge and a graded frontier
model shared one Anthropic account, that account hit its monthly spend cap
mid-run, and both died together -- quarantining 147/147 rows of one cell. Which
provider a run uses is configuration (``judge_provider`` in configs/models.json,
``judge.provider`` in configs/g1.json), never a code edit.

Google-specific hazards handled here
------------------------------------
* **thinking tokens share the output budget.** Gemini 3 spends ~140-155 thought
  tokens on a ~25-token verdict, so :class:`GoogleJudge` sizes ``max_tokens``
  for both. A truncated body is a ``MAX_TOKENS`` finish reason with no text,
  and it raises.
* **safety refusal / empty candidate.** Raises. A refusal is not a 1 -- the
  judge never read the response, so there is no verdict to record.
* **the 400 shape differs.** google-genai's ``ClientError`` carries ``code``,
  not ``status_code``; :func:`_status_of` reads both.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

__all__ = [
    "JudgeError",
    "MockJudge",
    "AnthropicJudge",
    "GoogleJudge",
    "make_judge",
    "load_judge_model",
    "load_judge_provider",
    "JUDGE_PROVIDERS",
    "DEFAULT_JUDGE_PROVIDER",
    "parse_judge_response",
    "rubric_hash",
    "JUDGE_PROMPT",
    "JUDGE_RUBRIC_VERSION",
    "MIN_SCORE",
    "MAX_SCORE",
    "TEMPERATURE_DROPPED_MARKER",
]

REPO_ROOT = Path(__file__).resolve().parent
MODELS_JSON = REPO_ROOT / "configs" / "models.json"

#: Legal score range. Anything outside it is a broken judge, not a low score.
MIN_SCORE = 1
MAX_SCORE = 5

#: Bumped whenever JUDGE_PROMPT changes. Cross-run comparisons are only valid
#: within one version; :func:`rubric_hash` pins the exact bytes.
JUDGE_RUBRIC_VERSION = "quality-1.0"

#: Stamped into ``quality_reasoning`` when the pinned model rejected
#: ``temperature`` and the judge had to run without it. Greppable on purpose.
TEMPERATURE_DROPPED_MARKER = "[judge ran WITHOUT temperature=0: model rejected it]"

#: Substrings that identify a 400 as "this model does not take temperature"
#: rather than some other bad request.
_TEMPERATURE_400_MARKERS = ("temperature", "sampling")

#: Network-backed providers this module can dispatch. Order is not significance.
JUDGE_PROVIDERS = ("anthropic", "google")

#: What ``judge_provider`` means when configs/models.json does not carry the key.
#: models.json predates the second provider; a checkout that has not been
#: migrated must keep scoring with the judge its numbers were scored with, not
#: fail and not silently move to another family.
DEFAULT_JUDGE_PROVIDER = "anthropic"


class JudgeError(RuntimeError):
    """The judge could not produce a real score.

    Raised on API failure after retries, on an unparseable response, and on a
    response whose score is out of range. NEVER caught-and-converted into a
    score inside this module. Callers that need a run to survive a flaky judge
    must opt in explicitly (harness.score_scenario's ``on_judge_error``), and
    that path quarantines the row instead of scoring it.
    """


# ==========================================================================
# The rubric
# ==========================================================================
# Deliberately says nothing about constraints, banned words, letters, lists,
# punctuation or formatting. It cannot: it never sees them. The "unusual
# wording is not a defect" clause exists because a constraint-obeying response
# often reads oddly (no letter 'e', no commas), and penalising that would turn
# the judge into a covert constraint scorer -- the exact thing R4 forbids.

JUDGE_PROMPT = """You are grading ONE thing: does the response genuinely address the task it was given?

TASK GIVEN TO THE WRITER:
{task}

RESPONSE TO GRADE:
{output}

Grade on this 1-5 scale:

5 - Fully addresses the task. Every part of the request is answered with real,
    on-topic substance.
4 - Addresses the task. One minor part is thin or missing, but a reader who
    asked for this would be satisfied.
3 - Genuinely responsive but incomplete or shallow. It is about the right
    subject and makes a real attempt at the request.
2 - Barely responsive. Mostly evasive, mostly off-topic, or answers a
    different question than the one asked.
1 - Does not address the task at all. Empty, whitespace, gibberish, a refusal,
    a restatement of the task, or writing about an unrelated subject.

How to grade:
- Judge CONTENT ONLY: did it do what was asked?
- Unusual wording, stilted phrasing, odd vocabulary, missing punctuation or an
  awkward style are NOT defects. Do not lower the score for how it is written,
  only for whether it does the job.
- An empty or near-empty response is a 1. So is text about a different subject,
  however well written.
- Length alone is not quality. A short response that does the job scores high.

Reply in EXACTLY this format and nothing else:
SCORE: <a single digit 1-5>
REASON: <one sentence>"""


def rubric_hash() -> str:
    """Stable hash of the rubric bytes + version. Pin it in a test."""
    blob = (JUDGE_RUBRIC_VERSION + "\x00" + JUDGE_PROMPT).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


# ==========================================================================
# Parsing
# ==========================================================================

_SCORE_RE = re.compile(r"^\s*SCORE\s*:\s*(\d+)\s*$", re.IGNORECASE | re.MULTILINE)
_REASON_RE = re.compile(r"^\s*REASON\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE | re.DOTALL)


def parse_judge_response(text: Any) -> Tuple[int, str]:
    """``"SCORE: 4\\nREASON: ..."`` -> ``(4, "...")``.

    Raises :class:`JudgeError` on anything it cannot read as a real verdict --
    including an in-format score outside 1-5, which means the judge ignored the
    scale and its number means nothing.
    """
    if not isinstance(text, str):
        raise JudgeError(f"judge response is not text: {type(text).__name__}")
    m = _SCORE_RE.search(text)
    if m is None:
        raise JudgeError(f"no parseable 'SCORE: <n>' line in judge response: {text[:200]!r}")
    score = int(m.group(1))
    if not (MIN_SCORE <= score <= MAX_SCORE):
        raise JudgeError(
            f"judge returned score {score}, outside the {MIN_SCORE}-{MAX_SCORE} scale: "
            f"{text[:200]!r}")
    r = _REASON_RE.search(text)
    reasoning = " ".join(r.group(1).split()) if r else ""
    if not reasoning:
        raise JudgeError(f"judge response has no REASON line: {text[:200]!r}")
    return score, reasoning


# ==========================================================================
# Runtime model pin
# ==========================================================================

def _load_models_cfg(path: Optional[Any] = None) -> Tuple[Path, dict]:
    p = Path(path) if path is not None else MODELS_JSON
    if not p.exists():
        raise JudgeError(
            f"{p} does not exist; the judge model is pinned there and is not "
            f"hardcoded. Run scripts/ping_backends.py or pass model= explicitly.")
    try:
        cfg = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        raise JudgeError(f"{p} is not valid JSON: {exc}") from exc
    if not isinstance(cfg, dict):
        raise JudgeError(f"{p} is not a JSON object")
    return p, cfg


def load_judge_provider(path: Optional[Any] = None) -> str:
    """Read ``judge_provider`` out of configs/models.json AT RUNTIME.

    The key is optional and defaults to :data:`DEFAULT_JUDGE_PROVIDER`, because
    every models.json written before the second provider existed lacks it and
    those runs were scored by Anthropic. An unknown value raises rather than
    falling back: dispatching a judge nobody chose is the failure this whole
    module is built to refuse.
    """
    p, cfg = _load_models_cfg(path)
    provider = cfg.get("judge_provider", DEFAULT_JUDGE_PROVIDER)
    if not isinstance(provider, str) or provider.strip() not in JUDGE_PROVIDERS:
        raise JudgeError(
            f"{p} has judge_provider={provider!r}; expected one of {list(JUDGE_PROVIDERS)}")
    return provider.strip()


def load_judge_model(path: Optional[Any] = None,
                     provider: Optional[str] = None) -> str:
    """Read ``judge_model`` out of configs/models.json AT RUNTIME.

    Never hardcode a Claude id: the pin lives in the config so re-pinning is a
    config edit plus a re-run, not a code change. A missing file or missing key
    raises -- defaulting to some id would silently score a run against a model
    nobody chose.

    ``provider``, when given, is cross-checked against ``judge_provider``. Two
    providers share one ``judge_model`` pin, so without this check a
    ``GoogleJudge`` built with no explicit model would cheerfully ask Gemini for
    a ``claude-*`` id (or the reverse) and fail deep inside the SDK with an
    error about model names rather than about configuration.
    """
    p, cfg = _load_models_cfg(path)
    model = cfg.get("judge_model")
    if not isinstance(model, str) or not model.strip():
        raise JudgeError(f"{p} has no usable 'judge_model' key (got {model!r})")
    if provider is not None:
        configured = load_judge_provider(p)
        if provider != configured:
            raise JudgeError(
                f"{p} pins judge_model {model.strip()!r} for judge_provider "
                f"{configured!r}, but a {provider!r} judge asked for it. Set "
                f"'judge_provider' and 'judge_model' together, or pass model= "
                f"explicitly.")
    return model.strip()


# ==========================================================================
# Judges
# ==========================================================================

class MockJudge:
    """Fixed-score judge for network-free tests and dry runs.

    Records every call on :attr:`calls` as ``(task, output)`` so a test can
    prove what the judge was and was not shown.
    """

    def __init__(self, score: int = MAX_SCORE, reasoning: Optional[str] = None):
        if not isinstance(score, int) or isinstance(score, bool):
            raise ValueError(f"MockJudge score must be an int, got {score!r}")
        if not (MIN_SCORE <= score <= MAX_SCORE):
            raise ValueError(
                f"MockJudge score {score} is outside the {MIN_SCORE}-{MAX_SCORE} scale")
        self.score = score
        self.reasoning = reasoning if reasoning is not None else f"mock judge: fixed {score}"
        self.calls: list = []

    def __call__(self, task: str, output: str) -> Tuple[int, str]:
        self.calls.append((task, output))
        return self.score, self.reasoning


class _RetryingJudge:
    """Everything about the FAILURE POLICY, shared by every network judge.

    Provider subclasses supply one method, :meth:`_create`, which turns a
    rendered rubric prompt into raw text. Retry, the temperature-400 fallback,
    the transcript stamp and -- above all -- the rule that an exhausted judge
    RAISES instead of returning a number live here, once, so that adding a
    provider cannot quietly add a second, laxer failure policy alongside the
    first. Two judges that fail differently are two different instruments.
    """

    #: set by every subclass's ``__init__``; declared for readers, not enforced.
    model: str
    tries: int
    max_tokens: int
    #: None = never probed; True = the model 400'd on temperature and the judge
    #: is now running without it. Falsey means "keep sending it".
    temperature_dropped: Optional[bool]

    def _create(self, prompt: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def _note_temperature_dropped(self) -> None:
        self.temperature_dropped = True
        msg = (f"judge model {self.model!r} rejected temperature=0 with a 400; "
               f"re-running WITHOUT temperature. Scores are no longer greedy-decoded. "
               f"Every affected transcript row is stamped {TEMPERATURE_DROPPED_MARKER!r}.")
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        print(f"WARNING: {msg}", file=sys.stderr, flush=True)

    # -- the judge contract ------------------------------------------------
    def __call__(self, task: str, output: str) -> Tuple[int, str]:
        prompt = JUDGE_PROMPT.format(task=task, output=output)
        errors = []
        for attempt in range(self.tries):
            try:
                score, reasoning = parse_judge_response(self._create(prompt))
            except Exception as exc:  # noqa: BLE001 -- accumulated, then raised
                errors.append(f"attempt {attempt + 1}: {type(exc).__name__}: {exc}")
                if attempt < self.tries - 1:
                    self._sleep(2 ** attempt)
                continue
            if self.temperature_dropped:
                reasoning = f"{reasoning} {TEMPERATURE_DROPPED_MARKER}"
            return score, reasoning
        raise JudgeError(
            f"judge failed {self.tries}/{self.tries} attempts on model {self.model!r}; "
            f"NO score produced (deliberately: a fabricated score is worse than a "
            f"missing one). Errors: " + " | ".join(errors))


class AnthropicJudge(_RetryingJudge):
    """Anthropic-backed task-quality judge. temperature 0, runtime-pinned model.

    ``client`` is injectable purely so the parse/retry/temperature-fallback
    logic is testable without a network call; production leaves it ``None`` and
    gets ``anthropic.Anthropic()``.
    """

    def __init__(self, model: Optional[str] = None, client: Any = None,
                 tries: int = 3, sleep: Callable[[float], Any] = time.sleep,
                 max_tokens: int = 300):
        self.model = model if model is not None else load_judge_model(provider="anthropic")
        if client is None:
            import anthropic  # lazy: mock runs and tests never need the SDK
            client = anthropic.Anthropic()
        self.client = client
        self.tries = int(tries)
        if self.tries < 1:
            raise ValueError("tries must be >= 1")
        self._sleep = sleep
        self.max_tokens = int(max_tokens)
        self.temperature_dropped: Optional[bool] = None

    # -- transport ---------------------------------------------------------
    def _create(self, prompt: str) -> str:
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if not self.temperature_dropped:
            kwargs["temperature"] = 0
        try:
            resp = self.client.messages.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 -- re-raised or narrowed below
            if self.temperature_dropped or not _is_temperature_400(exc):
                raise
            self._note_temperature_dropped()
            kwargs.pop("temperature", None)
            resp = self.client.messages.create(**kwargs)
        return _text_of(resp)


class GoogleJudge(_RetryingJudge):
    """Google-backed task-quality judge. Identical rubric bytes, identical
    contract, identical failure policy -- only the transport differs.

    WHY A SECOND PROVIDER EXISTS (2026-08-20). Family B moved from Google to
    Anthropic, which put the judge and a graded frontier model on ONE account.
    That account hit its monthly spend cap mid-run and took out the judge and
    the model it was grading together, quarantining 147/147 rows of one cell.
    The fragility is structural, not bad luck: a judge sharing an account with a
    model it grades is one billing event away from ending a run. A Google judge
    also restores the property the project relaxed under duress -- three
    DISTINCT families, openai and anthropic ablated, google judging, nothing
    scoring itself and no two sharing a failure domain.

    ``client`` is injectable for the same reason as :class:`AnthropicJudge`:
    every branch below is reachable in tests without a network call. Requests
    are built from plain dicts rather than ``google.genai.types`` objects so
    that neither this module nor its tests import the SDK to construct one.

    ``max_tokens`` defaults far above the Anthropic judge's 300 on purpose.
    Gemini 3 models spend THINKING tokens out of the SAME output budget, and how
    much they think is not bounded by how short the answer is. Measured
    2026-08-20 over real run outputs: gemini-3.6-flash spends ~140-155 thought
    tokens on a ~25-token verdict, but gemini-3-flash-preview was observed
    spending 982 and truncating against a 1024 budget -- returning a body whose
    ``SCORE:`` line PARSED while the reason was cut off mid-word. The default
    therefore clears the worst observed thinker several times over. It costs
    nothing to set high: ``max_output_tokens`` is a cap, not a purchase.
    """

    def __init__(self, model: Optional[str] = None, client: Any = None,
                 tries: int = 3, sleep: Callable[[float], Any] = time.sleep,
                 max_tokens: int = 4096):
        self.model = model if model is not None else load_judge_model(provider="google")
        if client is None:
            from google import genai  # lazy: mock runs and tests never need the SDK
            client = genai.Client()
        self.client = client
        self.tries = int(tries)
        if self.tries < 1:
            raise ValueError("tries must be >= 1")
        self._sleep = sleep
        self.max_tokens = int(max_tokens)
        self.temperature_dropped: Optional[bool] = None

    # -- transport ---------------------------------------------------------
    def _create(self, prompt: str) -> str:
        contents = [{"role": "user", "parts": [{"text": prompt}]}]
        config = {"max_output_tokens": self.max_tokens}
        if not self.temperature_dropped:
            config["temperature"] = 0
        try:
            resp = self.client.models.generate_content(
                model=self.model, contents=contents, config=dict(config))
        except Exception as exc:  # noqa: BLE001 -- re-raised or narrowed below
            if self.temperature_dropped or not _is_temperature_400(exc):
                raise
            self._note_temperature_dropped()
            config.pop("temperature", None)
            resp = self.client.models.generate_content(
                model=self.model, contents=contents, config=dict(config))
        return _google_text_of(resp)


def _text_of(resp: Any) -> str:
    """Concatenate the text blocks of a Messages API response."""
    content = getattr(resp, "content", None)
    if content is None:
        raise JudgeError(f"judge response object has no .content: {resp!r}")
    parts = []
    for block in content:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", ""))
    if not parts:
        raise JudgeError(
            f"judge response carried no text block (stop_reason="
            f"{getattr(resp, 'stop_reason', None)!r})")
    return "".join(parts).strip()


#: finish_reason values that mean "the model finished saying what it meant to".
#: Everything else -- SAFETY, RECITATION, PROHIBITED_CONTENT, MAX_TOKENS,
#: BLOCKLIST -- is a non-answer, and a non-answer is not a low score.
_GOOGLE_OK_FINISH = (None, "STOP", "FINISH_REASON_UNSPECIFIED")


def _enum_name(value: Any) -> Optional[str]:
    """``FinishReason.SAFETY`` -> ``"SAFETY"``; passes plain strings through."""
    if value is None:
        return None
    return getattr(value, "name", None) or str(value).rsplit(".", 1)[-1]


def _google_text_of(resp: Any) -> str:
    """Text of a google-genai ``GenerateContentResponse``, or raise.

    A safety refusal, a recitation block, a truncated body and an empty
    candidate list all raise here rather than returning a string that would
    parse as a low score. That is the whole point: the judge never READ the
    response in those cases, so there is no verdict to record, and a 1 stamped
    in place of a missing one is a fabricated measurement. The caller's retry
    loop gets its attempts and then :class:`JudgeError` propagates, which
    quarantines the row.
    """
    block = _enum_name(getattr(getattr(resp, "prompt_feedback", None), "block_reason", None))
    if block is not None:
        raise JudgeError(
            f"google judge: the rubric prompt was blocked before generation "
            f"(block_reason={block}); NO verdict was produced")
    candidates = getattr(resp, "candidates", None)
    if not candidates:
        raise JudgeError(
            f"google judge: response carried no candidates (prompt_feedback="
            f"{getattr(resp, 'prompt_feedback', None)!r})")
    cand = candidates[0]
    finish = _enum_name(getattr(cand, "finish_reason", None))
    if finish not in _GOOGLE_OK_FINISH:
        raise JudgeError(
            f"google judge: generation stopped with finish_reason={finish}, not STOP; "
            f"NO verdict was produced")
    parts = getattr(getattr(cand, "content", None), "parts", None) or []
    chunks = []
    for part in parts:
        if getattr(part, "thought", None):
            continue  # a thought is reasoning ABOUT the verdict, not the verdict
        text = getattr(part, "text", None)
        if text:
            chunks.append(text)
    if not chunks:
        raise JudgeError(
            f"google judge: response carried no text part "
            f"(finish_reason={finish}); NO verdict was produced")
    return "".join(chunks).strip()


def _status_of(exc: Exception) -> Optional[int]:
    """HTTP status carried by a provider exception, whatever it calls the field.

    anthropic and openai expose ``status_code``; ``google.genai.errors.ClientError``
    exposes ``code`` and leaves ``status_code`` unset (measured 2026-08-20).
    ``status_code`` is consulted first so the Anthropic path resolves exactly as
    it did before ``code`` was ever looked at.
    """
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return getattr(getattr(exc, "response", None), "status_code", None)


def _is_temperature_400(exc: Exception) -> bool:
    """Is this the 'this model no longer accepts temperature' 400?"""
    if _status_of(exc) != 400:
        return False
    blob = f"{exc}".lower()
    return any(marker in blob for marker in _TEMPERATURE_400_MARKERS)


#: provider name -> judge class. make_judge dispatches through this, so a new
#: provider is one entry, not a new branch in every caller.
_JUDGE_CLASSES = {"anthropic": AnthropicJudge, "google": GoogleJudge}

def make_judge(spec: str = "anthropic", model: Optional[str] = None, **kwargs) -> Any:
    """``"anthropic"`` | ``"google"`` | ``"mock:<1-5>"`` -> a judge callable.

    Both network providers render the SAME :data:`JUDGE_PROMPT` bytes and share
    :class:`_RetryingJudge`'s failure policy, so which one a run used changes the
    transport and nothing about the rubric. ``rubric_hash()`` is identical across
    them by construction; tests/test_judge_google.py asserts the sent bytes match.
    """
    cls = _JUDGE_CLASSES.get(spec)
    if cls is not None:
        return cls(model=model, **kwargs)
    if spec.startswith("mock:"):
        raw = spec.split(":", 1)[1]
        try:
            score = int(raw)
        except ValueError as exc:
            raise ValueError(f"mock judge spec needs an integer score, got {raw!r}") from exc
        return MockJudge(score)
    raise ValueError(
        f"unknown judge spec {spec!r} (expected one of "
        f"{[*JUDGE_PROVIDERS, 'mock:<1-5>']})")
