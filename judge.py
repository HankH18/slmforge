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
    "make_judge",
    "load_judge_model",
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

def load_judge_model(path: Optional[Any] = None) -> str:
    """Read ``judge_model`` out of configs/models.json AT RUNTIME.

    Never hardcode a Claude id: the pin lives in the config so re-pinning is a
    config edit plus a re-run, not a code change. A missing file or missing key
    raises -- defaulting to some id would silently score a run against a model
    nobody chose.
    """
    p = Path(path) if path is not None else MODELS_JSON
    if not p.exists():
        raise JudgeError(
            f"{p} does not exist; the judge model is pinned there and is not "
            f"hardcoded. Run scripts/ping_backends.py or pass model= explicitly.")
    try:
        cfg = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        raise JudgeError(f"{p} is not valid JSON: {exc}") from exc
    model = cfg.get("judge_model")
    if not isinstance(model, str) or not model.strip():
        raise JudgeError(f"{p} has no usable 'judge_model' key (got {model!r})")
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


class AnthropicJudge:
    """Anthropic-backed task-quality judge. temperature 0, runtime-pinned model.

    ``client`` is injectable purely so the parse/retry/temperature-fallback
    logic is testable without a network call; production leaves it ``None`` and
    gets ``anthropic.Anthropic()``.
    """

    def __init__(self, model: Optional[str] = None, client: Any = None,
                 tries: int = 3, sleep: Callable[[float], Any] = time.sleep,
                 max_tokens: int = 300):
        self.model = model if model is not None else load_judge_model()
        if client is None:
            import anthropic  # lazy: mock runs and tests never need the SDK
            client = anthropic.Anthropic()
        self.client = client
        self.tries = int(tries)
        if self.tries < 1:
            raise ValueError("tries must be >= 1")
        self._sleep = sleep
        self.max_tokens = int(max_tokens)
        #: None = never probed; True = the model 400'd on temperature and the
        #: judge is now running without it. Falsey means "keep sending it".
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


def _is_temperature_400(exc: Exception) -> bool:
    """Is this the 'this model no longer accepts temperature' 400?"""
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if status != 400:
        return False
    blob = f"{exc}".lower()
    return any(marker in blob for marker in _TEMPERATURE_400_MARKERS)


def make_judge(spec: str = "anthropic", model: Optional[str] = None, **kwargs) -> Any:
    """``"anthropic"`` | ``"mock:<1-5>"`` -> a judge callable. For CLI wiring."""
    if spec == "anthropic":
        return AnthropicJudge(model=model, **kwargs)
    if spec.startswith("mock:"):
        raw = spec.split(":", 1)[1]
        try:
            score = int(raw)
        except ValueError as exc:
            raise ValueError(f"mock judge spec needs an integer score, got {raw!r}") from exc
        return MockJudge(score)
    raise ValueError(f"unknown judge spec {spec!r} (expected 'anthropic' or 'mock:<1-5>')")
