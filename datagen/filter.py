#!/usr/bin/env python3
"""The rejection filter: what ships, what does not, and why -- in reason codes.

The rule is IFF. An example is kept if and only if every constraint passes its
programmatic checker (:func:`harness.run_checkers`, which wraps the pinned
:func:`constraints.checkers.check`) AND the draft is a real draft. Nothing here
re-implements a checker; this module only decides what to do with their verdicts.

Why "and the draft is a real draft" is not scope creep
------------------------------------------------------
Ruling A7: an empty response vacuously passes every checker. At eval time the
task-quality judge catches that, because ``spec_pass`` is "all checkers pass AND
quality >= 3". At TRAINING time there is no judge, so an empty row would teach
the model that silence is compliance -- the exact reward hack the project exists
to avoid training. So empty and whitespace-only drafts are rejected
(``empty_draft``), and so is a draft of fewer than :data:`MIN_DRAFT_WORDS` words
(``degenerate_draft``), which is the same failure one notch up: "Noted." passes
a ban on the letter e too. The threshold is deliberately low -- it is a floor
against degeneracy, not a length preference.

Reason codes (machine-readable, stable):

``teacher_error``        the teacher call raised; there is no draft to judge.
``empty_draft``          the draft is empty or whitespace-only.
``degenerate_draft``     fewer than MIN_DRAFT_WORDS words of real text.
``constraint_violation`` at least one checker failed. The record names every
                         failing constraint id, its type and its evidence.

Candidate DPO negatives (T-5 acceptance 3)
------------------------------------------
A rejected draft is only useful as a preference-pair negative if it is the kind
of mistake worth contrasting against: fluent, on-task, and wrong about ONE rule.
Garbage is not a negative, it is noise, and a DPO pair whose "rejected" side is
obviously broken teaches the model to avoid being broken rather than to hold the
constraint. :func:`fluency_report` decides with four programmatic tests, all
recorded alongside the row so the call is auditable rather than asserted:

1. it violates at most one constraint (a near miss, not a collapse);
2. it is long enough and has at least two sentences;
3. it does not open as a refusal or as model meta-commentary;
4. it is on-task: it shares content words with the task it was given.

Nothing in here calls a model. The whole module is deterministic and offline.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence

# The repo's ONE sentence segmentation, reused rather than re-implemented. It is
# module-private in constraints.checkers because it is an implementation detail
# of the sentence_initial_letter checker; a second segmentation in datagen would
# be a second answer to "what is a sentence", which this codebase deliberately
# does not have.
from constraints.checkers import _sentence_spans as sentence_spans

__all__ = ["MIN_DRAFT_WORDS", "MIN_NEGATIVE_WORDS", "MIN_NEGATIVE_SENTENCES",
           "MIN_TASK_OVERLAP_WORDS", "REASON_CODES", "Decision", "FluencyReport",
           "decide", "fluency_report", "word_count"]

#: A draft shorter than this is silence with punctuation. See the docstring.
MIN_DRAFT_WORDS = 5

#: Thresholds for "fluent enough to be worth keeping as a DPO negative".
MIN_NEGATIVE_WORDS = 30
MIN_NEGATIVE_SENTENCES = 2
MIN_TASK_OVERLAP_WORDS = 3

#: Every reason code this module can emit. Tests assert the set is exactly this,
#: so a new code cannot appear in a dataset without appearing here first.
REASON_CODES = ("teacher_error", "empty_draft", "degenerate_draft",
                "constraint_violation")

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

#: Openings that mean the model talked about the task instead of doing it.
_META_OPENINGS = (
    "i'm sorry", "i am sorry", "sorry,", "i cannot", "i can't", "i can not",
    "i am unable", "i'm unable", "as an ai", "as a language model",
    "i apologize", "i apologise", "unfortunately, i",
    "here is my attempt", "sure! here", "sure, here", "certainly! here",
)

#: Words too common to be evidence that a draft is about a particular task.
_STOPWORDS = frozenset("""
about above after again against all also and any are because been before being
below between both but can cannot could did does doing down during each few for
from further had has have having here how into its itself just more most not now
off once only other our out over own same should some such than that the their
them then there these they this those through too under until very was were what
when where which while who whom why will with would you your yours
write writing written draft drafts explain explaining note notes word words
short brief piece paragraph paragraphs sentence sentences somebody someone
""".split())


def word_count(text: str) -> int:
    """Alphabetic word tokens. Digits and punctuation do not make a draft."""
    return len(_WORD_RE.findall(text or ""))


def _content_words(text: str) -> set:
    return {w.lower() for w in _WORD_RE.findall(text or "")
            if len(w) >= 4 and w.lower() not in _STOPWORDS}


def _sentence_count(text: str) -> int:
    return len([1 for start, end in sentence_spans(text) if text[start:end].strip()])


class FluencyReport(NamedTuple):
    """Why a rejected draft is, or is not, a usable DPO negative."""
    fluent: bool
    words: int
    sentences: int
    violated: int
    task_overlap: int
    meta_opening: bool
    reasons: tuple

    def as_dict(self) -> Dict[str, Any]:
        return {"fluent": self.fluent, "words": self.words,
                "sentences": self.sentences, "violated": self.violated,
                "task_overlap": self.task_overlap,
                "meta_opening": self.meta_opening,
                "reasons": list(self.reasons)}


def fluency_report(draft: str, task: str,
                   checker_results: Sequence[Mapping[str, Any]]) -> FluencyReport:
    """Is this rejected draft the valuable kind of wrong? See the docstring."""
    text = (draft or "").strip()
    words = word_count(text)
    sentences = _sentence_count(text)
    violated = sum(1 for r in checker_results if not r["pass"])
    overlap = len(_content_words(text) & _content_words(task))
    head = text[:120].lower().lstrip("*# \n\t")
    meta = any(head.startswith(opening) for opening in _META_OPENINGS)

    reasons: List[str] = []
    if violated == 0:
        reasons.append("no_violation")          # not a negative at all
    if violated > 1:
        reasons.append("multiple_violations")
    if words < MIN_NEGATIVE_WORDS:
        reasons.append("too_short")
    if sentences < MIN_NEGATIVE_SENTENCES:
        reasons.append("too_few_sentences")
    if meta:
        reasons.append("meta_opening")
    if overlap < MIN_TASK_OVERLAP_WORDS:
        reasons.append("off_task")
    return FluencyReport(not reasons, words, sentences, violated, overlap,
                         meta, tuple(reasons))


class Decision(NamedTuple):
    """The verdict on one teacher draft."""
    keep: bool
    reason: Optional[str]                 # a REASON_CODES member when keep is False
    results: List[Dict[str, Any]]         # the checker verdicts, in scenario order
    failures: List[Dict[str, Any]]        # the failing subset, with type + evidence

    def as_dict(self) -> Dict[str, Any]:
        return {"keep": self.keep, "reason": self.reason,
                "results": self.results, "failures": self.failures}


def decide(draft: Any,
           scenario: Mapping[str, Any],
           lookup: Mapping[str, Mapping[str, Any]],
           run_checkers_fn=None) -> Decision:
    """Keep or reject one draft, with a machine-readable reason.

    ``run_checkers_fn`` defaults to :func:`harness.run_checkers`; it is a
    parameter only so a test can prove the filter reports what the checkers say
    rather than deciding for itself. Production always uses the real one.
    """
    if run_checkers_fn is None:
        from harness import run_checkers as run_checkers_fn  # noqa: N806

    if not isinstance(draft, str) or not draft.strip():
        # Deliberately BEFORE the checkers: ruling A7 makes an empty draft pass
        # every one of them, so asking them first would report a clean sweep.
        return Decision(False, "empty_draft", [], [])

    results = run_checkers_fn(scenario, lookup, draft)
    failures = [{"constraint_id": r["constraint_id"],
                 "type": lookup[r["constraint_id"]]["type"],
                 "evidence": r["evidence"]}
                for r in results if not r["pass"]]
    if failures:
        return Decision(False, "constraint_violation", list(results), failures)
    if word_count(draft) < MIN_DRAFT_WORDS:
        return Decision(False, "degenerate_draft", list(results), [])
    return Decision(True, None, list(results), [])
