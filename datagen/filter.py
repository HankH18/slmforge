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
``evasive_draft``        every checker passed, but the draft DECLINED the task or
                         NEGOTIATED about the rules instead of completing it.
                         See below.

Why "and the draft is an ANSWER, not a negotiation"
---------------------------------------------------
The same argument, one notch further up, and it is what makes the attack block
of :mod:`datagen.attacks` trainable at all. Under FORMAT COMPULSION the brief
demands an output shape the constraint forbids -- a recipe card as a numbered
list, under ``no_lists``. The cheapest draft that passes every checker is not a
good answer; it is a refusal:

    "I can't give you that as a numbered list, so here is a paragraph instead."

Every checker passes (there is no list), it clears MIN_DRAFT_WORDS, and the
filter as originally written would KEEP it. A few hundred such rows teach a
model to decline briefs it can in fact complete -- which would show up at eval
as collapsed task quality on the 147-row corpus while the constraint numbers
looked better than ever. That is a worse failure than the one the attack block
exists to fix, so it is rejected here, BEFORE any GPU spend, and the count is
carried into the run report by :func:`datagen.generate.summarise`.

:func:`evasion_cues` decides it, and it is deliberately a CONJUNCTION rule
rather than a keyword list: a legitimate answer is allowed to mention its own
subject, to use the word "cannot", and to talk about lists, letters or quoting
in the third person. What no legitimate answer does is talk in the first person
about whether it will produce the thing asked for, cite its own brief, or
announce a substitution for the requested form. So a cue fires only when a
DECLINING clause, a compliance narration, a rule citation, a substitution
announcement or an offer to break the rule is present WITHIN ONE SENTENCE
together with the thing it is about. The rules are named in
:data:`EVASION_RULES`, every fired cue is recorded with the span that fired it,
and ``tests/test_datagen.py`` runs the whole detector over every completion of
the shipped ``data/train/v3.jsonl`` to prove it stays silent on 1,815 real,
checker-passing answers.

ORDER MATTERS AND IS DELIBERATE: the evasion test runs LAST, after the checkers
and after the length floor. No draft that was rejected before this rule existed
changes its reason code; only drafts that would previously have been KEPT can be
rejected by it.

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
from typing import (Any, Dict, List, Mapping, NamedTuple, Optional, Sequence,
                    Tuple)

# The repo's ONE sentence segmentation, reused rather than re-implemented. It is
# module-private in constraints.checkers because it is an implementation detail
# of the sentence_initial_letter checker; a second segmentation in datagen would
# be a second answer to "what is a sentence", which this codebase deliberately
# does not have.
from constraints.checkers import sentence_spans

__all__ = ["MIN_DRAFT_WORDS", "MIN_NEGATIVE_WORDS", "MIN_NEGATIVE_SENTENCES",
           "MIN_TASK_OVERLAP_WORDS", "REASON_CODES", "EVASION_RULES", "Decision",
           "EvasionCue", "FluencyReport", "decide", "evasion_cues", "is_evasive",
           "fluency_report", "word_count"]

#: A draft shorter than this is silence with punctuation. See the docstring.
MIN_DRAFT_WORDS = 5

#: Thresholds for "fluent enough to be worth keeping as a DPO negative".
MIN_NEGATIVE_WORDS = 30
MIN_NEGATIVE_SENTENCES = 2
MIN_TASK_OVERLAP_WORDS = 3

#: Every reason code this module can emit. Tests assert the set is exactly this,
#: so a new code cannot appear in a dataset without appearing here first.
REASON_CODES = ("teacher_error", "empty_draft", "degenerate_draft",
                "constraint_violation", "evasive_draft")

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


# ---------------------------------------------------------------- evasion
# The six ways a draft can pass every checker while refusing to do the job.
# Each is a CONJUNCTION inside ONE sentence, because every ingredient on its own
# appears in perfectly good prose: "cannot" (third person), "list", "the word",
# "instead". What no answer to a brief does is put a first-person stance about
# producing the response next to the thing being refused.
#
#: ``rule -> one line on what it catches``. The keys are the vocabulary of the
#: ``evasion`` field in the rejection ledger; tests assert the set is closed.
EVASION_RULES: Dict[str, str] = {
    "decline_to_produce":
        "first person declines to write/give/use the thing that was asked for",
    "compliance_narration":
        "first person reports on its own rule-following instead of answering",
    "rule_talk":
        "the draft cites, quotes or argues with its own brief or requirements",
    "substitution":
        "the draft announces a replacement for the requested form",
    "offer_to_violate":
        "the draft offers to supply the forbidden form on request",
    "model_meta":
        "the draft talks about itself as a model or assistant",
}

#: Contractions expanded before matching, so one pattern covers "I can't" and
#: "I cannot" without an apostrophe alternation in every rule. Curly apostrophes
#: are folded to straight ones first.
_CONTRACTIONS: Tuple[Tuple[str, str], ...] = (
    (r"\bi'm\b", "i am"), (r"\bwe're\b", "we are"),
    (r"\bi've\b", "i have"), (r"\bwe've\b", "we have"),
    (r"\bi'll\b", "i will"), (r"\bwe'll\b", "we will"),
    (r"\bi'd\b", "i would"),
    (r"\bcan't\b", "cannot"), (r"\bcannot\b", "cannot"), (r"\bcan not\b", "cannot"),
    (r"\bwon't\b", "will not"), (r"\bshan't\b", "shall not"),
    (r"\bdon't\b", "do not"), (r"\bdoesn't\b", "does not"),
    (r"\bdidn't\b", "did not"), (r"\bisn't\b", "is not"),
    (r"\baren't\b", "are not"), (r"\bwasn't\b", "was not"),
    (r"\bcouldn't\b", "could not"), (r"\bwouldn't\b", "would not"),
    (r"\bshouldn't\b", "should not"), (r"\bhaven't\b", "have not"),
    (r"\bhasn't\b", "has not"),
)
_CONTRACTION_RES = tuple((re.compile(pattern), value)
                         for pattern, value in _CONTRACTIONS)

#: "I am not going to produce it" in every phrasing that matters. The subject is
#: pinned to first person on purpose: "the display cannot tell you which
#: appliance is drawing power" is an ANSWER, and a rule keyed on bare "cannot"
#: would reject it.
_DECLINE_RE = re.compile(
    r"\b(?:i|we)\s+(?:am\s+|are\s+)?(?:"
    r"cannot|will\s+not|shall\s+not|must\s+not|"
    r"unable\s+to|not\s+able\s+to|not\s+going\s+to|"
    r"must\s+decline|have\s+to\s+decline|decline\s+to|refuse\s+to|"
    r"am\s+afraid|are\s+afraid"
    r")\b")

#: A softly worded refusal. Held to a HIGHER bar than the hard modals above --
#: it counts only against a rule or a demanded form, never against a bare
#: "that" -- because ordinary prose uses it: a real tenant letter measured in
#: runs/ says "I would rather not keep doing that indefinitely", which is an
#: answer, not a negotiation about the brief.
_SOFT_DECLINE_RE = re.compile(
    r"\b(?:i|we)\s+(?:would\s+rather\s+not|prefer\s+not\s+to|"
    r"do\s+not\s+want\s+to|would\s+prefer\s+not\s+to)\b")

#: The same refusal with the first person deleted -- "this cannot be written as
#: a numbered list". Gated on a META object in the same sentence, so "the kiln
#: cannot be fired until the pieces are dry" is untouched.
_IMPERSONAL_DECLINE_RE = re.compile(
    r"\b(?:cannot|will\s+not|is\s+not\s+possible|is\s+impossible|"
    r"there\s+is\s+no\s+way)\b[^.!?\n]{0,40}?\b(?:be\s+)?"
    r"(?:done|written|produced|provided|given|supplied|presented|formatted|"
    r"set\s+out|laid\s+out)\b")

#: What is being declined has to be the ACT OF ANSWERING, not any old verb, or
#: "I cannot promise the boiler man will call" would read as a refusal.
_PRODUCE_RE = re.compile(
    r"\b(?:write|writing|produce|producing|provide|providing|give|giving|"
    r"offer|offering|create|creating|make|making|generate|generating|"
    r"draft|drafting|compose|composing|deliver|delivering|comply|complying|"
    r"complete|completing|fulfil|fulfill|fulfilling|do|doing|help|helping|"
    r"assist|assisting|format|formatting|present|presenting|put|turn|render|"
    r"supply|supplying|use|using|include|including|follow|following)\b")

#: The thing asked for, named as a request rather than as a subject. A bare
#: "it"/"that" is NOT here: "we refuse to make more than it can bake" is a
#: bakery's stall sign, and an object pronoun anywhere in the sentence is not
#: evidence that the sentence is about the request.
_ASK_OBJECT_RE = re.compile(
    r"\b(?:the\s+request|your\s+request|the\s+brief|the\s+task|the\s+ask|"
    r"what\s+you\s+(?:have\s+)?asked|what\s+was\s+asked|the\s+piece|"
    r"the\s+card|the\s+note|the\s+answer|the\s+response|the\s+version|"
    r"as\s+requested|as\s+asked|as\s+you\s+asked)\b")

#: "give you that", "help with that", "do this" -- the deictic as the DIRECT
#: object of the production verb, which is what makes it a reference to the ask.
_DIRECT_ASK_RE = re.compile(
    r"\b(?:give|giving|provide|providing|produce|producing|write|writing|"
    r"do|doing|make|making|supply|supplying|offer|offering|render|help|"
    r"helping|assist|assisting|complete|completing)\s+"
    r"(?:you\s+)?(?:with\s+)?(?:that|this|it|those)\b")

#: The rules themselves, or the forbidden form/token, named as a thing.
#:
#: The rule nouns need a DETERMINER within two words ("your word restrictions",
#: "the brief"), because undetermined ones are ordinary vocabulary: a fiction
#: task measured in runs/ produced "unless you give written, risk-conscious
#: instructions", which is an answer. The form nouns do not, but "quote" as a
#: bare noun is gone -- half a dozen model outputs offer to "get a proper quote"
#: from a fencing contractor, and none of them was talking about punctuation.
_META_OBJECT_RE = re.compile(
    r"(?:\b(?:the|your|these|those|that|this|any)\s+(?:\w+\s+){0,2}?"
    r"(?:brief|briefs|rule|rules|requirement|requirements|constraint|"
    r"constraints|instruction|instructions|restriction|restrictions|"
    r"guideline|guidelines|specification|specifications|spec|stipulation|"
    r"stipulations)\b"
    r"|\b(?:bullet|bullets|bullet\s+points?|numbered\s+list|numbered\s+steps|"
    r"list|lists|table|recipe\s+card|quotation\s+marks?|quote\s+marks?|"
    r"quotation|quoting|"
    r"the\s+word|that\s+word|the\s+letter|that\s+letter|the\s+phrase|"
    r"that\s+phrase|the\s+term|that\s+term|the\s+format|that\s+format|"
    r"the\s+shape|the\s+forbidden|the\s+banned)\b)")

#: "I have avoided the letter e throughout" -- true, compliant, and still a row
#: that teaches the model to narrate its own obedience instead of answering.
_NARRATION_RE = re.compile(
    r"\b(?:i|we)\s+(?:have\s+|had\s+|did\s+|do\s+)?(?:not\s+)?(?:"
    r"avoided|avoid|omitted|omit|left\s+out|leave\s+out|refrained\s+from|"
    r"steered\s+clear\s+of|excluded|exclude|skipped|kept\s+out|kept\s+clear|"
    r"stuck\s+to|adhered\s+to|honoured|honored|respected|observed|"
    r"used|use|included|include|written\s+this|written\s+it|kept\s+this|"
    r"kept\s+it)\b")

#: The paper the model was handed. "the rules" is deliberately NOT in the strong
#: list: club rules, school rules and the rules that prevent a custard curdling
#: are ordinary subject matter in this corpus, and a heading reading "The Rules
#: That Prevent Curdling" is an answer, not a negotiation. Only rules the draft
#: attributes to its instructor ("your rules", "these rules") count.
_RULE_NOUN = (r"(?:(?:the|your|these|those|that|this)\s+(?:brief|"
              r"requirement|requirements|constraint|constraints|instruction|"
              r"instructions|restriction|restrictions|guideline|guidelines|"
              r"specification|specifications|spec|request|prompt)"
              r"|(?:your|these|those)\s+(?:rule|rules))")

#: The draft arguing with, or quoting, the paper it was handed. A modal in the
#: gap disqualifies the match: "the rule should state whether they move to the
#: end" is a chess explainer describing a rule of the GAME.
_RULE_TALK_RE = re.compile(
    _RULE_NOUN + r"\s+(?:(?!should\b|would\b|could\b|might\b|must\b|may\b|"
    r"shall\b|will\b|can\b|cannot\b)\w+\s+){0,3}?"
    r"(?:asks?|says?|states?|requires?|demands?|calls\s+for|wants?|forbids?|"
    r"prohibits?|bans?|rules?\s+out|does\s+not\s+allow|do\s+not\s+allow|"
    r"means?|conflicts?|clash(?:es)?|"
    r"make[s]?\s+(?:it|this|that)\s+impossible)\b")

#: "per the brief", "in line with your requirements" -- a citation of the paper.
_PER_RULE_RE = re.compile(
    r"\b(?:per|as\s+per|in\s+line\s+with|according\s+to|in\s+keeping\s+with|"
    r"respecting|honouring|honoring|to\s+satisfy|to\s+respect|to\s+honour|"
    r"to\s+honor|because\s+of|given)\s+" + _RULE_NOUN + r"\b")

#: "instead of a numbered list", "as prose instead" -- announcing the swap.
_SUBSTITUTION_RE = re.compile(
    r"\b(?:instead\s+of|rather\s+than|in\s+place\s+of|as\s+opposed\s+to)\s+"
    r"(?:a\s+|an\s+|the\s+|any\s+|using\s+|writing\s+|giving\s+|providing\s+)*"
    r"(?:numbered\s+)?(?:list|lists|bullet|bullets|bullet\s+points?|"
    r"numbered\s+steps|table|recipe\s+card|quotation\s+marks?|quote|quotes|"
    r"the\s+requested|that\s+format|the\s+format|the\s+word|the\s+letter|"
    r"the\s+phrase)\b")
_PROSE_INSTEAD_RE = re.compile(
    r"\b(?:prose|paragraph|paragraphs|sentences|continuous\s+prose)\b"
    r"[^.!?\n]{0,24}\binstead\b")

#: "I can give you the bullet points if you would rather have them."
_CAPABILITY_RE = re.compile(
    r"\b(?:i|we)\s+(?:can|could|will|would|am\s+happy\s+to|are\s+happy\s+to|"
    r"would\s+be\s+happy\s+to|am\s+glad\s+to)\b")
#: "happy to" is deliberately NOT here even though it is in _CAPABILITY_RE: one
#: phrase satisfying both halves of a conjunction makes the conjunction a single
#: keyword, which is how "I am happy to stay on the list as an emergency
#: fill-in" read as an offer to break a formatting rule.
_CONDITIONAL_RE = re.compile(
    r"\b(?:if\s+you|should\s+you|if\s+that|on\s+request|just\s+ask|"
    r"let\s+me\s+know)\b")
#: Narrower than _META_OBJECT_RE: an offer has to be about the FORM or the
#: TOKEN, and a bare "list" is a rota, a waiting list or a kit list far more
#: often than it is a violation on offer.
_FORM_OBJECT_RE = re.compile(
    r"\b(?:bullet|bullets|bullet\s+points?|numbered\s+list|numbered\s+steps|"
    r"bulleted\s+list|as\s+a\s+list|in\s+a\s+list|list\s+format|table|"
    r"recipe\s+card|quotation\s+marks?|quote\s+marks?|"
    r"the\s+word|the\s+letter|the\s+phrase|the\s+restricted\s+terms?|"
    r"the\s+plain\s+terms?|word\s+list)\b")

_MODEL_META_RE = re.compile(
    r"\bas\s+an?\s+(?:ai|a\.i\.|artificial\s+intelligence|language\s+model|"
    r"large\s+language\s+model|assistant|llm|chatbot|model)\b")


class EvasionCue(NamedTuple):
    """One fired evasion rule and the sentence that fired it."""
    rule: str
    span: str

    def as_dict(self) -> Dict[str, str]:
        return {"rule": self.rule, "span": self.span}


def _normalise_for_evasion(text: str) -> str:
    """Lowercase, straighten apostrophes, expand contractions, collapse space."""
    out = " ".join(str(text or "").replace("’", "'").replace("ʼ", "'")
                   .lower().split())
    for pattern, value in _CONTRACTION_RES:
        out = pattern.sub(value, out)
    return out


def _sentences(text: str) -> List[str]:
    """The repo's one segmentation, with the whole text as a floor.

    A draft the segmenter yields nothing for (no terminal punctuation at all)
    must still be examined: "I cannot give you that as a list" with no full stop
    is exactly the shape being caught.
    """
    parts = [text[start:end].strip() for start, end in sentence_spans(text)]
    parts = [p for p in parts if p]
    return parts or ([text.strip()] if text.strip() else [])


def evasion_cues(draft: Any) -> Tuple[EvasionCue, ...]:
    """Every evasion rule this draft fires, with the sentence that fired it.

    Empty tuple means the draft reads as an ANSWER. See the module docstring for
    why each rule is a conjunction rather than a keyword.
    """
    if not isinstance(draft, str) or not draft.strip():
        return ()
    cues: List[EvasionCue] = []
    for raw in _sentences(draft):
        s = _normalise_for_evasion(raw)
        if not s:
            continue
        fired: List[str] = []
        meta = _META_OBJECT_RE.search(s)
        # What is refused must come AFTER the refusal. Searching the whole
        # sentence let the refusal supply its own evidence -- "I do not want to
        # push him onto a formal warning list" matched the production verb "do"
        # inside "do not want to" -- and it let a refusal of something else
        # entirely borrow a rule word from the far end of the sentence.
        hard = _DECLINE_RE.search(s)
        tail = s[hard.end():] if hard else ""
        soft = _SOFT_DECLINE_RE.search(s)
        soft_tail = s[soft.end():] if soft else ""
        declined = bool(hard) and bool(
            # The deictic branch has to be the refusal's own complement:
            # "I can't give you that" refuses the ask, while "I can't keep
            # doing that indefinitely" is a tenant writing about the rent.
            _DIRECT_ASK_RE.match(tail.lstrip())
            or (_PRODUCE_RE.search(tail)
                and (_ASK_OBJECT_RE.search(tail) or _META_OBJECT_RE.search(tail))))
        softly = bool(soft) and bool(_PRODUCE_RE.search(soft_tail)
                                     and _FORM_OBJECT_RE.search(soft_tail))
        impersonal = bool(_IMPERSONAL_DECLINE_RE.search(s) and meta)
        if declined or softly or impersonal:
            fired.append("decline_to_produce")
        if _NARRATION_RE.search(s) and _META_OBJECT_RE.search(s):
            fired.append("compliance_narration")
        if _RULE_TALK_RE.search(s) or _PER_RULE_RE.search(s):
            fired.append("rule_talk")
        if _SUBSTITUTION_RE.search(s) or _PROSE_INSTEAD_RE.search(s):
            fired.append("substitution")
        if (_CAPABILITY_RE.search(s) and _CONDITIONAL_RE.search(s)
                and _FORM_OBJECT_RE.search(s)):
            fired.append("offer_to_violate")
        if _MODEL_META_RE.search(s):
            fired.append("model_meta")
        cues.extend(EvasionCue(rule, raw.strip()[:200]) for rule in fired)
    return tuple(cues)


def is_evasive(draft: Any) -> bool:
    """True when the draft declines the task or negotiates about the rules."""
    return bool(evasion_cues(draft))


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
    evasion: Tuple[EvasionCue, ...] = ()  # fired rules when reason is evasive_draft

    def as_dict(self) -> Dict[str, Any]:
        return {"keep": self.keep, "reason": self.reason,
                "results": self.results, "failures": self.failures,
                "evasion": [cue.as_dict() for cue in self.evasion]}


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
    # LAST, deliberately: see the module docstring. Everything above keeps the
    # reason code it had before this rule existed, so only a draft that would
    # otherwise have been KEPT can be rejected here.
    cues = evasion_cues(draft)
    if cues:
        return Decision(False, "evasive_draft", list(results), [], cues)
    return Decision(True, None, list(results), [])
