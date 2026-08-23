#!/usr/bin/env python3
"""Attack shapes the training corpus has never carried: FORMAT and SUBJECT compulsion.

WHY THIS MODULE EXISTS
======================
Every adversarial row in the shipped training corpus is the same attack. The
whole of :data:`datagen.task_pool.NUDGES` -- all ten entries -- is one family:
a sentence granting PERMISSION to relax a rule ("treat the word restrictions as
a nice-to-have", "two out of three is the usual standard around here"). A model
trained on that learns to ignore one sentence of polite permission. It learns
nothing about the two attacks that do not ask permission at all:

FORMAT COMPULSION
    The brief demands an OUTPUT SHAPE that cannot exist without the banned
    element. A recipe card is a numbered list; there is no way to produce one
    under ``no_lists``. Nothing in the brief says the rule is optional -- the
    rule and the requested artefact simply cannot both be satisfied, and the
    model has to notice that and refuse the shape.

SUBJECT COMPULSION
    The brief's TOPIC is the banned token itself: explain the word, the phrase,
    the letter, while that word, phrase or letter is banned. Again no permission
    clause. The pull toward the violation is the subject matter.

EROSION INTO FORMAT
    Comply first, then a follow-up whose form requires the violation. Modelled
    here as :data:`EROSION_RECAP`: a SINGLE-TURN surrogate in which the brief
    establishes the prior compliant answer inline and then makes the demand. It
    is single-turn on purpose -- the trainer accepts ``[user, assistant]`` and
    nothing else, so a genuine three-message transcript cannot become a training
    row. The surrogate keeps the attack's structure (an established compliant
    baseline, then a demand whose form breaks it) inside one user message.

WHAT IS PROVED HERE, AND WHAT IS NOT
====================================
* **Every emitted constraint set is constructively satisfiable.** Proved by
  :func:`scenarios.reachability.build_witness`, which returns a real paragraph
  the ground-truth checkers pass against every constraint of the row. A row
  whose witness cannot be built is QUARANTINED, never emitted. An impossible row
  scores zero for every model forever and fakes a ceiling; see the note at the
  top of :mod:`scenarios.reachability`.
* **Every emitted row renders through the one renderer.** ``prompt`` is
  :func:`scenarios.render.render_user_message` output, because training rows are
  byte-checked against exactly that renderer by
  ``training/modal_app.py:verify_prompt_contract``.
* **No held-out constraint type is ever emitted.** The allowed set is resolved
  at RUNTIME from :func:`constraints.registry.types_by_split`, never from a
  literal list here, and :func:`_finalise` raises if a held-out type reaches an
  emitted row.
* **Every row's shape label is carried by its TEXT.** :func:`classify` decides
  the shape from the row alone, using structural evidence (which decidable
  conflicts the task exhibits, whether the target's own checker rejects the
  task's topic span, whether a prior compliant answer is established inline).
  A ``format_compulsion`` row cannot satisfy the ``subject_compulsion`` guard
  and vice versa -- see :func:`shape_evidence`.

  Stated rather than hidden: the subject guard's checker component is WEAK on
  its own. A topic span is built as ``the <kind> <token>``, so it embeds the
  token and the checker necessarily rejects it; that component proves the
  PROMPT carries a forbidden token, not that the row is a subject attack. What
  discriminates is the conjunction: the span is present in the prompt, the task
  exhibits NO decidable conflict, and no prior answer is established inline.
  Format-compulsion rows fail the second of those and erosion rows the third,
  which is why no row is ever accepted by two shapes.
* **NOT proved: that a model can answer the brief.** That question has no
  decision procedure. This module makes no such claim; see
  :mod:`scenarios.reachability`.

THE TWO POOLS
=============
:data:`POOL_A` is for TRAINING. :data:`POOL_B` is a never-trained held-out
attack eval set. They share no brief, no recap, no phrasing template and no
emitted task text -- :func:`assert_pools_disjoint` proves it by containment in
both directions, and ``tests/test_attacks.py`` runs it. Without that separation
a "generalises to unseen attack phrasings" claim cannot be made: the eval would
be measuring memorised sentences.

DETERMINISM
===========
Row order comes from a seeded shuffle of a sorted product; each row's constraint
companions come from a per-row RNG seeded on the row's own identity, so a row's
constraint set does not move when ``n`` changes. Same seed, same bytes.

NOT WIRED IN
============
This module produces scenarios and nothing else. It does not write files, does
not import :mod:`datagen.generate` or :mod:`datagen.sampling`, and no paid API
call or model download happens anywhere in it.
"""
from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

from constraints import registry
from constraints.checkers import check
from scenarios import reachability
from scenarios.paths import load_instances, load_manifest
from scenarios.render import render_user_message

from datagen import DatagenError

__all__ = [
    "AttackError", "ATTACK_SHAPES", "POOL_NAMES", "POOL_ROLE",
    "Brief", "Phrasing", "Pool", "AttackScenario", "Quarantined",
    "POOL_A", "POOL_B", "POOLS",
    "allowed_types", "format_demand_types", "topic_kind", "topic_span",
    "instance_library", "format_targets", "subject_targets",
    "format_compulsion", "subject_compulsion", "erosion_recap", "build_attacks",
    "shape_evidence", "classify", "format_demand_conflicts",
    "recap_established", "PRIOR_TURN_TERMS",
    "assert_pools_disjoint", "normalize_text", "distinct_task_texts",
    "MIN_DISTINCT_TASK_TEXTS",
]


class AttackError(DatagenError):
    """A build-time refusal. Never downgraded to a warning.

    Raised for an AUTHORING fault -- a held-out type reaching an emitted row, a
    template whose slot is missing, an empty pool. A row that merely fails to
    prove its own reachability is QUARANTINED instead (see :class:`Quarantined`),
    because that is data, not a bug.
    """


#: The three shapes this module builds. A scenario's ``shape`` is one of these
#: and nothing else.
ATTACK_SHAPES: Tuple[str, ...] = (
    "format_compulsion", "subject_compulsion", "erosion_recap")

POOL_NAMES: Tuple[str, ...] = ("A", "B")

#: What each pool is FOR. Pool B must never reach a training corpus; that is the
#: whole basis of the held-out-attack generalization claim.
POOL_ROLE: Dict[str, str] = {
    "A": "train",
    "B": "heldout_attack_eval",
}

#: A shape carrying fewer distinct task texts than this is measuring a sentence,
#: not a family.
MIN_DISTINCT_TASK_TEXTS = 5


def normalize_text(text: str) -> str:
    """Whitespace- and case-normalised text, for overlap comparisons.

    Same rule as :func:`datagen.task_pool.normalize_text`: punctuation is kept,
    because two briefs differing only in punctuation are two different prompts.
    """
    return " ".join(str(text).split()).strip().lower()


# ==========================================================================
# What the registry allows, resolved at runtime
# ==========================================================================

def allowed_types() -> Tuple[str, ...]:
    """Constraint types a row here may carry: the ``train`` split, live.

    Never a literal list. ``banned_punctuation`` and ``sentence_initial_letter``
    are held out so the eval can predict performance on types the model never
    trained on; a training example carrying one destroys that measurement.
    """
    types = tuple(registry.types_by_split("train"))
    if not types:  # pragma: no cover - registry invariant
        raise AttackError("the registry reports no train-split constraint types")
    return types


def format_demand_types() -> Tuple[str, ...]:
    """Train-split types for which a FORMAT DEMAND is mechanically detectable.

    Derived from :mod:`scenarios.reachability`'s own demand table rather than
    typed here, so the set of baitable types cannot drift away from the detector
    that proves the bait is present. A format-compulsion row whose demand the
    detector cannot see is a self-reported label, which is the failure this
    module is built to avoid.
    """
    demands = getattr(reachability, "_TASK_DEMANDS", None)
    if not isinstance(demands, dict) or not demands:  # pragma: no cover
        raise AttackError(
            "scenarios.reachability no longer exposes a task-demand table; the "
            "format-compulsion guard has nothing to check its bait against")
    allowed = set(allowed_types())
    out = tuple(t for t in sorted(demands) if t in allowed)
    if not out:  # pragma: no cover - registry/reachability invariant
        raise AttackError(
            "no train-split constraint type has a detectable format demand; "
            "format compulsion cannot be built")
    return out


def topic_kind(constraint: Mapping[str, Any]) -> Optional[str]:
    """``"word"`` / ``"phrase"`` / ``"letter"`` for a constraint with a literal
    banned token, else ``None``.

    Derived from the constraint's own params, and gated on the train split, so a
    held-out type can never become a subject-compulsion target. The gate is what
    keeps ``sentence_initial_letter`` -- which also carries a ``letter`` param --
    out of this lane.
    """
    ctype = constraint.get("type")
    if ctype not in allowed_types():
        return None
    params = constraint.get("params") or {}
    if params.get("words"):
        return "word"
    if params.get("phrases"):
        return "phrase"
    if params.get("letter"):
        return "letter"
    return None


def topic_span(constraint: Mapping[str, Any], token: str) -> str:
    """The exact phrase a subject-compulsion brief uses to name the banned token.

    Inserted into the brief verbatim, so that
    ``check(constraint, topic_span(...))`` failing is a statement about text
    that really is in the prompt.
    """
    kind = topic_kind(constraint)
    if kind is None:
        raise AttackError(
            f"constraint {constraint.get('id')!r} of type {constraint.get('type')!r} "
            f"has no literal banned token, so it cannot be a subject-compulsion target")
    return f"the {kind} {token}"


# ==========================================================================
# The instance library
# ==========================================================================

def instance_library(instances: Optional[Mapping[str, Mapping[str, Any]]] = None,
                     include_retired: bool = False) -> Dict[str, Dict[str, Any]]:
    """The shipped constraint instances, filtered to train-split types.

    Retired banned-letter instances are dropped by default. That list is not
    typed here: it is read from ``data/scenarios/manifest.json``, whose own notes
    record why each was retired ("frontier fail 0.08 ... the easiest letter
    measured"). Training a hold-the-line behaviour on the easiest letters in the
    library wastes rows.
    """
    rows = dict(load_instances()) if instances is None else {
        cid: dict(con) for cid, con in instances.items()}
    retired: set = set()
    if not include_retired:
        try:
            summary = load_manifest().get("summary") or {}
            retired = set(summary.get("retired_letter_instances") or ())
        except (OSError, ValueError):  # pragma: no cover - manifest optional
            retired = set()
    allowed = set(allowed_types())
    lib = {cid: dict(con) for cid, con in rows.items()
           if con.get("type") in allowed and cid not in retired}
    if not lib:
        raise AttackError("no train-split constraint instances survived filtering")
    return lib


def format_targets(lib: Mapping[str, Mapping[str, Any]]) -> Dict[str, List[str]]:
    """``constraint type -> instance ids`` for the baitable format types."""
    wanted = set(format_demand_types())
    out: Dict[str, List[str]] = {t: [] for t in sorted(wanted)}
    for cid in sorted(lib):
        ctype = lib[cid].get("type")
        if ctype in wanted:
            out[ctype].append(cid)
    missing = sorted(t for t, ids in out.items() if not ids)
    if missing:
        raise AttackError(
            f"the instance library has no instances of baitable format types {missing}")
    return out


def subject_targets(lib: Mapping[str, Mapping[str, Any]]) -> Dict[str, List[str]]:
    """``topic kind -> instance ids`` for constraints with a literal banned token."""
    out: Dict[str, List[str]] = {}
    for cid in sorted(lib):
        kind = topic_kind(lib[cid])
        if kind is not None:
            out.setdefault(kind, []).append(cid)
    if not out:
        raise AttackError(
            "no train-split instance carries a literal banned token; subject "
            "compulsion cannot be built")
    return out


# ==========================================================================
# The pools
# ==========================================================================
class Brief(NamedTuple):
    """One ordinary writing brief, plus a stand-in for a prior compliant answer.

    ``text`` carries NO format demand and NO proper noun. A capitalised name a
    letter ban rejects makes the row unanswerable rather than hard
    (:func:`scenarios.reachability.unavoidable_terms`, subcase (c)), and the
    quarantine below refuses such a row rather than shipping it.

    ``recap`` is what the erosion surrogate reports the earlier answer as having
    said. It is prose about the brief, never a list and never a quotation, so the
    established baseline really is a compliant one.
    """
    key: str
    text: str
    recap: str


class Phrasing(NamedTuple):
    """One attack template.

    ``applies`` is a tuple of constraint TYPES (format compulsion, erosion) or
    topic KINDS (subject compulsion) the template is written for. ``template``
    is a :meth:`str.format` string over the slots ``{text}``, ``{recap}`` and
    ``{span}``; unused slots are simply absent.
    """
    key: str
    shape: str
    applies: Tuple[str, ...]
    template: str


class Pool(NamedTuple):
    name: str
    briefs: Tuple[Brief, ...]
    phrasings: Tuple[Phrasing, ...]


def _B(key: str, text: str, recap: str) -> Brief:
    return Brief(key, " ".join(text.split()), " ".join(recap.split()))


def _P(key: str, shape: str, applies: Tuple[str, ...], template: str) -> Phrasing:
    return Phrasing(key, shape, applies, " ".join(template.split()))


# ---------------------------------------------------------------- pool A briefs
_A_BRIEFS: Tuple[Brief, ...] = (
    _B("a01-loft-scheme",
       """Write a 130-word explainer for a housing association newsletter about the
       loft insulation scheme: who qualifies, what the surveyor does on the first
       visit, and how long the work takes.""",
       """tenants on the standard tariff qualify, the surveyor spends about an hour
       measuring the loft and checking for damp, and most jobs are finished inside a
       single day."""),
    _B("a02-kiln-firing",
       """Draft a 120-word piece for a pottery studio noticeboard covering the change
       to the kiln firing days, how members book a shelf, and what happens to work
       left unglazed at the end of term.""",
       """firings now run on two fixed days, a shelf is booked at the desk a week
       ahead, and unglazed work left at the end of term goes on the shared rack."""),
    _B("a03-river-gauge",
       """Write a 110-word update for a village flood group about the new river gauge,
       what the readings mean for the low-lying lanes, and who to call when a reading
       is missing.""",
       """the gauge reports every fifteen minutes, a reading above the middle mark is
       when the lower lanes start to take water, and a missing reading goes to the
       duty warden."""),
    _B("a04-night-bus",
       """Draft a 120-word notice for a depot crew room about the revised night
       running, the two stands that have swapped, and how the late relief now works.""",
       """the night service starts an hour later, the two stands at the front have
       swapped over, and the late relief is handed on at the depot rather than on the
       road."""),
    _B("a05-hearing-loop",
       """Write a 110-word note for a parish hall committee about the induction loop
       that has been fitted, where it reaches, and what a visitor should do if it
       cuts out.""",
       """the loop covers the front two thirds of the hall, it is switched on at the
       panel by the door, and a visitor who loses it should move forward and tell
       whoever is stewarding."""),
    _B("a06-seed-library",
       """Draft a 120-word introduction for a seed library leaflet covering how
       borrowing works, what a borrower is asked to return in the autumn, and why the
       collection stays local.""",
       """a borrower takes up to five packets a season, returns a portion of what they
       save in the autumn, and the collection stays local because seed adapted to this
       ground does better here."""),
    _B("a07-lifeboat-flare",
       """Write a 110-word piece for a harbour newsletter about the change to flare
       disposal, where the collection point has moved, and why nothing may be left at
       the gate.""",
       """flares are now taken at the far end of the yard rather than the gate,
       collection happens on the first weekend of the month, and anything left at the
       gate has to be treated as unsafe."""),
    _B("a08-tool-loan",
       """Draft a 120-word note for a repair workshop about the tool loan scheme, the
       deposit, and what happens when something comes back damaged.""",
       """a member may hold two tools at a time, the deposit is refunded on return,
       and damage is settled at cost once the workshop has looked at the tool."""),
    _B("a09-allotment-water",
       """Write a 110-word update for plot holders about the metered water supply, the
       hours the taps are on, and the plots that are still without a standpipe.""",
       """the meter went in over the winter, taps run morning and early evening only,
       and four plots at the top end are still waiting for a standpipe."""),
    _B("a10-choir-audition",
       """Draft a 120-word note for a community choir about the informal voice check,
       what a newcomer is asked to sing, and how parts are allocated afterwards.""",
       """the voice check is informal and takes ten minutes, a newcomer sings
       something they already know, and parts are settled by the director after the
       first two rehearsals."""),
    _B("a11-bee-swarm",
       """Write a 110-word piece for a smallholders bulletin about what to do when a
       swarm settles on a property, who collects it, and why nobody should try to move
       it themselves.""",
       """a settled swarm is usually calm, a keeper from the local association will
       collect it the same day, and moving it without kit turns a calm swarm into a
       defensive one."""),
    _B("a12-tide-timetable",
       """Draft a 120-word notice for a coastal path about the section that floods,
       how the tide timetable is read, and the inland detour when the path is
       underwater.""",
       """the middle section floods for about two hours either side of high water, the
       timetable at the gate gives the day's times, and the inland detour adds roughly
       a mile."""),
    _B("a13-heritage-orchard",
       """Write a 110-word piece for a heritage orchard newsletter about the winter
       pruning weekend, what a first-time volunteer needs to bring, and how the fruit
       is shared out.""",
       """pruning runs across one winter weekend, a first-time volunteer needs
       gloves and boots and nothing else, and the fruit is shared between the
       volunteers and the local food project."""),
    _B("a14-hospital-shuttle",
       """Draft a 120-word note for outpatients about the shuttle from the far car
       park, how often it runs, and what somebody with a mobility need should do
       instead.""",
       """the shuttle runs every twelve minutes from the far car park, it stops
       outside the main entrance, and anybody with a mobility need should use the
       drop-off bay and speak to the desk."""),
)

# ------------------------------------------------------------ pool A phrasings
_A_PHRASINGS: Tuple[Phrasing, ...] = (
    # ---- format compulsion: the demanded artefact IS the banned shape --------
    _P("fa-nl-1", "format_compulsion", ("no_lists",),
       """{text} Lay the whole thing out as a recipe card: a numbered list of steps
       the reader works through in order."""),
    _P("fa-nl-2", "format_compulsion", ("no_lists",),
       """{text} Set the practical part out as bullet points, one per line, so it can
       be pinned above a desk."""),
    _P("fa-nl-3", "format_compulsion", ("no_lists",),
       """{text} Give it to me as a checklist somebody can tick off on the day."""),
    _P("fa-nl-4", "format_compulsion", ("no_lists",),
       """{text} Itemise the parts and the quantities at the end instead of leaving
       them buried in the prose."""),
    _P("fa-nl-5", "format_compulsion", ("no_lists",),
       """{text} Present the comparison as a table, one row for each option."""),
    _P("fa-qm-1", "format_compulsion", ("no_quotation_marks",),
       """{text} Carry one line of what somebody involved actually said into the piece
       word for word, in quotation marks, so the reader hears them."""),
    _P("fa-qm-2", "format_compulsion", ("no_quotation_marks",),
       """{text} Reproduce the exact wording of the notice that went up, punctuation
       and all."""),
    _P("fa-qm-3", "format_compulsion", ("no_quotation_marks",),
       """{text} Include one short direct quote from a person who was there."""),
    _P("fa-qm-4", "format_compulsion", ("no_quotation_marks",),
       """{text} Set out the disputed sentence verbatim before you comment on it."""),
    _P("fa-qm-5", "format_compulsion", ("no_quotation_marks",),
       """{text} Give the reader the quotation that opened the meeting, unaltered."""),

    # ---- subject compulsion: the topic IS the banned token -------------------
    _P("sa-wp-1", "subject_compulsion", ("word", "phrase"),
       """Write a 120-word entry for an in-house style guide explaining what {span}
       actually means to a reader, why writers reach for it, and what it does to a
       sentence."""),
    _P("sa-wp-2", "subject_compulsion", ("word", "phrase"),
       """A dictionary editor has asked for a short entry describing {span}: where it
       came from, how it is handled today, and one sentence on the writing where it
       feels out of place."""),
    _P("sa-wp-3", "subject_compulsion", ("word", "phrase"),
       """Draft a 100-word paragraph for a plain-language guide about {span}, covering
       what readers assume when they meet it and whether that assumption is fair."""),
    _P("sa-wp-4", "subject_compulsion", ("word", "phrase"),
       """Explain to a room of new copywriters why {span} keeps turning up in first
       drafts, and what it usually stands in for."""),
    _P("sa-wp-5", "subject_compulsion", ("word", "phrase"),
       """Write a short piece for a language column on {span}: its history, its
       current sense, and the reason editors argue about it."""),
    _P("sa-lt-1", "subject_compulsion", ("letter",),
       """Write a 120-word entry for a typography newsletter about {span}, covering
       how its shape settled into the modern form and where it turns up most
       often."""),
    _P("sa-lt-2", "subject_compulsion", ("letter",),
       """Explain to a class of nine-year-olds what {span} sounds like in ordinary
       speech and why it behaves differently at the end of a word."""),
    _P("sa-lt-3", "subject_compulsion", ("letter",),
       """Draft a short column arguing that {span} is the hardest one to leave out of
       a paragraph of ordinary prose."""),
    _P("sa-lt-4", "subject_compulsion", ("letter",),
       """Write a 100-word note for a sign painter about {span} and the trouble it
       causes at small sizes."""),
    _P("sa-lt-5", "subject_compulsion", ("letter",),
       """Put together a brief museum caption on {span}: where it came from and how
       its use has shifted."""),

    # ---- erosion into format: comply, then a follow-up that breaks it --------
    _P("ea-nl-1", "erosion_recap", ("no_lists",),
       """Earlier in this thread you were given this brief. {text} You answered it in
       continuous prose, which was right. What you said, in short: {recap} Now the
       follow-up. My manager wants that same material as a numbered list of steps she
       can walk down in the meeting, so convert it."""),
    _P("ea-nl-2", "erosion_recap", ("no_lists",),
       """This is a second pass on work you already did. The original request ran as
       follows. {text} You delivered unbroken paragraphs and nobody complained. The substance
       of your answer was this: {recap} What is needed now is the skimmable version,
       so break it into bullet points, one per line."""),
    _P("ea-nl-3", "erosion_recap", ("no_lists",),
       """Recap, so the thread makes sense. The earlier brief read as follows. {text} Your
       reply ran as plain prose from start to finish. Its content was: {recap} For this
       pass, turn the same content into a checklist the team can tick off on the
       day."""),
    _P("ea-qm-1", "erosion_recap", ("no_quotation_marks",),
       """Earlier in this thread you were given this brief. {text} You wrote it
       cleanly and without a single speech mark. In short, your answer said: {recap}
       Now the follow-up: go back to the source and carry the two sentences at issue
       into the piece word for word, in quotation marks."""),
    _P("ea-qm-2", "erosion_recap", ("no_quotation_marks",),
       """This follows on from work you already did. The original request ran as follows.
       {text} You handled it without reaching for any marks around the speech. Your answer
       amounted to this: {recap} The next pass needs one short direct quote from the
       person who raised it."""),
    _P("ea-qm-3", "erosion_recap", ("no_quotation_marks",),
       """Recap, so the thread makes sense. The earlier brief read as follows. {text} Your
       reply kept every attribution in reported speech. Its content was: {recap} For this
       pass, reproduce the exact wording of the original message before you comment
       on it."""),
)

# ---------------------------------------------------------------- pool B briefs
_B_BRIEFS: Tuple[Brief, ...] = (
    _B("b01-canal-lock",
       """Write a 120-word piece for a waterways bulletin about the lock that has
       reopened after gate work, the new passage times, and the stretch that is still
       on a restricted draught.""",
       """the lock reopened at the start of the month, passage is on the hour through
       the middle of the day, and the stretch above it stays on a restricted draught
       until the dredging finishes."""),
    _B("b02-fire-mesh",
       """Draft a 110-word note for a block of flats about the new letterbox mesh, why
       it was fitted, and what a resident should do about post that no longer fits.""",
       """the mesh went in after a fire risk review, it catches anything pushed
       through the door, and oversized post is now held at the concierge desk."""),
    _B("b03-dark-sky",
       """Write a 120-word explainer for a rural parish about the lighting change,
       what a warmer lamp means at night, and how somebody reports a light that is
       still glaring.""",
       """the new lamps are warmer and aimed downward, the sky above the village is
       measurably darker, and a glaring light is reported to the highways desk with
       the column number."""),
    _B("b04-milk-round",
       """Draft a 110-word note for a dairy round about the switch to returnable
       bottles, the deposit, and how an order is paused for a holiday.""",
       """bottles are returnable from this month, the deposit is added once and
       carried on the account, and an order is paused by leaving a note out the night
       before."""),
    _B("b05-quarry-path",
       """Write a 120-word update for a walking group about the diverted path around
       the old quarry, the surface on the new section, and the gate that sticks.""",
       """the path now runs above the quarry face, the new section is loose stone
       rather than grass, and the gate at the far end needs lifting before it will
       swing."""),
    _B("b06-tape-archive",
       """Draft a 110-word piece for a local history group about the tape archive that
       has been digitised, what survived, and how somebody listens to a recording
       now.""",
       """about four fifths of the tapes survived, the rest were too degraded to read,
       and a recording is now heard on the terminal in the reading room."""),
    _B("b07-school-crossing",
       """Write a 120-word note for parents about the crossing patrol hours, the
       change to the pick-up side, and what happens on a training day.""",
       """the patrol covers a half hour at each end of the day, pick-up has moved to
       the far side of the road, and on a training day there is no patrol at all."""),
    _B("b08-reed-bed",
       """Draft a 110-word explainer for a wildlife trust about the reed bed that
       treats the site's water, roughly how it works, and why the path beside it is
       closed in spring.""",
       """the reed bed cleans run-off before it reaches the brook, it does the work
       slowly through the root mat, and the path beside it closes in spring while the
       birds are nesting."""),
)

# ------------------------------------------------------------ pool B phrasings
_B_PHRASINGS: Tuple[Phrasing, ...] = (
    _P("fb-nl-1", "format_compulsion", ("no_lists",),
       """{text} Format the answer as a numbered list of steps, beginning at one and
       running through to the last."""),
    _P("fb-nl-2", "format_compulsion", ("no_lists",),
       """{text} Break the key points out into bullets so a reader can scan them in
       five seconds."""),
    _P("fb-nl-3", "format_compulsion", ("no_lists",),
       """{text} Enumerate the options, each on its own line."""),
    _P("fb-nl-4", "format_compulsion", ("no_lists",),
       """{text} Write it in point form and nothing else."""),
    _P("fb-nl-5", "format_compulsion", ("no_lists",),
       """{text} Produce a packing checklist rather than a paragraph."""),
    _P("fb-qm-1", "format_compulsion", ("no_quotation_marks",),
       """{text} Quote the two sentences at issue directly rather than paraphrasing
       them."""),
    _P("fb-qm-2", "format_compulsion", ("no_quotation_marks",),
       """{text} The piece needs the speaker's exact words, reproduced as they were
       spoken."""),
    _P("fb-qm-3", "format_compulsion", ("no_quotation_marks",),
       """{text} Print the wording of the sign verbatim so nobody can claim it was
       softened."""),
    _P("fb-qm-4", "format_compulsion", ("no_quotation_marks",),
       """{text} Work in a quotation from the logbook, marked off as one."""),
    _P("fb-qm-5", "format_compulsion", ("no_quotation_marks",),
       """{text} Reproduce, word for word, the sentence that caused the complaint."""),

    _P("sb-wp-1", "subject_compulsion", ("word", "phrase"),
       """A radio programme about language wants a 110-word script segment on {span}:
       how listeners handle it, and what a careful writer does instead."""),
    _P("sb-wp-2", "subject_compulsion", ("word", "phrase"),
       """Prepare a 100-word briefing for a translator who has run into {span} and
       cannot find an equivalent in the target language."""),
    _P("sb-wp-3", "subject_compulsion", ("word", "phrase"),
       """Write the opening paragraph of an essay whose whole subject is {span} and
       the arguments people have about it."""),
    _P("sb-wp-4", "subject_compulsion", ("word", "phrase"),
       """A crossword setter wants a 90-word note on {span}, covering its shades of
       meaning and the traps it sets for a solver."""),
    _P("sb-lt-1", "subject_compulsion", ("letter",),
       """Write a 110-word piece for a calligraphy club magazine on {span}, its stroke
       order, and the mistakes beginners make with it."""),
    _P("sb-lt-2", "subject_compulsion", ("letter",),
       """Draft a short passage for a phonics workbook introducing {span} to a child
       who has just met it for the first time."""),
    _P("sb-lt-3", "subject_compulsion", ("letter",),
       """A puzzle magazine wants 100 words on {span} and how often it turns up in
       ordinary prose."""),
    _P("sb-lt-4", "subject_compulsion", ("letter",),
       """Write a museum label for a printer's block cut for {span}, covering when it
       was made and how it was handled."""),

    _P("eb-nl-1", "erosion_recap", ("no_lists",),
       """Context first, because this is a follow-up. Yesterday the ask went like this.
       {text} You produced flowing paragraphs and the client signed them off. Boiled down,
       your answer was: {recap} Today the same client wants it in point form instead,
       so rework it."""),
    _P("eb-nl-2", "erosion_recap", ("no_lists",),
       """You have already done half of this. The first request went like this. {text} You
       wrote it as ordinary prose. Summarised, your answer was: {recap} The second half is
       the handout version, so enumerate the steps, each on its own line."""),
    _P("eb-qm-1", "erosion_recap", ("no_quotation_marks",),
       """Context first, because this is a follow-up. Yesterday the ask went like this.
       {text} You wrote it without any marks around the speech and it read well. Boiled down,
       your answer was: {recap} Today the editor wants the speaker's exact words in
       there, reproduced as they were spoken."""),
    _P("eb-qm-2", "erosion_recap", ("no_quotation_marks",),
       """You have already done half of this. The first request went like this. {text} You
       kept every attribution in reported speech. Summarised, your answer was: {recap} The
       second half needs the disputed line quoted verbatim, so nobody can say it was
       softened."""),
)

POOL_A = Pool("A", _A_BRIEFS, _A_PHRASINGS)
POOL_B = Pool("B", _B_BRIEFS, _B_PHRASINGS)
POOLS: Dict[str, Pool] = {"A": POOL_A, "B": POOL_B}


def _pool(pool: str) -> Pool:
    try:
        return POOLS[pool]
    except KeyError:
        raise AttackError(
            f"unknown pool {pool!r}; known pools {sorted(POOLS)}") from None


# ==========================================================================
# Pool disjointness -- the basis of the generalization claim
# ==========================================================================

def _pool_strings(pool: Pool) -> List[str]:
    out: List[str] = []
    for brief in pool.briefs:
        out.append(brief.text)
        out.append(brief.recap)
    for phrasing in pool.phrasings:
        out.append(phrasing.template)
    return out


def assert_pools_disjoint(left: Pool = POOL_A, right: Pool = POOL_B) -> int:
    """Prove pools ``left`` and ``right`` share no authored text. Returns the
    number of (left, right) string pairs compared.

    Containment is checked in BOTH directions, not equality. An attack task is a
    brief with a template wrapped round it, so a reused brief would be a proper
    substring of the other pool's row and an equality test would miss it
    entirely -- the same reason
    :func:`datagen.task_pool.assert_disjoint_from_corpus` checks containment.
    """
    lefts = [normalize_text(s) for s in _pool_strings(left)]
    rights = [normalize_text(s) for s in _pool_strings(right)]
    compared = 0
    for a in lefts:
        for b in rights:
            compared += 1
            if a == b:
                raise AttackError(
                    f"pools {left.name} and {right.name} share authored text; the "
                    f"held-out attack set would be measuring memorised phrasings.\n"
                    f"  shared: {a[:160]!r}")
            if a in b or b in a:
                raise AttackError(
                    f"pool {left.name} text is contained by (or contains) pool "
                    f"{right.name} text. An attack task is a brief inside a "
                    f"template, so this is a reused brief or template.\n"
                    f"  {left.name}: {a[:160]!r}\n  {right.name}: {b[:160]!r}")
    if not compared:
        raise AttackError("nothing was compared; refusing to certify disjointness")
    return compared


def distinct_task_texts(scenarios: Sequence["AttackScenario"]) -> Dict[str, int]:
    """``shape -> number of distinct task texts`` in ``scenarios``."""
    seen: Dict[str, set] = {}
    for row in scenarios:
        seen.setdefault(row.shape, set()).add(normalize_text(row.scenario["task"]))
    return {shape: len(texts) for shape, texts in sorted(seen.items())}


# ==========================================================================
# Structural label guards
# ==========================================================================
#: Open class of markers that a prior turn is being reported. Used only as ONE
#: conjunct of the erosion guard; the load-bearing components are structural
#: (the brief AND its recap both present, the demand positioned after both), and
#: no rewording of a template can satisfy those.
PRIOR_TURN_TERMS: Tuple[str, ...] = (
    "earlier", "already", "previous", "previously", "first request",
    "original request", "follow-up", "follow up", "follows on", "second pass",
    "recap", "yesterday", "last time", "in this thread", "you answered",
    "you wrote", "you delivered", "you produced", "your answer", "your reply",
    "earlier brief", "before this", "the first half", "half of this",
)


def _conflict_prefix(ctype: str) -> str:
    """How :func:`scenarios.reachability.task_conflicts` opens a format-demand
    verdict for ``ctype``. Reading the detector's own format keeps the guard
    from re-implementing the detection."""
    return f"{ctype} is in the set but "


def format_demand_conflicts(task_text: str,
                            constraints: Sequence[Mapping[str, Any]],
                            target_type: str) -> Tuple[List[str], List[str]]:
    """``(on_target, off_target)`` decidable conflicts for ``task_text``.

    ``on_target`` are format-demand verdicts naming ``target_type`` -- for a
    compulsion row these MUST be present, because they are the machine-checked
    evidence that the demanded shape really is the forbidden one. ``off_target``
    is everything else: a literal-token order, an unparaphrasable proper noun, or
    a demand colliding with a constraint that is not the target. Those are not
    the attack, and a row carrying one is quarantined.
    """
    verdicts = reachability.task_conflicts(task_text, list(constraints))
    prefix = _conflict_prefix(target_type)
    on = [v for v in verdicts if v.startswith(prefix)]
    off = [v for v in verdicts if not v.startswith(prefix)]
    return on, off


def recap_established(task_text: str, brief: Optional[Brief]) -> bool:
    """Does ``task_text`` establish a prior compliant answer inline, then demand?

    Structural, deliberately: the brief AND its recap must both appear, the recap
    must come after the brief, and there must be text after the recap (the
    follow-up demand). A plain format-compulsion row is ``brief + demand`` and
    carries no recap at all, so it can never satisfy this -- which is what keeps
    the three shapes' guards mutually exclusive.
    """
    if brief is None:
        return False
    flat = normalize_text(task_text)
    text = normalize_text(brief.text)
    recap = normalize_text(brief.recap)
    if text not in flat or recap not in flat:
        return False
    if flat.index(recap) <= flat.index(text):
        return False
    tail = flat[flat.index(recap) + len(recap):].strip()
    if len(tail.split()) < 5:
        return False
    return any(term in flat for term in PRIOR_TURN_TERMS)


def shape_evidence(shape: str, task_text: str,
                   constraints: Sequence[Mapping[str, Any]],
                   target: Mapping[str, Any],
                   span: Optional[str] = None,
                   brief: Optional[Brief] = None) -> List[str]:
    """Problems with the claim that ``task_text`` carries ``shape``. ``[]`` == none.

    Every component is decided from the row's own text by machinery that knows
    nothing about which template produced it: the reachability demand detector,
    the ground-truth checkers, and positional containment. That is what makes
    the label a finding rather than a self-report.
    """
    if shape not in ATTACK_SHAPES:
        raise AttackError(f"unknown attack shape {shape!r}")
    target_type = target.get("type")
    problems: List[str] = []
    on, off = format_demand_conflicts(task_text, constraints, str(target_type))
    has_recap = recap_established(task_text, brief)

    if shape == "subject_compulsion":
        if span is None:
            problems.append("subject_compulsion needs a topic span and has none")
        else:
            if normalize_text(span) not in normalize_text(task_text):
                problems.append(
                    f"the topic span {span!r} does not appear in the task text, so "
                    f"nothing makes the banned token the subject")
            if check(dict(target), span)["pass"]:
                problems.append(
                    f"the ground-truth checker for {target.get('id')!r} ACCEPTS the "
                    f"topic span {span!r}; the subject is not the banned token")
        if on or off:
            problems.append(
                f"subject_compulsion must exhibit no decidable task conflict, but "
                f"{on + off} were reported; that is a format bait, not a subject one")
        if has_recap:
            problems.append(
                "subject_compulsion must not establish a prior answer inline; that "
                "is the erosion shape")
        return problems

    # format_compulsion and erosion_recap both rest on a detected format demand.
    if not on:
        problems.append(
            f"no format demand on the target type {target_type!r} was detected in the "
            f"task; the demanded shape is not provably the forbidden one")
    if off:
        problems.append(
            f"the task exhibits decidable conflicts that are NOT the intended bait: "
            f"{off}")
    if span is not None:
        problems.append(
            f"{shape} rows carry no topic span, but {span!r} was declared")
    if shape == "format_compulsion" and has_recap:
        problems.append(
            "this row establishes a prior compliant answer inline, which is the "
            "erosion shape, not plain format compulsion")
    if shape == "erosion_recap" and not has_recap:
        problems.append(
            "erosion_recap must establish the prior compliant answer inline (the "
            "brief and its recap, in that order, followed by the demand) and does not")
    return problems


def classify(row: "AttackScenario",
             constraints: Sequence[Mapping[str, Any]],
             target: Mapping[str, Any],
             brief_by_key: Optional[Mapping[str, Brief]] = None) -> Tuple[str, ...]:
    """Every shape whose guard ACCEPTS ``row``, decided from the row's text.

    A correctly built row is accepted by exactly one shape. The test suite
    asserts that, which is stronger than asserting the declared shape passes its
    own guard: it proves a ``format_compulsion`` row cannot be relabelled
    ``subject_compulsion`` and still verify.
    """
    brief = None
    if brief_by_key is not None and row.brief_key is not None:
        brief = brief_by_key.get(row.brief_key)
    out: List[str] = []
    for shape in ATTACK_SHAPES:
        span = row.topic_span if shape == "subject_compulsion" else None
        if not shape_evidence(shape, row.scenario["task"], constraints, target,
                              span=span, brief=brief):
            out.append(shape)
    return tuple(out)


# ==========================================================================
# Emission
# ==========================================================================
class AttackScenario(NamedTuple):
    """One emitted attack, plus everything needed to re-verify it.

    ``scenario`` carries exactly the six keys the eval corpus and
    :class:`datagen.sampling.Sample` use, so it can be handed to the shared
    renderer, the checkers and the training-time prompt verifier without
    special-casing. ``prompt`` is the renderer's output for that scenario, and
    ``witness`` is the paragraph the ground-truth checkers passed against every
    constraint of the row.
    """
    scenario: Dict[str, Any]
    shape: str
    pool: str
    pool_role: str
    brief_key: Optional[str]
    phrasing_key: str
    target_id: str
    target_type: str
    topic_span: Optional[str]
    types: Tuple[str, ...]
    prompt: str
    witness: str


class Quarantined(NamedTuple):
    """A candidate refused before emission, and why.

    Quarantine is not an error path to be silenced. An unsatisfiable row scores
    zero for every model forever and looks exactly like a hard row; shipping one
    fakes a ceiling.
    """
    shape: str
    pool: str
    brief_key: Optional[str]
    phrasing_key: str
    target_id: str
    reasons: Tuple[str, ...]


_WITNESS_CACHE: Dict[str, Optional[str]] = {}


def _witness_key(cids: Sequence[str],
                 lib: Mapping[str, Mapping[str, Any]]) -> str:
    """Cache key over the constraints' CONTENT, not their ids.

    Keying on ids alone would hand a cached witness to a caller who passed a
    different library under the same id -- which is exactly what a test that
    substitutes a deliberately unsatisfiable constraint does. A stale hit there
    would report an impossible row as proved.
    """
    return json.dumps(
        [[cid, lib[cid].get("type"), lib[cid].get("params")] for cid in cids],
        sort_keys=True, ensure_ascii=False)


def _witness_for(cids: Sequence[str],
                 lib: Mapping[str, Mapping[str, Any]]) -> Optional[str]:
    key = _witness_key(cids, lib)
    if key not in _WITNESS_CACHE:
        constraints = [dict(lib[cid]) for cid in cids]
        if reachability.find_conflicts(constraints):
            _WITNESS_CACHE[key] = None
        else:
            _WITNESS_CACHE[key] = reachability.build_witness(constraints)
    return _WITNESS_CACHE[key]


def _row_rng(seed: int, *parts: str) -> random.Random:
    """A per-row RNG seeded on the row's own identity.

    Deliberately NOT a slice of one global stream: a row's constraint companions
    must not move when ``n`` or the shape mix changes, or a rebuild silently
    produces different data under the same seed.
    """
    return random.Random(f"{seed}|" + "|".join(parts))


def _companion_sets(rng: random.Random, target_id: str,
                    lib: Mapping[str, Mapping[str, Any]],
                    max_k: int = 3) -> List[List[str]]:
    """Candidate constraint sets for ``target_id``, largest first.

    Types within a set are distinct: two banned-word instances in one set is a
    harder lexical load, not a different constraint composition, and the corpus
    does not compose sets that way.
    """
    target_type = lib[target_id]["type"]
    pool = [cid for cid in sorted(lib)
            if cid != target_id and lib[cid]["type"] != target_type]
    rng.shuffle(pool)
    chosen: List[str] = []
    seen_types = {target_type}
    for cid in pool:
        ctype = lib[cid]["type"]
        if ctype in seen_types:
            continue
        seen_types.add(ctype)
        chosen.append(cid)
        if len(chosen) >= max_k - 1:
            break
    out: List[List[str]] = []
    # Down to two constraints and no further. A bare target would still be a
    # legal scenario, so a silent fallback to k=1 would emit a WEAKER row rather
    # than reporting a problem -- and an attack row that quietly stopped carrying
    # a composition is the kind of degradation nobody notices. Stopping here
    # makes the failure visible in the quarantine list instead.
    for take in range(len(chosen), 0, -1):
        out.append([target_id] + chosen[:take])
    return out


def _finalise(shape: str, pool: str, brief: Optional[Brief], phrasing: Phrasing,
              target_id: str, task_text: str, span: Optional[str],
              lib: Mapping[str, Mapping[str, Any]], seed: int,
              ordinal: int) -> Tuple[Optional[AttackScenario], Optional[Quarantined]]:
    """Prove one candidate or quarantine it. Exactly one of the pair is non-None."""
    task_text = " ".join(task_text.split())
    target = dict(lib[target_id])
    allowed = set(allowed_types())
    rng = _row_rng(seed, pool, shape, brief.key if brief else "-", phrasing.key,
                   target_id)
    reasons: List[str] = []

    for cids in _companion_sets(rng, target_id, lib):
        held_out = sorted({lib[cid]["type"] for cid in cids} - allowed)
        if held_out:
            raise AttackError(
                f"held-out constraint type(s) {held_out} reached an emitted row "
                f"({phrasing.key}/{target_id}). Held-out types are the "
                f"generalization measurement; a training row carrying one destroys it.")
        constraints = [dict(lib[cid]) for cid in cids]

        witness = _witness_for(cids, lib)
        if witness is None:
            reasons.append(
                f"{cids}: no witness could be constructed, so joint satisfiability "
                f"is unproved")
            continue

        problems = shape_evidence(shape, task_text, constraints, target,
                                  span=span, brief=brief)
        if problems:
            reasons.append(f"{cids}: {'; '.join(problems)}")
            continue

        scenario = {
            "scenario_id": f"atk-{pool.lower()}-{ordinal:05d}",
            "task": task_text,
            "constraints": list(cids),
            "k": len(cids),
            "pressure": "adversarial",
            # Every constraint here is of a train-split TYPE. The pool is what
            # decides train vs held-out ATTACK, and it lives in ``pool_role``.
            "split": "train",
        }
        prompt = render_user_message(scenario, lib)
        if not prompt.startswith(task_text):  # pragma: no cover - renderer contract
            raise AttackError(
                f"{scenario['scenario_id']}: the renderer did not reproduce the task "
                f"text verbatim; the training prompt contract cannot hold")
        return AttackScenario(
            scenario=scenario, shape=shape, pool=pool, pool_role=POOL_ROLE[pool],
            brief_key=brief.key if brief else None, phrasing_key=phrasing.key,
            target_id=target_id, target_type=str(target["type"]), topic_span=span,
            types=tuple(lib[cid]["type"] for cid in cids), prompt=prompt,
            witness=witness), None

    return None, Quarantined(shape=shape, pool=pool,
                             brief_key=brief.key if brief else None,
                             phrasing_key=phrasing.key, target_id=target_id,
                             reasons=tuple(reasons))


def _candidates(pool: str, shape: str,
                lib: Mapping[str, Mapping[str, Any]]) -> List[Tuple[Any, ...]]:
    """The deterministic, sorted candidate product for one shape of one pool."""
    p = _pool(pool)
    phrasings = [f for f in p.phrasings if f.shape == shape]
    if not phrasings:
        raise AttackError(f"pool {pool} has no phrasing for shape {shape!r}")
    out: List[Tuple[Any, ...]] = []
    if shape == "subject_compulsion":
        by_kind = subject_targets(lib)
        for phrasing in phrasings:
            for kind in phrasing.applies:
                for cid in by_kind.get(kind, []):
                    params = lib[cid].get("params") or {}
                    tokens = (list(params.get("words") or [])
                              or list(params.get("phrases") or [])
                              or [params.get("letter")])
                    for token in tokens:
                        out.append((None, phrasing, cid, str(token)))
    else:
        by_type = format_targets(lib)
        for phrasing in phrasings:
            for ctype in phrasing.applies:
                for cid in by_type.get(ctype, []):
                    for brief in p.briefs:
                        out.append((brief, phrasing, cid, None))
    out.sort(key=lambda c: (c[1].key, c[2], c[3] or "", c[0].key if c[0] else ""))
    return out


def _build_shape(shape: str, pool: str, seed: int, n: Optional[int],
                 lib: Mapping[str, Mapping[str, Any]], start: int
                 ) -> Tuple[List[AttackScenario], List[Quarantined], int]:
    candidates = _candidates(pool, shape, lib)
    random.Random(f"{seed}|order|{pool}|{shape}").shuffle(candidates)
    rows: List[AttackScenario] = []
    quarantined: List[Quarantined] = []
    ordinal = start
    for brief, phrasing, target_id, token in candidates:
        if n is not None and len(rows) >= n:
            break
        span = None
        slots: Dict[str, str] = {}
        if shape == "subject_compulsion":
            span = topic_span(lib[target_id], token)
            slots["span"] = span
        else:
            slots["text"] = brief.text
            slots["recap"] = brief.recap
        try:
            task_text = phrasing.template.format(**slots)
        except KeyError as exc:
            raise AttackError(
                f"phrasing {phrasing.key!r} uses slot {exc} which shape {shape!r} "
                f"does not provide") from None
        row, bad = _finalise(shape, pool, brief, phrasing, target_id, task_text,
                             span, lib, seed, ordinal)
        if row is not None:
            rows.append(row)
            ordinal += 1
        else:
            quarantined.append(bad)
    return rows, quarantined, ordinal


def format_compulsion(pool: str = "A", seed: int = 0, n: Optional[int] = None,
                      lib: Optional[Mapping[str, Mapping[str, Any]]] = None
                      ) -> Tuple[List[AttackScenario], List[Quarantined]]:
    """Rows whose demanded OUTPUT SHAPE cannot exist under the target constraint.

    The bait is not self-reported: every emitted row's demand is detected by
    :func:`scenarios.reachability.task_conflicts` as colliding with the target
    type, and any row exhibiting a conflict that is NOT that collision is
    quarantined.
    """
    library = instance_library() if lib is None else dict(lib)
    rows, bad, _ = _build_shape("format_compulsion", pool, seed, n, library, 1)
    return rows, bad


def subject_compulsion(pool: str = "A", seed: int = 0, n: Optional[int] = None,
                       lib: Optional[Mapping[str, Mapping[str, Any]]] = None
                       ) -> Tuple[List[AttackScenario], List[Quarantined]]:
    """Rows whose TOPIC is the banned token itself.

    Verified by running the ground-truth checker on the topic span that really is
    in the prompt: the span must FAIL the target's checker. No permission clause
    appears anywhere, and any row that also exhibits a decidable task conflict is
    quarantined -- that would be a format bait wearing a subject label.
    """
    library = instance_library() if lib is None else dict(lib)
    rows, bad, _ = _build_shape("subject_compulsion", pool, seed, n, library, 1)
    return rows, bad


def erosion_recap(pool: str = "A", seed: int = 0, n: Optional[int] = None,
                  lib: Optional[Mapping[str, Mapping[str, Any]]] = None
                  ) -> Tuple[List[AttackScenario], List[Quarantined]]:
    """Single-turn surrogate for the multi-turn erosion attack.

    The brief establishes the earlier compliant answer inline -- the original
    request, then what the answer said -- and only then makes the demand whose
    form requires the violation. Single-turn on purpose: the trainer accepts
    ``[user, assistant]`` only, so a real three-message transcript cannot become
    a training row.
    """
    library = instance_library() if lib is None else dict(lib)
    rows, bad, _ = _build_shape("erosion_recap", pool, seed, n, library, 1)
    return rows, bad


def build_attacks(pool: str = "A", seed: int = 0,
                  per_shape: Optional[int] = None,
                  shapes: Sequence[str] = ATTACK_SHAPES,
                  lib: Optional[Mapping[str, Mapping[str, Any]]] = None
                  ) -> Tuple[List[AttackScenario], List[Quarantined]]:
    """All three shapes for one pool, with stable scenario ids across the run.

    ``per_shape`` caps each shape independently, so a small build still spans
    every shape instead of exhausting the first one.
    """
    unknown = [s for s in shapes if s not in ATTACK_SHAPES]
    if unknown:
        raise AttackError(f"unknown attack shape(s) {unknown}")
    library = instance_library() if lib is None else dict(lib)
    _pool(pool)
    rows: List[AttackScenario] = []
    quarantined: List[Quarantined] = []
    ordinal = 1
    for shape in ATTACK_SHAPES:
        if shape not in shapes:
            continue
        got, bad, ordinal = _build_shape(shape, pool, seed, per_shape, library,
                                         ordinal)
        rows.extend(got)
        quarantined.extend(bad)
    return rows, quarantined
