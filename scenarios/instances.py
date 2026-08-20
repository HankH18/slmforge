#!/usr/bin/env python3
"""The constraint INSTANCE library (Deliverable 1).

:mod:`constraints.registry` pins seven constraint TYPES. Scenarios cannot
reference a type -- they reference a concrete instance id like ``c-lex-014``,
so those ids have to exist somewhere and resolve. This module is that
somewhere; :func:`build_instances` renders it to
``data/scenarios/constraint_instances.jsonl``.

Rules this module enforces on itself (all of them tested):

* every instance is produced by :func:`constraints.registry.build_constraint`,
  so ``family``, ``split`` and the id prefix are DERIVED from the registry row
  for its type. There is no way to hand-write a per-instance split here, which
  is the point -- the split is a property of the type (Design "Interfaces").
* ids are unique and stable; params are unique within a type. Two instances
  banning the same word set would be two names for one experiment.
* every instance carries an author's note saying what it is for, and a
  ``lexicon`` tag: either a topic from :mod:`scenarios.lexicons` or
  :data:`~scenarios.lexicons.GENERIC`. The tag is metadata for corpus
  assembly, NOT part of the constraint object -- the pinned schema has exactly
  five keys and validate_constraint rejects a sixth.

The tag is what stops the "banned word that never bites" failure: a
lexicon-tagged instance may only be attached to a task on that same topic (see
:mod:`scenarios.generator`).

Word choice note (A1): the checkers match WHOLE WORDS, so "dogs" does not
violate a ban on "dog". Where a writer would naturally reach for the plural,
the plural is banned explicitly rather than left as a free escape hatch.
"""
from __future__ import annotations

from typing import Any, Dict, List, NamedTuple, Tuple

from constraints.registry import build_constraint, validate_constraint

from scenarios.lexicons import GENERIC, LEXICONS

__all__ = ["InstanceSpec", "INSTANCE_SPECS", "INSTANCES", "INSTANCE_META",
           "build_instances", "instance_meta"]


class InstanceSpec(NamedTuple):
    index: int              # per-FAMILY counter -> the numeric part of the id
    type: str               # a registry type; family/split/prefix follow from it
    params: Dict[str, Any]
    lexicon: str            # a LEXICONS key, or GENERIC
    note: str               # why this instance exists / what it bites on


#: Authored instances. Ordered by family then index so the emitted JSONL is
#: stable across runs (a diff in this file should be a diff in the data).
INSTANCE_SPECS: Tuple[InstanceSpec, ...] = (
    # ---------------- lexical: banned_words (train) ----------------
    InstanceSpec(1, "banned_words", {"words": ["coffee", "espresso", "latte", "lattes"]},
                 "coffee", "the four words a cafe piece cannot avoid by accident"),
    InstanceSpec(2, "banned_words", {"words": ["train", "trains", "subway", "rail"]},
                 "transit", "core rolling-stock nouns for a commuting brief"),
    InstanceSpec(3, "banned_words", {"words": ["climate", "carbon", "warming"]},
                 "climate", "the three terms a climate explainer reaches for first"),
    InstanceSpec(4, "banned_words", {"words": ["recipe", "recipes", "oven"]},
                 "cooking", "bans the frame words of any cooking instruction"),
    InstanceSpec(5, "banned_words", {"words": ["music", "song", "songs", "band"]},
                 "music", "forces a gig review to describe sound without naming it"),
    InstanceSpec(6, "banned_words", {"words": ["sleep", "nap", "naps"]},
                 "sleep", "the subject noun of any sleep-hygiene piece"),
    InstanceSpec(7, "banned_words", {"words": ["running", "marathon", "marathons"]},
                 "running", "bans the activity noun a race writeup is built on"),
    InstanceSpec(8, "banned_words", {"words": ["budget", "budgets", "savings"]},
                 "money", "finance-memo vocabulary a writer defaults to"),
    InstanceSpec(9, "banned_words", {"words": ["dog", "dogs", "puppy", "puppies"]},
                 "dogs", "singular and plural both banned: 'dogs' is a separate token"),
    InstanceSpec(10, "banned_words", {"words": ["garden", "gardens", "soil"]},
                 "gardening", "the ground and the place, so paraphrase is required"),
    InstanceSpec(11, "banned_words", {"words": ["film", "films", "movie", "movies"]},
                 "cinema", "both synonyms, closing the obvious swap"),
    InstanceSpec(12, "banned_words", {"words": ["bug", "bugs", "crash", "crashes"]},
                 "software", "incident-writeup nouns an engineer types automatically"),
    InstanceSpec(13, "banned_words", {"words": ["patient", "patients", "doctor", "doctors"]},
                 "medicine", "the two roles a clinic notice is about"),
    InstanceSpec(14, "banned_words", {"words": ["teacher", "teachers", "student", "students"]},
                 "school", "the pinned example id; the two roles a school note names"),
    InstanceSpec(15, "banned_words", {"words": ["ocean", "oceans", "sea", "seas"]},
                 "ocean", "both the formal and the plain word for the same thing"),
    InstanceSpec(16, "banned_words", {"words": ["bike", "bikes", "bicycle", "bicycles"]},
                 "bicycle", "colloquial and formal forms together"),
    InstanceSpec(17, "banned_words", {"words": ["chess", "pawn", "pawns"]},
                 "chess", "the game name plus its most-mentioned piece"),
    InstanceSpec(18, "banned_words", {"words": ["wine", "wines", "vineyard", "vineyards"]},
                 "wine", "product and place, the two nouns a tasting note leans on"),
    InstanceSpec(19, "banned_words", {"words": ["letter", "letters", "mail"]},
                 "mail", "bans the object a postal piece keeps naming"),
    InstanceSpec(20, "banned_words", {"words": ["snow", "winter"]},
                 "winter", "season and substance, both unavoidable in a weather notice"),
    InstanceSpec(21, "banned_words", {"words": ["bread", "dough", "loaf"]},
                 "bakery", "product, ingredient state and unit"),
    InstanceSpec(22, "banned_words", {"words": ["photo", "photos", "photograph", "camera"]},
                 "photography", "the artefact and the device"),
    InstanceSpec(23, "banned_words", {"words": ["library", "libraries", "book", "books"]},
                 "library", "place and object; forces 'volume', 'title', 'branch'"),
    InstanceSpec(24, "banned_words", {"words": ["tea", "kettle"]},
                 "tea", "short, high-frequency, and the topic of the task"),
    # generic word bans -- bite on register rather than topic, so they are legal
    # anywhere and are what the clean-pressure scenarios use.
    InstanceSpec(25, "banned_words", {"words": ["very", "really", "quite"]},
                 GENERIC, "intensifier filler; bites any register"),
    InstanceSpec(26, "banned_words", {"words": ["important", "significant"]},
                 GENERIC, "evaluation filler a model reaches for in every summary"),
    InstanceSpec(27, "banned_words", {"words": ["good", "bad", "great"]},
                 GENERIC, "the three flattest evaluative adjectives"),
    InstanceSpec(28, "banned_words", {"words": ["thing", "things", "stuff"]},
                 GENERIC, "vague-noun ban; forces specific nouns"),
    InstanceSpec(29, "banned_words", {"words": ["just", "simply"]},
                 GENERIC, "minimiser filler common in instructional writing"),
    InstanceSpec(30, "banned_words", {"words": ["actually", "basically"]},
                 GENERIC, "hedging filler common in explanatory writing"),

    # ---------------- lexical: banned_phrases (train) ----------------
    InstanceSpec(40, "banned_phrases", {"phrases": ["in conclusion", "to sum up"]},
                 GENERIC, "the two closing crutches of model prose"),
    InstanceSpec(41, "banned_phrases",
                 {"phrases": ["it is important to note", "it should be noted"]},
                 GENERIC, "throat-clearing openers"),
    InstanceSpec(42, "banned_phrases",
                 {"phrases": ["on the other hand", "that being said"]},
                 GENERIC, "pivot phrases; forces a real transition"),
    InstanceSpec(43, "banned_phrases",
                 {"phrases": ["first of all", "last but not least"]},
                 GENERIC, "enumeration crutches that survive a no-lists rule"),
    InstanceSpec(44, "banned_phrases", {"phrases": ["a wide range of", "a variety of"]},
                 GENERIC, "vague-quantity phrases"),
    InstanceSpec(45, "banned_phrases",
                 {"phrases": ["at the end of the day", "when all is said and done"]},
                 GENERIC, "summing-up idioms"),
    InstanceSpec(46, "banned_phrases", {"phrases": ["in today's world", "in this day and age"]},
                 GENERIC, "essay-opening cliches"),
    InstanceSpec(47, "banned_phrases", {"phrases": ["as an ai", "large language model"]},
                 GENERIC, "self-reference boilerplate"),

    # ---------------- character: banned_letter (train) ----------------
    InstanceSpec(1, "banned_letter", {"letter": "e"},
                 GENERIC, "the hardest lipogram in English; always bites"),
    InstanceSpec(2, "banned_letter", {"letter": "a"},
                 GENERIC, "second-hardest vowel ban"),
    InstanceSpec(3, "banned_letter", {"letter": "s"},
                 GENERIC, "kills every regular plural and third-person verb"),
    InstanceSpec(4, "banned_letter", {"letter": "o"},
                 GENERIC, "vowel ban; blocks 'of', 'to', 'not', 'for'"),
    InstanceSpec(5, "banned_letter", {"letter": "r"},
                 GENERIC, "high-frequency consonant"),
    InstanceSpec(6, "banned_letter", {"letter": "l"},
                 GENERIC, "moderate difficulty consonant ban"),
    InstanceSpec(7, "banned_letter", {"letter": "u"},
                 GENERIC, "blocks 'you', 'our', 'about', 'must'"),
    InstanceSpec(8, "banned_letter", {"letter": "i"},
                 GENERIC, "blocks 'is', 'in', 'it', 'this', '-ing'"),
    InstanceSpec(9, "banned_letter", {"letter": "m"},
                 GENERIC, "lower-frequency consonant; the easy end of the range"),
    InstanceSpec(10, "banned_letter", {"letter": "p"},
                 GENERIC, "lower-frequency consonant; the easy end of the range"),

    # ---------------- character: banned_punctuation (HELD-OUT TYPE) ----------------
    InstanceSpec(20, "banned_punctuation", {"marks": [","]},
                 GENERIC, "the comma alone; forces short clauses"),
    InstanceSpec(21, "banned_punctuation", {"marks": [";", ":"]},
                 GENERIC, "the two marks a model uses to glue clauses"),
    InstanceSpec(22, "banned_punctuation", {"marks": [",", ";"]},
                 GENERIC, "comma and semicolon together; no clause stacking at all"),
    InstanceSpec(23, "banned_punctuation", {"marks": ["-", "—"]},
                 GENERIC, "hyphen and em dash; blocks compound-word and aside habits"),
    InstanceSpec(24, "banned_punctuation", {"marks": ["!", "?"]},
                 GENERIC, "blocks exclamation and rhetorical question"),
    InstanceSpec(25, "banned_punctuation", {"marks": ["(", ")"]},
                 GENERIC, "no parenthetical asides"),
    InstanceSpec(26, "banned_punctuation", {"marks": [",", "-"]},
                 GENERIC, "comma plus hyphen"),

    # ---------------- structural: sentence_initial_letter (HELD-OUT TYPE) ----------------
    # The one POSITIVE (form-requiring) type in the registry. Never appears
    # alone in a scenario -- see NEGATIVE_TYPES in scenarios.generator.
    InstanceSpec(1, "sentence_initial_letter", {"letter": "t"},
                 GENERIC, "every sentence opens with T"),
    InstanceSpec(2, "sentence_initial_letter", {"letter": "s"},
                 GENERIC, "every sentence opens with S"),
    InstanceSpec(3, "sentence_initial_letter", {"letter": "a"},
                 GENERIC, "every sentence opens with A"),
    InstanceSpec(4, "sentence_initial_letter", {"letter": "w"},
                 GENERIC, "every sentence opens with W"),
    InstanceSpec(5, "sentence_initial_letter", {"letter": "i"},
                 GENERIC, "every sentence opens with I"),
    InstanceSpec(6, "sentence_initial_letter", {"letter": "m"},
                 GENERIC, "every sentence opens with M"),
    InstanceSpec(7, "sentence_initial_letter", {"letter": "e"},
                 GENERIC, "every sentence opens with E"),
    InstanceSpec(8, "sentence_initial_letter", {"letter": "o"},
                 GENERIC, "every sentence opens with O"),

    # ---------------- format_negative (train) ----------------
    # Both types take no params, so exactly one instance each exists: a second
    # would be a duplicate under a different name.
    InstanceSpec(1, "no_lists", {},
                 GENERIC, "prose only; no bullets, no numbered items"),
    InstanceSpec(2, "no_quotation_marks", {},
                 GENERIC, "no double quotation marks (apostrophes stay legal, A4)"),
)


def _build() -> Tuple[Dict[str, Dict[str, Any]], Dict[str, InstanceSpec]]:
    instances: Dict[str, Dict[str, Any]] = {}
    meta: Dict[str, InstanceSpec] = {}
    seen_params: Dict[str, List[Any]] = {}
    for spec in INSTANCE_SPECS:
        constraint = build_constraint(spec.type, spec.params, spec.index)
        validate_constraint(constraint)          # belt and braces; raises on drift
        cid = constraint["id"]
        if cid in instances:
            raise ValueError(f"duplicate constraint instance id {cid!r}")
        key = repr(sorted(spec.params.items()))
        bucket = seen_params.setdefault(spec.type, [])
        if key in bucket:
            raise ValueError(f"duplicate params for type {spec.type!r}: {spec.params!r}")
        bucket.append(key)
        if spec.lexicon != GENERIC and spec.lexicon not in LEXICONS:
            raise ValueError(f"{cid}: unknown lexicon tag {spec.lexicon!r}")
        instances[cid] = constraint
        meta[cid] = spec
    return instances, meta


#: id -> the pinned five-key constraint object.
INSTANCES: Dict[str, Dict[str, Any]]
#: id -> the authoring metadata (lexicon tag, note). NOT part of the object.
INSTANCE_META: Dict[str, InstanceSpec]
INSTANCES, INSTANCE_META = _build()


def instance_meta(cid: str) -> InstanceSpec:
    try:
        return INSTANCE_META[cid]
    except KeyError:
        raise KeyError(f"unknown constraint instance {cid!r}") from None


def build_instances() -> List[Dict[str, Any]]:
    """The instance library as a list of constraint objects, in authored order."""
    return [INSTANCES[spec_id] for spec_id in INSTANCES]
