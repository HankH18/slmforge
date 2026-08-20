#!/usr/bin/env python3
"""Topical lexicons: the machinery that makes DISTRACTOR PRESSURE checkable.

The ticket's warning, restated: *"banning 'zygote' from an essay about transit
does nothing and silently makes the corpus easier."* A banned word only creates
pressure when the task's own subject matter pulls the writer toward it. That is
a semantic claim, and semantic claims are exactly the kind of thing a corpus
author asserts and nobody ever verifies.

So it is made mechanical here. Each lexicon is a bag of terms that co-occur in
one topic. A ``banned_words``/``banned_phrases`` instance is tagged with the
lexicon it draws from (see :mod:`scenarios.instances`), and the corpus is only
allowed to attach that instance to a task whose subject is that same lexicon.
The distractor claim then reduces to two facts a test can check against the
shipped JSONL without trusting the author:

1. the banned token is a member of lexicon L;
2. the task text visibly uses at least two OTHER members of lexicon L.

Fact 2 is what rules out the zygote case: a transit task uses transit words, so
a coffee ban can never qualify as a distractor there -- and, by the
same-lexicon rule, can never be attached at all.

Membership is matched WHOLE-WORD and case-insensitively against the task text,
matching the A1 ruling in :mod:`constraints.checkers`, so "cafeteria" is not a
hit for "cafe" here any more than it is there.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

__all__ = ["LEXICONS", "GENERIC", "lexicon_of_term", "members_present", "is_member"]

#: Tag used by constraint instances whose bite does not depend on the topic
#: (filler words, transition phrases, formatting rules). A generic instance may
#: be attached to any task; a lexicon-tagged one may not.
GENERIC = "generic"

#: topic name -> the terms that make that topic recognisable in prose.
#: Every term is lowercase and single-word unless the phrase is idiomatic.
#: Plural forms are listed separately where a writer would really use them --
#: the A1 whole-word ruling means "dogs" is a different token from "dog".
LEXICONS: Dict[str, Tuple[str, ...]] = {
    "coffee": (
        "coffee", "espresso", "espressos", "latte", "lattes", "barista", "baristas",
        "cafe", "cafes", "brew", "brewing", "roast", "roasted", "roaster", "beans",
        "cappuccino", "grinder", "pour", "cup", "cups", "caffeine", "decaf",
    ),
    "transit": (
        "train", "trains", "subway", "rail", "railway", "bus", "buses", "commute",
        "commuters", "platform", "platforms", "fare", "fares", "transit", "station",
        "stations", "timetable", "schedule", "carriage", "turnstile", "ridership",
    ),
    "climate": (
        "climate", "carbon", "emissions", "warming", "drought", "glacier", "glaciers",
        "greenhouse", "temperature", "temperatures", "flooding", "heatwave",
        "rainfall", "wildfire", "wildfires", "sea-level", "degrees",
    ),
    "cooking": (
        "recipe", "recipes", "dough", "oven", "simmer", "knife", "seasoning", "sauce",
        "bake", "baking", "skillet", "garlic", "onions", "roasting", "pan", "stove",
        "chopping", "boiling", "kitchen", "cook", "cooking", "salt",
    ),
    "music": (
        "music", "song", "songs", "guitar", "chorus", "album", "albums", "tempo",
        "melody", "drummer", "drums", "stage", "riff", "band", "bands", "singer",
        "rehearsal", "setlist", "encore", "amplifier", "bass",
    ),
    "sleep": (
        "sleep", "sleeping", "insomnia", "mattress", "dream", "dreams", "nap", "naps",
        "bedtime", "pillow", "drowsy", "alarm", "snoring", "rested", "awake",
        "melatonin", "night", "nights",
    ),
    "running": (
        "running", "marathon", "marathons", "pace", "stride", "sneakers", "mileage",
        "trail", "trails", "cadence", "blister", "blisters", "finish", "kilometre",
        "kilometres", "jog", "jogging", "runner", "runners", "splits", "warmup",
    ),
    "money": (
        "budget", "budgets", "savings", "invoice", "invoices", "rent", "salary",
        "expense", "expenses", "refund", "receipt", "receipts", "spending", "payment",
        "payments", "deposit", "billing", "cost", "costs", "cash",
    ),
    "dogs": (
        "dog", "dogs", "puppy", "puppies", "leash", "kennel", "bark", "barking",
        "breed", "breeds", "groomer", "paws", "vet", "fetch", "collar", "walks",
        "obedience", "shelter",
    ),
    "gardening": (
        "garden", "gardens", "compost", "seedling", "seedlings", "mulch", "pruning",
        "soil", "trowel", "bloom", "blooms", "weeds", "weeding", "planting",
        "watering", "greenhouse", "shrubs", "beds",
    ),
    "cinema": (
        "film", "films", "movie", "movies", "director", "cinema", "screening",
        "screenings", "script", "actor", "actors", "camera", "subtitle", "subtitles",
        "festival", "editing", "scene", "scenes", "projector",
    ),
    "software": (
        "bug", "bugs", "crash", "crashes", "deploy", "deployment", "server",
        "servers", "database", "latency", "commit", "commits", "refactor", "release",
        "cache", "rollback", "logs", "outage", "endpoint", "queue",
    ),
    "medicine": (
        "patient", "patients", "doctor", "doctors", "nurse", "nurses", "clinic",
        "symptom", "symptoms", "dosage", "ward", "triage", "diagnosis", "referral",
        "prescription", "appointment", "waiting", "discharge",
    ),
    "school": (
        "teacher", "teachers", "student", "students", "classroom", "homework", "exam",
        "exams", "syllabus", "pupil", "pupils", "lesson", "lessons", "grading",
        "marking", "coursework", "term", "assembly",
    ),
    "ocean": (
        "ocean", "oceans", "sea", "seas", "tide", "tides", "reef", "reefs", "kelp",
        "plankton", "shoreline", "current", "currents", "harbour", "seabird",
        "seabirds", "trawler", "estuary", "waves",
    ),
    "bicycle": (
        "bike", "bikes", "bicycle", "bicycles", "pedal", "pedals", "gears", "saddle",
        "helmet", "tyres", "spokes", "chain", "cyclist", "cyclists", "lanes",
        "puncture", "handlebars", "brakes",
    ),
    "chess": (
        "chess", "pawn", "pawns", "endgame", "opening", "openings", "rook", "bishop",
        "tournament", "blunder", "blunders", "clock", "board", "move", "moves",
        "checkmate", "draw", "rating",
    ),
    "wine": (
        "wine", "wines", "vineyard", "vineyards", "cellar", "tannin", "tannins",
        "vintage", "grapes", "sommelier", "cork", "bottle", "bottles", "harvest",
        "tasting", "barrel", "barrels",
    ),
    "mail": (
        "letter", "letters", "mail", "envelope", "envelopes", "postage", "mailbox",
        "courier", "parcel", "parcels", "delivery", "stamp", "stamps", "postcard",
        "postcards", "sorting", "postal",
    ),
    "winter": (
        "snow", "winter", "ice", "shovel", "frost", "blizzard", "plough", "mittens",
        "sled", "sledding", "freezing", "slush", "gritting", "boots", "thaw",
    ),
    "bakery": (
        "bread", "sourdough", "crust", "yeast", "loaf", "loaves", "bakery", "crumb",
        "proofing", "flour", "starter", "rolls", "pastry", "baker", "bakers", "oven",
    ),
    "photography": (
        "photo", "photos", "photograph", "photographs", "lens", "lenses", "aperture",
        "shutter", "darkroom", "exposure", "tripod", "camera", "negatives", "prints",
        "framing", "focus",
    ),
    "library": (
        "library", "libraries", "book", "books", "shelf", "shelves", "catalogue",
        "borrow", "borrowing", "overdue", "librarian", "librarians", "reading",
        "loans", "returns", "stacks",
    ),
    "tea": (
        "tea", "teas", "kettle", "steep", "steeping", "leaves", "teapot", "brewing",
        "chamomile", "infuser", "mug", "mugs", "pot", "oolong",
    ),
}


def _word_re(term: str) -> "re.Pattern[str]":
    # Whole-word, case-insensitive; mirrors the A1 boundary ruling in
    # constraints.checkers (a boundary is any non-letter, non-digit character).
    escaped = r"\s+".join(re.escape(part) for part in term.split())
    return re.compile(r"(?<![^\W_])" + escaped + r"(?![^\W_])", re.IGNORECASE | re.UNICODE)


def is_member(lexicon: str, term: str) -> bool:
    """Is ``term`` a member of ``lexicon``? Unknown lexicon raises KeyError."""
    return term.lower() in {t.lower() for t in LEXICONS[lexicon]}


def lexicon_of_term(term: str) -> List[str]:
    """Every lexicon containing ``term``. Used to catch cross-topic word bans."""
    low = term.lower()
    return [name for name, terms in LEXICONS.items()
            if low in {t.lower() for t in terms}]


def members_present(lexicon: str, text: str, exclude: Tuple[str, ...] = ()) -> List[str]:
    """Members of ``lexicon`` that literally appear in ``text``, in lexicon order.

    ``exclude`` drops terms from consideration -- the caller passes the banned
    tokens so the "task uses OTHER members of the topic" test cannot be
    satisfied by the banned token itself.
    """
    skip = {t.lower() for t in exclude}
    found: List[str] = []
    for term in LEXICONS[lexicon]:
        if term.lower() in skip:
            continue
        if _word_re(term).search(text):
            found.append(term)
    return found
