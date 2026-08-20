#!/usr/bin/env python3
"""Reachability: proof that a scenario can actually be passed.

An impossible scenario is the worst kind of bad data. "Ban the letter e, and
begin every sentence with E" scores zero for every model, on every strategy,
forever -- and it looks exactly like a hard scenario the models cannot do yet.
It fakes a ceiling. So the corpus does not merely avoid the obvious cases: for
every scenario it CONSTRUCTS a passing text and runs the real checkers on it.

Two layers, both used by the generator at build time and by
``tests/test_scenarios.py`` on the shipped JSONL:

1. :func:`find_conflicts` -- analytic. Names impossibilities in terms a human
   can act on ("c-str-002 requires every sentence to start with 's' while
   c-chr-003 bans the letter 's'"). Fast, and it explains itself.
2. :func:`build_witness` -- constructive. Assembles a real multi-sentence
   paragraph out of a filtered word pool and returns it only if
   :func:`constraints.checkers.check` passes it against EVERY constraint in the
   set. This is the load-bearing one: it cannot be fooled by a contradiction
   nobody thought to write a rule for, because the proof is an artefact the
   ground-truth checkers signed off on.

The witness deliberately must be substantial -- :data:`MIN_WITNESS_CHARS` and
:data:`MIN_WITNESS_SENTENCES`. An empty string vacuously passes every checker
(ruling A7), so "there exists a passing text" is trivially true and worthless
unless the text is real prose. Do not lower those floors to make a scenario
build; fix the scenario.

The witness is a satisfiability proof for the CONSTRAINT SET, not a model
answer -- it is not expected to address the task.

==========================================================================
WHAT IS PROVED, AND WHAT IS NOT. Read this before quoting a number from the
corpus.
==========================================================================

There are TWO reachability questions and they have different epistemic status.
Conflating them was a real defect in an earlier revision of this file, which
advertised reachability as "proved constructively" when only the first half
was.

(1) IS THE CONSTRAINT SET JOINTLY SATISFIABLE?  **Proved, constructively.**
    :func:`build_witness` returns a real multi-sentence paragraph that the
    ground-truth checkers pass against every constraint in the set. The proof
    is an artefact, not an argument.

(2) CAN THE TASK BE ANSWERED UNDER THAT CONSTRAINT SET?  **Not proved, and
    not provable by any mechanical procedure this repo can run.** Deciding it
    means deciding whether the MEANING a task asks for is expressible in the
    surviving fragment of English, which is a semantic question with no
    decision procedure. The corpus therefore makes no such claim.

What :func:`task_conflicts` does is narrower and honest: it decides the
DECIDABLE SUBCASES of (2) -- the ones where a contradiction can be exhibited,
not argued -- and each verdict is signed by the real checkers or by a literal
match against the shipped task text:

  a. FORMAT DEMANDS. The brief asks for output formatting that a format
     constraint forbids ("give me five bullet points" + ``no_lists``). Decided
     by literal match against the task text.
  b. LITERAL-TOKEN ORDERS. The brief orders a specific token into the response
     ("use the word 'sustainable'") and a constraint bans it.
  c. UNPARAPHRASABLE NAMES. The brief names an entity -- a proper noun -- and
     the real checker REJECTS that name on its own (``Thursdays`` under a ban
     on the letter ``s``). A name has no synonym, so any response that names
     the entity fails. See :func:`unavoidable_terms`.

Subcase (c) is where the previous revision was weakest: the old implementation
was a nine-regex keyword list that could not see a name at all. It is now
decided by running :func:`constraints.checkers.check` on the name itself, so
it cannot drift away from the ground truth the corpus is graded against.

THE RESIDUE, STATED PLAINLY. (c) proves that NAMING the entity is impossible;
it does not prove the task is impossible, because a writer may be able to
answer without naming it ("the fourth weekday"). The corpus takes the
conservative side of that gap: a scenario whose brief collides with its own
constraints under (a), (b) or (c) is REFUSED by the generator rather than
shipped with an unproven claim attached, and the affected task ids are recorded
in :data:`scenarios.generator.QUARANTINE`. Beyond (a)-(c), no claim is made:
a scenario in the corpus is one for which no contradiction could be exhibited,
which is weaker than "answerable", and nothing downstream should read it as
more.

WHY THIS RUNS ON THE BASE TASK ONLY. An adversarial scenario appends a nudge
that deliberately invites a violation ("use bullets regardless of the
formatting rule"). That is the embedded pressure working as designed, not a
self-contradictory brief, so :func:`task_conflicts` is always given the
nudge-free base task. Handing it the rendered task would flag every
format-targeting nudge in the corpus.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from constraints.checkers import check

__all__ = [
    "MIN_WITNESS_CHARS",
    "MIN_WITNESS_SENTENCES",
    "FUNCTION_WORDS",
    "UNPARAPHRASABLE_TYPES",
    "find_conflicts",
    "task_conflicts",
    "task_reachability_report",
    "unavoidable_terms",
    "content_words",
    "whole_word_present",
    "build_witness",
    "witness_report",
]

#: A witness shorter than this is not evidence of anything (A7: empty text
#: passes every checker vacuously).
MIN_WITNESS_CHARS = 120
#: ...and it has to be several sentences, or a sentence-initial-letter
#: constraint is proven by a single lucky opening word.
MIN_WITNESS_SENTENCES = 4

# ==========================================================================
# Word pool
#
# Plain, concrete English. Chosen for spread across initial letters and for
# tolerance to a single-letter ban: whatever letter is banned, enough of each
# pool survives to build four sentences. The pool is not topical because a
# witness proves the CONSTRAINT SET is satisfiable, not that the task is
# answerable in prose (that is the judge's job at eval time).
#
# Two coverage notes earned the hard way, both found by running the builder
# over every (sentence_initial_letter x banned_letter) pair in the instance
# library rather than by inspection:
#
# * banning 's' deletes every regular plural, so the noun pool carries
#   irregular plurals ("children", "oxen", "geese") that survive it;
# * the letters a, e, i and o are all both openers AND ban targets, so each
#   needs several words that start with it and avoid the others.
# ==========================================================================

NOUNS: Tuple[str, ...] = (
    # irregular plurals -- the only nouns that survive a ban on 's'
    "children", "men", "women", "oxen", "geese", "mice", "deer", "elk",
    "people", "teeth", "feet", "cattle",
    # vowel-initial nouns, spread so every opener letter survives a ban on
    # any single other vowel
    "acorns", "attics", "apples", "abbeys", "anthems", "orchards", "orchids",
    "outposts", "oaks", "orbits", "onions", "owls", "inlets", "icicles",
    "islanders", "idols", "innkeepers", "embers", "elders", "echoes",
    "anchors", "artists", "banners", "baskets", "beacons", "benches", "boats",
    "borders", "bottles", "boulders", "branches", "bridges", "brooks", "builders",
    "candles", "canyons", "captains", "cellars", "chapters", "cliffs", "clocks",
    "clouds", "coasts", "corridors", "cottages", "currents", "dancers", "deserts",
    "divers", "doors", "drawings", "dunes", "eagles", "echoes", "elders", "embers",
    "engines", "evenings", "fables", "falcons", "farmers", "feathers", "ferries",
    "fields", "fires", "flowers", "forests", "fountains", "foxes", "gardens",
    "gates", "giants", "glaciers", "gliders", "groves", "guards", "harbors",
    "harvests", "hills", "horses", "houses", "islands", "journals", "keepers",
    "kettles", "kitchens", "lakes", "lamps", "lanterns", "ledges", "lions",
    "machines", "maps", "markets", "meadows", "mirrors", "mornings", "mountains",
    "nights", "notebooks", "oceans", "orchards", "otters", "owls", "painters",
    "paths", "pilots", "pines", "planets", "ponds", "porches", "quarries",
    "rabbits", "rafts", "rangers", "ravens", "rivers", "roads", "rooftops",
    "sailors", "seasons", "shadows", "ships", "singers", "slopes", "sparrows",
    "stables", "stars", "storms", "streams", "tailors", "temples", "thickets",
    "tigers", "towers", "trails", "tunnels", "valleys", "villages", "vines",
    "voyages", "walls", "wanderers", "waterfalls", "whales", "willows", "windows",
    "wolves", "workshops",
)

VERBS: Tuple[str, ...] = (
    "appear", "arrive", "awaken", "balance", "blossom", "brighten", "carry",
    "circle", "climb", "cross", "dance", "deepen", "depart", "drift", "echo",
    "endure", "expand", "explore", "fade", "flourish", "flow", "follow", "gather",
    "glide", "glow", "grow", "hover", "hurry", "journey", "kindle", "linger",
    "listen", "march", "meander", "mingle", "move", "murmur", "open", "orbit",
    "pause", "pulse", "quicken", "rally", "ramble", "reach", "recover", "remain",
    "return", "rise", "roam", "rumble", "rush", "scatter", "settle", "shift",
    "shimmer", "sing", "soar", "sparkle", "spin", "spiral", "stand", "stir",
    "stretch", "survive", "swim", "sway", "thrive", "tilt", "tower", "travel",
    "turn", "unfold", "vanish", "wait", "wander", "watch", "waver", "whirl",
    "widen", "wither",
)

ADJECTIVES: Tuple[str, ...] = (
    # vowel-initial adjectives carry the sentence_initial_letter openers when
    # a banned letter has emptied the matching noun pool
    "icy", "inky", "ivory", "indigo", "old", "oily", "oval", "arctic",
    "ample", "aged", "eldest", "even",
    "airy", "amber", "ancient", "arid", "azure", "bold", "brave", "bright",
    "broad", "calm", "clear", "cold", "cool", "coral", "crimson", "curious",
    "dark", "deep", "distant", "drowsy", "dusty", "eager", "early", "eight",
    "elder", "empty", "endless", "faint", "famous", "fierce", "fine", "fluid",
    "fresh", "frozen", "gentle", "giant", "glad", "golden", "grand", "green",
    "happy", "hardy", "heavy", "hidden", "high", "hollow", "honest", "huge",
    "ideal", "idle", "immense", "inland", "iron", "jagged", "jolly", "keen",
    "kind", "lively", "lofty", "lonely", "long", "loud", "lucid", "lush",
    "marble", "mellow", "mighty", "mild", "misty", "modern", "muddy", "narrow",
    "noble", "northern", "olive", "open", "orange", "ornate", "outer", "pale",
    "patient", "placid", "polar", "proud", "purple", "quiet", "radiant", "rapid",
    "rare", "restless", "rocky", "rosy", "round", "royal", "rugged", "sandy",
    "scarlet", "serene", "sharp", "silent", "silver", "slender", "slow", "small",
    "smooth", "solemn", "southern", "spare", "steady", "still", "stony", "strong",
    "sturdy", "sunny", "swift", "tall", "thin", "tidy", "timid", "tiny",
    "tranquil", "upland", "urban", "vast", "velvet", "vibrant", "violet", "vivid",
    "warm", "weary", "western", "white", "wide", "wild", "windy", "wise",
    "wooden", "young",
)

ADVERBS: Tuple[str, ...] = (
    "boldly", "brightly", "calmly", "clearly", "daily", "deeply", "evenly",
    "fully", "gently", "gladly", "greatly", "highly", "hourly", "keenly",
    "kindly", "lightly", "loudly", "mildly", "nightly", "nimbly", "often",
    "openly", "quickly", "quietly", "rapidly", "readily", "rightly", "sharply",
    "silently", "slowly", "softly", "steadily", "strongly", "surely", "sweetly",
    "swiftly", "truly", "warmly", "weekly", "widely", "wisely", "yearly",
)

PREPOSITIONS: Tuple[str, ...] = (
    "above", "across", "along", "amid", "among", "around", "atop", "behind",
    "below", "beneath", "beside", "between", "beyond", "by", "from", "inside",
    "into", "near", "onto", "over", "past", "through", "toward", "under",
    "upon", "with", "within", "without",
)

#: Sentence terminators the builder may use, in preference order.
TERMINATORS: Tuple[str, ...] = (".", "!", "?")

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


# ==========================================================================
# Reading a constraint set
# ==========================================================================

def _summarise(constraints: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Pull the fields the reachability logic needs out of a constraint set."""
    banned_letters: List[Tuple[str, str]] = []          # (letter, constraint id)
    initial_letters: List[Tuple[str, str]] = []
    banned_words: Set[str] = set()
    banned_phrases: List[str] = []
    banned_marks: Set[str] = set()
    flags: Set[str] = set()
    for constraint in constraints:
        ctype = constraint["type"]
        params = constraint.get("params") or {}
        cid = constraint.get("id")
        if ctype == "banned_letter":
            banned_letters.append((params["letter"].casefold(), cid))
        elif ctype == "sentence_initial_letter":
            initial_letters.append((params["letter"].casefold(), cid))
        elif ctype == "banned_words":
            banned_words |= {w.casefold() for w in params["words"]}
        elif ctype == "banned_phrases":
            banned_phrases += [p.casefold() for p in params["phrases"]]
        elif ctype == "banned_punctuation":
            banned_marks |= set(params["marks"])
        elif ctype == "no_lists":
            flags.add("no_lists")
        elif ctype == "no_quotation_marks":
            flags.add("no_quotation_marks")
    return {"banned_letters": banned_letters, "initial_letters": initial_letters,
            "banned_words": banned_words, "banned_phrases": banned_phrases,
            "banned_marks": banned_marks, "flags": flags}


# ==========================================================================
# Layer 1 -- analytic conflicts
# ==========================================================================

#: Banning every vowel leaves no writable English. Two banned vowels is already
#: brutal; five is impossible, and the line has to be drawn somewhere it can be
#: defended. The constructive witness is the real gate -- this rule exists so
#: the failure message says WHY rather than "no witness found".
VOWELS = frozenset("aeiou")


def find_conflicts(constraints: Sequence[Dict[str, Any]]) -> List[str]:
    """Analytic impossibilities in a constraint set. Empty list == none found.

    Not a completeness claim -- :func:`build_witness` is what proves
    satisfiability. This names the failures whose diagnosis is worth having in
    words.
    """
    problems: List[str] = []
    info = _summarise(constraints)

    ids = [c.get("id") for c in constraints]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        problems.append(f"constraint id(s) {duplicates} appear more than once in one scenario")

    initials = info["initial_letters"]
    if len({letter for letter, _ in initials}) > 1:
        problems.append(
            "two sentence_initial_letter constraints demand different opening letters: "
            + ", ".join(f"{cid} -> {letter!r}" for letter, cid in initials)
            + "; no sentence can start with both")

    banned = info["banned_letters"]
    for letter, cid in initials:
        for bad_letter, bad_cid in banned:
            if letter == bad_letter:
                problems.append(
                    f"{cid} requires every sentence to begin with {letter!r} while {bad_cid} "
                    f"bans the letter {letter!r}: no sentence can satisfy both")

    banned_set = {letter for letter, _ in banned}
    if VOWELS <= banned_set:
        problems.append(f"every vowel is banned ({sorted(banned_set)}): no English word survives")
    if len(banned_set) >= 3:
        problems.append(
            f"{len(banned_set)} letters banned at once ({sorted(banned_set)}); the corpus caps "
            "this at two because beyond that the scenario stops measuring composition and "
            "starts measuring lipogram trivia")

    # A banned word that is a single letter also bans that letter as a word.
    # Harmless on its own; contradictory when the sentence-initial rule forces
    # a one-letter opener, which only 'a' and 'i' can be.
    for letter, cid in initials:
        if letter in {"a", "i"} and letter in info["banned_words"]:
            problems.append(
                f"{cid} demands sentences opening with {letter!r} while the word {letter!r} "
                "is banned outright; the natural opener is unavailable")

    marks = info["banned_marks"]
    if set(TERMINATORS) <= marks:
        problems.append(
            f"every sentence terminator is banned ({sorted(marks)}); the response can only be "
            "one newline-separated fragment, which is not prose")

    return problems


#: Task wordings that demand exactly the OUTPUT FORMAT a format constraint
#: forbids. Keyed by constraint type; each entry is (regex, human explanation).
#:
#: These match demands for a FORMAT, deliberately not mentions of a concept.
#: "Explain the steps in order" is not a conflict with ``no_lists`` -- prose
#: can narrate steps -- and flagging it would manufacture a contradiction where
#: none exists, which is the failure mode this lane is here to remove, not to
#: introduce. Only a demand that the OUTPUT be shaped as a list counts.
_TASK_DEMANDS: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "no_lists": (
        (r"\bbullet(?:ed|s|\s+points?)?\b", "the task asks for bullets"),
        (r"\bnumbered\s+(?:list|steps|points|items)\b", "the task asks for numbered items"),
        (r"\bas\s+a\s+(?:numbered\s+|bulleted\s+)?list\b", "the task asks for a list"),
        (r"\bin\s+list\s+form\b", "the task asks for list form"),
        (r"\bin\s+point\s+form\b", "the task asks for point form"),
        (r"\bone\s+per\s+line\b", "the task asks for one item per line"),
        (r"\bcheck\s?list\b", "the task asks for a checklist"),
        (r"\bitemi[sz]e\b", "the task asks the model to itemise"),
        (r"\benumerate\b", "the task asks the model to enumerate"),
        (r"\bas\s+a\s+table\b", "the task asks for a table"),
        (r"\blist (?:the|out|them|five|four|three|ten)\b", "the task asks the model to list items"),
    ),
    "no_quotation_marks": (
        (r"\bquote\b", "the task asks for a quote"),
        (r"\bquotes\b", "the task asks for quotes"),
        (r"\bquoting\b", "the task asks the model to quote"),
        (r"\bquotation\b", "the task asks for a quotation"),
        (r"\bverbatim\b", "the task asks for verbatim wording"),
        (r"\bword\s+for\s+word\b", "the task asks for word-for-word reproduction"),
        (r"\bexact\s+(?:wording|words|phrasing)\b", "the task asks for the exact wording"),
        (r"\bin\s+quotation\s+marks\b", "the task asks for quotation marks"),
    ),
}

#: A task that orders the model to use a token another constraint bans is a
#: direct contradiction rather than a hard scenario.
_USE_THE_WORD_RE = re.compile(
    r"\b(?:use|include|say|write)\s+the\s+(?:word|term|phrase|name)\s+[\"'“]?([\w'-]+)",
    re.IGNORECASE)

#: Constraint types a bare name can violate wherever it appears in a response.
#: ``sentence_initial_letter`` is deliberately excluded: it is POSITIONAL, so a
#: name that cannot open a sentence can still be written in the middle of one.
#: Including it here would flag every proper noun in the corpus as a conflict,
#: which would be a manufactured impossibility rather than a detected one.
UNPARAPHRASABLE_TYPES: Tuple[str, ...] = (
    "banned_letter", "banned_words", "banned_phrases", "banned_punctuation",
)

#: Closed-class English function words. Used to separate the words a task is
#: ABOUT from the grammatical scaffolding every task shares, both here (so "The"
#: is never mistaken for a name) and by the generator's derived pressure floors.
FUNCTION_WORDS: frozenset = frozenset("""
a an the this that these those and or but nor so yet for of to in on at by with
from into onto over under about after before between during through as is are
was were be been being am do does did done have has had having will would shall
should can could may might must it its they them their there here he she his
her him you your yours we us our ours i me my mine not no than then when where
which who whom whose what why how all any each few more most other some such
only own same too very just also if because while both either neither per up
down out off again once
""".split())

#: A word, allowing internal apostrophes and hyphens. No digits: "07.14" is not
#: a name, and a numeral carries no letters for a letter ban to bite on.
_TOKEN_RE = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*", re.UNICODE)
#: Sentence break for the purpose of "is this capital at the start of a
#: sentence". Deliberately crude: over-splitting only costs recall here.
_SENT_BREAK_RE = re.compile(r"(?<=[.!?])\s+")
#: Names shorter than this are too likely to be an initialism or an artefact.
_MIN_NAME_CHARS = 2


def whole_word_present(term: str, text: str) -> bool:
    """Does ``term`` occur in ``text`` as a whole word (or whole phrase)?

    Mirrors the A1 boundary ruling used by :mod:`constraints.checkers` and
    :mod:`scenarios.lexicons`: a boundary is any non-letter, non-digit
    character. Substring containment is NOT enough -- "us" must not match
    "museum".
    """
    if not term.strip():
        return False
    pattern = r"\s+".join(re.escape(part) for part in term.split())
    return re.search(r"(?<![^\W_])" + pattern + r"(?![^\W_])",
                     text, re.IGNORECASE | re.UNICODE) is not None


def content_words(text: str) -> List[str]:
    """The distinct, casefolded CONTENT words of ``text``, in sorted order.

    Content == not a closed-class function word and longer than two characters.
    Derived from the text alone; nothing here is declarable by an author.
    """
    return sorted({w.casefold() for w in _TOKEN_RE.findall(text)
                   if w.casefold() not in FUNCTION_WORDS and len(w) > 2})


def unavoidable_terms(task_text: str) -> List[str]:
    """Tokens a response cannot paraphrase away, derived from the task text.

    Today that means PROPER NOUNS: a capitalised word that is not the first
    word of its sentence. A name has no synonym -- a response about a meeting
    moved to ``Thursdays`` either writes ``Thursdays`` or stops naming the day
    -- so a checker that rejects the bare name rejects every response that
    names the entity.

    Known incompleteness, stated rather than hidden: a proper noun that happens
    to open a sentence is not detected (it is indistinguishable from ordinary
    capitalisation), and a lowercase unavoidable term -- a technical term with
    no synonym -- is not detected at all. This function is a lower bound on the
    unparaphrasable vocabulary of a task, never an enumeration of it.
    """
    found: List[str] = []
    seen: Set[str] = set()
    for sentence in _SENT_BREAK_RE.split(task_text):
        stripped = sentence.strip()
        if not stripped:
            continue
        opening = _TOKEN_RE.match(stripped)
        opening_span = opening.span() if opening else (-1, -1)
        for match in _TOKEN_RE.finditer(stripped):
            token = match.group(0)
            if match.span() == opening_span:
                continue                      # sentence-initial capital proves nothing
            if not token[0].isupper():
                continue
            if len(token) < _MIN_NAME_CHARS or token.casefold() in FUNCTION_WORDS:
                continue
            if token.casefold() in seen:
                continue
            seen.add(token.casefold())
            found.append(token)
    return found


def task_conflicts(task_text: str, constraints: Sequence[Dict[str, Any]]) -> List[str]:
    """Decidable contradictions between what the TASK demands and what the
    constraints forbid.

    Always call this with the NUDGE-FREE base task -- see the module docstring.
    An empty list means no contradiction could be EXHIBITED. It does not mean
    the task is answerable under the constraint set; that question has no
    decision procedure and the corpus does not claim it.
    """
    problems: List[str] = []
    types = {c["type"] for c in constraints}

    # (a) the brief demands an output format a constraint forbids
    for ctype, patterns in _TASK_DEMANDS.items():
        if ctype not in types:
            continue
        for pattern, explanation in patterns:
            match = re.search(pattern, task_text, re.IGNORECASE)
            if match:
                problems.append(
                    f"{ctype} is in the set but {explanation} ({match.group(0)!r})")

    # (b) the brief orders a literal token that a constraint bans
    info = _summarise(constraints)
    for match in _USE_THE_WORD_RE.finditer(task_text):
        token = match.group(1).casefold()
        if token in info["banned_words"]:
            problems.append(f"the task orders the model to use the banned word {token!r}")

    # (c) the brief names an entity the ground-truth checkers reject outright
    problems += _unparaphrasable_conflicts(task_text, constraints)
    return problems


def _unparaphrasable_conflicts(task_text: str,
                               constraints: Sequence[Dict[str, Any]]) -> List[str]:
    """Names in the task that a real checker rejects on their own.

    The verdict is produced by :func:`constraints.checkers.check`, not by a
    rule written here, so it cannot drift away from the ground truth the corpus
    is graded against.
    """
    problems: List[str] = []
    terms = unavoidable_terms(task_text)
    if not terms:
        return problems
    for constraint in constraints:
        if constraint["type"] not in UNPARAPHRASABLE_TYPES:
            continue
        for term in terms:
            if check(constraint, term)["pass"]:
                continue
            problems.append(
                f"{constraint.get('id')} ({constraint['type']}, "
                f"{constraint.get('params')}) rejects the name {term!r}, which the task "
                "names; a proper noun has no synonym, so no response that names that "
                "entity can pass")
    return problems


def task_reachability_report(task_text: str,
                             constraints: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Diagnostics for the task-vs-constraint half, including what is NOT decided.

    The ``undecided`` field is not decoration: it is the part of reachability
    the corpus does not prove, kept next to the part it does so that a reader
    cannot mistake one for the other.
    """
    terms = unavoidable_terms(task_text)
    return {
        "conflicts": task_conflicts(task_text, constraints),
        "unavoidable_terms": terms,
        "checked_types": [c.get("id") for c in constraints
                          if c["type"] in UNPARAPHRASABLE_TYPES],
        "undecided": ("whether the task's MEANING is expressible under this constraint "
                      "set. No mechanical procedure decides it; absence of a conflict "
                      "above is not a proof of answerability."),
    }


# ==========================================================================
# Layer 2 -- constructive witness
# ==========================================================================

def _usable(words: Iterable[str], info: Dict[str, Any]) -> List[str]:
    """Pool members that survive the banned letters / banned words / marks."""
    banned_letters = {letter for letter, _ in info["banned_letters"]}
    banned_words = info["banned_words"]
    banned_marks = info["banned_marks"]
    out: List[str] = []
    for word in words:
        low = word.casefold()
        if low in banned_words:
            continue
        if any(letter in low for letter in banned_letters):
            continue
        if any(mark in word for mark in banned_marks):
            continue
        out.append(word)
    return out


def _starts_with(words: Sequence[str], letter: Optional[str]) -> List[str]:
    if letter is None:
        return list(words)
    return [w for w in words if w.casefold().startswith(letter)]


def _terminator(info: Dict[str, Any]) -> Optional[str]:
    for mark in TERMINATORS:
        if mark not in info["banned_marks"]:
            return mark
    return None


def _sentence_templates() -> Tuple[Tuple[str, ...], ...]:
    """Slot orders to try, first slot first (it carries the initial letter)."""
    return (
        ("adj", "noun", "verb", "adv"),
        ("noun", "verb", "prep", "noun"),
        ("adj", "noun", "verb", "prep", "adj", "noun"),
        ("noun", "verb", "adv"),
    )


def build_witness(constraints: Sequence[Dict[str, Any]],
                  sentences: int = MIN_WITNESS_SENTENCES,
                  attempts: int = 4000) -> Optional[str]:
    """Construct a paragraph that passes EVERY constraint, or return ``None``.

    The returned text is verified with the real
    :func:`constraints.checkers.check` before it is handed back, so a non-None
    return is a proof and not a hope. ``None`` means "this builder could not do
    it", which for the corpus is treated as a failure to prove reachability --
    the generator refuses to emit such a scenario, and the test reports it.

    Fully deterministic: no RNG, no set iteration. The same constraint set
    always yields byte-identical prose, so a corpus rebuild is a no-op diff.
    """
    if sentences < MIN_WITNESS_SENTENCES:
        raise ValueError(f"a witness needs at least {MIN_WITNESS_SENTENCES} sentences")
    if find_conflicts(constraints):
        return None

    info = _summarise(constraints)
    terminator = _terminator(info)
    if terminator is None:
        return None

    initials = {letter for letter, _ in info["initial_letters"]}
    if len(initials) > 1:
        return None
    initial = next(iter(initials)) if initials else None

    pools = {
        "adj": _usable(ADJECTIVES, info),
        "noun": _usable(NOUNS, info),
        "verb": _usable(VERBS, info),
        "adv": _usable(ADVERBS, info),
        "prep": _usable(PREPOSITIONS, info),
    }
    if any(not pool for pool in pools.values()):
        return None

    templates = _sentence_templates()
    chosen: List[str] = []
    seen: Set[str] = set()

    for attempt in range(attempts):
        template = templates[attempt % len(templates)]
        openers = _starts_with(pools[template[0]], initial)
        if not openers:
            continue
        round_index = attempt // len(templates)
        words: List[str] = []
        for slot_index, slot in enumerate(template):
            pool = openers if slot_index == 0 else pools[slot]
            # Coprime-ish strides keep successive sentences from rhyming while
            # staying a pure function of (attempt, slot).
            step = (round_index * (13 + 6 * slot_index) + slot_index * 7) % len(pool)
            words.append(pool[step])
        sentence = " ".join(words)
        sentence = sentence[0].upper() + sentence[1:] + terminator
        if sentence in seen:
            continue
        candidate = " ".join(chosen + [sentence])
        if not _phrase_free(candidate, info["banned_phrases"]):
            continue
        chosen.append(sentence)
        seen.add(sentence)
        if len(chosen) >= sentences and len(" ".join(chosen)) >= MIN_WITNESS_CHARS:
            break

    if len(chosen) < sentences:
        return None
    text = " ".join(chosen)
    if len(text) < MIN_WITNESS_CHARS:
        # Never lower the floor to make a scenario build -- A7 makes short text
        # a vacuous proof. Failing here is the correct outcome.
        return None

    # The real checkers are the authority. If any of them disagrees with the
    # construction above, the witness is not a witness.
    for constraint in constraints:
        if not check(constraint, text)["pass"]:
            return None
    return text


def _phrase_free(text: str, phrases: Sequence[str]) -> bool:
    low = " ".join(text.casefold().split())
    return not any(phrase in low for phrase in phrases)


def witness_report(constraints: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """``{"conflicts": [...], "witness": str|None, "checks": [...]}`` for diagnostics."""
    conflicts = find_conflicts(constraints)
    witness = None if conflicts else build_witness(constraints)
    checks = ([{"constraint_id": c.get("id"), **check(c, witness)} for c in constraints]
              if witness else [])
    return {"conflicts": conflicts, "witness": witness, "checks": checks}
