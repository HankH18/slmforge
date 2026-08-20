#!/usr/bin/env python3
"""Deterministic constraint checkers -- the ground-truth layer.

Binding contract (docs/PIVOT_DESIGN.md, "Interfaces")::

    check(constraint, text) -> {"pass": bool, "evidence": str}

Every function here is PURE: no network, no model calls, no filesystem, no
randomness, no global mutable state, no caches. Calling a checker a thousand
times on the same input returns byte-identical results. ``evidence`` is either
the violating substring with enough surrounding context to locate it, or
exactly ``"clean"``.

Raising policy, stated once and applied everywhere:

* a malformed CONSTRAINT raises :class:`constraints.registry.ConstraintError` --
  a checker that silently "passes" a constraint it does not understand would
  inflate every downstream number;
* a hostile TEXT never raises. Empty, whitespace-only, 100k characters, lone
  surrogates, mixed scripts, astral emoji: all return a verdict.

The G1 prompt-ceiling gate, the training-data rejection filter and every
base-vs-tuned number are only as trustworthy as this file, so it is written for
an auditor rather than for brevity: explicit tables over clever regexes, one
named module-level constant per judgement call.

Ambiguity rulings
-----------------
Each ruling below is a named constant, greppable and cheap to flip. Hank rules
at the sign-off gate; these are defaults, not authority.

============ ============================================ =========================================
Ambiguity    Ruling                                       Constant
============ ============================================ =========================================
A1 words     whole-word only: "coffeehouse" does NOT      ``BANNED_WORDS_MATCH_MODE``
             violate "coffee"; neither does "coffees"
A2 letter    case-, diacritic- and homoglyph-insensitive; ``BANNED_LETTER_FOLD_DIACRITICS``,
             digits are never letters (no leetspeak)      ``FOLD_CONFUSABLES``, ``LEETSPEAK_DIGITS_ARE_LETTERS``
A3 punct     "anywhere" means anywhere, quotes included;  ``BANNED_PUNCTUATION_EXEMPT_QUOTED``,
             unicode variants of a listed mark count;     ``PUNCTUATION_LOOKALIKES``,
             unlisted marks (em-dash) never count         ``BANNED_PUNCTUATION_LISTED_MARKS_ONLY``
A4 quotes    double quotes only; the apostrophe in        ``COUNT_SINGLE_QUOTES``,
             "don't" is not a quotation mark              ``DOUBLE_QUOTATION_MARKS``
A5 sentence  abbreviation/decimal aware split; a newline  ``SENTENCE_ABBREVIATIONS``,
             ends a sentence; an unterminated tail is a   ``NEWLINE_TERMINATES_SENTENCE``,
             sentence; leading quotes/brackets are        ``TRAILING_FRAGMENT_IS_SENTENCE``,
             skipped but a leading DIGIT violates         ``SENTENCE_START_SKIPS_NON_ALNUM``
A6 lists     line-leading marker + space + content; a     ``LIST_BULLET_CHARS``,
             >=3-digit "1985." is prose, not a marker;    ``LIST_ORDINAL_MAX_DIGITS``,
             bare newline-separated lines are not a list  ``MIN_LIST_ITEMS``, ``BARE_LINES_ARE_LISTS``
A7 empty     empty/whitespace-only text passes every      ``EMPTY_TEXT_PASSES``,
             checker vacuously; the judge rejects it      ``NONE_TEXT_IS_EMPTY``
============ ============================================ =========================================
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple

from constraints.registry import ConstraintError, TYPES, normalize_constraint, validate_params

__all__ = [
    "check",
    "check_banned_words",
    "check_banned_phrases",
    "check_banned_letter",
    "check_banned_punctuation",
    "check_sentence_initial_letter",
    "check_no_lists",
    "check_no_quotation_marks",
    "CLEAN_EVIDENCE",
]

# ==========================================================================
# Evidence formatting
# ==========================================================================

#: The exact evidence string for a passing check. Pinned by the contract.
CLEAN_EVIDENCE = "clean"

#: Characters of original text shown either side of a violation.
EVIDENCE_CONTEXT_CHARS = 40

#: The violating span is wrapped in these so an auditor sees exactly what fired.
EVIDENCE_OPEN_MARK = "[["
EVIDENCE_CLOSE_MARK = "]]"

#: Evidence is a single line (it lands in JSONL transcripts), so whitespace runs
#: inside the window are collapsed to one space.
EVIDENCE_COLLAPSES_WHITESPACE = True

#: Unencodable/control characters in evidence are escaped as \uXXXX so a
#: transcript row can always be JSON-serialised.
EVIDENCE_ESCAPES_UNENCODABLE = True

# INVARIANT (not a knob -- there is nothing to flip): every checker reports the
# FIRST violation only, chosen by lowest offset with an explicit deterministic
# tie-break. Evidence is therefore a function of the input alone and never of
# dict or set iteration order. The transcript format (Design "Interfaces")
# stores one evidence string per constraint, so "all violations" has no
# consumer; do not add one speculatively.


def _safe(text: str) -> str:
    """Make a fragment safe to embed in a JSONL evidence string."""
    if not EVIDENCE_ESCAPES_UNENCODABLE:
        return text
    out: List[str] = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat in ("Cs", "Cc", "Cn"):          # surrogate / control / unassigned
            out.append("\\u%04x" % ord(ch))
            continue
        try:
            ch.encode("utf-8")
        except UnicodeEncodeError:
            out.append("\\u%04x" % ord(ch))
            continue
        out.append(ch)
    return "".join(out)


def _window(text: str, start: int, end: int) -> str:
    """``...before [[match]] after...`` around ``text[start:end]``."""
    start = max(0, min(start, len(text)))
    end = max(start, min(end, len(text)))
    left = max(0, start - EVIDENCE_CONTEXT_CHARS)
    right = min(len(text), end + EVIDENCE_CONTEXT_CHARS)
    before, match, after = text[left:start], text[start:end], text[end:right]
    if EVIDENCE_COLLAPSES_WHITESPACE:
        before = " ".join(before.split())
        match = " ".join(match.split())
        after = " ".join(after.split())
        if text[left:start] and text[left:start][-1].isspace():
            before += " "
        if text[end:right] and text[end:right][0].isspace():
            after = " " + after
    lead = "..." if left > 0 else ""
    tail = "..." if right < len(text) else ""
    return (f"{lead}{_safe(before)}{EVIDENCE_OPEN_MARK}{_safe(match)}"
            f"{EVIDENCE_CLOSE_MARK}{_safe(after)}{tail}")


def _fail(label: str, text: str, start: int, end: int) -> Dict[str, Any]:
    return {"pass": False, "evidence": f"{label} at offset {start}: {_window(text, start, end)}"}


def _clean() -> Dict[str, Any]:
    return {"pass": True, "evidence": CLEAN_EVIDENCE}


# ==========================================================================
# A7 -- empty input
# ==========================================================================

#: An empty or whitespace-only response VACUOUSLY PASSES every checker: there is
#: no violating substring to point at, so there is no violation. This is a
#: deliberate separation of concerns -- the checkers measure constraint
#: adherence only, and the task-quality judge (>=3, Design "Metrics") is what
#: rejects a model that games the constraints by saying nothing. Do not turn the
#: checkers into a quality gate; ``spec_pass`` already ANDs the two.
EMPTY_TEXT_PASSES = True

#: ``None`` is treated as an empty response (a model that returned nothing),
#: not as a caller bug. Any other non-str raises TypeError, because silently
#: ``str()``-ing a dict or a list would check the repr of a bug.
NONE_TEXT_IS_EMPTY = True


#: Evidence reported when EMPTY_TEXT_PASSES is flipped off.
EMPTY_TEXT_EVIDENCE = "empty response: no text was produced to satisfy the constraint"


def _empty_verdict(text: str) -> Optional[Dict[str, Any]]:
    """Shared A7 gate. Returns a verdict for empty/whitespace-only text, else None.

    Every checker routes empty input through here so the ruling has exactly one
    implementation and flipping ``EMPTY_TEXT_PASSES`` actually changes behaviour.
    """
    if text.strip():
        return None
    if EMPTY_TEXT_PASSES:
        return _clean()
    return {"pass": False, "evidence": EMPTY_TEXT_EVIDENCE}


def _coerce_text(text: Any) -> str:
    if text is None and NONE_TEXT_IS_EMPTY:
        return ""
    if not isinstance(text, str):
        raise TypeError(f"text must be a string (or None), got {type(text).__name__}")
    return text


# ==========================================================================
# A2 -- unicode folding
#
# One folding pipeline, shared by the three text-matching checkers
# (banned_words, banned_phrases, banned_letter) so they cannot disagree about
# what a character "is":
#
#     NFKD (or NFKC when diacritic folding is off)
#       -> drop combining marks        [BANNED_LETTER_FOLD_DIACRITICS]
#       -> map homoglyphs to Latin     [FOLD_CONFUSABLES]
#       -> casefold                    [per-call]
#
# The fold is index-mapped: every folded character remembers which ORIGINAL
# character produced it, so evidence always quotes the user's real text rather
# than the normalized form.
# ==========================================================================

#: NFKD is applied to every character. It is what makes fullwidth forms
#: (U+FF43 U+FF4F U+FF46 ... rendering as 'coffee') and mathematical
#: alphanumerics (U+1D400 and friends) fold onto plain ASCII for free.
UNICODE_NORMALIZATION = "NFKD"

#: A2: "cafe" with an acute accent still contains the letter e; a lipogram
#: author who writes "protege" with accents has not avoided the letter. Combining
#: marks are stripped after decomposition.
BANNED_LETTER_FOLD_DIACRITICS = True

#: A2: a Cyrillic "Ðµ" (U+0435) inside an otherwise-Latin English word is an
#: evasion, not a word in another language, and a human grader reading the
#: rendered text calls it the letter e. Homoglyphs listed in CONFUSABLE_TO_LATIN
#: are mapped to their Latin twin.
FOLD_CONFUSABLES = True

#: A2: digits are NOT letters. "c0ffee" does not violate a ban on "coffee" and,
#: far more importantly, "2020" must not violate a ban on the letter o. Mapping
#: leetspeak digits would make every year and quantity in the corpus a
#: violation, so it stays off.
LEETSPEAK_DIGITS_ARE_LETTERS = False

#: Homoglyph -> Latin. Cyrillic and Greek characters that render identically (or
#: near-identically) to a Latin letter in ordinary body text. Case is preserved
#: so the table also works with casefolding turned off. Deliberately excludes
#: anything that merely *resembles* a letter at a squint (no leetspeak digits,
#: no mathematical operators).
CONFUSABLE_TO_LATIN: Dict[str, str] = {
    # --- Cyrillic ---
    "а": "a", "А": "A",   # a / A
    "В": "B",                  # B
    "с": "c", "С": "C",   # c / C
    "е": "e", "Е": "E",   # e / E
    "Н": "H",                  # H
    "і": "i", "І": "I",   # i / I
    "ј": "j", "Ј": "J",   # j / J
    "к": "k", "К": "K",   # k / K
    "М": "M",                  # M
    "о": "o", "О": "O",   # o / O
    "р": "p", "Р": "P",   # p / P
    "ѕ": "s", "Ѕ": "S",   # s / S
    "Т": "T",                  # T
    "у": "y", "У": "Y",   # y / Y
    "х": "x", "Х": "X",   # x / X
    "ԁ": "d",                  # d
    # --- Greek ---
    "α": "a", "Α": "A",   # alpha / Alpha
    "Β": "B",                  # Beta
    "ε": "e", "Ε": "E",   # epsilon / Epsilon
    "Ζ": "Z",                  # Zeta
    "Η": "H",                  # Eta
    "Ι": "I",                  # Iota
    "κ": "k", "Κ": "K",   # kappa / Kappa
    "Μ": "M",                  # Mu
    "Ν": "N",                  # Nu
    "ο": "o", "Ο": "O",   # omicron / Omicron
    "ρ": "p", "Ρ": "P",   # rho / Rho
    "τ": "t", "Τ": "T",   # tau / Tau
    "Υ": "Y",                  # Upsilon
    "χ": "x", "Χ": "X",   # chi / Chi
    "ϲ": "c",                  # lunate sigma
    "ν": "v",                  # nu
    # --- Latin-script lookalikes ---
    "ı": "i",                  # dotless i
    "ɡ": "g",                  # script g
}


def _fold_char(ch: str, casefold: bool) -> str:
    """Fold ONE original character; may return 0, 1 or several characters."""
    form = UNICODE_NORMALIZATION if BANNED_LETTER_FOLD_DIACRITICS else "NFKC"
    try:
        piece = unicodedata.normalize(form, ch)
    except (TypeError, ValueError):          # defensive: never raise on text
        piece = ch
    if BANNED_LETTER_FOLD_DIACRITICS:
        piece = "".join(c for c in piece if not unicodedata.combining(c))
    if FOLD_CONFUSABLES:
        piece = "".join(CONFUSABLE_TO_LATIN.get(c, c) for c in piece)
    if casefold:
        piece = piece.casefold()
    return piece


def _fold(text: str, casefold: bool = True) -> Tuple[str, Optional[List[int]]]:
    """Return ``(folded_text, index_map)``.

    ``index_map[i]`` is the index in ``text`` of the character that produced
    ``folded_text[i]``; ``None`` means the identity map. Pure ASCII text takes
    the identity fast path -- ASCII casefolding is 1:1 and length preserving --
    which keeps a 100k-character response cheap.
    """
    if text.isascii():
        return (text.casefold() if casefold else text), None
    out: List[str] = []
    imap: List[int] = []
    for i, ch in enumerate(text):
        piece = _fold_char(ch, casefold)
        for c in piece:
            out.append(c)
            imap.append(i)
    return "".join(out), imap


def _orig_start(imap: Optional[List[int]], idx: int) -> int:
    return idx if imap is None else imap[idx]


def _orig_end(imap: Optional[List[int]], idx: int) -> int:
    """Exclusive original end offset for a folded exclusive end ``idx``."""
    if imap is None:
        return idx
    if idx <= 0:
        return 0
    return imap[idx - 1] + 1


# ==========================================================================
# A1 -- banned_words / banned_phrases
# ==========================================================================

#: A1 ruling. "must not use any listed word" bans the WORD, not the letter
#: sequence: "coffeehouse" and "scoffee" do not violate a ban on "coffee", the
#: same way a ban on "the" must not fire on "theory". A word boundary here is
#: any character that is not a unicode letter or digit -- so "coffee-house",
#: "coffee_break" and "(coffee)" all DO violate.
#:
#: Known consequence, flagged for sign-off: the inflected form "coffees" also
#: passes. The mitigation available to the scenario generator is to list the
#: inflections in ``params["words"]``. Flip this to "substring" to catch them at
#: the cost of "theory"-style false positives.
BANNED_WORDS_MATCH_MODE = "word_boundary"      # "word_boundary" | "substring"

#: Guard: a typo in the mode above must be loud. Silently degrading to the
#: default would mean the constant no longer describes what the checker does.
BANNED_WORDS_MATCH_MODES = ("word_boundary", "substring")

#: Both lexical checkers are case-insensitive (plain-English spec) and share the
#: unicode fold above.
BANNED_WORDS_CASE_INSENSITIVE = True

#: A1, phrases: "a contiguous run of words" -- words separated by whitespace
#: only. "in conclusion" fires on "In Conclusion" and across a line break, but
#: not on "in, conclusion" (intervening punctuation) and not inside a larger
#: word at either end.
PHRASE_WORD_SEPARATOR = r"\s+"

#: The word-boundary lookarounds. ``[^\W_]`` is "unicode letter or digit",
#: i.e. underscore counts as a boundary.
_WORD_CHAR = r"[^\W_]"
_LEFT_BOUNDARY = r"(?<!" + _WORD_CHAR + r")"
_RIGHT_BOUNDARY = r"(?!" + _WORD_CHAR + r")"


def _check_match_mode() -> None:
    if BANNED_WORDS_MATCH_MODE not in BANNED_WORDS_MATCH_MODES:
        raise ConstraintError(
            f"BANNED_WORDS_MATCH_MODE is {BANNED_WORDS_MATCH_MODE!r}; "
            f"expected one of {list(BANNED_WORDS_MATCH_MODES)}")


def _pattern_for_word(folded_word: str) -> str:
    _check_match_mode()
    if BANNED_WORDS_MATCH_MODE == "substring":
        return re.escape(folded_word)
    return _LEFT_BOUNDARY + re.escape(folded_word) + _RIGHT_BOUNDARY


def _pattern_for_phrase(folded_phrase: str) -> str:
    _check_match_mode()
    words = folded_phrase.split()
    if not words:
        return ""
    body = PHRASE_WORD_SEPARATOR.join(re.escape(w) for w in words)
    if BANNED_WORDS_MATCH_MODE == "substring":
        return body
    return _LEFT_BOUNDARY + body + _RIGHT_BOUNDARY


def _first_lexical_hit(text: str, needles: Sequence[str], make_pattern) -> Optional[Tuple[int, int, str]]:
    """Earliest match over ``needles``. Returns (orig_start, orig_end, needle).

    Deterministic tie-breaking: lowest folded start, then longest match, then
    the needle's position in the constraint's list.
    """
    folded, imap = _fold(text, casefold=BANNED_WORDS_CASE_INSENSITIVE)
    best: Optional[Tuple[int, int, int, str]] = None       # (start, -len, order, needle)
    for order, needle in enumerate(needles):
        folded_needle, _ = _fold(needle, casefold=BANNED_WORDS_CASE_INSENSITIVE)
        pattern = make_pattern(folded_needle)
        if not pattern:
            continue
        match = re.search(pattern, folded)
        if match is None:
            continue
        candidate = (match.start(), -(match.end() - match.start()), order, needle)
        if best is None or candidate[:3] < best[:3]:
            best = candidate
    if best is None:
        return None
    start_f, neg_len, _order, needle = best
    end_f = start_f - neg_len
    return _orig_start(imap, start_f), _orig_end(imap, end_f), needle


def check_banned_words(constraint: Dict[str, Any], text: str) -> Dict[str, Any]:
    """No listed word may be used. Case-insensitive, whole-word (A1)."""
    text = _coerce_text(text)
    validate_params("banned_words", constraint.get("params"))
    words = constraint["params"]["words"]
    empty = _empty_verdict(text)
    if empty is not None:
        return empty
    hit = _first_lexical_hit(text, words, _pattern_for_word)
    if hit is None:
        return _clean()
    start, end, word = hit
    return _fail(f"banned word {word!r}", text, start, end)


def check_banned_phrases(constraint: Dict[str, Any], text: str) -> Dict[str, Any]:
    """No listed phrase may appear as a contiguous run of words. Case-insensitive."""
    text = _coerce_text(text)
    validate_params("banned_phrases", constraint.get("params"))
    phrases = constraint["params"]["phrases"]
    empty = _empty_verdict(text)
    if empty is not None:
        return empty
    hit = _first_lexical_hit(text, phrases, _pattern_for_phrase)
    if hit is None:
        return _clean()
    start, end, phrase = hit
    return _fail(f"banned phrase {phrase!r}", text, start, end)


# ==========================================================================
# A2 -- banned_letter
# ==========================================================================

def check_banned_letter(constraint: Dict[str, Any], text: str) -> Dict[str, Any]:
    """The letter must not appear anywhere, including inside words (A2)."""
    text = _coerce_text(text)
    validate_params("banned_letter", constraint.get("params"))
    letter = constraint["params"]["letter"]
    empty = _empty_verdict(text)
    if empty is not None:
        return empty
    folded, imap = _fold(text, casefold=True)
    folded_letter, _ = _fold(letter, casefold=True)
    if not folded_letter:
        return _clean()
    idx = folded.find(folded_letter)
    if idx < 0:
        return _clean()
    start = _orig_start(imap, idx)
    end = _orig_end(imap, idx + len(folded_letter))
    return _fail(f"banned letter {letter!r}", text, start, end)


# ==========================================================================
# A3 -- banned_punctuation
# ==========================================================================

#: A3: the spec says "may not appear ANYWHERE". A comma inside quoted speech,
#: inside a number ("1,000") or inside a URL is still a comma on the page, and a
#: human grader counting commas counts those. No exemption.
#:
#: The exemption is implemented, not merely described: flip this to True and
#: marks strictly inside a double-quoted span stop counting. Quoting a source
#: verbatim is the one case where a writer cannot control the punctuation, so
#: the flip is a coherent alternative reading -- it is False because the pinned
#: plain-English spec says "anywhere", not because the alternative is absurd.
BANNED_PUNCTUATION_EXEMPT_QUOTED = False

#: A3: only the marks the constraint LISTS are banned. When ``marks`` is
#: ``[",", ";"]`` an em-dash, a colon and an ellipsis are all legal -- the
#: checker never generalises to "punctuation" as a family. Widening the ban is
#: the scenario author's job, done by listing more marks.
BANNED_PUNCTUATION_LISTED_MARKS_ONLY = True

#: A3: unicode variants of a listed mark DO count. Two mechanisms, both
#: explicit: (1) the fullwidth form of any printable ASCII mark, derived by the
#: standard +0xFEE0 offset; (2) this lookalike table for variants that are not
#: fullwidth forms. A fullwidth comma is a comma.
PUNCTUATION_LOOKALIKES: Dict[str, Tuple[str, ...]] = {
    ",": ("،",    # Arabic comma
          "、",    # ideographic comma
          "﹐",    # small comma
          "︐"),   # presentation form comma
    ";": ("؛",    # Arabic semicolon
          ";",    # Greek question mark (renders as a semicolon)
          "﹔",    # small semicolon
          "︔"),
    ":": ("﹕", "︓", "∶"),
    ".": ("。",    # ideographic full stop
          "｡", "﹒"),
    "!": ("﹗",),
    "?": ("؟",    # Arabic question mark
          "﹖"),
    "-": ("‐",    # hyphen
          "‑",    # non-breaking hyphen
          "−"),   # minus sign
    "(": ("﹙", "⦅"),
    ")": ("﹚", "⦆"),
    "'": ("‘", "’", "ʼ"),
    '"': ("“", "”", "„", "‟"),
    "/": ("⁄", "∕"),
    "&": ("﹠",),
    "%": ("﹪",),
    "@": ("﹫",),
    "#": ("﹟",),
    "*": ("﹡", "∗"),
    "+": ("﹢",),
    "=": ("﹦",),
}

#: The fullwidth block sits exactly 0xFEE0 above printable ASCII.
_FULLWIDTH_OFFSET = 0xFEE0


def _variants_of(mark: str) -> Tuple[str, ...]:
    """Every character that counts as ``mark``. Deterministic order."""
    seen: List[str] = [mark]
    if len(mark) == 1 and 0x21 <= ord(mark) <= 0x7E:
        fullwidth = chr(ord(mark) + _FULLWIDTH_OFFSET)
        if fullwidth not in seen:
            seen.append(fullwidth)
    for variant in PUNCTUATION_LOOKALIKES.get(mark, ()):
        if variant not in seen:
            seen.append(variant)
    return tuple(seen)


def _quoted_spans(text: str) -> List[Tuple[int, int]]:
    """Half-open spans strictly INSIDE double-quoted regions.

    A span opens at any character in DOUBLE_QUOTATION_MARKS and closes at the
    next one; an unclosed quote runs to the end of the text. The quote
    characters themselves are not inside the span. Only consulted when
    BANNED_PUNCTUATION_EXEMPT_QUOTED is on.
    """
    quotes = set(DOUBLE_QUOTATION_MARKS)
    spans: List[Tuple[int, int]] = []
    open_at: Optional[int] = None
    for idx, ch in enumerate(text):
        if ch not in quotes:
            continue
        if open_at is None:
            open_at = idx
        else:
            spans.append((open_at + 1, idx))
            open_at = None
    if open_at is not None:
        spans.append((open_at + 1, len(text)))
    return spans


def check_banned_punctuation(constraint: Dict[str, Any], text: str) -> Dict[str, Any]:
    """None of the listed marks may appear anywhere (A3)."""
    text = _coerce_text(text)
    validate_params("banned_punctuation", constraint.get("params"))
    marks = constraint["params"]["marks"]
    empty = _empty_verdict(text)
    if empty is not None:
        return empty
    # char -> the listed mark it stands for; built per call, no global state.
    banned: Dict[str, str] = {}
    for mark in marks:
        for variant in _variants_of(mark):
            banned.setdefault(variant, mark)
    exempt = _quoted_spans(text) if BANNED_PUNCTUATION_EXEMPT_QUOTED else []
    for idx, ch in enumerate(text):
        if ch in banned:
            if any(start <= idx < end for start, end in exempt):
                continue
            mark = banned[ch]
            label = (f"banned punctuation {mark!r}" if ch == mark
                     else f"banned punctuation {mark!r} (unicode variant {ch!r})")
            return _fail(label, text, idx, idx + 1)
    return _clean()


# ==========================================================================
# A4 -- no_quotation_marks
# ==========================================================================

#: A4: these always count. Double quotes in every common form, including the
#: French guillemets and the CJK/fullwidth variants.
DOUBLE_QUOTATION_MARKS: Tuple[str, ...] = (
    '"',            # U+0022 quotation mark
    "“",       # left double quotation mark
    "”",       # right double quotation mark
    "„",       # double low-9
    "‟",       # double high-reversed-9
    "«",       # left guillemet
    "»",       # right guillemet
    "＂",       # fullwidth quotation mark
    "〝",       # reversed double prime quotation mark
    "〞",       # double prime quotation mark
    "〟",       # low double prime quotation mark
    "❝",       # heavy double turned comma quotation mark ornament
    "❞",       # heavy double comma quotation mark ornament
)

#: A4 ruling. Single quotes do NOT count. The apostrophe in "don't", "it's",
#: "o'clock" and "the dogs' bowl" is an apostrophe, not a quotation mark, and a
#: checker that failed a model for writing "don't" would be obviously broken to
#: any human grader. Straight and curly are treated alike in both directions:
#: every double form fires, no single form does.
#:
#: Known consequence, flagged for sign-off: 'single-quoted' text passes. Flip
#: COUNT_SINGLE_QUOTES to True to ban them too -- at which point "don't" fails,
#: which is why it is False.
COUNT_SINGLE_QUOTES = False

SINGLE_QUOTATION_MARKS: Tuple[str, ...] = (
    "'",            # U+0027 apostrophe
    "‘",       # left single quotation mark
    "’",       # right single quotation mark / curly apostrophe
    "‚",       # single low-9
    "‛",       # single high-reversed-9
    "‹",       # left single guillemet
    "›",       # right single guillemet
    "＇",       # fullwidth apostrophe
)


def check_no_quotation_marks(constraint: Dict[str, Any], text: str) -> Dict[str, Any]:
    """No quotation marks (A4: doubles only; apostrophes are legal)."""
    text = _coerce_text(text)
    validate_params("no_quotation_marks", constraint.get("params"))
    empty = _empty_verdict(text)
    if empty is not None:
        return empty
    banned = set(DOUBLE_QUOTATION_MARKS)
    if COUNT_SINGLE_QUOTES:
        banned |= set(SINGLE_QUOTATION_MARKS)
    for idx, ch in enumerate(text):
        if ch in banned:
            return _fail(f"quotation mark {ch!r}", text, idx, idx + 1)
    return _clean()


# ==========================================================================
# A5 -- sentence_initial_letter
# ==========================================================================

#: Characters that end a sentence. '.' is the overloaded one and gets the extra
#: guards below; '!', '?' and the ellipsis character are unambiguous.
SENTENCE_TERMINATORS = ".!?…"

# A5, ellipses: '\u2026' is in SENTENCE_TERMINATORS above, and so is '.', so
# both "Wait..." and "Wait\u2026" end a sentence -- the text after them is a new
# sentence that must obey the rule. A RUN of terminators ("...", "?!") is one
# boundary. That is stated as an invariant rather than offered as a switch
# because the alternative is not observable: splitting a run into several
# boundaries only manufactures extra segments that contain no letters, and
# EMPTY_SEGMENT_IS_SENTENCE already skips those. A knob nothing can flip is a
# lie about the code, so there isn't one.

#: A5: closing punctuation after the terminator belongs to the sentence that
#: just ended, so ``She said "stop." Then...`` splits after the quote.
SENTENCE_TRAILERS = "\"'”’)]}»›"

#: A5: a line break ends a sentence. Models emit unwrapped prose, so a newline
#: is an author-intended boundary rather than soft wrapping; without this rule a
#: headline or a line of a stanza would be silently glued to the next sentence
#: and never checked.
NEWLINE_TERMINATES_SENTENCE = True

#: A5: the last sentence often has no terminator ("...and that is why"). It is
#: still a sentence and still must obey the rule.
TRAILING_FRAGMENT_IS_SENTENCE = True

#: A5: a '.' that is followed immediately by a non-space, non-closing character
#: is not a sentence end. Covers "3.5", "example.com", "U.S.A." and "1.5x"
#: without a URL regex.
PERIOD_REQUIRES_TRAILING_SPACE = True

#: A5: a '.' between two digits is a decimal point.
DECIMAL_POINT_IS_NOT_A_TERMINATOR = True

#: A5: tokens that end in '.' without ending the sentence. Compared lowercased,
#: including the dot. Any token that already contains a '.' (``e.g.``, ``U.S.``,
#: ``a.m.``) and any single-letter initial (``J. Smith``) are also treated as
#: abbreviations by rule, so this list only needs the ordinary ones.
SENTENCE_ABBREVIATIONS: Tuple[str, ...] = (
    "mr.", "mrs.", "ms.", "dr.", "prof.", "st.", "mt.", "rev.", "hon.",
    "jr.", "sr.", "gen.", "col.", "capt.", "lt.", "sgt.", "gov.", "sen.", "rep.",
    "etc.", "vs.", "approx.", "est.", "no.", "fig.", "vol.", "dept.", "inc.",
    "ltd.", "co.", "corp.", "univ.", "ave.", "blvd.", "rd.", "sq.",
    "jan.", "feb.", "mar.", "apr.", "jun.", "jul.", "aug.",
    "sep.", "sept.", "oct.", "nov.", "dec.",
    "mon.", "tue.", "tues.", "wed.", "thu.", "thurs.", "fri.", "sat.", "sun.",
)

#: A5: leading characters that are neither letters nor digits are skipped before
#: reading a sentence's first character, so ``"Something," she said.`` and
#: ``(Something happened.)`` both begin with S. A leading DIGIT is NOT skipped:
#: "1985 was the year" does not begin with the letter S, and a grader would mark
#: it a violation rather than looking past it.
SENTENCE_START_SKIPS_NON_ALNUM = True

#: A5: a segment with no letters or digits at all (a lone "---" or "...") is not
#: a sentence and is not checked.
EMPTY_SEGMENT_IS_SENTENCE = False

_SINGLE_INITIAL_RE = re.compile(r"^[^\W\d_]\.$", re.UNICODE)


def _preceding_token(text: str, end: int) -> str:
    """The whitespace-delimited token ending at ``end`` (exclusive), lowercased."""
    start = end
    while start > 0 and not text[start - 1].isspace():
        start -= 1
    return text[start:end].lower()


def _is_sentence_end(text: str, start: int, run_end: int) -> bool:
    """Does the terminator run ``text[start:run_end]`` really end a sentence?"""
    run = text[start:run_end]
    if run != ".":
        # '!', '?', the ellipsis character, and any multi-character run always
        # terminate: none of them are overloaded the way a lone '.' is.
        return True
    # --- single '.' ---
    if DECIMAL_POINT_IS_NOT_A_TERMINATOR:
        prev_ch = text[start - 1] if start > 0 else ""
        next_ch = text[run_end] if run_end < len(text) else ""
        if prev_ch.isdigit() and next_ch.isdigit():
            return False
    if PERIOD_REQUIRES_TRAILING_SPACE and run_end < len(text):
        next_ch = text[run_end]
        if not next_ch.isspace() and next_ch not in SENTENCE_TRAILERS:
            return False
    token = _preceding_token(text, run_end)
    if token in SENTENCE_ABBREVIATIONS:
        return False
    if _SINGLE_INITIAL_RE.match(token):
        return False
    if token.count(".") > 1:            # "u.s.", "e.g.", "a.m."
        return False
    return True


def _sentence_spans(text: str) -> List[Tuple[int, int]]:
    """Sentence spans ``(start, end)`` over the ORIGINAL text (A5)."""
    spans: List[Tuple[int, int]] = []
    n = len(text)
    i = 0
    seg_start: Optional[int] = None
    while i < n:
        ch = text[i]
        if seg_start is None and not ch.isspace():
            seg_start = i
        if ch == "\n" and NEWLINE_TERMINATES_SENTENCE:
            if seg_start is not None:
                spans.append((seg_start, i))
                seg_start = None
            i += 1
            continue
        if ch in SENTENCE_TERMINATORS:
            run_end = i
            while run_end < n and text[run_end] in SENTENCE_TERMINATORS:
                run_end += 1
            if _is_sentence_end(text, i, run_end):
                end = run_end
                while end < n and text[end] in SENTENCE_TRAILERS:
                    end += 1
                if seg_start is not None:
                    spans.append((seg_start, end))
                    seg_start = None
                i = end
                continue
            i = run_end
            continue
        i += 1
    if seg_start is not None and TRAILING_FRAGMENT_IS_SENTENCE:
        spans.append((seg_start, n))
    return spans


def _first_significant(text: str, start: int, end: int) -> int:
    """Index of the character a sentence "begins with", or -1 (A5)."""
    for idx in range(start, end):
        ch = text[idx]
        if ch.isalnum():
            return idx
        if not SENTENCE_START_SKIPS_NON_ALNUM and not ch.isspace():
            return idx
    return -1


def check_sentence_initial_letter(constraint: Dict[str, Any], text: str) -> Dict[str, Any]:
    """Every sentence must begin with the given letter. Case-insensitive (A5)."""
    text = _coerce_text(text)
    validate_params("sentence_initial_letter", constraint.get("params"))
    letter = constraint["params"]["letter"]
    empty = _empty_verdict(text)
    if empty is not None:
        return empty
    folded_letter, _ = _fold(letter, casefold=True)
    for ordinal, (start, end) in enumerate(_sentence_spans(text), start=1):
        idx = _first_significant(text, start, end)
        if idx < 0:
            if EMPTY_SEGMENT_IS_SENTENCE:
                return _fail(f"sentence {ordinal} has no initial letter (expected {letter!r})",
                             text, start, end)
            continue
        folded_ch, _ = _fold(text[idx], casefold=True)
        if not folded_ch.startswith(folded_letter):
            return _fail(f"sentence {ordinal} begins with {text[idx]!r}, not {letter!r}",
                         text, idx, min(end, idx + 1))
    return _clean()


# ==========================================================================
# A6 -- no_lists
# ==========================================================================

#: A6: characters that open a bullet when they START a line and are followed by
#: whitespace and then content. The space requirement is what keeps "-5 degrees"
#: and "*emphasis*" out. En- and em-dashes are included because models really do
#: bullet with them; a line that opens "- " or ">- " in prose is vanishingly
#: rare next to that.
LIST_BULLET_CHARS: Tuple[str, ...] = (
    "-", "*", "+",
    "•",   # bullet
    "‣",   # triangular bullet
    "◦",   # white bullet
    "▪",   # black small square
    "●",   # black circle
    "⁃",   # hyphen bullet
    "–",   # en dash
    "—",   # em dash
    "·",   # middle dot
    "∙",   # bullet operator
)

#: A6: an ordered marker is at most this many digits, which is what separates a
#: list item ("1. Preheat the oven") from a sentence opening with a year
#: ("1985. That year everything changed"). Two digits covers every realistic
#: enumeration.
LIST_ORDINAL_MAX_DIGITS = 2

#: A6: single-letter and roman-numeral markers ("a) ", "iv. ") count as ordered
#: lists too. This is the ruling most likely to bite -- a line that opens
#: "A. Lincoln said" is read as a list marker -- so it is its own switch.
LIST_ORDINAL_ALPHA = True

#: A6: one marker is enough. A response with a single "- item" line has used
#: list formatting, and a human grader marks it a violation of "no lists".
MIN_LIST_ITEMS = 1

#: A6: newline-separated fragments with no marker character are NOT a list.
#: Short lines are prose formatting, not enumeration, and inferring a list from
#: line breaks alone would fail every poem, address and dialogue.
BARE_LINES_ARE_LISTS = False

#: A6 support: markdown-style markers may be indented.
LIST_MARKER_MAY_BE_INDENTED = True

#: A6: a list ITEM has to say something. Without this, a thematic break
#: ("* * *", "- - -") reads as a bullet whose content is more bullets, and a
#: horizontal rule is not a list. The item text must contain a letter or digit.
LIST_ITEM_NEEDS_WORD_CONTENT = True

_ORDINAL_DIGIT_RE = re.compile(
    r"^\(?(\d{1,%d})[.)\]]" % LIST_ORDINAL_MAX_DIGITS)
_ORDINAL_ALPHA_RE = re.compile(
    r"^\(?([A-Za-z]|[ivxIVX]{1,4})[.)\]]")


def _is_list_item_body(rest: str) -> bool:
    """Does the text after a marker look like an item rather than more marker?"""
    if not rest.strip():
        return False
    if LIST_ITEM_NEEDS_WORD_CONTENT and not any(ch.isalnum() for ch in rest):
        return False
    return True


def _list_marker_at(line: str) -> Optional[Tuple[int, int, str]]:
    """``(offset_in_line, marker_len, description)`` for a list marker, else None."""
    stripped = line.lstrip()
    offset = len(line) - len(stripped)
    if offset and not LIST_MARKER_MAY_BE_INDENTED:
        return None
    if not stripped:
        return None
    # --- bullet ---
    if stripped[0] in LIST_BULLET_CHARS:
        rest = stripped[1:]
        if rest[:1].isspace() and _is_list_item_body(rest):
            return offset, 1, f"bullet {stripped[0]!r}"
    # --- ordered ---
    for regex, kind in ((_ORDINAL_DIGIT_RE, "numbered"),
                        (_ORDINAL_ALPHA_RE, "lettered") if LIST_ORDINAL_ALPHA else (None, "")):
        if regex is None:
            continue
        match = regex.match(stripped)
        if match is None:
            continue
        rest = stripped[match.end():]
        if rest[:1].isspace() and _is_list_item_body(rest):
            return offset, match.end(), f"{kind} marker {match.group(0)!r}"
    return None


def check_no_lists(constraint: Dict[str, Any], text: str) -> Dict[str, Any]:
    """No bulleted or numbered list (A6)."""
    text = _coerce_text(text)
    validate_params("no_lists", constraint.get("params"))
    empty = _empty_verdict(text)
    if empty is not None:
        return empty
    hits: List[Tuple[int, int, str, int]] = []
    line_start = 0
    for lineno, line in enumerate(text.split("\n"), start=1):
        marker = _list_marker_at(line)
        if marker is not None:
            offset, length, description = marker
            hits.append((line_start + offset, length, description, lineno))
        line_start += len(line) + 1
    if len(hits) < MIN_LIST_ITEMS:
        return _clean()
    start, length, description, lineno = hits[0]
    return _fail(f"list {description} on line {lineno}", text, start, start + length)


# ==========================================================================
# Dispatcher
# ==========================================================================

# The registry names each type's checker as a string; prove at import that
# every name resolves here, so a rename can never leave a type dispatching into
# a KeyError at eval time.
for _spec in TYPES.values():
    assert _spec.checker_name in globals(), \
        f"registry names checker {_spec.checker_name!r} but constraints.checkers has no such function"
del _spec


def check(constraint: Dict[str, Any], text: str) -> Dict[str, Any]:
    """The pinned entry point: ``check(constraint, text) -> {"pass", "evidence"}``.

    Validates the constraint against the registry (raising
    :class:`ConstraintError` on anything malformed) and dispatches to the one
    checker registered for its type. Pure.
    """
    normalized = normalize_constraint(constraint)
    checker = globals()[TYPES[normalized["type"]].checker_name]
    result = checker(normalized, text)
    return {"pass": bool(result["pass"]), "evidence": str(result["evidence"])}
