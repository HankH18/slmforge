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
A2 letter    case-, diacritic-, letterform- and            ``BANNED_LETTER_FOLD_DIACRITICS``,
             homoglyph-insensitive; invisible characters  ``FOLD_CONFUSABLES``, ``FOLD_LETTERFORMS``,
             are ignored; a symbol that merely            ``IGNORE_INVISIBLE_CHARACTERS``,
             DECOMPOSES into letters is not an            ``NFKD_MAY_MANUFACTURE_LETTERS``,
             occurrence of them; digits are never         ``LEETSPEAK_DIGITS_ARE_LETTERS``
             letters (no leetspeak)
A3 punct     "anywhere" means anywhere, quotes included;  ``BANNED_PUNCTUATION_EXEMPT_QUOTED``,
             unicode variants of a listed mark count;     ``PUNCTUATION_LOOKALIKES``
             unlisted marks (em-dash) never count -- an
             invariant, enforced by construction in
             ``check_banned_punctuation``, not a knob
A4 quotes    double quotes only; the apostrophe in        ``COUNT_SINGLE_QUOTES``,
             "don't" is not a quotation mark, but two     ``DOUBLE_QUOTATION_MARKS``,
             adjacent apostrophes are a typewriter one    ``ADJACENT_SINGLE_QUOTES_ARE_A_DOUBLE_QUOTE``
A5 sentence  abbreviation/decimal aware split; ANY line   ``SENTENCE_ABBREVIATIONS``,
             break ends a sentence; an unterminated tail  ``SENTENCE_ABBREVIATIONS_ALSO_ORDINARY_WORDS``,
             is a sentence; leading quotes/brackets are   ``SENTENCE_ABBREVIATIONS_CLAUSE_FINAL``,
             skipped but a leading DIGIT violates; a word ``SINGLE_INITIAL_REQUIRES_NAME_EVIDENCE``,
             that is both an abbreviation and an ordinary ``NEWLINE_TERMINATES_SENTENCE``,
             word ("no.", "etc.") ends the sentence in    ``SENTENCE_LINE_BREAK_CHARS``,
             its lowercase form before a capital; a       ``TRAILING_FRAGMENT_IS_SENTENCE``,
             one-letter token needs name evidence         ``SENTENCE_START_SKIPS_NON_ALNUM``
A6 lists     line-leading marker (after any blockquote    ``LIST_BULLET_CHARS``,
             prefix); >=2 markers make a list; a          ``LIST_BULLET_UNICODE_BLOCKS``,
             >=3-digit "1985." is prose, not a marker;    ``LIST_BULLET_CHARS_NEEDING_SEPARATOR``,
             bare newline-separated lines are never a     ``LIST_ORDINAL_MAX_DIGITS``,
             list (enforced invariant, not a knob)        ``MIN_LIST_ITEMS``
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
#     drop invisible characters       [IGNORE_INVISIBLE_CHARACTERS]
#       -> NFKD (or NFKC when diacritic folding is off)
#       -> refuse manufactured letters [NFKD_MAY_MANUFACTURE_LETTERS]
#       -> drop combining marks        [BANNED_LETTER_FOLD_DIACRITICS]
#       -> map letterforms to Latin    [FOLD_LETTERFORMS]
#       -> map homoglyphs to Latin     [FOLD_CONFUSABLES]
#       -> map leetspeak digits        [LEETSPEAK_DIGITS_ARE_LETTERS]
#       -> casefold                    [per-call]
#
# The fold is index-mapped: every folded character remembers which ORIGINAL
# character produced it, so evidence always quotes the user's real text rather
# than the normalized form. Dropping a character drops its map entry, so an
# evidence offset still points at the user's real bytes: the fold is never a
# lossy replace.
# ==========================================================================

#: NFKD is applied to every character. It is what makes fullwidth forms
#: (U+FF43 U+FF4F U+FF46 ... rendering as 'coffee') and mathematical
#: alphanumerics (U+1D400 and friends) fold onto plain ASCII for free.
UNICODE_NORMALIZATION = "NFKD"

# --------------------------------------------------------------------------
# A2, invisible characters
# --------------------------------------------------------------------------

#: A2: a character that renders as NOTHING cannot change what a human grader
#: reads. ``"cof​fee"`` is the word "coffee" on the page, so the invisible
#: codepoint is dropped before any matching happens. This is the cheapest
#: evasion available to a tuned model -- one zero-width codepoint defeats every
#: lexical constraint at no cost to fluency -- so it is closed for the whole
#: class, not per-character.
#:
#: THREE mechanisms, because no single one covers the class:
#:   (1) ``INVISIBLE_CATEGORIES`` -- general categories Cf (format) and Cc
#:       (control). Cf catches U+00AD, U+200B-U+200F, U+2060, U+FEFF and the
#:       bidi overrides. Cc is here because a C0/C1 control (U+0001, U+007F,
#:       U+0090) renders as nothing too and split a banned word exactly the way
#:       a zero-width space did -- same evasion, one category over. WHITESPACE
#:       IS EXEMPT (see ``_is_invisible``): '\n', '\t', '\v', '\f', '\r' and
#:       U+0085 are Cc but they SEPARATE, and dropping them would glue
#:       "tea\ncoffee" into one token and hide both words.
#:   (2) ``DEFAULT_IGNORABLE_RANGES`` -- the Default_Ignorable_Code_Point
#:       ranges. These include the variation selectors U+FE00-U+FE0F, which are
#:       category **Mn with canonical combining class 0**: the combining-mark
#:       filter below does NOT remove them and neither does (1). Separate hole,
#:       separate mechanism.
#:   (3) ``INVISIBLE_CHARS`` -- the singletons spelled out by name so an
#:       auditor can read the important ones without a unicode table.
IGNORE_INVISIBLE_CHARACTERS = True

#: (1) General categories whose members are invisible formatting or control.
INVISIBLE_CATEGORIES: Tuple[str, ...] = ("Cf", "Cc")

#: The EXEMPTION that makes admitting Cc safe: whitespace RENDERS -- as a gap
#: (Zs) or as a line break (Zl/Zp and the layout controls) -- so it separates
#: and is never "nothing". Dropping it would weld "tea\ncoffee" into one token
#: and hide BOTH words from every lexical checker.
#:
#: Spelled out rather than taken from ``str.isspace()`` ON PURPOSE: that
#: predicate also claims the C0 separators U+001C-U+001F, which render as
#: nothing in prose -- this file's own A5 line-break ruling already says they
#: are not line breaks -- and which therefore split a banned word exactly the
#: way a zero-width space does. They are invisible characters that Python calls
#: whitespace, and the invisible ruling is the one that applies to them.
WHITESPACE_CATEGORIES: Tuple[str, ...] = ("Zs", "Zl", "Zp")

#: Layout controls that render as a break or an advance. Category Cc, every one.
LAYOUT_WHITESPACE_CHARS: Tuple[str, ...] = (
    "\t",        # character tabulation
    "\n",        # line feed
    "\v",        # line tabulation
    "\f",        # form feed
    "\r",        # carriage return
    "\x85",      # NEXT LINE (NEL)
)

_LAYOUT_WHITESPACE_SET = frozenset(LAYOUT_WHITESPACE_CHARS)


def _renders_as_whitespace(ch: str) -> bool:
    """Does ``ch`` render as a gap or a break, rather than as nothing?"""
    return (ch in _LAYOUT_WHITESPACE_SET
            or unicodedata.category(ch) in WHITESPACE_CATEGORIES)

#: (2) Default_Ignorable_Code_Point ranges (Unicode 14.0), inclusive, one line
#: per range with what lives in it. The variation selectors and the Mongolian
#: free variation selectors are the entries category Cf does not cover.
DEFAULT_IGNORABLE_RANGES: Tuple[Tuple[int, int, str], ...] = (
    (0x00AD, 0x00AD, "SOFT HYPHEN"),
    (0x034F, 0x034F, "COMBINING GRAPHEME JOINER (Mn, ccc=0)"),
    (0x061C, 0x061C, "ARABIC LETTER MARK"),
    (0x115F, 0x1160, "HANGUL CHOSEONG/JUNGSEONG FILLER"),
    (0x17B4, 0x17B5, "KHMER VOWEL INHERENT AQ / AA"),
    (0x180B, 0x180F, "MONGOLIAN FREE VARIATION SELECTORS (Mn, ccc=0) + MVS"),
    (0x200B, 0x200F, "ZERO WIDTH SPACE .. RIGHT-TO-LEFT MARK"),
    (0x202A, 0x202E, "bidi embedding and override controls"),
    (0x2060, 0x206F, "WORD JOINER .. deprecated formatting characters"),
    (0x3164, 0x3164, "HANGUL FILLER"),
    (0xFE00, 0xFE0F, "VARIATION SELECTOR-1 .. -16 (Mn, ccc=0)"),
    (0xFEFF, 0xFEFF, "ZERO WIDTH NO-BREAK SPACE / byte order mark"),
    (0xFFA0, 0xFFA0, "HALFWIDTH HANGUL FILLER"),
    (0xFFF0, 0xFFF8, "unassigned specials, ignorable by property"),
    (0x1BCA0, 0x1BCA3, "SHORTHAND FORMAT controls"),
    (0x1D173, 0x1D17A, "MUSICAL SYMBOL BEGIN/END formatting"),
    (0xE0000, 0xE0FFF, "TAG characters and VARIATION SELECTOR-17 .. -256"),
)

#: (3) The named singletons. Every one of these is a real, observed evasion.
INVISIBLE_CHARS: Tuple[str, ...] = (
    "­",       # SOFT HYPHEN -- models emit it innocently, so it poisons data
    "͏",       # COMBINING GRAPHEME JOINER (Mn, ccc=0)
    "​",       # ZERO WIDTH SPACE
    "‌",       # ZERO WIDTH NON-JOINER
    "‍",       # ZERO WIDTH JOINER
    "⁠",       # WORD JOINER
    "︎",       # VARIATION SELECTOR-15 (text presentation)
    "️",       # VARIATION SELECTOR-16 (emoji presentation)
    "﻿",       # ZERO WIDTH NO-BREAK SPACE / BOM
)

_INVISIBLE_CHAR_SET = frozenset(INVISIBLE_CHARS)
_DEFAULT_IGNORABLE_CODEPOINTS = frozenset(
    cp for low, high, _name in DEFAULT_IGNORABLE_RANGES for cp in range(low, high + 1))


def _is_invisible(ch: str) -> bool:
    """Does ``ch`` render as nothing, and therefore never count?

    Whitespace is never "nothing": it renders as a gap and it SEPARATES. The
    exemption is load-bearing now that Cc is an invisible category -- '\n',
    '\t', '\v', '\f', '\r' and U+0085 are all category Cc, and dropping them
    would join the words on either side and hide both from every lexical
    checker. It also protects the line-break ruling in A5, which reads those
    same characters. It is ``_renders_as_whitespace``, not ``str.isspace()``:
    see WHITESPACE_CATEGORIES for why the two differ.
    """
    if _renders_as_whitespace(ch):
        return False
    if ch in _INVISIBLE_CHAR_SET:
        return True
    if ord(ch) in _DEFAULT_IGNORABLE_CODEPOINTS:
        return True
    return unicodedata.category(ch) in INVISIBLE_CATEGORIES


# --------------------------------------------------------------------------
# A2, manufactured letters
# --------------------------------------------------------------------------

#: The unicode general-category prefix that means "letter" (Lu/Ll/Lt/Lm/Lo).
#: Nl (ROMAN NUMERAL ONE) and No (SUPERSCRIPT TWO) are deliberately NOT letters.
LETTER_CATEGORY_PREFIX = "L"

#: A2: NFKD is a COMPATIBILITY decomposition, so left alone it expands symbols
#: into letters that were never on the page: U+2122 -> "TM", U+2116 -> "No",
#: U+2105 -> "c/o", U+2103 -> "degree C", U+2160 -> "I". 953 characters in
#: Unicode 14.0 have this property.
#:
#: Folding those manufactures violations out of thin air. That direction is the
#: expensive one: a false FAIL silently REJECTS a compliant training example and
#: depresses the measured base-model rate, and it is invisible in aggregate
#: because it looks like the model simply failing.
#:
#: RULING: a character folds to a letter only when the SOURCE IS that letter in
#: another form -- operationally, when the source is itself a unicode letter
#: (fullwidth 'c', mathematical bold 'A', the ligature 'ff' all are). A symbol
#: that merely DECOMPOSES into letters is not an occurrence of them, so it is
#: left as the original character -- which is not a word character either, so it
#: still acts as a word boundary. Flip this to True to restore literal NFKD.
NFKD_MAY_MANUFACTURE_LETTERS = False


def _is_letter(ch: str) -> bool:
    return unicodedata.category(ch).startswith(LETTER_CATEGORY_PREFIX)


#: A2, the NARROW exception to the ruling above, and the reason it is narrow.
#: The ruling refuses to let a SYMBOL manufacture letters. Applied to general
#: category alone it over-fires: an ENCLOSED letterform -- circled A, squared A,
#: parenthesized a -- is category So, not L, yet it is not a symbol that merely
#: decomposes into letters. It IS the letter A, drawn inside a box, and its own
#: unicode name says so: "CIRCLED LATIN CAPITAL LETTER A". A grader reading the
#: page reads an A, and "Ⓐpple" is the word apple.
#:
#: So the test is not "is the source category L" but "is the source AN
#: OCCURRENCE OF a letter": a letter, or a decorative enclosure of exactly one
#: named Latin letter. U+2122 TRADE MARK SIGN is neither -- its name is a sign,
#: not a letter -- so it still folds to nothing and "Zoom™" still passes a ban
#: on 't' (fx-character-19/20).
#:
#: DERIVATION (run once, pasted here as data): every codepoint of general
#: category So whose unicode NAME says it is one named Latin letter -- the name
#: ends in ``LATIN {CAPITAL,SMALL} LETTER <X>``, or is ``REGIONAL INDICATOR
#: SYMBOL LETTER <X>``, which is the same statement in the flag block's wording.
#: 213 characters in Unicode 14.0.
#:
#: THE DERIVATION USED TO REQUIRE AN NFKD DECOMPOSITION and that dropped 80 of
#: them on the floor: ``NEGATIVE CIRCLED``/``NEGATIVE SQUARED`` capitals and the
#: regional indicators have no decomposition at all, so "🅐pple" walked past a
#: ban on 'a' while "Ⓐpple" was correctly caught -- the identical ruling
#: enforced for one enclosure style and not the next. Having a decomposition is
#: a property of the unicode tables, not of what a reader sees on the page, so
#: it cannot be what decides. The letter is now read off the NAME for every
#: entry (``_ENCLOSED_LETTERFORM_TO_LATIN`` below), which is the same evidence
#: the ruling already claimed to use.
#:
#: COST ACCEPTED, and it is a real one: the regional indicators are the halves a
#: FLAG emoji is built from, so "🇺🇸" now folds to "US" and fails a ban on 'u' or
#: 's'. Nobody reads a flag as two letters. It is admitted anyway because its
#: own unicode name says LETTER U, because a single unpaired indicator renders
#: as a boxed capital, and because leaving it out leaves a one-codepoint
#: evasion open. Flagged for sign-off: if it is ruled the wrong way round,
#: delete the ``REGIONAL INDICATOR`` row -- nothing else depends on it.
ENCLOSED_LETTERFORM_RANGES: Tuple[Tuple[int, int, str], ...] = (
    (0x249C, 0x24B5, "PARENTHESIZED LATIN SMALL LETTER A..Z"),
    (0x24B6, 0x24CF, "CIRCLED LATIN CAPITAL LETTER A..Z"),
    (0x24D0, 0x24E9, "CIRCLED LATIN SMALL LETTER A..Z"),
    (0x1F110, 0x1F129, "PARENTHESIZED LATIN CAPITAL LETTER A..Z"),
    (0x1F12A, 0x1F12A, "TORTOISE SHELL BRACKETED LATIN CAPITAL LETTER S"),
    (0x1F12B, 0x1F12C, "CIRCLED ITALIC LATIN CAPITAL LETTER C..R"),
    (0x1F130, 0x1F149, "SQUARED LATIN CAPITAL LETTER A..Z"),
    (0x1F150, 0x1F169, "NEGATIVE CIRCLED LATIN CAPITAL LETTER A..Z"),
    (0x1F170, 0x1F18A, "NEGATIVE SQUARED LATIN CAPITAL LETTER A..Z, then "
                       "CROSSED NEGATIVE SQUARED LATIN CAPITAL LETTER P"),
    (0x1F1A5, 0x1F1A5, "SQUARED LATIN SMALL LETTER D"),
    (0x1F1E6, 0x1F1FF, "REGIONAL INDICATOR SYMBOL LETTER A..Z"),
)

_ENCLOSED_LETTERFORMS = frozenset(
    chr(cp) for low, high, _name in ENCLOSED_LETTERFORM_RANGES
    for cp in range(low, high + 1))

#: A2: the enclosed letterform -> the letter it encloses, read off each
#: character's own unicode NAME rather than off its NFKD decomposition, because
#: 80 of the 213 have no decomposition and the name is the evidence the ruling
#: rests on. Built once at import over the pasted ranges above.
_ENCLOSED_LETTERFORM_NAME_RE = re.compile(
    r"^(?:.*\s)?LATIN (CAPITAL|SMALL) LETTER ([A-Z])$")
_REGIONAL_INDICATOR_NAME_RE = re.compile(r"^REGIONAL INDICATOR SYMBOL LETTER ([A-Z])$")


def _enclosed_letterform_map() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for ch in _ENCLOSED_LETTERFORMS:
        try:
            name = unicodedata.name(ch)
        except ValueError:                   # unassigned in this unicode build
            continue
        match = _ENCLOSED_LETTERFORM_NAME_RE.match(name)
        if match:
            letter = match.group(2)
            mapping[ch] = letter if match.group(1) == "CAPITAL" else letter.lower()
            continue
        match = _REGIONAL_INDICATOR_NAME_RE.match(name)
        if match:                            # drawn as a boxed capital
            mapping[ch] = match.group(1)
    return mapping


_ENCLOSED_LETTERFORM_TO_LATIN: Dict[str, str] = _enclosed_letterform_map()


def _source_is_a_letter(ch: str) -> bool:
    """Is ``ch`` an OCCURRENCE of a letter, rather than a symbol that happens
    to decompose into letters? True for a letter in any form (fullwidth,
    mathematical, ligature) and for an enclosed letterform (circled A)."""
    return _is_letter(ch) or ch in _ENCLOSED_LETTERFORMS


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
#:
#: WIRED, not merely described: when True, ``LEETSPEAK_TO_LATIN`` is applied in
#: ``_fold_char`` and "c0ffee" does violate a ban on "coffee". It is False
#: because of the "2020" cost above, not because the mapping is missing.
LEETSPEAK_DIGITS_ARE_LETTERS = False

#: The mapping ``LEETSPEAK_DIGITS_ARE_LETTERS`` switches on. Consulted only when
#: that constant is True; it is the whole of what flipping it does. Restricted
#: to digits whose glyph is a letter substitution in ordinary leetspeak -- no
#: punctuation ("|" for l, "()" for o), which would collide with A3.
LEETSPEAK_TO_LATIN: Dict[str, str] = {
    "0": "o",
    "1": "l",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
}

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
    "ԁ": "d", "Ԁ": "D",   # d / D
    "һ": "h", "Һ": "H",   # shha renders as h
    "ԛ": "q", "Ԛ": "Q",   # qa renders as q
    "ԝ": "w", "Ԝ": "W",   # we renders as w
    "ѵ": "v", "Ѵ": "V",   # izhitsa renders as v
    "ӏ": "l", "Ӏ": "I",   # palochka renders as l / I
    "є": "e", "Є": "E",   # ukrainian ie renders as e
    "ҽ": "e", "Ҽ": "E",   # abkhasian che renders as e
    "ҿ": "e", "Ҿ": "E",   # abkhasian che with descender renders as e
    "ә": "e", "Ә": "E",   # cyrillic schwa renders as a turned e
    "ԑ": "e", "Ԑ": "E",   # cyrillic reversed ze renders as an epsilon
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
    "ɛ": "e", "Ɛ": "E",   # open e
    "ǝ": "e", "Ǝ": "E",   # turned e / reversed e
    "ə": "e", "Ə": "E",   # latin schwa
    "ⱻ": "e",                  # small capital turned e
    "ꬲ": "e",                  # blackletter e
    # --- letterlike symbols that render as a letter ---
    "℮": "e",                  # ESTIMATED SYMBOL renders as an e
}

#: A2: a MODIFIED LETTERFORM is still the letter. The checker already catches
#: "e-acute" and "e-macron", because NFKD decomposes them; it did NOT catch
#: "e with stroke" (U+0247) or "e with notch" (U+2C78), because those encode the
#: modification in the codepoint itself and have no decomposition at all. That
#: is the same ruling enforced for one sibling codepoint and not the next, so
#: the table below closes it. Applied before CONFUSABLE_TO_LATIN.
#:
#: KNOWN CONSEQUENCE, flagged for sign-off: this folds letters that are
#: independent members of another alphabet -- Danish/Norwegian "o with stroke",
#: Polish "l with stroke", Croatian "d with stroke". Fixture fx-character-17
#: (Soup with o-stroke, banned letter 'o') is the pinned case and it is an OPEN
#: ambiguity; its own rationale says it must be ruled TOGETHER with
#: fx-character-15, and fx-character-15 is ruled a violation. Flip
#: FOLD_LETTERFORMS to False to fold neither.
FOLD_LETTERFORMS = True

#: A2, stroke/bar/hook forms and small capitals -> Latin. DERIVATION (run once,
#: pasted here so the table is data, not a rule applied at runtime): every
#: character whose unicode NAME is ``LATIN {SMALL,CAPITAL} LETTER <B> <MOD>``
#: or ``LATIN LETTER SMALL CAPITAL <B>`` -- i.e. the name itself says the source
#: IS the letter <B> in another form -- and whose NFKD decomposition does NOT
#: already yield <B>, so the pipeline above cannot catch it. 257 entries across
#: all 26 letters. Entries whose NFKD already recovers the base letter
#: (``é``, ``ā``, ``ẚ``) are deliberately absent: they are already handled.
#:
#: <MOD> IS ANY MODIFIER, not the literal word "WITH". The derivation used to
#: require " WITH ", which is why Ħ, Ŧ, Đ and Ł were caught while U+0244
#: "LATIN CAPITAL LETTER U BAR" and U+0289 "LATIN SMALL LETTER U BAR" -- the
#: same stroke through the same kind of letter, named without the preposition
#: -- walked straight through a ban on 'u'. Widening the regex adds exactly
#: seven characters, listed together at the end of the table.
LETTERFORM_TO_LATIN: Dict[str, str] = {
    # --- modified Latin letterforms (stroke / bar / hook / descender / curl) ---
    "Ø": "O",  # U+00D8 LATIN CAPITAL LETTER O WITH STROKE
    "ø": "o",  # U+00F8 LATIN SMALL LETTER O WITH STROKE
    "Đ": "D",  # U+0110 LATIN CAPITAL LETTER D WITH STROKE
    "đ": "d",  # U+0111 LATIN SMALL LETTER D WITH STROKE
    "Ħ": "H",  # U+0126 LATIN CAPITAL LETTER H WITH STROKE
    "ħ": "h",  # U+0127 LATIN SMALL LETTER H WITH STROKE
    "Ł": "L",  # U+0141 LATIN CAPITAL LETTER L WITH STROKE
    "ł": "l",  # U+0142 LATIN SMALL LETTER L WITH STROKE
    "Ŧ": "T",  # U+0166 LATIN CAPITAL LETTER T WITH STROKE
    "ŧ": "t",  # U+0167 LATIN SMALL LETTER T WITH STROKE
    "ƀ": "b",  # U+0180 LATIN SMALL LETTER B WITH STROKE
    "Ɓ": "B",  # U+0181 LATIN CAPITAL LETTER B WITH HOOK
    "Ƃ": "B",  # U+0182 LATIN CAPITAL LETTER B WITH TOPBAR
    "ƃ": "b",  # U+0183 LATIN SMALL LETTER B WITH TOPBAR
    "Ƈ": "C",  # U+0187 LATIN CAPITAL LETTER C WITH HOOK
    "ƈ": "c",  # U+0188 LATIN SMALL LETTER C WITH HOOK
    "Ɗ": "D",  # U+018A LATIN CAPITAL LETTER D WITH HOOK
    "Ƌ": "D",  # U+018B LATIN CAPITAL LETTER D WITH TOPBAR
    "ƌ": "d",  # U+018C LATIN SMALL LETTER D WITH TOPBAR
    "Ƒ": "F",  # U+0191 LATIN CAPITAL LETTER F WITH HOOK
    "ƒ": "f",  # U+0192 LATIN SMALL LETTER F WITH HOOK
    "Ɠ": "G",  # U+0193 LATIN CAPITAL LETTER G WITH HOOK
    "Ɨ": "I",  # U+0197 LATIN CAPITAL LETTER I WITH STROKE
    "Ƙ": "K",  # U+0198 LATIN CAPITAL LETTER K WITH HOOK
    "ƙ": "k",  # U+0199 LATIN SMALL LETTER K WITH HOOK
    "ƚ": "l",  # U+019A LATIN SMALL LETTER L WITH BAR
    "Ɲ": "N",  # U+019D LATIN CAPITAL LETTER N WITH LEFT HOOK
    "ƞ": "n",  # U+019E LATIN SMALL LETTER N WITH LONG RIGHT LEG
    "Ɵ": "O",  # U+019F LATIN CAPITAL LETTER O WITH MIDDLE TILDE
    "Ƥ": "P",  # U+01A4 LATIN CAPITAL LETTER P WITH HOOK
    "ƥ": "p",  # U+01A5 LATIN SMALL LETTER P WITH HOOK
    "ƫ": "t",  # U+01AB LATIN SMALL LETTER T WITH PALATAL HOOK
    "Ƭ": "T",  # U+01AC LATIN CAPITAL LETTER T WITH HOOK
    "ƭ": "t",  # U+01AD LATIN SMALL LETTER T WITH HOOK
    "Ʈ": "T",  # U+01AE LATIN CAPITAL LETTER T WITH RETROFLEX HOOK
    "Ʋ": "V",  # U+01B2 LATIN CAPITAL LETTER V WITH HOOK
    "Ƴ": "Y",  # U+01B3 LATIN CAPITAL LETTER Y WITH HOOK
    "ƴ": "y",  # U+01B4 LATIN SMALL LETTER Y WITH HOOK
    "Ƶ": "Z",  # U+01B5 LATIN CAPITAL LETTER Z WITH STROKE
    "ƶ": "z",  # U+01B6 LATIN SMALL LETTER Z WITH STROKE
    "Ǥ": "G",  # U+01E4 LATIN CAPITAL LETTER G WITH STROKE
    "ǥ": "g",  # U+01E5 LATIN SMALL LETTER G WITH STROKE
    "Ǿ": "O",  # U+01FE LATIN CAPITAL LETTER O WITH STROKE AND ACUTE
    "ǿ": "o",  # U+01FF LATIN SMALL LETTER O WITH STROKE AND ACUTE
    "Ƞ": "N",  # U+0220 LATIN CAPITAL LETTER N WITH LONG RIGHT LEG
    "ȡ": "d",  # U+0221 LATIN SMALL LETTER D WITH CURL
    "Ȥ": "Z",  # U+0224 LATIN CAPITAL LETTER Z WITH HOOK
    "ȥ": "z",  # U+0225 LATIN SMALL LETTER Z WITH HOOK
    "ȴ": "l",  # U+0234 LATIN SMALL LETTER L WITH CURL
    "ȵ": "n",  # U+0235 LATIN SMALL LETTER N WITH CURL
    "ȶ": "t",  # U+0236 LATIN SMALL LETTER T WITH CURL
    "Ⱥ": "A",  # U+023A LATIN CAPITAL LETTER A WITH STROKE
    "Ȼ": "C",  # U+023B LATIN CAPITAL LETTER C WITH STROKE
    "ȼ": "c",  # U+023C LATIN SMALL LETTER C WITH STROKE
    "Ƚ": "L",  # U+023D LATIN CAPITAL LETTER L WITH BAR
    "Ⱦ": "T",  # U+023E LATIN CAPITAL LETTER T WITH DIAGONAL STROKE
    "ȿ": "s",  # U+023F LATIN SMALL LETTER S WITH SWASH TAIL
    "ɀ": "z",  # U+0240 LATIN SMALL LETTER Z WITH SWASH TAIL
    "Ƀ": "B",  # U+0243 LATIN CAPITAL LETTER B WITH STROKE
    "Ɇ": "E",  # U+0246 LATIN CAPITAL LETTER E WITH STROKE
    "ɇ": "e",  # U+0247 LATIN SMALL LETTER E WITH STROKE
    "Ɉ": "J",  # U+0248 LATIN CAPITAL LETTER J WITH STROKE
    "ɉ": "j",  # U+0249 LATIN SMALL LETTER J WITH STROKE
    "ɋ": "q",  # U+024B LATIN SMALL LETTER Q WITH HOOK TAIL
    "Ɍ": "R",  # U+024C LATIN CAPITAL LETTER R WITH STROKE
    "ɍ": "r",  # U+024D LATIN SMALL LETTER R WITH STROKE
    "Ɏ": "Y",  # U+024E LATIN CAPITAL LETTER Y WITH STROKE
    "ɏ": "y",  # U+024F LATIN SMALL LETTER Y WITH STROKE
    "ɓ": "b",  # U+0253 LATIN SMALL LETTER B WITH HOOK
    "ɕ": "c",  # U+0255 LATIN SMALL LETTER C WITH CURL
    "ɖ": "d",  # U+0256 LATIN SMALL LETTER D WITH TAIL
    "ɗ": "d",  # U+0257 LATIN SMALL LETTER D WITH HOOK
    "ɠ": "g",  # U+0260 LATIN SMALL LETTER G WITH HOOK
    "ɦ": "h",  # U+0266 LATIN SMALL LETTER H WITH HOOK
    "ɨ": "i",  # U+0268 LATIN SMALL LETTER I WITH STROKE
    "ɫ": "l",  # U+026B LATIN SMALL LETTER L WITH MIDDLE TILDE
    "ɬ": "l",  # U+026C LATIN SMALL LETTER L WITH BELT
    "ɭ": "l",  # U+026D LATIN SMALL LETTER L WITH RETROFLEX HOOK
    "ɱ": "m",  # U+0271 LATIN SMALL LETTER M WITH HOOK
    "ɲ": "n",  # U+0272 LATIN SMALL LETTER N WITH LEFT HOOK
    "ɳ": "n",  # U+0273 LATIN SMALL LETTER N WITH RETROFLEX HOOK
    "ɼ": "r",  # U+027C LATIN SMALL LETTER R WITH LONG LEG
    "ɽ": "r",  # U+027D LATIN SMALL LETTER R WITH TAIL
    "ɾ": "r",  # U+027E LATIN SMALL LETTER R WITH FISHHOOK
    "ʂ": "s",  # U+0282 LATIN SMALL LETTER S WITH HOOK
    "ʈ": "t",  # U+0288 LATIN SMALL LETTER T WITH RETROFLEX HOOK
    "ʋ": "v",  # U+028B LATIN SMALL LETTER V WITH HOOK
    "ʐ": "z",  # U+0290 LATIN SMALL LETTER Z WITH RETROFLEX HOOK
    "ʑ": "z",  # U+0291 LATIN SMALL LETTER Z WITH CURL
    "ʝ": "j",  # U+029D LATIN SMALL LETTER J WITH CROSSED-TAIL
    "ʠ": "q",  # U+02A0 LATIN SMALL LETTER Q WITH HOOK
    "ᵬ": "b",  # U+1D6C LATIN SMALL LETTER B WITH MIDDLE TILDE
    "ᵭ": "d",  # U+1D6D LATIN SMALL LETTER D WITH MIDDLE TILDE
    "ᵮ": "f",  # U+1D6E LATIN SMALL LETTER F WITH MIDDLE TILDE
    "ᵯ": "m",  # U+1D6F LATIN SMALL LETTER M WITH MIDDLE TILDE
    "ᵰ": "n",  # U+1D70 LATIN SMALL LETTER N WITH MIDDLE TILDE
    "ᵱ": "p",  # U+1D71 LATIN SMALL LETTER P WITH MIDDLE TILDE
    "ᵲ": "r",  # U+1D72 LATIN SMALL LETTER R WITH MIDDLE TILDE
    "ᵳ": "r",  # U+1D73 LATIN SMALL LETTER R WITH FISHHOOK AND MIDDLE TILDE
    "ᵴ": "s",  # U+1D74 LATIN SMALL LETTER S WITH MIDDLE TILDE
    "ᵵ": "t",  # U+1D75 LATIN SMALL LETTER T WITH MIDDLE TILDE
    "ᵶ": "z",  # U+1D76 LATIN SMALL LETTER Z WITH MIDDLE TILDE
    "ᵽ": "p",  # U+1D7D LATIN SMALL LETTER P WITH STROKE
    "ᶀ": "b",  # U+1D80 LATIN SMALL LETTER B WITH PALATAL HOOK
    "ᶁ": "d",  # U+1D81 LATIN SMALL LETTER D WITH PALATAL HOOK
    "ᶂ": "f",  # U+1D82 LATIN SMALL LETTER F WITH PALATAL HOOK
    "ᶃ": "g",  # U+1D83 LATIN SMALL LETTER G WITH PALATAL HOOK
    "ᶄ": "k",  # U+1D84 LATIN SMALL LETTER K WITH PALATAL HOOK
    "ᶅ": "l",  # U+1D85 LATIN SMALL LETTER L WITH PALATAL HOOK
    "ᶆ": "m",  # U+1D86 LATIN SMALL LETTER M WITH PALATAL HOOK
    "ᶇ": "n",  # U+1D87 LATIN SMALL LETTER N WITH PALATAL HOOK
    "ᶈ": "p",  # U+1D88 LATIN SMALL LETTER P WITH PALATAL HOOK
    "ᶉ": "r",  # U+1D89 LATIN SMALL LETTER R WITH PALATAL HOOK
    "ᶊ": "s",  # U+1D8A LATIN SMALL LETTER S WITH PALATAL HOOK
    "ᶌ": "v",  # U+1D8C LATIN SMALL LETTER V WITH PALATAL HOOK
    "ᶍ": "x",  # U+1D8D LATIN SMALL LETTER X WITH PALATAL HOOK
    "ᶎ": "z",  # U+1D8E LATIN SMALL LETTER Z WITH PALATAL HOOK
    "ᶏ": "a",  # U+1D8F LATIN SMALL LETTER A WITH RETROFLEX HOOK
    "ᶑ": "d",  # U+1D91 LATIN SMALL LETTER D WITH HOOK AND TAIL
    "ᶒ": "e",  # U+1D92 LATIN SMALL LETTER E WITH RETROFLEX HOOK
    "ᶖ": "i",  # U+1D96 LATIN SMALL LETTER I WITH RETROFLEX HOOK
    "ᶙ": "u",  # U+1D99 LATIN SMALL LETTER U WITH RETROFLEX HOOK
    "Ỿ": "Y",  # U+1EFE LATIN CAPITAL LETTER Y WITH LOOP
    "ỿ": "y",  # U+1EFF LATIN SMALL LETTER Y WITH LOOP
    "Ⱡ": "L",  # U+2C60 LATIN CAPITAL LETTER L WITH DOUBLE BAR
    "ⱡ": "l",  # U+2C61 LATIN SMALL LETTER L WITH DOUBLE BAR
    "Ɫ": "L",  # U+2C62 LATIN CAPITAL LETTER L WITH MIDDLE TILDE
    "Ᵽ": "P",  # U+2C63 LATIN CAPITAL LETTER P WITH STROKE
    "Ɽ": "R",  # U+2C64 LATIN CAPITAL LETTER R WITH TAIL
    "ⱥ": "a",  # U+2C65 LATIN SMALL LETTER A WITH STROKE
    "ⱦ": "t",  # U+2C66 LATIN SMALL LETTER T WITH DIAGONAL STROKE
    "Ⱨ": "H",  # U+2C67 LATIN CAPITAL LETTER H WITH DESCENDER
    "ⱨ": "h",  # U+2C68 LATIN SMALL LETTER H WITH DESCENDER
    "Ⱪ": "K",  # U+2C69 LATIN CAPITAL LETTER K WITH DESCENDER
    "ⱪ": "k",  # U+2C6A LATIN SMALL LETTER K WITH DESCENDER
    "Ⱬ": "Z",  # U+2C6B LATIN CAPITAL LETTER Z WITH DESCENDER
    "ⱬ": "z",  # U+2C6C LATIN SMALL LETTER Z WITH DESCENDER
    "Ɱ": "M",  # U+2C6E LATIN CAPITAL LETTER M WITH HOOK
    "ⱱ": "v",  # U+2C71 LATIN SMALL LETTER V WITH RIGHT HOOK
    "Ⱳ": "W",  # U+2C72 LATIN CAPITAL LETTER W WITH HOOK
    "ⱳ": "w",  # U+2C73 LATIN SMALL LETTER W WITH HOOK
    "ⱴ": "v",  # U+2C74 LATIN SMALL LETTER V WITH CURL
    "ⱸ": "e",  # U+2C78 LATIN SMALL LETTER E WITH NOTCH
    "ⱺ": "o",  # U+2C7A LATIN SMALL LETTER O WITH LOW RING INSIDE
    "Ȿ": "S",  # U+2C7E LATIN CAPITAL LETTER S WITH SWASH TAIL
    "Ɀ": "Z",  # U+2C7F LATIN CAPITAL LETTER Z WITH SWASH TAIL
    "Ꝁ": "K",  # U+A740 LATIN CAPITAL LETTER K WITH STROKE
    "ꝁ": "k",  # U+A741 LATIN SMALL LETTER K WITH STROKE
    "Ꝃ": "K",  # U+A742 LATIN CAPITAL LETTER K WITH DIAGONAL STROKE
    "ꝃ": "k",  # U+A743 LATIN SMALL LETTER K WITH DIAGONAL STROKE
    "Ꝅ": "K",  # U+A744 LATIN CAPITAL LETTER K WITH STROKE AND DIAGONAL STROKE
    "ꝅ": "k",  # U+A745 LATIN SMALL LETTER K WITH STROKE AND DIAGONAL STROKE
    "Ꝉ": "L",  # U+A748 LATIN CAPITAL LETTER L WITH HIGH STROKE
    "ꝉ": "l",  # U+A749 LATIN SMALL LETTER L WITH HIGH STROKE
    "Ꝋ": "O",  # U+A74A LATIN CAPITAL LETTER O WITH LONG STROKE OVERLAY
    "ꝋ": "o",  # U+A74B LATIN SMALL LETTER O WITH LONG STROKE OVERLAY
    "Ꝍ": "O",  # U+A74C LATIN CAPITAL LETTER O WITH LOOP
    "ꝍ": "o",  # U+A74D LATIN SMALL LETTER O WITH LOOP
    "Ꝑ": "P",  # U+A750 LATIN CAPITAL LETTER P WITH STROKE THROUGH DESCENDER
    "ꝑ": "p",  # U+A751 LATIN SMALL LETTER P WITH STROKE THROUGH DESCENDER
    "Ꝓ": "P",  # U+A752 LATIN CAPITAL LETTER P WITH FLOURISH
    "ꝓ": "p",  # U+A753 LATIN SMALL LETTER P WITH FLOURISH
    "Ꝕ": "P",  # U+A754 LATIN CAPITAL LETTER P WITH SQUIRREL TAIL
    "ꝕ": "p",  # U+A755 LATIN SMALL LETTER P WITH SQUIRREL TAIL
    "Ꝗ": "Q",  # U+A756 LATIN CAPITAL LETTER Q WITH STROKE THROUGH DESCENDER
    "ꝗ": "q",  # U+A757 LATIN SMALL LETTER Q WITH STROKE THROUGH DESCENDER
    "Ꝙ": "Q",  # U+A758 LATIN CAPITAL LETTER Q WITH DIAGONAL STROKE
    "ꝙ": "q",  # U+A759 LATIN SMALL LETTER Q WITH DIAGONAL STROKE
    "Ꝟ": "V",  # U+A75E LATIN CAPITAL LETTER V WITH DIAGONAL STROKE
    "ꝟ": "v",  # U+A75F LATIN SMALL LETTER V WITH DIAGONAL STROKE
    "ꞎ": "l",  # U+A78E LATIN SMALL LETTER L WITH RETROFLEX HOOK AND BELT
    "Ꞑ": "N",  # U+A790 LATIN CAPITAL LETTER N WITH DESCENDER
    "ꞑ": "n",  # U+A791 LATIN SMALL LETTER N WITH DESCENDER
    "Ꞓ": "C",  # U+A792 LATIN CAPITAL LETTER C WITH BAR
    "ꞓ": "c",  # U+A793 LATIN SMALL LETTER C WITH BAR
    "ꞔ": "c",  # U+A794 LATIN SMALL LETTER C WITH PALATAL HOOK
    "ꞕ": "h",  # U+A795 LATIN SMALL LETTER H WITH PALATAL HOOK
    "Ꞗ": "B",  # U+A796 LATIN CAPITAL LETTER B WITH FLOURISH
    "ꞗ": "b",  # U+A797 LATIN SMALL LETTER B WITH FLOURISH
    "Ꞙ": "F",  # U+A798 LATIN CAPITAL LETTER F WITH STROKE
    "ꞙ": "f",  # U+A799 LATIN SMALL LETTER F WITH STROKE
    "Ꞡ": "G",  # U+A7A0 LATIN CAPITAL LETTER G WITH OBLIQUE STROKE
    "ꞡ": "g",  # U+A7A1 LATIN SMALL LETTER G WITH OBLIQUE STROKE
    "Ꞣ": "K",  # U+A7A2 LATIN CAPITAL LETTER K WITH OBLIQUE STROKE
    "ꞣ": "k",  # U+A7A3 LATIN SMALL LETTER K WITH OBLIQUE STROKE
    "Ꞥ": "N",  # U+A7A4 LATIN CAPITAL LETTER N WITH OBLIQUE STROKE
    "ꞥ": "n",  # U+A7A5 LATIN SMALL LETTER N WITH OBLIQUE STROKE
    "Ꞧ": "R",  # U+A7A6 LATIN CAPITAL LETTER R WITH OBLIQUE STROKE
    "ꞧ": "r",  # U+A7A7 LATIN SMALL LETTER R WITH OBLIQUE STROKE
    "Ꞩ": "S",  # U+A7A8 LATIN CAPITAL LETTER S WITH OBLIQUE STROKE
    "ꞩ": "s",  # U+A7A9 LATIN SMALL LETTER S WITH OBLIQUE STROKE
    "Ɦ": "H",  # U+A7AA LATIN CAPITAL LETTER H WITH HOOK
    "Ɬ": "L",  # U+A7AD LATIN CAPITAL LETTER L WITH BELT
    "Ʝ": "J",  # U+A7B2 LATIN CAPITAL LETTER J WITH CROSSED-TAIL
    "Ꞹ": "U",  # U+A7B8 LATIN CAPITAL LETTER U WITH STROKE
    "ꞹ": "u",  # U+A7B9 LATIN SMALL LETTER U WITH STROKE
    "Ꞔ": "C",  # U+A7C4 LATIN CAPITAL LETTER C WITH PALATAL HOOK
    "Ʂ": "S",  # U+A7C5 LATIN CAPITAL LETTER S WITH HOOK
    "Ᶎ": "Z",  # U+A7C6 LATIN CAPITAL LETTER Z WITH PALATAL HOOK
    "Ꟈ": "D",  # U+A7C7 LATIN CAPITAL LETTER D WITH SHORT STROKE OVERLAY
    "ꟈ": "d",  # U+A7C8 LATIN SMALL LETTER D WITH SHORT STROKE OVERLAY
    "Ꟊ": "S",  # U+A7C9 LATIN CAPITAL LETTER S WITH SHORT STROKE OVERLAY
    "ꟊ": "s",  # U+A7CA LATIN SMALL LETTER S WITH SHORT STROKE OVERLAY
    "ꬴ": "e",  # U+AB34 LATIN SMALL LETTER E WITH FLOURISH
    "ꬷ": "l",  # U+AB37 LATIN SMALL LETTER L WITH INVERTED LAZY S
    "ꬸ": "l",  # U+AB38 LATIN SMALL LETTER L WITH DOUBLE MIDDLE TILDE
    "ꬹ": "l",  # U+AB39 LATIN SMALL LETTER L WITH MIDDLE RING
    "ꬺ": "m",  # U+AB3A LATIN SMALL LETTER M WITH CROSSED-TAIL
    "ꬻ": "n",  # U+AB3B LATIN SMALL LETTER N WITH CROSSED-TAIL
    "ꭉ": "r",  # U+AB49 LATIN SMALL LETTER R WITH CROSSED-TAIL
    "ꭎ": "u",  # U+AB4E LATIN SMALL LETTER U WITH SHORT RIGHT LEG
    "ꭒ": "u",  # U+AB52 LATIN SMALL LETTER U WITH LEFT HOOK
    "ꭖ": "x",  # U+AB56 LATIN SMALL LETTER X WITH LOW RIGHT RING
    "ꭗ": "x",  # U+AB57 LATIN SMALL LETTER X WITH LONG LEFT LEG
    "ꭘ": "x",  # U+AB58 LATIN SMALL LETTER X WITH LONG LEFT LEG AND LOW RIGHT RING
    "ꭙ": "x",  # U+AB59 LATIN SMALL LETTER X WITH LONG LEFT LEG WITH SERIF
    "ꭚ": "y",  # U+AB5A LATIN SMALL LETTER Y WITH SHORT RIGHT LEG
    "𝼉": "t",  # U+1DF09 LATIN SMALL LETTER T WITH HOOK AND RETROFLEX HOOK
    "𝼑": "l",  # U+1DF11 LATIN SMALL LETTER L WITH FISHHOOK
    "𝼓": "l",  # U+1DF13 LATIN SMALL LETTER L WITH BELT AND PALATAL HOOK
    "𝼖": "r",  # U+1DF16 LATIN SMALL LETTER R WITH FISHHOOK AND PALATAL HOOK
    "𝼚": "i",  # U+1DF1A LATIN SMALL LETTER I WITH STROKE AND RETROFLEX HOOK
    "𝼛": "o",  # U+1DF1B LATIN SMALL LETTER O WITH RETROFLEX HOOK
    "𝼝": "c",  # U+1DF1D LATIN SMALL LETTER C WITH RETROFLEX HOOK
    "𝼞": "s",  # U+1DF1E LATIN SMALL LETTER S WITH CURL
    # --- small capitals ---
    "ɢ": "g",  # U+0262 LATIN LETTER SMALL CAPITAL G
    "ɪ": "i",  # U+026A LATIN LETTER SMALL CAPITAL I
    "ɴ": "n",  # U+0274 LATIN LETTER SMALL CAPITAL N
    "ʀ": "r",  # U+0280 LATIN LETTER SMALL CAPITAL R
    "ʏ": "y",  # U+028F LATIN LETTER SMALL CAPITAL Y
    "ʙ": "b",  # U+0299 LATIN LETTER SMALL CAPITAL B
    "ʜ": "h",  # U+029C LATIN LETTER SMALL CAPITAL H
    "ʟ": "l",  # U+029F LATIN LETTER SMALL CAPITAL L
    "ᴀ": "a",  # U+1D00 LATIN LETTER SMALL CAPITAL A
    "ᴄ": "c",  # U+1D04 LATIN LETTER SMALL CAPITAL C
    "ᴅ": "d",  # U+1D05 LATIN LETTER SMALL CAPITAL D
    "ᴇ": "e",  # U+1D07 LATIN LETTER SMALL CAPITAL E
    "ᴊ": "j",  # U+1D0A LATIN LETTER SMALL CAPITAL J
    "ᴋ": "k",  # U+1D0B LATIN LETTER SMALL CAPITAL K
    "ᴍ": "m",  # U+1D0D LATIN LETTER SMALL CAPITAL M
    "ᴏ": "o",  # U+1D0F LATIN LETTER SMALL CAPITAL O
    "ᴘ": "p",  # U+1D18 LATIN LETTER SMALL CAPITAL P
    "ᴛ": "t",  # U+1D1B LATIN LETTER SMALL CAPITAL T
    "ᴜ": "u",  # U+1D1C LATIN LETTER SMALL CAPITAL U
    "ᴠ": "v",  # U+1D20 LATIN LETTER SMALL CAPITAL V
    "ᴡ": "w",  # U+1D21 LATIN LETTER SMALL CAPITAL W
    "ᴢ": "z",  # U+1D22 LATIN LETTER SMALL CAPITAL Z
    "ꜰ": "f",  # U+A730 LATIN LETTER SMALL CAPITAL F
    "ꜱ": "s",  # U+A731 LATIN LETTER SMALL CAPITAL S
    "ꞯ": "q",  # U+A7AF LATIN LETTER SMALL CAPITAL Q
    # --- named without the preposition "WITH": the seven the old derivation
    # --- regex missed. Same ruling, same shape, no new judgement call.
    "Ʉ": "U",  # U+0244 LATIN CAPITAL LETTER U BAR
    "ʉ": "u",  # U+0289 LATIN SMALL LETTER U BAR
    "Ꝛ": "R",  # U+A75A LATIN CAPITAL LETTER R ROTUNDA
    "ꝛ": "r",  # U+A75B LATIN SMALL LETTER R ROTUNDA
    "ꬱ": "a",  # U+AB31 LATIN SMALL LETTER A REVERSED-SCHWA
    "ꭇ": "r",  # U+AB47 LATIN SMALL LETTER R WITHOUT HANDLE
    "ꭏ": "u",  # U+AB4F LATIN SMALL LETTER U BAR WITH SHORT RIGHT LEG
}


#: The ASCII characters the fold DROPS. Derived from ``_is_invisible`` itself
#: rather than re-listed, so the fast-path guard in ``_fold`` can never disagree
#: with the rule it is guarding.
_ASCII_INVISIBLE_RE = re.compile(
    "[%s]" % "".join(re.escape(chr(cp)) for cp in range(0x80) if _is_invisible(chr(cp))))


def _fold_char(ch: str, casefold: bool) -> str:
    """Fold ONE original character; may return 0, 1 or several characters."""
    if IGNORE_INVISIBLE_CHARACTERS and _is_invisible(ch):
        return ""                            # renders as nothing -> counts as nothing
    form = UNICODE_NORMALIZATION if BANNED_LETTER_FOLD_DIACRITICS else "NFKC"
    try:
        piece = unicodedata.normalize(form, ch)
    except (TypeError, ValueError):          # defensive: never raise on text
        piece = ch
    if (not NFKD_MAY_MANUFACTURE_LETTERS
            and not _source_is_a_letter(ch)
            and any(_is_letter(c) for c in piece)):
        # A symbol is not an occurrence of the letters it decomposes into:
        # U+2122 is not a 't' followed by an 'm'. Keep the original character.
        piece = ch
    if IGNORE_INVISIBLE_CHARACTERS:
        piece = "".join(c for c in piece if not _is_invisible(c))
    if BANNED_LETTER_FOLD_DIACRITICS:
        piece = "".join(c for c in piece if not unicodedata.combining(c))
    # An enclosed letterform whose NFKD did NOT recover its letter (🅐, 🅰, 🇦)
    # is still that letter to a reader; the name-derived table supplies what the
    # decomposition does not. Entries NFKD already handled (Ⓐ) never reach here,
    # because ``piece`` is the decomposed "A" by now, not the enclosed source.
    piece = "".join(_ENCLOSED_LETTERFORM_TO_LATIN.get(c, c) for c in piece)
    if FOLD_LETTERFORMS:
        piece = "".join(LETTERFORM_TO_LATIN.get(c, c) for c in piece)
    if FOLD_CONFUSABLES:
        piece = "".join(CONFUSABLE_TO_LATIN.get(c, c) for c in piece)
    if LEETSPEAK_DIGITS_ARE_LETTERS:
        piece = "".join(LEETSPEAK_TO_LATIN.get(c, c) for c in piece)
    if casefold:
        piece = piece.casefold()
    return piece


def _fold(text: str, casefold: bool = True) -> Tuple[str, Optional[List[int]]]:
    """Return ``(folded_text, index_map)``.

    ``index_map[i]`` is the index in ``text`` of the character that produced
    ``folded_text[i]``; ``None`` means the identity map. Pure ASCII text takes
    the identity fast path -- ASCII casefolding is 1:1 and length preserving --
    which keeps a 100k-character response cheap. The fast path is disabled when
    ``LEETSPEAK_DIGITS_ARE_LETTERS`` is on, because that is the one ruling that
    rewrites an ASCII character; leaving it in would half-wire the knob.

    It is ALSO disabled for text containing an ASCII control character. Those
    are category Cc, so the fold DROPS them, so the fast path is no longer
    length preserving for such text -- and taking it anyway is what let
    ``"cof<U+0001>fee"`` walk past a ban on "coffee" while the non-ASCII
    zero-width version was caught. One `re.search` over the response is the
    price of keeping the fast path for the ordinary case.
    """
    if (text.isascii() and not LEETSPEAK_DIGITS_ARE_LETTERS
            and not _ASCII_INVISIBLE_RE.search(text)):
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
        if not folded_needle:
            # A needle made only of invisible characters folds away to nothing.
            # An empty pattern matches at offset 0 of every response, which
            # would fail every model on a malformed constraint instead of
            # reporting the truth, so it is skipped.
            continue
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

# INVARIANT (not a knob -- there is nothing to flip): only the marks the
# constraint LISTS are banned. When ``marks`` is ``[",", ";"]`` an em-dash, a
# colon and an ellipsis are all legal; the checker never generalises to
# "punctuation" as a family. Widening the ban is the scenario author's job,
# done by listing more marks.
#
# ENFORCED BY CONSTRUCTION: the banned set in ``check_banned_punctuation`` is
# built solely from ``_variants_of(mark)`` over ``params["marks"]``, and there
# is no other source of banned characters for it to consult -- no category
# sweep, no punctuation family, no fallback table. tests/test_constraints.py
# pins it in both directions
# (``test_banned_punctuation_bans_only_the_listed_marks_and_their_variants``).
#
# This replaces ``BANNED_PUNCTUATION_LISTED_MARKS_ONLY = True``, a constant no
# code read: flipping it changed nothing, so it described the behaviour without
# controlling it -- a trap for whoever next tried to flip it.

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

#: A4 (fx-fmt-18) -- OFF, and it stays off until Hank rules. THE STANDING
#: RULING IS DOUBLES ONLY: "every double form fires, no single form does". An
#: agent turned this on unilaterally on the reading that ``''a rounding error''``
#: is how a typewriter writes a double quotation mark; the defence it recorded
#: for doing so was checked and found factually wrong, so the change is reverted
#: rather than kept on a rationale nobody signed. fx-fmt-18 is an AMBIGUOUS row
#: and goes back onto the sign-off ledger, which is where an unruled reading
#: belongs -- a checker constant is not the place to record a guess.
#:
#: Set to True ONLY on a recorded ruling. Doing so bans ``''...''`` while
#: leaving every apostrophe in "don't", "o'clock" and "the dogs' bowl" alone
#: (they stand singly), and does NOT extend to U+2033 DOUBLE PRIME (fx-fmt-19):
#: that codepoint's job is inches and arcseconds, and telling the two uses apart
#: needs the paired-quoting machinery fx-fmt-08 shows cannot be had by flipping
#: a codepoint switch.
ADJACENT_SINGLE_QUOTES_ARE_A_DOUBLE_QUOTE = False

#: How many single-quote characters in a row read as one double quotation mark.
ADJACENT_SINGLE_QUOTE_RUN = 2

#: A4: a quotation has two ends. Requiring the closing run keeps a lone stray
#: ``''`` -- a typo, or a code snippet's empty string literal -- out of the
#: violation count, so the rule fires on quoting rather than on a keystroke.
DOUBLED_SINGLE_QUOTES_NEED_A_CLOSING_RUN = True

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


def _doubled_single_quote_run(text: str) -> Optional[Tuple[int, int]]:
    """A4: ``(start, end)`` of the OPENING typewriter double quote, else None.

    A run is ADJACENT_SINGLE_QUOTE_RUN or more single-quote characters in a row;
    a run counts only when the text also carries a closing one (unless
    DOUBLED_SINGLE_QUOTES_NEED_A_CLOSING_RUN is off). Scanning left to right
    keeps the reported violation the lowest-offset one, as the file's evidence
    invariant requires.
    """
    if not ADJACENT_SINGLE_QUOTES_ARE_A_DOUBLE_QUOTE:
        return None
    singles = set(SINGLE_QUOTATION_MARKS)
    runs: List[Tuple[int, int]] = []
    idx, length = 0, len(text)
    while idx < length:
        if text[idx] not in singles:
            idx += 1
            continue
        end = idx
        while end < length and text[end] in singles:
            end += 1
        if end - idx >= ADJACENT_SINGLE_QUOTE_RUN:
            runs.append((idx, end))
        idx = end
    needed = 2 if DOUBLED_SINGLE_QUOTES_NEED_A_CLOSING_RUN else 1
    return runs[0] if len(runs) >= needed else None


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
    doubled = _doubled_single_quote_run(text) if not COUNT_SINGLE_QUOTES else None
    for idx, ch in enumerate(text):
        if doubled is not None and idx == doubled[0]:
            return _fail(f"quotation mark {text[doubled[0]:doubled[1]]!r}",
                         text, doubled[0], doubled[1])
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

#: A5/fx-str-10: EVERY character that renders as a line break, not just '\n'.
#: A lone '\r' (Windows- or transcript-authored text), U+2028 LINE SEPARATOR,
#: U+2029 PARAGRAPH SEPARATOR, U+0085 NEL and the vertical tab / form feed all
#: break a line for the reader, so ``NEWLINE_TERMINATES_SENTENCE`` has to apply
#: to all of them or the ruling is implemented for one character out of seven.
#: A "\r\n" pair is two breaks around an empty segment, which is skipped, so it
#: stays one boundary. This is ``str.splitlines()``'s set minus the three C0
#: separators U+001C..U+001E, which are not line breaks in prose.
SENTENCE_LINE_BREAK_CHARS: Tuple[str, ...] = (
    "\n",        # line feed
    "\r",        # carriage return
    "\u2028",    # U+2028 line separator
    "\u2029",    # U+2029 paragraph separator
    "\u0085",    # U+0085 next line (NEL)
    "\v",        # vertical tab
    "\f",        # form feed
)

#: A5: the last sentence often has no terminator ("...and that is why"). It is
#: still a sentence and still must obey the rule.
TRAILING_FRAGMENT_IS_SENTENCE = True

#: A5: a '.' that is followed immediately by a non-space, non-closing character
#: is not a sentence end. Covers "3.5", "example.com", "U.S.A." and "1.5x"
#: without a URL regex.
PERIOD_REQUIRES_TRAILING_SPACE = True

#: A5: a '.' between two digits is a decimal point.
DECIMAL_POINT_IS_NOT_A_TERMINATOR = True

#: A5 -- THE ABBREVIATION RULING. An abbreviation's '.' is a sentence boundary
#: or not according to whether the abbreviation CAN END A SENTENCE. It is NOT
#: decided by how many dots the token contains: dot-count is a proxy and it
#: failed in BOTH directions at once.
#:
#:   * one dot, treated as never-final -> "Sales rose in Aug. Tom took the
#:     account." was ONE sentence to the checker and two to every reader, so
#:     every sentence after "Aug.", "Sept.", "Fri.", "Inc.", "Ave." went
#:     ungraded. They COMPOSE: a response can switch the constraint off outright
#:     in ordinary business prose, with no adversarial character anywhere.
#:   * two dots, treated as follower-decides -> "Several teams already use it,
#:     e.g. Tom's group in Denver." was split, and its second half "Tom's group
#:     in Denver." was reported as a sentence beginning with T. That is a FALSE
#:     FAIL on ordinary prose: "e.g." and "i.e." are clause-OPENING, they
#:     introduce what follows and never end an English sentence.
#:
#: Both errors are expensive and in opposite directions -- one makes the
#: constraint look easier than it is, the other harder -- and
#: sentence_initial_letter is the HELD-OUT type whose score IS the project's
#: generalisation claim. So the tables below are tiered by the property that
#: actually decides, in three tiers, one rule each. Tokens are compared
#: lowercased, dot included.

#: TIER 1 -- NEVER FINAL. **A '.' after one of these is never a sentence
#: boundary, whatever follows.** Each one introduces the thing after it (a name,
#: an example, a figure number) and so cannot be the last token of a sentence.
#: Multi-dot members are here BY NAME, not by dot-count: "e.g." and "i.e." are
#: the two that used to split ordinary prose.
#: Cost accepted and named: "as shown by Smith et al. Tom reproduced it." keeps
#: the boundary suppressed, because "al." is listed here.
NEVER_FINAL_ABBREVIATIONS: Tuple[str, ...] = (
    # titles and honorifics -- always followed by the name they modify
    "mr.", "mrs.", "ms.", "dr.", "prof.", "rev.", "hon.",
    "jr.", "sr.", "col.", "capt.", "lt.", "sgt.", "gov.", "sen.", "mt.",
    # reference and comparison markers -- always followed by what they point at
    "vs.", "approx.", "fig.", "vol.", "al.",
    # clause-OPENING connectives; these are the multi-dot false-fail carriers
    "e.g.", "i.e.", "cf.", "viz.",
)

#: TIER 2 -- CAN BE FINAL. **A '.' after one of these is a boundary exactly when
#: the FOLLOWER begins a new sentence** (whitespace, then an uppercase letter).
#: These really do end English sentences ("Sales rose in Aug."), and they really
#: do continue them ("Feb. 3", "Inc. of Delaware"), so nothing about the token
#: itself decides -- only what comes next.
#: Residual ambiguity, stated rather than hidden: "Sales rose in Aug. Tom took
#: it." is ambiguous to a human too. CAN-BE-FINAL plus a capitalised follower is
#: the best rule available to a segmenter, not a perfect one; it splits "the
#: U.S. Navy liaison" and "9 a.m. Monday morning", which are the same surface
#: shape as "in the U.S. Tom signed for them." and "at 9 a.m. Tom never
#: arrives." -- texts this file's own tests require to be split.
CAN_BE_FINAL_ABBREVIATIONS: Tuple[str, ...] = (
    # months
    "jan.", "feb.", "apr.", "jun.", "jul.", "aug.",
    "sep.", "sept.", "oct.", "nov.", "dec.",
    # weekdays
    "mon.", "tue.", "tues.", "wed.", "thu.", "thurs.", "fri.",
    # organisation, address and unit suffixes -- they END a name
    "inc.", "ltd.", "corp.", "univ.", "ave.", "blvd.", "rd.", "sq.", "dept.",
    # list-closing
    "etc.",
    # inner-dotted members, listed by name for the auditor; any OTHER inner-
    # dotted token ("U.S.A.", "Ph.D.") gets this same rule by default below.
    "u.s.", "u.k.", "a.m.", "p.m.",
)

#: TIER 3 -- CAN BE FINAL ONLY IN ITS ORDINARY-WORD FORM. These are abbreviations
#: that are ALSO ordinary English words. **A '.' after one of these is a boundary
#: when the token is written in its ordinary-word (lowercase) form AND the
#: follower begins a new sentence.** The abbreviation form is written capitalised
#: and introduces what follows ("St. Louis", "No. 7 Station", "Acme Co. Ltd",
#: "Rep. Smith"), while the word form ends a clause in lowercase ("Sam said no.",
#: "Sam sat."), so the pair (own case, follower's case) separates them without
#: removing any entry -- removal would re-break "No. 7 Station", and keeping them
#: unconditional hides every sentence after "no.", "co." or "sat." (fx-str-08).
#: This tier is TIER 2 plus one extra piece of evidence, not a different ruling.
#: Cost accepted: lowercase "no. 7" keeps its boundary suppressed (the follower
#: is a digit, not a capital), and a capitalised "No." that genuinely ends a
#: sentence stays glued.
CAN_BE_FINAL_ONLY_AS_ORDINARY_WORD: Tuple[str, ...] = (
    "no.", "co.", "st.", "sat.", "sun.", "est.", "rep.", "gen.", "mar.",
)

#: TIER 2, SECOND HALF OF THE RULE -- WHAT THE FOLLOWER IS, not just how it is
#: cased. Tiering weekdays and months as CAN-BE-FINAL made a DATE split between
#: its own halves: "Shipments resume Fri. Oct. 3." became ["Shipments resume
#: Fri.", "Oct. 3."] and the second span was reported as a sentence beginning
#: with 'O'. All 63 weekday x non-S-month pairs false-failed. No human ruling is
#: needed for that one -- no reading of "Fri. Oct. 3" contains two sentences --
#: and the direction it errs in is the expensive one for this project: it
#: manufactures violations out of clean prose and so biases
#: sentence_initial_letter, the HELD-OUT type, toward "harder than it is".
#:
#: The rule: a tier-2 '.' is not a boundary when the token after it CONTINUES
#: the same expression -- the next piece of a date ("Fri. Oct. 3", "Oct. 3"),
#: the next piece of an organisation's name ("the U.S. Dept. of Commerce"), or
#: the other half of a clock time ("Mon. a.m."). A capitalised ordinary word
#: ("Fri. Nobody noticed.") is untouched and still splits.
#:
#: Why this is a LITERAL table and not "any abbreviation": tier 1 is full of
#: tokens that routinely OPEN a sentence -- "Sales rose in Aug. Dr. Rivera
#: resigned." is two sentences -- and so are the tier-3 titles "Rep." and
#: "Gen.". Suppressing the boundary before those would trade a false fail for a
#: swallowed boundary, and a swallowed boundary hides an UNBOUNDED amount of
#: following text while a spurious one costs exactly one extra graded segment.
#: Only tokens that cannot be the first word of a new sentence *in this
#: position* are listed. Every member is an abbreviation that appears in one of
#: the three tiers above (asserted in the tests).
#: Cost accepted and named: "Sales rose in Aug. Sept. was worse." keeps its
#: boundary suppressed -- a real, and to a human also ambiguous, sentence pair.
EXPRESSION_CONTINUING_FOLLOWERS: Tuple[str, ...] = (
    # a weekday is followed by its month: "Fri. Oct. 3"
    "jan.", "feb.", "mar.", "apr.", "jun.", "jul.", "aug.",
    "sep.", "sept.", "oct.", "nov.", "dec.",
    # name suffixes: these END a name, so they can only be continuing the name
    # that the previous abbreviation was already part of ("Acme Corp. Ltd.")
    "inc.", "ltd.", "corp.", "co.", "univ.", "dept.",
    # address elements, same argument ("Third Ave. Sq. entrance")
    "ave.", "blvd.", "rd.", "sq.",
    # the halves of a clock time: "Mon. a.m."
    "a.m.", "p.m.",
)

#: A5/ITEM-1: a bare number after a can-be-final abbreviation is the next piece
#: of the expression ("Oct. 3", "No. 7"), never the first word of a sentence.
#: Stated here as its own ruling rather than left to fall out of "a digit is not
#: an uppercase letter", so that relaxing the capital rule later cannot silently
#: take the day off a date.
NUMBER_AFTER_ABBREVIATION_CONTINUES_THE_EXPRESSION = True

_EXPRESSION_CONTINUING_FOLLOWER_SET = frozenset(EXPRESSION_CONTINUING_FOLLOWERS)

#: A5: the DEFAULT for an inner-dotted token that is in no table above
#: ("U.S.A.", "Ph.D.", "3.p.m."): treat it as TIER 2, i.e. the follower decides.
#: The alternative -- "a token with two dots never terminates" -- is a
#: 30-character reward hack, not an edge case: "The meeting is at 3 p.m. Sales
#: rose." made EVERY following sentence invisible to sentence_initial_letter.
#: One swallowed boundary hides an unbounded amount of text; a spurious boundary
#: adds exactly one extra graded segment, and that asymmetry decides the default.
#: Setting this False restores the old behaviour for every inner-dotted token,
#: listed or not; it does NOT affect TIER 1, whose members are never final for a
#: reason that has nothing to do with dot-count.
#: Name initials ("J. R. R. Tolkien") are decided earlier, by ``_is_name_initial``.
MULTI_DOT_ABBREVIATION_CAN_END_A_SENTENCE = True

#: A5/fx-str-09: titles that make a following one-letter token a name initial
#: ("Dr. J. Smith"). Used ONLY as evidence for the initial rule below.
NAME_TITLE_ABBREVIATIONS: Tuple[str, ...] = (
    "mr.", "mrs.", "ms.", "dr.", "prof.", "rev.", "hon.", "jr.", "sr.",
    "gen.", "col.", "capt.", "lt.", "sgt.", "gov.", "sen.", "rep.", "st.", "mt.",
)

#: A5/fx-str-09: a one-letter token before a '.' suppresses the sentence boundary
#: only when there is POSITIVE evidence that it is a name initial -- it follows a
#: title ("Dr. J. Smith"), follows another initial, or is followed by another
#: initial ("J. R. R. Tolkien"). Defence in one line: "Sam scored an A. Tom
#: scored a B." and "The author is J. Smith." have identical surface shape, and
#: the asymmetry decides the tie -- a missed boundary hides an UNBOUNDED amount
#: of following text from the checker, while a spurious boundary adds exactly one
#: extra graded segment. Cost accepted: a lone initial in mid-sentence prose
#: ("a memo from J. Smith") is split.
SINGLE_INITIAL_REQUIRES_NAME_EVIDENCE = True

#: A5: leading characters that are neither letters nor digits are skipped before
#: reading a sentence's first character, so ``"Something," she said.`` and
#: ``(Something happened.)`` both begin with S. A leading DIGIT is NOT skipped:
#: "1985 was the year" does not begin with the letter S, and a grader would mark
#: it a violation rather than looking past it.
SENTENCE_START_SKIPS_NON_ALNUM = True

#: A5: a segment with no letters or digits at all (a lone "---" or "...") is not
#: a sentence and is not checked.
EMPTY_SEGMENT_IS_SENTENCE = False

#: A5/ITEM-4: an abbreviation means the same thing with a bracket or a quote in
#: front of it, but ``_preceding_token`` returns the token VERBATIM, so the
#: lookup key for "(e.g." was "(e.g." and missed EVERY table. The whole
#: parenthesised form of the clause openers therefore false-failed at HEAD and
#: at base: "Several suppliers already use it (e.g. Rivera's group in Denver)."
#: was split and its second half reported as a sentence beginning with 'R' --
#: again a violation manufactured out of ordinary prose. Leading characters that
#: are neither letters nor digits are stripped before a table lookup, which is
#: the rule the reader already uses to find a sentence's first character
#: (``SENTENCE_START_SKIPS_NON_ALNUM``). A token that is ALL punctuation keeps
#: its verbatim form, so nothing that used to be looked up becomes empty.
STRIP_WRAPPERS_BEFORE_ABBREVIATION_LOOKUP = True

_SINGLE_INITIAL_RE = re.compile(r"^[^\W\d_]\.$", re.UNICODE)


def _lookup_form(token: str) -> str:
    """``token`` with any leading brackets/quotes removed, for a table lookup.

    Case is preserved: the ordinary-word tier keys on whether the author wrote
    ``St.`` or ``st.``, and "(St." must still read as the capitalised form.
    """
    if not STRIP_WRAPPERS_BEFORE_ABBREVIATION_LOOKUP:
        return token
    idx = 0
    while idx < len(token) and not token[idx].isalnum():
        idx += 1
    return token[idx:] or token


def _preceding_token(text: str, end: int) -> str:
    """The whitespace-delimited token ending at ``end`` (exclusive), VERBATIM.

    Returned with its original case on purpose: the ordinary-word abbreviation
    ruling keys on whether the author wrote ``St.`` or ``st.``. Call sites that
    only want a table lookup lowercase it themselves.
    """
    start = end
    while start > 0 and not text[start - 1].isspace():
        start -= 1
    return text[start:end]


def _token_before(text: str, end: int) -> str:
    """The whitespace-delimited token that precedes the one ending at ``end``."""
    start = end - len(_preceding_token(text, end))
    while start > 0 and text[start - 1].isspace():
        start -= 1
    return _preceding_token(text, start)


def _next_visible(text: str, idx: int) -> int:
    """First index at or after ``idx`` holding a character that RENDERS (A5).

    fx-str-07: the segmenter reads the page a human reads. A zero-width space,
    a word joiner or a right-to-left mark sitting after a terminator is not
    ``str.isspace()``, so without this the terminator failed the trailing-space
    test, the boundary vanished, and the ENTIRE response collapsed into one
    graded sentence -- the constraint disabled outright by three invisible bytes.
    """
    n = len(text)
    while idx < n and _is_invisible(text[idx]):
        idx += 1
    return idx


def _token_after(text: str, idx: int) -> str:
    """The next whitespace-delimited token at or after ``idx`` ("" if none).

    Invisible characters are skipped alongside whitespace, so an interleaved
    zero-width space cannot turn the following word into an empty token and
    make ``_next_word_starts_a_new_sentence`` answer False by starvation.
    """
    n = len(text)
    while idx < n and (text[idx].isspace() or _is_invisible(text[idx])):
        idx += 1
    end = idx
    while end < n and not text[end].isspace():
        end += 1
    return text[idx:end]


def _next_word_starts_a_new_sentence(text: str, idx: int) -> bool:
    """Does the text after ``idx`` look like the START of a new sentence?

    True only when the next alphanumeric character is an UPPERCASE letter;
    leading quotes and brackets are skipped, matching
    ``SENTENCE_START_SKIPS_NON_ALNUM``. A lowercase word ("etc. were on the
    table") or a digit ("no. 7 Station") is a continuation of this sentence.

    Only the next whitespace-delimited token is inspected, so this stays O(word)
    on a 100k-character hostile input.
    """
    for ch in _token_after(text, idx):
        if ch.isalnum():
            return ch.isupper()
    return False


def _is_name_initial(text: str, run_end: int, token_lower: str) -> bool:
    """Is the one-letter token ending at ``run_end`` a NAME initial (fx-str-09)?

    Positive evidence only: a title before it ("Dr. J. Smith"), another initial
    before it, or another initial after it ("J. R. R. Tolkien"). Without
    evidence, a lone one-letter token is an ordinary word or a label ("scored an
    A.", "call it X.") and the '.' after it ends the sentence.
    """
    if not _SINGLE_INITIAL_RE.match(token_lower):
        return False
    if not SINGLE_INITIAL_REQUIRES_NAME_EVIDENCE:
        return True
    before = _lookup_form(_token_before(text, run_end)).lower()
    if before in NAME_TITLE_ABBREVIATIONS or _SINGLE_INITIAL_RE.match(before):
        return True
    # RAW token: this looks for a second initial inside the SAME name run, so an
    # intervening bracket or quote means the following initial belongs elsewhere.
    # Stripping it would manufacture name evidence and swallow a real boundary.
    after = _token_after(text, run_end).lower()
    return bool(_SINGLE_INITIAL_RE.match(after))


def _follower_continues_the_expression(text: str, idx: int) -> bool:
    """Is the token after ``idx`` the NEXT PIECE of the expression (A5/ITEM-1)?

    Only asked of a can-be-final abbreviation, where the follower's case would
    otherwise decide alone. "Fri. Oct. 3" and "the U.S. Dept. of Commerce" are
    one expression each; "Fri. Nobody noticed." is two sentences, and stays two.
    """
    follower = _lookup_form(_token_after(text, idx))
    if not follower:
        return False
    if follower.lower() in _EXPRESSION_CONTINUING_FOLLOWER_SET:
        return True
    return (NUMBER_AFTER_ABBREVIATION_CONTINUES_THE_EXPRESSION
            and follower[:1].isdigit())


def _is_sentence_end(text: str, start: int, run_end: int) -> bool:
    """Does the terminator run ``text[start:run_end]`` really end a sentence?"""
    run = text[start:run_end]
    if run != ".":
        # '!', '?', the ellipsis character, and any multi-character run always
        # terminate: none of them are overloaded the way a lone '.' is.
        return True
    # --- single '.' ---
    visible = _next_visible(text, run_end)
    if DECIMAL_POINT_IS_NOT_A_TERMINATOR:
        prev_ch = text[start - 1] if start > 0 else ""
        next_ch = text[visible] if visible < len(text) else ""
        if prev_ch.isdigit() and next_ch.isdigit():
            return False
    if PERIOD_REQUIRES_TRAILING_SPACE and visible < len(text):
        next_ch = text[visible]
        if not next_ch.isspace() and next_ch not in SENTENCE_TRAILERS:
            return False
    token = _lookup_form(_preceding_token(text, run_end))
    lowered = token.lower()
    if lowered in NEVER_FINAL_ABBREVIATIONS:
        # TIER 1: "Dr.", "e.g." -- introduces what follows, so never a boundary.
        return False
    if lowered in CAN_BE_FINAL_ONLY_AS_ORDINARY_WORD:
        # TIER 3: abbreviation form ("St. Louis") keeps the sentence open;
        # ordinary-word form followed by a new capitalised word ("Sam said no.
        # Tom left.") ends it.
        # RAW token, not the stripped one: _lookup_form drops a leading quote or
        # bracket, so '"No.' opening quoted dialogue would read as capitalised and
        # take the abbreviation branch, swallowing the boundary. Strip for the TABLE
        # LOOKUP only; the case test is about how the author wrote the word.
        written_as_word = not _preceding_token(text, run_end)[:1].isupper()
        return written_as_word and _next_word_starts_a_new_sentence(text, run_end)
    if _is_name_initial(text, run_end, lowered):
        return False
    inner_dotted = lowered.count(".") > 1        # "u.s.", "a.m.", "U.S.A."
    if inner_dotted and not MULTI_DOT_ABBREVIATION_CAN_END_A_SENTENCE:
        return False
    if lowered in CAN_BE_FINAL_ABBREVIATIONS or inner_dotted:
        # TIER 2, and the same rule as the default for an unlisted inner-dotted
        # token: the follower decides -- WHAT it is first ("Fri. Oct. 3" is one
        # date, not two sentences), then how it is cased.
        if _follower_continues_the_expression(text, run_end):
            return False
        return _next_word_starts_a_new_sentence(text, run_end)
    return True


def _sentence_spans(text: str) -> List[Tuple[int, int]]:
    """Sentence spans ``(start, end)`` over the ORIGINAL text (A5)."""
    spans: List[Tuple[int, int]] = []
    n = len(text)
    i = 0
    seg_start: Optional[int] = None
    while i < n:
        ch = text[i]
        if seg_start is None and not ch.isspace() and not _is_invisible(ch):
            seg_start = i     # a character that renders as nothing starts nothing
        if ch in SENTENCE_LINE_BREAK_CHARS and NEWLINE_TERMINATES_SENTENCE:
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

#: A6: the hand-written half of the bullet table -- the ASCII markers and the
#: dedicated bullet glyphs models reach for. En- and em-dashes are here because
#: models really do bullet with them. This table alone is NOT the ruling: a
#: closed allowlist is always one codepoint from being defeated (fx-fmt-10), so
#: LIST_BULLET_UNICODE_BLOCKS below decides membership for everything else.
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

#: A6 (fx-fmt-10): a line-leading glyph is bullet-like because of WHAT IT IS,
#: not because someone remembered to list it. Keeping the list bulleted and
#: swapping "•" for "▶", "→", "✓", "★", "❖" or an emoji was the cheapest gaming
#: path in the whole format family -- no invisible characters, no unicode
#: tricks, just a different bullet -- and any hand-grown allowlist reopens it at
#: the next glyph. A character qualifies when its unicode general category is
#: one of LIST_BULLET_SYMBOL_CATEGORIES *and* its codepoint falls inside one of
#: the arrow / geometric / pictographic blocks below.
#:
#: ASCII is excluded on purpose: "<" and ">" are Sm, and admitting them would
#: turn every blockquote and every "<ul>" into a bullet. Blocks that text is
#: drawn WITH rather than bulleted with -- Box Drawing, Block Elements,
#: Mathematical Operators, Miscellaneous Technical -- are deliberately absent,
#: so an ASCII-art table or a line of algebra is not read as a list.
LIST_BULLET_SYMBOL_CATEGORIES: Tuple[str, ...] = ("So", "Sm", "Sk")

#: (first, last, unicode block name) -- inclusive, block names verbatim from the
#: standard so an auditor can check the ranges instead of trusting them.
LIST_BULLET_UNICODE_BLOCKS: Tuple[Tuple[int, int, str], ...] = (
    (0x2190, 0x21FF, "Arrows"),
    (0x25A0, 0x25FF, "Geometric Shapes"),
    (0x2600, 0x26FF, "Miscellaneous Symbols"),
    (0x2700, 0x27BF, "Dingbats"),
    (0x27F0, 0x27FF, "Supplemental Arrows-A"),
    (0x2900, 0x297F, "Supplemental Arrows-B"),
    (0x2B00, 0x2BFF, "Miscellaneous Symbols and Arrows"),
    (0x1F300, 0x1F5FF, "Miscellaneous Symbols and Pictographs"),
    (0x1F600, 0x1F64F, "Emoticons"),
    (0x1F650, 0x1F67F, "Ornamental Dingbats"),
    (0x1F680, 0x1F6FF, "Transport and Map Symbols"),
    (0x1F780, 0x1F7FF, "Geometric Shapes Extended"),
    (0x1F800, 0x1F8FF, "Supplemental Arrows-C"),
    (0x1F900, 0x1F9FF, "Supplemental Symbols and Pictographs"),
    (0x1FA70, 0x1FAFF, "Symbols and Pictographs Extended-A"),
)

#: A6 (fx-fmt-09): the separating space is a guard against "-5 degrees" and
#: "*emphasis*", not part of what makes a bullet a bullet. A dedicated bullet
#: glyph never needed it -- "•" is nobody's minus sign -- and demanding it let a
#: compact "•Preheat the oven" list walk straight through, which is exactly the
#: style a small model copies. So the requirement is narrowed to the characters
#: that really are ambiguous in prose rather than applied to every glyph. Every
#: entry must also be in LIST_BULLET_CHARS (asserted below).
LIST_BULLET_CHARS_NEEDING_SEPARATOR: Tuple[str, ...] = (
    "-", "*", "+", "–", "—", "·", "∙",
)

#: A6: an ORDERED marker always needs the separator. Without it "3.14" and
#: "1.5x" open a numbered list, which is a far worse trade than the compact
#: "1.Preheat" style this misses.
LIST_ORDINAL_NEEDS_SEPARATOR = True

#: A6: an ordered marker is at most this many digits, which is what separates a
#: list item ("1. Preheat the oven") from a sentence opening with a year
#: ("1985. That year everything changed"). Two digits covers every realistic
#: enumeration.
LIST_ORDINAL_MAX_DIGITS = 2

#: A6: single-letter and roman-numeral markers ("a) ", "iv. ") count as ordered
#: lists too. This is the ruling most likely to bite -- a line that opens
#: "A. Lincoln said" is read as a list marker -- so it is its own switch.
LIST_ORDINAL_ALPHA = True

#: A6 -- HANK'S RULING R-A6, this session: a list needs TWO items. A single
#: line-leading marker is not a list. A lone "- " line is usually a dash
#: continuing a sentence across a line break (fx-fmt-03), models write that
#: constantly, and firing on it rejects compliant prose and manufactures fake
#: difficulty in the training-data rejection filter. The cost is accepted and
#: named: a model can dodge "no lists" with exactly one bullet.
MIN_LIST_ITEMS = 2

#: A6 (fx-fmt-14): a list nested behind a quote prefix is still a list -- "> - "
#: is what a model emits when it quotes a plan back at you, and the literal
#: markdown marker is right there in the text. The prefix is consumed BEFORE the
#: marker is looked for. A blockquote with no marker inside it ("> Preheat the
#: oven") still is not a list, because consuming ">" leaves ordinary prose.
LIST_STRIPS_BLOCKQUOTE_PREFIX = True
LIST_BLOCKQUOTE_PREFIX_CHARS: Tuple[str, ...] = (">",)

#: A6 (fx-fmt-11): characters with no visual width can neither separate a marker
#: from its content nor stand in for the separating space. "-<U+200B> Preheat"
#: renders as an ordinary markdown bullet, but U+200B is not isspace(), so
#: keying on the raw next character rejected it. Cf covers the zero-width
#: space/joiners, the word joiner, the BOM and the bidi controls; Mn covers the
#: emoji variation selectors that trail a pictographic bullet. They are SKIPPED:
#: never counted as content, never accepted as a separator.
#:
#: HONEST STATUS -- this is a BACKSTOP, not a live switch. With the layer's
#: current settings the A2 fold already deletes these characters
#: (IGNORE_INVISIBLE_CHARACTERS), so flipping this one changes no verdict today;
#: it becomes observable the moment that knob or LIST_FOLDS_UNICODE_MARKERS is
#: turned off. It is kept deliberately: fx-fmt-11 is a no_lists repair and must
#: not silently depend on another checker's ruling staying where it is.
LIST_SKIPS_INVISIBLES = True
LIST_INVISIBLE_CATEGORIES: Tuple[str, ...] = ("Cf", "Mn")

#: A6 (fx-fmt-13): ONE normalization policy across the layer. check_banned_-
#: punctuation already folds fullwidth forms, so a fullwidth comma IS a comma;
#: a fullwidth "１．" that renders as an ordinary numbered list must then be a
#: list by the same rule. Marker detection therefore runs on the shared _fold()
#: output (NFKD + confusables, no casefold) and maps its offsets back through
#: the index map, so evidence still quotes the model's real characters.
LIST_FOLDS_UNICODE_MARKERS = True

#: A6 support: markdown-style markers may be indented.
LIST_MARKER_MAY_BE_INDENTED = True

#: A6: a list ITEM has to say something. Without this, a thematic break
#: ("* * *", "- - -") reads as a bullet whose content is more bullets, and a
#: horizontal rule is not a list. The item text must contain a letter or digit.
LIST_ITEM_NEEDS_WORD_CONTENT = True

#: A6 (fx-fmt-12, the same line-break gap as fx-str-10 in another function):
#: "line-leading" means leading a LINE, and \r, \x0b, \x0c, \x1c-\x1e, U+0085,
#: U+2028 and U+2029 all end one. Splitting on "\n" alone made a carriage-
#: return-separated list a single line with no marker anywhere in it. CRLF is
#: one break, not two.
LIST_LINE_BREAK_CHARS: Tuple[str, ...] = (
    "\n", "\r", "\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029",
)

# INVARIANT (not a knob). ``BARE_LINES_ARE_LISTS = False`` used to sit here and
# did nothing: it appeared only in its own definition and in the module
# docstring, so flipping it changed no behaviour. It is deleted rather than
# wired up, and the ruling it claimed to carry is restated here as enforced
# behaviour: a list requires an EXPLICIT marker, so newline-separated fragments
# with no marker character are never a list however parallel or short they look
# -- inferring enumeration from line breaks alone would fail every poem,
# address, signature block and line of dialogue. Wiring it up was the wrong
# repair: honouring a True branch means shipping an unruled parallelism
# heuristic into the ground-truth layer, and fx-fmt-04 (three stacked names) is
# an OPEN ambiguity on the sign-off ledger, not a confirmed defect. If bare
# lines are ever ruled to be lists, that is new code and new fixtures, never a
# constant flip.

_ORDINAL_DIGIT_RE = re.compile(
    r"^\(?(\d{1,%d})[.)\]]" % LIST_ORDINAL_MAX_DIGITS)
_ORDINAL_ALPHA_RE = re.compile(
    r"^\(?([A-Za-z]|[ivxIVX]{1,4})[.)\]]")

_LIST_LINE_BREAK_RE = re.compile(
    "\r\n|[%s]" % "".join(re.escape(c) for c in LIST_LINE_BREAK_CHARS))

assert set(LIST_BULLET_CHARS_NEEDING_SEPARATOR) <= set(LIST_BULLET_CHARS), \
    "a bullet that needs a separator must itself be a bullet"


def _iter_list_lines(text: str):
    """``(lineno, start_offset, line)`` for every line, split on ANY line break.

    Named for its caller on purpose: check_sentence_initial_letter has the same
    line-break gap (fx-str-10) but is owned by another lane; when that lands the
    two should collapse into one shared splitter.
    """
    pos, lineno = 0, 1
    for match in _LIST_LINE_BREAK_RE.finditer(text):
        yield lineno, pos, text[pos:match.start()]
        pos, lineno = match.end(), lineno + 1
    yield lineno, pos, text[pos:]


def _is_bullet_char(ch: str) -> bool:
    """A6: is ``ch`` a bullet glyph -- by table, or by category-and-block?"""
    if ch in LIST_BULLET_CHARS:
        return True
    if ch.isascii():
        return False
    if unicodedata.category(ch) not in LIST_BULLET_SYMBOL_CATEGORIES:
        return False
    code = ord(ch)
    return any(low <= code <= high for low, high, _name in LIST_BULLET_UNICODE_BLOCKS)


def _is_list_invisible(ch: str) -> bool:
    """A6: zero-width/format character -- present in the bytes, absent on screen.

    Deliberately NOT named ``_is_invisible``: the A2 normalization lane owns a
    module-level function by that name and a duplicate here silently rebound
    theirs. The two policies should be merged once that lane settles -- this one
    adds Mn so an emoji variation selector cannot pad a bullet.
    """
    return LIST_SKIPS_INVISIBLES and unicodedata.category(ch) in LIST_INVISIBLE_CATEGORIES


def _drop_leading_invisibles(text: str) -> str:
    idx = 0
    while idx < len(text) and _is_list_invisible(text[idx]):
        idx += 1
    return text[idx:]


def _marker_is_separated(rest: str) -> bool:
    """Is the marker followed by REAL whitespace? Invisibles do not separate."""
    return _drop_leading_invisibles(rest)[:1].isspace()


def _is_list_item_body(rest: str) -> bool:
    """Does the text after a marker look like an item rather than more marker?"""
    if not rest.strip():
        return False
    if LIST_ITEM_NEEDS_WORD_CONTENT and not any(ch.isalnum() for ch in rest):
        return False
    return True


def _list_marker_at(line: str) -> Optional[Tuple[int, int, str]]:
    """``(start, end, description)`` for a list marker on ``line``, else None.

    ``start`` and ``end`` are offsets into the ORIGINAL line even when detection
    ran on the folded form, so evidence always quotes the model's own text.
    """
    folded, imap = (_fold(line, casefold=False) if LIST_FOLDS_UNICODE_MARKERS
                    else (line, None))
    # indent, invisibles and any blockquote prefix are consumed first
    idx, indented = 0, False
    while idx < len(folded):
        ch = folded[idx]
        if ch.isspace():
            indented = True
        elif _is_list_invisible(ch):
            pass
        elif LIST_STRIPS_BLOCKQUOTE_PREFIX and ch in LIST_BLOCKQUOTE_PREFIX_CHARS:
            pass
        else:
            break
        idx += 1
    if idx >= len(folded):
        return None
    if indented and not LIST_MARKER_MAY_BE_INDENTED:
        return None
    body = folded[idx:]

    def _span(length: int, kind: str) -> Tuple[int, int, str]:
        start = _orig_start(imap, idx)
        end = _orig_end(imap, idx + length)
        return start, end, f"{kind} {line[start:end]!r}"

    # --- bullet ---
    if _is_bullet_char(body[0]):
        rest = body[1:]
        needs_separator = body[0] in LIST_BULLET_CHARS_NEEDING_SEPARATOR
        if _is_list_item_body(rest) and (_marker_is_separated(rest) or not needs_separator):
            return _span(1, "bullet")
    # --- ordered ---
    for regex, kind in ((_ORDINAL_DIGIT_RE, "numbered marker"),
                        (_ORDINAL_ALPHA_RE, "lettered marker") if LIST_ORDINAL_ALPHA
                        else (None, "")):
        if regex is None:
            continue
        match = regex.match(body)
        if match is None:
            continue
        rest = body[match.end():]
        if _is_list_item_body(rest) and (_marker_is_separated(rest)
                                         or not LIST_ORDINAL_NEEDS_SEPARATOR):
            return _span(match.end(), kind)
    return None


def check_no_lists(constraint: Dict[str, Any], text: str) -> Dict[str, Any]:
    """No bulleted or numbered list (A6)."""
    text = _coerce_text(text)
    validate_params("no_lists", constraint.get("params"))
    empty = _empty_verdict(text)
    if empty is not None:
        return empty
    hits: List[Tuple[int, int, str, int]] = []
    for lineno, line_start, line in _iter_list_lines(text):
        marker = _list_marker_at(line)
        if marker is not None:
            start, end, description = marker
            hits.append((line_start + start, line_start + end, description, lineno))
    if len(hits) < MIN_LIST_ITEMS:
        return _clean()
    start, end, description, lineno = hits[0]
    return _fail(f"list {description} on line {lineno}", text, start, end)


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
