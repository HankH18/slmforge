"""Sentence segmentation for ``sentence_initial_letter`` (A5).

``sentence_initial_letter`` is a HELD-OUT constraint type: it is the number that
proves the tuned model generalises to a type it never trained on. A boundary the
splitter swallows is not a wrong verdict on one sentence -- it hides EVERY
sentence after it, so the constraint silently becomes vacuous and the held-out
number measures nothing.

The golden fixtures pin one text per defect (fx-str-08/09/10). This file pins the
CLASS around each of them -- the sibling characters, the sibling words and the
non-regressions the repair had to keep -- because a fix that only satisfies the
one fixture text is a fix to the fixture, not to the checker.
"""
import pytest

from constraints.checkers import (
    CAN_BE_FINAL_ABBREVIATIONS,
    CAN_BE_FINAL_ONLY_AS_ORDINARY_WORD,
    NAME_TITLE_ABBREVIATIONS,
    NEVER_FINAL_ABBREVIATIONS,
    SENTENCE_LINE_BREAK_CHARS,
    _sentence_spans,
    check_sentence_initial_letter,
)

S = {"id": "c-str-t", "type": "sentence_initial_letter",
     "params": {"letter": "s"}, "family": "structural"}


def _spans(text):
    return [text[a:b] for a, b in _sentence_spans(text)]


def _passes(text):
    verdict = check_sentence_initial_letter(S, text)
    assert type(verdict["pass"]) is bool
    return verdict["pass"]


# --------------------------------------------------------------------------
# fx-str-10 -- every character that renders as a line break ends a sentence
# --------------------------------------------------------------------------

@pytest.mark.parametrize("brk", SENTENCE_LINE_BREAK_CHARS,
                         ids=[hex(ord(c)) for c in SENTENCE_LINE_BREAK_CHARS])
def test_every_line_break_character_ends_a_sentence(brk):
    """NEWLINE_TERMINATES_SENTENCE is a ruling about LINE BREAKS, not about the
    single codepoint U+000A. A response authored on Windows or round-tripped
    through a transcript carries '\\r', and U+2028/U+2029/U+0085 render as breaks
    too -- each one used to glue two visible lines into one graded sentence."""
    text = "Sam ran the numbers again" + brk + "Tom walked out before the end"
    assert _spans(text) == ["Sam ran the numbers again", "Tom walked out before the end"]
    assert _passes(text) is False


def test_crlf_is_one_boundary_not_two_sentences():
    """The empty segment between '\\r' and '\\n' is not a sentence."""
    text = "Sam ran the numbers.\r\nTom walked out."
    assert _spans(text) == ["Sam ran the numbers.", "Tom walked out."]


def test_line_break_table_is_the_set_python_itself_calls_a_line_break():
    """Anchored to ``str.splitlines()`` so a future edit cannot quietly shrink
    the table: the only permitted omissions are the three C0 separators."""
    probe = "a" + "b".join(SENTENCE_LINE_BREAK_CHARS) + "z"
    assert len(probe.splitlines()) == len(SENTENCE_LINE_BREAK_CHARS) + 1
    missing = {c for c in "\x1c\x1d\x1e"} - set(SENTENCE_LINE_BREAK_CHARS)
    assert missing == {"\x1c", "\x1d", "\x1e"}


# --------------------------------------------------------------------------
# fx-str-08 -- abbreviations that are also ordinary English words
# --------------------------------------------------------------------------

#: (text, expected spans) -- the ordinary-word form ends the sentence.
ORDINARY_WORD_ENDS_THE_SENTENCE = (
    ("Sam said no. Tom left the room.", ["Sam said no.", "Tom left the room."]),
    ("Sam sat. Tom stood.", ["Sam sat.", "Tom stood."]),
    ("Sam watched the sun. Tom watched the moon.", ["Sam watched the sun.",
                                                    "Tom watched the moon."]),
    ("Sam works for the co. Tom does not.", ["Sam works for the co.", "Tom does not."]),
    ("Sam did this, that, etc. Tom left.", ["Sam did this, that, etc.", "Tom left."]),
    # the same recipe with the abbreviation capitalised: "etc." never introduces
    # what follows, so its own case carries no information.
    ("Sam did this, that, Etc. Tom left.", ["Sam did this, that, Etc.", "Tom left."]),
)

#: (text, why) -- the abbreviation form keeps the sentence open.
ABBREVIATION_FORM_KEEPS_THE_SENTENCE_OPEN = (
    ("Sam moved to St. Louis so nobody argued.", "St. Louis"),
    ("Sam works at No. 7 Station so he walks.", "No. 7"),
    ("Sam signed with Acme Co. Ltd so the deal closed.", "Co. Ltd"),
    ("Sam met Rep. Smith so plans shifted.", "Rep. Smith"),
    ("Some sign read Est. 1920 so nobody argued.", "Est. 1920"),
    ("Shipment lands Sat. 14 March so we wait.", "Sat. 14"),
    ("Sam filed issue no. 7 so the queue cleared.", "lowercase before a digit"),
    ("Sam left the co. before anyone noticed.", "lowercase before a lowercase word"),
)


@pytest.mark.parametrize("text,expected", ORDINARY_WORD_ENDS_THE_SENTENCE)
def test_ordinary_word_ending_a_sentence_does_not_swallow_the_boundary(text, expected):
    """fx-str-08's gaming recipe: end every sentence with "etc." (or "no.", or
    "sat.") and nothing after the first sentence is ever graded."""
    assert _spans(text) == expected
    assert _passes(text) is False


@pytest.mark.parametrize("text,why", ABBREVIATION_FORM_KEEPS_THE_SENTENCE_OPEN)
def test_the_abbreviation_reading_still_suppresses_the_boundary(text, why):
    """The repair may not be "delete the entries": removing "no." wholesale
    re-breaks "No. 7 Station" and manufactures a violation out of compliant
    prose."""
    assert _spans(text) == [text], why
    assert _passes(text) is True


def test_abbreviation_tiers_are_disjoint_and_lowercased():
    """Three tables, one ruling each; an entry in two tiers would make the
    ruling depend on which membership test runs first. Note what is NOT asserted
    any more: that every entry has exactly one dot. Dot-count was the old tiering
    proxy and it was wrong in both directions, so multi-dot tokens are now placed
    by name in whichever tier their behaviour belongs to."""
    tiers = (NEVER_FINAL_ABBREVIATIONS, CAN_BE_FINAL_ABBREVIATIONS,
             CAN_BE_FINAL_ONLY_AS_ORDINARY_WORD)
    for tier in tiers:
        for token in tier:
            assert token == token.lower() and token.endswith("."), token
    for i, left in enumerate(tiers):
        for right in tiers[i + 1:]:
            assert not set(left) & set(right), sorted(set(left) & set(right))


def test_the_tiers_are_cut_by_finality_not_by_dot_count():
    """The ruling in one assertion. Dot-count was a proxy that failed BOTH ways,
    so each tier must contain a counterexample to it: a one-dot token that CAN
    end a sentence, and a two-dot token that never can."""
    assert "aug." in CAN_BE_FINAL_ABBREVIATIONS, "one dot, yet 'Sales rose in Aug.' ends there"
    assert "e.g." in NEVER_FINAL_ABBREVIATIONS, "two dots, yet it never ends a sentence"
    assert "i.e." in NEVER_FINAL_ABBREVIATIONS
    assert "u.s." in CAN_BE_FINAL_ABBREVIATIONS


def test_the_words_that_caused_fx_str_08_left_the_unconditional_tier():
    """The exact entries the audit named. If one of these is ever moved into
    ``NEVER_FINAL_ABBREVIATIONS``, every sentence after it goes ungraded."""
    for token in ("no.", "co.", "st.", "sat.", "sun.", "est.", "rep.", "etc."):
        assert token not in NEVER_FINAL_ABBREVIATIONS, token


def test_the_month_and_company_abbreviations_left_the_unconditional_tier():
    """The second half of the same defect, and the wider one: these were
    UNCONDITIONALLY never-final, so "Sales rose in Aug. Tom took the account."
    was one sentence to the checker and two to every reader. They compose, so a
    response could switch the whole constraint off in ordinary business prose."""
    for token in ("aug.", "sept.", "oct.", "fri.", "inc.", "ltd.", "corp.",
                  "univ.", "ave.", "blvd."):
        assert token not in NEVER_FINAL_ABBREVIATIONS, token
        assert token in CAN_BE_FINAL_ABBREVIATIONS, token


# --------------------------------------------------------------------------
# TIER 1 -- never final: the FALSE-FAIL direction
# --------------------------------------------------------------------------

CLAUSE_OPENING_IS_ONE_SENTENCE = (
    ("Several teams already use it, e.g. Tom's group in Denver.", "e.g. opens a clause"),
    ("Shipping slows in winter, i.e. December through February.", "i.e. opens a clause"),
    ("Several sources disagree, cf. Nelson on the same point.", "cf. points forward"),
    ("Several formats work, viz. Markdown and plain text.", "viz. points forward"),
    ("Sam met Dr. Smith so plans shifted.", "a title introduces the name"),
    ("Some memo cited Fig. Three and stopped there.", "fig. introduces its number"),
)


@pytest.mark.parametrize("text,why", CLAUSE_OPENING_IS_ONE_SENTENCE, ids=[w for _, w in CLAUSE_OPENING_IS_ONE_SENTENCE])
def test_a_never_final_abbreviation_is_not_a_boundary_even_before_a_capital(text, why):
    """The false-FAIL direction, and the one that biases the held-out number
    toward "harder than it is". A clause-opening abbreviation introduces what
    follows and never ends an English sentence, so a capitalised follower is not
    evidence of anything -- "e.g. Tom's group" is one sentence, and reporting a
    sentence beginning with T is a violation manufactured out of ordinary prose."""
    assert _spans(text) == [text], why
    assert _passes(text) is True


@pytest.mark.parametrize("token", NEVER_FINAL_ABBREVIATIONS)
def test_every_never_final_entry_holds_the_sentence_open(token):
    """The class, not the samples: EVERY entry means the same thing, so every
    entry is checked against a capitalised follower -- the case that used to
    split them."""
    text = f"Something noted {token} Tom left the room."
    assert _spans(text) == [text], token
    assert _passes(text) is True


# --------------------------------------------------------------------------
# TIER 2 -- can be final: the FALSE-PASS direction
# --------------------------------------------------------------------------

@pytest.mark.parametrize("token", CAN_BE_FINAL_ABBREVIATIONS)
def test_every_can_be_final_entry_splits_before_a_capital_and_not_before_a_lowercase(token):
    """Both halves of the tier-2 rule over the whole table. Before the repair the
    single-dot half of this table was UNCONDITIONALLY never-final, so each entry
    swallowed the sentence after it; a table that only got the samples right
    would leave the next entry open."""
    split = f"Something ended in {token} Tom left the room."
    assert len(_sentence_spans(split)) == 2, f"{token}: capitalised follower is a new sentence"
    assert _passes(split) is False
    kept = f"Something ended in {token} so nobody noticed."
    assert _spans(kept) == [kept], f"{token}: a lowercase follower continues the sentence"
    assert _passes(kept) is True


ORDINARY_BUSINESS_PROSE_IS_FULLY_GRADED = (
    ("Sales rose in Aug. Tom took the account. Shipments fell in Sept. "
     "Rachel escalated it. Supplies ran short by Fri. Nobody noticed.", 6),
    ("Shipments left Acme Inc. Tom signed for them.", 2),
    ("Sam walked down Third Ave. Tom waited on the corner.", 2),
    ("Staff met at the Univ. Nobody else came.", 2),
)


@pytest.mark.parametrize("text,count", ORDINARY_BUSINESS_PROSE_IS_FULLY_GRADED,
                         ids=["composes", "inc", "ave", "univ"])
def test_can_be_final_abbreviations_no_longer_hide_the_sentences_after_them(text, count):
    """The defect in the shape it would really arrive in: natural business prose,
    no adversarial characters. The six-sentence case is the one that matters --
    the abbreviations COMPOSE, so before the repair three of the six sentences
    were graded and the three beginning T, R and N were invisible."""
    assert len(_sentence_spans(text)) == count
    assert _passes(text) is False


# --------------------------------------------------------------------------
# fx-str-09 -- one-letter tokens
# --------------------------------------------------------------------------

ONE_LETTER_WORD_ENDS_THE_SENTENCE = (
    "Sam scored an A. Tom scored a B.",
    "Sam set the variable to X. Tom reset it.",
    "Sam answered I. Tom answered II.",
)

NAME_INITIAL_KEEPS_THE_SENTENCE_OPEN = (
    ("Some report from Dr. J. Smith says so.", "initial after a title"),
    ("Sam read J. R. R. Tolkien slowly.", "initial run"),
    ("Sam cited Prof. K. Nelson so the claim stands.", "initial after a title"),
)


@pytest.mark.parametrize("text", ONE_LETTER_WORD_ENDS_THE_SENTENCE)
def test_a_lone_one_letter_token_does_not_suppress_the_boundary(text):
    """fx-str-09: a grade, a label or a variable name is not a name initial, and
    treating it as one hid every following sentence."""
    assert len(_sentence_spans(text)) == 2
    assert _passes(text) is False


@pytest.mark.parametrize("text,why", NAME_INITIAL_KEEPS_THE_SENTENCE_OPEN)
def test_an_initial_with_name_evidence_still_suppresses_the_boundary(text, why):
    assert _spans(text) == [text], why
    assert _passes(text) is True


def test_name_titles_are_recognised_abbreviations_in_their_own_right():
    """The initial rule reads ``NAME_TITLE_ABBREVIATIONS`` for evidence; each
    entry must also be handled by one of the abbreviation tiers, or "Dr." would
    end a sentence while still vouching for the initial after it."""
    known = (set(NEVER_FINAL_ABBREVIATIONS) | set(CAN_BE_FINAL_ABBREVIATIONS)
             | set(CAN_BE_FINAL_ONLY_AS_ORDINARY_WORD))
    assert set(NAME_TITLE_ABBREVIATIONS) <= known


# --------------------------------------------------------------------------
# Non-regressions the repair had to keep (fx-str-01..fx-str-05)
# --------------------------------------------------------------------------

UNSPLIT = (
    ("Sam cut it to 3.5 percent so nobody noticed.", "decimal point"),
    ("Sam checked example.com so nobody worried.", "domain"),
    ("Sam left the U.S.A. so nobody noticed.", "multi-dot abbreviation"),
    ("Sam met Dr. Whitfield so plans shifted.", "title"),
    ("Sam waited until Feb. 3 so the invoice cleared.", "month"),
)


@pytest.mark.parametrize("text,why", UNSPLIT)
def test_the_split_is_still_abbreviation_and_decimal_aware(text, why):
    assert _spans(text) == [text], why
    assert _passes(text) is True


def test_a_terminated_and_an_unterminated_sentence_are_both_graded():
    text = "Sam shipped it. Tom did not"
    assert _spans(text) == ["Sam shipped it.", "Tom did not"]
    assert _passes(text) is False


# --------------------------------------------------------------------------
# fx-str-07 -- cross-lane: invisible characters
# --------------------------------------------------------------------------

def test_segmentation_is_correct_once_invisible_characters_are_stripped():
    """fx-str-07 (U+200B after a terminator collapsing the response into one
    unchecked sentence) is repaired in the normalisation path, not here. This is
    the other half of that contract: with the invisibles removed, segmentation
    must already produce the four sentences -- so the only thing the
    normalisation fix has to deliver is the stripping itself."""
    hostile = ("Sam went to the market.​ Tom bought bread.​ "
               "Rachel paid.​ Nobody complained.")
    zero_width_space = "​"
    assert zero_width_space in hostile
    stripped = hostile.replace(zero_width_space, "")
    assert _spans(stripped) == ["Sam went to the market.", "Tom bought bread.",
                                "Rachel paid.", "Nobody complained."]
    assert check_sentence_initial_letter(S, stripped)["pass"] is False
