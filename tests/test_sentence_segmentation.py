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
    EXPRESSION_CONTINUING_FOLLOWERS,
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


#: The tier-1 members this file requires BY NAME. Written out as literals on
#: purpose: the behaviour test below used to parametrize over
#: ``NEVER_FINAL_ABBREVIATIONS`` itself, which made it self-referential --
#: deleting "al." from the table deleted its own test case, and the suite
#: reported 532 passed / 0 failed for a table that had lost a member. A literal
#: list turns a membership change into a LOUD failure instead of a quieter
#: suite. New members are still covered: the parametrization below is the union
#: of this list and the live table.
NEVER_FINAL_TOKENS_THAT_MUST_BE_PRESENT = (
    "mr.", "mrs.", "ms.", "dr.", "prof.", "rev.", "hon.",
    "jr.", "sr.", "col.", "capt.", "lt.", "sgt.", "gov.", "sen.", "mt.",
    "vs.", "approx.", "fig.", "vol.", "al.",
    "e.g.", "i.e.", "cf.", "viz.",
)


@pytest.mark.parametrize("token", NEVER_FINAL_TOKENS_THAT_MUST_BE_PRESENT)
def test_the_never_final_table_still_contains_every_token_it_is_required_to(token):
    """Membership, asserted against a literal list rather than against the table
    under test. Removing an entry now fails here AND fails behaviourally below;
    before, it merely removed a test case."""
    assert token in NEVER_FINAL_ABBREVIATIONS, (
        f"{token!r} left NEVER_FINAL_ABBREVIATIONS: every sentence after it "
        f"is now split off and graded as its own sentence")


@pytest.mark.parametrize(
    "token",
    sorted(set(NEVER_FINAL_TOKENS_THAT_MUST_BE_PRESENT) | set(NEVER_FINAL_ABBREVIATIONS)))
def test_every_never_final_entry_holds_the_sentence_open(token):
    """The class, not the samples: EVERY entry means the same thing, so every
    entry is checked against a capitalised follower -- the case that used to
    split them. Parametrized over the union of the required literals and the
    live table, so a token dropped from the table is still exercised (and
    fails), and a token added to it is covered on arrival."""
    text = f"Something noted {token} Tom left the room."
    assert _spans(text) == [text], token
    assert _passes(text) is True


#: The same defence for tier 2, whose behaviour test is parametrized over the
#: live table for the same reason and would shrink the same silent way.
CAN_BE_FINAL_TOKENS_THAT_MUST_BE_PRESENT = (
    "jan.", "feb.", "apr.", "jun.", "jul.", "aug.", "sep.", "sept.",
    "oct.", "nov.", "dec.",
    "mon.", "tue.", "tues.", "wed.", "thu.", "thurs.", "fri.",
    "inc.", "ltd.", "corp.", "univ.", "ave.", "blvd.", "rd.", "sq.", "dept.",
    "etc.", "u.s.", "u.k.", "a.m.", "p.m.",
)


@pytest.mark.parametrize("token", CAN_BE_FINAL_TOKENS_THAT_MUST_BE_PRESENT)
def test_the_can_be_final_table_still_contains_every_token_it_is_required_to(token):
    """Dropping one of these hides every sentence after it (if it moves to tier
    1) or is simply untested (if it is deleted). Literals, not the table."""
    assert token in CAN_BE_FINAL_ABBREVIATIONS, token


# --------------------------------------------------------------------------
# TIER 2 -- can be final: the FALSE-PASS direction
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "token",
    sorted(set(CAN_BE_FINAL_TOKENS_THAT_MUST_BE_PRESENT) | set(CAN_BE_FINAL_ABBREVIATIONS)))
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


# --------------------------------------------------------------------------
# ITEM 1 -- a can-be-final abbreviation followed by the NEXT PIECE of the same
# expression. Re-tiering weekdays and months as can-be-final split a date
# between its own halves ("Shipments resume Fri. Oct. 3." -> ["Shipments resume
# Fri.", "Oct. 3."]), a false FAIL on clean prose in the direction that inflates
# the apparent difficulty of the HELD-OUT type.
# --------------------------------------------------------------------------

#: Literal, not derived from the tables under test: this is the list the class
#: is defined by, so shrinking a table must fail here rather than quietly
#: shrinking the number of cases.
WEEKDAY_ABBREVIATIONS = ("Mon.", "Tue.", "Tues.", "Wed.", "Thu.", "Thurs.", "Fri.")

#: Months that do NOT begin with S, so a spurious second span is a real failure
#: of a letter-"s" constraint rather than an accidental pass.
MONTH_ABBREVIATIONS_NOT_BEGINNING_WITH_S = (
    "Jan.", "Feb.", "Apr.", "Jun.", "Jul.", "Aug.", "Oct.", "Nov.", "Dec.")

DATE_PAIRS = tuple((w, m) for w in WEEKDAY_ABBREVIATIONS
                   for m in MONTH_ABBREVIATIONS_NOT_BEGINNING_WITH_S)


def test_the_date_abbreviations_are_the_tokens_this_class_is_built_from():
    """Binds the literals above to the real tables. If a weekday or a month
    leaves tier 2, or a month leaves the continuing-follower table, the class
    below stops testing what it claims to and this says so first."""
    assert len(DATE_PAIRS) == 63
    for token in WEEKDAY_ABBREVIATIONS + MONTH_ABBREVIATIONS_NOT_BEGINNING_WITH_S:
        assert token.lower() in CAN_BE_FINAL_ABBREVIATIONS, token
    for month in MONTH_ABBREVIATIONS_NOT_BEGINNING_WITH_S:
        assert month.lower() in EXPRESSION_CONTINUING_FOLLOWERS, month


def test_every_continuing_follower_is_a_known_abbreviation():
    """The follower table may not smuggle in a token no tier knows about: it is
    a statement about abbreviations that continue an expression, not a private
    allowlist that can suppress boundaries after arbitrary words."""
    known = (set(NEVER_FINAL_ABBREVIATIONS) | set(CAN_BE_FINAL_ABBREVIATIONS)
             | set(CAN_BE_FINAL_ONLY_AS_ORDINARY_WORD))
    assert set(EXPRESSION_CONTINUING_FOLLOWERS) <= known
    for token in EXPRESSION_CONTINUING_FOLLOWERS:
        assert token == token.lower() and token.endswith("."), token
    # tier 1 is deliberately NOT a continuing follower: "Dr.", "Mr." and the
    # clause openers routinely START a sentence ("Sales rose in Aug. Dr. Rivera
    # resigned."), and suppressing that boundary would hide unbounded text.
    assert not set(EXPRESSION_CONTINUING_FOLLOWERS) & set(NEVER_FINAL_ABBREVIATIONS)


@pytest.mark.parametrize("weekday,month", DATE_PAIRS,
                         ids=[f"{w}{m}" for w, m in DATE_PAIRS])
def test_a_date_is_never_split_between_its_own_halves(weekday, month):
    """All 63 weekday x non-S-month pairs. Every one of them false-failed after
    the re-tiering: the second span began with the month's capital letter, so a
    letter-"s" constraint was reported violated by prose that never violated
    it."""
    text = f"Shipments resume {weekday} {month} 3."
    assert _spans(text) == [text], f"{weekday} {month} split inside the date"
    assert _passes(text) is True


@pytest.mark.parametrize("weekday", WEEKDAY_ABBREVIATIONS)
def test_a_weekday_that_really_does_end_the_sentence_still_splits(weekday):
    """The other direction of the same rule, over the same tokens: the fix is
    "the follower is the next piece of this expression", NOT "a weekday never
    ends a sentence". An ordinary capitalised word after it is still a new
    sentence and is still graded."""
    text = f"Supplies ran short by {weekday} Nobody noticed the delay."
    assert len(_sentence_spans(text)) == 2, weekday
    assert _passes(text) is False


CONTINUING_EXPRESSION_IS_ONE_SENTENCE = (
    ("Shipments resume Fri. Oct. 3.", "weekday then month then day"),
    ("Shipments resume Fri. Oct. 3, 2026, as planned.", "the same date in full"),
    ("Sam waited until Oct. 3 so the invoice cleared.", "month then bare day"),
    ("Staff met at the U.S. Dept. of Commerce so nobody travelled.",
     "an organisation's name continuing through two abbreviations"),
    ("Sam signed for Acme Corp. Ltd. so the deal closed.", "name suffixes in a row"),
    ("Standup moved to Mon. a.m. so the room is free.", "the halves of a clock time"),
)

STILL_TWO_SENTENCES = (
    ("Supplies ran short by Fri. Nobody noticed.", "a capitalised ordinary word"),
    ("Sales rose in Aug. Tom took the account.", "a name, not an abbreviation"),
    ("Shipments resume Fri. October arrived badly.",
     "the month spelled out is an ordinary word, so it opens a sentence"),
    ("Shipments left Acme Inc. Tom signed for them.", "the pinned tier-2 case"),
    ("Sales rose in Aug. Dr. Rivera resigned that week.",
     "a title opens a sentence, so tier 1 is not a continuing follower"),
)


@pytest.mark.parametrize("text,why", CONTINUING_EXPRESSION_IS_ONE_SENTENCE,
                         ids=[w for _, w in CONTINUING_EXPRESSION_IS_ONE_SENTENCE])
def test_a_continuing_follower_is_not_a_boundary(text, why):
    assert _spans(text) == [text], why
    assert _passes(text) is True


@pytest.mark.parametrize("text,why", STILL_TWO_SENTENCES,
                         ids=[w for _, w in STILL_TWO_SENTENCES])
def test_a_genuinely_sentence_final_abbreviation_still_splits(text, why):
    """The repair may not be bought with swallowed boundaries: everything the
    re-tiering was FOR still splits."""
    assert len(_sentence_spans(text)) == 2, why
    assert _passes(text) is False


def test_a_bare_number_after_a_month_is_the_day_not_a_new_sentence():
    """NUMBER_AFTER_ABBREVIATION_CONTINUES_THE_EXPRESSION, stated where it can
    be seen: the day of a date is a continuation on its own evidence, not
    because a digit happens not to be an uppercase letter."""
    from constraints.checkers import _follower_continues_the_expression
    text = "Sam waited until Oct. 3."
    assert _follower_continues_the_expression(text, text.index("Oct.") + 4) is True
    assert _spans(text) == ["Sam waited until Oct. 3."]
    assert _spans("Sam waited until Oct. 30 so the invoice cleared.") == [
        "Sam waited until Oct. 30 so the invoice cleared."]


# --------------------------------------------------------------------------
# ITEM 4 -- an abbreviation inside brackets or quotes. ``_preceding_token``
# returns the token verbatim, so the lookup key was "(e.g." and missed every
# table: the parenthesised form of the clause openers false-failed at HEAD and
# at base, and the whole class was invisible to the suite in BOTH directions.
# --------------------------------------------------------------------------

BRACKETED_ABBREVIATION_IS_ONE_SENTENCE = (
    ("Several suppliers already use it (e.g. Rivera's group in Denver).",
     "(e.g. -- the exact text the verifier proved false-failing"),
    ("Sourcing is the bottleneck (i.e. Rivera cannot buy the parts).", "(i.e."),
    ("Sign-off came from the lead reviewer (Dr. Ramirez) last week.", "(Dr."),
    ("Several teams use it [e.g. Rivera's group] already.", "[e.g."),
    ("Several sources disagree (cf. Nelson on the same point).", "(cf."),
    ("Some memo said “e.g. Rivera signs for them” and stopped there.",
     "a curly opening quote"),
    ("Some note read 'i.e. Rivera signs' and nothing else.", "an apostrophe"),
    ("Shipments resume (Fri. Oct. 3) as planned.",
     "both repairs at once: a bracketed weekday and a continuing follower"),
    ("Some report from (Dr. J. Smith) says so.",
     "a bracketed title still vouches for the initial after it"),
)

BRACKETED_TOKEN_THAT_IS_NOT_AN_ABBREVIATION_STILL_SPLITS = (
    ("Several suppliers use it (Rivera. Tom disagreed loudly.)",
     "stripping brackets does not make every bracketed token an abbreviation"),
    ("Shipments resume (Fri. Tom left the room.)",
     "a bracketed tier-2 token is placed in tier 2, not blanket-suppressed"),
)


@pytest.mark.parametrize("text,why", BRACKETED_ABBREVIATION_IS_ONE_SENTENCE,
                         ids=[w for _, w in BRACKETED_ABBREVIATION_IS_ONE_SENTENCE])
def test_a_bracketed_or_quoted_abbreviation_is_looked_up_like_a_bare_one(text, why):
    """One sentence, beginning with S. Splitting it reports a sentence beginning
    with R (or J), which is a violation manufactured out of ordinary prose."""
    assert _spans(text) == [text], why
    assert _passes(text) is True


@pytest.mark.parametrize("text,why", BRACKETED_TOKEN_THAT_IS_NOT_AN_ABBREVIATION_STILL_SPLITS,
                         ids=["ordinary word", "tier-2 token"])
def test_stripping_wrappers_does_not_suppress_boundaries_by_itself(text, why):
    """The other direction: the strip feeds a TABLE LOOKUP, it is not a licence
    to swallow a boundary after anything wearing a bracket."""
    assert len(_sentence_spans(text)) == 2, why
    assert _passes(text) is False
