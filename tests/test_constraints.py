"""T-1 Verify: the ground-truth layer's binding test suite.

Two independent halves, deliberately kept apart:

**(a) Golden fixtures.** ``data/fixtures/checker_golden.jsonl`` holds human
rulings written against the plain-English spec. Rows split three ways:

* ``ambiguous == false`` and no ``known_defect`` -> HARD ASSERTION. These are the
  rows where the spec is unarguable; the checker must agree.
* ``ambiguous == true`` -> ``xfail(strict=False)``. Hank has not ruled yet, so
  neither verdict is a bug. The xfail reason carries the question and both
  readings, so ``pytest -rxX`` prints the whole ledger he signs off on. When he
  rules, drop the flag and the row becomes a hard assertion.
* ``known_defect`` present -> ``xfail(strict=True)``. A CONFIRMED defect: the
  checker is wrong by any careful reading, reproduced and named. Strict, so that
  repairing the checker turns the row red and forces whoever fixes it to delete
  the marker rather than leaving a stale one behind.

**(b) Contract tests.** Fixture-independent. They bind the shape of the
interface, purity, the raising policy, the split invariant and the type-safety
of every numeric/param validation, so a rewritten checker still cannot drift.

Nothing here touches ``tests/test_checkers.py`` (abstainer phase) or any other
existing suite file.
"""
from __future__ import annotations

import ast
import copy
import json
import os
import re
import subprocess
import sys
import textwrap

import pytest

from conftest import REPO_ROOT

from constraints import checkers as CHECKERS_MODULE
from constraints.checkers import CLEAN_EVIDENCE, check
from constraints.registry import (
    FAMILIES,
    SPLITS,
    TYPES,
    ConstraintError,
    build_constraint,
    checker_for,
    list_types,
    types_by_split,
    validate_constraint,
)

FIXTURES_PATH = REPO_ROOT / "data" / "fixtures" / "checker_golden.jsonl"
CHECKERS_PATH = REPO_ROOT / "constraints" / "checkers.py"
REGISTRY_PATH = REPO_ROOT / "constraints" / "registry.py"

CATEGORIES = ("clean", "clear_violation", "near_miss", "ambiguous")

#: Every ambiguity a fixture may cite. A1-A7 are the ticket's; A8 was discovered
#: by adversarial probing (raw text vs. rendered surface) and is carried here so
#: the sign-off ledger is closed -- a fixture may not invent an unlisted code.
AMBIGUITY_CODES = ("A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8")

#: The one and only shape ``check`` may return.
VERDICT_KEYS = {"pass", "evidence"}

_OFFSET_RE = re.compile(r" at offset (\d+): ")
_MARKED_RE = re.compile(r"\[\[(.*)\]\]", re.S)


# ==========================================================================
# Fixture loading
# ==========================================================================

def _load_fixtures():
    rows = []
    for line in FIXTURES_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


FIXTURES = _load_fixtures()

HARD_ROWS = [r for r in FIXTURES if not r.get("known_defect") and not r["ambiguous"]]
AMBIGUOUS_ROWS = [r for r in FIXTURES if not r.get("known_defect") and r["ambiguous"]]
DEFECT_ROWS = [r for r in FIXTURES if r.get("known_defect")]


def _short(text, limit=260):
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _hard_params():
    return [pytest.param(r, id=r["fixture_id"]) for r in HARD_ROWS]


def _ambiguous_params():
    out = []
    for row in AMBIGUOUS_ROWS:
        reason = f"AMBIGUOUS, awaiting sign-off -- {_short(row['rationale'])}"
        out.append(pytest.param(row, id=row["fixture_id"],
                                marks=pytest.mark.xfail(reason=reason, strict=False)))
    return out


def _defect_params():
    if not DEFECT_ROWS:
        # Every confirmed defect has been repaired. The harness stays wired for
        # the next one; the placeholder keeps pytest from reporting an
        # empty-parameter-set SKIP, which reads like a switched-off test rather
        # than an empty ledger.
        return [pytest.param(None, id="defect-ledger-is-empty")]
    out = []
    for row in DEFECT_ROWS:
        reason = (f"KNOWN DEFECT in constraints/checkers.py -- {_short(row['known_defect'])} "
                  f"| when this is repaired, delete the 'known_defect' key from "
                  f"{row['fixture_id']} so the row becomes a hard assertion")
        out.append(pytest.param(row, id=row["fixture_id"],
                                marks=pytest.mark.xfail(reason=reason, strict=True)))
    return out


# ==========================================================================
# Shared assertions
# ==========================================================================

def _assert_verdict_shape(result, text):
    """The pinned contract: exactly {"pass": bool, "evidence": str}, and evidence
    is either exactly "clean" or a locating description of a real offset."""
    assert isinstance(result, dict), f"check must return a dict, got {type(result).__name__}"
    assert set(result) == VERDICT_KEYS, f"verdict keys must be exactly {VERDICT_KEYS}, got {set(result)}"
    assert type(result["pass"]) is bool, (
        f"'pass' must be a real bool, got {type(result['pass']).__name__} "
        f"({result['pass']!r}) -- an int or numpy bool would silently pass 'is True' checks")
    assert isinstance(result["evidence"], str), "'evidence' must be a str"

    if result["pass"]:
        assert result["evidence"] == CLEAN_EVIDENCE, (
            f"a passing check must report evidence exactly {CLEAN_EVIDENCE!r}, "
            f"got {result['evidence']!r}")
        return

    evidence = result["evidence"]
    assert evidence, "a failing check must report non-empty evidence"
    assert evidence != CLEAN_EVIDENCE, "a failing check must not report 'clean'"
    if not (text or "").strip():
        return          # degenerate-input evidence carries no offset by design
    match = _OFFSET_RE.search(evidence)
    assert match, f"failing evidence must locate the violation with 'at offset N:', got {evidence!r}"
    offset = int(match.group(1))
    assert 0 <= offset < len(text), (
        f"evidence offset {offset} is not an index into a {len(text)}-character response")
    marked = _MARKED_RE.search(evidence)
    assert marked and marked.group(1), (
        f"failing evidence must mark the violating span with [[...]], got {evidence!r}")


def _assert_fixture(row):
    result = check(row["constraint"], row["text"])
    _assert_verdict_shape(result, row["text"])
    assert result["pass"] == row["expected_pass"], (
        f"{row['fixture_id']} [{row['constraint']['type']} / {row['category']}]: "
        f"the human ruling is expected_pass={row['expected_pass']}, the checker returned "
        f"{result['pass']} (evidence {result['evidence']!r}).\nRATIONALE: {row['rationale']}")


# ==========================================================================
# (a) Golden-fixture tests
# ==========================================================================

def test_fixture_file_is_well_formed_and_schema_valid():
    assert FIXTURES, "checker_golden.jsonl is empty"
    seen_ids, seen_pairs = set(), set()
    for row in FIXTURES:
        fid = row["fixture_id"]
        assert fid not in seen_ids, f"duplicate fixture_id {fid}"
        seen_ids.add(fid)

        known = {"fixture_id", "constraint", "text", "expected_pass",
                 "category", "rationale", "ambiguous", "known_defect"}
        extra = set(row) - known
        assert not extra, f"{fid}: unknown fixture key(s) {sorted(extra)}"
        for key in ("fixture_id", "constraint", "text", "expected_pass",
                    "category", "rationale", "ambiguous"):
            assert key in row, f"{fid}: missing required key {key!r}"

        assert isinstance(row["text"], str), f"{fid}: text must be a string"
        assert type(row["expected_pass"]) is bool, f"{fid}: expected_pass must be a real bool"
        assert type(row["ambiguous"]) is bool, f"{fid}: ambiguous must be a real bool"
        assert row["category"] in CATEGORIES, f"{fid}: unknown category {row['category']!r}"
        assert len(row["rationale"]) >= 40, f"{fid}: rationale is too thin to audit"

        # A constraint that does not validate would make the row untestable.
        validate_constraint(row["constraint"])

        if row["category"] == "ambiguous":
            assert row["ambiguous"], f"{fid}: category 'ambiguous' must set ambiguous=true"
        if "known_defect" in row:
            assert isinstance(row["known_defect"], str) and row["known_defect"], \
                f"{fid}: known_defect must be a non-empty string"
            assert not row["ambiguous"], (
                f"{fid}: a confirmed defect is not an open question -- it is either a defect "
                f"(checker wrong by any reading) or an ambiguity (spec undecided), never both")
            assert row["category"] == "near_miss", f"{fid}: a defect row is a near_miss"

        pair = (json.dumps(row["constraint"], sort_keys=True), row["text"])
        assert pair not in seen_pairs, f"{fid}: duplicate (constraint, text) pair"
        seen_pairs.add(pair)


def test_every_ambiguous_fixture_names_the_ambiguity_it_rules_on():
    """An ambiguous row that does not say WHICH question it answers cannot be
    signed off -- Hank would be approving an unlabelled opinion."""
    unlabelled = []
    for row in FIXTURES:
        if not row["ambiguous"]:
            continue
        if not re.search(r"\bA[1-8]\b", row["rationale"]):
            unlabelled.append(row["fixture_id"])
    assert not unlabelled, (
        f"ambiguous fixtures citing no ambiguity code from {list(AMBIGUITY_CODES)}: {unlabelled}")


def test_every_type_is_covered_by_fixtures_in_both_directions():
    by_type = {}
    for row in FIXTURES:
        by_type.setdefault(row["constraint"]["type"], []).append(row)
    assert set(by_type) == set(TYPES), (
        f"fixture coverage gap: {sorted(set(TYPES) - set(by_type))} have no fixtures")
    for ctype, rows in sorted(by_type.items()):
        assert len(rows) >= 3, f"{ctype}: only {len(rows)} fixture(s)"
        verdicts = {r["expected_pass"] for r in rows}
        assert verdicts == {True, False}, (
            f"{ctype}: fixtures only ever expect {verdicts} -- a checker hardcoded to that "
            f"answer would pass every row")


@pytest.mark.parametrize("row", _hard_params())
def test_golden_fixture_unambiguous(row):
    """The binding set: the spec admits one reading and the checker must match it."""
    _assert_fixture(row)


@pytest.mark.parametrize("row", _ambiguous_params())
def test_golden_fixture_ambiguous_pending_signoff(row):
    """Reported, not enforced. Becomes a hard assertion once Hank rules."""
    _assert_fixture(row)


@pytest.mark.parametrize("row", _defect_params())
def test_golden_fixture_known_defect(row):
    """A reproduced defect in constraints/checkers.py. Strict xfail: fixing the
    checker fails this test, which is the signal to delete the known_defect key."""
    if row is None:
        assert not DEFECT_ROWS, "placeholder ran with defect rows present"
        return
    _assert_fixture(row)


def test_signoff_ledger_is_complete():
    """Every open question is either ruled (hard) or on the sign-off ledger, and
    the ledger is non-empty -- silence here would mean the ambiguities were
    resolved unilaterally instead of surfaced."""
    assert AMBIGUOUS_ROWS, "no ambiguities surfaced; A1-A7 are pinned as open by the ticket"
    codes = set()
    for row in AMBIGUOUS_ROWS:
        codes.update(re.findall(r"\bA[1-8]\b", row["rationale"]))
    missing = [c for c in AMBIGUITY_CODES if c not in codes]
    assert not missing, f"no fixture puts these ambiguities on the ledger: {missing}"
    print("\nSIGN-OFF LEDGER: %d ambiguous rows across %s; %d confirmed defects; "
          "%d hard assertions." % (len(AMBIGUOUS_ROWS), sorted(codes),
                                   len(DEFECT_ROWS), len(HARD_ROWS)))


# ==========================================================================
# (b) Contract tests -- fixture-independent
# ==========================================================================

def _representative(ctype):
    """A pinned-shape constraint for ``ctype`` built from the registry's own
    example params, so these tests never encode a second source of truth."""
    return build_constraint(ctype, copy.deepcopy(TYPES[ctype].example_params), 1)


#: (type, violating text, the offset a correct checker must report)
VIOLATION_SAMPLES = (
    ("banned_words", "I need coffee now.", 7),
    ("banned_phrases", "In conclusion, we ship.", 0),
    ("banned_letter", "The end.", 2),
    ("banned_punctuation", "First, we ship.", 5),
    ("sentence_initial_letter", "Tom left.", 0),
    ("no_lists", "Steps:\n- one\n- two", 7),
    ("no_quotation_marks", 'He said "no".', 8),
)

#: (type, obeying text)
CLEAN_SAMPLES = (
    ("banned_words", "Just water today."),
    ("banned_phrases", "To sum up, we ship."),
    ("banned_letter", "Only calm plans in play."),
    ("banned_punctuation", "First we ship. Nothing broke."),
    ("sentence_initial_letter", "Sam left. Sarah stayed."),
    ("no_lists", "We shipped it on Tuesday without any fuss."),
    ("no_quotation_marks", "He said no and left."),
)

_VIOLATION_BY_TYPE = {t: text for t, text, _ in VIOLATION_SAMPLES}
_CLEAN_BY_TYPE = dict(CLEAN_SAMPLES)

assert set(_VIOLATION_BY_TYPE) == set(TYPES), "violation samples must cover every type"
assert set(_CLEAN_BY_TYPE) == set(TYPES), "clean samples must cover every type"


def test_every_type_has_exactly_one_checker_and_every_checker_is_registered():
    registered = {spec.checker_name for spec in TYPES.values()}
    assert len(registered) == len(TYPES), "two types share a checker"

    exported = {n for n in CHECKERS_MODULE.__all__ if n.startswith("check_")}
    assert exported == registered, (
        f"checkers.__all__ and the registry disagree: only in __all__ "
        f"{sorted(exported - registered)}, only in registry {sorted(registered - exported)}")

    defined = {n for n, v in vars(CHECKERS_MODULE).items()
               if n.startswith("check_") and callable(v)}
    assert defined == registered, (
        f"an unregistered checker exists in constraints/checkers.py: "
        f"{sorted(defined - registered)} -- every checker must be reachable through the registry")

    for ctype, spec in TYPES.items():
        assert checker_for(ctype) is getattr(CHECKERS_MODULE, spec.checker_name)


def test_checkers_called_directly_also_honour_the_verdict_contract():
    """``checker_for(type)`` is public registry API -- T-2 and T-5 can call a
    checker without going through ``check``, which is the only place the verdict
    is coerced. The underlying functions must already satisfy the contract, or
    the coercion in ``check`` is hiding a checker that returns 1 for True."""
    for ctype in list_types():
        constraint = _representative(ctype)
        checker = checker_for(ctype)
        for text in (_CLEAN_BY_TYPE[ctype], _VIOLATION_BY_TYPE[ctype], "", "   "):
            _assert_verdict_shape(checker(constraint, text), text)


def test_check_dispatches_to_the_registered_checker_and_not_a_default():
    for ctype, text, _offset in VIOLATION_SAMPLES:
        constraint = _representative(ctype)
        direct = checker_for(ctype)(constraint, text)
        assert check(constraint, text) == {"pass": bool(direct["pass"]),
                                           "evidence": str(direct["evidence"])}, ctype


@pytest.mark.parametrize("ctype", list_types())
def test_check_returns_the_pinned_verdict_shape(ctype):
    constraint = _representative(ctype)
    for text in ("", "   \n\t ", "Nothing much happened here.",
                 _VIOLATION_BY_TYPE[ctype], _CLEAN_BY_TYPE[ctype]):
        _assert_verdict_shape(check(constraint, text), text)


@pytest.mark.parametrize("ctype,text", CLEAN_SAMPLES, ids=[t for t, _ in CLEAN_SAMPLES])
def test_evidence_is_exactly_clean_on_pass(ctype, text):
    result = check(_representative(ctype), text)
    assert result["pass"] is True, f"{ctype}: sample was meant to obey, got {result['evidence']!r}"
    assert result["evidence"] == CLEAN_EVIDENCE == "clean"


@pytest.mark.parametrize("ctype,text,offset", VIOLATION_SAMPLES,
                         ids=[t for t, _, _ in VIOLATION_SAMPLES])
def test_evidence_locates_the_violation_on_fail(ctype, text, offset):
    result = check(_representative(ctype), text)
    assert result["pass"] is False, f"{ctype}: sample was meant to violate"
    _assert_verdict_shape(result, text)
    reported = int(_OFFSET_RE.search(result["evidence"]).group(1))
    assert reported == offset, (
        f"{ctype}: evidence points at offset {reported} ({text[reported]!r}) but the violation "
        f"is at {offset} ({text[offset]!r}); evidence={result['evidence']!r}")


# --------------------------- purity ---------------------------------------

_ALLOWED_IMPORTS = {"__future__", "re", "unicodedata", "typing", "constraints"}
_BANNED_CALLS = {"open", "eval", "exec", "compile", "__import__", "input", "print"}


def _imported_top_level_modules(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            mods.add(node.module.split(".")[0])
    return mods


@pytest.mark.parametrize("path", [CHECKERS_PATH, REGISTRY_PATH],
                         ids=["checkers.py", "registry.py"])
def test_ground_truth_layer_imports_nothing_impure(path):
    """No os, random, time, requests, urllib, socket, subprocess, pathlib. The
    allowlist is the assertion: a new import has to be argued for, not merely
    avoid a blocklist someone forgot to extend."""
    mods = _imported_top_level_modules(path)
    assert mods <= _ALLOWED_IMPORTS, (
        f"{path.name} imports {sorted(mods - _ALLOWED_IMPORTS)}; the ground-truth layer may "
        f"only import {sorted(_ALLOWED_IMPORTS)} -- no network, filesystem, clock or RNG")
    for banned in ("os", "random", "time", "requests", "urllib", "socket",
                   "subprocess", "pathlib", "datetime", "secrets"):
        assert banned not in mods, f"{path.name} imports {banned}"


@pytest.mark.parametrize("path", [CHECKERS_PATH, REGISTRY_PATH],
                         ids=["checkers.py", "registry.py"])
def test_ground_truth_layer_makes_no_impure_calls(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _BANNED_CALLS:
                found.append(f"{node.func.id}() at line {node.lineno}")
    assert not found, f"{path.name} performs impure/IO calls: {found}"


def test_repeated_calls_are_byte_identical():
    """Purity in the only way that matters downstream: the same (constraint, text)
    always yields the same verdict AND the same evidence string."""
    for ctype, text, _offset in VIOLATION_SAMPLES:
        constraint = _representative(ctype)
        first = check(constraint, text)
        for _ in range(100):
            assert check(constraint, text) == first, f"{ctype}: non-deterministic verdict"
    for row in FIXTURES:
        first = check(row["constraint"], row["text"])
        for _ in range(3):
            assert check(row["constraint"], row["text"]) == first, row["fixture_id"]


def test_check_does_not_mutate_its_inputs():
    for ctype, text, _offset in VIOLATION_SAMPLES:
        constraint = _representative(ctype)
        before = copy.deepcopy(constraint)
        check(constraint, text)
        assert constraint == before, f"{ctype}: check() mutated the constraint it was given"


def test_verdicts_are_identical_across_processes_and_hash_seeds():
    """Dict and set iteration order must never leak into evidence. Three fresh
    interpreters, three PYTHONHASHSEEDs, one expected digest."""
    script = textwrap.dedent(
        """
        import hashlib, json, sys
        sys.path.insert(0, %r)
        from constraints.checkers import check
        digest = hashlib.sha256()
        with open(%r, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                verdict = check(row["constraint"], row["text"])
                digest.update(json.dumps([row["fixture_id"], verdict],
                                         sort_keys=True).encode("utf-8"))
        print(digest.hexdigest())
        """
    ) % (str(REPO_ROOT), str(FIXTURES_PATH))

    digests = set()
    for seed in ("0", "1", "12345"):
        proc = subprocess.run([sys.executable, "-c", script], capture_output=True,
                              text=True, cwd=str(REPO_ROOT),
                              env=dict(os.environ, PYTHONHASHSEED=seed))
        assert proc.returncode == 0, f"PYTHONHASHSEED={seed} run failed: {proc.stderr}"
        digests.add(proc.stdout.strip())
    assert len(digests) == 1, f"verdicts vary with hash seed: {digests}"


# --------------------------- raising policy -------------------------------

def _words(**over):
    base = {"id": "c-lex-001", "type": "banned_words", "params": {"words": ["coffee"]},
            "family": "lexical", "split": "train"}
    base.update(over)
    return base


MALFORMED_CONSTRAINTS = [
    ("not a mapping at all", None),
    ("a bare string", "banned_words"),
    ("a list", [("type", "banned_words")]),
    ("missing 'type'", {"params": {"words": ["coffee"]}}),
    ("missing 'params'", {"type": "banned_words"}),
    ("unknown type", {"type": "banned_lettters", "params": {"letter": "e"}}),
    ("type is not a string", {"type": 7, "params": {}}),
    ("params is a list", _words(params=["coffee"])),
    ("params is None", _words(params=None)),
    ("params is a string", _words(params="words")),
    ("non-string params key", _words(params={1: ["coffee"]})),
    ("misspelled param key", _words(params={"word": ["coffee"]})),
    ("extra param key", _words(params={"words": ["coffee"], "regex": True})),
    ("empty word list", _words(params={"words": []})),
    ("words given as a bare string", _words(params={"words": "coffee"})),
    ("non-string word", _words(params={"words": [1]})),
    ("bool-as-string word", _words(params={"words": [True]})),
    ("blank word", _words(params={"words": ["   "]})),
    ("multi-word banned_word", _words(params={"words": ["in conclusion"]})),
    ("empty phrase list", {"type": "banned_phrases", "params": {"phrases": []}}),
    ("phrases with a non-string", {"type": "banned_phrases", "params": {"phrases": ["ok", 2]}}),
    ("letter longer than one char", {"type": "banned_letter", "params": {"letter": "ee"}}),
    ("empty letter", {"type": "banned_letter", "params": {"letter": ""}}),
    ("letter as an int", {"type": "banned_letter", "params": {"letter": 5}}),
    ("letter as a bool", {"type": "banned_letter", "params": {"letter": True}}),
    ("letter is a digit", {"type": "banned_letter", "params": {"letter": "5"}}),
    ("letter param missing", {"type": "banned_letter", "params": {}}),
    ("marks with a multi-char entry", {"type": "banned_punctuation", "params": {"marks": [",;"]}}),
    ("marks with an alphanumeric", {"type": "banned_punctuation", "params": {"marks": ["a"]}}),
    ("marks with a whitespace mark", {"type": "banned_punctuation", "params": {"marks": [" "]}}),
    ("marks as ints", {"type": "banned_punctuation", "params": {"marks": [44]}}),
    ("marks as bools", {"type": "banned_punctuation", "params": {"marks": [True]}}),
    ("empty marks", {"type": "banned_punctuation", "params": {"marks": []}}),
    ("no_lists given params", {"type": "no_lists", "params": {"strict": True}}),
    ("no_quotation_marks given params", {"type": "no_quotation_marks", "params": {"marks": ['"']}}),
    ("family contradicts the registry", _words(family="character")),
    ("split contradicts the registry", _words(split="heldout_type")),
    ("id in the wrong shape", _words(id="lex-1")),
    ("id with the wrong family prefix", _words(id="c-chr-001")),
    ("id as an int", _words(id=7)),
    ("id as a bool", _words(id=True)),
    ("unknown top-level key", dict(_words(), note="hand-written")),
]


@pytest.mark.parametrize("label,constraint", MALFORMED_CONSTRAINTS,
                         ids=[lbl.replace(" ", "-") for lbl, _ in MALFORMED_CONSTRAINTS])
def test_malformed_constraints_raise_rather_than_vacuously_pass(label, constraint):
    """The worst available failure mode is a checker that silently 'passes' a
    constraint it did not understand -- it would inflate every downstream
    number. Every malformed constraint must raise."""
    with pytest.raises(ConstraintError):
        check(constraint, "The espresso machine broke, again; nobody said anything.")


def test_validate_constraint_requires_the_whole_pinned_object():
    for missing in ("id", "type", "params", "family", "split"):
        candidate = _words()
        candidate.pop(missing)
        with pytest.raises(ConstraintError):
            validate_constraint(candidate)
    with pytest.raises(ConstraintError):
        validate_constraint(_words(id=None))


@pytest.mark.parametrize("bad_text", [123, 1.5, True, b"bytes", ["a"], {"text": "a"}, object()],
                         ids=["int", "float", "bool", "bytes", "list", "dict", "object"])
def test_non_string_text_raises_instead_of_being_stringified(bad_text):
    """str()-ing a dict or a list would check the repr of a caller bug and report
    a confident verdict about it."""
    with pytest.raises(TypeError):
        check(_representative("banned_words"), bad_text)


def test_none_text_is_treated_as_an_empty_response():
    for ctype in list_types():
        result = check(_representative(ctype), None)
        _assert_verdict_shape(result, "")
        assert result["pass"] is True


# --------------------------- hostile text ---------------------------------

HOSTILE_TEXTS = {
    "empty": "",
    "spaces_only": "     ",
    "whitespace_zoo": "\n\t\r\v\f ",
    "100k_prose": "The quick brown fox jumps over the lazy dog. " * 2500,
    "100k_bullets": "- item one\n" * 12000,
    "mixed_scripts": "Sam said 你好 and مرحبا then шёл.",
    "lone_high_surrogate": "abc\ud800def",
    "lone_low_surrogate": "abc\udfffdef",
    "astral_emoji": "Sure \U0001f600\U0001f1fa\U0001f1f8 done.",
    "nul_byte": "a\x00b",
    "combining_storm": "e" + "́" * 500,
    "bidi_overrides": "‮reversed‬ text",
    "terminators_only": "... !!! ???",
    "no_terminator": "Sam ran",
}


@pytest.mark.parametrize("ctype", list_types())
@pytest.mark.parametrize("label", sorted(HOSTILE_TEXTS))
def test_hostile_text_never_raises_and_always_returns_a_verdict(ctype, label):
    """A hostile TEXT is never a caller bug -- it is what a model produced -- so
    it always gets a verdict, never an exception."""
    text = HOSTILE_TEXTS[label]
    result = check(_representative(ctype), text)
    _assert_verdict_shape(result, text)


def test_evidence_survives_json_serialisation_for_hostile_text():
    """Evidence lands in a JSONL transcript; a lone surrogate in the response
    must not make that row unwritable."""
    for ctype in list_types():
        for text in HOSTILE_TEXTS.values():
            evidence = check(_representative(ctype), text)["evidence"]
            round_tripped = json.loads(json.dumps({"evidence": evidence}))["evidence"]
            assert round_tripped == evidence
            evidence.encode("utf-8")


# --------------------------- registry invariants --------------------------

def test_type_level_split_invariant():
    for ctype, spec in TYPES.items():
        assert spec.split in SPLITS, f"{ctype}: split {spec.split!r} is not in {SPLITS}"
        assert spec.split in ("train", "heldout_type")
    assert set(types_by_split("heldout_type")) == {"banned_punctuation", "sentence_initial_letter"}, (
        "the held-out set is pinned: the structural family must be entirely unseen and the "
        "character family half-seen, or the local eval stops predicting the staff held-out set")
    assert set(types_by_split("train")) == set(TYPES) - {"banned_punctuation",
                                                         "sentence_initial_letter"}


def test_train_split_spans_at_least_three_families():
    families = {TYPES[t].family for t in types_by_split("train")}
    assert len(families) >= 3, f"train split covers only {sorted(families)} (Spec R1 needs >=3)"
    assert families <= set(FAMILIES)


def test_held_out_families_are_unseen_or_half_seen_as_designed():
    heldout = {TYPES[t].family for t in types_by_split("heldout_type")}
    train = {TYPES[t].family for t in types_by_split("train")}
    assert "structural" in heldout and "structural" not in train, (
        "the structural family must be entirely absent from the train split")
    assert "character" in heldout and "character" in train, (
        "the character family must be half-seen: banned_letter trains, banned_punctuation does not")


def test_every_type_carries_the_family_id_prefix_it_claims():
    for ctype, spec in TYPES.items():
        constraint = build_constraint(ctype, copy.deepcopy(spec.example_params), 14)
        validate_constraint(constraint)
        assert constraint["family"] == spec.family
        assert constraint["split"] == spec.split
        assert constraint["id"].endswith("014"), constraint["id"]


# --------------------------- numeric type confusion -----------------------

@pytest.mark.parametrize("bad_index", [True, False, 1.0, "1", None, [1], 1 + 0j, -1, -0],
                         ids=["True", "False", "float", "str", "None", "list", "complex",
                              "negative", "negative-zero"])
def test_build_constraint_rejects_bool_as_int_and_type_confused_indices(bad_index):
    """``True`` is an ``int`` in Python and ``isinstance(True, int)`` is True, so a
    naive index guard accepts it and mints ``c-lex-001`` for every constraint the
    caller meant to number differently. ``-0`` is the control: it IS 0 and must be
    accepted, proving the guard rejects on type and sign, not on truthiness."""
    if bad_index == 0 and bad_index is not False:      # -0 is a valid index
        assert build_constraint("banned_words", {"words": ["coffee"]}, bad_index)["id"] == "c-lex-000"
        return
    with pytest.raises(ConstraintError):
        build_constraint("banned_words", {"words": ["coffee"]}, bad_index)


def test_bool_params_are_not_accepted_as_their_int_or_str_shadows():
    """Nothing in the params schema may accept a bool by way of int/str coercion."""
    for constraint in (
        {"type": "banned_letter", "params": {"letter": True}},
        {"type": "banned_words", "params": {"words": [True]}},
        {"type": "banned_phrases", "params": {"phrases": [False]}},
        {"type": "banned_punctuation", "params": {"marks": [True]}},
        {"type": "banned_words", "params": {"words": True}},
        {"type": "no_lists", "params": {"enabled": False}},
    ):
        with pytest.raises(ConstraintError):
            check(constraint, "anything at all")


def test_a_true_verdict_is_a_bool_not_a_truthy_value():
    """Downstream code ANDs spec_pass across constraints and writes it to JSONL;
    a truthy non-bool would serialise as 1 and quietly change the schema."""
    for ctype, text in CLEAN_SAMPLES:
        assert check(_representative(ctype), text)["pass"] is True
    for ctype, text, _offset in VIOLATION_SAMPLES:
        assert check(_representative(ctype), text)["pass"] is False


# ==========================================================================
# (c) A2 folding pipeline -- the character/lexical rulings, enforced as a CLASS
#
# The golden fixtures pin one representative of each defect. These pin the
# whole class, because the cheapest reward hack in the project is a model that
# discovers the one sibling codepoint nobody wrote a fixture for.
# ==========================================================================

#: One probe per invisible-character mechanism, named so a failure says which.
INVISIBLE_PROBES = {
    "U+200B ZERO WIDTH SPACE": "​",
    "U+200C ZERO WIDTH NON-JOINER": "‌",
    "U+200D ZERO WIDTH JOINER": "‍",
    "U+200E LEFT-TO-RIGHT MARK": "‎",
    "U+200F RIGHT-TO-LEFT MARK": "‏",
    "U+202D LEFT-TO-RIGHT OVERRIDE": "‭",
    "U+2060 WORD JOINER": "⁠",
    "U+2064 INVISIBLE PLUS": "⁤",
    "U+FEFF ZERO WIDTH NO-BREAK SPACE": "﻿",
    "U+00AD SOFT HYPHEN": "­",
    "U+034F COMBINING GRAPHEME JOINER (Mn, ccc=0)": "͏",
    "U+180B MONGOLIAN FREE VARIATION SELECTOR ONE (Mn, ccc=0)": "᠋",
    "U+FE0E VARIATION SELECTOR-15 (Mn, ccc=0)": "︎",
    "U+FE0F VARIATION SELECTOR-16 (Mn, ccc=0)": "️",
    "U+061C ARABIC LETTER MARK": "؜",
    "U+3164 HANGUL FILLER": "ㅤ",
    "U+E0101 VARIATION SELECTOR-18": "\U000e0101",
    # Category Cc. A C0/C1 control renders as nothing exactly the way a
    # zero-width space does; INVISIBLE_CATEGORIES used to list only Cf, so
    # these three split a banned word while their Cf twins were caught.
    "U+0001 START OF HEADING (Cc)": "\x01",
    "U+0007 BELL (Cc)": "\x07",
    "U+007F DELETE (Cc)": "\x7f",
    "U+0090 DEVICE CONTROL STRING (C1, Cc)": "\x90",
}


@pytest.mark.parametrize("probe", list(INVISIBLE_PROBES.values()),
                         ids=list(INVISIBLE_PROBES))
def test_invisible_characters_cannot_hide_a_banned_word(probe):
    """A codepoint that renders as nothing cannot make a visible word invisible
    to the checker. This is the cheapest evasion in the project: one zero-width
    character costs a model nothing and, unfixed, the training data teaches it."""
    text = f"I drank a cof{probe}fee this morning."
    result = check(_representative("banned_words"), text)
    assert result["pass"] is False, (
        f"{probe!r} hid the banned word; the rendered line still reads 'coffee'")
    _assert_verdict_shape(result, text)


@pytest.mark.parametrize("probe", list(INVISIBLE_PROBES.values()),
                         ids=list(INVISIBLE_PROBES))
def test_invisible_characters_cannot_break_a_banned_phrase(probe):
    """The phrase separator is ``\\s+`` and none of these probes is ``\\s``."""
    text = f"The pilot shipped. In{probe} conclusion, we recommend it."
    result = check(_representative("banned_phrases"), text)
    assert result["pass"] is False, f"{probe!r} broke the phrase join"
    _assert_verdict_shape(result, text)


@pytest.mark.parametrize("probe", list(INVISIBLE_PROBES.values()),
                         ids=list(INVISIBLE_PROBES))
def test_invisible_characters_cannot_hide_a_banned_letter(probe):
    text = f"Th{probe}e draft is short."
    result = check(_representative("banned_letter"), text)
    assert result["pass"] is False, f"{probe!r} hid the banned letter"
    _assert_verdict_shape(result, text)


def test_invisible_stripping_is_an_index_map_not_a_lossy_replace():
    """Evidence must quote the user's REAL bytes at REAL offsets after the fold
    drops characters -- otherwise an auditor reading a transcript is shown a
    response the model never produced."""
    text = "I drank a cof​fee this morning."
    result = check(_representative("banned_words"), text)
    assert result["pass"] is False
    offset = int(_OFFSET_RE.search(result["evidence"]).group(1))
    assert text[offset:offset + 7] == "cof​fee", (
        f"offset {offset} does not land on the real violating span")
    marked = _MARKED_RE.search(result["evidence"]).group(1)
    assert marked == "cof​fee", (
        f"evidence quotes {marked!r}, not the user's real bytes 'cof\\u200bfee'")


def test_a_response_made_only_of_invisible_characters_still_gets_a_verdict():
    for ctype in list_types():
        for text in ("​​​", "﻿", "­͏"):
            _assert_verdict_shape(check(_representative(ctype), text), text)


# --------------------------- manufactured letters -------------------------

MANUFACTURED_LETTER_SAMPLES = (
    ("™", "TRADE MARK SIGN", "t"),
    ("№", "NUMERO SIGN", "o"),
    ("℅", "CARE OF", "c"),
    ("℠", "SERVICE MARK", "s"),
    ("℡", "TELEPHONE SIGN", "t"),
    ("℻", "FACSIMILE SIGN", "f"),
    ("℃", "DEGREE CELSIUS", "c"),
    ("℉", "DEGREE FAHRENHEIT", "f"),
    ("₨", "RUPEE SIGN", "r"),
    ("Ⅰ", "ROMAN NUMERAL ONE", "i"),
    ("㎠", "SQUARE CM SQUARED", "m"),
    ("㎒", "SQUARE MHZ", "h"),
)


@pytest.mark.parametrize("char,name,letter", MANUFACTURED_LETTER_SAMPLES,
                         ids=[n for _, n, _ in MANUFACTURED_LETTER_SAMPLES])
def test_a_symbol_is_not_an_occurrence_of_the_letters_it_decomposes_into(char, name, letter):
    """FALSE FAILS are the expensive direction: they silently reject a compliant
    training example and depress the base-model rate with nothing to show for it."""
    constraint = {"id": "c-chr-001", "type": "banned_letter", "params": {"letter": letter},
                  "family": "character", "split": "train"}
    # A deliberately letter-free carrier: the ONLY thing in the response that
    # could possibly be read as a letter is the symbol under test.
    carrier = "5 {} 5"
    assert not any(c.isalpha() for c in carrier.replace("{}", "")), "carrier must be letter-free"
    assert check(constraint, carrier.replace("{}", ""))["pass"] is True, "carrier control"
    text = carrier.format(char)
    result = check(constraint, text)
    assert result["pass"] is True, (
        f"{name} ({char!r}) manufactured a {letter!r} that is not on the page; "
        f"evidence={result['evidence']!r}")


def test_no_non_letter_in_unicode_folds_to_a_latin_letter():
    """The whole class, not the two fixtures. Every character that is NOT a
    unicode letter but whose NFKD expands into letters must fold to something
    with no ASCII letter in it -- unless it is an explicit, named entry in one
    of the confusable tables or in ENCLOSED_LETTERFORM_RANGES, each of which is
    a deliberate, name-derived ruling that the source IS a letter."""
    deliberate = (set(CHECKERS_MODULE.CONFUSABLE_TO_LATIN)
                  | set(CHECKERS_MODULE.LETTERFORM_TO_LATIN)
                  | set(CHECKERS_MODULE._ENCLOSED_LETTERFORMS))
    manufactured = []
    for cp in range(0xA0, 0x1FB00):
        ch = chr(cp)
        if ch in deliberate or CHECKERS_MODULE._is_letter(ch):
            continue
        folded = CHECKERS_MODULE._fold_char(ch, True)
        if any(c.isascii() and c.isalpha() for c in folded):
            manufactured.append(f"U+{cp:04X} -> {folded!r}")
    assert not manufactured, (
        f"{len(manufactured)} non-letter characters still manufacture Latin letters, "
        f"e.g. {manufactured[:10]}")


def test_the_manufactured_letter_guard_did_not_switch_off_real_folding():
    """Control for the test above: characters that ARE the letter in another
    form must still fold. A guard that simply stopped folding would pass the
    class sweep and reopen every evasion it was meant to leave closed."""
    for source, expected in (("ｃ", "c"),          # FULLWIDTH LATIN SMALL LETTER C
                             ("\U0001d400", "a"),      # MATHEMATICAL BOLD CAPITAL A
                             ("é", "e"),          # e with acute
                             ("ē", "e"),          # e with macron
                             ("ﬀ", "ff"),         # ligature ff
                             ("K", "k")):         # KELVIN SIGN
        assert CHECKERS_MODULE._fold_char(source, True) == expected, source


# ------------- BLOCKER 1: symbols vs. enclosed letterforms ----------------
#
# The manufactured-letter guard ("a symbol is not an occurrence of the letters
# it decomposes into") was implemented as "the source's general category is not
# L". That over-applies: an ENCLOSED letterform -- Ⓐ, ⒜, 🄰 -- is category So and
# stopped counting as its letter, so 133 characters HEAD caught correctly began
# evading banned_letter entirely. The rule has two directions and both are
# proved here: a SYMBOL never manufactures a letter, a LETTER in a decorative
# form always still folds.

#: The PROPERTY the enclosed-letterform ruling is stated in: "a decorative
#: enclosure of exactly one named Latin letter". A character has it when its own
#: unicode name says it is one Latin letter -- in the flag block that sentence is
#: spelled ``REGIONAL INDICATOR SYMBOL LETTER A`` instead of ``... LATIN CAPITAL
#: LETTER A``, which is a difference in wording, not in what is on the page.
_ENCLOSED_BY_NAME = re.compile(r"^(?:.*\s)?LATIN (CAPITAL|SMALL) LETTER ([A-Z])$")
_REGIONAL_INDICATOR_BY_NAME = re.compile(r"^REGIONAL INDICATOR SYMBOL LETTER ([A-Z])$")


def _base_letter_of(char):
    """The letter an enclosed form encloses, read off its own unicode name."""
    import unicodedata as _ud

    name = _ud.name(char)
    match = _ENCLOSED_BY_NAME.match(name)
    if match:
        return match.group(2).lower()
    match = _REGIONAL_INDICATOR_BY_NAME.match(name)
    assert match, f"{char!r} ({name}) is not a named Latin letterform"
    return match.group(1).lower()


def test_the_enclosed_letterform_table_is_bound_to_the_property_not_to_a_decomposition():
    """The ranges are pasted data; the PROPERTY is the ruling. A codepoint
    belongs because it is a SYMBOL (category So) whose own unicode name says it
    is exactly one Latin letter -- nothing else.

    What this test deliberately no longer asks is whether the character has an
    NFKD decomposition. That was the derivation the table used to be checked
    against, and it re-ran the very rule the checker implemented, so the test
    could not fail while the checker was wrong in the same way -- and it was:
    the ~80 NEGATIVE CIRCLED, NEGATIVE SQUARED and REGIONAL INDICATOR letters
    have no decomposition, so "🅐pple" walked past a ban on 'a' while "Ⓐpple"
    was caught, and the derivation test agreed with both. Having a decomposition
    is a property of the unicode tables, not of what a reader sees."""
    import unicodedata as _ud

    derived = set()
    for cp in range(0x110000):
        ch = chr(cp)
        if _ud.category(ch) != "So":
            continue
        try:
            name = _ud.name(ch)
        except ValueError:
            continue
        if _ENCLOSED_BY_NAME.match(name) or _REGIONAL_INDICATOR_BY_NAME.match(name):
            derived.add(ch)
    table = set(CHECKERS_MODULE._ENCLOSED_LETTERFORMS)
    assert table == derived, (
        f"ENCLOSED_LETTERFORM_RANGES drifted from the property it claims: "
        f"{sorted(hex(ord(c)) for c in table - derived)} listed but not derived, "
        f"{sorted(hex(ord(c)) for c in derived - table)} derived but not listed")
    assert len(derived) == 213, f"the property yields {len(derived)}, not the pinned 213"
    undecomposed = {c for c in derived if _ud.normalize("NFKD", c) == c}
    assert len(undecomposed) == 80, (
        "the 80 enclosures with no NFKD decomposition are the whole point of this "
        f"test; found {len(undecomposed)}")


@pytest.mark.parametrize("char,name", [
    ("\U0001F150", "NEGATIVE CIRCLED LATIN CAPITAL LETTER A"),
    ("\U0001F170", "NEGATIVE SQUARED LATIN CAPITAL LETTER A"),
    ("\U0001F1E6", "REGIONAL INDICATOR SYMBOL LETTER A"),
])
def test_an_enclosure_without_a_decomposition_is_still_the_letter_it_encloses(char, name):
    """The evasion, end to end: a reader sees "apple" in all four spellings, so a
    ban on 'a' must catch all four. "Ⓐpple" always failed; these three did not."""
    constraint = build_constraint("banned_letter", {"letter": "a"}, 1)
    assert check(constraint, "Ⓐpple")["pass"] is False, "control: the decomposing form"
    assert check(constraint, char + "pple")["pass"] is False, f"{name} evaded a ban on 'a'"


def test_a_symbol_never_manufactures_a_letter_but_an_enclosed_letter_still_folds():
    """BOTH directions of the one ruling, in one test, because fixing either
    direction alone is what broke the other.

    DIRECTION 1 -- a symbol whose name is a SIGN (™, №, ℡) folds to nothing with
    a letter in it: a false FAIL silently rejects a compliant training example.
    DIRECTION 2 -- a letter drawn inside a circle, a box or brackets is still
    that letter: a false PASS hands the model a one-codepoint evasion."""
    manufactured = []
    for char, name, _letter in MANUFACTURED_LETTER_SAMPLES:
        folded = CHECKERS_MODULE._fold_char(char, True)
        if any(c.isascii() and c.isalpha() for c in folded):
            manufactured.append(f"{name} ({char!r}) -> {folded!r}")
    assert not manufactured, (
        f"a SYMBOL manufactured letters that are not on the page: {manufactured}")

    lost = []
    for char in sorted(CHECKERS_MODULE._ENCLOSED_LETTERFORMS):
        expected = _base_letter_of(char)
        folded = CHECKERS_MODULE._fold_char(char, True)
        if expected not in folded:
            lost.append(f"U+{ord(char):04X} -> {folded!r}, expected to contain {expected!r}")
    assert not lost, (
        f"{len(lost)} enclosed letterform(s) stopped folding to the letter they "
        f"enclose, e.g. {lost[:10]}")


def test_every_enclosed_letterform_violates_a_ban_on_the_letter_it_encloses():
    """End to end through ``check``, not just the fold: the 213 characters the
    over-applied guard let through are the whole regression, so the whole class
    is asserted rather than the one fixture-shaped sample."""
    evaded = []
    for char in sorted(CHECKERS_MODULE._ENCLOSED_LETTERFORMS):
        letter = _base_letter_of(char)
        constraint = build_constraint("banned_letter", {"letter": letter}, 1)
        carrier = f"5 {char} 5"
        if check(constraint, carrier)["pass"] is not False:
            evaded.append(f"U+{ord(char):04X} evaded a ban on {letter!r}")
    assert not evaded, f"{len(evaded)} enclosed letterform(s) evaded, e.g. {evaded[:10]}"


def test_an_enclosed_letterform_is_read_as_its_letter_inside_a_word():
    """The evasion is not academic: the enclosed form is legible in running
    prose, so it also has to count inside a banned WORD."""
    result = check(_representative("banned_words"), "I drank a Ⓒoffee this morning.")
    assert result["pass"] is False, "a circled C is a C to every reader of the page"
    _assert_verdict_shape(result, "I drank a Ⓒoffee this morning.")


# ------------- the seven letterforms the "WITH" derivation missed ----------

NON_PREPOSITION_LETTERFORMS = ("Ʉ", "ʉ", "Ꝛ", "ꝛ", "ꬱ", "ꭇ", "ꭏ")


@pytest.mark.parametrize("char", NON_PREPOSITION_LETTERFORMS,
                         ids=[f"U+{ord(c):04X}" for c in NON_PREPOSITION_LETTERFORMS])
def test_a_letterform_named_without_the_word_with_still_folds(char):
    """``LATIN CAPITAL LETTER U BAR`` is the same ruling as ``LATIN CAPITAL
    LETTER T WITH STROKE``; only the preposition differs. The old derivation
    regex required the literal word "WITH", so these seven walked through."""
    assert char in CHECKERS_MODULE.LETTERFORM_TO_LATIN, f"{char!r} missing from the table"
    letter = CHECKERS_MODULE.LETTERFORM_TO_LATIN[char].lower()
    constraint = build_constraint("banned_letter", {"letter": letter}, 1)
    text = f"5 {char} 5"
    assert check(constraint, text)["pass"] is False, (
        f"U+{ord(char):04X} evaded a ban on {letter!r}")


# ------------- category Cc, and the whitespace that must survive it --------

def test_a_control_character_cannot_split_a_banned_word():
    """INVISIBLE_CATEGORIES listed only Cf. A C0/C1 control renders as nothing
    just as a zero-width space does, and ASCII controls additionally slipped
    past the ASCII fast path in ``_fold``, which is not length preserving for
    them."""
    for probe in ("\x01", "\x07", "\x1c", "\x1f", "\x7f", "\x90"):
        text = f"I drank a cof{probe}fee this morning."
        assert check(_representative("banned_words"), text)["pass"] is False, (
            f"U+{ord(probe):04X} (category Cc) hid the banned word")
    assert CHECKERS_MODULE._fold("cof\x01fee")[0] == "coffee"


WHITESPACE_THAT_SEPARATES = (" ", "\t", "\n", "\r", "\v", "\f", "\x85", "\xa0",
                             "\u2000", "\u3000", "\u2028", "\u2029")

#: The C0 separators. ``str.isspace()`` calls them whitespace; they render as
#: NOTHING, so the invisible ruling is the one that applies -- and the test
#: above proves they no longer split a banned word.
C0_SEPARATORS_ARE_INVISIBLE_NOT_WHITESPACE = ("\x1c", "\x1d", "\x1e", "\x1f")


@pytest.mark.parametrize("ws", WHITESPACE_THAT_SEPARATES,
                         ids=[f"U+{ord(c):04X}" for c in WHITESPACE_THAT_SEPARATES])
def test_whitespace_is_never_invisible_even_when_its_category_is_cc(ws):
    """The non-regression that pays for admitting Cc. '\n', '\t', '\v', '\f',
    '\r' and U+0085 are category Cc, but they SEPARATE -- dropping them would
    glue the words on either side into one token and hide BOTH from every
    lexical checker, and would silently delete the A5 line-break boundaries."""
    assert CHECKERS_MODULE._is_invisible(ws) is False, f"{ws!r} renders as a gap, not as nothing"
    assert ws.isspace() or ws == " ", f"{ws!r} must really be whitespace"
    glued = build_constraint("banned_words", {"words": ["teacoffee"]}, 1)
    assert check(glued, f"We serve tea{ws}coffee here.")["pass"] is True, (
        f"{ws!r} was dropped, welding two words into a banned one")


@pytest.mark.parametrize("sep", C0_SEPARATORS_ARE_INVISIBLE_NOT_WHITESPACE,
                         ids=[f"U+{ord(c):04X}" for c in C0_SEPARATORS_ARE_INVISIBLE_NOT_WHITESPACE])
def test_the_c0_separators_are_ruled_invisible_although_python_calls_them_space(sep):
    """The one place ``_renders_as_whitespace`` deliberately disagrees with
    ``str.isspace()``. U+001C-U+001F are record/unit separators from the ASCII
    era: they render as nothing in prose, this file's A5 ruling already refuses
    to treat them as line breaks, and treating them as whitespace let them split
    a banned word the way a zero-width space used to."""
    assert sep.isspace(), "premise: Python calls these whitespace"
    assert CHECKERS_MODULE._is_invisible(sep) is True, (
        f"U+{ord(sep):04X} renders as nothing, so it must count as nothing")


# ------------- BLOCKER 2: multi-dot abbreviations ---------------------------

S_LETTER = build_constraint("sentence_initial_letter", {"letter": "s"}, 1)

#: Abbreviations that CAN end an English sentence and here do. Before the repair
#: each of these hid EVERY following sentence from the checker.
#:
#: Two rows LEFT this tuple: "Several teams use it, e.g. Tom's group in Denver."
#: and "Sourcing is the bottleneck, i.e. Tom cannot buy the parts." both asserted
#: ``pass is False``, and both assertions were factually wrong about English.
#: Each text is ONE sentence, and it begins with S. "e.g." and "i.e." are
#: clause-OPENING: they introduce what follows ("for example ...") and never end
#: a sentence, so there is no second sentence beginning with T to fail on. They
#: now live in NOT_A_BOUNDARY below, which asserts the opposite verdict. The
#: reason the TEST was wrong rather than the checker is a fact about the
#: language, not a convenience: no reading of "..., e.g. Tom's group in Denver."
#: has two sentences in it.
MULTI_DOT_CARRIERS = (
    "Sales rose sharply at 3 p.m. Tom left the room.",
    "Standup starts at 9 a.m. Tom never arrives on time.",
    "Shipments cleared customs in the U.S. Tom signed for them.",
    "Sam wrote the memo at 4 p.m. Everyone read it by five.",
)

#: The same defect one dot down, and the wider half of it: these were
#: UNCONDITIONALLY never-final, so the sentence after them was never graded.
#: Ordinary business prose, no adversarial character anywhere -- and they
#: COMPOSE, so a whole response can be taken out of the constraint's reach.
SINGLE_DOT_CARRIERS = (
    "Sales rose in Aug. Tom took the account.",
    "Shipments fell in Sept. Rachel escalated it.",
    "Supplies ran short by Fri. Nobody noticed.",
    "Shipments left Acme Inc. Tom signed for them.",
    "Sam walked down Third Ave. Tom waited on the corner.",
    ("Sales rose in Aug. Tom took the account. Shipments fell in Sept. "
     "Rachel escalated it. Supplies ran short by Fri. Nobody noticed."),
)


@pytest.mark.parametrize("text", MULTI_DOT_CARRIERS + SINGLE_DOT_CARRIERS)
def test_an_abbreviation_that_can_be_final_no_longer_hides_the_sentences_after_it(text):
    """30 characters of ordinary prose used to switch the whole constraint off,
    with no adversarial intent required -- and sentence_initial_letter is the
    HELD-OUT type whose score is the project's generalisation claim, so this
    inflated precisely the number the project exists to prove."""
    result = check(S_LETTER, text)
    assert result["pass"] is False, (
        f"the sentence after the abbreviation was never graded: {text!r}")
    _assert_verdict_shape(result, text)


def test_the_composing_case_grades_every_sentence_not_merely_one_more():
    """The verdict alone is too weak here: a single recovered boundary would
    already turn the paragraph red. Six sentences to a human, so six graded
    spans -- three of them beginning T, R and N."""
    text = ("Sales rose in Aug. Tom took the account. Shipments fell in Sept. "
            "Rachel escalated it. Supplies ran short by Fri. Nobody noticed.")
    assert len(CHECKERS_MODULE._sentence_spans(text)) == 6


NOT_A_BOUNDARY = (
    ("Sam left the U.S.A. so nobody noticed.", "lowercase follower continues the sentence"),
    ("Sam runs standup at 9 a.m. so the room is free.", "lowercase follower"),
    ("Sam met J. R. R. Tolkien so the story goes.", "name initials, decided earlier"),
    ("Sam cut it to 3.5 percent so nobody noticed.", "decimal point"),
    ("Sam checked example.com so nobody worried.", "domain"),
    # --- capitalised followers: where the false FAILS live -------------------
    # A lowercase-only table is structurally blind to the direction that
    # over-splits, because a lowercase follower suppresses the boundary by
    # itself no matter which tier the abbreviation is in.
    ("Several teams use it, e.g. Tom's group in Denver.",
     "e.g. is clause-opening, so a capitalised follower is not a new sentence"),
    ("Sourcing is the bottleneck, i.e. Tom cannot buy the parts.",
     "i.e. is clause-opening"),
    ("Several sources disagree, cf. Nelson on the same point.", "cf. points forward"),
    ("Sam met Dr. Smith so the plan shifted.", "a title introduces the name it modifies"),
    ("Sam moved to St. Louis and stayed there.", "capitalised ordinary-word form is the abbreviation"),
    ("Sam signed with Acme Co. Ltd so the deal closed.", "Co. Ltd, capitalised follower"),
    ("Sam works at No. 7 Station so he walks.", "No. before a digit"),
    ("Sam waited until Feb. 3 so the invoice cleared.", "month before a date"),
)


@pytest.mark.parametrize("text,why", NOT_A_BOUNDARY, ids=[w for _, w in NOT_A_BOUNDARY])
def test_the_abbreviation_repair_does_not_split_a_continuing_sentence(text, why):
    """A spurious boundary costs one extra graded segment; the repair may not
    buy the hidden-text fix with a wave of them. Every row here is ONE sentence
    beginning with S, so a False verdict is a violation manufactured out of
    ordinary prose -- the direction that biases the held-out number toward
    "harder than it is"."""
    assert check(S_LETTER, text)["pass"] is True, why


def test_the_multi_dot_ruling_constant_is_wired_and_not_merely_documented():
    hack = "Sales rose sharply at 3 p.m. Tom left the room."
    assert CHECKERS_MODULE.MULTI_DOT_ABBREVIATION_CAN_END_A_SENTENCE is True
    assert check(S_LETTER, hack)["pass"] is False
    saved = CHECKERS_MODULE.MULTI_DOT_ABBREVIATION_CAN_END_A_SENTENCE
    try:
        CHECKERS_MODULE.MULTI_DOT_ABBREVIATION_CAN_END_A_SENTENCE = False
        assert check(S_LETTER, hack)["pass"] is True, (
            "flipping the constant changed nothing -- the ruling is not wired")
    finally:
        CHECKERS_MODULE.MULTI_DOT_ABBREVIATION_CAN_END_A_SENTENCE = saved
    assert CHECKERS_MODULE.MULTI_DOT_ABBREVIATION_CAN_END_A_SENTENCE is True


# ------------- fx-str-07: invisible characters after a terminator -----------

@pytest.mark.parametrize("probe", list(INVISIBLE_PROBES.values()),
                         ids=list(INVISIBLE_PROBES))
def test_an_invisible_character_after_a_terminator_cannot_hide_a_sentence(probe):
    """fx-str-07, the whole class rather than the one fixture codepoint. A
    character that renders as nothing sat between the '.' and the space, so
    PERIOD_REQUIRES_TRAILING_SPACE saw a non-space, dropped the boundary, and
    the entire response collapsed into ONE graded sentence."""
    text = f"Sam went to the market.{probe} Tom bought bread.{probe} Rachel paid."
    result = check(S_LETTER, text)
    assert result["pass"] is False, (
        f"{probe!r} after the terminator hid every following sentence")
    _assert_verdict_shape(result, text)


def test_invisible_characters_do_not_change_the_sentence_count():
    """The stronger statement: the segmentation of the rendered text is what
    counts, so sprinkling invisibles changes nothing at all."""
    clean = "Sam went to the market. Tom bought bread. Rachel paid. Nobody complained."
    hostile = clean.replace(". ", ".\u200b ").replace(".\u200b ", ".\u2060 ", 1)
    assert CHECKERS_MODULE._sentence_spans(clean) != []
    assert len(CHECKERS_MODULE._sentence_spans(hostile)) == len(
        CHECKERS_MODULE._sentence_spans(clean)), "invisibles changed the sentence count"


def test_a_sentence_never_begins_on_a_character_that_renders_as_nothing():
    """Evidence has to quote the user's real bytes at a real offset; a segment
    that opens on a zero-width space would report the violation one character
    early and point an auditor at nothing."""
    text = "Sam shipped it.\u200b Tom did not."
    spans = CHECKERS_MODULE._sentence_spans(text)
    for start, _end in spans:
        assert not CHECKERS_MODULE._is_invisible(text[start]), (
            f"sentence span opens on U+{ord(text[start]):04X}, which renders as nothing")
    result = check(S_LETTER, text)
    assert result["pass"] is False
    offset = int(_OFFSET_RE.search(result["evidence"]).group(1))
    assert text[offset] == "T", f"evidence points at {text[offset]!r}, not the real initial"


# ------------- A4: the doubled-single-quote reading stays unruled ----------

def test_the_adjacent_single_quote_reading_is_off_and_on_the_signoff_ledger():
    """The standing A4 ruling is DOUBLES ONLY. Treating ``\'\'`` as one double
    quotation mark is a reading, not a ruling: it was switched on unilaterally
    and the defence recorded for it did not survive checking, so it is off and
    fx-fmt-18 goes back on the ledger where an unruled reading belongs.

    The knob is asserted WIRED in both directions so that a future ruling is a
    one-line change and cannot be mistaken for dead documentation."""
    assert CHECKERS_MODULE.ADJACENT_SINGLE_QUOTES_ARE_A_DOUBLE_QUOTE is False
    quotes = _representative("no_quotation_marks")
    typewriter = "She called it ''a rounding error'' in her note to staff."
    assert check(quotes, typewriter)["pass"] is True, (
        "doubles only: no single-quote form fires under the standing ruling")
    assert check(quotes, 'She called it "a rounding error" in her note.')["pass"] is False
    assert check(quotes, "The dogs' bowl is empty and Sam won't fill it.")["pass"] is True

    ledger = [r for r in FIXTURES if r["fixture_id"] == "fx-fmt-18"]
    assert ledger and ledger[0]["ambiguous"] is True, (
        "fx-fmt-18 is the row this reading turns on; it must stay on the sign-off ledger")

    saved = CHECKERS_MODULE.ADJACENT_SINGLE_QUOTES_ARE_A_DOUBLE_QUOTE
    try:
        CHECKERS_MODULE.ADJACENT_SINGLE_QUOTES_ARE_A_DOUBLE_QUOTE = True
        assert check(quotes, typewriter)["pass"] is False, (
            "flipping the constant changed nothing -- it is dead documentation")
        assert check(quotes, "The dogs' bowl is empty and Sam won't fill it.")["pass"] is True, (
            "even switched on, the reading may not touch a lone apostrophe")
    finally:
        CHECKERS_MODULE.ADJACENT_SINGLE_QUOTES_ARE_A_DOUBLE_QUOTE = saved
    assert CHECKERS_MODULE.ADJACENT_SINGLE_QUOTES_ARE_A_DOUBLE_QUOTE is False


# --------------------------- letterforms and homoglyphs -------------------

def test_letterform_table_entries_agree_with_their_unicode_names():
    """The table is data, but the DERIVATION is the ruling: an entry may only
    exist because the character's own unicode name says it IS that letter in
    another form. This blocks a future entry added on a squint.

    The modifier is ANY modifier, not the literal word "WITH". Requiring the
    preposition is what let U+0244/U+0289 "LATIN ... LETTER U BAR" evade a ban
    on 'u' while Ħ, Ŧ, Đ and Ł -- the same stroke, named with it -- were caught.
    """
    import unicodedata as _ud

    with_form = re.compile(r"^LATIN (SMALL|CAPITAL) LETTER ([A-Z]) .+$")
    small_cap = re.compile(r"^LATIN LETTER SMALL CAPITAL ([A-Z])$")
    for source, target in CHECKERS_MODULE.LETTERFORM_TO_LATIN.items():
        assert len(source) == 1, f"{source!r}: table keys are single characters"
        assert len(target) == 1 and target.isascii() and target.isalpha(), (
            f"{source!r} -> {target!r}: targets are single ASCII letters")
        name = _ud.name(source)
        match = with_form.match(name)
        if match:
            expected = match.group(2) if match.group(1) == "CAPITAL" else match.group(2).lower()
        else:
            match = small_cap.match(name)
            assert match, f"{source!r} ({name}) is not a named letterform of any letter"
            expected = match.group(1).lower()
        assert target == expected, f"{source!r} ({name}) maps to {target!r}, not {expected!r}"


def test_every_letterform_folds_to_its_base_letter():
    for source, target in CHECKERS_MODULE.LETTERFORM_TO_LATIN.items():
        assert CHECKERS_MODULE._fold_char(source, True) == target.lower(), source


def test_the_two_confusable_tables_never_disagree():
    both = set(CHECKERS_MODULE.CONFUSABLE_TO_LATIN) & set(CHECKERS_MODULE.LETTERFORM_TO_LATIN)
    conflicts = [c for c in both
                 if CHECKERS_MODULE.CONFUSABLE_TO_LATIN[c] != CHECKERS_MODULE.LETTERFORM_TO_LATIN[c]]
    assert not conflicts, f"a character folds two different ways depending on table order: {conflicts}"


#: Every e-rendering codepoint named in fx-character-15/16 as evading the A2
#: homoglyph ruling. The fixtures pin two; the ruling covers all of them.
E_LOOKALIKES = ("ɇ", "ⱸ", "ҽ", "ᴇ", "ꬲ", "ɛ",
                "є", "ⱻ", "ǝ", "ә", "℮", "ə",
                "ᶒ", "ꬴ")


@pytest.mark.parametrize("char", E_LOOKALIKES, ids=[f"U+{ord(c):04X}" for c in E_LOOKALIKES])
def test_every_named_e_lookalike_violates_a_ban_on_e(char):
    constraint = {"id": "c-chr-001", "type": "banned_letter", "params": {"letter": "e"},
                  "family": "character", "split": "train"}
    text = f"Anna will h{char}lp with that portion of our layout."
    result = check(constraint, text)
    assert result["pass"] is False, (
        f"U+{ord(char):04X} renders as an e and evades the A2 ruling the code claims to enforce")


# --------------------------- the two former dead knobs --------------------

def test_leetspeak_knob_is_wired_and_not_merely_documented():
    """LEETSPEAK_DIGITS_ARE_LETTERS used to appear only in the docstring table
    and its own definition: flipping it changed nothing. A documented knob that
    controls nothing is a trap for whoever next tries to flip it."""
    constraint = _representative("banned_words")
    assert CHECKERS_MODULE.LEETSPEAK_DIGITS_ARE_LETTERS is False
    assert check(constraint, "I need c0ffee now.")["pass"] is True, (
        "with the knob off, a leetspeak digit is not a letter (so '2020' is safe)")

    saved = CHECKERS_MODULE.LEETSPEAK_DIGITS_ARE_LETTERS
    try:
        CHECKERS_MODULE.LEETSPEAK_DIGITS_ARE_LETTERS = True
        flipped = check(constraint, "I need c0ffee now.")
        assert flipped["pass"] is False, "flipping the knob changed nothing -- it is still dead"
        _assert_verdict_shape(flipped, "I need c0ffee now.")
        ban_o = {"id": "c-chr-001", "type": "banned_letter", "params": {"letter": "o"},
                 "family": "character", "split": "train"}
        assert check(ban_o, "In 2020 that plan fell apart.")["pass"] is False, (
            "the documented COST of flipping it -- '2020' violating a ban on 'o' -- must be real")
    finally:
        CHECKERS_MODULE.LEETSPEAK_DIGITS_ARE_LETTERS = saved
    assert CHECKERS_MODULE.LEETSPEAK_DIGITS_ARE_LETTERS is False


def test_banned_punctuation_bans_only_the_listed_marks_and_their_variants():
    """The invariant that replaced BANNED_PUNCTUATION_LISTED_MARKS_ONLY, which
    no code read. Enforced in both directions: unlisted marks never fire, and
    every unicode variant of a LISTED mark does."""
    comma_only = {"id": "c-chr-001", "type": "banned_punctuation", "params": {"marks": [","]},
                  "family": "character", "split": "heldout_type"}
    allowed = "".join(sorted(set("!\"#$%&'()*+-./:;<=>?@[\\]^_`{|}~") -
                             set(CHECKERS_MODULE._variants_of(","))))
    for mark in allowed + "—–…·":      # em dash, en dash, ellipsis, middot
        text = f"First we ship {mark} nothing broke."
        result = check(comma_only, text)
        assert result["pass"] is True, (
            f"{mark!r} is not in params['marks'] but fired; the checker generalised to "
            f"'punctuation' as a family. evidence={result['evidence']!r}")
    for variant in CHECKERS_MODULE._variants_of(","):
        text = f"First{variant} we ship."
        assert check(comma_only, text)["pass"] is False, (
            f"{variant!r} is a unicode variant of a listed mark and must count")


def test_no_named_ruling_constant_is_dead():
    """Every module-level ruling constant must be READ by the module, not merely
    defined and documented. This is the check that caught the two dead knobs."""
    source = CHECKERS_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assigned, loaded = [], set()
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    assigned.append(target.id)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            loaded.add(node.id)
    exported = set(CHECKERS_MODULE.__all__)
    dead = [name for name in assigned if name not in loaded and name not in exported]
    assert not dead, (
        f"module-level ruling constant(s) defined but never read: {dead} -- flipping one "
        f"changes nothing, so it documents behaviour it does not control. Wire it up or "
        f"delete it and restate the ruling as an enforced invariant.")
