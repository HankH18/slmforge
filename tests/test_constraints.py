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
