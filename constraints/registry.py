#!/usr/bin/env python3
"""Constraint registry: the PINNED type table, the constraint validator, and
per-type parameter schemas.

Binding contract (docs/PIVOT_DESIGN.md, "Interfaces"). A constraint is a plain
JSON object::

    {"id": "c-lex-014", "type": "banned_words",
     "params": {"words": ["coffee", "espresso"]},
     "family": "lexical", "split": "train"}

The seven types below are pinned by the ticket: do not add, remove or rename
them here. ``split`` is a property of the TYPE, never of an instance --
``sentence_initial_letter`` and ``banned_punctuation`` are held out so the local
eval predicts performance on constraint types the model never trained on
(Spec R3).

Every type maps to exactly ONE checker (asserted at import). Parameter
validation is strict and LOUD: wrong params for a type raise
:class:`ConstraintError`, they never silently pass. A checker that quietly
accepts a malformed constraint would report a vacuous "pass" for every
response, which is the single worst failure mode available to this layer.

This module imports nothing from :mod:`constraints.checkers` at module scope;
``checker_for`` resolves the function lazily so the two modules can reference
each other without an import cycle.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Callable, Dict, Iterable, List, NamedTuple, Tuple

__all__ = [
    "ConstraintError",
    "FAMILIES",
    "SPLITS",
    "TYPES",
    "ID_PREFIX_BY_FAMILY",
    "ID_RE",
    "TypeSpec",
    "list_types",
    "types_by_family",
    "types_by_split",
    "family_of",
    "split_of",
    "checker_for",
    "validate_params",
    "validate_constraint",
    "normalize_constraint",
    "build_constraint",
]


class ConstraintError(ValueError):
    """A constraint object or its params violate the pinned schema.

    Raised -- never swallowed. See the module docstring for why.
    """


#: The four constraint families (Spec R1).
FAMILIES: Tuple[str, ...] = ("lexical", "character", "structural", "format_negative")

#: Type-level train/held-out split (Design "Interfaces").
SPLITS: Tuple[str, ...] = ("train", "heldout_type")

#: family -> the ``id`` prefix every instance of that family must carry.
ID_PREFIX_BY_FAMILY: Dict[str, str] = {
    "lexical": "c-lex-",
    "character": "c-chr-",
    "structural": "c-str-",
    "format_negative": "c-fmt-",
}

#: Constraint ids look like ``c-lex-014``. Digits are not length-pinned so the
#: scenario generator (T-2) can number as it likes.
ID_RE = re.compile(r"^c-(?:lex|chr|str|fmt)-\d+$")


# --------------------------------------------------------------------------
# Per-type parameter schemas.
#
# Each validator raises ConstraintError with a message naming the type and the
# offending key. They are deliberately written out longhand rather than driven
# by a generic schema mini-language: this is the ground-truth layer, and an
# auditor must be able to read the rule for one type without decoding a
# framework.
# --------------------------------------------------------------------------

def _require_dict(ctype: str, params: Any) -> Dict[str, Any]:
    if not isinstance(params, dict):
        raise ConstraintError(f"{ctype}: params must be an object, got {type(params).__name__}")
    for key in params:
        if not isinstance(key, str):
            raise ConstraintError(f"{ctype}: params keys must be strings, got {key!r}")
    return params


def _require_exact_keys(ctype: str, params: Dict[str, Any], allowed: Tuple[str, ...]) -> None:
    missing = [k for k in allowed if k not in params]
    if missing:
        raise ConstraintError(f"{ctype}: params missing required key(s) {missing}")
    extra = sorted(k for k in params if k not in allowed)
    if extra:
        raise ConstraintError(f"{ctype}: params has unknown key(s) {extra}; allowed {list(allowed)}")


def _require_str_list(ctype: str, params: Dict[str, Any], key: str) -> List[str]:
    value = params[key]
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ConstraintError(f"{ctype}: params[{key!r}] must be a list of strings, "
                              f"got {type(value).__name__}")
    items = list(value)
    if not items:
        raise ConstraintError(f"{ctype}: params[{key!r}] must not be empty")
    for item in items:
        if not isinstance(item, str):
            raise ConstraintError(f"{ctype}: params[{key!r}] entries must be strings, got {item!r}")
        if not item.strip():
            raise ConstraintError(f"{ctype}: params[{key!r}] entries must not be blank")
    return items


def _validate_banned_words(ctype: str, params: Any) -> None:
    params = _require_dict(ctype, params)
    _require_exact_keys(ctype, params, ("words",))
    for word in _require_str_list(ctype, params, "words"):
        # A multi-word entry can never match under whole-word matching
        # (ruling A1), so it would silently never fire. Refuse it and point
        # the author at the type that does handle it.
        if any(ch.isspace() for ch in word):
            raise ConstraintError(
                f"{ctype}: params['words'] entry {word!r} contains whitespace; "
                "use type 'banned_phrases' for multi-word entries")


def _validate_banned_phrases(ctype: str, params: Any) -> None:
    params = _require_dict(ctype, params)
    _require_exact_keys(ctype, params, ("phrases",))
    _require_str_list(ctype, params, "phrases")


def _validate_single_letter(ctype: str, params: Any) -> None:
    params = _require_dict(ctype, params)
    _require_exact_keys(ctype, params, ("letter",))
    letter = params["letter"]
    if not isinstance(letter, str):
        raise ConstraintError(f"{ctype}: params['letter'] must be a string, "
                              f"got {type(letter).__name__}")
    if len(letter) != 1:
        raise ConstraintError(f"{ctype}: params['letter'] must be exactly one character, "
                              f"got {letter!r} (length {len(letter)})")
    if not letter.isalpha():
        raise ConstraintError(f"{ctype}: params['letter'] must be alphabetic, got {letter!r}")


def _validate_banned_punctuation(ctype: str, params: Any) -> None:
    params = _require_dict(ctype, params)
    _require_exact_keys(ctype, params, ("marks",))
    for mark in _require_str_list(ctype, params, "marks"):
        if len(mark) != 1:
            raise ConstraintError(f"{ctype}: params['marks'] entries must be single characters, "
                                  f"got {mark!r} (length {len(mark)})")
        if mark.isalnum() or mark.isspace():
            raise ConstraintError(f"{ctype}: params['marks'] entry {mark!r} is not punctuation")
        if unicodedata.category(mark)[0] not in ("P", "S"):
            raise ConstraintError(f"{ctype}: params['marks'] entry {mark!r} has unicode category "
                                  f"{unicodedata.category(mark)}; expected a P* or S* character")


def _validate_no_params(ctype: str, params: Any) -> None:
    params = _require_dict(ctype, params)
    if params:
        raise ConstraintError(f"{ctype}: params must be empty, got keys {sorted(params)}")


class TypeSpec(NamedTuple):
    """One row of the pinned registry."""
    type: str
    family: str
    split: str
    checker_name: str                       # attribute name in constraints.checkers
    validate: Callable[[str, Any], None]    # (type, params) -> None; raises ConstraintError
    example_params: Dict[str, Any]          # the ticket's illustrative params, for docs/tests


#: The PINNED registry. Do not add, remove or rename rows.
TYPES: Dict[str, TypeSpec] = {
    spec.type: spec
    for spec in (
        TypeSpec("banned_words", "lexical", "train",
                 "check_banned_words", _validate_banned_words,
                 {"words": ["coffee", "espresso"]}),
        TypeSpec("banned_phrases", "lexical", "train",
                 "check_banned_phrases", _validate_banned_phrases,
                 {"phrases": ["in conclusion", "as an AI"]}),
        TypeSpec("banned_letter", "character", "train",
                 "check_banned_letter", _validate_single_letter,
                 {"letter": "e"}),
        TypeSpec("banned_punctuation", "character", "heldout_type",
                 "check_banned_punctuation", _validate_banned_punctuation,
                 {"marks": [",", ";"]}),
        TypeSpec("sentence_initial_letter", "structural", "heldout_type",
                 "check_sentence_initial_letter", _validate_single_letter,
                 {"letter": "s"}),
        TypeSpec("no_lists", "format_negative", "train",
                 "check_no_lists", _validate_no_params, {}),
        TypeSpec("no_quotation_marks", "format_negative", "train",
                 "check_no_quotation_marks", _validate_no_params, {}),
    )
}

# --- structural invariants of the registry itself, checked at import ------
assert len({spec.checker_name for spec in TYPES.values()}) == len(TYPES), \
    "registry invariant: exactly one checker per constraint type"
assert all(spec.family in FAMILIES for spec in TYPES.values()), \
    "registry invariant: every type belongs to a known family"
assert all(spec.split in SPLITS for spec in TYPES.values()), \
    "registry invariant: every type has a known split"
assert set(FAMILIES) == {spec.family for spec in TYPES.values()}, \
    "registry invariant: every family is populated (Spec R1 needs >=3)"
assert set(FAMILIES) == set(ID_PREFIX_BY_FAMILY), \
    "registry invariant: every family has an id prefix"


# --------------------------------------------------------------------------
# Lookup helpers
# --------------------------------------------------------------------------

def list_types() -> List[str]:
    """Every registered type, in registry order (stable)."""
    return list(TYPES)


def types_by_family(family: str) -> List[str]:
    """Types belonging to ``family``, in registry order. Unknown family raises."""
    if family not in FAMILIES:
        raise ConstraintError(f"unknown family {family!r}; known families {list(FAMILIES)}")
    return [t for t, spec in TYPES.items() if spec.family == family]


def types_by_split(split: str) -> List[str]:
    """Types belonging to ``split``, in registry order. Unknown split raises."""
    if split not in SPLITS:
        raise ConstraintError(f"unknown split {split!r}; known splits {list(SPLITS)}")
    return [t for t, spec in TYPES.items() if spec.split == split]


def _spec(ctype: Any) -> TypeSpec:
    if not isinstance(ctype, str):
        raise ConstraintError(f"constraint type must be a string, got {type(ctype).__name__}")
    try:
        return TYPES[ctype]
    except KeyError:
        raise ConstraintError(f"unknown constraint type {ctype!r}; "
                              f"known types {list_types()}") from None


def family_of(ctype: str) -> str:
    return _spec(ctype).family


def split_of(ctype: str) -> str:
    return _spec(ctype).split


def checker_for(ctype: str) -> Callable[[Dict[str, Any], str], Dict[str, Any]]:
    """The one checker function for ``ctype``.

    Imported lazily so :mod:`constraints.registry` stays free of any dependency
    on :mod:`constraints.checkers` at module-import time.
    """
    from constraints import checkers  # local import: breaks the import cycle

    return getattr(checkers, _spec(ctype).checker_name)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def validate_params(ctype: str, params: Any) -> None:
    """Raise :class:`ConstraintError` unless ``params`` fit ``ctype``'s schema."""
    spec = _spec(ctype)
    spec.validate(spec.type, params)


def normalize_constraint(constraint: Any) -> Dict[str, Any]:
    """Validate the *checkable* core of a constraint and return it filled in.

    Requires ``type`` and ``params`` and validates the params strictly. ``id``,
    ``family`` and ``split`` are optional here, but if present they must AGREE
    with the pinned registry -- a constraint claiming
    ``{"type": "no_lists", "split": "heldout_type"}`` is a drift bug and raises
    rather than being quietly corrected. The returned dict always carries all
    five keys (``id`` may be ``None``).

    This is what :func:`constraints.checkers.check` calls. Use
    :func:`validate_constraint` for the stricter whole-object check.
    """
    if not isinstance(constraint, dict):
        raise ConstraintError(f"constraint must be an object, got {type(constraint).__name__}")
    if "type" not in constraint:
        raise ConstraintError("constraint is missing 'type'")
    if "params" not in constraint:
        raise ConstraintError(f"constraint {constraint.get('id')!r}: missing 'params'")

    spec = _spec(constraint["type"])
    validate_params(spec.type, constraint["params"])

    for key, expected in (("family", spec.family), ("split", spec.split)):
        actual = constraint.get(key)
        if actual is not None and actual != expected:
            raise ConstraintError(
                f"{spec.type}: {key} is {actual!r} but the registry pins it to {expected!r}")

    cid = constraint.get("id")
    if cid is not None:
        _validate_id(spec, cid)

    known = {"id", "type", "params", "family", "split"}
    extra = sorted(k for k in constraint if k not in known)
    if extra:
        raise ConstraintError(f"{spec.type}: constraint has unknown key(s) {extra}")

    return {"id": cid, "type": spec.type, "params": constraint["params"],
            "family": spec.family, "split": spec.split}


def _validate_id(spec: TypeSpec, cid: Any) -> None:
    if not isinstance(cid, str):
        raise ConstraintError(f"{spec.type}: id must be a string, got {type(cid).__name__}")
    if not ID_RE.match(cid):
        raise ConstraintError(f"{spec.type}: id {cid!r} does not match {ID_RE.pattern}")
    prefix = ID_PREFIX_BY_FAMILY[spec.family]
    if not cid.startswith(prefix):
        raise ConstraintError(f"{spec.type}: id {cid!r} must start with {prefix!r} "
                              f"for family {spec.family!r}")


def validate_constraint(constraint: Any) -> Dict[str, Any]:
    """Strict whole-object validation: all five pinned fields must be present.

    Returns the normalized constraint. Raises :class:`ConstraintError` on any
    deviation. This is the validator the scenario generator (T-2) and the
    dataset writer (T-5) should use on every constraint they emit.
    """
    if not isinstance(constraint, dict):
        raise ConstraintError(f"constraint must be an object, got {type(constraint).__name__}")
    for key in ("id", "type", "params", "family", "split"):
        if key not in constraint:
            raise ConstraintError(f"constraint is missing required key {key!r}")
    if constraint["id"] is None:
        raise ConstraintError("constraint 'id' must not be null")
    return normalize_constraint(constraint)


def build_constraint(ctype: str, params: Dict[str, Any], index: int) -> Dict[str, Any]:
    """Build a pinned-shape constraint, deriving ``family``/``split``/``id`` prefix
    from the registry so callers cannot drift from it.

    ``index`` is the caller's per-family counter; the id is zero-padded to three
    digits (``c-lex-014``) and widens beyond 999 rather than wrapping.
    """
    spec = _spec(ctype)
    validate_params(spec.type, params)
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ConstraintError(f"{spec.type}: index must be a non-negative int, got {index!r}")
    return {"id": f"{ID_PREFIX_BY_FAMILY[spec.family]}{index:03d}",
            "type": spec.type,
            "params": params,
            "family": spec.family,
            "split": spec.split}
