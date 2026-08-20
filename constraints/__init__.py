"""Constraints package: the pinned constraint registry and its deterministic checkers.

The ground-truth layer for the Forbidden-Constraint Holder target. Everything
downstream -- the G1 prompt-ceiling gate, the training-data rejection filter and
every base-vs-tuned number -- trusts these checkers, so they are pure,
deterministic and audited against human-approved golden fixtures.
"""
from constraints.checkers import CLEAN_EVIDENCE, check
from constraints.registry import (
    FAMILIES,
    SPLITS,
    TYPES,
    ConstraintError,
    build_constraint,
    checker_for,
    list_types,
    types_by_family,
    types_by_split,
    validate_constraint,
    validate_params,
)

__all__ = [
    "check",
    "CLEAN_EVIDENCE",
    "ConstraintError",
    "FAMILIES",
    "SPLITS",
    "TYPES",
    "build_constraint",
    "checker_for",
    "list_types",
    "types_by_family",
    "types_by_split",
    "validate_constraint",
    "validate_params",
]
