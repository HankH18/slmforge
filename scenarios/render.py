"""Scenario -> user message. The one definition of the prompt.

This module is the load-bearing contract of the project: the eval harness and the
training-data builder must render a scenario the SAME way, byte for byte. If they
diverge, the tuned model is measured under a prompt it never trained on and every
reported number is meaningless. So there is exactly one renderer, imported by both
-- never a second copy in training code.

Two deliberate choices, both of which would bias the measurement if made the other
way:

1. The requirement block is plain sentences, one per line, with no bullet or number
   markers. One of the constraint types is ``no_lists``; priming the model with a
   list-formatted prompt would confound the very thing being measured.

2. Requirements come after the task and are stated plainly and firmly, exactly once.
   Adversarial scenarios carry their pressure inside ``task`` (a nudge arguing the
   rules are optional). The renderer does not argue back -- pushing harder here
   would neutralise the pressure the scenario exists to apply.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Sequence

from constraints import registry

__all__ = ["render_user_message", "render_constraint", "RenderError", "RENDERERS"]

HEADER = "Requirements -- your response must satisfy every one of these:"


class RenderError(RuntimeError):
    """A scenario or constraint cannot be rendered. Never raised for style."""


# Punctuation is named rather than shown, because a comma-separated list of marks
# that includes a comma is ambiguous to a reader and to a model.
_MARK_NAMES = {
    ",": "commas", ";": "semicolons", ":": "colons", ".": "periods",
    "!": "exclamation marks", "?": "question marks", "-": "hyphens",
    "(": "opening parentheses", ")": "closing parentheses",
    "—": "em dashes", "–": "en dashes", "'": "apostrophes",
}


def _mark_name(mark: str) -> str:
    return _MARK_NAMES.get(mark, f"the {mark!r} character")


def _join(items: Sequence[str]) -> str:
    return ", ".join(items)


def _banned_words(params: Mapping[str, Any]) -> str:
    words = params["words"]
    if len(words) == 1:
        return f"Do not use the word {words[0]}."
    return f"Do not use any of these words: {_join(words)}."


def _banned_phrases(params: Mapping[str, Any]) -> str:
    phrases = params["phrases"]
    if len(phrases) == 1:
        return f"Do not use the phrase {phrases[0]}."
    return f"Do not use any of these phrases: {_join(phrases)}."


def _banned_letter(params: Mapping[str, Any]) -> str:
    return (f"Do not use the letter {params['letter']} anywhere in your response, "
            f"including inside words.")


def _banned_punctuation(params: Mapping[str, Any]) -> str:
    names = [_mark_name(m) for m in params["marks"]]
    if len(names) == 1:
        return f"Do not use {names[0]}."
    return f"Do not use {_join(names[:-1])} or {names[-1]}."


def _sentence_initial_letter(params: Mapping[str, Any]) -> str:
    return f"Every sentence must begin with the letter {params['letter']}."


def _no_lists(_params: Mapping[str, Any]) -> str:
    return "Do not use bulleted or numbered lists."


def _no_quotation_marks(_params: Mapping[str, Any]) -> str:
    return "Do not use quotation marks."


RENDERERS: Dict[str, Callable[[Mapping[str, Any]], str]] = {
    "banned_words": _banned_words,
    "banned_phrases": _banned_phrases,
    "banned_letter": _banned_letter,
    "banned_punctuation": _banned_punctuation,
    "sentence_initial_letter": _sentence_initial_letter,
    "no_lists": _no_lists,
    "no_quotation_marks": _no_quotation_marks,
}

# A registered type with no prompt text would be silently unstated in the prompt
# while still being scored -- an unannounced constraint the model cannot obey.
# Asserted at import so it cannot reach a run.
_missing = set(registry.list_types()) - set(RENDERERS)
_extra = set(RENDERERS) - set(registry.list_types())
if _missing or _extra:  # pragma: no cover - import-time guard
    raise RenderError(
        f"renderer/registry mismatch: unrendered types {sorted(_missing)}, "
        f"unregistered renderers {sorted(_extra)}")


def render_constraint(constraint: Mapping[str, Any]) -> str:
    """One constraint -> one plain-English sentence, validated first."""
    if not isinstance(constraint, Mapping):
        raise RenderError(f"constraint must be an object, got {type(constraint).__name__}")
    ctype = constraint.get("type")
    if ctype not in RENDERERS:
        raise RenderError(f"unknown constraint type {ctype!r}")
    params = constraint.get("params")
    # Validate through the registry so a malformed constraint cannot be rendered
    # into a plausible-looking instruction the checker will then score differently.
    registry.validate_params(ctype, params)
    return RENDERERS[ctype](params)


def render_user_message(scenario: Mapping[str, Any],
                        constraint_lookup: Mapping[str, Mapping[str, Any]]) -> str:
    """Scenario + constraint lookup -> the exact user message the model sees.

    Deterministic: constraints are rendered in the scenario's declared order, so
    the same scenario always produces byte-identical output. Raises rather than
    degrading -- a dangling constraint id must not silently become a prompt that
    omits a constraint the scorer will still enforce.
    """
    if not isinstance(scenario, Mapping):
        raise RenderError(f"scenario must be an object, got {type(scenario).__name__}")

    task = scenario.get("task")
    if not isinstance(task, str) or not task.strip():
        raise RenderError(f"scenario {scenario.get('scenario_id')!r} has no usable 'task'")

    cids = scenario.get("constraints")
    if not isinstance(cids, (list, tuple)) or not cids:
        raise RenderError(f"scenario {scenario.get('scenario_id')!r} has no constraints")

    lines = []
    for cid in cids:
        if cid not in constraint_lookup:
            raise RenderError(
                f"scenario {scenario.get('scenario_id')!r} references unknown "
                f"constraint id {cid!r}")
        lines.append(render_constraint(constraint_lookup[cid]))

    return f"{task.strip()}\n\n{HEADER}\n" + "\n".join(lines)
