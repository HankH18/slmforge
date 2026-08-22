#!/usr/bin/env python3
"""Everything the Space actually does. No Gradio import lives in this file.

WHY THE SPLIT
-------------
``app.py`` is layout: components, labels, and one ``.click()`` whose ``fn`` IS
:func:`compare` -- not a wrapper around it, not a copy of it. So the function a
visitor triggers in the browser is the same object the local verification
harness (``test_space_app.py``) drives with a stub generator and the REAL
checkers. There is no second code path to fall out of sync, and the callback can
be proven to work on a machine where Gradio is not installed at all.

NOTHING HERE IS A SECOND IMPLEMENTATION
---------------------------------------
Same rule as ``scripts/demo.py``, for the same reason: a demo that renders its
own prompt or runs its own checkers demonstrates a different system that happens
to share a repo. Every load-bearing step is imported from the pinned objects,
and most of them are imported from ``scripts/demo.py`` itself so that the Space
and the CLI cannot diverge:

  constraint parsing  ``demo.parse_constraint_spec`` / ``demo.resolve_constraints``
  scenario shape      ``demo.build_scenario``
  prompt              ``scenarios.render.render_user_message`` via
                      ``eval.build_prompt`` inside ``demo.generate``
  generation          ``demo.generate`` -> ``eval.make_generator`` (one model
                      resident at a time; it frees the weights itself)
  checking            ``demo.run_checks`` -> ``constraints.checkers.check``
  spec_pass           ``harness.compute_spec_pass`` -- the single source of truth
  scoreboard/footer   ``demo._format_scoreboard`` / ``demo.format_footer``
  model pins          ``demo.load_model_specs`` reading ``configs/models.json``

The measured headline numbers shown in the footer are NOT typed in here. They
come from ``demo.format_footer()``, whose table ``tests/test_demo.py`` recomputes
from ``runs/final/`` and fails on drift. This module states no rate of its own.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
No judge, no API keys, no spend. ``spec_pass`` is "all checkers pass AND judged
quality >= 3"; with no judge the true value is *undecided*, and that is what
``harness.compute_spec_pass(rows, None)`` returns and what the page displays.
The Space never guesses the missing half.
"""
from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import _repo  # noqa: E402

ROOT: Path = _repo.install()

import demo                                              # noqa: E402  scripts/demo.py
import eval as eval_mod                                  # noqa: E402
import harness                                           # noqa: E402
from constraints import ConstraintError                  # noqa: E402
from constraints import registry                         # noqa: E402
from scenarios.render import RenderError                 # noqa: E402
from scenarios.render import render_user_message         # noqa: E402

__all__ = [
    "ROOT", "OUTPUT_FIELDS", "EXAMPLE_SPECS", "EXAMPLE_CHOICES", "Pins",
    "SpaceError", "compare", "hardware_notice", "load_pins",
    "parse_constraint_lines", "thinking_flag", "MAX_NEW_TOKENS_CEILING",
    "append_specs",
]


class SpaceError(RuntimeError):
    """A visitor-fixable problem: bad constraint spec, empty task, no model."""


# --------------------------------------------------------------------------
# The evaluated generation budget.
#
# Not a new number: ``demo.EVAL_HF_MAX_NEW_TOKENS`` mirrors the literal in
# ``eval.HFGenerator.__call__`` and ``tests/test_demo.py`` reads eval.py's source
# to pin the two together. The Space adopts it as the DEFAULT so that a
# generation shown to a grader is produced under the budget the reported numbers
# were measured under.
# --------------------------------------------------------------------------
EVALUATED_MAX_NEW_TOKENS: int = demo.EVAL_HF_MAX_NEW_TOKENS

#: The slider's upper bound. Above the evaluated budget there is nothing to
#: compare against, so the UI does not offer it.
MAX_NEW_TOKENS_CEILING: int = EVALUATED_MAX_NEW_TOKENS

#: Mirrors the note ``demo.main`` prints when the budget is not the evaluated
#: one. Kept in the same words on purpose; ``test_space_app.py`` asserts the
#: Space refuses to stay silent about a lowered budget.
BUDGET_NOTE = ("--max-new-tokens {n} differs from the evaluated budget of {e}; a "
               "truncated answer can fail a constraint it would otherwise have held.")


# --------------------------------------------------------------------------
# enable_thinking parity
#
# The tuned model was trained, and both models were evaluated, with the chat
# template's thinking mode OFF. A Space generating under the other condition
# would be showing a grader a prompt neither model was measured on, so the two
# sources are cross-checked at import and disagreement is fatal.
# --------------------------------------------------------------------------
_THINKING_RE = re.compile(r"enable_thinking\s*=\s*(True|False)")


def thinking_flag() -> bool:
    """The one ``enable_thinking`` value, cross-checked across both sources.

    Raises :class:`SpaceError` if ``configs/train.json:data.enable_thinking``
    and every ``enable_thinking=`` literal in ``eval.py`` do not agree. This is
    a real check, not a docstring: delete either half and the Space refuses to
    import.
    """
    import json
    train_cfg = json.loads((ROOT / "configs" / "train.json").read_text(encoding="utf-8"))
    configured = train_cfg["data"]["enable_thinking"]
    if not isinstance(configured, bool):
        raise SpaceError(f"configs/train.json:data.enable_thinking is "
                         f"{configured!r}, not a bool")
    src = (ROOT / "eval.py").read_text(encoding="utf-8")
    found = {m.group(1) == "True" for m in _THINKING_RE.finditer(src)}
    if not found:
        raise SpaceError("eval.py names no enable_thinking value; the Space cannot "
                         "confirm it generates under the evaluated template condition")
    if found != {configured}:
        raise SpaceError(
            f"enable_thinking disagreement: configs/train.json says {configured} but "
            f"eval.py uses {sorted(found)}. The Space will not generate under a "
            f"template condition the numbers were not measured under.")
    return configured


ENABLE_THINKING: bool = thinking_flag()


# --------------------------------------------------------------------------
# Constraint quick-pick, DERIVED from the registry
#
# The example strings are not typed out: each one is built from
# ``registry.TYPES[t].example_params`` through the inverse of ``demo._SPEC_STYLE``
# and then round-tripped back through ``demo.parse_constraint_spec``. If the
# registry gains a type, changes a param key, or changes an example, either the
# build or the round-trip fails at import -- the UI can never offer a chip that
# the pinned parser rejects, and it can never omit a registered type.
# --------------------------------------------------------------------------

def _example_spec(ctype: str) -> str:
    key, style = demo._SPEC_STYLE[ctype]
    params = registry.TYPES[ctype].example_params
    if style == "none":
        return ctype
    value = params[key]
    if style == "csv":
        return f"{ctype}:{','.join(value)}"
    if style == "chars":
        return f"{ctype}:{''.join(value)}"
    return f"{ctype}:{value}"


def _build_examples() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for ctype in registry.list_types():
        spec = _example_spec(ctype)
        parsed_type, parsed_params = demo.parse_constraint_spec(spec)
        expected = registry.TYPES[ctype].example_params
        if parsed_type != ctype or parsed_params != expected:
            raise SpaceError(
                f"quick-pick example for {ctype!r} does not round-trip: built "
                f"{spec!r}, parsed back as {parsed_type!r} {parsed_params!r}, "
                f"registry example is {expected!r}")
        out[ctype] = spec
    covered = set(out)
    registered = set(registry.list_types())
    if covered != registered:
        raise SpaceError(f"quick-pick covers {sorted(covered)} but the registry pins "
                         f"{sorted(registered)}")
    return out


#: type -> a ready-to-paste --constraint spec, derived from the registry.
EXAMPLE_SPECS: Dict[str, str] = _build_examples()

#: (label, spec) for the UI chips. Label carries the pinned train/held-out split,
#: because "the tuned model collapsed on the held-out types" is the project's
#: headline and a visitor picking a chip should see which side they picked.
EXAMPLE_CHOICES: List[Tuple[str, str]] = [
    (f"{ctype}  [{registry.split_of(ctype)}]", EXAMPLE_SPECS[ctype])
    for ctype in registry.list_types()
]


# --------------------------------------------------------------------------
# Model pins, read at runtime from configs/models.json. No sha is written here.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Pins:
    root: Path
    specs: Dict[str, str]
    models_json: Path
    enable_thinking: bool
    evaluated_max_new_tokens: int

    def line(self, role: str) -> str:
        return f"{role}: `{self.specs[role]}`"


def load_pins(models_json: Optional[Path] = None) -> Pins:
    """``configs/models.json`` -> the two specs, through ``demo.load_model_specs``.

    ``demo.load_model_specs`` is what normalises the file's asymmetry (``base``
    carries its ``hf:`` provider, ``tuned`` does not) and refuses a file that
    pins the same model twice. Doing that here would be a second implementation
    of the pin contract.
    """
    path = Path(models_json) if models_json else (ROOT / "configs" / "models.json")
    return Pins(root=ROOT, specs=demo.load_model_specs(path), models_json=path,
                enable_thinking=ENABLE_THINKING,
                evaluated_max_new_tokens=EVALUATED_MAX_NEW_TOKENS)


# --------------------------------------------------------------------------
# Hardware / latency honesty
# --------------------------------------------------------------------------

def hardware_notice() -> str:
    """What this Space is running on and what that costs, stated up front.

    The tier comes from the Hub's own ``SPACE_HARDWARE`` env var when present.
    The latency figure is labelled an ESTIMATE because it has not been measured
    on this hardware -- and the app prints the real elapsed seconds for each
    model on every run, so the estimate is replaced by a measurement the first
    time anyone presses the button.
    """
    tier = os.environ.get("SPACE_HARDWARE") or "unknown (not reported by the host)"
    space_id = os.environ.get("SPACE_ID")
    cores = os.cpu_count()
    where = f"Hugging Face Space `{space_id}`" if space_id else "a local process"
    return (
        f"**Hardware:** {where}, tier `{tier}`, {cores} CPU core(s) visible. "
        f"There is no GPU on the free tier.\n\n"
        f"**Expect this to be slow.** Qwen3-1.7B generating "
        f"{EVALUATED_MAX_NEW_TOKENS} tokens on CPU is an order of minutes per "
        f"model, and this comparison runs two models one after the other. The "
        f"very first run additionally downloads ~3.4 GB of weights per model "
        f"into the Space's cache. These are estimates, not measurements on this "
        f"hardware; the real elapsed time is printed under each answer below.\n\n"
        f"**Why it is not faster:** the two models are loaded and freed one at a "
        f"time rather than kept resident together (`scripts/demo.py` does the "
        f"same, for the OOM reason recorded in its docstring), and the token "
        f"budget defaults to {EVALUATED_MAX_NEW_TOKENS} because that is the "
        f"budget the published numbers were measured under."
    )


# --------------------------------------------------------------------------
# Constraint input
# --------------------------------------------------------------------------

def parse_constraint_lines(text: Any) -> List[Tuple[str, str]]:
    """One constraint per line -> ``[("spec"|"id", line), ...]``.

    Same rule as ``demo._read_interactive``: a line matching
    ``registry.ID_RE`` reuses a shipped constraint instance, anything else is a
    spec. ``test_space_app.py`` asserts this function and that one classify a
    battery of lines identically, so the two cannot drift.
    """
    if text is None:
        return []
    items: List[Tuple[str, str]] = []
    for raw in str(text).splitlines():
        line = raw.strip()
        if not line:
            continue
        items.append(("id" if registry.ID_RE.match(line) else "spec", line))
    return items


def append_specs(current: Any, picked: Any) -> str:
    """Quick-pick handler: add the chosen example specs to the constraint box.

    Kept here rather than in ``app.py`` so it is testable without Gradio.
    Already-present lines are not duplicated, because ``resolve_constraints``
    rejects the same constraint twice and a visitor double-clicking a chip
    should not be shown an error for it.
    """
    lines = [ln.strip() for ln in str(current or "").splitlines() if ln.strip()]
    if picked is None:
        chosen: List[str] = []
    elif isinstance(picked, str):
        chosen = [picked]
    else:
        chosen = [str(p) for p in picked]
    for spec in chosen:
        if spec not in lines:
            lines.append(spec)
    return "\n".join(lines)


def _normalize_roles(roles: Any) -> List[str]:
    if roles is None:
        chosen = list(demo.ROLES)
    elif isinstance(roles, str):
        chosen = [roles]
    else:
        chosen = [str(r) for r in roles]
    unknown = [r for r in chosen if r not in demo.ROLES]
    if unknown:
        raise SpaceError(f"unknown model role(s) {unknown}; this Space compares "
                         f"exactly {list(demo.ROLES)}")
    ordered = [r for r in demo.ROLES if r in chosen]
    if not ordered:
        raise SpaceError("pick at least one model to run.")
    return ordered


def _normalize_budget(value: Any) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise SpaceError(f"max new tokens must be a whole number, got {value!r}") from None
    if n < 1:
        raise SpaceError("max new tokens must be at least 1")
    if n > MAX_NEW_TOKENS_CEILING:
        raise SpaceError(f"max new tokens is capped at {MAX_NEW_TOKENS_CEILING}, the "
                         f"evaluated budget; a longer answer would not be comparable "
                         f"with the published numbers")
    return n


# --------------------------------------------------------------------------
# Rendering the verdict
# --------------------------------------------------------------------------

_STATUS_ICON = {"PASS": "PASS", "FAIL": "FAIL", "ERROR": "ERROR"}


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _checks_table(checks: Sequence[Mapping[str, Any]]) -> str:
    rows = ["| # | verdict | type | id | constraint | evidence |",
            "| --- | --- | --- | --- | --- | --- |"]
    for i, c in enumerate(checks, start=1):
        detail = _md_escape(str(c["detail"]))
        if c["status"] == "FAIL":
            detail = f"**VIOLATION** -- {detail}"
        elif c["status"] == "ERROR":
            detail = f"**CHECKER ERROR (not a pass)** -- {detail}"
        # The id is printed because --constraint-id reuse is supported: a
        # visitor who typed `c-lex-001` must be able to see which instance
        # actually got checked, not just its rendered sentence.
        rows.append(f"| {i} | **{_STATUS_ICON[c['status']]}** | `{c['type']}` "
                    f"({registry.split_of(c['type'])}) | `{c['constraint_id']}` "
                    f"| {_md_escape(c['requirement'])} | {detail} |")
    return "\n".join(rows)


def _spec_pass_block(result: "demo.ModelResult") -> str:
    """The spec_pass half of the verdict, computed by ``harness.compute_spec_pass``.

    ``spec_pass`` = all checkers pass AND judged quality >= QUALITY_THRESHOLD.
    No judge runs here, so the honest value is the one the pinned function
    returns for a missing score: ``None``. The conditional line beside it is also
    a real call to the same function -- it is labelled as conditional and is not
    a claim that the quality bar was met.
    """
    if result.n_errored:
        return ("**spec_pass: WITHHELD.** A checker raised, so the constraint half "
                "is unknown, and an unknown verdict is not a pass.")
    rows = [{"pass": c["status"] == "PASS"} for c in result.checks]
    undecided = harness.compute_spec_pass(rows, None)
    conditional = harness.compute_spec_pass(rows, harness.QUALITY_THRESHOLD)
    assert undecided is None, "compute_spec_pass returned a verdict without a judge"
    return (
        f"**spec_pass: UNDECIDED.** `harness.compute_spec_pass(checks, "
        f"quality_score=None)` returns `None` -- no judge ran, and unknown is "
        f"neither pass nor fail.\n\n"
        f"- constraint half (all checkers passed): "
        f"**{'yes' if result.all_held else 'no'}**\n"
        f"- quality half (judged >= {harness.QUALITY_THRESHOLD}): **not measured here**\n"
        f"- *if* a judge scored this >= {harness.QUALITY_THRESHOLD}, spec_pass "
        f"would be **{conditional}** -- a conditional, not a measurement."
    )


def _verdict_md(result: "demo.ModelResult", pins: Pins) -> str:
    head = [f"### {result.role.upper()}", "", pins.line(result.role), ""]
    if not result.ran:
        head += [f"**Generation failed.** `{result.error}`", "", demo._verdict_line(result)]
        return "\n".join(head)
    took = f"{result.elapsed_s:.1f}s" if result.elapsed_s is not None else "n/a"
    head += [f"{len(result.output)} characters in **{took}** on this hardware "
             f"(load + generate, measured).", ""]
    if not (result.output or "").strip():
        head += ["**Empty output.** Every checker passes an empty string vacuously "
                 "(checkers ruling A7); that is not compliance.", ""]
    head += [_checks_table(result.checks), "", demo._verdict_line(result), "",
             _spec_pass_block(result)]
    return "\n".join(head)


def _scoreboard_md(constraints: Sequence[Mapping[str, Any]],
                   results: Sequence["demo.ModelResult"]) -> str:
    if not results:
        return ""
    body = "\n".join(demo._format_scoreboard(constraints, results))
    return ("### Side by side\n\n```text\n" + body + "\n```\n\n"
            "Rendered by `scripts/demo.py:_format_scoreboard` -- the same "
            "scoreboard the CLI prints.")


def _footer_md() -> str:
    return ("### The measured result -- read this before drawing a conclusion\n\n"
            "```text\n" + demo.format_footer() + "\n```\n\n"
            "Source: `docs/FINAL_REPORT.md` and the transcripts in `runs/final/`. "
            "This block is produced by `scripts/demo.py:format_footer()`, whose "
            "table `tests/test_demo.py` recomputes from those transcripts and "
            "fails on drift. This page states no number of its own.")


def _prompt_md(rendered: str, pins: Pins, budget: int, note: Optional[str]) -> str:
    lines = [
        "### The prompt both models received",
        "",
        "Rendered by `scenarios.render.render_user_message` -- the one renderer "
        "shared by the training data and the eval. No system message "
        f"(the `{demo.STRATEGY}` training condition), "
        f"`enable_thinking={pins.enable_thinking}` (matches "
        "`configs/train.json:data.enable_thinking` and `eval.HFGenerator`), "
        f"`max_new_tokens={budget}`.",
        "",
        "```text",
        rendered,
        "```",
    ]
    if note:
        lines += ["", f"> **Note.** {note}"]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# The callback
# --------------------------------------------------------------------------

#: The exact order ``compare`` yields, and the exact order ``app.py`` must list
#: its output components. Each of ``app.py``'s output components is tagged with
#: the name it carries, and ``app.py`` asserts the tags equal this tuple ELEMENT
#: BY ELEMENT at build time -- so transposing the ``base_*``/``tuned_*`` pair
#: here fails the UI on startup rather than showing a grader the tuned model's
#: answer in the base model's box. Editing this tuple without editing the
#: matching component in ``app.py`` is a startup error, by construction.
OUTPUT_FIELDS: Tuple[str, ...] = (
    "status",
    "rendered_prompt",
    "base_output",
    "base_verdict",
    "tuned_output",
    "tuned_verdict",
    "scoreboard",
    "footer",
)

_EMPTY: Tuple[str, ...] = ("",) * len(OUTPUT_FIELDS)


def _pack(**kwargs: str) -> Tuple[str, ...]:
    unknown = set(kwargs) - set(OUTPUT_FIELDS)
    if unknown:  # pragma: no cover - programming error, not a visitor error
        raise SpaceError(f"unknown output field(s) {sorted(unknown)}")
    return tuple(kwargs.get(name, "") for name in OUTPUT_FIELDS)


def _error(message: str, *, syntax: bool = True) -> Tuple[str, ...]:
    body = f"**Cannot run.** {message}"
    if syntax:
        body += "\n\n```text\n" + demo.spec_syntax_help() + "\n```"
    return _pack(status=body)


def compare(task: str,
            constraint_text: str,
            roles: Any = None,
            max_new_tokens: Any = EVALUATED_MAX_NEW_TOKENS,
            *,
            generator_factory: Any = None,
            models_json: Optional[Path] = None,
            ) -> Iterator[Tuple[str, ...]]:
    """THE callback. Yields one full output tuple per stage.

    A generator, so the page fills in as each model finishes rather than staying
    blank for the several minutes a CPU generation takes.

    ``generator_factory`` is the injectable model-loading seam. It is a
    KEYWORD-ONLY argument and ``app.py`` binds only the four positional inputs,
    so no visitor of the deployed Space can reach it -- it exists for
    ``test_space_app.py``, which drives this exact function with a stub generator
    and the real checkers.
    """
    try:
        pins = load_pins(models_json)
        chosen = _normalize_roles(roles)
        budget = _normalize_budget(max_new_tokens)
        items = parse_constraint_lines(constraint_text)
        instances = demo.load_instances(ROOT / "data" / "scenarios" / "constraint_instances.jsonl")
        constraints, lookup = demo.resolve_constraints(items, instances)
        scenario = demo.build_scenario(task, constraints)
        rendered = render_user_message(scenario, lookup)
    except (demo.DemoError, SpaceError, ConstraintError, RenderError,
            harness.HarnessError, FileNotFoundError) as exc:
        yield _error(str(exc))
        return

    note = None
    if budget != EVALUATED_MAX_NEW_TOKENS:
        note = BUDGET_NOTE.format(n=budget, e=EVALUATED_MAX_NEW_TOKENS)
    if len(constraints) < 2 or len(constraints) > 4:
        extra = (f"This run uses {len(constraints)} constraint(s); the graded spec "
                 f"uses 2-4 per scenario.")
        note = f"{note} {extra}" if note else extra

    prompt_block = _prompt_md(rendered, pins, budget, note)
    state: Dict[str, str] = {"rendered_prompt": prompt_block}

    results: List["demo.ModelResult"] = []
    for i, role in enumerate(chosen, start=1):
        state["status"] = (f"Running **{role}** ({i} of {len(chosen)}): loading "
                           f"`{pins.specs[role]}` and generating up to {budget} "
                           f"tokens on CPU. One model is resident at a time. "
                           f"This takes minutes -- leave the tab open.")
        yield _pack(**state)

        result = demo.ModelResult(role=role, spec=pins.specs[role])
        started = time.time()
        try:
            result.output = demo.generate(pins.specs[role], scenario, lookup, budget,
                                          generator_factory=generator_factory)
        except Exception as exc:  # noqa: BLE001 - surfaced in the table, never hidden
            result.error = f"{type(exc).__name__}: {exc}"
        result.elapsed_s = time.time() - started
        if result.output is not None:
            result.checks = demo.run_checks(constraints, result.output)
        results.append(result)

        state[f"{role}_output"] = result.output if result.output is not None else ""
        state[f"{role}_verdict"] = _verdict_md(result, pins)
        state["status"] = f"Finished **{role}** in {result.elapsed_s:.1f}s."
        yield _pack(**state)

    ran = [r.role for r in results if r.ran]
    failed = [r.role for r in results if not r.ran]
    summary = (f"Done. Generated: {', '.join(ran) if ran else 'nothing'}."
               + (f" FAILED: {', '.join(failed)}." if failed else ""))
    state["status"] = (summary + "\n\nVerdicts above are computed live by "
                       "`constraints.checkers.check` -- the same checkers every "
                       "published number in this project is re-derived from.")
    state["scoreboard"] = _scoreboard_md(constraints, results)
    state["footer"] = _footer_md()
    yield _pack(**state)
