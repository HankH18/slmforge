#!/usr/bin/env python3
"""Live demo (T-6): one grader-supplied prompt, base vs tuned, side by side.

    scripts/demo.py --prompt "Write a short paragraph about a coastal town." \
                    --constraint banned_words:however,moreover

    scripts/demo.py --interactive

WHAT THIS IS FOR
----------------
The project's claim is a base-vs-tuned comparison over a 147-scenario corpus.
This script makes that claim inspectable in one minute without opening any
JSON: a grader types their OWN task and their OWN forbidden constraint, and
sees both models' answers with the SAME deterministic verdict the eval uses
printed underneath each.

NOTHING HERE IS A SECOND IMPLEMENTATION
---------------------------------------
A demo that rendered its own prompt, or ran its own checkers, would prove
nothing about the evaluated system -- it would demonstrate a different system
that happens to live in the same repo. So every load-bearing step is imported:

  prompt      ``scenarios.render.render_user_message`` (via
              ``eval.build_prompt``) -- the ONE renderer, shared with
              training/modal_app.py and pinned in
              configs/train.json:data.prompt_renderer
  generation  ``eval.make_generator`` / ``eval.HFGenerator`` -- so the
              chat-template call (add_generation_prompt=True,
              enable_thinking=False) is the evaluated one, byte for byte
  checking    ``constraints.checkers.check`` -- the pinned entry point that
              scripts/gate_check.py and scripts/check_mvp.py re-derive every
              published number from
  types       ``constraints.registry`` -- the pinned type table. --constraint
              is validated against it, never against a local copy

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
No judge, no API calls, no spend. The LLM judge scores task quality only; this
script reports the programmatic half. A clean scoreboard here is therefore NOT
``spec_pass`` (which is "every checker passes AND judged quality >= 3"), and one
prompt is an anecdote either way. The footer prints the real measured numbers so
a lucky prompt cannot mislead in either direction.

MEMORY -- this has already killed two runs in this repo
-------------------------------------------------------
``eval.HFGenerator``'s docstring records a MEASURED 19.7GB peak on MPS for ONE
1.7B model on this 34GB machine, and MPS returns per-generation allocations only
inside ``__call__``, never on generator teardown. Holding base and tuned at the
same time can therefore exhaust the machine mid-demo. This script loads ONE
model, generates, drops every reference to it, ``gc.collect()``s and empties the
MPS cache, and only THEN loads the other. The two are never resident together.

Env: reads os.environ and loads no dotenv, exactly like eval.py::

    set -a && . ./.env && set +a && .venv/bin/python scripts/demo.py ...

Exit codes: 0 = both requested models answered and every checker returned a
verdict (PASS or FAIL); 1 = a model failed to generate or a checker raised;
2 = usage error.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import eval as eval_mod                                    # noqa: E402
import harness                                             # noqa: E402
from constraints import ConstraintError                    # noqa: E402
from constraints import check as check_constraint          # noqa: E402
from constraints import registry                           # noqa: E402
from scenarios.render import (RenderError, render_constraint,  # noqa: E402
                              render_user_message)

__all__ = [
    "DemoError", "ModelResult", "build_scenario", "format_report", "format_footer",
    "generate", "load_instances", "load_model_specs", "normalize_model_spec",
    "parse_constraint_spec", "release_generator", "resolve_constraints", "run_checks",
    "spec_syntax_help", "main",
]

# --------------------------------------------------------------------------
# Pinned constants
# --------------------------------------------------------------------------

#: The training condition. eval.py's zero_shot strategy carries NO system
#: message and the tuned model was fine-tuned on exactly that; demoing under any
#: other strategy would show the grader a prompt neither model was measured on.
STRATEGY = "zero_shot"

#: Mirrors the literal in ``eval.HFGenerator.__call__``. Used only to decide
#: whether --max-new-tokens differs from the EVALUATED budget (if it does not,
#: nothing is patched and the generation path is untouched).
#: tests/test_demo.py reads eval.py's source and pins the two together, so this
#: cannot silently drift.
EVAL_HF_MAX_NEW_TOKENS = 512

#: Providers ``eval.make_generator`` understands. Mirrored (not imported --
#: eval.py inlines them in an if-chain and exports no list); tests/test_demo.py
#: re-derives the set from eval.py's source and fails on drift.
KNOWN_PROVIDERS: Tuple[str, ...] = ("hf", "anthropic", "openai", "google", "mock")

DEFAULT_MODELS_JSON = REPO_ROOT / "configs" / "models.json"
DEFAULT_CONSTRAINT_INSTANCES = REPO_ROOT / eval_mod.DEFAULT_CONSTRAINTS

#: Ids minted for constraints typed on the command line. Started well above the
#: shipped library (65 instances, highest index < 100) so a demo constraint can
#: never collide with, or be mistaken for, a corpus instance. Collisions are
#: still checked for explicitly rather than assumed away.
DEMO_ID_INDEX_BASE = 900

ROLES: Tuple[str, ...] = ("base", "tuned")

# --------------------------------------------------------------------------
# The measured MVP result. HARD-CODED HONESTY.
#
# Source: runs/mvp/*.results.json (metrics.spec_adherence,
# metrics.by_constraint_type) and runs/mvp/*.transcripts.jsonl (quality_score),
# n = 147 scenarios, one eval set, one judge. tests/test_demo.py RECOMPUTES
# every number below from those files and fails if this table drifts from them.
#
# These are printed on every run because the honest headline is that the tuned
# model is WORSE on spec_adherence, and a grader who ran one lucky prompt must
# not walk away with the opposite impression.
# --------------------------------------------------------------------------
MVP_N = 147
MVP_SPEC_ADHERENCE: Tuple[float, float] = (0.0890, 0.0748)      # base, tuned
MVP_QUALITY_GE3: Tuple[float, float] = (61.2, 74.8)             # % of rows judged >= 3
#: type -> (base, tuned) share of individual checks that HELD, in percent.
MVP_BY_TYPE: Dict[str, Tuple[float, float]] = {
    "banned_words": (88.7, 95.2),
    "banned_phrases": (100.0, 100.0),
    "banned_letter": (0.0, 1.0),
    "banned_punctuation": (78.0, 43.9),
    "sentence_initial_letter": (72.0, 4.0),
    "no_lists": (98.1, 95.2),
    "no_quotation_marks": (100.0, 100.0),
}


class DemoError(RuntimeError):
    """A usage problem the grader can fix: bad spec, bad id, bad config."""


# --------------------------------------------------------------------------
# --constraint parsing
#
# The spec language is deliberately tiny:  <type>[:<value>]
#
# The param KEY for each type is not invented here -- it is asserted below
# against the registry's own example_params, so a registry change breaks the
# import rather than silently mis-parsing into a constraint that renders one way
# and checks another.
# --------------------------------------------------------------------------

#: type -> (param key, how the text after ':' becomes that param)
#:   "csv"    comma-separated items          banned_words:however,moreover
#:   "scalar" the whole value, verbatim      banned_letter:e
#:   "chars"  every non-space CHARACTER is   banned_punctuation:,;
#:            one item -- because a comma is
#:            itself a legal mark, so a
#:            comma-separated list of marks
#:            would be ambiguous
#:   "none"   the type takes no params       no_lists
_SPEC_STYLE: Dict[str, Tuple[Optional[str], str]] = {
    "banned_words": ("words", "csv"),
    "banned_phrases": ("phrases", "csv"),
    "banned_letter": ("letter", "scalar"),
    "banned_punctuation": ("marks", "chars"),
    "sentence_initial_letter": ("letter", "scalar"),
    "no_lists": (None, "none"),
    "no_quotation_marks": (None, "none"),
}

_missing = set(registry.list_types()) - set(_SPEC_STYLE)
_extra = set(_SPEC_STYLE) - set(registry.list_types())
if _missing or _extra:  # pragma: no cover - import-time guard
    raise DemoError(f"demo/registry mismatch: unparsable types {sorted(_missing)}, "
                    f"unregistered spec styles {sorted(_extra)}")
for _t, (_key, _style) in _SPEC_STYLE.items():  # pragma: no cover - import-time guard
    _expected = sorted(registry.TYPES[_t].example_params)
    _have = sorted([_key] if _key else [])
    if _have != _expected:
        raise DemoError(f"demo spec style for {_t!r} names params {_have} but the "
                        f"registry's schema uses {_expected}")
del _missing, _extra, _t, _key, _style, _expected, _have

_SPEC_USAGE: Dict[str, str] = {
    "banned_words": "banned_words:<word>[,<word>...]",
    "banned_phrases": "banned_phrases:<phrase>[,<phrase>...]",
    "banned_letter": "banned_letter:<single letter>",
    "banned_punctuation": "banned_punctuation:<marks, one character each, e.g. ,;>",
    "sentence_initial_letter": "sentence_initial_letter:<single letter>",
    "no_lists": "no_lists",
    "no_quotation_marks": "no_quotation_marks",
}


def spec_syntax_help() -> str:
    """Every valid --constraint form, registry order. Shown on any spec error."""
    lines = ["valid --constraint forms (type is pinned by constraints/registry.py):"]
    for ctype in registry.list_types():
        lines.append(f"    {_SPEC_USAGE[ctype]:<52}  [{registry.split_of(ctype)}]")
    lines.append("  or reuse a shipped instance:  --constraint-id c-lex-001")
    lines.append("  quote anything your shell would eat:  --constraint 'banned_punctuation:,;'")
    return "\n".join(lines)


def _dedupe(items: Sequence[str]) -> List[str]:
    seen, out = set(), []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def parse_constraint_spec(spec: Any) -> Tuple[str, Dict[str, Any]]:
    """``'banned_words:however,moreover'`` -> ``('banned_words', {'words': [...]})``.

    Raises :class:`DemoError` (never a silent default) on an unknown type, a
    missing value, a value where none is allowed, or params the registry
    rejects. The registry does the schema judging -- this function only shapes
    the text into params.
    """
    if not isinstance(spec, str) or not spec.strip():
        raise DemoError(f"--constraint needs a non-empty spec, got {spec!r}\n{spec_syntax_help()}")
    raw = spec.strip()
    ctype, sep, value = raw.partition(":")
    ctype = ctype.strip()
    if ctype not in _SPEC_STYLE:
        raise DemoError(f"unknown constraint type {ctype!r}\n{spec_syntax_help()}")

    key, style = _SPEC_STYLE[ctype]
    if style == "none":
        if sep or value.strip():
            raise DemoError(f"{ctype} takes no value, got {raw!r}\n"
                            f"    use exactly:  {_SPEC_USAGE[ctype]}")
        params: Dict[str, Any] = {}
    else:
        if not sep:
            raise DemoError(f"{ctype} requires a value\n    use:  {_SPEC_USAGE[ctype]}")
        if style == "csv":
            items = _dedupe([part.strip() for part in value.split(",") if part.strip()])
            if not items:
                raise DemoError(f"{ctype}: no {key} found in {value!r}\n"
                                f"    use:  {_SPEC_USAGE[ctype]}")
            params = {key: items}
        elif style == "chars":
            # every non-space character is one mark; a comma is a legal mark, so
            # splitting on commas here would be ambiguous
            marks = _dedupe([ch for ch in value if not ch.isspace()])
            if not marks:
                raise DemoError(f"{ctype}: no marks found in {value!r}\n"
                                f"    use:  {_SPEC_USAGE[ctype]}")
            params = {key: marks}
        else:  # scalar
            scalar = value.strip()
            if not scalar:
                raise DemoError(f"{ctype} requires a value\n    use:  {_SPEC_USAGE[ctype]}")
            params = {key: scalar}

    # The registry is the schema authority. Its message names the type and the
    # offending key, so surface it verbatim rather than paraphrasing it.
    try:
        registry.validate_params(ctype, params)
    except ConstraintError as exc:
        raise DemoError(f"{exc}\n    use:  {_SPEC_USAGE[ctype]}") from None
    return ctype, params


# --------------------------------------------------------------------------
# Constraint resolution
# --------------------------------------------------------------------------

def load_instances(path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """The shipped constraint library as ``{id: constraint}``.

    Loaded through eval.load_jsonl + harness.build_constraint_lookup, the same
    pair eval.py uses, so a duplicate id is rejected here exactly as it is there.
    """
    p = Path(path) if path is not None else DEFAULT_CONSTRAINT_INSTANCES
    return harness.build_constraint_lookup(eval_mod.load_jsonl(str(p)))


def resolve_constraints(items: Sequence[Tuple[str, str]],
                        instances: Mapping[str, Mapping[str, Any]],
                        ) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """``[("spec", "banned_words:x"), ("id", "c-lex-001")]`` -> (constraints, lookup).

    Order is the order the grader typed, because the renderer states requirements
    in scenario order. Ids minted for typed constraints come from
    ``registry.build_constraint`` so their family/split/id shape cannot drift from
    the pinned table.
    """
    if not items:
        raise DemoError("no constraints given. The demo exists to show a forbidden "
                        "constraint being held or broken, so at least one is required.\n"
                        + spec_syntax_help())
    constraints: List[Dict[str, Any]] = []
    lookup: Dict[str, Dict[str, Any]] = {}
    index = DEMO_ID_INDEX_BASE
    for kind, value in items:
        if kind == "id":
            cid = value.strip()
            if cid not in instances:
                sample = ", ".join(list(instances)[:4])
                raise DemoError(f"--constraint-id {cid!r} is not in "
                                f"{DEFAULT_CONSTRAINT_INSTANCES.name} "
                                f"({len(instances)} instances, e.g. {sample})")
            constraint = dict(instances[cid])
        else:
            ctype, params = parse_constraint_spec(value)
            while True:
                constraint = registry.build_constraint(ctype, params, index)
                index += 1
                if constraint["id"] not in instances:
                    break
        cid = constraint["id"]
        if cid in lookup:
            raise DemoError(f"constraint {cid!r} given twice; each constraint may only "
                            f"appear once in a scenario")
        lookup[cid] = constraint
        constraints.append(constraint)
    return constraints, lookup


def build_scenario(prompt: str, constraints: Sequence[Mapping[str, Any]],
                   scenario_id: str = "demo-0001") -> Dict[str, Any]:
    """A corpus-shaped scenario row for the grader's prompt.

    Mirrors data/scenarios/corpus.jsonl (scenario_id / task / constraints / k /
    pressure) so the renderer and the checkers see exactly the shape they see in
    a real eval row. ``split`` is deliberately absent: it is a property of a
    corpus row's provenance, and an ad-hoc prompt belongs to neither split --
    inventing one would put a fake provenance in front of a grader.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise DemoError("--prompt must be a non-empty task for the model to do")
    ids = [c["id"] for c in constraints]
    return {"scenario_id": scenario_id, "task": prompt.strip(),
            "constraints": ids, "k": len(ids), "pressure": "clean"}


# --------------------------------------------------------------------------
# Model specs
# --------------------------------------------------------------------------

def normalize_model_spec(value: Any, *, field_name: str) -> str:
    """configs/models.json value -> a spec ``eval.make_generator`` accepts.

    The file is ASYMMETRIC and that asymmetry is load-bearing elsewhere:
    ``base`` is stored WITH its provider ("hf:Qwen/Qwen3-1.7B@<sha>") because
    eval.py consumes it directly, while ``tuned`` is stored as a bare HF
    "<owner>/<repo>@<sha>" because scripts/check_mvp.py's TUNED_RE validates that
    exact shape and re-adds the prefix itself (``f"hf:{cfg['tuned']}"``).
    Hard-coding either form here would break the moment the other is edited.
    """
    if not isinstance(value, str) or not value.strip():
        raise DemoError(f"configs/models.json[{field_name!r}] must be a non-empty "
                        f"string, got {value!r}")
    spec = value.strip()
    head = spec.split(":", 1)[0]
    if ":" in spec and "/" not in head:
        if head not in KNOWN_PROVIDERS:
            raise DemoError(f"configs/models.json[{field_name!r}] names provider "
                            f"{head!r}; eval.make_generator knows {list(KNOWN_PROVIDERS)}")
        return spec
    return f"hf:{spec}"


def load_model_specs(path: Optional[Path] = None) -> Dict[str, str]:
    """-> ``{"base": "hf:...", "tuned": "hf:..."}`` from configs/models.json."""
    p = Path(path) if path is not None else DEFAULT_MODELS_JSON
    if not p.exists():
        raise DemoError(f"{p} does not exist; the demo will not guess a model id")
    try:
        cfg = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DemoError(f"{p}: invalid JSON: {exc}") from None
    if not isinstance(cfg, dict):
        raise DemoError(f"{p} is not a JSON object")
    specs = {}
    for role in ROLES:
        if role not in cfg:
            raise DemoError(f"{p} has no {role!r} key; the demo compares exactly "
                            f"the two models the MVP package pins")
        specs[role] = normalize_model_spec(cfg[role], field_name=role)
    if specs["base"] == specs["tuned"]:
        raise DemoError(f"{p} pins the same model for 'base' and 'tuned'; there is "
                        f"no comparison to show")
    return specs


# --------------------------------------------------------------------------
# Generation: ONE model resident at a time
# --------------------------------------------------------------------------

def _apply_token_budget(gen: Any, max_new_tokens: int) -> bool:
    """Make an HFGenerator honour --max-new-tokens without forking its call.

    eval.HFGenerator.__call__ hard-codes ``max_new_tokens=512``. Copying its body
    here to change one number would fork the chat-template call -- the exact
    thing this demo must not do -- so the budget is overridden one level down, on
    ``model.generate``, leaving the tokenizer/template/decode path untouched.
    Returns True if the override was installed (or was unnecessary).
    """
    if max_new_tokens == EVAL_HF_MAX_NEW_TOKENS:
        return True
    model = getattr(gen, "model", None)
    inner = getattr(model, "generate", None)
    if inner is None:
        return False

    def _generate(*args: Any, _inner: Any = inner, **kwargs: Any) -> Any:
        kwargs["max_new_tokens"] = max_new_tokens
        return _inner(*args, **kwargs)

    model.generate = _generate
    return True


def release_generator(gen: Any) -> None:
    """Drop a generator's weights NOW, before the next one is built.

    See the module docstring: 19.7GB measured peak for one 1.7B model on MPS,
    and MPS keeps per-generation allocations until something empties the cache.
    The caller must also drop its own reference; this frees the large attributes
    so that reference costs nothing.
    """
    torch = getattr(gen, "torch", None)
    for attr in ("model", "tok"):
        if hasattr(gen, attr):
            try:
                delattr(gen, attr)
            except Exception:  # noqa: BLE001 - best effort teardown, never fatal
                setattr(gen, attr, None)
    del gen
    gc.collect()
    if torch is None:
        return
    mps = getattr(torch, "mps", None)
    if mps is not None and hasattr(mps, "empty_cache"):
        try:
            mps.empty_cache()
        except Exception:  # noqa: BLE001
            pass
    cuda = getattr(torch, "cuda", None)
    if cuda is not None:
        try:
            if cuda.is_available():
                cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


def generate(spec: str, scenario: Mapping[str, Any],
             constraint_lookup: Mapping[str, Mapping[str, Any]],
             max_new_tokens: int = EVAL_HF_MAX_NEW_TOKENS,
             generator_factory: Any = None) -> str:
    """Build the generator, run ONE generation, free the weights. Always frees."""
    factory = generator_factory or eval_mod.make_generator
    gen = factory(spec)
    try:
        _apply_token_budget(gen, max_new_tokens)
        # eval.build_prompt is what the eval calls; it renders the user message
        # through scenarios.render.render_user_message and adds no system turn
        # under zero_shot.
        prompt = eval_mod.build_prompt(STRATEGY, scenario, constraint_lookup, None)
        rendered = render_user_message(scenario, constraint_lookup)
        if prompt["turns"][-1]["content"] != rendered:
            raise DemoError(
                "the prompt shown to the grader is not the prompt sent to the model; "
                "eval.build_prompt and scenarios.render.render_user_message disagree")
        output = gen(scenario, prompt, constraint_lookup)
    finally:
        release_generator(gen)
        del gen
        gc.collect()
    if output is None:
        raise DemoError(f"{spec} returned None rather than text")
    return output


# --------------------------------------------------------------------------
# Checking
# --------------------------------------------------------------------------

def run_checks(constraints: Sequence[Mapping[str, Any]], output: str) -> List[Dict[str, Any]]:
    """Every constraint against one output, through ``constraints.checkers.check``.

    Not harness.run_checkers: that aborts the whole batch on the first raising
    checker, and this report has to show the other verdicts plus the failure. A
    raising checker becomes status ERROR -- NEVER a pass. A constraint whose
    verdict is unknown is not a constraint that held.
    """
    results: List[Dict[str, Any]] = []
    for constraint in constraints:
        row: Dict[str, Any] = {
            "constraint_id": constraint.get("id"),
            "type": constraint.get("type"),
            "requirement": _requirement_text(constraint),
        }
        try:
            verdict = check_constraint(dict(constraint), output)
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            row["status"] = "ERROR"
            row["detail"] = f"{type(exc).__name__}: {exc}"
        else:
            row["status"] = "PASS" if verdict["pass"] else "FAIL"
            row["detail"] = str(verdict["evidence"])
        results.append(row)
    return results


def _requirement_text(constraint: Mapping[str, Any]) -> str:
    """The requirement sentence the model was actually given, or a fallback."""
    try:
        return render_constraint(constraint)
    except Exception:  # noqa: BLE001
        return f"{constraint.get('type')} {constraint.get('params')}"


@dataclass
class ModelResult:
    """One model's side of the comparison."""
    role: str
    spec: str
    output: Optional[str] = None
    error: Optional[str] = None
    checks: List[Dict[str, Any]] = field(default_factory=list)
    elapsed_s: Optional[float] = None

    @property
    def ran(self) -> bool:
        return self.error is None and self.output is not None

    @property
    def n_failed(self) -> int:
        return sum(1 for c in self.checks if c["status"] == "FAIL")

    @property
    def n_errored(self) -> int:
        return sum(1 for c in self.checks if c["status"] == "ERROR")

    @property
    def all_held(self) -> bool:
        """True only if every checker RETURNED and returned a pass."""
        return bool(self.checks) and all(c["status"] == "PASS" for c in self.checks)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

WIDTH = 78
RULE = "=" * WIDTH
THIN = "-" * WIDTH
_STATUS_WIDTH = 6


def _truncate(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _verdict_line(result: ModelResult) -> str:
    if not result.ran:
        return f"VERDICT {result.role}: NOT RUN -- {result.error}"
    n = len(result.checks)
    if result.n_errored:
        return (f"VERDICT {result.role}: VERDICT WITHHELD -- {result.n_errored} of {n} "
                f"checkers raised. An unknown verdict is not a pass.")
    if not (result.output or "").strip():
        return (f"VERDICT {result.role}: EMPTY OUTPUT. Every checker passes an empty "
                f"string vacuously (checkers ruling A7); this is NOT compliance.")
    if result.all_held:
        return f"VERDICT {result.role}: all {n} constraint(s) HELD"
    return f"VERDICT {result.role}: {result.n_failed} of {n} constraint(s) VIOLATED"


def format_footer() -> str:
    """The honesty footer. Printed on every run, win or lose."""
    base_sa, tuned_sa = MVP_SPEC_ADHERENCE
    lines = [
        "HOW TO READ THIS -- please read before drawing a conclusion",
        "",
        "This is ONE prompt, scored by the deterministic checkers ONLY. No judge ran, so",
        "this is not spec_pass (= every checker passes AND judged task quality >= 3).",
        "One prompt is an anecdote; it is not the measurement.",
        "",
        f"The measurement is {MVP_N} scenarios in runs/mvp/, written up in docs/MVP_REPORT.md.",
        "Its headline is that fine-tuning made the headline metric WORSE:",
        f"    spec_adherence          base {base_sa:.4f}   ->   tuned {tuned_sa:.4f}   (worse)",
        "",
        "Per constraint type it splits sharply -- share of individual checks that HELD:",
    ]
    for ctype in registry.list_types():
        base_pct, tuned_pct = MVP_BY_TYPE[ctype]
        if tuned_pct > base_pct + 0.05:
            direction = "better"
        elif tuned_pct < base_pct - 0.05:
            direction = "WORSE"
        else:
            direction = "flat"
        lines.append(f"    {ctype:<24} {registry.split_of(ctype):<13} "
                     f"{base_pct:5.1f}% -> {tuned_pct:5.1f}%  {direction}")
    q_base, q_tuned = MVP_QUALITY_GE3
    lines += [
        "",
        f"Task quality judged >= 3 rose {q_base:.1f}% -> {q_tuned:.1f}% of {MVP_N} rows.",
        "So the tuned model improved on the TRAINED types and on task quality, and",
        "collapsed on the two HELD-OUT types. Whichever way this one prompt lands, it",
        "does not settle that. Read docs/MVP_REPORT.md before quoting either side.",
    ]
    return "\n".join(lines)


def _format_scoreboard(constraints: Sequence[Mapping[str, Any]],
                       results: Sequence[ModelResult]) -> List[str]:
    label_w = WIDTH - (len(results) * (_STATUS_WIDTH + 2)) - 2
    header = "CONSTRAINT".ljust(label_w)
    for r in results:
        header += "  " + r.role.upper().ljust(_STATUS_WIDTH)
    lines = [header, THIN]
    by_role = [{c["constraint_id"]: c["status"] for c in r.checks} for r in results]
    for constraint in constraints:
        cid = constraint["id"]
        label = _truncate(f"{constraint['type']}: {_requirement_text(constraint)}", label_w)
        row = label.ljust(label_w)
        for statuses, r in zip(by_role, results):
            row += "  " + (statuses.get(cid, "-" if r.ran else "n/a")).ljust(_STATUS_WIDTH)
        lines.append(row)
    lines.append(THIN)
    summary = "ALL CONSTRAINTS HELD".ljust(label_w)
    for r in results:
        if not r.ran:
            mark = "n/a"
        elif r.n_errored:
            mark = "?"
        else:
            mark = "yes" if r.all_held else "NO"
        summary += "  " + mark.ljust(_STATUS_WIDTH)
    lines.append(summary)
    # rstrip: the report is meant to be piped to a file and diffed, and trailing
    # padding from the last column is noise in every one of those uses.
    return [line.rstrip() for line in lines]


def format_report(rendered_prompt: str, constraints: Sequence[Mapping[str, Any]],
                  results: Sequence[ModelResult], *,
                  max_new_tokens: int = EVAL_HF_MAX_NEW_TOKENS,
                  note: Optional[str] = None) -> str:
    """The whole terminal report. Pure: no models, no files, no clock."""
    out: List[str] = [RULE, "FORBIDDEN-CONSTRAINT HOLDER -- LIVE DEMO", RULE]
    for r in results:
        out.append(f"{r.role:<6} {r.spec}")
    out.append(f"strategy {STRATEGY}   max_new_tokens {max_new_tokens}   "
               f"judge NOT RUN (checkers only)")
    if len(constraints) < 2 or len(constraints) > 4:
        out.append(f"note: this run uses {len(constraints)} constraint(s); the graded "
                   f"spec uses 2-4 per scenario.")
    if note:
        out.append(f"note: {note}")

    out += ["", THIN,
            "PROMPT -- rendered by scenarios.render.render_user_message, the one",
            "renderer shared by the eval and the training data. Both models get this",
            "byte for byte, with no system message (the zero_shot training condition).",
            THIN, rendered_prompt, THIN, ""]

    for r in results:
        out += [RULE, f"{r.role.upper()}  {r.spec}", RULE]
        if not r.ran:
            out += [f"GENERATION FAILED: {r.error}", "", _verdict_line(r), ""]
            continue
        took = f", {r.elapsed_s:.1f}s" if r.elapsed_s is not None else ""
        # Printed VERBATIM and never re-wrapped: no_lists and
        # sentence_initial_letter are line-sensitive, so re-flowing the text
        # could visually manufacture or hide the very thing being checked.
        out += [f"--- output ({len(r.output)} chars{took}, verbatim) ---",
                r.output if r.output.strip() else "(empty output)",
                "--- checks (constraints.checkers.check -- the pinned checkers) ---"]
        for i, c in enumerate(r.checks, start=1):
            out.append(f"  [{i}] {c['status']:<5} {c['type']:<24} {c['constraint_id']}")
            out.append(f"      {c['requirement']}")
            if c["status"] == "PASS":
                out.append(f"      evidence: {c['detail']}")
            elif c["status"] == "FAIL":
                out.append(f"      VIOLATION: {c['detail']}")
            else:
                out.append(f"      CHECKER ERROR (not a pass): {c['detail']}")
        out += ["", _verdict_line(r), ""]

    out += [RULE, "SIDE BY SIDE", RULE]
    out += _format_scoreboard(constraints, results)
    out += ["", RULE, format_footer(), RULE]
    return "\n".join(out)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

class _CollectConstraint(argparse.Action):
    """Keep --constraint and --constraint-id in the order the grader typed them.

    Two separate append lists would lose the interleaving, and the renderer
    states requirements in scenario order -- so the prompt would not match what
    was asked for.
    """

    def __call__(self, parser, namespace, values, option_string=None):  # noqa: D102
        items = list(getattr(namespace, "constraint_specs", None) or [])
        items.append(("id" if option_string == "--constraint-id" else "spec", values))
        setattr(namespace, "constraint_specs", items)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="scripts/demo.py",
        description="Run one grader-supplied prompt through the base and tuned models "
                    "and print both answers with the pinned checkers' verdicts.",
        epilog=spec_syntax_help(),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", help="the task, in your own words (required unless --interactive)")
    ap.add_argument("--constraint", dest="constraint_specs", action=_CollectConstraint,
                    default=None, metavar="SPEC",
                    help="one forbidden constraint; repeatable. See the forms below.")
    ap.add_argument("--constraint-id", dest="constraint_specs", action=_CollectConstraint,
                    default=None, metavar="ID",
                    help="reuse a shipped constraint instance, e.g. c-lex-001; repeatable")
    ap.add_argument("--interactive", action="store_true",
                    help="type the task and the constraints on stdin")
    ap.add_argument("--models-json", default=str(DEFAULT_MODELS_JSON),
                    help="default: configs/models.json")
    ap.add_argument("--max-new-tokens", type=int, default=EVAL_HF_MAX_NEW_TOKENS,
                    help=f"default {EVAL_HF_MAX_NEW_TOKENS}, the evaluated budget")
    side = ap.add_mutually_exclusive_group()
    side.add_argument("--base-only", action="store_true", help="run only the base model")
    side.add_argument("--tuned-only", action="store_true", help="run only the tuned model")
    return ap


def _read_interactive(stdin=None, stdout=None) -> Tuple[str, List[Tuple[str, str]]]:
    """Ask for the task and the constraints. Returns (prompt, items)."""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    print("Task -- what should the model write?", file=stdout)
    prompt = (stdin.readline() or "").strip()
    print("", file=stdout)
    print(spec_syntax_help(), file=stdout)
    print("One per line; blank line when done.", file=stdout)
    items: List[Tuple[str, str]] = []
    while True:
        line = stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            break
        items.append(("id" if registry.ID_RE.match(line) else "spec", line))
    return prompt, items


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    items: List[Tuple[str, str]] = list(args.constraint_specs or [])
    prompt = args.prompt
    if args.interactive:
        typed_prompt, typed_items = _read_interactive()
        prompt = prompt or typed_prompt
        items += typed_items
    elif not prompt:
        parser.error("--prompt is required unless --interactive is given")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be >= 1")

    try:
        instances = load_instances()
        constraints, lookup = resolve_constraints(items, instances)
        scenario = build_scenario(prompt, constraints)
        specs = load_model_specs(Path(args.models_json))
        # Rendered BEFORE any model is loaded: a prompt that cannot be rendered
        # is a usage error, and finding that out after a 3.4GB load wastes the
        # grader's time.
        rendered_prompt = render_user_message(scenario, lookup)
    except (DemoError, ConstraintError, RenderError, harness.HarnessError,
            FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    roles = [r for r in ROLES
             if not (args.base_only and r != "base") and not (args.tuned_only and r != "tuned")]

    note = None
    if args.max_new_tokens != EVAL_HF_MAX_NEW_TOKENS:
        note = (f"--max-new-tokens {args.max_new_tokens} differs from the evaluated "
                f"budget of {EVAL_HF_MAX_NEW_TOKENS}; a truncated answer can fail a "
                f"constraint it would otherwise have held.")

    results: List[ModelResult] = []
    for role in roles:
        spec = specs[role]
        # Progress goes to stderr so a piped report stays clean.
        print(f"[demo] loading {role} ({spec}) -- one model at a time, by design",
              file=sys.stderr, flush=True)
        started = time.time()
        result = ModelResult(role=role, spec=spec)
        try:
            result.output = generate(spec, scenario, lookup, args.max_new_tokens)
        except Exception as exc:  # noqa: BLE001 - reported in the table, never hidden
            result.error = f"{type(exc).__name__}: {exc}"
        result.elapsed_s = time.time() - started
        if result.output is not None:
            result.checks = run_checks(constraints, result.output)
        results.append(result)
        print(f"[demo] {role} done in {result.elapsed_s:.1f}s", file=sys.stderr, flush=True)

    print(format_report(rendered_prompt, constraints, results,
                        max_new_tokens=args.max_new_tokens, note=note))

    failed = any(not r.ran for r in results) or any(r.n_errored for r in results)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
