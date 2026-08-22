#!/usr/bin/env python3
"""Gradio layer: components, labels, wiring. No logic.

Everything this file does with the visitor's input is ``core.compare``. The
``.click()`` below binds ``fn=core.compare`` DIRECTLY -- not a lambda around it,
not a copy of it -- so the function the browser triggers is byte-identically the
function ``test_space_app.py`` drives with a stub generator and the real
checkers on a machine with no Gradio installed.

Four things are checked at build time rather than trusted:

* every output component is TAGGED, at the one place it is constructed, with
  the ``core.OUTPUT_FIELDS`` name it carries, and the tags must equal
  ``core.OUTPUT_FIELDS`` ELEMENT BY ELEMENT -- not merely in count. Transposing
  the ``base_*``/``tuned_*`` pair in either list therefore raises on startup
  instead of showing a grader the tuned model's answer in the base model's box.
  A count-only check does not catch that: a transposition preserves the count;
* a labelled output tagged for one model must name that model in its visible
  label and must not name the other, so the heading a grader reads and the
  value routed underneath it cannot disagree;
* the same rule on the way in: every input component is tagged with the
  ``core.compare`` parameter it feeds -- the names read off the function, not
  retyped -- and the ``inputs`` list must be in that parameter order. Gradio
  passes ``inputs`` positionally, so a transposed pair would send the visitor's
  constraints as the task, silently, both being Textboxes;
* the model-loading seam ``generator_factory`` is KEYWORD-ONLY on
  ``core.compare`` and appears in none of the ``inputs`` lists, so no visitor of
  the deployed Space can substitute a fake generator.

Run locally:  python app.py     (needs gradio; see requirements.txt)
"""
from __future__ import annotations

import inspect
import sys
from itertools import zip_longest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr  # noqa: E402

import core  # noqa: E402

PINS = core.load_pins()

TITLE = "Forbidden-Constraint Holder -- base vs tuned, live"

#: The two model roles, taken from the repo's own list rather than retyped. The
#: unpacking fails loudly at import if that list ever stops being a pair -- the
#: two-column layout below assumes exactly two.
ROLE_BASE, ROLE_TUNED = core.demo.ROLES

INTRO = f"""
# Forbidden-Constraint Holder: base vs tuned, on **your** prompt

Type your own task, pick or type your own forbidden constraints, and press Run.
Both models answer the same prompt and each answer is scored by this project's
**pinned programmatic checkers** -- `constraints.checkers.check`, the same entry
point every published number in the repo is re-derived from. Nothing here is
pre-selected and nothing here is cached.

**Running these exact commits, read at startup from `configs/models.json`:**

* base &nbsp;`{PINS.specs['base']}`
* tuned `{PINS.specs['tuned']}`

Strategy `{core.demo.STRATEGY}` (no system message -- the training condition),
`enable_thinking={PINS.enable_thinking}`, matching
`configs/train.json:data.enable_thinking` and `eval.HFGenerator`'s chat-template
call, so what you see is generated under the same condition the reported numbers
were measured under. Generation budget defaults to
`max_new_tokens={PINS.evaluated_max_new_tokens}` -- the evaluated budget.
"""

CONSTRAINT_HELP = f"""
One constraint per line. Same forms as `scripts/demo.py --constraint`, parsed by
the same function, validated against `constraints/registry.py`:

```text
{core.demo.spec_syntax_help()}
```
"""

WHAT_IS_NOT_HERE = f"""
### What this page does **not** tell you

* **No judge runs.** `spec_pass` is *all checkers pass **and** judged task
  quality >= {core.harness.QUALITY_THRESHOLD}*. This page computes the first
  half live and leaves the second half unmeasured, so
  `harness.compute_spec_pass(checks, None)` returns `None` -- undecided. It is
  not reported as a pass.
* **One prompt is an anecdote.** The measurement is 147 scenarios in
  `runs/final/`, written up in `docs/FINAL_REPORT.md`, and its headline is that
  fine-tuning made the headline metric **worse**. The block at the bottom of
  every run prints those numbers whichever way your prompt lands.
* **Nothing here is a second implementation.** Prompt, checkers, model pins and
  scoreboard are all imported from the repo; this page adds a text box.
"""


def build() -> "gr.Blocks":
    # ``core.compare``'s own parameter names, read from the function rather than
    # retyped, so the inputs below are checked against the callback itself.
    compare_params = tuple(inspect.signature(core.compare).parameters)

    # (component, name, visible label) in creation order. ``_out``/``_in`` below
    # are the ONLY places a component is bound to a name, and they bind it at the
    # same call that sets the label -- so the tag, the heading and the value can
    # only be separated by editing one call site, and the checks after the layout
    # compare them.
    tags = []

    def _tag(name, factory, **kwargs):
        if any(n == name for _c, n, _l in tags):
            raise RuntimeError(f"two components claim the name {name!r}")
        # elem_id is the same name again, so the rendered DOM id is an
        # independent witness of the wiring that check_app_wiring.py reads back.
        comp = factory(elem_id=name, **kwargs)
        tags.append((comp, name, kwargs.get("label")))
        return comp

    def _out(field, factory, **kwargs):
        """Create an output component carrying the field name it is wired to."""
        if field not in core.OUTPUT_FIELDS:
            raise RuntimeError(
                f"app.py tagged an output component {field!r}, which is not one "
                f"of core.OUTPUT_FIELDS {core.OUTPUT_FIELDS}")
        return _tag(field, factory, **kwargs)

    def _in(param, factory, **kwargs):
        """Create an input component carrying the parameter it feeds."""
        if param not in compare_params:
            raise RuntimeError(
                f"app.py tagged an input component {param!r}, which is not a "
                f"parameter of core.compare {compare_params}")
        return _tag(param, factory, **kwargs)

    def _name_of(comp):
        """The name a component was tagged with -- by IDENTITY, not position."""
        for c, n, _l in tags:
            if c is comp:
                return n
        return None

    with gr.Blocks(title=TITLE, fill_height=False) as ui:
        gr.Markdown(INTRO)
        gr.Markdown(core.hardware_notice())

        with gr.Row():
            with gr.Column(scale=3):
                task = _in(
                    "task", gr.Textbox,
                    label="Your task -- what should the model write?",
                    placeholder="Write a short paragraph about a coastal town.",
                    lines=3)
                constraint_text = _in(
                    "constraint_text", gr.Textbox,
                    label="Your forbidden constraints -- one per line",
                    placeholder="banned_words:however,moreover\nno_lists",
                    lines=4)
                with gr.Row():
                    picker = gr.Dropdown(
                        choices=core.EXAMPLE_CHOICES, multiselect=True, value=[],
                        label="Quick-pick an example of each registered type "
                              "(train / heldout_type is the pinned split)")
                    add_btn = gr.Button("Add to the box", scale=0)
                with gr.Accordion("Constraint syntax", open=False):
                    gr.Markdown(CONSTRAINT_HELP)
            with gr.Column(scale=1):
                roles = _in(
                    "roles", gr.CheckboxGroup,
                    choices=list(core.demo.ROLES), value=list(core.demo.ROLES),
                    label="Models to run (in this order, one resident at a time)")
                max_new_tokens = _in(
                    "max_new_tokens", gr.Slider,
                    minimum=16, maximum=core.MAX_NEW_TOKENS_CEILING, step=16,
                    value=core.EVALUATED_MAX_NEW_TOKENS,
                    label=f"max_new_tokens (evaluated budget: "
                          f"{core.EVALUATED_MAX_NEW_TOKENS})")
                run_btn = gr.Button("Run base vs tuned", variant="primary")

        status = _out("status", gr.Markdown)
        rendered_prompt = _out("rendered_prompt", gr.Markdown)

        with gr.Row():
            with gr.Column():
                base_output = _out(
                    "base_output", gr.Textbox,
                    label=f"{ROLE_BASE.upper()} -- raw output, verbatim", lines=14,
                    show_copy_button=True, interactive=False)
                base_verdict = _out("base_verdict", gr.Markdown)
            with gr.Column():
                tuned_output = _out(
                    "tuned_output", gr.Textbox,
                    label=f"{ROLE_TUNED.upper()} -- raw output, verbatim", lines=14,
                    show_copy_button=True, interactive=False)
                tuned_verdict = _out("tuned_verdict", gr.Markdown)

        scoreboard = _out("scoreboard", gr.Markdown)
        footer = _out("footer", gr.Markdown)
        gr.Markdown(WHAT_IS_NOT_HERE)

        # The output list below MUST equal core.OUTPUT_FIELDS ELEMENT BY ELEMENT.
        # Asserted by name, not counted -- see the module docstring. A count-only
        # check accepts a transposed base/tuned pair, which is the one wiring
        # error that silently prints a model's answer under the other's heading.
        outputs = [status, rendered_prompt, base_output, base_verdict,
                   tuned_output, tuned_verdict, scoreboard, footer]

        wired = tuple(_name_of(c) for c in outputs)
        if wired != tuple(core.OUTPUT_FIELDS):
            rows = []
            for i, (got, want) in enumerate(zip_longest(wired, core.OUTPUT_FIELDS,
                                                        fillvalue="<missing>")):
                rows.append(f"  {'->' if got != want else '  '} position {i}: "
                            f"UI {got!r}   core.compare yields {want!r}")
            raise RuntimeError(
                "app.py's output components do not correspond to "
                "core.OUTPUT_FIELDS position by position, so a value core.compare "
                "yields for one field would be routed into another field's "
                "component -- e.g. the tuned model's answer under the BASE "
                "heading:\n" + "\n".join(rows))

        # A labelled output tagged for one model must name that model and only
        # that model. This is the other half of the same guarantee: the tag says
        # where the value comes from, the label says what a grader reads above it.
        for _comp, field, label in tags:
            if not label or field not in core.OUTPUT_FIELDS:
                continue
            owner = field.split("_", 1)[0]
            for role in core.demo.ROLES:
                names_role = role.upper() in label.upper()
                if role == owner and not names_role:
                    raise RuntimeError(
                        f"the component wired to {field!r} is labelled {label!r}, "
                        f"which does not name the {role} model")
                if role != owner and names_role:
                    raise RuntimeError(
                        f"the component wired to {field!r} is labelled {label!r}, "
                        f"which names the {role} model; the heading above an "
                        f"answer must name the model that produced it")

        # Same rule on the way in. Gradio passes `inputs` POSITIONALLY, so a
        # transposed pair here sends the visitor's constraints as the task --
        # silently, since both are Textboxes. Checked by name, not by count.
        inputs = [task, constraint_text, roles, max_new_tokens]
        params = list(compare_params)
        expected_in = ["task", "constraint_text", "roles", "max_new_tokens"]
        if params[:len(inputs)] != expected_in:
            raise RuntimeError(f"core.compare's positional parameters {params} no "
                               f"longer match this UI's inputs")
        wired_in = [_name_of(c) for c in inputs]
        if wired_in != expected_in:
            raise RuntimeError(
                "app.py's input components are not in core.compare's parameter "
                "order, so a visitor's value would arrive as a different "
                f"argument:\n  UI      : {wired_in}\n  compare : {expected_in}")
        seam = inspect.signature(core.compare).parameters["generator_factory"]
        if seam.kind is not inspect.Parameter.KEYWORD_ONLY:
            raise RuntimeError("core.compare's generator_factory seam must stay "
                               "keyword-only so a Space visitor cannot reach it")

        add_btn.click(fn=core.append_specs, inputs=[constraint_text, picker],
                      outputs=[constraint_text])
        run_btn.click(fn=core.compare, inputs=inputs, outputs=outputs)

    return ui


if __name__ == "__main__":
    app = build()
    # One job at a time: the box has no GPU and two 1.7B generations already
    # saturate it. A second concurrent visitor would not go faster, it would make
    # both slower and risk the OOM scripts/demo.py's one-at-a-time rule avoids.
    app.queue(default_concurrency_limit=1, max_size=8)
    app.launch(server_name="0.0.0.0", show_api=False)
