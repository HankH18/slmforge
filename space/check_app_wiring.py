#!/usr/bin/env python3
"""Execute ``app.py``'s ``build()`` with a RECORDING SHIM in place of Gradio.

    .venv/bin/python space/check_app_wiring.py      # exit 0 = wiring holds

WHY THIS EXISTS
---------------
Gradio is not installed in this repo's venv and this build does not install it,
so ``app.py`` would otherwise be an unexecuted file. This runs it: the component
tree is built, the structural guards inside ``build()`` are all evaluated, and
the Run button's binding is inspected.

WHAT IT PROVES
--------------
* ``app.py`` imports and ``build()`` runs top to bottom.
* The Run button's ``fn`` **is** ``core.compare`` -- identity, not a copy or a
  wrapper -- so what a visitor triggers is what ``test_space_app.py`` verified.
* The output components carry ``core.OUTPUT_FIELDS``' names, in that order,
  read back off the ``elem_id`` each component was constructed with -- an
  independent witness of ``build()``'s own tag list, checked NAME BY NAME and
  not merely counted. The guards that enforce that actually fire when the
  wiring is wrong, including for a transposed ``base_*``/``tuned_*`` pair, which
  a count-only check accepts (proved by the negative cases at the bottom).
* No commit sha and no measured rate is written into ``app.py``'s own text.

WHAT IT DOES **NOT** PROVE -- read this before believing the exit code
----------------------------------------------------------------------
The shim is not Gradio. It accepts every component name and every keyword
argument. So this says NOTHING about whether real Gradio accepts
``gr.Textbox(show_copy_button=True)``, whether it accepts ``elem_id=`` on every
output component (a base-class keyword, but unexercised here), whether
``gr.Dropdown`` takes
``(label, value)`` choice tuples in the pinned ``sdk_version``, whether a
generator ``fn`` streams, or whether the page renders at all. Those are
exercised the first time the Space boots, and DEPLOY.md's verification step
exists for exactly that reason.

The shim is deliberately NOT forgiving about the things it can judge: an unknown
attribute on ``gr`` raises rather than auto-vivifying, so a typo'd component
name fails here instead of at boot.
"""
from __future__ import annotations

import inspect
import re
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

#: Only the components app.py is allowed to use. An unknown name is an error,
#: not a silently created stub -- that is the one thing a shim can honestly check.
KNOWN_COMPONENTS = {
    "Blocks", "Row", "Column", "Accordion", "Markdown", "Textbox",
    "Dropdown", "Button", "CheckboxGroup", "Slider",
}
_CONTAINERS = {"Blocks", "Row", "Column", "Accordion"}

RECORD = {"components": [], "clicks": []}


class _Component:
    def __init__(self, kind, *args, **kwargs):
        self.kind = kind
        self.args = args
        self.kwargs = kwargs
        RECORD["components"].append(self)

    def __repr__(self):
        return f"<{self.kind} {self.kwargs.get('label') or ''!r}>"

    def __enter__(self):
        if self.kind not in _CONTAINERS:
            raise TypeError(f"gr.{self.kind} is not a context manager")
        return self

    def __exit__(self, *exc):
        return False

    def click(self, fn=None, inputs=None, outputs=None, **kwargs):
        if self.kind != "Button":
            raise TypeError(f"gr.{self.kind}.click does not exist")
        RECORD["clicks"].append({"button": self, "fn": fn,
                                 "inputs": list(inputs or []),
                                 "outputs": list(outputs or []), "kwargs": kwargs})
        return self

    def queue(self, **kwargs):
        return self

    def launch(self, **kwargs):  # pragma: no cover - never called by build()
        raise AssertionError("check_app_wiring must never launch a server")


class _GradioShim(types.ModuleType):
    __version__ = "shim"

    def __getattr__(self, name):
        if name not in KNOWN_COMPONENTS:
            raise AttributeError(
                f"app.py used gr.{name}, which this shim does not know about. "
                f"Either it is a typo, or KNOWN_COMPONENTS needs updating after "
                f"a deliberate change -- do not auto-create it.")
        return lambda *a, **kw: _Component(name, *a, **kw)


def _install_shim():
    sys.modules["gradio"] = _GradioShim("gradio")


def _reset():
    RECORD["components"] = []
    RECORD["clicks"] = []
    for mod in ("app",):
        sys.modules.pop(mod, None)


def check() -> None:
    _install_shim()
    _reset()
    import app  # noqa: E402
    import core  # noqa: E402

    ui = app.build()
    assert ui.kind == "Blocks", f"build() returned {ui!r}"

    kinds = [c.kind for c in RECORD["components"]]
    for required in ("Textbox", "CheckboxGroup", "Slider", "Button", "Markdown",
                     "Dropdown"):
        assert required in kinds, f"app.py never created a {required}"

    assert len(RECORD["clicks"]) == 2, (
        f"expected 2 click bindings (add-to-box, run), got {len(RECORD['clicks'])}")
    by_fn = {c["fn"]: c for c in RECORD["clicks"]}
    assert core.compare in by_fn, (
        "the Run button is NOT bound to core.compare itself; a wrapper or a copy "
        "means the verified callback is not the one visitors trigger")
    run = by_fn[core.compare]
    assert len(run["inputs"]) == 4, run["inputs"]
    # Inputs are passed positionally by Gradio, so their order is checked by name
    # too -- against core.compare's own parameter list, read from the function.
    compare_params = list(inspect.signature(core.compare).parameters)[:4]
    bound_in = [c.kwargs.get("elem_id") for c in run["inputs"]]
    assert bound_in == compare_params, (
        "the Run button's input components are not in core.compare's parameter "
        f"order:\n  bound  : {bound_in}\n  params : {compare_params}")
    assert len(run["outputs"]) == len(core.OUTPUT_FIELDS), (
        f"{len(run['outputs'])} outputs bound, core.OUTPUT_FIELDS has "
        f"{len(core.OUTPUT_FIELDS)}")
    # Names, not just count. Read off each component's elem_id -- the id real
    # Gradio renders into the DOM -- so this is a second, independent statement
    # of the wiring, not a re-read of the list build() already checked.
    wired = [c.kwargs.get("elem_id") for c in run["outputs"]]
    assert wired == list(core.OUTPUT_FIELDS), (
        "the Run button's output components do not carry core.OUTPUT_FIELDS' "
        "names in order; a value would land under the wrong heading:\n"
        f"  bound  : {wired}\n  fields : {list(core.OUTPUT_FIELDS)}")
    assert core.append_specs in by_fn, "the quick-pick button is not bound"

    # the seam a visitor must not reach
    assert not any(getattr(i, "kwargs", {}).get(k) == "generator_factory"
                   for i in run["inputs"] for k in ("label", "elem_id"))

    # the intro text really carries the pins read from configs/models.json
    pins = core.load_pins()
    intro = app.INTRO
    for role in ("base", "tuned"):
        assert pins.specs[role] in intro, f"the page does not state the {role} pin"

    # nothing hardcoded into the page's own source
    src = (HERE / "app.py").read_text(encoding="utf-8")
    assert not re.search(r"\b[0-9a-f]{40}\b", src), "app.py hardcodes a commit sha"
    assert not re.search(r"\b\d{1,3}\.\d\s*%", src), "app.py states a rate of its own"
    assert "147" in app.WHAT_IS_NOT_HERE, "the page omits the size of the real measurement"

    print("app.py wiring OK:")
    print(f"  components built ......... {len(RECORD['components'])} "
          f"({', '.join(sorted(set(kinds)))})")
    print(f"  Run button fn ............ core.compare  (identity check passed)")
    print(f"  inputs ................... {len(run['inputs'])}, names match "
          f"core.compare's parameter order")
    print(f"  outputs .................. {len(run['outputs'])}, names match "
          f"core.OUTPUT_FIELDS element by element")
    print(f"  pins rendered from ....... {pins.models_json}")
    print("")
    print("NOT PROVEN HERE: that real Gradio accepts these components/kwargs, or")
    print("that the page renders. Gradio is not installed; see this file's docstring.")


def _negative_cases() -> None:
    """Prove build()'s guards are load-bearing, by breaking the wiring."""
    import app  # noqa: E402
    import core  # noqa: E402

    cases = []

    # 1. a dropped output component
    original_fields = core.OUTPUT_FIELDS
    try:
        core.OUTPUT_FIELDS = original_fields + ("extra",)
        _reset()
        try:
            app.build()
        except RuntimeError as exc:
            cases.append(("dropped output component", str(exc)))
        else:
            raise AssertionError("build() accepted a mismatched output count")
    finally:
        core.OUTPUT_FIELDS = original_fields

    # 2. the injectable seam made positional (a visitor could then reach it)
    original = core.compare

    def _positional_seam(task, constraint_text, roles, max_new_tokens,
                         generator_factory=None, models_json=None):
        yield ()

    try:
        core.compare = _positional_seam
        _reset()
        try:
            app.build()
        except RuntimeError as exc:
            cases.append(("generator_factory made positional", str(exc)))
        else:
            raise AssertionError("build() accepted a reachable model-loading seam")
    finally:
        core.compare = original

    # 3. THE transposition: base_*/tuned_* swapped in core.OUTPUT_FIELDS. The
    #    COUNT IS UNCHANGED, so a length-only guard accepts it and the deployed
    #    page prints the tuned model's answer under the BASE heading -- a grader
    #    would read the base model apparently holding the constraint.
    try:
        pos = {name: i for i, name in enumerate(original_fields)}
        swapped = list(original_fields)
        for a, b in (("base_output", "tuned_output"),
                     ("base_verdict", "tuned_verdict")):
            swapped[pos[a]], swapped[pos[b]] = b, a
        core.OUTPUT_FIELDS = tuple(swapped)
        assert len(core.OUTPUT_FIELDS) == len(original_fields), "not a transposition"
        assert core.OUTPUT_FIELDS != original_fields, "the swap changed nothing"
        _reset()
        try:
            app.build()
        except RuntimeError as exc:
            cases.append(("base/tuned outputs transposed (same count)", str(exc)))
        else:
            raise AssertionError(
                "build() accepted the tuned model's answer wired under the BASE "
                "heading -- the guard is counting, not naming")
    finally:
        core.OUTPUT_FIELDS = original_fields

    # 4. The other half: the tag and the visible label disagree, so the value
    #    is routed correctly but a grader reads the wrong model's name above it.
    original_role = app.ROLE_BASE
    try:
        app.ROLE_BASE = "tuned"
        _reset()
        try:
            app.build()
        except RuntimeError as exc:
            cases.append(("base answer labelled TUNED", str(exc)))
        else:
            raise AssertionError("build() accepted a heading naming the wrong model")
    finally:
        app.ROLE_BASE = original_role

    print("\nbuild()'s guards fire when the wiring is broken:")
    for name, msg in cases:
        print(f"  {name}\n      -> RuntimeError: {msg.splitlines()[0]}")


if __name__ == "__main__":
    check()
    _negative_cases()
    print("\nOK")
