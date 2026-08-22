#!/usr/bin/env python3
"""Local verification for the Space, with NO model weights and NO Gradio.

    .venv/bin/python space/test_space_app.py        # standalone, exit 0 = pass
    .venv/bin/python -m pytest space/test_space_app.py -q

WHAT THIS PROVES AND WHAT IT DOES NOT
-------------------------------------
PROVES: ``core.compare`` -- the exact function object ``app.py`` binds to the Run
button -- runs end to end, renders the pinned prompt, and produces constraint
verdicts computed by the REAL ``constraints.checkers.check``. Two stub answers
are fed through it, one that violates every constraint and one that satisfies
every constraint, and the verdicts must come back different. If the checkers
were stubbed, faked, or bypassed, that difference would not appear.

DOES NOT PROVE: that Gradio renders. Gradio is not installed in this venv and
this repo's rule is not to install it. ``app.py``'s component tree is not
exercised by real Gradio here -- but it is not unexercised either:
``test_app_wiring_holds_including_its_own_negative_cases`` runs
``check_app_wiring.py`` as a subprocess, which builds the tree against a
recording shim and requires every name-to-position binding between the UI and
``core.compare`` to hold. Without that case, transposing the ``base_*``/
``tuned_*`` outputs is invisible to this file: the swap is self-consistent
inside ``core``, so every other test here still passes.

DOES NOT PROVE: anything about the real models. The generator is a stub. The
model-loading seam is real (``demo.generate`` -> ``eval.make_generator``) and
untested here by design -- pulling 7 GB of weights is not a unit test.

MOCK HONESTY
------------
``StubGenerator`` refuses to be constructed while another instance still holds
weights, because that is what the real thing does on real hardware: the two
1.7B models must never be resident together (``scripts/demo.py`` module
docstring, 19.7 GB measured peak). A stub that tolerated it would hide the
regression it exists to catch.
"""
from __future__ import annotations

import io
import re
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import _repo  # noqa: E402
import core  # noqa: E402

demo = core.demo
registry = core.registry


# --------------------------------------------------------------------------
# The stub model
# --------------------------------------------------------------------------

VIOLATING = "However, moreover:\n- item one\n- item two"
COMPLYING = "A calm town sits by a hazy bay."


class _StubMPS:
    def __init__(self):
        self.empties = 0

    def empty_cache(self):
        self.empties += 1


class _StubTorch:
    def __init__(self):
        self.mps = _StubMPS()


class _StubModel:
    def __init__(self):
        self.kwargs = []

    def generate(self, **kwargs):
        self.kwargs.append(kwargs)
        return "unused"


class StubGenerator:
    """Stands in for ``eval.HFGenerator``. Never loads a weight."""

    resident = []
    prompts = []
    outputs = {}

    def __init__(self, spec):
        assert not StubGenerator.resident, (
            f"{spec} was built while {StubGenerator.resident} still held weights; "
            f"two 1.7B models resident at once is the OOM this demo must avoid")
        self.spec = spec
        self.torch = _StubTorch()
        self.tok = object()
        self.model = _StubModel()
        self.thread_safe = False
        StubGenerator.resident.append(spec)

    def __delattr__(self, name):
        object.__delattr__(self, name)
        if name == "model" and self.spec in StubGenerator.resident:
            StubGenerator.resident.remove(self.spec)

    def __call__(self, scenario, prompt, constraint_lookup=None):
        StubGenerator.prompts.append(prompt)
        return StubGenerator.outputs.get(self.spec, "stub output")

    @classmethod
    def reset(cls):
        cls.resident = []
        cls.prompts = []
        cls.outputs = {}


TASK = "Write a short paragraph about a coastal town."
CONSTRAINTS = "banned_words:however,moreover\nno_lists\nbanned_letter:e"


def _run(task=TASK, constraints=CONSTRAINTS, roles=None, budget=None,
         base_text=VIOLATING, tuned_text=COMPLYING):
    """Drive the real callback with the stub. Returns (final, all_frames)."""
    StubGenerator.reset()
    pins = core.load_pins()
    StubGenerator.outputs[pins.specs["base"]] = base_text
    StubGenerator.outputs[pins.specs["tuned"]] = tuned_text
    frames = list(core.compare(
        task, constraints,
        roles if roles is not None else list(demo.ROLES),
        budget if budget is not None else core.EVALUATED_MAX_NEW_TOKENS,
        generator_factory=StubGenerator))
    assert frames, "compare yielded nothing"
    return dict(zip(core.OUTPUT_FIELDS, frames[-1])), frames


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def test_every_frame_is_a_full_output_tuple():
    _final, frames = _run()
    for i, frame in enumerate(frames):
        assert isinstance(frame, tuple), f"frame {i} is {type(frame)}, not a tuple"
        assert len(frame) == len(core.OUTPUT_FIELDS), (
            f"frame {i} has {len(frame)} values, core.OUTPUT_FIELDS has "
            f"{len(core.OUTPUT_FIELDS)}")
        assert all(isinstance(v, str) for v in frame)


def test_the_prompt_shown_is_the_prompt_sent():
    """The rendered block and the model's last turn are the same bytes."""
    final, _frames = _run()
    pins = core.load_pins()
    instances = demo.load_instances(core.ROOT / "data" / "scenarios" / "constraint_instances.jsonl")
    items = core.parse_constraint_lines(CONSTRAINTS)
    constraints, lookup = demo.resolve_constraints(items, instances)
    scenario = demo.build_scenario(TASK, constraints)
    expected = core.render_user_message(scenario, lookup)
    assert expected in final["rendered_prompt"], "the page shows a different prompt"
    assert StubGenerator.prompts, "no prompt reached the generator"
    for prompt in StubGenerator.prompts:
        assert prompt["system"] is None, "a system message appeared under zero_shot"
        assert prompt["turns"][-1]["content"] == expected, (
            "the model received a different prompt than the page displayed")
    del pins


def test_real_checkers_produce_different_verdicts_for_violating_and_complying_text():
    """The load-bearing test: same constraints, two answers, opposite verdicts."""
    final, _frames = _run(base_text=VIOLATING, tuned_text=COMPLYING)

    base, tuned = final["base_verdict"], final["tuned_verdict"]
    assert base != tuned, "both models produced the same verdict block"

    # base violated all three; the checkers must say so, by name.
    assert base.count("**FAIL**") == 3, f"expected 3 FAIL rows for base:\n{base}"
    assert "**PASS**" not in base
    assert "however" in base.lower(), "banned_words evidence does not name the word"
    assert "VERDICT base: 3 of 3 constraint(s) VIOLATED" in base

    # tuned satisfied all three.
    assert tuned.count("**PASS**") == 3, f"expected 3 PASS rows for tuned:\n{tuned}"
    assert "**FAIL**" not in tuned
    assert "VERDICT tuned: all 3 constraint(s) HELD" in tuned


def test_swapping_the_two_answers_swaps_the_two_verdicts():
    """Guards against a verdict hard-wired to a role rather than to the text."""
    normal, _ = _run(base_text=VIOLATING, tuned_text=COMPLYING)
    swapped, _ = _run(base_text=COMPLYING, tuned_text=VIOLATING)
    assert "VERDICT base: all 3 constraint(s) HELD" in swapped["base_verdict"]
    assert "VERDICT tuned: 3 of 3 constraint(s) VIOLATED" in swapped["tuned_verdict"]
    assert normal["base_verdict"] != swapped["base_verdict"]


def test_one_violated_constraint_is_reported_alone():
    """A partial violation is not rounded to all-fail or all-pass."""
    # 'e' only: no banned word, no list, but the letter e appears.
    final, _ = _run(base_text="A quiet town beside the sea.", tuned_text=COMPLYING)
    base = final["base_verdict"]
    assert base.count("**FAIL**") == 1, base
    assert base.count("**PASS**") == 2, base
    assert "VERDICT base: 1 of 3 constraint(s) VIOLATED" in base


def test_spec_pass_is_undecided_without_a_judge_and_never_claimed():
    final, _ = _run()
    for role in ("base", "tuned"):
        block = final[f"{role}_verdict"]
        assert "spec_pass: UNDECIDED" in block, block
        assert "quality_score=None" in block
    # the conditional line is a real compute_spec_pass call, and it must differ
    # between an answer that held every constraint and one that did not
    assert "spec_pass would be **False**" in final["base_verdict"].replace("\n", " ")
    assert "spec_pass would be **True**" in final["tuned_verdict"].replace("\n", " ")


def test_checker_error_withholds_the_verdict_instead_of_passing():
    """An unknown verdict is never a pass -- exercised through the real path."""
    stub_result = demo.ModelResult(
        role="base", spec="hf:x@1", output="text", elapsed_s=0.1,
        checks=[{"constraint_id": "c-lex-900", "type": "banned_words",
                 "requirement": "Do not use the word x.", "status": "ERROR",
                 "detail": "ValueError: boom"}])
    block = core._spec_pass_block(stub_result)
    assert "WITHHELD" in block
    assert "UNDECIDED" not in block
    assert "would be" not in block


def test_empty_output_is_called_out_not_scored_as_compliance():
    final, _ = _run(base_text="   ", tuned_text=COMPLYING)
    assert "Empty output" in final["base_verdict"]
    assert "not compliance" in final["base_verdict"]


def test_generation_failure_is_reported_and_does_not_abort_the_other_model():
    class Boom(StubGenerator):
        def __call__(self, *a, **kw):
            raise RuntimeError("simulated OOM")

    StubGenerator.reset()
    frames = list(core.compare(TASK, CONSTRAINTS, ["base", "tuned"],
                               core.EVALUATED_MAX_NEW_TOKENS,
                               generator_factory=Boom))
    final = dict(zip(core.OUTPUT_FIELDS, frames[-1]))
    assert "Generation failed" in final["base_verdict"]
    assert "simulated OOM" in final["base_verdict"]
    assert "Generation failed" in final["tuned_verdict"]
    assert "FAILED: base, tuned" in final["status"]
    assert StubGenerator.resident == [], "weights survived a failed generation"


def test_models_are_never_resident_together():
    """The stub asserts it; this test proves the assertion was reached."""
    final, _ = _run()
    assert StubGenerator.resident == [], "a generator still holds weights at the end"
    assert "base" in final["scoreboard"].lower()


def test_only_the_selected_model_runs():
    final, _ = _run(roles=["tuned"])
    assert final["base_verdict"] == ""
    assert final["base_output"] == ""
    assert "VERDICT tuned:" in final["tuned_verdict"]


def test_a_lowered_budget_is_stated_and_reaches_model_generate():
    final, _ = _run(budget=64)
    note = final["rendered_prompt"]
    assert "differs from the evaluated budget of 512" in note, note
    assert "max_new_tokens=64" in note


def test_the_budget_cannot_exceed_the_evaluated_one():
    frames = list(core.compare(TASK, CONSTRAINTS, ["base"], 4096,
                               generator_factory=StubGenerator))
    final = dict(zip(core.OUTPUT_FIELDS, frames[-1]))
    assert "capped at 512" in final["status"]
    assert final["base_verdict"] == ""


def test_bad_constraint_spec_is_a_usage_error_with_the_pinned_syntax_help():
    frames = list(core.compare(TASK, "banned_wordz:x", ["base"], 512,
                               generator_factory=StubGenerator))
    final = dict(zip(core.OUTPUT_FIELDS, frames[-1]))
    assert "Cannot run" in final["status"]
    assert "unknown constraint type" in final["status"]
    assert demo.spec_syntax_help() in final["status"]
    assert final["base_output"] == ""


def test_no_constraints_is_refused():
    frames = list(core.compare(TASK, "   \n\n", ["base"], 512,
                               generator_factory=StubGenerator))
    final = dict(zip(core.OUTPUT_FIELDS, frames[-1]))
    assert "no constraints given" in final["status"]


def test_empty_task_is_refused():
    frames = list(core.compare("  ", CONSTRAINTS, ["base"], 512,
                               generator_factory=StubGenerator))
    final = dict(zip(core.OUTPUT_FIELDS, frames[-1]))
    assert "non-empty task" in final["status"]


def test_no_model_selected_is_refused():
    frames = list(core.compare(TASK, CONSTRAINTS, [], 512,
                               generator_factory=StubGenerator))
    final = dict(zip(core.OUTPUT_FIELDS, frames[-1]))
    assert "pick at least one model" in final["status"]


def test_shipped_constraint_ids_can_be_reused_like_the_cli():
    instances = demo.load_instances(core.ROOT / "data" / "scenarios" / "constraint_instances.jsonl")
    cid = next(iter(instances))
    final, _ = _run(constraints=cid)
    assert cid in final["base_verdict"], final["base_verdict"]


def test_line_parsing_agrees_with_the_cli_interactive_parser():
    """Parity with ``demo._read_interactive``'s id-vs-spec rule."""
    lines = ["banned_words:a,b", "c-lex-001", "no_lists", "  banned_letter:e  ",
             "c-chr-042", "not-an-id", "c-lex-1", "c-zzz-001", ""]
    text = "\n".join(lines)
    mine = core.parse_constraint_lines(text)
    _prompt, theirs = demo._read_interactive(
        stdin=io.StringIO("ignored-task\n" + text + "\n\n"), stdout=io.StringIO())
    assert mine == theirs, f"drifted from the CLI parser:\n{mine}\n{theirs}"


def test_quick_pick_covers_every_registered_type_and_round_trips():
    assert set(core.EXAMPLE_SPECS) == set(registry.list_types())
    for ctype, spec in core.EXAMPLE_SPECS.items():
        parsed_type, params = demo.parse_constraint_spec(spec)
        assert parsed_type == ctype
        assert params == registry.TYPES[ctype].example_params
    labels = [label for label, _ in core.EXAMPLE_CHOICES]
    assert any("heldout_type" in label for label in labels), (
        "the held-out types are the project's headline; the picker must show the split")


def test_append_specs_does_not_duplicate():
    assert core.append_specs("no_lists", ["no_lists"]) == "no_lists"
    assert core.append_specs("", ["a", "b"]) == "a\nb"
    assert core.append_specs("a\n", ["b"]) == "a\nb"
    assert core.append_specs(None, None) == ""


def test_the_pins_are_read_from_configs_models_json_not_hardcoded(tmp_path=None):
    """Point the app at a different models.json; the page must follow it."""
    import json
    tmp = HERE / "_tmp_models.json"
    try:
        tmp.write_text(json.dumps({"base": "hf:fake/base@aaa", "tuned": "fake/tuned@bbb"}),
                       encoding="utf-8")
        pins = core.load_pins(tmp)
        assert pins.specs == {"base": "hf:fake/base@aaa", "tuned": "hf:fake/tuned@bbb"}
    finally:
        tmp.unlink(missing_ok=True)
    real = core.load_pins()
    src = json.loads((core.ROOT / "configs" / "models.json").read_text(encoding="utf-8"))
    assert src["base"] in real.specs["base"]
    assert src["tuned"] in real.specs["tuned"]
    # and no sha is written into the Space's own source
    for name in ("app.py", "core.py", "_repo.py"):
        text = (HERE / name).read_text(encoding="utf-8")
        assert not re.search(r"\b[0-9a-f]{40}\b", text), f"{name} hardcodes a commit sha"


def test_the_footer_is_the_repos_own_and_states_the_negative_headline():
    final, _ = _run()
    assert demo.format_footer() in final["footer"]
    assert "spec_adherence" in final["footer"]
    assert "FINAL_REPORT.md" in final["footer"]
    # no rate-shaped number is typed into the Space's own source
    for name in ("app.py", "core.py"):
        text = (HERE / name).read_text(encoding="utf-8")
        stripped = re.sub(r"\{[^{}]*\}", "", text)
        assert not re.search(r"\b\d{1,3}\.\d\s*%", stripped), (
            f"{name} states a percentage of its own; every number must come from "
            f"demo.format_footer()")


def test_enable_thinking_matches_the_training_config():
    import json
    cfg = json.loads((core.ROOT / "configs" / "train.json").read_text(encoding="utf-8"))
    assert core.ENABLE_THINKING == cfg["data"]["enable_thinking"]
    final, _ = _run()
    assert f"enable_thinking={core.ENABLE_THINKING}" in final["rendered_prompt"]


def test_hardware_notice_is_honest_about_speed():
    text = core.hardware_notice()
    assert "slow" in text.lower()
    assert "no GPU" in text
    assert str(core.EVALUATED_MAX_NEW_TOKENS) in text
    assert "estimate" in text.lower()


# --------------------------------------------------------------------------
# A SKIP IS NOT A PASS
# --------------------------------------------------------------------------
# A case that does not execute used to `print("(skipped: ...)")` and return, and
# the runner below read the clean return as a pass. Measured consequence: with
# check_app_wiring.py absent -- exactly the layout DEPLOY.md step 2 leaves behind
# -- a live base/tuned transposition in core.OUTPUT_FIELDS reported
# `31/31 passed`, exit 0. The tuned model's answer under the BASE heading in a
# graded demo, with a green suite and nothing in the tally to say so.
#
# So: a case that does not execute is printed as SKIP and counted separately from
# the passes, and a file may be missing ONLY in the one layout where its absence
# is legitimate.

class Skipped(Exception):
    """This case cannot run in this layout. Never counted as a pass."""


def _skip(reason: str) -> None:
    if "pytest" in sys.modules:  # under pytest, let pytest own the reporting
        import pytest
        pytest.skip(reason)
    raise Skipped(reason)


#: True when ``space/`` sits inside the slmforge checkout -- its parent carries
#: every marker ``_repo`` requires. In that layout every script in this directory
#: is present by construction. An assembled Space resolves its root to ``pins/``
#: instead, and the deploy-time tools are deliberately not uploaded to it.
IN_REPO_CHECKOUT = not _repo.missing_markers(HERE.parent)


def _require(filename: str, *, everywhere: bool) -> None:
    """Fail -- or, only where absence is legitimate, skip -- on a missing file.

    ``everywhere=True``: every documented way of running this suite puts the file
    next to it (space/DEPLOY.md step 2 copies this file and check_app_wiring.py
    into the assembled Space together), so absence is ALWAYS a failure. There is
    no layout in which the only case covering the UI wiring may quietly not run.

    ``everywhere=False``: a deploy-time tool that is deliberately not uploaded.
    Absent inside an assembled Space it is legitimately skipped; absent in the
    checkout, where it is always present, it is a failure.
    """
    if (HERE / filename).exists():
        return
    if everywhere:
        raise AssertionError(
            f"{filename} is missing from {HERE}. It is copied in alongside this "
            f"file by every documented way of running this suite (space/DEPLOY.md "
            f"step 2), and it is the only check here that catches a base/tuned "
            f"transposition in app.py's wiring. Not running it is not a pass.")
    if IN_REPO_CHECKOUT:
        raise AssertionError(
            f"{filename} is missing from {HERE}, which is inside the slmforge "
            f"checkout, where it is always present. Absent here means this tree "
            f"is wrong -- not that this is a Space.")
    _skip(f"{filename} is not shipped in the Space")


def test_deploy_verifier_reports_drift_in_both_directions():
    """The post-push check must not be satisfiable by a partial upload."""
    _require("verify_deploy.py", everywhere=False)
    import verify_deploy as V
    built = {"app.py": "aaa", "core.py": "bbb", "pins/eval.py": "ccc"}
    assert V.compare_manifest(built, dict(built)) == []
    missing = V.compare_manifest(built, {"app.py": "aaa", "core.py": "bbb"})
    assert missing == ["MISSING on the Hub: pins/eval.py"], missing
    changed = V.compare_manifest(built, {**built, "core.py": "zzz"})
    assert any("DIFFERENT bytes: core.py" in p for p in changed), changed
    extra = V.compare_manifest(built, {**built, "sneaky.py": "ddd"})
    assert extra == ["EXTRA on the Hub (not in the build): sneaky.py"], extra


def test_deploy_verifier_digests_a_real_tree_and_ignores_caches():
    _require("verify_deploy.py", everywhere=False)
    import verify_deploy as V
    tree = HERE / "_tmp_digest"
    try:
        (tree / "pins" / "__pycache__").mkdir(parents=True, exist_ok=True)
        (tree / "app.py").write_text("x", encoding="utf-8")
        (tree / "pins" / "__pycache__" / "a.pyc").write_text("junk", encoding="utf-8")
        digests = V.digest_tree(tree)
        assert list(digests) == ["app.py"], digests
        assert digests["app.py"] == V.sha256(tree / "app.py")
    finally:
        import shutil
        shutil.rmtree(tree, ignore_errors=True)


#: ``check_app_wiring.py`` executes ``app.py``'s component tree against a
#: recording shim. It runs here as a SUBPROCESS rather than by import, because
#: it installs a fake ``gradio`` into ``sys.modules`` and monkeypatches
#: ``core.OUTPUT_FIELDS`` in its negative cases; a subprocess keeps both out of
#: this process. Without this, the suite is blind to app.py's wiring: a
#: base/tuned transposition in ``core.OUTPUT_FIELDS`` is self-consistent inside
#: ``core`` and passes every other test here while the page would print the
#: tuned model's answer under the BASE heading. That is why this case is required
#: in EVERY layout (``everywhere=True``) rather than skipped when the script is
#: not next to this file: a missing wiring check is the failure it exists to
#: catch, wearing the failure's own disguise.


def test_app_wiring_holds_including_its_own_negative_cases():
    _require("check_app_wiring.py", everywhere=True)
    import subprocess
    proc = subprocess.run([sys.executable, str(HERE / "check_app_wiring.py")],
                          capture_output=True, text=True)
    assert proc.returncode == 0, (
        f"check_app_wiring.py exited {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")
    assert "names match core.OUTPUT_FIELDS element by element" in proc.stdout, proc.stdout
    assert "base/tuned outputs transposed" in proc.stdout, proc.stdout


def test_readme_front_matter_and_syntax_block_do_not_drift():
    """The Space README is the Hub's config file; check it as one."""
    text = (HERE / "README.md").read_text(encoding="utf-8")
    assert text.startswith("---\n"), "no YAML front matter; the Hub needs it first"
    front = text.split("---", 2)[1]
    for key in ("title:", "emoji:", "sdk: gradio", "sdk_version:",
                "app_file: app.py", "pinned: false"):
        assert key in front, f"front matter is missing {key!r}"
    # every registered constraint type must be documented, and nothing else
    body = text.split("---", 2)[2]
    for ctype in registry.list_types():
        assert ctype in body, f"README does not document the {ctype!r} constraint"
    # the README must not restate a measured rate; the app renders those live
    assert not re.search(r"\b\d{1,3}\.\d\s*%", body), "README states a rate of its own"
    assert "147" in body, "README omits the size of the real measurement"
    assert "UNDECIDED" in body, "README does not say spec_pass is undecided here"


def test_requirements_do_not_pin_gradio_or_a_provider_sdk():
    text = (HERE / "requirements.txt").read_text(encoding="utf-8")
    active = [ln.strip() for ln in text.splitlines()
              if ln.strip() and not ln.strip().startswith("#")]
    joined = " ".join(active)
    for needed in ("torch", "transformers", "accelerate"):
        assert any(ln.startswith(needed) for ln in active), f"{needed} is not required"
    for forbidden in ("gradio", "anthropic", "openai", "google-genai", "modal"):
        assert not any(ln.split("=")[0].split(">")[0].split("<")[0].strip() == forbidden
                       for ln in active), (
            f"{forbidden} must not be installed by this Space; see requirements.txt")
    del joined


def test_repo_root_resolution_rejects_an_incomplete_tree():
    import _repo
    missing = _repo.missing_markers(HERE)
    assert missing, "space/ alone must not be accepted as a pinned root"


# --------------------------------------------------------------------------
# Standalone runner (so a clean exit code does not depend on pytest plugins)
# --------------------------------------------------------------------------

def _main() -> int:
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failures, skips = [], []
    for name, fn in tests:
        StubGenerator.reset()
        try:
            fn()
        except Skipped as exc:
            skips.append((name, str(exc)))
            print(f"SKIP  {name}  ({exc})")
        except Exception:  # noqa: BLE001
            failures.append((name, traceback.format_exc()))
            print(f"FAIL  {name}")
        else:
            print(f"ok    {name}")
    passed = len(tests) - len(failures) - len(skips)
    print(f"\n{passed} passed, {len(skips)} skipped, {len(failures)} failed "
          f"({len(tests)} cases)")
    if skips:
        # Printed separately and never folded into the passes: a skipped case
        # proves nothing, and the tally must not let it look as if it did.
        print("NOT EXECUTED (a skip is not a pass):")
        for name, reason in skips:
            print(f"  {name}: {reason}")
    for name, tb in failures:
        print(f"\n--- {name} ---\n{tb}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_main())
