"""T-6 -- proof that scripts/demo.py shows a grader the EVALUATED system.

The demo's whole value is that it reuses the graded machinery: if it rendered
its own prompt, or ran its own checkers, a grader watching it would be watching
a different system that happens to share a repo with the measured one. So the
tests below are mostly BINDING tests -- they assert identity with eval.py,
scenarios/render.py and constraints/ rather than re-asserting behaviour those
modules already own.

The GPU path cannot run here (no model download, no MPS), so generation is
faked -- but the fake is used to prove the two things the real path must do:
the prompt handed to the generator is byte-identical to the rendered one, and
the two models are NEVER resident at the same time (eval.HFGenerator's docstring
records a measured 19.7GB peak for one 1.7B model on this 34GB machine).
"""
import inspect
import io
import json
import re

import pytest
from conftest import REPO_ROOT

import eval as eval_mod
import scripts.demo as D
import scenarios.render as render_mod
from constraints import registry
from constraints.registry import ConstraintError

BASE_SPEC = "hf:Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
TUNED_BARE = "HankH18/qwen3-1.7b-forbidden-constraint-holder@b97eb115de1f1837db60550361669950d158e95e"


# ==========================================================================
# --constraint parsing: the valid forms
# ==========================================================================

@pytest.mark.parametrize("spec,expected", [
    ("banned_words:however,moreover", ("banned_words", {"words": ["however", "moreover"]})),
    ("banned_words:coffee", ("banned_words", {"words": ["coffee"]})),
    ("banned_words: however , moreover ", ("banned_words", {"words": ["however", "moreover"]})),
    ("banned_phrases:in conclusion,to sum up",
     ("banned_phrases", {"phrases": ["in conclusion", "to sum up"]})),
    ("banned_letter:e", ("banned_letter", {"letter": "e"})),
    ("sentence_initial_letter:s", ("sentence_initial_letter", {"letter": "s"})),
    # a comma is itself a legal mark, so marks are read CHARACTER-wise
    ("banned_punctuation:,;", ("banned_punctuation", {"marks": [",", ";"]})),
    ("banned_punctuation:,", ("banned_punctuation", {"marks": [","]})),
    ("banned_punctuation: , ; ", ("banned_punctuation", {"marks": [",", ";"]})),
    ("no_lists", ("no_lists", {})),
    ("no_quotation_marks", ("no_quotation_marks", {})),
    ("  no_lists  ", ("no_lists", {})),
])
def test_parse_constraint_spec_valid(spec, expected):
    assert D.parse_constraint_spec(spec) == expected


def test_every_registered_type_is_reachable_from_the_command_line():
    """A pinned type the grader cannot type is a type the demo cannot show."""
    for ctype in registry.list_types():
        assert ctype in D._SPEC_STYLE
        assert ctype in D._SPEC_USAGE
        assert ctype in D.spec_syntax_help()
    # including BOTH held-out types, which are the interesting half of the story
    heldout = registry.types_by_split("heldout_type")
    assert set(heldout) == {"banned_punctuation", "sentence_initial_letter"}
    for ctype in heldout:
        assert ctype in D.spec_syntax_help()


def test_parsed_specs_survive_the_registry_and_the_renderer():
    """Whatever the grader types must be checkable AND renderable."""
    for usage in ("banned_words:coffee,tea", "banned_phrases:in conclusion",
                  "banned_letter:e", "banned_punctuation:,;",
                  "sentence_initial_letter:t", "no_lists", "no_quotation_marks"):
        ctype, params = D.parse_constraint_spec(usage)
        constraint = registry.build_constraint(ctype, params, 900)
        registry.validate_constraint(constraint)          # raises if malformed
        assert render_mod.render_constraint(constraint)   # raises if unrenderable


# ==========================================================================
# --constraint parsing: the malformed forms
# ==========================================================================

def test_unknown_type_is_rejected_and_lists_every_valid_type():
    with pytest.raises(D.DemoError) as exc:
        D.parse_constraint_spec("banned_wordz:however")
    msg = str(exc.value)
    assert "banned_wordz" in msg
    for ctype in registry.list_types():
        assert ctype in msg, f"error message hides valid type {ctype}"


def test_missing_required_param_is_rejected():
    for spec in ("banned_words", "banned_letter", "banned_punctuation",
                 "sentence_initial_letter", "banned_phrases"):
        with pytest.raises(D.DemoError) as exc:
            D.parse_constraint_spec(spec)
        assert "requires a value" in str(exc.value)


@pytest.mark.parametrize("spec,needle", [
    ("banned_words:", "no words found"),
    ("banned_words:  ,  ", "no words found"),
    ("banned_letter:", "requires a value"),
    ("banned_letter:ee", "exactly one character"),
    ("banned_letter:4", "alphabetic"),
    ("banned_punctuation: ", "no marks found"),
    ("banned_punctuation:a", "not punctuation"),
    ("no_lists:yes", "takes no value"),
    ("no_quotation_marks:", "takes no value"),
    ("", "non-empty spec"),
    ("   ", "non-empty spec"),
])
def test_malformed_specs_raise_demo_error(spec, needle):
    with pytest.raises(D.DemoError) as exc:
        D.parse_constraint_spec(spec)
    assert needle in str(exc.value)


def test_multi_word_banned_word_is_refused_by_the_registry_not_silently_accepted():
    """A whitespace 'word' can never match under whole-word matching (ruling A1),
    so it would be an unenforceable constraint printed in the prompt."""
    with pytest.raises(D.DemoError) as exc:
        D.parse_constraint_spec("banned_words:in conclusion")
    assert "banned_phrases" in str(exc.value)


def test_spec_style_param_keys_match_the_registry_schema():
    """The demo must not invent a param name the checkers do not read."""
    for ctype, (key, _style) in D._SPEC_STYLE.items():
        expected = sorted(registry.TYPES[ctype].example_params)
        assert sorted([key] if key else []) == expected


# ==========================================================================
# Constraint resolution: typed specs and corpus ids
# ==========================================================================

@pytest.fixture(scope="module")
def instances():
    return D.load_instances()


def test_shipped_instances_load_and_cover_the_examples(instances):
    assert len(instances) >= 60
    assert "c-lex-001" in instances
    assert instances["c-lex-001"]["type"] == "banned_words"


def test_resolve_constraints_preserves_typed_order(instances):
    items = [("spec", "banned_letter:e"), ("id", "c-lex-001"), ("spec", "no_lists")]
    constraints, lookup = D.resolve_constraints(items, instances)
    assert [c["type"] for c in constraints] == ["banned_letter", "banned_words", "no_lists"]
    assert constraints[1]["id"] == "c-lex-001"
    assert set(lookup) == {c["id"] for c in constraints}


def test_minted_ids_never_collide_with_the_shipped_library():
    """The demo mints ids above the library, but checks rather than assumes."""
    fake_library = {f"c-lex-{i:03d}": {"id": f"c-lex-{i:03d}", "type": "no_lists",
                                       "params": {}, "family": "format_negative",
                                       "split": "train"}
                    for i in (D.DEMO_ID_INDEX_BASE, D.DEMO_ID_INDEX_BASE + 1)}
    constraints, _ = D.resolve_constraints([("spec", "banned_words:coffee")], fake_library)
    assert constraints[0]["id"] not in fake_library
    registry.validate_constraint(constraints[0])


def test_unknown_constraint_id_is_rejected(instances):
    with pytest.raises(D.DemoError) as exc:
        D.resolve_constraints([("id", "c-lex-999")], instances)
    assert "c-lex-999" in str(exc.value)


def test_duplicate_constraint_id_is_rejected(instances):
    with pytest.raises(D.DemoError):
        D.resolve_constraints([("id", "c-lex-001"), ("id", "c-lex-001")], instances)


def test_no_constraints_at_all_is_rejected(instances):
    with pytest.raises(D.DemoError) as exc:
        D.resolve_constraints([], instances)
    assert "at least one" in str(exc.value)


# ==========================================================================
# models.json resolution -- the "hf:" asymmetry
# ==========================================================================

def test_real_models_json_still_has_the_asymmetry_this_code_handles():
    """If the file is ever normalised, this test says so instead of the demo
    silently building 'hf:hf:...'."""
    cfg = json.loads((REPO_ROOT / "configs" / "models.json").read_text())
    assert cfg["base"].startswith("hf:")
    assert not cfg["tuned"].startswith("hf:")


def test_normalize_model_spec_handles_both_stored_forms():
    assert D.normalize_model_spec(BASE_SPEC, field_name="base") == BASE_SPEC
    assert D.normalize_model_spec(TUNED_BARE, field_name="tuned") == f"hf:{TUNED_BARE}"
    # already-prefixed values are never double-prefixed
    assert D.normalize_model_spec(f"hf:{TUNED_BARE}", field_name="tuned") == f"hf:{TUNED_BARE}"
    assert D.normalize_model_spec("mock:compliant", field_name="base") == "mock:compliant"


@pytest.mark.parametrize("value", ["", "   ", None, 7, {}])
def test_normalize_model_spec_rejects_junk(value):
    with pytest.raises(D.DemoError):
        D.normalize_model_spec(value, field_name="base")


def test_normalize_model_spec_rejects_an_unknown_provider():
    with pytest.raises(D.DemoError) as exc:
        D.normalize_model_spec("llamacpp:some-model", field_name="base")
    assert "llamacpp" in str(exc.value)


def test_load_model_specs_reads_the_real_config():
    specs = D.load_model_specs()
    assert specs["base"].startswith("hf:")
    assert specs["tuned"].startswith("hf:")
    assert specs["base"] != specs["tuned"]
    # the demo must be able to hand both straight to eval.make_generator
    for spec in specs.values():
        provider, _, rest = spec.partition(":")
        assert provider in D.KNOWN_PROVIDERS and rest


def test_load_model_specs_rejects_a_config_missing_a_role(tmp_path):
    p = tmp_path / "models.json"
    p.write_text(json.dumps({"base": BASE_SPEC}))
    with pytest.raises(D.DemoError) as exc:
        D.load_model_specs(p)
    assert "tuned" in str(exc.value)


def test_load_model_specs_rejects_identical_roles(tmp_path):
    p = tmp_path / "models.json"
    p.write_text(json.dumps({"base": BASE_SPEC, "tuned": BASE_SPEC.removeprefix("hf:")}))
    with pytest.raises(D.DemoError) as exc:
        D.load_model_specs(p)
    assert "no comparison" in str(exc.value)


def test_load_model_specs_rejects_missing_or_broken_file(tmp_path):
    with pytest.raises(D.DemoError):
        D.load_model_specs(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(D.DemoError):
        D.load_model_specs(bad)


# ==========================================================================
# Scenario construction + the ONE renderer
# ==========================================================================

def _demo_setup(items=(("spec", "banned_words:however,moreover"),)):
    instances = D.load_instances()
    constraints, lookup = D.resolve_constraints(list(items), instances)
    scenario = D.build_scenario("Write a short paragraph about a coastal town.", constraints)
    return constraints, lookup, scenario


def test_build_scenario_mirrors_a_corpus_row():
    constraints, _lookup, scenario = _demo_setup(
        (("spec", "banned_words:however"), ("spec", "no_lists")))
    corpus_row = json.loads((REPO_ROOT / "data" / "scenarios" / "corpus.jsonl")
                            .read_text().splitlines()[0])
    assert set(scenario) <= set(corpus_row), "demo scenario invents fields a corpus row lacks"
    for key in ("scenario_id", "task", "constraints", "k", "pressure"):
        assert key in scenario
    assert scenario["constraints"] == [c["id"] for c in constraints]
    assert scenario["k"] == 2
    # 'split' is provenance of a corpus row; a grader's prompt has none
    assert "split" not in scenario


def test_build_scenario_rejects_an_empty_task():
    for task in ("", "   "):
        with pytest.raises(D.DemoError):
            D.build_scenario(task, [])


def test_demo_uses_the_shared_renderer_object_itself():
    """Identity, not similarity: no second implementation of the prompt."""
    assert D.render_user_message is render_mod.render_user_message
    assert D.render_constraint is render_mod.render_constraint
    assert eval_mod.render_user_message is render_mod.render_user_message


def test_rendered_prompt_is_byte_identical_to_the_shared_renderer():
    _constraints, lookup, scenario = _demo_setup()
    expected = render_mod.render_user_message(scenario, lookup)
    prompt = eval_mod.build_prompt(D.STRATEGY, scenario, lookup, None)
    assert prompt["turns"][-1]["content"] == expected
    assert prompt["system"] is None, "zero_shot is the training condition: no system turn"
    assert render_mod.HEADER in expected
    assert "Do not use any of these words: however, moreover." in expected


def test_renderer_binding_is_declared_in_the_training_config():
    cfg = json.loads((REPO_ROOT / "configs" / "train.json").read_text())
    assert cfg["data"]["prompt_renderer"] == "scenarios.render.render_user_message"


# ==========================================================================
# Generation -- faked, but proving the real contract
# ==========================================================================

class _FakeMPS:
    def __init__(self):
        self.empties = 0

    def empty_cache(self):
        self.empties += 1


class _FakeTorch:
    def __init__(self):
        self.mps = _FakeMPS()
        self.cuda = None


class _FakeModel:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return "generated"


class FakeGenerator:
    """Stands in for eval.HFGenerator. Tracks residency so a test can prove the
    demo never holds base and tuned at once."""
    resident = []
    seen_prompts = []
    outputs = {}
    torches = []

    def __init__(self, spec):
        assert not FakeGenerator.resident, (
            f"{spec} was built while {FakeGenerator.resident} still held weights -- "
            f"two 1.7B models resident at once is the OOM this demo must avoid")
        self.spec = spec
        self.torch = _FakeTorch()
        self.tok = object()
        self.model = _FakeModel()
        self.thread_safe = False
        FakeGenerator.resident.append(spec)
        FakeGenerator.torches.append(self.torch)

    def __delattr__(self, name):
        object.__delattr__(self, name)
        if name == "model" and self.spec in FakeGenerator.resident:
            FakeGenerator.resident.remove(self.spec)

    def __call__(self, scenario, prompt, constraint_lookup=None):
        FakeGenerator.seen_prompts.append(prompt)
        return FakeGenerator.outputs.get(self.spec, "fake output")


@pytest.fixture(autouse=True)
def _reset_fake_generator_state():
    """Residency is class state, so it must not leak between tests -- a leaked
    entry would make the never-two-resident assertion fire in the wrong test."""
    FakeGenerator.resident = []
    FakeGenerator.seen_prompts = []
    FakeGenerator.outputs = {}
    FakeGenerator.torches = []
    yield
    FakeGenerator.resident = []


@pytest.fixture
def fake_generators(monkeypatch):
    monkeypatch.setattr(eval_mod, "make_generator", FakeGenerator)
    return FakeGenerator


def test_generate_sends_exactly_the_rendered_prompt(fake_generators):
    _constraints, lookup, scenario = _demo_setup()
    fake_generators.outputs["mock:x"] = "a quiet town by the water"
    out = D.generate("mock:x", scenario, lookup, generator_factory=fake_generators)
    assert out == "a quiet town by the water"
    prompt = fake_generators.seen_prompts[-1]
    assert prompt["turns"][-1]["content"] == render_mod.render_user_message(scenario, lookup)
    assert prompt["system"] is None


def test_generate_releases_the_model_even_when_generation_raises(fake_generators):
    _constraints, lookup, scenario = _demo_setup()

    class Boom(FakeGenerator):
        def __call__(self, *a, **kw):
            raise RuntimeError("mps out of memory")

    with pytest.raises(RuntimeError):
        D.generate("mock:boom", scenario, lookup, generator_factory=Boom)
    assert FakeGenerator.resident == [], "weights survived a failed generation"
    assert FakeGenerator.torches[-1].mps.empties >= 1


def test_release_generator_drops_weights_and_empties_the_mps_cache():
    gen = FakeGenerator("mock:release")
    torch = gen.torch
    D.release_generator(gen)
    assert not hasattr(gen, "model") or gen.model is None
    assert not hasattr(gen, "tok") or gen.tok is None
    assert torch.mps.empties == 1


def test_release_generator_survives_a_generator_with_no_torch():
    class Bare:
        pass
    D.release_generator(Bare())   # must not raise


def test_default_token_budget_leaves_the_generation_path_untouched():
    gen = FakeGenerator("mock:budget")
    assert D._apply_token_budget(gen, D.EVAL_HF_MAX_NEW_TOKENS) is True
    # nothing shadowed on the instance: model.generate is still the class's own
    assert "generate" not in vars(gen.model), "the evaluated budget must not be patched"
    gen.model.generate(max_new_tokens=512, do_sample=False)
    assert gen.model.calls == [{"max_new_tokens": 512, "do_sample": False}]


def test_custom_token_budget_overrides_only_max_new_tokens():
    gen = FakeGenerator("mock:budget2")
    assert D._apply_token_budget(gen, 64) is True
    assert "generate" in vars(gen.model), "the override is installed on the model"
    gen.model.generate(max_new_tokens=512, do_sample=False)
    assert gen.model.calls == [{"max_new_tokens": 64, "do_sample": False}]


def test_token_budget_reports_when_it_cannot_be_applied():
    class NoModel:
        pass
    assert D._apply_token_budget(NoModel(), 64) is False


# ==========================================================================
# Checking -- and the error path that must never read as a pass
# ==========================================================================

def test_run_checks_uses_the_pinned_checkers_and_reports_evidence():
    constraints, _lookup, _scenario = _demo_setup(
        (("spec", "banned_words:however"), ("spec", "no_lists")))
    rows = D.run_checks(constraints, "The tide turns early.\n- a bullet\n- another bullet")
    assert [r["status"] for r in rows] == ["PASS", "FAIL"]
    assert rows[0]["detail"] == "clean"
    assert "[[" in rows[1]["detail"], "a violation must carry the checker's own evidence"
    assert rows[1]["type"] == "no_lists"


def test_run_checks_reports_a_raising_checker_as_ERROR_never_as_PASS(monkeypatch):
    constraints, _lookup, _scenario = _demo_setup(
        (("spec", "banned_words:however"), ("spec", "no_lists")))

    def exploding_check(constraint, text):
        if constraint["type"] == "no_lists":
            raise ConstraintError("checker blew up")
        return {"pass": True, "evidence": "clean"}

    monkeypatch.setattr(D, "check_constraint", exploding_check)
    rows = D.run_checks(constraints, "some text")
    assert [r["status"] for r in rows] == ["PASS", "ERROR"]
    assert "checker blew up" in rows[1]["detail"]
    assert rows[1]["status"] != "PASS"

    result = D.ModelResult(role="tuned", spec="mock:x", output="some text", checks=rows)
    assert result.all_held is False, "an unknown verdict is not a pass"
    assert result.n_errored == 1
    report = D.format_report("prompt", constraints, [result])
    assert "CHECKER ERROR (not a pass)" in report
    assert "VERDICT WITHHELD" in report
    assert "all 2 constraint(s) HELD" not in report


def test_a_checker_error_does_not_stop_the_other_verdicts(monkeypatch):
    constraints, _l, _s = _demo_setup(
        (("spec", "banned_words:however"), ("spec", "no_lists"),
         ("spec", "no_quotation_marks")))
    calls = []

    def flaky(constraint, text):
        calls.append(constraint["type"])
        if constraint["type"] == "banned_words":
            raise RuntimeError("boom")
        return {"pass": True, "evidence": "clean"}

    monkeypatch.setattr(D, "check_constraint", flaky)
    rows = D.run_checks(constraints, "text")
    assert [r["status"] for r in rows] == ["ERROR", "PASS", "PASS"]
    assert len(calls) == 3


def test_empty_output_is_not_reported_as_compliance():
    """Ruling A7: empty text passes every checker vacuously. The demo must say so."""
    constraints, _l, _s = _demo_setup()
    rows = D.run_checks(constraints, "")
    assert all(r["status"] == "PASS" for r in rows)
    result = D.ModelResult(role="tuned", spec="mock:x", output="", checks=rows)
    report = D.format_report("prompt", constraints, [result])
    assert "EMPTY OUTPUT" in report
    assert "NOT compliance" in report


# ==========================================================================
# The report
# ==========================================================================

def _synthetic_report():
    constraints, _lookup, _scenario = _demo_setup(
        (("spec", "banned_words:however,moreover"), ("spec", "banned_letter:e")))
    base = D.ModelResult(
        role="base", spec=BASE_SPEC,
        output="However, the town is quiet.", elapsed_s=12.5,
        checks=D.run_checks(constraints, "However, the town is quiet."))
    tuned = D.ModelResult(
        role="tuned", spec=f"hf:{TUNED_BARE}",
        output="A small port town naps.", elapsed_s=11.0,
        checks=D.run_checks(constraints, "A small port town naps."))
    return constraints, base, tuned, D.format_report(
        "TASK\n\nRequirements -- ...", constraints, [base, tuned])


def test_report_shows_both_models_the_prompt_and_both_outputs():
    _c, base, tuned, report = _synthetic_report()
    assert base.spec in report and tuned.spec in report
    assert base.output in report and tuned.output in report
    assert "TASK" in report
    assert "scenarios.render.render_user_message" in report
    assert "constraints.checkers.check" in report


def test_report_makes_the_winner_obvious_at_a_glance():
    _c, base, tuned, report = _synthetic_report()
    assert base.n_failed == 2 and tuned.all_held
    assert "SIDE BY SIDE" in report
    scoreboard = report.split("SIDE BY SIDE")[1]
    assert "BASE" in scoreboard and "TUNED" in scoreboard
    summary = [ln for ln in scoreboard.splitlines() if ln.startswith("ALL CONSTRAINTS HELD")]
    assert len(summary) == 1
    assert summary[0].split()[-2:] == ["NO", "yes"]
    assert "VERDICT base: 2 of 2 constraint(s) VIOLATED" in report
    assert "VERDICT tuned: all 2 constraint(s) HELD" in report


def test_report_prints_the_violating_evidence():
    _c, base, _t, report = _synthetic_report()
    violation = [c for c in base.checks if c["status"] == "FAIL"][0]
    assert violation["detail"] in report
    assert "VIOLATION:" in report


def test_report_never_re_wraps_the_model_output():
    """no_lists and sentence_initial_letter are line-sensitive; re-flowing the
    text would show the grader something the checker never saw."""
    constraints, _l, _s = _demo_setup((("spec", "no_lists"),))
    output = "a line\n- bullet one\n- bullet two\n" + ("x" * 400)
    result = D.ModelResult(role="base", spec="mock:x", output=output,
                           checks=D.run_checks(constraints, output))
    report = D.format_report("prompt", constraints, [result])
    assert output in report


def test_report_handles_a_model_that_failed_to_generate():
    constraints, _l, _s = _demo_setup()
    dead = D.ModelResult(role="tuned", spec="hf:x", error="RuntimeError: MPS OOM")
    alive = D.ModelResult(role="base", spec="hf:y", output="fine",
                          checks=D.run_checks(constraints, "fine"))
    report = D.format_report("prompt", constraints, [alive, dead])
    assert "GENERATION FAILED: RuntimeError: MPS OOM" in report
    assert "NOT RUN" in report
    assert "n/a" in report.split("SIDE BY SIDE")[1]


def test_report_runs_with_a_single_model_side():
    constraints, _l, _s = _demo_setup()
    only = D.ModelResult(role="tuned", spec="hf:x", output="ok",
                         checks=D.run_checks(constraints, "ok"))
    report = D.format_report("prompt", constraints, [only])
    assert "TUNED" in report
    assert "BASE" not in report.split("SIDE BY SIDE")[1]


def test_report_lines_carry_no_trailing_whitespace():
    _c, _b, _t, report = _synthetic_report()
    offenders = [ln for ln in report.splitlines() if ln != ln.rstrip()]
    assert offenders == []


# ==========================================================================
# Honesty footer
# ==========================================================================

def test_footer_states_the_negative_headline_result():
    footer = D.format_footer()
    assert "WORSE" in footer
    assert "0.0890" in footer and "0.0816" in footer
    assert "docs/FINAL_REPORT.md" in footer
    assert "anecdote" in footer
    lowered = footer.lower()
    for claim in ("tuned wins", "the tuned model is better", "proves the tuned"):
        assert claim not in lowered


def test_footer_names_every_constraint_type_with_its_split():
    footer = D.format_footer()
    for ctype in registry.list_types():
        assert ctype in footer
    assert "heldout_type" in footer
    # the two collapses the write-up turns on
    assert "78.0% ->  41.5%" in footer
    assert "72.0% ->   4.0%" in footer


def test_footer_is_printed_on_every_report():
    _c, _b, _t, report = _synthetic_report()
    assert D.format_footer() in report


def test_footer_numbers_match_the_committed_mvp_package():
    """The footer's numbers are RE-DERIVED here from the package that holds the
    checkpoint models.json currently pins. If the package is ever re-run, or the
    pin moves to a different checkpoint, this fails rather than letting the demo
    quote figures for a model the grader will not download.

    The run dir is runs/final rather than runs/mvp because the pin moved: runs/mvp
    is the frozen MVP package and is pinned by `tuned_mvp`."""
    run_dir = REPO_ROOT / "runs" / "final"
    results = sorted(run_dir.glob("*.results.json"))
    if len(results) != 2:
        pytest.skip("runs/final is not the two-cell final package")
    cfg = json.loads((REPO_ROOT / "configs" / "models.json").read_text())
    want = {"base": cfg["base"], "tuned": f"hf:{cfg['tuned']}"}
    cells = {}
    for path in results:
        data = json.loads(path.read_text())
        for role, spec in want.items():
            if data["model"] == spec:
                cells[role] = (data, path)
    assert set(cells) == {"base", "tuned"}, "runs/final does not hold both pinned models"

    for i, role in enumerate(("base", "tuned")):
        data, path = cells[role]
        assert round(data["metrics"]["spec_adherence"], 4) == D.MVP_SPEC_ADHERENCE[i]
        types = data["metrics"]["by_constraint_type"]["types"]
        for ctype, stats in types.items():
            held = round(100.0 * (1.0 - stats["violation_rate"]), 1)
            assert held == D.MVP_BY_TYPE[ctype][i], f"{role} {ctype}"
        rows = [json.loads(ln) for ln
                in path.with_suffix("").with_suffix(".transcripts.jsonl").read_text().splitlines()
                if ln.strip()]
        assert len(rows) == D.MVP_N
        ge3 = [r for r in rows if r.get("quality_score") is not None and r["quality_score"] >= 3]
        assert round(100.0 * len(ge3) / len(rows), 1) == D.MVP_QUALITY_GE3[i]


def test_the_headline_is_actually_a_regression():
    """Guards the one claim a reader will quote."""
    base_sa, tuned_sa = D.MVP_SPEC_ADHERENCE
    assert tuned_sa < base_sa
    for ctype in registry.types_by_split("heldout_type"):
        base_pct, tuned_pct = D.MVP_BY_TYPE[ctype]
        assert tuned_pct < base_pct


# ==========================================================================
# Drift guards against eval.py
# ==========================================================================

def test_eval_hf_token_budget_constant_matches_eval_py():
    src = inspect.getsource(eval_mod.HFGenerator.__call__)
    found = re.findall(r"max_new_tokens=(\d+)", src)
    assert found == [str(D.EVAL_HF_MAX_NEW_TOKENS)], (
        "eval.HFGenerator's token budget moved; scripts/demo.py's "
        "EVAL_HF_MAX_NEW_TOKENS must move with it")


def test_known_providers_match_eval_make_generator():
    src = inspect.getsource(eval_mod.make_generator)
    providers = set(re.findall(r'provider == "(\w+)"', src))
    assert providers == set(D.KNOWN_PROVIDERS)


def test_demo_uses_evals_own_generator_factory_and_prompt_builder():
    assert D.eval_mod is eval_mod
    assert "make_generator" in inspect.getsource(D.generate)
    assert "build_prompt" in inspect.getsource(D.generate)
    assert D.STRATEGY in eval_mod.STRATEGIES


# ==========================================================================
# CLI
# ==========================================================================

def _write_models_json(tmp_path):
    """Mirrors the real file's asymmetry on purpose."""
    p = tmp_path / "models.json"
    p.write_text(json.dumps({"base": "hf:fake/base@aaa", "tuned": "fake/tuned@bbb"}))
    return p


def test_main_runs_both_sides_in_order_and_never_holds_two_models(
        tmp_path, fake_generators, capsys):
    fake_generators.outputs = {
        "hf:fake/base@aaa": "However the harbour is calm.",
        "hf:fake/tuned@bbb": "A calm harbour, at last.",
    }
    rc = D.main(["--prompt", "Describe a harbour.",
                 "--constraint", "banned_words:however",
                 "--models-json", str(_write_models_json(tmp_path))])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.index("BASE") < out.index("TUNED")
    assert "However the harbour is calm." in out and "A calm harbour, at last." in out
    assert fake_generators.resident == []
    assert len(fake_generators.seen_prompts) == 2
    # both models saw the SAME prompt -- otherwise the comparison is not one
    assert (fake_generators.seen_prompts[0]["turns"][-1]["content"]
            == fake_generators.seen_prompts[1]["turns"][-1]["content"])
    assert "VERDICT base: 1 of 1 constraint(s) VIOLATED" in out
    assert "VERDICT tuned: all 1 constraint(s) HELD" in out


def test_main_base_only_and_tuned_only(tmp_path, fake_generators, capsys):
    models = _write_models_json(tmp_path)
    assert D.main(["--prompt", "x", "--constraint", "no_lists",
                   "--models-json", str(models), "--base-only"]) == 0
    out = capsys.readouterr().out
    assert "hf:fake/base@aaa" in out and "hf:fake/tuned@bbb" not in out

    assert D.main(["--prompt", "x", "--constraint", "no_lists",
                   "--models-json", str(models), "--tuned-only"]) == 0
    out = capsys.readouterr().out
    assert "hf:fake/tuned@bbb" in out and "hf:fake/base@aaa" not in out


def test_main_reports_a_generation_failure_and_exits_nonzero(
        tmp_path, monkeypatch, fake_generators, capsys):
    def boom(spec):
        if "tuned" in spec:
            raise RuntimeError("MPS backend out of memory")
        return FakeGenerator(spec)

    monkeypatch.setattr(eval_mod, "make_generator", boom)
    rc = D.main(["--prompt", "x", "--constraint", "no_lists",
                 "--models-json", str(_write_models_json(tmp_path))])
    out = capsys.readouterr().out
    assert rc == 1
    assert "MPS backend out of memory" in out
    assert "NOT RUN" in out


def test_main_rejects_bad_usage(tmp_path, fake_generators, capsys):
    models = str(_write_models_json(tmp_path))
    assert D.main(["--prompt", "x", "--constraint", "banned_wordz:y",
                   "--models-json", models]) == 2
    assert D.main(["--prompt", "x", "--models-json", models]) == 2   # no constraints
    assert D.main(["--prompt", "x", "--constraint-id", "c-nope-001",
                   "--models-json", models]) == 2
    with pytest.raises(SystemExit):
        D.main(["--constraint", "no_lists", "--models-json", models])  # no --prompt
    with pytest.raises(SystemExit):
        D.main(["--prompt", "x", "--constraint", "no_lists", "--base-only", "--tuned-only",
                "--models-json", models])


def test_main_keeps_constraint_flags_in_the_typed_order(tmp_path, fake_generators, capsys):
    D.main(["--prompt", "x",
            "--constraint", "banned_letter:e",
            "--constraint-id", "c-lex-001",
            "--constraint", "no_lists",
            "--models-json", str(_write_models_json(tmp_path)), "--base-only"])
    out = capsys.readouterr().out
    prompt = fake_generators.seen_prompts[0]["turns"][-1]["content"]
    assert prompt.index("letter e") < prompt.index("coffee") < prompt.index("bulleted")


def test_main_warns_when_the_token_budget_is_not_the_evaluated_one(
        tmp_path, fake_generators, capsys):
    D.main(["--prompt", "x", "--constraint", "no_lists", "--max-new-tokens", "32",
            "--models-json", str(_write_models_json(tmp_path)), "--base-only"])
    out = capsys.readouterr().out
    assert "max_new_tokens 32" in out
    assert "differs from the evaluated budget" in out


def test_main_exits_nonzero_when_a_checker_errors(
        tmp_path, monkeypatch, fake_generators, capsys):
    monkeypatch.setattr(D, "check_constraint",
                        lambda c, t: (_ for _ in ()).throw(RuntimeError("checker down")))
    rc = D.main(["--prompt", "x", "--constraint", "no_lists",
                 "--models-json", str(_write_models_json(tmp_path)), "--base-only"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "CHECKER ERROR (not a pass)" in out


def test_interactive_reads_a_task_and_both_constraint_forms():
    stdin = io.StringIO("Describe a harbour.\nbanned_words:however\nc-lex-001\n\n")
    prompt, items = D._read_interactive(stdin=stdin, stdout=io.StringIO())
    assert prompt == "Describe a harbour."
    assert items == [("spec", "banned_words:however"), ("id", "c-lex-001")]


def test_interactive_mode_end_to_end(tmp_path, monkeypatch, fake_generators, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("Describe a harbour.\nno_lists\n\n"))
    rc = D.main(["--interactive", "--base-only",
                 "--models-json", str(_write_models_json(tmp_path))])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Describe a harbour." in out
    assert "Do not use bulleted or numbered lists." in out


def test_the_help_text_documents_the_constraint_language(capsys):
    parser = D.build_parser()
    text = parser.format_help()
    for flag in ("--prompt", "--constraint", "--constraint-id", "--interactive",
                 "--models-json", "--max-new-tokens", "--base-only", "--tuned-only"):
        assert flag in text
    assert "banned_words:<word>" in text
