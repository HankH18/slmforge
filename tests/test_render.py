"""The prompt renderer is the contract binding training to eval.

If these fail, the tuned model is being measured under a prompt it never saw.
"""
import re

import pytest

from constraints import registry
from harness import build_constraint_lookup
from scenarios.paths import load_corpus, load_instances, load_smoke
from scenarios.render import (HEADER, RENDERERS, RenderError, render_constraint,
                              render_user_message)


@pytest.fixture(scope="module")
def lookup():
    return build_constraint_lookup(load_instances().values())


def _c(ctype, params, cid="c-lex-999"):
    return {"id": cid, "type": ctype, "params": params,
            "family": registry.family_of(ctype), "split": registry.split_of(ctype)}


# ---------------------------------------------------------------- the binding
def test_every_registered_type_has_prompt_text():
    """A registered type with no rendering would be scored but never stated --
    an unannounced constraint the model cannot possibly obey."""
    assert set(RENDERERS) == set(registry.list_types())


def test_every_corpus_scenario_renders(lookup):
    corpus = load_corpus()
    assert len(corpus) == 150
    for s in corpus:
        msg = render_user_message(s, lookup)
        assert msg.strip()
        assert s["task"].strip() in msg
        assert HEADER in msg


def test_smoke_set_renders(lookup):
    for s in load_smoke():
        assert render_user_message(s, lookup).strip()


# ---------------------------------------------------------------- determinism
def test_rendering_is_byte_identical_across_calls(lookup):
    """Training renders once and eval renders again later. Any instability here
    silently breaks the training-condition-equals-eval-condition contract."""
    for s in load_corpus()[:20]:
        first = render_user_message(s, lookup)
        assert all(render_user_message(s, lookup) == first for _ in range(50))


def test_constraint_order_follows_the_scenario_not_the_lookup(lookup):
    """Declared order, not dict/set iteration order."""
    s = {"scenario_id": "t", "task": "Write something.",
         "constraints": ["c-lex-001", "c-fmt-001"], "k": 2,
         "pressure": "clean", "split": "train"}
    flipped = dict(s, constraints=["c-fmt-001", "c-lex-001"])
    a, b = render_user_message(s, lookup), render_user_message(flipped, lookup)
    assert a != b
    assert a.index("Do not use any of these words") < a.index("bulleted")
    assert b.index("bulleted") < b.index("Do not use any of these words")


# ------------------------------------------------- the no_lists priming guard
LIST_MARKER = re.compile(r"^\s*(?:[-*+•●▪‣⁃]|\d+[.)]|[a-z][.)])\s+",
                         re.IGNORECASE | re.MULTILINE)


def test_the_prompt_never_uses_list_formatting(lookup):
    """One constraint type is no_lists. A list-formatted requirement block would
    prime the very output format being forbidden, so the measurement would be of
    the prompt, not the model."""
    for s in load_corpus():
        msg = render_user_message(s, lookup)
        assert not LIST_MARKER.search(msg), f"{s['scenario_id']}: list marker in prompt"


def test_requirements_come_after_the_task(lookup):
    """Adversarial pressure lives in the task; requirements sit closest to
    generation. Reversing this changes what is being measured."""
    for s in load_corpus()[:30]:
        msg = render_user_message(s, lookup)
        assert msg.index(s["task"].strip()[:40]) < msg.index(HEADER)


def test_every_constraint_in_the_scenario_appears_in_the_prompt(lookup):
    """A constraint that is scored but not stated makes the scenario unpassable
    by construction and fakes difficulty."""
    for s in load_corpus():
        msg = render_user_message(s, lookup)
        body = msg.split(HEADER, 1)[1]
        assert len([ln for ln in body.strip().split("\n") if ln.strip()]) == s["k"]


# ---------------------------------------------------------------- the text
@pytest.mark.parametrize("ctype,params,must_contain", [
    ("banned_words", {"words": ["coffee", "espresso"]}, ["coffee", "espresso"]),
    ("banned_words", {"words": ["coffee"]}, ["the word coffee"]),
    ("banned_phrases", {"phrases": ["in conclusion", "to sum up"]},
     ["in conclusion", "to sum up"]),
    ("banned_letter", {"letter": "e"}, ["letter e", "inside words"]),
    ("banned_punctuation", {"marks": [","]}, ["commas"]),
    ("banned_punctuation", {"marks": [",", ";"]}, ["commas", "semicolons"]),
    ("sentence_initial_letter", {"letter": "s"}, ["Every sentence", "letter s"]),
    ("no_lists", {}, ["bulleted", "numbered"]),
    ("no_quotation_marks", {}, ["quotation marks"]),
])
def test_constraint_text_names_what_it_forbids(ctype, params, must_contain):
    text = render_constraint(_c(ctype, params))
    for frag in must_contain:
        assert frag in text, f"{ctype}: {frag!r} missing from {text!r}"


def test_punctuation_is_named_not_shown_as_a_bare_mark():
    """'Do not use these marks: ,' is ambiguous -- the separator is the mark."""
    text = render_constraint(_c("banned_punctuation", {"marks": [",", ";"]},
                                cid="c-chr-020"))
    assert "commas" in text and "semicolons" in text


# ---------------------------------------------------------------- refusals
def test_dangling_constraint_id_raises(lookup):
    s = {"scenario_id": "t", "task": "Write something.",
         "constraints": ["c-lex-001", "c-nope-999"], "k": 2,
         "pressure": "clean", "split": "train"}
    with pytest.raises(RenderError, match="unknown constraint id"):
        render_user_message(s, lookup)


@pytest.mark.parametrize("scenario,match", [
    ({"scenario_id": "t", "task": "", "constraints": ["c-lex-001"]}, "no usable 'task'"),
    ({"scenario_id": "t", "task": "   ", "constraints": ["c-lex-001"]}, "no usable 'task'"),
    ({"scenario_id": "t", "constraints": ["c-lex-001"]}, "no usable 'task'"),
    ({"scenario_id": "t", "task": "x", "constraints": []}, "no constraints"),
    ({"scenario_id": "t", "task": "x"}, "no constraints"),
])
def test_malformed_scenarios_raise(scenario, match, lookup):
    with pytest.raises(RenderError, match=match):
        render_user_message(scenario, lookup)


def test_non_mapping_inputs_raise(lookup):
    with pytest.raises(RenderError, match="scenario must be an object"):
        render_user_message("not a scenario", lookup)
    with pytest.raises(RenderError, match="constraint must be an object"):
        render_constraint("not a constraint")


def test_unknown_constraint_type_raises():
    with pytest.raises(RenderError, match="unknown constraint type"):
        render_constraint({"id": "c-lex-001", "type": "banned_vibes", "params": {}})


def test_malformed_params_raise_rather_than_rendering():
    """A malformed constraint must not become a plausible-looking instruction
    that the checker then scores by a different rule."""
    with pytest.raises(Exception):
        render_constraint(_c("banned_words", {"words": "coffee"}))
    with pytest.raises(Exception):
        render_constraint(_c("banned_letter", {"letter": "ee"}))


# ---------------------------------------------------------------- call shape
def test_arity_is_two_positional_args():
    """scripts/check_loop.py and training/modal_app.py both resolve this renderer
    by dotted path and call it positionally. A signature change breaks the
    training/eval binding at a distance."""
    import inspect
    params = inspect.signature(render_user_message).parameters
    positional = [p for p in params.values()
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    assert len(positional) == 2


def test_renderer_is_reachable_at_the_path_configs_declare():
    """configs/train.json names this renderer by dotted path; the loop checker
    restricts it to the scenarios/harness packages."""
    import importlib
    import json
    from pathlib import Path
    cfg = json.loads(Path(__file__).resolve().parent.parent.joinpath(
        "configs/train.json").read_text())
    dotted = cfg["data"]["prompt_renderer"]
    module_path, _, fn_name = dotted.rpartition(".")
    assert module_path.split(".")[0] in ("scenarios", "harness")
    fn = getattr(importlib.import_module(module_path), fn_name)
    assert fn is render_user_message
