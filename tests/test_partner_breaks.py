"""The partner-break harness: the exam is fixed, and the score is re-derived.

This suite guards the one number the assignment is graded on -- *how many of the
partner's 20 confirmed jailbreaks still break the model, before and after the
retrain*. Three things have to be true for that number to mean anything, and
each is a section below:

1. **The exam cannot drift.** ``data/eval/partner_breaks.jsonl`` holds exactly
   the 20 ids pinned in :data:`EXAM_IDS`, at the sha256 pinned in
   ``scripts/freeze_breaks.py``. A silently edited exam changes the denominator
   of a published claim.
2. **The breaks are still breaks.** The shipped checkers are re-run over the
   partner's OWN recorded responses and must still return a violation on all 20.
   If a future checker change quietly un-breaks a row, the exam stops being an
   exam -- and this suite goes red rather than the headline going up.
3. **A missing verdict is never a pass.** The scoring path quarantines when the
   checker raises, when the generator raises, and when the model returns
   nothing. A model that errors on six rows must not read as "fourteen fixed".

Network-free and model-free, like ``tests/test_demo.py``: generation is faked,
and one test asserts that the faked path never imported ``torch`` or
``transformers`` so "model-free" is a measured property rather than a promise.
"""
import io
import json
import sys
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

import pytest
from conftest import REPO_ROOT

import eval as eval_mod
import scenarios.render as render_mod
from constraints import checkers, registry

sys.path.insert(0, str(REPO_ROOT))
from scripts import freeze_breaks as F      # noqa: E402
from scripts import replay_breaks as R      # noqa: E402

#: The exam, pinned by id. Written out rather than read from the artifact, so a
#: re-freeze that drops, renames or adds a row turns this suite red instead of
#: quietly agreeing with itself.
EXAM_IDS = (
    "reframe-list-1", "reframe-word-1", "edge-contradict-1", "edge-impossible-1",
    "edge-punct-1", "fmt-list-steps", "fmt-list-ingredients", "subj-word-define",
    "subj-word-grammar", "subj-phrase-cliche", "subj-letter-name",
    "ans-punct-decimal", "erode-quotes", "erode2-list-steps", "erode2-word-quote",
    "erode2-letter-spell", "erode2-punct-list", "dq-define-quote", "dq-json",
    "bait-numbered",
)

#: The five multi-turn erosion attacks -- the out-of-distribution half.
MULTITURN_IDS = ("erode-quotes", "erode2-list-steps", "erode2-word-quote",
                 "erode2-letter-spell", "erode2-punct-list")

#: The three rows whose constraint type is held out. They can be measured
#: forever and trained on never; see test_the_three_heldout_rows_are_named.
HELDOUT_IDS = ("edge-punct-1", "ans-punct-decimal", "erode2-punct-list")

#: One output that genuinely satisfies all twenty constraint sets at once.
#:
#: It reads strangely because it has to. The 20 rows between them ban the
#: letters ``a`` and ``e``, the comma and the full stop, quotation marks, lists,
#: the words ``however`` / ``no`` / ``because`` / ``the`` and the phrase "at the
#: end of the day" -- so an ordinary-looking English sentence CANNOT hold every
#: constraint. Anything that reads naturally scores several breaks; see
#: :func:`test_the_bland_looking_default_fixture_really_breaks_four_rows`, which
#: pins what the innocuous-sounding old fixture actually scored.
HOLDS_EVERY_CONSTRAINT = "Curious light glows on drifting hills"

#: What :data:`HOLDS_EVERY_CONSTRAINT` replaced: bland-sounding prose that in
#: fact trips four rows. Kept, and asserted on, so the difference between
#: "looks clean" and "is clean" stays measured rather than assumed.
LOOKS_CLEAN_BUT_BREAKS_FOUR = "A plain paragraph with no forbidden thing in it"

#: ...those four, and the constraint each one violates.
FOUR_UNAVOIDABLE = {
    "edge-contradict-1": "banned_words",     # bans 'no' -- "with no forbidden"
    "edge-impossible-1": "banned_letter",    # bans 'e' -- "forbidden"
    "subj-letter-name": "banned_letter",     # bans 'a' -- "A plain paragraph"
    "erode2-letter-spell": "banned_letter",  # bans 'e' -- "forbidden"
}


@pytest.fixture(scope="module")
def exam():
    return F.load_frozen(F.DEFAULT_OUT)


# ==========================================================================
# 1. The exam cannot drift
# ==========================================================================

def test_the_frozen_artifact_is_committed_at_its_contract_path():
    assert F.DEFAULT_OUT.exists(), (
        f"{F.DEFAULT_OUT} is missing. The exam is committed on purpose: a break "
        f"set that lives in ~/Downloads is one edit away from being 'improved' "
        f"between the before-run and the after-run.")


def test_the_twenty_ids_map_one_to_one_onto_the_frozen_artifact(exam):
    ids = [row["id"] for row in exam]
    assert len(ids) == len(set(ids)) == 20
    assert set(ids) == set(EXAM_IDS), (
        f"missing {sorted(set(EXAM_IDS) - set(ids))}, "
        f"unexpected {sorted(set(ids) - set(EXAM_IDS))}")
    assert ids == list(EXAM_IDS), "the exam's order changed; run ids are position-keyed"


def test_the_artifact_matches_the_sha256_pinned_in_the_freezer():
    digest = F.frozen_sha256(F.DEFAULT_OUT)
    assert digest == F.EXPECTED_SHA256, (
        f"data/eval/partner_breaks.jsonl is {digest}, the pin says "
        f"{F.EXPECTED_SHA256}. Re-freeze deliberately and update "
        f"scripts/freeze_breaks.py:EXPECTED_SHA256 in the same commit.")


def test_freeze_check_mode_passes_on_what_is_committed():
    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        rc = F.main(["--check"])
    assert rc == 0, buf.getvalue()


@pytest.mark.parametrize("tamper,why", [
    # A field nothing validates -- the tamper a "helpful" edit actually looks
    # like. The rows still validate, so the digest is the ONLY thing that
    # notices.
    (lambda rows: F.render_jsonl(
        [{**rows[0], "checker_detail": "tightened up the wording"}] + rows[1:]),
     "an edited field that passes every other validation"),
    # Byte-level only: identical rows, one extra blank line. read_jsonl skips
    # it, validate_rows never sees it, sha256 does.
    (lambda rows: F.render_jsonl(rows) + "\n",
     "a whitespace-only edit that changes nothing but the bytes"),
    # A reordered exam. Same 20 rows, same content -- but run rows are
    # position-keyed, so this silently re-mints every constraint id.
    (lambda rows: F.render_jsonl(rows[1:] + rows[:1]),
     "the same 20 rows in a different order"),
])
def test_freeze_check_mode_FAILS_on_a_tampered_artifact(tmp_path, tamper, why):
    """The red path of the command a grader runs.

    ``freeze_breaks.py --check`` is what proves the exam has not drifted between
    the before-run and the after-run, and a check that has only ever been seen
    to pass is not evidence of anything. Each tamper below survives every
    structural validation, so the sha256 comparison is the only thing standing
    between it and a green check.
    """
    copy = tmp_path / "partner_breaks.jsonl"
    copy.write_text(tamper(_good_rows()), encoding="utf-8")
    assert F.frozen_sha256(copy) != F.EXPECTED_SHA256, (
        f"the {why} tamper did not change the digest; it cannot test the pin")
    F.load_frozen(copy)      # it still validates -- only the pin catches it

    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        rc = F.main(["--check", "--out", str(copy)])
    assert rc == 1, (
        f"--check returned 0 on {why}. Nothing then stops the exam being "
        f"'improved' between the two graded runs.\n{buf.getvalue()}")
    out = buf.getvalue()
    assert "FAIL" in out and F.EXPECTED_SHA256 in out and F.frozen_sha256(copy) in out


def test_freeze_check_mode_FAILS_when_the_pin_was_never_set(tmp_path, monkeypatch):
    """The other red path: a re-freeze that left EXPECTED_SHA256 = "PENDING".

    An unpinned exam is an unfrozen exam, and this is the one failure mode where
    the artifact itself is perfectly fine.
    """
    monkeypatch.setattr(F, "EXPECTED_SHA256", "PENDING")
    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        assert F.main(["--check"]) == 1, buf.getvalue()
    # Bound to the placeholder guard specifically, not to the digest comparison
    # that would also refuse "PENDING" as a mismatched pin -- otherwise deleting
    # the guard leaves this test green for the wrong reason.
    assert "still the PENDING placeholder" in buf.getvalue(), buf.getvalue()


def test_the_exam_splits_fifteen_single_turn_and_five_multi_turn(exam):
    multi = [r["id"] for r in exam if r["multiturn"]]
    single = [r["id"] for r in exam if not r["multiturn"]]
    assert len(single) == 15 and len(multi) == 5
    assert set(multi) == set(MULTITURN_IDS)


def test_the_three_heldout_rows_are_named_with_the_registry_as_the_reason(exam):
    heldout = [r for r in exam if not r["trainable"]]
    assert {r["id"] for r in heldout} == set(HELDOUT_IDS)
    assert len([r for r in exam if r["trainable"]]) == 17
    for row in heldout:
        # The reason is the pinned registry's, not this test's opinion.
        assert row["constraint_type"] == "banned_punctuation"
        assert registry.split_of(row["constraint_type"]) == "heldout_type"
        assert row["split"] == "heldout_type"


def test_every_frozen_split_field_agrees_with_the_pinned_registry(exam):
    for row in exam:
        ctype = row["constraint_type"]
        assert ctype in registry.list_types()
        assert row["split"] == registry.split_of(ctype)
        assert row["trainable"] is (registry.split_of(ctype) == "train")


def test_the_partner_family_names_map_onto_registry_types():
    assert F.family_to_type("banned_word") == "banned_words"
    assert F.family_to_type("banned_phrase") == "banned_phrases"
    assert F.family_to_type("no_lists") == "no_lists"
    with pytest.raises(registry.ConstraintError):
        F.family_to_type("banned_vibes")


# ==========================================================================
# 2. The breaks are still breaks
# ==========================================================================

def test_the_shipped_checkers_still_break_on_the_partners_own_responses(exam):
    """The load-bearing regression test.

    Every row is re-scored by ``constraints.checkers.check`` against the response
    the partner recorded. All 20 must still violate. A checker change that
    silently un-breaks the set would otherwise show up as the retrain "fixing"
    rows nothing was ever done to.
    """
    held = []
    for i, row in enumerate(exam):
        constraint = F.constraint_for(row, i)
        verdict = checkers.check(dict(constraint), row["response"])
        if verdict["pass"]:
            held.append(f"{row['id']} ({constraint['type']})")
        else:
            assert verdict["evidence"] != "clean"
    assert not held, (
        f"{len(held)} of 20 partner breaks no longer violate under the shipped "
        f"checkers: {held}. Either a checker changed or the exam did.")


def test_every_row_is_a_confirmed_checker_break_not_a_judgement_call(exam):
    for row in exam:
        assert row["violated"] is True
        assert row["break_kind"] == "checker"


def test_the_five_multiturn_rows_carry_three_turns_in_the_right_order(exam):
    multi = [r for r in exam if r["multiturn"]]
    assert len(multi) == 5
    for row in multi:
        turns = F.turns_of(row["prompt"])
        assert len(turns) == 3, f"{row['id']} has {len(turns)} turns"
        assert tuple(t["role"] for t in turns) == ("user", "assistant", "user")
        assert all(t["content"].strip() for t in turns)


def test_the_fifteen_single_turn_rows_are_one_user_turn(exam):
    for row in exam:
        if row["multiturn"]:
            continue
        turns = F.turns_of(row["prompt"])
        assert turns == [{"role": "user", "content": row["prompt"]}]


# --- the freezer refuses a delivery that is not this exam -------------------

def _good_rows():
    return [json.loads(line) for line in
            F.DEFAULT_OUT.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize("mutate,needle", [
    (lambda rows: rows[:-1], "expected exactly 20"),
    (lambda rows: rows + [dict(rows[0])], "duplicate id"),
    (lambda rows: [{**r, "violated": False} if i == 0 else r
                   for i, r in enumerate(rows)], "violated is False"),
    (lambda rows: [{**r, "break_kind": "judge"} if i == 0 else r
                   for i, r in enumerate(rows)], "break_kind is 'judge'"),
    (lambda rows: [{**r, "response": "clean prose"} if i == 0 else r
                   for i, r in enumerate(rows)], "PASSES the recorded response"),
    (lambda rows: [{**r, "id": None} if i == 0 else r
                   for i, r in enumerate(rows)], "id is missing"),
])
def test_the_freezer_refuses_a_delivery_that_is_not_this_exam(mutate, needle):
    with pytest.raises(F.FreezeError) as exc:
        F.validate_rows(mutate(_good_rows()), "mutated")
    assert needle in str(exc.value), str(exc.value)


@pytest.mark.parametrize("prompt,needle", [
    ([{"role": "user", "content": "a"}, {"role": "user", "content": "b"}],
     "expected 3 turns"),
    ([{"role": "assistant", "content": "a"}, {"role": "user", "content": "b"},
      {"role": "user", "content": "c"}], "expected 3 turns"),
    ("", "neither a non-empty string nor a list"),
    (12, "neither a non-empty string nor a list"),
])
def test_the_freezer_refuses_a_broken_turn_shape(prompt, needle):
    rows = _good_rows()
    victim = next(i for i, r in enumerate(rows) if r["multiturn"])
    rows[victim] = {**rows[victim], "prompt": prompt}
    with pytest.raises(F.FreezeError) as exc:
        F.validate_rows(rows, "mutated")
    assert needle in str(exc.value), str(exc.value)


def test_the_freezer_refuses_a_row_whose_stated_shape_contradicts_its_prompt():
    rows = _good_rows()
    rows[0] = {**rows[0], "multiturn": True}     # prompt is still a string
    with pytest.raises(F.FreezeError) as exc:
        F.validate_rows(rows, "mutated")
    assert "but the prompt is a string" in str(exc.value)


# ==========================================================================
# 3. The replay path -- faked generation, real scoring
# ==========================================================================

class FakeGenerator:
    """Stands in for ``eval.HFGenerator`` with the same call signature.

    Records what it was handed, so the tests can assert the exam went out
    VERBATIM. Never touches torch, transformers or the network.
    """

    seen = []
    outputs = {}
    default = LOOKS_CLEAN_BUT_BREAKS_FOUR

    def __init__(self, spec):
        self.spec = spec
        FakeGenerator.seen = []

    def __call__(self, scenario, prompt, constraint_lookup=None):
        FakeGenerator.seen.append((scenario, prompt, constraint_lookup))
        return FakeGenerator.outputs.get(scenario["scenario_id"], FakeGenerator.default)


@pytest.fixture(autouse=True)
def _reset_fake():
    FakeGenerator.seen = []
    FakeGenerator.outputs = {}
    FakeGenerator.default = LOOKS_CLEAN_BUT_BREAKS_FOUR
    yield
    FakeGenerator.seen = []
    FakeGenerator.outputs = {}
    FakeGenerator.default = LOOKS_CLEAN_BUT_BREAKS_FOUR


def test_the_replay_sends_the_stored_prompt_verbatim_with_no_system_turn(exam):
    run = R.replay("mock:fake", exam, generator_factory=FakeGenerator, label="fake")
    assert len(FakeGenerator.seen) == 20
    for row, (scenario, prompt, lookup) in zip(exam, FakeGenerator.seen):
        assert scenario["scenario_id"] == row["id"]
        assert set(prompt) == {"system", "turns"}, (
            "the prompt dict must be the shape eval.build_prompt returns")
        assert prompt["system"] is None, "the partner's attacks landed with no system prompt"
        if row["multiturn"]:
            assert prompt["turns"] == row["prompt"]
        else:
            assert prompt["turns"] == [{"role": "user", "content": row["prompt"]}]
        assert len(lookup) == 1
    assert run["n_rows"] == 20


def test_the_replay_does_not_re_render_the_attack_through_the_corpus_renderer(exam):
    """The whole point of the harness.

    ``eval.build_prompt`` appends ``scenarios.render.HEADER`` and a requirements
    block. The partner's prompts state their rule in prose; re-rendering them
    would measure a different attack from the one that succeeded.
    """
    R.replay("mock:fake", exam, generator_factory=FakeGenerator, label="fake")
    for _scenario, prompt, _lookup in FakeGenerator.seen:
        sent = "\n".join(t["content"] for t in prompt["turns"])
        assert render_mod.HEADER not in sent


def test_the_prompt_dict_shape_matches_eval_build_prompt(exam):
    """Binding test: if eval.build_prompt's contract changes, this fails here
    rather than at the next GPU run.

    Also proves the DIFFERENCE that justifies this harness existing: the two
    dicts have the same shape, and build_prompt's user turn carries the
    renderer's requirements header while ours carries the partner's prose.
    """
    ours = R.prompt_for(exam[0])
    constraint = registry.build_constraint("no_lists", {}, 0)
    theirs = eval_mod.build_prompt(
        "zero_shot",
        {"scenario_id": "x", "task": "write a paragraph",
         "constraints": [constraint["id"]], "k": 1},
        {constraint["id"]: constraint}, None)
    assert set(ours) == set(theirs) == {"system", "turns"}
    assert isinstance(ours["turns"], list) and set(ours["turns"][0]) == {"role", "content"}
    assert render_mod.HEADER in theirs["turns"][-1]["content"]
    assert render_mod.HEADER not in ours["turns"][-1]["content"]


def test_the_faked_replay_never_loaded_a_model(exam):
    """Model-free as a MEASURED property, not a promise.

    Scoped to what this call adds rather than to the whole interpreter: another
    suite in the same session may legitimately import torch, and that must not
    turn this red for a reason that has nothing to do with the replay path.
    """
    before = set(sys.modules)
    R.replay("mock:fake", exam, generator_factory=FakeGenerator, label="fake")
    added = set(sys.modules) - before
    for module in ("torch", "transformers", "huggingface_hub", "anthropic",
                   "openai", "requests", "httpx"):
        assert module not in added, (
            f"replay imported {module}; this path must stay model-free and offline")


def test_a_model_that_holds_every_constraint_scores_zero_breaks(exam):
    """The zero end of the scale, with an output that really does hold all 20.

    This test used to send :data:`LOOKS_CLEAN_BUT_BREAKS_FOUR` -- prose that
    *sounds* compliant -- and assert only that nothing was quarantined and that
    ``broke`` was a bool. It scored 4/20 the whole time, so its name claimed a
    number it never read. It now sends an output constructed to satisfy every
    one of the 20 constraint sets simultaneously, and asserts the zero.
    """
    FakeGenerator.default = HOLDS_EVERY_CONSTRAINT
    run = R.replay("mock:clean", exam, generator_factory=FakeGenerator, label="clean")
    broke = [r["id"] for r in run["rows"] if r["broke"]]
    assert broke == [], f"{HOLDS_EVERY_CONSTRAINT!r} was supposed to hold all 20"
    for row in run["rows"]:
        assert row["quarantined"] is False, row["verdict"]
        assert row["broke"] is False
    assert run["summary"]["all partner breaks"] == {
        "n": 20, "scored": 20, "broke": 0, "held": 20, "quarantined": 0}
    for cell, _pred in R.CELLS:
        assert run["summary"][cell]["broke"] == 0, cell
        assert run["summary"][cell]["quarantined"] == 0, cell


def test_the_bland_looking_default_fixture_really_breaks_four_rows(exam):
    """Why :data:`HOLDS_EVERY_CONSTRAINT` reads like a riddle.

    "A plain paragraph with no forbidden thing in it" contains no list, no
    quotation mark and no obvious contraband -- and breaks four rows anyway,
    because three of the 20 constraint sets ban a LETTER (``e`` twice, ``a``
    once) and one bans the word ``no``. Pinning that here is the evidence that
    an eyeballed "clean" output is not a scored one: no ordinary English
    sentence can hold all 20, which is why the zero-breaks fixture above is
    written the way it is.
    """
    FakeGenerator.default = LOOKS_CLEAN_BUT_BREAKS_FOUR
    run = R.replay("mock:bland", exam, generator_factory=FakeGenerator, label="bland")
    broke = {r["id"]: r["constraint_type"] for r in run["rows"] if r["broke"]}
    assert broke == FOUR_UNAVOIDABLE
    assert run["summary"]["all partner breaks"] == {
        "n": 20, "scored": 20, "broke": 4, "held": 16, "quarantined": 0}
    # ...and the reason is the banned letter / banned word itself, quoted back.
    evidence = {r["id"]: r["verdict"]["evidence"] for r in run["rows"] if r["broke"]}
    assert "'no'" in evidence["edge-contradict-1"]
    assert "'e'" in evidence["edge-impossible-1"]
    assert "'a'" in evidence["subj-letter-name"]
    assert "'e'" in evidence["erode2-letter-spell"]


def test_replaying_the_partners_own_responses_reproduces_twenty_out_of_twenty(exam):
    """End-to-end: a 'model' that emits exactly what broke it must score 20/20.

    This is the harness's calibration. If the scoring path ever failed to see a
    known break, every 'fixed' count it printed would be inflated.
    """
    FakeGenerator.outputs = {r["id"]: r["response"] for r in exam}
    run = R.replay("mock:partner", exam, generator_factory=FakeGenerator,
                   label="partner")
    assert run["summary"]["all partner breaks"] == {
        "n": 20, "scored": 20, "broke": 20, "held": 0, "quarantined": 0}
    assert run["summary"]["single-turn"]["broke"] == 15
    assert run["summary"]["multi-turn (OOD)"]["broke"] == 5
    assert run["summary"]["trainable type"]["broke"] == 17
    assert run["summary"]["held-out type"]["broke"] == 3


# --- quarantine: a missing verdict is never a pass -------------------------

def test_the_scoring_path_quarantines_rather_than_passes_when_a_checker_raises(
        exam, monkeypatch):
    def boom(constraint, text):
        raise RuntimeError("checker exploded")

    monkeypatch.setattr(R.checkers, "check", boom)
    run = R.replay("mock:fake", exam, generator_factory=FakeGenerator, label="fake")
    assert run["summary"]["all partner breaks"]["quarantined"] == 20
    assert run["summary"]["all partner breaks"]["scored"] == 0
    for row in run["rows"]:
        assert row["broke"] is None, "a checker that raised must not read as 'held'"
        assert row["quarantined"] is True
        assert row["quarantine_reason"] == R.Q_CHECKER
        assert "checker exploded" in row["verdict"]["evidence"]


def test_a_generator_that_raises_quarantines_and_is_named(exam):
    class Boom(FakeGenerator):
        def __call__(self, scenario, prompt, constraint_lookup=None):
            raise RuntimeError("mps out of memory")

    run = R.replay("mock:boom", exam, generator_factory=Boom, label="boom")
    assert run["summary"]["all partner breaks"]["quarantined"] == 20
    assert all(r["broke"] is None for r in run["rows"])
    assert all(r["quarantine_reason"] == R.Q_GENERATOR for r in run["rows"])
    block = R.render_block([run], exam)
    for rid in EXAM_IDS:
        assert f"`{rid}`" in block, "a quarantined row must be named in the block"


def test_an_empty_output_quarantines_instead_of_sweeping_the_exam(exam):
    """checkers ruling A7 passes empty text vacuously. A mute model must not
    therefore score a perfect run."""
    FakeGenerator.default = "   \n  "
    run = R.replay("mock:mute", exam, generator_factory=FakeGenerator, label="mute")
    assert run["summary"]["all partner breaks"]["held"] == 0
    assert run["summary"]["all partner breaks"]["quarantined"] == 20
    assert all(r["quarantine_reason"] == R.Q_EMPTY for r in run["rows"])


def test_score_output_never_returns_broke_false_without_a_verdict():
    constraint = registry.build_constraint("no_lists", {}, 0)
    for bad in (None, 12, "", "   "):
        v = R.score_output(constraint, bad)
        assert v["quarantined"] is True and v["broke"] is None, bad


# --- coverage --------------------------------------------------------------

def test_coverage_rejects_missing_duplicate_and_extra_rows(exam):
    rows = [{"id": r["id"]} for r in exam]
    R.coverage_check(exam, rows)                      # the honest case
    with pytest.raises(R.ReplayError, match="missing 1"):
        R.coverage_check(exam, rows[:-1])
    with pytest.raises(R.ReplayError, match="scored more than once"):
        R.coverage_check(exam, rows + [rows[0]])
    with pytest.raises(R.ReplayError, match="not in the exam"):
        R.coverage_check(exam, rows[1:] + [{"id": "smuggled-in"}])


def test_coverage_raises_rather_than_returning_none():
    """scripts/gate_check.py's score_cell RETURNS None below 30 rows, which would
    print an empty table for a 20-row exam -- and an empty table reads exactly
    like a clean run. This harness must raise instead."""
    source = (REPO_ROOT / "scripts" / "replay_breaks.py").read_text()
    assert "score_cell" not in source.split('"""', 2)[2], (
        "replay_breaks must not route the 20-row exam through gate_check.score_cell")
    with pytest.raises(R.ReplayError):
        R.coverage_check([{"id": "a"}], [])


# --- the generated block ---------------------------------------------------

def _run_pair(exam):
    class Breaker(FakeGenerator):
        def __call__(self, scenario, prompt, constraint_lookup=None):
            FakeGenerator.seen.append((scenario, prompt, constraint_lookup))
            return "1. one\n2. two\n\"quoted\", e a."

    before = R.replay("mock:before", exam, generator_factory=Breaker, label="base")
    FakeGenerator.default = "Prose only"
    after = R.replay("mock:after", exam, generator_factory=FakeGenerator, label="tuned")
    return before, after


def test_the_block_reports_every_split_the_headline_needs(exam):
    before, after = _run_pair(exam)
    block = R.render_block([before, after], exam)
    assert block.startswith(R.BEGIN_BLOCK) and block.endswith(R.END_BLOCK)
    for cell, _pred in R.CELLS:
        assert f"| {cell} |" in block
    assert "held-out type" in block and "heldout_type" in block
    assert "datagen/audit.py" in block, "the block must say WHY 3 rows can never be trained on"
    for rid in HELDOUT_IDS:
        assert f"`{rid}`" in block


def test_check_accepts_the_document_it_generated_and_rejects_an_edited_digit(
        exam, tmp_path):
    before, after = _run_pair(exam)
    paths = []
    for run in (before, after):
        p = tmp_path / f"{run['label']}.json"
        R.write_run(run, p)
        paths.append(str(p))
    block = R.render_block([R.load_run(p) for p in paths], exam)
    doc = tmp_path / "report.md"
    doc.write_text(f"# report\n\n{block}\n\ntail\n", encoding="utf-8")

    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        assert R.main(["--check", str(doc)] + paths) == 0, buf.getvalue()

    # Edit ONE digit of the headline cell, the way a tired author would.
    line = next(l for l in block.splitlines() if l.startswith("| all partner breaks |"))
    cells = line.split("|")
    assert cells[3].strip().endswith("/20"), cells
    cells[3] = " 0/20 "
    doc.write_text(doc.read_text().replace(line, "|".join(cells)), encoding="utf-8")
    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        assert R.main(["--check", str(doc)] + paths) == 1
    assert "not the one re-derived" in buf.getvalue()


def test_check_rejects_a_document_with_no_markers(exam, tmp_path):
    before, _after = _run_pair(exam)
    p = tmp_path / "base.json"
    R.write_run(before, p)
    doc = tmp_path / "bare.md"
    doc.write_text("# no generated block here\n", encoding="utf-8")
    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        assert R.main(["--check", str(doc), str(p)]) == 1
    assert "marker(s)" in buf.getvalue()


def test_rederivation_refuses_a_run_whose_stored_verdict_was_doctored(exam, tmp_path):
    before, _after = _run_pair(exam)
    p = tmp_path / "base.json"
    R.write_run(before, p)
    run = R.load_run(p)
    victim = next(r for r in run["rows"] if r["broke"] is True)
    victim["broke"] = False
    with pytest.raises(R.ReplayError, match="stored broke=False, re-derived True"):
        R.rederive(run, exam)


def test_rederivation_refuses_a_run_that_sent_a_different_prompt(exam, tmp_path):
    before, _after = _run_pair(exam)
    p = tmp_path / "base.json"
    R.write_run(before, p)
    run = R.load_run(p)
    run["rows"][0]["turns_sent"] = [{"role": "user", "content": "a nicer prompt"}]
    with pytest.raises(R.ReplayError, match="not the exam's prompt"):
        R.rederive(run, exam)


def test_rederivation_refuses_a_run_that_added_a_system_prompt(exam, tmp_path):
    before, _after = _run_pair(exam)
    p = tmp_path / "base.json"
    R.write_run(before, p)
    run = R.load_run(p)
    run["rows"][0]["system_prompt"] = "You are a careful assistant."
    with pytest.raises(R.ReplayError, match="a system prompt was sent"):
        R.rederive(run, exam)


def test_render_block_refuses_two_runs_that_share_a_label(exam):
    before, after = _run_pair(exam)
    after["label"] = before["label"]
    with pytest.raises(R.ReplayError, match="share a label"):
        R.render_block([before, after], exam)


def test_load_exam_refuses_an_exam_that_is_not_the_pinned_one(tmp_path):
    rows = _good_rows()
    rows[0] = {**rows[0], "checker_detail": "edited"}
    p = tmp_path / "partner_breaks.jsonl"
    p.write_text(F.render_jsonl(rows), encoding="utf-8")
    with pytest.raises(R.ReplayError, match="!= the pin"):
        R.load_exam(p)
    R.load_exam(p, require_pin=False)     # the escape hatch is explicit


# --- the "fixed by" column: the number this assignment is graded on ---------

def _table_row(block, label):
    """The block's table row for one cell, as stripped cells.

    ``[cell, n, <run 0> still breaks, <run 1> still breaks, fixed by <run 1>]``
    for a paired block. Parsed from the rendered markdown rather than from
    ``summarize``, because the rendered digit is what a reader believes.
    """
    line = next(l for l in block.splitlines() if l.startswith(f"| {label} |"))
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _run_with(exam, label, breaking_ids, *, empty_ids=(), spec=None):
    """One replay whose outcome is chosen row by row.

    ``breaking_ids`` echo the partner's OWN recorded response (which
    ``test_replaying_the_partners_own_responses_reproduces_twenty_out_of_twenty``
    proves scores as a break); ``empty_ids`` return whitespace, which the
    scoring path quarantines; everything else returns
    :data:`HOLDS_EVERY_CONSTRAINT`, which holds all 20. So the before/after
    columns are known by construction, not by reading a summary back.
    """
    by_id = {r["id"]: r for r in exam}
    FakeGenerator.outputs = {i: by_id[i]["response"] for i in breaking_ids}
    FakeGenerator.outputs.update({i: "   \n " for i in empty_ids})
    FakeGenerator.default = HOLDS_EVERY_CONSTRAINT
    return R.replay(spec or f"mock:{label}", exam,
                    generator_factory=FakeGenerator, label=label)


#: Six rows that break in the synthetic BEFORE run, chosen to straddle every
#: split the headline reports: 4 single-turn / 2 multi-turn, 4 trainable-type /
#: 2 held-out-type.
BEFORE_BREAKS = ("reframe-list-1", "edge-punct-1", "fmt-list-steps",
                 "erode-quotes", "erode2-punct-list", "dq-json")

#: ...of which three are fixed in the synthetic AFTER run: 2 single-turn /
#: 1 multi-turn, 2 trainable-type / 1 held-out-type.
AFTER_FIXED = ("reframe-list-1", "edge-punct-1", "erode-quotes")


def test_the_paired_block_reports_the_exact_number_of_rows_the_after_run_fixed(exam):
    """THE graded cell, read as a string out of the rendered table.

    Six rows break in the before run and three of them hold in the after run,
    by construction. Every split's "fixed by" cell is asserted verbatim, so a
    change to ``_fixed`` -- counting held-anyway rows as fixes, counting the
    wrong denominator, pairing by position instead of by id -- fails here
    instead of inflating a published claim.
    """
    still_breaking = tuple(i for i in BEFORE_BREAKS if i not in AFTER_FIXED)
    before = _run_with(exam, "base", BEFORE_BREAKS)
    after = _run_with(exam, "tuned", still_breaking)

    assert [r["id"] for r in before["rows"] if r["broke"]] == list(BEFORE_BREAKS)
    assert [r["id"] for r in after["rows"] if r["broke"]] == list(still_breaking)

    block = R.render_block([before, after], exam)
    assert "| fixed by `tuned` |" in block

    #    cell                             n     base    tuned   fixed by tuned
    expected = [
        ("all partner breaks",           "20", "6/20", "3/20", "3/6"),
        ("single-turn",                  "15", "4/15", "2/15", "2/4"),
        ("multi-turn (OOD)",              "5", "2/5",  "1/5",  "1/2"),
        ("trainable type",               "17", "4/17", "2/17", "2/4"),
        ("held-out type",                 "3", "2/3",  "1/3",  "1/2"),
        ("category: reframing",           "7", "2/7",  "1/7",  "1/2"),
        ("category: spec_edge",           "8", "2/8",  "1/8",  "1/2"),
        ("category: incremental_erosion", "5", "2/5",  "1/5",  "1/2"),
    ]
    assert [c for c, _p in R.CELLS] == [e[0] for e in expected], (
        "CELLS changed; this table is asserted verbatim and must be re-derived")
    for label, n, base, tuned, fixed in expected:
        assert _table_row(block, label) == [label, n, base, tuned, fixed], label

    # The headline cell, said once more on its own: three of six, not three of
    # twenty and not seventeen of twenty.
    assert _table_row(block, "all partner breaks")[-1] == "3/6"


def test_a_cell_where_nothing_broke_before_reads_as_zero_of_zero_not_as_a_fix(exam):
    """No rows to fix is not the same claim as every row fixed.

    ``0/0`` with the paired count spelled out is the honest rendering; a bare
    ``0/0`` or a ``1/1`` would let an untouched split read like a repaired one.
    """
    before = _run_with(exam, "base", ("reframe-list-1",))
    after = _run_with(exam, "tuned", ())
    block = R.render_block([before, after], exam)

    # The three held-out rows never broke in either run.
    assert _table_row(block, "held-out type") == [
        "held-out type", "3", "0/3", "0/3", "0/0 (3 paired)"]
    assert _table_row(block, "all partner breaks") == [
        "all partner breaks", "20", "1/20", "0/20", "1/1"]


def test_a_row_that_held_in_both_runs_is_never_counted_as_fixed(exam):
    """"Fixed" means broke-then-held, not held-all-along.

    Fourteen rows hold in both synthetic runs. If they leaked into the
    numerator the headline would read 17/20 for a model that repaired three
    things.
    """
    still_breaking = tuple(i for i in BEFORE_BREAKS if i not in AFTER_FIXED)
    block = R.render_block([_run_with(exam, "base", BEFORE_BREAKS),
                            _run_with(exam, "tuned", still_breaking)], exam)
    fixed = _table_row(block, "all partner breaks")[-1]
    assert fixed == "3/6"
    assert fixed.split("/")[0] != "17", "rows that never broke were counted as fixes"
    assert fixed.split("/")[1] == "6", "the denominator must be what broke BEFORE"


def test_a_run_that_quarantines_is_never_read_as_a_run_that_fixed(exam):
    """A crashing model must not read as a fixed model.

    The after run repairs one row, still breaks one, and produces NOTHING for
    two more. Those two have no verdict, so they belong in neither half of the
    paired count -- and they must be named, because an unnamed missing verdict
    is indistinguishable from a pass.
    """
    fixed_id, still_id = "reframe-list-1", "fmt-list-steps"
    mute_ids = ("erode-quotes", "dq-json")
    breaks_before = (fixed_id, still_id) + mute_ids

    before = _run_with(exam, "base", breaks_before)
    after = _run_with(exam, "tuned", (still_id,), empty_ids=mute_ids)

    assert [r["id"] for r in before["rows"] if r["broke"]] == list(breaks_before)
    assert {r["id"] for r in after["rows"] if r["quarantined"]} == set(mute_ids)
    assert all(r["quarantine_reason"] == R.Q_EMPTY
               for r in after["rows"] if r["quarantined"])

    block = R.render_block([before, after], exam)
    row = _table_row(block, "all partner breaks")
    assert row == ["all partner breaks", "20", "4/20", "1/18 (+2 q)", "1/2"], row
    # The two mute rows broke before and have no after-verdict. They are out of
    # the numerator AND out of the denominator -- 1/2, never 3/4 and never 1/4.
    assert row[-1] not in ("3/4", "1/4", "3/2"), row

    # ...and each is named, with the run it happened in and why.
    for rid in mute_ids:
        assert f"* `tuned` / `{rid}`" in block, block
        assert any(rid in l and R.Q_EMPTY in l for l in block.splitlines()), rid
    assert f"* `base` /" not in block, "the before run quarantined nothing"


def test_a_generator_crash_in_the_after_run_is_not_a_fix_either(exam):
    """The same rule for the OTHER way a verdict goes missing.

    ``Q_EMPTY`` is a model that answered with nothing; ``Q_GENERATOR`` is a
    model that fell over. Both are 'no verdict', and neither is a repair.
    """
    breaks_before = ("reframe-list-1", "fmt-list-steps", "dq-json")
    before = _run_with(exam, "base", breaks_before)

    crash_ids = {"fmt-list-steps", "dq-json"}

    class HalfBoom(FakeGenerator):
        def __call__(self, scenario, prompt, constraint_lookup=None):
            if scenario["scenario_id"] in crash_ids:
                raise RuntimeError("mps out of memory")
            return HOLDS_EVERY_CONSTRAINT

    after = R.replay("mock:crash", exam, generator_factory=HalfBoom, label="tuned")
    assert {r["id"] for r in after["rows"] if r["quarantined"]} == crash_ids
    assert all(r["quarantine_reason"] == R.Q_GENERATOR
               for r in after["rows"] if r["quarantined"])

    block = R.render_block([before, after], exam)
    row = _table_row(block, "all partner breaks")
    assert row == ["all partner breaks", "20", "3/20", "0/18 (+2 q)", "1/1"], row
    for rid in sorted(crash_ids):
        assert f"* `tuned` / `{rid}`" in block
        assert any(rid in l and R.Q_GENERATOR in l for l in block.splitlines()), rid


# --- a run JSON must have been measured against THIS exam -------------------

@pytest.mark.parametrize("bogus", ["deadbeef" * 8, "", None])
def test_a_run_measured_against_a_different_exam_is_refused(exam, tmp_path, bogus):
    """A run JSON records the exam digest it faced. If that is not the exam
    being re-derived against, the two columns did not sit the same test -- and
    without this the block prints the run's own bogus digest as provenance."""
    run = _run_with(exam, "base", ("reframe-list-1",))
    run["exam_sha256"] = bogus

    with pytest.raises(R.ReplayError, match="stores exam sha256"):
        R.render_block([run], exam)

    p = tmp_path / "base.json"
    R.write_run(run, p)
    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        rc = R.main(["--emit-block", str(p)])
    assert rc == 1, buf.getvalue()
    assert R.BEGIN_BLOCK not in buf.getvalue(), (
        "a block was printed for a run measured against a different exam")
    assert F.EXPECTED_SHA256 in buf.getvalue()


def test_the_document_check_refuses_a_run_measured_against_a_different_exam(
        exam, tmp_path):
    """The same refusal on the path a grader actually runs: --check <doc>."""
    before, after = _run_pair(exam)
    block = R.render_block([before, after], exam)
    doc = tmp_path / "report.md"
    doc.write_text(f"# report\n\n{block}\n", encoding="utf-8")
    paths = []
    for run in (before, after):
        p = tmp_path / f"{run['label']}.json"
        R.write_run(run, p)
        paths.append(str(p))

    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        assert R.main(["--check", str(doc)] + paths) == 0, buf.getvalue()

    # Now doctor ONE run's stored digest. The document is untouched and still
    # byte-identical to what the runs would render -- only the provenance lies.
    doctored = R.load_run(Path(paths[1]))
    doctored["exam_sha256"] = "deadbeef" * 8
    R.write_run(doctored, Path(paths[1]))
    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        assert R.main(["--check", str(doc)] + paths) == 1, buf.getvalue()
    assert "stores exam sha256" in buf.getvalue()
