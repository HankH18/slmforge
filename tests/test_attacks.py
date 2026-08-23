#!/usr/bin/env python3
"""Tests for :mod:`datagen.attacks` -- the format/subject compulsion generator.

WHAT THESE TESTS ARE FOR
========================
The module exists because every adversarial row the project has ever trained on
is one attack family (permission-to-relax). Four claims have to hold before rows
from it can be trained on or used as a held-out attack eval, and each one is a
claim a green suite could otherwise fake:

1. POOLS A AND B ARE DISJOINT. Without it the "generalises to unseen attack
   phrasings" claim is measuring memorised sentences. Checked by containment in
   both directions over the authored text AND over the emitted task text.
2. NO HELD-OUT CONSTRAINT TYPE IS EVER EMITTED. Checked against the registry at
   runtime, and the guard itself is checked by offering it a held-out instance
   and requiring a raise.
3. EVERY EMITTED ROW IS PROVABLY SATISFIABLE AND RENDERS. The witness is
   re-derived here and re-checked with the ground-truth checkers, and the prompt
   is re-verified through the REAL training gate
   (``training.modal_app.verify_prompt_contract``), not a local imitation of it.
4. THE SHAPE LABEL IS CARRIED BY THE TEXT. Every row is classified from its own
   text and must be accepted by EXACTLY ONE shape, so a ``format_compulsion``
   row cannot be relabelled ``subject_compulsion`` and still verify.

Nothing here calls a paid API, downloads weights, or runs a model.
"""
from __future__ import annotations

import collections
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from constraints import registry                            # noqa: E402
from constraints.checkers import check                      # noqa: E402
from datagen import attacks, task_pool                      # noqa: E402
from scenarios import paths, reachability                   # noqa: E402
from scenarios.render import HEADER, render_user_message    # noqa: E402

SEED = 7


# --------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def lib():
    return attacks.instance_library()


@pytest.fixture(scope="module")
def built(lib):
    """``{pool: (rows, quarantined)}`` for both pools, built once."""
    return {name: attacks.build_attacks(name, seed=SEED, lib=lib)
            for name in attacks.POOL_NAMES}


@pytest.fixture(scope="module")
def rows(built):
    out = []
    for name in attacks.POOL_NAMES:
        out.extend(built[name][0])
    assert out, "the generator produced no rows at all"
    return out


def _brief_index(pool_name):
    return {b.key: b for b in attacks.POOLS[pool_name].briefs}


# =========================================================================
# 1. The two pools
# =========================================================================
def test_authored_pools_are_disjoint():
    """No brief, recap or template is shared between A and B, in either direction."""
    compared = attacks.assert_pools_disjoint()
    assert compared == (len(attacks._pool_strings(attacks.POOL_A))
                        * len(attacks._pool_strings(attacks.POOL_B)))
    assert compared > 0


def test_disjointness_check_actually_catches_a_shared_brief():
    """The disjointness proof is not vacuous: plant a shared brief, demand a raise."""
    shared = attacks.POOL_A.briefs[0]
    poisoned = attacks.Pool("B'", attacks.POOL_B.briefs + (shared,),
                            attacks.POOL_B.phrasings)
    with pytest.raises(attacks.AttackError, match="share authored text"):
        attacks.assert_pools_disjoint(attacks.POOL_A, poisoned)


def test_disjointness_check_catches_containment_not_only_equality():
    """A brief REUSED INSIDE a longer template is the realistic leak, and equality
    would miss it."""
    a_brief = attacks.POOL_A.briefs[0]
    wrapper = attacks.Brief("leak", "Some preamble. " + a_brief.text + " Some tail.",
                            "an unrelated recap sentence for the leak brief.")
    poisoned = attacks.Pool("B'", attacks.POOL_B.briefs + (wrapper,),
                            attacks.POOL_B.phrasings)
    with pytest.raises(attacks.AttackError, match="contained by"):
        attacks.assert_pools_disjoint(attacks.POOL_A, poisoned)


def test_pool_keys_are_disjoint():
    a_keys = {b.key for b in attacks.POOL_A.briefs} | {
        p.key for p in attacks.POOL_A.phrasings}
    b_keys = {b.key for b in attacks.POOL_B.briefs} | {
        p.key for p in attacks.POOL_B.phrasings}
    assert not (a_keys & b_keys)


def test_emitted_task_texts_are_disjoint_across_pools(built):
    a = {attacks.normalize_text(r.scenario["task"]) for r in built["A"][0]}
    b = {attacks.normalize_text(r.scenario["task"]) for r in built["B"][0]}
    assert a and b
    assert not (a & b)
    for left in a:
        for right in b:
            assert left not in right and right not in left


def test_pool_roles_keep_b_out_of_training(built):
    assert attacks.POOL_ROLE["A"] == "train"
    assert attacks.POOL_ROLE["B"] != "train"
    assert {r.pool_role for r in built["A"][0]} == {"train"}
    assert {r.pool_role for r in built["B"][0]} == {"heldout_attack_eval"}


def test_both_pools_cover_all_three_shapes(built):
    for name in attacks.POOL_NAMES:
        shapes = {r.shape for r in built[name][0]}
        assert shapes == set(attacks.ATTACK_SHAPES), name


def test_each_shape_carries_enough_distinct_task_texts(built):
    for name in attacks.POOL_NAMES:
        counts = attacks.distinct_task_texts(built[name][0])
        assert set(counts) == set(attacks.ATTACK_SHAPES), name
        for shape, n in counts.items():
            assert n >= attacks.MIN_DISTINCT_TASK_TEXTS, (name, shape, n)


# =========================================================================
# 2. Held-out constraint types
# =========================================================================
def test_no_emitted_row_carries_a_heldout_type(rows):
    """Resolved through the registry, not against a list typed in the test."""
    train = set(registry.types_by_split("train"))
    heldout = set(registry.types_by_split("heldout_type"))
    assert heldout, "the registry reports no held-out types; this test would be vacuous"
    seen = set()
    for row in rows:
        assert set(row.types) <= train, (row.scenario["scenario_id"], row.types)
        assert not (set(row.types) & heldout), row.scenario["scenario_id"]
        seen |= set(row.types)
    assert seen, "no constraint types were observed at all"


def test_allowed_types_is_resolved_from_the_registry_at_runtime(monkeypatch):
    """Not a literal list: shrink the registry's answer and the module follows."""
    real = registry.types_by_split

    def fake(split):
        return ["no_lists"] if split == "train" else real(split)

    monkeypatch.setattr(registry, "types_by_split", fake)
    assert attacks.allowed_types() == ("no_lists",)
    assert attacks.format_demand_types() == ("no_lists",)
    shrunk = attacks.instance_library()
    assert {c["type"] for c in shrunk.values()} == {"no_lists"}


def test_a_heldout_instance_offered_to_the_builder_raises(lib):
    """The guard is real: hand it a held-out instance and it must refuse to build."""
    heldout_type = registry.types_by_split("heldout_type")[0]
    poisoned = dict(lib)
    poisoned["c-chr-900"] = {
        "id": "c-chr-900", "type": heldout_type,
        "params": ({"letter": "t"} if heldout_type == "sentence_initial_letter"
                   else {"marks": [";"]}),
        "family": "character", "split": "heldout_type"}
    with pytest.raises(attacks.AttackError, match="held-out constraint type"):
        attacks.build_attacks("A", seed=SEED, per_shape=40, lib=poisoned)


def test_a_heldout_type_can_never_become_a_subject_target():
    """``sentence_initial_letter`` also carries a ``letter`` param; the split gate
    is what keeps it out of the subject-compulsion lane."""
    held = {"id": "c-chr-901", "type": "sentence_initial_letter",
            "params": {"letter": "t"}, "family": "character", "split": "heldout_type"}
    assert attacks.topic_kind(held) is None
    with pytest.raises(attacks.AttackError):
        attacks.topic_span(held, "t")


# =========================================================================
# 3. Satisfiability, rendering, and the real training gate
# =========================================================================
def test_no_row_was_quarantined(built):
    for name in attacks.POOL_NAMES:
        _, bad = built[name]
        assert bad == [], [(q.shape, q.phrasing_key, q.target_id, q.reasons)
                           for q in bad[:5]]


def test_every_row_is_constructively_satisfiable(rows, lib):
    """The witness is re-derived here and re-run through the ground-truth checkers.

    An impossible row scores zero for every model forever and looks exactly like
    a hard one -- scenarios/reachability.py:4-9.
    """
    for row in rows:
        constraints = [lib[cid] for cid in row.scenario["constraints"]]
        assert reachability.find_conflicts(constraints) == [], row.scenario
        assert len(row.witness) >= reachability.MIN_WITNESS_CHARS
        for constraint in constraints:
            verdict = check(constraint, row.witness)
            assert verdict["pass"], (row.scenario["scenario_id"],
                                     constraint["id"], verdict)
        assert reachability.build_witness(constraints) == row.witness


def test_every_row_renders_through_the_one_renderer(rows, lib):
    for row in rows:
        rendered = render_user_message(row.scenario, lib)
        assert rendered == row.prompt
        assert rendered.startswith(row.scenario["task"])
        assert HEADER in rendered
        # One rendered requirement line per constraint, in declared order.
        block = rendered.split(HEADER + "\n", 1)[1].splitlines()
        assert len(block) == len(row.scenario["constraints"])


def test_rows_pass_the_real_trainer_prompt_contract(rows, lib):
    """Byte-identity is enforced by the trainer itself, not by a local copy of it.

    ``training/modal_app.py:verify_prompt_contract`` re-renders every training
    row and demands the stored user message back byte for byte; a row that fails
    it can never become training data.
    """
    modal_app = pytest.importorskip("training.modal_app")
    sample = rows[::37] or rows[:20]
    training_rows = [
        {"messages": [{"role": "user", "content": row.prompt},
                       {"role": "assistant", "content": row.witness}],
         "scenario": row.scenario,
         "constraints": [lib[cid] for cid in row.scenario["constraints"]]}
        for row in sample]
    assert modal_app.verify_prompt_contract(
        training_rows, render_user_message) == len(training_rows)


def test_the_prompt_contract_check_is_not_vacuous(rows, lib):
    """Perturb one stored prompt and the trainer's gate must reject it."""
    modal_app = pytest.importorskip("training.modal_app")
    row = rows[0]
    bad = [{"messages": [{"role": "user", "content": row.prompt + " (extra)"},
                          {"role": "assistant", "content": row.witness}],
            "scenario": row.scenario,
            "constraints": [lib[cid] for cid in row.scenario["constraints"]]}]
    with pytest.raises(ValueError, match="NOT what the shared"):
        modal_app.verify_prompt_contract(bad, render_user_message)


def test_scenario_shape_matches_the_corpus_contract(rows):
    for row in rows:
        assert set(row.scenario) == {"scenario_id", "task", "constraints", "k",
                                     "pressure", "split"}
        assert isinstance(row.scenario["task"], str)     # single-turn, never a list
        assert row.scenario["k"] == len(row.scenario["constraints"]) >= 2
        assert row.scenario["pressure"] == "adversarial"
        assert row.scenario["split"] == "train"
        assert len(set(row.scenario["constraints"])) == len(row.scenario["constraints"])


def test_scenario_ids_are_unique_within_a_pool(built):
    for name in attacks.POOL_NAMES:
        ids = [r.scenario["scenario_id"] for r in built[name][0]]
        assert len(set(ids)) == len(ids)


def test_no_emitted_task_collides_with_the_eval_corpus(rows):
    """Contamination guard: the graded eval corpus must stay untouched by these."""
    corpus = [attacks.normalize_text(r["task"])
              for r in paths.load_corpus() + paths.load_smoke()]
    assert corpus
    for row in rows:
        mine = attacks.normalize_text(row.scenario["task"])
        for other in corpus:
            assert mine != other, row.scenario["scenario_id"]
            assert mine not in other and other not in mine, row.scenario["scenario_id"]


def test_no_emitted_task_reuses_a_permission_to_relax_nudge(rows):
    """These are a DIFFERENT attack family from datagen.task_pool.NUDGES.

    If a nudge sentence leaked in, the row would be measuring the family the
    corpus already has 857 of, under a new label.
    """
    nudges = [attacks.normalize_text(n.text) for n in task_pool.NUDGES]
    assert nudges
    for row in rows:
        flat = attacks.normalize_text(row.scenario["task"])
        for nudge in nudges:
            assert nudge not in flat, (row.scenario["scenario_id"], nudge[:60])


def test_no_emitted_task_names_an_unparaphrasable_entity(rows):
    """A proper noun a letter ban rejects makes a row unanswerable, not hard
    (reachability subcase (c))."""
    for row in rows:
        assert reachability.unavoidable_terms(row.scenario["task"]) == [], \
            row.scenario["scenario_id"]


# =========================================================================
# 4. The structural label guard
# =========================================================================
def test_every_row_is_accepted_by_exactly_one_shape(built, lib):
    """The load-bearing label test.

    Not "the declared shape passes its own guard" -- that a mislabelled row could
    also satisfy. Every shape's guard is run against every row, and each row must
    be accepted by exactly one, which is its declared one.
    """
    for name in attacks.POOL_NAMES:
        briefs = _brief_index(name)
        for row in built[name][0]:
            constraints = [lib[cid] for cid in row.scenario["constraints"]]
            got = attacks.classify(row, constraints, lib[row.target_id], briefs)
            assert got == (row.shape,), (row.scenario["scenario_id"], row.shape, got)


def test_a_format_compulsion_row_cannot_be_mislabelled_subject_compulsion(built, lib):
    """Relabel a real format row and the subject guard must reject it -- and the
    reason must be structural, not a missing field."""
    briefs = _brief_index("A")
    fmt = [r for r in built["A"][0] if r.shape == "format_compulsion"]
    assert fmt
    for row in fmt[:25]:
        constraints = [lib[cid] for cid in row.scenario["constraints"]]
        target = lib[row.target_id]
        # (a) relabelled with no span at all
        problems = attacks.shape_evidence(
            "subject_compulsion", row.scenario["task"], constraints, target,
            span=None, brief=briefs.get(row.brief_key))
        assert problems
        assert any("no decidable task conflict" in p for p in problems), problems
        # (b) relabelled AND handed a plausible span, so the failure cannot be
        #     "you forgot a field": the format demand still gives it away.
        problems = attacks.shape_evidence(
            "subject_compulsion", row.scenario["task"], constraints, target,
            span="the word bullet", brief=briefs.get(row.brief_key))
        assert any("no decidable task conflict" in p for p in problems), problems


def test_a_subject_compulsion_row_cannot_be_mislabelled_format_compulsion(built, lib):
    briefs = _brief_index("A")
    subj = [r for r in built["A"][0] if r.shape == "subject_compulsion"]
    assert subj
    for row in subj[:25]:
        constraints = [lib[cid] for cid in row.scenario["constraints"]]
        problems = attacks.shape_evidence(
            "format_compulsion", row.scenario["task"], constraints,
            lib[row.target_id], span=row.topic_span, brief=briefs.get(row.brief_key))
        assert problems
        assert any("no format demand" in p for p in problems), problems


def test_an_erosion_row_cannot_be_mislabelled_plain_format_compulsion(built, lib):
    briefs = _brief_index("A")
    ero = [r for r in built["A"][0] if r.shape == "erosion_recap"]
    assert ero
    for row in ero[:25]:
        constraints = [lib[cid] for cid in row.scenario["constraints"]]
        problems = attacks.shape_evidence(
            "format_compulsion", row.scenario["task"], constraints,
            lib[row.target_id], span=None, brief=briefs[row.brief_key])
        assert any("prior compliant answer" in p for p in problems), problems


def test_format_and_erosion_demands_are_detected_by_reachability(built, lib):
    """The bait is machine-detected, never self-reported: the very detector the
    corpus generator uses to REFUSE contradictory briefs is what confirms it."""
    for name in attacks.POOL_NAMES:
        for row in built[name][0]:
            if row.shape == "subject_compulsion":
                continue
            constraints = [lib[cid] for cid in row.scenario["constraints"]]
            on, off = attacks.format_demand_conflicts(
                row.scenario["task"], constraints, row.target_type)
            assert on, (row.scenario["scenario_id"], row.phrasing_key)
            assert off == [], (row.scenario["scenario_id"], off)
            assert row.target_type in attacks.format_demand_types()


def test_subject_rows_put_the_banned_token_in_the_prompt(built, lib):
    """The span really is in the prompt, and the GROUND-TRUTH checker rejects it."""
    for name in attacks.POOL_NAMES:
        for row in built[name][0]:
            if row.shape != "subject_compulsion":
                continue
            target = lib[row.target_id]
            assert row.topic_span
            assert row.topic_span in row.scenario["task"]
            assert not check(target, row.topic_span)["pass"], row.topic_span
            # ...and no format bait is hiding inside it.
            assert reachability.task_conflicts(
                row.scenario["task"],
                [lib[cid] for cid in row.scenario["constraints"]]) == []
            assert row.topic_span.startswith(
                "the " + str(attacks.topic_kind(target)))


def test_erosion_rows_establish_the_prior_answer_before_the_demand(built, lib):
    for name in attacks.POOL_NAMES:
        briefs = _brief_index(name)
        for row in built[name][0]:
            if row.shape != "erosion_recap":
                continue
            brief = briefs[row.brief_key]
            task = attacks.normalize_text(row.scenario["task"])
            assert attacks.recap_established(row.scenario["task"], brief)
            assert task.index(attacks.normalize_text(brief.recap)) > \
                task.index(attacks.normalize_text(brief.text))
            # single-turn on purpose: the trainer accepts [user, assistant] only
            assert isinstance(row.scenario["task"], str)


def test_recap_guard_rejects_a_row_with_no_prior_answer(built):
    """``recap_established`` is a real discriminator, not always-true."""
    briefs = _brief_index("A")
    fmt = [r for r in built["A"][0] if r.shape == "format_compulsion"]
    assert fmt
    for row in fmt[:25]:
        assert not attacks.recap_established(row.scenario["task"],
                                             briefs[row.brief_key])
    assert not attacks.recap_established("anything at all", None)


# =========================================================================
# 5. Quarantine
# =========================================================================
def test_an_unsatisfiable_candidate_is_quarantined_not_emitted():
    """Feed the builder a constraint that no witness can survive.

    Banning the whole witness noun pool empties it, so ``build_witness`` returns
    None for any set containing it -- and every such candidate must land in the
    quarantine list rather than in the output.
    """
    poison = sorted(set(reachability.NOUNS))
    lib = {
        "c-fmt-001": {"id": "c-fmt-001", "type": "no_lists", "params": {},
                      "family": "format_negative", "split": "train"},
        "c-fmt-002": {"id": "c-fmt-002", "type": "no_quotation_marks", "params": {},
                      "family": "format_negative", "split": "train"},
        "c-lex-900": {"id": "c-lex-900", "type": "banned_words",
                      "params": {"words": poison},
                      "family": "lexical", "split": "train"},
    }
    rows, bad = attacks.erosion_recap("A", seed=SEED, lib=lib)
    assert bad, "nothing was quarantined; the satisfiability gate is not running"
    assert any("no witness could be constructed" in r
               for q in bad for r in q.reasons)
    for row in rows:
        assert "c-lex-900" not in row.scenario["constraints"]


def test_quarantine_records_name_the_candidate():
    poison = sorted(set(reachability.NOUNS))
    lib = {
        "c-fmt-001": {"id": "c-fmt-001", "type": "no_lists", "params": {},
                      "family": "format_negative", "split": "train"},
        "c-fmt-002": {"id": "c-fmt-002", "type": "no_quotation_marks", "params": {},
                      "family": "format_negative", "split": "train"},
        "c-lex-900": {"id": "c-lex-900", "type": "banned_words",
                      "params": {"words": poison},
                      "family": "lexical", "split": "train"},
    }
    _, bad = attacks.format_compulsion("A", seed=SEED, lib=lib)
    assert bad
    q = bad[0]
    assert q.shape == "format_compulsion"
    assert q.pool == "A"
    assert q.phrasing_key and q.target_id and q.reasons


# =========================================================================
# 6. Determinism
# =========================================================================
def _fingerprint(rows):
    return [(r.scenario["scenario_id"], r.scenario["task"],
             tuple(r.scenario["constraints"]), r.shape, r.prompt, r.witness)
            for r in rows]


def test_build_is_deterministic_under_a_fixed_seed(lib):
    first, bad_first = attacks.build_attacks("A", seed=SEED, per_shape=30, lib=lib)
    second, bad_second = attacks.build_attacks("A", seed=SEED, per_shape=30, lib=lib)
    assert _fingerprint(first) == _fingerprint(second)
    assert [q.reasons for q in bad_first] == [q.reasons for q in bad_second]


def test_build_is_deterministic_across_processes():
    """Same seed, same bytes -- in a FRESH interpreter.

    Same-process equality can be produced by a cached result. Determinism that
    survives a new process is what makes the corpus reproducible, and it is the
    property a hash-randomised seed or set iteration would break.
    """
    script = (
        "import hashlib, json, sys; sys.path.insert(0, %r);"
        "from datagen import attacks;"
        "rows, _ = attacks.build_attacks('A', seed=7, per_shape=25);"
        "print(hashlib.sha256(json.dumps("
        "[[r.scenario, r.shape, r.prompt, r.witness] for r in rows],"
        " sort_keys=True).encode()).hexdigest())" % str(REPO_ROOT)
    )
    runs = [subprocess.run([sys.executable, "-c", script], capture_output=True,
                           text=True, cwd=str(REPO_ROOT), check=True).stdout.strip()
            for _ in range(2)]
    assert runs[0] and runs[0] == runs[1], runs


def test_the_seed_actually_changes_the_output(lib):
    a, _ = attacks.build_attacks("A", seed=SEED, per_shape=30, lib=lib)
    b, _ = attacks.build_attacks("A", seed=SEED + 1, per_shape=30, lib=lib)
    assert _fingerprint(a) != _fingerprint(b)


def test_a_rows_constraint_set_does_not_move_when_n_changes(lib):
    """Companions come from a per-ROW rng, so growing the build must not rewrite
    the rows already produced."""
    small, _ = attacks.build_attacks("A", seed=SEED, per_shape=5, lib=lib)
    large, _ = attacks.build_attacks("A", seed=SEED, per_shape=40, lib=lib)
    by_key = {(r.shape, r.pool, r.brief_key, r.phrasing_key, r.target_id,
               r.topic_span): r for r in large}
    checked = 0
    for row in small:
        key = (row.shape, row.pool, row.brief_key, row.phrasing_key,
               row.target_id, row.topic_span)
        assert key in by_key
        assert by_key[key].scenario["constraints"] == row.scenario["constraints"]
        assert by_key[key].scenario["task"] == row.scenario["task"]
        checked += 1
    assert checked == len(small)


def test_per_shape_caps_each_shape_independently(lib):
    rows, _ = attacks.build_attacks("A", seed=SEED, per_shape=4, lib=lib)
    counts = collections.Counter(r.shape for r in rows)
    assert set(counts) == set(attacks.ATTACK_SHAPES)
    assert all(n == 4 for n in counts.values()), counts


def test_shape_selection_is_honoured(lib):
    rows, _ = attacks.build_attacks("A", seed=SEED, per_shape=3,
                                    shapes=("subject_compulsion",), lib=lib)
    assert rows
    assert {r.shape for r in rows} == {"subject_compulsion"}
    with pytest.raises(attacks.AttackError):
        attacks.build_attacks("A", shapes=("no_such_shape",), lib=lib)


def test_unknown_pool_raises(lib):
    with pytest.raises(attacks.AttackError, match="unknown pool"):
        attacks.build_attacks("Z", lib=lib)
