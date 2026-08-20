"""T-2 acceptance: the constraint instance library and the scenario corpus.

Everything here reads the SHIPPED artefacts under ``data/scenarios/`` and
checks them against the real registry in :mod:`constraints.registry`. Where a
property could be checked either by asking the generator or by recomputing it
from the data, these tests recompute it -- a test that asks the code under test
whether it did the right thing proves nothing.

Three tests deserve their names read carefully:

``test_split_invariant_*``      the one that protects the experiment: a
                                held-out constraint type leaking into a train
                                scenario would make "unseen type" a lie.
``test_every_scenario_is_reachable`` constructs a passing text per scenario and
                                runs the ground-truth checkers on it. An
                                impossible scenario scores zero for every model
                                forever and fakes a ceiling.
``test_guards_actually_guard``  proves the two above can fail. Green from a
                                check that cannot go red is not evidence.
"""
import json
import re

import pytest

from conftest import REPO_ROOT  # noqa: F401  (adds the repo root to sys.path)

from constraints.checkers import CLEAN_EVIDENCE, check
from constraints.registry import (ID_PREFIX_BY_FAMILY, TYPES, ConstraintError,
                                  family_of, list_types, split_of, validate_constraint)
from scenarios import build as build_mod
from scenarios.generator import (ALLOWED_K, MIN_ADVERSARIAL_SHARE, MIN_SCENARIOS,
                                 NEGATIVE_TYPES, POSITIVE_TYPES, GeneratorError,
                                 _build_one, generate)
from scenarios.instances import INSTANCE_META
from scenarios.lexicons import GENERIC, LEXICONS
from scenarios.paths import (CORPUS_PATH, INSTANCES_PATH, MANIFEST_PATH, SMOKE_PATH,
                             load_corpus, load_instances, load_jsonl, load_manifest,
                             load_smoke)
from scenarios.pressure import NUDGES
from scenarios.reachability import (MIN_WITNESS_CHARS, MIN_WITNESS_SENTENCES,
                                    build_witness, find_conflicts, task_conflicts)

SCENARIO_KEYS = {"scenario_id", "task", "constraints", "k", "pressure", "split"}
CONSTRAINT_KEYS = {"id", "type", "params", "family", "split"}
HELDOUT_TYPES = {t for t, spec in TYPES.items() if spec.split == "heldout_type"}


@pytest.fixture(scope="module")
def instances():
    return load_instances()


@pytest.fixture(scope="module")
def corpus():
    return load_corpus()


@pytest.fixture(scope="module")
def manifest():
    return load_manifest()


@pytest.fixture(scope="module")
def by_id(manifest):
    return {entry["scenario_id"]: entry for entry in manifest["scenarios"]}


def _whole_word(term, text):
    pattern = r"\s+".join(re.escape(part) for part in term.split())
    return re.search(r"(?<![^\W_])" + pattern + r"(?![^\W_])", text, re.IGNORECASE)


# ==========================================================================
# The artefacts exist. A missing file is a failure, never a skip -- an absent
# deliverable behind a green suite is exactly the failure mode this repo has
# already been bitten by.
# ==========================================================================

def test_every_deliverable_is_on_disk():
    for path in (INSTANCES_PATH, CORPUS_PATH, SMOKE_PATH, MANIFEST_PATH):
        assert path.exists(), f"{path} is missing: T-2 is not delivered"
        assert path.stat().st_size > 0, f"{path} is empty"


# ==========================================================================
# Deliverable 1 -- the constraint instance library
# ==========================================================================

def test_every_instance_is_schema_valid(instances):
    for cid, constraint in instances.items():
        assert set(constraint) == CONSTRAINT_KEYS, f"{cid}: keys {sorted(constraint)}"
        validate_constraint(constraint)          # raises ConstraintError on drift


def test_instance_family_and_split_come_from_the_type(instances):
    """split is a property of the TYPE, never of an instance (Design, Interfaces)."""
    for cid, constraint in instances.items():
        ctype = constraint["type"]
        assert constraint["family"] == family_of(ctype), f"{cid}: family drifted from registry"
        assert constraint["split"] == split_of(ctype), f"{cid}: split drifted from registry"
        prefix = ID_PREFIX_BY_FAMILY[constraint["family"]]
        assert cid.startswith(prefix), f"{cid}: wrong prefix for family {constraint['family']}"


def test_instance_ids_are_unique_and_params_do_not_repeat():
    rows = load_jsonl(INSTANCES_PATH)
    ids = [row["id"] for row in rows]
    assert len(set(ids)) == len(ids), "duplicate constraint instance id"
    seen = {}
    for row in rows:
        key = (row["type"], json.dumps(row["params"], sort_keys=True))
        assert key not in seen, (f"{row['id']} duplicates {seen.get(key)}: same type and params "
                                 "under two names is two names for one experiment")
        seen[key] = row["id"]


def test_every_registry_type_has_instances(instances):
    """A type with no instance is a type the corpus can never exercise."""
    present = {c["type"] for c in instances.values()}
    assert present == set(list_types()), f"types with no instance: {set(list_types()) - present}"


def test_instance_library_has_a_spread_per_type(instances):
    counts = {}
    for constraint in instances.values():
        counts[constraint["type"]] = counts.get(constraint["type"], 0) + 1
    # The two no-param types can only ever have one instance; the rest are
    # authored with a spread so scenarios are not all the same experiment.
    for ctype in ("banned_words", "banned_phrases", "banned_letter",
                  "banned_punctuation", "sentence_initial_letter"):
        assert counts[ctype] >= 4, f"{ctype}: only {counts.get(ctype)} instance(s)"
    for ctype in ("no_lists", "no_quotation_marks"):
        assert counts[ctype] == 1, f"{ctype} takes no params; a second instance is a duplicate"


def test_banned_letter_instances_cover_distinct_letters(instances):
    letters = [c["params"]["letter"] for c in instances.values()
               if c["type"] == "banned_letter"]
    assert len(set(letters)) == len(letters)
    # A ban only bites if the letter is common enough to appear in ordinary
    # prose. Banning 'q' would pass every checker by accident.
    assert set(letters) <= set("etaoinsrhldcumpfg"), f"low-frequency letters banned: {letters}"


# ==========================================================================
# Deliverable 2 -- the scenario corpus, schema and size
# ==========================================================================

def test_every_scenario_is_schema_valid(corpus):
    for scenario in corpus:
        sid = scenario.get("scenario_id")
        assert set(scenario) == SCENARIO_KEYS, f"{sid}: keys {sorted(scenario)}"
        assert re.fullmatch(r"s-\d{4}", scenario["scenario_id"]), f"bad id {sid!r}"
        assert isinstance(scenario["task"], str) and scenario["task"].strip()
        assert isinstance(scenario["constraints"], list)
        assert all(isinstance(cid, str) for cid in scenario["constraints"])
        assert scenario["pressure"] in ("clean", "adversarial"), sid
        assert scenario["split"] in ("train", "heldout_type"), sid
        assert isinstance(scenario["k"], int) and not isinstance(scenario["k"], bool)


def test_corpus_is_big_enough_and_every_cell_is_usable(corpus):
    assert len(corpus) >= MIN_SCENARIOS, f"{len(corpus)} scenarios, need >= {MIN_SCENARIOS}"
    cells = {}
    for scenario in corpus:
        key = (scenario["split"], scenario["pressure"])
        cells[key] = cells.get(key, 0) + 1
    assert len(cells) == 4, f"missing split x pressure cells: {sorted(cells)}"
    for key, count in sorted(cells.items()):
        assert count >= 30, f"cell {key} has {count} scenarios, need >= 30 after sampling"


def test_k_equals_len_constraints_and_is_in_range(corpus):
    for scenario in corpus:
        assert scenario["k"] == len(scenario["constraints"]), scenario["scenario_id"]
        assert scenario["k"] in ALLOWED_K, f"{scenario['scenario_id']}: k={scenario['k']}"
        assert len(set(scenario["constraints"])) == scenario["k"], "repeated constraint id"


def test_k_covers_two_three_and_four_in_both_splits(corpus):
    for split in ("train", "heldout_type"):
        ks = {s["k"] for s in corpus if s["split"] == split}
        assert set(ALLOWED_K) <= ks, f"{split}: k values {sorted(ks)}; k must not confound split"


def test_no_duplicate_scenario_ids_or_tasks(corpus, by_id):
    ids = [s["scenario_id"] for s in corpus]
    assert len(set(ids)) == len(ids), "duplicate scenario_id"
    tasks = [s["task"] for s in corpus]
    assert len(set(tasks)) == len(tasks), "duplicate task text"
    # ...and the BASE tasks are unique too, so a task cannot be reused by
    # bolting a different pressure sentence onto it.
    bases = [by_id[sid]["base_task"] for sid in ids]
    duplicates = sorted({b for b in bases if bases.count(b) > 1})
    assert not duplicates, f"base task reused across scenarios: {duplicates[:2]}"


def test_every_referenced_constraint_resolves(corpus, instances):
    for scenario in corpus:
        for cid in scenario["constraints"]:
            assert cid in instances, (f"{scenario['scenario_id']} references {cid!r}, "
                                      "which is not in constraint_instances.jsonl")


def test_no_instance_is_dead_weight(corpus, instances):
    used = {cid for scenario in corpus for cid in scenario["constraints"]}
    unused = sorted(set(instances) - used)
    assert not unused, f"instances defined but never used by any scenario: {unused}"


# ==========================================================================
# Spec R2 -- every scenario carries a negative constraint
# ==========================================================================

def test_negative_types_are_derived_from_the_registry():
    assert set(NEGATIVE_TYPES) | set(POSITIVE_TYPES) == set(list_types())
    assert not set(NEGATIVE_TYPES) & set(POSITIVE_TYPES)
    # sentence_initial_letter REQUIRES a form; the other six FORBID something.
    assert POSITIVE_TYPES == ("sentence_initial_letter",)


def test_every_scenario_has_at_least_one_negative_constraint(corpus, instances):
    for scenario in corpus:
        types = [instances[cid]["type"] for cid in scenario["constraints"]]
        negatives = [t for t in types if t in NEGATIVE_TYPES]
        assert negatives, (f"{scenario['scenario_id']} has no negative constraint "
                           f"(types: {types}) -- Spec R2 is the whole target")


# ==========================================================================
# THE SPLIT INVARIANT
# ==========================================================================

def test_split_invariant_no_heldout_type_in_any_train_scenario(corpus, instances):
    """No constraint whose TYPE is held out may appear in a train scenario."""
    offenders = []
    for scenario in corpus:
        if scenario["split"] != "train":
            continue
        for cid in scenario["constraints"]:
            if instances[cid]["type"] in HELDOUT_TYPES:
                offenders.append((scenario["scenario_id"], cid, instances[cid]["type"]))
    assert not offenders, f"held-out types leaked into train scenarios: {offenders}"


def test_split_invariant_stated_the_other_way_round(corpus, instances):
    """The scenario's split is exactly 'does it contain a held-out type'."""
    for scenario in corpus:
        held = any(instances[cid]["split"] == "heldout_type"
                   for cid in scenario["constraints"])
        expected = "heldout_type" if held else "train"
        assert scenario["split"] == expected, (
            f"{scenario['scenario_id']} labelled {scenario['split']!r} but its constraints "
            f"say {expected!r}")


def test_both_splits_are_produced(corpus):
    splits = {s["split"] for s in corpus}
    assert splits == {"train", "heldout_type"}, splits


def test_heldout_scenarios_really_exercise_the_heldout_types(corpus, instances):
    seen = set()
    for scenario in corpus:
        if scenario["split"] != "heldout_type":
            continue
        held = {instances[cid]["type"] for cid in scenario["constraints"]} & HELDOUT_TYPES
        assert held, f"{scenario['scenario_id']} is labelled heldout_type but contains none"
        seen |= held
    assert seen == HELDOUT_TYPES, f"held-out types never exercised: {HELDOUT_TYPES - seen}"


# ==========================================================================
# Pressure: both mechanisms, verified against the shipped text
# ==========================================================================

def test_adversarial_share_meets_the_floor(corpus):
    adversarial = [s for s in corpus if s["pressure"] == "adversarial"]
    share = len(adversarial) / len(corpus)
    assert share >= MIN_ADVERSARIAL_SHARE, f"adversarial share {share:.1%}"


def test_adversarial_scenarios_carry_embedded_pressure(corpus, by_id, instances):
    for scenario in corpus:
        if scenario["pressure"] != "adversarial":
            continue
        entry = by_id[scenario["scenario_id"]]
        embedded = entry["embedded_pressure"]
        assert embedded, f"{scenario['scenario_id']}: adversarial with no embedded pressure"
        assert embedded["nudge_id"] in NUDGES
        # The claim is checkable against the shipped task, not just the manifest.
        assert embedded["text"] in scenario["task"], (
            f"{scenario['scenario_id']}: nudge text is not present in the task")
        assert scenario["task"] == f"{entry['base_task']} {embedded['text']}"
        types = {instances[cid]["type"] for cid in scenario["constraints"]}
        assert set(embedded["targets"]) & types, (
            f"{scenario['scenario_id']}: nudge undermines {embedded['targets']} but the "
            f"scenario has none of those constraint types ({sorted(types)})")


def test_adversarial_scenarios_carry_distractor_pressure(corpus, by_id, instances):
    """Re-derive every distractor claim from the shipped task text."""
    for scenario in corpus:
        if scenario["pressure"] != "adversarial":
            continue
        sid = scenario["scenario_id"]
        entry = by_id[sid]
        base = entry["base_task"]
        mechanisms = entry["distractor_pressure"]
        assert mechanisms, f"{sid}: adversarial with no distractor pressure"
        for mech in mechanisms:
            cid = mech["constraint_id"]
            assert cid in scenario["constraints"], f"{sid}: distractor cites unused {cid}"
            constraint = instances[cid]
            if mech["kind"] == "lexicon_collision":
                lex = mech["lexicon"]
                members = {m.lower() for m in LEXICONS[lex]}
                banned = {w.lower() for w in
                          (constraint["params"].get("words")
                           or constraint["params"].get("phrases"))}
                assert mech["banned_members"], f"{sid}: no banned member named"
                for token in mech["banned_members"]:
                    assert token.lower() in members, f"{sid}: {token!r} is not in lexicon {lex}"
                    assert token.lower() in banned, f"{sid}: {token!r} is not actually banned"
                others = mech["topic_words_in_task"]
                assert len(others) >= 2, f"{sid}: only {others} topic words in the task"
                for word in others:
                    assert word.lower() in members, f"{sid}: {word!r} not in lexicon {lex}"
                    assert word.lower() not in banned, f"{sid}: {word!r} is itself banned"
                    assert _whole_word(word, base), f"{sid}: {word!r} is not in the task text"
            elif mech["kind"] == "key_term_letter":
                letter = mech["letter"]
                assert constraint["params"]["letter"].lower() == letter
                hits = mech["key_terms_carrying_letter"]
                assert len(hits) >= 2, f"{sid}: banned letter appears in only {hits}"
                for term in hits:
                    assert letter in term.lower(), f"{sid}: {term!r} does not carry {letter!r}"
                    assert term.lower() in base.lower(), f"{sid}: {term!r} is not in the task"
            elif mech["kind"] == "punctuation_density":
                total = sum(base.count(mark) for mark in constraint["params"]["marks"])
                assert total >= 3, (f"{sid}: the prompt uses the banned mark(s) only {total} "
                                    "times, which is not a register that invites them")
            else:
                raise AssertionError(f"{sid}: unknown distractor kind {mech['kind']!r}")


def test_clean_scenarios_carry_neither_mechanism(corpus, by_id, instances):
    for scenario in corpus:
        if scenario["pressure"] != "clean":
            continue
        sid = scenario["scenario_id"]
        entry = by_id[sid]
        assert entry["embedded_pressure"] is None, f"{sid}: clean scenario with a nudge"
        assert entry["distractor_pressure"] == [], f"{sid}: clean scenario with a distractor"
        assert scenario["task"] == entry["base_task"], f"{sid}: clean task was modified"
        for nudge in NUDGES.values():
            assert nudge.text not in scenario["task"], f"{sid}: carries nudge text anyway"
        # ...and the pressure could not fire even if someone forgot to record it.
        for cid in scenario["constraints"]:
            constraint = instances[cid]
            if constraint["type"] == "banned_punctuation":
                total = sum(entry["base_task"].count(m) for m in constraint["params"]["marks"])
                assert total < 3, f"{sid}: clean prompt is dense in the banned mark ({total})"
            assert INSTANCE_META[cid].lexicon == GENERIC, (
                f"{sid}: clean scenario uses topic-tagged constraint {cid}")


def test_both_pressure_mechanisms_are_used_across_the_corpus(by_id):
    kinds = {mech["kind"] for entry in by_id.values()
             for mech in entry["distractor_pressure"]}
    assert {"lexicon_collision", "key_term_letter", "punctuation_density"} <= kinds
    nudges = {entry["embedded_pressure"]["nudge_id"] for entry in by_id.values()
              if entry["embedded_pressure"]}
    assert nudges == set(NUDGES), f"nudges never used: {sorted(set(NUDGES) - nudges)}"


def test_topic_tagged_constraints_only_appear_on_that_topic(corpus, by_id):
    """The anti-'ban zygote in a transit essay' rule, checked on the corpus."""
    for scenario in corpus:
        entry = by_id[scenario["scenario_id"]]
        for cid in scenario["constraints"]:
            tag = INSTANCE_META[cid].lexicon
            if tag == GENERIC:
                continue
            assert tag == entry["lexicon"], (
                f"{scenario['scenario_id']}: {cid} is a {tag!r} constraint on a "
                f"{entry['lexicon']!r} task, so it could never bite")


# ==========================================================================
# REACHABILITY -- no scenario is impossible or vacuously unsatisfiable
# ==========================================================================

def test_no_scenario_has_an_analytic_conflict(corpus, instances):
    for scenario in corpus:
        constraints = [instances[cid] for cid in scenario["constraints"]]
        conflicts = find_conflicts(constraints)
        assert not conflicts, f"{scenario['scenario_id']}: {conflicts}"


def test_every_scenario_is_reachable(corpus, instances, by_id):
    """Construct a passing text per scenario and let the REAL checkers judge it."""
    for scenario in corpus:
        sid = scenario["scenario_id"]
        constraints = [instances[cid] for cid in scenario["constraints"]]
        witness = build_witness(constraints)
        assert witness is not None, (
            f"{sid}: no text could be constructed that satisfies "
            f"{scenario['constraints']}; an impossible scenario scores zero for every "
            "model forever and fakes a ceiling")
        # Not a vacuous proof: empty text passes every checker (ruling A7).
        assert len(witness) >= MIN_WITNESS_CHARS, f"{sid}: witness too short to prove anything"
        assert witness.count(".") + witness.count("!") + witness.count("?") \
            >= MIN_WITNESS_SENTENCES, f"{sid}: witness is not multi-sentence"
        for constraint in constraints:
            result = check(constraint, witness)
            assert result["pass"], (f"{sid}: witness violates {constraint['id']}: "
                                    f"{result['evidence']}")
            assert result["evidence"] == CLEAN_EVIDENCE
        # ...and it matches the witness recorded at build time (determinism).
        assert witness == by_id[sid]["reachability_witness"], f"{sid}: witness is not stable"


def test_the_empty_response_is_why_the_witness_floors_exist(instances):
    """A7: empty text passes every checker, so a short witness proves nothing."""
    constraints = [instances["c-chr-001"], instances["c-fmt-001"]]
    for constraint in constraints:
        assert check(constraint, "")["pass"], "A7 changed; revisit the witness floors"
    assert MIN_WITNESS_CHARS >= 100 and MIN_WITNESS_SENTENCES >= 4


def test_no_task_demands_what_its_constraints_forbid(corpus, instances, by_id):
    """A brief asking for bullets under no_lists is a contradiction, not a hard task.

    Checked on the BASE task: an adversarial nudge that invites bullets is the
    embedded pressure working as designed, and must not be confused with a
    self-contradictory brief.
    """
    for scenario in corpus:
        constraints = [instances[cid] for cid in scenario["constraints"]]
        base = by_id[scenario["scenario_id"]]["base_task"]
        problems = task_conflicts(base, constraints)
        assert not problems, f"{scenario['scenario_id']}: {problems}"


# ==========================================================================
# Deliverable 3 -- the smoke set
# ==========================================================================

def test_smoke_set_is_five_scenarios_drawn_from_the_corpus(corpus):
    smoke = load_smoke()
    assert len(smoke) == 5, f"smoke set has {len(smoke)} scenarios"
    corpus_by_id = {s["scenario_id"]: s for s in corpus}
    for scenario in smoke:
        sid = scenario["scenario_id"]
        assert sid in corpus_by_id, f"{sid} is not a corpus scenario"
        assert scenario == corpus_by_id[sid], f"{sid} has drifted from the corpus row"


def test_smoke_set_mixes_k_and_pressure():
    smoke = load_smoke()
    assert len({s["k"] for s in smoke}) >= 2, "smoke set does not mix k"
    assert {s["pressure"] for s in smoke} == {"clean", "adversarial"}
    assert {s["split"] for s in smoke} == {"train", "heldout_type"}


# ==========================================================================
# Manifest / build integrity
# ==========================================================================

def test_manifest_describes_every_scenario(corpus, by_id, manifest):
    assert set(by_id) == {s["scenario_id"] for s in corpus}
    summary = manifest["summary"]
    assert summary["scenarios"] == len(corpus)
    counted = {}
    for scenario in corpus:
        counted[str(scenario["k"])] = counted.get(str(scenario["k"]), 0) + 1
    assert summary["k_distribution"] == counted, "the stated k distribution is not the real one"


def test_shipped_artefacts_match_a_fresh_build():
    """Catches a hand-edited JSONL row: the pipeline is deterministic."""
    for path, text in build_mod.render().items():
        assert path.read_text(encoding="utf-8") == text, (
            f"{path} differs from a regeneration; rerun python -m scenarios.build")


# ==========================================================================
# The guards actually guard. Without these, every assertion above could be
# passing because it is incapable of failing.
# ==========================================================================

def _spec(**overrides):
    spec = {"id": "rig", "text": "Write a paragraph about a harbour at dawn.",
            "domain": "test", "register": "prose",
            "constraints": ["c-chr-001", "c-fmt-001"], "pressure": "clean",
            "lexicon": None, "key_terms": (), "nudge": None}
    spec.update(overrides)
    return spec


def test_guards_actually_guard_impossible_constraint_sets(instances):
    """Ban the letter e, and demand every sentence start with E."""
    impossible = [instances["c-chr-001"], instances["c-str-007"]]
    conflicts = find_conflicts(impossible)
    assert conflicts, "the analytic detector missed an outright contradiction"
    assert "c-chr-001" in " ".join(conflicts) and "c-str-007" in " ".join(conflicts)
    assert build_witness(impossible) is None, "a witness was built for an impossible set"
    with pytest.raises(GeneratorError, match="unreachable"):
        _build_one(1, _spec(constraints=["c-chr-001", "c-str-007"]))


def test_guards_actually_guard_the_negative_constraint_rule(instances):
    """A scenario made only of the one positive type must be refused.

    Worth stating plainly rather than hiding behind a green assertion: with the
    PINNED seven-type registry, Spec R2 cannot actually be violated by a valid
    scenario. Only one type is positive, k >= 2, and two instances of the same
    structural type in one scenario are refused -- so every k >= 2 scenario
    contains a negative constraint by construction. That makes the corpus-level
    R2 assertion true but easy, so what is tested here is that the PREDICATE
    discriminates (it returns nothing for an all-positive set) and that such a
    set cannot be built at all. If a future registry adds a second positive
    type, the predicate is already correct and the corpus test starts biting.
    """
    from scenarios.generator import negative_constraints
    all_positive = [instances["c-str-001"], instances["c-str-002"]]
    assert negative_constraints(all_positive) == [], "the R2 predicate does not discriminate"
    assert [instances["c-fmt-001"]] and negative_constraints(
        [instances["c-fmt-001"], instances["c-str-001"]]) == ["c-fmt-001"]
    with pytest.raises(GeneratorError):
        _build_one(1, _spec(constraints=["c-str-001", "c-str-002"]))


def test_guards_actually_guard_k(instances):
    with pytest.raises(GeneratorError, match="k=1"):
        _build_one(1, _spec(constraints=["c-fmt-001"]))
    with pytest.raises(GeneratorError, match="k=5"):
        _build_one(1, _spec(constraints=["c-fmt-001", "c-fmt-002", "c-lex-025",
                                         "c-lex-026", "c-lex-027"]))


def test_guards_actually_guard_off_topic_word_bans():
    """Banning coffee words in a transit task must be refused, not shipped."""
    with pytest.raises(GeneratorError, match="never bite"):
        _build_one(1, _spec(constraints=["c-lex-001", "c-fmt-001"], lexicon="transit"))


def test_guards_actually_guard_pressure_labelling():
    with pytest.raises(GeneratorError, match="no distractor pressure"):
        _build_one(1, _spec(pressure="adversarial", nudge="letter_audit"))
    with pytest.raises(GeneratorError, match="clean scenarios carry no nudge"):
        _build_one(1, _spec(pressure="clean", nudge="letter_audit"))


def test_guards_actually_guard_task_constraint_contradiction():
    with pytest.raises(GeneratorError, match="contradicts"):
        _build_one(1, _spec(text="Give me the answer as a list of five bullet points.",
                            constraints=["c-fmt-001", "c-lex-025"]))


def test_guards_actually_guard_unknown_constraint_ids():
    with pytest.raises(GeneratorError, match="not in the instance library"):
        _build_one(1, _spec(constraints=["c-lex-999", "c-fmt-001"]))


def test_guards_actually_guard_undeclared_key_terms():
    with pytest.raises(GeneratorError, match="does not appear in the task text"):
        _build_one(1, _spec(pressure="adversarial", nudge="letter_audit",
                            key_terms=("nonexistent phrase", "another one")))


def test_a_malformed_constraint_is_rejected_by_the_registry():
    """The instance library cannot smuggle a per-instance split past the registry."""
    with pytest.raises(ConstraintError):
        validate_constraint({"id": "c-fmt-001", "type": "no_lists", "params": {},
                             "family": "format_negative", "split": "heldout_type"})


def test_generate_is_all_or_nothing():
    """The generator never quietly drops a bad scenario: it raises."""
    from scenarios.tasks import TASKS
    broken = [dict(TASKS[0], constraints=["c-str-001"], pressure="clean", nudge=None,
                   key_terms=())] + [dict(t) for t in TASKS[1:]]
    with pytest.raises(GeneratorError):
        generate(broken)
