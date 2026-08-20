"""T-2 acceptance: the constraint instance library and the scenario corpus.

Everything here reads the SHIPPED artefacts under ``data/scenarios/`` and
checks them against the real registry in :mod:`constraints.registry`. Where a
property could be checked either by asking the generator or by recomputing it
from the data, these tests recompute it -- a test that asks the code under test
whether it did the right thing proves nothing.

These deserve their names read carefully:

``test_split_invariant_*``      the one that protects the experiment: a
                                held-out constraint type leaking into a train
                                scenario would make "unseen type" a lie.
``test_every_scenario_is_reachable`` constructs a passing text per scenario and
                                runs the ground-truth checkers on it. An
                                impossible scenario scores zero for every model
                                forever and fakes a ceiling.
``test_task_conflicts_catches_an_unparaphrasable_name`` the other half of
                                reachability, and the half that used to be a
                                keyword list: a brief that names an entity its
                                own constraints reject. Decided by the real
                                checkers on the bare name.
``test_*_derived_floor*`` / ``test_*_is_a_rate*`` every distractor mechanism
                                clears a floor recomputed from the SHIPPED task
                                text. An author can attempt an adversarial
                                label; only the text can grant it.
``test_every_quarantined_task_still_reproduces_its_recorded_defect`` re-proves
                                every scenario held out of the corpus. Without
                                it the quarantine would be a way to delete an
                                inconvenient scenario with prose as evidence.
``test_guards_actually_guard``  proves the above can fail. Green from a
                                check that cannot go red is not evidence.
"""
import json
import re
from statistics import mean

import pytest

from conftest import REPO_ROOT  # noqa: F401  (adds the repo root to sys.path)

from constraints.checkers import CLEAN_EVIDENCE, check
from constraints.registry import (FAMILIES, ID_PREFIX_BY_FAMILY, TYPES, ConstraintError,
                                  family_of, list_types, split_of, validate_constraint)
from scenarios import build as build_mod
from scenarios.generator import (ALLOWED_K, CHARACTER_FAMILY, CHARACTER_TYPES,
                                 MIN_ADVERSARIAL_SHARE, MIN_BANNED_LETTER_SHARE,
                                 MIN_CHARACTER_CONSTRAINT_SHARE,
                                 MIN_NON_CHARACTER_CONTROL,
                                 MIN_KEY_TERM_HITS,
                                 MIN_LETTER_CONTENT_SHARE, MIN_LEXICON_WITNESSES,
                                 MIN_MARK_OCCURRENCES, MIN_MARK_RATE_PER_100_CHARS,
                                 MIN_SCENARIOS, NEGATIVE_TYPES, POSITIVE_TYPES, QUARANTINE,
                                 GeneratorError, _build_one, _corpus_invariants,
                                 banned_letter_of, banned_letter_share,
                                 carries_banned_letter,
                                 character_constraint_share, distractor_mechanisms,
                                 dominant_lexicons, generate, letter_content_share,
                                 letter_distribution, summarise)
import scenarios.generator as gen_mod
import scenarios.instances as inst_mod
from scenarios.instances import (INSTANCE_META, PERMITTED_BANNED_LETTERS,
                                 RETIRED_LETTER_INSTANCES, InstanceSpec,
                                 banned_letters_in_library)
from scenarios.lexicons import GENERIC, LEXICONS
from scenarios.paths import (CORPUS_PATH, INSTANCES_PATH, MANIFEST_PATH, SMOKE_PATH,
                             load_corpus, load_instances, load_jsonl, load_manifest,
                             load_smoke)
from scenarios.pressure import NUDGES
from scenarios.render import render_user_message
from scenarios.reachability import (MIN_WITNESS_CHARS, MIN_WITNESS_SENTENCES,
                                    UNPARAPHRASABLE_TYPES, build_witness, content_words,
                                    find_conflicts, task_conflicts,
                                    task_reachability_report, unavoidable_terms,
                                    whole_word_present)

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
    """Every instance is either USED by a scenario or explicitly RETIRED.

    Until 2026-08-20 this asserted `not unused` outright. Ladder rung r7-hifreq
    then retired five letter bans (r, l, u, m, p) that the corpus may no longer
    carry but whose ids MUST still resolve: they are written into the run
    transcripts under runs/, into 14 of the 56 rows of
    data/train/smoke_v0.jsonl, into the checker golden fixtures and into a
    few-shot exemplar, and deleting a definition does not delete its uses -- it
    turns them into dangling ids.
    The contract asserted here is therefore narrower, not looser: the unused set
    must be EXACTLY the declared retirements, so an instance still cannot become
    unused quietly, and a retirement cannot be declared for something the corpus
    is in fact still using.
    """
    used = {cid for scenario in corpus for cid in scenario["constraints"]}
    unused = sorted(set(instances) - used)
    assert unused == sorted(RETIRED_LETTER_INSTANCES), (
        f"instances defined but neither used by a scenario nor declared retired: "
        f"{sorted(set(unused) - set(RETIRED_LETTER_INSTANCES))}; and retirements "
        f"the corpus still uses: {sorted(set(RETIRED_LETTER_INSTANCES) & used)}")
    for cid, reason in RETIRED_LETTER_INSTANCES.items():
        assert instances[cid]["type"] == "banned_letter", (
            f"{cid} is declared retired but is not a letter ban; retirement is a "
            "r7-hifreq letter rule, not a general mute button")
        assert instances[cid]["params"]["letter"] not in PERMITTED_BANNED_LETTERS
        assert reason.strip(), f"{cid} is retired with no recorded reason"


# ==========================================================================
# LADDER RUNG r7-hifreq -- THE CURRENT RUNG. Two quantities, both recomputed
# here from the SHIPPED corpus rather than read out of the manifest:
#
#   (1) the share of scenarios carrying a banned_letter -- narrower than the
#       r6 character-family share on purpose, because the r6 run measured
#       banned_punctuation (the other character type) at 0.00-0.13 per frontier
#       cell, so the family overstates how much of the corpus does the work;
#   (2) the LETTERS those bans use. The r6 run measured per-scenario fail rate
#       as monotone in English letter frequency -- a 0.82, e 0.61, i 0.42,
#       o 0.41, s 0.33 against r 0.21, u 0.17, m 0.08 -- so a share met with
#       low-frequency letters would satisfy the number and measure almost
#       nothing. Neither half is sufficient alone, so both are bound.
# ==========================================================================

def test_only_permitted_letters_are_usable_by_a_scenario(instances):
    """The library may DEFINE a retired letter ban; a scenario may not use one."""
    usable = sorted({c["params"]["letter"] for cid, c in instances.items()
                     if c["type"] == "banned_letter"
                     and cid not in RETIRED_LETTER_INSTANCES})
    assert usable == sorted(PERMITTED_BANNED_LETTERS), (
        f"the shipped instance library offers {usable} for use; rung r7-hifreq "
        f"permits {sorted(PERMITTED_BANNED_LETTERS)}")
    assert banned_letters_in_library(include_retired=False) == usable
    # The retired ids still RESOLVE -- run transcripts, the shipped training set
    # and the checker fixtures all name them, and a dangling id there is a
    # scored artefact nobody can read back.
    for retired in RETIRED_LETTER_INSTANCES:
        assert retired in instances, (
            f"{retired} was deleted rather than retired; every artefact that "
            "references it now carries an id that resolves to nothing")


def test_a_scenario_may_not_use_a_retired_letter_instance():
    """Retirement is enforced at build time, not asked for in a comment."""
    retired = sorted(RETIRED_LETTER_INSTANCES)[0]
    spec = _rig(constraints=[retired, "c-fmt-001"])
    with pytest.raises(GeneratorError, match="outside the r7-hifreq permitted set"):
        _build_one(1, spec)


def test_the_instance_library_refuses_a_new_low_frequency_letter():
    """The guard bites: a NEW low-frequency ban cannot be authored.

    'd' rather than one of the retired letters, because a retired instance
    already exists and the duplicate-params rule would refuse it first -- which
    would leave this test green without ever reaching the r7 guard.
    """
    original = inst_mod.INSTANCE_SPECS
    try:
        inst_mod.INSTANCE_SPECS = original + (
            InstanceSpec(11, "banned_letter", {"letter": "d"}, GENERIC, "smuggled back in"),)
        with pytest.raises(ValueError, match="PERMITTED_BANNED_LETTERS"):
            inst_mod._build()
    finally:
        inst_mod.INSTANCE_SPECS = original


def test_banned_letter_share_meets_the_rung_floor(corpus, instances):
    carrying = [s["scenario_id"] for s in corpus
                if any(instances[cid]["type"] == "banned_letter"
                       for cid in s["constraints"])]
    share = len(carrying) / len(corpus)
    assert share >= MIN_BANNED_LETTER_SHARE, (
        f"{len(carrying)}/{len(corpus)} = {share:.1%} of scenarios carry a "
        f"banned_letter, below the {MIN_BANNED_LETTER_SHARE:.0%} that ladder rung "
        "r7-hifreq declares. The rung exists because the r6 run localised frontier "
        "failure to LETTER bans specifically; a corpus below the floor is not the "
        "corpus that rung names")
    # ...and the helper the generator enforces with agrees with this recomputation.
    assert banned_letter_share(corpus) == share


def test_every_letter_ban_in_the_corpus_is_a_permitted_letter(corpus, instances):
    used = {}
    for scenario in corpus:
        for cid in scenario["constraints"]:
            if instances[cid]["type"] != "banned_letter":
                continue
            used.setdefault(instances[cid]["params"]["letter"], []).append(
                scenario["scenario_id"])
    stray = sorted(set(used) - set(PERMITTED_BANNED_LETTERS))
    assert not stray, (
        f"scenarios ban {stray}, outside the r7-hifreq permitted set "
        f"{sorted(PERMITTED_BANNED_LETTERS)}")
    # Every permitted letter is actually exercised, or the set is wider than the
    # corpus and the rung claims a spread it does not have.
    assert set(used) == set(PERMITTED_BANNED_LETTERS), (
        f"permitted letters never used: {sorted(set(PERMITTED_BANNED_LETTERS) - set(used))}")
    assert letter_distribution(corpus) == {k: len(v) for k, v in sorted(used.items())}


def test_the_letter_mix_is_weighted_toward_the_hardest_letters(corpus):
    """Hank's ruling: the permitted set, WEIGHTED toward a and e.

    An audited quantity with a floor, not a decoration: 'a' and 'e' are the two
    letters the r6 run measured hardest (0.82 and 0.61), and a corpus that met
    the share with the easy end of the permitted set would project far lower.
    """
    dist = letter_distribution(corpus)
    n_letter = sum(dist.values())
    hardest = dist.get("a", 0) + dist.get("e", 0)
    assert hardest / n_letter >= 0.50, (
        f"only {hardest}/{n_letter} = {hardest / n_letter:.1%} of letter bans use "
        f"'a' or 'e'; the rung is weighted toward them ({dist})")


def test_the_rung_numbers_in_the_manifest_are_the_real_ones(corpus, manifest):
    summary = manifest["summary"]
    instances = load_instances()
    n_letter = sum(1 for s in corpus
                   if any(instances[cid]["type"] == "banned_letter"
                          for cid in s["constraints"]))
    assert summary["banned_letter_scenarios"] == n_letter
    assert summary["banned_letter_share"] == round(n_letter / len(corpus), 4)
    assert summary["banned_letter_share_floor"] == MIN_BANNED_LETTER_SHARE
    assert summary["permitted_banned_letters"] == list(PERMITTED_BANNED_LETTERS)
    assert summary["retired_letter_instances"] == dict(
        sorted(RETIRED_LETTER_INSTANCES.items()))
    used = {cid for s_ in corpus for cid in s_["constraints"]}
    assert not (set(summary["retired_letter_instances"]) & used), (
        "a retired instance is published as retired while the corpus still uses it")
    assert summary["letter_distribution"] == letter_distribution(corpus)
    assert summary["non_character_control_floor"] == MIN_NON_CHARACTER_CONTROL


def test_the_config_rung_declares_both_halves_and_they_are_the_built_ones():
    """The rung field and the floor the corpus is built to cannot drift apart.

    ``scripts/gate_check.py`` re-derives both numbers from whatever corpus a run
    actually used; this binds those declared numbers to the ones the generator
    builds and refuses below.
    """
    cfg = json.loads((REPO_ROOT / "configs" / "g1.json").read_text())
    ladder = cfg["ladder"]
    assert ladder["current"] == "r7-hifreq", (
        f"ladder.current is {ladder['current']!r}; the shipped corpus is built to "
        "r7-hifreq and a corpus labelled with another rung is a mislabelled corpus")
    rung = next(r for r in ladder["rungs"] if r["name"] == ladder["current"])
    assert rung.get("min_banned_letter_share") == MIN_BANNED_LETTER_SHARE, (
        f"config rung declares {rung.get('min_banned_letter_share')} but the "
        f"generator builds to {MIN_BANNED_LETTER_SHARE}")
    assert rung.get("permitted_banned_letters") == list(PERMITTED_BANNED_LETTERS), (
        f"config rung permits {rung.get('permitted_banned_letters')} but the "
        f"instance library permits {list(PERMITTED_BANNED_LETTERS)}")
    assert isinstance(rung.get("_not_pre_registered"), str) and \
        len(rung["_not_pre_registered"]) > 200, (
        "the rung must keep its not-pre-registered disclosure; a rung invented "
        "after seeing the numbers that does not say so is the one thing a grader "
        "would be right to throw out")


def test_guards_actually_guard_the_banned_letter_share(corpus, by_id):
    """The letter floor can fail the build, and it bites at the right place.

    Letter bans are swapped for a banned_punctuation instance rather than
    deleted, so the CHARACTER share stays where it is and the only floor this
    rig can trip is the letter one.
    """
    instances = load_instances()
    punctuation = "c-chr-020"
    assert instances[punctuation]["type"] == "banned_punctuation"

    def corpus_with_n_delettered(n):
        rows, manifests, swapped = [], [], 0
        for scenario in corpus:
            row = dict(scenario)
            if carries_banned_letter(scenario["constraints"]) and swapped < n:
                row["constraints"] = [punctuation if
                                      instances[cid]["type"] == "banned_letter" else cid
                                      for cid in scenario["constraints"]]
                if punctuation in row["constraints"]:
                    row["split"] = "heldout_type"
                swapped += 1
            rows.append(row)
            manifests.append({"base_task": by_id[scenario["scenario_id"]]["base_task"]})
        assert swapped == n, "not enough letter scenarios to run the rig"
        return rows, manifests

    n_letter = sum(1 for s in corpus if carries_banned_letter(s["constraints"]))
    n_total = len(corpus)
    ok = next(n for n in range(n_letter, -1, -1)
              if (n_letter - n) / n_total >= MIN_BANNED_LETTER_SHARE)
    _corpus_invariants(*corpus_with_n_delettered(ok))          # must not raise
    with pytest.raises(GeneratorError, match="banned-letter share"):
        _corpus_invariants(*corpus_with_n_delettered(ok + 1))


def test_guards_actually_guard_the_permitted_letter_set(corpus, by_id):
    """A corpus that bans a low-frequency letter is refused, not merely noted."""
    smuggled = "c-chr-099"
    original = dict(gen_mod.INSTANCES)
    try:
        gen_mod.INSTANCES[smuggled] = {"id": smuggled, "type": "banned_letter",
                                       "params": {"letter": "m"}, "family": "character",
                                       "split": "train"}
        rows, manifests = [], []
        for i, scenario in enumerate(corpus):
            row = dict(scenario)
            if i == 0:
                row["constraints"] = [smuggled] + [
                    cid for cid in scenario["constraints"]
                    if gen_mod.INSTANCES[cid]["type"] != "banned_letter"]
                row["split"] = "train" if all(
                    gen_mod.INSTANCES[cid]["split"] == "train"
                    for cid in row["constraints"]) else "heldout_type"
            rows.append(row)
            manifests.append({"base_task": by_id[scenario["scenario_id"]]["base_task"]})
        with pytest.raises(GeneratorError, match="outside the r7-hifreq permitted set"):
            _corpus_invariants(rows, manifests)
    finally:
        gen_mod.INSTANCES.clear()
        gen_mod.INSTANCES.update(original)


def test_guards_actually_guard_the_non_character_control_group(corpus, by_id):
    """The share floor must not be reachable by deleting the control arm."""
    instances = load_instances()
    letter_only = ["c-chr-002", "c-fmt-001"]      # a character scenario, train split
    rows, manifests = [], []
    for scenario in corpus:
        row = dict(scenario)
        if not any(instances[cid]["family"] == CHARACTER_FAMILY
                   for cid in scenario["constraints"]):
            row["constraints"] = list(letter_only)
            row["k"] = len(letter_only)
            row["split"] = "train"
        rows.append(row)
        manifests.append({"base_task": by_id[scenario["scenario_id"]]["base_task"]})
    with pytest.raises(GeneratorError, match="non-character control scenarios"):
        _corpus_invariants(rows, manifests)


def test_every_few_shot_exemplar_still_resolves_against_the_shipped_library():
    """Retiring an instance must not leave the few_shot cell pointing at a ghost.

    ADDED 2026-08-20: r7-hifreq retired five banned_letter instances, and one
    shipped exemplar demonstrated one of them. A dangling id there is not a
    cosmetic problem -- ``harness.run_checkers`` cannot score a constraint it
    cannot resolve, so the whole few_shot column of the ablation would have died
    at run time, after the money was spent.
    """
    exemplars_path = REPO_ROOT / "data" / "scenarios" / "fewshot_exemplars.jsonl"
    rows = [json.loads(line) for line in exemplars_path.read_text().splitlines()
            if line.strip()]
    assert rows, "no exemplars; the few_shot cell would be unrunnable"
    instances = load_instances()
    for row in rows:
        missing = [cid for cid in row["constraints"] if cid not in instances]
        assert not missing, (
            f"{row['exemplar_id']} references retired constraint(s) {missing}")
        assert row["k"] == len(row["constraints"])


def test_the_gate_re_derives_both_rung_halves_from_the_shipped_corpus():
    """The rung is verified where it matters: against the corpus a RUN used.

    ``scripts/gate_check.py`` is the only reader of these numbers that a run
    cannot talk its way past, so the binding is proved here rather than assumed:
    the shipped corpus passes, and a corpus that violates EITHER half is
    rejected with a message naming which half.
    """
    import scripts.gate_check as gate

    cfg = json.loads((REPO_ROOT / "configs" / "g1.json").read_text())
    definition = next(r for r in cfg["ladder"]["rungs"]
                      if r["name"] == cfg["ladder"]["current"])
    rung = {"rung": definition["name"], "definition": definition}
    instances = gate.load_constraint_instances(INSTANCES_PATH)
    eval_index = gate.load_eval_set(CORPUS_PATH)

    errors = []
    observed = gate.verify_rung_against_eval_set(rung, eval_index, instances,
                                                 CORPUS_PATH, errors)
    assert errors == [], f"the shipped corpus fails its own rung: {errors}"
    corpus = load_corpus()
    assert observed["n_banned_letter"] == sum(
        1 for s in corpus if carries_banned_letter(s["constraints"]))
    assert observed["letter_distribution"] == letter_distribution(corpus)
    assert observed["permitted_banned_letters"] == list(PERMITTED_BANNED_LETTERS)

    # (1) share below the floor: strip letter bans until it is.
    thin = {sid: dict(row) for sid, row in eval_index.items()}
    stripped = 0
    target = len(thin) - int(len(thin) * MIN_BANNED_LETTER_SHARE)
    for sid, row in thin.items():
        if stripped > target:
            break
        kept = [cid for cid in row["constraints"]
                if instances[cid]["type"] != "banned_letter"]
        if len(kept) != len(row["constraints"]) and kept:
            row["constraints"] = kept
            stripped += 1
    errors = []
    gate.verify_rung_against_eval_set(rung, thin, instances, CORPUS_PATH, errors)
    assert any("banned_letter constraint, below" in e for e in errors), errors

    # (2) a letter the rung does not permit, at full share.
    smuggled = dict(instances)
    smuggled["c-chr-099"] = {"id": "c-chr-099", "type": "banned_letter",
                             "params": {"letter": "m"}, "family": "character",
                             "split": "train"}
    swapped = {sid: dict(row) for sid, row in eval_index.items()}
    for sid, row in swapped.items():
        if any(instances[cid]["type"] == "banned_letter" for cid in row["constraints"]):
            row["constraints"] = ["c-chr-099" if
                                  instances[cid]["type"] == "banned_letter" else cid
                                  for cid in row["constraints"]]
            break
    errors = []
    gate.verify_rung_against_eval_set(rung, swapped, smuggled, CORPUS_PATH, errors)
    assert any("does not permit" in e for e in errors), errors


# ==========================================================================
# LADDER RUNG r6-char65 -- the character-constraint share.
#
# This is the number the whole escalation rests on, so it is recomputed here
# from the SHIPPED corpus and the pinned registry, never read out of the
# manifest. "Character" is decided by constraints.registry.family_of, not by a
# list of type names retyped in this file: if a character type is added later
# it counts automatically instead of silently deflating the share.
# ==========================================================================

def _character_types_from_the_registry():
    types = {t for t in TYPES if family_of(t) == CHARACTER_FAMILY}
    assert types, "the registry has no character family; the rung is meaningless"
    return types


def test_the_character_family_is_the_one_the_rung_means():
    """banned_letter (train) and banned_punctuation (held out), from the registry."""
    assert CHARACTER_FAMILY in FAMILIES
    assert set(CHARACTER_TYPES) == _character_types_from_the_registry()
    assert set(CHARACTER_TYPES) == {"banned_letter", "banned_punctuation"}, (
        f"the character family is now {sorted(CHARACTER_TYPES)}; the r6-char65 "
        "rung was approved on the letter-ban / punctuation-ban pair, so a change "
        "here changes what the rung claims and needs saying out loud")
    # The generalisation claim the split sets up: letters are trained on,
    # punctuation is held out. Same family, different constraint.
    assert split_of("banned_letter") == "train"
    assert split_of("banned_punctuation") == "heldout_type"


def test_character_constraint_share_meets_the_rung_floor(corpus, instances):
    character = _character_types_from_the_registry()
    carrying = [s["scenario_id"] for s in corpus
                if any(instances[cid]["type"] in character for cid in s["constraints"])]
    share = len(carrying) / len(corpus)
    assert share >= MIN_CHARACTER_CONSTRAINT_SHARE, (
        f"{len(carrying)}/{len(corpus)} = {share:.1%} of scenarios carry a "
        f"character-level constraint, below the {MIN_CHARACTER_CONSTRAINT_SHARE:.0%} "
        "the r6-char65 rung declares. The rung exists because the G1 ablation "
        "localised frontier failure to character constraints; a corpus below the "
        "floor is not the corpus that rung names")
    # ...and the helper the generator enforces with agrees with this recomputation.
    assert character_constraint_share(corpus) == share


def test_the_character_share_in_the_manifest_is_the_real_one(corpus, manifest):
    summary = manifest["summary"]
    character = _character_types_from_the_registry()
    instances = load_instances()
    n_char = sum(1 for s in corpus
                 if any(instances[cid]["type"] in character for cid in s["constraints"]))
    assert summary["character_constraint_scenarios"] == n_char
    assert summary["character_constraint_share"] == round(n_char / len(corpus), 4)
    assert summary["character_constraint_share_floor"] == MIN_CHARACTER_CONSTRAINT_SHARE
    assert set(summary["character_types"]) == character


def test_the_config_rung_and_the_generator_floor_are_the_same_number():
    """An unverified rung field is a claim. This is where the claim is bound.

    ``configs/g1.json`` names the rung the shipped corpus is built to and
    ``scripts/gate_check.py`` re-derives the share from whatever corpus a run
    actually used. Both read ``min_character_constraint_share``; if that number
    and the floor the generator builds to ever drift apart, the corpus would
    satisfy one and be labelled with the other.
    """
    cfg = json.loads((REPO_ROOT / "configs" / "g1.json").read_text())
    ladder = cfg["ladder"]
    rung = next(r for r in ladder["rungs"] if r["name"] == ladder["current"])
    declared = rung.get("min_character_constraint_share")
    assert declared is not None, (
        f"ladder.current is {ladder['current']!r} but that rung declares no "
        "min_character_constraint_share; the corpus is built to a floor the "
        "config does not state")
    assert declared == MIN_CHARACTER_CONSTRAINT_SHARE, (
        f"config rung {ladder['current']!r} declares {declared} but the generator "
        f"builds to {MIN_CHARACTER_CONSTRAINT_SHARE}")


def test_character_constraints_are_not_concentrated_in_one_cell(corpus, instances):
    """A share met by loading one cell would confound the family with pressure.

    Robustness is spec-adherence on the ADVERSARIAL subset. If character
    constraints lived mostly in that subset, the clean-vs-adversarial gap would
    be a character-vs-lexical gap wearing its name. The split is a different
    matter and is deliberately asymmetric -- banned_punctuation is a held-out
    character type, so the held-out arm is character-heavy by design -- so what
    is bound here is the PRESSURE arms, plus a floor per cell that keeps every
    cell usable on its own.
    """
    character = _character_types_from_the_registry()
    def carries(scenario):
        return any(instances[cid]["type"] in character for cid in scenario["constraints"])

    by_pressure = {}
    for pressure in ("clean", "adversarial"):
        rows = [s for s in corpus if s["pressure"] == pressure]
        by_pressure[pressure] = sum(1 for s in rows if carries(s)) / len(rows)
    gap = abs(by_pressure["clean"] - by_pressure["adversarial"])
    assert gap <= 0.15, (
        f"character-constraint share is {by_pressure['clean']:.1%} in the clean arm "
        f"and {by_pressure['adversarial']:.1%} in the adversarial arm; a gap that "
        "size makes robustness a measure of constraint family, not of pressure")

    for split in ("train", "heldout_type"):
        for pressure in ("clean", "adversarial"):
            rows = [s for s in corpus
                    if s["split"] == split and s["pressure"] == pressure]
            n_char = sum(1 for s in rows if carries(s))
            assert n_char >= 20, (
                f"cell {(split, pressure)} carries only {n_char} character "
                f"scenarios of {len(rows)}; the rung's claim cannot be read per cell")


def test_the_corpus_still_holds_a_non_character_control_group(corpus, instances):
    """Raising the share must not delete the contrast that made the finding legible.

    The G1 ablation is only interpretable because frontier models failed
    letter-ban scenarios 35-52% of the time and everything else 0.9-4.5%. That
    comparison needs a non-character arm big enough to carry a rate.
    """
    character = _character_types_from_the_registry()
    control = [s["scenario_id"] for s in corpus
               if not any(instances[cid]["type"] in character for cid in s["constraints"])]
    assert len(control) >= MIN_NON_CHARACTER_CONTROL, (
        f"only {len(control)} scenarios carry no character constraint; the "
        "non-character comparison arm is too small to state a rate against")
    # The generator refuses below the same number, so this is not a floor that
    # lives only in a test the corpus is not built against.
    assert MIN_NON_CHARACTER_CONTROL == 30


def test_guards_actually_guard_the_character_share(corpus, by_id):
    """The floor can fail the build, and it bites at exactly the right place.

    Without this the share assertions above could be green because nothing is
    capable of turning them red.
    """
    character = _character_types_from_the_registry()
    instances = load_instances()
    plain = ["c-lex-025", "c-fmt-001"]      # two train, non-character constraints
    assert not any(instances[cid]["type"] in character for cid in plain)

    def corpus_with_n_stripped(n):
        rows, manifests, stripped = [], [], 0
        for scenario in corpus:
            row = dict(scenario)
            carries = any(instances[cid]["type"] in character
                          for cid in scenario["constraints"])
            if carries and stripped < n:
                row["constraints"] = list(plain)
                row["k"] = len(plain)
                row["split"] = "train"
                stripped += 1
            rows.append(row)
            manifests.append({"base_task": by_id[scenario["scenario_id"]]["base_task"]})
        assert stripped == n, "not enough character scenarios to run the rig"
        return rows, manifests

    n_char = sum(1 for s in corpus
                 if any(instances[cid]["type"] in character for cid in s["constraints"]))
    n_total = len(corpus)
    # The largest strip that still clears the floor, and the first that does not.
    ok = next(n for n in range(n_char, -1, -1)
              if (n_char - n) / n_total >= MIN_CHARACTER_CONSTRAINT_SHARE)
    _corpus_invariants(*corpus_with_n_stripped(ok))          # must not raise
    with pytest.raises(GeneratorError, match="character-constraint share"):
        _corpus_invariants(*corpus_with_n_stripped(ok + 1))


# ==========================================================================
# Every shipped scenario is renderable into the exact user message a model
# sees. A scenario that cannot be rendered is not evaluable, so the corpus
# would be shorter than it claims without anything saying so.
# ==========================================================================

def test_every_scenario_renders_through_the_real_renderer(corpus, instances):
    for scenario in corpus:
        prompt = render_user_message(scenario, instances)
        assert isinstance(prompt, str) and prompt.strip(), scenario["scenario_id"]
        assert scenario["task"] in prompt, (
            f"{scenario['scenario_id']}: the rendered prompt dropped the task")
        assert render_user_message(scenario, instances) == prompt, "render is not stable"


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
                assert len(others) >= MIN_LEXICON_WITNESSES, (
                    f"{sid}: only {others} topic words in the task")
                for word in others:
                    assert word.lower() in members, f"{sid}: {word!r} not in lexicon {lex}"
                    assert word.lower() not in banned, f"{sid}: {word!r} is itself banned"
                    assert _whole_word(word, base), f"{sid}: {word!r} is not in the task text"
                # DERIVED floor: the tag is not the author's to assert -- the
                # shipped text has to make this lexicon the dominant topic.
                leaders, best = dominant_lexicons(base)
                assert lex in leaders, (
                    f"{sid}: tagged {lex!r} but the task text is more about {leaders} "
                    f"({best} members); the topic tag is not confirmed by the prompt")
                assert mech["dominant_lexicons"] == leaders, f"{sid}: stale dominance evidence"
            elif mech["kind"] == "key_term_letter":
                letter = mech["letter"]
                assert constraint["params"]["letter"].lower() == letter
                hits = mech["key_terms_carrying_letter"]
                assert len(hits) >= MIN_KEY_TERM_HITS, (
                    f"{sid}: banned letter appears in only {hits}")
                for term in hits:
                    assert letter in term.lower(), f"{sid}: {term!r} does not carry {letter!r}"
                    # WHOLE-WORD, not substring: "us" must not qualify off "museum".
                    assert _whole_word(term, base), f"{sid}: {term!r} is not in the task"
                # DERIVED floor: recomputed from the shipped text, with no
                # reference to what the author declared. This is the assertion
                # that makes the adversarial label unfakeable by declaration.
                share = letter_content_share(letter, base)
                assert share >= MIN_LETTER_CONTENT_SHARE, (
                    f"{sid}: only {share:.1%} of the task's content words carry {letter!r}; "
                    f"the floor is {MIN_LETTER_CONTENT_SHARE:.0%}. A declared key term "
                    "cannot make a letter bite a task it barely touches")
                assert mech["letter_content_share"] == round(share, 4), (
                    f"{sid}: recorded share {mech['letter_content_share']} != recomputed {share}")
                assert set(mech["content_words_carrying_letter"]) <= set(content_words(base))
            elif mech["kind"] == "punctuation_density":
                total = sum(base.count(mark) for mark in constraint["params"]["marks"])
                assert total >= MIN_MARK_OCCURRENCES, (
                    f"{sid}: the prompt uses the banned mark(s) only {total} "
                    "times, which is not a register that invites them")
                # DERIVED floor: a rate, so a long prompt cannot buy the label.
                rate = total / len(base) * 100
                assert rate >= MIN_MARK_RATE_PER_100_CHARS, (
                    f"{sid}: {rate:.2f} banned marks per 100 chars is below the "
                    f"{MIN_MARK_RATE_PER_100_CHARS} floor; the prompt is long, not dense")
                assert mech["marks_per_100_chars"] == round(rate, 4), f"{sid}: stale rate"
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


def test_the_name_detector_has_something_to_detect_in_the_shipped_corpus(corpus, by_id,
                                                                         instances):
    """Green from a detector with no input is not evidence.

    ``test_no_task_demands_what_its_constraints_forbid`` would pass trivially if
    no shipped task named anything, or if no scenario carried a constraint a
    bare name could violate. Neither is true, so its green means the check ran.
    """
    named = [s["scenario_id"] for s in corpus
             if unavoidable_terms(by_id[s["scenario_id"]]["base_task"])]
    assert len(named) >= 20, (
        f"only {len(named)} shipped tasks name anything; the unparaphrasable-name "
        "check is passing because it has nothing to look at")
    checked = [s["scenario_id"] for s in corpus
               if any(instances[cid]["type"] in UNPARAPHRASABLE_TYPES
                      for cid in s["constraints"])]
    assert len(checked) >= len(corpus) // 2, (
        f"only {len(checked)} scenarios carry a constraint a bare name could violate")


def test_task_conflicts_catches_an_unparaphrasable_name(instances):
    """BLOCKER 1's core case, decided by the REAL checkers rather than a keyword list.

    A brief that names a weekday under a ban on 's' cannot be answered in the
    obvious way, and no regex list of task wordings would ever have seen it.
    """
    banned_s = instances["c-chr-003"]                     # banned_letter 's'
    task = "Write a short note explaining that the meeting moves to Thursdays."
    assert unavoidable_terms(task) == ["Thursdays"]
    assert check(banned_s, "Thursdays")["pass"] is False, "the premise changed"
    problems = task_conflicts(task, [banned_s, instances["c-fmt-001"]])
    assert problems, "the unparaphrasable name was not detected"
    assert "Thursdays" in problems[0] and "c-chr-003" in problems[0]
    # ...and the same brief with a letter the name survives is NOT flagged.
    # Was c-chr-010 ('p') until 2026-08-20; that instance was retired with the
    # low-frequency letters under ladder rung r7-hifreq, so the counter-example
    # now uses a PERMITTED letter that 'Thursdays' still survives. The claim
    # being tested is unchanged: a name the checker accepts is not a conflict.
    assert check(instances["c-chr-001"], "Thursdays")["pass"] is True, (
        "the counter-example needs a letter the name survives")
    assert task_conflicts(task, [instances["c-chr-001"],       # banned_letter 'e'
                                 instances["c-fmt-001"]]) == []


def test_the_name_check_excludes_positional_constraints(instances):
    """A name that cannot OPEN a sentence can still be written inside one.

    Running sentence_initial_letter over bare names would flag almost every
    proper noun in the corpus as an impossibility. That would be a manufactured
    contradiction -- the failure mode this lane exists to remove -- so the
    positional type is excluded by construction, not by luck.
    """
    assert "sentence_initial_letter" not in UNPARAPHRASABLE_TYPES
    opens_with_q = instances["c-str-002"]                 # every sentence opens with 's'
    task = "Write a short note explaining that the meeting moves to Thursdays."
    assert check(opens_with_q, "Thursdays")["pass"] is False, (
        "premise: a bare name fails a positional checker, which is exactly why it "
        "must not be treated as an impossibility")
    assert task_conflicts(task, [opens_with_q, instances["c-fmt-001"]]) == []


def test_unavoidable_terms_ignores_sentence_openers_and_grammar():
    assert unavoidable_terms("Write a note about the harbour.") == []
    assert unavoidable_terms("Describe Ashford. The station is closed.") == ["Ashford"]
    assert unavoidable_terms("Write about the 07.14 service.") == [], "a numeral is not a name"


def test_widened_format_demands_still_catch_the_obvious_and_spare_the_innocent(instances):
    no_lists = instances["c-fmt-001"]
    no_quotes = instances["c-fmt-002"]
    for demand in ("Give me five bullet points.", "Set it out as a numbered list.",
                   "Write it in point form.", "Produce a checklist for the team.",
                   "Please itemise the charges.", "Put one per line."):
        assert task_conflicts(demand, [no_lists]), f"missed a list demand: {demand!r}"
    for demand in ("Reproduce her exact wording.", "Quote the inspector.",
                   "Copy it word for word.", "Put the name in quotation marks."):
        assert task_conflicts(demand, [no_quotes]), f"missed a quote demand: {demand!r}"
    # Prose can narrate steps and can report what somebody said. Flagging these
    # would invent an impossibility, which is worse than missing one.
    for innocent in ("Explain the steps in order, in flowing prose.",
                     "Say what the inspector concluded."):
        assert task_conflicts(innocent, [no_lists, no_quotes]) == [], (
            f"manufactured a conflict for {innocent!r}")


def test_the_report_says_out_loud_what_it_does_not_decide(instances):
    """The undecidable residue is published, not quietly implied by a green test."""
    report = task_reachability_report(
        "Write a short note explaining that the meeting moves to Thursdays.",
        [instances["c-chr-003"], instances["c-fmt-001"]])
    assert report["conflicts"], "the decidable half stopped working"
    assert report["unavoidable_terms"] == ["Thursdays"]
    assert "undecided" in report and "expressible" in report["undecided"], (
        "the report must state the part of reachability the corpus does NOT prove")


# ==========================================================================
# The QUARANTINE: scenarios held out of the corpus, and why
# ==========================================================================

def test_quarantined_tasks_are_absent_from_the_corpus(by_id):
    shipped = {entry["task_key"] for entry in by_id.values()}
    for task_key in QUARANTINE:
        assert task_key not in shipped, (
            f"{task_key} is quarantined but shipped anyway")


def test_every_quarantined_task_still_reproduces_its_recorded_defect():
    """A quarantine entry is a claim about a defect, so it is re-proved here.

    Without this, ``QUARANTINE`` would be a way to delete an inconvenient
    scenario from the corpus with a sentence of prose as the only evidence.
    """
    from scenarios.tasks import TASKS
    by_key = {spec["id"]: spec for spec in TASKS}
    assert set(QUARANTINE) <= set(by_key), (
        f"quarantine names tasks that do not exist: {sorted(set(QUARANTINE) - set(by_key))}")
    for task_key, entry in QUARANTINE.items():
        with pytest.raises(GeneratorError) as excinfo:
            _build_one(1, by_key[task_key])
        assert entry.marker in str(excinfo.value), (
            f"{task_key}: quarantined for {entry.marker!r} but now fails with "
            f"{excinfo.value}")
        assert entry.reason.strip(), f"{task_key}: quarantined with no reason given"


def test_the_quarantine_is_published_in_the_manifest(manifest):
    published = {row["task_key"]: row["reason"] for row in
                 manifest["summary"]["quarantined_tasks"]}
    assert published == {key: entry.reason for key, entry in QUARANTINE.items()}, (
        "the shipped manifest does not record what was held out of the corpus")


def test_the_quarantine_cannot_be_used_as_a_mute_button():
    """A quarantine entry for a task that builds fine must FAIL the build."""
    from scenarios.tasks import TASKS
    healthy = next(spec for spec in TASKS if spec["id"] not in QUARANTINE)
    from scenarios import generator as gen
    original = dict(gen.QUARANTINE)
    gen.QUARANTINE[healthy["id"]] = gen.Quarantined("invented", "never appears in any error")
    try:
        with pytest.raises(GeneratorError, match="no longer reproduces its defect"):
            generate()
    finally:
        gen.QUARANTINE.clear()
        gen.QUARANTINE.update(original)
    # ...and the restore worked, so the rest of the suite sees the real corpus.
    assert set(gen.QUARANTINE) == set(original)


def test_a_quarantine_entry_for_a_task_that_does_not_exist_fails_the_build():
    """Dead quarantine entries hide the next real one, so they are an error."""
    from scenarios import generator as gen
    original = dict(gen.QUARANTINE)
    gen.QUARANTINE["no-such-task"] = gen.Quarantined("ghost", "never matched")
    try:
        with pytest.raises(GeneratorError, match="not in scenarios.tasks"):
            generate()
    finally:
        gen.QUARANTINE.clear()
        gen.QUARANTINE.update(original)


def test_a_quarantine_marker_that_no_longer_matches_fails_the_build():
    from scenarios import generator as gen
    from scenarios.tasks import TASKS
    victim = next(iter(QUARANTINE))
    original = dict(gen.QUARANTINE)
    gen.QUARANTINE[victim] = gen.Quarantined("stale", "a reason that is not the real one")
    try:
        with pytest.raises(GeneratorError, match="fails for a different reason"):
            generate()
    finally:
        gen.QUARANTINE.clear()
        gen.QUARANTINE.update(original)


# ==========================================================================
# Distractor floors are DERIVED, not declared (BLOCKER 2)
# ==========================================================================

def _rig(**overrides):
    spec = {"id": "rig", "text": "Write a paragraph about a harbour at dawn.",
            "domain": "test", "register": "prose",
            "constraints": ["c-chr-001", "c-fmt-001"], "pressure": "clean",
            "lexicon": None, "key_terms": (), "nudge": None}
    spec.update(overrides)
    return spec


#: Declared key terms that all carry the banned letter, in a task whose
#: vocabulary barely does. Under the pre-fix rule this was enough to be called
#: adversarial.
#:
#: The rig letter was 'p' until 2026-08-20. Ladder rung r7-hifreq retired the
#: low-frequency instances, so the rig moved to 's' -- the least frequent letter
#: the rung still permits, and therefore the one that still leaves room for a
#: task whose vocabulary the letter barely touches. Nothing about what these
#: rigs PROVE changed: thin declaration below the floor must not qualify, dense
#: text at or above it must.
_RIG_LETTER = "s"
_RIG_LETTER_ID = "c-chr-003"
_THIN_LETTER_TASK = ("Write a calm note about the season tickets, the stairwell, the "
                     "wooden door, the iron gate and the linen cloth.")
_DENSE_LETTER_TASK = ("Prepare a summary assessing the season tickets, the stairwell "
                      "signs and the storeroom shelves.")
_THIN_KEY_TERMS = ("season tickets", "stairwell")
_DENSE_KEY_TERMS = ("season tickets", "storeroom shelves")


def test_key_term_letter_needs_the_derived_floor_not_just_a_declaration():
    thin = _rig(text=_THIN_LETTER_TASK, constraints=[_RIG_LETTER_ID, "c-fmt-001"],
                pressure="adversarial", key_terms=_THIN_KEY_TERMS,
                nudge="letter_audit")
    share = letter_content_share(_RIG_LETTER, _THIN_LETTER_TASK)
    assert share < MIN_LETTER_CONTENT_SHARE, "rig no longer exercises the floor"
    assert len([t for t in _THIN_KEY_TERMS
                if _RIG_LETTER in t]) >= MIN_KEY_TERM_HITS, (
        "rig must satisfy the OLD declaration-only rule, or it proves nothing")
    assert distractor_mechanisms(_THIN_LETTER_TASK, None, _THIN_KEY_TERMS,
                                 [load_instances()[_RIG_LETTER_ID]]) == [], (
        "the mechanism fired on a task whose vocabulary the letter barely touches")
    with pytest.raises(GeneratorError, match="no distractor pressure qualifies"):
        _build_one(1, thin)


def test_key_term_letter_still_fires_when_the_text_backs_the_claim():
    dense = _rig(text=_DENSE_LETTER_TASK, constraints=[_RIG_LETTER_ID, "c-fmt-001"],
                 pressure="adversarial",
                 key_terms=_DENSE_KEY_TERMS, nudge="letter_audit")
    built = _build_one(1, dense)
    kinds = [m["kind"] for m in built.manifest["distractor_pressure"]]
    assert kinds == ["key_term_letter"], f"the floor is now unreachable: {kinds}"
    assert built.manifest["distractor_pressure"][0]["letter_content_share"] >= \
        MIN_LETTER_CONTENT_SHARE


def test_a_declared_key_term_must_be_a_whole_word_in_the_task():
    """Substring matching is how a declared distractor becomes unfalsifiable."""
    assert whole_word_present("storeroom shelves", _DENSE_LETTER_TASK)
    assert not whole_word_present("store", _DENSE_LETTER_TASK), (
        "'store' is inside 'storeroom' only")
    with pytest.raises(GeneratorError, match="does not appear in the task text"):
        _build_one(1, _rig(text=_DENSE_LETTER_TASK,
                           constraints=[_RIG_LETTER_ID, "c-fmt-001"],
                           pressure="adversarial",
                           key_terms=("store", "storeroom shelves"),
                           nudge="letter_audit"))


def test_lexicon_collision_requires_the_text_to_confirm_the_topic_tag(instances):
    """A tea ban on a prompt that is really about baking must not qualify."""
    tea_ban = instances["c-lex-024"]
    off_topic = ("The mug and the teapot sat by the oven while the dough, the flour, "
                 "the yeast, the crust and the loaf waited.")
    on_topic = ("The mug and the teapot and the infuser and the oolong sat by the oven "
                "with the dough.")
    assert "tea" not in dominant_lexicons(off_topic)[0], "rig no longer exercises dominance"
    assert distractor_mechanisms(off_topic, "tea", (), [tea_ban]) == [], (
        "a topic tag the prompt does not support still bought a distractor claim")
    assert "tea" in dominant_lexicons(on_topic)[0]
    assert [m["kind"] for m in distractor_mechanisms(on_topic, "tea", (), [tea_ban])] == \
        ["lexicon_collision"], "the dominance floor is now unreachable"


def test_punctuation_density_is_a_rate_so_length_cannot_buy_the_label():
    short = ("Write a note, a short one, quickly, please.")
    long = ("Write a long brief for the team that explains at length what the finished "
            "piece must cover and how it should read to somebody who has never seen the "
            "material before and needs the whole background spelled out from the "
            "beginning without any shortcuts whatsoever and who will also want to know "
            "why every earlier attempt at this same brief was abandoned halfway "
            "through, and then, finally, wrap up.")
    assert short.count(",") == long.count(",") == MIN_MARK_OCCURRENCES, (
        "both rigs must clear the OLD count-only bar, or this proves nothing")
    assert long.count(",") / len(long) * 100 < MIN_MARK_RATE_PER_100_CHARS
    built = _build_one(1, _rig(text=short, constraints=["c-chr-020", "c-fmt-001"],
                               pressure="adversarial", nudge="punct_readability"))
    assert [m["kind"] for m in built.manifest["distractor_pressure"]] == ["punctuation_density"]
    with pytest.raises(GeneratorError, match="no distractor pressure qualifies"):
        _build_one(1, _rig(text=long, constraints=["c-chr-020", "c-fmt-001"],
                           pressure="adversarial", nudge="punct_readability"))


def test_the_clean_arms_latent_letter_pressure_is_disclosed(corpus, by_id, instances):
    """The one asymmetry left in the pressure labelling is published, not hidden.

    ``key_term_letter`` fires only where an author declared key terms, so a
    CLEAN task whose own vocabulary is dense in its banned letter keeps the
    clean label. That understates the clean-vs-adversarial gap rather than
    overstating it, but it is a real limit, so every scenario carrying a
    banned_letter records its derived share in the manifest -- clean ones
    included -- and the number is recomputable from the shipped text.
    """
    disclosed = 0
    for scenario in corpus:
        entry = by_id[scenario["scenario_id"]]
        letters = [instances[cid]["params"]["letter"] for cid in scenario["constraints"]
                   if instances[cid]["type"] == "banned_letter"]
        if not letters:
            assert entry["derived_letter_share"] is None
            continue
        expected = letter_content_share(letters[0], entry["base_task"])
        assert entry["derived_letter_share"] == round(expected, 4), (
            f"{scenario['scenario_id']}: recorded letter share is not the real one")
        disclosed += 1
    assert disclosed >= 20, f"only {disclosed} scenarios disclose a letter share"

    # ------------------------------------------------------------------
    # WHAT REPLACED THE OLD ABSOLUTE COUNT, and why, because a test that got
    # weaker without a reason is indistinguishable from one that was silenced.
    #
    # Until 2026-08-20 this asserted `len(latent) <= 10`, counting CLEAN
    # scenarios whose derived_letter_share reached MIN_LETTER_CONTENT_SHARE
    # (0.50). That count was informative while the corpus banned letters from
    # across the alphabet: clearing 0.50 was then a rare event that really did
    # signal an unusually letter-dense task.
    #
    # Ladder rung r7-hifreq restricts letter bans to {a, e, i, o, s}, and for
    # the frequent letters 0.50 is not a rare event -- it is the MEDIAN of
    # ordinary English. Measured over all 147 shipped base tasks, the fraction
    # whose content words clear 0.50 is 0.837 for 'e' and 0.252 for 'a'. So the
    # old statistic now counts English, not engineered pressure, and no
    # threshold on it can separate the arms. Raising 10 to 20 would have been
    # exactly the silencing this repo's rules forbid.
    #
    # The CONTRACT the old assertion was reaching for is that the clean arm is
    # still a control: it must not carry MORE letter pressure than the arm that
    # engineers it, and it must not be selected for letter-dense tasks. Both are
    # asserted directly below, against a baseline recomputed from the shipped
    # task texts, which is strictly stronger than a count that English frequency
    # alone can move.
    # ------------------------------------------------------------------
    def _shares(pressure, letter=None):
        out = []
        for scenario in corpus:
            if scenario["pressure"] != pressure:
                continue
            entry = by_id[scenario["scenario_id"]]
            share = entry["derived_letter_share"]
            if share is None:
                continue
            banned = [instances[cid]["params"]["letter"]
                      for cid in scenario["constraints"]
                      if instances[cid]["type"] == "banned_letter"][0]
            if letter is not None and banned != letter:
                continue
            out.append(share)
        return out

    clean, adversarial = _shares("clean"), _shares("adversarial")
    assert clean and adversarial, "both arms must carry letter bans to compare them"
    assert mean(clean) <= mean(adversarial), (
        f"the CLEAN arm's mean derived letter share is {mean(clean):.3f} against "
        f"{mean(adversarial):.3f} in the adversarial arm; the control arm carries "
        "more letter pressure than the arm that engineers it, so the "
        "clean-vs-adversarial gap no longer measures engineered pressure")

    # ...and per letter, against what that letter costs in this corpus's prose
    # generally. The baseline is derived from every shipped base task, so it is
    # a property of the corpus's English rather than of the pairing.
    all_texts = [by_id[s["scenario_id"]]["base_task"] for s in corpus]
    for letter in sorted({instances[cid]["params"]["letter"]
                          for s in corpus for cid in s["constraints"]
                          if instances[cid]["type"] == "banned_letter"}):
        rows = _shares("clean", letter)
        if not rows:
            continue
        baseline = mean([letter_content_share(letter, t) for t in all_texts])
        assert mean(rows) <= baseline + 0.10, (
            f"clean scenarios banning {letter!r} average a derived share of "
            f"{mean(rows):.3f} against a corpus baseline of {baseline:.3f}; the "
            "clean arm has been selected for letter-dense tasks and is no longer "
            "a control")


def test_the_adversarial_share_is_the_honest_one(corpus):
    """The floor is the ticket's 40%, and the number is recomputed, not quoted."""
    share = sum(1 for s in corpus if s["pressure"] == "adversarial") / len(corpus)
    assert share >= MIN_ADVERSARIAL_SHARE
    assert summarise(corpus)["adversarial_share"] == round(share, 4)


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
