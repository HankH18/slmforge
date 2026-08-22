"""The shipped DPO preference set is re-verified here, not taken on trust.

Everything the builder claims is re-derived from the frozen inputs by code that
does not import the builder's verdicts: the pinned checkers decide orientation
again, the shared renderer re-renders every prompt again, and the eval corpus is
re-rendered and compared again. A test that only asked the builder whether it was
happy would be a tautology.

The non-vacuity guards matter as much as the assertions. Every parametrised or
looped check below is paired with a counter assertion, because a filter bug that
empties the dataset would otherwise turn this whole file green.
"""
import json
from pathlib import Path

import pytest

from datagen import dpo_pairs as D
from constraints.checkers import check
from scenarios.paths import load_corpus, load_instances
from scenarios.render import render_user_message

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE = REPO_ROOT / "data" / "train" / "v3.dpo_negatives.jsonl"
PAIRS = REPO_ROOT / "data" / "train" / "v3.dpo_pairs.jsonl"
MANIFEST = REPO_ROOT / "data" / "train" / "v3.dpo_pairs.manifest.json"

INSTANCES = load_instances()


def _load_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@pytest.fixture(scope="module")
def rows():
    return _load_jsonl(PAIRS)


@pytest.fixture(scope="module")
def manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def source_rows():
    return _load_jsonl(SOURCE)


@pytest.fixture(scope="module")
def rebuilt():
    """One real build of the shipped source. Shared: it re-runs the checkers."""
    return D.build(SOURCE)


# ==========================================================================
# The files exist and are not empty -- every loop below depends on it
# ==========================================================================

def test_the_dataset_and_its_manifest_are_shipped_and_non_empty(rows, manifest):
    assert PAIRS.exists() and MANIFEST.exists()
    assert len(rows) > 0, "the pairs file is empty; every loop in this file is vacuous"
    assert manifest["output"]["rows"] == len(rows)


# ==========================================================================
# Orientation, re-decided by the pinned checkers
# ==========================================================================

def test_every_pair_reverifies_under_the_shipped_checkers(rows):
    """chosen passes every constraint; rejected fails at least one; texts differ."""
    checked = 0
    for index, row in enumerate(rows):
        constraints = [INSTANCES[cid] for cid in row["constraints"]]
        assert constraints, f"row {index}: no constraints to verify against"
        chosen = row["chosen"][0]["content"]
        rejected = row["rejected"][0]["content"]

        offenders = [c["id"] for c in constraints if not check(c, chosen)["pass"]]
        assert not offenders, f"row {index} ({row['scenario_id']}): chosen violates {offenders}"

        violated = [c["id"] for c in constraints if not check(c, rejected)["pass"]]
        assert violated, f"row {index} ({row['scenario_id']}): rejected violates nothing"

        assert chosen.strip() != rejected.strip(), f"row {index}: identical texts"
        assert sorted(violated) == sorted(row["violated_constraint_ids"]), (
            f"row {index}: recorded violations {row['violated_constraint_ids']} "
            f"!= re-derived {violated}")
        checked += 1
    assert checked == len(rows) > 0


def test_no_pair_is_backwards(rows):
    """The specific failure mode: rejected is clean and chosen is the violator."""
    backwards = []
    for row in rows:
        constraints = [INSTANCES[cid] for cid in row["constraints"]]
        rejected_clean = all(check(c, row["rejected"][0]["content"])["pass"]
                             for c in constraints)
        chosen_dirty = any(not check(c, row["chosen"][0]["content"])["pass"]
                           for c in constraints)
        if rejected_clean and chosen_dirty:
            backwards.append(row["scenario_id"])
    assert not backwards, f"backwards pairs: {backwards}"


def test_the_pair_constructor_refuses_a_backwards_pair():
    """The runtime chokepoint binds: a mis-oriented pair cannot be constructed."""
    banned_p = next(c for c in INSTANCES.values()
                    if c["type"] == "banned_letter" and c["params"]["letter"] == "p")
    good = "A clean line without that letter at all."
    other = "Another clean line, also entirely free of it."
    bad = "A sample line that does carry the letter."
    assert check(banned_p, good)["pass"] and check(banned_p, other)["pass"]
    assert not check(banned_p, bad)["pass"]

    ok = D.VerifiedPair("dg-00001", "prompt", good, bad, (banned_p,), 1)
    assert ok.violated_constraint_ids == (banned_p["id"],)
    assert ok.violated_types == ("banned_letter",)

    with pytest.raises(D.PairOrientationError, match=r"'chosen' violates"):
        D.VerifiedPair("dg-00001", "prompt", bad, good, (banned_p,), 1)   # backwards
    with pytest.raises(D.PairOrientationError, match="violates nothing"):
        D.VerifiedPair("dg-00001", "prompt", good, other, (banned_p,), 1)  # both clean
    with pytest.raises(D.PairOrientationError, match="same text"):
        D.VerifiedPair("dg-00001", "prompt", good, good + "  ", (banned_p,), 1)
    with pytest.raises(D.PairOrientationError, match="certifies nothing"):
        D.VerifiedPair("dg-00001", "prompt", good, bad, (), 1)


# ==========================================================================
# The prompt contract
# ==========================================================================

def test_every_prompt_round_trips_through_the_shared_renderer(rows, source_rows):
    """The DPO prompt must be the SFT/eval prompt, byte for byte.

    Re-rendered from the scenario the generator stored, through the same function
    configs/train.json names as the prompt renderer.
    """
    scenarios = {}
    for row in source_rows:
        scenarios.setdefault(row["scenario_id"], row["scenario"])
    checked = 0
    for row in rows:
        expected = render_user_message(scenarios[row["scenario_id"]], INSTANCES)
        actual = row["prompt"][0]["content"]
        assert actual == expected, f"{row['scenario_id']}: prompt drifted from the renderer"
        assert actual.encode("utf-8") == expected.encode("utf-8")
        checked += 1
    assert checked == len(rows) > 0


def test_the_schema_is_the_conversational_form_trl_reads(rows):
    for index, row in enumerate(rows):
        assert list(row) == list(D.SCHEMA_KEYS), f"row {index}: unexpected key order/set"
        assert [m["role"] for m in row["prompt"]] == ["user"]
        assert [m["role"] for m in row["chosen"]] == ["assistant"]
        assert [m["role"] for m in row["rejected"]] == ["assistant"]
        for key in ("prompt", "chosen", "rejected"):
            assert row[key][0]["content"].strip(), f"row {index}: empty {key}"


# ==========================================================================
# Train/eval separation
# ==========================================================================

def test_no_pair_prompt_appears_in_the_eval_corpus(rows):
    corpus_prompts = {render_user_message(s, INSTANCES) for s in load_corpus()}
    assert len(corpus_prompts) > 0, "eval corpus rendered nothing; this check is vacuous"
    overlap = {row["prompt"][0]["content"] for row in rows} & corpus_prompts
    assert not overlap, f"{len(overlap)} pair prompts are eval-corpus prompts"


def test_no_pair_scenario_id_appears_in_the_eval_corpus(rows):
    corpus_ids = {s["scenario_id"] for s in load_corpus()}
    assert corpus_ids
    pair_ids = {row["scenario_id"] for row in rows}
    assert not (pair_ids & corpus_ids)
    assert all(D.DATAGEN_ID.match(i) for i in pair_ids), (
        f"non-datagen ids in the pairs: "
        f"{sorted(i for i in pair_ids if not D.DATAGEN_ID.match(i))}")


def test_every_constraint_a_pair_uses_is_a_train_split_instance(rows):
    """A held-out-split constraint in training data would leak the held-out axis."""
    used = {cid for row in rows for cid in row["constraints"]}
    assert used
    held_out = sorted(cid for cid in used if INSTANCES[cid]["split"] != "train")
    assert not held_out, f"held-out constraints used by pairs: {held_out}"


# ==========================================================================
# The manifest describes the file it ships with
# ==========================================================================

def test_manifest_hashes_match_the_files_on_disk(manifest, rows):
    assert manifest["output"]["sha256"] == D.sha256_file(PAIRS)
    assert manifest["output"]["bytes"] == PAIRS.stat().st_size
    assert manifest["source"]["sha256"] == D.sha256_file(SOURCE)
    assert manifest["source"]["path"] == "data/train/v3.dpo_negatives.jsonl"
    assert manifest["output"]["path"] == "data/train/v3.dpo_pairs.jsonl"


def test_manifest_counts_match_the_file(manifest, rows, source_rows):
    from collections import Counter
    assert manifest["source"]["rows"] == len(source_rows)
    assert manifest["output"]["rows"] == len(rows)

    sids = sorted({row["scenario_id"] for row in rows})
    assert manifest["scenarios"]["distinct_scenario_ids"] == len(sids)
    assert manifest["scenarios"]["scenario_ids"] == sids
    assert manifest["scenarios"]["scenario_id_list_sha256"] == D.sha256_text("\n".join(sids))

    per_scenario = Counter(row["scenario_id"] for row in rows)
    assert manifest["scenarios"]["pairs_per_scenario"] == {
        str(n): c for n, c in sorted(Counter(per_scenario.values()).items())}
    assert manifest["scenarios"]["max_pairs_for_one_scenario"] == max(per_scenario.values())

    assert manifest["k_distribution"] == {
        str(k): c for k, c in sorted(Counter(row["k"] for row in rows).items())}

    violated = Counter(t for row in rows for t in row["violated_types"])
    assert manifest["disclosures"]["monoculture"]["violated_constraint_types"] == dict(violated)

    present = Counter(INSTANCES[cid]["type"] for row in rows for cid in row["constraints"])
    assert (manifest["disclosures"]["monoculture"]
            ["constraint_types_present_in_the_prompts"] == dict(present))


def test_manifest_drop_accounting_is_complete_and_enumerated(manifest, source_rows, rows):
    by_reason = manifest["dropped"]["by_reason"]
    assert list(by_reason) == list(D.DROP_REASONS), "drop reasons drifted from the module"
    assert sum(by_reason.values()) == manifest["dropped"]["total"]
    assert manifest["dropped"]["total"] + len(rows) == len(source_rows), (
        "kept + dropped must account for every source row")


def test_manifest_separation_block_is_measured_and_zero(manifest):
    sep = manifest["disclosures"]["train_eval_separation"]
    assert sep["rendered_prompt_overlap_with_eval_corpus"] == 0
    assert sep["scenario_id_overlap_with_eval_corpus"] == 0
    assert sep["non_datagen_scenario_ids_in_pairs"] == []
    assert sep["eval_corpus_distinct_rendered_prompts"] == len(load_corpus())


def test_manifest_attrition_block_matches_the_source(manifest, source_rows, rows):
    attr = manifest["disclosures"]["attrition"]
    absent = sum(1 for row in source_rows if row.get("chosen") is None)
    assert attr["rows_with_no_compliant_draft"] == absent
    assert attr["source_rows"] == len(source_rows)
    assert attr["distinct_scenarios_in_source"] == len({r["scenario_id"] for r in source_rows})
    assert attr["scenarios_with_at_least_one_pair"] == len({r["scenario_id"] for r in rows})
    freq = attr["banned_letter_english_frequency"]
    assert freq["per_letter"], "the attrition comparison covers no letters"
    for entry in freq["per_letter"]:
        total = entry["scenarios_with_a_pair"] + entry["scenarios_without_a_pair"]
        assert total > 0
        assert entry["pair_rate"] == pytest.approx(entry["scenarios_with_a_pair"] / total,
                                                   abs=1e-4)


def test_the_monoculture_disclosure_is_true_of_the_shipped_file(rows, manifest):
    """The headline disclosure, re-derived: every violation is banned_letter."""
    types = {t for row in rows for t in row["violated_types"]}
    assert types == {"banned_letter"}, f"violated types are {sorted(types)}"
    mono = manifest["disclosures"]["monoculture"]
    assert mono["fraction_banned_letter_only"] == 1.0
    assert mono["pairs_whose_violations_are_banned_letter_only"] == len(rows)
    for held in mono["held_out_types_derived_from_the_instance_library"]:
        assert held in mono["registered_types_never_violated_by_any_rejected_text"]
    assert set(mono["held_out_types_derived_from_the_instance_library"]) == {
        t for t in {c["type"] for c in INSTANCES.values()}
        if not any(c["type"] == t and c["split"] == "train" for c in INSTANCES.values())}


# ==========================================================================
# Determinism and the drop classifier
# ==========================================================================

def test_the_build_is_deterministic_and_reproduces_the_shipped_bytes(rebuilt, tmp_path):
    # Compared by digest, not by ==: these are multi-megabyte strings and a bare
    # equality assert makes pytest render a diff of the whole dataset.
    payload = D.dataset_text(rebuilt.pairs)
    assert D.sha256_text(payload) == D.sha256_file(PAIRS), (
        "rebuilding the shipped source does not reproduce the shipped file")
    assert len(payload.encode("utf-8")) == PAIRS.stat().st_size

    again = D.dataset_text(D.build(SOURCE).pairs)
    assert D.sha256_text(again) == D.sha256_text(payload), (
        "two builds of the same input differ")

    out = tmp_path / "v3.dpo_pairs.jsonl"
    manifest_path = tmp_path / "v3.dpo_pairs.manifest.json"
    written, manifest_text = D.write_dataset(SOURCE, out, manifest_path, rebuilt)
    assert D.sha256_text(written) == D.sha256_text(payload)
    assert json.loads(manifest_text)["output"]["sha256"] == D.sha256_file(PAIRS)
    # The manifest carries no wall clock, so it is byte-stable too.
    assert "utc" not in manifest_text.lower() and "timestamp" not in manifest_text.lower()


def test_output_paths_are_derived_from_the_sidecar_name():
    pairs, manifest = D.output_paths(SOURCE)
    assert pairs == PAIRS and manifest == MANIFEST
    with pytest.raises(ValueError):
        D.output_paths(Path("data/train/v3.jsonl"))


def _template_row(source_rows):
    return next(json.loads(json.dumps(r)) for r in source_rows if r.get("chosen"))


def test_every_drop_reason_is_reachable(source_rows):
    """The shipped data only exercises chosen_absent. The other branches are
    exercised here so none of them is dead code that could never fire."""
    corpus_prompts = D.corpus_prompt_index(INSTANCES)
    fired = set()

    base = _template_row(source_rows)
    pair, reason = D.classify_row(base, INSTANCES, corpus_prompts)
    assert pair is not None and reason is None, "the template row should build a pair"

    row = _template_row(source_rows); row["chosen"] = None
    fired.add(D.classify_row(row, INSTANCES, corpus_prompts)[1])

    row = _template_row(source_rows); row["scenario_id"] = "s-0001"
    fired.add(D.classify_row(row, INSTANCES, corpus_prompts)[1])

    row = _template_row(source_rows); row["prompt"] = row["prompt"] + " tampered"
    fired.add(D.classify_row(row, INSTANCES, corpus_prompts)[1])

    row = _template_row(source_rows); row["chosen"] = row["rejected"]
    fired.add(D.classify_row(row, INSTANCES, corpus_prompts)[1])

    # A chosen side that is the violating draft doubled: distinct text, still
    # violating -- so it reaches the orientation check rather than the identity one.
    row = _template_row(source_rows)
    row["chosen"] = row["rejected"] + "\n\n" + row["rejected"]
    fired.add(D.classify_row(row, INSTANCES, corpus_prompts)[1])

    # A rejected side that is the compliant draft doubled: distinct, and clean.
    row = _template_row(source_rows)
    row["rejected"] = row["chosen"] + "\n\n" + row["chosen"]
    fired.add(D.classify_row(row, INSTANCES, corpus_prompts)[1])

    # An eval-corpus prompt carried under a datagen id: the leak branch.
    eval_scenario = load_corpus()[0]
    row = _template_row(source_rows)
    row["scenario"] = dict(eval_scenario, scenario_id=row["scenario_id"])
    row["constraints"] = list(eval_scenario["constraints"])
    row["prompt"] = render_user_message(row["scenario"], INSTANCES)
    row["chosen"], row["rejected"] = "clean text", "clean text too"
    fired.add(D.classify_row(row, INSTANCES, corpus_prompts)[1])

    assert fired == set(D.DROP_REASONS), (
        f"unreachable drop reasons: {sorted(set(D.DROP_REASONS) - fired)}")


def test_build_refuses_an_empty_source(tmp_path):
    empty = tmp_path / "empty.dpo_negatives.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError):
        D.build(empty)


def test_the_frequency_table_covers_every_banned_letter_in_the_library():
    letters = {c["params"]["letter"].lower() for c in INSTANCES.values()
               if c["type"] == "banned_letter"}
    assert letters, "no banned_letter instances; the attrition comparison is vacuous"
    assert letters <= set(D.ENGLISH_LETTER_FREQUENCY_PCT)


# ==========================================================================
# The documentation carries the same numbers as the manifest
# ==========================================================================

def test_the_doc_quotes_the_manifest_not_a_memory(manifest):
    doc = (REPO_ROOT / "docs" / "DPO.md").read_text(encoding="utf-8")
    for number in (str(manifest["output"]["rows"]),
                   str(manifest["source"]["rows"]),
                   str(manifest["dropped"]["by_reason"]["chosen_absent"]),
                   str(manifest["scenarios"]["distinct_scenario_ids"]),
                   manifest["output"]["sha256"][:16]):
        assert number in doc, f"docs/DPO.md does not carry {number!r}"
    assert "No DPO model has been trained" in doc
