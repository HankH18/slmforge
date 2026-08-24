"""The shipped training data is audited by the suite, not by memory.

An adversarial verifier found datagen/audit.py was a manual CLI that nothing
called -- not check_loop, not modal_app, not any test -- so the only gate on the
dataset the project trains and reports on was a human remembering the command.
It also found the auditor certified verbatim eval-corpus prose as clean, because
it re-derived only (task, constraint-set) pairs while datagen's own documented
standard is task-level. Both are bound here.
"""
import json
from collections import Counter
from pathlib import Path

import pytest

from datagen.audit import audit
from scenarios.paths import load_corpus, load_instances, load_smoke

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = REPO_ROOT / "data" / "train"

#: CORPUS COMPONENTS: a pipeline output that is PART of a shipped corpus and is
#: not a corpus itself. Enumerated BY NAME, never by pattern, so this stays a
#: closed world: any other single-dot .jsonl in data/train/ is still a training
#: file and still gets the full audit below.
#:
#: `v4_attacks.jsonl` is the 180-row attack block, written verbatim into
#: `v4.jsonl` as its tail. It cannot satisfy two of the auditor's checks and was
#: never meant to: `k_span` (datagen/attacks.py::_companion_sets composes every
#: attack row at k=3, so no block of them ever spans k=2..4) and `type_share`
#: (the three attack shapes bait a handful of target types, so banned_words is
#: 31.9% of the block's slots against a 25% cap). Both are properties OF A WHOLE
#: CORPUS. docs/RETRAIN_PREREGISTRATION.md sec.5 R2 accordingly gates the audit
#: on v4, and sec.4 publishes the block's evasion count at
#: `data/train/v4_attacks.report.json`, i.e. the frozen spec already treats this
#: file as a component. It is NOT thereby unaudited: see
#: `test_a_corpus_component_is_row_clean_and_lands_in_an_audited_corpus`, which
#: runs the real auditor on it and permits EXACTLY those two findings and no
#: other, then requires every one of its rows to appear in a corpus that passes
#: the audit outright.
COMPONENTS = sorted(p for p in (TRAIN_DIR / "v4_attacks.jsonl",) if p.exists())

# Training datasets only. The sidecars -- <name>.rejected.jsonl and
# <name>.dpo_negatives.jsonl -- deliberately hold VIOLATING rows (that is their
# purpose: candidate DPO negatives), so auditing them for compliance is a category
# error. A training file has exactly one dot-segment before .jsonl.
SHIPPED = sorted(p for p in TRAIN_DIR.glob("*.jsonl")
                 if p.name.count(".") == 1 and p not in COMPONENTS)
SIDECARS = sorted(p for p in TRAIN_DIR.glob("*.jsonl")
                  if p.name.count(".") > 1)


@pytest.mark.parametrize("path", SHIPPED, ids=lambda p: p.name)
def test_every_shipped_training_file_passes_its_own_audit(path):
    """If this fails, the dataset is contaminated or non-compliant -- and nothing
    else in the project would notice."""
    result = audit(str(path), min_rows=1)
    violations, detail = result["violations"], result["detail"]
    assert not violations, f"{path.name} fails its audit: {violations}\n{detail}"


def test_there_is_at_least_one_shipped_training_file():
    """Guards the parametrization above from silently covering nothing."""
    assert SHIPPED, "no data/train/*.jsonl; the audit parametrization covers nothing"


def _row_with_task(task_text, corpus_row, instances):
    """A schema-valid, genuinely compliant row carrying the given task text."""
    from harness import build_constraint_lookup
    from scenarios.render import render_user_message
    from scenarios.reachability import build_witness
    lookup = build_constraint_lookup(instances.values())
    cids = [c for c in corpus_row["constraints"]
            if instances[c]["split"] == "train"] or corpus_row["constraints"][:1]
    scen = {"scenario_id": "planted", "task": task_text, "constraints": cids,
            "k": len(cids), "pressure": "clean", "split": "train"}
    witness = build_witness([instances[c] for c in cids], sentences=4)
    if witness is None:
        pytest.skip("no witness available for this constraint set")
    return {"messages": [{"role": "user", "content": render_user_message(scen, lookup)},
                         {"role": "assistant", "content": witness}],
            "constraints": cids,
            "provenance": {"teacher": "test", "attempt": 1},
            "scenario": scen}


def test_audit_rejects_verbatim_eval_prose_under_a_different_constraint_set(tmp_path):
    """FINDING B. Reusing an eval task's exact prose under other constraints was
    certified 'zero eval overlap'. A model that has seen the exact text it will be
    graded on is contaminated regardless of which constraints accompanied it."""
    instances = load_instances()
    corpus = list(load_corpus())
    victim = corpus[0]
    row = _row_with_task(victim["task"], victim, instances)
    f = tmp_path / "planted.jsonl"
    f.write_text(json.dumps(row) + "\n")
    result = audit(str(f), min_rows=1)
    violations = result["violations"]
    assert "eval_task_overlap" in violations, (
        f"verbatim eval prose was certified clean; violations={violations}")


def test_audit_still_rejects_the_exact_eval_pair(tmp_path):
    """The pair-level check must survive the task-level addition."""
    instances = load_instances()
    victim = list(load_corpus())[0]
    row = _row_with_task(victim["task"], victim, instances)
    row["constraints"] = victim["constraints"]
    row["scenario"]["constraints"] = victim["constraints"]
    row["scenario"]["k"] = len(victim["constraints"])
    f = tmp_path / "pair.jsonl"
    f.write_text(json.dumps(row) + "\n")
    result = audit(str(f), min_rows=1)
    assert result["violations"], "the exact eval pair was certified clean"


def test_audit_is_not_vacuous_on_an_empty_file(tmp_path):
    f = tmp_path / "empty.jsonl"
    f.write_text("")
    result = audit(str(f), min_rows=1)
    assert result["violations"], "an empty dataset passed the audit"


@pytest.mark.parametrize("path", SIDECARS, ids=lambda p: p.name)
def test_sidecars_are_not_mistaken_for_training_data(path):
    """The rejected/DPO-negative sidecars hold violating rows on purpose. This
    pins that they are NOT in the audited set, so a future rename cannot quietly
    slip violating rows into training."""
    assert path not in SHIPPED
    # ".dpo_pairs.jsonl" added 2026-08-22 for the DPO stretch rung. This gate is a
    # CLOSED-WORLD CLASSIFIER: it exists so a new multi-dot file in data/train/ cannot
    # arrive unclassified, and it correctly flagged v3.dpo_pairs.jsonl when that file
    # first appeared. The enumeration is extended rather than the assertion weakened --
    # an unrecognised name still fails, and the new kind carries its own contract below.
    # A .dpo_pairs.jsonl holds (prompt, chosen, rejected) preference rows whose REJECTED
    # side violates a constraint on purpose, exactly like the other two sidecars, which
    # is why it must stay out of SHIPPED and out of datagen.audit's `messages` schema.
    # Per-row verification of every pair against the pinned checkers lives in
    # tests/test_dpo_pairs.py; this only pins the classification and the shape.
    # ".draft.jsonl" added 2026-08-23 for the v4 attack block. This gate fired on
    # v4_format_compulsion.draft.jsonl the moment those files appeared, which is
    # the gate working. A .draft.jsonl holds TEACHER PROSE keyed by scenario id --
    # {"scenario_id", "draft"} -- the injected generator datagen/generate.py's
    # DraftFile reads instead of calling a provider. It is an INPUT to generation,
    # not an output of it: it carries no prompt, no checker verdict and no
    # provenance, so auditing it as training data would be the same category error
    # as auditing the rejection ledger. The rows it produced ARE audited, inside
    # the corpus they were written into. Contract pinned below.
    assert path.name.endswith(
        (".rejected.jsonl", ".dpo_negatives.jsonl", ".dpo_pairs.jsonl",
         ".draft.jsonl")), (
        f"{path.name} is neither a training file nor a recognised sidecar -- it is "
        f"unaudited and unclassified")

    if path.name.endswith(".dpo_pairs.jsonl"):
        # Bind the new category: a preference file must actually be preference-shaped,
        # so a bag of raw SFT rows cannot be renamed into this class to dodge the audit.
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        assert rows, f"{path.name} is empty"
        for i, row in enumerate(rows):
            missing = {"prompt", "chosen", "rejected"} - set(row)
            assert not missing, (
                f"{path.name} row {i} is missing {sorted(missing)} -- a .dpo_pairs.jsonl "
                f"must be (prompt, chosen, rejected) preference rows, not training rows")
            assert "messages" not in row, (
                f"{path.name} row {i} carries a 'messages' key: this looks like SFT "
                f"training data renamed into the preference-sidecar class")

    if path.name.endswith(".draft.jsonl"):
        # Bind the new category the same way: a drafts file must actually be
        # draft-shaped -- the two keys DraftFile reads and nothing that would let
        # a bag of finished training rows be renamed into this class to dodge the
        # audit. Duplicate scenario ids are checked too, because DraftFile refuses
        # a file carrying two answers for one scenario and that refusal should
        # surface here rather than mid-generation.
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        assert rows, f"{path.name} is empty"
        seen = set()
        for i, row in enumerate(rows):
            assert isinstance(row.get("scenario_id"), str) and row["scenario_id"], (
                f"{path.name} row {i} has no 'scenario_id'; DraftFile could not key it")
            drafts = row.get("drafts")
            if drafts is None and isinstance(row.get("draft"), str):
                drafts = [row["draft"]]
            assert isinstance(drafts, list) and drafts and all(
                isinstance(d, str) and d.strip() for d in drafts), (
                f"{path.name} row {i} ({row['scenario_id']}) carries no non-empty "
                f"draft text -- a .draft.jsonl is teacher prose, nothing else")
            assert "messages" not in row and "provenance" not in row, (
                f"{path.name} row {i} carries 'messages'/'provenance': this looks "
                f"like generated training data renamed into the drafts class")
            assert row["scenario_id"] not in seen, (
                f"{path.name} row {i} repeats scenario {row['scenario_id']}; "
                f"DraftFile refuses a file with two answers for one scenario")
            seen.add(row["scenario_id"])


# ==========================================================================
# The nested family the data-efficiency curve rests on
# ==========================================================================

def _row_key(row):
    """Content identity, ignoring the wall-clock stamp a rerun necessarily moves."""
    import copy
    row = copy.deepcopy(row)
    row.get("provenance", {}).pop("generated_utc", None)
    return json.dumps(row, sort_keys=True)


def test_v3_contains_every_v2_row():
    """The upward extension of the curve is only interpretable if the larger
    corpus CONTAINS the smaller one -- otherwise the points differ by which rows
    they drew as well as by how many, and the difference is unattributable.

    Set inclusion, deliberately not prefix equality. The v2 run abandoned rows on
    a provider error, a failed call is never cached, so those requests were
    re-issued during the v3 run and some succeeded; the two files agree for a
    few hundred rows and then interleave. Asserting prefix equality here would
    fail on a correct dataset.
    """
    v2 = REPO_ROOT / "data" / "train" / "v2.jsonl"
    v3 = REPO_ROOT / "data" / "train" / "v3.jsonl"
    if not v3.exists():
        pytest.skip("v3 has not been generated")
    keys2 = {_row_key(json.loads(l)) for l in v2.read_text().splitlines() if l.strip()}
    keys3 = {_row_key(json.loads(l)) for l in v3.read_text().splitlines() if l.strip()}
    missing = keys2 - keys3
    assert not missing, (
        f"{len(missing)} of {len(keys2)} v2 rows are absent from v3; the curve's "
        f"nested family is broken and the points are no longer comparable")
    assert len(keys3) > len(keys2), "v3 must be strictly larger than v2"


def test_v4_contains_every_v3_row():
    """The next rung of the same ladder, and the reason it is asserted separately.

    ``docs/RETRAIN_PREREGISTRATION.md`` sec.5 R3 fixes this in advance: "v4
    contains v3 -- all 1,815 v3 rows present, byte-compared". v4 is v3 plus a
    180-row adversarial attack block, and the whole point of the v3-vs-v4
    comparison is that the ONLY difference between them is those 180 rows. If a
    v3 row were dropped or rewritten during composition, a change in the tuned
    model could be caused by the loss as easily as by the addition, and nothing
    downstream would show it.

    Set inclusion again, matching the v2/v3 assertion above, and by the same
    content identity (the wall-clock stamp a rerun necessarily moves is ignored).
    """
    v3 = REPO_ROOT / "data" / "train" / "v3.jsonl"
    v4 = REPO_ROOT / "data" / "train" / "v4.jsonl"
    if not v4.exists():
        pytest.skip("v4 has not been generated")
    lines3 = [l for l in v3.read_text().splitlines() if l.strip()]
    lines4 = [l for l in v4.read_text().splitlines() if l.strip()]
    keys3 = {_row_key(json.loads(l)) for l in lines3}
    keys4 = {_row_key(json.loads(l)) for l in lines4}
    missing = keys3 - keys4
    assert not missing, (
        f"{len(missing)} of {len(keys3)} v3 rows are absent from v4; the curve's "
        f"nested family is broken and the points are no longer comparable")
    assert len(keys4) > len(keys3), "v4 must be strictly larger than v3"
    # R3 says BYTE-compared, so say it in bytes as well as in content identity:
    # v3 is copied through as the literal prefix of v4, line for line.
    assert lines4[:len(lines3)] == lines3, (
        "v4's first rows are not v3's lines verbatim; v3 was re-serialised or "
        "reordered during composition rather than copied through")


# ==========================================================================
# Corpus components: audited row-wise, and never audited as a corpus
# ==========================================================================

#: The two auditor checks that are properties of a WHOLE CORPUS rather than of any
#: row, and are therefore the only findings a component may carry. Everything else
#: the auditor computes -- schema, byte-identical prompt re-render, every checker
#: re-run, held-out types, eval-task and eval-pair overlap, partner-break overlap,
#: duplicate pairs, k in 2..4 per set, a negative per set -- still has to be clean.
WHOLE_CORPUS_CHECKS = {"type_share", "k_span"}


def test_there_is_at_least_one_corpus_component():
    """Guards the parametrization below from silently covering nothing, and pins
    that COMPONENTS is not quietly swallowing files that should be audited."""
    assert COMPONENTS, (
        "COMPONENTS is empty; either the v4 attack block was never written or the "
        "enumeration has drifted from the files on disk")


@pytest.mark.parametrize("path", COMPONENTS, ids=lambda p: p.name)
def test_a_corpus_component_is_row_clean_and_lands_in_an_audited_corpus(path):
    """A component is exempt from two checks, not from the auditor.

    This is the price of taking `v4_attacks.jsonl` out of SHIPPED. Two things are
    asserted and both have to hold:

    1. the REAL auditor runs on the component and reports findings that are a
       SUBSET of the two whole-corpus checks. Not "some findings are allowed" --
       one row with a held-out constraint type, one prompt that no longer
       re-renders byte-identically, one task that overlaps the partner's exam,
       and this fails exactly as it would for a shipped corpus.
    2. every row of the component appears in a file that passed the full audit
       above. A component that is not actually part of any corpus is a file
       nobody grades, which is the state this whole module exists to prevent.
    """
    result = audit(str(path), min_rows=1)
    violations = set(result["violations"])
    unexpected = violations - WHOLE_CORPUS_CHECKS
    assert not unexpected, (
        f"{path.name} fails {sorted(unexpected)}, which are ROW-level checks a "
        f"component is not excused from:\n{result['detail']}")

    component_keys = {_row_key(json.loads(l))
                      for l in path.read_text().splitlines() if l.strip()}
    assert component_keys, f"{path.name} is empty"
    homes = []
    for corpus in SHIPPED:
        keys = {_row_key(json.loads(l))
                for l in corpus.read_text().splitlines() if l.strip()}
        if component_keys <= keys:
            homes.append(corpus.name)
    assert homes, (
        f"every row of {path.name} should appear in some shipped corpus that "
        f"passes the full audit, and none does. The component is ungraded.")


def test_the_attack_block_names_its_teacher_truthfully():
    """The ledger must name whoever actually wrote the prose.

    These 180 completions were drafted by a Claude subagent iterating against
    `constraints.checkers`, injected through `datagen/generate.py --drafts`, and
    the run recorded `anthropic:claude-opus-5`. That string is in
    `configs/models.json:frontier`, so it is a real frontier teacher and the
    assignment's "distilling from a frontier teacher" holds on its face -- but
    the reason it is asserted here is the opposite one: no provider call was made,
    so an `openai:` spec on this text would be a fabricated provenance that
    nothing else in the project could detect.
    """
    block = REPO_ROOT / "data" / "train" / "v4_attacks.jsonl"
    if not block.exists():
        pytest.skip("the v4 attack block has not been generated")
    rows = [json.loads(l) for l in block.read_text().splitlines() if l.strip()]
    teachers = {row["provenance"]["teacher"] for row in rows}
    assert teachers == {"anthropic:claude-opus-5"}, (
        f"the attack block's teachers are {sorted(teachers)}; the drafts were "
        f"written by a Claude subagent and the ledger must say so")
    shapes = Counter(row["provenance"]["attack"]["shape"] for row in rows)
    assert set(shapes) == {"format_compulsion", "subject_compulsion",
                           "erosion_recap"}, (
        f"the block should span all three attack shapes, got {dict(shapes)}. A "
        f"block missing a shape teaches only the attacks it happens to contain.")
    pools = {row["provenance"]["attack"]["pool"] for row in rows}
    assert pools == {"A"}, (
        f"the block draws from pool(s) {sorted(pools)}; pool B is the never-trained "
        f"held-out attack eval and training on it makes the generalisation claim "
        f"unfalsifiable")


def test_the_evasion_count_is_published_and_is_a_real_number():
    """`docs/RETRAIN_PREREGISTRATION.md` sec.4 (falsifier F5) pre-registers this.

    Under format compulsion the cheapest draft that passes every checker is a
    refusal, and the pre-registration says the count of drafts rejected for it is
    published in `data/train/v4_attacks.report.json`. A missing key is worse than
    a bad number: it means the falsifier cannot be read at all.
    """
    report = REPO_ROOT / "data" / "train" / "v4_attacks.report.json"
    if not report.exists():
        pytest.skip("the v4 attack block has not been generated")
    payload = json.loads(report.read_text())
    assert "evasive_drafts" in payload, (
        "v4_attacks.report.json publishes no 'evasive_drafts'; F5 is unreadable")
    assert isinstance(payload["evasive_drafts"], int)
    assert payload["teacher"] == "anthropic:claude-opus-5"
    assert payload["teacher_resolved_by"] == "--teacher"
    assert payload["drafts_file"], (
        "the report does not name the drafts file, so the text's author is "
        "recorded nowhere a reader can check")


def test_the_evasion_gate_would_have_bitten_this_block():
    """The zero is a real zero, and this is what makes it readable as one.

    `evasive_drafts: 0` has two possible causes -- the teacher never wrote an
    evasion, or the gate is dead on rows of this shape -- and the report cannot
    tell them apart. So: plant the cheapest reward hack into the REAL scenarios
    of the shipped block, assert the precondition that it passes every real
    checker (i.e. the pre-gate filter would have KEPT it), and require the gate to
    reject every one. Without this, a near-zero count is uninterpretable, which is
    precisely the reading `docs/RETRAIN_PREREGISTRATION.md` sec.4 warns about.
    """
    from datagen import filter as F
    from harness import build_constraint_lookup, run_checkers
    from scenarios.reachability import build_witness

    block = REPO_ROOT / "data" / "train" / "v4_attacks.jsonl"
    if not block.exists():
        pytest.skip("the v4 attack block has not been generated")
    instances = load_instances()
    lookup = build_constraint_lookup(instances.values())
    rows = [json.loads(l) for l in block.read_text().splitlines() if l.strip()]

    # No shipped completion trips the gate -- that is the reported zero.
    flagged = [r["scenario"]["scenario_id"] for r in rows
               if F.is_evasive(r["messages"][1]["content"])]
    assert not flagged, f"shipped rows flagged as evasive: {flagged[:5]}"

    # And the gate is not asleep: the planted refusal is caught on every row where
    # the checkers alone would have let it through.
    hack_prefix = ("I cannot give you that as a numbered list, so here is a "
                   "paragraph instead. ")
    would_have_been_kept = 0
    escaped = []
    for row in rows:
        scen = row["scenario"]
        witness = build_witness([instances[c] for c in scen["constraints"]],
                                sentences=6)
        if witness is None:
            continue
        hack = hack_prefix + witness
        if not all(v["pass"] for v in run_checkers(scen, lookup, hack)):
            continue  # the row's own constraints already kill this phrasing
        would_have_been_kept += 1
        if not F.is_evasive(hack):
            escaped.append(scen["scenario_id"])
    assert would_have_been_kept >= 50, (
        f"only {would_have_been_kept} of {len(rows)} block rows admit a "
        f"checker-passing refusal, so this test proves almost nothing; the "
        f"non-vacuity of the reported zero is no longer established here")
    assert not escaped, (
        f"{len(escaped)} planted refusals passed every checker AND the evasion "
        f"gate ({escaped[:5]}); the reported evasive_drafts=0 cannot be read as "
        f"'the teacher never evaded'")
