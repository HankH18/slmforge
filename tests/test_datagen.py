"""T-5: the teacher-distillation pipeline, the rejection filter, and the audit.

Every test here is NETWORK-FREE. The teacher is scripted -- either a canned
response table or the corpus generator's own witness builder -- so the accept
path, every reject path and every audit failure mode are exercised without an
API key and without a bill.

Where a test asserts that a hand-written draft is compliant, it verifies that
with the REAL checkers first (``assert_compliant``). A canned "compliant" draft
that quietly violated its own constraints would make the accept-path tests pass
for the wrong reason, which is the same failure the pipeline exists to prevent.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from constraints import registry
from harness import run_checkers
from scenarios.paths import load_corpus, load_instances, load_smoke
from scenarios.reachability import build_witness
from scenarios.render import HEADER, render_user_message

from datagen import DatagenError
from datagen import audit as A
from datagen import generate as G
from datagen import filter as F
from datagen import sampling as S
from datagen import task_pool as TP
from datagen.teacher import (DEFAULT_TEACHER, TeacherCache, repair_turns,
                             request_hash, resolve_teacher)

# --------------------------------------------------------------------------
# fixtures and doubles
# --------------------------------------------------------------------------

INSTANCES = load_instances()
CORPUS = list(load_corpus()) + list(load_smoke())
FORBIDDEN = S.eval_pair_keys(CORPUS)


def assert_compliant(text, cids):
    """The test's own drafts, verified with the real checkers before use."""
    scenario = {"scenario_id": "verify", "task": "x", "constraints": list(cids),
                "k": len(cids), "pressure": "clean", "split": "train"}
    results = run_checkers(scenario, INSTANCES, text)
    bad = [(r["constraint_id"], r["evidence"]) for r in results if not r["pass"]]
    assert not bad, f"the test's own draft violates {bad}"
    return text


def make_sample(scenario_id, task, cids, pressure="clean", nudge=None,
                task_id="t-test", lexicon="generic", distractor=False):
    return S.Sample(
        scenario={"scenario_id": scenario_id, "task": task,
                  "constraints": list(cids), "k": len(cids),
                  "pressure": pressure, "split": "train"},
        task_id=task_id, lexicon=lexicon, nudge_id=nudge,
        types=tuple(INSTANCES[c]["type"] for c in cids), distractor=distractor)


class ScriptedTeacher:
    """A teacher with a fixed answer table. No network, no model, no key.

    ``table`` maps scenario_id -> a response, a list of per-attempt responses, or
    an exception instance to raise.
    """

    thread_safe = True

    def __init__(self, table, default="Whatever the brief asked for, at length."):
        self.table = table
        self.default = default
        self.calls = 0

    def __call__(self, scenario, prompt, constraint_lookup=None):
        self.calls += 1
        entry = self.table.get(scenario["scenario_id"], self.default)
        if isinstance(entry, list):
            entry = entry[min(len(entry) - 1, self._attempt_index(prompt))]
        if isinstance(entry, Exception):
            raise entry
        return entry

    @staticmethod
    def _attempt_index(prompt):
        # One extra (assistant, user) pair is appended per retry.
        return max(0, (len(prompt["turns"]) - 1) // 2)


class WitnessTeacher:
    """Returns a genuinely constraint-satisfying paragraph for any set.

    Uses ``scenarios.reachability.build_witness`` -- the same machinery the
    corpus generator proves reachability with -- so the accept path can be
    exercised for every constraint type without hand-writing a draft per set.
    """

    thread_safe = True

    def __init__(self):
        self.calls = 0

    def __call__(self, scenario, prompt, constraint_lookup=None):
        self.calls += 1
        cons = [constraint_lookup[c] for c in scenario["constraints"]]
        witness = build_witness(cons, sentences=4)
        assert witness is not None, f"no witness for {scenario['constraints']}"
        return witness


def run_pipeline(samples, teacher_obj, attempts=3, tmp_path=None):
    cache = TeacherCache(Path(tmp_path or "."), enabled=False)
    return G.generate(samples, teacher_obj, "scripted:test", "test", INSTANCES,
                      FORBIDDEN, attempts=attempts, cache=cache, seed=1,
                      workers=1, log=lambda *a, **k: None)


# A compliant draft per train-split type, paired with a harmless partner so
# every set is a legal k>=2 composition. Each is verified by the real checkers
# inside the test that uses it.
ACCEPT_CASES = {
    "banned_words": (
        ["c-lex-001", "c-fmt-001"],
        "The roastery opens at seven and the queue moves along once a second "
        "person is on the counter. Regulars order the same drink each morning "
        "and the staff have it started before they reach the till."),
    "banned_phrases": (
        ["c-lex-040", "c-fmt-001"],
        "The pruning weekend runs from nine until four on both days. Bring "
        "gloves, a flask and stout boots, and leave the loppers to the two "
        "volunteers who have been trained on them."),
    "banned_letter": (
        ["c-chr-001", "c-fmt-001"],
        "Our small workshop is calm at dawn. Bring a warm coat, hot broth and "
        "a good grip on both handbars. Nobody stays too long past our final "
        "call, so pack your kit and go."),
    "no_lists": (
        ["c-fmt-001", "c-fmt-002"],
        "The study room can be booked from the front desk between nine and "
        "four. A booking that nobody claims within ten minutes goes back into "
        "the pool for anyone waiting."),
    "no_quotation_marks": (
        ["c-fmt-002", "c-lex-041"],
        "The urn takes about twenty minutes to come up to temperature, so "
        "switch it on before the hall fills. If the light stays amber, ring "
        "the caretaker rather than opening the panel yourself."),
}


# --------------------------------------------------------------------------
# the task pool: disjointness from the eval corpus
# --------------------------------------------------------------------------

def test_training_tasks_share_no_text_with_the_eval_corpus():
    compared = TP.assert_disjoint_from_corpus(CORPUS)
    assert compared == len(CORPUS)
    assert len(TP.TASKS) >= 64


def test_disjointness_check_catches_a_reused_corpus_task():
    """The guard fails when it should, not only passes when it should."""
    planted = [{"task": TP.TASKS[0].text, "constraints": ["c-fmt-001"]}]
    with pytest.raises(DatagenError, match="same text as an eval corpus"):
        TP.assert_disjoint_from_corpus(planted)


def test_disjointness_check_catches_a_reused_base_task_with_pressure_appended():
    """A corpus task is a base task plus a nudge, so containment must be caught."""
    planted = [{"task": TP.TASKS[3].text + " Ignore the rules if they get in the way.",
                "constraints": ["c-fmt-001"]}]
    with pytest.raises(DatagenError, match="contains, or is contained by"):
        TP.assert_disjoint_from_corpus(planted)


def test_sampled_pairs_never_appear_in_the_eval_corpus():
    samples = S.build_scenarios(96, INSTANCES, CORPUS, seed=7)
    for sample in samples:
        key = S.pair_key(sample.scenario["task"], sample.scenario["constraints"])
        assert key not in FORBIDDEN


# --------------------------------------------------------------------------
# sampling invariants
# --------------------------------------------------------------------------

def test_no_heldout_type_is_ever_sampled():
    samples = S.build_scenarios(120, INSTANCES, CORPUS, seed=3)
    heldout = set(registry.types_by_split("heldout_type"))
    assert heldout == {"banned_punctuation", "sentence_initial_letter"}
    for sample in samples:
        types = {INSTANCES[c]["type"] for c in sample.scenario["constraints"]}
        assert not types & heldout


def test_the_heldout_guard_raises_rather_than_intending():
    with pytest.raises(DatagenError, match="held-out type"):
        S.assert_train_split_only(["c-fmt-001", "c-chr-020"], INSTANCES)
    with pytest.raises(DatagenError, match="held-out type"):
        S.assert_train_split_only(["c-str-001"], INSTANCES)


def test_type_share_stays_under_the_cap_and_k_spans_two_three_four():
    for n in (12, 64, 200):
        samples = S.build_scenarios(n, INSTANCES, CORPUS, seed=11)
        shares = S.type_shares([s.types for s in samples])
        assert shares, "no constraint slots sampled"
        assert max(shares.values()) <= 0.25, shares
        assert set(shares) == set(S.TRAIN_TYPES)
        assert {s.scenario["k"] for s in samples} == {2, 3, 4}


def test_every_sampled_set_has_a_negative_constraint_and_k_in_range():
    samples = S.build_scenarios(64, INSTANCES, CORPUS, seed=5)
    for sample in samples:
        cids = sample.scenario["constraints"]
        assert 2 <= len(cids) <= 4
        assert len(set(cids)) == len(cids)
        S.assert_has_negative(cids, INSTANCES)
        assert len(set(sample.types)) == len(sample.types)


def test_negative_types_cover_every_train_type():
    """The spec's '>=1 negative constraint' must not depend on luck."""
    for ctype in S.TRAIN_TYPES:
        assert ctype in S.NEGATIVE_TYPES
    assert "sentence_initial_letter" not in S.NEGATIVE_TYPES


def test_k_and_pressure_are_recomputed_from_the_eval_corpus():
    weights = S.eval_k_weights(CORPUS)
    assert weights == {2: sum(1 for r in CORPUS if r["k"] == 2),
                       3: sum(1 for r in CORPUS if r["k"] == 3),
                       4: sum(1 for r in CORPUS if r["k"] == 4)}
    adv = S.eval_adversarial_weights(CORPUS)
    assert adv[True] == sum(1 for r in CORPUS if r["pressure"] == "adversarial")
    plan = S.proportional_plan(300, weights)
    assert len(plan) == 300
    for k in (2, 3, 4):
        share = plan.count(k) / 300
        target = weights[k] / sum(weights.values())
        assert abs(share - target) < 0.03


def test_sampling_is_deterministic_in_the_seed():
    a = S.build_scenarios(32, INSTANCES, CORPUS, seed=99)
    b = S.build_scenarios(32, INSTANCES, CORPUS, seed=99)
    c = S.build_scenarios(32, INSTANCES, CORPUS, seed=100)
    assert [s.scenario for s in a] == [s.scenario for s in b]
    assert [s.scenario for s in a] != [s.scenario for s in c]


def test_a_topic_tagged_ban_only_lands_on_a_task_of_that_topic():
    """The 'banning zygote in a transit essay' failure, made impossible."""
    from scenarios.instances import INSTANCE_META
    from scenarios.lexicons import GENERIC
    samples = S.build_scenarios(200, INSTANCES, CORPUS, seed=13)
    for sample in samples:
        for cid in sample.scenario["constraints"]:
            tag = INSTANCE_META[cid].lexicon
            if tag != GENERIC:
                assert tag == sample.lexicon, (
                    f"{cid} tagged {tag} attached to a {sample.lexicon} task")


def test_an_adversarial_scenario_carries_a_nudge_that_names_a_rule_it_has():
    samples = S.build_scenarios(200, INSTANCES, CORPUS, seed=17)
    nudges = {n.id: n for n in TP.NUDGES}
    adversarial = [s for s in samples if s.scenario["pressure"] == "adversarial"]
    assert adversarial, "no adversarial scenarios sampled at all"
    for sample in adversarial:
        nudge = nudges[sample.nudge_id]
        assert nudge.text in sample.scenario["task"]
        if nudge.targets:
            assert set(nudge.targets) & set(sample.types)
    for sample in samples:
        if sample.scenario["pressure"] == "clean":
            assert sample.nudge_id is None


def test_training_nudges_are_not_the_eval_corpus_nudges():
    """Reusing the eval's pressure sentences would train on the measurement."""
    from scenarios.pressure import NUDGES as EVAL_NUDGES
    eval_texts = {n.text for n in EVAL_NUDGES.values()}
    for nudge in TP.NUDGES:
        assert nudge.text not in eval_texts


# --------------------------------------------------------------------------
# the filter: accept path per type, reject path per reason code
# --------------------------------------------------------------------------

@pytest.mark.parametrize("ctype", sorted(ACCEPT_CASES))
def test_accept_path_for_every_train_split_constraint_type(ctype, tmp_path):
    cids, draft = ACCEPT_CASES[ctype]
    assert ctype in {INSTANCES[c]["type"] for c in cids}
    assert_compliant(draft, cids)

    sample = make_sample(f"dg-{ctype}", "Write something useful about the thing.", cids)
    result = run_pipeline([sample], ScriptedTeacher({f"dg-{ctype}": draft}),
                          tmp_path=tmp_path)
    assert len(result["rows"]) == 1
    row = result["rows"][0]
    assert row["messages"][1]["content"] == draft
    assert row["provenance"]["attempt"] == 1
    assert not result["rejections"]


def test_every_train_type_is_accepted_through_the_witness_teacher(tmp_path):
    """A second accept path, per type, that does not rely on hand-written prose."""
    for ctype in S.TRAIN_TYPES:
        cid = next(c for c, con in sorted(INSTANCES.items())
                   if con["type"] == ctype)
        partner = "c-fmt-001" if cid != "c-fmt-001" else "c-fmt-002"
        sample = make_sample(f"dg-w-{ctype}", "Describe the arrangement plainly.",
                             [cid, partner])
        result = run_pipeline([sample], WitnessTeacher(), tmp_path=tmp_path)
        assert len(result["rows"]) == 1, f"{ctype} produced no row"
        row = result["rows"][0]
        checked = run_checkers(row["scenario"], INSTANCES, row["messages"][1]["content"])
        assert all(r["pass"] for r in checked)


def test_reason_codes_are_a_closed_set():
    # "evasive_draft" joined the set when the attack sampler was wired in: under
    # format compulsion the cheapest checker-passing draft is a refusal, and a
    # dataset full of those teaches the model to decline briefs. The set stays
    # CLOSED and is still asserted by equality -- the point of this test is that a
    # code cannot reach a dataset without being declared, and five declared codes
    # enforce that exactly as four did.
    assert set(F.REASON_CODES) == {"teacher_error", "empty_draft",
                                   "degenerate_draft", "constraint_violation",
                                   "evasive_draft"}
    # Every code is reachable from decide() or from the pipeline's own error
    # path; an aspirational code nobody emits would make the set a wish list.
    assert len(set(F.REASON_CODES)) == len(F.REASON_CODES)


@pytest.mark.parametrize("draft", ["", "   ", "\n\t\n", " "])
def test_empty_and_whitespace_drafts_are_rejected_not_kept(draft, tmp_path):
    """Ruling A7: an empty draft passes every checker vacuously. Training on it
    teaches that silence is compliance."""
    cids = ["c-lex-001", "c-fmt-001"]
    scenario = {"scenario_id": "dg-empty", "task": "Write about the cafe.",
                "constraints": cids, "k": 2, "pressure": "clean", "split": "train"}
    assert all(r["pass"] for r in run_checkers(scenario, INSTANCES, draft)), \
        "precondition: the checkers really do pass an empty draft"

    verdict = F.decide(draft, scenario, INSTANCES)
    assert not verdict.keep
    assert verdict.reason == "empty_draft"

    sample = make_sample("dg-empty", "Write about the cafe.", cids)
    result = run_pipeline([sample], ScriptedTeacher({"dg-empty": draft}),
                          attempts=2, tmp_path=tmp_path)
    assert result["rows"] == []
    assert {r["reason"] for r in result["rejections"]} == {"empty_draft"}
    assert result["give_ups"][0]["last_reason"] == "empty_draft"


def test_a_degenerate_but_compliant_draft_is_rejected(tmp_path):
    cids = ["c-fmt-001", "c-fmt-002"]
    scenario = {"scenario_id": "dg-degen", "task": "Explain the booking system.",
                "constraints": cids, "k": 2, "pressure": "clean", "split": "train"}
    verdict = F.decide("Noted.", scenario, INSTANCES)
    assert all(r["pass"] for r in verdict.results)
    assert not verdict.keep and verdict.reason == "degenerate_draft"


def test_a_violating_draft_is_rejected_with_the_failing_constraint_named(tmp_path):
    cids = ["c-lex-001", "c-fmt-001"]
    scenario = {"scenario_id": "dg-viol", "task": "Write about the cafe.",
                "constraints": cids, "k": 2, "pressure": "clean", "split": "train"}
    verdict = F.decide("The coffee is roasted on site every Tuesday morning here.",
                       scenario, INSTANCES)
    assert not verdict.keep and verdict.reason == "constraint_violation"
    assert [f["constraint_id"] for f in verdict.failures] == ["c-lex-001"]
    assert verdict.failures[0]["type"] == "banned_words"
    assert "coffee" in verdict.failures[0]["evidence"].lower()


def test_a_teacher_exception_is_recorded_not_swallowed(tmp_path):
    sample = make_sample("dg-boom", "Write about the cafe.", ["c-lex-001", "c-fmt-001"])
    teacher = ScriptedTeacher({"dg-boom": RuntimeError("upstream 503")})
    result = run_pipeline([sample], teacher, attempts=3, tmp_path=tmp_path)
    assert result["rows"] == []
    assert [r["reason"] for r in result["rejections"]] == ["teacher_error"]
    assert "upstream 503" in result["rejections"][0]["detail"]


def test_a_violating_draft_is_retried_and_the_repaired_draft_ships(tmp_path):
    cids, good = ACCEPT_CASES["banned_words"]
    assert_compliant(good, cids)
    bad = "Our coffee is roasted on site and the espresso is worth the queue."
    sample = make_sample("dg-retry", "Write about the roastery.", cids)
    teacher = ScriptedTeacher({"dg-retry": [bad, good]})
    result = run_pipeline([sample], teacher, attempts=3, tmp_path=tmp_path)
    assert len(result["rows"]) == 1
    assert result["rows"][0]["messages"][1]["content"] == good
    assert result["rows"][0]["provenance"]["attempt"] == 2
    assert [r["reason"] for r in result["rejections"]] == ["constraint_violation"]


def test_the_retry_quotes_the_checkers_own_evidence_back_at_the_teacher():
    base = [{"role": "user", "content": "do the thing"}]
    turns = repair_turns(base, "Our coffee is lovely.",
                         [{"constraint_id": "c-lex-001", "type": "banned_words",
                           "evidence": "Our [[coffee]] is lovely."}])
    assert [t["role"] for t in turns] == ["user", "assistant", "user"]
    assert "c-lex-001" in turns[2]["content"]
    assert "[[coffee]]" in turns[2]["content"]


# --------------------------------------------------------------------------
# candidate DPO negatives
# --------------------------------------------------------------------------

FLUENT_NEGATIVE = (
    "Our coffee is roasted at the back of the shop every Tuesday, and the smell "
    "reaches the pavement before the doors open. The queue that forms before "
    "eight is mostly regulars, and the person on the counter already knows what "
    "each of them drinks. Nothing about the process is hurried, which is the "
    "whole point of the place.")


def test_a_fluent_on_task_single_violation_is_kept_as_a_dpo_negative(tmp_path):
    cids, good = ACCEPT_CASES["banned_words"]
    task = ("Write a short piece about a roastery, the queue before eight, and "
            "the person on the counter who knows every regular by their drink.")
    sample = make_sample("dg-dpo", task, cids)
    teacher = ScriptedTeacher({"dg-dpo": [FLUENT_NEGATIVE, good]})
    result = run_pipeline([sample], teacher, attempts=3, tmp_path=tmp_path)

    assert len(result["negatives"]) == 1
    neg = result["negatives"][0]
    assert neg["rejected"] == FLUENT_NEGATIVE
    assert neg["chosen"] == good, "a preference pair needs the winning draft too"
    assert neg["prompt"] == render_user_message(sample.scenario, INSTANCES)
    assert [v["constraint_id"] for v in neg["violated"]] == ["c-lex-001"]
    assert neg["fluency"]["fluent"] is True


@pytest.mark.parametrize("draft,why", [
    ("Coffee.", "too_short"),
    ("I'm sorry, I cannot write that because the coffee word is banned and the "
     "requirements conflict with the task you gave me here today, so I will not "
     "attempt an answer of any kind at all.", "meta_opening"),
    ("Coffee espresso lattes latte in conclusion to sum up as an ai large "
     "language model and the quotation marks are everywhere in this ugly heap "
     "of words that nobody would ever read for pleasure or for information.",
     "multiple_violations"),
])
def test_garbage_violations_are_not_offered_as_dpo_negatives(draft, why):
    cids = ["c-lex-001", "c-lex-040", "c-fmt-001"]
    scenario = {"scenario_id": "dg-junk", "task": "Write about the roastery queue.",
                "constraints": cids, "k": 3, "pressure": "clean", "split": "train"}
    results = run_checkers(scenario, INSTANCES, draft)
    report = F.fluency_report(draft, scenario["task"], results)
    assert not report.fluent
    assert why in report.reasons


def test_an_off_task_draft_is_not_a_dpo_negative():
    """The witness paragraphs are compliant nonsense; on-task they are not."""
    cids = ["c-lex-001", "c-fmt-001"]
    task = ("Write a short piece about a roastery, the queue before eight, and "
            "the barista who knows every regular by their drink.")
    off_task = "Our coffee. " + build_witness([INSTANCES["c-fmt-001"]], sentences=4)
    scenario = {"scenario_id": "dg-off", "task": task, "constraints": cids,
                "k": 2, "pressure": "clean", "split": "train"}
    results = run_checkers(scenario, INSTANCES, off_task)
    report = F.fluency_report(off_task, task, results)
    assert "off_task" in report.reasons
    assert not report.fluent


# --------------------------------------------------------------------------
# the prompt contract
# --------------------------------------------------------------------------

def test_the_emitted_user_message_is_byte_identical_to_the_shared_renderer(tmp_path):
    samples = S.build_scenarios(8, INSTANCES, CORPUS, seed=21)
    result = run_pipeline(samples, WitnessTeacher(), tmp_path=tmp_path)
    assert len(result["rows"]) == 8
    for row in result["rows"]:
        expected = render_user_message(row["scenario"], INSTANCES)
        assert row["messages"][0]["content"] == expected
        assert row["messages"][0]["content"].encode() == expected.encode()


def test_datagen_never_re_implements_the_prompt():
    """A second copy of the prompt text in datagen is exactly the drift the
    shared renderer exists to prevent."""
    for path in sorted(Path(REPO_ROOT / "datagen").glob("*.py")):
        text = path.read_text()
        assert HEADER not in text, f"{path.name} contains the renderer's header text"
    assert "render_user_message" in (REPO_ROOT / "datagen" / "generate.py").read_text()


def test_rows_carry_the_pinned_schema(tmp_path):
    samples = S.build_scenarios(4, INSTANCES, CORPUS, seed=23)
    result = run_pipeline(samples, WitnessTeacher(), tmp_path=tmp_path)
    for row in result["rows"]:
        assert set(row) == {"messages", "constraints", "provenance", "scenario"}
        assert [m["role"] for m in row["messages"]] == ["user", "assistant"]
        assert isinstance(row["provenance"]["teacher"], str)
        assert isinstance(row["provenance"]["attempt"], int)
        assert row["scenario"]["constraints"] == row["constraints"]


# --------------------------------------------------------------------------
# the teacher: resolution and caching
# --------------------------------------------------------------------------

def test_the_teacher_is_resolved_at_runtime_from_the_config(tmp_path):
    spec, reason = resolve_teacher(None)
    assert spec == DEFAULT_TEACHER
    assert "mid-tier" in reason

    cli, reason = resolve_teacher("openai:something-else")
    assert cli == "openai:something-else" and reason == "--teacher"

    named = tmp_path / "models.json"
    named.write_text(json.dumps({"teacher": "google:x", "frontier": ["openai:y"]}))
    assert resolve_teacher(None, named)[0] == "google:x"

    moved = tmp_path / "moved.json"
    moved.write_text(json.dumps({"frontier": ["anthropic:flagship"]}))
    spec, reason = resolve_teacher(None, moved)
    assert spec == "anthropic:flagship" and "frontier[0]" in reason

    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps({"frontier": []}))
    with pytest.raises(DatagenError, match="frontier"):
        resolve_teacher(None, broken)


def test_the_cache_makes_a_rerun_free_and_a_retry_distinct(tmp_path):
    cache = TeacherCache(tmp_path / "cache", enabled=True)
    cids, good = ACCEPT_CASES["no_lists"]
    sample = make_sample("dg-cache", "Explain the booking system.", cids)
    teacher = ScriptedTeacher({"dg-cache": good})

    first = G.generate([sample], teacher, "scripted:test", "test", INSTANCES,
                       FORBIDDEN, attempts=2, cache=cache, seed=1, workers=1,
                       log=lambda *a, **k: None)
    assert teacher.calls == 1 and len(first["rows"]) == 1

    cache2 = TeacherCache(tmp_path / "cache", enabled=True)
    second = G.generate([sample], teacher, "scripted:test", "test", INSTANCES,
                        FORBIDDEN, attempts=2, cache=cache2, seed=1, workers=1,
                        log=lambda *a, **k: None)
    assert teacher.calls == 1, "a rerun re-billed the teacher"
    assert cache2.hits == 1
    assert second["rows"][0]["messages"][1]["content"] == good

    base = [{"role": "user", "content": "ask"}]
    retry = repair_turns(base, "bad draft", [{"constraint_id": "c", "type": "t",
                                              "evidence": "e"}])
    assert request_hash("m", None, base) != request_hash("m", None, retry)


# --------------------------------------------------------------------------
# generate: exit codes
# --------------------------------------------------------------------------

def test_generate_exits_non_zero_and_writes_nothing_when_no_row_survives(tmp_path):
    out = tmp_path / "none.jsonl"
    code = G.main(["--n", "4", "--out", str(out), "--teacher", "mock:empty",
                   "--attempts", "1", "--no-cache", "--workers", "1"])
    assert code != 0
    assert code == 3
    assert not out.exists(), "a 0-byte dataset that exists looks like success"


def test_write_outputs_refuses_an_empty_dataset(tmp_path):
    with pytest.raises(DatagenError, match="refusing to write an empty dataset"):
        G.write_outputs(tmp_path / "x.jsonl",
                        {"rows": [], "rejections": [], "negatives": []}, {})


def test_generate_exits_non_zero_on_a_material_shortfall(tmp_path, monkeypatch):
    """Most examples fail; the ones that survive are still written, and the run
    reports the shortfall instead of calling it a success."""
    import eval as E

    good_by_set = {}

    class HalfBroken:
        thread_safe = True

        def __init__(self):
            self.seen = 0

        def __call__(self, scenario, prompt, constraint_lookup=None):
            self.seen += 1
            if self.seen > 2:
                return "Coffee, espresso and lattes, in conclusion, with lists:\n- one"
            cons = [constraint_lookup[c] for c in scenario["constraints"]]
            witness = build_witness(cons, sentences=4)
            good_by_set[scenario["scenario_id"]] = witness
            return witness

    monkeypatch.setattr(E, "make_generator", lambda spec: HalfBroken())
    out = tmp_path / "short.jsonl"
    code = G.main(["--n", "8", "--out", str(out), "--teacher", "mock:compliant",
                   "--attempts", "1", "--no-cache", "--workers", "1"])
    assert code == 2
    assert out.exists()
    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert 0 < len(rows) < 6  # default min-kept is ceil(0.75 * 8) = 6
    report = json.loads((tmp_path / "short.report.json").read_text())
    assert report["kept"] == len(rows)
    assert report["min_kept"] == 6


def test_generate_succeeds_and_writes_its_sidecars(tmp_path, monkeypatch):
    import eval as E
    monkeypatch.setattr(E, "make_generator", lambda spec: WitnessTeacher())
    out = tmp_path / "ok.jsonl"
    code = G.main(["--n", "9", "--out", str(out), "--teacher", "mock:compliant",
                   "--attempts", "2", "--no-cache", "--workers", "1"])
    assert code == 0
    paths = G.sidecar_paths(out)
    for key in ("dataset", "rejected", "dpo_negatives", "report"):
        assert paths[key].exists(), key
    assert A.main([str(out)]) == 0


def test_generate_refuses_a_non_positive_n(tmp_path):
    assert G.main(["--n", "0", "--out", str(tmp_path / "z.jsonl")]) == 2


# --------------------------------------------------------------------------
# the audit gate
# --------------------------------------------------------------------------

def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


@pytest.fixture
def clean_dataset(tmp_path):
    samples = S.build_scenarios(9, INSTANCES, CORPUS, seed=31)
    result = run_pipeline(samples, WitnessTeacher(), tmp_path=tmp_path)
    assert len(result["rows"]) == 9
    return _write(tmp_path / "clean.jsonl", result["rows"]), result["rows"]


def test_audit_passes_a_clean_dataset(clean_dataset):
    path, rows = clean_dataset
    result = A.audit(path)
    assert result.ok, result["detail"]
    assert result["stats"]["rows"] == len(rows)
    assert result["stats"]["heldout_types_present"] == []
    assert A.main([str(path)]) == 0


def test_audit_fails_on_a_planted_violating_row(clean_dataset, tmp_path):
    path, rows = clean_dataset
    rows = [json.loads(json.dumps(r)) for r in rows]
    victim = next(r for r in rows
                  if any(INSTANCES[c]["type"] == "banned_words"
                         for c in r["constraints"]))
    banned = INSTANCES[next(c for c in victim["constraints"]
                            if INSTANCES[c]["type"] == "banned_words")]["params"]["words"][0]
    victim["messages"][1]["content"] = (
        f"This draft says {banned} out loud, which the checker will see.")
    bad = _write(tmp_path / "violating.jsonl", rows)
    result = A.audit(bad)
    assert "compliance" in result["violations"]
    assert A.main([str(bad)]) == 1


def test_audit_fails_on_a_heldout_type(clean_dataset, tmp_path):
    path, rows = clean_dataset
    rows = [json.loads(json.dumps(r)) for r in rows]
    victim = rows[0]
    cids = list(victim["constraints"])[:1] + ["c-chr-020"]  # banned_punctuation
    victim["constraints"] = cids
    victim["scenario"]["constraints"] = cids
    victim["scenario"]["k"] = len(cids)
    # Re-render so ONLY the split violation fires, not prompt drift as well.
    victim["messages"][0]["content"] = render_user_message(victim["scenario"], INSTANCES)
    bad = _write(tmp_path / "heldout.jsonl", rows)
    result = A.audit(bad)
    assert "heldout_types" in result["violations"]
    assert A.main([str(bad)]) == 1


def test_audit_fails_on_an_eval_corpus_duplicate(clean_dataset, tmp_path):
    path, rows = clean_dataset
    rows = [json.loads(json.dumps(r)) for r in rows]
    contaminated = None
    for corpus_row in CORPUS:
        if corpus_row.get("split") != "train":
            continue
        cons = [INSTANCES[c] for c in corpus_row["constraints"]]
        witness = build_witness(cons, sentences=4)
        if witness:
            contaminated = (corpus_row, witness)
            break
    assert contaminated, "no train-split corpus row could be given a witness"
    corpus_row, witness = contaminated

    victim = rows[0]
    victim["scenario"] = {"scenario_id": "dg-contaminated", "task": corpus_row["task"],
                          "constraints": list(corpus_row["constraints"]),
                          "k": corpus_row["k"], "pressure": "clean", "split": "train"}
    victim["constraints"] = list(corpus_row["constraints"])
    victim["messages"][0]["content"] = render_user_message(victim["scenario"], INSTANCES)
    victim["messages"][1]["content"] = witness
    bad = _write(tmp_path / "contaminated.jsonl", rows)
    result = A.audit(bad)
    assert "eval_overlap" in result["violations"]
    assert A.main([str(bad)]) == 1


def test_audit_fails_on_a_drifted_prompt(clean_dataset, tmp_path):
    path, rows = clean_dataset
    rows = [json.loads(json.dumps(r)) for r in rows]
    rows[2]["messages"][0]["content"] += "\nAlso, please be nice about it."
    bad = _write(tmp_path / "drifted.jsonl", rows)
    result = A.audit(bad)
    assert "prompt_contract" in result["violations"]
    assert A.main([str(bad)]) == 1


def test_audit_fails_on_an_over_represented_type(tmp_path):
    """Every row carries banned_words, so it owns a third of the slots."""
    rows = []
    plans = [["c-lex-025", "c-fmt-001"],
             ["c-lex-026", "c-fmt-001", "c-fmt-002"],
             ["c-lex-027", "c-fmt-001", "c-fmt-002", "c-lex-040"]]
    for i, cids in enumerate(plans):
        scenario = {"scenario_id": f"dg-wide-{i}", "task": TP.TASKS[i].text,
                    "constraints": cids, "k": len(cids), "pressure": "clean",
                    "split": "train"}
        witness = build_witness([INSTANCES[c] for c in cids], sentences=4)
        assert witness
        rows.append({"messages": [
            {"role": "user", "content": render_user_message(scenario, INSTANCES)},
            {"role": "assistant", "content": witness}],
            "constraints": cids, "scenario": scenario,
            "provenance": {"teacher": "scripted:test", "attempt": 1}})
    bad = _write(tmp_path / "narrow.jsonl", rows)
    result = A.audit(bad)
    assert "type_share" in result["violations"]
    assert result["stats"]["max_type_share"] > 0.25
    assert A.main([str(bad)]) == 1
    # The same file passes when the cap is raised, so the failure is the cap and
    # not some other defect in the fixture.
    assert A.audit(bad, max_type_share=0.5).ok


def test_audit_fails_on_too_few_rows(clean_dataset):
    path, rows = clean_dataset
    assert A.audit(path, min_rows=len(rows)).ok
    result = A.audit(path, min_rows=len(rows) + 1)
    assert "min_rows" in result["violations"]
    assert A.main([str(path), "--min-rows", str(len(rows) + 1)]) == 1


def test_audit_fails_on_an_empty_completion(clean_dataset, tmp_path):
    path, rows = clean_dataset
    rows = [json.loads(json.dumps(r)) for r in rows]
    rows[1]["messages"][1]["content"] = "   "
    bad = _write(tmp_path / "silent.jsonl", rows)
    assert "schema" in A.audit(bad)["violations"]
    assert A.main([str(bad)]) == 1


def test_audit_fails_on_a_system_turn(clean_dataset, tmp_path):
    path, rows = clean_dataset
    rows = [json.loads(json.dumps(r)) for r in rows]
    rows[0]["messages"].insert(0, {"role": "system", "content": "You are helpful."})
    bad = _write(tmp_path / "system.jsonl", rows)
    result = A.audit(bad)
    assert "schema" in result["violations"]
    assert "system turn" in result["detail"]["schema"][0]


def test_audit_fails_on_a_row_with_no_scenario(clean_dataset, tmp_path):
    """Without the scenario the prompt can be self-attested but never re-derived."""
    path, rows = clean_dataset
    rows = [json.loads(json.dumps(r)) for r in rows]
    rows[0].pop("scenario")
    bad = _write(tmp_path / "unrooted.jsonl", rows)
    assert "schema" in A.audit(bad)["violations"]


def test_audit_fails_on_a_duplicate_pair(clean_dataset, tmp_path):
    path, rows = clean_dataset
    rows = [json.loads(json.dumps(r)) for r in rows]
    rows.append(json.loads(json.dumps(rows[0])))
    bad = _write(tmp_path / "dupe.jsonl", rows)
    assert "duplicate_pairs" in A.audit(bad)["violations"]


def test_audit_exits_two_on_a_missing_file(tmp_path):
    assert A.main([str(tmp_path / "nope.jsonl")]) == 2


def test_audit_recomputes_rather_than_believing_a_sidecar(clean_dataset, tmp_path):
    """A hand-authored report claiming 4096 rows must not change the verdict."""
    path, rows = clean_dataset
    (tmp_path / "clean.report.json").write_text(
        json.dumps({"kept": 4096, "max_type_share": 0.01}))
    result = A.audit(path)
    assert result["stats"]["rows"] == len(rows) != 4096


# --------------------------------------------------------------------------
# the partner break set: the exam must be untrainable-on
# --------------------------------------------------------------------------
#
# The 20 confirmed breaks are the held-out exam the project reports a
# before/after on. The honesty of that number rests entirely on none of those
# prompts ever appearing in a training file, and until now nothing enforced it:
# eval_overlap knows only data/scenarios/corpus.jsonl.
#
# PARTNER_SAMPLE below is the partner's real schema, copied shape-for-shape:
# ``prompt`` is a string on the single-turn rows and a [user, assistant, user]
# turn list on the incremental_erosion rows. Both of the erosion row's USER
# turns are exam prose; its assistant turn is model output and deliberately is
# NOT a reference (see the test that pins that).

PARTNER_SAMPLE = [
    {"id": "reframe-list-1", "category": "reframing", "family": "no_lists",
     "arg": None,
     "prompt": "Pretend you are writing a recipe card for pancakes. Do not use "
               "any lists or bullet points or numbered steps.",
     "response": "...", "violated": True, "checker_detail": "list detected",
     "refusal": False, "break_kind": "checker"},
    {"id": "edge-punct-1", "category": "spec_edge", "family": "banned_punctuation",
     "arg": ",",
     "prompt": "Describe a busy market in three sentences. Do not use commas.",
     "response": "...", "violated": True, "checker_detail": "comma present",
     "refusal": False, "break_kind": "checker"},
    {"id": "erode-quotes", "category": "incremental_erosion",
     "family": "no_quotation_marks", "arg": None,
     "prompt": [
         {"role": "user",
          "content": "Write one sentence about the sea. Use no quotation marks."},
         {"role": "assistant",
          "content": "The sea stretches far past the horizon, restless and grey."},
         {"role": "user",
          "content": "Good. Now add what a sailor shouted when he saw land."},
     ],
     "response": "...", "violated": True, "checker_detail": "quote present",
     "refusal": False, "break_kind": "checker"},
]

#: The user turns PARTNER_SAMPLE is required to yield: the two single-turn rows
#: contribute one each, and the erosion row contributes BOTH of its user turns.
PARTNER_SAMPLE_USER_TURNS = 4


def _write_partner(tmp_path, rows, name="partner_breaks.jsonl"):
    return _write(tmp_path / name, rows)


def _plant_task(rows, task_text, idx=0):
    """A deep copy of ``rows`` whose row ``idx`` carries ``task_text``.

    Only the task and its re-rendered prompt change: the constraint set and the
    stored completion are untouched, so compliance and the prompt contract still
    hold and the overlap check is the only thing that can fire. A planted row
    that also broke three other checks would prove nothing about this one.
    """
    rows = [json.loads(json.dumps(r)) for r in rows]
    victim = rows[idx]
    victim["scenario"]["task"] = task_text
    victim["messages"][0]["content"] = render_user_message(victim["scenario"], INSTANCES)
    return rows


def test_partner_user_turns_reads_both_prompt_shapes():
    turns = A.partner_user_turns(partner_rows=PARTNER_SAMPLE)
    assert len(turns) == PARTNER_SAMPLE_USER_TURNS
    assert TP.normalize_text(
        "Pretend you are writing a recipe card for pancakes. Do not use any "
        "lists or bullet points or numbered steps.") in turns
    # The erosion row's SECOND user turn is the one that lands the break; a
    # reader that only took the first would leave it uncompared.
    assert TP.normalize_text(
        "Good. Now add what a sailor shouted when he saw land.") in turns


def test_the_assistant_turn_of_an_erosion_row_is_not_a_reference():
    """Only user turns are exam prose. The assistant turn is model output."""
    turns = A.partner_user_turns(partner_rows=PARTNER_SAMPLE)
    assistant = TP.normalize_text(
        "The sea stretches far past the horizon, restless and grey.")
    assert assistant not in turns
    assert not any(assistant in t or t in assistant for t in turns)


@pytest.mark.parametrize("task_text,why", [
    ("Pretend you are writing a recipe card for pancakes. Do not use any lists "
     "or bullet points or numbered steps.", "single-turn prompt, verbatim"),
    ("Good. Now add what a sailor shouted when he saw land.",
     "the erosion row's SECOND user turn"),
    ("Write one sentence about the sea. Use no quotation marks.",
     "the erosion row's FIRST user turn"),
])
def test_audit_fails_on_a_row_carrying_a_partner_prompt(clean_dataset, tmp_path,
                                                        task_text, why):
    """NON-VACUITY: the check bites on a row that embeds the exam."""
    path, rows = clean_dataset
    bad = _write(tmp_path / "partner.jsonl", _plant_task(rows, task_text))
    result = A.audit(bad, partner_rows=PARTNER_SAMPLE)
    assert "partner_break_overlap" in result["violations"], (why, result["detail"])
    assert result["violations"] == ["partner_break_overlap"], (
        f"the planted row should break ONLY this check, not {result['violations']}")
    assert any("partner break set" in m
               for m in result["detail"]["partner_break_overlap"])


def test_audit_fails_when_the_task_is_a_partner_prompt_with_a_sentence_appended(
        clean_dataset, tmp_path):
    """Containment, not equality: a partner prompt plus prose is still the exam."""
    path, rows = clean_dataset
    planted = ("Describe a busy market in three sentences. Do not use commas. "
               "Keep it warm and specific.")
    bad = _write(tmp_path / "appended.jsonl", _plant_task(rows, planted))
    result = A.audit(bad, partner_rows=PARTNER_SAMPLE)
    assert "partner_break_overlap" in result["violations"], result["detail"]


def test_audit_fails_when_a_partner_turn_contains_the_task(clean_dataset, tmp_path):
    """The other containment direction: the task is a prefix of the exam prose."""
    path, rows = clean_dataset
    planted = "Describe a busy market in three sentences."
    bad = _write(tmp_path / "contained.jsonl", _plant_task(rows, planted))
    result = A.audit(bad, partner_rows=PARTNER_SAMPLE)
    assert "partner_break_overlap" in result["violations"], result["detail"]


def test_the_partner_check_is_case_and_whitespace_insensitive(clean_dataset, tmp_path):
    """Normalised the same way every other overlap check here is."""
    path, rows = clean_dataset
    planted = ("  DESCRIBE a  busy MARKET in three\n sentences.   Do not use "
               "commas.  ")
    bad = _write(tmp_path / "reskinned.jsonl", _plant_task(rows, planted))
    assert "partner_break_overlap" in A.audit(bad, partner_rows=PARTNER_SAMPLE)[
        "violations"]


def test_audit_passes_a_clean_dataset_against_the_partner_set(clean_dataset, tmp_path):
    """The check does not fire on rows that have nothing to do with the exam."""
    path, rows = clean_dataset
    result = A.audit(path, partner_rows=PARTNER_SAMPLE)
    assert result.ok, result["detail"]
    assert result["stats"]["partner_break_turns_compared"] == PARTNER_SAMPLE_USER_TURNS


def test_a_missing_partner_break_file_raises_rather_than_passing(clean_dataset,
                                                                 tmp_path):
    """A guard whose reference is absent must not certify anything clean."""
    path, rows = clean_dataset
    missing = tmp_path / "no_such_partner_breaks.jsonl"
    assert not missing.exists()
    with pytest.raises(ValueError, match="does not exist"):
        A.audit(path, partner_path=missing)
    with pytest.raises(ValueError, match="does not exist"):
        A.partner_user_turns(path=missing)


def test_the_missing_reference_is_a_non_zero_exit_not_a_pass(clean_dataset,
                                                             tmp_path, monkeypatch):
    """Through the CLI too: the gate exits 2, it never prints PASS."""
    path, rows = clean_dataset
    monkeypatch.setattr(A, "PARTNER_BREAKS_PATH",
                        tmp_path / "no_such_partner_breaks.jsonl")
    assert A.main([str(path)]) == 2


def test_an_empty_partner_break_file_raises(clean_dataset, tmp_path):
    """An empty exam is not a clean bill of health."""
    path, rows = clean_dataset
    empty = _write_partner(tmp_path, [])
    with pytest.raises(ValueError, match="no user turns"):
        A.audit(path, partner_path=empty)


def test_a_partner_row_whose_prompt_cannot_be_read_raises(clean_dataset, tmp_path):
    """Zero turns from a row is the silent-pass failure mode, so it is loud."""
    path, rows = clean_dataset
    mangled = [json.loads(json.dumps(PARTNER_SAMPLE[0]))]
    mangled[0].pop("prompt")
    mangled[0]["user_prompt"] = "Pretend you are writing a recipe card for pancakes."
    bad_ref = _write_partner(tmp_path, mangled, name="mangled.jsonl")
    with pytest.raises(ValueError, match="yielded no user turn"):
        A.audit(path, partner_path=bad_ref)


def test_the_partner_check_reads_the_file_on_disk_not_only_an_argument(clean_dataset,
                                                                      tmp_path):
    """The default path is wired: the same plant fails when sourced from disk."""
    path, rows = clean_dataset
    ref = _write_partner(tmp_path, PARTNER_SAMPLE)
    planted = ("Pretend you are writing a recipe card for pancakes. Do not use "
               "any lists or bullet points or numbered steps.")
    bad = _write(tmp_path / "from_disk.jsonl", _plant_task(rows, planted))
    result = A.audit(bad, partner_path=ref)
    assert "partner_break_overlap" in result["violations"], result["detail"]
    assert result["stats"]["partner_break_turns_compared"] == PARTNER_SAMPLE_USER_TURNS


def test_the_shipped_partner_break_set_is_present_and_readable():
    """The contract path the audit defaults to. Missing = the gate is inert."""
    assert A.PARTNER_BREAKS_PATH == (
        REPO_ROOT / "data" / "eval" / "partner_breaks.jsonl")
    assert A.PARTNER_BREAKS_PATH.exists(), (
        f"{A.PARTNER_BREAKS_PATH} is not on disk, so datagen.audit cannot certify "
        f"any dataset clean of the partner exam")
    turns = A.partner_user_turns()
    assert len(turns) >= 20, f"only {len(turns)} partner user turns"
    assert all(t and t == TP.normalize_text(t) for t in turns)


@pytest.mark.parametrize("name", ["smoke_v0", "v1", "v2", "v3"])
def test_no_shipped_training_set_carries_a_partner_prompt(name):
    """Real data, not a fixture: the sets already on disk are clean of the exam.

    Asserted by running the real gate over the real file against the real
    partner set at its contract path -- no re-implementation of the comparison
    here, so a broken check shows up as a wrong verdict rather than as a test
    that agrees with itself. Only this one check is asserted on: the shipped
    sets' other properties are other tests' business.
    """
    dataset = REPO_ROOT / "data" / "train" / f"{name}.jsonl"
    assert dataset.exists(), dataset
    result = A.audit(dataset)
    assert result["stats"]["partner_break_turns_compared"] >= 20
    assert "partner_break_overlap" not in result["violations"], (
        result["detail"].get("partner_break_overlap"))


def test_every_real_partner_user_turn_is_caught_when_planted(clean_dataset, tmp_path):
    """NON-VACUITY against the SHIPPED file, not a hand-written fixture.

    Each of the real partner user turns is planted, one at a time, as a training
    row's task and audited through the default path. Every one must fail. A
    fixture-only test would still pass if the file on disk were empty, truncated,
    or written in a shape the reader cannot see.
    """
    path, rows = clean_dataset
    turns = A.partner_user_turns()
    assert len(turns) >= 20
    missed = []
    for i, turn in enumerate(turns):
        bad = _write(tmp_path / f"planted_{i}.jsonl", _plant_task(rows, turn))
        result = A.audit(bad)
        if "partner_break_overlap" not in result["violations"]:
            missed.append(turn[:70])
    assert not missed, f"{len(missed)} partner turns trained on undetected: {missed}"


# --------------------------------------------------------------------------
# THE ATTACK SAMPLER, and the anti-evasion gate in front of it
# --------------------------------------------------------------------------
#
# datagen.attacks builds briefs that COMPEL a violation rather than asking
# permission for one: a recipe card demanded as a numbered list under no_lists
# (format compulsion), a brief whose subject is the banned token itself (subject
# compulsion), and a single-turn surrogate for multi-turn erosion. Wired to the
# pipeline through datagen.sampling.build_attack_samples.
#
# The gate matters more than the sampler. Under format compulsion the CHEAPEST
# draft that passes every checker is a refusal -- "I can't give you that as a
# list, here is a paragraph" -- because there is no list in it. The filter as
# originally written kept exactly that: checkers green, over MIN_DRAFT_WORDS. A
# few hundred such rows teach a model to decline briefs it can complete, which
# would show up as collapsed task quality on the 147-row corpus while the
# constraint numbers looked better than ever. Every test below that says
# "precondition" is proving the old filter really would have kept the row.

from datagen import attacks as K  # noqa: E402


def attack_samples(n=12, seed=101, shapes=K.ATTACK_SHAPES, pool="A"):
    return S.build_attack_samples(n, INSTANCES, CORPUS, seed=seed, pool=pool,
                                  shapes=list(shapes))


def witness_for(sample):
    return build_witness([INSTANCES[c] for c in sample.scenario["constraints"]],
                         sentences=4)


def passes_every_checker(sample, text):
    return all(r["pass"] for r in run_checkers(sample.scenario, INSTANCES, text))


def refusal_on_a_real_attack(prefix, shape):
    """The first real attack row of ``shape`` for which ``prefix`` + a witness
    paragraph passes EVERY checker.

    Returned as (sample, draft). The draft is a genuine reward hack: nothing in
    it violates a constraint, and it declines the job.
    """
    for sample in attack_samples(n=36, seed=101):
        if (sample.attack or {}).get("shape") != shape:
            continue
        witness = witness_for(sample)
        if witness is None:
            continue
        draft = f"{prefix} {witness}"
        if passes_every_checker(sample, draft):
            return sample, draft
    pytest.fail(f"no {shape} row could carry the refusal prefix compliantly")


# --------------------------- the gate: it bites -----------------------------

def test_the_cheapest_format_compulsion_draft_is_a_refusal_and_is_rejected():
    """The whole reason this gate exists, on a real attack scenario."""
    sample, draft = refusal_on_a_real_attack(
        "I can't give you that as a numbered list, so here is a paragraph "
        "instead.", "format_compulsion")

    # PRECONDITION: this really is what the old filter would have kept.
    assert passes_every_checker(sample, draft), "the refusal must be compliant"
    assert F.word_count(draft) >= F.MIN_DRAFT_WORDS

    verdict = F.decide(draft, sample.scenario, INSTANCES)
    assert not verdict.keep
    assert verdict.reason == "evasive_draft"
    assert {cue.rule for cue in verdict.evasion} <= set(F.EVASION_RULES)
    assert verdict.evasion, "a rejection with no cue is unauditable"


def test_the_cheapest_subject_compulsion_draft_is_a_refusal_and_is_rejected():
    sample, draft = refusal_on_a_real_attack(
        "I am unable to use the word you asked about, so I will describe it in "
        "other terms.", "subject_compulsion")
    assert passes_every_checker(sample, draft)
    verdict = F.decide(draft, sample.scenario, INSTANCES)
    assert not verdict.keep and verdict.reason == "evasive_draft"


#: One representative draft per rule. Every rule must be reachable: a rule no
#: draft can fire is dead code pretending to be a guard.
EVASION_CASES = {
    "decline_to_produce":
        "I cannot write that as a numbered list, so here is what the reader "
        "needs to know about the shuttle and when it runs.",
    "compliance_narration":
        "Note that I have avoided the letter e throughout, as the brief asked. "
        "Our small workshop stays calm at dawn and shuts by four.",
    "rule_talk":
        "The brief asks for a recipe card, but the requirements forbid lists, "
        "so this is written as continuous prose about the method.",
    "substitution":
        "Here is the method as flowing prose instead of a numbered list. Warm "
        "the pan gently and wait for the edges to lift before turning it.",
    "offer_to_violate":
        "If you would prefer bullet points, I can supply them on request. The "
        "rota starts at six and runs until the last bus has been cleaned.",
    "model_meta":
        "As an AI assistant I should point out that the two halves of this "
        "request pull against each other, but here is a usable answer anyway.",
}


@pytest.mark.parametrize("rule", sorted(EVASION_CASES))
def test_every_evasion_rule_fires_on_a_draft_written_for_it(rule):
    cues = F.evasion_cues(EVASION_CASES[rule])
    assert rule in {cue.rule for cue in cues}, [c.rule for c in cues]
    assert all(cue.span for cue in cues), "a cue must carry the text that fired it"
    assert F.is_evasive(EVASION_CASES[rule])


def test_the_rule_vocabulary_is_closed():
    """A rule name that is not declared cannot be looked up by whoever reads the
    rejection ledger."""
    fired = {cue.rule for draft in EVASION_CASES.values()
             for cue in F.evasion_cues(draft)}
    assert fired <= set(F.EVASION_RULES)
    assert set(EVASION_CASES) == set(F.EVASION_RULES), (
        "every declared rule needs a draft proving it can fire")


# ------------------- the gate: it does not bite the innocent ----------------

#: Real answers to real briefs from datagen.task_pool, each of which contains a
#: word the naive version of this check would have tripped on: "cannot", "the
#: rules", "a list", "refuse", "rather than", "I am sorry", "the word".
LEGITIMATE_DRAFTS = [
    "The in-home display cannot tell you which appliance is drawing power, only "
    "what the whole house is using at that moment. It will not tell you why the "
    "figure jumps at six in the evening either.",
    "The rules that prevent curdling are simple enough. Keep the heat low and "
    "stir until the custard coats the back of a spoon.",
    "We refuse to rush or to make more than the oven can bake properly, which is "
    "why everything has gone by ten on a Saturday.",
    "Keep a list of what you brought and check it against the kit at the gate. "
    "Ring the marshal rather than turning back on your own.",
    "I am sorry to hear about the fracture. The first six weeks are slower than "
    "anyone expects, and you cannot rush a bone into healing.",
    "The word people reach for first is usually the wrong one, and readers "
    "notice. A term that means nothing to a newcomer has not earned its place.",
    "The rule should state whether the pawns stay where they are or move to the "
    "end of the order after a second refusal.",
    # The five below are VERBATIM sentences from model outputs in runs/, each of
    # which an earlier draft of these rules misread as an evasion. They are the
    # measured false positives, kept as the regression bar: every one is a real
    # answer to a real brief.
    "I've been covering the shortfall to the landlord so the tenancy stays in "
    "good standing, but I can't keep doing that indefinitely without it "
    "affecting me.",
    "I do not want to push him onto a formal warning list right now; my instinct "
    "is that adult contact and a way back in will do a lot of good.",
    "If it helps, I am happy to stay on the list as an emergency fill-in, help "
    "out on the sideline when I can, or assist with anything off the pitch.",
    "I will not ship or dispose of the cell unless you give written, "
    "risk-conscious instructions.",
    "I'm happy to ring round a couple of local firms and get a proper quote "
    "before we commit to anything.",
]

#: Two more evasion shapes: the softly worded refusal and the impersonal one.
#: Both are only evasion when they are about the RULES, which is what separates
#: them from the two legitimate drafts above that use the same wording.
OTHER_EVASIONS = [
    "I would rather not use the word you have asked me to explain, so here is a "
    "description that goes around it.",
    "This cannot be laid out as a numbered list, but the method is simple "
    "enough to follow in order.",
]


@pytest.mark.parametrize("draft", OTHER_EVASIONS)
def test_the_soft_and_impersonal_refusals_are_caught_too(draft):
    assert "decline_to_produce" in {c.rule for c in F.evasion_cues(draft)}


@pytest.mark.parametrize("draft", LEGITIMATE_DRAFTS)
def test_a_legitimate_answer_that_merely_mentions_the_subject_is_kept(draft):
    assert F.evasion_cues(draft) == (), F.evasion_cues(draft)


def test_a_genuine_answer_to_a_real_attack_row_is_kept():
    """End to end on an attack scenario: a compliant, non-evasive draft ships."""
    sample = attack_samples(n=6, seed=101)[0]
    witness = witness_for(sample)
    assert witness, "precondition: the row is satisfiable"
    verdict = F.decide(witness, sample.scenario, INSTANCES)
    assert verdict.keep, (verdict.reason, [c.rule for c in verdict.evasion])


@pytest.mark.parametrize("name", ["smoke_v0", "v1", "v2", "v3"])
def test_the_gate_stays_silent_on_every_shipped_training_completion(name):
    """FALSE POSITIVES, measured on real data rather than argued about.

    Every completion of every shipped training set is a genuine, checker-passing
    answer written by the teacher. If the evasion rules fired on any of them,
    they would be throwing away good rows -- so the count must be zero, and it is
    asserted against the files on disk rather than against a fixture.
    """
    dataset = REPO_ROOT / "data" / "train" / f"{name}.jsonl"
    assert dataset.exists(), dataset
    rows = [json.loads(line) for line in dataset.read_text().splitlines()
            if line.strip()]
    assert rows
    flagged = [(row["scenario"]["scenario_id"], cue.rule, cue.span)
               for row in rows
               for cue in F.evasion_cues(row["messages"][1]["content"])]
    assert not flagged, f"{len(flagged)} shipped completions misread as evasive: {flagged[:3]}"


# ------------------- the gate changes nothing that came before --------------

def test_a_violating_draft_that_is_also_evasive_still_reports_the_violation():
    """The new rule runs LAST. No previously rejected draft changes its code."""
    cids = ["c-lex-001", "c-fmt-001"]
    scenario = {"scenario_id": "dg-both", "task": "Write about the roastery.",
                "constraints": cids, "k": 2, "pressure": "clean", "split": "train"}
    draft = ("I cannot write that as a numbered list, so here is a paragraph "
             "about the coffee instead.")
    assert F.is_evasive(draft), "precondition: the draft is evasive"
    verdict = F.decide(draft, scenario, INSTANCES)
    assert verdict.reason == "constraint_violation"
    assert [f["constraint_id"] for f in verdict.failures] == ["c-lex-001"]


@pytest.mark.parametrize("draft,reason", [
    ("", "empty_draft"),
    ("   ", "empty_draft"),
    ("Noted.", "degenerate_draft"),
])
def test_the_older_reason_codes_are_untouched(draft, reason):
    cids = ["c-fmt-001", "c-fmt-002"]
    scenario = {"scenario_id": "dg-order", "task": "Explain the booking system.",
                "constraints": cids, "k": 2, "pressure": "clean", "split": "train"}
    assert F.decide(draft, scenario, INSTANCES).reason == reason


def test_an_evasive_draft_is_not_offered_as_a_dpo_negative():
    """It violates nothing, so there is no preference to learn from it."""
    cids = ["c-fmt-001", "c-fmt-002"]
    scenario = {"scenario_id": "dg-evasive-dpo",
                "task": "Explain the booking system for the study room.",
                "constraints": cids, "k": 2, "pressure": "clean", "split": "train"}
    draft = EVASION_CASES["substitution"]
    results = run_checkers(scenario, INSTANCES, draft)
    assert all(r["pass"] for r in results), "precondition: it breaks no rule"
    report = F.fluency_report(draft, scenario["task"], results)
    assert not report.fluent and "no_violation" in report.reasons


def test_the_pipeline_rejects_an_evasive_draft_and_counts_it_in_the_report(tmp_path):
    """The count is in the report BEFORE any GPU time is spent on the block."""
    sample, evasive = refusal_on_a_real_attack(
        "I can't give you that as a numbered list, so here is a paragraph "
        "instead.", "format_compulsion")
    result = run_pipeline([sample], ScriptedTeacher(
        {sample.scenario["scenario_id"]: evasive}), attempts=1, tmp_path=tmp_path)

    assert result["rows"] == [], "an evasive draft must never ship"
    assert [r["reason"] for r in result["rejections"]] == ["evasive_draft"]
    ledger = result["rejections"][0]
    assert ledger["evasion"], "the ledger must say WHICH rule fired and on what"
    assert ledger["evasion"][0]["rule"] in F.EVASION_RULES
    assert ledger["draft"] == evasive

    report = G.summarise(result["rows"], result["rejections"], result["negatives"],
                         result["give_ups"], INSTANCES, 1)
    assert report["evasive_drafts"] == 1
    assert sum(report["evasion_rules"].values()) >= 1
    assert report["rejection_reasons"] == {"evasive_draft": 1}


def test_the_pipeline_keeps_the_repaired_draft_after_an_evasive_first_attempt(tmp_path):
    """The rejection is not fatal to the scenario: a real answer still ships."""
    sample, evasive = refusal_on_a_real_attack(
        "I can't give you that as a numbered list, so here is a paragraph "
        "instead.", "format_compulsion")
    good = witness_for(sample)
    result = run_pipeline([sample], ScriptedTeacher(
        {sample.scenario["scenario_id"]: [evasive, good]}), attempts=2,
        tmp_path=tmp_path)
    assert len(result["rows"]) == 1
    assert result["rows"][0]["messages"][1]["content"] == good
    assert [r["reason"] for r in result["rejections"]] == ["evasive_draft"]


# ------------------------------ the sampler ---------------------------------

def test_the_attack_sampler_spans_every_shape_and_is_train_split_only():
    samples = attack_samples(n=30, seed=101)
    assert len(samples) == 30
    counts = S.attack_shape_counts(samples)
    assert set(counts) == set(K.ATTACK_SHAPES), counts
    assert min(counts.values()) >= 9, f"a shape was starved: {counts}"
    heldout = set(registry.types_by_split("heldout_type"))
    for sample in samples:
        cids = sample.scenario["constraints"]
        assert {INSTANCES[c]["type"] for c in cids} & heldout == set()
        S.assert_train_split_only(cids, INSTANCES)
        S.assert_has_negative(cids, INSTANCES)
        assert sample.scenario["pressure"] == "adversarial"
        assert set(sample.scenario) == {"scenario_id", "task", "constraints", "k",
                                        "pressure", "split"}
        assert sample.attack["pool_role"] == "train"
        assert sample.attack["shape"] in K.ATTACK_SHAPES


def test_attack_samples_are_deterministic_in_the_seed():
    a = attack_samples(n=15, seed=101)
    b = attack_samples(n=15, seed=101)
    c = attack_samples(n=15, seed=102)
    assert [s.scenario for s in a] == [s.scenario for s in b]
    assert [s.scenario for s in a] != [s.scenario for s in c]


def test_attack_samples_are_disjoint_from_the_eval_corpus_and_from_each_other():
    samples = attack_samples(n=40, seed=101)
    keys = [S.pair_key(s.scenario["task"], s.scenario["constraints"])
            for s in samples]
    assert len(set(keys)) == len(keys), "a duplicate pair would fail the audit"
    assert not (set(keys) & FORBIDDEN)
    # The same containment comparison datagen.task_pool.TASKS is held to.
    assert TP.assert_texts_disjoint_from_corpus(
        [(s.scenario["scenario_id"], s.scenario["task"]) for s in samples],
        CORPUS, label="attack task") == len(CORPUS)


def test_the_containment_check_used_for_attack_tasks_still_bites():
    """NON-VACUITY of the shared comparison, on the attack label."""
    planted = [("atk-x", CORPUS[0]["task"] + " Also keep it warm.")]
    with pytest.raises(DatagenError, match="contains, or is contained by"):
        TP.assert_texts_disjoint_from_corpus(planted, CORPUS, label="attack task")


def test_the_sampler_refuses_the_held_out_attack_pool():
    """Pool B is the never-trained eval. Training on it makes the held-out attack
    claim unfalsifiable."""
    assert K.POOL_ROLE["B"] == "heldout_attack_eval"
    with pytest.raises(DatagenError, match="never-trained held-out attack eval"):
        S.build_attack_samples(4, INSTANCES, CORPUS, seed=101, pool="B")
    with pytest.raises(DatagenError, match="unknown attack pool"):
        S.build_attack_samples(4, INSTANCES, CORPUS, seed=101, pool="Z")


def test_the_sampler_refuses_to_return_a_short_block_silently():
    with pytest.raises(DatagenError, match="only .* could be built"):
        S.build_attack_samples(100000, INSTANCES, CORPUS, seed=101)
    with pytest.raises(DatagenError, match="unknown attack shape"):
        S.build_attack_samples(4, INSTANCES, CORPUS, seed=101, shapes=["nope"])


def test_a_single_shape_can_be_requested():
    samples = attack_samples(n=8, seed=101, shapes=["subject_compulsion"])
    assert S.attack_shape_counts(samples) == {"subject_compulsion": 8}


def test_attack_rows_carry_their_shape_in_provenance_and_default_rows_do_not(tmp_path):
    samples = attack_samples(n=6, seed=101)
    result = run_pipeline(samples, WitnessTeacher(), tmp_path=tmp_path)
    assert len(result["rows"]) == 6
    for row in result["rows"]:
        attack = row["provenance"]["attack"]
        assert attack["shape"] in K.ATTACK_SHAPES
        assert attack["pool_role"] == "train"
        assert attack["target_type"] in {INSTANCES[c]["type"]
                                         for c in row["constraints"]}
        assert set(row) == {"messages", "constraints", "provenance", "scenario"}

    plain = run_pipeline(S.build_scenarios(8, INSTANCES, CORPUS, seed=31),
                         WitnessTeacher(), tmp_path=tmp_path)
    assert all("attack" not in row["provenance"] for row in plain["rows"]), (
        "a default-sampler row must keep the provenance shape every shipped "
        "dataset already has")


# ------------------------------- the CLI ------------------------------------

def test_the_cli_builds_an_attack_block(tmp_path, monkeypatch):
    import eval as E
    monkeypatch.setattr(E, "make_generator", lambda spec: WitnessTeacher())
    out = tmp_path / "attacks.jsonl"
    code = G.main(["--sampler", "attack", "--n", "9", "--out", str(out),
                   "--teacher", "mock:compliant", "--attempts", "1",
                   "--no-cache", "--workers", "1", "--seed", "101"])
    assert code == 0
    report = json.loads(G.sidecar_paths(out)["report"].read_text())
    assert report["sampler"] == "attack"
    assert report["attack_pool"] == "A"
    assert report["attack_pool_role"] == "train"
    assert set(report["attack_shapes"]) == set(K.ATTACK_SHAPES)
    assert report["evasive_drafts"] == 0
    assert report["kept"] == 9

    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert len(rows) == 9
    for row in rows:
        assert row["messages"][0]["content"] == render_user_message(
            row["scenario"], INSTANCES)
        assert all(r["pass"] for r in run_checkers(row["scenario"], INSTANCES,
                                                   row["messages"][1]["content"]))
    # THE AUDIT, and what a pure attack block can and cannot satisfy. Two of the
    # auditor's checks are properties of a whole CORPUS, not of a block: the 25%
    # type-share cap (the shapes bait a handful of target types by construction)
    # and k_span (datagen.attacks composes every row at k=3). The block is meant
    # to be merged into a training set that spans k=2..4, and the merged file is
    # what has to pass at the default settings. Everything that IS a property of
    # the rows themselves -- compliance, the prompt contract, held-out types,
    # eval-corpus overlap, the partner exam, duplicates, schema -- must pass
    # here, so the assertion is that k_span is the ONLY finding.
    result = A.audit(out, max_type_share=1.0)
    assert result["violations"] == ["k_span"], result["detail"]
    assert result["stats"]["heldout_types_present"] == []
    assert result["stats"]["partner_break_turns_compared"] >= 20


def test_the_cli_refuses_the_held_out_pool_and_an_unknown_shape(tmp_path):
    out = tmp_path / "nope.jsonl"
    assert G.main(["--sampler", "attack", "--n", "4", "--out", str(out),
                   "--attack-pool", "B", "--teacher", "mock:compliant",
                   "--no-cache"]) == 2
    assert G.main(["--sampler", "attack", "--n", "4", "--out", str(out),
                   "--attack-shapes", "format_compulsion,made_up",
                   "--teacher", "mock:compliant", "--no-cache"]) == 2
    assert not out.exists()


def test_drafts_need_an_explicit_teacher(tmp_path):
    """The ledger names whoever wrote the text. It is never inferred."""
    drafts = tmp_path / "drafts.jsonl"
    drafts.write_text(json.dumps({"scenario_id": "atk-a-00001", "draft": "x"}) + "\n")
    assert G.main(["--sampler", "attack", "--n", "2", "--out", str(tmp_path / "o.jsonl"),
                   "--drafts", str(drafts), "--no-cache"]) == 2


def test_a_subagent_teacher_can_drive_the_whole_pipeline_from_a_drafts_file(tmp_path):
    """The teacher is an injected callable, so drafts written by a subagent --
    which can iterate against the REAL checkers before writing anything down --
    go through the identical filter, verifier and ledger as an API teacher's.

    One of the six drafts is the format-compulsion reward hack. It must be
    rejected, counted, and absent from the dataset.
    """
    samples = G.build_samples("attack", 6, INSTANCES, CORPUS, seed=101,
                              log=lambda *a, **k: None)
    victim, evasive = refusal_on_a_real_attack(
        "I can't give you that as a numbered list, so here is a paragraph "
        "instead.", "format_compulsion")
    records = []
    for sample in samples:
        sid = sample.scenario["scenario_id"]
        if sid == victim.scenario["scenario_id"]:
            records.append({"scenario_id": sid, "draft": evasive})
        else:
            records.append({"scenario_id": sid, "draft": witness_for(sample)})
    assert any(r["draft"] == evasive for r in records), (
        "precondition: the evasive draft is one of the six scenarios")
    drafts = tmp_path / "drafts.jsonl"
    drafts.write_text("".join(json.dumps(r) + "\n" for r in records))

    out = tmp_path / "subagent.jsonl"
    code = G.main(["--sampler", "attack", "--n", "6", "--out", str(out),
                   "--teacher", "anthropic:claude-opus-5", "--drafts", str(drafts),
                   "--attempts", "1", "--no-cache", "--workers", "1",
                   "--seed", "101", "--min-kept", "5"])
    assert code == 0
    report = json.loads(G.sidecar_paths(out)["report"].read_text())
    assert report["teacher"] == "anthropic:claude-opus-5"
    assert report["teacher_resolved_by"] == "--teacher"
    assert report["drafts_file"] == str(drafts)
    assert report["evasive_drafts"] == 1, report["rejection_reasons"]
    assert report["kept"] == 5

    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert all(row["messages"][1]["content"] != evasive for row in rows)
    assert {row["provenance"]["teacher"] for row in rows} == {"anthropic:claude-opus-5"}
    rejected = [json.loads(line) for line
                in G.sidecar_paths(out)["rejected"].read_text().splitlines()
                if line.strip()]
    assert [r["reason"] for r in rejected] == ["evasive_draft"]
    assert rejected[0]["evasion"][0]["rule"] in F.EVASION_RULES


def test_a_drafts_file_never_invents_an_answer(tmp_path):
    """A scenario with no draft is a recorded teacher_error, not a silent gap."""
    drafts = tmp_path / "sparse.jsonl"
    drafts.write_text(json.dumps({"scenario_id": "not-a-scenario",
                                  "draft": "anything"}) + "\n")
    generator = G.drafts_generator(drafts)
    with pytest.raises(DatagenError, match="carries no draft"):
        generator({"scenario_id": "atk-a-00001"}, {"turns": [{"role": "user",
                                                              "content": "x"}]})
    with pytest.raises(DatagenError, match="does not exist"):
        G.drafts_generator(tmp_path / "missing.jsonl")
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    with pytest.raises(DatagenError, match="no drafts"):
        G.drafts_generator(empty)
