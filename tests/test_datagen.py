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
    assert set(F.REASON_CODES) == {"teacher_error", "empty_draft",
                                   "degenerate_draft", "constraint_violation"}


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
