"""P3 datagen tests. Network-free: every teacher reply here is canned.

Covers the binding "Datagen verifier" and "Training example schema" contracts
from docs/BUILD_PLAN.md: one accept per taxonomy type, one reject per reason
code, byte-identity of the emitted prompt with the zero_shot eval condition,
and the audit gate's exit status.
"""
import json
import re
import subprocess
import sys

import pytest
from conftest import REPO_ROOT

import eval as E
from datagen import generate as G
from datagen.filter import (ExampleFilter, LeakageIndex, build_row, normalize,
                            split_user_msg)

AUDIT = str(REPO_ROOT / "datagen" / "audit.py")


# ---------------------------------------------------------------- canned data
def cand(type_, gold_label, passage, question, assistant):
    return {"type": type_, "gold_label": gold_label, "passage": passage,
            "question": question, "assistant": assistant}


CLEAN_PASSAGE = (
    "The Harrowgate Beekeeping Cooperative registered 214 new hives across its member "
    "farms during the most recent season. Inspectors recorded a winter survival rate of "
    "78 percent for the registered colonies, the strongest figure since the cooperative "
    "began keeping its own records. Members share extraction equipment from a central "
    "barn and rotate the maintenance duty every fortnight. A grant from the regional "
    "agricultural board paid for two replacement centrifuges last autumn. The cooperative "
    "plans to add a queen-rearing programme once its mentoring roster is full.")
CLEAN_QUESTION = "What winter survival rate did inspectors record for the registered colonies?"
CLEAN_SPAN = ("Inspectors recorded a winter survival rate of 78 percent for the "
              "registered colonies")
CLEAN_ASSISTANT = f'Answer: 78 percent.\nCitation: "{CLEAN_SPAN}"'

TWIN_PASSAGE = (
    "The Harrowgate Beekeeping Cooperative registered 214 new hives across its member "
    "farms during the most recent season. Members share extraction equipment from a "
    "central barn and rotate the maintenance duty every fortnight. A grant from the "
    "regional agricultural board paid for two replacement centrifuges last autumn. The "
    "cooperative plans to add a queen-rearing programme once its mentoring roster is full.")

DISTRACTOR_PASSAGE = (
    "Pellworth Glassworks operates a studio on Ferrier Lane where three resident makers "
    "share the furnace schedule. Ines Abara joined the studio in March 2022 and now runs "
    "the evening blowing sessions. Rueben Tiegs came aboard in August 2023 to manage the "
    "annealing ovens and the apprentice rota. The newest resident, Odile Marchetti, "
    "started in February 2024 and focuses on cast panels for architectural commissions. "
    "The studio sells seconds at a quarterly sale that funds its scholarship fund.")
DISTRACTOR_QUESTION = "What is the annual revenue of Pellworth Glassworks?"

PARAMETRIC_PASSAGE = (
    "Conservators working on the Colosseum have spent recent seasons stabilising the "
    "travertine facade against vibration from nearby traffic. Crews inject lime-based "
    "grout into hairline fractures and replace corroded iron cramps left by earlier "
    "restorations. Rainwater management has become a second priority, since pooling on "
    "the upper walkways accelerates salt damage in the porous stone. Visitor routes are "
    "adjusted each season so that scaffolded sections stay closed while work proceeds. "
    "A monitoring network of sensors now reports movement in the outer ring continuously.")
PARAMETRIC_QUESTION = "In what year was the Colosseum completed?"

NEAR_MISS_PASSAGE = (
    "The Selbury Aquatic Centre finished a refit of its lap pool and family changing "
    "rooms over the spring shutdown. Staff note that the centre never locks its doors, "
    "at any hour and on any date in the calendar, with lifeguards on deck during busy "
    "stretches and badge access otherwise. Membership covers a second site beside the "
    "athletics track, and single-visit tickets are sold at the desk while it is staffed. "
    "The refit added a sauna and replaced a filtration plant dating from the original "
    "build. Swim-club bookings resume on the first weekend of the new term.")
NEAR_MISS_QUESTION = "Is the aquatic centre open 24/7?"
NEAR_MISS_SPAN = "the centre never locks its doors, at any hour and on any date in the calendar"
NEAR_MISS_ASSISTANT = f'Answer: Yes.\nCitation: "{NEAR_MISS_SPAN}"'

REFUSAL = E.REFUSAL_STRING

ACCEPT_CASES = {
    "clean_answerable": cand("clean_answerable", "answerable", CLEAN_PASSAGE,
                             CLEAN_QUESTION, CLEAN_ASSISTANT),
    "support_deleted": cand("support_deleted", "unanswerable", TWIN_PASSAGE,
                            CLEAN_QUESTION, REFUSAL),
    "distractor": cand("distractor", "unanswerable", DISTRACTOR_PASSAGE,
                       DISTRACTOR_QUESTION, REFUSAL),
    "parametric_trap": cand("parametric_trap", "unanswerable", PARAMETRIC_PASSAGE,
                            PARAMETRIC_QUESTION, REFUSAL),
    "near_miss": cand("near_miss", "answerable", NEAR_MISS_PASSAGE,
                      NEAR_MISS_QUESTION, NEAR_MISS_ASSISTANT),
}


# ---------------------------------------------------------------- accepts
@pytest.mark.parametrize("type_", sorted(ACCEPT_CASES))
def test_accepts_one_canned_example_per_taxonomy_type(type_):
    v = ExampleFilter().check(ACCEPT_CASES[type_])
    assert v.ok, f"{type_} rejected as {v.reason}: {v.detail}"
    assert v.parse_class == ("refusal" if ACCEPT_CASES[type_]["gold_label"] == "unanswerable"
                             else "answer")


def test_every_taxonomy_type_is_covered_by_the_accept_cases():
    from datagen.filter import TAXONOMY_TYPES
    assert set(ACCEPT_CASES) == set(TAXONOMY_TYPES)


# ---------------------------------------------------------------- rejects
def test_reject_bad_format():
    c = cand("clean_answerable", "answerable", CLEAN_PASSAGE, CLEAN_QUESTION,
             "Sure! Based on the passage, I think the answer is probably 78 percent.")
    v = ExampleFilter().check(c)
    assert not v.ok and v.reason == "parse_malformed"


def test_reject_answer_line_without_citation_is_malformed():
    c = cand("clean_answerable", "answerable", CLEAN_PASSAGE, CLEAN_QUESTION,
             "Answer: 78 percent.")
    v = ExampleFilter().check(c)
    assert not v.ok and v.reason == "parse_malformed"


def test_reject_wrong_class_for_gold_unanswerable():
    """Teacher answered an item whose gold label is unanswerable."""
    c = cand("parametric_trap", "unanswerable", PARAMETRIC_PASSAGE, PARAMETRIC_QUESTION,
             'Answer: 80 AD.\nCitation: "Conservators working on the Colosseum"')
    v = ExampleFilter().check(c)
    assert not v.ok and v.reason == "class_mismatch"
    assert v.parse_class == "answer"


def test_reject_wrong_class_for_gold_answerable():
    c = cand("clean_answerable", "answerable", CLEAN_PASSAGE, CLEAN_QUESTION, REFUSAL)
    v = ExampleFilter().check(c)
    assert not v.ok and v.reason == "class_mismatch"
    assert v.parse_class == "refusal"


def test_reject_nonverbatim_span():
    c = cand("clean_answerable", "answerable", CLEAN_PASSAGE, CLEAN_QUESTION,
             'Answer: 78 percent.\nCitation: "inspectors logged a winter survival rate of 78%"')
    v = ExampleFilter().check(c)
    assert not v.ok and v.reason == "nonverbatim_citation"


def test_reject_quote_in_passage():
    passage = CLEAN_PASSAGE.replace("Harrowgate", 'the "Harrowgate"')
    c = cand("clean_answerable", "answerable", passage, CLEAN_QUESTION, CLEAN_ASSISTANT)
    v = ExampleFilter().check(c)
    assert not v.ok and v.reason == "quote_in_passage"


def test_reject_6gram_leakage_against_the_real_eval_set():
    """A generated passage that reuses six consecutive words from a frozen eval
    scenario must be rejected -- this is the leakage guard."""
    eval_rows = E.load_jsonl(str(REPO_ROOT / "data" / "eval" / "scenarios_v1.jsonl"))
    stolen = " ".join(eval_rows[0]["passage"].split()[8:20])
    passage = ("Notes from the Ferndale survey office describe the coming season. " +
               stolen + " Field crews will revisit the site in the autumn. The office "
               "publishes its schedule each spring. Volunteers handle the transcription.")
    c = cand("distractor", "unanswerable", passage,
             "How many staff does the Ferndale survey office employ?", REFUSAL)
    v = ExampleFilter().check(c)
    assert not v.ok and v.reason == "leakage_6gram"


def test_reject_6gram_leakage_against_the_fewshot_exemplars():
    ex = E.load_jsonl(str(REPO_ROOT / "data" / "eval" / "fewshot_exemplars.jsonl"))
    stolen = " ".join(ex[0]["passage"].split()[5:18])
    passage = ("A short note follows. " + stolen +
               " Nothing else was recorded. The clerk filed the note. It sits in a box.")
    c = cand("distractor", "unanswerable", passage, "Who filed the note in 1994?", REFUSAL)
    v = ExampleFilter().check(c)
    assert not v.ok and v.reason == "leakage_6gram"


def test_reject_duplicate():
    f = ExampleFilter()
    assert f.check(ACCEPT_CASES["clean_answerable"]).ok
    again = dict(ACCEPT_CASES["clean_answerable"])
    again["passage"] = "  " + again["passage"].upper() + "  "   # normalized dupe
    again["assistant"] = REFUSAL
    again["gold_label"] = "unanswerable"
    again["type"] = "support_deleted"
    v = f.check(again)
    assert not v.ok and v.reason == "duplicate"


def test_reject_unknown_type_and_bad_schema():
    f = ExampleFilter()
    v = f.check(cand("not_a_type", "answerable", CLEAN_PASSAGE, CLEAN_QUESTION,
                     CLEAN_ASSISTANT))
    assert not v.ok and v.reason == "unknown_type"
    v = f.check({"type": "clean_answerable", "gold_label": "answerable",
                 "passage": CLEAN_PASSAGE, "question": CLEAN_QUESTION})
    assert not v.ok and v.reason == "bad_schema"


def test_every_rejection_carries_a_known_reason_code():
    from datagen.filter import REASONS
    f = ExampleFilter()
    f.check(cand("clean_answerable", "answerable", CLEAN_PASSAGE, CLEAN_QUESTION, "nope"))
    f.check(cand("clean_answerable", "answerable", CLEAN_PASSAGE, CLEAN_QUESTION, REFUSAL))
    assert f.rejected and set(f.rejected) <= set(REASONS)
    assert all(isinstance(v, int) for v in f.rejected.values())


# ---------------------------------------------------------------- schema
def test_emitted_prompt_is_byte_identical_to_the_zero_shot_eval_condition():
    c = ACCEPT_CASES["clean_answerable"]
    v = ExampleFilter().check(c)
    row = build_row(c, "unit_test", v)
    sys_msg, user, asst = row["messages"]
    assert sys_msg["content"] == E.SYSTEM_CORE
    assert user["content"] == E.user_msg(c["passage"], c["question"])
    assert asst["content"] == c["assistant"]
    # and identical to what eval.build_prompt would send in the zero_shot cell
    prompt = E.build_prompt("zero_shot", {"passage": c["passage"],
                                          "question": c["question"]}, None)
    assert prompt["system"] == sys_msg["content"]
    assert prompt["turns"][-1]["content"] == user["content"]
    assert split_user_msg(user["content"]) == (c["passage"], c["question"])


def test_row_meta_matches_the_training_example_schema():
    c = ACCEPT_CASES["near_miss"]
    row = build_row(c, "batch_x", ExampleFilter().check(c))
    assert set(row) == {"messages", "meta"}
    assert set(row["meta"]) == {"type", "gold_label", "source_batch", "verifier"}
    assert row["meta"]["type"] == "near_miss"
    assert row["meta"]["gold_label"] == "answerable"
    assert row["meta"]["source_batch"] == "batch_x"
    assert row["meta"]["verifier"]["rubric_hash"] == E.rubric_hash()
    assert "eval.parse_output:answer" in row["meta"]["verifier"]["checks"]


# ---------------------------------------------------------------- helpers
def test_support_deleted_twin_is_built_by_sentence_deletion():
    twin, reason = G.delete_support(CLEAN_PASSAGE, [CLEAN_SPAN])
    assert reason == ""
    assert twin == TWIN_PASSAGE
    assert CLEAN_SPAN not in twin
    assert "214 new hives" in twin           # the rest of the passage survives


def test_support_deletion_refuses_when_too_little_would_remain():
    passage = "Alpha one two. Beta three four. Gamma five six."
    twin, reason = G.delete_support(passage, ["Beta three four"])
    assert twin is None and reason == "twin_too_short"


def test_allocation_sums_to_n_and_is_label_balanced():
    for n in (8, 64, 1024):
        alloc = G.allocate(n, G.QUOTAS)
        assert sum(alloc.values()) == n
        counts = {"answerable": 0, "unanswerable": 0}
        for bucket, k in alloc.items():
            counts[G.BUCKETS[bucket][2]] += k
        assert G.label_imbalance(counts) <= 0.05, (n, alloc)


def test_parse_items_accepts_a_fenced_json_array():
    items = G.parse_items('```json\n[{"passage": "p", "question": "q"}]\n```')
    assert items == [{"passage": "p", "question": "q"}]
    with pytest.raises(ValueError):
        G.parse_items("I would rather not.")


def test_missing_models_config_is_a_clear_stop(tmp_path):
    with pytest.raises(SystemExit) as e:
        G.load_teacher_spec(tmp_path / "models.json")
    assert "configs/models.json" in str(e.value)
    bad = tmp_path / "m.json"
    bad.write_text(json.dumps({"frontier": []}))
    with pytest.raises(SystemExit):
        G.load_teacher_spec(bad)
    good = tmp_path / "g.json"
    good.write_text(json.dumps({"frontier": ["openai:some-model", "google:other"]}))
    assert G.load_teacher_spec(good)[0] == "openai:some-model"


# ---------------------------------------------------------------- scripted teacher
class ScriptedTeacherGenerator:
    """A canned frontier teacher: JSON items on stage 1, eval-format targets on
    stage 2. Zero network."""

    def __init__(self):
        self.counter = {b: 0 for b in G.BUCKETS}
        self.support = {}          # normalized question -> {spans, answer}
        self.gen_calls = 0
        self.target_calls = 0

    def _bucket_of(self, user):
        for bucket, spec in G.ITEM_SPECS.items():
            if spec.splitlines()[0] in user:
                return bucket
        raise AssertionError("unrecognised datagen prompt")

    def __call__(self, scenario, prompt):
        user = prompt["turns"][-1]["content"]
        if user.startswith("Context passage:\n"):
            self.target_calls += 1
            passage, question = split_user_msg(user)
            info = self.support.get(normalize(question))
            if info and all(s in passage for s in info["spans"]):
                cites = "\n".join(f'Citation: "{s}"' for s in info["spans"])
                return f"Answer: {info['answer']}\n{cites}"
            return REFUSAL
        self.gen_calls += 1
        bucket = self._bucket_of(user)
        k = int(re.search(r"Write (\d+) items", user).group(1))
        items = []
        for _ in range(k):
            i = self.counter[bucket]
            self.counter[bucket] = i + 1
            item = ITEM_FACTORIES[bucket](i)
            if item.get("gold_spans"):
                self.support[normalize(item["question"])] = {
                    "spans": item["gold_spans"], "answer": item["gold_answer"]}
            items.append(item)
        return "```json\n" + json.dumps(items) + "\n```"


def _answerable_item(i):
    name = f"Harrowgate {i}"
    span = (f"Inspectors recorded a winter survival rate of {70 + i} percent for the "
            f"{name} colonies")
    passage = (
        f"The {name} Beekeeping Cooperative registered {200 + i} new hives across its "
        f"member farms during the most recent season. {span}, the strongest figure since "
        "the cooperative began keeping its own records. Members share extraction "
        "equipment from a central barn and rotate the maintenance duty every fortnight. "
        "A grant from the regional agricultural board paid for two replacement "
        "centrifuges last autumn. The cooperative plans to add a queen-rearing programme "
        "once its mentoring roster is full.")
    return {"domain": "beekeeping", "passage": passage,
            "question": f"What winter survival rate did inspectors record for the {name} colonies?",
            "gold_label": "answerable", "gold_answer": f"{70 + i} percent.",
            "gold_spans": [span]}


def _competing_item(i):
    name = f"Pellworth {i}"
    span = f"Rueben Tiegs came aboard at {name} in August {2010 + i}"
    passage = (
        f"{name} Glassworks operates a studio on Ferrier Lane where three resident makers "
        f"share the furnace schedule. Ines Abara joined the studio in March {2005 + i} and "
        f"now runs the evening blowing sessions. {span} to manage the annealing ovens and "
        f"the apprentice rota. The newest resident, Odile Marchetti, started in February "
        f"{2018 + i} and focuses on cast panels for architectural commissions. The studio "
        "sells seconds at a quarterly sale that funds its scholarship fund.")
    return {"domain": "glassblowing", "passage": passage,
            "question": f"When did Rueben Tiegs come aboard at {name}?",
            "gold_label": "answerable", "gold_answer": f"August {2010 + i}.",
            "gold_spans": [span]}


def _paraphrase_item(i):
    name = f"Selbury {i}"
    span = f"the {name} centre never locks its doors, at any hour and on any date in the calendar"
    passage = (
        f"The {name} Aquatic Centre finished a refit of its lap pool and family changing "
        f"rooms over the spring shutdown. Staff note that {span}, with lifeguards on deck "
        "during busy stretches and badge access otherwise. Membership covers a second site "
        "beside the athletics track, and single-visit tickets are sold at the desk while it "
        "is staffed. The refit added a sauna and replaced a filtration plant dating from "
        "the original build. Swim-club bookings resume on the first weekend of the new term.")
    return {"domain": "campus-housing", "passage": passage,
            "question": f"Is the {name} aquatic centre open 24/7?",
            "gold_label": "answerable", "gold_answer": "Yes.", "gold_spans": [span]}


def _absent_item(i):
    name = f"Pellworth {i}"
    passage = (
        f"{name} Glassworks operates a studio on Ferrier Lane where three resident makers "
        "share the furnace schedule. Ines Abara runs the evening blowing sessions and "
        "teaches a beginner course twice each term. Rueben Tiegs manages the annealing "
        "ovens and the apprentice rota. Odile Marchetti focuses on cast panels for "
        "architectural commissions and keeps the kiln log. The studio sells seconds at a "
        "quarterly sale that funds its scholarship fund.")
    return {"domain": "glassblowing", "passage": passage,
            "question": f"What was the annual revenue of {name} Glassworks?",
            "gold_label": "unanswerable", "gold_answer": None, "gold_spans": None}


def _parametric_item(i):
    passage = (
        "Conservators working on the Colosseum have spent recent seasons stabilising the "
        f"travertine facade against vibration from nearby traffic in sector {i}. Crews "
        "inject lime-based grout into hairline fractures and replace corroded iron cramps "
        "left by earlier restorations. Rainwater management has become a second priority, "
        "since pooling on the upper walkways accelerates salt damage in the porous stone. "
        "Visitor routes are adjusted each season so that scaffolded sections stay closed "
        "while work proceeds. A monitoring network of sensors now reports movement in the "
        "outer ring continuously.")
    return {"domain": "the Colosseum", "passage": passage,
            "question": f"In what year was the Colosseum completed, per record {i}?",
            "gold_label": "unanswerable", "gold_answer": None, "gold_spans": None}


def _shifted_item(i):
    name = f"Larkfield {i}"
    passage = (
        f"{name} Outfitters told analysts this week that turnover for the period climbed "
        f"{10 + i} percent above the matching three months a year earlier, helped by "
        "orders for its trail shoes. Managers lifted the full-year sales outlook and said "
        "stock levels had settled after two seasons of markdowns. Three shops opened "
        "during the period, each beside a college campus in the mountain west. Freight "
        "bills were named as the main cost pressure heading into winter. Spring price "
        "lists reach wholesale partners next month.")
    return {"domain": "microbrewing", "passage": passage,
            "question": f"What was the profit growth rate at {name} Outfitters for the period?",
            "gold_label": "unanswerable", "gold_answer": None, "gold_spans": None}


ITEM_FACTORIES = {
    "clean_answerable": _answerable_item,
    "distractor_competing": _competing_item,
    "near_miss_paraphrase": _paraphrase_item,
    "distractor_absent": _absent_item,
    "parametric_trap": _parametric_item,
    "near_miss_shifted": _shifted_item,
}


def _run_pipeline(tmp_path, n=8, scripted=None, batch="unit_batch"):
    scripted = scripted or ScriptedTeacherGenerator()
    teacher = G.Teacher("mock:scripted", cache_dir=tmp_path / "cache", verbose=False)
    teacher._gen = scripted
    flt = ExampleFilter()
    gen = G.Generator(teacher, flt, batch, items_per_call=4, seed=42, verbose=False)
    cands, extra = gen.run(n, 0.05)
    rows = [build_row(c, batch, c["verdict"]) for c in cands]
    return rows, flt, teacher, scripted, extra, cands


def test_end_to_end_pipeline_with_a_canned_teacher(tmp_path):
    rows, flt, teacher, scripted, _, _c = _run_pipeline(tmp_path, n=8)
    assert len(rows) == 8
    types = {r["meta"]["type"] for r in rows}
    assert types == set(G.BUCKETS[b][0] for b in G.BUCKETS)
    labels = [r["meta"]["gold_label"] for r in rows]
    assert labels.count("answerable") == 4 and labels.count("unanswerable") == 4
    assert all(r["messages"][0]["content"] == E.SYSTEM_CORE for r in rows)
    assert teacher.api_calls > 0 and scripted.target_calls > 0
    # the support_deleted rows really are sentence-deleted twins that refuse
    twins = [r for r in rows if r["meta"]["type"] == "support_deleted"]
    assert twins
    for t in twins:
        assert t["messages"][2]["content"] == REFUSAL


def test_teacher_replies_are_cached_so_a_rerun_does_not_rebill(tmp_path):
    _rows, _flt, teacher1, _s, _e, _c = _run_pipeline(tmp_path, n=8)
    assert teacher1.api_calls > 0

    class Exploding:
        def __call__(self, scenario, prompt):
            raise AssertionError("cache miss: the teacher was called again")

    rows2, _flt2, teacher2, _s2, _e2, _c2 = _run_pipeline(tmp_path, n=8, scripted=Exploding())
    assert teacher2.api_calls == 0
    assert teacher2.cache.hits > 0
    assert len(rows2) == 8


# ------------------------------------------------- DEFECT 1: budget scaling
@pytest.mark.parametrize("n", [64, 256, 1024])
def test_generator_reaches_the_requested_size_with_a_perfect_teacher(tmp_path, n):
    """A hard ceiling of max_calls_per_bucket x items_per_call used to cap the
    output at ~320 rows: n=256 emitted 238 and n=1024 emitted 320 (31%). With a
    teacher whose items are never rejected, --n must actually be delivered."""
    rows, _flt, _t, _s, extra, _c = _run_pipeline(tmp_path, n=n)
    assert len(rows) == n, f"asked for {n}, emitted {len(rows)}"
    assert extra["short_buckets"] == []
    assert all(s["stop"] == "target_met" for s in extra["bucket_stops"])


def test_call_budget_scales_with_the_request_not_a_constant(tmp_path):
    teacher = G.Teacher("mock:scripted", cache_dir=tmp_path / "cache", verbose=False)
    teacher._gen = ScriptedTeacherGenerator()
    gen = G.Generator(teacher, ExampleFilter(), "b", items_per_call=4, verbose=False)
    small, large = gen.call_budget(4), gen.call_budget(266)
    assert large > small
    # enough calls to cover the need even if most items are rejected
    assert large >= 266 / gen.items_per_call
    # an explicit cap is still honoured as an operator override
    capped = G.Generator(teacher, ExampleFilter(), "b", items_per_call=4,
                         max_calls_per_bucket=3, verbose=False)
    assert capped.call_budget(1000) == 3


def test_a_hopeless_teacher_stops_early_with_a_reported_reason(tmp_path):
    """The budget must not be burned in full on a teacher that cannot produce
    the shape: a genuine stopping condition fires and is named."""
    class Garbage:
        def __call__(self, scenario, prompt):
            return "I would rather not."

    teacher = G.Teacher("mock:scripted", cache_dir=tmp_path / "cache", verbose=False)
    teacher._gen = Garbage()
    gen = G.Generator(teacher, ExampleFilter(), "b", items_per_call=4, verbose=False)
    cands, extra = gen.run(64, 0.05)
    assert cands == []
    stops = {s["stop"] for s in extra["bucket_stops"]}
    assert stops <= set(G.STOP_REASONS)
    assert "teacher_exhausted" in stops
    for s in extra["bucket_stops"]:
        if s["bucket"] != "support_deleted":
            assert s["calls"] <= gen.stall_calls, s
    assert extra["short_buckets"]


# ------------------------------- DEFECT 2: exit status is a success signal
def _main_with_teacher(tmp_path, gen_obj, monkeypatch, n=16, extra_argv=()):
    monkeypatch.setattr(G.E, "make_generator", lambda spec: gen_obj)
    cfg = tmp_path / "models.json"
    cfg.write_text(json.dumps({"frontier": ["mock:scripted", "mock:other"],
                               "judge": "mock:judge"}))
    out = tmp_path / "out.jsonl"
    rc = G.main(["--n", str(n), "--out", str(out), "--batch-name", "cli_batch",
                 "--models-config", str(cfg), "--cache-dir", str(tmp_path / "cache"),
                 "--quiet", *extra_argv])
    return rc, out


def test_main_exits_zero_and_emits_the_full_batch(tmp_path, monkeypatch):
    rc, out = _main_with_teacher(tmp_path, ScriptedTeacherGenerator(), monkeypatch, n=16)
    assert rc == 0
    assert len(out.read_text().splitlines()) == 16
    report = json.loads(out.with_suffix(".report.json").read_text())
    assert report["n_kept"] == 16 and report["shortfall"] == 0


def test_main_exits_nonzero_and_writes_nothing_when_it_keeps_zero(tmp_path, monkeypatch):
    """Used to print 'wrote 0 examples' + 'WARNING: emitted 0' and exit 0,
    leaving a 0-byte file that reads as a deliverable."""
    class Incapable:
        def __call__(self, scenario, prompt):
            return "I would rather not."

    rc, out = _main_with_teacher(tmp_path, Incapable(), monkeypatch, n=16)
    assert rc != 0
    assert not out.exists(), "an empty run must not leave a 0-byte dataset behind"


def test_an_empty_run_does_not_clobber_an_existing_dataset(tmp_path, monkeypatch):
    good = _main_with_teacher(tmp_path, ScriptedTeacherGenerator(), monkeypatch, n=16)[1]
    before = good.read_text()
    assert before.strip()

    class Incapable:
        def __call__(self, scenario, prompt):
            return "no."

    monkeypatch.setattr(G.E, "make_generator", lambda spec: Incapable())
    cfg = tmp_path / "models.json"
    rc = G.main(["--n", "16", "--out", str(good), "--batch-name", "cli_batch2",
                 "--models-config", str(cfg), "--cache-dir", str(tmp_path / "cache2"),
                 "--quiet"])
    assert rc != 0
    assert good.read_text() == before


def test_main_exits_nonzero_on_a_material_shortfall(tmp_path, monkeypatch):
    """Some rows, but far fewer than --n: still not a success."""
    class HalfBroken(ScriptedTeacherGenerator):
        def __call__(self, scenario, prompt):
            user = prompt["turns"][-1]["content"]
            if not user.startswith("Context passage:\n"):
                if self._bucket_of(user) != "clean_answerable":
                    return "I would rather not."
            return super().__call__(scenario, prompt)

    rc, out = _main_with_teacher(tmp_path, HalfBroken(), monkeypatch, n=16)
    assert rc != 0
    kept = len(out.read_text().splitlines())
    assert 0 < kept < 16
    report = json.loads(out.with_suffix(".report.json").read_text())
    assert report["n_kept"] == kept
    assert report["shortfall"] > 0
    assert report["short_buckets"]


def test_min_kept_makes_a_tolerated_shortfall_explicit(tmp_path, monkeypatch):
    class HalfBroken(ScriptedTeacherGenerator):
        def __call__(self, scenario, prompt):
            user = prompt["turns"][-1]["content"]
            if not user.startswith("Context passage:\n"):
                if self._bucket_of(user) != "clean_answerable":
                    return "I would rather not."
            return super().__call__(scenario, prompt)

    rc, out = _main_with_teacher(tmp_path, HalfBroken(), monkeypatch, n=16,
                                 extra_argv=("--min-kept", "2"))
    assert rc == 0
    assert len(out.read_text().splitlines()) >= 2


def test_min_kept_defaults_to_a_tight_fraction_of_n():
    assert G.MIN_KEPT_FRACTION >= 0.9
    import math as _m
    assert _m.ceil(G.MIN_KEPT_FRACTION * 1024) >= 972


# ---------------------------------------------------------------- audit gate
def _sidecar_for(rows, **override):
    """A sidecar report that HONESTLY describes ``rows``.

    The audit now cross-checks every reported number against the rows, so a
    fixture must supply the report a real generator run would have written.
    ``override`` is how a test makes the sidecar lie on purpose.
    """
    label_counts, type_counts, batches = {}, {}, []
    for r in rows:
        meta = r.get("meta", {})
        lab, typ = meta.get("gold_label"), str(meta.get("type"))
        label_counts[lab] = label_counts.get(lab, 0) + 1
        type_counts[typ] = type_counts.get(typ, 0) + 1
        batches.append(str(meta.get("source_batch")))
    report = {
        "batch_name": batches[0] if batches else "audit_test",
        "teacher": "mock:scripted",
        "n_requested": len(rows), "n_kept": len(rows),
        "passed_verifier": len(rows),
        "label_counts": label_counts,
        "label_imbalance": round(G.label_imbalance(label_counts), 6),
        "type_counts": type_counts,
        "rejected": {}, "rejected_total": 0,
        "rubric_hash": E.rubric_hash(),
        "teacher_api_calls": 0, "cache_hits": 0,
    }
    report.update(override)
    return report


def _write(path, rows, report=None, sidecar=True):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    if sidecar:
        rep = _sidecar_for(rows) if report is None else report
        path.with_suffix(".report.json").write_text(json.dumps(rep, indent=2))
    return path


def _audit(path, *extra):
    return subprocess.run([sys.executable, AUDIT, str(path), *extra],
                          capture_output=True, text=True, cwd=str(REPO_ROOT))


def _balanced_rows():
    f = ExampleFilter()
    rows = []
    for key in ("clean_answerable", "support_deleted", "distractor", "near_miss"):
        c = ACCEPT_CASES[key]
        rows.append(build_row(c, "audit_test", f.check(c)))
    return rows


def test_audit_exits_zero_on_a_balanced_file(tmp_path):
    rows = _balanced_rows()
    assert sum(r["meta"]["gold_label"] == "answerable" for r in rows) == 2
    r = _audit(_write(tmp_path / "balanced.jsonl", rows), "--max-imbalance", "0.05")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PASS" in r.stdout


def test_audit_exits_nonzero_on_an_imbalanced_file(tmp_path):
    rows = [r for r in _balanced_rows() if r["meta"]["gold_label"] == "unanswerable"]
    r = _audit(_write(tmp_path / "imbalanced.jsonl", rows), "--max-imbalance", "0.05")
    assert r.returncode != 0
    assert "label imbalance" in r.stdout


def test_audit_exits_nonzero_on_leakage(tmp_path):
    rows = _balanced_rows()
    eval_rows = E.load_jsonl(str(REPO_ROOT / "data" / "eval" / "scenarios_v1.jsonl"))
    stolen = " ".join(eval_rows[2]["passage"].split()[4:16])
    leaked = dict(ACCEPT_CASES["distractor"])
    leaked["passage"] = "A note follows. " + stolen + " The file was closed."
    f = ExampleFilter()
    rows.append(build_row(leaked, "audit_test", f.check(leaked)))
    rows.append(build_row(ACCEPT_CASES["parametric_trap"], "audit_test",
                          f.check(ACCEPT_CASES["parametric_trap"])))
    r = _audit(_write(tmp_path / "leaky.jsonl", rows))
    assert r.returncode != 0
    assert "leaking row" in r.stdout


def test_audit_exits_nonzero_on_a_schema_violation(tmp_path):
    rows = _balanced_rows()
    rows[0]["messages"][0]["content"] = E.SYSTEM_CORE + " (helpful extra guidance)"
    r = _audit(_write(tmp_path / "badschema.jsonl", rows))
    assert r.returncode != 0
    assert "system_not_eval_SYSTEM_CORE" in r.stdout


def test_audit_exits_nonzero_when_the_user_message_is_not_eval_user_msg(tmp_path):
    rows = _balanced_rows()
    rows[1]["messages"][1]["content"] = "Passage: whatever\nQ: whatever"
    r = _audit(_write(tmp_path / "baduser.jsonl", rows))
    assert r.returncode != 0
    assert "user_not_eval_user_msg" in r.stdout


def test_audit_exits_nonzero_on_duplicates_and_on_an_empty_file(tmp_path):
    rows = _balanced_rows()
    dup = json.loads(json.dumps(rows[0]))
    dup["meta"]["gold_label"] = "unanswerable"
    dup["messages"][2]["content"] = REFUSAL
    dup["meta"]["type"] = "support_deleted"
    rows.append(dup)
    rows.append(json.loads(json.dumps(rows[1])))
    r = _audit(_write(tmp_path / "dupes.jsonl", rows))
    assert r.returncode != 0
    assert "duplicate" in r.stdout
    empty = _audit(_write(tmp_path / "empty.jsonl", []))
    assert empty.returncode != 0


def test_audit_passes_the_generated_pipeline_output(tmp_path):
    """The full offline pipeline output clears the real P3 gate.

    The sidecar is built from the run's own candidates -- ``build_report([])``
    would describe an empty batch beside an 8-row file, which the gate now
    (correctly) rejects; see test_audit_fails_when_the_sidecar_disagrees.
    """
    rows, flt, teacher, _s, extra, cands = _run_pipeline(tmp_path, n=8)
    report = G.build_report(
        cands, flt, teacher, "unit_batch", 8, tmp_path / "models.json", 0.05, extra)
    assert report["n_kept"] == len(rows)
    out = _write(tmp_path / "smoke.jsonl", rows, report=report)
    r = _audit(out, "--max-imbalance", "0.05")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PASS" in r.stdout
    assert "kept / rejected" in r.stdout


# --------------------------------- DEFECT 3: the sidecar is a claim, not fact
def test_audit_fails_when_the_sidecar_disagrees_with_the_rows(tmp_path):
    """A 4-row file whose sidecar claims 64 kept / 32+32 labels used to print
    'requested 64  passed verifier 64  kept in file 64' and exit 0."""
    rows = _balanced_rows()
    liar = _sidecar_for(rows, n_requested=64, n_kept=64, passed_verifier=64,
                        label_counts={"answerable": 32, "unanswerable": 32},
                        type_counts={"clean_answerable": 32, "distractor": 32})
    r = _audit(_write(tmp_path / "liar.jsonl", rows, report=liar))
    assert r.returncode != 0, r.stdout
    assert "report_mismatch n_kept" in r.stdout
    assert "report_mismatch label_counts" in r.stdout
    assert "report_mismatch type_counts" in r.stdout
    assert "PASS" not in r.stdout


@pytest.mark.parametrize("field,value,marker", [
    ("label_imbalance", 0.4242, "report_mismatch label_imbalance"),
    ("rubric_hash", "deadbeefdeadbeef", "report_mismatch rubric_hash"),
    ("batch_name", "some_other_batch", "report_mismatch batch_name"),
    ("passed_verifier", 1, "report_mismatch passed_verifier"),
    ("rejected_total", 99, "report_mismatch rejected_total"),
    ("rejected", {"not_a_real_code": 3}, "report_mismatch rejected"),
    ("bucket_counts", {"clean_answerable": 999}, "report_mismatch bucket_counts"),
])
def test_audit_recomputes_each_reported_number_and_fails_on_disagreement(
        tmp_path, field, value, marker):
    rows = _balanced_rows()
    rep = _sidecar_for(rows, **{field: value})
    r = _audit(_write(tmp_path / "bad.jsonl", rows, report=rep))
    assert r.returncode != 0, r.stdout
    assert marker in r.stdout, r.stdout


def test_audit_fails_on_a_short_dataset_even_with_an_honest_sidecar(tmp_path):
    """A 3x-short batch must not clear the gate silently: the sidecar honestly
    reports 21 kept against 64 requested, and the size floor binds."""
    rows = (_balanced_rows() * 6)[:21]
    for i, row in enumerate(rows):          # keep them distinct for the dedupe check
        row = json.loads(json.dumps(row))
        row["messages"][1]["content"] = E.user_msg(
            f"Note {i}. " + split_user_msg(row["messages"][1]["content"])[0],
            split_user_msg(row["messages"][1]["content"])[1])
        rows[i] = row
    rep = _sidecar_for(rows, n_requested=64)
    r = _audit(_write(tmp_path / "short.jsonl", rows, report=rep))
    assert r.returncode != 0, r.stdout
    assert "below the floor" in r.stdout
    ok = _audit(_write(tmp_path / "short2.jsonl", rows, report=_sidecar_for(rows)),
                "--max-imbalance", "0.05")
    assert ok.returncode == 0, ok.stdout          # same rows, honest n_requested


def test_audit_fails_when_the_sidecar_is_missing(tmp_path):
    rows = _balanced_rows()
    p = _write(tmp_path / "nosidecar.jsonl", rows, sidecar=False)
    r = _audit(p)
    assert r.returncode != 0
    assert "missing_report" in r.stdout
    r2 = _audit(p, "--allow-missing-report")
    assert r2.returncode == 0, r2.stdout


def test_audit_honours_an_explicit_min_rows_floor(tmp_path):
    rows = _balanced_rows()
    p = _write(tmp_path / "tiny.jsonl", rows)
    assert _audit(p).returncode == 0
    r = _audit(p, "--min-rows", "64")
    assert r.returncode != 0 and "below the floor of 64" in r.stdout


# ------------------------- DEFECT 4: the leakage guard binds, independently
def _leaky_candidate(text: str):
    passage = ("A short note from the parish office follows. " + text +
               " Nothing else was recorded that week. The clerk filed the note. "
               "It sits in a numbered box.")
    return cand("distractor", "unanswerable", passage,
                "How many boxes does the parish office hold?", REFUSAL)


@pytest.mark.parametrize("path_name", ["scenarios_v1.jsonl", "fewshot_exemplars.jsonl"])
@pytest.mark.parametrize("field", ["passage", "question"])
def test_a_6gram_from_any_eval_row_is_rejected_by_construction(path_name, field):
    """Take a 6-gram window out of EVERY eval passage and question in turn and
    confirm the generator's filter rejects a candidate carrying it."""
    eval_rows = E.load_jsonl(str(REPO_ROOT / "data" / "eval" / path_name))
    assert eval_rows
    checked = 0
    for row in eval_rows:
        words = str(row.get(field, "")).split()
        if len(words) < 6:
            continue
        for start in range(0, len(words) - 6 + 1, 7):     # sample windows
            stolen = " ".join(words[start:start + 6])
            v = ExampleFilter().check(_leaky_candidate(stolen))
            assert not v.ok and v.reason == "leakage_6gram", (
                f"{path_name} {field} window at {start} slipped through: {stolen!r}")
            checked += 1
    assert checked > 0


def test_a_5gram_is_not_enough_to_trip_the_guard():
    """The rule is 6-gram overlap; a 5-word borrow must NOT be rejected, or the
    'guard binds' evidence would just be a guard that rejects everything."""
    eval_rows = E.load_jsonl(str(REPO_ROOT / "data" / "eval" / "scenarios_v1.jsonl"))
    words = eval_rows[0]["passage"].split()
    v = ExampleFilter().check(_leaky_candidate(" ".join(words[3:8])))
    assert v.ok or v.reason != "leakage_6gram", v.reason


def test_the_audit_guard_is_a_second_implementation_that_agrees():
    import importlib
    audit = importlib.import_module("datagen.audit")
    from datagen.filter import default_leakage_index
    mine, theirs = audit.AuditLeakageIndex(), default_leakage_index()
    assert mine.grams == theirs.grams
    assert mine.grams is not theirs.grams
    assert len(mine.grams) > 1000


def test_audit_catches_leakage_even_when_the_generator_filter_is_bypassed(
        tmp_path, monkeypatch, capsys):
    """Blind datagen.filter's guard entirely. The audit must still fail the
    leaking row from its OWN index, and must report that the two guards
    disagree -- that is what makes the re-check evidence and not theatre."""
    import importlib
    audit = importlib.import_module("datagen.audit")
    import datagen.filter as F

    eval_rows = E.load_jsonl(str(REPO_ROOT / "data" / "eval" / "scenarios_v1.jsonl"))
    stolen = " ".join(eval_rows[1]["passage"].split()[6:18])
    leaked = _leaky_candidate(stolen)
    rows = _balanced_rows()
    rows.append(build_row(leaked, "audit_test", ExampleFilter().check(leaked)))
    rows.append(build_row(ACCEPT_CASES["parametric_trap"], "audit_test",
                          ExampleFilter().check(ACCEPT_CASES["parametric_trap"])))
    p = _write(tmp_path / "bypassed.jsonl", rows)

    blind = F.LeakageIndex([])                    # a filter that sees nothing
    monkeypatch.setattr(F, "_DEFAULT_INDEX", blind)
    monkeypatch.setattr(audit, "default_leakage_index", lambda: blind)
    assert blind.hits(*split_user_msg(rows[-2]["messages"][1]["content"])) == []

    rc = audit.main([str(p)])
    out = capsys.readouterr().out
    assert rc != 0
    assert "leakage_6gram" in out                 # caught by the audit's own guard
    assert "guard_disagreement" in out            # and the divergence is named
    assert "DISAGREE" in out


@pytest.mark.parametrize("mutation,expected", [
    ("nonverbatim", "nonverbatim_citation"),
    ("wrong_class", "class_mismatch"),
    ("quote", "quote_in_passage"),
    ("bad_meta", "bad_schema"),
    ("bad_type", "unknown_type"),
])
def test_audit_re_runs_the_verifier_and_fails_on_a_tampered_row(tmp_path, mutation, expected):
    """The gate re-derives the verifier contract from the emitted row itself --
    it does not trust meta.verifier."""
    rows = _balanced_rows()
    row = rows[0]                                     # clean_answerable, answerable
    if mutation == "nonverbatim":
        row["messages"][2]["content"] = ('Answer: 78 percent.\n'
                                         'Citation: "a span that is not in the passage"')
    elif mutation == "wrong_class":
        row["messages"][2]["content"] = REFUSAL
    elif mutation == "quote":
        row["messages"][1]["content"] = row["messages"][1]["content"].replace(
            "Harrowgate", 'the "Harrowgate"', 1)
        row["messages"][2]["content"] = REFUSAL       # keep the class check quiet
        row["meta"]["gold_label"] = "unanswerable"
    elif mutation == "bad_meta":
        row["meta"]["extra_key"] = "not in the schema"
    elif mutation == "bad_type":
        row["meta"]["type"] = "made_up_type"
    r = _audit(_write(tmp_path / f"{mutation}.jsonl", rows))
    assert r.returncode != 0, r.stdout
    assert expected in r.stdout
