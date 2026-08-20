"""configs/g1.json must name the right corpus, not merely an existing file.

The config previously pointed at the retired abstainer hedge set. That file
exists, so every path-existence check passed, and G1 would have scored
grounded-abstention scenarios with constraint checkers. These bind the schema.
"""
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
G1 = json.loads((REPO_ROOT / "configs" / "g1.json").read_text())

SCENARIO_KEYS = {"scenario_id", "task", "constraints", "k", "pressure", "split"}
ABSTAINER_KEYS = {"gold_label", "passage", "question", "gold_spans"}


def _rows(path):
    return [json.loads(l) for l in (REPO_ROOT / path).read_text().splitlines() if l.strip()]


def test_eval_set_is_the_constraint_corpus_not_a_retired_one():
    rows = _rows(G1["eval_set"])
    assert rows, f"{G1['eval_set']} is empty"
    for r in rows[:5]:
        assert SCENARIO_KEYS <= set(r), (
            f"{G1['eval_set']} rows are not constraint scenarios: missing "
            f"{SCENARIO_KEYS - set(r)}")
        assert not (ABSTAINER_KEYS & set(r)), (
            f"{G1['eval_set']} carries abstainer fields {ABSTAINER_KEYS & set(r)} -- "
            f"this is the retired target's data")


def test_eval_set_meets_the_r6_scenario_floor():
    """R6 requires >= 30 scenarios per model x strategy cell."""
    assert len(_rows(G1["eval_set"])) >= 30


def test_every_eval_scenario_resolves_against_the_pinned_instances():
    lookup = {c["id"] for c in _rows("data/scenarios/constraint_instances.jsonl")}
    for s in _rows(G1["eval_set"]):
        missing = [c for c in s["constraints"] if c not in lookup]
        assert not missing, f"{s['scenario_id']} references unknown constraints {missing}"


def test_exemplars_are_verified_compliant_and_leak_free():
    """Few-shot exemplars are shown to every model in the few_shot cell. One that
    violates its own constraints teaches the failure it exists to prevent."""
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from harness import build_constraint_lookup, run_checkers
    lookup = build_constraint_lookup(_rows("data/scenarios/constraint_instances.jsonl"))
    corpus_tasks = {s["task"].strip() for s in _rows(G1["eval_set"])}
    exemplars = _rows(G1["exemplars"])
    assert exemplars, "no exemplars; the few_shot cell would be unrunnable"
    for ex in exemplars:
        scen = {"scenario_id": ex["exemplar_id"], "task": "x",
                "constraints": ex["constraints"], "k": ex["k"],
                "pressure": "clean", "split": "train"}
        results = run_checkers(scen, lookup, ex["target_output"])
        bad = [r for r in results if not r["pass"]]
        assert not bad, f"{ex['exemplar_id']} violates its own constraints: {bad}"
        assert ex["rendered_prompt"].split("\n")[0].strip() not in corpus_tasks, (
            f"{ex['exemplar_id']} reuses a graded scenario's task -- leakage")


def test_exemplars_never_demonstrate_a_held_out_type():
    """A worked example of a held-out type hands the model exactly the in-context
    help the held-out split exists to withhold."""
    from constraints import registry
    instances = {c["id"]: c for c in _rows("data/scenarios/constraint_instances.jsonl")}
    for ex in _rows(G1["exemplars"]):
        for cid in ex["constraints"]:
            ctype = instances[cid]["type"]
            assert registry.split_of(ctype) == "train", (
                f"{ex['exemplar_id']} demonstrates held-out type {ctype}")


def test_models_agree_with_configs_models_json():
    models = json.loads((REPO_ROOT / "configs" / "models.json").read_text())
    assert G1["models"]["frontier"] == models["frontier"]
    assert G1["models"]["local_base"] == models["base"]
    assert G1["judge"]["model"] == models["judge_model"]


def test_two_distinct_frontier_families():
    fams = {m.split(":", 1)[0] for m in G1["models"]["frontier"]}
    assert len(fams) >= 2, f"R6 requires >=2 frontier families, got {fams}"
    assert G1["judge"]["provider"] not in fams, "the judge must not be an ablated family"


def test_google_worker_cap_is_encoded():
    """Measured: gemini-3.1-pro is capped at 25 req/min and 8 workers 429s mid-run."""
    assert G1["workers"]["by_provider"]["google"] <= 2
