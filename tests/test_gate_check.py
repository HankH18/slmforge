"""T-4 — proof that scripts/gate_check.py is a real gate, not a formality.

Every fixture is built in tmp_path. Nothing here reads or writes the repo's
runs/ directory: a previous agent destroyed a deliverable while sabotage-testing
against a live artifact, and these tests deliberately corrupt what they build.

The suite is organised as:
  * happy path + the three verdicts + both threshold boundaries
  * a parametrized SABOTAGE table — each entry mutates an otherwise-PROCEEDing
    run and must be rejected (exit 1), never rated
  * type-confusion, because `isinstance(True, int)` is True in Python and that
    bypass has already defeated guards in this repo
  * contract tests over configs/g1.json
"""
import ast
import hashlib
import json
import math
import os
import re
from fractions import Fraction
from pathlib import Path

import pytest
from conftest import REPO_ROOT

import scripts.gate_check as G
from constraints.checkers import check as real_check

FRONTIER_A = "openai:gpt-5.6-sol"
FRONTIER_B = "google:gemini-3.1-pro-preview"
LOCAL_BASE = "hf:Qwen/Qwen3-1.7B@" + "7" * 40
STRATEGIES = ("zero_shot", "few_shot", "structured")

RUNGS = [
    {"name": "r1-k2", "k_values": [2], "adversarial_share": 0.40},
    {"name": "r2-k3", "k_values": [2, 3], "adversarial_share": 0.40},
    {"name": "r3-k4", "k_values": [2, 3, 4], "adversarial_share": 0.40},
    {"name": "r4-adv60", "k_values": [2, 3, 4], "adversarial_share": 0.60},
    {"name": "r5-adv80", "k_values": [2, 3, 4], "adversarial_share": 0.80},
]
LAST_RUNG = RUNGS[-1]["name"]

N = 40  # >= R6's floor of 30, and divides the fail rates the table exercises

# --------------------------------------------------------- real constraints
# The gate re-runs constraints.checkers.check over (constraint, output), so a
# fixture can no longer wave a "pass": false flag at it. Every instance below is
# a real registry-valid constraint, every "violation" snippet genuinely trips
# EXACTLY its own checker (test_fixture_rig_is_genuine proves both halves), and
# every checker_results block in the rig is produced by calling the real
# checker. A sabotage that fabricates a verdict is therefore fabricating it
# against ground truth, which is the only way this suite can prove the gate.
INSTANCES = [
    {"id": "c-lex-001", "type": "banned_words", "family": "lexical",
     "split": "train", "params": {"words": ["coffee", "espresso", "latte"]}},
    {"id": "c-lex-002", "type": "banned_words", "family": "lexical",
     "split": "train", "params": {"words": ["quantum", "entanglement"]}},
    {"id": "c-lex-003", "type": "banned_phrases", "family": "lexical",
     "split": "train", "params": {"phrases": ["in conclusion", "as an AI"]}},
    {"id": "c-fmt-001", "type": "no_lists", "family": "format_negative",
     "split": "train", "params": {}},
    {"id": "c-fmt-002", "type": "no_quotation_marks", "family": "format_negative",
     "split": "train", "params": {}},
    {"id": "c-chr-001", "type": "banned_punctuation", "family": "character",
     "split": "heldout_type", "params": {"marks": [";"]}},
]
INSTANCE_BY_ID = {c["id"]: c for c in INSTANCES}
INSTANCE_IDS = [c["id"] for c in INSTANCES]

#: id -> text that genuinely violates that constraint and no other one here.
VIOLATION = {
    "c-lex-001": " The roastery served fresh coffee to everyone.",
    "c-lex-002": " A quantum result arrived overnight.",
    "c-lex-003": " In conclusion the schedule holds.",
    "c-fmt-001": "\n- first point\n- second point\n",
    "c-fmt-002": ' The chair said "ready" and sat down.',
    "c-chr-001": " It worked; it held.",
}


def compliant_output(scenario_id):
    """Prose that genuinely satisfies every constraint in INSTANCES."""
    return (f"Response for {scenario_id}. The team wrote a plain paragraph "
            f"about the topic and kept every sentence ordinary.")


# --------------------------------------------------------------- fixture rig
def stem(model, strategy):
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in model)
    return f"{safe}__{strategy}"


def write_instances(path, instances=INSTANCES):
    path.write_text("".join(json.dumps(c) + "\n" for c in instances))
    return path


def make_eval_set(path, n=N, k_values=(2,), adversarial_share=0.40):
    """A schema-valid v2 scenario corpus. k cycles through k_values so the
    highest k the rung claims is actually present. Constraint ids rotate through
    the real instance library so each scenario names k DISTINCT real ones."""
    n_adv = math.ceil(n * adversarial_share)
    scenarios = []
    for i in range(n):
        k = k_values[i % len(k_values)]
        # guarantee max(k_values) appears even when n % len(k_values) skips it
        if i == n - 1:
            k = max(k_values)
        cons = [INSTANCE_IDS[(i + j) % len(INSTANCE_IDS)] for j in range(k)]
        assert len(set(cons)) == k, "a scenario must not repeat a constraint id"
        scenarios.append({
            "scenario_id": f"s-{i:04d}",
            "task": f"Write a short paragraph about topic {i}.",
            "constraints": cons,
            "k": k,
            "pressure": "adversarial" if i < n_adv else "clean",
            "split": "train",
        })
    path.write_text("".join(json.dumps(s) + "\n" for s in scenarios))
    return scenarios


def score_output(scenario, output):
    """checker_results the way the harness makes them: by running the checker."""
    results = []
    for cid in scenario["constraints"]:
        verdict = real_check(INSTANCE_BY_ID[cid], output)
        results.append({"constraint_id": cid, "pass": bool(verdict["pass"]),
                        "evidence": str(verdict["evidence"])})
    return results


def make_row(scenario, model, strategy, passing, quality=4):
    """One transcript row in the pinned format.

    `passing` drives a REAL constraint violation in the model output, and the
    checker_results are then produced by the real checker — never asserted.
    """
    output = compliant_output(scenario["scenario_id"])
    if not passing:
        output += VIOLATION[scenario["constraints"][0]]
    results = score_output(scenario, output)
    assert all(r["pass"] for r in results) is passing, (
        f"fixture is not honest: passing={passing} but the real checkers say "
        f"{[(r['constraint_id'], r['pass']) for r in results]}")
    spec_pass = all(r["pass"] for r in results) and quality >= 3
    return {
        "scenario_id": scenario["scenario_id"],
        "model": model,
        "strategy": strategy,
        "output": output,
        "checker_results": results,
        "quality_score": quality,
        "quality_reasoning": "Addresses the task with a coherent paragraph.",
        "spec_pass": spec_pass,
    }


def write_cell(run_dir, scenarios, model, strategy, n_fail):
    """Write one model x strategy cell: transcripts JSONL + results JSON whose
    stored aggregate genuinely matches the rows."""
    rows = [make_row(s, model, strategy, passing=(i >= n_fail))
            for i, s in enumerate(scenarios)]
    base = stem(model, strategy)
    tp = run_dir / f"{base}.transcripts.jsonl"
    tp.write_text("".join(json.dumps(r) + "\n" for r in rows))

    n = len(rows)
    n_pass = sum(1 for r in rows if r["spec_pass"])
    adv = [(r, s) for r, s in zip(rows, scenarios) if s["pressure"] == "adversarial"]
    results = {
        "model": model,
        "strategy": strategy,
        "judge_model": "claude-sonnet-4-6",
        "timestamp_utc": "2026-08-20T07:00:00+00:00",
        "metrics": {
            "n": n,
            "spec_adherence": float(Fraction(n_pass, n)),
            "fail_rate": float(Fraction(n - n_pass, n)),
            "robustness": (float(Fraction(sum(1 for r, _ in adv if r["spec_pass"]),
                                          len(adv))) if adv else None),
        },
    }
    rp = run_dir / f"{base}.results.json"
    rp.write_text(json.dumps(results, indent=2))
    return rp, tp


def build_run(tmp_path, *, best_fail=10, other_fails=(14, 18), base_fail=30,
              rung="r1-k2", k_values=(2,), adversarial_share=0.40, n=N,
              name="run"):
    """A complete, internally consistent G1 run directory + its config.

    `best_fail` is the failure COUNT of the best frontier cell (the minimum over
    frontier cells, since best = highest adherence).
    """
    run_dir = tmp_path / name
    run_dir.mkdir()
    eval_path = tmp_path / f"{name}_scenarios.jsonl"
    instances_path = write_instances(tmp_path / f"{name}_instances.jsonl")
    scenarios = make_eval_set(eval_path, n=n, k_values=k_values,
                              adversarial_share=adversarial_share)

    fails = [best_fail, *other_fails]
    frontier_cells = [(m, s) for m in (FRONTIER_A, FRONTIER_B) for s in STRATEGIES]
    for i, (m, s) in enumerate(frontier_cells):
        write_cell(run_dir, scenarios, m, s, fails[i % len(fails)])
    for s in STRATEGIES:
        write_cell(run_dir, scenarios, LOCAL_BASE, s, base_fail)

    (run_dir / "g1_table.json").write_text(json.dumps({
        "run_id": "g1", "ladder_rung": rung, "eval_set": str(eval_path)}, indent=2))

    cfg_path = tmp_path / f"{name}_g1.json"
    cfg_path.write_text(json.dumps({
        "run_id": "g1", "out_dir": str(run_dir),
        "models": {"frontier": [FRONTIER_A, FRONTIER_B], "local_base": LOCAL_BASE},
        "strategies": list(STRATEGIES),
        "eval_set": str(eval_path),
        "constraint_instances": str(instances_path),
        "judge": {"provider": "anthropic", "model": "claude-sonnet-4-6"},
        "ladder": {"current": rung, "rungs": RUNGS},
    }, indent=2))
    return run_dir, cfg_path, eval_path, scenarios


def run_gate(run_dir, cfg_path, extra=()):
    return G.main([str(run_dir), "--config", str(cfg_path), *extra])


def read_report(run_dir):
    return json.loads((run_dir / "gate_report.json").read_text())


def rewrite_results(run_dir, model, strategy, mutate):
    path = run_dir / f"{stem(model, strategy)}.results.json"
    data = json.loads(path.read_text())
    mutate(data)
    path.write_text(json.dumps(data, indent=2))
    return path


def read_rows(run_dir, model, strategy):
    path = run_dir / f"{stem(model, strategy)}.transcripts.jsonl"
    return path, [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def write_rows(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def restate_aggregate(run_dir, model, strategy, rows, scenarios):
    """Rewrite the stored metrics so they match `rows` exactly.

    Sabotage that leaves the summary contradicting its own rows is caught by the
    aggregate check and proves nothing about the guard under test, so every
    mutation here restates the aggregate first.
    """
    n = len(rows)
    n_pass = sum(1 for r in rows if r["spec_pass"])
    by_id = {s["scenario_id"]: s for s in scenarios}
    adv = [r for r in rows if by_id[r["scenario_id"]]["pressure"] == "adversarial"]
    rewrite_results(run_dir, model, strategy, lambda d: d["metrics"].update(
        n=n, spec_adherence=n_pass / n, fail_rate=(n - n_pass) / n,
        robustness=(sum(1 for r in adv if r["spec_pass"]) / len(adv)) if adv else None))


def forge_failures(run_dir, model, strategy, scenarios, n_fail):
    """Claim `n_fail` checker failures on rows whose outputs genuinely comply.

    Nothing about the row is left inconsistent: the failing checker keeps a
    plausible evidence string, spec_pass follows the pinned definition from the
    forged verdicts, and the stored aggregate is restated to match. Only
    re-running the checker over (constraint, output) can see it.
    """
    path, rows = read_rows(run_dir, model, strategy)
    forged = 0
    for r in rows:
        if forged >= n_fail:
            break
        if not r["spec_pass"]:
            continue
        r["checker_results"][0] = {
            "constraint_id": r["checker_results"][0]["constraint_id"],
            "pass": False,
            "evidence": "banned word 'coffee' at offset 12: ...served coffee...",
        }
        r["spec_pass"] = False
        forged += 1
    assert forged == n_fail, "the fixture must have enough passing rows to forge"
    write_rows(path, rows)
    restate_aggregate(run_dir, model, strategy, rows, scenarios)


def demote_quality(run_dir, model, strategy, scenarios, n_demote, score=2):
    """Drop the judge's score below 3 on rows whose checkers genuinely pass.

    This is the ONE failure the gate cannot reproduce (a judge call cannot be
    replayed offline), so it is a legitimate row shape — the point of the test
    is that the report says out loud how many failures rest on it.
    """
    path, rows = read_rows(run_dir, model, strategy)
    done = 0
    for r in rows:
        if done >= n_demote:
            break
        if not r["spec_pass"]:
            continue
        r["quality_score"] = score
        r["spec_pass"] = False
        done += 1
    assert done == n_demote
    write_rows(path, rows)
    restate_aggregate(run_dir, model, strategy, rows, scenarios)


# ------------------------------------------------------------- happy path
def test_well_formed_run_at_fail_rate_025_proceeds_and_exits_zero(tmp_path, capsys):
    run_dir, cfg, _, _ = build_run(tmp_path, best_fail=10)  # 10/40 = 0.25
    assert run_gate(run_dir, cfg) == G.EXIT_PROCEED

    report = read_report(run_dir)
    assert report["verdict"] == "PROCEED"
    assert report["gate_fail_rate"] == pytest.approx(0.25)
    assert report["gate_spec_adherence"] == pytest.approx(0.75)
    assert report["best_frontier_cell"]["model"] in (FRONTIER_A, FRONTIER_B)
    assert report["thresholds"]["proceed_fail_rate_min"] == 0.20

    out = capsys.readouterr().out.strip().splitlines()
    assert out[-1].startswith("PROCEED"), "the verdict must be readable in one line"
    assert "fail_rate 0.2500" in out[-1]


def test_report_records_the_frontier_vs_local_base_classification(tmp_path):
    run_dir, cfg, _, _ = build_run(tmp_path, best_fail=10)
    run_gate(run_dir, cfg)
    report = read_report(run_dir)

    cls = report["classification"]
    assert cls["n_frontier_cells"] == 6
    assert cls["n_local_base_cells"] == 3
    assert cls["frontier_providers"] == ["google", "openai"]
    assert cls["local_base_providers"] == ["hf"]

    by_cell = {(c["model"], c["strategy"]): c for c in report["cells"]}
    assert len(by_cell) == 9
    for s in STRATEGIES:
        assert by_cell[(LOCAL_BASE, s)]["classification"] == "local_base"
        assert by_cell[(LOCAL_BASE, s)]["counted_in_gate"] is False
        assert by_cell[(FRONTIER_A, s)]["classification"] == "frontier"
        assert by_cell[(FRONTIER_A, s)]["counted_in_gate"] is True
    # every cell carries its own fail_rate, not just the gate's
    assert all(isinstance(c["fail_rate"], float) for c in report["cells"])


def test_local_base_row_cannot_become_the_ceiling(tmp_path):
    """A perfect local base row must not raise the ceiling and block the gate."""
    run_dir, cfg, _, _ = build_run(tmp_path, best_fail=10, base_fail=0)
    assert run_gate(run_dir, cfg) == G.EXIT_PROCEED
    report = read_report(run_dir)
    assert report["best_frontier_cell"]["model"] != LOCAL_BASE
    assert report["gate_fail_rate"] == pytest.approx(0.25)


def test_report_names_the_ladder_rung_that_produced_the_run(tmp_path):
    run_dir, cfg, _, _ = build_run(tmp_path, best_fail=10, rung="r2-k3",
                                   k_values=(2, 3))
    assert run_gate(run_dir, cfg) == G.EXIT_PROCEED
    ladder = read_report(run_dir)["ladder"]
    assert ladder["rung"] == "r2-k3"
    assert ladder["rung_source"] == "run_manifest"
    assert (ladder["rung_index"], ladder["n_rungs"]) == (2, 5)
    assert ladder["exhausted"] is False
    assert ladder["next_rung"] == "r3-k4"
    assert ladder["observed_in_eval_set"]["k_values"] == [2, 3]


# --------------------------------------------------------------- verdicts
def test_fail_rate_005_escalates_and_exits_non_zero(tmp_path, capsys):
    run_dir, cfg, _, _ = build_run(tmp_path, best_fail=2)  # 2/40 = 0.05
    code = run_gate(run_dir, cfg)
    assert code == G.EXIT_ESCALATE
    assert code != 0

    report = read_report(run_dir)
    assert report["verdict"] == "ESCALATE"
    assert report["gate_fail_rate"] == pytest.approx(0.05)
    assert report["ladder"]["next_rung"] == "r2-k3"
    assert capsys.readouterr().out.strip().splitlines()[-1].startswith("ESCALATE")


def test_best_cell_adherence_095_after_the_full_ladder_hard_stops(tmp_path, capsys):
    run_dir, cfg, _, _ = build_run(tmp_path, best_fail=2, rung=LAST_RUNG,
                                   k_values=(2, 3, 4), adversarial_share=0.80)
    code = run_gate(run_dir, cfg)
    assert code == G.EXIT_HARD_STOP
    assert code != 0

    report = read_report(run_dir)
    assert report["verdict"] == "HARD_STOP"
    assert report["gate_spec_adherence"] == pytest.approx(0.95)
    assert report["ladder"]["exhausted"] is True
    assert "0.90" in report["verdict_reason"]
    assert capsys.readouterr().out.strip().splitlines()[-1].startswith("HARD_STOP")


def test_adherence_095_before_the_ladder_is_exhausted_escalates_not_hard_stops(tmp_path):
    """R6's hard stop is 'adherence >= 0.90 AFTER the full ladder'. On rung 1 the
    remedy is to climb, not to stop — stated here so the reading is testable."""
    run_dir, cfg, _, _ = build_run(tmp_path, best_fail=2, rung="r1-k2")
    assert run_gate(run_dir, cfg) == G.EXIT_ESCALATE
    assert read_report(run_dir)["verdict"] == "ESCALATE"


def test_exhausted_ladder_below_the_threshold_but_under_090_also_hard_stops(tmp_path):
    """adherence 0.85: no rung left, and R6 permits PROCEED only at >= 0.20."""
    run_dir, cfg, _, _ = build_run(tmp_path, best_fail=6, rung=LAST_RUNG,
                                   k_values=(2, 3, 4), adversarial_share=0.80)
    assert run_gate(run_dir, cfg) == G.EXIT_HARD_STOP
    report = read_report(run_dir)
    assert report["verdict"] == "HARD_STOP"
    assert report["gate_spec_adherence"] == pytest.approx(0.85)


# -------------------------------------------------------------- boundaries
def test_exactly_020_falls_on_the_proceed_side(tmp_path):
    """R6 says 'fail-rate >= 0.20', so the boundary is INCLUSIVE: 8/40 PROCEEDs.
    Computed as an exact Fraction — 1 - 0.8 in float is 0.19999999999999996 and
    would silently flip this run to ESCALATE."""
    run_dir, cfg, _, _ = build_run(tmp_path, best_fail=8)
    assert run_gate(run_dir, cfg) == G.EXIT_PROCEED
    report = read_report(run_dir)
    assert report["gate_fail_rate"] == pytest.approx(0.20)
    assert report["verdict"] == "PROCEED"
    assert report["thresholds"]["proceed_boundary"].startswith("inclusive")


def test_one_scenario_below_the_boundary_escalates(tmp_path):
    run_dir, cfg, _, _ = build_run(tmp_path, best_fail=7)  # 7/40 = 0.175
    assert run_gate(run_dir, cfg) == G.EXIT_ESCALATE


def test_decide_is_exact_at_both_thresholds():
    f = Fraction
    assert G.decide(f("0.20"), f("0.80"), False)[0] == "PROCEED"
    assert G.decide(f(24, 120), f(96, 120), False)[0] == "PROCEED"   # float trap
    assert G.decide(f("0.1999"), f("0.8001"), False)[0] == "ESCALATE"
    assert G.decide(f("0.10"), f("0.90"), True)[0] == "HARD_STOP"
    assert G.decide(f("0.1001"), f("0.8999"), True)[0] == "HARD_STOP"
    assert G.decide(f("0.20"), f("0.80"), True)[0] == "PROCEED"


# ---------------------------------------------------------- sabotage table
def sab_inflate_stored_aggregate(run_dir, best, **_):
    """Authentic rows, a hand-edited summary."""
    rewrite_results(run_dir, *best,
                    lambda d: d["metrics"].update(spec_adherence=0.99))


def sab_deflate_stored_fail_rate(run_dir, best, **_):
    rewrite_results(run_dir, *best, lambda d: d["metrics"].update(fail_rate=0.42))


def sab_stored_n_lies(run_dir, best, **_):
    rewrite_results(run_dir, *best, lambda d: d["metrics"].update(n=999))


def sab_duplicate_row(run_dir, best, **_):
    """A passing row repeated: every row is authentic, the mean is not.

    The stored aggregate is updated to exactly what a duplicate-BLIND recompute
    would produce (40 distinct scenario ids, 31 spec_pass increments), so the
    only guard that can catch this is the duplicate check itself. The duplicated
    row carries pressure="clean", so robustness is untouched.
    """
    path, rows = read_rows(run_dir, *best)
    dup = rows[-1]
    assert dup["spec_pass"], "the fixture must duplicate a PASSING row"
    write_rows(path, rows + [dup])
    n = len(rows)
    n_pass = sum(1 for r in rows if r["spec_pass"]) + 1
    rewrite_results(run_dir, *best, lambda d: d["metrics"].update(
        n=n, spec_adherence=n_pass / n, fail_rate=(n - n_pass) / n))


def sab_drop_failing_rows(run_dir, best, **_):
    """A strict subset of correctly-scored rows, re-weighted upward."""
    path, rows = read_rows(run_dir, *best)
    kept = [r for r in rows if r["spec_pass"]]
    write_rows(path, kept)
    n = len(kept)
    rewrite_results(run_dir, *best,
                    lambda d: d["metrics"].update(n=n, spec_adherence=1.0,
                                                  fail_rate=0.0, robustness=1.0))


def sab_row_for_unknown_scenario(run_dir, best, **_):
    path, rows = read_rows(run_dir, *best)
    ghost = dict(rows[-1], scenario_id="s-9999")
    write_rows(path, rows + [ghost])


def sab_missing_strategy_field(run_dir, best, **_):
    rewrite_results(run_dir, *best, lambda d: d.pop("strategy"))


def sab_unpinned_model(run_dir, best, **_):
    """A coherent, self-consistent cell from a model configs/g1.json never pinned
    — rows relabelled too, so only the pinned-model check sees it as wrong."""
    rogue = "openai:gpt-4o-mini"
    path, rows = read_rows(run_dir, *best)
    for r in rows:
        r["model"] = rogue
    write_rows(path, rows)
    rewrite_results(run_dir, *best, lambda d: d.update(model=rogue))


def sab_unknown_provider(run_dir, best, **_):
    rewrite_results(run_dir, *best, lambda d: d.update(model="mock:always_pass"))


def sab_row_model_mismatch(run_dir, best, **_):
    path, rows = read_rows(run_dir, *best)
    rows[3]["model"] = FRONTIER_B if best[0] == FRONTIER_A else FRONTIER_A
    write_rows(path, rows)


def sab_bad_strategy_name(run_dir, best, **_):
    rewrite_results(run_dir, *best, lambda d: d.update(strategy="chain_of_thought"))


def sab_missing_cell(run_dir, best, **_):
    (run_dir / f"{stem(FRONTIER_B, 'structured')}.results.json").unlink()
    (run_dir / f"{stem(FRONTIER_B, 'structured')}.transcripts.jsonl").unlink()


def sab_missing_transcripts(run_dir, best, **_):
    (run_dir / f"{stem(*best)}.transcripts.jsonl").unlink()


def sab_spec_pass_true_over_failing_checker(run_dir, best, **_):
    path, rows = read_rows(run_dir, *best)
    for r in rows:
        if not r["spec_pass"]:
            r["spec_pass"] = True
    write_rows(path, rows)
    rewrite_results(run_dir, *best,
                    lambda d: d["metrics"].update(spec_adherence=1.0, fail_rate=0.0,
                                                  robustness=1.0))


def sab_drop_the_failing_checker_result(run_dir, best, **_):
    """Delete the one failing checker so the row 'legitimately' passes."""
    path, rows = read_rows(run_dir, *best)
    for r in rows:
        if not r["spec_pass"]:
            r["checker_results"] = [c for c in r["checker_results"] if c["pass"]]
            r["spec_pass"] = True
    write_rows(path, rows)
    rewrite_results(run_dir, *best,
                    lambda d: d["metrics"].update(spec_adherence=1.0, fail_rate=0.0,
                                                  robustness=1.0))


def sab_quality_below_threshold_still_passes(run_dir, best, **_):
    path, rows = read_rows(run_dir, *best)
    for r in rows:
        if r["spec_pass"]:
            r["quality_score"] = 2      # < 3, so spec_pass must be False
    write_rows(path, rows)


def sab_empty_evidence(run_dir, best, **_):
    path, rows = read_rows(run_dir, *best)
    rows[0]["checker_results"][0]["evidence"] = ""
    write_rows(path, rows)


def sab_no_ladder_rung(run_dir, best, **_):
    p = run_dir / "g1_table.json"
    d = json.loads(p.read_text())
    d.pop("ladder_rung")
    p.write_text(json.dumps(d))


def sab_unknown_ladder_rung(run_dir, best, **_):
    p = run_dir / "g1_table.json"
    d = json.loads(p.read_text())
    d["ladder_rung"] = "r9-turbo"
    p.write_text(json.dumps(d))


def sab_rung_overstates_k(run_dir, best, **_):
    """Claim the ladder was climbed to k=4 over a k=2 corpus."""
    p = run_dir / "g1_table.json"
    d = json.loads(p.read_text())
    d["ladder_rung"] = "r3-k4"
    p.write_text(json.dumps(d))


def sab_rung_overstates_adversarial_share(run_dir, best, **_):
    p = run_dir / "g1_table.json"
    d = json.loads(p.read_text())
    d["ladder_rung"] = "r4-adv60"   # corpus is 40% adversarial
    p.write_text(json.dumps(d))


def sab_eval_set_deleted(run_dir, best, eval_path=None, **_):
    eval_path.unlink()


def sab_eval_set_shrunk_below_r6_floor(run_dir, best, eval_path=None, **_):
    lines = eval_path.read_text().splitlines()[:20]
    eval_path.write_text("\n".join(lines) + "\n")


# --- BLOCKER 1: forged programmatic verdicts over authentic model outputs -----
def sab_fabricated_checker_failures(run_dir, best, scenarios=None, **_):
    """Rows whose outputs genuinely comply, claiming checker failures anyway.

    Self-consistent at every level the gate used to look at: the verdict, the
    spec_pass derived from it and the stored aggregate all agree with each
    other. They just disagree with the checker.
    """
    forge_failures(run_dir, *best, scenarios, 4)


def sab_forged_evidence_behind_a_true_verdict(run_dir, best, **_):
    """The pass flag is right; the evidence naming the violation is invented."""
    path, rows = read_rows(run_dir, *best)
    for r in rows:
        if r["spec_pass"]:
            continue
        for cr in r["checker_results"]:
            if not cr["pass"]:
                cr["evidence"] = "the model came close to a banned term here"
        break
    write_rows(path, rows)


def sab_output_edited_after_scoring(run_dir, best, **_):
    """A failing row's output is quietly cleaned up, its failing verdict kept.

    The row still 'fails', so the metric is untouched — but the transcript no
    longer shows the violation the gate's evidence rests on, and the archived
    output is what an Architecture Defense would be shown.
    """
    path, rows = read_rows(run_dir, *best)
    for r in rows:
        if not r["spec_pass"]:
            r["output"] = compliant_output(r["scenario_id"])
            break
    write_rows(path, rows)


# --- BLOCKER 2: the run choosing the corpus it is graded against -------------
def sab_manifest_names_a_softer_corpus(run_dir, best, tmp_path=None, **_):
    """A second, easier corpus, named by the run manifest instead of the config."""
    soft = tmp_path / "soft_corpus.jsonl"
    make_eval_set(soft, n=30, k_values=(2,), adversarial_share=0.40)
    p = run_dir / "g1_table.json"
    d = json.loads(p.read_text())
    d["eval_set"] = str(soft)
    p.write_text(json.dumps(d))


def sab_manifest_names_a_softer_instance_library(run_dir, best, tmp_path=None, **_):
    """Same trick one level down: laxer constraint objects behind the same ids."""
    lax = write_instances(tmp_path / "lax_instances.jsonl", [
        dict(c, params={"words": ["zzzzznotawordzzzzz"]})
        if c["type"] == "banned_words" else c for c in INSTANCES])
    p = run_dir / "g1_table.json"
    d = json.loads(p.read_text())
    d["constraint_instances"] = str(lax)
    p.write_text(json.dumps(d))


def sab_transcripts_declared_outside_the_run(run_dir, best, tmp_path=None, **_):
    elsewhere = tmp_path / "elsewhere.transcripts.jsonl"
    _, rows = read_rows(run_dir, *best)
    write_rows(elsewhere, [r for r in rows if r["spec_pass"]])
    rewrite_results(run_dir, *best,
                    lambda d: d.update(transcripts=f"../{elsewhere.name}"))


def sab_scenario_names_an_unresolvable_constraint(run_dir, best, eval_path=None, **_):
    """A constraint id no instance library holds: its checker cannot be re-run."""
    lines = [json.loads(ln) for ln in eval_path.read_text().splitlines() if ln.strip()]
    lines[0]["constraints"][0] = "c-lex-987"
    eval_path.write_text("".join(json.dumps(s) + "\n" for s in lines))



# ------------------------------- substituting the file a cell is scored against
def donor_run(tmp_path, scenarios, n_fail=12, name="donor"):
    """A second, REAL run outside the run under test.

    Same scenarios, same cells, same schema — but its model outputs genuinely
    violate their constraints, so its rows survive every content check the gate
    makes. It is the file a path substitution is trying to steal, and the reason
    a substitution has to be caught by PATH RESOLUTION: by the time the rows are
    read, there is nothing wrong with them.
    """
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    for m in (FRONTIER_A, FRONTIER_B):
        for s in STRATEGIES:
            write_cell(d, scenarios, m, s, n_fail)
    return d


def substitute_transcripts(run_dir, model, strategy, donor, scenarios, how):
    """Point one cell at the donor's rows by `how`, and restate its aggregate.

    Every mechanism here keeps the run looking untouched from inside: the file
    the results JSON names sits in the run directory, and the stored metrics
    match the rows that will actually be read.
    """
    base = stem(model, strategy)
    src = donor / f"{base}.transcripts.jsonl"
    tp = run_dir / f"{base}.transcripts.jsonl"
    if how == "symlink_file":            # the conventional name IS the symlink
        tp.unlink()
        tp.symlink_to(src)
    elif how == "hardlink":              # a second name for the donor's bytes
        tp.unlink()
        os.link(src, tp)
    elif how == "declared_symlink":      # an innocuous in-run name in the key
        link = run_dir / f"{base}.audit.jsonl"
        link.symlink_to(src)
        rewrite_results(run_dir, model, strategy,
                        lambda d: d.update(transcripts=link.name))
    elif how == "symlinked_dir":         # relative, no "..", still leaves the run
        raw = run_dir / "raw"
        if not raw.exists():
            raw.symlink_to(donor, target_is_directory=True)
        rewrite_results(run_dir, model, strategy, lambda d: d.update(
            transcripts=f"raw/{base}.transcripts.jsonl"))
    else:                                 # pragma: no cover - test bug
        raise AssertionError(how)
    rows = [json.loads(ln) for ln in src.read_text().splitlines() if ln.strip()]
    restate_aggregate(run_dir, model, strategy, rows, scenarios)


SUBSTITUTIONS = ("symlink_file", "declared_symlink", "symlinked_dir", "hardlink")


def sab_transcripts_symlinked_to_a_donor_file(run_dir, best, tmp_path=None,
                                              scenarios=None, **_):
    substitute_transcripts(run_dir, *best, donor_run(tmp_path, scenarios), scenarios,
                           "symlink_file")


def sab_transcripts_declared_through_a_symlinked_dir(run_dir, best, tmp_path=None,
                                                     scenarios=None, **_):
    substitute_transcripts(run_dir, *best, donor_run(tmp_path, scenarios), scenarios,
                           "symlinked_dir")


def sab_transcripts_hardlinked_to_a_donor_file(run_dir, best, tmp_path=None,
                                               scenarios=None, **_):
    substitute_transcripts(run_dir, *best, donor_run(tmp_path, scenarios), scenarios,
                           "hardlink")


def sab_results_json_symlinked_out_of_the_run(run_dir, best, tmp_path=None,
                                              scenarios=None, **_):
    """The summary itself comes from another run; only its NAME is in this one."""
    donor = donor_run(tmp_path, scenarios)
    base = stem(*best)
    rp = run_dir / f"{base}.results.json"
    rp.unlink()
    rp.symlink_to(donor / f"{base}.results.json")


def sab_manifest_symlinked_out_of_the_run(run_dir, best, tmp_path=None, **_):
    """The manifest sets the ladder rung, so it decides ESCALATE vs HARD_STOP."""
    foreign = tmp_path / "foreign_g1_table.json"
    foreign.write_text(json.dumps({"run_id": "g1", "ladder_rung": LAST_RUNG}))
    p = run_dir / "g1_table.json"
    p.unlink()
    p.symlink_to(foreign)


def sab_two_cells_scoring_one_transcript_file(run_dir, best, **_):
    """One row set counted as two cells, via a second cell's transcripts key."""
    other = run_dir / f"{stem(FRONTIER_A, 'few_shot')}.transcripts.jsonl"
    rewrite_results(run_dir, *best, lambda d: d.update(transcripts=other.name))


SABOTAGE = [
    # (label, mutation, a substring of the error THIS guard must produce)
    ("stored aggregate inflated above its own rows", sab_inflate_stored_aggregate,
     "disagrees with its own rows"),
    ("stored fail_rate contradicts its own rows", sab_deflate_stored_fail_rate,
     "metrics.fail_rate"),
    ("stored n contradicts the transcript", sab_stored_n_lies, "metrics.n"),
    ("a duplicated authentic row re-weights the mean", sab_duplicate_row,
     "duplicate row for scenario"),
    ("failing rows dropped: a re-weighted strict subset", sab_drop_failing_rows,
     "strict subset re-weights"),
    ("a row for a scenario outside the eval set", sab_row_for_unknown_scenario,
     "is not in the eval set"),
    ("results JSON missing its strategy field", sab_missing_strategy_field,
     "model/strategy must be non-empty strings"),
    ("a cell from a model the config does not pin", sab_unpinned_model,
     "is not one of the models"),
    ("a cell from an unclassifiable provider", sab_unknown_provider,
     "neither a frontier family"),
    ("a transcript row whose model is not its cell's", sab_row_model_mismatch,
     "cell's model"),
    ("a strategy outside the pinned three", sab_bad_strategy_name,
     "'chain_of_thought' is not one of"),
    ("an incomplete grid", sab_missing_cell, "grid incomplete"),
    ("a cell shipping only its summary", sab_missing_transcripts,
     "ships only its summary"),
    ("spec_pass true over a failing checker", sab_spec_pass_true_over_failing_checker,
     "the pinned definition"),
    ("the failing checker result deleted", sab_drop_the_failing_checker_result,
     "dropped or invented checker result"),
    ("quality below 3 still marked spec_pass", sab_quality_below_threshold_still_passes,
     "the pinned definition"),
    ("a checker result with no evidence", sab_empty_evidence,
     "requires the violating substring"),
    ("no ladder rung recorded anywhere", sab_no_ladder_rung,
     "which escalation-ladder rung"),
    ("a ladder rung the config never declared", sab_unknown_ladder_rung,
     "is not a rung declared"),
    ("a rung claiming a k the corpus never reached", sab_rung_overstates_k,
     "overstates this run"),
    ("a rung claiming more adversarial pressure than the corpus has",
     sab_rung_overstates_adversarial_share, "adversarial share is"),
    ("the eval set missing", sab_eval_set_deleted, "not found"),
    ("an eval set below R6's 30-scenario floor", sab_eval_set_shrunk_below_r6_floor,
     "per model x strategy cell"),
    # BLOCKER 1 — the programmatic half must be re-derived, not just re-read
    ("fabricated checker failures over compliant outputs",
     sab_fabricated_checker_failures, "the stored verdict is fabricated"),
    ("forged evidence behind an otherwise correct verdict",
     sab_forged_evidence_behind_a_true_verdict,
     "not produced by the checker it claims"),
    ("a failing row's output edited after scoring", sab_output_edited_after_scoring,
     "the stored verdict is fabricated"),
    # BLOCKER 2 — the config pins the corpus, the run does not choose it
    ("a manifest naming a softer corpus than the config pins",
     sab_manifest_names_a_softer_corpus, "provenance, not an override"),
    ("a manifest naming a laxer constraint instance library",
     sab_manifest_names_a_softer_instance_library, "provenance, not an override"),
    ("a results JSON pointing at transcripts outside its run",
     sab_transcripts_declared_outside_the_run, "escapes the run directory"),
    ("a scenario naming a constraint no library holds",
     sab_scenario_names_an_unresolvable_constraint,
     "not in the pinned instance library"),
    # THE PATH THE REPAIR OPENED — the file scored must be the file shipped
    ("the conventional transcripts file replaced by a symlink out of the run",
     sab_transcripts_symlinked_to_a_donor_file, "escapes the run directory"),
    ("a declared transcripts path that hops through a symlinked directory",
     sab_transcripts_declared_through_a_symlinked_dir, "escapes the run directory"),
    ("transcripts hard-linked to another run's rows",
     sab_transcripts_hardlinked_to_a_donor_file, "is a hard link"),
    ("a results JSON that is a symlink to another run's summary",
     sab_results_json_symlinked_out_of_the_run, "escapes the run directory"),
    ("a run manifest symlinked to another run's manifest",
     sab_manifest_symlinked_out_of_the_run, "escapes the run directory"),
    ("one transcripts file scored as two cells",
     sab_two_cells_scoring_one_transcript_file, "cannot be two cells"),
]


@pytest.mark.parametrize("label,sabotage,expected", SABOTAGE,
                         ids=[s[0] for s in SABOTAGE])
def test_sabotaged_run_is_rejected_not_rated(tmp_path, capsys, label, sabotage,
                                             expected):
    """Each entry must be rejected BY ITS OWN GUARD.

    Asserting only on the exit code lets one guard mask another: a mutation
    testing pass showed the duplicate-row and unpinned-model checks could both be
    deleted while this table stayed green. The expected-message assertion binds
    each guard to the sabotage it exists to stop.
    """
    run_dir, cfg, eval_path, scenarios = build_run(tmp_path, best_fail=10)
    assert run_gate(run_dir, cfg) == G.EXIT_PROCEED, "control run must PROCEED first"
    (run_dir / "gate_report.json").unlink()
    capsys.readouterr()

    best = (FRONTIER_A, "zero_shot")   # deterministic best frontier cell above
    sabotage(run_dir, best, eval_path=eval_path, scenarios=scenarios,
             tmp_path=tmp_path)

    code = run_gate(run_dir, cfg)
    assert code == G.EXIT_INVALID, f"{label}: expected rejection, got exit {code}"
    err = capsys.readouterr().err
    assert "FAIL" in err
    assert expected in err, (
        f"{label}: rejected, but not by the guard that should have caught it.\n"
        f"expected {expected!r} in:\n{err}")
    assert not (run_dir / "gate_report.json").exists(), (
        f"{label}: a rejected run must not leave a verdict behind")


def test_a_stale_report_from_an_earlier_run_is_called_out(tmp_path, capsys):
    run_dir, cfg, _, _ = build_run(tmp_path, best_fail=10)
    assert run_gate(run_dir, cfg) == G.EXIT_PROCEED
    capsys.readouterr()

    sab_inflate_stored_aggregate(run_dir, (FRONTIER_A, "zero_shot"))
    assert run_gate(run_dir, cfg) == G.EXIT_INVALID
    err = capsys.readouterr().err
    assert "STALE" in err, "a leftover PROCEED report must be flagged, not trusted"


# ------------------------- the headline: a genuinely compliant run cannot PROCEED
def test_a_fully_compliant_run_cannot_fabricate_its_way_to_a_proceed(tmp_path, capsys):
    """THE attack this gate exists to stop.

    Every model output in this run genuinely satisfies every constraint — the
    checkers say so, and the ceiling claim R6 is about is therefore FALSE for
    this corpus. The run then writes itself a 30% fail rate: forged checker
    verdicts, spec_pass derived correctly FROM the forgery, aggregates restated
    to match, coverage intact, every row type-clean. Before the gate re-derived
    checker_results from (constraint, output) this produced a PROCEED, which
    would have carried a false ceiling into the Architecture Defense.
    """
    run_dir, cfg, _, scenarios = build_run(tmp_path, best_fail=0, other_fails=(0, 0),
                                           base_fail=0)

    # control: the truth about this run
    assert run_gate(run_dir, cfg) == G.EXIT_ESCALATE
    control = read_report(run_dir)
    assert control["gate_fail_rate"] == 0.0
    assert control["gate_spec_adherence"] == 1.0
    assert control["best_frontier_cell"]["n_fail_from_checkers"] == 0
    (run_dir / "gate_report.json").unlink()
    capsys.readouterr()

    # the forgery: 12/40 = 0.30 fail rate in every frontier cell
    for model in (FRONTIER_A, FRONTIER_B):
        for strategy in STRATEGIES:
            forge_failures(run_dir, model, strategy, scenarios, 12)

    code = run_gate(run_dir, cfg)
    err = capsys.readouterr().err
    assert code == G.EXIT_INVALID, (
        "a run whose outputs genuinely comply must never be rated PROCEED on "
        "failures it invented")
    assert "the stored verdict is fabricated" in err
    assert not (run_dir / "gate_report.json").exists()


@pytest.mark.parametrize("how", SUBSTITUTIONS)
def test_a_fully_compliant_run_cannot_substitute_its_way_to_a_proceed(tmp_path, capsys,
                                                                     how):
    """THE second way to fake a ceiling: keep honest rows, score somebody else's.

    Every model output in this run genuinely satisfies every constraint, so R6's
    ceiling claim is FALSE for it — the honest verdict is ESCALATE at fail_rate
    0.0000. The run then points each cell at a DONOR run's rows (30% genuine
    failures) without touching anything the gate looks at inside a row: the
    donor's verdicts are real, its coverage is exact, and the stored aggregates
    are restated to match. Only resolving the path can see it.

    Every mechanism here defeated the previous check, which tested a declared
    path lexically (`is_absolute()` / `".." in parts`) and nothing else: a
    symlink, a symlink named in the transcripts key, a relative path through a
    symlinked directory, and a hard link all satisfy it. Live, through the real
    CLI, they turned `ESCALATE, adherence 1.0000` into `exit 0, PROCEED,
    fail_rate 0.3000`, with the report recording the innocuous in-run filename.
    """
    run_dir, cfg, _, scenarios = build_run(tmp_path, best_fail=0, other_fails=(0, 0),
                                           base_fail=0)
    assert run_gate(run_dir, cfg) == G.EXIT_ESCALATE, "control: the truth is 0.0000"
    control = read_report(run_dir)
    assert control["gate_fail_rate"] == 0.0
    (run_dir / "gate_report.json").unlink()
    capsys.readouterr()

    donor = donor_run(tmp_path, scenarios, n_fail=12)
    for model in (FRONTIER_A, FRONTIER_B):
        for strategy in STRATEGIES:
            substitute_transcripts(run_dir, model, strategy, donor, scenarios, how)

    code = run_gate(run_dir, cfg)
    err = capsys.readouterr().err
    assert code == G.EXIT_INVALID, (
        f"{how}: a run whose outputs genuinely comply was rated {code} on another "
        f"run's rows")
    assert "escapes the run directory" in err or "is a hard link" in err
    assert not (run_dir / "gate_report.json").exists()


def test_the_report_names_the_file_actually_scored_not_the_name_it_was_given(tmp_path):
    """The audit trail must survive a reader who checks it.

    A path the gate accepted still deserves to be recorded as what it resolved
    to, with a content hash, so 'this cell was scored against these bytes' is
    checkable after the fact instead of resting on a filename.
    """
    run_dir, cfg, _, _ = build_run(tmp_path, best_fail=10)
    assert run_gate(run_dir, cfg) == G.EXIT_PROCEED
    report = read_report(run_dir)

    for cell in report["cells"]:
        tp = Path(cell["transcripts_jsonl_path"])
        rp = Path(cell["results_json_path"])
        assert tp.is_absolute() and rp.is_absolute()
        assert tp == tp.resolve() and rp == rp.resolve(), "a recorded path is resolved"
        assert tp.parent == run_dir.resolve(), "the file scored lives in the run"
        assert cell["transcripts_jsonl_sha256"] == \
            hashlib.sha256(tp.read_bytes()).hexdigest()
        assert cell["results_json_sha256"] == \
            hashlib.sha256(rp.read_bytes()).hexdigest()

    # and the hash tracks the bytes, rather than being decoration
    best = next(c for c in report["cells"] if c["model"] == FRONTIER_A
                and c["strategy"] == "zero_shot")
    path, rows = read_rows(run_dir, FRONTIER_A, "zero_shot")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == \
        best["transcripts_jsonl_sha256"]


def test_a_run_directory_reached_through_a_symlink_is_still_rated(tmp_path):
    """Containment must not become 'no symlink anywhere'.

    The run dir itself is resolved first, so pointing the CLI at an alias for it
    is ordinary usage — every file inside is still contained by the real dir.
    """
    run_dir, cfg, _, _ = build_run(tmp_path, best_fail=10)
    alias = tmp_path / "alias_run"
    alias.symlink_to(run_dir, target_is_directory=True)

    assert run_gate(alias, cfg) == G.EXIT_PROCEED
    report = read_report(run_dir)
    assert report["run_dir"] == str(run_dir.resolve())
    assert report["gate_fail_rate"] == pytest.approx(0.25)


def test_a_repo_relative_pinned_path_may_not_symlink_out_of_the_repo(tmp_path,
                                                                     monkeypatch):
    """The eval set and the instance library are resolved the same way.

    'Repo-relative by contract' is not true of a checked-in symlink that leaves
    the tree, and the corpus is the other half of every number the gate reports.
    """
    fake_repo = tmp_path / "repo"
    fake_repo.mkdir()
    outside = tmp_path / "outside.jsonl"
    make_eval_set(outside, n=30)
    (fake_repo / "corpus.jsonl").symlink_to(outside)
    honest = fake_repo / "honest.jsonl"
    make_eval_set(honest, n=30)
    monkeypatch.setattr(G, "REPO_ROOT", fake_repo)

    with pytest.raises(G.GateError) as exc:
        G.resolve_pinned_path("corpus.jsonl", "eval set")
    assert "escapes" in str(exc.value) and "repo root" in str(exc.value)

    # the honest pin still resolves, and resolves to a real path
    assert G.resolve_pinned_path("honest.jsonl", "eval set") == honest.resolve()
    # an ABSOLUTE pin is the config naming a location outright: allowed, and
    # recorded resolved rather than second-guessed
    assert G.resolve_pinned_path(str(outside), "eval set") == outside.resolve()


def test_a_single_forged_row_is_enough_to_make_the_run_unratable(tmp_path, capsys):
    """No quorum, no tolerance: one fabricated verdict fails the whole run."""
    run_dir, cfg, _, scenarios = build_run(tmp_path, best_fail=10)
    assert run_gate(run_dir, cfg) == G.EXIT_PROCEED
    (run_dir / "gate_report.json").unlink()
    capsys.readouterr()

    forge_failures(run_dir, FRONTIER_A, "zero_shot", scenarios, 1)
    assert run_gate(run_dir, cfg) == G.EXIT_INVALID
    assert "fabricated" in capsys.readouterr().err


def test_checker_verdicts_are_recomputed_not_reread(tmp_path, capsys):
    """A row can also forge a PASS. Same guard, opposite direction."""
    run_dir, cfg, _, _ = build_run(tmp_path, best_fail=10)
    path, rows = read_rows(run_dir, FRONTIER_A, "zero_shot")
    victim = next(r for r in rows if not r["spec_pass"])
    for cr in victim["checker_results"]:
        cr["pass"] = True
        cr["evidence"] = "clean"
    victim["spec_pass"] = True
    write_rows(path, rows)

    assert run_gate(run_dir, cfg) == G.EXIT_INVALID
    err = capsys.readouterr().err
    assert "fabricated" in err and victim["scenario_id"] in err


@pytest.mark.parametrize("bogus", [12345, None, ["text"], {"text": "x"}, True],
                         ids=["int", "null", "list", "dict", "true"])
def test_an_unusable_output_is_reported_not_crashed_on(tmp_path, capsys, bogus):
    """constraints.checkers.check raises TypeError on a non-string text.

    The gate now feeds every row's output to it, so a row carrying a number
    where its output should be must come back as a NAMED problem in the FAIL
    list. A traceback would abort the run before the other rows were examined
    and would print no verdict at all — indistinguishable, to a script, from a
    crash for an unrelated reason.
    """
    run_dir, cfg, _, _ = build_run(tmp_path, best_fail=10)
    path, rows = read_rows(run_dir, FRONTIER_A, "zero_shot")
    rows[0]["output"] = bogus
    write_rows(path, rows)

    assert run_gate(run_dir, cfg) == G.EXIT_INVALID
    err = capsys.readouterr().err
    assert "output is" in err and "not a string" in err
    assert "Traceback" not in err


def test_the_gate_imports_the_pinned_checker_and_does_not_reimplement_it():
    """The ground-truth layer has exactly one implementation."""
    src = (REPO_ROOT / "scripts" / "gate_check.py").read_text()
    assert "from constraints.checkers import check" in src
    tree = ast.parse(src)
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for banned in ("check_banned_words", "check_banned_letter", "check_no_lists",
                   "check_no_quotation_marks", "check_banned_punctuation"):
        assert banned not in defined, (
            f"gate_check.py defines {banned}; a second copy of a checker is a "
            f"second ground truth")


def test_fixture_rig_is_genuine():
    """The rig's own honesty, checked against the real checkers.

    Every test above rests on 'this output genuinely complies' and 'this snippet
    genuinely violates exactly one constraint'. Both are asserted here so a
    silent drift in constraints/checkers.py breaks the claim loudly rather than
    quietly turning the sabotage table into theatre.
    """
    base = compliant_output("s-0000")
    for c in INSTANCES:
        verdict = real_check(c, base)
        assert verdict["pass"] is True, f"{c['id']} rejects the clean fixture output"
        assert verdict["evidence"] == "clean"
    for cid, snippet in VIOLATION.items():
        text = base + snippet
        tripped = {c["id"] for c in INSTANCES if not real_check(c, text)["pass"]}
        assert tripped == {cid}, (
            f"{cid}'s violation snippet trips {sorted(tripped)}, not exactly "
            f"{{{cid!r}}} — a spillover would make the rig's failure counts wrong")


# ------------------------- the corpus is pinned by the config, not by the run
def test_report_records_the_hash_of_the_corpus_actually_scored(tmp_path):
    run_dir, cfg, eval_path, _ = build_run(tmp_path, best_fail=10)
    assert run_gate(run_dir, cfg) == G.EXIT_PROCEED
    report = read_report(run_dir)

    assert report["eval_set"] == str(eval_path)
    assert report["eval_set_sha256"] == hashlib.sha256(eval_path.read_bytes()).hexdigest()
    assert "config" in report["eval_set_source"] or "pinned" in report["eval_set_source"]

    inst = Path(report["constraint_instances"])
    assert report["constraint_instances_sha256"] == \
        hashlib.sha256(inst.read_bytes()).hexdigest()

    # the hash must track the corpus, not be a constant
    eval_path.write_text(eval_path.read_text() + "\n")
    (run_dir / "gate_report.json").unlink()
    assert run_gate(run_dir, cfg) == G.EXIT_PROCEED
    assert read_report(run_dir)["eval_set_sha256"] != report["eval_set_sha256"]


def test_a_manifest_agreeing_with_the_config_is_fine(tmp_path):
    """Recording provenance is allowed; outranking the config is not."""
    run_dir, cfg, eval_path, _ = build_run(tmp_path, best_fail=10)
    p = run_dir / "g1_table.json"
    d = json.loads(p.read_text())
    assert d["eval_set"] == str(eval_path)
    p.write_text(json.dumps(d))
    assert run_gate(run_dir, cfg) == G.EXIT_PROCEED


def test_a_config_with_no_eval_set_is_an_error_not_a_manifest_lookup(tmp_path, capsys):
    run_dir, cfg, eval_path, _ = build_run(tmp_path, best_fail=10)
    data = json.loads(cfg.read_text())
    data.pop("eval_set")
    cfg.write_text(json.dumps(data))
    # the manifest still names a perfectly good corpus — it must not be used
    assert json.loads((run_dir / "g1_table.json").read_text())["eval_set"] \
        == str(eval_path)

    assert run_gate(run_dir, cfg) == G.EXIT_INVALID
    err = capsys.readouterr().err
    assert "eval_set" in err and "pinned by the config" in err


def test_there_is_no_eval_set_override_flag(tmp_path):
    """--eval-set let the invocation pick the exam. It is gone, not softened."""
    run_dir, cfg, eval_path, _ = build_run(tmp_path, best_fail=10)
    with pytest.raises(SystemExit):
        G.main([str(run_dir), "--config", str(cfg), "--eval-set", str(eval_path)])
    src = (REPO_ROOT / "scripts" / "gate_check.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    assert "eval" not in arg.value, f"{arg.value} re-opens the vector"


def test_a_pinned_path_is_not_searched_for_in_the_cwd(tmp_path, capsys, monkeypatch):
    """resolve_path used to try cwd, then the run dir, then the repo root, so a
    decoy file next to the caller could answer to the pinned name."""
    run_dir, cfg, eval_path, scenarios = build_run(tmp_path, best_fail=10)
    decoy_dir = tmp_path / "decoy"
    decoy_dir.mkdir()
    decoy = decoy_dir / "corpus.jsonl"
    make_eval_set(decoy, n=30)

    data = json.loads(cfg.read_text())
    data["eval_set"] = "corpus.jsonl"        # repo-relative, does not exist there
    cfg.write_text(json.dumps(data))
    monkeypatch.chdir(decoy_dir)

    assert run_gate(run_dir, cfg) == G.EXIT_INVALID
    assert "not found" in capsys.readouterr().err


def test_the_default_constraint_instance_library_is_the_shipped_one():
    """The gate's default is a real repo artifact, and it loads."""
    path = REPO_ROOT / G.DEFAULT_INSTANCES_PATH
    assert path.exists(), (
        f"{G.DEFAULT_INSTANCES_PATH} is the library the gate re-runs checkers "
        f"against; its absence must break the gate, not be worked around")
    lib = G.load_constraint_instances(path)
    assert len(lib) >= 1 and all(cid.startswith("c-") for cid in lib)


# --------------------------------------- what the gate cannot recompute, it says
def test_failures_are_attributed_to_the_checkers_or_to_the_judge(tmp_path):
    """quality_score is the one input the gate cannot replay offline.

    It is trusted (there is no way not to), so the report states how much of the
    fail rate rests on it. A reader can then see at a glance whether a PROCEED
    was earned by constraint violations or by a judge's scores.
    """
    run_dir, cfg, _, scenarios = build_run(tmp_path, best_fail=10)
    assert run_gate(run_dir, cfg) == G.EXIT_PROCEED
    best = read_report(run_dir)["best_frontier_cell"]
    assert best["n_fail_from_checkers"] == 10
    assert best["n_fail_from_quality_only"] == 0

    # now a run whose entire fail rate is judge-attributed
    run2, cfg2, _, scen2 = build_run(tmp_path, best_fail=0, other_fails=(0, 0),
                                     base_fail=0, name="quality_only")
    for model in (FRONTIER_A, FRONTIER_B):
        for strategy in STRATEGIES:
            demote_quality(run2, model, strategy, scen2, 10)
    assert run_gate(run2, cfg2) == G.EXIT_PROCEED   # 0.25 by the pinned definition
    report = read_report(run2)
    best2 = report["best_frontier_cell"]
    assert best2["n_fail_from_checkers"] == 0, (
        "no constraint was violated in this run")
    assert best2["n_fail_from_quality_only"] == 10, (
        "the report must say that every counted failure rests on the judge alone")
    assert "quality_score" in report["definition"]


# ----------------------------------------------------------- type confusion
@pytest.mark.parametrize("bogus", [True, False, "0.25", None, [0.25],
                                   float("nan"), float("inf")],
                         ids=["true", "false", "str", "null", "list", "nan", "inf"])
def test_non_numeric_stored_adherence_never_passes_a_number_check(tmp_path, capsys,
                                                                  bogus):
    run_dir, cfg, _, _ = build_run(tmp_path, best_fail=10)
    rewrite_results(run_dir, FRONTIER_A, "zero_shot",
                    lambda d: d["metrics"].update(spec_adherence=bogus))
    assert run_gate(run_dir, cfg) == G.EXIT_INVALID
    err = capsys.readouterr().err
    assert "spec_adherence" in err and "not a finite" in err


@pytest.mark.parametrize("bogus", [True, "0.25", None], ids=["true", "str", "null"])
def test_non_numeric_stored_fail_rate_never_passes_a_number_check(tmp_path, capsys,
                                                                  bogus):
    run_dir, cfg, _, _ = build_run(tmp_path, best_fail=10)
    rewrite_results(run_dir, FRONTIER_A, "zero_shot",
                    lambda d: d["metrics"].update(fail_rate=bogus))
    assert run_gate(run_dir, cfg) == G.EXIT_INVALID
    assert "fail_rate" in capsys.readouterr().err


@pytest.mark.parametrize("bogus", [1, 0, "true", "false", None, 1.0],
                         ids=["int1", "int0", "strtrue", "strfalse", "null", "float"])
def test_spec_pass_must_be_a_real_boolean(tmp_path, capsys, bogus):
    run_dir, cfg, _, _ = build_run(tmp_path, best_fail=10)
    path, rows = read_rows(run_dir, FRONTIER_A, "zero_shot")
    rows[0]["spec_pass"] = bogus
    write_rows(path, rows)
    assert run_gate(run_dir, cfg) == G.EXIT_INVALID
    assert "spec_pass" in capsys.readouterr().err


@pytest.mark.parametrize("bogus", [1, 0, "true", None], ids=["int1", "int0", "str", "null"])
def test_checker_pass_must_be_a_real_boolean(tmp_path, capsys, bogus):
    run_dir, cfg, _, _ = build_run(tmp_path, best_fail=10)
    path, rows = read_rows(run_dir, FRONTIER_A, "zero_shot")
    rows[0]["checker_results"][0]["pass"] = bogus
    write_rows(path, rows)
    assert run_gate(run_dir, cfg) == G.EXIT_INVALID
    assert "not a JSON boolean" in capsys.readouterr().err


@pytest.mark.parametrize("bogus", [True, "4", 4.0, None, 0, 6],
                         ids=["true", "str", "float", "null", "zero", "six"])
def test_quality_score_must_be_an_integer_in_range(tmp_path, capsys, bogus):
    run_dir, cfg, _, _ = build_run(tmp_path, best_fail=10)
    path, rows = read_rows(run_dir, FRONTIER_A, "zero_shot")
    rows[0]["quality_score"] = bogus
    write_rows(path, rows)
    assert run_gate(run_dir, cfg) == G.EXIT_INVALID
    assert "quality_score" in capsys.readouterr().err


def test_is_real_number_rejects_bools_directly():
    assert G.is_real_number(0.25) and G.is_real_number(1) and G.is_real_number(0)
    for x in (True, False, "1", None, [1], float("nan"), float("inf")):
        assert not G.is_real_number(x), f"{x!r} must not read as a number"
    assert G.is_real_bool(True) and not G.is_real_bool(1)
    assert G.is_real_int(3) and not G.is_real_int(True) and not G.is_real_int(3.0)


# -------------------------------------------------- the gate cannot be softened
def test_no_cli_flag_can_move_the_threshold():
    parser_opts = set()
    src = (REPO_ROOT / "scripts" / "gate_check.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    parser_opts.add(arg.value.lstrip("-").replace("-", "_"))
    banned = ("threshold", "min_fail_rate", "fail_rate", "force", "allow", "override",
              "soften", "proceed", "skip", "no_check", "adherence")
    for opt in parser_opts:
        assert not any(b in opt for b in banned), (
            f"--{opt} looks like a threshold/override flag; the kickoff forbids "
            f"softening the gate")


def test_thresholds_are_module_constants_never_reassigned():
    tree = ast.parse((REPO_ROOT / "scripts" / "gate_check.py").read_text())
    names = ("PROCEED_FAIL_RATE_MIN", "HARD_STOP_ADHERENCE_MIN", "QUALITY_SCORE_MIN",
             "MIN_SCENARIOS_PER_CELL")
    counts = {n: 0 for n in names}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Name) and t.id in counts:
                    counts[t.id] += 1
    for n, c in counts.items():
        assert c == 1, f"{n} is assigned {c} times; a threshold must be a constant"


def test_thresholds_hold_the_values_r6_states():
    assert G.PROCEED_FAIL_RATE_MIN == Fraction(1, 5)
    assert G.HARD_STOP_ADHERENCE_MIN == Fraction(9, 10)
    assert G.QUALITY_SCORE_MIN == 3
    assert G.MIN_SCENARIOS_PER_CELL == 30
    assert G.EXIT_FOR_VERDICT["PROCEED"] == 0
    assert all(v != 0 for k, v in G.EXIT_FOR_VERDICT.items() if k != "PROCEED")


def test_gate_never_reads_the_repos_real_runs_directory(tmp_path):
    """These tests must not be able to touch a live artifact.

    The previous form asserted `not runs/g1.exists() or <report>.stat()`, which
    raised FileNotFoundError whenever runs/g1 existed WITHOUT a gate report --
    the normal state while an ablation is mid-run, since run_ablation.py creates
    the directory long before any gate report exists. It tested a coincidence of
    repo state rather than the guarantee. Compare before/after instead.
    """
    live = REPO_ROOT / "runs" / "g1" / "gate_report.json"
    before = live.stat().st_mtime_ns if live.exists() else None

    run_dir, cfg, _, _ = build_run(tmp_path, best_fail=10)
    assert run_gate(run_dir, cfg) == G.EXIT_PROCEED

    after = live.stat().st_mtime_ns if live.exists() else None
    assert before == after, "the gate wrote into the repo's live runs/g1"
    report = read_report(run_dir)
    assert str(tmp_path) in report["run_dir"]


# ------------------------------------------------------- configs/g1.json
G1_PATH = REPO_ROOT / "configs" / "g1.json"


@pytest.fixture(scope="module")
def g1():
    assert G1_PATH.exists(), "configs/g1.json is a T-4 deliverable and must exist"
    return json.loads(G1_PATH.read_text())


@pytest.fixture(scope="module")
def models_json():
    path = REPO_ROOT / "configs" / "models.json"
    assert path.exists(), "configs/models.json must exist for g1.json to pin from"
    return json.loads(path.read_text())


def test_g1_config_passes_the_gates_own_loader(g1):
    cfg = G.load_config(G1_PATH)      # raises GateError on any contract breach
    assert cfg["out_dir"] == "runs/g1"


def test_g1_models_are_the_ids_pinned_in_models_json(g1, models_json):
    assert g1["models"]["frontier"] == models_json["frontier"], (
        "g1.json invented or drifted from the verified frontier ids")
    assert g1["models"]["local_base"] == models_json["base"]


def test_g1_frontier_is_two_different_families(g1):
    families = [m.split(":", 1)[0] for m in g1["models"]["frontier"]]
    assert len(families) == 2 and len(set(families)) == 2, families


def test_g1_strategies_are_the_pinned_three(g1):
    assert g1["strategies"] == ["zero_shot", "few_shot", "structured"]


def test_g1_judge_is_a_third_family_that_never_scores_itself(g1, models_json):
    assert g1["judge"]["model"] == models_json["judge_model"]
    assert g1["judge"]["provider"] == "anthropic"
    families = {m.split(":", 1)[0] for m in g1["models"]["frontier"]}
    assert "anthropic" not in families


def test_g1_eval_set_and_exemplars_are_jsonl_paths_under_data(g1):
    for key in ("eval_set", "exemplars"):
        p = g1[key]
        assert isinstance(p, str) and p.endswith(".jsonl"), f"{key}={p!r}"
        assert not Path(p).is_absolute(), f"{key} must be repo-relative"
        assert p.startswith("data/"), f"{key}={p!r} must live under data/"
    # Corrected: this previously pinned data/eval/scenarios_v2_hardened.jsonl, the
    # RETIRED abstainer hedge corpus. That file exists, so the assertion passed while
    # pinning a corpus of grounded-abstention scenarios that constraint checkers
    # cannot meaningfully score. tests/test_g1_config.py binds the schema, which is
    # the check that would have caught it.
    assert g1["eval_set"] == "data/scenarios/corpus.jsonl"
    # Existence is deliberately NOT asserted here: both files are other tickets'
    # in-flight deliverables. scripts/gate_check.py FAILS at run time if the eval
    # set is absent (test_sabotaged_run_is_rejected_not_rated covers it), which is
    # where the absence must bite.


def test_g1_google_worker_count_respects_the_measured_25_rpm_cap(g1):
    workers = g1["workers"]
    assert workers["by_provider"]["google"] <= 2, (
        "google:gemini-3.1-pro-preview is capped at 25 rpm; 8 workers hard-429s")
    assert workers["by_provider"]["google"] < workers["default"]
    assert workers["by_provider"]["hf"] == 1
    why = json.dumps({k: v for k, v in workers.items() if k.startswith("_")})
    assert "25" in why and "429" in why, (
        "the per-provider worker cap must carry the reason it exists")


def test_g1_ladder_raises_k_then_adversarial_share(g1):
    rungs = g1["ladder"]["rungs"]
    assert len(rungs) >= 4
    max_k = [max(r["k_values"]) for r in rungs]
    shares = [r["adversarial_share"] for r in rungs]
    assert max_k == sorted(max_k), f"k must be non-decreasing along the ladder: {max_k}"
    assert shares == sorted(shares), f"adversarial share must be non-decreasing: {shares}"
    assert max_k[0] == 2 and max_k[-1] == 4, "R6's ladder is k 2->3->4"
    assert shares[-1] > shares[0], "the ladder must also raise adversarial pressure"
    k_phase = [i for i, m in enumerate(max_k) if m < 4]
    share_phase = [i for i, s in enumerate(shares) if s > shares[0]]
    assert max(k_phase) < min(share_phase), (
        "R6 raises k FIRST, then the adversarial share")
    assert all(min(r["k_values"]) >= 2 for r in rungs), "the design forbids k=1"
    assert g1["ladder"]["current"] in {r["name"] for r in rungs}


def test_g1_config_carries_no_threshold_override(g1):
    blob = json.dumps(g1).lower()
    for marker in ("fail_rate_threshold", "proceed_threshold", "gate_threshold",
                   "min_fail_rate", "force_proceed", "skip_gate"):
        assert marker not in blob, (
            f"configs/g1.json contains {marker!r} — thresholds live in "
            f"scripts/gate_check.py and are not configurable")


def test_g1_config_holds_no_secret_material(g1):
    """Key-SHAPED patterns, anchored on a word boundary: a bare "sk-" substring
    test fires on the English word "task-quality"."""
    blob = json.dumps(g1)
    for pattern in (r"\bsk-ant-", r"\bsk-[A-Za-z0-9]{8}", r"\bhf_[A-Za-z0-9]{8}",
                    r"\bAIza[A-Za-z0-9]{8}", r"API_KEY"):
        assert not re.search(pattern, blob), (
            f"configs/g1.json matches the key pattern {pattern!r}")
