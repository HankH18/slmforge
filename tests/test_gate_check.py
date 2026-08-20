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
import json
import math
import re
from fractions import Fraction
from pathlib import Path

import pytest
from conftest import REPO_ROOT

import scripts.gate_check as G

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


# --------------------------------------------------------------- fixture rig
def stem(model, strategy):
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in model)
    return f"{safe}__{strategy}"


def make_eval_set(path, n=N, k_values=(2,), adversarial_share=0.40):
    """A schema-valid v2 scenario corpus. k cycles through k_values so the
    highest k the rung claims is actually present."""
    n_adv = math.ceil(n * adversarial_share)
    scenarios = []
    for i in range(n):
        k = k_values[i % len(k_values)]
        # guarantee max(k_values) appears even when n % len(k_values) skips it
        if i == n - 1:
            k = max(k_values)
        scenarios.append({
            "scenario_id": f"s-{i:04d}",
            "task": f"Write a short paragraph about topic {i}.",
            "constraints": [f"c-lex-{i:03d}-{j}" for j in range(k)],
            "k": k,
            "pressure": "adversarial" if i < n_adv else "clean",
            "split": "train",
        })
    path.write_text("".join(json.dumps(s) + "\n" for s in scenarios))
    return scenarios


def make_row(scenario, model, strategy, passing):
    """One transcript row in the pinned format. `passing` drives a real checker
    failure, never just the spec_pass flag."""
    cons = scenario["constraints"]
    results = [{"constraint_id": c, "pass": True, "evidence": "clean"} for c in cons]
    if not passing:
        results[0] = {"constraint_id": cons[0], "pass": False,
                      "evidence": "...used the banned word 'coffee' here..."}
    quality = 4
    spec_pass = all(r["pass"] for r in results) and quality >= 3
    return {
        "scenario_id": scenario["scenario_id"],
        "model": model,
        "strategy": strategy,
        "output": f"Response for {scenario['scenario_id']}.",
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
    run_dir, cfg, eval_path, _ = build_run(tmp_path, best_fail=10)
    assert run_gate(run_dir, cfg) == G.EXIT_PROCEED, "control run must PROCEED first"
    (run_dir / "gate_report.json").unlink()
    capsys.readouterr()

    best = (FRONTIER_A, "zero_shot")   # deterministic best frontier cell above
    sabotage(run_dir, best, eval_path=eval_path)

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
    """These tests must not be able to touch a live artifact."""
    run_dir, cfg, _, _ = build_run(tmp_path, best_fail=10)
    assert run_gate(run_dir, cfg) == G.EXIT_PROCEED
    assert not (REPO_ROOT / "runs" / "g1").exists() or \
        (REPO_ROOT / "runs" / "g1" / "gate_report.json").stat().st_mtime > 0
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
    assert g1["eval_set"] == "data/eval/scenarios_v2_hardened.jsonl"
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
