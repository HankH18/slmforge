"""Proof that the phase checkers are GATES, not printers.

For each of scripts/check_ablation.py, scripts/check_mvp.py and
scripts/push_model.py --check: it must exit non-zero on a missing/empty artifact
set, exit non-zero on a subtly broken one, and exit zero only on a well-formed
one. A checker that cannot be made to fail proves nothing.

Network-free: every artifact set here is synthesized on disk from the frozen eval
set with the frozen `eval.write_outputs`/`eval.aggregate`, and no test reaches an
API, a GPU, or the Hugging Face Hub.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

import eval as E
from conftest import REPO_ROOT, SCENARIOS_PATH

SCRIPTS = REPO_ROOT / "scripts"
RUBRIC = "c0adb876c3e6c4c9"
JUDGE = "claude-sonnet-4-6"

TUNED_REPO = "tester/grounded-abstainer-qwen3-1.7b"
TUNED_SHA = "0123456789abcdef0123456789abcdef01234567"
TUNED_PIN = f"{TUNED_REPO}@{TUNED_SHA}"
TUNED_SPEC = f"hf:{TUNED_PIN}"
BASE_SHA = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
BASE_REPO = f"Qwen/Qwen3-1.7B@{BASE_SHA}"
BASE_SPEC = f"hf:{BASE_REPO}"
EVAL_SHA = "89abcdef0123456789abcdef0123456789abcdef"

ABLATION_MODELS = ["openai:gpt-5.6-sol", "google:gemini-3.1-pro-preview",
                   f"hf:Qwen/Qwen3-1.7B@{BASE_SHA}"]
STRATEGIES = ["zero_shot", "few_shot", "structured"]
# per (model index, strategy index) count of answerable scenarios scored as
# entailment failures -- gives each cell a distinct spec_adherence
FAIL_COUNTS = {0: [10, 9, 8], 1: [11, 10, 9], 2: [13, 13, 13]}


def run(script, *args):
    return subprocess.run([sys.executable, str(SCRIPTS / script), *map(str, args)],
                          cwd=str(REPO_ROOT), capture_output=True, text=True)


def out(proc):
    return proc.stdout + proc.stderr


# --------------------------------------------------------------- synthesis
def synth_records(scenarios, model, strategy, n_answerable_fail=0):
    """Plausible transcript rows: refusals on unanswerable, cited answers on
    answerable, with `n_answerable_fail` of the answers failing entailment."""
    records, left = [], n_answerable_fail
    for s in scenarios:
        rec = {"scenario_id": s["id"], "type": s.get("type"),
               "adversarial": bool(s.get("adversarial")),
               "gold_label": s["gold_label"], "raw_output": E.REFUSAL_STRING,
               "pred_class": "refusal", "judge_verdict": None,
               "judge_reasoning": None, "passed": True, "failure": None}
        if s["gold_label"] == "answerable":
            span = s["gold_spans"][0]
            rec.update(raw_output=f'Answer: {s["gold_answer"]}\nCitation: "{span}"',
                       pred_class="answer", judge_verdict=True,
                       judge_reasoning="YES. The cited span states the answer.",
                       passed=True, failure=None)
            if left:
                left -= 1
                rec.update(judge_verdict=False, passed=False,
                           judge_reasoning="NO. The span does not entail the answer.",
                           failure="entailment_fail")
        rec.update(model=model, strategy=strategy)
        records.append(rec)
    return records


def write_cell(out_dir, scenarios, model, strategy, n_answerable_fail):
    records = synth_records(scenarios, model, strategy, n_answerable_fail)
    metrics = E.aggregate(records)
    E.write_outputs(Path(out_dir), model, strategy, records, metrics, JUDGE)
    return metrics


def build_ablation(out_dir, scenarios, models=None, strategies=None,
                   fail_counts=None, gate_report=True):
    """A minimal but well-formed runs/ablation/ that check_ablation.py accepts.

    `gate_report=False` leaves gate_report.json out, which is the state
    run_ablation.py actually leaves behind — scripts/make_gate_report.py is what
    fills it in.
    """
    models = models or ABLATION_MODELS
    strategies = strategies or STRATEGIES
    fail_counts = fail_counts or FAIL_COUNTS
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cells, rows = {}, []
    for mi, model in enumerate(models):
        for si, strategy in enumerate(strategies):
            n_fail = fail_counts.get(mi, [8, 8, 8])[si % 3]
            metrics = write_cell(out_dir, scenarios, model, strategy, n_fail)
            cells[(model, strategy)] = metrics
            rows.append({"model": model, "strategy": strategy, "metrics": metrics})

    (out_dir / "ablation_summary.json").write_text(json.dumps(
        {"rubric_version": E.RUBRIC_VERSION, "rubric_hash": RUBRIC,
         "judge_model": JUDGE, "n_scenarios": len(scenarios),
         "eval_set": str(SCENARIOS_PATH), "timestamp_utc": "2026-08-20T00:00:00+00:00",
         "cells": rows}, indent=2))

    header = ("| model | " + " | ".join(strategies) + " |\n"
              "|---|" + "---|" * len(strategies) + "\n")
    body = ""
    for model in models:
        body += (f"| `{model}` | "
                 + " | ".join(f"{100 * cells[(model, s)]['spec_adherence']:.1f}% / "
                              f"{100 * (cells[(model, s)]['robustness'] or 0):.1f}%"
                              for s in strategies) + " |\n")
    (out_dir / "ablation_table.md").write_text(
        f"# Prompt-Ceiling Ablation — rubric v1.0 (hash `{RUBRIC}`), "
        f"n={len(scenarios)} common scenarios, judge={JUDGE}\n\n"
        + header + body +
        "\n## Failure mode that survives the best prompting attempt\n\n"
        + FAILURE_PARAGRAPH + "\n")

    if gate_report:
        frontier = [m for m in models if m.split(":", 1)[0] in
                    ("openai", "google", "anthropic")] or models
        best = max(cells[(m, s)]["spec_adherence"] for m in frontier for s in strategies)
        rate = 1.0 - best
        (out_dir / "gate_report.json").write_text(json.dumps(
            {"rubric_hash": RUBRIC, "max_spec_adherence": best, "gate_fail_rate": rate,
             "escalation_needed": rate < 0.20,
             "frontier_models": frontier}, indent=2))
    return out_dir


FAILURE_PARAGRAPH = (
    "The failure that survives every prompting strategy is silent parametric "
    "completion on near-miss passages: the model answers from world knowledge "
    "when the passage mentions the entity but not the attribute asked about. "
    "Scenario A1 is answered correctly under all three strategies, but D2 and "
    "NM1 are answered from memory rather than refused, and P3 shows the same "
    "trap with a date the passage never states.")


def build_mvp(root, scenarios, tuned_fail=3, base_fail=11):
    """A minimal but well-formed MVP submission tree that check_mvp.py accepts."""
    root = Path(root)
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "docs").mkdir(parents=True, exist_ok=True)
    (root / "data" / "eval").mkdir(parents=True, exist_ok=True)
    shutil.copy(SCENARIOS_PATH, root / "data" / "eval" / "scenarios_v1.jsonl")

    (root / "configs" / "models.json").write_text(json.dumps(
        {"frontier": ABLATION_MODELS[:2], "judge_model": JUDGE,
         "base": BASE_SPEC, "verified_utc": "2026-08-20T00:00:00+00:00",
         "sources": ["https://example.invalid/docs"], "tuned": TUNED_PIN},
        indent=2))

    run_dir = root / "runs" / "mvp"
    tuned_metrics = write_cell(run_dir, scenarios, TUNED_SPEC, "zero_shot", tuned_fail)
    base_metrics = write_cell(run_dir, scenarios, BASE_SPEC, "zero_shot", base_fail)

    tuned_rel = f"runs/mvp/{E.sanitize(TUNED_SPEC)}__zero_shot.transcripts.jsonl"
    base_rel = f"runs/mvp/{E.sanitize(BASE_SPEC)}__zero_shot.transcripts.jsonl"
    repro = (f"python eval.py --model {TUNED_SPEC} \\\n"
             f"               --compare-with {BASE_SPEC} \\\n"
             f"               --strategy zero_shot \\\n"
             f"               --eval-set data/eval/scenarios_v1.jsonl \\\n"
             f"               --out runs/mvp")

    def pct(m, k):
        return "n/a" if m[k] is None else f"{100 * m[k]:.1f}%"

    (root / "docs" / "MVP_REPORT.md").write_text(f"""# MVP Report — Grounded Abstainer

Rubric v1.0, hash `{RUBRIC}`, judge `{JUDGE}`, n={len(scenarios)} scenarios.

## Pinned versions

| what | pin |
|---|---|
| tuned model (HF, public) | `{TUNED_PIN}` |
| base model | `{BASE_REPO}` |
| eval-code commit | `{EVAL_SHA}` |
| rubric hash | `{RUBRIC}` |

## Base vs tuned

| model | spec_adherence | robustness | false_answer_rate | false_refusal_rate |
|---|---|---|---|---|
| base `{BASE_REPO}` | {pct(base_metrics, 'spec_adherence')} | \
{pct(base_metrics, 'robustness')} | {pct(base_metrics, 'false_answer_rate')} | \
{pct(base_metrics, 'false_refusal_rate')} |
| tuned `{TUNED_PIN}` | {pct(tuned_metrics, 'spec_adherence')} | \
{pct(tuned_metrics, 'robustness')} | {pct(tuned_metrics, 'false_answer_rate')} | \
{pct(tuned_metrics, 'false_refusal_rate')} |

## Delta (tuned minus base)

| metric | delta |
|---|---|
| spec_adherence | {100 * (tuned_metrics['spec_adherence'] - base_metrics['spec_adherence']):+.1f} pp |
| robustness | {100 * ((tuned_metrics['robustness'] or 0) - (base_metrics['robustness'] or 0)):+.1f} pp |

## One-command repro

```bash
{repro}
```

## Raw judge transcripts

- tuned: `{tuned_rel}`
- base: [base transcripts]({base_rel})

Every cell is under `runs/mvp/*.results.json`; the checkpoint itself is at
https://huggingface.co/{TUNED_REPO} (public).
""")

    (root / "README.md").write_text(f"""# Grounded Abstainer

Base vs. tuned, the MVP comparison:

```bash
{repro}
```

Requires `ANTHROPIC_API_KEY` (judge) and a Python 3.11 venv.
""")
    return root


# --------------------------------------------------------------- fixtures
@pytest.fixture
def ablation_dir(tmp_path, scenarios):
    return build_ablation(tmp_path / "ablation", scenarios)


@pytest.fixture
def mvp_root(tmp_path, scenarios):
    return build_mvp(tmp_path / "repo", scenarios, )


def edit_json(path, mutate):
    data = json.loads(Path(path).read_text())
    mutate(data)
    Path(path).write_text(json.dumps(data, indent=2))


def rows_of(path):
    return [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]


def write_rows(path, rows):
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))


# ============================================================ check_ablation
def test_ablation_checker_fails_on_missing_directory(tmp_path):
    r = run("check_ablation.py", tmp_path / "nope")
    assert r.returncode != 0
    assert "does not exist" in out(r)


def test_ablation_checker_fails_on_empty_directory(tmp_path):
    empty = tmp_path / "ablation"
    empty.mkdir()
    r = run("check_ablation.py", empty)
    assert r.returncode != 0
    assert "empty" in out(r)


def test_ablation_checker_passes_on_a_well_formed_set(ablation_dir):
    r = run("check_ablation.py", ablation_dir)
    assert r.returncode == 0, out(r)
    assert "PASS" in r.stdout
    # the gate number is recomputed, not read: best frontier cell is 24/32
    assert "gate_fail_rate 0.2500" in r.stdout


def _drop_a_cell(d):
    for p in d.glob("openai_gpt-5.6-sol__structured.*"):
        p.unlink()


def _mixed_rubric(d):
    edit_json(d / "google_gemini-3.1-pro-preview__few_shot.results.json",
              lambda x: x.update(rubric_hash="deadbeefdeadbeef"))


def _table_still_todo(d):
    p = d / "ablation_table.md"
    p.write_text(p.read_text().split("## Failure mode")[0]
                 + "## Failure mode that survives the best prompting attempt\n\n"
                 + "_TODO: written after inspecting transcripts — name the specific\n"
                 + "failure mode, with scenario IDs as evidence._\n")


def _too_few_scenario_ids(d):
    p = d / "ablation_table.md"
    p.write_text(p.read_text().replace(
        "but D2 and NM1 are answered from memory rather than refused, and P3 "
        "shows the same trap with a date the passage never states.",
        "but ZZ9 and QQ7 are answered from memory rather than refused, and QQ8 "
        "shows the same trap with a date the passage never states."))


def _gate_rate_disagrees(d):
    edit_json(d / "gate_report.json", lambda x: x.update(gate_fail_rate=0.05))


def _escalation_flag_wrong(d):
    edit_json(d / "gate_report.json", lambda x: x.update(escalation_needed=True))


def _missing_gate_report(d):
    (d / "gate_report.json").unlink()


def _blank_judge_reasoning(d):
    p = d / "openai_gpt-5.6-sol__zero_shot.transcripts.jsonl"
    rows = rows_of(p)
    for r in rows:
        if r["gold_label"] == "answerable":
            r["judge_reasoning"] = ""
            break
    write_rows(p, rows)


def _truncated_transcripts(d):
    p = d / "google_gemini-3.1-pro-preview__structured.transcripts.jsonl"
    write_rows(p, rows_of(p)[:-1])


def _doctored_metrics(d):
    def bump(x):
        x["metrics"]["spec_adherence"] = 0.97
    edit_json(d / "openai_gpt-5.6-sol__few_shot.results.json", bump)


def _stale_summary(d):
    edit_json(d / "ablation_summary.json",
              lambda x: x.update(cells=x["cells"][:-1]))


ABLATION_MUTATIONS = [
    ("a dropped cell", _drop_a_cell, "expected 9 results JSONs"),
    ("one cell on a different rubric", _mixed_rubric, "mixed-rubric"),
    ("the _TODO placeholder still in the table", _table_still_todo, "_TODO"),
    ("fewer than 3 real scenario IDs cited", _too_few_scenario_ids,
     "real scenario ID"),
    ("gate_fail_rate disagreeing with the results", _gate_rate_disagrees,
     "disagrees with the committed results"),
    ("the escalation flag inconsistent with the rate", _escalation_flag_wrong,
     "escalation_needed"),
    ("no gate_report.json", _missing_gate_report, "gate_report.json missing"),
    ("an empty judge_reasoning on a judged example", _blank_judge_reasoning,
     "judge_reasoning is empty"),
    ("a truncated transcripts file", _truncated_transcripts, "rows, expected 32"),
    ("a doctored spec_adherence", _doctored_metrics,
     "the transcripts give"),
    ("a stale ablation_summary.json", _stale_summary, "stale"),
]


@pytest.mark.parametrize("label,mutate,expected",
                         ABLATION_MUTATIONS,
                         ids=[m[0] for m in ABLATION_MUTATIONS])
def test_ablation_checker_fails_on_a_broken_set(ablation_dir, label, mutate, expected):
    assert run("check_ablation.py", ablation_dir).returncode == 0, \
        "fixture must start valid, otherwise the mutation proves nothing"
    mutate(ablation_dir)
    r = run("check_ablation.py", ablation_dir)
    assert r.returncode != 0, f"checker accepted {label}\n{out(r)}"
    assert expected in out(r), out(r)


def test_ablation_checker_fails_on_a_cell_below_the_30_scenario_floor(
        ablation_dir, scenarios):
    write_cell(ablation_dir, scenarios[:29], ABLATION_MODELS[0], "zero_shot", 5)
    r = run("check_ablation.py", ablation_dir)
    assert r.returncode != 0
    assert "n == 29" in out(r) and "floor is 30" in out(r)


def test_ablation_checker_fails_when_the_judge_never_ran(tmp_path, scenarios):
    """Aggregates alone are not evidence: if no row was ever judged, the cells
    are not LLM-as-judge results no matter what the numbers say."""
    d = build_ablation(tmp_path / "ablation", scenarios)
    for p in d.glob("*.transcripts.jsonl"):
        rows = rows_of(p)
        for r in rows:
            if r["gold_label"] == "answerable":
                r.update(pred_class="answer", judge_verdict=None,
                         judge_reasoning=None, passed=False,
                         failure="nonverbatim_citation")
        write_rows(p, rows)
        results = p.with_name(p.name.replace(".transcripts.jsonl", ".results.json"))
        edit_json(results, lambda x: x.update(metrics=E.aggregate(rows_of(p))))
    r = run("check_ablation.py", d)
    assert r.returncode != 0
    assert "judge never ran" in out(r)


def test_ablation_checker_fails_when_both_frontier_cells_are_one_family(
        tmp_path, scenarios):
    d = build_ablation(tmp_path / "ablation", scenarios,
                       models=["openai:gpt-5.6-sol", "openai:gpt-5.6-terra",
                               f"hf:Qwen/Qwen3-1.7B@{BASE_SHA}"])
    r = run("check_ablation.py", d)
    assert r.returncode != 0
    assert "DIFFERENT families" in out(r)


def test_ablation_checker_fails_without_the_local_base_model(tmp_path, scenarios):
    d = build_ablation(tmp_path / "ablation", scenarios,
                       models=["openai:gpt-5.6-sol", "google:gemini-3.1-pro-preview",
                               "anthropic:claude-opus-4-5"])
    r = run("check_ablation.py", d)
    assert r.returncode != 0
    assert "no local base model" in out(r)


def test_ablation_checker_rejects_a_mock_sweep(tmp_path, scenarios):
    d = build_ablation(tmp_path / "ablation", scenarios,
                       models=["mock:oracle", "mock:always_refuse",
                               f"hf:Qwen/Qwen3-1.7B@{BASE_SHA}"])
    r = run("check_ablation.py", d)
    assert r.returncode != 0
    assert "mock model" in out(r)


def test_ablation_checker_fails_on_the_wrong_strategy_set(tmp_path, scenarios):
    d = build_ablation(tmp_path / "ablation", scenarios,
                       strategies=["zero_shot", "few_shot", "chain_of_thought"])
    r = run("check_ablation.py", d)
    assert r.returncode != 0
    assert "strategies must be exactly" in out(r)


# ================================================================= check_mvp
def test_mvp_checker_fails_on_an_empty_root(tmp_path):
    r = run("check_mvp.py", "--root", tmp_path)
    assert r.returncode != 0
    assert "missing or empty" in out(r)


def test_mvp_checker_passes_on_a_well_formed_package(mvp_root):
    r = run("check_mvp.py", "--root", mvp_root)
    assert r.returncode == 0, out(r)
    assert "PASS" in r.stdout
    assert "byte-for-byte" in r.stdout


def _mvp_delete_base_run(root):
    for p in (root / "runs" / "mvp").glob(f"{E.sanitize(BASE_SPEC)}__*"):
        p.unlink()


def _mvp_ghost_path_in_report(root):
    p = root / "docs" / "MVP_REPORT.md"
    p.write_text(p.read_text()
                 + "\n- held-out: `runs/mvp/heldout.transcripts.jsonl`\n")


def _mvp_transcripts_unreferenced(root):
    p = root / "docs" / "MVP_REPORT.md"
    rel = f"runs/mvp/{E.sanitize(BASE_SPEC)}__zero_shot.transcripts.jsonl"
    p.write_text(p.read_text().replace(f"- base: [base transcripts]({rel})\n", ""))


def _mvp_ghost_markdown_link(root):
    p = root / "docs" / "MVP_REPORT.md"
    p.write_text(p.read_text()
                 + "\nSee also [the held-out run](runs/heldout/results.json).\n")


def _mvp_readme_repro_drifted(root):
    p = root / "README.md"
    p.write_text(p.read_text().replace("--strategy zero_shot", "--strategy few_shot"))


def _mvp_null_judge_verdict(root):
    p = (root / "runs" / "mvp" /
         f"{E.sanitize(TUNED_SPEC)}__zero_shot.transcripts.jsonl")
    rows = rows_of(p)
    for r in rows:
        if r["gold_label"] == "answerable" and r["pred_class"] == "answer":
            r["judge_verdict"] = None
            break
    write_rows(p, rows)


def _mvp_report_pins_a_different_commit(root):
    p = root / "docs" / "MVP_REPORT.md"
    other = "f" * 40
    p.write_text(p.read_text().replace(TUNED_SHA, other))


def _mvp_rubric_hash_differs_between_models(root):
    edit_json(root / "runs" / "mvp" /
              f"{E.sanitize(BASE_SPEC)}__zero_shot.results.json",
              lambda x: x.update(rubric_hash="deadbeefdeadbeef"))


def _mvp_doctored_metrics(root):
    def bump(x):
        x["metrics"]["spec_adherence"] = 0.99
    edit_json(root / "runs" / "mvp" /
              f"{E.sanitize(TUNED_SPEC)}__zero_shot.results.json", bump)


def _mvp_no_eval_code_commit(root):
    p = root / "docs" / "MVP_REPORT.md"
    p.write_text(p.read_text().replace(f"| eval-code commit | `{EVAL_SHA}` |\n", ""))


def _mvp_no_rubric_hash(root):
    p = root / "docs" / "MVP_REPORT.md"
    p.write_text(p.read_text().replace(RUBRIC, "unstated"))


def _mvp_tuned_pin_missing_from_models_json(root):
    edit_json(root / "configs" / "models.json", lambda x: x.pop("tuned"))


def _mvp_wrong_strategy(root):
    edit_json(root / "runs" / "mvp" /
              f"{E.sanitize(TUNED_SPEC)}__zero_shot.results.json",
              lambda x: x.update(strategy="few_shot"))


def _mvp_shortened_eval_set(root):
    p = (root / "runs" / "mvp" /
         f"{E.sanitize(BASE_SPEC)}__zero_shot.transcripts.jsonl")
    write_rows(p, rows_of(p)[:20])


MVP_MUTATIONS = [
    ("no base-model run", _mvp_delete_base_run, "no base results JSON"),
    ("a report path that does not exist", _mvp_ghost_path_in_report,
     "which does not exist on disk"),
    ("a markdown-linked path that does not exist", _mvp_ghost_markdown_link,
     "runs/heldout/results.json which does not exist"),
    ("a transcripts file the report never points at", _mvp_transcripts_unreferenced,
     "does not reference the transcripts file"),
    ("a README repro line that drifted", _mvp_readme_repro_drifted,
     "not byte-consistent with README.md"),
    ("a null judge verdict on a scored answerable example", _mvp_null_judge_verdict,
     "judge_verdict is null"),
    ("a report pinning a different commit", _mvp_report_pins_a_different_commit,
     "does not contain the pinned tuned model"),
    ("two models scored under different rubrics",
     _mvp_rubric_hash_differs_between_models, "rubric_hash"),
    ("a doctored spec_adherence", _mvp_doctored_metrics, "the transcripts give"),
    ("no eval-code commit hash", _mvp_no_eval_code_commit,
     "no eval-code commit hash"),
    ("no rubric hash in the report", _mvp_no_rubric_hash, "does not state the rubric hash"),
    ("no tuned pin in models.json", _mvp_tuned_pin_missing_from_models_json,
     "expected '<owner>/<name>@<40-hex-sha>'"),
    ("the wrong prompting strategy", _mvp_wrong_strategy, "zero_shot"),
    ("the two models on different eval sets", _mvp_shortened_eval_set,
     "transcript rows but metrics.n"),
]


@pytest.mark.parametrize("label,mutate,expected", MVP_MUTATIONS,
                         ids=[m[0] for m in MVP_MUTATIONS])
def test_mvp_checker_fails_on_a_broken_package(mvp_root, label, mutate, expected):
    assert run("check_mvp.py", "--root", mvp_root).returncode == 0, \
        "fixture must start valid, otherwise the mutation proves nothing"
    mutate(mvp_root)
    r = run("check_mvp.py", "--root", mvp_root)
    assert r.returncode != 0, f"checker accepted {label}\n{out(r)}"
    assert expected in out(r), out(r)


# ========================================================= push_model --check
def test_push_model_check_fails_on_a_missing_models_json(tmp_path):
    r = run("push_model.py", "--check", tmp_path / "models.json", "--no-hub")
    assert r.returncode != 0
    assert "does not exist" in out(r)


def test_push_model_check_passes_on_a_well_formed_pin(tmp_path):
    p = tmp_path / "models.json"
    p.write_text(json.dumps({"base": BASE_SPEC, "judge_model": JUDGE,
                             "frontier": ABLATION_MODELS[:2], "tuned": TUNED_PIN}))
    r = run("push_model.py", "--check", p, "--no-hub")
    assert r.returncode == 0, out(r)
    assert TUNED_REPO in r.stdout and TUNED_SHA in r.stdout


BAD_PINS = [
    ("no tuned key at all", {"base": BASE_SPEC}, "has no 'tuned' key"),
    ("a short sha", {"tuned": f"{TUNED_REPO}@0123456789abcdef"}, "not well formed"),
    ("no owner", {"tuned": f"grounded-abstainer@{TUNED_SHA}"}, "not well formed"),
    ("no commit at all", {"tuned": TUNED_REPO}, "not well formed"),
    ("an uppercase sha", {"tuned": f"{TUNED_REPO}@{TUNED_SHA.upper()}"},
     "not well formed"),
    ("a branch instead of a commit", {"tuned": f"{TUNED_REPO}@main"},
     "not well formed"),
    ("an hf: prefix the contract forbids", {"tuned": f"hf:{TUNED_PIN}"},
     "not well formed"),
    ("a non-string pin", {"tuned": 12345}, "expected a string"),
]


@pytest.mark.parametrize("label,cfg,expected", BAD_PINS,
                         ids=[b[0] for b in BAD_PINS])
def test_push_model_check_fails_on_a_malformed_pin(tmp_path, label, cfg, expected):
    p = tmp_path / "models.json"
    p.write_text(json.dumps({"base": BASE_SPEC, **cfg}))
    r = run("push_model.py", "--check", p, "--no-hub")
    assert r.returncode != 0, f"checker accepted {label}"
    assert expected in out(r), out(r)


def test_push_model_check_fails_on_unreadable_json(tmp_path):
    p = tmp_path / "models.json"
    p.write_text("{ not json")
    r = run("push_model.py", "--check", p, "--no-hub")
    assert r.returncode != 0
    assert "not readable JSON" in out(r)


def test_push_model_check_refuses_to_double_as_a_push(tmp_path):
    p = tmp_path / "models.json"
    p.write_text(json.dumps({"tuned": TUNED_PIN}))
    r = run("push_model.py", "--check", p, "--no-hub", "--repo", "x")
    assert r.returncode != 0
    assert "read-only mode" in out(r)


def test_push_model_live_mode_rejects_missing_artifact_dirs(tmp_path):
    """Fails on argument validation, before any credential or network use."""
    r = run("push_model.py", "--adapter", tmp_path / "adapter",
            "--merged", tmp_path / "merged", "--repo", "some-model",
            "--env-file", tmp_path / "no.env")
    assert r.returncode != 0
    assert "is not a directory" in out(r)


def test_push_model_live_mode_stops_and_names_the_key_when_it_cannot_write(tmp_path):
    adapter = tmp_path / "adapter"
    merged = tmp_path / "merged"
    adapter.mkdir()
    merged.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    (merged / "config.json").write_text("{}")
    (merged / "model.safetensors").write_bytes(b"\0")

    env = {k: v for k, v in os.environ.items()
           if k not in ("HF_TOKEN", "HF_USERNAME", "HUGGING_FACE_HUB_TOKEN")}
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "push_model.py"),
         "--adapter", str(adapter), "--merged", str(merged),
         "--repo", "some-model", "--env-file", str(tmp_path / "no.env")],
        cwd=str(REPO_ROOT), capture_output=True, text=True, env=env)
    assert r.returncode != 0
    assert "HF_TOKEN" in out(r)
    assert "Nothing was pushed" in out(r)


def test_push_model_live_mode_rejects_a_directory_that_is_not_an_adapter(tmp_path):
    adapter = tmp_path / "adapter"
    merged = tmp_path / "merged"
    adapter.mkdir()
    merged.mkdir()
    (adapter / "notes.txt").write_text("nope")
    (merged / "config.json").write_text("{}")
    (merged / "model.safetensors").write_bytes(b"\0")
    r = run("push_model.py", "--adapter", adapter, "--merged", merged,
            "--repo", "some-model", "--env-file", tmp_path / "no.env")
    assert r.returncode != 0
    assert "adapter_config.json" in out(r)


def test_push_model_appends_tuned_without_clobbering_existing_keys(tmp_path):
    """The one live-mode side effect that is testable offline: the models.json
    append must preserve every key P1 wrote."""
    sys.path.insert(0, str(SCRIPTS))
    import push_model  # noqa: PLC0415  (script dir is not on the path at import time)

    p = tmp_path / "models.json"
    original = {"frontier": ABLATION_MODELS[:2], "judge_model": JUDGE,
                "base": BASE_SPEC, "verified_utc": "2026-08-20T00:00:00+00:00",
                "sources": ["https://example.invalid/docs"]}
    p.write_text(json.dumps(original, indent=2))

    assert push_model.update_models_json(p, TUNED_REPO, TUNED_SHA) is None
    after = json.loads(p.read_text())
    assert after["tuned"] == TUNED_PIN
    for k, v in original.items():
        assert after[k] == v, f"{k} was clobbered"

    # and re-publishing replaces only the pin
    new_sha = "f" * 40
    assert push_model.update_models_json(p, TUNED_REPO, new_sha) is None
    again = json.loads(p.read_text())
    assert again["tuned"] == f"{TUNED_REPO}@{new_sha}"
    for k, v in original.items():
        assert again[k] == v

    # the freshly written file is what --check reads
    r = run("push_model.py", "--check", p, "--no-hub")
    assert r.returncode == 0, out(r)


def test_push_model_refuses_to_overwrite_a_non_object_models_json(tmp_path):
    sys.path.insert(0, str(SCRIPTS))
    import push_model  # noqa: PLC0415

    p = tmp_path / "models.json"
    p.write_text(json.dumps(["not", "an", "object"]))
    err = push_model.update_models_json(p, TUNED_REPO, TUNED_SHA)
    assert err and "refusing to overwrite" in err
    assert json.loads(p.read_text()) == ["not", "an", "object"]


# ------------------------------------------------------------- the checkers
def test_every_checker_is_executable_and_self_documenting():
    for name in ("check_ablation.py", "check_mvp.py", "push_model.py",
                 "make_gate_report.py"):
        path = SCRIPTS / name
        assert path.exists(), f"scripts/{name} missing"
        r = run(name, "--help")
        assert r.returncode == 0, out(r)


# ============================================ anti-fabrication: the raw_output
# `eval.aggregate` reads the STORED pred_class/passed/adversarial fields, so a
# self-consistent hand edit moves every aggregate in a run without touching a
# single raw_output. These forgeries all keep the results JSON, the summary and
# the gate report in perfect agreement with the transcripts -- exactly what a
# careful fabricator would do -- so the only thing that can catch them is
# re-scoring each row from its own raw_output with the frozen rubric.
def resync_ablation(d):
    """Recompute results JSONs, summary and gate report from the (edited)
    transcripts, leaving the artifact set internally self-consistent."""
    d = Path(d)
    metrics_by_cell = {}
    for tp in sorted(d.glob("*.transcripts.jsonl")):
        rows = rows_of(tp)
        rp = tp.with_name(tp.name.replace(".transcripts.jsonl", ".results.json"))
        res = json.loads(rp.read_text())
        res["metrics"] = E.aggregate(rows)
        rp.write_text(json.dumps(res, indent=2))
        metrics_by_cell[(res["model"], res["strategy"])] = res["metrics"]

    summary_path = d / "ablation_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        for row in summary.get("cells", []):
            key = (row.get("model"), row.get("strategy"))
            if key in metrics_by_cell:
                row["metrics"] = metrics_by_cell[key]
        summary_path.write_text(json.dumps(summary, indent=2))

    gate_path = d / "gate_report.json"
    if gate_path.exists():
        frontier = [k for k in metrics_by_cell
                    if k[0].split(":", 1)[0] in ("openai", "google", "anthropic")]
        best = max(metrics_by_cell[k]["spec_adherence"] for k in frontier)
        report = json.loads(gate_path.read_text())
        report.update(max_spec_adherence=best, gate_fail_rate=1.0 - best,
                      escalation_needed=(1.0 - best) < 0.20)
        gate_path.write_text(json.dumps(report, indent=2))


def _forge_a_whole_cell_into_a_perfect_score(d):
    """Every row claims a clean cited pass; the raw_output still shows the model
    emitting prose with no Answer/Citation structure at all."""
    p = d / "openai_gpt-5.6-sol__structured.transcripts.jsonl"
    rows = rows_of(p)
    for r in rows:
        if not r["passed"] or r["gold_label"] == "unanswerable":
            r["raw_output"] = "Sure! Based on the passage, the answer is probably 42."
        r.update(pred_class="answer", passed=True, failure=None,
                 judge_verdict=True, judge_reasoning="YES. The span entails it.")
    write_rows(p, rows)
    resync_ablation(d)


def _hide_a_false_answer(d):
    """The model answered a question the passage cannot support; the row records a
    refusal. Aggregates are untouched, so nothing downstream notices."""
    p = d / "google_gemini-3.1-pro-preview__zero_shot.transcripts.jsonl"
    rows = rows_of(p)
    for r in rows:
        if r["gold_label"] == "unanswerable":
            r["raw_output"] = 'Answer: 1996.\nCitation: "the treaty was signed"'
            break
    write_rows(p, rows)
    resync_ablation(d)


def _fabricate_a_citation(d):
    """A cited span that appears nowhere in the passage, waved through by a stored
    judge YES -- the verdict the checker is allowed to trust cannot rescue a span
    the rubric would have rejected before the judge was ever called."""
    p = d / "openai_gpt-5.6-sol__zero_shot.transcripts.jsonl"
    rows = rows_of(p)
    for r in rows:
        if r["gold_label"] == "answerable" and r["passed"]:
            r["raw_output"] = ('Answer: the figure given in the passage.\n'
                               'Citation: "THIS SPAN IS NOT IN THE PASSAGE"')
            break
    write_rows(p, rows)
    resync_ablation(d)


def _relabel_an_adversarial_row(d):
    """Drop a failing scenario out of the adversarial subset to lift robustness."""
    p = d / "google_gemini-3.1-pro-preview__few_shot.transcripts.jsonl"
    rows = rows_of(p)
    for r in rows:
        if r["adversarial"] and not r["passed"]:
            r["adversarial"] = False
            break
    write_rows(p, rows)
    resync_ablation(d)


FORGERIES = [
    ("a whole cell forged to a perfect score", _forge_a_whole_cell_into_a_perfect_score,
     "re-deriving from this row's own raw_output"),
    ("a false answer recorded as a refusal", _hide_a_false_answer,
     "re-deriving from this row's own raw_output"),
    ("a fabricated citation blessed by a stored judge verdict", _fabricate_a_citation,
     "not verbatim in the passage"),
    ("a failing row relabelled out of the adversarial subset", _relabel_an_adversarial_row,
     "adversarial == False"),
]


@pytest.mark.parametrize("label,forge,expected", FORGERIES,
                         ids=[f[0] for f in FORGERIES])
def test_ablation_checker_catches_a_self_consistent_transcript_forgery(
        ablation_dir, label, forge, expected):
    assert run("check_ablation.py", ablation_dir).returncode == 0, \
        "fixture must start valid, otherwise the forgery proves nothing"
    forge(ablation_dir)

    # the forgery is self-consistent: eval.aggregate over the edited transcripts
    # reproduces every committed number, so the aggregate layer alone is blind
    for tp in sorted(Path(ablation_dir).glob("*.transcripts.jsonl")):
        rp = tp.with_name(tp.name.replace(".transcripts.jsonl", ".results.json"))
        assert E.aggregate(rows_of(tp)) == json.loads(rp.read_text())["metrics"]

    r = run("check_ablation.py", ablation_dir)
    assert r.returncode != 0, f"checker accepted {label}\n{out(r)}"
    assert expected in out(r), out(r)


def test_ablation_forgery_lifts_the_gate_number_before_the_fix_catches_it(ablation_dir):
    """The forgery is not academic: it moves the G1 gate number itself."""
    before = json.loads((ablation_dir / "gate_report.json").read_text())
    _forge_a_whole_cell_into_a_perfect_score(ablation_dir)
    after = json.loads((ablation_dir / "gate_report.json").read_text())
    assert before["gate_fail_rate"] == pytest.approx(0.25)
    assert after["gate_fail_rate"] == pytest.approx(0.0)
    assert after["escalation_needed"] is True and before["escalation_needed"] is False
    assert run("check_ablation.py", ablation_dir).returncode != 0


def test_ablation_checker_trusts_only_the_judge_verdict_it_cannot_recompute(
        ablation_dir):
    """The documented trust boundary, both halves of it.

    The entailment verdict cannot be recomputed offline, so a flipped verdict on a
    row that really did parse and really did cite verbatim is accepted. Everything
    the rubric decides BEFORE the judge is re-derived and is not negotiable.
    """
    p = ablation_dir / "openai_gpt-5.6-sol__zero_shot.transcripts.jsonl"
    rows = rows_of(p)
    for r in rows:
        if r["failure"] == "entailment_fail":
            r.update(judge_verdict=True, passed=True, failure=None,
                     judge_reasoning="YES. On reflection the span entails it.")
            break
    else:
        pytest.fail("fixture has no entailment_fail row to flip")
    write_rows(p, rows)
    resync_ablation(ablation_dir)
    assert run("check_ablation.py", ablation_dir).returncode == 0, \
        "a flipped judge verdict on an otherwise valid row cannot be disproved offline"

    _fabricate_a_citation(ablation_dir)
    r = run("check_ablation.py", ablation_dir)
    assert r.returncode != 0
    assert "never sends this row to the judge" in out(r)


def test_ablation_checker_fails_when_the_eval_set_is_gone(ablation_dir, tmp_path):
    """Without the scenarios the rows can only be checked against themselves,
    which is the hole in the first place — say so instead of passing."""
    r = run("check_ablation.py", ablation_dir, "--scenarios", tmp_path / "nope.jsonl")
    assert r.returncode != 0
    assert "cannot be re-derived" in out(r)


def test_mvp_checker_catches_a_transcript_that_hides_a_false_answer(mvp_root):
    """Same hole, same fix, in the G2 gate: this edit leaves every aggregate,
    every results JSON and every reported number untouched."""
    assert run("check_mvp.py", "--root", mvp_root).returncode == 0
    p = (mvp_root / "runs" / "mvp" /
         f"{E.sanitize(TUNED_SPEC)}__zero_shot.transcripts.jsonl")
    rows = rows_of(p)
    for r in rows:
        if r["gold_label"] == "unanswerable":
            r["raw_output"] = 'Answer: 1996.\nCitation: "the treaty was signed"'
            break
    write_rows(p, rows)
    rp = p.with_name(p.name.replace(".transcripts.jsonl", ".results.json"))
    assert E.aggregate(rows_of(p)) == json.loads(rp.read_text())["metrics"], \
        "the edit must be aggregate-neutral, or it proves the wrong thing"

    r = run("check_mvp.py", "--root", mvp_root)
    assert r.returncode != 0
    assert "re-deriving from this row's own raw_output" in out(r)


# ================================ check_mvp: the numbers the grader actually reads
def _rewrite_report(root, fn):
    p = Path(root) / "docs" / "MVP_REPORT.md"
    p.write_text(fn(p.read_text()))


def _set_cell(line, index, value):
    cells = line.split("|")
    cells[index] = f" {value} "
    return "|".join(cells)


def _mvp_inflated_tuned_spec_adherence(root):
    def fn(text):
        return "\n".join(_set_cell(ln, 2, "99.9%") if ln.startswith("| tuned ") else ln
                         for ln in text.splitlines()) + "\n"
    _rewrite_report(root, fn)


def _mvp_deflated_base_spec_adherence(root):
    """Flattering the delta from the other end."""
    def fn(text):
        return "\n".join(_set_cell(ln, 2, "11.1%") if ln.startswith("| base ") else ln
                         for ln in text.splitlines()) + "\n"
    _rewrite_report(root, fn)


def _mvp_inflated_tuned_robustness(root):
    def fn(text):
        return "\n".join(_set_cell(ln, 3, "100.0%") if ln.startswith("| base ") else ln
                         for ln in text.splitlines()) + "\n"
    _rewrite_report(root, fn)


def _mvp_inflated_delta(root):
    _rewrite_report(root, lambda t: re.sub(
        r"^\| spec_adherence \| [-+][\d.]+ pp \|", "| spec_adherence | +42.0 pp |",
        t, count=1, flags=re.M))


def _mvp_metric_column_nothing_backs(root):
    def fn(text):
        head, sep, rest = text.partition("## Base vs tuned")
        body, sep2, tail = rest.partition("## Delta")
        lines = []
        for ln in body.splitlines():
            if ln.startswith("| model |"):
                ln = ln[:-1] + " hallucination_rate |"
            elif ln.startswith("|---|"):
                ln = ln + "---|"
            elif ln.startswith(("| base ", "| tuned ")):
                ln = ln[:-1] + " 0.0% |"
            lines.append(ln)
        return head + sep + "\n".join(lines) + "\n\n" + sep2 + tail
    _rewrite_report(root, fn)


def _mvp_metric_row_nothing_backs(root):
    _rewrite_report(root, lambda t: re.sub(
        r"^\| robustness \| ", "| abstention_accuracy | +99.0 pp |\n| robustness | ",
        t, count=1, flags=re.M))


def _mvp_metrics_table_for_a_model_that_never_ran(root):
    """A fourth row of headline numbers for a model with no results JSON at all."""
    def fn(text):
        lines = []
        for ln in text.splitlines():
            lines.append(ln)
            if ln.startswith("| tuned "):
                lines.append("| `openai:gpt-5.6-sol` | 99.9% | 99.9% | 0.0% | 0.0% |")
        return "\n".join(lines) + "\n"
    _rewrite_report(root, fn)


REPORT_NUMBER_MUTATIONS = [
    ("an inflated tuned spec_adherence", _mvp_inflated_tuned_spec_adherence,
     "does not match the artifact it claims to summarize"),
    ("a deflated base spec_adherence", _mvp_deflated_base_spec_adherence,
     "does not match the artifact it claims to summarize"),
    ("an inflated robustness", _mvp_inflated_tuned_robustness,
     "does not match the artifact it claims to summarize"),
    ("an inflated delta", _mvp_inflated_delta,
     "the delta spec_adherence cell reports '+42.0 pp'"),
    ("a metric column no results JSON backs", _mvp_metric_column_nothing_backs,
     "the eval harness never emits"),
    ("a metric row no results JSON backs", _mvp_metric_row_nothing_backs,
     "the eval harness never emits"),
    ("metric numbers for a model that never ran",
     _mvp_metrics_table_for_a_model_that_never_ran,
     "names neither the pinned tuned model"),
]


@pytest.mark.parametrize("label,mutate,expected", REPORT_NUMBER_MUTATIONS,
                         ids=[m[0] for m in REPORT_NUMBER_MUTATIONS])
def test_mvp_checker_fails_when_the_report_numbers_are_not_the_committed_ones(
        mvp_root, label, mutate, expected):
    """The artifacts stay untouched: only the prose a grader reads is doctored."""
    assert run("check_mvp.py", "--root", mvp_root).returncode == 0, \
        "fixture must start valid, otherwise the mutation proves nothing"
    before = {p.name: p.read_bytes() for p in (mvp_root / "runs" / "mvp").iterdir()}
    mutate(mvp_root)
    after = {p.name: p.read_bytes() for p in (mvp_root / "runs" / "mvp").iterdir()}
    assert before == after, "the mutation must only touch the report"

    r = run("check_mvp.py", "--root", mvp_root)
    assert r.returncode != 0, f"checker accepted {label}\n{out(r)}"
    assert expected in out(r), out(r)


def test_mvp_checker_reports_how_many_numbers_it_verified(mvp_root):
    r = run("check_mvp.py", "--root", mvp_root)
    assert r.returncode == 0, out(r)
    assert "reported metric value(s) match the committed results JSONs" in r.stdout


def test_mvp_checker_accepts_the_precision_the_report_is_written_at(mvp_root):
    """65.6% for 0.65625 is right; 65.7% is not. The tolerance is half a unit in
    the last printed place, not a blanket epsilon."""
    _rewrite_report(mvp_root, lambda t: t.replace("| 65.6% |", "| 65.7% |"))
    r = run("check_mvp.py", "--root", mvp_root)
    assert r.returncode != 0, out(r)
    assert "does not match the artifact it claims to summarize" in out(r)


# ============================================================ make_gate_report
def test_make_gate_report_makes_the_p2_verify_command_passable(tmp_path, scenarios):
    """run_ablation.py writes no gate_report.json, and check_ablation.py requires
    one — without this script the P2 verify command cannot pass at all."""
    d = build_ablation(tmp_path / "ablation", scenarios, gate_report=False)
    assert not (d / "gate_report.json").exists()
    r = run("check_ablation.py", d)
    assert r.returncode != 0 and "gate_report.json missing" in out(r)

    made = run("make_gate_report.py", d)
    assert made.returncode == 0, out(made)
    assert run("check_ablation.py", d).returncode == 0, out(run("check_ablation.py", d))

    report = json.loads((d / "gate_report.json").read_text())
    assert report["gate_fail_rate"] == pytest.approx(0.25)
    assert report["max_spec_adherence"] == pytest.approx(0.75)
    assert report["escalation_needed"] is False
    assert report["rubric_hash"] == RUBRIC


def test_make_gate_report_records_which_cells_it_treated_as_frontier(tmp_path, scenarios):
    d = build_ablation(tmp_path / "ablation", scenarios, gate_report=False)
    assert run("make_gate_report.py", d).returncode == 0
    report = json.loads((d / "gate_report.json").read_text())

    assert report["n_frontier_cells"] == 6 and report["n_base_cells"] == 3
    assert {c["model"] for c in report["frontier_cells"]} == set(ABLATION_MODELS[:2])
    assert {c["model"] for c in report["base_cells"]} == {ABLATION_MODELS[2]}
    assert all(c["provider"] in ("openai", "google") for c in report["frontier_cells"])
    assert {c["strategy"] for c in report["frontier_cells"]} == set(STRATEGIES)
    assert report["best_frontier_cell"] == {"model": "openai:gpt-5.6-sol",
                                            "strategy": "structured"}


def test_make_gate_report_excludes_the_local_base_row_from_the_max(tmp_path, scenarios):
    """The gate measures the FRONTIER ceiling. A base row that scores better than
    every frontier cell must not lower gate_fail_rate."""
    d = build_ablation(tmp_path / "ablation", scenarios, gate_report=False,
                       fail_counts={0: [10, 9, 8], 1: [11, 10, 9], 2: [0, 0, 0]})
    assert run("make_gate_report.py", d).returncode == 0
    report = json.loads((d / "gate_report.json").read_text())
    assert all(c["spec_adherence"] == 1.0 for c in report["base_cells"])
    assert report["max_spec_adherence"] == pytest.approx(0.75)
    assert report["gate_fail_rate"] == pytest.approx(0.25)
    assert run("check_ablation.py", d).returncode == 0


def test_make_gate_report_sets_escalation_needed_below_the_threshold(tmp_path, scenarios):
    d = build_ablation(tmp_path / "ablation", scenarios, gate_report=False,
                       fail_counts={0: [10, 9, 1], 1: [11, 10, 9], 2: [13, 13, 13]})
    assert run("make_gate_report.py", d).returncode == 0
    report = json.loads((d / "gate_report.json").read_text())
    assert report["gate_fail_rate"] == pytest.approx(1 / 32)
    assert report["escalation_needed"] is True
    assert run("check_ablation.py", d).returncode == 0


def test_make_gate_report_writes_where_it_is_told(tmp_path, scenarios):
    d = build_ablation(tmp_path / "ablation", scenarios, gate_report=False)
    dest = tmp_path / "elsewhere" / "gate_report.json"
    assert run("make_gate_report.py", d, "--out", dest).returncode == 0
    assert dest.exists()
    assert not (d / "gate_report.json").exists()


def _mock_sweep(tmp_path, scenarios):
    return build_ablation(tmp_path / "a", scenarios, gate_report=False,
                          models=["mock:oracle", "mock:always_refuse",
                                  f"hf:Qwen/Qwen3-1.7B@{BASE_SHA}"])


def _no_local_base(tmp_path, scenarios):
    return build_ablation(tmp_path / "a", scenarios, gate_report=False,
                          models=["openai:gpt-5.6-sol",
                                  "google:gemini-3.1-pro-preview",
                                  "anthropic:claude-opus-4-5"])


def _one_frontier_family(tmp_path, scenarios):
    return build_ablation(tmp_path / "a", scenarios, gate_report=False,
                          models=["openai:gpt-5.6-sol", "openai:gpt-5.6-terra",
                                  f"hf:Qwen/Qwen3-1.7B@{BASE_SHA}"])


def _mixed_rubric_cells(tmp_path, scenarios):
    d = build_ablation(tmp_path / "a", scenarios, gate_report=False)
    edit_json(d / "openai_gpt-5.6-sol__few_shot.results.json",
              lambda x: x.update(rubric_hash="deadbeefdeadbeef"))
    return d


UNGATEABLE = [
    ("a mock sweep", _mock_sweep, "neither a frontier family"),
    ("no local base row", _no_local_base, "expected 6 frontier cells"),
    ("one frontier family", _one_frontier_family, "DIFFERENT frontier families"),
    ("cells on different rubrics", _mixed_rubric_cells, "mixed-rubric"),
]


@pytest.mark.parametrize("label,build,expected", UNGATEABLE,
                         ids=[u[0] for u in UNGATEABLE])
def test_make_gate_report_refuses_a_grid_it_cannot_gate(tmp_path, scenarios,
                                                        label, build, expected):
    """Refusing to write beats writing a number that means nothing."""
    d = build(tmp_path, scenarios)
    r = run("make_gate_report.py", d)
    assert r.returncode != 0, f"gate report produced for {label}\n{out(r)}"
    assert expected in out(r), out(r)
    assert not (d / "gate_report.json").exists()


def test_make_gate_report_refuses_an_empty_directory(tmp_path):
    empty = tmp_path / "ablation"
    empty.mkdir()
    r = run("make_gate_report.py", empty)
    assert r.returncode != 0
    assert "no *.results.json" in out(r)
    r = run("make_gate_report.py", tmp_path / "nope")
    assert r.returncode != 0
    assert "does not exist" in out(r)


def test_check_ablation_still_catches_a_tampered_generated_gate_report(tmp_path,
                                                                       scenarios):
    """The generated report is not trusted either: check_ablation.py recomputes
    the number independently, so editing the file after the fact fails."""
    d = build_ablation(tmp_path / "ablation", scenarios, gate_report=False)
    assert run("make_gate_report.py", d).returncode == 0
    assert run("check_ablation.py", d).returncode == 0

    edit_json(d / "gate_report.json", lambda x: x.update(gate_fail_rate=0.05))
    r = run("check_ablation.py", d)
    assert r.returncode != 0
    assert "disagrees with the committed results" in out(r)

    assert run("make_gate_report.py", d).returncode == 0      # regenerating fixes it
    assert run("check_ablation.py", d).returncode == 0

    edit_json(d / "gate_report.json", lambda x: x.update(escalation_needed=True))
    r = run("check_ablation.py", d)
    assert r.returncode != 0
    assert "escalation_needed" in out(r)


# ================================================= push_model: the SUCCESS path
PUSHED_SHA = "9f8e7d6c5b4a39281706f5e4d3c2b1a098765432"


def _fake_hub(monkeypatch, commit_sha, private=False, gated=False):
    """A fully working fake Hub: whoami + create_repo + upload_folder + model_info.

    The three existing live-mode tests all stop at a failure BEFORE the Hub is
    touched, which is exactly why a NameError on the success path survived: the
    publish, both uploads and the models.json write all completed and the run
    still exited non-zero. Returns the call log so the test can prove the whole
    sequence ran.
    """
    log = []

    class _FakeHubHTTPError(Exception):
        pass

    class _Commit:
        def __init__(self, oid):
            self.oid = oid

    class _Info:
        def __init__(self):
            self.sha, self.private, self.gated = commit_sha, private, gated

    class _Api:
        def __init__(self, token=None):
            log.append(("HfApi", bool(token)))

        def whoami(self):
            log.append(("whoami",))
            return {"name": "tester", "auth": {"accessToken": {"role": "write"}}}

        def create_repo(self, repo_id, repo_type=None, private=None, exist_ok=None):
            log.append(("create_repo", repo_id, private))
            return None

        def upload_folder(self, repo_id=None, folder_path=None, path_in_repo=None,
                          commit_message=None):
            log.append(("upload_folder", repo_id, path_in_repo))
            return _Commit(commit_sha)

        def model_info(self, repo_id, revision=None):
            log.append(("model_info", repo_id, revision))
            return _Info()

    hub = types.ModuleType("huggingface_hub")
    hub.HfApi = _Api
    errors = types.ModuleType("huggingface_hub.errors")
    errors.HfHubHTTPError = _FakeHubHTTPError
    errors.RepositoryNotFoundError = type("RepositoryNotFoundError",
                                          (_FakeHubHTTPError,), {})
    errors.RevisionNotFoundError = type("RevisionNotFoundError",
                                        (_FakeHubHTTPError,), {})
    hub.errors = errors
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)
    return log


def _publishable_artifacts(tmp_path):
    adapter, merged = tmp_path / "adapter", tmp_path / "merged"
    adapter.mkdir()
    merged.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    (merged / "config.json").write_text("{}")
    (merged / "model.safetensors").write_bytes(b"\0")
    return adapter, merged


P1_KEYS = {"frontier": ABLATION_MODELS[:2], "judge_model": JUDGE,
           "base": BASE_SPEC, "verified_utc": "2026-08-20T00:00:00+00:00",
           "sources": ["https://example.invalid/docs"]}


def test_push_model_exits_zero_and_pins_the_commit_on_a_successful_publish(
        tmp_path, monkeypatch, capsys):
    """The path P5 actually takes when everything works."""
    sys.path.insert(0, str(SCRIPTS))
    import push_model  # noqa: PLC0415

    log = _fake_hub(monkeypatch, PUSHED_SHA)
    adapter, merged = _publishable_artifacts(tmp_path)
    models_json = tmp_path / "models.json"
    models_json.write_text(json.dumps(P1_KEYS, indent=2))
    monkeypatch.setenv("HF_TOKEN", "hf_this_is_not_a_real_token")
    monkeypatch.setenv("HF_USERNAME", "tester")

    rc = push_model.main(["--adapter", str(adapter), "--merged", str(merged),
                          "--repo", "grounded-abstainer-qwen3-1.7b",
                          "--models-json", str(models_json),
                          "--env-file", str(tmp_path / "no.env")])
    printed = capsys.readouterr().out
    assert rc == 0, f"successful publish exited {rc}\n{printed}"

    # the publish really happened, in order, before the pin was written
    assert ("create_repo", "tester/grounded-abstainer-qwen3-1.7b", False) in log
    uploads = [c for c in log if c[0] == "upload_folder"]
    assert len(uploads) == 2
    assert [c[2] for c in uploads] == [None, "adapter"]
    assert any(c[0] == "model_info" for c in log)

    after = json.loads(models_json.read_text())
    assert after["tuned"] == f"tester/grounded-abstainer-qwen3-1.7b@{PUSHED_SHA}"
    for k, v in P1_KEYS.items():
        assert after[k] == v, f"{k} was clobbered by the publish"
    assert "PASS published" in printed
    assert "hf_this_is_not_a_real_token" not in printed   # the token is never echoed

    # and the pin it just wrote is the pin the P5 verify command accepts
    assert run("push_model.py", "--check", models_json, "--no-hub").returncode == 0


def test_push_model_success_path_reports_a_failed_pin_write(tmp_path, monkeypatch):
    """Sabotage of the same path: the Hub half succeeds, the models.json half
    cannot, and the run must exit non-zero saying so rather than claiming PASS."""
    sys.path.insert(0, str(SCRIPTS))
    import push_model  # noqa: PLC0415

    _fake_hub(monkeypatch, PUSHED_SHA)
    adapter, merged = _publishable_artifacts(tmp_path)
    models_json = tmp_path / "models.json"
    models_json.write_text(json.dumps(["not", "an", "object"]))
    monkeypatch.setenv("HF_TOKEN", "hf_this_is_not_a_real_token")
    monkeypatch.setenv("HF_USERNAME", "tester")

    rc = push_model.main(["--adapter", str(adapter), "--merged", str(merged),
                          "--repo", "grounded-abstainer-qwen3-1.7b",
                          "--models-json", str(models_json),
                          "--env-file", str(tmp_path / "no.env")])
    assert rc != 0
    assert json.loads(models_json.read_text()) == ["not", "an", "object"]


def test_push_model_success_path_refuses_a_private_repo(tmp_path, monkeypatch):
    """The read-back is a guard, not a formality: no pin is written for a repo a
    grader could not pull."""
    sys.path.insert(0, str(SCRIPTS))
    import push_model  # noqa: PLC0415

    _fake_hub(monkeypatch, PUSHED_SHA, private=True)
    adapter, merged = _publishable_artifacts(tmp_path)
    models_json = tmp_path / "models.json"
    models_json.write_text(json.dumps(P1_KEYS, indent=2))
    monkeypatch.setenv("HF_TOKEN", "hf_this_is_not_a_real_token")
    monkeypatch.setenv("HF_USERNAME", "tester")

    rc = push_model.main(["--adapter", str(adapter), "--merged", str(merged),
                          "--repo", "grounded-abstainer-qwen3-1.7b",
                          "--models-json", str(models_json),
                          "--env-file", str(tmp_path / "no.env")])
    assert rc != 0
    assert "tuned" not in json.loads(models_json.read_text())
