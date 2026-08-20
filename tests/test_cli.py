"""End-to-end CLI paths, network-free via mock generators + mock judge. Covers the
one-command repro shape the project doc grades on, and the --compare-with
base-vs-tuned mechanism required at MVP."""
import json
import subprocess
import sys

import pytest

from conftest import EXEMPLARS_PATH, REPO_ROOT, SCENARIOS_PATH


def run_cli(*args, cwd=None):
    return subprocess.run([sys.executable, str(REPO_ROOT / "eval.py"), *args],
                          cwd=str(cwd or REPO_ROOT), capture_output=True, text=True)


@pytest.fixture
def out_dir(tmp_path):
    return tmp_path / "run"


def test_single_model_run_writes_transcripts_and_results(out_dir):
    r = run_cli("--model", "mock:oracle", "--eval-set", str(SCENARIOS_PATH),
                "--judge", "mock:yes", "--out", str(out_dir))
    assert r.returncode == 0, r.stderr
    assert "c0adb876c3e6c4c9" in r.stdout, "run must print the rubric hash it scored under"

    transcripts = out_dir / "mock_oracle__zero_shot.transcripts.jsonl"
    results = out_dir / "mock_oracle__zero_shot.results.json"
    assert transcripts.exists() and results.exists()

    rows = [json.loads(ln) for ln in transcripts.read_text().splitlines() if ln.strip()]
    assert len(rows) == 32
    assert {"scenario_id", "raw_output", "judge_verdict", "judge_reasoning"} <= set(rows[0])

    data = json.loads(results.read_text())
    assert data["rubric_hash"] == "c0adb876c3e6c4c9"
    assert data["metrics"]["spec_adherence"] == 1.0
    assert data["model"] == "mock:oracle" and data["strategy"] == "zero_shot"
    assert data["timestamp_utc"] and data["judge_model"]


def test_compare_with_emits_both_models_and_a_delta(out_dir):
    """base-vs-tuned mechanism: two models, one rubric, one delta table."""
    r = run_cli("--model", "mock:oracle", "--compare-with", "mock:always_refuse",
                "--eval-set", str(SCENARIOS_PATH), "--judge", "mock:yes",
                "--strategy", "zero_shot", "--out", str(out_dir))
    assert r.returncode == 0, r.stderr
    assert "delta (mock:oracle minus mock:always_refuse)" in r.stdout
    assert "spec_adherence" in r.stdout and "pp" in r.stdout

    for stem in ("mock_oracle__zero_shot", "mock_always_refuse__zero_shot"):
        assert (out_dir / f"{stem}.transcripts.jsonl").exists()
        assert (out_dir / f"{stem}.results.json").exists()

    oracle = json.loads((out_dir / "mock_oracle__zero_shot.results.json").read_text())
    refuse = json.loads((out_dir / "mock_always_refuse__zero_shot.results.json").read_text())
    delta = oracle["metrics"]["spec_adherence"] - refuse["metrics"]["spec_adherence"]
    assert f"{100 * delta:+.1f} pp" in r.stdout


def test_few_shot_cli_path(out_dir):
    r = run_cli("--model", "mock:oracle", "--eval-set", str(SCENARIOS_PATH),
                "--strategy", "few_shot", "--exemplars", str(EXEMPLARS_PATH),
                "--judge", "mock:yes", "--out", str(out_dir))
    assert r.returncode == 0, r.stderr
    assert (out_dir / "mock_oracle__few_shot.results.json").exists()


def test_few_shot_without_exemplars_fails_loudly(out_dir):
    r = run_cli("--model", "mock:oracle", "--eval-set", str(SCENARIOS_PATH),
                "--strategy", "few_shot", "--judge", "mock:yes", "--out", str(out_dir))
    assert r.returncode != 0
    assert "requires --exemplars" in (r.stderr + r.stdout)


def test_failure_breakdown_reaches_the_printed_table(out_dir):
    r = run_cli("--model", "mock:format_breaker", "--eval-set", str(SCENARIOS_PATH),
                "--judge", "mock:yes", "--out", str(out_dir))
    assert r.returncode == 0, r.stderr
    assert "malformed=32" in r.stdout


def test_ablation_runner_sweeps_every_cell(tmp_path):
    """run_ablation.py is the frozen Prompt-Ceiling driver: models x strategies,
    one shared rubric, a markdown table for the Architecture Defense."""
    out = tmp_path / "ablation"
    r = subprocess.run(
        [sys.executable, str(REPO_ROOT / "run_ablation.py"),
         "--models", "mock:oracle,mock:always_refuse",
         "--strategies", "zero_shot,few_shot,structured",
         "--eval-set", str(SCENARIOS_PATH), "--exemplars", str(EXEMPLARS_PATH),
         "--judge", "mock:yes", "--out", str(out)],
        cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr

    summary = json.loads((out / "ablation_summary.json").read_text())
    assert len(summary["cells"]) == 6, "2 models x 3 strategies"
    assert summary["n_scenarios"] == 32
    assert summary["rubric_hash"] == "c0adb876c3e6c4c9"

    table = (out / "ablation_table.md").read_text()
    for strategy in ("zero_shot", "few_shot", "structured"):
        assert strategy in table
    assert "Failure mode that survives" in table
    for model in ("mock_oracle", "mock_always_refuse"):
        for strategy in ("zero_shot", "few_shot", "structured"):
            assert (out / f"{model}__{strategy}.transcripts.jsonl").exists()
