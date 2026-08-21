"""The gate now guards TWO frozen packages, and the flags that select them.

`tests/test_mvp_check.py` covers the re-derivation itself, but every one of its
cases invokes the gate with `--root <tmp>` and nothing else — the default path.
When the repo grew a second submission package the flags that pick between them
had no coverage at all, and the default invocation silently came to mean an
incoherent triple: the MVP report, the MVP run directory, and the pin of a
checkpoint no file in that directory holds.

Every assertion here binds one of those: a default that names a package that
does not exist, a flag combination that gates a report against the wrong
checkpoint, or a PASS line that attests to a package it did not examine.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECK_MVP = REPO_ROOT / "scripts" / "check_mvp.py"


def gate(*args, cwd=None):
    return subprocess.run([sys.executable, str(CHECK_MVP), *map(str, args)],
                          cwd=str(cwd or REPO_ROOT), capture_output=True, text=True)


def out(r):
    return r.stdout + r.stderr


@pytest.fixture(scope="module")
def models_json():
    return json.loads((REPO_ROOT / "configs" / "models.json").read_text())


# ------------------------------------------------------- the two packages
def test_the_bare_default_command_still_gates_the_mvp_package():
    """THE ONE THAT MATTERS. `python scripts/check_mvp.py` with no flags is the
    documented verify step. When 'tuned' was repointed to a newer checkpoint it
    began pairing the MVP report and runs/mvp with a pin those transcripts do
    not hold, and the command that had always passed started failing."""
    r = gate()
    assert r.returncode == 0, out(r)
    assert "PASS MVP submission check" in r.stdout
    assert "pin=tuned_mvp" in r.stdout and "run=runs/mvp" in r.stdout


def test_naming_the_final_report_selects_the_final_pin():
    r = gate("--report", "docs/FINAL_REPORT.md", "--run-dir", "runs/final")
    assert r.returncode == 0, out(r)
    assert "PASS FINAL submission check" in r.stdout
    assert "pin=tuned" in r.stdout and "run=runs/final" in r.stdout


def test_the_banner_names_the_pin_and_run_dir_not_just_the_filename(tmp_path):
    """A PASS line is quoted as evidence, so it has to say what was checked.
    Deriving it from the report's filename alone let a copy of one report,
    renamed, print a banner for a package it had not gated."""
    r = gate("--report", "docs/FINAL_REPORT.md", "--run-dir", "runs/final")
    banner = [l for l in r.stdout.splitlines() if l.startswith("PASS")][0]
    assert "pin=tuned" in banner
    assert "run=runs/final" in banner


def test_the_two_packages_pin_different_checkpoints(models_json):
    """If these ever collapse to one value the split-package arrangement is
    pointless and one of the two reports is describing the wrong model."""
    assert models_json["tuned"] != models_json["tuned_mvp"]
    assert models_json["tuned"].split("@")[0] == models_json["tuned_mvp"].split("@")[0]


# ------------------------------------------------------------- refusals
def test_a_pin_the_run_dir_does_not_hold_is_refused():
    """The cross-product must fail: the final pin against the MVP transcripts."""
    r = gate("--pin-key", "tuned", "--report", "docs/MVP_REPORT.md",
             "--run-dir", "runs/mvp")
    assert r.returncode != 0
    assert "neither the pinned tuned nor the pinned base" in out(r) \
        or "no tuned results JSON" in out(r)


def test_the_reverse_cross_product_is_also_refused():
    r = gate("--pin-key", "tuned_mvp", "--report", "docs/FINAL_REPORT.md",
             "--run-dir", "runs/final")
    assert r.returncode != 0


def test_a_report_outside_the_root_is_a_verdict_not_a_traceback(tmp_path):
    """report_path.relative_to(root) raises on a non-descendant path. Reached
    from inside check_report it ended the run with an unhandled ValueError
    after the whole re-derivation had already been paid for."""
    outside = tmp_path / "elsewhere.md"
    outside.write_text("# not part of this package\n")
    r = gate("--report", str(outside), "--run-dir", "runs/final")
    assert r.returncode != 0
    assert "Traceback" not in out(r), out(r)
    assert "not inside --root" in out(r)


def test_a_run_dir_outside_the_root_is_a_verdict_not_a_traceback(tmp_path):
    r = gate("--report", "docs/FINAL_REPORT.md", "--run-dir", str(tmp_path))
    assert r.returncode != 0
    assert "Traceback" not in out(r), out(r)


def test_a_missing_report_is_named_the_way_the_rest_of_the_gate_names_it(tmp_path):
    r = gate("--report", "docs/NOT_A_REPORT.md", "--run-dir", "runs/final")
    assert r.returncode != 0
    assert "docs/NOT_A_REPORT.md missing" in out(r)


# ------------------------------------------------------------ emit path
def test_emit_metrics_does_not_require_the_report_to_exist_yet():
    """--emit-metrics produces the block an author pastes INTO the report.
    Requiring the report first inverts the only workflow it has."""
    r = gate("--emit-metrics", "--report", "docs/DOES_NOT_EXIST_YET.md",
             "--run-dir", "runs/final", "--pin-key", "tuned")
    assert r.returncode == 0, out(r)
    assert r.stdout.startswith("<!-- BEGIN GENERATED METRICS -->")
    assert r.stdout.rstrip().endswith("<!-- END GENERATED METRICS -->")


def test_emit_metrics_for_each_package_produces_that_package_s_numbers():
    """The two packages must not emit the same block, or one report is
    documenting the other's checkpoint."""
    mvp = gate("--emit-metrics")
    final = gate("--emit-metrics", "--report", "docs/FINAL_REPORT.md",
                 "--run-dir", "runs/final")
    assert mvp.returncode == 0 and final.returncode == 0
    assert mvp.stdout != final.stdout


def test_each_report_carries_its_own_emitted_block():
    for report, run_dir in (("docs/MVP_REPORT.md", "runs/mvp"),
                            ("docs/FINAL_REPORT.md", "runs/final")):
        emitted = gate("--emit-metrics", "--report", report, "--run-dir", run_dir)
        assert emitted.returncode == 0, out(emitted)
        assert emitted.stdout.rstrip() in (REPO_ROOT / report).read_text(), report
