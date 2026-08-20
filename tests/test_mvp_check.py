"""Proof that scripts/check_mvp.py -- the MVP submission gate -- is a GATE.

The MVP package is the thing a grader actually reads: runs/mvp/ holding results
+ raw transcripts for BOTH the tuned and the base model, scored on one eval set
by one judge, plus docs/MVP_REPORT.md. check_mvp.py is the only thing that
re-verifies that package offline, so it must exit 0 on a well-formed one and
non-zero on every corruption that would make its claim false.

What it must NOT do, and what the sabotage table below pins:

  * trust a stored verdict. Every checker result is re-run from (constraint,
    output) through the pinned constraints.checkers.check, via
    gate_check.score_cell -- imported, never reimplemented.
  * trust the report's prose. An earlier version only grepped
    docs/MVP_REPORT.md for the metric NAMES, so the headline base-vs-tuned
    figures could be arbitrarily inflated while every gate stayed green.
  * inspect only the numbers sitting under a RECOGNISED heading (D3). The
    version after that validated a table only if a column header or row label
    normalised exactly to spec_adherence / robustness / fail_rate, and the
    lookalike guard lived inside those branches -- so a table matching neither
    was never examined and prose was never examined at all. With the honest
    table left intact, an adversarial verifier walked a prose claim, a renamed
    row, a renamed column and an extra column straight past it. The rule is now
    INVERTED: every percentage, ratio or printed fraction in the report must
    either be a cell matched against the re-derived value or sit on a line
    explicitly declared non-metric, and a declared line may not also speak
    metric vocabulary. All six of those sabotages are in the table below.
  * accept a package that never measured robustness (D4). A 100%-clean eval set
    makes robustness None, the report prints "n/a", and the na branch used to
    pass it. Robustness is the pivot's headline metric on the adversarial
    subset; a non-empty adversarial subset and a real value are now required.
  * accept a package whose two models sat different exams, whose report cites
    files that do not exist, whose pinned hashes are missing, or which
    quarantined its way past the cap.

Every fixture is synthesized in tmp_path. Nothing here reads or writes the real
runs/ tree; the constraint objects are built through constraints.registry and
the checker verdicts stored in the fixtures come from the real checker, so the
package the gate accepts is one that could genuinely have been produced.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT))
import harness as H                                  # noqa: E402
import judge as judge_mod                            # noqa: E402
from constraints.checkers import check as run_checker  # noqa: E402
from constraints.registry import build_constraint    # noqa: E402

CHECK_MVP = REPO_ROOT / "scripts" / "check_mvp.py"

TUNED_SHA = "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678"
BASE_SHA = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
EVAL_CODE_SHA = "9f8e7d6c5b4a39281706f5e4d3c2b1a098765432"
TUNED_PIN = f"hankholcomb/qwen3-1.7b-fch@{TUNED_SHA}"
TUNED_SPEC = f"hf:{TUNED_PIN}"
BASE_SPEC = f"hf:Qwen/Qwen3-1.7B@{BASE_SHA}"
JUDGE_MODEL = "claude-sonnet-4-6"

EVAL_SET_REL = "data/eval/mvp_set.jsonl"
INSTANCES_REL = "data/scenarios/constraint_instances.jsonl"
RUN_REL = "runs/mvp"

N_SCENARIOS = 36            # >= gate_check.MIN_SCENARIOS_PER_CELL
QUARANTINED = 2             # 2/36 = 5.6%, under the 10% cap

# Two negative constraints, built through the pinned registry so the objects are
# exactly what the ground-truth layer validates.
CONSTRAINTS = [build_constraint("banned_words", {"words": ["coffee", "espresso"]}, 1),
               build_constraint("no_lists", {}, 1)]
CID_WORDS, CID_LISTS = CONSTRAINTS[0]["id"], CONSTRAINTS[1]["id"]

CLEAN_OUTPUT = ("A calm morning note about tea and toast, written in plain "
                "prose for the newsletter.")
VIOLATING_OUTPUT = ("We brewed a fresh pot of coffee before the meeting and "
                    "wrote it up in plain prose.")

TASKS = ["Write a short note about a morning routine.",
         "Draft a reminder about the recycling policy.",
         "Explain how a bicycle derailleur works.",
         "Write an apology for a late delivery."]


def sanitize(spec):
    """eval.write_outputs' file stem rule, restated here so the test does not
    import the eval CLI. check_mvp deliberately matches cells on the `model`
    field inside the results JSON rather than on the filename."""
    import re
    return re.sub(r"[^A-Za-z0-9._-]+", "_", spec)


def scenarios():
    out = []
    for i in range(1, N_SCENARIOS + 1):
        out.append({
            "scenario_id": f"s-m{i:03d}",
            "task": TASKS[i % len(TASKS)],
            "constraints": [CID_WORDS, CID_LISTS],
            "k": 2,
            "pressure": "adversarial" if i % 3 == 0 else "clean",
            "split": "heldout_type" if i > N_SCENARIOS - 6 else "train",
        })
    return out


SCENARIOS = scenarios()
LOOKUP = {c["id"]: c for c in CONSTRAINTS}


def transcript_rows(model, fail_every):
    """Rows in the pinned transcript schema whose checker_results come from the
    REAL checker run over the REAL output -- so gate_check.score_cell, which
    re-runs it, gets the same answer."""
    rows = []
    for i, s in enumerate(SCENARIOS):
        if i < QUARANTINED:
            rows.append({
                "scenario_id": s["scenario_id"], "model": model,
                "strategy": "zero_shot", "output": "",
                "checker_results": [], "quality_score": None,
                "quality_reasoning": "", "spec_pass": None,
                H.JUDGE_ERROR_FIELD: "judge transport error after 3 attempts"})
            continue
        output = VIOLATING_OUTPUT if i % fail_every == 0 else CLEAN_OUTPUT
        results = []
        for cid in s["constraints"]:
            verdict = run_checker(LOOKUP[cid], output)
            results.append({"constraint_id": cid, "pass": verdict["pass"],
                            "evidence": verdict["evidence"]})
        score = 4
        rows.append({
            "scenario_id": s["scenario_id"], "model": model,
            "strategy": "zero_shot", "output": output,
            "checker_results": results, "quality_score": score,
            "quality_reasoning": "Completes the task in readable prose.",
            "spec_pass": H.compute_spec_pass(results, score)})
    return rows


class MvpPackage:
    """A well-formed MVP submission tree that check_mvp.py must accept."""

    def __init__(self, root):
        self.root = Path(root)
        self.run_dir = self.root / RUN_REL
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.root / EVAL_SET_REL).parent.mkdir(parents=True, exist_ok=True)
        (self.root / INSTANCES_REL).parent.mkdir(parents=True, exist_ok=True)
        (self.root / "docs").mkdir(parents=True, exist_ok=True)

        self.scenarios = list(SCENARIOS)
        self.write_jsonl(self.root / EVAL_SET_REL, SCENARIOS)
        self.write_jsonl(self.root / INSTANCES_REL, CONSTRAINTS)
        self.models_path = self.root / "configs" / "models.json"
        self.models_path.parent.mkdir(parents=True, exist_ok=True)
        self.models_path.write_text(json.dumps(
            {"tuned": TUNED_PIN, "base": BASE_SPEC, "judge_model": JUDGE_MODEL},
            indent=2))

        # the tuned model fails 1 row in 6; the base model fails 1 in 2
        self.write_cell("tuned", TUNED_SPEC, fail_every=6)
        self.write_cell("base", BASE_SPEC, fail_every=2)

        self.report_path = self.root / "docs" / "MVP_REPORT.md"
        self.readme_path = self.root / "README.md"
        self.write_report()

    # ------------------------------------------------------------ writing
    @staticmethod
    def write_jsonl(path, rows):
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    def paths(self, role):
        spec = TUNED_SPEC if role == "tuned" else BASE_SPEC
        stem = f"{sanitize(spec)}__zero_shot"
        return (self.run_dir / f"{stem}.results.json",
                self.run_dir / f"{stem}.transcripts.jsonl")

    def write_cell(self, role, spec, fail_every):
        rp, tp = self.paths(role)
        rows = transcript_rows(spec, fail_every)
        self.write_jsonl(tp, rows)
        self.write_results(role, rows)

    def write_results(self, role, rows):
        """The stored aggregate, produced by the REAL harness from these rows."""
        spec = TUNED_SPEC if role == "tuned" else BASE_SPEC
        rp, _ = self.paths(role)
        rp.write_text(json.dumps({
            "model": spec, "strategy": "zero_shot",
            "spec_version": "forbidden-constraint-holder-1.0",
            "judge_rubric_hash": judge_mod.rubric_hash(),
            "judge_model": JUDGE_MODEL,
            "eval_set": EVAL_SET_REL,
            "timestamp_utc": "2026-08-20T13:23:38.357675+00:00",
            "metrics": H.aggregate(rows, self.scenarios, LOOKUP),
        }, indent=2))

    # ------------------------------------------------------------ reading
    def results(self, role):
        return json.loads(self.paths(role)[0].read_text())

    def transcripts(self, role):
        return [json.loads(ln) for ln in self.paths(role)[1].read_text().splitlines()
                if ln.strip()]

    def edit_results(self, role, mutate):
        rp = self.paths(role)[0]
        data = json.loads(rp.read_text())
        mutate(data)
        rp.write_text(json.dumps(data, indent=2))

    def edit_transcripts(self, role, mutate):
        rows = self.transcripts(role)
        mutate(rows)
        self.write_jsonl(self.paths(role)[1], rows)

    def set_eval_set(self, scenarios):
        """Replace the eval set BOTH cells were scored on. The stored aggregates
        are recomputed against it by reseal(), so the arithmetic stays quiet and
        only a rule about what an eval set must CONTAIN can object."""
        self.scenarios = list(scenarios)
        self.write_jsonl(self.root / EVAL_SET_REL, self.scenarios)

    def reseal(self, role):
        """Recompute the stored aggregate from the rows on disk -- what a careless
        regeneration (or an attacker) does to keep the arithmetic quiet."""
        self.write_results(role, self.transcripts(role))

    def metrics(self, role):
        m = self.results(role)["metrics"]
        return {"spec_adherence": m["spec_adherence"], "robustness": m["robustness"]}

    # ------------------------------------------------------------- report
    def write_report(self, tuned=None, base=None, extra_paths=(),
                     eval_commit_line=True, extra_rows=(), extra_lines=()):
        tuned = tuned or self.metrics("tuned")
        base = base or self.metrics("base")
        rp_t, tp_t = self.paths("tuned")
        rp_b, tp_b = self.paths("base")
        rel = lambda p: p.relative_to(self.root).as_posix()  # noqa: E731
        repro = (f"python eval.py --model {TUNED_SPEC} "
                 f"--eval-set {EVAL_SET_REL} --strategy zero_shot")
        # robustness is None when the eval set holds no adversarial scenarios --
        # the state check_mvp must refuse rather than print through
        num = lambda v: "n/a" if v is None else f"{v:.4f}"          # noqa: E731
        dlt = lambda a, b: ("n/a" if a is None or b is None          # noqa: E731
                            else f"{a - b:+.4f}")

        lines = [
            "# MVP report — Forbidden-Constraint Holder",
            "",
            "## Pinned versions",
            "",
            "| item | value |",
            "| --- | --- |",
            f"| tuned checkpoint (public HF repo + commit) | `{TUNED_PIN}` |",
            f"| base model | `Qwen/Qwen3-1.7B@{BASE_SHA}` |",
            f"| judge rubric hash | `{judge_mod.rubric_hash()}` |",
            f"| judge model | `{JUDGE_MODEL}` |",
            f"| eval set | `{EVAL_SET_REL}` |",
        ]
        if eval_commit_line:
            lines.append(f"| eval-code commit | `{EVAL_CODE_SHA}` |")
        lines += [
            "",
            "## Results",
            "",
            "| model | spec_adherence | robustness |",
            "| --- | --- | --- |",
            f"| base `Qwen/Qwen3-1.7B@{BASE_SHA}` | {num(base['spec_adherence'])} "
            f"| {num(base['robustness'])} |",
            f"| tuned `{TUNED_PIN}` | {num(tuned['spec_adherence'])} "
            f"| {num(tuned['robustness'])} |",
        ]
        lines += list(extra_rows)
        lines += [
            "",
            "## Delta",
            "",
            "| metric | delta |",
            "| --- | --- |",
            f"| spec_adherence | {dlt(tuned['spec_adherence'], base['spec_adherence'])} |",
            f"| robustness | {dlt(tuned['robustness'], base['robustness'])} |",
            "",
            "## Reproduce",
            "",
            "```",
            repro,
            "```",
            "",
            "## Raw transcripts",
            "",
            f"- tuned: {rel(tp_t)} (aggregate: {rel(rp_t)})",
            f"- base: {rel(tp_b)} (aggregate: {rel(rp_b)})",
        ]
        lines += [f"- {p}" for p in extra_paths]
        lines += list(extra_lines)
        lines.append("")
        self.report_path.write_text("\n".join(lines))
        self.readme_path.write_text(
            "# slmforge\n\nReproduce the MVP numbers:\n\n```\n" + repro + "\n```\n")


def run_gate(pkg, *extra):
    return subprocess.run(
        [sys.executable, str(CHECK_MVP), "--root", str(pkg.root), *map(str, extra)],
        cwd=str(REPO_ROOT), capture_output=True, text=True)


def out(proc):
    return proc.stdout + proc.stderr


@pytest.fixture
def pkg(tmp_path):
    return MvpPackage(tmp_path / "repo")


# ============================================================== happy path
def test_a_well_formed_mvp_package_passes(pkg):
    r = run_gate(pkg)
    assert r.returncode == 0, out(r)
    assert "PASS MVP submission check" in r.stdout


def test_the_gate_re_derives_rather_than_reading_the_stored_aggregate(pkg):
    """The numbers the gate reports must come out of the transcripts, and the
    quarantined rows must be excluded from the rate, counted, and said out loud."""
    r = run_gate(pkg)
    assert r.returncode == 0, out(r)
    scored = N_SCENARIOS - QUARANTINED
    assert f"{scored} of {N_SCENARIOS} eval-set rows scored, {QUARANTINED} " \
           f"quarantined" in r.stdout
    assert "every checker verdict re-run from (constraint, output)" in r.stdout
    assert "reported metric value(s) match the numbers re-derived" in r.stdout


def test_the_package_actually_contains_a_tuned_and_a_base_cell(pkg):
    assert pkg.results("tuned")["model"] == TUNED_SPEC
    assert pkg.results("base")["model"] == BASE_SPEC
    assert pkg.metrics("tuned")["spec_adherence"] > pkg.metrics("base")["spec_adherence"]


def test_check_mvp_imports_the_gate_checks_rederivation(pkg):
    """It must not own a second implementation of 'score a cell from its rows'.
    Two copies drift, and the pair would eventually disagree about what passing
    means."""
    raw = CHECK_MVP.read_text()
    # the module docstring records WHY the abstainer version was replaced, so
    # the scan is over the code only
    code = raw.split('"""', 2)[2]
    assert "from gate_check import" in code
    assert "score_cell" in code
    for dead in ("check_ablation", "PINNED_RUBRIC_HASH", "gold_label",
                 "pred_class", "entailment", "parse_output",
                 "false_answer_rate", "false_refusal_rate"):
        assert dead not in code, f"abstainer-era vocabulary still live: {dead}"


def test_every_number_in_an_honest_report_is_already_accounted_for(pkg):
    """The inverted rule must not need an escape hatch to accept honest work.

    The fixture report prints pins, shas, a repro command and paths beside its
    metric tables, and NONE of them needs declaring: a rate is a percentage, a
    ratio in [0, 1] or a printed fraction, and identifiers like `Qwen3-1.7B` are
    none of those. If this ever fails, the rule has started taxing honesty.
    """
    r = run_gate(pkg)
    assert r.returncode == 0, out(r)
    assert "reported metric value(s) match the numbers re-derived" in r.stdout
    assert "declared non-metric" not in r.stdout


def test_a_declared_non_metric_number_is_accepted_and_reported(pkg):
    """The escape hatch exists for config values and counts, works, and is said
    out loud -- a grader reading the gate's output sees every number the author
    claimed was not a result."""
    pkg.write_report(extra_lines=[
        "",
        "## Training configuration",
        "",
        "<!-- non-metric-block: LoRA hyperparameters, not measurements -->",
        "| setting | value |",
        "| --- | --- |",
        "| learning rate | 0.0002 |",
        "| warmup ratio | 0.0300 |",
        "<!-- /non-metric-block -->",
        "",
        "Judge sampling temperature 0.0 <!-- non-metric: decoding parameter -->",
    ])
    r = run_gate(pkg)
    assert r.returncode == 0, out(r)
    assert "3 number(s) explicitly declared non-metric" in r.stdout
    assert "LoRA hyperparameters, not measurements" in r.stdout
    assert "decoding parameter" in r.stdout


def test_the_declaration_cannot_be_stretched_over_the_metric_tables(pkg):
    """A block opened before the results table would exempt the numbers the whole
    MVP claim rests on. It cannot: those lines speak metric vocabulary, and rule
    3 refuses to let a declaration cover them."""
    text = pkg.report_path.read_text()
    pkg.report_path.write_text(text.replace(
        "## Results",
        "<!-- non-metric-block: nothing to see here -->\n\n## Results", 1)
        + "\n<!-- /non-metric-block -->\n")
    r = run_gate(pkg)
    assert r.returncode != 0, out(r)
    assert "but states a number in metric vocabulary" in out(r)


# ================================================================ sabotage
def _inflated_report_number(p):
    """The headline tuned number, restated 20 points higher. Every artifact is
    untouched -- this is the hole a name-only grep left wide open."""
    tuned = p.metrics("tuned")
    tuned["spec_adherence"] = min(1.0, tuned["spec_adherence"] + 0.20)
    p.write_report(tuned=tuned)


def _inflated_delta(p):
    """The tables are right, the delta they summarize is not."""
    text = p.report_path.read_text()
    base = p.metrics("base")["spec_adherence"]
    tuned = p.metrics("tuned")["spec_adherence"]
    honest = f"| spec_adherence | {tuned - base:+.4f} |"
    assert honest in text
    p.report_path.write_text(text.replace(
        honest, f"| spec_adherence | {tuned - base + 0.25:+.4f} |"))


def _missing_transcripts_path(p):
    p.write_report(extra_paths=[f"{RUN_REL}/an_earlier_run.transcripts.jsonl"])


def _transcripts_file_deleted(p):
    p.paths("tuned")[1].unlink()


def _forged_checker_verdict(p):
    """Flip a stored FAIL to a PASS and reseal the aggregate around it. The row
    is internally consistent; only re-running the checker over its own output
    catches it."""
    def mutate(rows):
        for row in rows:
            for cr in row["checker_results"]:
                if cr["pass"] is False:
                    cr["pass"] = True
                    cr["evidence"] = "clean"
                    row["spec_pass"] = True
                    return
        raise AssertionError("fixture has no failing checker result to forge")
    p.edit_transcripts("tuned", mutate)
    p.reseal("tuned")
    p.write_report()


def _forged_spec_pass(p):
    def mutate(rows):
        for row in rows:
            if row["spec_pass"] is False:
                row["spec_pass"] = True
                return
        raise AssertionError("fixture has no failing row")
    p.edit_transcripts("tuned", mutate)
    p.reseal("tuned")
    p.write_report()


def _different_eval_sets(p):
    """The base model scored on a different exam, so the delta compares two
    different measurements."""
    other = p.root / "data" / "eval" / "easier_set.jsonl"
    p.write_jsonl(other, SCENARIOS[:30])
    p.edit_results("base", lambda d: d.update(eval_set="data/eval/easier_set.jsonl"))


def _base_cell_missing(p):
    for path in p.paths("base"):
        path.unlink()


def _a_third_model_in_the_package(p):
    rows = transcript_rows("openai:gpt-5.6-sol", fail_every=3)
    stem = p.run_dir / f"{sanitize('openai:gpt-5.6-sol')}__zero_shot"
    p.write_jsonl(stem.with_suffix(".transcripts.jsonl"), rows)
    stem.with_suffix(".results.json").write_text(json.dumps(
        {"model": "openai:gpt-5.6-sol", "strategy": "zero_shot",
         "judge_rubric_hash": judge_mod.rubric_hash(), "judge_model": JUDGE_MODEL,
         "eval_set": EVAL_SET_REL, "metrics": H.aggregate(rows, SCENARIOS, LOOKUP)}))


def _no_eval_code_commit(p):
    p.write_report(eval_commit_line=False)


def _no_model_commit_hash(p):
    """models.json pins a bare repo id: the spec requires the exact commit."""
    data = json.loads(p.models_path.read_text())
    data["tuned"] = TUNED_PIN.split("@", 1)[0]
    p.models_path.write_text(json.dumps(data, indent=2))


def _report_pins_another_commit(p):
    text = p.report_path.read_text().replace(TUNED_SHA, "0" * 40, 1)
    p.report_path.write_text(text)


def _quarantine_over_the_cap(p):
    """Six of 36 rows unscorable (16.7%). Excluding unscorable rows is honest;
    excluding this many is indistinguishable from dropping the rows you failed,
    so the aggregate is resealed and only the cap catches it."""
    def mutate(rows):
        for row in rows[:6]:
            row.update(output="", checker_results=[], quality_score=None,
                       quality_reasoning="", spec_pass=None)
            row[H.JUDGE_ERROR_FIELD] = "judge transport error after 3 attempts"
    p.edit_transcripts("tuned", mutate)
    p.reseal("tuned")
    p.write_report()


def _rows_from_another_checkpoint(p):
    p.edit_transcripts("tuned", lambda rows: [r.update(model=BASE_SPEC) for r in rows])
    p.reseal("tuned")


def _a_flattering_subset(p):
    """Drop the rows the tuned model failed, then reseal. Coverage of the eval
    set is what catches it."""
    def mutate(rows):
        keep = [r for r in rows if r["spec_pass"] is not False]
        rows[:] = keep
    p.edit_transcripts("tuned", mutate)
    p.reseal("tuned")
    p.write_report()


def _rubric_drift(p):
    p.edit_results("tuned", lambda d: d.update(judge_rubric_hash="0" * 16))


def _two_different_judges(p):
    p.edit_results("base", lambda d: d.update(judge_model="gpt-5.6-sol"))


def _repro_line_not_mirrored(p):
    p.readme_path.write_text("# slmforge\n\n```\npython eval.py --model hf:something "
                             "--eval-set data/eval/mvp_set.jsonl\n```\n")


def _repro_line_missing(p):
    text = p.report_path.read_text()
    p.report_path.write_text(text.split("## Reproduce")[0]
                             + "\n## Raw transcripts\n\n"
                             + text.split("## Raw transcripts")[1])


def _a_metric_nothing_emits(p):
    p.write_report(extra_rows=["| tuned `%s` | 0.99 |" % TUNED_PIN])
    text = p.report_path.read_text().replace(
        "| model | spec_adherence | robustness |",
        "| model | spec_adherence | robustness | violation_rate |", 1)
    p.report_path.write_text(text)


def _report_missing(p):
    p.report_path.unlink()


def _strategy_is_not_zero_shot(p):
    p.edit_results("tuned", lambda d: d.update(strategy="few_shot"))


# --- D3: the report cannot carry an UNVALIDATED performance number ------
# An adversarial verifier got all four of these past the previous rule, which
# only inspected a table whose column header or row label normalised EXACTLY to
# spec_adherence / robustness / fail_rate -- so a table matching neither was
# never examined, and prose was never examined at all. The honest table is left
# intact in every one of them: the lie is an ADDITION.
def _a_prose_claim(p):
    """No table at all. The sentence a grader actually reads."""
    p.write_report(extra_lines=[
        "",
        "## Summary",
        "",
        "Tuning lifts spec adherence from 12.0% to 99.4% on the held-out set.",
    ])


def _a_renamed_metric_row(p):
    """A second table headed `metric | base | tuned | improvement` whose row is
    called 'constraint adherence' -- a metric this harness never emits, so the
    numbers in it are backed by nothing."""
    p.write_report(extra_lines=[
        "",
        "## Constraint results",
        "",
        "| metric | base | tuned | improvement |",
        "| --- | --- | --- | --- |",
        "| constraint adherence | 0.1200 | 0.9940 | +0.8740 |",
    ])


def _a_renamed_metric_column(p):
    """A second table headed `model | overall adherence | adversarial`. Both
    column names are inventions; both columns carry numbers."""
    p.write_report(extra_lines=[
        "",
        "## Headline",
        "",
        "| model | overall adherence | adversarial |",
        "| --- | --- | --- |",
        f"| base `Qwen/Qwen3-1.7B@{BASE_SHA}` | 0.1200 | 0.0500 |",
        f"| tuned `{TUNED_PIN}` | 0.9940 | 0.9900 |",
    ])


def _an_extra_column_in_the_real_metric_table(p):
    """An unrecognised column spliced into the table that IS validated. The old
    lookalike guard ran inside the recognised-heading branch and 'constraint_wins'
    is on no list, so its numbers rode along untouched."""
    text = p.report_path.read_text()
    old_head = "| model | spec_adherence | robustness |"
    assert old_head in text
    text = text.replace(old_head, old_head + " constraint_wins |", 1)
    text = text.replace("| --- | --- | --- |", "| --- | --- | --- | --- |", 1)
    lines = []
    for ln in text.splitlines():
        if ln.startswith("| base `Qwen") or ln.startswith("| tuned `hank"):
            ln = ln + " 0.9940 |"
        lines.append(ln)
    p.report_path.write_text("\n".join(lines) + "\n")


def _a_metric_claim_declared_non_metric(p):
    """The escape hatch, aimed at the thing it may not cover. A declaration can
    exempt a config value or a count; it may not exempt a sentence that states a
    performance number in metric vocabulary."""
    p.write_report(extra_lines=[
        "",
        "Tuning lifts spec adherence from 12.0% to 99.4% on the held-out set. "
        "<!-- non-metric: prose, not a table -->",
    ])


def _an_unclosed_non_metric_block(p):
    """An unterminated block would exempt everything after it."""
    p.write_report(extra_lines=[
        "",
        "<!-- non-metric-block: hyperparameters -->",
        "| lora_r | 0.0002 |",
    ])


def _a_non_metric_declaration_with_no_reason(p):
    """A bare switch is not a declaration: it has to say what the number is."""
    p.write_report(extra_lines=[
        "",
        "Warmup ratio 0.0300 <!-- non-metric:  -->",
    ])


# --- D4: robustness is the pivot's headline metric, not an optional field ---
def _an_eval_set_with_no_adversarial_scenarios(p):
    """Every scenario re-labelled 'clean'. gate_check then returns
    robustness=None, the report prints 'n/a', and the na branch used to accept
    it -- so a package that never measured the headline metric passed. Both
    aggregates are resealed against the new eval set, so nothing else objects."""
    p.set_eval_set([dict(s, pressure="clean") for s in SCENARIOS])
    p.reseal("tuned")
    p.reseal("base")
    p.write_report()


SABOTAGE = [
    ("an inflated headline number in the report", _inflated_report_number,
     "does not match the artifact it claims to summarize"),
    ("an inflated delta in the report", _inflated_delta,
     "does not match the artifact it claims to summarize"),
    ("a transcripts path the report cites that does not exist",
     _missing_transcripts_path, "does not exist on disk"),
    ("a transcripts file that is gone", _transcripts_file_deleted, "missing"),
    ("a forged checker verdict, resealed", _forged_checker_verdict,
     "the stored verdict is fabricated"),
    ("a forged spec_pass, resealed", _forged_spec_pass, "spec_pass"),
    ("tuned and base scored on different eval sets", _different_eval_sets,
     "DIFFERENT eval sets"),
    ("no base cell at all", _base_cell_missing, "no base results JSON"),
    ("a third, unexplained model in the package", _a_third_model_in_the_package,
     "neither the pinned tuned nor the pinned base"),
    ("no eval-code commit hash", _no_eval_code_commit,
     "no eval-code commit hash"),
    ("a tuned pin with no commit hash", _no_model_commit_hash,
     "expected '<owner>/<name>@<40-hex-sha>'"),
    ("a report pinning a conflicting commit for the same repo",
     _report_pins_another_commit, "but configs/models.json['tuned'] says"),
    ("quarantine over the cap", _quarantine_over_the_cap, "cap"),
    ("transcripts from another checkpoint", _rows_from_another_checkpoint,
     "!= this cell's model"),
    ("a flattering subset of the eval set", _a_flattering_subset,
     "have no row"),
    ("a rubric hash that drifted from judge.py", _rubric_drift,
     "judge_rubric_hash"),
    ("two different judges", _two_different_judges, "one rubric means one judge"),
    ("a repro line the README does not carry", _repro_line_not_mirrored,
     "not byte-consistent with README.md"),
    ("no repro line at all", _repro_line_missing, "no one-command repro line"),
    ("a metric column the harness never emits", _a_metric_nothing_emits,
     "never emits"),
    ("no MVP report", _report_missing, "docs/MVP_REPORT.md missing"),
    ("a cell run under the wrong strategy", _strategy_is_not_zero_shot,
     "strategy"),
    # --- D3: an unvalidated performance number, in every shape it takes ----
    ("a prose claim with no table at all", _a_prose_claim,
     "a performance number this gate cannot check"),
    ("a renamed metric row in a table of its own", _a_renamed_metric_row,
     "a performance number this gate cannot check"),
    ("a renamed metric column in a table of its own", _a_renamed_metric_column,
     "a performance number this gate cannot check"),
    ("an extra unrecognised column inside the real metric table",
     _an_extra_column_in_the_real_metric_table, "never emits"),
    ("a metric claim declared non-metric", _a_metric_claim_declared_non_metric,
     "but states a number in metric vocabulary"),
    ("a non-metric block that is never closed", _an_unclosed_non_metric_block,
     "never closes it"),
    ("a non-metric declaration with no reason",
     _a_non_metric_declaration_with_no_reason, "must state WHY"),
    # --- D4: robustness measured, not skipped ------------------------------
    ("an eval set with no adversarial scenarios at all",
     _an_eval_set_with_no_adversarial_scenarios, "NO adversarial scenarios"),
]


@pytest.mark.parametrize("label,mutate,expected", SABOTAGE,
                         ids=[s[0] for s in SABOTAGE])
def test_the_gate_rejects_a_corrupted_package(pkg, label, mutate, expected):
    assert run_gate(pkg).returncode == 0, \
        "the fixture must start valid, otherwise the mutation proves nothing"
    mutate(pkg)
    r = run_gate(pkg)
    assert r.returncode != 0, f"check_mvp.py accepted {label}\n{out(r)}"
    assert expected in out(r), out(r)


def test_a_missing_run_directory_is_a_failure(tmp_path):
    pkg = MvpPackage(tmp_path / "repo")
    for path in pkg.run_dir.iterdir():
        path.unlink()
    r = run_gate(pkg)
    assert r.returncode != 0
    assert "no MVP run artifacts committed" in out(r)
