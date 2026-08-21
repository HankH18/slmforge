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
    row, a renamed column and an extra column straight past it.
  * try to PARSE the claim back out of prose at all (D5). The rule after that
    was inverted -- every percentage, ratio or printed fraction had to be a
    matched cell or a declared non-metric line -- and 18 adversarial report
    bodies were run through it with the honest table left intact. ALL 18
    PASSED: it knew '%' and 'pp' but not `percent`, `pct`, `p.p.`, `bps`,
    `8.3x`, `.994`, `0,994`, `99.4 %`, `**99.4**%`, `99.4&#37;`, `99.4％`,
    `146/147.`, `146 of 147` or a claim split across a newline; it never looked
    inside a table it could not orient, and a body row with no leading pipe or a
    separator row with no trailing pipe made a table parse with an EMPTY BODY;
    and its 15-word vocabulary could be worded around. The report therefore does
    not STATE its numbers any more: check_mvp.py --emit-metrics GENERATES them
    from the re-derived stats between two literal markers, the gate requires
    that block back byte for byte, and outside it the report may print no
    rate-shaped number at all unless the line is declared non-metric and speaks
    no metric vocabulary. Every one of those bypasses is in the table below,
    and MvpPackage.write_report pastes the real emitted block -- this file owns
    no second renderer that could disagree with the checker.
  * let the GENERATED-BLOCK design be worked around either (D6). Four auditors
    attacked the version above and got six fabricated, inflated bodies past it
    with the honest block left byte-perfect: an of-form ratio with one article
    in it ('146 of the 147' -- the pattern demanded the integers be ADJACENT),
    'times out of' and 'in' ratios, a whole table of such cells, a rate wearing
    a version tag ('0.994-v1' -- the identifier mask blanked the ENTIRE token,
    rate included, so every tier that reads the masked line found nothing), an
    invisible U+2066 between '99' and '%' (the blanking list was an enumeration
    of an open set), and a fabricated base-vs-tuned table wrapped in a
    `<!-- non-metric-block: ... -->` that renders as NOTHING. That last one is
    the important one: an invisible declaration costs a dishonest author zero,
    and policing it by vocabulary is the defect that killed the design before
    this one. It is policed by VISIBILITY now -- a declaration that hides a
    number must say so where the reader sees it.
  * tax honest content. The tests marked `test_honest_content_is_writable` are
    the other half of the same rule: a LoRA hyperparameter table of bare
    integers, counts and durations in prose, a thousands-separated integer, a
    prior-run ablation table and a corpus-composition footnote all have to be
    writable, because a rule that cannot express a real report gets deleted by
    the next person rather than obeyed.
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

WHAT THE GREEN HERE IS EVIDENCE OF, STATED HONESTLY. MvpPackage.write_report
pastes the block straight from `--emit-metrics`, so "the honest report passes"
is guaranteed by construction for the BLOCK. That is deliberate (this file owns
no second renderer that could disagree with the checker) but an audit named the
consequence correctly: it proves nothing about a report a PERSON writes. So
test_a_hand_authored_report_can_pass assembles one line by line -- prose, a
hyperparameter table, a corpus footnote, a repro command -- with only the block
pasted, and the honest-content table above pins the false positives that would
make such a report unwritable.
"""
import json
import re
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
        self.block = None
        self.block = self.emit_block()
        self.write_report(refresh=False)

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
    def emit_block(self, required=True):
        """The generated metrics block, taken from the CLI flag a report author
        would actually run. This file deliberately owns no renderer of its own:
        if check_mvp's renderer and its checker ever disagreed, a second copy
        here is exactly what would hide it.

        A package that no longer re-derives has no honest block to emit, and the
        author is left holding the last one -- which is what happens in real life
        when someone edits the transcripts and forgets to regenerate."""
        r = subprocess.run(
            [sys.executable, str(CHECK_MVP), "--root", str(self.root),
             "--emit-metrics"], cwd=str(REPO_ROOT), capture_output=True, text=True)
        if r.returncode != 0:
            assert not required, \
                f"--emit-metrics failed on a package that must be valid\n{out(r)}"
            return self.block
        assert r.stdout.startswith("<!-- BEGIN GENERATED METRICS -->"), out(r)
        return r.stdout.rstrip("\n")

    def write_report(self, extra_paths=(), eval_commit_line=True, extra_rows=(),
                     extra_lines=(), block=None, refresh=True):
        """The report a grader reads. Everything here is hand-written except the
        fenced metric block, which is pasted verbatim from --emit-metrics.

        `block` pastes a DIFFERENT block (the tampering the gate must catch);
        `refresh=False` re-pastes the last one without regenerating it."""
        if block is None:
            block = self.emit_block(required=False) if refresh else self.block
        self.block = block
        rp_t, tp_t = self.paths("tuned")
        rp_b, tp_b = self.paths("base")
        rel = lambda p: p.relative_to(self.root).as_posix()  # noqa: E731
        repro = (f"python eval.py --model {TUNED_SPEC} "
                 f"--eval-set {EVAL_SET_REL} --strategy zero_shot")

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
            "Every number below is generated by `check_mvp.py --emit-metrics` "
            "and re-derived from",
            "the committed transcripts; the gate refuses this report unless it "
            "can regenerate",
            "the block below byte for byte.",
            "",
        ]
        lines += block.splitlines()
        lines += list(extra_rows)
        lines += [
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


def run_emit(pkg):
    return run_gate(pkg, "--emit-metrics")


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
    # the report's numbers are the re-derived ones because they ARE the
    # re-derivation: the gate regenerates the block and demands it back verbatim
    assert ("byte-identical to the one re-derived from the committed transcripts"
            in r.stdout)
    assert f"| n_scored | {scored} | {scored} | +0 |" in pkg.block
    assert f"| n_quarantined | {QUARANTINED} | {QUARANTINED} | +0 |" in pkg.block


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
    """The sweep must not need an escape hatch to accept honest work.

    The fixture report prints pins, 40-hex shas, a rubric hash, a judge model, a
    repro command, an ISO timestamp and half a dozen paths OUTSIDE the generated
    block, and NONE of them needs declaring: a rate is a percentage, a printed
    fraction, a bare fractional number or a number wearing a rate unit, and
    `Qwen3-1.7B`, `data/eval/mvp_set.jsonl` and a commit hash are none of those.
    If this ever fails, the rule has started taxing honesty -- and a rule that
    cannot express a real report gets deleted by the next person, not obeyed.
    """
    r = run_gate(pkg)
    assert r.returncode == 0, out(r)
    assert ("byte-identical to the one re-derived from the committed transcripts"
            in r.stdout)
    # and the clean bill of health is not vacuous: the sweep really did look at
    # every line outside the block
    m = re.search(r"(\d+) line\(s\) outside the generated block swept", r.stdout)
    assert m, r.stdout
    total = len(pkg.report_path.read_text().splitlines())
    outside = total - len(pkg.block.splitlines())
    assert int(m.group(1)) == outside >= 25, (m.group(1), outside, r.stdout)
    assert "declared non-metric" not in r.stdout


DECLARED_CONFIG = [
    "",
    "## Training configuration",
    "",
    "<!-- non-metric-block: LoRA hyperparameters, not measurements -->",
    "These are the LoRA settings the run was launched with, not a measurement "
    "of either checkpoint.",
    "",
    "| setting | value |",
    "| --- | --- |",
    "| learning rate | 0.0002 |",
    "| warmup ratio | 0.0300 |",
    "<!-- /non-metric-block -->",
    "",
    "Judge sampling temperature 0.0 — a decoding parameter, not a measurement. "
    "<!-- non-metric: decoding parameter -->",
]


def test_a_declared_non_metric_number_is_accepted_and_reported(pkg):
    """The escape hatch exists for config values and counts, works, and is said
    out loud -- a grader reading the gate's output sees every number the author
    claimed was not a result."""
    pkg.write_report(extra_lines=DECLARED_CONFIG)
    r = run_gate(pkg)
    assert r.returncode == 0, out(r)
    assert "3 number(s) explicitly declared non-metric" in r.stdout
    assert "LoRA hyperparameters, not measurements" in r.stdout
    assert "decoding parameter" in r.stdout


def test_an_invisible_declaration_is_not_a_declaration(pkg):
    """`<!-- non-metric: ... -->` renders as NOTHING.

    So a hatch made only of one costs a dishonest author zero: an audit wrapped a
    whole fabricated base-vs-tuned table in one and the gate called the report
    clean while a grader read an unqualified results table. The hatch cannot be
    policed by VOCABULARY (that is the defect that destroyed the design before
    this one, and it is one rewrite away from useless), so it is policed by
    VISIBILITY: a declaration that actually hides a number has to say so where
    the reader sees it. Same numbers, same declaration, one visible sentence
    added -- refused, then accepted."""
    body = [
        "",
        "<!-- non-metric-block: LoRA hyperparameters -->",
        "| setting | value |",
        "| --- | --- |",
        "| learning rate | 0.0002 |",
        "<!-- /non-metric-block -->",
    ]
    pkg.write_report(extra_lines=body)
    r = run_gate(pkg)
    assert r.returncode != 0, out(r)
    assert "renders as NOTHING" in out(r)

    caption = "These are the LoRA settings the run was launched with, not a " \
              "measurement of either checkpoint."
    pkg.write_report(extra_lines=body[:1] + [caption, ""] + body[1:])
    r = run_gate(pkg)
    assert r.returncode == 0, out(r)
    assert "1 number(s) explicitly declared non-metric" in r.stdout


def test_a_declaration_over_prose_with_no_numbers_needs_no_disclaimer(pkg):
    """The visible-disclaimer rule fires only where a declaration actually hid
    something. Demanding one everywhere would teach authors to paste it blind,
    and a disclaimer nobody reads is worth exactly nothing."""
    pkg.write_report(extra_lines=[
        "",
        "<!-- non-metric-block: narrative, no numbers in it -->",
        "The tuned checkpoint was trained on the forbidden-constraint corpus.",
        "<!-- /non-metric-block -->"])
    r = run_gate(pkg)
    assert r.returncode == 0, out(r)


# ------------------------------------------- honest content must stay writable
# A rule that cannot express a real report gets deleted by the next person, not
# obeyed. Each body below carries NO performance claim of any kind, and each one
# was refused by the version of this gate these tests were written against.
HONEST_BODIES = [
    ("a LoRA hyperparameter table of bare integers", [
        "",
        "## Training configuration",
        "",
        "| hyperparameter | value |",
        "| --- | --- |",
        "| r | 16 |",
        "| lora_alpha | 32 |",
        "| max_seq_length | 2048 |",
        "| num_train_epochs | 3 |",
        "| warmup_steps | 500 |",
        "| seed | 42 |",
     ]),
    ("counts, sizes, durations and versions in prose", [
        "",
        "The eval set holds 147 scenarios, 48 of them adversarial, built from 881",
        "rows over 3 epochs in 500 steps with seed 1234. The adapter is 121 MB and",
        "training took 2 GPU-hours. Pinned: transformers 4.57.2, trl 0.26.0.",
     ]),
    ("a thousands-separated integer", [
        "",
        "The corpus holds 12,000 rows across 3 shards and 1,000 held-out cases.",
     ]),
    ("a prior-run ablation table, declared honestly and disclaimed visibly", [
        "",
        "## Prior evidence: the G1 ablation",
        "",
        "The table below is not a measurement of this package: it is the prior G1",
        "ablation, measured in an earlier run, that authorised this fine-tune.",
        "",
        "<!-- not-a-gate-metric-block: prior G1 ablation from an earlier run -->",
        "| model | fail_rate zero_shot | fail_rate few_shot | delta |",
        "| --- | --- | --- | --- |",
        "| `openai:gpt-5.6-sol` | 0.381 | 0.218 | 0.293 |",
        "| `anthropic:claude-x` | 0.911 | 0.702 | 0.209 |",
        "<!-- /not-a-gate-metric-block -->",
     ]),
    ("corpus composition on a declared, disclaimed line", [
        "",
        "Character constraints are 0.598 of train and 0.850 of held-out — corpus "
        "composition, not a measurement. <!-- non-metric: corpus composition -->",
     ]),
]


@pytest.mark.parametrize("label,body", HONEST_BODIES,
                         ids=[b[0] for b in HONEST_BODIES])
def test_honest_content_is_writable(pkg, label, body):
    pkg.write_report(extra_lines=body)
    r = run_gate(pkg)
    assert r.returncode == 0, f"the gate refused honest content ({label})\n{out(r)}"


def test_a_hand_authored_report_can_pass(pkg):
    """WHAT THE REST OF THIS FILE IS AND IS NOT EVIDENCE OF.

    MvpPackage.write_report pastes the block straight from `--emit-metrics`, on
    purpose -- this file owns no second renderer that could disagree with the
    checker. But the consequence is real and an audit named it: the positive
    control is circular by construction, so nothing above proves that a report a
    PERSON writes can pass. The block has to be pasted (that is the design), but
    everything around it does not.

    So this report is assembled here, line by line, the way an author would write
    one: prose, a hyperparameter table, a prior-run ablation, a corpus footnote,
    a repro command -- with the emitted block dropped in the middle and nothing
    else copied from the fixture's writer."""
    tp_t, tp_b = pkg.paths("tuned")[1], pkg.paths("base")[1]
    rel = lambda p: p.relative_to(pkg.root).as_posix()          # noqa: E731
    repro = (f"python eval.py --model {TUNED_SPEC} "
             f"--eval-set {EVAL_SET_REL} --strategy zero_shot")
    report = "\n".join([
        "# Forbidden-Constraint Holder — MVP",
        "",
        "We fine-tuned `Qwen/Qwen3-1.7B` to hold hard output constraints, and",
        f"published the result at `{TUNED_PIN}`. The base model it is compared",
        f"against is `Qwen/Qwen3-1.7B@{BASE_SHA}`; both checkpoints sat the same",
        f"exam, `{EVAL_SET_REL}`, judged by `{JUDGE_MODEL}` under rubric hash",
        f"`{judge_mod.rubric_hash()}`.",
        "",
        f"The eval code is commit `{EVAL_CODE_SHA}`.",
        "",
        "## What the numbers are",
        "",
        "Every figure below is generated by the gate from the committed",
        "transcripts. spec_adherence is the share of scenarios where every",
        "constraint held and the judge scored the task done; robustness is the",
        "same measure restricted to the adversarial subset.",
        "",
        *pkg.block.splitlines(),
        "",
        "## How it was trained",
        "",
        "The settings below are the ones the run was launched with — a",
        "configuration, not a measurement.",
        "",
        "<!-- non-metric-block: LoRA hyperparameters -->",
        "| hyperparameter | value |",
        "| --- | --- |",
        "| r | 16 |",
        "| lora_alpha | 32 |",
        "| learning_rate | 0.0002 |",
        "| num_train_epochs | 3 |",
        "| seed | 42 |",
        "<!-- /non-metric-block -->",
        "",
        "Character constraints are 0.598 of the training split against 0.850 of",
        "the held-out split — a property of the corpus, not a measured result.",
        "<!-- non-metric: corpus composition -->",
        "",
        "## Reproduce",
        "",
        "```",
        repro,
        "```",
        "",
        "## Raw judge transcripts",
        "",
        f"- tuned: {rel(tp_t)}",
        f"- base: {rel(tp_b)}",
        "",
    ])
    pkg.report_path.write_text(report)
    r = run_gate(pkg)
    assert r.returncode == 0, \
        f"a hand-authored report cannot pass this gate:\n{out(r)}"
    assert "byte-identical to the one re-derived" in r.stdout
    # ...and the clean bill is not vacuous: the sweep really read the prose
    m = re.search(r"(\d+) line\(s\) outside the generated block swept", r.stdout)
    assert m and int(m.group(1)) >= 30, r.stdout


def test_the_declaration_cannot_be_stretched_over_the_generated_block(pkg):
    """A non-metric block opened before the results would exempt the numbers the
    whole MVP claim rests on. It cannot -- and now for a structural reason rather
    than a lexical one. The old rule protected those lines only because they
    happened to speak metric vocabulary, so renaming the rows would have freed
    them; the generated block may not sit inside a declaration AT ALL, whatever
    its rows are called, because the block IS the measurement."""
    text = pkg.report_path.read_text()
    pkg.report_path.write_text(text.replace(
        "## Results",
        "<!-- non-metric-block: nothing to see here -->\n\n## Results", 1)
        + "\n<!-- /non-metric-block -->\n")
    r = run_gate(pkg)
    assert r.returncode != 0, out(r)
    assert "the generated metrics block sits inside a non-metric declaration" \
        in out(r)
    assert "nothing to see here" in out(r)


# ================================================================ sabotage
def block_row(block, metric):
    """The generated block's row for one metric, as [label, base, tuned, delta]
    cells. The tampering below edits the block where a dishonest author would --
    inside the only region of the report allowed to state a number."""
    row = next(ln for ln in block.splitlines()
               if ln.startswith(f"| {metric} |"))
    return row, row.split("|")


def _inflated_report_number(p):
    """The headline tuned number, restated 20 points higher INSIDE the generated
    block -- the only place the report may state a number at all. Every artifact
    is untouched: this is the hole a name-only grep left wide open, and the hole
    a parser could always be worded around."""
    row, cells = block_row(p.block, "spec_adherence")
    cells[3] = f" {min(1.0, float(cells[3]) + 0.20):.4f} "
    p.write_report(block=p.block.replace(row, "|".join(cells)), refresh=False)


def _inflated_delta(p):
    """The base and tuned columns are right, the delta they summarize is not."""
    row, cells = block_row(p.block, "spec_adherence")
    cells[4] = f" {float(cells[4]) + 0.25:+.4f} "
    p.write_report(block=p.block.replace(row, "|".join(cells)), refresh=False)


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
    """A metric this harness never emits, in a table of its own next to the
    block. The old rule refused the COLUMN NAME (a whitelist a rename walks
    past); the number is what is refused now, whatever it is filed under."""
    p.write_report(extra_rows=[
        "",
        "| model | violation_rate |",
        "| --- | --- |",
        f"| tuned `{TUNED_PIN}` | 0.99 |"])


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
    """An unrecognised column spliced INTO the generated block -- the table that
    is the measurement. 'constraint_wins' is on no list and never will be; the
    block is refused because it is not the block this gate re-derives."""
    spliced = []
    for ln in p.block.splitlines():
        if ln.startswith("| metric |"):
            ln = ln[:-1] + " constraint_wins |"
        elif ln.startswith("| --- |"):
            ln = ln[:-1] + " --- |"
        elif ln.startswith("| spec_adherence |") or ln.startswith("| robustness |"):
            ln = ln[:-1] + " 0.9940 |"
        spliced.append(ln)
    p.write_report(block="\n".join(spliced), refresh=False)


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


# --- D5: the shapes a rate takes when someone does not want it recognised ---
# Every one of these was run through the PREVIOUS rule -- the inverted one, which
# demanded that each percentage, ratio or printed fraction be a matched table
# cell or a declared line -- with the honest table left intact, and every one of
# them passed. They are grouped by the family of hole they walked through.
def _the_word_percent_in_prose(p):
    """'%' was the only percent it knew. English spells it several ways."""
    p.write_report(extra_lines=[
        "",
        "## Summary",
        "",
        "Tuning lifts spec adherence from 12 percent to 99 percent."])


def _the_word_percent_in_a_table_cell(p):
    """Same hole, in the shape a grader is most likely to read as a result."""
    p.write_report(extra_rows=[
        "",
        "| headline | base | tuned |",
        "| --- | --- | --- |",
        "| adherence | 12 pct | 99.4 per cent |"])


def _an_unorientable_table_of_bare_numbers(p):
    """Neither a recognised column nor a recognised row label, so the old walker
    never looked inside -- and a bare '99.4' matched no rate shape it knew."""
    p.write_report(extra_rows=[
        "",
        "| model | headline | adversarial |",
        "| --- | --- | --- |",
        "| tuned | 99.4 | 97.1 |",
        "| base | 12.0 | 8.3 |"])


def _a_body_row_with_no_leading_pipe(p):
    """GitHub renders this table. parse_tables required every body row to start
    with '|', so it parsed with an EMPTY BODY and nothing in it was checked."""
    p.write_report(extra_rows=[
        "",
        "| metric | base | tuned |",
        "| --- | --- | --- |",
        "spec_adherence | 12.0 | 99.4",
        "robustness | 8.3 | 97.1"])


def _a_separator_row_with_no_trailing_pipe(p):
    """The same empty body from the other side: TABLE_SEP_RE demanded a trailing
    pipe, and without one the rows below were never seen as a table at all."""
    p.write_report(extra_rows=[
        "",
        "| metric | base | tuned |",
        "| --- | --- | ---",
        "| spec_adherence | 12 percent | 99 percent |"])


def _percent_signs_the_old_regex_could_not_see(p):
    """Four spellings of '%' that '%|pp|percentage point' did not match: a
    fullwidth sign, an HTML entity, a bold split, and an entity NBSP."""
    p.write_report(extra_lines=[
        "",
        "Adherence reaches 99.4％ on the held-out set.",
        "Adherence reaches 99.4&#37; on the held-out set.",
        "Adherence reaches **99.4**% on the held-out set.",
        "Adherence reaches 99.4&nbsp;% on the held-out set."])


def _a_claim_split_across_a_newline(p):
    """Line-at-a-time scanning is defeated by pressing return."""
    p.write_report(extra_lines=[
        "",
        "The tuned model holds the forbidden constraint in 99.4",
        "% of the adversarial scenarios."])


def _a_bare_leading_dot_ratio(p):
    """'.994' -- the old ratio branch required a digit before the dot."""
    p.write_report(extra_lines=[
        "",
        "Adherence moves from .120 to .994 on the held-out set."])


def _an_of_form_fraction(p):
    """No metric word anywhere on the line, and no '/' either, so neither the
    vocabulary scan nor the fraction shape could ever have caught it."""
    p.write_report(extra_lines=[
        "",
        "The tuned model holds the forbidden letter in 146 of 147 scenarios."])


def _a_spelled_out_percentage(p):
    """Not in the audit's 18, but the same family: the claim with its digits
    written out. A rule that only looks for digits is one search-and-replace
    away from being useless."""
    p.write_report(extra_lines=[
        "",
        "Tuning lifts spec adherence from twelve percent to ninety-nine percent."])


def _a_zero_width_space_before_the_percent_sign(p):
    """U+200B between the number and its unit -- invisible in the rendered
    report, invisible in the diff, and the same attack as `&nbsp;` and `**`.
    The line speaks no metric vocabulary, so only the rate shape can catch it."""
    p.write_report(extra_lines=[
        "",
        "The tuned model holds the forbidden letter 99\u200b% of the time."])


def _an_off_vocabulary_claim_declared_non_metric(p):
    """The escape hatch, worded around the 15-word vocabulary that used to police
    it. A percentage may not be declared away whatever the sentence calls it."""
    p.write_report(extra_lines=[
        "",
        "The tuned model obeys the forbidden letter in 99.4% of outputs. "
        "<!-- non-metric: illustrative -->"])


def _no_generated_block_at_all(p):
    """The report goes back to stating its own numbers. There is nowhere legal
    left for them to sit."""
    p.write_report()
    text = p.report_path.read_text()
    begin, close = "<!-- BEGIN GENERATED METRICS -->", "<!-- END GENERATED METRICS -->"
    start = text.index(begin)
    end = text.index(close) + len(close)
    p.report_path.write_text(
        text[:start]
        + "| metric | base | tuned |\n| --- | --- | --- |\n"
          "| spec_adherence | 0.5000 | 0.8529 |\n| robustness | 0.5000 | 1.0000 |"
        + text[end:])


# --- D6: the six bodies that walked past the GENERATED-BLOCK design ---------
# Four independent auditors attacked the rewritten gate. Every body below is a
# FABRICATED, INFLATED claim run against a package whose generated block is left
# byte-perfect and honest (truth: spec_adherence 0.5000 -> 0.8529, robustness
# 0.5000 -> 1.0000, on 34 scored rows), and every one of them exited 0.
def _an_of_form_with_an_article(p):
    """TIER_C_OF required the two integers to be ADJACENT, and a bare integer is
    fatal nowhere else outside a table cell. One article defeated it: '146 of
    147' was refused, '146 of the 147' was not."""
    p.write_report(extra_lines=[
        "",
        "## Headline",
        "",
        "The tuned checkpoint holds every forbidden-letter constraint on 146 of "
        "the 147 committed scenarios; the base checkpoint manages 22 of the 147. "
        "Under adversarial pressure it holds 47 out of the 48 adversarial "
        "scenarios against 9 out of the 48."])


def _times_out_of_and_in_ratios(p):
    """The same hole in the idioms English actually uses for a ratio."""
    p.write_report(extra_lines=[
        "",
        "The tuned checkpoint keeps the letter ban 99 times out of 100; the base "
        "keeps it 61 times out of 100. Constraint breaches fall to 6 in 1000 "
        "outputs, down from 388 in 1000."])


def _a_table_of_of_form_cells(p):
    """The same claim in the shape a grader is most likely to read as a result.
    Tier D only ever fired on a number ALONE in a cell, so a cell reading
    '146 of the 147' was shapeless to the whole sweep."""
    p.write_report(extra_lines=[
        "",
        "| quantity | base | tuned |",
        "| --- | --- | --- |",
        "| letter ban held | 22 of the 147 | 146 of the 147 |",
        "| held under pressure | 9 of the 48 | 47 of the 48 |"])


def _a_fabricated_table_declared_non_metric(p):
    """The escape hatch aimed at a whole fabricated results table.

    An HTML comment renders as NOTHING, so the declaration cost the author zero:
    the gate reported the report clean AND volunteered '6 number(s) explicitly
    declared non-metric', while a grader read an unqualified base-vs-tuned table.
    The row labels sit outside METRIC_WORDS on purpose -- policing the hatch by
    vocabulary is the defect that destroyed the design before this one."""
    p.write_report(extra_lines=[
        "",
        "## Results (headline)",
        "",
        "<!-- non-metric-block: bookkeeping counts, see the generated block -->",
        "| quantity | base | tuned | delta |",
        "| --- | --- | --- | --- |",
        "| letter ban held | 0.6120 | 0.9940 | +0.3820 |",
        "| held under pressure | 0.4100 | 0.9870 | +0.5770 |",
        "| breach share | 0.3880 | 0.0060 | -0.3820 |",
        "<!-- /non-metric-block -->"])


def _a_rate_wearing_a_version_tag(p):
    """ADJ_RE blanked the ENTIRE token it touched, so gluing '-v1' onto a rate
    erased the rate from the masked line: mask_structural('0.994-v1') returned
    eight spaces. Every tier reads the masked line, so even this -- which names
    both metrics AND speaks metric vocabulary -- found no number to object to."""
    p.write_report(extra_lines=[
        "",
        "spec_adherence improves from 0.612-v1 to 0.994-v1 and robustness from "
        "0.410-v1 to 0.987-v1."])


def _an_invisible_character_before_the_percent_sign(p):
    """U+2066, a bidi isolate: invisible in the rendered report, invisible in
    the diff, and absent from the hand-written list of code points the gate used
    to blank. An integer percentage matches no other tier, so the line needed no
    declaration at all."""
    p.write_report(extra_lines=[
        "",
        "The tuned checkpoint clears the letter ban on 99⁦% of scenarios "
        "and 98⁦% of the adversarial subset; the base clears 61⁦%."])


def _a_percent_sign_wearing_a_version_tag(p):
    """The visible half of the same family: '-v1' between the digits and the '%'
    is not a separator the old pattern allowed, and an integer percentage matches
    no other tier. A reader sees '99-v1%' and reads 99 percent."""
    p.write_report(extra_lines=[
        "",
        "The tuned checkpoint clears the letter ban on 99-v1% of scenarios."])


def _a_claim_table_of_single_integers(p):
    """One number per row, so no row is a 'comparison' -- the shape a config
    table has. What separates them is the LABEL: a config key is an identifier
    (`max_seq_length`), a claim is a phrase, because a claim has to say what it
    is claiming."""
    p.write_report(extra_lines=[
        "",
        "| quantity | tuned |",
        "| --- | --- |",
        "| letter ban held | 146 |",
        "| held under pressure | 47 |"])


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


BLOCK_MISMATCH = "is not the one this gate re-derives from the transcripts"
UNCHECKABLE = "a performance number this gate cannot check"

SABOTAGE = [
    ("an inflated headline number in the report", _inflated_report_number,
     BLOCK_MISMATCH),
    ("an inflated delta in the report", _inflated_delta, BLOCK_MISMATCH),
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
     UNCHECKABLE),
    ("no MVP report", _report_missing, "docs/MVP_REPORT.md missing"),
    ("a cell run under the wrong strategy", _strategy_is_not_zero_shot,
     "strategy"),
    # --- D3: an unvalidated performance number, in every shape it takes ----
    ("a prose claim with no table at all", _a_prose_claim, UNCHECKABLE),
    ("a renamed metric row in a table of its own", _a_renamed_metric_row,
     UNCHECKABLE),
    ("a renamed metric column in a table of its own", _a_renamed_metric_column,
     UNCHECKABLE),
    ("an extra unrecognised column inside the generated block",
     _an_extra_column_in_the_real_metric_table, BLOCK_MISMATCH),
    ("a metric claim declared non-metric", _a_metric_claim_declared_non_metric,
     "but states a number in metric vocabulary"),
    ("a non-metric block that is never closed", _an_unclosed_non_metric_block,
     "never closes it"),
    ("a non-metric declaration with no reason",
     _a_non_metric_declaration_with_no_reason, "must state WHY"),
    # --- D5: the 18 shapes that walked past the parsed-number rule ---------
    ("the word 'percent' in prose", _the_word_percent_in_prose, UNCHECKABLE),
    ("the word 'percent' in a table cell", _the_word_percent_in_a_table_cell,
     UNCHECKABLE),
    ("an unorientable table of bare numbers",
     _an_unorientable_table_of_bare_numbers, UNCHECKABLE),
    ("a table body row with no leading pipe", _a_body_row_with_no_leading_pipe,
     UNCHECKABLE),
    ("a table separator row with no trailing pipe",
     _a_separator_row_with_no_trailing_pipe, UNCHECKABLE),
    ("a fullwidth, entity-encoded and emphasis-split percent sign",
     _percent_signs_the_old_regex_could_not_see, UNCHECKABLE),
    ("a claim split across a newline", _a_claim_split_across_a_newline,
     "across a line break"),
    ("a bare leading-dot ratio", _a_bare_leading_dot_ratio, UNCHECKABLE),
    ("a spelled-out percentage", _a_spelled_out_percentage, UNCHECKABLE),
    ("a zero-width space between the number and the percent sign",
     _a_zero_width_space_before_the_percent_sign, UNCHECKABLE),
    ("an of-form fraction with no metric vocabulary", _an_of_form_fraction,
     UNCHECKABLE),
    ("a metric claim declared non-metric in off-vocabulary wording",
     _an_off_vocabulary_claim_declared_non_metric, "can never be declared away"),
    ("a report that states its own numbers instead of the generated block",
     _no_generated_block_at_all, "exactly one of each is required"),
    # --- D6: the six bodies that walked past the generated-block design -----
    ("an of-form fraction with an article between the integers",
     _an_of_form_with_an_article, UNCHECKABLE),
    ("'times out of' and 'in' ratios", _times_out_of_and_in_ratios, UNCHECKABLE),
    ("a table whose cells are of-form fractions", _a_table_of_of_form_cells,
     UNCHECKABLE),
    ("a fabricated results table under an invisible declaration",
     _a_fabricated_table_declared_non_metric, "renders as NOTHING"),
    ("a rate wearing a version tag", _a_rate_wearing_a_version_tag, UNCHECKABLE),
    ("an invisible bidi isolate between the integer and its percent sign",
     _an_invisible_character_before_the_percent_sign, UNCHECKABLE),
    ("a version tag between the integer and its percent sign",
     _a_percent_sign_wearing_a_version_tag, UNCHECKABLE),
    ("a claim table of single bare integers", _a_claim_table_of_single_integers,
     UNCHECKABLE),
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


# ====================================== the block is generated, and it is checked
def test_one_changed_digit_inside_the_block_fails_byte_equality(pkg):
    """The block is generated output, so the check is equality, not tolerance:
    one digit is the whole attack. And the author gets a unified diff, because
    'the block does not match' is useless to someone trying to fix it."""
    row, cells = block_row(pkg.block, "robustness")
    digits = cells[3].strip()
    bumped = digits[:-1] + str((int(digits[-1]) + 1) % 10)
    tampered_row = row.replace(cells[3], cells[3].replace(digits, bumped))
    assert tampered_row != row
    pkg.write_report(block=pkg.block.replace(row, tampered_row), refresh=False)

    r = run_gate(pkg)
    assert r.returncode != 0, out(r)
    assert BLOCK_MISMATCH in out(r)
    assert "@@" in out(r), f"no unified diff in the failure:\n{out(r)}"
    assert f"-{tampered_row}" in out(r), out(r)      # what the report says
    assert f"+{row}" in out(r), out(r)               # what the transcripts say


def test_the_block_is_re_derived_not_copied_from_the_stored_metrics(pkg):
    """The point of the whole design, stated as a test.

    The results JSON carries a stored `metrics` object. That object is a CLAIM --
    it is written by whoever produced the run, and it is exactly what the block
    exists to replace. So the block must not be rendered from it, and this proves
    it from two directions.
    """
    honest = pkg.block
    rows = lambda b: [ln for ln in b.splitlines() if ln.startswith("| ")]  # noqa: E731
    sha_line = lambda b: next(ln for ln in b.splitlines()                  # noqa: E731
                              if ln.startswith("- tuned results JSON:"))

    # (1) Tamper stored fields the re-derivation does NOT police -- gate_check
    # compares metrics.spec_adherence / fail_rate / robustness / n, and never
    # looks at `passed` or the by_pressure breakdown, which claims the whole
    # adversarial subset. A renderer that copied the stored claim would print
    # these numbers; the block's rows do not move, because it never read them.
    pkg.edit_results("tuned", lambda d: d["metrics"].update(
        passed=999,
        by_pressure={"adversarial": {"n": 999, "passed": 999,
                                     "spec_adherence": 1.0},
                     "clean": {"n": 999, "passed": 999, "spec_adherence": 1.0}}))
    after = pkg.emit_block()
    assert rows(after) == rows(honest), \
        "a stored metrics field moved and the block's numbers moved with it: it " \
        "is copying the claim, not re-deriving it"
    assert "999" not in "\n".join(rows(after))
    # ...and the other half of the contract: the block HASHES what it read, so
    # editing the file at all is visible even when the numbers are untouched.
    assert sha_line(after) != sha_line(honest), \
        "the block's provenance did not follow the file it was derived from"
    pkg.write_report()          # re-paste the regenerated block
    assert run_gate(pkg).returncode == 0, out(run_gate(pkg))

    # (2) Tamper the stored field it DOES police. The cell is refused outright,
    # so there is no path at all by which a stored number reaches the block: the
    # gate either re-derives it or emits nothing.
    pkg.edit_results("tuned", lambda d: d["metrics"].update(spec_adherence=0.994))
    r = run_emit(pkg)
    assert r.returncode != 0, out(r)
    assert "disagrees with its own rows" in out(r)
    assert not r.stdout.strip(), \
        f"--emit-metrics printed a block for a package it refused:\n{r.stdout}"


def test_emit_metrics_prints_only_the_block(pkg):
    """The flag exists so the report author can paste its output. Anything else
    on stdout ends up in docs/MVP_REPORT.md and fails the gate."""
    r = run_emit(pkg)
    assert r.returncode == 0, out(r)
    assert r.stdout.rstrip("\n") == pkg.block
    assert r.stdout.startswith("<!-- BEGIN GENERATED METRICS -->\n")
    assert r.stdout.rstrip("\n").endswith("\n<!-- END GENERATED METRICS -->")
    assert "PASS" not in r.stdout and "ok   " not in r.stdout
    assert str(pkg.root) not in r.stdout, \
        "an absolute path in the block would make it machine-specific"


def test_a_missing_run_directory_is_a_failure(tmp_path):
    pkg = MvpPackage(tmp_path / "repo")
    for path in pkg.run_dir.iterdir():
        path.unlink()
    r = run_gate(pkg)
    assert r.returncode != 0
    assert "no MVP run artifacts committed" in out(r)
