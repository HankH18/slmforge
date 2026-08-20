"""Proof that scripts/check_loop.py -- the loop gate -- is a GATE, not a printer.

The MVP bar the spec grades is "full loop, generate -> train -> eval, runs end to
end". check_loop.py is the only thing that re-verifies that claim offline, so it
must exit 0 on a well-formed run tree and non-zero on every corruption that would
make the claim false -- above all on THE contract that makes the tuned model's
numbers mean anything: the prompt a training row carries must be exactly the
prompt the eval harness builds for the same scenario.

An adversarial verifier previously proved four holes in this checker, each of
which is regression-tested by name below:

  1. the prompt leg trusted ``system_prompt_equals_eval_SYSTEM_CORE``, a boolean
     the trainer writes as a hardcoded literal. Every row could be rewritten,
     the dataset hash recomputed, and the loop still reported VERIFIED. There is
     now no attestation boolean at all: the prompt is re-rendered from each
     row's own scenario through the shared renderer and compared byte for byte,
     which also makes TRAILING DRIFT appended to the user message visible -- the
     old prefix/suffix shape test could not see appended text.
  2. the checkpoint leg accepted `any(marker in model_spec)` over a marker set
     seeded with the bare literals 'adapter' and 'merged_fp16', so any path
     containing the word "adapter" passed. Matching is now whole-reference.
  3. bare ``isinstance(x, int)`` read ``true`` as 1, so ``"bytes": true`` passed
     the leg whose job is proving a non-zero byte count. Every numeric field in
     the log and in the metrics is parametrized against a boolean below.
  4. the wall-clock check was bare truthiness. 0 / false / "fast" are all tested.

Every fixture is synthesized in tmp_path. Nothing here reads or writes the real
runs/, data/ or scenarios/ trees.
"""
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT))
import harness as H  # noqa: E402

SCRIPTS = REPO_ROOT / "scripts"
CHECK_LOOP = SCRIPTS / "check_loop.py"

RUN_NAME = "smoke-20260820-001"
BASE_REPO = "Qwen/Qwen3-1.7B"
BASE_SHA = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
ADAPTER_PATH = f"/vol/{RUN_NAME}/adapter"
MERGED_PATH = f"/vol/{RUN_NAME}/merged_fp16"
TUNED_SPEC = f"hf:{MERGED_PATH}"
DATASET_REL = "data/train/train_smoke.jsonl"
RENDER_FN = "scenarios.render.render_prompt"
STEM = "merged_fp16__zero_shot"
TEACHER = "claude-sonnet-4-6"


# ------------------------------------------------------------------ renderer
# Stand-in for the SHARED prompt renderer owned by the scenario/harness lane
# (see DECLARED ASSUMPTIONS in the task report). check_loop.py imports whatever
# prompt_contract.render_fn names out of the repo it is pointed at, so the test
# repo ships a real one and the training rows below are built THROUGH it -- the
# checker is never handed a prompt this test wrote by hand.
RENDERER_SRC = '''\
"""The shared scenario -> user-message renderer."""

_DESCRIBE = {
    "banned_words": lambda p: "never use the word(s): " + ", ".join(p["words"]),
    "banned_phrases": lambda p: "never use the phrase(s): " + ", ".join(p["phrases"]),
    "banned_letter": lambda p: "never use the letter " + p["letter"],
    "banned_punctuation": lambda p: "never use the mark(s): " + " ".join(p["marks"]),
    "sentence_initial_letter": lambda p: "start every sentence with " + p["letter"],
    "no_lists": lambda p: "never use a list or bullet points",
    "no_quotation_marks": lambda p: "never use quotation marks",
}


def describe(constraint):
    return _DESCRIBE[constraint["type"]](constraint["params"])


def render_prompt(scenario, constraint_lookup):
    lines = ["Task: " + scenario["task"], "", "Hard constraints -- all must hold:"]
    for cid in scenario["constraints"]:
        lines.append("- " + describe(constraint_lookup[cid]))
    lines.append("")
    lines.append("Write the response now. Violating any constraint fails the task.")
    return "\\n".join(lines)
'''


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ corpora
CONSTRAINTS = [
    {"id": "c-lex-001", "type": "banned_words",
     "params": {"words": ["coffee", "espresso"]},
     "family": "lexical", "split": "train"},
    {"id": "c-fmt-001", "type": "no_lists", "params": {},
     "family": "format_negative", "split": "train"},
    {"id": "c-chr-001", "type": "banned_letter", "params": {"letter": "z"},
     "family": "character", "split": "train"},
]
LOOKUP = {c["id"]: c for c in CONSTRAINTS}

TRAIN_TASKS = [
    "Summarize the history of the espresso machine for a cafe newsletter.",
    "Draft a short welcome note for a new teammate.",
    "Explain why sourdough starters need feeding.",
    "Write a two-sentence product blurb for a wool blanket.",
    "Describe a rainy morning commute in three sentences.",
    "Write a polite decline to a meeting invitation.",
    "Explain photosynthesis to a ten year old.",
    "Write a caption for a photo of a mountain lake.",
]


def train_scenarios():
    out = []
    for i, task in enumerate(TRAIN_TASKS, 1):
        ids = ["c-lex-001", "c-fmt-001"] if i % 2 else ["c-fmt-001", "c-chr-001"]
        out.append({"scenario_id": f"s-t{i:03d}", "task": task, "constraints": ids,
                    "k": len(ids),
                    "pressure": "adversarial" if i % 3 == 0 else "clean",
                    "split": "train"})
    return out


EVAL_TASKS = [
    "Write a three-sentence review of a public library.",
    "Draft a reminder about the office recycling policy.",
    "Explain how a bicycle derailleur works.",
    "Write an apology for a late delivery.",
    "Describe the sound of a summer thunderstorm.",
    "Write a one-paragraph bio for a conference program.",
]


def eval_scenarios():
    out = []
    for i, task in enumerate(EVAL_TASKS, 1):
        ids = ["c-lex-001", "c-fmt-001"] if i % 2 else ["c-fmt-001", "c-chr-001"]
        out.append({"scenario_id": f"s-e{i:03d}", "task": task, "constraints": ids,
                    "k": len(ids),
                    "pressure": "adversarial" if i % 2 == 0 else "clean",
                    "split": "heldout_type" if i > 4 else "train"})
    return out


def transcript_rows(scenarios):
    """Per-example rows in the harness's PINNED transcript schema, with
    spec_pass computed by harness.compute_spec_pass -- so the metrics below come
    out of the real harness.aggregate, not out of a hand-written dict."""
    rows = []
    for i, s in enumerate(scenarios):
        cid_list = s["constraints"]
        # rows 2 and 4 violate their first constraint; row 5 is quarantined.
        violates = i in (2, 4)
        results = [{"constraint_id": cid,
                    "pass": not (violates and j == 0),
                    "evidence": "[[coffee]]" if (violates and j == 0) else "clean"}
                   for j, cid in enumerate(cid_list)]
        if i == 5:
            rows.append({
                "scenario_id": s["scenario_id"], "model": TUNED_SPEC,
                "strategy": "zero_shot", "output": "A short reply.",
                "checker_results": results, "quality_score": None,
                "quality_reasoning": "", "spec_pass": None,
                H.JUDGE_ERROR_FIELD: "judge transport error after 3 attempts"})
            continue
        score = 4
        rows.append({
            "scenario_id": s["scenario_id"], "model": TUNED_SPEC,
            "strategy": "zero_shot", "output": f"Reply number {i}.",
            "checker_results": results, "quality_score": score,
            "quality_reasoning": "Fulfills the task in plain prose.",
            "spec_pass": H.compute_spec_pass(results, score)})
    return rows


class LoopRun:
    """A well-formed committed-artifact tree that check_loop.py must accept."""

    def __init__(self, root):
        self.root = Path(root)
        self.run_dir = self.root / "runs" / "smoke"
        self.eval_dir = self.run_dir / "eval"
        self.ds_path = self.root / DATASET_REL
        self.log_path = self.run_dir / "train_log.json"
        self.results_path = self.eval_dir / f"{STEM}.results.json"
        self.transcripts_path = self.eval_dir / f"{STEM}.transcripts.jsonl"

        # the shared renderer, inside the repo under test
        pkg = self.root / "scenarios"
        pkg.mkdir(parents=True, exist_ok=True)
        (pkg / "__init__.py").write_text("")
        (pkg / "render.py").write_text(RENDERER_SRC)
        self.render = _load_module(pkg / "render.py", f"_render_{id(self)}")

        # generate: training rows built THROUGH the shared renderer
        self.ds_path.parent.mkdir(parents=True, exist_ok=True)
        self.write_rows([self.training_row(s) for s in train_scenarios()])

        # eval: transcripts + metrics from the REAL harness
        scen = eval_scenarios()
        rows = transcript_rows(scen)
        metrics = H.aggregate(rows, scen, LOOKUP)
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        self.transcripts_path.write_text(
            "".join(json.dumps(r) + "\n" for r in rows))
        self.results_path.write_text(json.dumps(
            {"model": TUNED_SPEC, "strategy": "zero_shot", "judge": TEACHER,
             "metrics": metrics}, indent=2))

        self.log_path.write_text(json.dumps(self.train_log(), indent=2))

    # ---------------------------------------------------------- generation
    def training_row(self, scenario, attempt=1):
        constraints = [LOOKUP[c] for c in scenario["constraints"]]
        prompt = self.render.render_prompt(scenario, LOOKUP)
        return {"messages": [{"role": "user", "content": prompt},
                             {"role": "assistant",
                              "content": f"A clean completion for "
                                         f"{scenario['scenario_id']}."}],
                "constraints": constraints,
                "provenance": {"teacher": TEACHER, "attempt": attempt},
                "scenario": scenario}

    def train_log(self):
        rows = self.rows()
        return {
            "base_model": {"repo": BASE_REPO, "revision": BASE_SHA},
            "config_resolved": {"lora_r": 16, "lora_alpha": 32,
                                "learning_rate": 0.0002, "max_steps": 20,
                                "batch_size": 4, "smoke": True},
            "dataset": {"path": DATASET_REL, "rows": len(rows),
                        "sha256": self.sha256()},
            "prompt_contract": {
                "render_fn": RENDER_FN,
                "sample_row_index": 0,
                "sample_rendered_prompt": rows[0]["messages"][0]["content"]},
            "steps": 20,
            "loss_history": [[i, round(1.9 - 0.07 * i, 4)] for i in range(1, 21)],
            "wall_clock_seconds": 559.2,
            "artifacts": {"adapter": ADAPTER_PATH, "adapter_bytes": 73_400_320,
                          "merged_fp16": MERGED_PATH,
                          "merged_bytes": 3_400_000_000},
            "started_utc": "2026-08-20T00:00:00+00:00",
            "finished_utc": "2026-08-20T00:12:00+00:00",
        }

    # ---------------------------------------------------------- accessors
    def rows(self):
        return [json.loads(ln) for ln in self.ds_path.read_text().splitlines()
                if ln.strip()]

    def write_rows(self, rows):
        self.ds_path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    def sha256(self):
        return hashlib.sha256(self.ds_path.read_bytes()).hexdigest()

    def log(self):
        return json.loads(self.log_path.read_text())

    def edit_log(self, mutate):
        data = self.log()
        mutate(data)
        self.log_path.write_text(json.dumps(data, indent=2))

    def edit_results(self, mutate):
        data = json.loads(self.results_path.read_text())
        mutate(data)
        self.results_path.write_text(json.dumps(data, indent=2))

    def edit_metrics(self, mutate):
        self.edit_results(lambda d: mutate(d["metrics"]))

    def reseal(self):
        """Recompute dataset.sha256, dataset.rows and the sample prompt -- what
        an attacker (or a careless regeneration) does to keep every integrity
        check quiet after rewriting the rows."""
        rows = self.rows()
        self.edit_log(lambda d: (
            d["dataset"].update(sha256=self.sha256(), rows=len(rows)),
            d["prompt_contract"].update(
                sample_rendered_prompt=rows[d["prompt_contract"]
                                            ["sample_row_index"]]
                ["messages"][0]["content"])))


def run_checker(run_dir, repo_root, *extra):
    return subprocess.run(
        [sys.executable, str(CHECK_LOOP), str(run_dir),
         "--repo-root", str(repo_root), *map(str, extra)],
        cwd=str(REPO_ROOT), capture_output=True, text=True)


def out(proc):
    return proc.stdout + proc.stderr


@pytest.fixture
def loop_run(tmp_path):
    return LoopRun(tmp_path / "repo")


# ============================================================ happy path
def test_checker_passes_on_a_well_formed_run(loop_run):
    r = run_checker(loop_run.run_dir, loop_run.root)
    assert r.returncode == 0, out(r)
    assert "LOOP VERIFIED: 4/4 legs pass" in r.stdout
    # the contract is reported as re-derived through the shared renderer
    assert RENDER_FN in r.stdout


def test_json_report_is_machine_readable_and_agrees(loop_run):
    r = run_checker(loop_run.run_dir, loop_run.root, "--json")
    assert r.returncode == 0, out(r)
    report = json.loads(r.stdout)
    assert report["ok"] is True
    assert len(report["legs"]) == 4
    assert all(leg["ok"] for leg in report["legs"])


def test_checker_fails_on_a_missing_run_directory(tmp_path):
    r = run_checker(tmp_path / "nope", tmp_path)
    assert r.returncode != 0
    assert "is not a directory" in out(r)


def test_no_self_attestation_boolean_is_read_anywhere(loop_run):
    """DEFECT 1, structurally. The checker must not have re-grown a flag to
    trust: nothing in the log outside prompt_contract may certify the prompt."""
    reads = [ln.strip() for ln in CHECK_LOOP.read_text().splitlines()
             if "system_prompt_equals" in ln
             and (".get(" in ln or '["' in ln or "['" in ln)]
    assert not reads, "the checker still reads an attestation flag:\n" + "\n".join(reads)
    # and a run that carries such a flag is not helped by it
    loop_run.edit_log(lambda d: d.update(
        prompt_contract_verified=True,
        system_prompt_equals_eval_SYSTEM_CORE=True))
    rows = loop_run.rows()
    rows[0]["messages"][0]["content"] = "You are a helpful assistant. Do the task."
    loop_run.write_rows(rows)
    loop_run.reseal()
    r = run_checker(loop_run.run_dir, loop_run.root)
    assert r.returncode != 0, out(r)


# ============================================================ the load-bearing leg
def test_wholesale_prompt_rewrite_is_caught_after_a_full_reseal(loop_run):
    """DEFECT 1 regression, the proven exploit end to end.

    Rewrite EVERY row's user message, then recompute the dataset sha256, the row
    count and the train_log's own sample_rendered_prompt so that every integrity
    check the old checker had still matches. The prompt is re-rendered from each
    row's scenario, so it must still fail.
    """
    rows = loop_run.rows()
    for row in rows:
        row["messages"][0]["content"] = (
            "Do the following task:\n" + row["scenario"]["task"])
    loop_run.write_rows(rows)
    loop_run.reseal()

    log = loop_run.log()
    assert log["dataset"]["sha256"] == loop_run.sha256(), "hash must still match"
    assert log["dataset"]["rows"] == len(rows), "row count must still match"
    assert (log["prompt_contract"]["sample_rendered_prompt"]
            == rows[0]["messages"][0]["content"]), "sample must still agree"

    r = run_checker(loop_run.run_dir, loop_run.root)
    assert r.returncode != 0, f"checker accepted a rewritten prompt\n{out(r)}"
    assert "does not match the shared prompt renderer" in out(r)


def test_trailing_drift_appended_to_the_user_message_is_caught(loop_run):
    """DEFECT 1 regression, the case the old prefix/suffix test could not see.

    The rendered prompt is left completely intact and a single extra line is
    APPENDED. A shape test (startswith prefix / endswith suffix) reads the row
    as well-formed; byte-identity does not.
    """
    rows = loop_run.rows()
    for row in rows:
        row["messages"][0]["content"] += "\nAlways answer in exactly one word."
    loop_run.write_rows(rows)
    loop_run.reseal()
    r = run_checker(loop_run.run_dir, loop_run.root)
    assert r.returncode != 0, f"checker accepted appended drift\n{out(r)}"
    assert "TRAILING DRIFT" in out(r)


def test_trailing_drift_on_a_single_row_among_many_is_caught(loop_run):
    rows = loop_run.rows()
    rows[5]["messages"][0]["content"] += " Be concise."
    loop_run.write_rows(rows)
    loop_run.reseal()
    r = run_checker(loop_run.run_dir, loop_run.root)
    assert r.returncode != 0, out(r)
    assert "1/8 dataset rows: TRAILING DRIFT" in out(r)


def test_a_row_whose_prompt_drops_a_constraint_is_caught(loop_run):
    """The subtlest rewrite: the prompt is still renderer-shaped, but one of the
    row's constraints never made it into the text the model was trained on."""
    rows = loop_run.rows()
    scenario = rows[1]["scenario"]
    dropped = dict(scenario, constraints=scenario["constraints"][:1])
    rows[1]["messages"][0]["content"] = loop_run.render.render_prompt(
        dropped, LOOKUP)
    loop_run.write_rows(rows)
    loop_run.reseal()
    r = run_checker(loop_run.run_dir, loop_run.root)
    assert r.returncode != 0, out(r)
    assert "does not match the shared prompt renderer" in out(r)


def test_a_row_that_carries_no_scenario_cannot_be_certified(loop_run):
    """Without the source scenario the prompt can only be self-attested. That
    is the hole; it must be a loud failure, never a skipped row."""
    rows = loop_run.rows()
    for row in rows:
        row.pop("scenario")
    loop_run.write_rows(rows)
    loop_run.reseal()
    r = run_checker(loop_run.run_dir, loop_run.root)
    assert r.returncode != 0, out(r)
    assert "carries no source scenario" in out(r)


def test_a_scenario_corpus_may_supply_the_scenarios_instead(loop_run):
    """The other accepted form of the same evidence: rows name a scenario_id and
    the train_log declares the corpus they resolve against."""
    corpus = loop_run.root / "data" / "scenarios" / "train_smoke.jsonl"
    corpus.parent.mkdir(parents=True, exist_ok=True)
    corpus.write_text("".join(json.dumps(s) + "\n" for s in train_scenarios()))
    rows = loop_run.rows()
    for row in rows:
        row["scenario_id"] = row.pop("scenario")["scenario_id"]
    loop_run.write_rows(rows)
    loop_run.edit_log(lambda d: d["dataset"].update(
        scenarios_path="data/scenarios/train_smoke.jsonl"))
    loop_run.reseal()
    r = run_checker(loop_run.run_dir, loop_run.root)
    assert r.returncode == 0, out(r)


def test_a_scenario_corpus_declared_in_config_resolved_is_honoured(loop_run):
    """The trainer lane records its data config under config_resolved, so a
    corpus declared there resolves exactly as dataset.scenarios_path does."""
    corpus = loop_run.root / "data" / "scenarios" / "train_smoke.jsonl"
    corpus.parent.mkdir(parents=True, exist_ok=True)
    corpus.write_text("".join(json.dumps(s) + "\n" for s in train_scenarios()))
    rows = loop_run.rows()
    for row in rows:
        row["scenario_id"] = row.pop("scenario")["scenario_id"]
    loop_run.write_rows(rows)
    loop_run.edit_log(lambda d: d["config_resolved"].update(
        data={"scenarios_path": "data/scenarios/train_smoke.jsonl"}))
    loop_run.reseal()
    r = run_checker(loop_run.run_dir, loop_run.root)
    assert r.returncode == 0, out(r)


def test_a_declared_scenario_corpus_that_is_missing_is_a_failure(loop_run):
    loop_run.edit_log(lambda d: d["dataset"].update(
        scenarios_path="data/scenarios/gone.jsonl"))
    r = run_checker(loop_run.run_dir, loop_run.root)
    assert r.returncode != 0, out(r)
    assert "does not exist" in out(r)


def test_the_renderer_must_live_in_the_repo_under_test(loop_run):
    """A run may not nominate its own prompt renderer -- that is self-attestation
    with extra steps."""
    loop_run.edit_log(lambda d: d["prompt_contract"].update(
        render_fn="json.dumps"))
    r = run_checker(loop_run.run_dir, loop_run.root)
    assert r.returncode != 0, out(r)
    assert "does not live in the shared renderer modules" in out(r)


def test_a_renderer_that_is_not_on_disk_is_not_stubbed(loop_run):
    (loop_run.root / "scenarios" / "render.py").unlink()
    r = run_checker(loop_run.run_dir, loop_run.root)
    assert r.returncode != 0, out(r)
    assert "cannot import the shared prompt renderer" in out(r)


# ============================================================ leg 4 identity
def test_a_checkpoint_path_merely_containing_the_word_adapter_is_rejected(loop_run):
    """DEFECT 2 regression.

    _checkpoint_markers() injected the bare literals 'adapter' and 'merged_fp16'
    into the marker set and accepted `any(m in spec)`, so ANY path with the word
    "adapter" in it certified the eval as run against the trained weights. The
    reference below would have passed that test.
    """
    impostor = "hf:/models/public/some-other-adapter-run"
    assert "adapter" in impostor, "the fixture must defeat the old substring test"
    loop_run.edit_results(lambda d: d.update(model=impostor))
    r = run_checker(loop_run.run_dir, loop_run.root)
    assert r.returncode != 0, f"checker accepted {impostor}\n{out(r)}"
    assert "in whole, not merely contain a word" in out(r)


def test_a_checkpoint_path_containing_the_run_name_is_rejected(loop_run):
    """The marker set also carried run_dir.name, so any path mentioning the run
    passed. Only the recorded artifact references count now."""
    loop_run.edit_results(lambda d: d.update(
        model=f"hf:/vol/{RUN_NAME}/merged_fp16_backup"))
    r = run_checker(loop_run.run_dir, loop_run.root)
    assert r.returncode != 0, out(r)
    assert "does not reference this run's checkpoint" in out(r)


def test_the_adapter_artifact_is_an_accepted_checkpoint_reference(loop_run):
    """The positive half of the same rule: the exact recorded adapter path is a
    match, so the guard is identity-based and not merely restrictive."""
    loop_run.edit_results(lambda d: d.update(model=f"hf:{ADAPTER_PATH}"))
    rows = [json.loads(ln) for ln
            in loop_run.transcripts_path.read_text().splitlines() if ln.strip()]
    for row in rows:
        row["model"] = f"hf:{ADAPTER_PATH}"
    loop_run.transcripts_path.write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    r = run_checker(loop_run.run_dir, loop_run.root)
    assert r.returncode == 0, out(r)


def test_metrics_that_the_transcripts_do_not_support_are_rejected(loop_run):
    """The headline number is re-derived from the raw transcripts, so inflating
    it without touching the per-example records fails."""
    loop_run.edit_metrics(lambda m: m.update(passed=m["n_scored"],
                                             spec_adherence=1.0))
    r = run_checker(loop_run.run_dir, loop_run.root)
    assert r.returncode != 0, out(r)
    assert "the transcripts do not support the metrics" in out(r)


def test_a_forged_pass_verdict_in_a_transcript_is_rejected(loop_run):
    """Flipping spec_pass in a transcript row (and the metrics with it) is
    caught by harness.validate_transcript_row: spec_pass must agree with the
    checker verdicts and the quality score in the same row."""
    lines = [json.loads(ln) for ln
             in loop_run.transcripts_path.read_text().splitlines() if ln.strip()]
    flipped = next(r for r in lines if r["spec_pass"] is False)
    flipped["spec_pass"] = True
    loop_run.transcripts_path.write_text(
        "".join(json.dumps(r) + "\n" for r in lines))
    loop_run.edit_metrics(lambda m: m.update(
        passed=m["passed"] + 1, spec_adherence=(m["passed"] + 1) / m["n_scored"]))
    r = run_checker(loop_run.run_dir, loop_run.root)
    assert r.returncode != 0, out(r)
    assert "not a row this harness scored" in out(r)


def test_a_run_cannot_smuggle_in_its_own_scoring_harness(loop_run):
    """The checker resolves the run's prompt renderer out of the run's own repo,
    which means that repo is on sys.path. It must not be able to substitute the
    harness that revalidates the transcripts: a forged pass verdict stays caught
    even when the repo ships a harness.py that would wave it through.
    """
    (loop_run.root / "harness.py").write_text(
        "JUDGE_ERROR_FIELD = 'judge_error'\n"
        "def validate_transcript_row(row):\n    return None\n")
    lines = [json.loads(ln) for ln
             in loop_run.transcripts_path.read_text().splitlines() if ln.strip()]
    flipped = next(r for r in lines if r["spec_pass"] is False)
    flipped["spec_pass"] = True
    loop_run.transcripts_path.write_text(
        "".join(json.dumps(r) + "\n" for r in lines))
    loop_run.edit_metrics(lambda m: m.update(
        passed=m["passed"] + 1, spec_adherence=(m["passed"] + 1) / m["n_scored"]))
    r = run_checker(loop_run.run_dir, loop_run.root)
    assert r.returncode != 0, f"a planted harness.py defeated leg 4\n{out(r)}"
    assert "not a row this harness scored" in out(r)


def test_check_loop_requires_exactly_what_the_real_harness_emits(loop_run):
    """Binds the checker's metric contract to harness.aggregate() itself: if the
    harness stops emitting one of these keys, this fails instead of the checker
    silently demanding a key nobody writes."""
    module = _load_module(CHECK_LOOP, "_check_loop_under_test")
    scen = eval_scenarios()
    real = H.aggregate(transcript_rows(scen), scen, LOOKUP)
    missing = [k for k in module.REQUIRED_METRIC_KEYS if k not in real]
    assert not missing, f"check_loop requires metric keys aggregate() does not emit: {missing}"


# ============================================================ sabotage table
def _no_train_log(d):
    d.log_path.unlink()


def _train_log_not_json(d):
    d.log_path.write_text("{not json")


def _empty_loss_history(d):
    d.edit_log(lambda x: x.update(loss_history=[]))


def _no_loss_history(d):
    d.edit_log(lambda x: x.pop("loss_history"))


def _zero_steps(d):
    d.edit_log(lambda x: x.update(steps=0))


def _all_zero_losses(d):
    d.edit_log(lambda x: x.update(
        loss_history=[[i, 0.0] for i in range(1, 21)]))


def _non_finite_loss(d):
    d.edit_log(lambda x: x["loss_history"].__setitem__(3, [4, float("nan")]))


def _loss_beyond_the_step_count(d):
    d.edit_log(lambda x: x["loss_history"].append([x["steps"] + 5, 0.31]))


def _no_config_resolved(d):
    d.edit_log(lambda x: x.pop("config_resolved"))


def _empty_config_resolved(d):
    d.edit_log(lambda x: x.update(config_resolved={}))


def _unpinned_base_revision(d):
    d.edit_log(lambda x: x["base_model"].update(revision="main"))


def _no_wall_clock(d):
    d.edit_log(lambda x: x.pop("wall_clock_seconds"))


def _zero_wall_clock(d):
    d.edit_log(lambda x: x.update(wall_clock_seconds=0))


def _false_wall_clock(d):
    d.edit_log(lambda x: x.update(wall_clock_seconds=False))


def _string_wall_clock(d):
    d.edit_log(lambda x: x.update(wall_clock_seconds="fast"))


def _impossible_wall_clock(d):
    d.edit_log(lambda x: x.update(wall_clock_seconds=99_999.0))


def _finished_before_started(d):
    d.edit_log(lambda x: x.update(finished_utc="2026-08-19T23:00:00+00:00"))


def _bad_timestamp(d):
    d.edit_log(lambda x: x.update(started_utc="yesterday"))


def _no_adapter(d):
    d.edit_log(lambda x: x["artifacts"].pop("adapter"))


def _no_merged(d):
    d.edit_log(lambda x: x["artifacts"].pop("merged_fp16"))


def _zero_merged_bytes(d):
    d.edit_log(lambda x: x["artifacts"].update(merged_bytes=0))


def _same_path_twice(d):
    d.edit_log(lambda x: x["artifacts"].update(merged_fp16=ADAPTER_PATH))


def _no_prompt_contract(d):
    d.edit_log(lambda x: x.pop("prompt_contract"))


def _no_render_fn(d):
    d.edit_log(lambda x: x["prompt_contract"].pop("render_fn"))


def _no_sample_prompt(d):
    d.edit_log(lambda x: x["prompt_contract"].pop("sample_rendered_prompt"))


def _sample_prompt_disagrees(d):
    d.edit_log(lambda x: x["prompt_contract"].update(
        sample_rendered_prompt="Task: something else entirely."))


def _sample_index_out_of_range(d):
    d.edit_log(lambda x: x["prompt_contract"].update(sample_row_index=999))


def _no_eval_results(d):
    d.results_path.unlink()


def _results_are_the_base_model(d):
    d.edit_results(lambda x: x.update(model=f"hf:{BASE_REPO}@{BASE_SHA}"))


def _results_n_zero(d):
    d.edit_metrics(lambda m: m.update(n=0))


def _results_no_spec_adherence(d):
    d.edit_metrics(lambda m: m.pop("spec_adherence"))


def _results_no_robustness(d):
    d.edit_metrics(lambda m: m.pop("robustness"))


def _results_null_robustness(d):
    d.edit_metrics(lambda m: m.update(robustness=None))


def _spec_adherence_not_the_ratio(d):
    d.edit_metrics(lambda m: m.update(spec_adherence=0.99))


def _no_adversarial_cell(d):
    d.edit_metrics(lambda m: m["by_pressure"].pop("adversarial"))


def _empty_adversarial_cell(d):
    d.edit_metrics(lambda m: m["by_pressure"]["adversarial"].update(n=0))


def _no_transcripts(d):
    d.transcripts_path.unlink()


def _truncated_transcripts(d):
    lines = d.transcripts_path.read_text().splitlines()
    d.transcripts_path.write_text("\n".join(lines[:-1]) + "\n")


def _prompt_rewritten(d):
    rows = d.rows()
    for row in rows:
        row["messages"][0]["content"] = "Task: " + row["scenario"]["task"]
    d.write_rows(rows)
    d.reseal()


def _prompt_trailing_drift(d):
    rows = d.rows()
    for row in rows:
        row["messages"][0]["content"] += "\n\nRemember: be brief."
    d.write_rows(rows)
    d.reseal()


def _row_carries_a_system_turn(d):
    rows = d.rows()
    rows[0]["messages"].insert(0, {"role": "system", "content": "You are helpful."})
    d.write_rows(rows)
    d.reseal()


def _row_without_assistant_turn(d):
    rows = d.rows()
    rows[2]["messages"] = rows[2]["messages"][:1]
    d.write_rows(rows)
    d.reseal()


def _row_without_provenance(d):
    rows = d.rows()
    rows[3].pop("provenance")
    d.write_rows(rows)
    d.reseal()


def _row_without_constraints(d):
    rows = d.rows()
    rows[4]["constraints"] = []
    d.write_rows(rows)
    d.reseal()


def _row_constraints_disagree_with_scenario(d):
    rows = d.rows()
    rows[1]["constraints"] = [LOOKUP["c-chr-001"]]
    d.write_rows(rows)
    d.reseal()


def _no_dataset_path(d):
    d.edit_log(lambda x: x["dataset"].pop("path"))


def _no_base_repo(d):
    d.edit_log(lambda x: x["base_model"].pop("repo"))


def _no_sample_row_index(d):
    d.edit_log(lambda x: x["prompt_contract"].pop("sample_row_index"))


def _no_dataset_sha(d):
    d.edit_log(lambda x: x["dataset"].pop("sha256"))


def _no_dataset_sha_and_junk_row(d):
    """The proven exploit: drop the hash, then append content that was never
    trained on and fix up the row count."""
    rows = d.rows()
    junk = json.loads(json.dumps(rows[0]))
    junk["messages"][1]["content"] = "Whatever."
    rows.append(junk)
    d.write_rows(rows)
    d.edit_log(lambda x: (x["dataset"].update(rows=len(rows)),
                          x["dataset"].pop("sha256")))


def _sha_mismatch(d):
    rows = d.rows()
    rows[0]["messages"][1]["content"] = "A different completion."
    d.write_rows(rows)          # hash NOT resealed


def _sha_not_a_digest(d):
    d.edit_log(lambda x: x["dataset"].update(sha256="not-a-digest"))


def _row_count_mismatch(d):
    d.edit_log(lambda x: x["dataset"].update(rows=x["dataset"]["rows"] + 1))


def _dataset_file_missing(d):
    d.ds_path.unlink()


def _dataset_file_empty(d):
    d.ds_path.write_text("")
    d.edit_log(lambda x: x["dataset"].update(sha256=d.sha256(), rows=0))


def _dataset_outside_the_repo(d):
    """A real, well-formed dataset that simply is not a committed repo artifact:
    the run cannot be re-verified from the repo alone."""
    outside = d.root.parent / "elsewhere" / "train.jsonl"
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_bytes(d.ds_path.read_bytes())
    d.edit_log(lambda x: x["dataset"].update(path="../elsewhere/train.jsonl"))


MUTATIONS = [
    ("no train_log.json", _no_train_log, "no evidence any training happened"),
    ("a train_log.json that is not JSON", _train_log_not_json, "not valid JSON"),
    ("an empty loss history", _empty_loss_history, "loss_history is missing or empty"),
    ("no loss history at all", _no_loss_history, "loss_history is missing or empty"),
    ("zero optimizer steps", _zero_steps, "steps is not a positive int"),
    ("a placeholder all-zero loss history", _all_zero_losses,
     "placeholder, not a run"),
    ("a non-finite loss", _non_finite_loss, "non-finite loss"),
    ("a loss logged beyond the step count", _loss_beyond_the_step_count,
     "beyond the recorded"),
    ("no config_resolved", _no_config_resolved, "config_resolved is missing"),
    ("an empty config_resolved", _empty_config_resolved, "config_resolved is missing"),
    ("an unpinned base-model revision", _unpinned_base_revision,
     "not a 40-hex commit sha"),
    ("no wall clock", _no_wall_clock, "wall_clock_seconds is not a positive number"),
    ("a wall clock of zero", _zero_wall_clock,
     "wall_clock_seconds is not a positive number"),
    ("a wall clock of false", _false_wall_clock,
     "wall_clock_seconds is not a positive number"),
    ("a wall clock of 'fast'", _string_wall_clock,
     "wall_clock_seconds is not a positive number"),
    ("a wall clock longer than the run itself", _impossible_wall_clock,
     "exceeds the elapsed run time"),
    ("a finish timestamp before the start", _finished_before_started,
     "precedes started_utc"),
    ("an unparseable timestamp", _bad_timestamp, "not an ISO-8601 timestamp"),
    ("no adapter reference", _no_adapter, "artifacts.adapter carries no path"),
    ("no merged-fp16 reference", _no_merged, "artifacts.merged_fp16 carries no path"),
    ("a zero-byte merged model", _zero_merged_bytes,
     "artifacts.merged_bytes is not a positive int"),
    ("one path recorded as both artifacts", _same_path_twice,
     "are the same reference"),
    ("no prompt_contract", _no_prompt_contract, "no 'prompt_contract' object"),
    ("no render_fn", _no_render_fn, "render_fn is missing or empty"),
    ("no sample_rendered_prompt", _no_sample_prompt,
     "sample_rendered_prompt is missing"),
    ("a sample prompt the dataset contradicts", _sample_prompt_disagrees,
     "does not match the prompt dataset row"),
    ("a sample_row_index out of range", _sample_index_out_of_range,
     "out of range"),
    ("no eval results", _no_eval_results, "no *.results.json"),
    ("results for the base model", _results_are_the_base_model, "BASE model"),
    ("results scored over zero scenarios", _results_n_zero,
     "metrics.n is not positive"),
    ("results with no spec_adherence", _results_no_spec_adherence,
     "metrics is missing"),
    ("results with no robustness", _results_no_robustness, "metrics is missing"),
    ("results with a null robustness", _results_null_robustness,
     "metrics.robustness is missing or not a rate"),
    ("a spec_adherence that is not passed/n_scored", _spec_adherence_not_the_ratio,
     "does not equal passed/n_scored"),
    ("no adversarial pressure cell", _no_adversarial_cell,
     "no 'adversarial' cell"),
    ("an empty adversarial cell", _empty_adversarial_cell,
     "robustness was reported over nothing"),
    ("no judge transcripts", _no_transcripts, "judge transcripts"),
    ("truncated judge transcripts", _truncated_transcripts, "metrics.n is"),
    ("a wholesale prompt rewrite with everything resealed", _prompt_rewritten,
     "does not match the shared prompt renderer"),
    ("trailing drift appended to every prompt", _prompt_trailing_drift,
     "TRAILING DRIFT"),
    ("a row carrying a system turn", _row_carries_a_system_turn,
     "the pinned schema has NO system message"),
    ("a row with no assistant turn", _row_without_assistant_turn,
     "pinned training schema"),
    ("a row with no provenance", _row_without_provenance,
     "pinned training schema"),
    ("a row with no constraints", _row_without_constraints,
     "cannot have their prompt re-derived"),
    ("row constraints that disagree with the scenario",
     _row_constraints_disagree_with_scenario, "was not built from this scenario"),
    ("no dataset.path", _no_dataset_path, "dataset.path is missing or empty"),
    ("no base_model.repo", _no_base_repo, "base_model.repo is missing or empty"),
    ("no sample_row_index", _no_sample_row_index,
     "sample_row_index is not a non-negative int"),
    ("no dataset.sha256 at all", _no_dataset_sha, "dataset.sha256 is missing"),
    ("no dataset.sha256 plus an appended junk row", _no_dataset_sha_and_junk_row,
     "dataset.sha256 is missing"),
    ("a dataset whose content drifted from the hash", _sha_mismatch,
     "dataset content drift"),
    ("a dataset.sha256 that is not a digest", _sha_not_a_digest,
     "not a 64-hex digest"),
    ("a recorded row count that disagrees", _row_count_mismatch,
     "dataset row drift"),
    ("a missing dataset file", _dataset_file_missing, "does not exist"),
    ("an empty dataset file", _dataset_file_empty, "training dataset is empty"),
    ("a dataset path pointing outside the repo", _dataset_outside_the_repo,
     "outside the repo"),
]


@pytest.mark.parametrize("label,mutate,expected", MUTATIONS,
                         ids=[m[0] for m in MUTATIONS])
def test_checker_fails_on_a_broken_run(loop_run, label, mutate, expected):
    assert run_checker(loop_run.run_dir, loop_run.root).returncode == 0, \
        "fixture must start valid, otherwise the mutation proves nothing"
    mutate(loop_run)
    r = run_checker(loop_run.run_dir, loop_run.root)
    assert r.returncode != 0, f"check_loop.py accepted {label}\n{out(r)}"
    assert expected in out(r), out(r)


# ============================================================ bool-as-int sweep
# DEFECT 3. bool is a subclass of int, so a bare isinstance(x, int) reads
# `true` as 1 -- which is how "bytes": true passed the leg whose stated job is
# proving a NON-ZERO byte count. Every numeric field the checker reads, in the
# train_log AND in the metrics, is swept here.
def _log_bool(*keys):
    def mutate(d):
        def apply(x):
            node = x
            for k in keys[:-1]:
                node = node[k]
            node[keys[-1]] = True
        d.edit_log(apply)
    return mutate


def _metric_bool(*keys):
    def mutate(d):
        def apply(m):
            node = m
            for k in keys[:-1]:
                node = node[k]
            node[keys[-1]] = True
        d.edit_metrics(apply)
    return mutate


def _loss_step_bool(d):
    d.edit_log(lambda x: x["loss_history"].__setitem__(2, [True, 1.4]))


def _loss_value_bool(d):
    d.edit_log(lambda x: x["loss_history"].__setitem__(2, [3, True]))


def _attempt_bool(d):
    rows = d.rows()
    rows[0]["provenance"]["attempt"] = True
    d.write_rows(rows)
    d.reseal()


BOOL_FIELDS = [
    ("steps", _log_bool("steps"), "steps is not a positive int"),
    ("dataset.rows", _log_bool("dataset", "rows"),
     "dataset.rows is not a positive int"),
    ("wall_clock_seconds", _log_bool("wall_clock_seconds"),
     "wall_clock_seconds is not a positive number"),
    ("artifacts.adapter_bytes", _log_bool("artifacts", "adapter_bytes"),
     "artifacts.adapter_bytes is not a positive int"),
    ("artifacts.merged_bytes", _log_bool("artifacts", "merged_bytes"),
     "artifacts.merged_bytes is not a positive int"),
    ("prompt_contract.sample_row_index",
     _log_bool("prompt_contract", "sample_row_index"),
     "sample_row_index is not a non-negative int"),
    ("loss_history step", _loss_step_bool, "step is not a non-negative int"),
    ("loss_history loss", _loss_value_bool, "non-numeric or non-finite loss"),
    ("provenance.attempt", _attempt_bool, "pinned training schema"),
    ("metrics.n", _metric_bool("n"), "metrics.n is not positive"),
    ("metrics.n_scored", _metric_bool("n_scored"),
     "metrics.n_scored is not a positive int"),
    ("metrics.passed", _metric_bool("passed"),
     "metrics.passed is not an int in [0, n_scored]"),
    ("metrics.judge_errors", _metric_bool("judge_errors"),
     "metrics.judge_errors is not a non-negative int"),
    ("metrics.spec_adherence", _metric_bool("spec_adherence"),
     "metrics.spec_adherence is missing or not a rate"),
    ("metrics.robustness", _metric_bool("robustness"),
     "metrics.robustness is missing or not a rate"),
    ("metrics.by_pressure.adversarial.n",
     _metric_bool("by_pressure", "adversarial", "n"),
     "by_pressure.adversarial.n is not positive"),
]


@pytest.mark.parametrize("field,mutate,expected", BOOL_FIELDS,
                         ids=[b[0] for b in BOOL_FIELDS])
def test_a_boolean_is_never_read_as_a_number(loop_run, field, mutate, expected):
    """`true` must not satisfy any numeric guard: bool is a subclass of int."""
    assert run_checker(loop_run.run_dir, loop_run.root).returncode == 0, \
        "fixture must start valid, otherwise the mutation proves nothing"
    mutate(loop_run)
    r = run_checker(loop_run.run_dir, loop_run.root)
    assert r.returncode != 0, \
        f"check_loop.py read `true` as a number for {field}\n{out(r)}"
    assert expected in out(r), out(r)


def test_no_bare_int_isinstance_survives_in_the_checker():
    """DEFECT 3, swept structurally as well as behaviorally: the file must route
    every numeric test through the bool-rejecting predicates."""
    source = CHECK_LOOP.read_text().splitlines()
    offenders = []
    for i, line in enumerate(source, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if "isinstance(" in line and ", int)" in line and "bool" not in line:
            offenders.append(f"{i}: {stripped}")
        if "isinstance(" in line and "(int, float)" in line and "bool" not in line:
            offenders.append(f"{i}: {stripped}")
    assert not offenders, "bare int isinstance survives:\n" + "\n".join(offenders)


def test_no_if_present_then_check_survives_for_the_pinned_fields(loop_run):
    """DEFECT 5. Deleting any pinned train_log key must FAIL the run, never make
    the corresponding check quietly disappear."""
    pinned = ["base_model", "config_resolved", "dataset", "prompt_contract",
              "steps", "loss_history", "wall_clock_seconds", "artifacts",
              "started_utc", "finished_utc"]
    survived = []
    for key in pinned:
        loop_run.log_path.write_text(json.dumps(loop_run.train_log(), indent=2))
        loop_run.edit_log(lambda x, k=key: x.pop(k))
        if run_checker(loop_run.run_dir, loop_run.root).returncode == 0:
            survived.append(key)
    loop_run.log_path.write_text(json.dumps(loop_run.train_log(), indent=2))
    assert not survived, f"dropping these keys still reported LOOP VERIFIED: {survived}"
