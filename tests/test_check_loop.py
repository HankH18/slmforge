"""Proof that scripts/check_loop.py -- the P4 loop gate -- is a GATE, not a printer.

The MVP bar the spec grades is "full loop, generate -> train -> eval, runs end to
end". check_loop.py is the only thing that re-verifies that claim offline. So it
must exit 0 on a well-formed run tree and non-zero on every corruption that would
make the claim false -- above all on the one contract that makes the tuned
model's numbers mean anything: every training row's system message is
byte-identical to eval.SYSTEM_CORE and its user message is eval.user_msg(). That
contract has to be re-derived FROM THE DATASET, never read off the
`system_prompt_equals_eval_SYSTEM_CORE` boolean, which training/modal_app.py
writes as a hardcoded literal True.

Network-free: the run tree is synthesized in tmp_path from the frozen eval set
with the frozen eval.write_outputs/eval.aggregate. Nothing here reads or writes
the real runs/ directory.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

import eval as E
from conftest import REPO_ROOT

SCRIPTS = REPO_ROOT / "scripts"
RUBRIC = "c0adb876c3e6c4c9"
JUDGE = "claude-sonnet-4-6"

RUN_NAME = "smoke-20260820-001"
BASE_SHA = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
BASE_SPEC = f"hf:Qwen/Qwen3-1.7B@{BASE_SHA}"
TUNED_SHA = "0123456789abcdef0123456789abcdef01234567"
TUNED_SPEC = f"hf:tester/grounded-abstainer-{RUN_NAME}@{TUNED_SHA}"
DATASET_REL = "data/train/train_smoke.jsonl"
N_TRAIN_ROWS = 12


def run_checker(run_dir, repo_root, *extra):
    return subprocess.run(
        [sys.executable, str(SCRIPTS / "check_loop.py"), str(run_dir),
         "--repo-root", str(repo_root), *map(str, extra)],
        cwd=str(REPO_ROOT), capture_output=True, text=True)


def out(proc):
    return proc.stdout + proc.stderr


# --------------------------------------------------------------- synthesis
def training_rows(scenarios, n=N_TRAIN_ROWS):
    """P3-schema training rows, built through eval.SYSTEM_CORE / eval.user_msg
    exactly as datagen.filter.build_row does.

    The passages come from the frozen eval set only because it is real,
    on-contract text; check_loop.py does no leakage checking, so nothing here
    depends on train/eval separation.
    """
    rows = []
    for s in scenarios[:n]:
        if s["gold_label"] == "answerable":
            assistant = f'Answer: {s["gold_answer"]}\nCitation: "{s["gold_spans"][0]}"'
        else:
            assistant = E.REFUSAL_STRING
        rows.append({
            "messages": [
                {"role": "system", "content": E.SYSTEM_CORE},
                {"role": "user", "content": E.user_msg(s["passage"], s["question"])},
                {"role": "assistant", "content": assistant},
            ],
            "meta": {"type": s.get("type"), "gold_label": s["gold_label"],
                     "source_batch": "smoke-batch-1",
                     "verifier": {"rubric_hash": E.rubric_hash()}},
        })
    return rows


def eval_records(scenarios, n_answerable_fail=2):
    """Plausible per-example judge transcripts over the frozen eval set."""
    records, left = [], n_answerable_fail
    for s in scenarios:
        rec = {"scenario_id": s["id"], "type": s.get("type"),
               "adversarial": bool(s.get("adversarial")),
               "gold_label": s["gold_label"], "raw_output": E.REFUSAL_STRING,
               "pred_class": "refusal", "judge_verdict": None,
               "judge_reasoning": None, "passed": True, "failure": None,
               "model": TUNED_SPEC, "strategy": "zero_shot"}
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
        records.append(rec)
    return records


class LoopRun:
    """A well-formed committed-artifact tree that check_loop.py must accept."""

    def __init__(self, root, scenarios):
        self.root = Path(root)
        self.run_dir = self.root / "runs" / "smoke"
        self.eval_dir = self.run_dir / "eval"
        self.ds_path = self.root / DATASET_REL
        self.log_path = self.run_dir / "train_log.json"
        self.stem = f"{E.sanitize(TUNED_SPEC)}__zero_shot"
        self.results_path = self.eval_dir / f"{self.stem}.results.json"
        self.transcripts_path = self.eval_dir / f"{self.stem}.transcripts.jsonl"

        self.ds_path.parent.mkdir(parents=True, exist_ok=True)
        self.write_rows(training_rows(scenarios))

        records = eval_records(scenarios)
        E.write_outputs(self.eval_dir, TUNED_SPEC, "zero_shot", records,
                        E.aggregate(records), JUDGE)

        self.log_path.write_text(json.dumps(self.train_log(), indent=2))

    # ---------------------------------------------------------- accessors
    def rows(self):
        return [json.loads(ln) for ln in self.ds_path.read_text().splitlines()
                if ln.strip()]

    def write_rows(self, rows):
        self.ds_path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    def sha256(self):
        import hashlib
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

    def reseal(self):
        """Recompute dataset.sha256 and dataset.rows -- what an attacker (or a
        careless regeneration) does to keep the integrity checks quiet."""
        self.edit_log(lambda d: d["dataset"].update(
            sha256=self.sha256(), rows=len(self.rows())))

    def train_log(self):
        n = len(self.rows())
        return {
            "run_name": RUN_NAME,
            "smoke": True,
            "started_utc": "2026-08-20T00:00:00+00:00",
            "finished_utc": "2026-08-20T00:12:00+00:00",
            "steps": 20,
            "epochs_completed": 1.0,
            "final_train_loss": 0.4123,
            "loss_curve": [{"step": i, "loss": 1.9 - 0.07 * i} for i in range(1, 21)],
            "wall_time_s": {"model_load": 61.2, "tokenize": 3.4, "train": 412.5,
                            "save_adapter": 8.1, "save_merged": 74.0,
                            "total": 559.2},
            "dataset": {"path": DATASET_REL, "rows": n, "rows_trained_on": n,
                        "rows_truncated": 0, "rows_skipped": 0,
                        "prefix_mismatch": 0, "sha256": self.sha256()},
            "base_model": {"repo": "Qwen/Qwen3-1.7B", "revision": BASE_SHA,
                           "spec": BASE_SPEC},
            "rubric_hash": RUBRIC,
            "rubric_version": E.RUBRIC_VERSION,
            "system_prompt_equals_eval_SYSTEM_CORE": True,
            "artifacts": {
                "volume": "slmforge-artifacts",
                "run_dir_remote": f"/vol/{RUN_NAME}",
                "adapter": {"remote_path": f"/vol/{RUN_NAME}/adapter",
                            "volume_path": f"{RUN_NAME}/adapter",
                            "files": ["adapter_config.json",
                                      "adapter_model.safetensors"],
                            "bytes": 73_400_320},
                "merged": {"remote_path": f"/vol/{RUN_NAME}/merged_fp16",
                           "volume_path": f"{RUN_NAME}/merged_fp16",
                           "files": ["config.json", "model.safetensors",
                                     "tokenizer.json"],
                           "bytes": 3_400_000_000,
                           "save_method": "merged_16bit"},
                "train_log": f"/vol/{RUN_NAME}/train_log.json",
            },
            "gpu": "L4",
        }


@pytest.fixture
def loop_run(tmp_path, scenarios):
    return LoopRun(tmp_path / "repo", scenarios)


# ============================================================ happy path
def test_checker_passes_on_a_well_formed_run(loop_run):
    r = run_checker(loop_run.run_dir, loop_run.root)
    assert r.returncode == 0, out(r)
    assert "LOOP VERIFIED: 5/5 legs pass" in r.stdout
    # the prompt contract is reported as re-derived from the rows, not attested
    assert "eval.SYSTEM_CORE" in r.stdout


def test_json_report_is_machine_readable_and_agrees(loop_run):
    r = run_checker(loop_run.run_dir, loop_run.root, "--json")
    assert r.returncode == 0, out(r)
    report = json.loads(r.stdout)
    assert report["ok"] is True
    assert len(report["legs"]) == 5
    assert all(leg["ok"] for leg in report["legs"])


def test_checker_fails_on_a_missing_run_directory(tmp_path):
    r = run_checker(tmp_path / "nope", tmp_path)
    assert r.returncode != 0
    assert "is not a directory" in out(r)


# ============================================================ the load-bearing one
def test_checker_does_not_trust_the_self_attested_system_prompt_flag(loop_run):
    """DEFECT 1 regression.

    training/modal_app.py writes system_prompt_equals_eval_SYSTEM_CORE as a
    hardcoded literal True -- the checked thing attesting to itself. Rewrite every
    row's system message, reseal the hash and the row count so leg 1's integrity
    checks stay quiet, leave the flag True: the checker must still fail, because
    it re-derives the contract from the dataset.
    """
    rows = loop_run.rows()
    for row in rows:
        row["messages"][0]["content"] = "You are a helpful assistant."
    loop_run.write_rows(rows)
    loop_run.reseal()

    log = loop_run.log()
    assert log["system_prompt_equals_eval_SYSTEM_CORE"] is True
    assert log["dataset"]["sha256"] == loop_run.sha256(), "hash must still match"
    assert log["dataset"]["rows"] == len(rows), "row count must still match"

    r = run_checker(loop_run.run_dir, loop_run.root)
    assert r.returncode != 0, f"checker accepted a drifted system prompt\n{out(r)}"
    assert "byte-identical to eval.SYSTEM_CORE" in out(r)


def test_checker_catches_a_single_drifted_row_among_many(loop_run):
    """One bad row in twelve is still a broken training condition."""
    rows = loop_run.rows()
    rows[7]["messages"][0]["content"] = E.SYSTEM_CORE + "\nAlways be concise."
    loop_run.write_rows(rows)
    loop_run.reseal()
    r = run_checker(loop_run.run_dir, loop_run.root)
    assert r.returncode != 0, out(r)
    assert "1/12 dataset rows" in out(r)


def test_checker_catches_a_user_message_not_built_by_eval_user_msg(loop_run):
    rows = loop_run.rows()
    for row in rows:
        content = row["messages"][1]["content"]
        row["messages"][1]["content"] = content.replace("Context passage:\n",
                                                        "Passage:\n")
    loop_run.write_rows(rows)
    loop_run.reseal()
    r = run_checker(loop_run.run_dir, loop_run.root)
    assert r.returncode != 0, out(r)
    assert "eval.user_msg" in out(r)


def test_checker_survives_a_passage_containing_the_question_separator(loop_run,
                                                                     scenarios):
    """The round-trip must not false-positive on a passage that itself contains
    the '\\n\\nQuestion: ' separator -- eval.user_msg is still what built it."""
    rows = loop_run.rows()
    tricky = ("Line one.\n\nQuestion: this text is part of the passage.\n"
              "Line three.")
    rows[0]["messages"][1]["content"] = E.user_msg(tricky, "What is line three?")
    loop_run.write_rows(rows)
    loop_run.reseal()
    r = run_checker(loop_run.run_dir, loop_run.root)
    assert r.returncode == 0, out(r)


# ============================================================ sabotage table
def _no_train_log(d):
    d.log_path.unlink()


def _train_log_not_json(d):
    d.log_path.write_text("{not json")


def _empty_loss_curve(d):
    d.edit_log(lambda x: x.update(loss_curve=[]))


def _zero_steps(d):
    d.edit_log(lambda x: x.update(steps=0))


def _steps_is_a_boolean(d):
    d.edit_log(lambda x: x.update(steps=True))


def _all_zero_losses(d):
    d.edit_log(lambda x: x.update(
        loss_curve=[{"step": i, "loss": 0.0} for i in range(1, 21)]))


def _non_finite_loss(d):
    d.edit_log(lambda x: x["loss_curve"].__setitem__(
        3, {"step": 4, "loss": float("nan")}))


def _no_train_wall_time(d):
    d.edit_log(lambda x: x.pop("wall_time_s"))


def _no_adapter(d):
    d.edit_log(lambda x: x["artifacts"].pop("adapter"))


def _no_merged(d):
    d.edit_log(lambda x: x["artifacts"].pop("merged"))


def _adapter_files_empty(d):
    d.edit_log(lambda x: x["artifacts"]["adapter"].update(files=[]))


def _merged_zero_bytes(d):
    d.edit_log(lambda x: x["artifacts"]["merged"].update(bytes=0))


def _no_artifact_volume(d):
    d.edit_log(lambda x: x["artifacts"].pop("volume"))


def _no_base_model_spec(d):
    d.edit_log(lambda x: x["base_model"].pop("spec"))


def _train_log_rubric_drift(d):
    d.edit_log(lambda x: x.update(rubric_hash="deadbeefdeadbeef"))


def _attestation_false(d):
    d.edit_log(lambda x: x.update(system_prompt_equals_eval_SYSTEM_CORE=False))


def _no_eval_results(d):
    d.results_path.unlink()


def _results_wrong_rubric(d):
    d.edit_results(lambda x: x.update(rubric_hash="deadbeefdeadbeef"))


def _results_are_the_base_model(d):
    d.edit_results(lambda x: x.update(model=BASE_SPEC))


def _results_are_some_other_model(d):
    d.edit_results(lambda x: x.update(model="hf:someone/unrelated-model@" + TUNED_SHA))


def _results_n_zero(d):
    d.edit_results(lambda x: x["metrics"].update(n=0))


def _results_no_spec_adherence(d):
    d.edit_results(lambda x: x["metrics"].pop("spec_adherence"))


def _no_transcripts(d):
    d.transcripts_path.unlink()


def _truncated_transcripts(d):
    lines = d.transcripts_path.read_text().splitlines()
    d.transcripts_path.write_text("\n".join(lines[:-1]) + "\n")


def _drifted_system_prompt(d):
    rows = d.rows()
    for row in rows:
        row["messages"][0]["content"] = "You are a helpful assistant."
    d.write_rows(rows)
    d.reseal()          # hash + row count resealed, attestation left True


def _drifted_user_message(d):
    rows = d.rows()
    for row in rows:
        msgs = row["messages"]
        msgs[1]["content"] = msgs[1]["content"].replace("\n\nQuestion: ", "\nQ: ")
    d.write_rows(rows)
    d.reseal()


def _row_without_assistant_turn(d):
    rows = d.rows()
    rows[2]["messages"] = rows[2]["messages"][:2]
    d.write_rows(rows)
    d.reseal()


def _no_dataset_sha(d):
    d.edit_log(lambda x: x["dataset"].pop("sha256"))


def _no_dataset_sha_and_junk_row(d):
    """The proven exploit: drop the hash, then append content that was never
    trained on and fix up the row count."""
    rows = d.rows()
    junk = json.loads(json.dumps(rows[0]))
    junk["messages"][2]["content"] = "Answer: whatever\nCitation: \"nope\""
    rows.append(junk)
    d.write_rows(rows)
    d.edit_log(lambda x: x["dataset"].update(rows=len(rows)))
    d.edit_log(lambda x: x["dataset"].pop("sha256"))


def _sha_mismatch(d):
    rows = d.rows()
    rows[0]["messages"][2]["content"] = E.REFUSAL_STRING
    d.write_rows(rows)          # hash NOT resealed, row count unchanged


def _row_count_mismatch(d):
    d.edit_log(lambda x: x["dataset"].update(rows=x["dataset"]["rows"] + 1))


def _rows_trained_on_zero(d):
    d.edit_log(lambda x: x["dataset"].update(rows_trained_on=0))


def _no_rows_trained_on(d):
    d.edit_log(lambda x: x["dataset"].pop("rows_trained_on"))


def _dataset_file_missing(d):
    d.ds_path.unlink()


def _dataset_file_empty(d):
    d.ds_path.write_text("")
    d.edit_log(lambda x: x["dataset"].update(sha256=d.sha256()))


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
    ("an empty loss curve", _empty_loss_curve, "loss_curve is missing or empty"),
    ("zero optimizer steps", _zero_steps, "steps is not a positive int"),
    ("a step count recorded as a boolean", _steps_is_a_boolean,
     "steps is not a positive int"),
    ("a placeholder all-zero loss curve", _all_zero_losses,
     "placeholder, not a run"),
    ("a non-finite loss", _non_finite_loss, "non-finite loss"),
    ("no train wall time", _no_train_wall_time, "wall_time_s.train"),
    ("no adapter reference", _no_adapter, "artifacts.adapter is missing"),
    ("no merged-model reference", _no_merged, "artifacts.merged is missing"),
    ("an adapter with an empty file listing", _adapter_files_empty,
     "artifacts.adapter.files is empty"),
    ("a zero-byte merged model", _merged_zero_bytes,
     "artifacts.merged.bytes is not positive"),
    ("no artifacts volume", _no_artifact_volume, "artifacts.volume is missing"),
    ("no base_model.spec", _no_base_model_spec, "base_model.spec is missing"),
    ("a train_log stamped with another rubric", _train_log_rubric_drift,
     "train_log.json:rubric_hash"),
    ("the system-prompt attestation set to false", _attestation_false,
     "does not attest"),
    ("no eval results", _no_eval_results, "no *.results.json"),
    ("results on a different rubric", _results_wrong_rubric, "rubric_hash"),
    ("results for the base model", _results_are_the_base_model, "BASE model"),
    ("results for an unrelated checkpoint", _results_are_some_other_model,
     "does not reference this run's checkpoint"),
    ("results scored over zero scenarios", _results_n_zero,
     "metrics.n is not positive"),
    ("results with no spec_adherence", _results_no_spec_adherence,
     "spec_adherence is missing"),
    ("no judge transcripts", _no_transcripts, "judge transcripts"),
    ("truncated judge transcripts", _truncated_transcripts, "metrics.n is"),
    ("a drifted system prompt with the hash resealed", _drifted_system_prompt,
     "byte-identical to eval.SYSTEM_CORE"),
    ("a user message not built by eval.user_msg", _drifted_user_message,
     "eval.user_msg"),
    ("a row with no assistant turn", _row_without_assistant_turn,
     "training schema"),
    ("no dataset.sha256 at all", _no_dataset_sha, "dataset.sha256 is missing"),
    ("no dataset.sha256 plus an appended junk row", _no_dataset_sha_and_junk_row,
     "dataset.sha256 is missing"),
    ("a dataset whose content drifted from the hash", _sha_mismatch,
     "dataset content drift"),
    ("a recorded row count that disagrees", _row_count_mismatch,
     "dataset row drift"),
    ("zero rows actually trained on", _rows_trained_on_zero, "rows_trained_on"),
    ("no rows_trained_on recorded", _no_rows_trained_on, "rows_trained_on"),
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
