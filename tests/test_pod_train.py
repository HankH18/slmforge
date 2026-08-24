"""Proof that training/pod_train.py swaps the HOST and nothing else.

The four curve checkpoints were trained by the function body in
training/modal_app.py. Modal is gone, so the retrain runs that SAME body on a
rented CUDA pod via modal.Function.local(). The thing worth testing is therefore
not "does the driver train" -- it is "is the driver incapable of quietly
changing the recipe while claiming it did not". Four ways it could:

  1. a divergent CLI, so a flag the Modal path honoured is silently dropped;
  2. SLMFORGE_MICRO_BATCH read in the wrong place, or defaulted, so the batch
     factorization in train_log.config_resolved stops describing the run;
  3. a train_log.json whose shape scripts/check_loop.py rejects at gate time,
     after the GPU is paid for;
  4. artifact references in the Volume-relative form, which leg 4 -- whole
     canonical string, never substring -- can never match against an eval.

Every test here is network-free, GPU-free and model-free: the trainer is
replaced by a fake, and the modules the real body imports first (torch, peft,
bitsandbytes, transformers) are replaced by tripwires that raise on ANY
attribute access, so a test cannot pass while the real body secretly ran.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import shutil
import sys
import types
from pathlib import Path

import pytest

from conftest import REPO_ROOT

POD_TRAIN = REPO_ROOT / "training" / "pod_train.py"
MODAL_APP = REPO_ROOT / "training" / "modal_app.py"
CHECK_LOOP = REPO_ROOT / "scripts" / "check_loop.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def pod():
    assert POD_TRAIN.exists(), f"{POD_TRAIN} missing -- the off-Modal driver"
    return _load("slmforge_pod_train", POD_TRAIN)


@pytest.fixture(scope="module")
def app_mod(pod):
    """The SAME modal_app module object pod_train imported.

    Fetched off the driver rather than re-imported, because monkeypatching
    `train` on a second copy would patch a module the driver never calls -- the
    exact way a "no model was loaded" test can be green and prove nothing.
    """
    return pod.M


@pytest.fixture(scope="module")
def check_loop():
    return _load("slmforge_check_loop_podtest", CHECK_LOOP)


@pytest.fixture(scope="module")
def train_cfg():
    return json.loads((REPO_ROOT / "configs" / "train.json").read_text())


# =====================================================================
# 1. the CLI surface
# =====================================================================
def _modal_main_params():
    """The parameter names of modal_app.main, read off the UNDECORATED function.

    modal.App.local_entrypoint replaces main with a LocalEntrypoint whose
    __call__ is (*args, **kwargs), so inspect.signature on the wrapper says
    nothing. .info.raw_f is the real function -- deriving the expected flag set
    from it means this test breaks when the Modal CLI gains or loses an option,
    which is the whole point of calling it a mirror.
    """
    spec = importlib.util.spec_from_file_location("slmforge_modal_app_sig",
                                                  MODAL_APP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    raw = mod.main.info.raw_f
    return list(inspect.signature(raw).parameters)


def test_the_argparse_surface_is_exactly_modal_mains(pod):
    expected = {"--" + name.replace("_", "-") for name in _modal_main_params()}
    parser = pod.build_parser()
    got = set()
    for action in parser._actions:
        if action.dest == "help":
            continue
        got.update(action.option_strings)
    assert got == expected, (
        f"pod_train's CLI {sorted(got)} does not mirror modal_app.main's "
        f"{sorted(expected)} -- a flag that exists on one host and not the "
        f"other means the two runs are not the same run")


def test_every_modal_main_option_parses_and_lands_on_the_same_dest(pod):
    args = pod.build_parser().parse_args(
        ["--smoke", "--dataset", "data/train/x.jsonl",
         "--out-dir", "runs/x", "--run-name", "x"])
    for name in _modal_main_params():
        assert hasattr(args, name), f"--{name} did not land on dest {name!r}"
    assert args.smoke is True
    assert args.dataset == "data/train/x.jsonl"
    assert args.out_dir == "runs/x"
    assert args.run_name == "x"


def test_smoke_is_a_flag_not_a_value_taking_option(pod):
    args = pod.build_parser().parse_args([])
    assert args.smoke is False
    args = pod.build_parser().parse_args(["--smoke"])
    assert args.smoke is True


def test_defaults_resolve_the_same_dataset_and_out_dir_modal_main_does(pod,
                                                                      train_cfg):
    parser = pod.build_parser()
    smoke = pod.resolve_paths(parser.parse_args(["--smoke"]), train_cfg)
    full = pod.resolve_paths(parser.parse_args([]), train_cfg)
    assert smoke["dataset"] == train_cfg["data"]["smoke_train_jsonl"]
    assert full["dataset"] == train_cfg["data"]["train_jsonl"]
    assert smoke["out_dir"] == train_cfg["output"]["smoke_run_dir"]
    assert full["out_dir"] == train_cfg["output"]["run_dir"]
    # run_name follows out_dir so the weights land INSIDE the run directory the
    # eval and scripts/check_loop.py already read.
    assert smoke["run_name"] == Path(smoke["out_dir"]).name
    assert full["run_name"] == Path(full["out_dir"]).name


# =====================================================================
# 2. SLMFORGE_MICRO_BATCH -- read in the driver, threaded through the call
# =====================================================================
def test_micro_batch_is_read_from_the_environment(pod):
    assert pod.read_micro_batch({"SLMFORGE_MICRO_BATCH": "4"}) == 4


@pytest.mark.parametrize("env", [{}, {"SLMFORGE_MICRO_BATCH": ""}])
def test_unset_micro_batch_means_the_pinned_config_not_a_default(pod, env):
    """0 is what apply_micro_batch_split treats as 'leave the pinned config
    alone'. A driver that defaulted to 4 would halve the micro-batch of a run
    whose train_log claimed the pinned recipe."""
    assert pod.read_micro_batch(env) == 0


def test_unset_micro_batch_leaves_the_pinned_batch_untouched(pod, app_mod,
                                                            train_cfg):
    resolved = app_mod.apply_micro_batch_split(
        app_mod.resolve_config(train_cfg, smoke=False), pod.read_micro_batch({}))
    t = resolved["training"]
    assert t["per_device_train_batch_size"] == \
        train_cfg["training"]["per_device_train_batch_size"]
    assert t["gradient_accumulation_steps"] == \
        train_cfg["training"]["gradient_accumulation_steps"]
    assert "_micro_batch_split" not in resolved, (
        "an UNSET micro batch recorded a split; config_resolved would then "
        "describe a factorization that never happened")


def test_micro_batch_reaches_the_trainer_call(pod, run_driver):
    """The value must arrive as an ARGUMENT. Reading it inside the body would
    work on a pod and silently stop working the day this runs on Modal again,
    where the container inherits none of the launching shell's environment."""
    res = run_driver(env={"SLMFORGE_MICRO_BATCH": "4"})
    assert res["calls"], "the trainer was never called"
    assert res["calls"][0]["micro_batch"] == 4


def test_unset_micro_batch_reaches_the_trainer_call_as_zero(pod, run_driver):
    res = run_driver(env={})
    assert res["calls"][0]["micro_batch"] == 0


def test_the_trainer_is_invoked_through_local_not_remote(pod, run_driver):
    """`.remote()` is a control-plane call and would need the Modal account that
    no longer exists. `.local()` runs the real decorated body in-process, which
    is what makes this a host swap rather than a second trainer."""
    res = run_driver(env={})
    assert res["how"] == ["local"], (
        f"the trainer was reached via {res['how']}, not .local()")


# =====================================================================
# 3. the emitted artifacts have exactly the pinned key sets
# =====================================================================
def test_emitted_train_log_key_sets_equal_the_pinned_constants(pod, app_mod,
                                                               run_driver):
    res = run_driver(env={})
    log = json.loads((res["out_path"] / "train_log.json").read_text())

    assert set(log) == set(app_mod.TRAIN_LOG_KEYS)
    assert set(log["base_model"]) == set(app_mod.BASE_MODEL_KEYS)
    assert set(log["prompt_contract"]) == set(app_mod.PROMPT_CONTRACT_KEYS)
    assert set(log["artifacts"]) == set(app_mod.ARTIFACT_KEYS)
    assert set(app_mod.DATASET_KEYS) <= set(log["dataset"])
    assert set(log["dataset"]) - set(app_mod.DATASET_KEYS) <= \
        set(pod.OPTIONAL_DATASET_KEYS)


def test_the_driver_reexports_the_writers_own_contract_objects(pod, app_mod):
    """Not a copy of the key tuples. A second copy could drift from the writer's
    and each copy would still agree with itself."""
    for name in ("TRAIN_LOG_KEYS", "BASE_MODEL_KEYS", "DATASET_KEYS",
                 "PROMPT_CONTRACT_KEYS", "ARTIFACT_KEYS"):
        assert getattr(pod, name) is getattr(app_mod, name)


def test_a_train_log_missing_a_pinned_key_is_refused_before_it_is_written(
        pod, app_mod, run_driver):
    def drop_a_key(manifest):
        manifest["train_log"].pop("loss_history")
        return manifest

    res = run_driver(env={}, mutate=drop_a_key, expect_rc=3)
    assert not (res["out_path"] / "train_log.json").exists(), (
        "a train_log the loop checker would reject was written to disk anyway")


def test_a_train_log_carrying_an_extra_key_is_refused(pod, run_driver):
    def add_a_key(manifest):
        manifest["train_log"]["verified"] = True
        return manifest

    res = run_driver(env={}, mutate=add_a_key, expect_rc=3)
    assert not (res["out_path"] / "train_log.json").exists()


def test_run_manifest_carries_what_a_grader_needs_to_judge_the_host_swap(
        pod, run_driver):
    res = run_driver(env={})
    man = json.loads((res["out_path"] / "run_manifest.json").read_text())
    assert "train_log" not in man, (
        "run_manifest duplicated train_log; the pinned artifact must be the "
        "only place that shape lives")
    ev = man["host_swap"]
    assert ev["gpu"] == "NVIDIA A100-SXM4-80GB"
    assert ev["peak_vram_bytes"] > 0
    assert ev["compute_dtype_4bit"] == "torch.bfloat16"
    for name in ("torch", "transformers", "peft", "bitsandbytes"):
        assert ev["versions"][name] not in (None, "", "unknown"), (
            f"{name} version is not recorded; the host swap cannot be judged "
            f"without knowing which stack ran")
    assert ev["unverified"] == []


def test_the_pinned_stack_is_read_from_the_modal_image_not_restated(pod,
                                                                    app_mod):
    pinned = pod.pinned_stack()
    assert pinned == {s.split("==")[0]: s.split("==")[1]
                      for s in app_mod.PIP_PACKAGES if "==" in s}
    assert pinned["torch"] == "2.12.1"
    assert pinned["transformers"] == "4.57.6"


def test_a_matching_stack_reports_no_divergence(pod):
    assert pod.stack_divergence(pod.pinned_stack()) == []


def test_a_different_torch_is_reported_as_divergence(pod):
    observed = dict(pod.pinned_stack())
    observed["torch"] = "2.13.0"
    diffs = pod.stack_divergence(observed)
    assert any("torch: pinned 2.12.1, host has 2.13.0" in d for d in diffs)


def test_an_absent_package_is_divergence_not_a_pass(pod):
    """An unknown version is not a matching one. transformers 5.x removed
    TrainingArguments.warmup_ratio, which configs/train.json pins -- 'not
    reported' must never read as 'fine'."""
    observed = dict(pod.pinned_stack())
    observed["peft"] = None
    observed["bitsandbytes"] = "unknown"
    diffs = pod.stack_divergence(observed)
    assert any(d.startswith("peft: pinned") for d in diffs)
    assert any(d.startswith("bitsandbytes: pinned") for d in diffs)


def test_the_run_manifest_records_the_stack_comparison(pod, run_driver):
    res = run_driver(env={})
    ev = json.loads((res["out_path"] / "run_manifest.json").read_text())["host_swap"]
    assert ev["pinned_stack"]["torch"] == "2.12.1"
    # The fake manifest reports exactly the pinned versions for the five
    # packages the trainer measures, so the only divergences are the packages
    # the trainer does not probe at all.
    assert not [d for d in ev["stack_divergence"]
                if d.split(":")[0] in ("torch", "transformers", "peft",
                                       "bitsandbytes", "accelerate")]


def test_a_run_on_the_wrong_stack_is_flagged_in_the_manifest(pod, run_driver):
    def wrong_torch(manifest):
        manifest["env"]["torch"] = "2.13.0"
        return manifest

    res = run_driver(env={}, mutate=wrong_torch)
    ev = json.loads((res["out_path"] / "run_manifest.json").read_text())["host_swap"]
    assert any("torch: pinned 2.12.1, host has 2.13.0" in d
               for d in ev["stack_divergence"]), (
        "a run on a different torch was recorded as if the recipe were held "
        "constant")


def test_missing_host_evidence_is_named_rather_than_passed_over(pod):
    ev = pod.host_evidence({"gpu": "", "peak_vram_bytes": 0,
                            "env": {"torch": "2.12.1", "peft": "unknown"}})
    assert "gpu" in ev["unverified"]
    assert "peak_vram_bytes" in ev["unverified"]
    assert "compute_dtype_4bit" in ev["unverified"]
    assert "versions.peft" in ev["unverified"]


# =====================================================================
# 4. artifact references: repo-relative, and what leg 4 actually compares
# =====================================================================
def test_artifact_refs_are_repo_relative_paths_under_runs(pod, run_driver):
    res = run_driver(env={})
    art = json.loads((res["out_path"] / "train_log.json").read_text())["artifacts"]
    for key in ("adapter", "merged_fp16"):
        ref = art[key]
        assert not ref.startswith("/"), f"{key}={ref!r} is absolute"
        assert Path(ref).parts[0] == "runs", f"{key}={ref!r} is not under runs/"
        assert (REPO_ROOT / ref).resolve().is_relative_to(REPO_ROOT.resolve())


def test_artifact_refs_canonicalize_to_what_leg_four_compares(pod, check_loop,
                                                              run_driver):
    """The free win. check_loop.checkpoint_identities builds the set of
    canonical references leg 4 will accept, and _canon_ref strips the scheme
    from the eval's model spec. Both sides must land on the same string."""
    res = run_driver(env={})
    log = json.loads((res["out_path"] / "train_log.json").read_text())
    identities = check_loop.checkpoint_identities(log)

    for key, dirname in (("adapter", "adapter"), ("merged_fp16", "merged")):
        eval_spec = f"hf:{res['run_dir_rel']}/{dirname}"
        canon = check_loop._canon_ref(eval_spec)
        assert canon in identities, (
            f"an eval naming {eval_spec!r} canonicalizes to {canon!r}, which is "
            f"not in {sorted(identities)} -- leg 4 could never pass")
        assert identities[canon] == key


def test_the_old_volume_relative_form_is_what_leg_four_could_never_match(
        pod, check_loop, run_driver):
    """Regression proof that the change was load-bearing rather than cosmetic:
    with the Modal-era reference, the same eval spec does NOT match."""
    res = run_driver(env={})
    log = json.loads((res["out_path"] / "train_log.json").read_text())
    volume_era = dict(log)
    volume_era["artifacts"] = dict(log["artifacts"])
    volume_era["artifacts"]["adapter"] = f"{res['run_name']}/adapter"
    volume_era["artifacts"]["merged_fp16"] = f"{res['run_name']}/merged"

    identities = check_loop.checkpoint_identities(volume_era)
    canon = check_loop._canon_ref(f"hf:{res['run_dir_rel']}/merged")
    assert canon not in identities, (
        "the volume-relative reference matched, so this test proves nothing "
        "about why leg 4 used to fail")


def test_artifact_ref_is_a_no_op_inside_a_modal_container(app_mod, monkeypatch):
    """The whole edit to modal_app.py is only safe if it changes NOTHING on
    Modal. Force the on-Modal branch and check the old literal comes back."""
    monkeypatch.setattr(app_mod, "running_on_modal", lambda: True)
    assert app_mod.artifact_ref("v2", "merged") == "v2/merged"
    assert app_mod.artifact_root() == Path(app_mod.REMOTE_VOL)


def test_artifact_ref_off_modal_is_repo_relative(app_mod, monkeypatch):
    monkeypatch.setattr(app_mod, "running_on_modal", lambda: False)
    assert app_mod.artifact_ref("train_v4", "merged") == "runs/train_v4/merged"
    assert app_mod.artifact_root() == app_mod.REPO_ROOT / "runs"


def test_a_volume_relative_ref_is_refused_before_the_log_is_written(pod,
                                                                   run_driver):
    def volume_era(manifest):
        manifest["train_log"]["artifacts"]["merged_fp16"] = "v4/merged"
        return manifest

    res = run_driver(env={}, mutate=volume_era, expect_rc=3)
    assert not (res["out_path"] / "train_log.json").exists()


# =====================================================================
# 5. nothing loaded a model
# =====================================================================
def test_no_torch_model_was_constructed(pod, run_driver):
    """The trainer body's first act is `import torch` followed by
    `from peft import ...`. Both are replaced by tripwires for the whole driver
    run (see the run_driver fixture), so any attribute touch raises. Reaching
    this assertion at all is the proof."""
    res = run_driver(env={})
    assert res["tripped"] == [], (
        f"the real trainer body executed -- it touched {res['tripped']}")
    assert res["calls"], "the fake trainer was never called either; test is inert"


def test_the_tripwires_would_actually_fire(pod, tripwires):
    """A guard nobody has seen fail is not a guard. Prove the mechanism by
    tripping it deliberately."""
    tripped = tripwires["tripped"]
    with pytest.raises(AssertionError):
        sys.modules["torch"].cuda
    assert any("torch.cuda" in t for t in tripped)
    tripped.clear()


def test_the_preflights_are_the_ones_modal_main_runs(pod, train_cfg):
    """Not a smoke check of preflight() -- proof it resolves the REAL renderer
    and the REAL side corpora, so a run that would die inside the container
    dies here instead, before a pod-second is billed."""
    pre = pod.preflight(train_cfg, train_cfg["data"]["train_jsonl"])
    assert pre["render_fn"] == \
        train_cfg["data"]["prompt_renderer"].rsplit(".", 1)[1]
    assert pre["scenarios"] > 0, "the scenario corpus loaded empty"
    assert pre["constraints"] > 0, "the constraint corpus loaded empty"


def test_a_missing_dataset_fails_the_preflight_not_the_gpu(pod, train_cfg):
    with pytest.raises(pod.PodTrainError) as exc:
        pod.preflight(train_cfg, "data/train/does_not_exist.jsonl")
    assert "not found" in str(exc.value)


def test_a_declared_corpus_that_is_missing_is_an_error_not_a_skip(pod,
                                                                  train_cfg):
    """load_side_corpus refuses to silently skip a declared path: the rows that
    reference it by id would otherwise be unverifiable while everything passed."""
    cfg = json.loads(json.dumps(train_cfg))
    cfg["data"]["scenarios_path"] = "data/scenarios/gone.jsonl"
    with pytest.raises(FileNotFoundError):
        pod.preflight(cfg, cfg["data"]["train_jsonl"])


# =====================================================================
# fixtures: the fake trainer, the tripwires, and one driver run
# =====================================================================
class _Tripwire(types.ModuleType):
    """A module that raises on ANY attribute access, recording what was asked
    for. Installed over torch / peft / bitsandbytes / transformers so the real
    trainer body cannot run without saying so."""

    def __init__(self, name, log):
        super().__init__(name)
        self.__dict__["_log"] = log

    def __getattr__(self, item):
        if item.startswith("__"):
            raise AttributeError(item)
        self.__dict__["_log"].append(f"{self.__name__}.{item}")
        raise AssertionError(
            f"the real trainer body ran: it touched {self.__name__}.{item}")


@pytest.fixture
def tripwires(monkeypatch):
    tripped: list[str] = []
    for name in ("torch", "peft", "bitsandbytes", "transformers", "accelerate",
                 "huggingface_hub"):
        monkeypatch.setitem(sys.modules, name, _Tripwire(name, tripped))
    return {"tripped": tripped}


def _fake_manifest(app_mod, run_name: str, dataset: str) -> dict:
    """A manifest shaped exactly as train() returns one, with a train_log built
    by modal_app.build_train_log -- the real writer, so the key sets under test
    are the ones production emits and not a hand-typed imitation."""
    train_log = app_mod.build_train_log(
        base_repo="Qwen/Qwen3-1.7B",
        base_revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        config_resolved={"training": {"per_device_train_batch_size": 8}},
        dataset_path=dataset,
        dataset_rows=1815,
        dataset_sha256="ab" * 32,
        scenarios_path="data/scenarios/corpus.jsonl",
        constraints_path="data/scenarios/constraint_instances.jsonl",
        render_fn_dotted="scenarios.render.render_user_message",
        sample_row_index=0,
        sample_rendered_prompt="<|im_start|>user\nhello<|im_end|>\n",
        sample_prompt_messages=[{"role": "user", "content": "hello"}],
        steps=340,
        loss_history=[[1, 1.5], [2, 1.1]],
        wall_clock_seconds=1234.5,
        adapter_path=app_mod.artifact_ref(run_name, "adapter"),
        adapter_bytes=85670948,
        merged_path=app_mod.artifact_ref(run_name, "merged"),
        merged_bytes=3457069101,
        started_utc="2026-08-23T00:00:00+00:00",
        finished_utc="2026-08-23T00:20:34+00:00",
    )
    return {
        "run_name": run_name,
        "smoke": False,
        "volume": "slmforge-artifacts",
        "run_dir_remote": str(app_mod.artifact_root() / run_name),
        "host": "local_cuda_host",
        "gpu": "NVIDIA A100-SXM4-80GB",
        "gpu_requested": "pod",
        "peak_vram_bytes": 41 * 1024 ** 3,
        "peak_vram_gib": 41.0,
        "tokenization": {"kept_row_indices": [0]},
        "wall_time_s": {"train": 1200.0},
        "artifact_files": {"adapter": ["adapter_model.safetensors"],
                           "merged": ["model.safetensors"],
                           "merge_path": "peft.merge_and_unload(...)"},
        "env": {"python": "3.11.9", "torch": "2.12.1",
                "transformers": "4.57.6", "peft": "0.20.0",
                "bitsandbytes": "0.50.1", "accelerate": "1.14.0",
                "quantization": {"load_in_4bit": True,
                                 "bnb_4bit_quant_type": "nf4",
                                 "bnb_4bit_use_double_quant": True,
                                 "bnb_4bit_compute_dtype": "torch.bfloat16"},
                "gradient_checkpointing": True},
        "pull_commands": ["# already on this host"],
        "train_log": train_log,
    }


@pytest.fixture
def run_driver(pod, app_mod, tripwires, monkeypatch, tmp_path):
    """Run pod_train.main() end to end with the trainer replaced.

    The output directory is a real path under runs/ because check_artifact_refs
    resolves references against the repo root -- pointing REPO_ROOT at tmp_path
    would also move the dataset and the corpora, and the preflight this driver
    exists to run would stop being exercised. The directory is removed after.
    """
    made: list[Path] = []

    def _run(env, mutate=None, expect_rc=0, argv=None, run_name=None):
        run_name = run_name or f"_pytest_pod_{tmp_path.name}"
        out_dir = f"runs/{run_name}"
        out_path = REPO_ROOT / out_dir
        made.append(out_path)

        calls, how = [], []

        class FakeTrainer:
            def local(self, **kwargs):
                how.append("local")
                calls.append(kwargs)
                man = _fake_manifest(app_mod, kwargs["run_name"],
                                     kwargs["dataset_path"])
                return mutate(man) if mutate else man

            def remote(self, **kwargs):  # pragma: no cover - must never run
                how.append("remote")
                raise AssertionError(
                    ".remote() needs the Modal control plane this host cannot "
                    "reach; the driver must use .local()")

        monkeypatch.setattr(app_mod, "train", FakeTrainer())
        monkeypatch.setattr(app_mod, "running_on_modal", lambda: False)
        monkeypatch.setattr(pod.os, "environ", dict(env))

        rc = pod.main(argv if argv is not None else ["--out-dir", out_dir])
        assert rc == expect_rc, f"pod_train.main returned {rc}, expected {expect_rc}"
        return {"rc": rc, "calls": calls, "how": how, "out_path": out_path,
                "run_name": run_name, "run_dir_rel": out_dir,
                "tripped": tripwires["tripped"]}

    yield _run

    for path in made:
        shutil.rmtree(path, ignore_errors=True)
