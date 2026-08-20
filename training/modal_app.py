#!/usr/bin/env python3
"""
Grounded Abstainer -- QLoRA fine-tune on Modal.

EXACT INVOCATION (run from the repo root). Modal turns this file's
@app.local_entrypoint signature into the CLI, so `--smoke` is a real flag --
verified with `modal run training/modal_app.py::main --help`, which prints:
`--run-name TEXT`, `--out-dir TEXT`, `--dataset TEXT`, `--smoke / --no-smoke`.

    # P4 smoke loop: trains on data/train/smoke_v0.jsonl with smoke_overrides
    # (max_steps 20) applied, writes runs/smoke/train_log.json + run_manifest.json
    .venv/bin/modal run training/modal_app.py::main --smoke

    # P5 real run: trains on data/train/v1.jsonl with configs/train.json untouched,
    # writes runs/train_v1/
    .venv/bin/modal run training/modal_app.py::main

    # overrides (both optional)
    .venv/bin/modal run training/modal_app.py::main --smoke \
        --dataset data/train/smoke_v0.jsonl --out-dir runs/smoke --run-name smoke

    # see the generated CLI without touching the network
    .venv/bin/modal run training/modal_app.py::main --help

Artifacts. Weights are NOT returned through the control plane -- the adapter and
the merged fp16 model are written into the Modal Volume named in
configs/train.json:output.modal_volume and committed. The call returns only a
small JSON manifest (paths, file lists, byte counts) plus train_log.json, which
the local entrypoint writes under the run directory so scripts/check_loop.py can
re-verify the loop offline, without re-billing a GPU. Pull the weights with:

    .venv/bin/modal volume ls  slmforge-artifacts smoke
    .venv/bin/modal volume get slmforge-artifacts smoke/merged  runs/smoke/merged
    .venv/bin/modal volume get slmforge-artifacts smoke/adapter runs/smoke/adapter

Then close the loop (leg 3) against the pulled checkpoint:

    .venv/bin/python eval.py --model hf:runs/smoke/merged \
        --eval-set data/eval/scenarios_v1.jsonl --out runs/smoke/eval
    .venv/bin/python scripts/check_loop.py runs/smoke

Environment knobs (deliberately NOT in configs/train.json -- they are infra, not
hyperparameters, and train.json is frozen by policy):
    SLMFORGE_GPU         Modal GPU type,   default "L4"   (24GB, bf16-capable)
    SLMFORGE_TIMEOUT_S   function timeout, default 14400  (Modal's own default
                         is 300s and WILL kill a real run partway through)

Written against modal client 1.5.x: modal.App (modal.Stub does not exist),
gpu= as a string kwarg, secrets= as a list, Volume.from_name + vol.commit().
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "train.json"
REMOTE_REPO = "/repo"          # where the mounted repo slice lands in the container
REMOTE_VOL = "/vol"            # where the artifacts Volume is mounted
REMOTE_HF_CACHE = "/root/.cache/huggingface"

GPU = os.environ.get("SLMFORGE_GPU", "L4")
TIMEOUT_S = int(os.environ.get("SLMFORGE_TIMEOUT_S", "14400"))

# The one thing configs/train.json does NOT pin: the container's python deps.
# Unsloth needs a CUDA runtime and a torch build matched to it, so we start from
# NVIDIA's image rather than debian_slim. Versions are left to pip resolution on
# purpose -- pinning a version string that was never built here would be a lie.
# If the image build fails, this list is the single place to adjust.
PIP_PACKAGES = [
    "unsloth",
    "unsloth_zoo",
    "bitsandbytes",
    "accelerate",
    "peft",
    "transformers",
    "datasets",
    "trl",
    "sentencepiece",
    "protobuf",
    "huggingface_hub",
    "hf_transfer",
]


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


_CFG = load_config()


# ---------------------------------------------------------------- modal objects
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git", "build-essential")
    .pip_install(*PIP_PACKAGES)
    .env(
        {
            "HF_HOME": REMOTE_HF_CACHE,
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    # Mounted (copy=False) rather than baked, so editing the dataset or the
    # config does not rebuild the image. eval.py rides along so the trainer can
    # assert the training condition equals the zero_shot eval condition.
    .add_local_file(REPO_ROOT / "eval.py", f"{REMOTE_REPO}/eval.py")
    .add_local_dir(REPO_ROOT / "configs", f"{REMOTE_REPO}/configs")
    .add_local_dir(REPO_ROOT / "data", f"{REMOTE_REPO}/data")
)

artifacts_vol = modal.Volume.from_name(
    _CFG["output"]["modal_volume"], create_if_missing=True
)
hf_cache_vol = modal.Volume.from_name("slmforge-hf-cache", create_if_missing=True)


def _hf_secret() -> modal.Secret:
    """Least-privilege secret sourced from .env at launch, never baked into the
    image. Only the two Hub keys go to the GPU container -- the frontier/judge
    API keys in .env have no business there.

    The whole-file alternative is modal.Secret.from_dotenv(path=str(REPO_ROOT)),
    which ships every key in .env; we deliberately do not use it.
    """
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        try:
            from dotenv import load_dotenv

            load_dotenv(str(env_path))
        except ImportError:  # python-dotenv absent -> fall back to real env
            pass
    payload = {}
    for key in ("HF_TOKEN", "HF_USERNAME"):
        val = os.environ.get(key)
        if val:
            payload[key] = val
    return modal.Secret.from_dict(payload)


app = modal.App("slmforge-train")


# ---------------------------------------------------------------- config merge
def _is_section(cfg: dict, key: str) -> bool:
    return isinstance(cfg.get(key), dict)


def resolve_config(cfg: dict, smoke: bool) -> dict:
    """Apply smoke_overrides on top of cfg when smoke is True.

    A key that names a top-level section deep-merges into that section; every
    other key merges into 'training' (so {"max_steps": 20} lands where HF's
    TrainingArguments expects it).
    """
    resolved = json.loads(json.dumps(cfg))  # deep copy, JSON-only by construction
    resolved["_smoke"] = bool(smoke)
    if not smoke:
        return resolved
    for key, val in cfg.get("smoke_overrides", {}).items():
        if _is_section(resolved, key) and isinstance(val, dict):
            resolved[key].update(val)
        else:
            resolved["training"][key] = val
    return resolved


# ---------------------------------------------------------------- data plumbing
def build_training_features(tokenizer, rows, max_seq_length, enable_thinking,
                            completion_only_loss=True):
    """P3 rows -> {input_ids, attention_mask, labels}, using the model's own chat
    template.

    The prompt half is rendered with EXACTLY the call eval.py's HFGenerator makes
    (add_generation_prompt=True, enable_thinking=False), and the target is
    appended to that rendered string. Building it this way -- rather than
    templating the whole conversation in one shot -- guarantees the token prefix
    the model is trained to continue is byte-identical to the prefix it sees at
    eval time.

    Returns (features, stats).
    """
    def render_prompt(messages):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=enable_thinking)
        except TypeError:  # tokenizer without the enable_thinking kwarg
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

    eos = tokenizer.eos_token or ""
    features, stats = [], {"rows": len(rows), "truncated": 0,
                           "prefix_mismatch": 0, "skipped": 0}
    for row in rows:
        messages = row["messages"]
        if len(messages) < 2 or messages[-1].get("role") != "assistant":
            stats["skipped"] += 1
            continue
        prompt_text = render_prompt(messages[:-1])
        target_text = messages[-1]["content"] + eos

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(prompt_text + target_text,
                             add_special_tokens=False)["input_ids"]

        if full_ids[:len(prompt_ids)] != prompt_ids:
            # Retokenization at the join changed the prompt tokens. Fall back to
            # concatenating the two token sequences so the mask stays exact.
            stats["prefix_mismatch"] += 1
            target_ids = tokenizer(target_text,
                                   add_special_tokens=False)["input_ids"]
            full_ids = prompt_ids + target_ids

        if completion_only_loss:
            labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        else:
            labels = list(full_ids)

        if len(full_ids) > max_seq_length:
            stats["truncated"] += 1
            full_ids = full_ids[:max_seq_length]
            labels = labels[:max_seq_length]
        if all(t == -100 for t in labels):
            stats["skipped"] += 1  # nothing left to learn from after truncation
            continue

        features.append({"input_ids": full_ids,
                         "attention_mask": [1] * len(full_ids),
                         "labels": labels})
    stats["kept"] = len(features)
    return features, stats


class PadCollator:
    """Right-pads a batch; label padding is -100 so it never contributes loss."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        import torch

        width = max(len(b["input_ids"]) for b in batch)
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for b in batch:
            gap = width - len(b["input_ids"])
            out["input_ids"].append(b["input_ids"] + [self.pad_token_id] * gap)
            out["attention_mask"].append(b["attention_mask"] + [0] * gap)
            out["labels"].append(b["labels"] + [-100] * gap)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}


def _dir_listing(path):
    """(files, total_bytes) for one artifact directory -- the committed proof
    that the artifact exists, without shipping the weights back."""
    files, total = [], 0
    root = Path(path)
    if not root.exists():
        return files, total
    for p in sorted(root.rglob("*")):
        if p.is_file():
            size = p.stat().st_size
            files.append({"name": str(p.relative_to(root)), "bytes": size})
            total += size
    return files, total


# ---------------------------------------------------------------- the GPU job
@app.function(
    image=image,
    gpu=GPU,
    timeout=TIMEOUT_S,
    volumes={REMOTE_VOL: artifacts_vol, REMOTE_HF_CACHE: hf_cache_vol},
    secrets=[_hf_secret()],
)
def train(smoke: bool, dataset_path: str, run_name: str) -> dict:
    import hashlib
    import time
    from datetime import datetime, timezone

    t_zero = time.time()
    started_utc = datetime.now(timezone.utc).isoformat()

    # unsloth must be imported before transformers/trl for its patches to land.
    from unsloth import FastLanguageModel  # noqa: F401  (import order matters)
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from transformers import Trainer, TrainingArguments

    sys.path.insert(0, REMOTE_REPO)
    import eval as E  # the frozen harness -- imported, never reimplemented

    cfg = load_config(Path(REMOTE_REPO) / "configs" / "train.json")
    resolved = resolve_config(cfg, smoke)
    tcfg = resolved["training"]
    lcfg = resolved["lora"]
    qcfg = resolved["quantization"]
    dcfg = resolved["data"]
    ocfg = resolved["output"]

    # ---- dataset (leg 1 of the loop: generate) -------------------------------
    ds_file = Path(REMOTE_REPO) / dataset_path
    if not ds_file.exists():
        raise FileNotFoundError(
            f"training dataset {dataset_path!r} not found at {ds_file} -- P3 must "
            f"commit it before P4 can train on it")
    raw_bytes = ds_file.read_bytes()
    rows = [json.loads(ln) for ln in raw_bytes.decode().splitlines() if ln.strip()]
    if not rows:
        raise ValueError(f"training dataset {dataset_path!r} is empty")

    # Contract guard: training condition == zero_shot eval condition.
    bad_system = [i for i, r in enumerate(rows)
                  if r["messages"][0].get("content") != E.SYSTEM_CORE]
    if bad_system:
        raise ValueError(
            f"{len(bad_system)} training rows do not carry eval.SYSTEM_CORE "
            f"verbatim as the system message (first: row {bad_system[0]}). The "
            f"training condition must equal the zero_shot eval condition.")

    label_counts, type_counts = {}, {}
    for r in rows:
        meta = r.get("meta", {}) or {}
        gl = meta.get("gold_label", "unknown")
        ty = meta.get("type", "unknown")
        label_counts[gl] = label_counts.get(gl, 0) + 1
        type_counts[ty] = type_counts.get(ty, 0) + 1

    # ---- base model, pinned by revision --------------------------------------
    t_load = time.time()
    base_repo = resolved["base_model"]["repo"]
    base_rev = resolved["base_model"]["revision"]
    # Resolve the pin ourselves rather than trusting a revision= kwarg to survive
    # Unsloth's wrapper: snapshot_download hard-pins the SHA, and Unsloth then
    # loads a plain local directory.
    snapshot_path = snapshot_download(repo_id=base_repo, revision=base_rev)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=snapshot_path,
        max_seq_length=tcfg["max_seq_length"],
        dtype=qcfg["dtype"],
        load_in_4bit=qcfg["load_in_4bit"],
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lcfg["r"],
        lora_alpha=lcfg["lora_alpha"],
        lora_dropout=lcfg["lora_dropout"],
        target_modules=lcfg["target_modules"],
        bias=lcfg["bias"],
        use_rslora=lcfg["use_rslora"],
        use_gradient_checkpointing=lcfg["use_gradient_checkpointing"],
        random_state=lcfg["random_state"],
    )
    load_s = time.time() - t_load

    # ---- tokenize (leg 2 of the loop: train) ---------------------------------
    t_tok = time.time()
    features, ds_stats = build_training_features(
        tokenizer, rows,
        max_seq_length=tcfg["max_seq_length"],
        enable_thinking=dcfg["enable_thinking"],
        completion_only_loss=dcfg["completion_only_loss"],
    )
    if not features:
        raise ValueError("every training row was dropped during tokenization")
    tokenize_s = time.time() - t_tok

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    transformers.set_seed(tcfg["seed"])

    out_root = Path(REMOTE_VOL) / run_name
    args_kwargs = dict(
        output_dir=str(out_root / "hf_trainer"),
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        num_train_epochs=tcfg["num_train_epochs"],
        learning_rate=tcfg["learning_rate"],
        warmup_ratio=tcfg["warmup_ratio"],
        lr_scheduler_type=tcfg["lr_scheduler_type"],
        weight_decay=tcfg["weight_decay"],
        max_grad_norm=tcfg["max_grad_norm"],
        optim=tcfg["optim"],
        logging_steps=tcfg["logging_steps"],
        seed=tcfg["seed"],
        data_seed=tcfg["seed"],
        save_strategy="no",
        report_to="none",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
    )
    if "max_steps" in tcfg:  # only present under --smoke
        args_kwargs["max_steps"] = tcfg["max_steps"]

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**args_kwargs),
        train_dataset=features,
        data_collator=PadCollator(pad_id),
    )

    t_train = time.time()
    train_result = trainer.train()
    train_s = time.time() - t_train

    loss_curve = [
        {"step": h["step"], "loss": h["loss"],
         "learning_rate": h.get("learning_rate"), "epoch": h.get("epoch")}
        for h in trainer.state.log_history if "loss" in h
    ]

    # ---- artifacts: adapter, merged fp16, train_log --------------------------
    adapter_dir = out_root / ocfg["adapter_dirname"]
    merged_dir = out_root / ocfg["merged_dirname"]
    adapter_dir.mkdir(parents=True, exist_ok=True)

    t_adapter = time.time()
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    adapter_s = time.time() - t_adapter

    t_merge = time.time()
    merge_method = ocfg["merged_save_method"]
    if hasattr(model, "save_pretrained_merged"):
        model.save_pretrained_merged(str(merged_dir), tokenizer,
                                     save_method=merge_method)
        merge_path = "unsloth.save_pretrained_merged"
    else:
        # Honest fallback: reload the pinned base in fp16 on CPU and merge the
        # adapter into it. Slower, but produces a genuine fp16 merged model.
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        base = AutoModelForCausalLM.from_pretrained(
            snapshot_path, torch_dtype=torch.float16, device_map="cpu")
        merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
        merged.save_pretrained(str(merged_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(merged_dir))
        merge_path = "peft.merge_and_unload(fp16, cpu)"
    merge_s = time.time() - t_merge

    adapter_files, adapter_bytes = _dir_listing(adapter_dir)
    merged_files, merged_bytes = _dir_listing(merged_dir)
    if not adapter_files:
        raise RuntimeError(f"adapter directory {adapter_dir} is empty after save")
    if not merged_files:
        raise RuntimeError(f"merged directory {merged_dir} is empty after save")

    try:
        import unsloth as _unsloth
        unsloth_version = getattr(_unsloth, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        unsloth_version = "unknown"

    total_s = time.time() - t_zero
    train_log = {
        "run_name": run_name,
        "smoke": bool(smoke),
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "steps": int(trainer.state.global_step),
        "epochs_completed": float(trainer.state.epoch or 0.0),
        "final_train_loss": float(train_result.training_loss),
        "loss_curve": loss_curve,
        "wall_time_s": {
            "model_load": round(load_s, 2),
            "tokenize": round(tokenize_s, 2),
            "train": round(train_s, 2),
            "save_adapter": round(adapter_s, 2),
            "save_merged": round(merge_s, 2),
            "total": round(total_s, 2),
        },
        "dataset": {
            "path": dataset_path,
            "rows": len(rows),
            "rows_trained_on": ds_stats["kept"],
            "rows_truncated": ds_stats["truncated"],
            "rows_skipped": ds_stats["skipped"],
            "prefix_mismatch": ds_stats["prefix_mismatch"],
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "gold_label_counts": label_counts,
            "type_counts": type_counts,
        },
        "base_model": {
            "repo": base_repo,
            "revision": base_rev,
            "spec": resolved["base_model"]["spec"],
            "snapshot_path": snapshot_path,
        },
        "rubric_hash": E.rubric_hash(),
        "rubric_version": E.RUBRIC_VERSION,
        "system_prompt_equals_eval_SYSTEM_CORE": True,
        "resolved_config": resolved,
        "artifacts": {
            "volume": ocfg["modal_volume"],
            "run_dir_remote": str(out_root),
            "adapter": {
                "remote_path": str(adapter_dir),
                "volume_path": f"{run_name}/{ocfg['adapter_dirname']}",
                "files": adapter_files,
                "bytes": adapter_bytes,
            },
            "merged": {
                "remote_path": str(merged_dir),
                "volume_path": f"{run_name}/{ocfg['merged_dirname']}",
                "files": merged_files,
                "bytes": merged_bytes,
                "save_method": merge_method,
                "merge_path": merge_path,
            },
            "train_log": str(out_root / ocfg["train_log_filename"]),
        },
        "gpu": GPU,
        "env": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "unsloth": unsloth_version,
        },
    }

    (out_root / ocfg["train_log_filename"]).write_text(
        json.dumps(train_log, indent=2))
    artifacts_vol.commit()

    return {
        "run_name": run_name,
        "volume": ocfg["modal_volume"],
        "run_dir_remote": str(out_root),
        "pull_commands": [
            f"modal volume get {ocfg['modal_volume']} "
            f"{run_name}/{ocfg['merged_dirname']} <local-dir>",
            f"modal volume get {ocfg['modal_volume']} "
            f"{run_name}/{ocfg['adapter_dirname']} <local-dir>",
        ],
        "train_log": train_log,
    }


# ---------------------------------------------------------------- local driver
@app.local_entrypoint()
def main(smoke: bool = False, dataset: str = "", out_dir: str = "",
         run_name: str = ""):
    """Drive one training run and land its evidence under out_dir.

    smoke     apply configs/train.json:smoke_overrides (max_steps 20)
    dataset   repo-relative training JSONL (default: from configs/train.json)
    out_dir   repo-relative directory for train_log.json + run_manifest.json
    run_name  name of the run directory inside the Modal Volume
    """
    cfg = load_config()
    dcfg, ocfg = cfg["data"], cfg["output"]

    dataset = dataset or (dcfg["smoke_train_jsonl"] if smoke else dcfg["train_jsonl"])
    out_dir = out_dir or (ocfg["smoke_run_dir"] if smoke else ocfg["run_dir"])
    run_name = run_name or ("smoke" if smoke else "v1")

    local_ds = REPO_ROOT / dataset
    if not local_ds.exists():
        raise SystemExit(
            f"training dataset not found: {local_ds}\n"
            f"P3 must generate and commit it before P4 can train on it.")

    print(f"[slmforge] gpu={GPU} timeout={TIMEOUT_S}s smoke={smoke}")
    print(f"[slmforge] dataset={dataset} run_name={run_name} out_dir={out_dir}")

    manifest = train.remote(smoke=smoke, dataset_path=dataset, run_name=run_name)

    train_log = manifest["train_log"]
    out_path = REPO_ROOT / out_dir
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / ocfg["train_log_filename"]).write_text(
        json.dumps(train_log, indent=2))
    (out_path / ocfg["manifest_filename"]).write_text(json.dumps(
        {k: v for k, v in manifest.items() if k != "train_log"}, indent=2))

    art = train_log["artifacts"]
    print(f"[slmforge] steps={train_log['steps']} "
          f"final_loss={train_log['final_train_loss']:.4f} "
          f"train_s={train_log['wall_time_s']['train']}")
    print(f"[slmforge] wrote {out_dir}/{ocfg['train_log_filename']} and "
          f"{out_dir}/{ocfg['manifest_filename']}")
    print("\n[slmforge] pull the weights, then close the loop:")
    for name in (ocfg["merged_dirname"], ocfg["adapter_dirname"]):
        print(f"  modal volume get {art['volume']} {run_name}/{name} "
              f"{out_dir}/{name}")
    print(f"  python eval.py --model hf:{out_dir}/{ocfg['merged_dirname']} "
          f"--eval-set data/eval/scenarios_v1.jsonl --out {out_dir}/eval")
    print(f"  python scripts/check_loop.py {out_dir}")
