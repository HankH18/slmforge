#!/usr/bin/env python3
"""DPO (Direct Preference Optimization) on top of the SFT checkpoint, on Apple Silicon.

Stretch Ladder rung 1 of docs/project_spec.pdf p.4: "build preference pairs
(on-spec vs. off-spec) and run DPO on top of your SFT model; measure the delta
over SFT alone."

WHAT THIS IS NOT
================
This is NOT a reproduction of the SFT recipe on a second device. Three things
in configs/train.json are CUDA-only and are deliberately absent here. Saying so
in the docstring rather than silently substituting is the point:

  * quantization.load_in_4bit -- bitsandbytes ships CUDA kernels. There is no
    MPS build. The policy is loaded in bfloat16 (or float32), not NF4.
  * training.optim = "adamw_8bit" -- also bitsandbytes. This trainer uses a
    plain torch optimizer ("adamw_torch" by default).
  * training.warmup_ratio -- removed from transformers.TrainingArguments in
    transformers 5.x, which is what .venv-dpo resolves. Warmup is expressed as
    warmup_steps instead. It is a DIFFERENT knob, not a rename.

configs/train.json is pinned by contract and describes the SFT run. This file
never reads a hyperparameter out of it. It does *cross-check* its LoRA block
against the SFT recipe and record the delta in the train log, so a divergence is
visible evidence rather than a surprise.

WHY A LOCAL DPO RESULT IS STILL ADMISSIBLE
==========================================
The comparison the spec asks for is "the delta over SFT alone". Both sides of
that comparison are the SAME CUDA-trained SFT checkpoint (runs/train_v2/merged);
only the DPO step differs. A data-efficiency curve point trained on a different
stack would be confounded -- this is not, because the stack is held constant
across the contrast. See docs/DPO_TRAINING.md.

THE REFERENCE MODEL
===================
No second copy of the weights is loaded. With a PEFT LoRA policy and
ref_model=None, TRL recovers the reference by disabling the adapter. Verbatim
from the installed trl 1.10.0, trl/trainer/dpo_trainer.py:918-924:

    # Reference model
    if ref_model is None:
        if is_peft_model(self.model) or args.precompute_ref_log_probs:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled to revert to the
            # initial model. If precompute_ref_log_probs is True, the reference model does not need to be kept in
            # memory during training.
            self.ref_model = None

and the loss path at dpo_trainer.py:1217-1222 does exactly that:

    if self.ref_model is None and is_peft_model(self.model):
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        with use_adapter(
            unwrapped_model, adapter_name="ref" if "ref" in unwrapped_model.peft_config else None
        ):
            ref_outputs = model(**model_kwargs)

`use_adapter(model, adapter_name=None)` is TRL's disable-adapter context. So the
reference policy IS the SFT checkpoint, exactly, at zero extra memory.

TWO INPUT SHAPES, ONE INTERNAL SHAPE
====================================
--pairs accepts either of the two preference files this repo produces:

  * data/train/*.dpo_negatives.jsonl -- prompt/chosen/rejected are STRINGS and
    the row carries its source ``scenario`` object inline. ``chosen`` is null on
    the prompts where the teacher never produced a compliant draft; those rows
    are dropped and counted.
  * data/train/*.dpo_pairs.jsonl -- TRL *conversational* form: prompt/chosen/
    rejected are one-message lists, and the row carries only ``scenario_id``.
    The dg-* scenario ids in it are NOT in data/scenarios/corpus.jsonl, so the
    scenario has to come from somewhere: the sibling
    ``<stem>.manifest.json``'s ``source.path`` is read for it, and that file's
    sha256 is checked against the manifest's ``source.sha256`` first. Override
    with --scenario-source, or disable the whole lookup with
    --no-scenario-source (which also disables the prompt-contract gate, loudly).

Both are normalised to three plain strings before anything else happens.

THE PROMPT CONTRACT
===================
The dataset is handed to TRL in its *standard* (plain-string) format, not its
conversational format, because that is the only way to guarantee the prompt
prefix is byte-identical to the one eval.py feeds the model. The prompt string
is built with the same call training/modal_app.py:render_prompt_text and
eval.py:HFGenerator both make:

    tokenizer.apply_chat_template(msgs, tokenize=False,
                                  add_generation_prompt=True,
                                  enable_thinking=False)

TRL's standard path tokenizes that string with the tokenizer's __call__
(trl/data_utils.py:433, `processing_class(text=input)`), which defaults to
add_special_tokens=True; preflight asserts that this is a no-op for the
checkpoint's tokenizer and refuses to train if it is not. Every row's user
message is additionally re-rendered through the shared renderer
(scenarios.render.render_user_message) and must come back byte-identical, the
same gate training/modal_app.py:verify_prompt_contract applies.

USAGE
=====
    # preflight only -- loads no weights, costs nothing
    .venv-dpo/bin/python training/dpo_local.py --preflight-only

    # cheap end-to-end check against a tiny CPU model
    .venv-dpo/bin/python training/dpo_local.py --smoke --device cpu \\
        --sft-checkpoint /tmp/tiny --pairs /tmp/pairs.jsonl --out-dir runs/dpo_smoke

    # the real run
    .venv-dpo/bin/python training/dpo_local.py \\
        --pairs data/train/v3.dpo_pairs.jsonl --run-name dpo_v1

    # merge an already-trained adapter into standalone weights eval.py can load
    .venv-dpo/bin/python training/dpo_local.py --merge-only \\
        --adapter runs/dpo_v1/adapter --out-dir runs/dpo_v1

Run it with .venv-dpo, never .venv: .venv is the graded environment and must not
grow a trl/peft dependency. See training/requirements-dpo.txt.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import inspect
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# --------------------------------------------------------------------- config
#
# Every hyperparameter lives HERE or on the command line. Nothing is read out of
# configs/train.json, which is pinned and describes the SFT run.
#
# The training numbers are UNSWEPT defaults inside TRL's documented DPO range.
# No sweep was run on this project's data, and this file does not pretend one
# was: --config <path>.json deep-merges over these so a tuned recipe can be
# pinned later without editing code.
DEFAULTS = {
    "sft_checkpoint": "runs/train_v2/merged",
    "data": {
        "pairs": "data/train/v3.dpo_pairs.jsonl",
        "constraints_path": "data/scenarios/constraint_instances.jsonl",
        "scenarios_path": "data/scenarios/corpus.jsonl",
        "prompt_renderer": "scenarios.render.render_user_message",
        "enable_thinking": False,
        "validate_pairs": True,
    },
    "lora": {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
        "bias": "none",
        "use_rslora": False,
        "task_type": "CAUSAL_LM",
    },
    "training": {
        "beta": 0.1,
        "loss_type": "sigmoid",
        "learning_rate": 5e-6,
        "num_train_epochs": 1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "max_length": 2048,
        "seed": 42,
        "warmup_steps": 10,
        "lr_scheduler_type": "linear",
        "optim": "adamw_torch",
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "logging_steps": 1,
        "gradient_checkpointing": False,
    },
    "smoke_overrides": {
        "max_steps": 2,
        "warmup_steps": 0,
        "gradient_accumulation_steps": 1,
    },
    "output": {
        "run_dir_root": "runs",
        "adapter_dirname": "adapter",
        "merged_dirname": "merged",
        "train_log_filename": "train_log.json",
        "manifest_filename": "run_manifest.json",
        "merge_load_dtype": "float32",
        "merge_save_dtype": "bfloat16",
    },
}

# The SFT recipe's LoRA block is not read as configuration -- it is read as a
# CROSS-CHECK, so a divergence between the two recipes shows up as recorded
# evidence in the train log instead of as an unexamined difference.
SFT_CONFIG_PATH = REPO_ROOT / "configs" / "train.json"

# Same restriction training/modal_app.py:resolve_render_fn enforces: a training
# run may not nominate its own prompt renderer.
ALLOWED_RENDERER_ROOTS = ("scenarios", "harness", "eval")

TRAIN_LOG_KEYS = ("base_model", "config_resolved", "dataset", "prompt_contract",
                  "steps", "loss_history", "wall_clock_seconds", "artifacts",
                  "started_utc", "finished_utc")


class PreflightError(RuntimeError):
    """Something that must be true BEFORE any weights load is not true."""


# ------------------------------------------------------------------ utilities
def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_digest(path: Path, mode: str = "content") -> str:
    """A content digest for a local checkpoint directory.

    ``content`` hashes every file's bytes (ordered by relative path, with the
    path and size folded in, so a rename or a truncation changes the digest).
    ``listing`` hashes only the (path, size) table -- cheap, and honestly
    weaker. The mode is recorded next to the digest in the train log so nobody
    can mistake the cheap one for the strong one.
    """
    h = hashlib.sha256()
    for p in sorted(path.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(path))
        h.update(rel.encode())
        h.update(str(p.stat().st_size).encode())
        if mode == "content":
            h.update(sha256_file(p).encode())
    return h.hexdigest()


def dir_listing(path: Path):
    """(files, total_bytes) -- same shape as training/modal_app.py:_dir_listing."""
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


def resolve_render_fn(dotted: str):
    """Import the shared prompt renderer named by its dotted path.

    Deliberately the same rule as training/modal_app.py:resolve_render_fn --
    including the module-root allowlist. There is no fallback: a stubbed
    renderer would make the DPO prompt differ from the eval prompt while every
    check still passed.
    """
    if not isinstance(dotted, str) or not dotted.strip():
        raise PreflightError(f"prompt_renderer must be a non-empty dotted path, got {dotted!r}")
    path = dotted.strip().replace(":", ".")
    if "." not in path:
        raise PreflightError(f"prompt_renderer {dotted!r} names no module")
    module_name, _, attr = path.rpartition(".")
    if module_name.split(".")[0] not in ALLOWED_RENDERER_ROOTS:
        raise PreflightError(
            f"prompt_renderer {dotted!r} does not live in the shared renderer "
            f"modules {list(ALLOWED_RENDERER_ROOTS)} -- a training run may not "
            f"nominate its own prompt renderer")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise PreflightError(
            f"the shared prompt renderer {dotted!r} is not importable "
            f"({module_name}: {exc})") from exc
    fn = getattr(module, attr, None)
    if fn is None:
        raise PreflightError(f"module {module_name!r} has no attribute {attr!r}")
    if not callable(fn):
        raise PreflightError(f"the shared prompt renderer {dotted!r} is not callable")
    return fn


def render_arity(fn) -> int:
    """2 -> called (scenario, constraint_lookup); 1 -> (scenario).

    Same rule as training/modal_app.py:render_arity and scripts/check_loop.py.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return 2
    positional, var_positional = 0, False
    for param in sig.parameters.values():
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            var_positional = True
        elif param.kind in (inspect.Parameter.POSITIONAL_ONLY,
                            inspect.Parameter.POSITIONAL_OR_KEYWORD):
            positional += 1
    return 2 if (var_positional or positional >= 2) else 1


def load_side_corpus(path: Path, id_key: str) -> dict:
    if not path.exists():
        raise PreflightError(f"declared corpus {path} does not exist")
    corpus = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        key = obj.get(id_key)
        if not isinstance(key, str) or not key:
            raise PreflightError(f"{path}: a row has no {id_key!r}")
        corpus[key] = obj
    if not corpus:
        raise PreflightError(f"declared corpus {path} is empty")
    return corpus


# ------------------------------------------------------------------- the data
def _one_message(value, role: str, where: str) -> str:
    """A TRL conversational field -> its single message's content.

    Multi-turn is rejected rather than flattened: this project's eval condition
    is exactly one user turn, and a silently concatenated second turn would tune
    the model under a prompt shape it is never measured with.
    """
    if not isinstance(value, list) or len(value) != 1:
        raise PreflightError(
            f"{where}: expected exactly one {role!r} message, got "
            f"{len(value) if isinstance(value, list) else type(value).__name__}")
    msg = value[0]
    if not isinstance(msg, dict) or msg.get("role") != role:
        raise PreflightError(f"{where}: message role is {msg.get('role') if isinstance(msg, dict) else '?'!r}, "
                             f"expected {role!r}")
    content = msg.get("content")
    if not isinstance(content, str) or not content.strip():
        raise PreflightError(f"{where}: {role!r} message has no text content")
    return content


def normalize_row(row: dict, where: str) -> dict:
    """Either input shape -> {"prompt", "chosen", "rejected"} as plain strings.

    Returns None for a row whose ``chosen`` is null -- the caller counts those.
    """
    if row.get("chosen") is None:
        return None
    out = {}
    for field, role in (("prompt", "user"), ("chosen", "assistant"), ("rejected", "assistant")):
        value = row.get(field)
        if isinstance(value, str):
            if not value.strip():
                raise PreflightError(f"{where}: {field!r} is an empty string")
            out[field] = value
        elif isinstance(value, list):
            out[field] = _one_message(value, role, f"{where}: {field!r}")
        else:
            raise PreflightError(
                f"{where}: {field!r} is a {type(value).__name__}; expected a string "
                f"(standard form) or a one-message list (conversational form)")
    return out


def build_scenario_index(paths, scenarios_corpus: dict) -> dict:
    """{scenario_id: scenario} from the declared corpus plus any side files.

    A side file contributes a scenario only if its row carries BOTH a
    ``scenario_id`` and a ``scenario`` object. Two different scenario objects
    under one id is a hard error: it would make the prompt a coin flip.
    """
    index = dict(scenarios_corpus or {})
    for path in paths:
        for i, line in enumerate(Path(path).read_text().splitlines()):
            if not line.strip():
                continue
            obj = json.loads(line)
            sid, scenario = obj.get("scenario_id"), obj.get("scenario")
            if not isinstance(sid, str) or not isinstance(scenario, dict):
                continue
            prior = index.get(sid)
            if prior is not None and prior != scenario:
                raise PreflightError(
                    f"{path} row {i}: scenario {sid!r} disagrees with the scenario "
                    f"already indexed for that id. The prompt would depend on which "
                    f"copy was read.")
            index[sid] = scenario
    return index


def resolve_scenario_source(pairs_path: Path, explicit, disabled: bool):
    """Where the scenario objects for a scenario_id-only pairs file come from.

    Order: --scenario-source, then the sibling ``<stem>.manifest.json``'s
    ``source.path``. When the manifest also declares ``source.sha256``, the file
    is hashed and a mismatch is fatal -- a pairs file re-derived from a
    different source than the one it names is not the file it claims to be.
    Returns (paths, provenance).
    """
    if disabled:
        return [], {"mode": "disabled", "_warning": "prompt-contract gate is OFF"}
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = REPO_ROOT / p
        if not p.exists():
            raise PreflightError(f"--scenario-source {p} does not exist")
        return [p], {"mode": "explicit", "paths": [str(p)]}

    manifest_path = pairs_path.with_name(pairs_path.name.replace(".jsonl", ".manifest.json"))
    if not manifest_path.exists():
        return [], {"mode": "none", "_note": "no sibling manifest; rows must carry "
                                             "their own 'scenario' object"}
    manifest = json.loads(manifest_path.read_text())
    source = manifest.get("source") or {}
    rel = source.get("path")
    if not rel:
        return [], {"mode": "manifest_without_source", "manifest": str(manifest_path)}
    src = REPO_ROOT / rel
    if not src.exists():
        raise PreflightError(
            f"{manifest_path} declares source {rel!r}, which does not exist at {src}. "
            f"The scenario objects the pairs file's prompts were rendered from cannot "
            f"be recovered, so the prompt contract cannot be re-derived.")
    prov = {"mode": "sibling_manifest", "manifest": str(manifest_path),
            "paths": [str(src)], "declared_sha256": source.get("sha256")}
    if source.get("sha256"):
        actual = sha256_file(src)
        prov["actual_sha256"] = actual
        if actual != source["sha256"]:
            raise PreflightError(
                f"{manifest_path} declares source sha256 {source['sha256']} but {src} "
                f"hashes to {actual}. The pairs file was built from a different source "
                f"than the one it names.")
    return [src], prov


def load_pairs(pairs_path: Path, constraint_corpus: dict, render_fn, arity: int,
               validate: bool, max_rows=None, scenario_index=None,
               contract_gate: bool = True):
    """The preference file -> (records, stats).

    A record is TRL's *standard* preference format: three plain strings, with
    ``prompt`` left as the raw user message here; templating happens in
    build_dataset once the tokenizer is loaded.

    Rows whose ``chosen`` is null are the prompts where the teacher never
    produced a compliant draft. They are DROPPED and COUNTED, never silently
    passed through -- a null chosen is not a preference.

    Everything else is a hard error, because a mis-oriented pair teaches the
    model the opposite of the behavior spec:
      * prompt not byte-identical to the shared renderer's output
      * chosen == rejected
      * (validate) chosen fails any constraint, or rejected passes all of them,
        under the SHIPPED checkers -- constraints.checkers.check
    """
    from constraints.checkers import check

    rows = []
    for line in pairs_path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise PreflightError(f"{pairs_path} has no rows")

    stats = {"rows_in_file": len(rows), "dropped_null_chosen": 0,
             "kept": 0, "validated": bool(validate),
             "prompt_contract_checked": 0, "prompt_contract_unverifiable": 0,
             "violated_types": {}, "kept_scenario_ids": []}
    records = []
    for i, row in enumerate(rows):
        where = f"{pairs_path} row {i}"
        norm = normalize_row(row, where)
        if norm is None:
            stats["dropped_null_chosen"] += 1
            continue
        if norm["chosen"] == norm["rejected"]:
            raise PreflightError(f"{where}: chosen == rejected; not a preference")

        scenario = row.get("scenario")
        if not isinstance(scenario, dict):
            sid = row.get("scenario_id")
            scenario = (scenario_index or {}).get(sid) if isinstance(sid, str) else None
        if scenario is None and contract_gate:
            raise PreflightError(
                f"{where}: carries no 'scenario' object and its scenario_id "
                f"{row.get('scenario_id')!r} resolves in neither the declared scenario "
                f"corpus nor any --scenario-source, so its prompt can only be "
                f"self-attested, never re-derived. Point --scenario-source at a file "
                f"that carries the scenario objects, or pass --no-scenario-source to "
                f"train without the prompt-contract gate.")
        cids = row.get("constraints")
        if not isinstance(cids, list) or not cids:
            raise PreflightError(f"{where}: 'constraints' is missing or empty")
        lookup = {}
        for cid in cids:
            if isinstance(cid, dict):
                if not isinstance(cid.get("id"), str):
                    raise PreflightError(f"{where}: a constraint object has no 'id'")
                lookup[cid["id"]] = cid
            elif isinstance(cid, str):
                if cid not in constraint_corpus:
                    raise PreflightError(
                        f"{where}: constraint id {cid!r} is not in the declared "
                        f"constraint corpus")
                lookup[cid] = constraint_corpus[cid]
            else:
                raise PreflightError(f"{where}: constraint entry is a "
                                     f"{type(cid).__name__}, not an object or an id")

        if scenario is not None:
            rendered = render_fn(scenario, lookup) if arity == 2 else render_fn(scenario)
            if rendered != norm["prompt"]:
                raise PreflightError(
                    f"{where}: the stored prompt is NOT what the shared renderer "
                    f"produces for this scenario + constraint set. Training on it would "
                    f"tune the model under a prompt it is never evaluated with."
                    f"\n  stored:   {norm['prompt'][:300]!r}"
                    f"\n  rendered: {rendered[:300]!r}")
            stats["prompt_contract_checked"] += 1
        else:
            stats["prompt_contract_unverifiable"] += 1

        if validate:
            failures = []
            for cid, constraint in lookup.items():
                verdict = check(constraint, norm["chosen"])
                if not verdict["pass"]:
                    failures.append((cid, verdict["evidence"]))
            if failures:
                raise PreflightError(
                    f"{where}: 'chosen' FAILS {len(failures)} constraint(s) under the "
                    f"shipped checkers -- {failures[:2]}. This pair is backwards or "
                    f"mislabelled; DPO on it teaches the violation.")
            rejected_failures = [cid for cid, constraint in lookup.items()
                                 if not check(constraint, norm["rejected"])["pass"]]
            if not rejected_failures:
                raise PreflightError(
                    f"{where}: 'rejected' PASSES every constraint, so the pair carries "
                    f"no preference signal about the behavior spec.")
            for cid in rejected_failures:
                ctype = str(lookup[cid].get("type", "?"))
                stats["violated_types"][ctype] = stats["violated_types"].get(ctype, 0) + 1

        records.append(norm)
        stats["kept_scenario_ids"].append(row.get("scenario_id"))
        if max_rows is not None and len(records) >= max_rows:
            break

    stats["kept"] = len(records)
    if not records:
        raise PreflightError(
            f"{pairs_path}: every row was dropped ({stats['dropped_null_chosen']} had a "
            f"null 'chosen'). There is nothing to train on.")
    return records, stats


def build_dataset(records, tokenizer, enable_thinking: bool):
    """Records -> a datasets.Dataset in TRL's standard preference format, with the
    prompt already templated by the model's own chat template.

    This is the single place the prompt prefix is constructed, and it makes the
    same call eval.py:HFGenerator makes.
    """
    from datasets import Dataset

    out = []
    for rec in records:
        msgs = [{"role": "user", "content": rec["prompt"]}]
        try:
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=enable_thinking)
        except TypeError:  # tokenizer without the enable_thinking kwarg
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
        out.append({"prompt": text, "chosen": rec["chosen"], "rejected": rec["rejected"]})
    return Dataset.from_list(out), out


# -------------------------------------------------------------------- devices
def resolve_device(requested: str) -> str:
    import torch

    if requested == "cpu":
        return "cpu"
    mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    if requested == "mps":
        if not mps:
            raise PreflightError(
                "--device mps was requested but torch.backends.mps.is_available() is "
                "False. Nothing has been loaded; fix the environment or pass --device cpu.")
        return "mps"
    if requested == "auto":
        if mps:
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    raise PreflightError(f"unknown --device {requested!r}")


def resolve_dtype(name: str, device: str):
    import torch

    if name == "auto":
        name = "float32" if device == "cpu" else "bfloat16"
    try:
        dtype = getattr(torch, name)
    except AttributeError as exc:
        raise PreflightError(f"unknown dtype {name!r}") from exc
    return dtype, name


# ------------------------------------------------------------------ preflight
def preflight(cfg: dict, args) -> dict:
    """Everything that can be checked BEFORE a byte of the checkpoint is read.

    This repo has been bitten by validation that ran after the money was spent,
    so the order here is deliberate: paths, renderer, corpora, pair orientation,
    tokenizer, device -- and only then, back in train(), the weights.
    """
    ck = Path(args.sft_checkpoint or cfg["sft_checkpoint"])
    if not ck.is_absolute():
        ck = REPO_ROOT / ck
    if not ck.is_dir():
        raise PreflightError(f"SFT checkpoint {ck} is not a directory. Nothing was loaded.")
    if not (ck / "config.json").exists():
        raise PreflightError(f"SFT checkpoint {ck} has no config.json")
    weights = list(ck.glob("*.safetensors")) + list(ck.glob("*.bin"))
    if not weights:
        raise PreflightError(f"SFT checkpoint {ck} holds no *.safetensors or *.bin weights")

    pairs = Path(args.pairs or cfg["data"]["pairs"])
    if not pairs.is_absolute():
        pairs = REPO_ROOT / pairs
    if not pairs.exists():
        raise PreflightError(
            f"preference pair file {pairs} does not exist. Refusing to train on a "
            f"silently smaller or absent set.")

    render_fn = resolve_render_fn(cfg["data"]["prompt_renderer"])
    arity = render_arity(render_fn)

    constraints_path = REPO_ROOT / cfg["data"]["constraints_path"]
    constraint_corpus = load_side_corpus(constraints_path, "id")

    scenarios_path = REPO_ROOT / cfg["data"]["scenarios_path"]
    scenarios_corpus = load_side_corpus(scenarios_path, "scenario_id")
    source_paths, source_provenance = resolve_scenario_source(
        pairs, args.scenario_source, args.no_scenario_source)
    scenario_index = build_scenario_index(source_paths, scenarios_corpus)

    device = resolve_device(args.device)

    records, pair_stats = load_pairs(
        pairs, constraint_corpus, render_fn, arity,
        validate=bool(cfg["data"]["validate_pairs"]) and not args.no_validate_pairs,
        max_rows=args.max_rows, scenario_index=scenario_index,
        contract_gate=not args.no_scenario_source)
    pair_stats["scenario_source"] = source_provenance

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(ck))
    if getattr(tokenizer, "chat_template", None) in (None, ""):
        raise PreflightError(f"{ck}: the tokenizer has no chat template, so the eval "
                             f"prompt cannot be reproduced")

    # TRL's standard-format path tokenizes the prompt with the tokenizer's
    # __call__ (trl/data_utils.py:433), which defaults to add_special_tokens=True.
    # eval.py:HFGenerator does the same. If that is NOT a no-op for this
    # tokenizer, the DPO prompt would carry a leading special token the eval
    # prompt also carries -- but the SFT features were built with
    # add_special_tokens=False, so the three would not agree. Fail here.
    probe = tokenizer.apply_chat_template(
        [{"role": "user", "content": "probe"}], tokenize=False,
        add_generation_prompt=True, enable_thinking=cfg["data"]["enable_thinking"])
    if tokenizer(probe)["input_ids"] != tokenizer(probe, add_special_tokens=False)["input_ids"]:
        raise PreflightError(
            "this tokenizer injects special tokens on __call__, so TRL's prompt "
            "tokenization would not match training/modal_app.py's "
            "add_special_tokens=False features. Refusing to train.")

    return {"checkpoint": ck, "pairs": pairs, "render_fn": render_fn,
            "render_arity": arity, "constraint_corpus": constraint_corpus,
            "records": records, "pair_stats": pair_stats,
            "tokenizer": tokenizer, "device": device}


# ------------------------------------------------------------------ train log
def sft_lora_delta(lora_cfg: dict) -> dict:
    """How this run's LoRA block differs from the pinned SFT recipe.

    Recorded, never enforced and never read as configuration. configs/train.json
    is pinned to describe the SFT run; this is a cross-check so a divergence is
    evidence in the log instead of an unexamined difference.
    """
    try:
        sft = json.loads(SFT_CONFIG_PATH.read_text())["lora"]
    except Exception as exc:  # noqa: BLE001
        return {"_unavailable": f"{type(exc).__name__}: {exc}"}
    delta = {}
    for key in ("r", "lora_alpha", "lora_dropout", "target_modules", "bias", "use_rslora"):
        mine, theirs = lora_cfg.get(key), sft.get(key)
        if isinstance(mine, list) and isinstance(theirs, list):
            same = sorted(mine) == sorted(theirs)
        else:
            same = mine == theirs
        if not same:
            delta[key] = {"dpo": mine, "sft_recipe": theirs}
    return delta


def build_train_log(*, checkpoint: Path, checkpoint_digest: str, digest_mode: str,
                    config_resolved: dict, dataset_path: Path, dataset_rows: int,
                    dataset_sha256: str, render_fn_dotted: str, sample_row_index: int,
                    sample_rendered_prompt: str, sample_prompt_messages, steps: int,
                    loss_history, wall_clock_seconds: float, adapter_path,
                    adapter_bytes: int, merged_path, merged_bytes: int,
                    started_utc: str, finished_utc: str, extras: dict) -> dict:
    """The SFT train_log.json shape, mirrored where it applies.

    Same ten keys as training/modal_app.py:TRAIN_LOG_KEYS, so the DPO run's
    evidence is auditable the same way the SFT runs' is. Two honest departures,
    both visible rather than papered over:

      * base_model.revision is NOT a 40-hex Hub commit -- the base of a DPO run
        is a LOCAL directory. It carries a sha256 content digest of that
        directory, and base_model.revision_kind says which kind of digest it is.
      * an extra "dpo" key holds what has no SFT counterpart (beta, loss type,
        the reference-model strategy, pair statistics, the environment).

    Refuses to emit a log with empty evidence, for the same reason the SFT
    builder does: a checker must never trust a flag written by the thing it is
    checking, so prompt_contract carries the rendered text itself.
    """
    if not isinstance(sample_rendered_prompt, str) or not sample_rendered_prompt:
        raise ValueError("prompt_contract.sample_rendered_prompt is empty. It is "
                         "EVIDENCE -- the literal prompt text fed to the model.")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
        raise ValueError(f"steps must be a positive int, got {steps!r}")
    if not isinstance(dataset_rows, int) or dataset_rows <= 0:
        raise ValueError(f"dataset.rows must be a positive int, got {dataset_rows!r}")
    if len(dataset_sha256) != 64:
        raise ValueError(f"dataset.sha256 must be 64 hex chars, got {dataset_sha256!r}")

    log = {
        "base_model": {
            "repo": str(checkpoint),
            "revision": checkpoint_digest,
            "revision_kind": f"sha256_of_local_checkpoint_{digest_mode}",
            "_note": ("The base of this DPO run is the SFT checkpoint on disk, not a "
                      "Hub commit. configs/models.json:tuned names the published "
                      "checkpoint the orchestrator states these weights match; this "
                      "log does NOT verify that claim, it records the local digest."),
        },
        "config_resolved": config_resolved,
        "dataset": {"path": str(dataset_path), "rows": int(dataset_rows),
                    "sha256": dataset_sha256},
        "prompt_contract": {"render_fn": render_fn_dotted,
                            "sample_row_index": int(sample_row_index),
                            "sample_rendered_prompt": sample_rendered_prompt,
                            "sample_prompt_messages": sample_prompt_messages},
        "steps": int(steps),
        "loss_history": [[int(s), float(v)] for s, v in loss_history],
        "wall_clock_seconds": float(wall_clock_seconds),
        "artifacts": {"adapter": str(adapter_path), "adapter_bytes": int(adapter_bytes),
                      "merged_fp16": str(merged_path) if merged_path else None,
                      "merged_bytes": int(merged_bytes)},
        "started_utc": started_utc,
        "finished_utc": finished_utc,
    }
    missing = [k for k in TRAIN_LOG_KEYS if k not in log]
    if missing:
        raise ValueError(f"train_log is missing pinned keys: {missing}")
    log["dpo"] = extras
    return log


def environment() -> dict:
    import accelerate
    import peft
    import torch
    import transformers
    import trl

    return {"python": platform.python_version(),
            "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "trl": trl.__version__,
            "peft": peft.__version__,
            "accelerate": accelerate.__version__,
            "quantization": None,
            "_quantization_note": ("no 4-bit: bitsandbytes has no MPS build. The SFT "
                                   "recipe's load_in_4bit and adamw_8bit are CUDA-only "
                                   "and are NOT reproduced here.")}


# --------------------------------------------------------------------- merge
def merge_adapter(checkpoint: Path, adapter_dir: Path, out_dir: Path,
                  load_dtype: str, save_dtype: str) -> dict:
    """LoRA adapter + SFT checkpoint -> a standalone weights directory.

    eval.py --model <dir> loads merged weights with
    AutoModelForCausalLM.from_pretrained, so an adapter alone is not evaluable.

    Mirrors the SFT run's recorded merge path -- run_manifest.json says
    "peft.merge_and_unload(fp32 cpu -> merged_16bit)" -- by loading float32 on
    CPU, merging, then casting to bfloat16 before saving. Always CPU: merging is
    a one-shot weight arithmetic pass, and doing it on the accelerator would
    contend for memory for no speed that matters.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    ldtype = getattr(torch, load_dtype)
    sdtype = getattr(torch, save_dtype)
    base = AutoModelForCausalLM.from_pretrained(
        str(checkpoint), dtype=ldtype, low_cpu_mem_usage=True)
    base = base.to("cpu")
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    merged = model.merge_and_unload()
    merged = merged.to(sdtype)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out_dir), safe_serialization=True)
    try:
        tok = AutoTokenizer.from_pretrained(str(adapter_dir))
    except Exception:  # noqa: BLE001 -- adapter dirs need not carry a tokenizer
        tok = AutoTokenizer.from_pretrained(str(checkpoint))
    tok.save_pretrained(str(out_dir))
    files, total = dir_listing(out_dir)
    return {"out_dir": str(out_dir), "bytes": total, "files": files,
            "seconds": time.time() - t0,
            "path": f"peft.merge_and_unload({load_dtype} cpu -> {save_dtype})"}


# --------------------------------------------------------------------- train
def train(cfg: dict, args, pre: dict) -> int:
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, set_seed
    from trl import DPOConfig, DPOTrainer

    # MEASURED, not assumed: without this, two identical runs produce DIFFERENT
    # adapters. DPOConfig(seed=...) seeds inside Trainer.__init__, but
    # DPOTrainer calls peft.get_peft_model BEFORE that point, so lora_A is drawn
    # from an unseeded RNG. Two runs of the CPU smoke differed by max |A - A'| =
    # 0.249 and by their step-2 loss. Seeding here, before any weight is created,
    # is what makes the run reproducible.
    set_seed(int(cfg["training"]["seed"]))

    started = _utc()
    t_start = time.time()

    checkpoint = pre["checkpoint"]
    tokenizer = pre["tokenizer"]
    device = pre["device"]
    dtype, dtype_name = resolve_dtype(args.dtype, device)

    run_name = args.run_name or ("dpo_smoke" if args.smoke else "dpo_v1")
    out_root = Path(args.out_dir) if args.out_dir else (REPO_ROOT / cfg["output"]["run_dir_root"] / run_name)
    if not out_root.is_absolute():
        out_root = REPO_ROOT / out_root
    adapter_dir = out_root / cfg["output"]["adapter_dirname"]
    merged_dir = out_root / cfg["output"]["merged_dirname"]
    out_root.mkdir(parents=True, exist_ok=True)

    ds, templated = build_dataset(pre["records"], tokenizer, cfg["data"]["enable_thinking"])
    print(f"[dpo] {len(ds)} preference pairs; device={device}; dtype={dtype_name}", flush=True)

    t_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint), dtype=dtype, low_cpu_mem_usage=True)
    model.config.use_cache = False
    load_seconds = time.time() - t_load

    lora = LoraConfig(**cfg["lora"])

    tcfg = dict(cfg["training"])
    dpo_args = DPOConfig(
        output_dir=str(out_root / "hf_trainer"),
        beta=tcfg["beta"],
        loss_type=tcfg["loss_type"],
        max_length=tcfg["max_length"],
        learning_rate=tcfg["learning_rate"],
        num_train_epochs=tcfg["num_train_epochs"],
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        seed=tcfg["seed"],
        warmup_steps=tcfg["warmup_steps"],
        lr_scheduler_type=tcfg["lr_scheduler_type"],
        optim=tcfg["optim"],
        weight_decay=tcfg["weight_decay"],
        max_grad_norm=tcfg["max_grad_norm"],
        logging_steps=tcfg["logging_steps"],
        gradient_checkpointing=tcfg["gradient_checkpointing"],
        max_steps=int(tcfg.get("max_steps", -1)),
        use_cpu=(device == "cpu"),
        bf16=False,   # weights are already in `dtype`; no autocast/GradScaler path
        fp16=False,
        report_to=[],
        save_strategy="no",
        dataset_num_proc=None,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,       # PEFT policy -> TRL disables the adapter for the reference
        args=dpo_args,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=lora,
    )
    if trainer.ref_model is not None:
        raise RuntimeError(
            "TRL built a SEPARATE reference model. With a PEFT policy it must derive "
            "the reference by disabling the adapter (dpo_trainer.py:918-924); a second "
            "copy of a 1.7B model on this device is not what this script claims to do.")

    t_train = time.time()
    result = trainer.train()
    train_seconds = time.time() - t_train

    loss_history = [(int(e["step"]), float(e["loss"]))
                    for e in trainer.state.log_history if "loss" in e and "step" in e]
    metrics_history = [{k: v for k, v in e.items()
                        if k.startswith("rewards/") or k in ("step", "epoch", "grad_norm",
                                                             "learning_rate", "logps/chosen",
                                                             "logps/rejected")}
                       for e in trainer.state.log_history if "loss" in e]
    if not loss_history:
        raise RuntimeError("training produced no loss entries; there is nothing to log")
    bad = [v for _, v in loss_history if v != v or v in (float("inf"), float("-inf"))]
    if bad:
        raise RuntimeError(f"loss history contains non-finite values: {bad[:5]}")

    t_save = time.time()
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    save_seconds = time.time() - t_save
    adapter_files, adapter_bytes = dir_listing(adapter_dir)

    merged_info = None
    if not args.no_merge:
        merged_info = merge_adapter(
            checkpoint, adapter_dir, merged_dir,
            cfg["output"]["merge_load_dtype"], cfg["output"]["merge_save_dtype"])

    finished = _utc()
    wall = time.time() - t_start

    digest_mode = args.checkpoint_digest
    ck_digest = dir_digest(checkpoint, digest_mode)

    resolved = deep_merge(cfg, {"_smoke": bool(args.smoke), "_device": device,
                                "_dtype": dtype_name, "_run_name": run_name})
    log = build_train_log(
        checkpoint=checkpoint, checkpoint_digest=ck_digest, digest_mode=digest_mode,
        config_resolved=resolved, dataset_path=pre["pairs"],
        dataset_rows=len(ds), dataset_sha256=sha256_file(pre["pairs"]),
        render_fn_dotted=cfg["data"]["prompt_renderer"], sample_row_index=0,
        sample_rendered_prompt=templated[0]["prompt"],
        sample_prompt_messages=[{"role": "user", "content": pre["records"][0]["prompt"]}],
        steps=int(trainer.state.global_step), loss_history=loss_history,
        wall_clock_seconds=wall, adapter_path=str(adapter_dir.relative_to(REPO_ROOT))
        if str(adapter_dir).startswith(str(REPO_ROOT)) else str(adapter_dir),
        adapter_bytes=adapter_bytes,
        merged_path=(str(merged_dir.relative_to(REPO_ROOT))
                     if merged_info and str(merged_dir).startswith(str(REPO_ROOT))
                     else (str(merged_dir) if merged_info else None)),
        merged_bytes=(merged_info["bytes"] if merged_info else 0),
        started_utc=started, finished_utc=finished,
        extras={
            "beta": tcfg["beta"],
            "loss_type": tcfg["loss_type"],
            "reference_model": {
                "strategy": "peft_adapter_disabled",
                "separate_weights_loaded": False,
                "evidence": "trl 1.10.0 dpo_trainer.py:918-924 and 1217-1222; "
                            "asserted at runtime: trainer.ref_model is None",
            },
            "pairs": pre["pair_stats"],
            "lora_vs_sft_recipe": sft_lora_delta(cfg["lora"]),
            "final_metrics": {k: (float(v) if isinstance(v, (int, float)) else v)
                              for k, v in (result.metrics or {}).items()},
            "metrics_history": metrics_history,
            "environment": environment(),
        })

    (out_root / cfg["output"]["train_log_filename"]).write_text(json.dumps(log, indent=2))

    manifest = {
        "run_name": run_name,
        "smoke": bool(args.smoke),
        "device": device,
        "dtype": dtype_name,
        "pairs": pre["pair_stats"],
        "wall_time_s": {"model_load": round(load_seconds, 2),
                        "train": round(train_seconds, 2),
                        "save_adapter": round(save_seconds, 2),
                        "save_merged": round(merged_info["seconds"], 2) if merged_info else None},
        "artifact_files": {"adapter": adapter_files,
                           "merged": merged_info["files"] if merged_info else [],
                           "merge_path": merged_info["path"] if merged_info else None},
        "env": environment(),
        "eval_command": (f".venv/bin/python eval.py --model {merged_dir} "
                         f"--eval-set data/scenarios/corpus.jsonl" if merged_info else None),
    }
    (out_root / cfg["output"]["manifest_filename"]).write_text(json.dumps(manifest, indent=2))

    print(f"[dpo] steps={trainer.state.global_step} "
          f"first_loss={loss_history[0][1]:.6f} last_loss={loss_history[-1][1]:.6f} "
          f"wall={wall:.1f}s", flush=True)
    print(f"[dpo] adapter -> {adapter_dir} ({adapter_bytes} bytes)", flush=True)
    if merged_info:
        print(f"[dpo] merged  -> {merged_dir} ({merged_info['bytes']} bytes)", flush=True)
    print(f"[dpo] train_log -> {out_root / cfg['output']['train_log_filename']}", flush=True)
    return 0


# ---------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="DPO LoRA on top of the SFT checkpoint, on Apple Silicon.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--sft-checkpoint", default=None,
                   help=f"local SFT weights dir (default {DEFAULTS['sft_checkpoint']})")
    p.add_argument("--pairs", default=None,
                   help=f"preference jsonl (default {DEFAULTS['data']['pairs']})")
    p.add_argument("--config", default=None,
                   help="JSON file deep-merged over the built-in DEFAULTS")
    p.add_argument("--run-name", default=None, help="run dir name under runs/")
    p.add_argument("--out-dir", default=None, help="explicit run dir; overrides --run-name")
    p.add_argument("--device", default="auto", choices=("auto", "mps", "cpu"))
    p.add_argument("--dtype", default="auto",
                   help="auto | bfloat16 | float32 (auto: bfloat16 off-CPU, float32 on CPU)")
    p.add_argument("--smoke", action="store_true",
                   help="apply smoke_overrides (tiny max_steps) for a cheap end-to-end check")
    p.add_argument("--max-rows", type=int, default=None, help="cap the number of pairs used")
    p.add_argument("--no-validate-pairs", action="store_true",
                   help="skip the shipped-checker orientation gate (records validated=false)")
    p.add_argument("--scenario-source", default=None,
                   help="jsonl carrying {scenario_id, scenario} for a pairs file that "
                        "only names scenarios by id; default: the sibling manifest's "
                        "source.path")
    p.add_argument("--no-scenario-source", action="store_true",
                   help="do not look up scenarios at all -- DISABLES the prompt-contract "
                        "gate for rows that carry no scenario object")
    p.add_argument("--no-merge", action="store_true",
                   help="stop after the adapter; do not build merged weights")
    p.add_argument("--merge-only", action="store_true",
                   help="merge an existing adapter into standalone weights and exit")
    p.add_argument("--adapter", default=None, help="--merge-only: the adapter dir")
    p.add_argument("--checkpoint-digest", default="content", choices=("content", "listing"),
                   help="how base_model.revision is computed for the train log")
    p.add_argument("--preflight-only", action="store_true",
                   help="run every pre-weights check and exit; loads no model")
    p.add_argument("--print-config", action="store_true",
                   help="print the resolved config as JSON and exit")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    cfg = copy.deepcopy(DEFAULTS)
    if args.config:
        cfg = deep_merge(cfg, json.loads(Path(args.config).read_text()))
    if args.smoke:
        cfg["training"] = deep_merge(cfg["training"], cfg["smoke_overrides"])
    if args.print_config:
        print(json.dumps(cfg, indent=2))
        return 0

    if args.merge_only:
        if not args.adapter:
            print("FAIL --merge-only requires --adapter <dir>", file=sys.stderr)
            return 2
        ck = Path(args.sft_checkpoint or cfg["sft_checkpoint"])
        if not ck.is_absolute():
            ck = REPO_ROOT / ck
        adapter = Path(args.adapter)
        if not adapter.is_absolute():
            adapter = REPO_ROOT / adapter
        out = Path(args.out_dir) if args.out_dir else adapter.parent
        if not out.is_absolute():
            out = REPO_ROOT / out
        if out.name != cfg["output"]["merged_dirname"]:
            out = out / cfg["output"]["merged_dirname"]
        for path, what in ((ck, "SFT checkpoint"), (adapter, "adapter")):
            if not path.is_dir():
                print(f"FAIL {what} {path} is not a directory", file=sys.stderr)
                return 2
        info = merge_adapter(ck, adapter, out,
                             cfg["output"]["merge_load_dtype"],
                             cfg["output"]["merge_save_dtype"])
        print(json.dumps({k: v for k, v in info.items() if k != "files"}, indent=2))
        return 0

    try:
        pre = preflight(cfg, args)
    except PreflightError as exc:
        print(f"FAIL preflight: {exc}", file=sys.stderr)
        return 2

    stats = pre["pair_stats"]
    print(f"[dpo] preflight OK: checkpoint={pre['checkpoint']} device={pre['device']} "
          f"pairs={stats['kept']} kept / {stats['rows_in_file']} in file "
          f"({stats['dropped_null_chosen']} dropped for null chosen), "
          f"validated={stats['validated']} violated_types={stats['violated_types']} "
          f"prompt_contract={stats['prompt_contract_checked']} checked / "
          f"{stats['prompt_contract_unverifiable']} unverifiable "
          f"(scenario source: {stats['scenario_source'].get('mode')})",
          flush=True)
    if args.preflight_only:
        return 0

    return train(cfg, args, pre)


if __name__ == "__main__":
    sys.exit(main())
