#!/usr/bin/env python3
"""
Forbidden-Constraint Holder -- QLoRA fine-tune on Modal.

EXACT INVOCATION (run from the repo root). Modal turns this file's
@app.local_entrypoint signature into the CLI, so `--smoke` is a real flag:
`--run-name TEXT`, `--out-dir TEXT`, `--dataset TEXT`, `--smoke / --no-smoke`.

    # smoke loop: trains on configs/train.json:data.smoke_train_jsonl with
    # smoke_overrides (max_steps 20) applied
    .venv/bin/modal run training/modal_app.py::main --smoke

    # real run: configs/train.json untouched, full 3 epochs
    .venv/bin/modal run training/modal_app.py::main

    # see the generated CLI without touching the network
    .venv/bin/modal run training/modal_app.py::main --help

THE EXAMPLE SCHEMA THIS READS (pinned, docs/PIVOT_DESIGN.md "Data models")
-------------------------------------------------------------------------
    {"messages": [{"role": "user",      "content": "<task + constraints rendered>"},
                  {"role": "assistant", "content": "<clean completion>"}],
     "constraints": [<constraint objects or ids>],
     "provenance": {"teacher": "...", "attempt": <int>}}

There is NO system message. This module does not synthesise one: a system turn
that training invents and the eval harness never sends is exactly the drift the
prompt contract exists to prevent. A row carrying a system message is a hard
error here, not something to quietly drop.

THE PROMPT CONTRACT (the thing that makes every number mean anything)
---------------------------------------------------------------------
The training condition must equal the eval condition. Two layers:

  layer 1  scenario (task + constraints) -> user message.
           Owned by the SHARED RENDERER, which lives in the scenarios/ ---
           harness lane. This module IMPORTS it by the dotted path in
           configs/train.json:data.prompt_renderer and never reimplements it.
           Every row is re-rendered through it and compared, byte for byte,
           against the user message the row actually carries. A single
           mismatch aborts the run before the base model is even downloaded,
           and main() resolves the renderer locally before the container
           starts at all.

  layer 2  user message -> the token sequence fed to the model.
           tokenizer.apply_chat_template(..., add_generation_prompt=True),
           the same call the eval generator makes.

train_log.json records EVIDENCE of both, not a claim about them:
prompt_contract.sample_rendered_prompt is the literal layer-2 text for a
named row index, so an offline checker can re-derive it from the dataset and
compare -- rather than trusting a boolean this file wrote about itself. There
is deliberately no self-attestation flag anywhere in this module.

DECLARED ASSUMPTIONS (the shared renderer is not on disk yet)
-------------------------------------------------------------
At the time this was written the scenarios/harness renderer did not exist.
Nothing here stubs or emulates it. What IS assumed is stated so integration can
reconcile it by editing configuration, not logic:

  A. the dotted path in configs/train.json:data.prompt_renderer resolves to a
     callable inside the `scenarios` or `harness` package -- change the JSON,
     not the code, if the lane lands elsewhere;
  B. it is called render(scenario, constraint_lookup) -> str, or render(scenario)
     if it takes a single positional argument. This is deliberately the SAME
     call scripts/check_loop.py makes (its `_render_arity`/`render_prompt`), so
     the writer and the validator of train_log.json agree by construction;
  C. a training row identifies its source scenario, either inline as
     row["scenario"] or by row["scenario_id"] against the corpus declared in
     configs/train.json:data.scenarios_path. The pinned example schema stores
     only the *rendered* user message, so without one of the two the prompt
     could be self-attested but never re-derived.

If any of these is wrong, resolve_render_fn / scenario_of /
verify_prompt_contract raise with the assumption spelled out. Nothing degrades
to a guess.

Artifacts. Weights are NOT returned through the control plane -- the adapter and
the merged fp16 model are written into the Modal Volume named in
configs/train.json:output.modal_volume and committed. The call returns a small
manifest plus train_log.json, which the local entrypoint writes under the run
directory. Pull the weights with:

    .venv/bin/modal volume get slmforge-artifacts smoke/merged  runs/smoke/merged
    .venv/bin/modal volume get slmforge-artifacts smoke/adapter runs/smoke/adapter

Environment knobs (deliberately NOT in configs/train.json -- they are infra, not
hyperparameters, and train.json is frozen by policy):
    SLMFORGE_GPU         Modal GPU type,   default "L4"   (24GB, bf16-capable)
    SLMFORGE_TIMEOUT_S   function timeout, default 14400  (Modal's own default
                         is 300s and WILL kill a real run partway through)

Written against modal client 1.5.4 (verified with the installed package):
modal.App (modal.Stub does not exist in 1.5.x), gpu= as a string kwarg,
secrets= as a list, Volume.from_name(create_if_missing=True) + vol.commit().
"""
from __future__ import annotations

import importlib
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

#: The pinned train_log.json shape. One lane writes it, one lane validates it,
#: so the key set is a contract and not an implementation detail.
TRAIN_LOG_KEYS = ("base_model", "config_resolved", "dataset", "prompt_contract",
                  "steps", "loss_history", "wall_clock_seconds", "artifacts",
                  "started_utc", "finished_utc")
BASE_MODEL_KEYS = ("repo", "revision")
DATASET_KEYS = ("path", "rows", "sha256")
PROMPT_CONTRACT_KEYS = ("render_fn", "sample_row_index", "sample_rendered_prompt",
                        "sample_prompt_messages")
ARTIFACT_KEYS = ("adapter", "adapter_bytes", "merged_fp16", "merged_bytes")

# The one thing configs/train.json does NOT pin: the container's python deps.
# Unsloth needs a CUDA runtime and a torch build matched to it, so we start from
# NVIDIA's image rather than debian_slim. Versions are left to pip resolution on
# purpose -- pinning a version string that was never built here would be a lie.
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
    # config does not rebuild the image. The scoring/scenario lane rides along
    # so the container can IMPORT the shared prompt renderer instead of
    # reimplementing it.
    .add_local_file(REPO_ROOT / "harness.py", f"{REMOTE_REPO}/harness.py")
    .add_local_dir(REPO_ROOT / "scenarios", f"{REMOTE_REPO}/scenarios")
    .add_local_dir(REPO_ROOT / "constraints", f"{REMOTE_REPO}/constraints")
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


# ------------------------------------------------------- the shared renderer
#: The renderer must live in the repo's shared modules. A training run may not
#: nominate its own prompt renderer -- that would be self-attestation with extra
#: steps. Same allowlist scripts/check_loop.py enforces on the other side.
ALLOWED_RENDERER_ROOTS = ("scenarios", "harness")

_RENDERER_ASSUMPTION = (
    "DECLARED ASSUMPTION: configs/train.json:data.prompt_renderer names the "
    "shared scenario->user-message renderer written by the scenarios/harness "
    "lane, called as render(scenario, constraint_lookup) -> str (or "
    "render(scenario) if it takes one positional argument). This module imports "
    "it and never reimplements it. If the lane published it under a different "
    "name, fix the dotted path in configs/train.json -- do NOT add a local "
    "renderer here, because a second implementation of the prompt IS the drift "
    "this contract exists to prevent."
)


def resolve_render_fn(dotted: str):
    """Import the shared prompt renderer named by its dotted path.

    Raises loudly if it is not importable. There is deliberately no fallback:
    a stubbed renderer would make the training prompt differ from the eval
    prompt while every check still passed.
    """
    if not isinstance(dotted, str) or not dotted.strip():
        raise ValueError(
            f"prompt_renderer must be a non-empty dotted path, got {dotted!r}. "
            + _RENDERER_ASSUMPTION)
    path = dotted.strip().replace(":", ".")
    if "." not in path:
        raise ValueError(
            f"prompt_renderer {dotted!r} names no module. " + _RENDERER_ASSUMPTION)
    module_name, _, attr = path.rpartition(".")
    if module_name.split(".")[0] not in ALLOWED_RENDERER_ROOTS:
        raise ValueError(
            f"prompt_renderer {dotted!r} does not live in the shared renderer "
            f"modules {list(ALLOWED_RENDERER_ROOTS)} -- a training run may not "
            f"nominate its own prompt renderer. " + _RENDERER_ASSUMPTION)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"the shared prompt renderer {dotted!r} is not importable "
            f"({module_name}: {exc}). " + _RENDERER_ASSUMPTION) from exc
    try:
        fn = getattr(module, attr)
    except AttributeError as exc:
        raise AttributeError(
            f"module {module_name!r} has no attribute {attr!r}, named by "
            f"configs/train.json:data.prompt_renderer. " + _RENDERER_ASSUMPTION
        ) from exc
    if not callable(fn):
        raise TypeError(
            f"the shared prompt renderer {dotted!r} is not callable "
            f"(got {type(fn).__name__}). " + _RENDERER_ASSUMPTION)
    return fn


def render_arity(fn) -> int:
    """2 -> the renderer is called (scenario, constraint_lookup); 1 -> (scenario).

    Deliberately identical to scripts/check_loop.py's rule, so the writer and
    the validator of train_log.json call the shared renderer the same way.
    """
    import inspect

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
    if var_positional or positional >= 2:
        return 2
    return 1


def scenario_of(row, index, scenario_corpus=None) -> dict:
    """The source scenario a training row was rendered from.

    The row carries it (``scenario``) or names it (``scenario_id``) in a corpus
    declared by configs/train.json:data.scenarios_path. Without one of the two
    the row's prompt can only be self-attested, never re-derived -- so this is a
    hard error, never a silent pass.
    """
    scenario = row.get("scenario")
    if isinstance(scenario, dict):
        if not isinstance(scenario.get("task"), str) or not scenario["task"].strip():
            raise ValueError(f"row {index}: the row's scenario has no 'task' text, "
                             f"so its prompt cannot be re-rendered")
        return scenario
    sid = row.get("scenario_id")
    if isinstance(sid, str) and sid:
        if not scenario_corpus:
            raise ValueError(
                f"row {index}: names scenario {sid!r} but no scenario corpus is "
                f"declared (configs/train.json:data.scenarios_path), so the "
                f"prompt contract cannot be re-derived from the dataset")
        if sid not in scenario_corpus:
            raise ValueError(f"row {index}: scenario {sid!r} is not in the "
                             f"declared scenario corpus")
        return scenario_corpus[sid]
    raise ValueError(
        f"row {index}: carries neither a 'scenario' object nor a 'scenario_id', "
        f"so its user message can only be self-attested, never re-derived. "
        + _RENDERER_ASSUMPTION)


def constraint_lookup_of(row, index, constraint_corpus=None) -> dict:
    """``{id: constraint}`` for one row, from its own constraint objects, or from
    ids resolved against a declared constraint corpus."""
    raw = row.get("constraints")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"row {index}: 'constraints' is missing or empty; the "
                         f"behavior spec requires at least one")
    lookup = {}
    for entry in raw:
        if isinstance(entry, dict):
            cid = entry.get("id")
            if not isinstance(cid, str) or not cid:
                raise ValueError(f"row {index}: a constraint object has no 'id'")
            lookup[cid] = entry
        elif isinstance(entry, str):
            if not constraint_corpus or entry not in constraint_corpus:
                raise ValueError(
                    f"row {index}: constraint id {entry!r} cannot be resolved "
                    f"(declare configs/train.json:data.constraints_path, or "
                    f"carry constraint objects in the row)")
            lookup[entry] = constraint_corpus[entry]
        else:
            raise ValueError(f"row {index}: constraint entry is a "
                             f"{type(entry).__name__}, not an object or an id")
    return lookup


def call_renderer(render_fn, arity, scenario, lookup) -> str:
    out = render_fn(scenario, lookup) if arity == 2 else render_fn(scenario)
    if not isinstance(out, str):
        raise TypeError(f"the shared prompt renderer returned "
                        f"{type(out).__name__}, not a string. "
                        + _RENDERER_ASSUMPTION)
    return out


def verify_prompt_contract(rows, render_fn, arity=None, scenario_corpus=None,
                           constraint_corpus=None) -> int:
    """Re-render every row through the shared renderer and demand the row's own
    user message back, byte for byte.

    This replaces the abstainer-era guard that compared each row's system
    message against the frozen eval system prompt -- there is no system message
    any more, so the renderer is what binds the two conditions together.
    Returns the number of rows verified; raises on the first mismatch.
    """
    if arity is None:
        arity = render_arity(render_fn)
    for i, row in enumerate(rows):
        user_content = row["messages"][0]["content"]
        scenario = scenario_of(row, i, scenario_corpus)
        lookup = constraint_lookup_of(row, i, constraint_corpus)
        rendered = call_renderer(render_fn, arity, scenario, lookup)
        if rendered != user_content:
            raise ValueError(
                f"row {i}: the stored user message is NOT what the shared "
                f"renderer produces for this scenario + constraint set. Training "
                f"on it would measure the tuned model under a prompt it never saw."
                f"\n  stored:   {user_content[:300]!r}"
                f"\n  rendered: {rendered[:300]!r}")
    return len(rows)


def load_side_corpus(repo_root: Path, rel_path, id_key: str):
    """``{id: object}`` from a declared JSONL side-corpus, or None when the
    config declares none. A declared path that does not load is an error --
    never a silent skip, because the rows that need it would then be
    unverifiable."""
    if rel_path in (None, ""):
        return None
    corpus_file = Path(repo_root) / rel_path
    if not corpus_file.exists():
        raise FileNotFoundError(
            f"declared corpus {rel_path!r} does not exist at {corpus_file}; the "
            f"rows that reference it by id could not be re-rendered")
    corpus = {}
    for line in corpus_file.read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        key = obj.get(id_key)
        if not isinstance(key, str) or not key:
            raise ValueError(f"{rel_path}: a row has no {id_key!r}")
        corpus[key] = obj
    if not corpus:
        raise ValueError(f"declared corpus {rel_path!r} is empty")
    return corpus


# ---------------------------------------------------------------- data plumbing
def render_prompt_text(tokenizer, messages, enable_thinking):
    """The exact layer-2 prompt text: the chat template with a generation prompt
    appended, which is the same call the eval generator makes. Single source --
    build_training_features and the train_log evidence both go through here."""
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking)
    except TypeError:  # tokenizer without the enable_thinking kwarg
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


def validate_example_schema(row, index) -> None:
    """The pinned two-message schema, enforced. A violation raises: a silently
    dropped row is an unnoticed hole in the corpus, and a synthesised system
    message is prompt drift."""
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"row {index}: 'messages' must be a list, got "
                         f"{type(messages).__name__}")
    roles = [m.get("role") for m in messages if isinstance(m, dict)]
    if "system" in roles:
        raise ValueError(
            f"row {index}: the example carries a system message. The pinned "
            f"schema is exactly [user, assistant]; the eval harness sends no "
            f"system turn, so training with one is prompt drift. This module "
            f"will not strip it and will not synthesise one.")
    if roles != ["user", "assistant"]:
        raise ValueError(
            f"row {index}: messages must be exactly [user, assistant], got "
            f"{roles!r}")
    for m, role in zip(messages, roles):
        if not isinstance(m.get("content"), str):
            raise ValueError(f"row {index}: {role} content must be a string")
    if not messages[1]["content"].strip():
        raise ValueError(f"row {index}: the assistant completion is empty -- "
                         f"there is nothing to learn from it")


def build_training_features(tokenizer, rows, max_seq_length, enable_thinking,
                            completion_only_loss=True):
    """Pinned rows -> {input_ids, attention_mask, labels}, using the model's own
    chat template.

    The prompt half is rendered with EXACTLY the call the eval generator makes
    (add_generation_prompt=True), and the target is appended to that rendered
    string. Building it this way -- rather than templating the whole
    conversation in one shot -- guarantees the token prefix the model is trained
    to continue is byte-identical to the prefix it sees at eval time.

    Label masking: prompt tokens -> -100, completion tokens -> themselves. The
    boundary is len(prompt_ids), verified against the joint tokenization so a
    merge across the join cannot shift it by one.

    Returns (features, stats).
    """
    eos = tokenizer.eos_token or ""
    features = []
    stats = {"rows": len(rows), "truncated": 0, "prefix_mismatch": 0,
             "skipped": 0, "kept_row_indices": []}

    for index, row in enumerate(rows):
        validate_example_schema(row, index)
        messages = row["messages"]
        prompt_text = render_prompt_text(tokenizer, messages[:1], enable_thinking)
        target_text = messages[1]["content"] + eos

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

        stats["kept_row_indices"].append(index)
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


# ------------------------------------------------------------------- train_log
def _hex(value, length, field):
    if not isinstance(value, str) or len(value) != length or not all(
            c in "0123456789abcdef" for c in value.lower()):
        raise ValueError(f"{field} must be a {length}-hex string, got {value!r}")
    return value.lower()


def build_train_log(*, base_repo, base_revision, config_resolved, dataset_path,
                    dataset_rows, dataset_sha256, render_fn_dotted,
                    sample_row_index, sample_rendered_prompt, sample_prompt_messages, steps,
                    loss_history, wall_clock_seconds, adapter_path,
                    adapter_bytes, merged_path, merged_bytes, started_utc,
                    finished_utc, scenarios_path=None,
                    constraints_path=None) -> dict:
    """The pinned train_log.json, and nothing else.

    Every field is derived from something that actually happened; the function
    refuses to emit a log with empty evidence. Note what is NOT here: no
    self-attestation boolean. A checker must never trust a flag written by the
    thing it is checking, so prompt_contract carries the rendered text itself
    and the checker re-derives the claim from the dataset.
    """
    if not isinstance(sample_rendered_prompt, str) or not sample_rendered_prompt:
        raise ValueError(
            "prompt_contract.sample_rendered_prompt is empty. It is EVIDENCE -- "
            "the literal prompt text fed to the model for that row -- and an "
            "empty string proves nothing.")
    if not isinstance(sample_row_index, int) or isinstance(sample_row_index, bool) \
            or sample_row_index < 0:
        raise ValueError(f"sample_row_index must be a non-negative int, got "
                         f"{sample_row_index!r}")
    if not isinstance(render_fn_dotted, str) or "." not in render_fn_dotted:
        raise ValueError(f"prompt_contract.render_fn must be the dotted path of "
                         f"the shared renderer, got {render_fn_dotted!r}")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
        raise ValueError(f"steps must be a positive int, got {steps!r}")
    if not isinstance(dataset_rows, int) or dataset_rows <= 0:
        raise ValueError(f"dataset.rows must be a positive int, got "
                         f"{dataset_rows!r}")

    history = []
    for entry in loss_history:
        step, loss = entry
        history.append([int(step), float(loss)])

    # The three pinned dataset keys, plus -- only when the run actually used
    # them -- the side corpora the rows reference by id. Without those declared,
    # a checker holding an id-carrying dataset cannot resolve the scenario and
    # the contract stops being re-derivable.
    dataset = {"path": str(dataset_path),
               "rows": int(dataset_rows),
               "sha256": _hex(dataset_sha256, 64, "dataset.sha256")}
    if scenarios_path:
        dataset["scenarios_path"] = str(scenarios_path)
    if constraints_path:
        dataset["constraints_path"] = str(constraints_path)

    return {
        "base_model": {"repo": str(base_repo),
                       "revision": _hex(base_revision, 40, "base_model.revision")},
        "config_resolved": config_resolved,
        "dataset": dataset,
        "prompt_contract": {"render_fn": render_fn_dotted,
                            "sample_row_index": int(sample_row_index),
                            "sample_rendered_prompt": sample_rendered_prompt,
                            "sample_prompt_messages": sample_prompt_messages},
        "steps": int(steps),
        "loss_history": history,
        "wall_clock_seconds": float(wall_clock_seconds),
        "artifacts": {"adapter": str(adapter_path),
                      "adapter_bytes": int(adapter_bytes),
                      "merged_fp16": str(merged_path),
                      "merged_bytes": int(merged_bytes)},
        "started_utc": started_utc,
        "finished_utc": finished_utc,
    }


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

    cfg = load_config(Path(REMOTE_REPO) / "configs" / "train.json")
    resolved = resolve_config(cfg, smoke)
    tcfg = resolved["training"]
    lcfg = resolved["lora"]
    qcfg = resolved["quantization"]
    dcfg = resolved["data"]
    ocfg = resolved["output"]

    # ---- dataset -------------------------------------------------------------
    ds_file = Path(REMOTE_REPO) / dataset_path
    if not ds_file.exists():
        raise FileNotFoundError(
            f"training dataset {dataset_path!r} not found at {ds_file} -- the "
            f"data pipeline must commit it before a GPU run can train on it")
    raw_bytes = ds_file.read_bytes()
    rows = [json.loads(ln) for ln in raw_bytes.decode().splitlines() if ln.strip()]
    if not rows:
        raise ValueError(f"training dataset {dataset_path!r} is empty")

    # ---- THE PROMPT CONTRACT, checked before the base model is downloaded ----
    render_fn_dotted = dcfg["prompt_renderer"]
    render_fn = resolve_render_fn(render_fn_dotted)
    scenarios_path = dcfg.get("scenarios_path")
    constraints_path = dcfg.get("constraints_path")
    scenario_corpus = load_side_corpus(REMOTE_REPO, scenarios_path, "scenario_id")
    constraint_corpus = load_side_corpus(REMOTE_REPO, constraints_path, "id")
    verify_prompt_contract(rows, render_fn, scenario_corpus=scenario_corpus,
                           constraint_corpus=constraint_corpus)

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

    # ---- tokenize ------------------------------------------------------------
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

    # Evidence for prompt_contract: the literal layer-2 text for a row that was
    # actually trained on, produced by the same function that built its tokens.
    sample_row_index = ds_stats["kept_row_indices"][0]
    # The MESSAGES that were templated, recorded verbatim alongside the rendered
    # text. An offline checker can compare these EXACTLY -- no tokenizer needed and
    # nothing to parse. Recording only the layer-2 string forced the checker into
    # locating the prompt inside chat-template scaffolding, and "scaffolding" is
    # attacker-supplied: an injected system turn, or a bare <note> tag, both read
    # as scaffold and passed as verified.
    sample_prompt_messages = [dict(m) for m in rows[sample_row_index]["messages"][:1]]
    sample_rendered_prompt = render_prompt_text(
        tokenizer, rows[sample_row_index]["messages"][:1], dcfg["enable_thinking"])

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
    trainer.train()
    train_s = time.time() - t_train

    loss_history = [[h["step"], h["loss"]]
                    for h in trainer.state.log_history if "loss" in h]

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

    finished_utc = datetime.now(timezone.utc).isoformat()
    train_log = build_train_log(
        base_repo=base_repo,
        base_revision=base_rev,
        config_resolved=resolved,
        dataset_path=dataset_path,
        dataset_rows=len(rows),
        dataset_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        scenarios_path=scenarios_path,
        constraints_path=constraints_path,
        render_fn_dotted=render_fn_dotted,
        sample_row_index=sample_row_index,
        sample_rendered_prompt=sample_rendered_prompt,
        sample_prompt_messages=sample_prompt_messages,
        steps=int(trainer.state.global_step),
        loss_history=loss_history,
        wall_clock_seconds=time.time() - t_zero,
        adapter_path=f"{run_name}/{ocfg['adapter_dirname']}",
        adapter_bytes=adapter_bytes,
        merged_path=f"{run_name}/{ocfg['merged_dirname']}",
        merged_bytes=merged_bytes,
        started_utc=started_utc,
        finished_utc=finished_utc,
    )

    (out_root / ocfg["train_log_filename"]).write_text(
        json.dumps(train_log, indent=2))
    artifacts_vol.commit()

    try:
        import unsloth as _unsloth
        unsloth_version = getattr(_unsloth, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        unsloth_version = "unknown"

    # Everything that is NOT part of the pinned train_log shape rides in the
    # manifest instead, so the pinned artifact stays exactly the agreed keys.
    return {
        "run_name": run_name,
        "smoke": bool(smoke),
        "volume": ocfg["modal_volume"],
        "run_dir_remote": str(out_root),
        "gpu": GPU,
        "tokenization": ds_stats,
        "wall_time_s": {"model_load": round(load_s, 2),
                        "tokenize": round(tokenize_s, 2),
                        "train": round(train_s, 2),
                        "save_adapter": round(adapter_s, 2),
                        "save_merged": round(merge_s, 2)},
        "artifact_files": {"adapter": adapter_files, "merged": merged_files,
                           "merge_path": merge_path},
        "env": {"python": sys.version.split()[0], "torch": torch.__version__,
                "transformers": transformers.__version__,
                "unsloth": unsloth_version},
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
            f"the data pipeline must generate and commit it before a GPU run.")

    # Fail here, locally and for free, rather than after a container spins up.
    resolve_render_fn(dcfg["prompt_renderer"])
    load_side_corpus(REPO_ROOT, dcfg.get("scenarios_path"), "scenario_id")
    load_side_corpus(REPO_ROOT, dcfg.get("constraints_path"), "id")

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

    last_loss = train_log["loss_history"][-1][1] if train_log["loss_history"] else None
    print(f"[slmforge] steps={train_log['steps']} last_loss={last_loss} "
          f"wall_clock_s={train_log['wall_clock_seconds']:.1f}")
    print(f"[slmforge] wrote {out_dir}/{ocfg['train_log_filename']} and "
          f"{out_dir}/{ocfg['manifest_filename']}")
    print("\n[slmforge] pull the weights, then close the loop:")
    for name in (ocfg["merged_dirname"], ocfg["adapter_dirname"]):
        print(f"  modal volume get {manifest['volume']} {run_name}/{name} "
              f"{out_dir}/{name}")
