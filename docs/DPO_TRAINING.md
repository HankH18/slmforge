# DPO on top of the SFT checkpoint

Stretch Ladder rung 1, docs/project_spec.pdf p.4, verbatim: *"DPO / preference
tuning — build preference pairs (on-spec vs. off-spec) and run DPO on top of your
SFT model; measure the delta over SFT alone."*

Code: `training/dpo_local.py`. Environment: `training/requirements-dpo.txt`
(`.venv-dpo`, **never** `.venv`).

---

## Status: what has been run, and what has not

**NOT RUN TO COMPLETION — the real DPO job was started on MPS and deliberately
abandoned. This is the headline of this document.** The finding is not about the
data or the rig; both worked. It is that **Qwen3-1.7B DPO on Apple Silicon MPS
degrades until it is unusable.**

Run: `dpo_local.py --pairs data/train/v3.dpo_pairs.jsonl --run-name dpo_v1
--device mps`, all 1211 pairs, bf16, `max_length` 2048, per-device batch 1 ×
grad-accum 8 → 152 optimizer steps. Preflight passed on every row: 1211 kept,
0 dropped, prompt contract 1211/1211 re-derived through the shared renderer.

| optimizer step | elapsed | seconds/step |
|---:|---:|---:|
| 1 | 00:23 | 23.7 |
| 8 | 03:28 | 32.1 |
| 16 | 16:24 | 100.0 |
| 32 | 46:52 | 98.3 |
| 40 | 1:19:42 | 345.0 |
| 48 | 3:05:21 | 745.2 |
| 65 | 6:23:22 | 731.7 |

**Step time rose 30.8× — 23.7 s to 731.7 s — and the rise is monotone after
roughly step 32.** It is not sequence-length variance: the per-step token counts do
not vary by anything near thirty-fold, and the degradation does not reverse.
System memory free fell 43% → 30% → 18% while the run held the only large
allocation on the machine, and kept falling after other work on the host had
exited, so the trainer is accumulating unified memory rather than merely competing
for it. The projected finish moved from 59 minutes at step 1 to **17h40m** at step
65. The run was stopped there.

**Nothing was lost by stopping.** The DPO loss over those 65 steps moved
0.6958 → 0.6914 against a chance value of ln 2 = 0.6931 — at `lr` 5e-6 the policy
had not yet separated chosen from rejected, so there was no partial result to
salvage. Nothing checkpoints mid-run, and no artifacts were written.

**What this means for the stretch rung.** The preference dataset is built,
published and independently verified (`data/train/v3.dpo_pairs.jsonl`, 1211 pairs,
`docs/DPO.md`); the trainer is built, guarded and smoked end to end; the reference-
model behaviour and the seeding defect were found and fixed by running the real
library. What is missing is a trained checkpoint and therefore the delta over SFT
alone, and the blocker is compute, not method: the CUDA workspaces this project
used are over their spend limit (see `docs/EFFICIENCY_CURVE.md` §5), and MPS is
demonstrably not a substitute at this model size. **No DPO delta is reported
anywhere in this repository, because none has been measured.**

Anyone with a working CUDA card can produce it with the commands in *How to run
it* below; the measured cost to beat is 152 optimizer steps.


**RUN — a CPU end-to-end smoke against a randomly-initialised 9,780,928-parameter
Qwen3 model.** 4 real preference pairs, `max_steps=2`, `--device cpu`. It loaded,
stepped, produced finite losses (`0.693147 → 0.693135`), wrote `train_log.json`
and `run_manifest.json`, saved an 11,574,032-byte LoRA adapter, and merged that
adapter into a 30,992,686-byte standalone directory that reloads and runs a
forward pass. Comparing the merged weights against the bf16-cast base shows
**exactly 14 of 24 parameter tensors changed — 2 layers × the 7 targeted
projections, and nothing else.** All 14 `lora_B` tensors are non-zero
(max |B| = 7.5e-06), so the adapter carries a real update rather than its
zero initialisation.

**FOUND AND FIXED — the run was not reproducible.** Two identical smoke runs
produced *different* adapters: max |A − A′| = 0.249, and different step-2 losses
(0.6931167 vs 0.6931350). `DPOConfig(seed=42)` seeds inside `Trainer.__init__`,
but `DPOTrainer` calls `peft.get_peft_model` **before** that point, so `lora_A`
was drawn from an unseeded RNG. `dpo_local.py` now calls
`transformers.set_seed(seed)` before any weight is created. After the fix, two
identical runs produce **bit-identical adapters (max abs diff 0.0)** and an
identical loss history.

**RUN — the preflight gate against all four preference files on the real data:**

| file | rows in file | pairs kept | dropped (`chosen` null) | prompt contract re-derived | violation types |
|---|---|---|---|---|---|
| `data/train/v1.dpo_negatives.jsonl` | 474 | 184 | 290 | 184 / 184 | `banned_letter` 184 |
| `data/train/v2.dpo_negatives.jsonl` | 1053 | 522 | 531 | 522 / 522 | `banned_letter` 522 |
| `data/train/v3.dpo_negatives.jsonl` | 2266 | 1211 | 1055 | 1211 / 1211 | `banned_letter` 1211 |
| `data/train/v3.dpo_pairs.jsonl` | 1211 | 1211 | 0 | 1211 / 1211 | `banned_letter` 1211 |

Every pair was re-checked with the shipped `constraints.checkers.check`: `chosen`
passes every constraint, `rejected` fails at least one. Every prompt was
re-rendered through `scenarios.render.render_user_message` and came back
byte-identical. 100% of violations are `banned_letter` — that is the whole
preference signal, not a sample of it.

The two v3 files produce a **byte-identical** 1,211-pair training set; they are
the same data in two serialisations.

**NOT COMPLETED — the real DPO training, and therefore its evaluation.** No delta
over SFT alone has been measured. The commands below the "How to run it" heading
*have* now been exercised against Qwen3-1.7B on MPS — a 2-step smoke that completed
in 23.7 s and wrote a real adapter and a merged directory, and the 65-step partial
run recorded at the top of this document — so they are known to work; what is not
known is how they behave to completion, because no run has reached completion.

**MEASURED — wall clock, and it is the reason there is no result.** 23.7 s for the
first optimizer step, rising monotonically to 731.7 s by step 65: see the table at
the top. The full 152-step pass was projected at 17h40m and rising when it was
stopped. Peak memory was not instrumented directly; system memory free fell
43% → 18% over the run and did not recover until the process exited.

```bash
./.venv-dpo/bin/python training/dpo_local.py --smoke --max-rows 8 --run-name dpo_probe
```

---

## Why a locally-trained DPO result is admissible

Both Modal workspaces are over their spend limit, so CUDA is unavailable and the
DPO step runs on this machine's MPS device.

For a data-efficiency curve point that would be fatal: the curve compares runs to
each other, so a point trained on a different stack is confounded by the stack.

**The DPO contrast is not confounded, because the stack is held constant across
it.** The spec asks for "the delta over SFT alone". Both sides of that comparison
are the *same* CUDA-trained SFT checkpoint — `runs/train_v2/merged`, the weights
behind the published `tuned` pin in `configs/models.json`:

- the SFT side of the delta **is** that checkpoint, evaluated as-is;
- the DPO side is that checkpoint plus a LoRA adapter, and TRL's reference policy
  during training is that checkpoint again (the adapter, disabled).

Only the DPO step differs between the two arms. Nothing that varies between CUDA
and MPS varies *across* the comparison.

Two conditions this argument depends on, both enforced in code rather than
asserted here:

1. **Both arms are evaluated by the same harness on the same device.** Run
   `eval.py` for the SFT arm and the DPO arm the same way, or the delta measures
   the harness.
2. **Both arms see the same prompt.** `dpo_local.py` re-renders every training
   prompt through `scenarios.render.render_user_message` — the one renderer
   `eval.py` also uses — and refuses to train on a single byte of divergence.

---

## What differs from the SFT recipe — stated as limitations, not footnotes

`configs/train.json` is pinned by contract and describes the SFT run.
`training/dpo_local.py` never reads a hyperparameter out of it. Three parts of
that recipe **cannot** be reproduced here, and are not:

| SFT recipe (CUDA) | DPO run (MPS) | Why |
|---|---|---|
| `quantization.load_in_4bit: true` | no quantization; bf16 weights | bitsandbytes ships CUDA kernels only. There is no MPS build. |
| `training.optim: "adamw_8bit"` | `adamw_torch` | also bitsandbytes. |
| `training.warmup_ratio: 0.03` | `warmup_steps: 10` | `warmup_ratio` was **removed** from `transformers.TrainingArguments` in transformers 5.x, which is what `.venv-dpo` resolves. A different knob, not a rename. |
| `use_gradient_checkpointing: "unsloth"` | `gradient_checkpointing: false` | unsloth is not installed in either environment. |
| trained on Modal, NVIDIA A10 | trained on Apple Silicon, MPS | both Modal workspaces are over their spend limit. |
| transformers 4.57.6 | transformers 5.15.1 | `.venv-dpo` resolved forward; peft is 0.20.0 in both. |

Consequences worth naming plainly:

- **The DPO adapter is trained against un-quantized weights while the SFT adapter
  was trained against NF4-quantized weights.** The *base* for both merges is the
  same fp16 merged checkpoint, so the artifacts are comparable; the optimisation
  conditions were not identical.
- **The training hyperparameters are unswept defaults** inside TRL's documented
  DPO range (`beta=0.1`, `lr=5e-6`, 1 epoch, effective batch 8). No sweep was run
  on this project's data. A DPO run that shows no delta is therefore evidence
  about *these* hyperparameters on *this* data, not about DPO.
- **Only one constraint type is in the preference data.** 1211 of 1211 pairs
  differ on `banned_letter`. Any delta this run produces is a delta on
  banned-letter behaviour; a delta on `no_lists`, `banned_words`,
  `no_quotation_marks` or `banned_punctuation` cannot be attributed to it,
  because no pair in the set carries that signal.

---

## The reference model

With a PEFT LoRA policy and `ref_model=None`, TRL derives the reference by
disabling the adapter. No second copy of the 1.7B weights is loaded. This was
read out of the installed library, not assumed — `trl 1.10.0`,
`trl/trainer/dpo_trainer.py:918-924`:

```python
# Reference model
if ref_model is None:
    if is_peft_model(self.model) or args.precompute_ref_log_probs:
        # If PEFT is used, the reference model is not needed since the adapter can be disabled to revert to the
        # initial model. If precompute_ref_log_probs is True, the reference model does not need to be kept in
        # memory during training.
        self.ref_model = None
```

and the loss path, `dpo_trainer.py:1217-1222`:

```python
if self.ref_model is None and is_peft_model(self.model):
    unwrapped_model = self.accelerator.unwrap_model(self.model)
    with use_adapter(
        unwrapped_model, adapter_name="ref" if "ref" in unwrapped_model.peft_config else None
    ):
        ref_outputs = model(**model_kwargs)
```

`dpo_local.py` asserts `trainer.ref_model is None` immediately after
construction and aborts otherwise. The assertion is not vacuous: the same
construction *without* `peft_config` leaves `trainer.ref_model` a full
`Qwen3ForCausalLM`, which is what the guard exists to catch.

---

## Hyperparameters

Defaults live in `DEFAULTS` at the top of `training/dpo_local.py`. Print the
resolved set with `--print-config`; every value also lands in
`train_log.json:config_resolved`.

| knob | default | note |
|---|---|---|
| `beta` | 0.1 | DPO regularisation strength |
| `loss_type` | `sigmoid` | the original DPO loss |
| `learning_rate` | 5e-6 | unswept |
| `num_train_epochs` | 1 | 1211 pairs ÷ effective batch 8 ≈ 151 optimizer steps |
| `per_device_train_batch_size` | 1 | |
| `gradient_accumulation_steps` | 8 | effective batch 8 |
| `max_length` | 2048 | matches the SFT `max_seq_length`; TRL 1.10 has no separate `max_prompt_length` |
| `seed` | 42 | matches the SFT recipe |
| `warmup_steps` | 10 | see the limitations table |
| `lr_scheduler_type` | `linear` | matches the SFT recipe |
| `optim` | `adamw_torch` | see the limitations table |
| `weight_decay` | 0.0 | |
| `max_grad_norm` | 1.0 | matches the SFT recipe |
| `logging_steps` | 1 | matches the SFT recipe |
| LoRA `r` / `alpha` / `dropout` | 16 / 32 / 0.05 | **identical to the SFT recipe** |
| LoRA `target_modules` | the 7 projections | **identical to the SFT recipe** |
| `--smoke` overrides | `max_steps=2`, `warmup_steps=0`, `grad_accum=1` | |

The LoRA block is cross-checked against `configs/train.json` at write time and
the difference is recorded in `train_log.json:dpo.lora_vs_sft_recipe`. It is
currently `{}` — no divergence. Override anything with `--config <file>.json`,
which deep-merges over the defaults.

---

## How to run it

### 1. Recreate the environment

```bash
/opt/homebrew/bin/python3.11 -m venv .venv-dpo
./.venv-dpo/bin/pip install -r training/requirements-dpo.txt
```

Resolved on 2026-08-22, Python 3.11.15, pip 26.2.1, macOS 25.4.0 arm64:
`torch==2.13.0`, `transformers==5.15.1`, `trl==1.10.0`, `peft==0.20.0`,
`datasets==5.0.1`, `accelerate==1.14.0`. The full transitive freeze is in
`training/requirements-dpo.txt`.

`.venv-dpo/` is gitignored. **Nothing was installed into `.venv`**, which still
holds `transformers 5.15.1` and `torch 2.13.0` and no `trl` or `peft`. trl 1.10.0
did *not* drag transformers backward — the two environments resolved to the same
transformers version — but they stay separate, because that outcome is a fact
about one resolution and not a guarantee.

### 2. Preflight — loads no weights, costs nothing

```bash
./.venv-dpo/bin/python training/dpo_local.py --preflight-only
```

Checks, in this order, **before a byte of the checkpoint is read**: the SFT
checkpoint directory and its weights exist; the pair file exists; the shared
prompt renderer imports; the constraint and scenario corpora load; every pair is
well-oriented under the shipped checkers; every prompt re-renders byte-identically;
the tokenizer has a chat template and does not inject special tokens on `__call__`;
the requested device is available. Exit code 2 and a `FAIL preflight:` line on any
failure.

### 3. Train

```bash
./.venv-dpo/bin/python training/dpo_local.py \
    --pairs data/train/v3.dpo_pairs.jsonl \
    --run-name dpo_v1
```

Writes `runs/dpo_v1/adapter`, `runs/dpo_v1/merged`, `runs/dpo_v1/train_log.json`
and `runs/dpo_v1/run_manifest.json`. `--no-merge` stops after the adapter.

### 4. Merge separately, if the adapter already exists

```bash
./.venv-dpo/bin/python training/dpo_local.py --merge-only \
    --adapter runs/dpo_v1/adapter --out-dir runs/dpo_v1
```

Mirrors the SFT run's recorded merge path — `run_manifest.json` for
`runs/train_v2` says `peft.merge_and_unload(fp32 cpu -> merged_16bit)` — by
loading float32 on CPU, merging, casting to bfloat16, saving. Always CPU: merging
is one-shot weight arithmetic and has no reason to contend for accelerator memory.

### 5. Evaluate, with the graded environment

`eval.py` accepts a local directory as a model spec
(`eval.py:normalize_model_spec`, rule 3), so the merged directory is evaluable
directly. **Run eval with `.venv`, not `.venv-dpo`** — `.venv` is the graded
environment:

```bash
set -a && . ./.env && set +a

# the DPO arm
.venv/bin/python eval.py --model runs/dpo_v1/merged \
    --eval-set data/scenarios/corpus.jsonl --out runs/dpo_v1_eval

# the SFT arm -- the delta's other side, same harness, same device
.venv/bin/python eval.py --model runs/train_v2/merged \
    --eval-set data/scenarios/corpus.jsonl --out runs/sft_arm_eval
```

Both arms must be scored by the same judge instrument end to end. Per
`configs/models.json`, swapping judges between the two arms shifts the
task-quality floor and makes the numbers non-comparable.

---

## Evidence the run leaves behind

`train_log.json` carries the same ten keys as the SFT runs
(`training/modal_app.py:TRAIN_LOG_KEYS`): `base_model`, `config_resolved`,
`dataset`, `prompt_contract`, `steps`, `loss_history`, `wall_clock_seconds`,
`artifacts`, `started_utc`, `finished_utc`. Two honest departures:

- **`base_model.revision` is not a 40-hex Hub commit.** The base of a DPO run is
  a local directory. The field carries a sha256 content digest of that directory
  and `base_model.revision_kind` says which kind of digest it is. The log records
  the local digest; it does **not** verify the claim that these weights equal the
  published `tuned` pin.
- **An extra `dpo` key** holds what has no SFT counterpart: `beta`, `loss_type`,
  the reference-model strategy and its runtime assertion, the pair statistics
  (including the scenario-source provenance and its sha256 check), the LoRA
  cross-check against the SFT recipe, the final metrics, the per-step
  reward/logp history, and the resolved package versions.

`run_manifest.json` mirrors the SFT manifest's shape: run name, device, dtype,
pair statistics, per-phase wall times, artifact file listings, the merge path
string, the environment, and the `eval.py` command line for the merged directory.

---

## Input shapes

`--pairs` takes either of the two files this repo produces.

`data/train/*.dpo_negatives.jsonl` — `prompt`/`chosen`/`rejected` are strings and
each row carries its source `scenario` object inline. `chosen` is `null` on the
prompts where the teacher never produced a compliant draft; those rows are
dropped and **counted** in `dpo.pairs.dropped_null_chosen`, never silently passed
through.

`data/train/*.dpo_pairs.jsonl` — TRL conversational form: `prompt`/`chosen`/
`rejected` are one-message lists and the row carries only `scenario_id`. The
`dg-*` ids in it are **not** in `data/scenarios/corpus.jsonl`, so without a
scenario object the prompt could only be self-attested. The sibling
`<stem>.manifest.json`'s `source.path` supplies the scenarios, and that file is
hashed against the manifest's `source.sha256` first — a pairs file built from a
different source than the one it names is fatal. `--scenario-source <path>`
overrides the lookup; `--no-scenario-source` disables it and, with it, the
prompt-contract gate, recording `prompt_contract_unverifiable` in the log rather
than claiming a verification that did not happen.

Both shapes are normalised to three plain strings before anything else runs. A
multi-turn `prompt` is rejected rather than flattened: this project's eval
condition is exactly one user turn.

---

## Failure modes that were exercised

Each of these was run against the real code and exits 2 before any weights load:

| sabotage | result |
|---|---|
| SFT checkpoint path does not exist | `FAIL preflight: SFT checkpoint … is not a directory. Nothing was loaded.` |
| pair file does not exist | `FAIL preflight: preference pair file … does not exist. Refusing to train on a silently smaller or absent set.` |
| a pair with `chosen` and `rejected` swapped | `FAIL preflight: … row 1: 'chosen' FAILS 1 constraint(s) under the shipped checkers -- [('c-chr-003', "banned letter 's' at offset 24: …")]` |
| `chosen == rejected` | `FAIL preflight: … row 0: chosen == rejected; not a preference` |
| a prompt edited away from the renderer's output | `FAIL preflight: … row 2: the stored prompt is NOT what the shared renderer produces …` |
| every row has `chosen: null` | `FAIL preflight: … every row was dropped (3 had a null 'chosen'). There is nothing to train on.` |
| conversational pairs with no manifest and unresolvable `scenario_id` | `FAIL preflight: … carries no 'scenario' object and its scenario_id 'dg-00002' resolves in neither …` |
| manifest declaring a `source.sha256` the source file does not match | `FAIL preflight: … declares source sha256 000… but … hashes to 7d02d7af… The pairs file was built from a different source than the one it names.` |
| a two-message `prompt` | `FAIL preflight: … row 1: 'prompt': expected exactly one 'user' message, got 2` |
| a tokenizer that injects a special token on `__call__` | `FAIL preflight: this tokenizer injects special tokens on __call__, so TRL's prompt tokenization would not match …` |
| `DPOTrainer` built without `peft_config` (would load a second reference model) | guard fires; the same construction *with* `peft_config` leaves `ref_model` `None` |

---

## What remains unproven

- **The delta over SFT alone.** Not measured. That is the deliverable this file
  exists to make runnable, and it has not been run.
- **That the trainer works on MPS at all.** Every execution so far was
  `--device cpu` against a 9.8M-parameter random model. bf16 on MPS, memory
  headroom for a 1.7B policy at `max_length=2048`, and MPS kernel coverage for
  every op TRL's DPO loss uses are all untested here.
- **That `runs/train_v2/merged` is byte-identical to the published `tuned` pin.**
  Taken from the orchestrator and recorded, not verified.
- **That these hyperparameters are the right ones.** They are unswept defaults.
- **Anything about constraint types other than `banned_letter`.** No pair in the
  set carries a different violation.
