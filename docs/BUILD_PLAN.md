> **SUPERSEDED — this document plans the RETIRED grounded-abstainer target.**
> The behaviour it calls "frozen" below is **not** the graded Behavior Spec. That
> target failed its Prompt-Ceiling Ablation (the frontier models scored a perfect
> adherence across every strategy, so there was no ceiling to beat) and the project
> pivoted to composed constraint-holding.
>
> The current Behavior Spec is [`docs/PIVOT_SPEC.md`](PIVOT_SPEC.md) and
> [`README.md`](../README.md). The current Prompt-Ceiling Ablation is
> [`docs/ABLATION_G1.md`](ABLATION_G1.md) over `runs/g1/` — **not** `runs/ablation/`.
>
> Kept unedited because it is cited by `configs/train.json` and by the frozen
> `train_log.json` in every committed training run. It is history, not instruction.

# Grounded Abstainer — MVP Build Plan

One document. Everything Claude Code needs to take the frozen artifacts to the
Gauntlet MVP: phases in order, the contracts that prevent drift, and a verify
command per phase.

## Frozen inputs (never edit)

`eval.py`, `run_ablation.py`, `data/eval/scenarios_v1.jsonl` (32 scenarios),
`data/eval/fewshot_exemplars.jsonl` (6 exemplars). The test suite pins
`eval.rubric_hash() == "c0adb876c3e6c4c9"` — if that assertion ever fails, the
rubric drifted: stop and ask. A genuine defect in frozen code is also stop-and-ask,
never a silent fix.

**NOT THE GRADED SPEC — this is the retired abstainer behaviour. See the banner at
the top of this file; the graded Behavior Spec is in `docs/PIVOT_SPEC.md`.**

Behavior Spec (frozen): Given a question and a context passage, the model answers
only if the passage contains the answer, outputting exactly one `Answer: <answer>`
line followed by one or more `Citation: "<span>"` lines where every cited span is a
verbatim substring of the passage and the spans jointly entail the answer;
otherwise its entire output is exactly `Not answerable from the provided context.`

## Documentation policy (deliberate, do not "improve")

Documentation is opt-in, not default. Always committed: source, tests, configs,
datasets, the run artifacts the project doc grades (transcripts JSONL, results
JSON, ablation table, MVP_REPORT.md), and one README. Beyond that, create a doc
only when something genuinely needs it — a non-obvious setup step a grader would
otherwise trip on, a decision that must outlive the session — and prefer adding
to the README over adding a new file. Never: process journals, per-step logs,
changelogs, work summaries, or restating in prose what code and configs already
say. Code comments only where the code is non-obvious. The test for any new doc:
who reads this, and what breaks if it doesn't exist?

## Contracts (the few that prevent drift)

- **configs/models.json** — written in P1, read everywhere after:
  `{"frontier": ["openai:<id>", "google:<id>"], "judge_model": "<anthropic id>",
  "base": "hf:Qwen/Qwen3-1.7B@<revision>", "verified_utc": "...", "sources": [...],
  "tuned": "<repo>@<commit>"  // added in P5}`
  IDs are verified against provider docs at runtime (browser, read-only), never
  trusted from memory. Judge stays a third family: no model scores itself.
- **configs/train.json** — pinned hyperparameters, written in P4:
  Qwen3-1.7B-Instruct @ the same revision; LoRA r16/alpha32/dropout 0.05 on
  q,k,v,o,gate,up,down projections; lr 2e-4, 3 epochs, batch 8×2 accum, seq 2048,
  seed 42, warmup 0.03; `smoke_overrides: {max_steps: 20}`. A disappointing model
  is a DATA problem — changing these to chase metrics is prohibited; ask first.
- **Training example schema** (P3 output, P4/P5 input) — JSONL:
  `{"messages": [system, user, assistant], "meta": {type, gold_label, source_batch,
  verifier}}` where system = `eval.SYSTEM_CORE` verbatim and user is built with
  `eval.user_msg()`. Training condition must equal the zero_shot eval condition.
- **Datagen verifier** — keep an example iff its assistant content passes
  `eval.parse_output` (imported, never reimplemented): exact refusal for gold
  unanswerable; valid format + every span a verbatim passage substring for gold
  answerable. Plus: no `"` in passages, no 6-gram overlap (lowercased) with eval
  scenarios or exemplars, dedupe on normalized (passage, question). Teacher calls
  cached to disk so reruns don't re-bill.
- **Model publishing** — merged fp16 model (adapter too) to a public HF repo so
  `eval.py --model hf:<repo>@<commit>` runs it directly; exact commit hash recorded.

## Phases

Work in order. A phase is done when its verify command exits 0 AND
`python3 -m pytest tests -q` is green (network-free). Commit per phase.

**P0 — Skeleton + tests.** Place frozen files (eval.py, run_ablation.py at root;
JSONLs under data/eval/). Port the delivered mock smoke checks into pytest:
rubric-hash pin; the five mock-behavior fixtures (oracle SA=1.000; always_refuse
SA=0.594 FRR=1.0; nonverbatim SA=0; format_breaker malformed=32; bad_entail
entailment_fail=13); few_shot and --compare-with paths. `.gitignore` covers .env
from the first commit; `.env.example` lists required keys; requirements.txt.
Verify: `python3 -m pytest tests -q`

**P1 — Backends live, IDs pinned.** `scripts/ping_backends.py`: 2-scenario
round-trip through both frontier backends, the judge, and hf:Qwen/Qwen3-1.7B;
writes configs/models.json per the contract. Missing/invalid key → stop and ask,
naming the key.
Verify: `python3 scripts/ping_backends.py --check configs/models.json`

**P2 — Prompt-Ceiling Ablation. [GATE G1]** Run `run_ablation.py` with the three
models × three strategies on the 32 scenarios; commit runs/ablation/ (9 result
JSONs n=32 each, per-cell transcripts). Replace the table's failure-mode
placeholder with a written paragraph naming the surviving failure mode, citing ≥3
scenario IDs from transcripts. Compute gate_fail_rate = 1 − max(spec_adherence
over the 6 frontier cells) into runs/ablation/gate_report.json; escalation-needed
if < 0.20. Then STOP and present G1 (this is live Architecture Defense material);
P3–P4 may proceed while waiting, P5 may not.
Verify: `python3 scripts/check_ablation.py runs/ablation`

**P3 — Datagen pipeline + audited smoke batch.** `datagen/generate.py` producing
all five taxonomy types (support-deleted twins by sentence deletion from generated
answerable examples), rejection filter per the verifier contract,
`datagen/audit.py` (kept/rejected counts per reason, label balance, leakage).
Filter unit tests with canned teacher outputs covering accept + every reject
reason. Commit data/train/smoke_v0.jsonl (~64 kept, balance ±5pp, zero leakage).
Verify: `python3 -m pytest tests/test_datagen.py -q && python3 datagen/audit.py data/train/smoke_v0.jsonl --max-imbalance 0.05`

**P4 — Training pipeline + full smoke loop.** `training/modal_app.py`: Unsloth
QLoRA per configs/train.json, `--smoke` applies overrides, returns adapter +
merged model + train_log.json. Run the smoke loop end to end: smoke_v0 → smoke
train → `eval.py` against the smoke checkpoint (numbers irrelevant); artifacts
under runs/smoke/. This closes the project doc's loop-demo requirement before any
real spend.
Verify: `python3 scripts/check_loop.py runs/smoke`

**P5 — Dataset v1 + real run + publish.** Generate data/train/v1.jsonl (~1024
kept, audit green). Train with configs/train.json untouched; log under
runs/train_v1/. Push merged model + adapter to public HF repo under HF_USERNAME;
append `tuned` repo@commit to configs/models.json; confirm public visibility
(browser, read-only). Requires G1 resolved.
Verify: `python3 datagen/audit.py data/train/v1.jsonl --max-imbalance 0.05 && python3 scripts/push_model.py --check configs/models.json`

**P6 — Base-vs-tuned + MVP report. [GATE G2]** Run
`eval.py --model hf:<tuned>@<commit> --compare-with hf:Qwen/Qwen3-1.7B@<rev>
--strategy zero_shot --eval-set data/eval/scenarios_v1.jsonl`; commit results,
delta table, judge transcripts under runs/mvp/. Write docs/MVP_REPORT.md: both
tables, HF repo + commit hash, eval-code git commit hash, rubric hash, the exact
one-command repro line, paths to every transcripts file. Mirror the repro line +
env prerequisites in README. Then STOP and present G2 — submission is manual.
Verify: `python3 scripts/check_mvp.py`

Checker scripts (`scripts/ping_backends.py`, `check_ablation.py`, `check_loop.py`,
`check_mvp.py`, `push_model.py --check`) are part of their phase's work: each
validates committed artifacts and exits non-zero on failure, so a phase's done-ness
is re-checkable later without re-billing APIs or GPUs.

## Decisions (do not "improve" these away)

1. Qwen3-**1.7B**, not 4B — the Data-Efficiency phase later multiplies training
   cost by 4+ checkpoints; 4B is a fallback only if tuned-1.7B underfits at G2.
2. **Modal** for GPU — scriptable end to end, per-second billing. Don't switch.
3. Ablation families OpenAI + Google, judge Anthropic — fixed role split.
4. Deployment prompt in training data — see contract; measuring the behavior, not
   prompt transfer.
5. No CI service — the pytest suite run locally per phase is the regression gate.
6. transformers-based inference for eval — no serving stack for 32-scenario sets.

## Non-goals

Data-Efficiency curve, DPO, adversarial stretch evals, demo video, Brainlift
content, staff held-out handling, scenario edits (only G1 can direct escalation),
Qwen3-4B, portal submission, and any documentation beyond the policy above.

## Cost & safety rails

Cache teacher responses; smoke before real GPU spend; single GPU; ask before any
single step expected to exceed ~$25 compute. `.env` never committed. Browser
strictly read-only (model-ID verification, HF visibility, Modal status) — logins,
token creation, and submissions are the human's.
