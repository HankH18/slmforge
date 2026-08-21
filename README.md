# Forbidden-Constraint Holder

Instilling one falsifiable behavior into a small open model (Qwen3-1.7B) and
proving with numbers that the tuned model beats the prompted base model at it.

## Start here

| Document | What it answers |
|---|---|
| [`docs/BRAINLIFT.md`](docs/BRAINLIFT.md) | The behavior thesis, and whether data → behavior held. Read this first. |
| [`docs/FINAL_REPORT.md`](docs/FINAL_REPORT.md) | The final base-vs-tuned table, measured against the checkpoint pulled from the Hub. |
| [`docs/EFFICIENCY_CURVE.md`](docs/EFFICIENCY_CURVE.md) | Data-Efficiency Curve: performance vs dataset size, and what the minimum viable N can and cannot be. |
| [`docs/ABLATION_G1.md`](docs/ABLATION_G1.md) | Prompt-Ceiling Ablation: 2 frontier families × 3 strategies × 147 scenarios, and the failure mode that survives the best prompt. |
| [`docs/CURVE_PREREGISTRATION.md`](docs/CURVE_PREREGISTRATION.md) | Predictions for the upward extension of the curve, committed before that checkpoint was trained. |
| [`docs/MVP_REPORT.md`](docs/MVP_REPORT.md) | The frozen MVP record. Pins the v1 checkpoint; still re-derives. |
| [`docs/PIVOT_SPEC.md`](docs/PIVOT_SPEC.md) | The Behavior Spec and the numbered requirements. |

Published artifacts, both public and both pinned by commit hash in
`configs/models.json`:

| | |
|---|---|
| Model | [`HankH18/qwen3-1.7b-forbidden-constraint-holder`](https://huggingface.co/HankH18/qwen3-1.7b-forbidden-constraint-holder) — merged fp16 at the repo root, LoRA adapter under `adapter/` |
| Dataset | [`HankH18/forbidden-constraint-holder-sft`](https://huggingface.co/datasets/HankH18/forbidden-constraint-holder-sft) — the training rows, **plus every rejected teacher attempt** and the generation ledger |

`configs/models.json` carries two model pins on purpose: `tuned` is the current
published checkpoint, `tuned_mvp` is the one `docs/MVP_REPORT.md` documents. Both
are commits of the same repo, so either still pulls, and `scripts/check_mvp.py`
gates each package against its own pin.

Commands worth knowing before the rest of this file:

```bash
# re-derive every number in the final base-vs-tuned report from the committed transcripts
python scripts/check_mvp.py --pin-key tuned --report docs/FINAL_REPORT.md --run-dir runs/final

# the MVP package, still re-deriving against the checkpoint it was submitted with
python scripts/check_mvp.py --pin-key tuned_mvp --report docs/MVP_REPORT.md --run-dir runs/mvp

# re-derive the whole data-efficiency curve from the committed transcripts
python scripts/curve_report.py --check --markdown

# run your own prompt and your own forbidden constraint against base and tuned, side by side
python scripts/demo.py --prompt "Write a short notice about a library reopening." \
                       --constraint banned_punctuation:, --constraint banned_words:however
```

**Before running the harness on a scenario file it has never seen** — the staff
held-out set, or anything else — check it for free first:

```bash
python scripts/preflight_evalset.py <scenarios.jsonl> [constraints.jsonl]
```

It renders every row, resolves every constraint id, runs each constraint through
the pinned checker and prints what the set contains. `eval.py` performs the same
validation before it generates a token, because three of the fields it needs used
to be read only *after* every generation and judge call had been paid for.

## Behavior Spec (the graded spec)

> Given a task prompt and a set of 2–4 hard constraints that always includes at
> least one negative constraint, the model completes the task while violating
> zero constraints. Pass = every constraint passes its programmatic checker AND
> the response genuinely addresses the task (judge task-quality ≥ 3).

A stranger can mark any output pass/fail against that sentence.

**Constraint pass/fail is 100% programmatic.** `constraints/checkers.py` holds one
pure, deterministic checker per constraint type — no model call, no network. The
LLM judge scores **task-quality only** and can never override a checker; it exists
solely to block the degenerate strategy of satisfying every constraint with
garbage text. `spec_pass = all(checkers pass) AND quality_score >= 3`.

## Why this target

Single-constraint instruction following is largely solved by current frontier
models. The documented surviving gap is **composition under distractor pressure**
— holding several constraints at once, at least one of them negative, while the
task's own subject matter pulls toward violating them. Constraint count `k` varies
difficulty *within* the one behavior; it is not a second target.

## Setup

Python **3.11+** is required.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then fill in the five keys
```

**The harness reads `os.environ` directly and loads no dotenv.** Export before any
live run, or every provider call fails on a missing key:

```bash
set -a && . ./.env && set +a
```

| Variable | Used for |
|---|---|
| `ANTHROPIC_API_KEY` | Task-quality judge (a third model family — nothing scores itself) |
| `OPENAI_API_KEY` | Ablation family A, and the distillation teacher |
| `GOOGLE_API_KEY` | Ablation family B |
| `HF_TOKEN` | Publishing the tuned model (needs **write** scope) |
| `HF_USERNAME` | Target namespace |

## Eval harness

```bash
# base model, zero-shot
python eval.py --model hf:Qwen/Qwen3-1.7B --eval-set data/scenarios/corpus.jsonl

# base vs. tuned — the FINAL comparison, against the published checkpoint
python eval.py --model hf:HankH18/qwen3-1.7b-forbidden-constraint-holder@c19712a72932b503f28fe063d1962acc9e8c3fb8 \
               --compare-with hf:Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
               --eval-set data/scenarios/corpus.jsonl --out runs/final

# base vs. tuned — the MVP comparison
python eval.py --model hf:HankH18/qwen3-1.7b-forbidden-constraint-holder@b97eb115de1f1837db60550361669950d158e95e \
               --compare-with hf:Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
               --eval-set data/scenarios/corpus.jsonl --out runs/mvp

# few-shot needs exemplars
python eval.py --model openai:<id> --strategy few_shot \
               --exemplars data/scenarios/fewshot_exemplars.jsonl \
               --eval-set data/scenarios/corpus.jsonl

# any unseen scenario set, same harness and same checkers
python eval.py --model hf:<repo>@<commit> --eval-set <path-to-scenarios.jsonl>

# the staff held-out set, with their own constraint library — preflight it first
python eval.py --model hf:HankH18/qwen3-1.7b-forbidden-constraint-holder@c19712a72932b503f28fe063d1962acc9e8c3fb8 \
               --eval-set <path-to-staff-set.jsonl> \
               --constraints <path-to-their-constraints.jsonl> --out <your-run-dir>
```

Note the `hf:` provider prefix. The project doc writes the repro line as
`eval.py --model <hf-repo-id>`; this harness requires `provider:model`, and the
commands above are the ones that actually run.

`google:gemini-3.1-pro` is rate-limited to **25 requests/minute** — pass
`--workers 2` for it or the run 429s partway through.

Every run writes per-example transcripts (`*.transcripts.jsonl`) alongside the
aggregate (`*.results.json`).

Model specs: `hf:<repo>[@revision]`, `openai:<id>`, `google:<id>`,
`anthropic:<id>`, and `mock:<behavior>` (`compliant`, `violator`, `empty`,
`offtask`) for network-free tests.

## Metrics

`spec_adherence` (fraction of scenarios where all checkers pass and quality ≥ 3),
`robustness` (the same over the `pressure=adversarial` subset), plus breakdowns by
constraint type, by `k`, by pressure, and by split. The per-`k` breakdown is the
composition evidence: whether compliance degrades as `k` rises 2 → 3 → 4.

## Constraint registry

Seven types across four families. `split` is assigned at the **type** level.

| type | family | split |
|---|---|---|
| `banned_words`, `banned_phrases` | lexical | train |
| `banned_letter` | character | train |
| `banned_punctuation` | character | **held-out** |
| `sentence_initial_letter` | structural | **held-out** |
| `no_lists`, `no_quotation_marks` | format_negative | train |

Held-out types never appear in training data. The structural family is entirely
unseen in training, so the held-out split is a genuine test of generalization to
constraint types the model never trained on — the local proxy for the staff
held-out set.

Checkers are validated against `data/fixtures/checker_golden.jsonl`, hand-labeled
by authors who could not read the implementation.

## Scenario corpus

`data/scenarios/corpus.jsonl` — task + constraint set (`k` ∈ 2,3,4, always ≥1
negative) + pressure level + split. Adversarial scenarios carry **both** a
distractor (the task's subject matter pulls toward the banned token) and an
embedded nudge arguing the rules are optional. Every scenario ships with a
constructive proof that a compliant response exists; the generator quarantines
anything it cannot prove satisfiable, because an impossible scenario scores zero
for every model forever and silently fakes a ceiling.

Regenerate and verify the artifacts match a fresh build:

```bash
python -m scenarios.build --check
```

## The prompt renderer

`scenarios/render.py` turns a scenario into the user message. **It is the one
definition, imported by both the eval harness and the training data builder.** If
they ever render differently the tuned model is measured under a prompt it never
saw, and every reported number is meaningless. `zero_shot` carries no system
message, matching the training example schema exactly.

## Tests

```bash
python -m pytest tests -q                      # the live suite
python -m pytest archive/abstainer/tests -q    # the retired target, still reproducible
```

Network-free — mock generators and mock judges throughout.

## Repo layout

| Path | What |
|---|---|
| `eval.py` | The graded one-command entry point; CLI plumbing and strategy prompts only |
| `harness.py`, `judge.py` | Scoring, metrics, transcripts; task-quality judge |
| `constraints/` | Registry and the deterministic checkers — the ground-truth layer |
| `scenarios/` | Corpus generator, reachability prover, the shared prompt renderer |
| `training/` | Modal QLoRA app |
| `scripts/` | the gates and the publishing tools — see below |
| `data/scenarios/` | Corpus, constraint instances, few-shot exemplars, smoke set |
| `data/fixtures/` | Golden checker fixtures |
| `runs/` | Ablation, smoke and MVP run artifacts |
| `archive/abstainer/` | The retired first target — see below |
| `docs/` | Project spec (source of truth), pivot spec/design/tasks |

### `scripts/`

| Script | What it does |
|---|---|
| `check_mvp.py` | Gates a submission package: re-derives every number from the transcripts and requires the report's generated block back byte for byte. `--pin-key` / `--report` / `--run-dir` select which package. |
| `curve_report.py` | Re-derives the whole data-efficiency curve from the committed transcripts. `--check` fails on a hand-edited `curve.json`. |
| `compare_checkpoints.py` | Paired comparison of two eval cells — McNemar on `spec_pass`, sign test and bootstrap on banned-letter occurrences. Two checkpoints on one eval set are one sample measured twice. |
| `preflight_evalset.py` | Validates an unseen scenario file for free, before any model loads. |
| `gate_check.py` | The G1 prompt-ceiling ablation gate. |
| `check_loop.py` | The generate → train → eval loop checker. |
| `push_model.py` | Publishes the checkpoint to a public HF repo and pins the resulting commit. `--check` re-verifies the pin. |
| `push_dataset.py` | The same for the dataset, with a secret/PII scan that runs before anything is uploaded. |
| `model_card.py` | Renders the HF model card from the run artifacts, so it cannot drift from the run it describes. |
| `make_subset.py` | Deterministic nested subsets for the curve. |
| `demo.py` | Base-vs-tuned on your own prompt and your own constraints. |

## The abstainer archive

The first target was grounded abstention: answer only from a passage, with
verbatim citations, else refuse. It **failed its own Prompt-Ceiling Ablation** —
both frontier families scored 100% across all three prompting strategies, so the
"prompting has a ceiling" claim could not be supported and the target could not
clear the gate. `archive/abstainer/` holds its harness, scenarios and all nine
ablation cells, with a README covering how to reproduce those numbers. It is kept
because the ablation working as designed — sending us back to pick a harder
target — is part of this project's evidence, not an embarrassment to hide.
