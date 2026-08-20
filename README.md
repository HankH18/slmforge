# Grounded Abstainer

Instilling one falsifiable behavior into a small open model (Qwen3-1.7B), and
proving with numbers that the tuned model beats the prompted base model at it.

## Behavior Spec (frozen)

> Given a question and a context passage, the model answers only if the passage
> contains the answer, outputting exactly one `Answer: <answer>` line followed by
> one or more `Citation: "<span>"` lines where every cited span is a verbatim
> substring of the passage and the spans jointly entail the answer; otherwise its
> entire output is exactly `Not answerable from the provided context.`

A stranger can mark any output pass/fail against that sentence. It is
simultaneously the data-generation rubric and the eval criterion.

**Pass rule.** `gold=unanswerable` passes iff the output is exactly the refusal
string. `gold=answerable` passes iff the format parses, every cited span is a
verbatim substring of the passage (both programmatic), and an LLM judge answers
YES to a single binary entailment question. The judge is consulted **only after**
the programmatic checks pass — it can flip a would-be pass to fail, never rescue
a programmatic failure.

## Setup

Python **3.11+** is required: the frozen `eval.py` uses PEP 604 (`list | None`)
annotations that are evaluated at import time, so it will not load on 3.9.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then fill in the five keys
```

| Variable | Used for |
|---|---|
| `ANTHROPIC_API_KEY` | LLM-as-judge (third model family — nothing scores itself) |
| `OPENAI_API_KEY` | Ablation family A + teacher model for distillation |
| `GOOGLE_API_KEY` | Ablation family B |
| `HF_TOKEN` | Publishing the tuned model (needs **write** scope) |
| `HF_USERNAME` | Target namespace for the model repo |

`.env` is gitignored and must never be committed.

## Eval harness

```bash
# base model, zero-shot
python eval.py --model hf:Qwen/Qwen3-1.7B --eval-set data/eval/scenarios_v1.jsonl

# base vs. tuned, the MVP comparison
python eval.py --model hf:<tuned-repo>@<commit> \
               --compare-with hf:Qwen/Qwen3-1.7B@<revision> \
               --strategy zero_shot --eval-set data/eval/scenarios_v1.jsonl \
               --out runs/mvp

# any held-out scenario set, same harness and same rubric
python eval.py --model hf:<tuned-repo>@<commit> --eval-set <path-to-scenarios.jsonl>
```

Every run writes per-example judge transcripts (`*.transcripts.jsonl`, score +
reasoning) alongside the aggregate `*.results.json`, both stamped with the rubric
hash they were scored under.

Model specs: `hf:<repo>[@revision]`, `openai:<id>`, `google:<id>`,
`anthropic:<id>`, and `mock:<behavior>` test doubles for network-free tests.

## Metrics

`spec_adherence` (pass rate), `robustness` (pass rate on the adversarial subset),
`false_answer_rate` = P(answered | gold unanswerable), `false_refusal_rate` =
P(refused | gold answerable), plus a per-failure-mode breakdown and a gold ×
predicted confusion matrix.

**The abstention floor is 0.594.** Refusing every scenario scores 19/32 on this
eval set for free. A tuned model has to beat that number, not zero.

## Eval set

`data/eval/scenarios_v1.jsonl` — 32 scenarios (the project doc requires ≥30 per
model × strategy cell): 13 answerable, 19 unanswerable, 16 adversarial. Five
types: `clean_answerable`, `support_deleted` (answerable twin with the supporting
sentence removed), `distractor`, `parametric_trap` (answerable from world
knowledge but not from the passage), `near_miss`.

## Tests

```bash
python -m pytest tests -q
```

Network-free — mock generators and mock judges throughout. The suite pins
`rubric_hash() == c0adb876c3e6c4c9`; if that assertion fails, scoring drifted and
every previously reported number is invalid.

## Repo layout

| Path | What |
|---|---|
| `eval.py`, `run_ablation.py` | Frozen harness + Prompt-Ceiling Ablation driver |
| `data/eval/` | Frozen eval scenarios and few-shot exemplars |
| `configs/` | `models.json` (pinned model IDs), `train.json` (pinned hyperparameters) |
| `datagen/` | Distillation pipeline + rejection filter + dataset audit |
| `training/` | Modal QLoRA app |
| `scripts/` | Per-phase checkers that re-verify committed artifacts without re-billing |
| `runs/` | Ablation, smoke, and MVP run artifacts |
| `docs/` | Project spec (source of truth), build plan, MVP report |

Phases and their verify commands: `docs/BUILD_PLAN.md`. Grading requirements:
`docs/project_spec.pdf`.
