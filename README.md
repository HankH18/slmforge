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

**Python 3.11+ is required.** The frozen `eval.py` uses PEP 604 (`list | None`)
annotations that are evaluated at import time. On the macOS system interpreter
(`python3` → 3.9) importing it fails outright:

```
TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'
```

so every command below must run under the 3.11 venv, not `python3`.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then fill in the five keys
python --version          # must print 3.11.x
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

`eval.py` and `run_ablation.py` are frozen: import them, never edit them.

```bash
# base model, zero-shot, judged by the mock judge (no API spend)
python eval.py --model hf:Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
               --eval-set data/eval/scenarios_v1.jsonl \
               --judge mock:yes --out /tmp/eval-smoke

# network-free smoke: the abstention floor, straight out of the harness
python eval.py --model mock:always_refuse --eval-set data/eval/scenarios_v1.jsonl \
               --judge mock:yes --out /tmp/eval-smoke
```

`runs/` is **not** gitignored — it holds graded artifacts. Point throwaway runs at
`/tmp` and keep `runs/<phase>/` for the ones the project doc grades.

Drop `--judge mock:yes` to score with the real Anthropic judge (`ANTHROPIC_API_KEY`).
Add `--compare-with hf:Qwen/Qwen3-1.7B@<revision>` for the base-vs-tuned delta table.

**The `--model` value needs a provider prefix.** The project spec writes the repro
line as `eval.py --model <hf-repo-id> --eval-set <path>`, but `make_generator`
splits on `:` and rejects a bare repo id:

```
$ python eval.py --model Qwen/Qwen3-1.7B --eval-set data/eval/scenarios_v1.jsonl
ValueError: model spec must be provider:model, got 'Qwen/Qwen3-1.7B'
```

The working form is `hf:Qwen/Qwen3-1.7B@<revision>`. Pin the revision — an
unpinned `hf:<repo>` runs whatever `main` points at today, and a bad revision
fails loudly (`OSError: <rev> is not a valid git identifier`). The pinned base
revision lives in `configs/models.json`.

Model specs: `hf:<repo>[@revision]`, `openai:<id>`, `google:<id>`,
`anthropic:<id>`, and `mock:<behavior>` test doubles for network-free runs.

Every run writes per-example judge transcripts (`*.transcripts.jsonl`, score +
reasoning) alongside the aggregate `*.results.json`, both stamped with the rubric
hash they were scored under. With no `--out`, output goes to `runs/<utc timestamp>/`.

## Held-out eval sets

At grading time the harness is pointed at a scenario file nobody here has seen.
`eval.py` is frozen and **validates nothing about its input**, which gives two
failure modes — one loud, one much worse:

- **Crash.** It subscripts `scenario["id"]`, `["gold_label"]`, `["question"]` and
  `["passage"]` directly, so a missing key is a `KeyError` partway through the run.
- **Silent misscore.** `score_example` branches on
  `if scenario["gold_label"] == "unanswerable":` with no else-validation, so
  **every other value falls through to the "gold answerable" branch**. A set that
  spells the label `not_answerable`, `Unanswerable`, `no_answer` or `false` is
  scored as if all of its unanswerable rows were answerable. Measured on the same
  refusing model over the same three rows, changing only that one word:

  | gold_label | spec_adherence | false_answer_rate | failures |
  |---|---|---|---|
  | `unanswerable` | 100.0% | 0.0% | – |
  | `not_answerable` | 0.0% | n/a | `false_refusal=3` |

  Exit code 0 both times. No warning, no traceback, a clean-looking table.

So preflight the file **before** scoring anything with it:

```bash
python scripts/check_eval_set.py <path-to-held-out.jsonl>
python scripts/check_eval_set.py <path-to-held-out.jsonl> --json   # machine-readable
```

Read-only — it never edits `eval.py` and never rewrites the input. Exit 0 means
the frozen harness will score the file correctly; exit 1 lists, per line and per
scenario id, anything that would crash the run or bend the numbers:

| Check | Why it matters |
|---|---|
| missing `id` / `gold_label` / `question` / `passage` | subscripted directly → `KeyError` mid-run |
| `gold_label` outside `{answerable, unanswerable}` | **silently scored as answerable** |
| missing / empty / null `gold_spans` on an answerable row | `mock:oracle` sanity double raises `KeyError`/`IndexError` |
| a gold span that is not a verbatim substring of its passage | row is unpassable; even an exact citation scores `nonverbatim_citation` |
| `"` in a passage | the score then depends on the model's quoting convention: raw quotes pass, escaped (`\"`) and curly (`“ ”`) renderings of the same span fail the verbatim check |
| missing or non-boolean `adversarial` | read via `.get()` and coerced with `bool()`, so `robustness` silently reports `n/a` — and the string `"false"` counts as **true** |
| duplicate `id`s | transcripts key on `scenario_id`; duplicates are indistinguishable afterwards |

Warnings (fewer than 30 scenarios, no adversarial rows, `gold_spans` set on an
unanswerable row) are printed but do not fail the check.

Then run the unseen set through the same harness and the same rubric:

```bash
HELDOUT=path/to/staff_scenarios.jsonl

python scripts/check_eval_set.py "$HELDOUT" \
  && python eval.py --model hf:<tuned-repo>@<commit> \
                    --eval-set "$HELDOUT" \
                    --out runs/heldout
```

The `&&` is the point: a failed preflight stops the run instead of producing a
plausible-looking number.

If the preflight fails, the numbers from that file are not evidence of anything —
report the defect rather than the score.

## Metrics

`spec_adherence` (pass rate), `robustness` (pass rate on the adversarial subset),
`false_answer_rate` = P(answered | gold unanswerable), `false_refusal_rate` =
P(refused | gold answerable), plus a per-failure-mode breakdown and a gold ×
predicted confusion matrix.

**The abstention floor is 0.594.** Refusing every scenario scores 19/32 on this
eval set for free (`mock:always_refuse` above prints exactly that). A tuned model
has to beat that number, not zero.

## Eval set

`data/eval/scenarios_v1.jsonl` — 32 scenarios (the project doc requires ≥30 per
model × strategy cell): 13 answerable, 19 unanswerable, 16 adversarial. Five
types: `clean_answerable`, `support_deleted` (answerable twin with the supporting
sentence removed), `distractor`, `parametric_trap` (answerable from world
knowledge but not from the passage), `near_miss`.
`data/eval/fewshot_exemplars.jsonl` — 6 exemplars for `--strategy few_shot`.

Both are frozen inputs. Per-row schema, and what happens if a row deviates from
it: **Held-out eval sets** above.

## Training data

The training sets under `data/train/` are produced by **P3**; they are not part of
the skeleton. The pipeline that writes them lives in `datagen/` (generation,
rejection filter, audit) and is driven by the commands in `docs/BUILD_PLAN.md` § P3.

The contract it implements: every row carries `eval.SYSTEM_CORE` verbatim as its
system message and `eval.user_msg()` as its user message, so the training
condition is byte-identical to the `zero_shot` eval condition. An example is kept
only if `eval.parse_output` (imported from the frozen harness, never
reimplemented) returns the class its gold label demands, every cited span is a
verbatim substring of the passage, the passage holds no `"` character, it shares
no 6-gram with `data/eval/*.jsonl`, and it is not a duplicate. Every rejection is
recorded under a machine-readable reason code. `support_deleted` twins are built
by deleting the supporting sentence from a generated answerable example, and
survive only if the teacher then refuses on the twin. Teacher replies are cached
in `datagen/.cache/` (gitignored), so reruns do not re-bill.

## Tests

```bash
python -m pytest tests -q
```

Network-free — mock generators and mock judges throughout. The suite pins
`rubric_hash() == c0adb876c3e6c4c9`; if that assertion fails, scoring drifted and
every previously reported number is invalid.

## Repo layout

| Path | What | Status |
|---|---|---|
| `eval.py`, `run_ablation.py` | Frozen harness + Prompt-Ceiling Ablation driver | present |
| `data/eval/` | Frozen eval scenarios (32) and few-shot exemplars (6) | present |
| `configs/models.json` | Pinned model IDs + base revision, verified against provider docs | present (P1); gains a `tuned` entry in P5 |
| `configs/train.json` | Pinned training hyperparameters | present |
| `datagen/` | Distillation pipeline + rejection filter + dataset audit | present |
| `training/` | Modal QLoRA app | present |
| `scripts/` | `check_eval_set.py` (held-out preflight) plus one checker per phase, each re-verifying committed artifacts without re-billing APIs or GPUs | present |
| `tests/` | The regression gate — network-free pytest suite | present |
| `docs/BUILD_PLAN.md`, `docs/project_spec.pdf` | Build plan (contracts, phases) and the graded spec | present |
| `data/train/` | Distilled training sets | produced by P3 |
| `runs/ablation/` | 9 model × strategy result JSONs, transcripts, ablation table | produced by P2 |
| `runs/smoke/`, `runs/train_v1/`, `runs/mvp/` | Smoke loop, real training run, base-vs-tuned MVP comparison | produced by P4 / P5 / P6 |
| `docs/MVP_REPORT.md` | Both tables, HF repo + commit, rubric hash, one-command repro | produced by P6 |

Run artifacts are written by the phase that produces them; the skeleton does not
ship them. Phases and their verify commands: `docs/BUILD_PLAN.md`. Grading
requirements: `docs/project_spec.pdf`.
