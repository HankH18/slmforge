# DPO Preference Pairs (Stretch Ladder, rung 1)

**Headline: 1211 verified preference pairs over 451 prompts, and 100 % of them teach exactly one thing — do not use the banned letter. This dataset cannot move the two held-out constraint types, which is where the tuned model is already worse than base. No DPO model has been trained from it.**

`data/train/v3.dpo_pairs.jsonl` — 1211 rows, 2 634 363 bytes, sha256 `f13d6821d3a45369…83d6efe`
`data/train/v3.dpo_pairs.manifest.json` — every number below, derived at build time by `datagen/dpo_pairs.py`

---

## 1. What it is

The generator (`datagen/generate.py`) retries a rejected draft up to `--attempts` times. When a later attempt finally satisfies every constraint, the earlier fluent-but-violating drafts are written to `data/train/v3.dpo_negatives.jsonl` with the winning draft back-filled as `chosen`. That sidecar is a **candidate pool**, not a dataset: 2266 rows, of which 1055 have no positive side at all.

`datagen/dpo_pairs.py` turns the pool into a dataset by re-deriving every claim:

| Source field | Treatment |
|---|---|
| `violated` | **Ignored.** Both texts are re-run through `constraints.checkers.check` against the constraints resolved from `data/scenarios/constraint_instances.jsonl`. |
| `prompt` | **Ignored as an input.** Re-rendered from the stored scenario through `scenarios.render.render_user_message` — the function `configs/train.json:data.prompt_renderer` names — and asserted byte-identical to the stored string. |

A row becomes a pair **iff** `chosen` passes every constraint, `rejected` fails at least one, and the two texts differ. **0 rows failed the prompt re-render** and 0 failed on orientation; the entire 1055-row loss is `chosen_absent`.

| Drop reason | Rows |
|---|---|
| `non_datagen_scenario_id` | 0 |
| `chosen_absent` | 1055 |
| `prompt_render_mismatch` | 0 |
| `eval_corpus_prompt_collision` | 0 |
| `texts_identical` | 0 |
| `chosen_violates_a_constraint` | 0 |
| `rejected_violates_nothing` | 0 |
| **kept** | **1211** |

Composition: 451 distinct prompts, 1 to 9 pairs each (176 prompts contribute one pair, 5 contribute nine — the same `chosen` against several distinct rejected drafts). `k` distribution: 120 pairs at k=2, 344 at k=3, 747 at k=4.

The orientation guarantee is a **runtime chokepoint, not a type-level one**: every emitted row is built by `VerifiedPair`, whose only constructor re-runs the checkers and raises. That binds every path inside the builder; it does not police the JSONL after it is written, which is why `tests/test_dpo_pairs.py` re-verifies the shipped file independently with the same checkers.

## 2. Schema

TRL's **conversational** form, one JSON object per line:

```json
{"prompt":   [{"role": "user",      "content": "..."}],
 "chosen":   [{"role": "assistant", "content": "..."}],
 "rejected": [{"role": "assistant", "content": "..."}],
 "scenario_id": "dg-00002",
 "constraints": ["c-chr-008", "c-fmt-002", "c-lex-043"],
 "violated_constraint_ids": ["c-chr-008"],
 "violated_types": ["banned_letter"],
 "k": 3}
```

`DPOTrainer` reads `prompt`/`chosen`/`rejected` and ignores the rest; the trailing keys exist so an auditor can re-verify a row without joining back to the sidecar.

**Why conversational and not three plain strings.** SFT feeds the model `tokenizer.apply_chat_template(messages[:1], add_generation_prompt=True, enable_thinking=False)`, and the eval generator makes the same call — the chat scaffolding is part of the trained prompt. TRL applies the chat template for conversational rows and does *not* apply it for the plain-string form, so the plain form would hand the trainer bare prose with no `<|im_start|>` scaffolding: a prompt distribution this checkpoint has never seen.

**The chat-template mismatch, and where it is closed.** The Qwen3 template emits an
empty `<think>\n\n</think>` block after the assistant header when
`enable_thinking=False`, which is what `configs/train.json:data.enable_thinking`
pins for SFT. Verified directly against `runs/train_v2/merged`'s tokenizer — the two
renderings differ exactly there:

```
enable_thinking=True   ... <|im_end|>\n<|im_start|>assistant\n
enable_thinking=False  ... <|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
```

This file stores conversational rows and cannot itself force a consumer's template
kwargs, so the obligation belongs to whoever trains on it. **The shipped trainer
discharges it:** `training/dpo_local.py:build_dataset` applies the template itself
with `enable_thinking` read from `configs/train.json`, and hands TRL the
*already-templated plain-string* form — which TRL does not re-template — so the DPO
prompt is byte-identical to the SFT and eval prompt. A different consumer feeding
these conversational rows to TRL directly must pass the same kwarg, or it trains on
a prompt distribution this checkpoint has never seen.

---

## 3. Disclosure 1 — the monoculture

**Every violation in every pair is `banned_letter`. 1211 of 1211, fraction 1.0000. Not a sample: re-derived on all rows by the shipped checkers, twice (builder and test).**

The prompts are not monolithic — the 1211 pairs carry 826 `banned_words`, 790 `banned_phrases`, 734 `no_quotation_marks` and 699 `no_lists` constraints alongside their 1211 `banned_letter` constraints. But the *rejected* text never breaks any of them. Six of the seven registered constraint types are never violated by any rejected draft:

`banned_phrases`, `banned_punctuation`, `banned_words`, `no_lists`, `no_quotation_marks`, `sentence_initial_letter`.

**Consequence, stated plainly.** The preference signal is one-dimensional. DPO trained on this file can move the `banned_letter` axis and nothing else. The two held-out types — `banned_punctuation` and `sentence_initial_letter`, derived from the instance library as the types with no train-split instance — appear in **no** pair's prompt and are violated by **no** rejected text. They are also exactly where the tuned checkpoint already scores below base. If a DPO run improves them, this data does not explain it.

Within the one axis it *can* move, coverage of the graded letter mix is complete:

| Banned letter | Pair scenarios | Eval-corpus scenarios |
|---|---|---|
| a | 134 | 31 |
| e | 115 | 35 |
| i | 60 | 13 |
| o | 51 | 14 |
| r | 35 | — |
| u | 26 | — |
| s | 21 | 12 |
| l | 4 | — |
| m | 3 | — |
| p | 2 | — |

All 105 of the eval corpus's `banned_letter` scenarios ban a letter that at least one pair scenario also bans; `eval_letters_with_no_pair_scenario` is empty. This is the v1 failure (`docs/BRAINLIFT.md` §4: 60.9 % of letter rows banned a letter the eval corpus never uses) not recurring.

## 4. Disclosure 2 — the attrition

**1055 of 2266 source rows (46.56 %) have `chosen = None`: the teacher never produced a compliant draft for that prompt. 165 of 616 scenarios (26.79 %) contribute no pair at all.**

The pairs that exist are therefore drawn from the prompts the teacher found *easier*. That is the same selection effect this project already documented for v1's rejection sampling — `docs/BRAINLIFT.md` §5, "rejection sampling for quality selects against difficulty."

Quantified. Every scenario in the sidecar carries exactly one `banned_letter` constraint, so "difficulty" has a cheap proxy: how common that letter is in English (Lewand's relative-frequency table). Comparing the 451 scenarios that produced a pair against the 165 that did not:

| | Scenarios | Mean English frequency of the banned letter |
|---|---|---|
| Produced at least one pair | 451 | **8.42 %** |
| Produced no pair | 165 | **9.59 %** |

Delta **−1.17 percentage points**; point-biserial correlation between letter frequency and "produced a pair" **r = −0.182** over all 616 scenarios. Per letter, the pair rate falls as the letter gets commoner:

| Letter | English freq | With a pair | Without | Pair rate |
|---|---|---|---|---|
| e | 12.70 % | 115 | 65 | 0.639 |
| a | 8.17 % | 134 | 65 | 0.673 |
| o | 7.51 % | 51 | 11 | 0.823 |
| i | 6.97 % | 60 | 12 | 0.833 |
| s | 6.33 % | 21 | 0 | 1.000 |
| r | 5.99 % | 35 | 8 | 0.814 |
| l | 4.03 % | 4 | 1 | 0.800 |
| u | 2.76 % | 26 | 3 | 0.897 |
| m | 2.41 % | 3 | 0 | 1.000 |
| p | 1.93 % | 2 | 0 | 1.000 |

The two hardest cells, `e` and `a`, are the two with the lowest pair rate — and they are also the eval corpus's two largest cells (35 and 31 of 105). The pairs cover those letters, but they cover the *easier instances of them*: for `e` and `a`, roughly a third of the prompts contributed no preference signal because the teacher could not solve them.

**Consequence.** A DPO gain measured on this data is a gain on the easier half of the distribution. The hardest prompts are absent from the training signal entirely, and the eval set does not share that bias.

## 5. Disclosure 3 — train/eval separation

Measured, not assumed:

| Check | Result |
|---|---|
| Pair scenario ids matching `^dg-\d+$` | 451 / 451 |
| Pair prompts (rendered) also in `data/scenarios/corpus.jsonl` | **0** |
| Pair scenario ids also in the eval corpus | **0** |
| Held-out-split constraint instances used by any pair | **0** |

The comparison renders all 147 eval scenarios through the same `render_user_message` and compares the rendered user messages, not the raw task strings — a pair whose task text differed only in the requirement block would still be caught. A non-zero count on either axis is a **blocker**, not a caveat: `datagen.dpo_pairs.build` raises `EvalLeak` and the CLI exits 2, because an SFT→DPO delta measured on prompts the DPO model trained on is not a delta.

---

## 6. Rebuilding

```
python -m datagen.dpo_pairs                 # writes both files
python -m datagen.dpo_pairs --check         # verifies and prints, writes nothing
python -m datagen.dpo_pairs --source data/train/v2.dpo_negatives.jsonl
```

The build is deterministic: same sidecar plus same constraint library gives byte-identical output and a byte-identical manifest. Neither file carries a timestamp — `tests/test_dpo_pairs.py::test_the_build_is_deterministic_and_reproduces_the_shipped_bytes` rebuilds from `data/train/v3.dpo_negatives.jsonl` (sha256 `7d02d7af238309e2…fee3e7`) and asserts the result equals the shipped bytes.

Verification, all of it re-derived from the frozen inputs rather than read off the manifest:

```
.venv/bin/python -m pytest tests/test_dpo_pairs.py -q
```

## 7. Results

**No DPO model has been trained from this dataset.** This document ships the data and its disclosures only; the training run, the SFT-vs-DPO comparison and the delta the spec asks for are not done.

<!-- RESULTS: intentionally empty until a DPO run exists. Fill in below. -->

| Checkpoint | spec_adherence | robustness | banned_letter pass rate | held-out pass rate |
|---|---|---|---|---|
| SFT (`runs/train_v2/merged`) | — | — | — | — |
| SFT + DPO | — | — | — | — |

Anyone filling this table should read §3 and §4 first. The prior this dataset supports is: movement on `banned_letter`, nothing on `banned_punctuation` or `sentence_initial_letter`, and a gain concentrated on the easier letters. A result outside that shape needs an explanation that does not come from this file.
