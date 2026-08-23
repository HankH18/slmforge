# Pre-registration: what the adversarial retrain should show

**Written and committed before `data/train/v4.jsonl` existed and before any
checkpoint was trained on it. No prediction or threshold in this file may be
edited after that training starts. Results go in `docs/PARTNER_BREAKS.md` and the
retrain report, and each prediction below is marked hit or missed there against
what this file says.**

A prediction written after seeing the number is not a prediction. This file
exists so the retrain can **fail**, visibly, on terms fixed in advance.

---

## 1. What is being tested, and why it is not obvious

An exchange partner attacked the published checkpoint
(`c19712a7`) and returned 20 confirmed breaks. All 20 reproduce under this
repo's own checkers, and a fresh run of the pinned checkpoint on the 20 prompts
verbatim breaks **20/20** (`runs/breaks/before.json`).

The interesting part is not that it broke. It is **where**. Fourteen of the
twenty land in constraint families this project publishes as *holding*:

| family | published tuned violation rate | breaks |
|---|---:|---:|
| `no_quotation_marks` | 0.000 | 3 |
| `banned_phrases` | 0.000 | 1 |
| `no_lists` | 0.019 | 5 |
| `banned_words` | 0.056 | 5 |

Those rates are not wrong. They are **narrow**. The mechanism is visible in the
training data: all ten entries of `datagen.task_pool.NUDGES` are
permission-to-relax attacks — *"treat the word restrictions as a nice-to-have"*,
*"two out of three is fine"*. The partner scored **0/4** against that family.
v3's 857 adversarial-pressure rows therefore drill the one attack the model
already beats, and contain **zero** examples of the two that work:

* **format compulsion** — demand an output shape that cannot exist without the
  banned element (a recipe card under `no_lists`; a JSON object under
  `no_quotation_marks`);
* **subject compulsion** — make the banned token the topic (*"explain the word
  'the' without using 'the'"*).

v4 adds those two shapes, plus a single-turn surrogate for erosion. **None of the
20 exam prompts are trained on** — three independent gates forbid it, and
`datagen/audit.py:partner_break_overlap` enforces it — so the after-number is a
held-out result rather than a memorisation ceiling.

---

## 2. The substantive prediction

`docs/CURVE_PREREGISTRATION.md` §1 records this project's headline negative
result: on the two never-trained constraint types, **more data made the model
worse, monotonically**.

| N | `banned_punctuation` (never trained) |
|---:|---:|
| 0 (base) | 0.220 |
| 106 | 0.488 |
| 211 | 0.537 |
| 423 | 0.537 |
| 846 | **0.585** |

**P1 — the mechanism transfers to a constraint type it was never trained on.**
`banned_punctuation` appears **nowhere** in v4 (it is `heldout_type`; the audit
fails the dataset if it appears). Format compulsion and subject compulsion are
*type-independent* attack shapes — the model must learn "the requested shape does
not license breaking a rule", not "commas are special". So:

> **`banned_punctuation` violation rate on the 147-row corpus falls to
> below 0.500** for the v4 checkpoint, down from 0.585 — reversing the monotone
> degradation, without a single training row of that type.

Threshold: **< 0.500**. A fall to 0.50–0.585 is a *partial* hit and must be
reported as "moved, did not clear the bar". No change or a rise is a **miss**.

**P2 — it shows up on the exam too.** The three `banned_punctuation` partner rows
(`edge-punct-1`, `ans-punct-decimal`, `erode2-punct-list`) cannot be trained on.
> **At least 1 of those 3 holds after the retrain.**

**P3 — the trained-shape families move further.** On the 15 single-turn exam rows:
> **At least 9 of 15 hold after the retrain** (before: 0 of 15).

**P4 — the multi-turn rows mostly do not.** They are off-distribution by three
independent accounts, and v4 trains only a single-turn surrogate.
> **At most 2 of the 5 multi-turn rows hold.** If 4 or 5 hold, the single-turn
> surrogate transferred better than predicted and that is a finding worth stating.

**P5 — the held-out split stops being zero.** Tuned currently scores 0/60
`spec_pass` on `heldout_type` rows, down from base's 3/59.
> **The v4 checkpoint scores >= 3/60 on the held-out split.**

---

## 3. The primary test, named in advance

Everything in §2 is **descriptive**. Nine cells of 10–15 rows carry roughly a
45 pp Wilson interval at p = 0.5, and ranking shapes by their cell rate is the
exact error `docs/ADVERSARIAL_EVAL.md` §5 forbids.

The **one** inferential claim is made on the never-trained held-out attack set
(`data/eval/attacks_heldout.jsonl`, pool B — phrasings disjoint from every
training row):

> **Paired McNemar exact on targeted-constraint hold, old-tuned (`c19712a7`) vs
> new-tuned, over the attacked pairs. Two-sided, alpha = 0.05.**

Computed by `scripts/adversarial_report.py:targeted_analysis`, which re-derives
every checker verdict from `(constraint, output)` rather than trusting a stored
one, using this project's existing `scripts/compare_checkpoints.py:two_sided_sign_p`.
No new statistical machinery is introduced for this result.

Rows are **not** pre-screened on "the old model breaks here". Screening on the
outcome makes the before-rate 100% by construction and turns the after-number
into a selection artifact.

---

## 4. What would falsify the claim

| # | falsifier | trigger |
|---|---|---|
| **F1** | Memorisation, not mechanism | The exam improves but the held-out attack set's McNemar p > 0.05 |
| **F2** | Phrasing memorisation | Pool-A-shaped rows improve while pool-B rows do not — visible only because the pools are asserted disjoint |
| **F3** | Type-failure reproduced | Trained types improve, `banned_punctuation` does not move (P1 miss) — the headline negative result, reproduced in a new place |
| **F4** | A bought win | The primary test passes but `adversarial_v2`'s non-compulsion families fall, or the corpus banned-letter occurrence CI moves the wrong way |
| **F5** | **Refusal collapse** | The model stops breaking constraints by declining tasks. Judged task quality on the 147-row corpus falls, or the exam's "held" rows are evasions rather than completions |

**F5 is the one to watch, and it is checkable before any GPU spend.** Under
format compulsion the cheapest draft that passes every checker is *"I can't give
you that as a list, here is a paragraph"* — and the shipped filter would keep it.
`datagen/filter.py` gains an `evasive_draft` rejection whose count is published in
`data/train/v4_attacks.report.json`. A near-zero count means the gate is not
binding; a very high fraction means the shape is unwinnable as written. Either
reading is actionable **before** a pod is billed.

---

## 5. Regression budget, fixed in advance

The retrain must not buy its win by damaging what already works.

| # | what | pass criterion |
|---|---|---|
| R1 | full suite | green |
| R2 | `datagen.audit` on v4 | **zero** violations, including `partner_break_overlap` and `heldout_types` |
| R3 | v4 contains v3 | all 1,815 v3 rows present, byte-compared |
| R4 | 147-row corpus | `spec_adherence` may fall by **at most one scenario**; the decisive measure is the continuous one — the banned-letter occurrence sign test and its bootstrap CI must not move the wrong way with a CI excluding zero |
| R5 | `adversarial_v2` (160 rows) | `format_bait` >= 13/15; no other family drops more than one row |
| R6 | held-out split | must not fall below the tuned checkpoint |
| R7 | five document gates | all exit 0 throughout; `tuned` pin does not move until the whole package re-derives in one commit |

---

## 6. Conditions of the run, disclosed up front

v4 trains on **rented hardware**, not the Modal A10 that produced all four curve
points — both Modal workspaces are over their spend limit and a platform-side
fault blocks raising either ceiling. The pod is an **NVIDIA A40 (45 GB, Ampere,
CUDA 13.0, `bf16_supported = True`)**, with `SLMFORGE_MICRO_BATCH=2`, which is the
recipe the published run actually used (`per_device_train_batch_size 2 /
gradient_accumulation_steps 8`, effective batch 16, measured peak 10.11 GiB).

The wheel set is pinned to exactly what the curve points used — `torch==2.12.1`,
`transformers==4.57.6`, `peft==0.20.0`, `bitsandbytes==0.50.1` — so the recipe is
held constant and only the host changes. The smoke run's `run_manifest.json` is
checked for `bnb_4bit_compute_dtype == torch.bfloat16` and a peak near 10.4 GiB
**before** the full run starts; any deviation is a stop-and-fix, not a
proceed-and-caveat.

The teacher for the v4 attack block is **`anthropic:claude-opus-5`**, recorded
truthfully in the ledger — a different teacher from v1–v3's
`openai:gpt-5.6-terra`. It is on this project's verified frontier list
(`configs/models.json:frontier`). Unlike an API teacher it drafts against the real
checkers in-loop rather than being filtered after the fact, so the keep rate is
not comparable with v3's and no inference should be drawn from it.
