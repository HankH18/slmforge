# Pre-registration: what the upward extension of the curve should show

**Written and committed before the N≈1690 checkpoint was trained or evaluated.
Nothing below may be edited after that run starts. The results go in
[`EFFICIENCY_CURVE.md`](EFFICIENCY_CURVE.md), and each prediction is marked
hit or missed there against what this file says.**

The five committed points end at N=846, which is the whole v2 dataset. The
report states the minimum viable N as a **bound** — the error was still falling
steeply at the largest N, so no minimum had been located. Refining downward
would only measure the flat part of the curve, so the sweep is being extended
**upward** instead, by generating a superset dataset (v3) and training the same
recipe on it.

A prediction written after seeing the number is not a prediction. This file
exists so that the upward extension can *fail*.

> **Status: predictions stand, untested.** The 2× corpus was generated and an
> inward point at N=634 was prepared, but the training platform refused both runs
> — *"Workspace has exceeded its spend limit"* — so neither checkpoint exists.
> Nothing below has been edited since it was committed, and the corpus and the
> subset are on disk, so testing these predictions is one training command per
> point once GPU access is restored. See `EFFICIENCY_CURVE.md` §5.

---

## 1. What is actually being tested

Re-deriving the five committed points per constraint-type split
(`scripts/curve_report.py`) turns up something the aggregate metric hides. The
curve is not flat. It has **opposite signs on the two halves of the eval set**:

| N | `banned_words` (trained) | `sentence_initial_letter` (never trained) | `banned_punctuation` (never trained) |
|---:|---:|---:|---:|
| 0 (base) | 0.114 | 0.292 | 0.220 |
| 106 | 0.105 | 0.280 | 0.488 |
| 211 | 0.073 | 0.600 | 0.537 |
| 423 | 0.065 | 0.920 | 0.537 |
| 846 | 0.057 | 0.960 | 0.585 |

Violation rate, lower is better. The trained type improves monotonically with
N. Both untrained types degrade monotonically with N. `spec_adherence` requires
every constraint in a scenario to pass, and 60 of the 147 eval scenarios carry
an untrained constraint type, so the aggregate is the sum of a real gain and a
real loss and reads as noise.

That is the claim the extension tests. Stated so it can be killed: **dataset
size buys compliance on the constraint families the data contains and spends
generalization on the families it does not, and both effects continue past
N=846.**

## 2. The predictions

The v3 dataset is generated with the identical recipe, seed and teacher as v2,
and the sampler is prefix-stable, so v3's rows are a strict superset of v2's
846 — verified before generation by replaying the first 24 samples from cache
and comparing them byte-for-byte against `data/train/v2.jsonl`. The training
config is the pinned `configs/train.json`, untouched. Only the row count
changes.

Evaluated on the identical 147-scenario eval set, identical judge, identical
rubric hash:

| # | Prediction at N≈1690 | Fails if |
|---|---|---|
| P1 | `banned_words` violation rate **≤ 0.057** (continues to fall or holds) | > 0.065, i.e. back to the N=423 level |
| P2 | `sentence_initial_letter` violation rate **≥ 0.92** (saturated near 1.0) | < 0.92 |
| P3 | `banned_punctuation` violation rate **≥ 0.585** (no recovery) | < 0.537, i.e. back below the N=211/423 level |
| P4 | held-out-split `spec_adherence` **≤ 0.0167** (≤ 1 of 60) | ≥ 2 of 60 |
| P5 | train-split `spec_adherence` **≥ 0.1379** (holds its best or improves) | < 0.126 |
| P6 | median banned-letter occurrences **< 14** (keeps falling) | ≥ 14 |
| P7 | overall `spec_adherence` **≤ 0.11**, i.e. still no decisive win over base's 0.0890 | > 0.11 |
| P8 | `banned_letter` checker passes **≤ 8 / 105** (still pinned near the floor) | ≥ 9 / 105 |

P6 and P8 are the two that matter most, and they point in opposite directions
on purpose. P6 says the underlying error keeps shrinking. P8 says the graded
metric still cannot see it. If **P6 holds and P8 fails** — the count keeps
falling *and* the checker starts passing — the cliff has been crossed and a
minimum viable N exists between 846 and 1690. That is the outcome that would
make this project's headline finding wrong in the most interesting way, and it
is the reason the sweep goes up rather than down.

## 3. What each outcome means for the minimum viable N

* **P6 holds, P8 holds** (expected): no N in the measured range makes the model
  hold the full composed spec reliably. The honest report is a two-sided
  answer — a minimum N for the *trained* families, and the observation that the
  aggregate has no minimum because generalization damage grows with N. The
  bound tightens to N > 1690.
* **P6 holds, P8 fails**: the cliff is crossed between 846 and 1690. The
  minimum viable N is bracketed and the sweep should be refined *inside* that
  interval, not extended further.
* **P6 fails** (median stops falling): the 34 → 14 decline saturated, and the
  claim that data volume is still buying behaviour at N=846 was reading a floor
  effect. The v2 → v3 gain would then be attributable to something other than
  volume and the curve's headline needs rewriting.
* **P4 or P5 fails**: the split-divergence claim in §1 is wrong, and the
  aggregate's flatness needs a different explanation.

## 4. What is deliberately not predicted

`banned_phrases` and `no_quotation_marks` sit at a 0.000 violation rate at every
committed point. They have no headroom, cannot move down, and any movement up
would be a single row. `no_lists` runs 0.019 / 0.029 / 0.029 / 0.038 / 0.019 —
inside sampling noise on 105 checks. None of the three is evidence either way
and none is predicted.

`robustness` is not predicted separately. It is `spec_adherence` restricted to
the 69 adversarial scenarios, so it inherits whatever P7 does, on a smaller
denominator.
