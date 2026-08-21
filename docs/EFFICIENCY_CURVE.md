# Data-Efficiency Curve (Ablation 2)

**Headline: the graded metric says data does nothing; the underlying behaviour says otherwise, and the two disagree because the metric is a cliff.**

Five checkpoints — the base model plus four QLoRA fine-tunes at N = 106 / 211 / 423 / 846 — evaluated on the identical 147-scenario eval set with the identical judge and rubric. On `spec_adherence` the curve is flat-to-noisy and no trained point clears base. Counting how often the banned letter actually appears, the same five checkpoints run **34 → 35 → 33 → 29 → 14** occurrences in the median response. The behaviour is being learned. The graded metric cannot see it until the count reaches zero.

Every number below was re-derived from the raw transcripts at commit `d85a12a`, not read from a stored aggregate. See [§7](#7-how-these-numbers-were-produced).

---

## 1. Setup

| Item | Value |
|---|---|
| Base model | `hf:Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| Training corpus | `data/train/v2.jsonl`, 846 rows, sha256 `9d9fc3f3…0105c8` |
| Eval set | `data/scenarios/corpus.jsonl` (147 scenarios), sha256 `0ad00c19…a9de72` |
| Constraint instances | `data/scenarios/constraint_instances.jsonl`, sha256 `8cad778c…532ed6` |
| Strategy | `zero_shot`, no system message (the training condition, exactly) |
| Judge | `claude-sonnet-4-6`, rubric hash `12a0dcaaf29080cb` |
| Spec version | `forbidden-constraint-holder-1.0` |
| Metric definitions | `spec_pass` = all programmatic checkers pass **and** judge task-quality ≥ 3 (`harness.QUALITY_THRESHOLD`) |

`spec_adherence` = mean `spec_pass` over scored rows. `robustness` = the same quantity restricted to the adversarial subset (69 of 147 scenarios; 68 for the base run, which quarantined one row). Constraint pass/fail is 100 % programmatic; the judge never sees the constraints and cannot overturn a checker.

All five points share the same `eval_set`, `spec_version`, `judge_model` and `judge_rubric_hash` — checked field by field across the five `*.results.json` files. No new rubric was introduced for this ablation, as the spec requires.

The four trained points also share an identical `config_resolved` block in their training logs — compared key by key, zero differences: LoRA r=16 / α=32 on all seven projections, lr 2e-4, 3 epochs, effective batch 16, seed 42, 4-bit NF4. **Nothing but the row count changed.**

---

## 2. Spacing, and why the subsets are nested

The spec suggests "a log-spaced sweep such as N, N/2, N/4, N/8". Applied to the corpus actually on disk, that is 846 / 423 / 211 / 106 — 846 halved three times, rounded to integers. Four points, each half the size of the next, spanning an 8× range — the spec's own spacing on the real N rather than a rounder set of numbers that would have required generating or discarding data to hit.

Log spacing is the right shape here because the question is *order of magnitude*, not precision: if the behaviour appears somewhere between 100 and 850 rows, doubling steps find it in four points, whereas even linear steps would spend three of the four points in the same regime.

The subsets are **nested**: each is a prefix of one seeded shuffle (`scripts/make_subset.py`, seed `20260821`), so the N=106 rows are a strict subset of N=211, which is a strict subset of N=423, which is a strict subset of the full 846. This is the property that makes the curve interpretable. Drawn independently, a small point could win or lose on *which* rows it happened to receive, and that wiggle would be misread as an effect of dataset size. Nested, the points differ only by volume.

Verified directly rather than taken on trust — all three set-inclusion checks and all three prefix-equality checks pass, and the on-disk sha256 of every subset matches the manifest:

| N | rows | sha256 (first 8) | banned-letter share of constraint slots | letter mix (top 4) |
|---:|---:|---|---:|---|
| 106 | 106 | `5ac85bf5` | 16.6 % | e .254, o .136, a .136, i .136 |
| 211 | 211 | `d8f1f517` | 18.3 % | e .254, a .169, o .123, i .115 |
| 423 | 423 | `4cd9c73e` | 18.7 % | e .286, a .180, o .109, i .109 |
| 846 (full) | 846 | `9d9fc3f3` | 18.6 % | e .301, a .202, o .112, i .108 |

The letter mix matters more than the type mix. A subset that happened to drop the hard, high-frequency letters would post a flattering number at small N and misplace the minimum entirely. It does not: `e` and `a` — which together are 50 % of the parent's letter bans and 63 % of the eval set's — hold their share down to N=106, drifting by a few points at the smallest size as expected from sampling 59 letter rows.

---

## 3. The curve

`spec_adherence` and `robustness` are the two the spec asks for. The remaining columns are the diagnostics that explain them.

| N | `spec_adherence` | `robustness` | passes | adv. passes | all checkers pass | median banned-letter occurrences |
|---:|---:|---:|---:|---:|---:|---:|
| 0 (base) | 0.0890 | 0.1029 | 13 / 146 | 7 / 68 | 27 | 34 |
| 106 | 0.0612 | 0.0580 | 9 / 147 | 4 / 69 | 24 | 35 |
| 211 | 0.0884 | 0.1014 | 13 / 147 | 7 / 69 | 22 | 33 |
| 423 | 0.0816 | 0.0870 | 12 / 147 | 6 / 69 | 18 | 29 |
| 846 | 0.0816 | 0.0870 | 12 / 147 | 6 / 69 | 20 | **14** |

Read the first two columns alone and the conclusion is "eight times the data changes nothing". The whole range of `spec_adherence` across the sweep is 0.0612 to 0.0890 — a spread of four passing scenarios, which a single-seed run on 147 scenarios cannot distinguish from noise. `robustness` tracks it, because it is the same quantity on a 69-scenario subset and is therefore noisier still.

The last column is the same five models measured without the threshold.

---

## 4. The central finding: `spec_adherence` is a cliff metric

`spec_adherence` requires **zero** occurrences of the banned letter anywhere in the response. It cannot distinguish one slip from forty. At N=846 the median response still contains 14 occurrences — so every one of those responses scores exactly the same as the base model's 34: a fail.

Counting occurrences instead, the picture inverts. The count declines at every point after N=106 (35 → 33 → 29 → 14) and falls off a shelf at the largest N. The banned-letter checker itself passes 0, 1, 0, 0 and 1 times out of 105 across the five points — pinned at the floor throughout, because the checker *is* the threshold and nothing short of a perfectly clean response registers on it.

Two supporting cuts, both re-derived from the transcripts:

| N | median occurrences | per 1,000 chars | median response length | median occurrences, `e`-ban rows only |
|---:|---:|---:|---:|---:|
| 0 (base) | 34 | 67.7 | 484 | 42 |
| 106 | 35 | 71.8 | 499 | 43 |
| 211 | 33 | 72.9 | 468 | 47 |
| 423 | 29 | 65.0 | 456 | 36 |
| 846 | **14** | **36.5** | 416 | **16** |

The length-normalised column is the honest test of whether the model is avoiding the letter or merely writing less. **It only clearly separates at N=846**, where density falls 46 % against a 14 % fall in response length. At N=423 the raw count improves 15 % but density improves only 4 %, so most of that point's apparent gain is shorter answers, not better constraint-holding. The raw-count curve is therefore real but flattering in its middle; the signal that survives normalisation lives entirely in the last doubling.

**The finding is about the instrument as much as the model.** A pass/fail constraint metric is a cliff: progress accumulates invisibly on the far side of it and then arrives all at once. Somebody watching only `spec_adherence` across this sweep would conclude the data work was worthless while the underlying error count fell 59 % (46 % length-normalised) — and would go looking for a hyperparameter, which is exactly the move this project's rules forbid.

---

## 5. Minimum viable N

> **This sweep does not contain the minimum viable N.** On the binary metric, no checkpoint in the range 106–846 holds the behaviour reliably: every trained point sits at or below the N=0 base model, so there is no smallest N that qualifies. On the sensitive measure the error is still falling steeply at the largest N — the biggest single drop in the whole sweep is the last doubling, 29 → 14 — with no sign of a plateau anywhere. The threshold therefore lies **above 846**, and these four points have not bracketed it. The honest answer is a bound, not a value: **minimum viable N > 846**.

What would bracket it: **extend the sweep upward**, at roughly 2× and 4× the current N (≈ 1,700 and ≈ 3,400 rows), with the larger corpus built so the present 846 rows stay a prefix of it — otherwise the existing four points are no longer members of the same nested family and the whole curve has to be re-run. Refining downward — adding points at 50, 75, 150 — would only measure the flat part more precisely, and the flat part is already the least interesting region on the chart.

Two costs belong with that number. Generation is the expensive half: `data/train/v2.report.json` records 3,142 uncached teacher generations to keep 846 rows under the hard letter mix — 3.7 per surviving row, with 178 of 1,024 requests giving up entirely after ten attempts, and 2,302 of the 2,367 rejections caused by a `banned_letter` violation — so a 2× point is on the order of 6,300 teacher generations. Training is cheap by comparison: about 24 GPU-minutes on an A10G at the pinned configuration, extrapolating from the 707 s the 846-row run took. Neither is the blocker. The blocker is that the answer to "what is the minimum viable N" is currently "more than we have measured", and saying anything narrower would be inventing a number the data does not support.

---

## 6. Caveats, unsmoothed

**Small-N fine-tuning makes the model worse before it makes it better.** The N=106 point is below base on *both* graded metrics (0.0612 vs 0.0890; 0.0580 vs 0.1029) and carries the highest median occurrence count of any point in the sweep (35, against base's 34) and the highest density (71.8 per 1,000 chars). Whatever a hundred rows teaches, it is not the target behaviour, and it costs something. The curve is not monotone from the origin.

**The binary column is non-monotone; the count column is not the whole story either.** Both are stated above rather than averaged away. The `spec_adherence` sequence 0.0890 → 0.0612 → 0.0884 → 0.0816 → 0.0816 is noise at this sample size, and the report does not claim a shape for it.

**Held-out constraint types erode as N grows.** The v2 corpus contains five constraint types and the eval set contains seven; the two it does not train on regress steadily with more data:

| constraint type | in corpus? | N=0 | N=106 | N=211 | N=423 | N=846 |
|---|---|---:|---:|---:|---:|---:|
| `banned_letter` | yes | 0/105 | 1/105 | 0/105 | 0/105 | 1/105 |
| `banned_words` | yes | 109/123 | 111/124 | 115/124 | 116/124 | 117/124 |
| `banned_phrases` | yes | 43/43 | 42/43 | 43/43 | 43/43 | 43/43 |
| `no_lists` | yes | 102/104 | 102/105 | 102/105 | 101/105 | 103/105 |
| `no_quotation_marks` | yes | 49/49 | 50/50 | 50/50 | 50/50 | 50/50 |
| `banned_punctuation` | **no** | 32/41 | 21/41 | 19/41 | 19/41 | 17/41 |
| `sentence_initial_letter` | **no** | 17/24 | 18/25 | 10/25 | 2/25 | 1/25 |

This is why the "all checkers pass" column in §3 *falls* as N rises while the letter behaviour improves: the trained types hold or improve, the two untrained types collapse, and the all-or-nothing aggregate follows the collapse. Generalisation across constraint *shape* is a separate, unsolved problem from generalisation within a trained type, and this curve measures the second, not the first.

**Single seed, one eval set, 147 scenarios.** No error bars. Differences of a few passing scenarios are not resolvable, and nothing here should be read as resolving them.

---

## 7. How these numbers were produced

`spec_adherence` and `robustness` were recomputed with `scripts/gate_check.py:score_cell`, the same function that gates the G1 ablation and the MVP report — the checker verdicts are re-run from `(constraint, output)` against the pinned instance library, `spec_pass` is re-derived from *those* verdicts plus the row's quality score, and the stored aggregate is compared to the result rather than trusted. All five cells returned clean, with zero disagreements against their stored `metrics` blocks and full 147-scenario coverage.

The occurrence column is not stored anywhere and was computed from the transcripts directly: for each of the 105 eval scenarios carrying a `banned_letter` constraint, the response is put through `constraints.checkers._fold` (the same normalisation the checker uses — invisible-character stripping, NFKD, diacritic and homoglyph folding, casefold) and occurrences of the banned letter are counted, then the median is taken across the 105 rows. Using the checker's own fold means a response cannot lower its count with a zero-width space or a fullwidth character.

The base run quarantined one row (`s-0103`, three consecutive judge refusals, no score produced), which is why its denominator is 146 rather than 147. That row still contributes to the occurrence count, which needs no judge.

---

## 8. Reproducibility — rerunning one sample point

Everything below is on disk. Substitute `106` → `211` / `423` for the other points; the N=846 point is the full corpus and lives in `runs/train_v2` + `runs/mvp_v2`.

**Training logs (the artefact the spec asks for):**

| N | training log | run manifest | transcripts | results |
|---:|---|---|---|---|
| 106 | `runs/efficiency/n106/train_log.json` | `runs/efficiency/n106/run_manifest.json` | `runs/efficiency/n106/hf_runs_efficiency_n106_merged__zero_shot.transcripts.jsonl` | `…__zero_shot.results.json` |
| 211 | `runs/efficiency/n211/train_log.json` | `runs/efficiency/n211/run_manifest.json` | `runs/efficiency/n211/…transcripts.jsonl` | `…results.json` |
| 423 | `runs/efficiency/n423/train_log.json` | `runs/efficiency/n423/run_manifest.json` | `runs/efficiency/n423/…transcripts.jsonl` | `…results.json` |
| 846 | `runs/train_v2/train_log.json` | `runs/train_v2/run_manifest.json` | `runs/mvp_v2/hf_runs_train_v2_merged__zero_shot.transcripts.jsonl` | `…results.json` |
| 0 (base) | — | — | `runs/mvp/hf_Qwen_Qwen3-1.7B_70d244cc…__zero_shot.transcripts.jsonl` | `…results.json` |

Each `train_log.json` carries the base-model repo and revision, the resolved config, the dataset path with its sha256, the full per-step loss history, and `prompt_contract.sample_rendered_prompt` — the literal tokenised prompt for a named row index, so the training condition can be re-derived offline and compared rather than taken on faith.

**Step 1 — regenerate the subsets** (deterministic; the seed is recorded in the manifest and is the script default):

```sh
.venv/bin/python scripts/make_subset.py data/train/v2.jsonl \
    --sizes 106 211 423 --out-dir data/train/subsets --seed 20260821
```

The sha256 of each regenerated file must match `data/train/subsets/v2.subsets.json`. The script asserts nesting itself and aborts if a subset is not contained in the next.

**Step 2 — train.** Modal turns the `@app.local_entrypoint` signature into the CLI, so these are real flags:

```sh
SLMFORGE_GPU=A10G SLMFORGE_MICRO_BATCH=2 \
.venv/bin/modal run training/modal_app.py::main \
    --dataset data/train/subsets/v2.n106.jsonl \
    --out-dir runs/efficiency/n106 \
    --run-name v2_n106
```

`SLMFORGE_GPU` overrides the `A100-80GB` default; all four points ran with `SLMFORGE_GPU=A10G` at a peak VRAM of 8.05–10.11 GiB, so the sweep is reproducible on a small GPU. `SLMFORGE_MICRO_BATCH=2` splits the pinned effective batch of 16 into batch 2 × accum 8; it preserves the effective batch exactly and refuses to run if it would not. Neither is a hyperparameter — `configs/train.json` is frozen by policy, and these are infra knobs held outside it for that reason.

**Step 3 — pull the merged weights** out of the Modal volume (the commands are also written into each run manifest):

```sh
.venv/bin/modal volume get slmforge-artifacts v2_n106/merged runs/efficiency/n106/merged
```

**Step 4 — evaluate**, same harness, same eval set, no new rubric:

```sh
set -a && . ./.env && set +a          # eval.py reads os.environ and loads no dotenv
.venv/bin/python eval.py \
    --model hf:runs/efficiency/n106/merged \
    --eval-set data/scenarios/corpus.jsonl \
    --out runs/efficiency/n106
```

Defaults supply the rest: `--strategy zero_shot`, `--constraints data/scenarios/constraint_instances.jsonl`, `--judge anthropic` with `judge_model` read from `configs/models.json` (`claude-sonnet-4-6`). `--workers` is forced to 1 for `hf:` specs because the local generator is not thread-safe, so it need not be passed.

**Observed cost per point**, from the run manifests and training logs:

| N | optimizer steps | train wall-clock | mean loss, last 10 steps |
|---:|---:|---:|---:|
| 106 | 21 | 94 s | 2.8437 |
| 211 | 42 | 171 s | 2.6545 |
| 423 | 81 | 394 s | 2.4014 |
| 846 | 159 | 707 s | 2.1010 |

Steps scale as 3 epochs × N / 16, as they should for a fixed effective batch. Training loss falls monotonically with N — the model is fitting the data at every size. It is the *metric*, not the fit, that stays flat.
