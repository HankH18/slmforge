# Brainlift — Forbidden-Constraint Holder

Every number in this document is re-derivable from artifacts committed in this
repo. The headline tables come from `scripts/check_mvp.py` and
`scripts/curve_report.py`, which recompute them from the raw transcripts and
refuse to agree with a figure typed in by hand.

---

## 1. The behavior thesis

**Thesis.** A small model can be made to hold a hard *negative* constraint
reliably by controlling the composition of its training data, in a place where
prompting a frontier model plateaus. The lever is which examples exist, not how
many, and not the optimizer settings.

The behavior being trained, quoted exactly from the Behavior Spec
(`docs/PIVOT_SPEC.md:12`, spec version `forbidden-constraint-holder-1.0`):

> "Given a task prompt and a set of 2–4 hard constraints that always includes at
> least one negative constraint (e.g., never use a listed word, exclude a letter,
> no commas), the model completes the task while violating zero constraints. Pass
> = every constraint passes its programmatic checker AND the response genuinely
> addresses the task (judge task-quality at or above threshold)."

Why this behavior. It has three properties that make a data-versus-prompting
claim falsifiable:

1. **The pass condition is programmatic.** Every constraint has a deterministic
   checker. The LLM judge scores task quality 1–5 only; it never sees the
   constraints and cannot overturn a checker. So "did the behavior happen" is not
   a matter of taste.
2. **It cannot be gamed by silence.** Pass requires *all checkers pass* **and**
   quality ≥ 3. Refusing, hedging, or writing nothing scores zero. A model cannot
   buy constraint compliance with degenerate output — which turns out to matter
   enormously in §4.
3. **Generalization is measurable inside the project.** Two constraint types —
   `banned_punctuation` and `sentence_initial_letter` — are withheld from
   training entirely (`docs/PIVOT_SPEC.md` R3). They are the local proxy for a
   held-out grader set, and they are the reason the first tuned model produced a
   usable finding rather than a flat number.

Scoring throughout: `spec_pass = all checkers pass AND quality_score >= 3`;
**spec-adherence** = mean `spec_pass`; **robustness** = spec-adherence on the
adversarial subset. The eval set is `data/scenarios/corpus.jsonl`, 147 scenarios,
sha256 `0ad00c19…`, held fixed across every run reported here.

---

## 2. Prompting really does plateau

Source: `docs/ABLATION_G1.md`, artifacts in `runs/g1/`.

Nine cells: two frontier families × three prompting strategies × 147 scenarios,
plus the local base model as a reference row.

| Model | Strategy | Spec-adherence | Fail rate |
|---|---|---:|---:|
| `openai:gpt-5.6-sol` | **zero_shot** | **0.7823** | **0.2177** |
| `openai:gpt-5.6-sol` | structured (CoT/verify) | 0.7075 | 0.2925 |
| `openai:gpt-5.6-sol` | few_shot (5 exemplars) | 0.6190 | 0.3810 |
| `anthropic:claude-opus-5` | structured | 0.5205 | 0.4795 |
| `anthropic:claude-opus-5` | zero_shot | 0.4714 | 0.5286 |
| `anthropic:claude-opus-5` | few_shot | 0.4422 | 0.5578 |
| `hf:Qwen/Qwen3-1.7B` (base) | zero_shot | 0.0890 | 0.9110 |

**The ceiling is not an under-engineered prompt.** On the best model the
*plainest* prompt won. Five verified compliant exemplars made it worse
(0.6190). A system prompt explicitly ordering a silent character-by-character
verification pass — restate each requirement, draft, re-read against each
requirement, rewrite, re-check — also made it worse (0.7075). The strategy
designed specifically to catch this failure lost to no strategy at all. On the
Anthropic row the ordering differs but no strategy clears 0.53. There is no
prompt in the ablation that removes the gap.

**What survives the best prompt.** In the best cell, 32 scenarios fail. 30 are
checker failures and **every one of them is a `banned_letter` violation**; the
other 2 are refusals the judge scored 1. Every other constraint type in that cell
records zero violations: `banned_words` 0/124, `banned_phrases` 0/43,
`banned_punctuation` 0/41, `no_lists` 0/105, `no_quotation_marks` 0/50,
`sentence_initial_letter` 0/25.

The failure mode is **silent single-token leakage of a high-frequency banned
letter into an unmarked function or inflectional word**, inside a response that
is otherwise fully compliant and well written. 15 of the 30 leak the banned
letter exactly once in the entire response; 23 of 30 leak it at most twice. The
judge scored 22 of those 30 a **5** and the remaining 8 a **4** — the model does
the task well and loses only on the character constraint. The leaked tokens are
the ones a writer never inspects: 1,063 clean characters of sewing-machine advice
under a ban on *e*, ending "Hook timing may also b**e** off"; a 1,410-character
toast containing exactly one *e*, in "cook**e**d".

It is neither a composition effect nor an adversarial-pressure effect. Across the
42 scenarios carrying **no** letter ban the best cell fails 1/42 (0.0238); across
the 105 that carry one it fails 31/105 (0.2952). Adversarial pressure costs about
8 points (clean 0.8205, adversarial 0.7391) — real, but not where the ceiling
lives.

**Honest limitation, restated from the ablation.** The corpus was escalated
toward high-frequency letter bans after seeing which constraint type frontier
models fail, and two ladder rungs were not pre-registered. The bar itself never
moved (thresholds are hard-coded in `scripts/gate_check.py`), every scenario
carries a constructive witness proving it is satisfiable, and the control arm
(42 scenarios with no letter ban, failing at 0.0238) is what makes the result
specific rather than "we made it hard everywhere." The claim to take away is *"a
frontier ceiling exists on high-frequency letter exclusion within composed
constraint sets"*, not *"frontier models fail 22% of arbitrary constraint sets."*
The tuned model is measured on this identical pinned corpus, so both sides sit
the same exam.

---

## 3. What the first tuned model actually did

Source: `docs/MVP_REPORT.md`, artifacts in `runs/mvp/`. Base
`hf:Qwen/Qwen3-1.7B@70d244cc`, tuned
`HankH18/qwen3-1.7b-forbidden-constraint-holder@b97eb115`, same eval set, same
strategy (zero_shot, no system message), same judge model and rubric hash.

**It lost on the headline.**

| metric | base | tuned | delta |
|---|---:|---:|---:|
| spec_adherence | 0.0890 | **0.0748** | −0.0142 |
| robustness | 0.1029 | 0.1014 | −0.0015 |
| n_scored | 146 | 147 | +1 |

That is the result. It is not being dressed up. The decomposition is where the
information is, and it moves in two opposite directions at once.

**Judged task quality rose.**

| judged quality | base | tuned |
|---|---:|---:|
| answers at or above the quality threshold | 90 | **110** |
| failures caused by quality alone | 14 | **8** |

**Trained constraint types improved; the two held-out types collapsed.**

| constraint type | split | base | tuned |
|---|---|---:|---:|
| `banned_words` | train | 109/123 | **118/124** |
| `no_quotation_marks` | train | 49/49 | 50/50 |
| `banned_phrases` | train | 43/43 | 43/43 |
| `no_lists` | train | 102/104 | 100/105 |
| `banned_letter` | train | 0/105 | 1/105 |
| `banned_punctuation` | **held-out** | 32/41 | **18/41** |
| `sentence_initial_letter` | **held-out** | 17/24 | **1/25** |

By split, `spec_pass` went train 10 → 11 and held-out **3 → 0**. Held-out types
are a large enough share of the corpus that their collapse outweighs the gains on
trained types and drags the headline under the base.

**The collapse has a measured mechanism, not a guessed one.** At an unchanged
median of 6 sentences and 88 → 89 words per answer, sentence-internal
punctuation per answer (`,` `;` `:`) rose from a mean of **4.2 to 10.2** (median
3 → 8). Counting the wider punctuation class the base-vs-tuned report uses
(`, ; : ! ? - -- ( ) [ ] " '`), the median rises **4 → 10**. Same length, same
sentence count, roughly two and a half times the punctuation either way.

The base model was not honouring `banned_punctuation`. It was passing it
**incidentally**, by writing flat, comma-free prose. Training bought fluency;
fluency brought commas; the incidental pass died. `sentence_initial_letter`
breaks the same way — more varied sentence openings, a type never trained on, and
no reason for the model to be watching for it.

> Part of the base model's apparent competence on unseen constraint types was an
> artifact of its duller prose, and improving the prose destroyed it. A
> constraint a model satisfies by writing badly is not a constraint it can hold.

This is the finding the held-out split exists to produce, and it is the reason a
"tuned model got worse" line is not the end of the story.

---

## 4. What the failure decomposition revealed

The base-vs-tuned report leads with the held-out collapse, because that is what
the split table shows. Partitioning the tuned model's failures scenario by
scenario says the held-out collapse is **not the dominant failure**. Numbers
below are recomputed directly from
`runs/mvp/…forbidden-constraint-holder…transcripts.jsonl` (147 scored, 11 passes,
136 failures).

**Which constraint types actually fail, by failure set:**

| the scenario's failing types | count |
|---|---:|
| `banned_letter` **alone** | **74** |
| `banned_letter` + `banned_punctuation` | 14 |
| `sentence_initial_letter` alone | 11 |
| `banned_letter` + `sentence_initial_letter` | 9 |
| quality only, no checker failure | 8 |
| `banned_punctuation` alone | 5 |
| everything else (6 patterns) | 15 |

- **74 of 147 scenarios (50.3%) fail on `banned_letter` and nothing else** — 54.4%
  of all failures.
- `banned_letter` is a constraint in **105 of 147 scenarios (71.4%)** of the eval
  corpus, and the tuned model violates it in **104 of those 105** (violation rate
  0.9905). It appears in the failure set of 104 of the 136 failures (76.5%).

**The two candidate diagnoses, settled on arithmetic.** Both are counterfactuals
computed over the real transcripts: hold every other checker verdict and the
judge's quality score fixed, and ask what spec-adherence would be if a given
constraint type were solved.

| counterfactual | passes | spec-adherence |
|---|---:|---:|
| as measured | 11 | 0.0748 |
| solve both **held-out** types (`banned_punctuation` + `sentence_initial_letter`) | 29 | 0.1973 |
| solve **`banned_letter`** alone | 64 | **0.4354** |
| solve all three | 102 | 0.6939 |

Fixing the generalization story the report leads with buys **+0.12**. Fixing the
one type the model was trained on and still cannot do buys **+0.36** — three times
as much, from a single constraint type. `banned_letter` is not a secondary
regression; it is the binding constraint on the whole metric, and it is also
exactly where the frontier ceiling lives (§2). Both the ceiling and the tuned
model's dominant failure sit on the same constraint type.

**So why couldn't the tuned model do the one thing it was trained to do?** The
answer is in the training data's own generation report, `data/train/v1.report.json`:

```
"rejections": 918,
"rejection_reasons":        { "constraint_violation": 918 },
"rejected_constraint_types": { "banned_letter": 918 }
```

918 of 918 rejections. Across an entire 1,024-scenario generation run against the
teacher, **no constraint type other than `banned_letter` ever failed the
rejection filter, not once**. Every other type was free; the letter ban was the
only thing the teacher could not reliably do.

And within `banned_letter`, difficulty is wildly uneven. Re-measured from
`data/train/v1.jsonl` and `data/train/v1.rejected.jsonl` (these are the exact
counts recorded in `datagen/sampling.py:V1_LETTER_OUTCOMES`):

| letter | kept | rejected | reject rate |
|---|---:|---:|---:|
| e | 32 | 166 | **83.8%** |
| a | 33 | 168 | **83.6%** |
| i | 44 | 113 | 72.0% |
| o | 48 | 113 | 70.2% |
| r | 56 | 125 | 69.1% |
| u | 44 | 79 | 64.2% |
| s | 56 | 64 | 53.3% |
| l | 83 | 34 | 29.1% |
| p | 71 | 27 | 27.6% |
| m | 78 | 29 | **27.1%** |

Instance choice was **uniform**, so every letter was *requested* equally often.
What *survived* was whatever the teacher already found easy. The resulting
training corpus — 545 of 881 kept rows carry a letter ban — came out:

```
kept:   l 83   m 78   p 71   s 56   r 56   o 48   i 44   u 44   a 33   e 32
eval:   e 35   a 31   o 14   i 13   s 12                (105 letter scenarios)
```

**332 of the 545 letter rows (60.9%) ban a letter the eval corpus never uses**
(`l`, `m`, `p`, `r`, `u`), while `e` and `a` — the eval set's two largest cells —
are the two *smallest* cells in training. The model was trained on the easy tail
of the exact skill it is graded on. It satisfied **1 of 105** `banned_letter`
eval instances.

---

## 5. The transferable insight

> **Rejection sampling for quality selects against difficulty.**

A quality gate does not merely remove bad examples. It removes *hard* examples,
because hard is where the teacher fails — and it does so silently, because the
gate reports a healthy accept rate (0.860 here) while the composition of what it
kept has quietly shifted underneath. The filter builds a curriculum out of
whatever the teacher already does well, and the student then learns exactly the
part of the skill nobody needed it to learn.

This is a general hazard of distillation pipelines, not a quirk of this project.
It recurs anywhere a teacher's output is filtered by a correctness or quality
check: the filter is a difficulty-biased sampler, and the bias is invisible
unless you measure the *kept* distribution against the *requested* one rather
than trusting the accept rate. Two consequences worth carrying elsewhere:

1. **Report the kept distribution, not just the accept rate.** The accept rate
   said 86% and looked fine. `rejected_constraint_types` and the per-letter
   kept/rejected split are what exposed the problem, and both were already in the
   report — they simply had to be read as a *curriculum* description rather than
   as diagnostics.
2. **A difficulty-neutral request policy is not a difficulty-neutral dataset.**
   Uniform sampling is only unbiased before the filter runs.

---

## 6. The v2 intervention

Implemented in `datagen/sampling.py` as `letter_mix="hard"`
(`datagen/generate.py --letter-mix hard`); `uniform` remains the default and is
byte-for-byte the historical behavior.

Weight the *request* by inverse difficulty so that what *survives* lands where it
should:

```
raw(letter) = ENGLISH_LETTER_FREQUENCY[letter] / LETTER_KEEP_RATES[letter]
weight      = raw(letter) / sum(raw)
```

Since `kept ∝ request × keep_rate` and `request ∝ frequency / keep_rate`, the
keep rate **cancels algebraically** and the kept distribution comes out
proportional to English letter frequency. Verified by importing the module and
projecting the kept shares at the measured v1 keep rates — they reproduce the
English shares to four decimal places:

| letter | request weight | projected kept share | English share |
|---|---:|---:|---:|
| e | 0.3403 | 21.6% | 21.6% |
| a | 0.2163 | 13.9% | 13.9% |
| o | 0.1089 | 12.8% | 12.8% |
| i | 0.1082 | 11.9% | 11.9% |
| s | 0.0585 | 10.7% | 10.7% |
| r | 0.0840 | 10.2% | 10.2% |
| l | 0.0244 | 6.8% | 6.8% |
| u | 0.0339 | 4.8% | 4.8% |
| m | 0.0143 | 4.1% | 4.1% |
| p | 0.0114 | 3.2% | 3.2% |

**Why English frequency and not our own eval histogram.** Our eval corpus uses
e(35), a(31), o(14), i(13), s(12). Copying that histogram would train against our
own answer key: the improvement would be real on this eval and unfalsifiable
anywhere else. English letter frequency is a property of the *language*, so it
predicts what any held-out eval built from English text will emphasise. The two
targets happen to agree in their ordering, which is precisely why choosing the
honest one costs nothing. The weights are *derived* in code from the frequency
table and the measured keep rates rather than pasted in, and `build_scenarios`
refuses to run in `hard` mode if the instance library contains a letter the table
does not cover — a silently unweighted letter would be the original bug again.

**Predicted effect.** The kept letter distribution inverts: `e` goes from 5.9% of
kept letter rows to a projected 21.6%, `a` from 6.1% to 13.9%, while `l`+`m`+`p`
fall from 42.6% to a projected 14.1%. The model should see the letters it is
actually graded on. §4's counterfactual bounds the prize: with everything else
held where it is, solving `banned_letter` alone moves spec-adherence 0.0748 →
0.4354.

**Predicted cost, stated up front.** Hard letters survive the filter less often,
so the same number of *requests* yields fewer kept rows: mean keep rate falls
from 0.4201 under `uniform` to 0.2546 under `hard`, about 61% as many surviving
letter rows per request. Buying a harder curriculum costs teacher calls, and it
costs dataset size at fixed budget — which is one more reason the data-efficiency
curve in §8 matters.

**One caveat on the algebra.** The measured keep rates are per *attempt*, not per
*scenario* (each scenario gets up to 4 attempts), so they function as a
relative-difficulty weight rather than a literal success probability, and the
cancellation is exact only under the proportionality assumption.
`datagen.generate.summarise` re-measures the same kept/rejected counts on every
run, so the realised distribution is checked against this projection rather than
assumed.

---

## 7. Did data → behavior hold?

**The verdict so far: data *composition* is what moved the behavior — not
volume, and not hyperparameters.** Three pieces of evidence, all from the first
iteration:

1. **Hyperparameters were never the variable.** `configs/train.json` is pinned
   (LoRA r=16, α=32, lr 2e-4, 3 epochs, effective batch 16, seed 42) and the
   project rules forbid tuning them to chase a metric. The tuned model's losses
   and gains are therefore attributable to what was in the data.
2. **The data changed behavior in exactly the direction its composition
   predicted, in both directions at once.** Types present in training improved
   (`banned_words` 109/123 → 118/124; trained-split `spec_pass` 10 → 11). Types
   absent from training collapsed (`banned_punctuation` 32/41 → 18/41;
   `sentence_initial_letter` 17/24 → 1/25; held-out `spec_pass` 3 → 0). Fluency,
   which the teacher's data is full of, went up (quality-passing answers 90 →
   110). Data → behavior held; it simply held for the behavior the data actually
   described.
3. **The one trained type that did not improve is the one the data barely
   contained in gradable form.** `banned_letter` was 545 of 881 rows by count but,
   by *letter identity*, was 83 rows of `l` and 32 of `e` against an eval set that
   is 35 `e` and 31 `a`. The composition, not the count, explains the 1/105.

That is a coherent causal story, and it is also a **testable prediction**: if
composition is the lever, then changing *only* the letter mix — same generator,
same teacher, same checkers, same pinned hyperparameters, same eval set — should
move `banned_letter` and therefore the headline. If it does not, the thesis is
wrong in a way I would have to report, and volume or model capacity becomes the
better explanation.

### The answer: partly, and the metric nearly hid it

v2 was generated under `--letter-mix hard`, trained on the same pinned config,
and evaluated on the same 147-scenario corpus. The letter distribution inverted
as intended — kept `e` rows went 32 → 159, `a` 33 → 107, while `p` fell 71 → 7.

On the graded metric, the intervention looks like a failure:

| | base | v1 | v2 |
|---|---|---|---|
| `spec_adherence` | 0.0890 | 0.0748 | 0.0816 |
| `robustness` | 0.1029 | 0.1014 | 0.0870 |
| `banned_letter` instances passed | 0/105 | 1/105 | 1/105 |

Five times the `e` examples bought one extra passing instance. Taken alone, that
says the thesis is wrong.

It is not alone. `spec_adherence` requires **zero** occurrences of the banned
letter in a response; it cannot distinguish "one slip" from "forty". Counting how
often the letter actually appears — the quantity the behavior is made of — the
three models are nowhere near each other:

| | base | v1 | v2 |
|---|---|---|---|
| median occurrences per response | 34.0 | 20.0 | **14.0** |
| per 1,000 characters | 67.7 | 50.5 | **36.5** |
| letter `e` specifically | 42.0 | 32.0 | **16.0** |

A 59% reduction from base, monotone across both iterations, and `e` — the exact
letter v2's rebalancing targeted — halved again from v1. The model is
demonstrably learning to avoid the letter. It has not learned to avoid it
*completely*, and completely is the only thing the checker can see.

**So data → behavior held, on the behavior. It did not hold on the metric.** The
composition hypothesis was right: changing only which letters the training set
contains moved the error rate by more than half, with no hyperparameter touched.
What it did not do is cross the threshold that scores.

That distinction is the most useful thing this project has produced, because it
generalises past this task: **a pass/fail constraint metric is a cliff.**
Progress accumulates invisibly on the far side of it and then arrives all at
once. A team watching only `spec_adherence` across these two iterations would
have concluded their data work was worthless while the underlying error fell 59%
— and would have gone looking for a hyperparameter, which is precisely the move
the project's own rules forbid.

### The held-out collapse: answered, and worse than expected

The open question above was whether the held-out collapse would persist. It does
— and it does not merely persist, it *deepens monotonically with dataset size*,
which is a stronger and more uncomfortable result than "the fix does not
generalize."

Violation rate by constraint type across the whole sweep, lower is better
(`scripts/curve_report.py`, re-derived from the transcripts):

| N | `banned_words` (trained) | `banned_punctuation` (never trained) | `sentence_initial_letter` (never trained) |
|---:|---:|---:|---:|
| 0 (base) | 0.114 | 0.220 | 0.292 |
| 106 | 0.105 | 0.488 | 0.280 |
| 211 | 0.073 | 0.537 | 0.600 |
| 423 | 0.065 | 0.537 | 0.920 |
| 846 | 0.057 | 0.585 | 0.960 |

Every trained-family column falls with N. Every untrained-family column rises
with N. Split the eval set the same way and the two halves move in opposite
directions: train-split `spec_adherence` goes 0.1149 → 0.1379, held-out-split
goes 0.0508 → 0.0000.

So the aggregate metric is not flat because nothing happened. **It is flat
because two real effects of opposite sign are cancelling inside it**, and 60 of
the 147 eval scenarios carry an untrained constraint type. This is the second
way the metric hid the data work, and it is independent of the cliff: even a
metric with no threshold would report ~zero here, because the gain and the loss
genuinely offset.

The honest reading is the one that was pre-committed as the bad case:
composition fixes the type you compose for, and generalization across constraint
*shape* is a separate, unsolved problem. Buying compliance on the trained
families cost generalization on the untrained ones, and the more data was spent,
the more it cost.

**Incidental passes** remain unmeasured. A constraint passed by writing badly is
not a constraint held, and any comparison against the base model on held-out
types flatters the duller writer until an incidental-pass rate exists. Note that
this cuts *against* the base model here — base wins on the held-out families,
and some of that win may be incidental rather than held.

---

## 8. Minimum viable dataset size

Four checkpoints at N = 106 / 211 / 423 / 846 — the spec's own log spacing on
the v2 corpus — each evaluated on the identical eval set, with the base model as
the N=0 reference:

| N | `spec_adherence` | `robustness` | median banned-letter occurrences |
|---|---|---|---|
| 0 (base) | 0.0890 | 0.1029 | 34.0 |
| 106 | 0.0612 | 0.0580 | 35.0 |
| 211 | 0.0884 | 0.1014 | 33.0 |
| 423 | 0.0816 | 0.0870 | 29.0 |
| 846 | 0.0816 | 0.0870 | **14.0** |

**The honest answer is that this sweep does not contain the minimum viable N.**
On the binary metric no point holds the behavior — every one sits at or below
base. On the sensitive metric the error is still falling steeply at the largest N
with no hint of a plateau, so the threshold lies *above* 846 and the sweep has
not bracketed it. Extending upward (2× and 4× the current N) would bracket it;
refining downward would only measure the flat part more precisely.

Three things in that table are worth not smoothing over.

The N=106 point is *worse* than base on both metrics and has the highest
occurrence count of any point — small-N fine-tuning appears to damage the
behavior before it helps it. The binary column is non-monotone while the
occurrence column is monotone from N=211 — the same cliff effect as above,
showing up as noise in one column and signal in the other.

And the most important qualification: **normalising for length, most of the
middle of the curve disappears.** Occurrences per 1,000 characters run
67.7 → 71.8 → 72.9 → 65.0 → 36.5. Only the last doubling separates. At N=423 the
raw count improves 15% but the density only 4%, so most of that point's apparent
gain is the model writing *shorter answers*, not holding the constraint better.
The signal that survives normalisation lives almost entirely in the final
doubling — which cuts against reading the curve as a smooth trend, and
strengthens rather than weakens the conclusion that the threshold is above 846.

That length check matters for the base→v1→v2 comparison too, and it survives it:
that reduction is 67.7 → 50.5 → 36.5 per 1,000 characters, i.e. 46% *after*
normalising, so the headline improvement is not an artifact of terser writing.

The mechanism is already in the repo. `scripts/make_subset.py` builds
**nested** (each smaller subset is a prefix of the same seeded shuffle, so a
small point cannot win or lose on which rows it happened to draw) and
**distribution-preserving** subsets, writing each subset's sha256 and its realised
composition — constraint types, k, pressure, and which letter each
`banned_letter` row bans — to a manifest, so the claim that only *size* varied is
auditable rather than asserted. That last property is load-bearing here: after
§4, a subset that happened to drop the hard letters would post a flattering
number at small N and misplace the minimum entirely.

**The definition above does not survive contact with the data, and saying why
is the finding.** The rule as originally written — *the smallest point whose
spec-adherence is indistinguishable from the largest point's* — silently assumes
the metric can distinguish points at all. It cannot. Here are the 95% Wilson
intervals, re-derived by `scripts/curve_report.py`:

| N | `spec_adherence` | 95% CI |
|---:|---:|---|
| 0 (base) | 0.0890 | [0.053, 0.146] |
| 106 | 0.0612 | [0.033, 0.112] |
| 211 | 0.0884 | [0.052, 0.145] |
| 423 | 0.0816 | [0.047, 0.137] |
| 846 | 0.0816 | [0.047, 0.137] |

Every interval contains every other point's estimate. The entire observed range
of `spec_adherence` across an 8× sweep is 2.8 percentage points; one point's
interval is about 9 wide. Applied literally, the rule therefore names **N=106**
— the smallest checkpoint trained — which is plainly false, since N=106 has the
*worst* measured behaviour on the sweep and a median of 35 banned-letter
occurrences against the full set's 14.

A definition that returns the wrong answer on data this clean is the wrong
definition. The mistake is using a cliff metric to locate a knee: the metric
resolves nothing until compliance is perfect, so it has no knee to find.

**The definition this project actually uses.** Locate the minimum viable N on a
measure that *moves*, and state the cliff metric's blindness as a separate
result rather than burying it. On the paired continuous measure — banned-letter
occurrences per response, counted with the checker's own normalisation
(`scripts/compare_checkpoints.py`) — the picture is not ambiguous at all:

- base → v2 (N=846): median 34 → 14, lower on 89 of 105 paired scenarios,
  sign-test **p = 1.9 × 10⁻¹³**, bootstrap 95% CI on the difference of
  medians **[−26, −15]**.
- v1 → v2, the two data iterations: median 20 → 14, lower on 75 of 105,
  **p = 6.1 × 10⁻⁸**, CI **[−9, −4]**.

Both are overwhelming. On the same rows, `spec_pass` gives McNemar exact
**p = 1.000** for both comparisons — 12 passes against 13, and 12 against 11.

**That contrast is the single most useful sentence this project produced:** the
graded metric cannot distinguish two checkpoints that a paired test on the same
147 scenarios separates at p ≈ 10⁻⁸. Anyone reporting only `spec_adherence`
would have called both data iterations worthless.

**How N is justified.** Three things stated together, or the number is just a
point on a chart:

1. **Which measure, and what it can resolve.** Stated above: `spec_adherence` on
   147 scenarios resolves nothing below ~9 percentage points, so it cannot carry
   the claim; the occurrence measure resolves the difference decisively and is
   what the minimum is defined on.
2. **The composition at that N**, from the subset manifest — specifically whether
   the hard letters are still represented at the intended share. A minimum viable
   size is only meaningful alongside a minimum viable *mix*.
3. **What it cost, measured rather than projected.** v2 spent 3,142 teacher
   calls for 846 kept rows; v1 spent 1,764 for 881. That is 3.71 calls per kept
   row against 2.00 — the harder curriculum cost **1.86x** as much per example,
   and 178 of 1,024 requests were abandoned after ten attempts. (§6 predicted
   this from keep rates before the run; these are the observed figures.) The
   useful statement for anyone reproducing this is N *and* the request budget
   that produced it, because at a fixed budget a harder mix buys a smaller N.

`docs/PIVOT_SPEC.md` R8 names N = 64 / 256 / 1024 / 4096 as the log-spaced ladder.
The top of that ladder is above what a single generation run at this scale yields
(v1 kept 881 rows from 1,024 requested), so the shipped curve uses log-spaced
points up to the size actually generated; the spacing, not the literal endpoints,
is what the requirement is for. Any deviation will be stated with the realised N
values rather than silently rounded.

---

## 9. What I would do differently, and what is still open

**Would do differently:**

1. **Measure the kept distribution before spending a GPU-hour.** Everything in §4
   was already sitting in `data/train/v1.report.json` the moment generation
   finished. `rejected_constraint_types` said `{"banned_letter": 918}` — 918 of
   918 — before any training happened. A five-minute look at the composition of
   what survived the filter, compared against the eval set's composition, would
   have caught this ahead of the training run rather than after the evaluation.
2. **Treat the difficulty distribution as a first-class dataset parameter.** The
   sampler had `k`, pressure, and type share as explicit, reported, balanced
   knobs. *Which letter* was left to a uniform draw, as if letters were
   interchangeable — and it turned out to be the single most consequential axis in
   the dataset, varying reject rate from 27% to 84%.
3. **Baseline the incidental passes at the same time as the model.** The
   base-model number on held-out types was partly an artifact of dull prose, and
   that was only discovered by explaining a regression. It should have been a
   measurement, not an inference.
4. **Put the per-type counterfactual in the eval report itself.** The §4
   partition redirected the whole project and was a hand-rolled analysis over raw
   transcripts. "What would spec-adherence be if type X were solved" belongs in
   the standard output — it is what tells you which number to work on.

**Open questions:**

- **Does composition-fixing generalize across constraint *shape*, or only within
  a type?** The letter fix is a within-type correction. Nothing in it teaches the
  model to obey a *stated negative constraint it has never seen*, which is the
  more interesting version of the thesis and what the held-out split measures.
- **Is quality-vs-constraint a genuine trade-off at this model size?** The first
  tuned model bought fluency and paid for it in incidental constraint passes. If
  that is a frontier rather than an accident, the training objective needs to
  hold both explicitly (the DPO negatives already collected — 474 of them in v1 —
  are the obvious next lever, and are so far unused).
- **How much of the frontier ceiling is tokenization?** The failure mode in §2 is
  sub-token: a banned letter hiding inside a token the model never inspects
  character by character. Whether a small model can be trained past a ceiling that
  is partly architectural, rather than merely learning the surface task, is not
  something this project has evidence on yet.
- **One seed, one dataset version, one model size, no confidence intervals.** The
  base-vs-tuned headline gap (−0.0142) is small relative to what 147 scenarios can
  resolve. The held-out collapse and the letter partition are far larger than that
  and are what the conclusions rest on — but the headline number itself should not
  be over-read in either direction.

---

## Sources

| Claim | File |
|---|---|
| Behavior Spec (quoted) | `docs/PIVOT_SPEC.md:12` |
| Prompting ceiling, failure mode, ladder | `docs/ABLATION_G1.md`, `runs/g1/` |
| Base-vs-tuned, generated metrics block | `docs/MVP_REPORT.md`, `runs/mvp/` |
| Failure partition & counterfactuals (§4) | recomputed from `runs/mvp/*.transcripts.jsonl` + `data/scenarios/constraint_instances.jsonl` |
| Rejection ledger, letter outcomes | `data/train/v1.report.json`, `data/train/v1.jsonl`, `data/train/v1.rejected.jsonl` |
| v2 weighting derivation | `datagen/sampling.py` (`LETTER_REQUEST_WEIGHTS`) |
| Pinned training config | `configs/train.json` |
| Data-efficiency subsets | `scripts/make_subset.py` |
