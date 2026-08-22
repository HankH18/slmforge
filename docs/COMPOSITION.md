# Composed Behaviour (Stretch Ladder rung 3)

**Headline: the corpus proves 147 constraint sets are held simultaneously, and it cannot prove that a second constraint type was TAUGHT. On the question it can answer — does an individual constraint get dropped more often when more constraints share the prompt — the honest answer is that no trend survives the confound. The largest effect, base `banned_words` at −16.0 pp from k≤3 to k=4, has raw permutation p = 0.0515, Holm-adjusted p = 0.4635, and loses most of its magnitude when adversarial pressure is held fixed. `k` and pressure are entangled at trend p = 0.00005.**

**Second headline, and the sharper one: the corpus contains almost no genuine constraint-vs-constraint tension. Over all 147 scenarios the vocabulary surviving every constraint jointly is within 0.22 % of what independence predicts (ratio range [0.9979, 1.0022]), and not one scenario falls below 0.9. That is not an accident — `scenarios/reachability.py` refuses contradictory sets by construction — but it bounds the rung-3 claim hard: what is demonstrated is simultaneous holding of near-INDEPENDENT constraints, not the resolution of competing ones.**

Every number below is re-derived from the committed transcripts by `scripts/composition_report.py`. The tables in [§4](#4-the-tables) are generated, not transcribed, and the figures quoted in the prose around them are bound the same way: `python scripts/composition_report.py --check docs/COMPOSITION.md` re-derives both and exits 1 if this document and the transcripts disagree.

---

## 1. What the spec asks, and what is actually addressed

The requirement, verbatim from `docs/project_spec.pdf` page 4:

> Composed behavior — instill a second, potentially competing constraint and show the model holds both at once.

That sentence has three clauses, and they are not equally satisfied.

| Clause | Status | Where |
|---|---|---|
| "show the model holds both at once" | **Addressed.** Every scenario carries k = 2, 3 or 4 simultaneous constraints, at least one negative (`docs/PIVOT_SPEC.md`), scored per constraint by programmatic checkers. The k cut below measures whether holding degrades as k rises. | [§4](#4-the-tables) |
| "instill a second … constraint" | **NOT addressed.** No new constraint type was trained. Both cells are checkpoints that already exist: the pinned base and the pinned fine-tune. Instilling a new type means a training run, and both Modal workspaces are over their spend limit, so no GPU was available to do it. Nothing in this document should be read as evidence that a second constraint type was successfully taught. | — |
| "potentially competing" | **Addressed, and answered in the negative.** Measured rather than assumed: the corpus's constraint sets are essentially independent on the vocabulary, and 0/147 carry an analytic conflict. | [§7](#7-how-much-genuine-tension-exists) |

A weaker but real substitute for the missing clause exists in the data already: the tuned model was trained on `banned_words`, `banned_letter`, `banned_phrases`, `no_lists` and `no_quotation_marks`, and NEVER on `banned_punctuation` or `sentence_initial_letter` (`constraints/registry.py`, `types_by_split("heldout_type")`). So the corpus does ask a trained model to hold a type it was never taught, alongside types it was. That is a generalization test, not an instillation test, and it is reported as such in [§6](#6-base-against-tuned-the-only-comparison-with-power).

**Do not read the k cut as the spec's clause 1 being satisfied by training.** It is satisfied by the DATA — the eval set poses composed constraints and both models are scored against all of them — and the measurement is of degradation, not of acquisition.

---

## 2. Setup

| Item | Value |
|---|---|
| Eval set | `data/scenarios/corpus.jsonl`, 147 scenarios, sha256 `0ad00c19…a9de72` |
| Constraint instances | `data/scenarios/constraint_instances.jsonl`, 65 instances, sha256 `8cad778c…532ed6` |
| Base cell | `hf:Qwen/Qwen3-1.7B@70d244cc…f908b1ad5e`, transcripts sha256 `461a38b6…b86751cd` |
| Tuned cell | `hf:HankH18/qwen3-1.7b-forbidden-constraint-holder@c19712a7…9e8c3fb8`, transcripts sha256 `abfd3a38…e2e19b1e` |
| Run directory | `runs/final/`, strategy `zero_shot` |
| Cell resolution | from `configs/models.json` pins `base` and `tuned`, not from filenames — a moved pin fails the script instead of relabelling a different checkpoint |
| k distribution | k=2: 23 scenarios, k=3: 49, k=4: 75 |
| Outcome | one programmatic checker verdict, re-run from (constraint, output) with `constraints.checkers.check` — the same function the harness and `scripts/gate_check.py` call |

**Nothing stored is trusted, and the stored verdicts are still checked.** Every verdict is re-derived; the transcript's own `checker_results` block is then required to agree with the re-derivation, and a single disagreement stops the script. All 493 constraint checks per cell agree, and all 493 are required to have a stored verdict to agree with: agreement alone would be satisfied vacuously by a transcript whose `checker_results` block had been deleted, which is why the count is asserted and printed as well. A checker changed after the run, a flipped verdict, or a stripped verdict block therefore fails loudly instead of quietly rewriting a table.

**Quarantine.** A row is scored here if it has a usable `output`. Unlike `scripts/curve_report.py`, a row whose JUDGE failed is not dropped — this document measures programmatic verdicts only, and the judge takes no part in them. On these runs that is exactly one row: base `s-0103`, whose judge failed three attempts and whose real 336-character output passes all four of its checkers. `--exclude-quarantined` reproduces `curve_report`'s denominator; the difference is that row and nothing else, and it moves the largest effect from −16.0 pp (p = 0.0515, Holm 0.4635) to −16.2 pp (p = 0.0481, Holm 0.4329). No conclusion in this document turns on which convention is used.

---

## 3. The test, and why this one

**The unit is the scenario, not the check.** 27 scenarios carry two `banned_words` constraints and one carries three, so per-check counts are clustered: those scenarios contribute 2–3 correlated Bernoullis from a single response. A test over checks would treat them as independent and overstate its own significance. A scenario "holds type T" iff every type-T constraint in it passes. The per-check table is printed anyway, because it is the table the finding was first stated in, and dropping it would hide the difference rather than explain it.

**The test is a stratified exact permutation test of the Cochran–Armitage linear trend statistic**

```
T = Σ over strata s, units i in s of (k_i − mean_k(s)) · hold_i
```

with the hold outcomes permuted **within** each stratum, 20 000 draws, seed 7. The two-sided p is `(1 + #{|T*| ≥ |T_obs|}) / (1 + B)`.

Chosen once and used everywhere, for these reasons:

* Cochran–Armitage is the standard test for a trend in a binary outcome across **ordered** groups, which is what k = 2, 3, 4 is. A 3×2 chi-square would discard the ordering; three pairwise exact tests would spend the power on multiplicity.
* **Permutation** rather than the usual normal approximation, because several cells are tiny — base `banned_words` at k=2 is 2 scenarios — and the asymptotic variance is not trustworthy there. Permutation is exact under exchangeability at any n.
* **Stratified**, because the confound in [§5](#5-reading-the-confound) is real and large. Permuting inside a stratum holds that stratum's pressure (and split) mix fixed, so the stratified p asks the only question worth asking: does k still predict holding once pressure is constant?
* The unstratified column is the same function called with one stratum, so there is exactly one implementation and no chance of two drifting apart.

The **assumption** the test makes is exchangeability of the hold outcomes within a stratum under the null — that scenarios inside a stratum are otherwise interchangeable. That assumption is exactly what [§5](#5-reading-the-confound) probes and, for some cells, breaks. It is stated here so a reader can hold the p-values at the right distance.

**Effect size is reported separately from the test**, as the k=4 minus k≤3 difference with a **Newcombe hybrid-score 95 % CI**, built on top of `curve_report.wilson_interval` — the interval this repo already committed to — rather than a second interval implementation. A Wald interval is useless at these rates: several cells are 0/17 or 25/25, where Wald has zero width and sits on a boundary. The CI is unstratified; the per-stratum rates are printed beside it so the adjustment is visible without a test.

**Multiplicity.** Nine trend tests have any variance to test. At α = 0.05 the family-wise error over nine tests is about 37 %, which is the machinery that manufactures findings, so a Holm–Bonferroni adjusted p is reported next to every raw one. Holm rather than Bonferroni: uniformly more powerful, same assumptions.

**Cells with no variance are refused, not scored.** `banned_letter` for base is 0/105 — it floors at zero at every k — and `banned_phrases` and `no_quotation_marks` are 100 % everywhere in both cells. Those print as **not testable** with the reason. "The test could not run" and "the test found nothing" are different statements, and this project has confused them before.

---

## 4. The tables

<!-- BEGIN composition_report -->

### Hold rate by constraint type and k

Unit: one **check**. Each cell is held/checks (rate) [95% Wilson interval, percentage points].

| model | constraint type | k=2 | k=3 | k=4 |
|---|---|---|---|---|
| base | `banned_letter` | 0/17 (0%) [0–18] | 0/43 (0%) [0–8] | 0/45 (0%) [0–8] |
| base | `banned_phrases` | — | 6/6 (100%) [61–100] | 37/37 (100%) [91–100] |
| base | `banned_punctuation` | — | 10/11 (91%) [62–98] | 22/30 (73%) [56–86] |
| base | `banned_words` | 2/2 (100%) [34–100] | 26/27 (96%) [82–99] | 82/95 (86%) [78–92] |
| base | `no_lists` | 15/15 (100%) [80–100] | 34/34 (100%) [90–100] | 54/56 (96%) [88–99] |
| base | `no_quotation_marks` | 8/8 (100%) [68–100] | 13/13 (100%) [77–100] | 29/29 (100%) [88–100] |
| base | `sentence_initial_letter` | 3/4 (75%) [30–95] | 10/13 (77%) [50–92] | 5/8 (62%) [31–86] |
| tuned | `banned_letter` | 0/17 (0%) [0–18] | 1/43 (2%) [0–12] | 0/45 (0%) [0–8] |
| tuned | `banned_phrases` | — | 6/6 (100%) [61–100] | 37/37 (100%) [91–100] |
| tuned | `banned_punctuation` | — | 5/11 (45%) [21–72] | 12/30 (40%) [25–58] |
| tuned | `banned_words` | 2/2 (100%) [34–100] | 26/27 (96%) [82–99] | 89/95 (94%) [87–97] |
| tuned | `no_lists` | 14/15 (93%) [70–99] | 33/34 (97%) [85–99] | 56/56 (100%) [94–100] |
| tuned | `no_quotation_marks` | 8/8 (100%) [68–100] | 13/13 (100%) [77–100] | 29/29 (100%) [88–100] |
| tuned | `sentence_initial_letter` | 0/4 (0%) [0–49] | 1/13 (8%) [1–33] | 0/8 (0%) [0–32] |

Unit: one **scenario** (a scenario holds a type iff every constraint of that type in it passes). This is the unit every test below uses; 27 scenarios carry two `banned_words` constraints and one carries three, so checks are clustered and the table above cannot be tested as if its rows were independent.

| model | constraint type | k=2 | k=3 | k=4 |
|---|---|---|---|---|
| base | `banned_letter` | 0/17 (0%) [0–18] | 0/43 (0%) [0–8] | 0/45 (0%) [0–8] |
| base | `banned_phrases` | — | 6/6 (100%) [61–100] | 36/36 (100%) [90–100] |
| base | `banned_punctuation` | — | 10/11 (91%) [62–98] | 22/30 (73%) [56–86] |
| base | `banned_words` | 2/2 (100%) [34–100] | 26/27 (96%) [82–99] | 54/67 (81%) [70–88] |
| base | `no_lists` | 15/15 (100%) [80–100] | 34/34 (100%) [90–100] | 54/56 (96%) [88–99] |
| base | `no_quotation_marks` | 8/8 (100%) [68–100] | 13/13 (100%) [77–100] | 29/29 (100%) [88–100] |
| base | `sentence_initial_letter` | 3/4 (75%) [30–95] | 10/13 (77%) [50–92] | 5/8 (62%) [31–86] |
| tuned | `banned_letter` | 0/17 (0%) [0–18] | 1/43 (2%) [0–12] | 0/45 (0%) [0–8] |
| tuned | `banned_phrases` | — | 6/6 (100%) [61–100] | 36/36 (100%) [90–100] |
| tuned | `banned_punctuation` | — | 5/11 (45%) [21–72] | 12/30 (40%) [25–58] |
| tuned | `banned_words` | 2/2 (100%) [34–100] | 26/27 (96%) [82–99] | 61/67 (91%) [82–96] |
| tuned | `no_lists` | 14/15 (93%) [70–99] | 33/34 (97%) [85–99] | 56/56 (100%) [94–100] |
| tuned | `no_quotation_marks` | 8/8 (100%) [68–100] | 13/13 (100%) [77–100] | 29/29 (100%) [88–100] |
| tuned | `sentence_initial_letter` | 0/4 (0%) [0–49] | 1/13 (8%) [1–33] | 0/8 (0%) [0–32] |

### Does hold rate decline with k?

Stratified permutation test of the Cochran-Armitage trend statistic, scenario-level, 20000 resamples, seed 7. `p (raw)` is unstratified; `p (pressure)` permutes only within clean and within adversarial, so it asks whether k still predicts holding with pressure held fixed. `p (Holm)` adjusts the unstratified p across the family of tests actually run.

| model | type | k≤3 | k=4 | Δ (pp) | 95% CI on Δ (pp) | p (raw) | p (Holm) | p (pressure) | p (pressure×split) |
|---|---|---|---|---|---|---|---|---|---|
| base | `banned_letter` | 0/60 (0%) | 0/45 (0%) | — | — | not testable (every outcome identical) | — | — | — |
| base | `banned_phrases` | 6/6 (100%) | 36/36 (100%) | — | — | not testable (every outcome identical) | — | — | — |
| base | `banned_punctuation` | 10/11 (91%) | 22/30 (73%) | -17.6 | [-36.9, +13.7] | 0.3997 | 1.0000 | 0.4055 | 0.4055 |
| base | `banned_words` | 28/29 (97%) | 54/67 (81%) | -16.0 | [-27.3, -0.2] | 0.0515 | 0.4635 | 0.1753 | 0.0458 |
| base | `no_lists` | 49/49 (100%) | 54/56 (96%) | -3.6 | [-12.1, +4.2] | 0.3968 | 1.0000 | 0.3711 | 0.4157 |
| base | `no_quotation_marks` | 21/21 (100%) | 29/29 (100%) | — | — | not testable (every outcome identical) | — | — | — |
| base | `sentence_initial_letter` | 13/17 (76%) | 5/8 (62%) | -14.0 | [-48.8, +19.7] | 0.7399 | 1.0000 | 1.0000 | 1.0000 |
| tuned | `banned_letter` | 1/60 (2%) | 0/45 (0%) | -1.7 | [-8.9, +6.3] | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| tuned | `banned_phrases` | 6/6 (100%) | 36/36 (100%) | — | — | not testable (every outcome identical) | — | — | — |
| tuned | `banned_punctuation` | 5/11 (45%) | 12/30 (40%) | -5.5 | [-36.1, +24.5] | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| tuned | `banned_words` | 28/29 (97%) | 61/67 (91%) | -5.5 | [-15.2, +9.0] | 0.4620 | 1.0000 | 1.0000 | 0.3980 |
| tuned | `no_lists` | 47/49 (96%) | 56/56 (100%) | +4.1 | [-3.0, +13.7] | 0.1105 | 0.8844 | 0.0384 | 0.0396 |
| tuned | `no_quotation_marks` | 21/21 (100%) | 29/29 (100%) | — | — | not testable (every outcome identical) | — | — | — |
| tuned | `sentence_initial_letter` | 1/17 (6%) | 0/8 (0%) | -5.9 | [-27.0, +26.9] | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Same cells, split by pressure, so the confound is visible without a test.

| model | type | clean k≤3 | clean k=4 | adversarial k≤3 | adversarial k=4 |
|---|---|---|---|---|---|
| base | `banned_letter` | 0/38 (0%) | 0/20 (0%) | 0/22 (0%) | 0/25 (0%) |
| base | `banned_phrases` | 6/6 (100%) | 21/21 (100%) | — | 15/15 (100%) |
| base | `banned_punctuation` | 6/6 (100%) | 8/10 (80%) | 4/5 (80%) | 14/20 (70%) |
| base | `banned_words` | 18/18 (100%) | 23/25 (92%) | 10/11 (91%) | 31/42 (74%) |
| base | `no_lists` | 31/31 (100%) | 21/22 (95%) | 18/18 (100%) | 33/34 (97%) |
| base | `no_quotation_marks` | 16/16 (100%) | 13/13 (100%) | 5/5 (100%) | 16/16 (100%) |
| base | `sentence_initial_letter` | 11/13 (85%) | 2/3 (67%) | 2/4 (50%) | 3/5 (60%) |
| tuned | `banned_letter` | 1/38 (3%) | 0/20 (0%) | 0/22 (0%) | 0/25 (0%) |
| tuned | `banned_phrases` | 6/6 (100%) | 21/21 (100%) | — | 15/15 (100%) |
| tuned | `banned_punctuation` | 3/6 (50%) | 6/10 (60%) | 2/5 (40%) | 6/20 (30%) |
| tuned | `banned_words` | 18/18 (100%) | 25/25 (100%) | 10/11 (91%) | 36/42 (86%) |
| tuned | `no_lists` | 31/31 (100%) | 22/22 (100%) | 16/18 (89%) | 34/34 (100%) |
| tuned | `no_quotation_marks` | 16/16 (100%) | 13/13 (100%) | 5/5 (100%) | 16/16 (100%) |
| tuned | `sentence_initial_letter` | 1/13 (8%) | 0/3 (0%) | 0/4 (0%) | 0/5 (0%) |

### Does the tuned model pay a SMALLER composition cost?

The reading being tested is an interaction, so it is tested as one: the outcome is the paired per-scenario contrast d = (tuned holds) − (base holds), and the same trend statistic asks whether d grows with k. Both models answered the same prompts, so pairing removes topic, letter, pressure and split.

| type | scenarios | mean d, k≤3 | mean d, k=4 | p (raw) | p (Holm) | p (pressure) | p (pressure, Holm) |
|---|---|---|---|---|---|---|---|
| `banned_letter` | 105 | +0.0167 | +0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `banned_phrases` | 42 | +0.0000 | +0.0000 | not testable | — | not testable | — |
| `banned_punctuation` | 41 | -0.4545 | -0.3333 | 0.7359 | 1.0000 | 0.5102 | 1.0000 |
| `banned_words` | 96 | +0.0000 | +0.1045 | 0.2289 | 0.9158 | 0.3019 | 1.0000 |
| `no_lists` | 105 | -0.0408 | +0.0357 | 0.0783 | 0.3917 | 0.0184 | 0.0922 |
| `no_quotation_marks` | 50 | +0.0000 | +0.0000 | not testable | — | not testable | — |
| `sentence_initial_letter` | 25 | -0.7059 | -0.6250 | 0.7583 | 1.0000 | 1.0000 | 1.0000 |

### The confound: k is not randomly assigned

| adversarial share | k=2 | k=3 | k=4 | trend p |
|---|---|---|---|---|
| scenarios | 4/23 (17%) | 19/49 (39%) | 46/75 (61%) | 0.0001 |

| held-out-type share | k=2 | k=3 | k=4 | trend p |
|---|---|---|---|---|
| scenarios | 4/23 (17%) | 24/49 (49%) | 32/75 (43%) | 0.1398 |

Which types share the prompt, by k (scenario counts; `_scenarios` is the cell's own denominator):

| type | k=2 | k=3 | k=4 |
|---|---|---|---|
| `banned_letter` | 17 scen: no_lists 12, no_quotation_marks 5 | 43 scen: banned_phrases 2, banned_punctuation 11, banned_words 22, no_lists 30, no_quotation_marks 11, sentence_initial_letter 10 | 45 scen: banned_phrases 18, banned_punctuation 20, banned_words 38, no_lists 29, no_quotation_marks 18, sentence_initial_letter 2 |
| `banned_phrases` | — | 6 scen: banned_letter 2, banned_words 5, no_lists 3, no_quotation_marks 1, sentence_initial_letter 1 | 36 scen: banned_letter 18, banned_punctuation 4, banned_words 30, no_lists 30, no_quotation_marks 13, sentence_initial_letter 1 |
| `banned_punctuation` | — | 11 scen: banned_letter 11, no_lists 9, no_quotation_marks 2 | 30 scen: banned_letter 20, banned_phrases 4, banned_words 24, no_lists 22, no_quotation_marks 10, sentence_initial_letter 6 |
| `banned_words` | 2 scen: no_quotation_marks 2 | 27 scen: banned_letter 22, banned_phrases 5, no_lists 16, no_quotation_marks 9, sentence_initial_letter 2 | 67 scen: banned_letter 38, banned_phrases 30, banned_punctuation 24, no_lists 50, no_quotation_marks 24, sentence_initial_letter 6 |
| `no_lists` | 15 scen: banned_letter 12, sentence_initial_letter 3 | 34 scen: banned_letter 30, banned_phrases 3, banned_punctuation 9, banned_words 16, sentence_initial_letter 10 | 56 scen: banned_letter 29, banned_phrases 30, banned_punctuation 22, banned_words 50, no_quotation_marks 10, sentence_initial_letter 7 |
| `no_quotation_marks` | 8 scen: banned_letter 5, banned_words 2, sentence_initial_letter 1 | 13 scen: banned_letter 11, banned_phrases 1, banned_punctuation 2, banned_words 9, sentence_initial_letter 3 | 29 scen: banned_letter 18, banned_phrases 13, banned_punctuation 10, banned_words 24, no_lists 10, sentence_initial_letter 2 |
| `sentence_initial_letter` | 4 scen: no_lists 3, no_quotation_marks 1 | 13 scen: banned_letter 10, banned_phrases 1, banned_words 2, no_lists 10, no_quotation_marks 3 | 8 scen: banned_letter 2, banned_phrases 1, banned_punctuation 6, banned_words 6, no_lists 7, no_quotation_marks 2 |

Letters drawn, by k (letter difficulty is not constant, so a shifting letter mix would masquerade as a k effect):

| type | k=2 | k=3 | k=4 |
|---|---|---|---|
| `banned_letter` | a×6, e×6, i×3, o×1, s×1 | a×12, e×14, i×4, o×5, s×8 | a×13, e×15, i×6, o×8, s×3 |
| `sentence_initial_letter` | a×1, i×1, o×1, t×1 | a×1, e×2, i×1, m×1, o×1, s×3, t×2, w×2 | a×1, i×1, m×1, s×2, t×2, w×1 |

Clustering of checks within a scenario (why the tests use the scenario, not the check):

| type | scenarios | checks | scenarios with >1 | max in one scenario |
|---|---|---|---|---|
| `banned_letter` | 105 | 105 | 0 | 1 |
| `banned_phrases` | 42 | 43 | 1 | 2 |
| `banned_punctuation` | 41 | 41 | 0 | 1 |
| `banned_words` | 96 | 124 | 27 | 3 |
| `no_lists` | 105 | 105 | 0 | 1 |
| `no_quotation_marks` | 50 | 50 | 0 | 1 |
| `sentence_initial_letter` | 25 | 25 | 0 | 1 |

### Pressure at fixed k = 4

k is held at 4, so nothing here is a composition effect. What is left is the embedded instruction telling the model to break the rule — the only kind of competition this corpus contains in quantity.

| model | type | clean | adversarial | 95% CI on Δ (pp) |
|---|---|---|---|---|
| base | `banned_letter` | 0/20 (0%) | 0/25 (0%) | [-16.1, +13.3] |
| base | `banned_phrases` | 21/21 (100%) | 15/15 (100%) | [-20.4, +15.5] |
| base | `banned_punctuation` | 8/10 (80%) | 14/20 (70%) | [-36.2, +24.6] |
| base | `banned_words` | 23/25 (92%) | 31/42 (74%) | [-34.2, +2.0] |
| base | `no_lists` | 21/22 (95%) | 33/34 (97%) | [-10.9, +19.0] |
| base | `no_quotation_marks` | 13/13 (100%) | 16/16 (100%) | [-19.4, +22.8] |
| base | `sentence_initial_letter` | 2/3 (67%) | 3/5 (60%) | [-52.5, +47.2] |
| tuned | `banned_letter` | 0/20 (0%) | 0/25 (0%) | [-16.1, +13.3] |
| tuned | `banned_phrases` | 21/21 (100%) | 15/15 (100%) | [-20.4, +15.5] |
| tuned | `banned_punctuation` | 6/10 (60%) | 6/20 (30%) | [-57.9, +6.1] |
| tuned | `banned_words` | 25/25 (100%) | 36/42 (86%) | [-27.8, +1.0] |
| tuned | `no_lists` | 22/22 (100%) | 34/34 (100%) | [-10.2, +14.9] |
| tuned | `no_quotation_marks` | 13/13 (100%) | 16/16 (100%) | [-19.4, +22.8] |
| tuned | `sentence_initial_letter` | 0/3 (0%) | 0/5 (0%) | [-56.1, +43.5] |

### Base against tuned, paired on the same scenarios

McNemar exact. `base-only` counts scenarios where base held the type and tuned did not.

| type / k | paired | base-only | tuned-only | exact p | Bonferroni over 21 tests |
|---|---|---|---|---|---|
| `banned_letter` k<=3 | 60 | 0 | 1 | 1.0000 | 1.0000 |
| `banned_letter` k=4 | 45 | 0 | 0 | 1.0000 | 1.0000 |
| `banned_letter` all k | 105 | 0 | 1 | 1.0000 | 1.0000 |
| `banned_phrases` k<=3 | 6 | 0 | 0 | 1.0000 | 1.0000 |
| `banned_phrases` k=4 | 36 | 0 | 0 | 1.0000 | 1.0000 |
| `banned_phrases` all k | 42 | 0 | 0 | 1.0000 | 1.0000 |
| `banned_punctuation` k<=3 | 11 | 5 | 0 | 0.0625 | 1.0000 |
| `banned_punctuation` k=4 | 30 | 11 | 1 | 0.0063 | 0.1333 |
| `banned_punctuation` all k | 41 | 16 | 1 | 0.0003 | 0.0058 |
| `banned_words` k<=3 | 29 | 1 | 1 | 1.0000 | 1.0000 |
| `banned_words` k=4 | 67 | 1 | 8 | 0.0391 | 0.8203 |
| `banned_words` all k | 96 | 2 | 9 | 0.0654 | 1.0000 |
| `no_lists` k<=3 | 49 | 2 | 0 | 0.5000 | 1.0000 |
| `no_lists` k=4 | 56 | 0 | 2 | 0.5000 | 1.0000 |
| `no_lists` all k | 105 | 2 | 2 | 1.0000 | 1.0000 |
| `no_quotation_marks` k<=3 | 21 | 0 | 0 | 1.0000 | 1.0000 |
| `no_quotation_marks` k=4 | 29 | 0 | 0 | 1.0000 | 1.0000 |
| `no_quotation_marks` all k | 50 | 0 | 0 | 1.0000 | 1.0000 |
| `sentence_initial_letter` k<=3 | 17 | 12 | 0 | 0.0005 | 0.0103 |
| `sentence_initial_letter` k=4 | 8 | 5 | 0 | 0.0625 | 1.0000 |
| `sentence_initial_letter` all k | 25 | 17 | 0 | 0.0000 | 0.0003 |

### How much genuine constraint-vs-constraint tension exists?

* Scenarios: 147.
* `reachability.find_conflicts` fires on 0 of them.
* `reachability.build_witness` constructs a paragraph that the ground-truth checkers pass against every constraint at once for 147/147.
* `reachability.task_conflicts` fires on 0 of the nudge-free (clean) tasks.
* 72/368 banned-word entries are already excluded by a co-occurring banned letter — redundancy, not tension.

**Lexical interaction.** Joint vocabulary survival against the independence expectation, over 147 scenarios: median 1.0, range [0.9979, 1.0022]. 0 scenarios fall below 0.9.

| k | n | min | median | max | below 0.9 |
|---|---|---|---|---|---|
| 2 | 23 | 1.0 | 1.0 | 1.0 | 0 |
| 3 | 49 | 0.9979 | 1.0 | 1.0 | 0 |
| 4 | 75 | 0.9989 | 1.0 | 1.0022 | 0 |

**Positional tension**, the sharpest case the corpus can hold: a `sentence_initial_letter` meeting a letter ban. Ratio below 1 = the ban bites the required openers harder than the language at large. Over 25 scenarios: median 1.0, range [0.9366, 1.3265], 0 below 0.9.

| scenario | k | initial | openers surviving | pool survival | opener survival | ratio |
|---|---|---|---|---|---|---|
| `s-0117` | 3 | `w` | 14/32 | 0.4671 | 0.4375 | 0.9366 |
| `s-0098` | 3 | `s` | 55/55 | 1.0 | 1.0 | 1.0 |
| `s-0107` | 4 | `a` | 25/25 | 1.0 | 1.0 | 1.0 |
| `s-0119` | 4 | `t` | 22/22 | 1.0 | 1.0 | 1.0 |
| `s-0121` | 2 | `t` | 22/22 | 1.0 | 1.0 | 1.0 |
| `s-0122` | 3 | `s` | 55/55 | 1.0 | 1.0 | 1.0 |
| `s-0130` | 3 | `w` | 32/32 | 1.0 | 1.0 | 1.0 |
| `s-0131` | 2 | `i` | 17/17 | 1.0 | 1.0 | 1.0 |
| `s-0132` | 4 | `m` | 22/22 | 1.0 | 1.0 | 1.0 |
| `s-0133` | 2 | `o` | 23/23 | 1.0 | 1.0 | 1.0 |
| `s-0136` | 4 | `s` | 55/55 | 1.0 | 1.0 | 1.0 |
| `s-0141` | 4 | `i` | 17/17 | 1.0 | 1.0 | 1.0 |
| `s-0143` | 2 | `a` | 25/25 | 1.0 | 1.0 | 1.0 |
| `s-0103` | 4 | `w` | 32/32 | 0.9978 | 1.0 | 1.0022 |
| `s-0112` | 4 | `s` | 34/55 | 0.6031 | 0.6182 | 1.0251 |
| `s-0140` | 3 | `m` | 15/22 | 0.6623 | 0.6818 | 1.0295 |
| `s-0147` | 3 | `s` | 37/55 | 0.6382 | 0.6727 | 1.0542 |
| `s-0129` | 3 | `a` | 13/25 | 0.4671 | 0.52 | 1.1132 |
| `s-0145` | 3 | `o` | 12/23 | 0.4671 | 0.5217 | 1.117 |
| `s-0135` | 3 | `e` | 12/20 | 0.5154 | 0.6 | 1.1643 |
| `s-0110` | 3 | `t` | 17/22 | 0.6623 | 0.7727 | 1.1668 |
| `s-0138` | 3 | `t` | 16/22 | 0.6031 | 0.7273 | 1.206 |
| `s-0096` | 4 | `t` | 13/22 | 0.4671 | 0.5909 | 1.265 |
| `s-0149` | 3 | `i` | 13/17 | 0.6031 | 0.7647 | 1.268 |
| `s-0108` | 3 | `e` | 16/20 | 0.6031 | 0.8 | 1.3265 |

<!-- END composition_report -->

---

## 5. Reading the confound

**k is not randomly assigned, and the largest imbalance is the one that matters most.** The adversarial share of scenarios runs 4/23 (17 %) at k=2, 19/49 (39 %) at k=3, 46/75 (61 %) at k=4 — trend p = 0.00005 by the same permutation test. Adversarial scenarios append a nudge that explicitly invites a violation. So "hold rate falls with k" and "hold rate falls under pressure" are partially the same sentence, and any trend claim that does not separate them is measuring the wrong thing.

The held-out-type share is 4/23, 24/49, 32/75 — non-monotone and trend p = 0.1398, so split is a weaker confound than pressure, though it is not zero.

**Which types share the prompt also changes with k, and in one case it removes the comparison entirely.** The base `banned_words` k=2 cell is 2 scenarios whose only companion constraint is `no_quotation_marks`, both clean. The k=4 cell is 67 scenarios that additionally carry `banned_letter` (38), `no_lists` (50), `banned_phrases` (30) and `banned_punctuation` (24), 42 of them adversarial. "2/2 at k=2 against 54/67 at k=4" is not one variable moving; it is a different experiment. **No trend should be read off a k=2 cell for `banned_words`.** The trend tests use all three k levels, so this cell contributes — with a weight proportional to its 2 scenarios, which is why the raw k≤3 vs k=4 contrast in the table is the more honest summary.

**Letter mix.** The `banned_letter` mix is roughly stable across k (`e` and `a` dominate at every level), so letter difficulty is unlikely to drive the `banned_letter` floor. The `sentence_initial_letter` mix does shift, but on 4 / 13 / 8 instances nothing can be read from it either way, and that is stated rather than modelled.

**What the confound does and does not do to the headline effect.** Base `banned_words` is the largest effect in the table: 28/29 (97 %) at k≤3 against 54/67 (81 %) at k=4, Δ = −16.0 pp, 95 % CI [−27.3, −0.2]. Split by pressure it is 18/18 → 23/25 within clean and 10/11 → 31/42 within adversarial — a decline in **both** strata, so the effect is not purely pressure re-labelled. But the raw permutation p is 0.0515, the Holm-adjusted p is 0.4635, and the pressure-stratified p is 0.1753. **The claim fails on n as much as on the confound.** The pressure×split column reports 0.0458 for that same cell — a finer stratification producing a smaller p than a coarser one is a sign of an unstable estimate, that column carries no multiplicity adjustment, and with nine live trend tests it would not survive one. It is not evidence.

**A design note, since this is a limit of the corpus and not of the analysis.** k was never randomised against pressure, split, topic or letter, because the corpus was not built to answer this question. A corpus that could answer it would hold pressure, split, letter and topic fixed and vary only k — ideally the same task rendered at k=2 and k=4 with the extra constraints drawn from the same families. Nothing in `runs/final` is that design, and no amount of stratification after the fact substitutes for it.

---

## 6. Base against tuned: the only comparison with power

The two cells answered the **same** 147 prompts, so base-against-tuned is paired and McNemar's exact test applies. This is where the data has power, and two results are large enough to survive any reasonable correction for the 21 tests in that table:

* **`sentence_initial_letter`, all k: 17 scenarios base held and tuned did not, 0 the other way, exact p = 1.5 × 10⁻⁵.** Base holds 13/17 at k≤3 and 5/8 at k=4; tuned holds 1/25 in total.
* **`banned_punctuation`, all k: 16 base-only against 1 tuned-only, exact p = 2.7 × 10⁻⁴** (Bonferroni over 21 tests: 0.0058). Base 32/41, tuned 17/41.

Both are **held-out types** — types the training data never contained. Fine-tuning did not merely fail to generalize to them; it made them substantially worse than the untuned base. That is a negative result about the data intervention, and it is the strongest thing this document establishes.

**On the trained types, and on the reading this analysis was built to test.** `banned_words` at k=4 is 1 base-only against 8 tuned-only, exact p = 0.0391 — the tuned model holds better where k is highest — while at k≤3 it is 1 against 1, p = 1.0000. That pattern is exactly the shape of the reading "base pays a composition cost and the tuned model does not."

**It does not survive being tested as what it is.** "Significant in one subgroup and not in the other" is not a difference between subgroups. Tested directly, with the paired per-scenario contrast d = (tuned holds) − (base holds) as the outcome and the same trend statistic asking whether d grows with k: **p = 0.2289 raw, Holm-adjusted 0.9158, pressure-stratified 0.3019.** The interaction is not established. Separately, 0.0391 does not survive multiplicity across the 21 McNemar tests either (Bonferroni: 0.82). The mean advantage does move in the predicted direction — d = +0.0000 at k≤3 and +0.1045 at k=4 — and that is all that can be said.

**`no_lists` deserves an explicit refusal.** Its interaction is p = 0.0184 pressure-stratified (Holm 0.0922), the smallest p in that table. It rests on **four discordant scenarios out of 105**: 2 where base held and tuned did not at k≤3, and 2 the other way at k=4. This project has already caught itself calling a one-scenario difference "ten times worse" at McNemar p = 1.000. Four scenarios is that shape. It is reported because suppressing it would be worse, and it is not a finding.

---

## 7. How much genuine tension exists

The spec's phrase is "potentially competing". Whether the corpus contains competing constraints is a question about the DATA, decidable without running a model, and `scenarios/reachability.py` — the module the corpus was built against, so a set cannot be tense here and clean there — decides it.

**Hard contradiction: none.** `find_conflicts` fires on 0 of 147 scenarios, and `build_witness` constructs, for **147/147**, a real multi-sentence paragraph that the ground-truth checkers pass against every constraint in the set at once. That constructive result is the rung-3 "holds both at once" claim proven at the level of the data: the constraint sets are jointly satisfiable, by exhibited artefact rather than by argument. It says nothing about whether a model can do it.

**Lexical interaction: essentially zero.** For each scenario, the share of `reachability`'s word pool surviving every constraint jointly, against the product of each constraint's own survival share. Independence predicts a ratio of exactly 1. Observed: median 1.0 across all 147, range **[0.9979, 1.0022]**, and **0 scenarios below 0.9** at any k. The constraints cut near-disjoint slices of the vocabulary. They compose; they do not compete.

**Positional tension — the sharpest case the corpus can hold — is also absent.** 12 scenarios pair a `sentence_initial_letter` with a `banned_letter`, and `find_conflicts` guarantees the two letters differ. The measure is the share of required openers surviving the co-occurring constraints, divided by the share of the whole pool surviving them; below 1 means the ban bites the required openers harder than it bites the language. Observed over 25 scenarios: median 1.0, range **[0.9366, 1.3265]**, **0 below 0.9**. The single most tense scenario in the corpus is `s-0117` — open every sentence with `w`, ban the letter `e` — at 0.9366, and 12 of the 25 sit **above** 1.0, meaning the required openers survive the other constraints BETTER than the pool at large.

**Redundancy, which is the opposite of tension, is common.** 72 of 368 banned-word entries (20 %) are already excluded by a co-occurring banned letter — a constraint whose work another constraint has done.

**Task-vs-constraint collisions: 0** on the 78 nudge-free (clean) tasks. `task_conflicts` is run on clean rows only, because an adversarial row's stored task carries an appended nudge and `reachability`'s own contract requires the base task; feeding it the rendered task would flag every format nudge in the corpus as a contradiction. Reporting a number computed on the wrong input would be worse than reporting it on the honest subset.

**What this measure is, and is not.** It runs over `reachability`'s 456-word synthetic pool, which is a proxy for English and not English, and it measures LEXICAL feasibility, not model difficulty. Two constraints can be perfectly independent on the vocabulary and still interact badly inside a 1.7B model's decoding. So the correct statement is bounded: the corpus contains no measurable constraint-vs-constraint tension **by this measure**, and this measure cannot see a tension that lives in the model rather than in the language.

**What the corpus does contain in quantity is constraint-vs-INSTRUCTION tension** — the adversarial nudge telling the model to break the rule the constraint set imposes. At fixed k = 4, where nothing is a composition effect, base `banned_words` runs 23/25 clean against 31/42 adversarial and tuned runs 25/25 against 36/42. Every one of those confidence intervals includes zero, so even this is not established at n = 147; but it is the axis on which this eval's competition actually lives, and a future rung-3 attempt should build the "competing" clause there rather than in the constraint set.

---

## 8. Conclusion

Stated so the unflattering reading is the easy one.

1. **The spec's "instill a second constraint" clause is not addressed.** No new constraint type was trained. Both cells pre-date this analysis. No GPU was available for a training run.
2. **"Holds both at once" is demonstrated for the data and measured for the models.** All 147 constraint sets are jointly satisfiable by constructed witness. Both models are scored per constraint across k = 2, 3, 4.
3. **A composition cost is not established for either model.** The largest effect is base `banned_words`, −16.0 pp from k≤3 to k=4, 95 % CI [−27.3, −0.2], raw p = 0.0515, **Holm-adjusted p = 0.4635**, pressure-stratified p = 0.1753. The smallest Holm-adjusted p in the trend table is 0.4635. The point estimates are negative for base on all four testable types and mixed for tuned, which is consistent with a real cost — and consistent with noise. The corpus cannot separate the two.
4. **The claim that the tuned model pays a smaller composition cost than base is not established.** Tested as the interaction it is: p = 0.2289 raw, 0.9158 Holm-adjusted. The k=4 `banned_words` McNemar result (p = 0.0391) that motivates the reading does not survive multiplicity, and a significant subgroup beside a non-significant one is not a difference between them.
5. **The corpus contains no measurable constraint-vs-constraint tension.** Lexical interaction ratio within 0.22 % of independence for all 147 scenarios; 0 below 0.9; the most tense scenario in the corpus is at 0.9366. This is by construction — reachability refuses contradictory sets — and it **bounds what rung 3 can claim from this eval**: simultaneous holding of near-independent constraints, not the resolution of competing ones.
6. **The one thing this analysis does establish, with power, is negative and is about the fine-tune, not about composition.** On the two held-out constraint types the tuned model is decisively worse than the untuned base: `sentence_initial_letter` 17 base-only against 0 (p = 1.5 × 10⁻⁵) and `banned_punctuation` 16 against 1 (p = 2.7 × 10⁻⁴), paired on the same scenarios.

**Limits.** One strategy (`zero_shot`), one sample per scenario, so generation variance is unmeasured and every interval understates it. 147 scenarios and 65 constraint instances. k confounded with pressure at trend p = 0.00005. Constraint checking is 100 % programmatic; the judge takes no part in any number here.

---

## 9. Re-deriving every number

```bash
python scripts/composition_report.py --selftest      # known-answer checks on the statistics
python scripts/composition_report.py                 # the summary in §5–§8
python scripts/composition_report.py --markdown      # the tables in §4
python scripts/composition_report.py --json          # every intermediate, including per-scenario rows
python scripts/composition_report.py --check docs/COMPOSITION.md   # this document against the transcripts
python scripts/composition_report.py --check docs/COMPOSITION.md --show-facts   # every prose figure, and the line it is bound at
python scripts/composition_report.py --exclude-quarantined         # the curve_report denominator
```

`--check` compares the generated block in §4 against a fresh re-derivation, and then re-derives every figure the prose around it states: each numeral outside the block is matched by a pattern in the script's `PROSE_FACTS` registry, recomputed from the same transcripts, and compared at the precision the sentence writes it — four decimals for `0.0515`, none for `about 37 %`, two significant figures for `2.7 × 10⁻⁴`. A numeral that no pattern claims fails the check as loudly as a wrong one, so a number can neither be hand-edited into this document nor added to it unbound. Two kinds of numeral are exempt and both are printed by name under `--show-facts`: labels rather than measurements (section numbering and cross-references, the k levels, the spec's page number, an exit status), and the one figure quoted from an earlier analysis in this repo rather than derived from these runs. `--selftest` checks every statistic this script owns against a closed form or against complete enumeration: the permutation p against exhaustive relabeling, the Newcombe interval against its published worked example, Holm against a hand-computed step-down, and the within-stratum invariant against a deliberately-crossing relabeling. The Wilson interval and the McNemar exact p are imported from `scripts/curve_report.py` and `scripts/compare_checkpoints.py` and checked, not reimplemented.

The script additionally refuses to run rather than produce a misleading table if: a `configs/models.json` pin names a run that is not in the run directory; the transcripts do not cover exactly the eval set; a stored `checker_results` verdict disagrees with the re-derivation; or a permutation draw moves an outcome between strata.
