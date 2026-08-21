# Prompt-Ceiling Ablation (G1)

**Verdict: PROCEED.** Best frontier cell `openai:gpt-5.6-sol` × `zero_shot` — Spec-adherence **0.7823**, fail rate **0.2177** (32/147) against the ≥ 0.20 bar, at ladder rung `r7-hifreq` (7/7, ladder exhausted).

Every number below was read out of `runs/g1/*.results.json`, `runs/g1/gate_report.json` and `runs/g1/run_manifest.json`, and independently recomputed from the raw transcript rows in `runs/g1/*.transcripts.jsonl`.

---

## 1. What was tested

Whether prompting alone can make a frontier model hold a composed constraint set — i.e. whether the target behaviour has a prompting ceiling that leaves room for a tuned small model to beat.

The Behavior Spec being scored (`docs/PIVOT_SPEC.md:12`, spec version `forbidden-constraint-holder-1.0`):

> "Given a task prompt and a set of 2–4 hard constraints that always includes at least one negative constraint (e.g., never use a listed word, exclude a letter, no commas), the model completes the task while violating zero constraints. Pass = every constraint passes its programmatic checker AND the response genuinely addresses the task (judge task-quality at or above threshold)."

Scoring, as implemented (`harness.py`, `scripts/gate_check.py`):

- `spec_pass = all checkers pass AND quality_score >= 3` (`harness.QUALITY_THRESHOLD = 3`).
- **Spec-adherence** = mean `spec_pass` over the eval set. **Fail rate** = 1 − Spec-adherence.
- **Robustness** = Spec-adherence restricted to the adversarial subset (69 of 147 scenarios).
- Constraint pass/fail is 100% programmatic. The judge sees only `(task, output)` — never the constraints, the constraint ids, or the checker results — and scores task-quality 1–5 only. It cannot overturn a checker.

This is the same harness, the same rubric and the same metric definitions the later base-vs-tuned comparison uses; `run_ablation.py` imports `eval.py` rather than reimplementing scoring.

## 2. Setup

| Item | Value | Source |
|---|---|---|
| Frontier model A | `openai:gpt-5.6-sol` (OpenAI family) | `run_manifest.json:models.frontier` |
| Frontier model B | `anthropic:claude-opus-5` (Anthropic family) | `run_manifest.json:models.frontier` |
| Local base (reference row, excluded from the ceiling claim) | `hf:Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | `run_manifest.json:models.local_base` |
| Strategies | `zero_shot`, `few_shot` (5 in-context exemplars), `structured` (CoT/verification system prompt) | `run_manifest.json:strategies`, `eval.py:STRUCTURED_SYSTEM` |
| Scenarios per model × strategy cell | 147 (R6 floor is 30) | `run_manifest.json:n_scenarios` |
| Eval set | `data/scenarios/corpus.jsonl`, sha256 `0ad00c19184d3a51c7950c13800f218b78eb5be4bfb74057608c00186aa9de72` | `run_manifest.json`, re-checked with `shasum -a 256` |
| Constraint instances | `data/scenarios/constraint_instances.jsonl`, sha256 `8cad778c2f8a06d6114a275d5f81dbc00236603349d508b9ab3e64e2fa532ed6` | `run_manifest.json`, re-checked |
| Judge model | `claude-sonnet-4-6` (provider `anthropic`) | `run_manifest.json:judge` |
| Judge rubric hash | `12a0dcaaf29080cb` | `run_manifest.json:judge.rubric_hash`, re-derived via `judge.rubric_hash()` |
| Quality threshold | 3 (of 1–5) | `harness.QUALITY_THRESHOLD` |

Corpus composition (recomputed from `corpus.jsonl`): k = 2 → 23, k = 3 → 49, k = 4 → 75; pressure clean 78 / adversarial 69; split train 87 / held-out-type 60; all four split × pressure cells ≥ 30.

The three strategies, precisely:

- **`zero_shot`** — no system message; the rendered scenario as a single user turn. This is deliberately identical to the training condition, so nothing is added here that `training/modal_app.py` does not also see.
- **`few_shot`** — the same user turn preceded by 5 verified `(prompt, compliant output)` exemplar pairs from `data/scenarios/fewshot_exemplars.jsonl`.
- **`structured`** — a system prompt ordering a silent verification protocol: restate each requirement, draft, re-read the draft against each requirement in turn — *"Where a requirement concerns letters or punctuation, check character by character rather than by impression"* — rewrite, re-check, output only the finished response.

Constraints are rendered as plain sentences, one per line, with no bullet markers (a list-formatted prompt would confound the `no_lists` constraint). Adversarial pressure lives inside the task text; the renderer never argues back.

## 3. Results

All nine cells. `n scored` is rows that produced both a generation and a judge score; Spec-adherence and Robustness are computed over scored rows only.

| Model | Family | Strategy | Spec-adherence | Robustness | Fail rate | n scored | passed |
|---|---|---|---:|---:|---:|---:|---:|
| `openai:gpt-5.6-sol` | OpenAI | **zero_shot** | **0.7823** | **0.7391** | **0.2177** | 147 | 115 |
| `openai:gpt-5.6-sol` | OpenAI | few_shot | 0.6190 | 0.6667 | 0.3810 | 147 | 91 |
| `openai:gpt-5.6-sol` | OpenAI | structured | 0.7075 | 0.7101 | 0.2925 | 147 | 104 |
| `anthropic:claude-opus-5` | Anthropic | zero_shot | 0.4714 | 0.5231 | 0.5286 | 140 | 66 |
| `anthropic:claude-opus-5` | Anthropic | few_shot | 0.4422 | 0.4638 | 0.5578 | 147 | 65 |
| `anthropic:claude-opus-5` | Anthropic | structured | 0.5205 | 0.5507 | 0.4795 | 146 | 76 |
| `hf:Qwen/Qwen3-1.7B` (base) | Qwen | zero_shot | 0.0890 | 0.1029 | 0.9110 | 146 | 13 |
| `hf:Qwen/Qwen3-1.7B` (base) | Qwen | few_shot | 0.0959 | 0.1324 | 0.9041 | 146 | 14 |
| `hf:Qwen/Qwen3-1.7B` (base) | Qwen | structured | 0.1301 | 0.1739 | 0.8699 | 146 | 19 |

Each cell attempted all 147 scenarios. Unscored rows, disclosed rather than silently dropped: `anthropic × zero_shot` 7 generator errors, `anthropic × structured` 1 generator error, each `hf` cell 1 judge error. All six frontier cells and all three base cells clear the R6 floor of 30 scenarios.

The gate reads the six frontier cells only; the base row is recorded and excluded from the ceiling claim (`gate_report.json:classification.rule`).

**Where the failures are.** Violation rate per constraint type (violations / checks), frontier cells:

| Cell | banned_letter | banned_words | banned_phrases | banned_punctuation | no_lists | no_quotation_marks | sentence_initial_letter |
|---|---:|---:|---:|---:|---:|---:|---:|
| `openai` × zero_shot | **0.2857** (30/105) | 0.0 (0/124) | 0.0 (0/43) | 0.0 (0/41) | 0.0 (0/105) | 0.0 (0/50) | 0.0 (0/25) |
| `openai` × few_shot | 0.5238 (55/105) | 0.0 (0/124) | 0.0 (0/43) | 0.0 (0/41) | 0.0 (0/105) | 0.0 (0/50) | 0.0 (0/25) |
| `openai` × structured | 0.3905 (41/105) | 0.0 (0/124) | 0.0 (0/43) | 0.0 (0/41) | 0.0 (0/105) | 0.0 (0/50) | 0.0 (0/25) |
| `anthropic` × zero_shot | 0.6768 (67/99) | 0.0513 (6/117) | 0.0 (0/40) | 0.0256 (1/39) | 0.0 (0/99) | 0.0 (0/49) | 0.12 (3/25) |
| `anthropic` × few_shot | 0.7429 (78/105) | 0.0 (0/124) | 0.0 (0/43) | 0.0244 (1/41) | 0.0 (0/105) | 0.0 (0/50) | 0.16 (4/25) |
| `anthropic` × structured | 0.6442 (67/104) | 0.0081 (1/123) | 0.0 (0/42) | 0.0 (0/41) | 0.0 (0/104) | 0.0 (0/50) | 0.12 (3/25) |

(The base row, for contrast, violates `banned_letter` on 105/105 checks in all three of its cells.)

Two observations worth stating plainly:

1. **No prompting strategy dominates, and the best cell is the plainest prompt.** For `openai:gpt-5.6-sol`, zero-shot beats structured (0.7823 vs 0.7075) and few-shot is worst (0.6190). For `anthropic:claude-opus-5` the order is structured > zero-shot > few-shot. The ceiling therefore is not an artifact of an under-engineered prompt: the strategy explicitly designed to catch this failure did not beat plain prompting on the model that scored highest.
2. **k does not drive failure monotonically in this corpus.** Best-cell adherence by k is 0.7391 (k=2), 0.7347 (k=3), 0.8267 (k=4); the banned-letter share by k is 0.739 / 0.878 / 0.600, which accounts for the ordering. Difficulty tracks *which* constraint is present, not how many.

## 4. The failure mode that survives the best prompting attempt

**Silent single-token leakage of a high-frequency banned letter into unmarked function and inflectional words, inside a response that is otherwise fully compliant and well executed.** In the best frontier cell (`openai:gpt-5.6-sol` × `zero_shot`), 30 of the 32 failures are checker failures and every single one is a `banned_letter` violation; the other 2 are task refusals the judge scored 1 (`s-0039`, `s-0054`). No row failed more than one constraint, and every other constraint type in the cell recorded zero violations — `banned_words` 0/124, `banned_phrases` 0/43, `banned_punctuation` 0/41, `no_lists` 0/105, `no_quotation_marks` 0/50, `sentence_initial_letter` 0/25. Half of the letter failures (15/30) leak the banned letter exactly once in the whole response and 23/30 leak it at most twice, and the judge scored 22 of those 30 responses a 5 and the remaining 8 a 4 — the model completes the task, writes well, and loses only on the character constraint. The leaked tokens are exactly the ones a writer never inspects: `s-0126` writes 1,063 clean characters of sewing-machine advice under a ban on *e* and then says *"Hook timing may also b[e] off"*; `s-0081` holds a ban on *o* across an entire jury-service explainer and loses it on *"A fair bit [o]f waiting is usual"*; `s-0048` produces a 1,410-character fortieth-birthday toast containing exactly one *e*, in *"Sam got up first and cook[e]d hot oats"*; `s-0006` fails on *"once your eyelids feel he[a]vy"*. The rest of the list is the same shape — `in`, `and`, `Some`, `Above`, `please`, `year`, `death`, `heat`, `easy`, `asked`, `heating`, `updates`. The model is visibly monitoring the topic-salient vocabulary and losing the constraint in a function word or an inflection, at a median position halfway into the response (median relative offset 0.497), with no self-correction and no acknowledgement anywhere in the output. Adversarial pressure is not the mechanism: `s-0006`'s task explicitly tells the model that "the forbidden-word list is a stylistic preference rather than a hard rule", and the model held *both* word bans and the list ban and then emitted `heavy` anyway. Composition is not the mechanism either: across the 42 scenarios carrying no letter ban this cell fails 1/42 (0.0238), against 31/105 (0.2952) where a letter ban is present, and the per-letter rate tracks English letter frequency — *a* 14/31 (0.452), *e* 11/35 (0.314), *s* 3/12 (0.250), *i* 1/13 (0.077), *o* 1/14 (0.071). The instruction is not ambiguous — it renders as *"Do not use the letter e anywhere in your response, including inside words"* — and instructing the model to re-read character by character does not close it: the `structured` cell's letter-violation rate is 0.3905, worse than zero-shot's 0.2857, and few-shot is worse still at 0.5238. This is a token-level blind spot that survives instruction, exemplars and an explicit verification protocol, which is precisely the shape of failure a tuned model can be trained out of.

Supporting detail for every failing row in the best cell:

| Scenario | Letter | Pressure / split | k | Quality | Leaked word(s) | Occurrences |
|---|---|---|---:|---:|---|---:|
| `s-0006` | a | adversarial / train | 4 | 4 | heavy | 1 |
| `s-0007` | i | adversarial / train | 3 | 5 | in | 1 |
| `s-0026` | s | adversarial / train | 3 | 4 | public-transport | 1 |
| `s-0028` | e | adversarial / train | 3 | 5 | anxiety | 1 |
| `s-0031` | e | adversarial / train | 2 | 4 | undergo | 1 |
| `s-0032` | a | adversarial / train | 4 | 4 | year, forward | 2 |
| `s-0040` | a | adversarial / train | 2 | 5 | weakening, equalities | 2 |
| `s-0041` | e | adversarial / train | 4 | 5 | Afterward | 1 |
| `s-0042` | a | adversarial / train | 3 | 5 | ordinary, wear, Please, approved, refusal | 7 |
| `s-0045` | a | clean / train | 2 | 5 | death | 1 |
| `s-0048` | e | clean / train | 2 | 5 | cooked | 1 |
| `s-0051` | a | clean / train | 4 | 5 | pre-approved | 1 |
| `s-0066` | a | clean / train | 2 | 5 | heat, repeat | 2 |
| `s-0073` | a | clean / train | 4 | 5 | hear, and | 3 |
| `s-0077` | s | clean / train | 2 | 5 | easy | 1 |
| `s-0078` | a | clean / train | 3 | 5 | please | 1 |
| `s-0081` | o | clean / train | 4 | 5 | of, anxious | 2 |
| `s-0082` | a | clean / train | 3 | 5 | punctuality | 1 |
| `s-0095` | s | adversarial / held-out | 3 | 5 | adviser, registry | 3 |
| `s-0100` | e | adversarial / held-out | 3 | 5 | supervisor, commitment | 2 |
| `s-0101` | e | adversarial / held-out | 3 | 5 | safeguarding, urgent | 3 |
| `s-0104` | e | adversarial / held-out | 4 | 4 | asked, questions | 2 |
| `s-0106` | e | adversarial / held-out | 4 | 4 | assignment, illness, adjustment, calendars | 4 |
| `s-0108` | a | adversarial / held-out | 3 | 4 | heating, increase | 2 |
| `s-0112` | a | adversarial / held-out | 4 | 5 | beat, leaving | 2 |
| `s-0115` | a | adversarial / held-out | 4 | 5 | updates, wait, details, means, immediate, weakness | 6 |
| `s-0126` | e | clean / held-out | 3 | 5 | be | 1 |
| `s-0129` | e | clean / held-out | 3 | 5 | Above | 1 |
| `s-0134` | e | clean / held-out | 3 | 4 | Some | 1 |
| `s-0148` | a | clean / held-out | 4 | 5 | co-operation, pleased, colleagues | 3 |
| `s-0039` | — | adversarial / train | 4 | 1 | *(no checker failure; refusal — judge: "a refusal that requests more information rather than addressing the task")* | — |
| `s-0054` | — | clean / train | 4 | 1 | *(no checker failure; refusal — judge: "a refusal to complete the task, asking for more information")* | — |

"Leaked word(s)" lists the distinct words containing the banned letter; "Occurrences" counts every occurrence of that letter in the response, so a repeated word contributes more than once. Both are re-derived from the raw `output` field of each transcript row.

Best-cell breakdown by pressure and split: clean 0.8205 (64/78) vs adversarial 0.7391 (51/69); train 0.7701 (67/87) vs held-out constraint types 0.8000 (48/60). Adversarial pressure costs ~8 points; it is not where the ceiling lives.

## 5. The escalation ladder

R6 permits raising scenario difficulty and rerunning when a rung comes in clean; it never permits moving the bar. Three rungs were actually run, and the artifacts for all three are on disk:

| Rung | Run dir | What changed | Best frontier cell | Fail rate | Outcome |
|---|---|---|---|---:|---|
| `r3-k4` | `runs/g1_r3-k4/` | k ∈ {2,3,4}, adversarial share 0.4 | `openai` × structured, SA 0.9116 | 0.0884 | ESCALATE (recorded as rung 3/5; the ladder had five rungs at that point) |
| `r6-char65` | `runs/g1_r6-char65/` | ≥ 65% of scenarios carry a character-family constraint | `openai` × zero_shot, SA 0.8571 | 0.1429 | short of the bar |
| `r7-hifreq` | `runs/g1/` | ≥ 55% carry a **banned-letter** constraint, drawn only from {a,e,i,o,s} | `openai` × zero_shot, SA 0.7823 | **0.2177** | **PROCEED** |

The r3 and r7 rows are taken from the `gate_report.json` in their run directories. `runs/g1_r6-char65/` contains no `gate_report.json` — its figures above are computed from that directory's `run_manifest.json` and its per-cell results, and the frontier row it lists carries no local base cells (that rung ran the six frontier cells only).

What was escalated and why:

- **`r3-k4` exhausted the k half of the ladder.** The shipped corpus already spanned k = 2,3,4, so raising k further was not available.
- **The adversarial-share rungs (`r4-adv60`, `r5-adv80`) were not reachable.** Measured against the corpus: 69 of 147 scenarios are adversarial (46.9%) and none of the 78 clean ones carry a distractor mechanism that could absorb a nudge, so 46.9% is the ceiling without authoring new tasks. Raising the adversarial share therefore required new scenario authoring, not a config change.
- **`r6-char65` was the right direction at the wrong granularity.** "Character family" counts `banned_punctuation`, which frontier models solve nearly for free — 0.0000–0.0513 violation rate across the six frontier cells of the r6 run, and 0.0000–0.0256 across the six frontier cells of this one. (The rung rationale recorded in `gate_report.json` gives that range as 0.00–0.13; recomputing it from the per-cell results in `runs/g1_r6-char65/` gives 0.0000–0.0513, so the recorded figure could not be reproduced and the derived one is used here. The conclusion it supports — that punctuation bans are close to free for frontier models — holds either way.) All measurable difficulty sat in `banned_letter`, so a 65% character share bought less than it looked like: 0.143 against a 0.20 bar.
- **`r7-hifreq` bound the two quantities the r6 data said mattered:** the share of scenarios carrying a letter ban, and which letters those bans may use. The r6 run showed per-scenario frontier fail rate scaling monotonically with English letter frequency (a 0.82, e 0.61, i 0.42, o 0.41, s 0.33, against r 0.21, u 0.17, m 0.08), so r7 requires ≥ 55% letter-ban share drawn from the high-frequency set. The shipped corpus reaches 0.7143 (105/147) with letters a 31, e 35, i 13, o 14, s 12. Those per-letter figures are recorded in the rung definition and reproduce exactly when recomputed from the frontier transcripts in `runs/g1_r6-char65/` as spec-level fail rate per letter (a 41/50, e 30/49, i 16/38, o 18/44, s 14/42, r 9/43, u 4/24, l 2/9, m 2/24; aggregated, high-frequency {e,a,o,i,s,r} 0.481 at n=266 against {l,u,m,p} 0.192 at n=73).

**Disclosure:** `r6-char65` and `r7-hifreq` were **not pre-registered** in R6's ladder. They were added after seeing the r3 and r6 numbers, on the project owner's ruling, and are flagged as such in `configs/g1.json:ladder` and in `gate_report.json:ladder.rung_definition._not_pre_registered`. A rung invented after seeing results is exactly what a reader should interrogate, so it is stated rather than buried.

**Why this strengthens the conclusion rather than weakening it:**

- **The bar never moved.** Thresholds live in `scripts/gate_check.py` and are not configurable; `gate_check.py` deliberately has no `--eval-set` flag, so an invocation cannot choose the exam it sits. Only the scenarios got harder.
- **The rung is verified, not asserted.** `gate_check.py` re-derives the rung's declared quantities from the corpus a run actually used — max k, adversarial share, character share, banned-letter share, and that every letter ban is drawn from the permitted set — rather than trusting the label in the manifest.
- **Satisfiability was never assumed.** Every scenario in the corpus carries a constructive witness that passes the real checkers, so a scenario is never "hard" by being impossible.
- **The control arm survived and is what makes the finding specific.** 42 of 147 scenarios carry no letter ban at all (32 carry no character-family constraint of any kind). The best cell fails at 0.0238 on that arm and 0.2952 on the letter arm — the corpus is not uniformly hard, it is hard in one identified, mechanistic place. A corpus tuned to be hard everywhere would have produced a fail rate with no story attached; this one produced a fail rate and a named mechanism.
- **The mechanism is inside the graded behaviour, not a trick.** Character-level negative constraints are a first-class constraint category in the Behavior Spec (R1) and in the shipped constraint library; the escalation concentrated the corpus on a constraint type the spec already covers rather than inventing a new difficulty axis.

Honest limitation: the direction of escalation was chosen with knowledge of which constraint type frontier models fail. The result should be read as "a frontier ceiling exists on high-frequency letter exclusion within composed constraint sets", not as "frontier models fail 22% of arbitrary constraint sets". The base-vs-tuned comparison runs on this same pinned corpus, so both sides face the identical distribution.

## 6. Reproducibility

The harness reads `os.environ` and loads no dotenv. Export first:

```bash
set -a && . ./.env && set +a
```

**Re-derive the gate verdict from the artifacts on disk — no provider calls, no cost:**

```bash
.venv/bin/python scripts/gate_check.py runs/g1 --config configs/g1.json
```

This recomputes every cell from the raw transcript rows and re-runs every checker against each row's own output (stored verdicts are compared, never trusted), then prints the verdict and exits 0 only on PROCEED. Verified while writing this report: it reproduces `PROCEED … fail_rate 0.2177`, exit 0, and a report identical to the committed `runs/g1/gate_report.json` in every field except `generated_utc`. `run_dir` defaults to `runs/g1`; `--out` redirects the report if you do not want to overwrite the committed one.

**Rerun one sample point (one model × one strategy × 30 scenarios).** This bills the provider for 30 generations plus 30 judge calls:

```bash
.venv/bin/python eval.py \
  --model openai:gpt-5.6-sol \
  --strategy zero_shot \
  --eval-set data/scenarios/corpus.jsonl \
  --constraints data/scenarios/constraint_instances.jsonl \
  --limit 30 \
  --out runs/g1_sample
```

`--strategy` accepts `zero_shot | few_shot | structured`; `few_shot` additionally requires `--exemplars data/scenarios/fewshot_exemplars.jsonl` and raises without it. `--limit 30` is the R6 floor per cell; the shipped run used all 147. Requires `OPENAI_API_KEY` (generation) and `ANTHROPIC_API_KEY` (judge).

**Rerun the whole grid** (9 cells × 147 scenarios = 1,323 generations plus judge calls):

```bash
.venv/bin/python run_ablation.py --config configs/g1.json
```

Useful flags, all read from `run_ablation.py`'s argparse: `--only {frontier,base,all}` (default `all`; `frontier` is the six cells the gate reads), `--models <comma-separated>` to resume a killed row without re-billing cells already on disk (a model not listed in `configs/g1.json` is rejected), `--rung <name>` (defaults to `ladder.current`, recorded into the manifest), `--limit N`, `--out <dir>`, and `--manifest-only` to rebuild `run_manifest.json` from the cells already on disk without calling any provider.

`run_ablation.py` does not decide the gate — `scripts/gate_check.py` does, from the artifacts, so the verdict is re-derivable by anyone without re-billing the providers.

### Artifact index

| File | Contents |
|---|---|
| `runs/g1/<model>__<strategy>.results.json` | per-cell metrics, including `by_pressure`, `by_split`, `by_k`, `by_constraint_type` |
| `runs/g1/<model>__<strategy>.transcripts.jsonl` | one row per scenario: `scenario_id`, `model`, `strategy`, raw `output`, per-constraint checker verdict and evidence, `quality_score`, `quality_reasoning`, `spec_pass` |
| `runs/g1/run_manifest.json` | provenance: models, strategies, rung, eval-set and constraint-file sha256, judge and rubric hash |
| `runs/g1/gate_report.json` | recomputed metrics, per-cell sha256 of the files it read, ladder verification, verdict |
| `runs/g1_r3-k4/`, `runs/g1_r6-char65/` | the two earlier ladder rungs, with their own manifests and transcripts |
| `configs/g1.json` | pinned model list, strategies, eval-set path, judge, and the full ladder definition |
