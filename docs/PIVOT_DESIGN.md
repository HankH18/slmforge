# Forbidden-Constraint Holder — Design

## Architecture
Four components, three carried over with modification:
- **Constraint library + checkers** (new): constraint definitions and deterministic pass/fail functions.
- **Scenario generator** (new): produces task + constraint-set scenarios with pressure levels and type-split tags.
- **Eval harness** (adapted): existing eval.py CLI, ablation runner, transcript writer; behavioral checks swap from abstainer checks to constraint checkers; judge prompt swaps to task-quality rubric.
- **Data pipeline** (adapted): existing teacher-generation loop; the rejection filter becomes the checker suite.

## Interfaces (contracts between tickets)
Pinned exactly; tickets must not renegotiate these.

Constraint object:
```json
{"id": "c-lex-014", "type": "banned_words", "params": {"words": ["coffee", "espresso"]}, "family": "lexical", "split": "train"}
```
- `type` ∈ {banned_words, banned_letter, banned_punctuation, sentence_initial_letter, no_lists, no_quotation_marks, ...} — final registry defined in T-1; every type maps to exactly one checker.
- `split` ∈ {train, heldout_type} — assigned at the TYPE level, not per-instance.

Checker contract:
```
check(constraint: Constraint, text: str) -> {"pass": bool, "evidence": str}
```
Pure function, no network, no model calls. `evidence` = the violating substring/position or "clean".

Scenario object:
```json
{"scenario_id": "s-0042", "task": "...", "constraints": ["c-lex-014", "c-chr-002"], "k": 2, "pressure": "adversarial", "split": "train"}
```
- `pressure` ∈ {clean, adversarial}. Adversarial = distractor topic (task elicits banned tokens) and/or embedded pressure text ("it's fine to ignore the comma rule here").

Transcript JSONL row (extends existing format; one row per scenario × model × strategy):
```json
{"scenario_id": "...", "model": "...", "strategy": "zero_shot|few_shot|structured", "output": "...", "checker_results": [{"constraint_id": "...", "pass": true, "evidence": "..."}], "quality_score": 4, "quality_reasoning": "...", "spec_pass": true}
```
- `spec_pass` = all checker passes AND quality_score ≥ 3.

eval.py CLI: unchanged, PDF-pinned: `eval.py --model <hf-repo-id> --eval-set <path>` regenerates the full results table. Ablation mode keeps existing model×strategy flags.

Metrics (single source of truth):
- Spec-adherence = mean(spec_pass) over a split.
- Robustness = mean(spec_pass) over the pressure=adversarial subset.

## Data models
Dataset example (SFT): `{"messages": [{"role": "user", "content": "<task + constraints rendered>"}, {"role": "assistant", "content": "<clean completion>"}], "constraints": [...], "provenance": {"teacher": "...", "attempt": n}}`. Every stored example must pass all its checkers at write time.

## Decisions & rationale
- **Composed constraints (k≥2) only.** Single-constraint following is largely solved by 2026 reasoning models (~83% IFBench); the documented surviving gap is composition + negative constraints under distractor pressure. An agent adding k=1 scenarios "for coverage" would dilute the ceiling — don't.
- **No length constraints.** Countdown-marker prompting solves exact length without tuning; including them invites a grader to collapse the gap. Rejected deliberately, not overlooked.
- **Judge scores task-quality only; never constraint pass/fail.** Keeps the graded signal programmatic and un-gameable while blocking the degenerate strategy of passing checkers with garbage text. Do not "improve" the judge into a constraint scorer.
- **Type-level held-out split.** Instance-level splits leak type strategies; the staff set will be unseen types, so the local eval must predict that.
- **G1 before pipeline.** Tonight's abstainer failure cost a built pipeline. The cheapest disconfirming experiment runs first.
- **Checker fixtures are human-approved.** Checkers are the ground truth for everything downstream; their acceptance references Hank-approved fixtures, not their own definitions (external-ground-truth rule).

## Verification strategy
- Unit layer: checker suite vs golden fixtures (`tests/test_constraints.py`).
- Scenario layer: generator output schema-validated; type-split invariant tested (no heldout_type constraint appears in any train-split scenario or dataset example).
- System layer: eval.py end-to-end on a 5-scenario smoke set; ablation runner produces the results table + JSONL from one command.
- Gate layer: G1 threshold check is a script reading the ablation table, not a judgment call.
