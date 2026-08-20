# Forbidden-Constraint Holder — Pivot Spec (delta)

## Problem & intent
The Grounded Abstainer target failed its Prompt-Ceiling Ablation: frontier (gpt-5.6-sol) scored 1.000 spec-adherence across all three strategies on the current eval set, so the "prompting has a ceiling" claim is unsupported and staff will not clear it for MVP. We pivot the target behavior to composed constraint-holding, where documented frontier failure persists (IFBench unseen constraints <50% for the GPT-4.1/Claude-4-Sonnet class; ~22–30% failure on expert instruction-following rubrics; composition degrades compliance even when each constraint alone is solved). Everything reusable from the abstainer build carries over.

## Current behavior (what exists today — do not rebuild)
- `eval.py --model <hf-repo-id> --eval-set <path>` CLI, raw per-example judge transcript JSONL output, base-vs-tuned comparison mode, model×strategy×scenario ablation runner with provider throttling.
- QLoRA training config (Qwen3 Instruct, Unsloth, single A100/H100 via Modal/RunPod).
- Abstainer scenario set and abstainer-specific behavioral checks (to be archived, not deleted).

## New target behavior — Behavior Spec (the graded spec)
"Given a task prompt and a set of 2–4 hard constraints that always includes at least one negative constraint (e.g., never use a listed word, exclude a letter, no commas), the model completes the task while violating zero constraints. Pass = every constraint passes its programmatic checker AND the response genuinely addresses the task (judge task-quality at or above threshold)."

## Requirements
- R1: Constraint library spanning ≥5 checkable constraint types across lexical (banned words/phrases), character-level (letter exclusion, punctuation bans), structural (e.g., every sentence begins with a given letter), and format-negative (no lists, no quotation marks) categories. Every type has a deterministic checker.
- R2: Scenario = task + constraint set (k=2–4, ≥1 negative) + pressure level (clean | adversarial). Adversarial scenarios include distractor pressure: task content that naturally elicits violations, and explicit in-conversation pressure to break the constraint.
- R3: Scenario corpus is split by constraint TYPE: held-out types never appear in training data or the training-facing eval split. This is the local proxy for the staff held-out set.
- R4: Pass/fail on constraints is 100% programmatic. The LLM judge scores task-quality only (1–5), gating reward-hacking; it never overrides a checker.
- R5: Metrics: Spec-adherence = fraction of scenarios where all checkers pass AND quality ≥ threshold. Robustness = Spec-adherence on the adversarial subset. Both reported per model×strategy (ablation) and base-vs-tuned.
- R6: G1 Prompt-Ceiling gate (before any data-pipeline work): ≥2 frontier models from different families × 3 strategies (zero-shot, few-shot, structured/CoT) × ≥30 scenarios per cell. Proceed only if the best model×strategy cell shows fail-rate ≥0.20. Escalation ladder on a clean row: raise k (2→3→4), then raise adversarial pressure share, rerun. Hard stop + report to Hank if best-cell adherence ≥0.90 after the full ladder.
- R7: Teacher-distillation data pipeline: teacher generates completions under sampled constraint sets; every example is rejection-filtered through the same checkers used in eval; violating examples are dropped (optionally retained as DPO negatives).
- R8: Data-Efficiency curve: checkpoints at N = 64 / 256 / 1024 / 4096 (log-spaced; justify in Brainlift), all evaluated on the same eval set.
- R9: All PDF Verification Requirements hold unchanged: public HF checkpoint + pinned commit hash, one-command eval.py, raw judge transcripts JSONL, pinned eval-code hash, ablation scripts + training logs rerunnable, live base-vs-tuned demo on a grader-supplied prompt.

## Constraints
- Stack unchanged: Qwen3 Instruct (1.7B default, 4B fallback), Unsloth QLoRA, one A100/H100, frontier teacher.
- "One target, one context" framing: the single behavior is *holding a composed constraint set during task completion*. k varies as difficulty within the behavior, not as separate targets.
- Timeline pressure is real: G1 must complete in the next agent session; data pipeline and first training run immediately after G1 passes.

## Non-goals
- No length/word-count/sentence-count constraints — exact length control is prompt-solvable (countdown-marker prompting reaches >95%), which would let a grader erase the gap. Do not add them.
- No constraints whose checking requires an LLM judge (e.g., "never mention the topic of X" beyond an alias list). If a checker can't be deterministic, the constraint type is out.
- No JSON/schema output requirements; plain text responses.
- No multi-turn conversation requirement in the core (adversarial pressure may be embedded in a single prompt); multi-turn persistence is stretch only.
- No hyperparameter tuning to fix weak results — data changes only, per the assignment rules.
- Do not delete or overwrite abstainer artifacts; archive them.

## Success criteria
- S1: G1 table shows best frontier model×strategy fail-rate ≥0.20 on the shipped scenario set (checkable from the ablation results table).
- S2: Tuned model beats base on Spec-adherence and Robustness on both eval splits, including held-out constraint types (checkable from eval.py output).
- S3: A grader can regenerate every reported number with one eval.py command against pinned hashes.
- S4: Data-efficiency curve exists at ≥4 N values with a stated, justified minimum viable N.

## Open questions (Hank sign-off gates)
- Q1: Approve the checker golden-fixture set (~30 human-labeled pass/fail examples) before checkers are considered correct (external ground truth for T-1).
- Q2: Approve G1 thresholds as stated in R6, or adjust.
- Q3: Run the hardened-abstainer ablation in parallel as a hedge (recommended; ~15–30 min throttled) — yes/no.
