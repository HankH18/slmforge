# Claude Code Kickoff — Pivot to Forbidden-Constraint Holder

The Grounded Abstainer target failed its Prompt-Ceiling Ablation (frontier scored 1.000 across all strategies), so the project pivots to a new target behavior: composed constraint-holding. This handoff contains everything you need. Hank has signed off on the pivot package; the two standing human gates below still apply.

## Documents (read in this order)
1. `PIVOT_SPEC.md` — intent, requirements (R1–R9), non-goals, success criteria. The Behavior Spec in §New target behavior is the graded spec.
2. `PIVOT_DESIGN.md` — pinned interfaces (constraint/checker/scenario/transcript schemas, metrics definitions) and decisions with rationale. Do not renegotiate pinned interfaces or "improve away" the documented decisions.
3. `PIVOT_TASKS.md` — the task graph (T-0 … T-7) with acceptance criteria and verify commands.
4. The original assignment PDF remains the authority on submission requirements; the spec mirrors it, but if you find a conflict, stop and flag it rather than resolving it silently.

## Execution rules
1. Work the graph, not the docs: pull ONE ready ticket (all dependencies closed), load only its contract plus the files in its scope. Do not preload the whole graph.
2. A ticket closes only when its Verify command passes. No self-judgment substitutes for a passing check.
3. Human gates — hard stops, no exceptions:
   - T-1: present the checker golden fixtures to Hank and wait for approval before closing the ticket.
   - T-4: deliver the full G1 ablation table, gate verdict, and raw transcripts to Hank and wait for his go before starting T-5. If the escalation ladder (R6) exhausts without a ceiling appearing, stop and report — do not soften thresholds, do not proceed.
4. On a failed attempt: revert to the ticket's starting state before retrying. Do not iterate on top of your own failure.
5. On discovered reality (a plan assumption is wrong): stop, report what broke and the smallest re-plan, and wait — do not push through a stale plan.
6. Order is the point of this pivot: T-4 (the prompt-ceiling gate) runs BEFORE any data-pipeline or training work. Last night's failure mode was building downstream of an unproven ceiling; do not recreate it.
7. Data beats config: if eval numbers disappoint at any stage, the fix is a data change, never a hyperparameter change (assignment rule; Spec §Constraints).
8. Keep every reported number regenerable: pinned model hash, pinned eval-code hash, one-command eval.py, raw judge transcripts saved as JSONL for every eval you run.

## Reporting style
- Direct and terse. Lead with the number or the blocker, not narrative.
- All time estimates in agentic-execution terms (runs/sessions), not human hours.
- When you need a decision from Hank, present it as short concrete options, not open-ended prose questions.

## First actions this session
1. T-0 (archive + smoke).
2. T-1 (constraint library + checkers) → fixture sign-off gate.
3. T-2, T-3.
4. T-4 G1 ablation → deliver table to Hank → hold at the gate.
