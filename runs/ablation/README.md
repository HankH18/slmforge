# SUPERSEDED — this is the RETIRED abstainer ablation

**This directory is not the project's Prompt-Ceiling Ablation.** It grades the
retired *grounded abstainer* behaviour — answer only from a supplied passage, or
refuse — which the project pivoted away from. Its metrics (`false_answer_rate`,
`false_refusal_rate`, `nonverbatim_citation`) belong to that behaviour and have no
meaning for the graded one, and it was run over `data/eval/scenarios_v1.jsonl`,
which is not the graded eval set.

Its own conclusion was that the gate **failed**: the frontier models saturated the
task, so no prompting ceiling was found. That result is what caused the pivot.

**The Prompt-Ceiling Ablation that counts is [`docs/ABLATION_G1.md`](../../docs/ABLATION_G1.md)
over [`runs/g1/`](../g1/)** — two frontier families x three prompting strategies x
147 scenarios, scored on the graded Behavior Spec with the same judge rubric the
base-vs-tuned comparison uses. Re-derive it with:

```bash
python scripts/gate_check.py runs/g1 --config configs/g1.json
```

Kept, rather than deleted, because it is the evidence for why the target changed.
A byte-identical copy also lives at `archive/abstainer/ablation/`.
