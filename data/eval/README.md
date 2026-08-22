# `data/eval/` — retired eval sets

**The graded eval set is not here.** It is
[`data/scenarios/corpus.jsonl`](../scenarios/corpus.jsonl) (147 scenarios,
sha256 `0ad00c19…`), with its constraint library in
`data/scenarios/constraint_instances.jsonl`.

Everything in this directory belongs to the retired *grounded abstainer* target:

| File | What it was | Status |
|---|---|---|
| `scenarios_v1.jsonl` | 32 abstainer scenarios; the set `runs/ablation/` was scored on | Retired. **Pinned in place** by `tests/test_repo_hygiene.py` (`test_frozen_inputs_are_at_their_contract_paths`) — moving it turns the suite red. |
| `fewshot_exemplars.jsonl` | Few-shot exemplars for that target | Retired |
| `scenarios_v2_hardened.jsonl` | The abstainer "hedge" set | Retired, and **a known trap** — see below |

`configs/g1.json` records why the last one matters: the G1 ablation config
previously named `scenarios_v2_hardened.jsonl`, and because the file exists on
disk an existence check passed while the run would have scored grounded-abstention
scenarios with constraint checkers. Nothing references it today. If you are
pointing any harness at a file in this directory, you are pointing it at the wrong
target.

The retired track is preserved whole under [`archive/abstainer/`](../../archive/abstainer/),
including a runnable copy of its harness and tests.
