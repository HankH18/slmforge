# Abstainer track (superseded)

The original target: given a question and a context passage, answer only if the
passage supports it — one `Answer:` line plus verbatim `Citation:` lines — else
emit exactly `Not answerable from the provided context.`

**Why it was retired.** It failed its own Prompt-Ceiling Ablation. Both frontier
families solved it cold, so the "prompting has a ceiling" claim could not be
supported and the target could not clear G1. See `ablation/ablation_table.md`;
the numbers are summarised in commit `dd7ad0d`.

**What is here.** Pristine copies, not the live tree:

| Path | What |
|---|---|
| `eval.py`, `run_ablation.py` | The abstainer harness at rubric `c0adb876c3e6c4c9`. The live root `eval.py` diverges from this once the pivot adapts it. |
| `scenarios_v1.jsonl` | 32 abstainer scenarios (13 answerable / 19 unanswerable, 16 adversarial) |
| `fewshot_exemplars.jsonl` | 6 few-shot exemplars |
| `smoke.jsonl` | 5-scenario mini-set, one per taxonomy type |
| `ablation/` | All 9 cells: results JSON + raw judge transcripts, plus table and summary |

**Reproducing the archived numbers.** Use the archived harness, not the live one
— after the pivot they score different behaviors:

```bash
python archive/abstainer/run_ablation.py \
  --models "openai:gpt-5.6-sol,google:gemini-3.1-pro-preview,hf:Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e" \
  --strategies zero_shot,few_shot,structured \
  --eval-set archive/abstainer/scenarios_v1.jsonl \
  --exemplars archive/abstainer/fewshot_exemplars.jsonl \
  --judge anthropic --judge-model claude-sonnet-4-6 --workers 2
```

`--workers 2` is required: `gemini-3.1-pro` is capped at 25 requests/minute and
8 workers exhausts the quota mid-run. Export `.env` first — the harness reads
`os.environ` directly and loads no dotenv itself.

**The result worth carrying forward.** Qwen3-1.7B zero-shot scored 53.1%, below
the 59.4% floor a model gets for refusing everything, at false-refusal rate 0.0%.
The small model knew when to abstain; it could not copy spans verbatim. The
deficit was citation fidelity, not judgment.
