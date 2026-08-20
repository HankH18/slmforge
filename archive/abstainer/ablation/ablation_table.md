# Prompt-Ceiling Ablation — rubric v1.0 (hash `c0adb876c3e6c4c9`), n=32 common scenarios, judge=claude-sonnet-4-6

Cell = mean Spec-adherence / Robustness (adversarial subset).

| model | zero_shot | few_shot | structured |
|---|---|---|---|
| `openai:gpt-5.6-sol` | 100.0% / 100.0% | 100.0% / 100.0% | 96.9% / 93.8% |
| `google:gemini-3.1-pro-preview` | 100.0% / 100.0% | 100.0% / 100.0% | 100.0% / 100.0% |
| `hf:Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | 53.1% / 56.2% | 87.5% / 87.5% | 53.1% / 56.2% |

## Failure mode that survives the best prompting attempt

_TODO: written after inspecting transcripts — name the specific
failure mode, with scenario IDs as evidence._
