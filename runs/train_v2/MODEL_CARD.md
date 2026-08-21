---
license: apache-2.0
base_model: Qwen/Qwen3-1.7B
library_name: transformers
pipeline_tag: text-generation
datasets:
  - HankH18/forbidden-constraint-holder-sft
tags:
  - qlora
  - peft
  - constrained-generation
  - instruction-following
---

# Qwen3-1.7B — Forbidden-Constraint Holder

A QLoRA fine-tune of **Qwen/Qwen3-1.7B** trained to *hold negative
constraints* while completing an ordinary writing task — "write this, and
never use the letter e / the word 'however' / any quotation marks" —
including when the task itself pulls toward the banned token.

Every constraint has a deterministic checker, so adherence here is
**measured, not judged**: an LLM judge scores task quality only and can
never overturn a checker.

> **Read the results section before using this model.** It improves on the
> constraint families it was trained on and *regresses* on families it was
> not. That trade is the finding, and it is stated in full below.

## Provenance

| | |
|---|---|
| Base | `Qwen/Qwen3-1.7B` |
| Base revision (pinned) | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| Method | QLoRA (NF4 + double quant), LoRA adapter merged to fp16 |
| Hardware | NVIDIA A10, 10.11 GiB peak |
| Training wall clock | 740 s |

This repo holds the **merged fp16 model** at the root and the **LoRA
adapter** under `adapter/`.

## Training data

| | |
|---|---|
| Dataset | `data/train/v2.jsonl` |
| Rows | 846 |
| sha256 | `9d9fc3f32d522f5cd873ae9af6629e09538f642d26cc652b964f260aff0105c8` |
| Published at | [`HankH18/forbidden-constraint-holder-sft`](https://huggingface.co/datasets/HankH18/forbidden-constraint-holder-sft) |

Teacher-distilled and **rejection-filtered through the same deterministic
checkers used at evaluation**: a generated completion that violated its own
constraint set was discarded rather than trained on. Constraint types held
out for generalization testing appear in zero training rows.

## Hyperparameters

| | |
|---|---|
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Target modules | down_proj, gate_proj, k_proj, o_proj, q_proj, up_proj, v_proj |
| Learning rate | 0.0002 |
| Epochs | 3 |
| Effective batch size | 16 (2 x 8 accumulation) |
| Max sequence length | 2048 |
| Optimizer / schedule | adamw_8bit / linear, warmup ratio 0.03 |
| Seed | 42 |
| Steps | 159 |
| Training loss | 5.7666 -> 2.0033 |

Loss is computed on the completion only; prompt tokens are masked to -100.

### Prompt format

Use the chat template with a **single user turn and no system message**.
Training used no system turn, and every training prompt was re-rendered
through the evaluation harness's own renderer and compared byte-for-byte
before training began, so the training condition equals the evaluation
condition. Adding a system prompt puts the model off-distribution.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "HankH18/qwen3-1.7b-forbidden-constraint-holder"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, dtype="auto", device_map="auto")

messages = [{"role": "user", "content": "Write a short paragraph about the ocean. Do not use the letter 'e' anywhere in your response."}]
inputs = tok.apply_chat_template(messages, add_generation_prompt=True,
                                 enable_thinking=False, return_tensors="pt").to(model.device)
print(tok.decode(model.generate(inputs, max_new_tokens=512, do_sample=False)[0][inputs.shape[-1]:]))
```

`enable_thinking=False` matters: the training rows were rendered with it
off, and Qwen3's template emits a thinking block otherwise.

## Results

Eval set `data/scenarios/corpus.jsonl`, judge `claude-sonnet-4-6`, rubric hash `12a0dcaaf29080cb`.
`spec_pass` = every checker passes **and** judge task-quality >= 3.
`robustness` is the same quantity over the adversarial subset only.

Scored rows: **base 146, tuned 147**. The denominators differ because 1 row(s) were quarantined — the judge returned no score, so there is no verdict to count either way. Rates are over scored rows.

| Metric | base | tuned | delta |
|---|---|---|---|
| `spec_adherence` | 0.0890 | 0.0816 | -0.0074 |
| `robustness` | 0.1029 | 0.0870 | -0.0160 |

Violation rate per constraint type — **lower is better**. Types marked
*held out* appear in **zero** training rows.

| Constraint type | in training data | base | tuned |
|---|---|---|---|
| `banned_letter` | yes | 1.000 | 0.990 |
| `banned_phrases` | yes | 0.000 | 0.000 |
| `banned_punctuation` | no (held out) | 0.220 | 0.585 |
| `banned_words` | yes | 0.113 | 0.056 |
| `no_lists` | yes | 0.019 | 0.019 |
| `no_quotation_marks` | yes | 0.000 | 0.000 |
| `sentence_initial_letter` | no (held out) | 0.280 | 0.960 |

## Limitations, stated plainly

* **It does not reliably hold a banned letter.** That is the hardest
  constraint in the set and the checker requires *zero* occurrences. The
  tuned model reduces how often the banned letter appears — the median
  response goes from 34 occurrences to 14 — but a response with 14 is
  still a failure, and the pass rate stays near the floor.
* **It gets worse at constraint families it never saw.** Compare the
  held-out rows in the table above. Fine-tuning bought compliance on the
  trained families and spent generalization on the untrained ones. If your
  constraint type is not in the training list, the base model is the better
  choice.
* **Single user turn, no system prompt.** See the prompt format above.
* 1.7B parameters, English only, short-form writing tasks.

## Reproducing these numbers

```bash
python eval.py --model hf:HankH18/qwen3-1.7b-forbidden-constraint-holder@<commit> \
               --compare-with hf:Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
               --eval-set data/scenarios/corpus.jsonl
```

The harness, the checkers, the scenario corpus and the raw judge
transcripts are all in the project repo.
