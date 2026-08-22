---
title: Forbidden-Constraint Holder
emoji: 🚫
colorFrom: gray
colorTo: indigo
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
short_description: Base vs tuned Qwen3-1.7B on your own forbidden constraints
---

# Forbidden-Constraint Holder — base vs tuned, on your prompt

Type your own task, type or pick your own forbidden constraints, press Run. Both
models answer the same prompt, and each answer is scored live by this project's
pinned programmatic checkers. Nothing on this page is pre-selected and nothing is
cached.

## What is running

Both commits are read from `configs/models.json` when the app starts and printed
at the top of the page. They are not written into the application code — see
`pins/PINS_MANIFEST.txt` for the exact bytes of the project modules this Space is
carrying, and the commit of the source repo they came from.

| | |
|---|---|
| base | `Qwen/Qwen3-1.7B` at the pinned revision |
| tuned | `HankH18/qwen3-1.7b-forbidden-constraint-holder` at the pinned revision |
| strategy | `zero_shot` — no system message, the training condition |
| `enable_thinking` | matches `configs/train.json:data.enable_thinking`, cross-checked against `eval.py` at startup |
| `max_new_tokens` | defaults to 512, the evaluated budget |

The `enable_thinking` cross-check is a real boundary, not a note: if the training
config and `eval.py` disagree, the app raises on import instead of generating
under a template condition the published numbers were not measured under.

## This is slow, and the reason is boring

There is no GPU on the free CPU tier. Qwen3-1.7B generating 512 tokens on CPU
takes an order of minutes, and this comparison runs two models one after the
other rather than holding both in memory — the same one-model-at-a-time rule
`scripts/demo.py` follows, for the OOM reason recorded in its docstring. The
first run also downloads roughly 3.4 GB of weights per model into the Space's
cache. Elapsed seconds are printed under each answer, so the estimate is replaced
by a measurement as soon as you press the button once.

You can lower `max_new_tokens`, and the page says plainly when you have: a
truncated answer can fail a constraint it would otherwise have held, so a run
below 512 tokens is not comparable with the published numbers.

## What the verdicts are, and what they are not

Each constraint is checked by `constraints.checkers.check` — the pinned entry
point every published number in the project is re-derived from. Those verdicts
are computed live, on your text.

`spec_pass` is *all checkers pass **and** judged task quality ≥ 3*. No judge runs
here, so `harness.compute_spec_pass(checks, None)` returns `None` and the page
reports **UNDECIDED**. It does not report a pass. The page also shows what
`spec_pass` *would* be if a judge scored the answer ≥ 3, labelled as the
conditional it is.

One prompt is an anecdote either way. The measurement is 147 scenarios, written
up in `docs/FINAL_REPORT.md` in the source repo, and its headline is that
fine-tuning made the headline metric **worse** while improving the trained
constraint types and task quality. Every run prints those numbers underneath the
comparison, whichever way your prompt lands. They are rendered by the repo's own
`scripts/demo.py:format_footer()`, whose table the repo's tests recompute from
the run transcripts; this Space states no number of its own.

## Nothing here is a second implementation

The prompt renderer, the constraint parser, the checkers, the scoreboard and the
model pins are all imported from the source project (mirrored under `pins/`). A
demo that rendered its own prompt or ran its own checkers would demonstrate a
different system that happens to share a repo. `app.py` is layout; `core.py` is
the callback and imports everything load-bearing.

## Constraint syntax

One per line, the same forms as `scripts/demo.py --constraint`, validated against
`constraints/registry.py`:

```text
banned_words:<word>[,<word>...]
banned_phrases:<phrase>[,<phrase>...]
banned_letter:<single letter>
banned_punctuation:<marks, one character each, e.g. ,;>
sentence_initial_letter:<single letter>
no_lists
no_quotation_marks
```

A line that looks like a shipped constraint id (`c-lex-001`) reuses that instance
from the project's constraint library instead.
