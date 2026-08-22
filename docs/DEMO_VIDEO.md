# Demo video — runbook

The Final Submission asks for:

> "A 3–5 minute demo video showing the tuned model doing the thing the base model
> fails to do reliably, including one live, grader-supplied prompt."

and, separately, under Verification Requirements:

> "Live comparison in demo | Part of your demo video must show a grader-supplied
> prompt run live against base vs. tuned — not only pre-selected examples."

Both are satisfied by one segment: §3 below. Everything else is context around it.

---

## Before recording

`scripts/demo.py` loads **both** checkpoints — roughly 3.4 GB each — and the first
load is much slower than the rest. Warm the cache first so the recording does not
open on a progress bar:

```bash
set -a && . ./.env && set +a
python scripts/demo.py --prompt "warm the cache" --constraint banned_words:the --tuned-only
python scripts/demo.py --prompt "warm the cache" --constraint banned_words:the --base-only
```

Then check what will be on screen:

```bash
python scripts/demo.py --help          # the constraint syntax block is worth reading aloud
git rev-parse HEAD                     # the eval-code commit, if you want it in frame
```

Have these open in tabs:

- the model: <https://huggingface.co/HankH18/qwen3-1.7b-forbidden-constraint-holder>
- the dataset: <https://huggingface.co/datasets/HankH18/forbidden-constraint-holder-sft>
- the Space, once deployed (see [`space/DEPLOY.md`](../space/DEPLOY.md))
- `docs/FINAL_REPORT.md` and `docs/EFFICIENCY_CURVE.md`

---

## The shape, ~4:30

### §1 · What the behaviour is — 0:00–0:30

Read the Behavior Spec off the screen; it is two sentences and a stranger can mark
any output against it:

> Given a task prompt and a set of 2–4 hard constraints that always includes at
> least one negative constraint, the model completes the task while violating zero
> constraints. Pass = every constraint passes its programmatic checker AND the
> response genuinely addresses the task.

Say the one thing that makes the whole project falsifiable: **constraint pass/fail
is 100% programmatic** — `constraints/checkers.py`, no model call, no network. The
LLM judge scores task-quality only and can never overturn a checker; it exists to
stop the degenerate strategy of satisfying every constraint with garbage text.

### §2 · Why fine-tuning was justified at all — 0:30–1:15

Show `docs/ABLATION_G1.md`. Two frontier families × three prompting strategies ×
147 scenarios. The best prompt still leaves a real failure rate, and the failure
that survives is compositional: the model holds a constraint until the task's own
subject matter pulls against it.

Say the number the gate turned on rather than describing it. It is one command:

```bash
python scripts/gate_check.py runs/g1 --config configs/g1.json
```

### §3 · THE LIVE SEGMENT — 1:15–2:30 — *this is the graded part*

**Ask the grader for a task and a forbidden constraint, on camera, and type what
they say.** Do not pre-select it. The point of the requirement is that the input
was not chosen by you.

```bash
python scripts/demo.py --prompt "<whatever they ask for>" \
                       --constraint banned_letter:e \
                       --constraint no_lists
```

If they would rather choose the constraint themselves, `--help` prints every form:

```
banned_words:<word>[,<word>...]        banned_punctuation:<marks, e.g. ,;>
banned_phrases:<phrase>[,...]          sentence_initial_letter:<letter>
banned_letter:<single letter>          no_lists          no_quotation_marks
```

`--interactive` reads the task and the constraints from stdin, which films better
if the grader wants to type it themselves.

What to point at when the output lands, in this order:

1. **Both answers, side by side** — same prompt, same constraints, same generation
   budget (512 new tokens, the evaluated budget), same chat-template condition.
2. **The verdict table.** These are the *shipped* checkers, run live. On a
   violation the evidence shows the offending span and its offset, so nobody has to
   take the verdict on trust.
3. **The honesty footer.** It is bound to `runs/final/` by `tests/test_demo.py`, so
   the demo cannot quote numbers the committed transcripts do not support.

Two things worth saying out loud here, because a viewer will notice them anyway:

- `banned_letter` is the hard one. It is where the base model fails most and where
  the tuned model improves most.
- A single run is one sample. The claim that the difference is real comes from the
  paired test in §4, not from what is on screen.

### §4 · The headline, told honestly — 2:30–3:30

This is the part not to soften. Put `docs/FINAL_REPORT.md` on screen.

- On `spec_adherence` — the graded pass/fail metric — the tuned model **does not
  beat** the base model. Of 146 paired scenarios, 5 pass only for base and 4 only
  for tuned. McNemar exact **p = 1.000**. It is a tie the instrument cannot
  resolve, not a loss.
- On the same scenarios, counting banned-letter *occurrences* rather than
  thresholding them: median **34 → 14**, lower on 89 of 105 rows, sign test
  **p = 1.87 × 10⁻¹³**, bootstrap CI on the difference of medians **[−26, −15]**.

Then the finding the project is actually about: **`spec_adherence` requires zero
violations, so it is a cliff.** It cannot see 34 → 14. A metric that reports
p = 1.000 for a difference a paired test on the identical scenarios resolves at
p ≈ 10⁻¹³ is measuring the threshold, not the behaviour.

Re-derive it live if there is time — it reads from committed transcripts, so it is
fast and needs no model:

```bash
python scripts/compare_checkpoints.py \
  --a runs/final/hf_Qwen_Qwen3-1.7B_70d244cc86ccca08cf5af4e1e306ecf908b1ad5e__zero_shot.transcripts.jsonl \
  --b runs/final/hf_HankH18_qwen3-1.7b-forbidden-constraint-holder_c19712a72932b503f28fe063d1962acc9e8c3fb8__zero_shot.transcripts.jsonl \
  --label-a base --label-b v2
```

### §5 · Data-efficiency curve and minimum viable N — 3:30–4:15

```bash
python scripts/curve_report.py --check --markdown
```

Four trained checkpoints at 106 / 211 / 423 / 846 — the assignment's own N/8, N/4,
N/2, N spacing, anchored on the real corpus size. Two things to show:

- The trained constraint families improve monotonically with N; the never-trained
  families **degrade** monotonically. Sixty of the 147 eval scenarios carry an
  untrained type, so a real gain and a real loss cancel and the aggregate reads as
  noise. That is a second, independent way the metric hid the data work.
- The minimum viable N is stated as a **bound**, not a value, and why: on 147
  scenarios `spec_adherence` cannot resolve differences under about nine
  percentage points, so every interval overlaps every other. Locating a knee with a
  metric that has no knee is the mistake; the answer is given on the measure that
  moves.

### §6 · Where everything is — 4:15–4:30

Model and dataset on the Hub, both public, both pinned by commit in
`configs/models.json`. The running demo Space. And the one command a grader runs:

```bash
python eval.py --model HankH18/qwen3-1.7b-forbidden-constraint-holder \
               --eval-set data/scenarios/corpus.jsonl
```

Close on the honest sentence: *the data intervention moved the target behaviour by
a large and statistically overwhelming margin, and the pass/fail metric the
assignment grades on is structurally incapable of showing it.*

---

## Rehearsal run — what this actually looks like

Run end to end against the current pins, warm cache. **This is a rehearsal, not a
substitute for the live segment** — §3 still has to use the grader's own prompt.

Command:

```bash
python scripts/demo.py \
  --prompt "Write a short notice for a community noticeboard about the library reopening on Saturday." \
  --constraint banned_letter:e --constraint no_lists
```

Observed timings, warm cache: base load + generate **26.2 s**, tuned **3.8 s**.
Cold, the first load is far slower — warm it before recording.

What came back is a better illustration than a scripted example would have been,
and it is worth understanding before you record, because the *interesting*
difference is not the pass/fail column:

- **The base model degenerated.** 2,787 characters consisting of the sentence "The
  library is reopening on Saturday." repeated roughly seventy times. It is on-topic
  and it is useless.
- **The tuned model wrote the notice.** 105 characters: *"**Library Open** / On
  Saturday, our public library will open again. All public services are now
  available."*
- **Both failed `banned_letter:e`** — base at offset 17 (`notic[[e]]board`), tuned
  at offset 12 (`Op[[e]]n`). Both passed `no_lists`.

So the side-by-side reads `FAIL / FAIL`, and the demo's own footer says so. Do not
paper over that. The thing to point at is the **length and quality gap**, which is
exactly the effect the report quantifies: judged task quality ≥ 3 rose from 61.2 %
to 72.1 % of 147 rows, while `banned_letter` compliance went from 0.0 % to 1.0 % of
checks — the constraint is still almost never fully held, and the *occurrence count*
is where the improvement lives.

That is the honest story in one screen: **the tuned model got much better at the
task and measurably better at the constraint, and the pass/fail column cannot show
either.** If the live prompt lands the same way, you already have the words for it.

---

## If the live segment misbehaves on camera

- **Generation is slow.** Expected on a laptop. Say so and keep talking; the
  runbook's §4 material fills the gap naturally.
- **The tuned model violates the constraint.** Leave it in. It fails a majority of
  scenarios outright — that is what the numbers say, and the claim being made is
  about occurrence counts and about the metric, not about perfection. Cutting an
  unflattering live run would contradict §4.
- **The grader picks a held-out constraint type** (`banned_punctuation`,
  `sentence_initial_letter`). Worth welcoming rather than avoiding: those two types
  appear in **zero** training rows and the tuned model is *worse* than base on them.
  It is the project's least flattering result and it is already written up in
  `docs/BRAINLIFT.md` §7.
