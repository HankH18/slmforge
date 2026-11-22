# Demo video — runbook

The Final Submission asks for:

> "A 3–5 minute demo video showing the tuned model doing the thing the base model
> fails to do reliably, including one live, grader-supplied prompt."

and, separately, under Verification Requirements:

> "Live comparison in demo | Part of your demo video must show a grader-supplied
> prompt run live against base vs. tuned — not only pre-selected examples."

**F6 needs both halves, and they are different segments.** §3 is the pre-selected
example — the tuned model doing the thing the base model fails at. §4 is the live,
grader-supplied prompt. §4 alone does not satisfy F6, and §3 alone does not
satisfy V6.

---

## Read this before anything else — two traps that were in this runbook

**1. `python` is not the venv.** On this machine `which python` is
`~/opt/anaconda3/bin/python`, and that interpreter has no `torch`:

```
$ ~/opt/anaconda3/bin/python -c "import torch"
ModuleNotFoundError: No module named 'torch'
```

Every command below is written as `.venv/bin/python` for that reason. Do not
shorten them on camera.

**2. The prompt this runbook used to recommend fails on both models.** It was
`--constraint banned_letter:e`, and `banned_letter` is the one constraint type
*neither* checkpoint can hold: across the adversarial set the tuned model violates
it on **108 of 108 checks** (`docs/ADVERSARIAL_EVAL.md:299-300`), and on the final
eval `banned_letter` compliance is **0.0 % → 1.0 %** of checks
(`scripts/demo.py:145`, printed in the demo's own footer). Filming a
segment whose headline constraint fails on both sides is a self-inflicted wound.
§3 now uses `banned_words`, where a measured tuned-over-base win exists, and the
exact command has been **run end to end** — the transcript is in
[Rehearsal record](#rehearsal-record--actually-run-not-reconstructed) below.

---

## Before recording

`scripts/demo.py` loads **both** checkpoints — roughly 3.4 GB each — one at a time.
Both are already in `~/.cache/huggingface/hub`, but the process still reads them
from disk on every invocation. Warm the OS page cache by running **the §3 command
itself** once, an hour or less before you record. That is the rehearsal and the
warm-up in one action, and it is the only warm-up you need.

No API key is required for the demo: no judge runs, only `constraints/checkers.py`,
and both Hub repos are public. Sourcing `.env` is harmless but unnecessary — the
rehearsal below ran without it.

Then check what will be on screen:

```bash
.venv/bin/python scripts/demo.py --help    # the constraint syntax block reads well aloud
git rev-parse HEAD                         # the eval-code commit, if you want it in frame
```

Have these open in tabs:

- the model: <https://huggingface.co/HankH18/qwen3-1.7b-forbidden-constraint-holder>
- the dataset: <https://huggingface.co/datasets/HankH18/forbidden-constraint-holder-sft>
- the live Space: https://huggingface.co/spaces/HankH18/forbidden-constraint-holder
  (deployed and read back with `space/verify_deploy.py`, exit 0, stage RUNNING).
  It is on `cpu-basic`, so do NOT run a generation on camera — it takes minutes.
  Show the page, then do the live prompt locally with `scripts/demo.py`.
- `docs/FINAL_REPORT.md`, `docs/ADVERSARIAL_EVAL.md`, `docs/EFFICIENCY_CURVE.md`

---

## The shape, ~4:45

| § | what | in | out | length |
|---|---|---|---|---|
| 1 | What the behaviour is | 0:00 | 0:25 | 0:25 |
| 2 | Why fine-tuning was justified at all | 0:25 | 0:50 | 0:25 |
| 3 | **Pre-selected: base fails, tuned holds** | 0:50 | 1:40 | 0:50 |
| 4 | **THE LIVE SEGMENT — graded** | 1:40 | 2:50 | 1:10 |
| 5 | The headline, told honestly | 2:50 | 4:00 | 1:10 |
| 6 | Data-efficiency curve and minimum viable N | 4:00 | 4:30 | 0:30 |
| 7 | Where everything is | 4:30 | 4:45 | 0:15 |

4:45 total, inside the 3–5 minute window with 15 seconds of slack. §3 and §4 each
contain roughly 20 s of generation you cannot speed up; the talk track below is
written to be spoken *over* it, not before it.

### §1 · What the behaviour is — 0:00–0:25

Read the Behavior Spec off the screen; it is two sentences and a stranger can mark
any output against it:

> Given a task prompt and a set of 2–4 hard constraints that always includes at
> least one negative constraint, the model completes the task while violating zero
> constraints. Pass = every constraint passes its programmatic checker AND the
> response genuinely addresses the task.

Say the one thing that makes the whole project falsifiable: **constraint pass/fail
is 100 % programmatic** — `constraints/checkers.py`, no model call, no network. The
LLM judge scores task-quality only and can never overturn a checker; it exists to
stop the degenerate strategy of satisfying every constraint with garbage text.

### §2 · Why fine-tuning was justified at all — 0:25–0:50

Show `docs/ABLATION_G1.md`. Two frontier families × three prompting strategies ×
147 scenarios. The best prompt still leaves a real failure rate, and the failure
that survives is compositional: the model holds a constraint until the task's own
subject matter pulls against it.

Say the number the gate turned on rather than describing it. It is one command,
and it prints its own verdict line:

```bash
.venv/bin/python scripts/gate_check.py runs/g1 --config configs/g1.json
```

### §3 · Pre-selected: the base model fails, the tuned model holds — 0:50–1:40

This is the "doing the thing the base model fails to do" half of F6. **Type this
exactly.** It has been run; see the rehearsal record for the verbatim output.

```bash
.venv/bin/python scripts/demo.py \
  --prompt "Write a short notice for a park gate explaining the new fenced field where dogs can run off the lead, when it is open, and who to tell about a broken latch. Readers here know the subject inside out, so write it the way people who work in it actually talk about it, in the plain terms they use among themselves." \
  --constraint banned_words:dog,dogs,puppy,puppies \
  --constraint no_lists
```

**Why this prompt and not another one.** The task text is
`av2-b06-content_collision` from `data/scenarios/adversarial_v2.jsonl`, and the
banned-word list is the one carried by the shipped constraint `c-lex-009`. (A
typed `--constraint` gets a freshly minted id, so the on-screen table will read
`c-lex-900`, not `c-lex-009` — same four words, and `demo.py` guarantees a minted
id never collides with the shipped library. `--constraint-id c-lex-009` would show
the shipped id instead and renders the identical requirement text — but the
rehearsal below used the typed form, so type the typed form.) It belongs to the
`content_collision` attack family, whose whole design is to make the forbidden
words the topic-salient words — write about the dog field without saying "dog".
That family is where the measured tuned-over-base win lives: on the targeted
constraint the base model holds **5 of 15** and the tuned model **12 of 15**, seven
scenarios flipping to tuned and none to base, McNemar exact **p = 0.0156**
(`docs/ADVERSARIAL_EVAL.md:122`).

Talk track while it generates (about 21 s wall, base first):

- "Same prompt, same constraints, same 512-token budget, same chat template. The
  only difference between these two answers is the fine-tune."
- "This is a family of scenarios built to make the banned word the obvious word.
  The base model reaches for it. The point is whether the tuned model does."
- When it lands: **read the SIDE BY SIDE table off the screen.** Base violates
  `banned_words` — the checker prints the offending span and its character offset,
  so nobody takes the verdict on trust. Tuned holds both constraints.
- "One prompt is an anecdote. The demo says so itself in its footer, and the
  measurement is the ninety paired rows I'll show in a minute."

Do **not** claim robustness from this. Say the honest version if it comes up:
*"the family is called an attack, but when I remove the attack and re-ask the same
question of the paired clean control, most of this advantage is still there — so
it is constraint-holding, not attack resistance"*
(`docs/ADVERSARIAL_EVAL.md:206`, `:208-212`).

### §4 · THE LIVE SEGMENT — 1:40–2:50 — *this is the graded part*

**What `scripts/demo.py` actually requires from the grader — read this before you
ask them.** Verified against the argument parsing:

| requirement | where it is enforced | what happens if it is missing |
|---|---|---|
| a task prompt | `scripts/demo.py:789` — `--prompt` is required unless `--interactive` | `parser.error`, exit 2, nothing runs |
| **at least one constraint** | `scripts/demo.py:308-311` — `resolve_constraints` raises `DemoError` | `error: no constraints given…`, exit 2, nothing runs |

So a grader who supplies only a prompt cannot be filmed. **You must ask for both,
on camera, before you type anything.** Say this:

> "This next part I have not chosen and cannot rehearse. I need two things from
> you. First, a writing task — a notice, an email, a product description,
> anything, just say it in a sentence. Second, and the tool will not run without
> it: at least one forbidden constraint. That can be a word or a short list of
> words the model may not use, a letter it may not use, a punctuation mark it may
> not use, no bulleted or numbered lists, or no quotation marks. Give me the task
> and give me at least one constraint, and I will type exactly what you say."

If they give you a task but no constraint:

> "I need a constraint as well — the demo refuses to run without one, because the
> whole behaviour is about holding a forbidden rule. Any of these: a banned word,
> a banned letter, a banned punctuation mark, no lists, or no quotation marks."

If they say *you* pick the constraint, say the choice out loud so it is on the
record:

> "Then I will name it rather than quietly choose the easy one: I'm using a banned
> word, which is a constraint type the training data covered. If you'd rather see
> the least flattering case, ask me for banned punctuation or a sentence-initial
> letter — those two types appear in zero training rows and the tuned model is
> *worse* than base on them."

Then type it. `--constraint` is repeatable; the form is:

```
banned_words:<word>[,<word>...]                       [train]
banned_phrases:<phrase>[,<phrase>...]                 [train]
banned_letter:<single letter>                         [train]
banned_punctuation:<marks, one character each, e.g. ,;>  [heldout_type]
sentence_initial_letter:<single letter>               [heldout_type]
no_lists                                              [train]
no_quotation_marks                                    [train]
```

That block is `--help`'s own output, tags included, and the `[train]` /
`[heldout_type]` labels are worth reading aloud — they are the honest map of where
this model is good and where it is not.

```bash
.venv/bin/python scripts/demo.py \
  --prompt "<exactly what the grader said>" \
  --constraint "<exactly what the grader said>"
```

Quote anything the shell would eat: `--constraint 'banned_punctuation:,;'`.

`--interactive` reads the task and the constraints from stdin and films better if
the grader wants to type it themselves; it asks for the task first, then the
constraints one per line, blank line to finish.

What to point at when the output lands, in this order:

1. **Both answers, side by side** — same prompt, same constraints, same generation
   budget, same chat-template condition.
2. **The verdict table.** These are the *shipped* checkers, run live. On a
   violation the evidence shows the offending span and its offset.
3. **The honesty footer.** Its numbers are re-derived from `runs/final/` by
   `tests/test_demo.py:627`, so the demo cannot quote figures the committed
   transcripts do not support.

Two things worth saying out loud here:

- Decoding is **greedy** — `do_sample=False` at `eval.py:234`, the same call the
  eval uses. There is no temperature to blame and no lucky sample: the same input
  gives the same answer.
- A single run is one sample. The claim that the difference is real comes from the
  paired tests in §5, not from what is on screen.

### §5 · The headline, told honestly — 2:50–4:00

This is the part not to soften. Put `docs/FINAL_REPORT.md` on screen. Say the loss
**first**, before anything good. Every figure spoken below appears in
[the provenance table](#where-every-number-in-this-runbook-comes-from) with the
file and line it was read from and the command that re-derives it.

**Must say — roughly 120 words, fits in 50 s:**

> "Here is the number I am graded on, and I did not win it. `spec_adherence` —
> every constraint held, plus a judge scoring the writing at least three — is
> **0.0890** for the base model and **0.0816** for the tuned model. Fine-tuning
> moved the headline the wrong way. I'm not going to dress that up, and I'm not
> going to overstate it either: paired on the same 146 scenarios, five pass only
> for base and four only for tuned, McNemar exact **p = 1.000**. At this sample
> size the two models are indistinguishable on that metric. What I cannot claim is
> that the tuned model beat it."

> "So here is what did move, on the same scenarios, all of it paired and
> programmatic. `spec_adherence` requires **zero** occurrences of the banned
> letter, so it cannot tell one slip from forty. Count the occurrences instead:
> median **34 → 14**, lower on **89 of 105** paired rows, sign test
> **p = 1.87 × 10⁻¹³**, bootstrap CI on the difference of medians **[−26, −15]**."

> "And on the ninety attacked rows you just saw one of: pooled, the tuned model
> holds the targeted constraint on **12** scenarios where base loses it and loses
> it on **1** where base holds it. McNemar exact **p = 0.0034**."

**Then the finding the project is actually about:**

> "A metric that reports p = 1.000 for a difference a paired test on the identical
> scenarios resolves at p ≈ 10⁻¹³ is measuring the threshold, not the behaviour.
> `spec_adherence` requires zero violations, so it is a cliff — and a cliff has no
> knee to find."

**If you have room — the caveats, said before a grader can say them:**

> "Two caveats. The adversarial cut was chosen after the run, not pre-registered,
> and six family-level tests were computed — at Bonferroni 0.05/6 no individual
> family survives, so only the pooled number should be read inferentially. And the
> aggregate cancels because sixty of the 147 eval scenarios carry a constraint type
> the training data never contained: on trained families `spec_adherence` goes
> **0.1149 → 0.1379**, on never-trained families **0.0508 → 0.0000**."

Re-derive it live if there is time — it reads from committed transcripts, so it is
fast and needs no model:

```bash
.venv/bin/python scripts/compare_checkpoints.py \
  --a runs/final/hf_Qwen_Qwen3-1.7B_70d244cc86ccca08cf5af4e1e306ecf908b1ad5e__zero_shot.transcripts.jsonl \
  --b runs/final/hf_HankH18_qwen3-1.7b-forbidden-constraint-holder_c19712a72932b503f28fe063d1962acc9e8c3fb8__zero_shot.transcripts.jsonl \
  --label-a base --label-b v2

.venv/bin/python scripts/adversarial_report.py --check   # prints two ok lines, gates the numbers above
```

### §6 · Data-efficiency curve and minimum viable N — 4:00–4:30

```bash
.venv/bin/python scripts/curve_report.py --check --markdown
```

Four trained checkpoints at **106 / 211 / 423 / 846** — the assignment's own N/8,
N/4, N/2, N spacing, anchored on the real corpus size
(`docs/EFFICIENCY_CURVE.md:118`). Two things to show:

- The trained constraint families improve with N; the never-trained families
  **degrade**. That is a second, independent way the aggregate metric hid the data
  work — and the held-out-type split is a generalization test this project imposed
  on itself.
- The minimum viable N is stated as a **bound**, not a value, and why: on 147
  scenarios `spec_adherence` cannot resolve small differences, so every interval
  overlaps every other. Locating a knee with a metric that has no knee is the
  mistake; the answer is given on the measure that moves.

### §7 · Where everything is — 4:30–4:45

Model and dataset on the Hub, both public, both pinned by commit in
`configs/models.json`. The demo Space is live at
huggingface.co/spaces/HankH18/forbidden-constraint-holder — show the page, but do
not run a generation on it on camera: it is on `cpu-basic` and takes minutes. The
live prompt segment is done locally with `scripts/demo.py`. Then the one command a
grader runs:

```bash
.venv/bin/python eval.py \
  --model HankH18/qwen3-1.7b-forbidden-constraint-holder \
  --eval-set data/scenarios/corpus.jsonl
```

That is `README.md:120` verbatim apart from the interpreter prefix. **This one does
call the judge and costs money — never run it on camera**, just show it.

Close on the honest sentence: *the data intervention moved the target behaviour by
a large and statistically overwhelming margin, and the pass/fail metric the
assignment grades on is structurally incapable of showing it.*

---

## Rehearsal record — actually run, not reconstructed

The §3 command above was executed against the current pins on 2026-08-22, warm
cache, on this machine. **This is a rehearsal, not a substitute for §4** — the live
segment still has to use the grader's own prompt.

Timings, from `demo.py`'s own stderr: base load + generate **12.7 s**, tuned
**8.2 s**, **21.3 s** wall for the whole invocation. Decoding is greedy
(`eval.py:234`), so re-running it should reproduce these answers byte for byte;
that is what the warm-up run is for.

**Base output** (302 chars) — reaches for the banned word in its first clause:

> The new fenced field is open for **dogs** to run off the lead, and it's a good
> place to let them loose when the weather's nice. Just make sure to keep an eye on
> them and don't let them wander too far. If you ever notice a broken latch, don't
> hesitate to call the park staff. They're always on hand to help.

**Tuned output** (360 chars) — writes the same notice without any of the four
banned words:

> Fenced Field Notice
>
> A new fenced field is now open for running off the lead. It is available from
> 9:00 am until dusk, weather permitting. Please keep your animal close and follow
> the gate staff's instructions.
>
> If a latch breaks or cannot be opened, contact the gate office immediately. Staff
> will be able to repair it and provide information about the field.

**The side-by-side, verbatim:**

```
CONSTRAINT                                                    BASE    TUNED
------------------------------------------------------------------------------
banned_words: Do not use any of these words: dog, dogs, p...  FAIL    PASS
no_lists: Do not use bulleted or numbered lists.              PASS    PASS
------------------------------------------------------------------------------
ALL CONSTRAINTS HELD                                          NO      yes
```

Base's failure evidence was `banned word 'dogs' at offset 33`. That is the shot.

---

## Where every number in this runbook comes from

Nothing here is quoted from memory. Each row is a committed artifact and the
command that re-derives it.

| figure | source | re-derive with |
|---|---|---|
| `content_collision` targeted hold: base 5/15, tuned 12/15, 7 tuned-only, p = 0.0156 | `docs/ADVERSARIAL_EVAL.md:122` | `scripts/adversarial_report.py --markdown` |
| pooled targeted: 90 pairs, 52 vs 63, base-only 1, tuned-only 12, p = 0.0034 | `docs/ADVERSARIAL_EVAL.md:126`, `:128-129` | same |
| `banned_words` as a targeted *type*: base 29/41, tuned 37/41, p = 0.0078 | `docs/ADVERSARIAL_EVAL.md:216` | same |
| `banned_words` violation rate, scored rows: 0.275 (46/167) → 0.095 (16/168) | `docs/ADVERSARIAL_EVAL.md:303-304` | same |
| six family tests, Bonferroni 0.05/6 = 0.0083, no family survives | `docs/ADVERSARIAL_EVAL.md:170-172` | same |
| clean controls: base 58, tuned 65, p = 0.0654; attack-attributable +4.4 pp | `docs/ADVERSARIAL_EVAL.md:206`, `:208-212` | same |
| tuned violates `banned_letter` on 108/108 adversarial checks | `docs/ADVERSARIAL_EVAL.md:299-300` | same |
| `spec_adherence` 0.0890 → 0.0816; 5 base-only vs 4 tuned-only of 146; p = 1.000 | `docs/BRAINLIFT.md:536` — §7 table, row *"`spec_adherence`, whole eval set"* | `scripts/compare_checkpoints.py` |
| occurrences median 34 → 14, lower on 89/105, p = 1.87 × 10⁻¹³, CI [−26, −15] | `docs/BRAINLIFT.md:540` (§7 table) and `:740-742` (§8, *"base → v2 (N=846)"*) | same |
| trained families 0.1149 → 0.1379; never-trained 0.0508 → 0.0000 | `docs/BRAINLIFT.md:538-539` — §7 table | `scripts/curve_report.py --check --markdown` |
| sixty of 147 eval scenarios carry an untrained constraint type | `docs/BRAINLIFT.md:554` — §7, *"Sixty of the 147 eval scenarios carry a…"* | same |
| footer per-type compliance, incl. `banned_letter` 0.0 % → 1.0 % and `banned_words` 88.7 % → 94.4 % | `scripts/demo.py:139-150` | run any `demo.py` invocation; the footer prints it |
| `--prompt` required unless `--interactive` | `scripts/demo.py:789` | `.venv/bin/python scripts/demo.py --constraint no_lists` → exit 2 |
| at least one constraint required | `scripts/demo.py:308-311` | `.venv/bin/python scripts/demo.py --prompt x` → exit 2 |
| greedy decoding, `do_sample=False` | `eval.py:234` | read the generator |
| footer numbers bound to `runs/final/` | `tests/test_demo.py:627` | `pytest -q tests/test_demo.py` |
| the §3 task text and `c-lex-009` | `data/scenarios/adversarial_v2.jsonl` (`av2-b06-content_collision`), `data/scenarios/constraint_instances.jsonl` | grep the two files |

`docs/ADVERSARIAL_EVAL.md`'s results section is machine-gated:
`scripts/adversarial_report.py --check` re-derives every figure in it from the
committed transcripts and exits non-zero on any drift. `docs/BRAINLIFT.md` is not
gated and its line numbers move when it is edited, so each `BRAINLIFT` row above
also carries the quoted anchor text — if a line number no longer matches, grep the
anchor, and if the anchor is gone, re-derive with the command in the third column
rather than trusting the figure.

**Every row in this table was re-derived on 2026-08-22**, not copied: the two
`--check` gates were run (`adversarial_report.py --check` → *"66 figures
re-derived and matched"*), `compare_checkpoints.py` was run against the
`runs/final/` transcripts, and the §3 command was executed end to end.

---

## If the live segment misbehaves on camera

- **Generation is slow.** Expected on a laptop; §3 took 21 s wall. Say so and keep
  talking — §5's material fills the gap naturally.
- **The tuned model violates the grader's constraint.** Leave it in. It fails a
  majority of full 2-to-4 constraint specs outright — that is what the numbers say,
  and the claim being made is about occurrence counts, about the targeted
  constraint under collision, and about the metric, not about perfection. Cutting
  an unflattering live run would contradict §5.
- **The grader picks `banned_letter`.** Both models will almost certainly fail it —
  the tuned model violates it on 108 of 108 adversarial checks
  (`docs/ADVERSARIAL_EVAL.md:299-300`). Say that *before* the output lands, so it
  reads as a prediction rather than an excuse, and then use it: this is exactly the
  cliff §5 is about, and the occurrence measure is what sees the improvement the
  pass/fail column cannot.
- **The grader picks a held-out constraint type** (`banned_punctuation`,
  `sentence_initial_letter`). Worth welcoming rather than avoiding: those two types
  appear in **zero** training rows and the tuned model is *worse* than base on them
  (`docs/ADVERSARIAL_EVAL.md:305-306`). It is the project's least flattering result
  and it is already written up in `docs/BRAINLIFT.md` §7.
- **Both models pass.** Fine — say that the constraint was one a 1.7 B model has
  headroom on, and that the interesting cases are the ones where the topic pulls
  toward the banned words, which is §3.
