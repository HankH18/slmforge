# Deploying the Space

Target: a **public** Gradio Space that runs a grader-supplied prompt against the
pinned base and tuned models live and prints the pinned checkers' verdicts.

Every command below has been run from this machine. Four of them ran only as far
as argument parsing, and are marked where they appear:

* the three marked **NOT RUN** — creating the Space, uploading to it, and
  reading a live Space back — were run offline, so their flags are known to be
  accepted and nothing was pushed;
* step 6's `scripts/demo.py` call, which parses and then wants ~3.4 GB of
  weights per model.

Everything else ran to completion. Exit codes quoted are the ones observed.

Repeat the parse check on any command here that reaches the Hub:

```bash
env -u HF_TOKEN HF_HUB_OFFLINE=1 .venv/bin/hf <the exact arguments below>
```

Exit `1` naming `offline mode is enabled` or `Cannot reach
https://huggingface.co/...` means the arguments parsed and only the network was
refused — that is the pass. Exit `2` with `Got unexpected extra argument` is a
broken command line. The way to write one by accident: `hf`'s `--exclude` takes
**one** pattern per flag, so a second bare pattern after it is read as a
positional argument and lands in `PATH_IN_REPO`'s slot.

---

## 0. Prerequisites

```bash
cd /Users/hankholcomb/Documents/code_parent_folders/gauntlet_repos/slmforge
set -a && . ./.env && set +a
.venv/bin/hf auth whoami
```

Observed: `user=HankH18`, exit `0`. The token in `.env` must have **write**
scope to create and upload a Space; `whoami` succeeding does not prove write
scope, only that the token is valid.

Python is `.venv/bin/python` (3.11). `hf` is `huggingface_hub` 1.28.0's CLI —
`huggingface-cli` is deprecated and no longer works.

---

## 1. Verify locally first — before anything is pushed

Two checks, both offline, neither needs Gradio and neither downloads a weight.

```bash
.venv/bin/python space/test_space_app.py
.venv/bin/python space/check_app_wiring.py
```

Observed: `31 passed, 0 skipped, 0 failed (31 cases)`, exit `0`; and
`app.py wiring OK`, exit `0`. Skips are counted and printed separately from
passes and never folded into them, so a case that did not execute cannot be read
as one that did.

The first drives `core.compare` — the exact function the Run button is bound to —
with a stub generator and the **real** checkers, feeding it one answer that
violates every constraint and one that satisfies every constraint, and requires
the verdicts to come back different. One of its cases runs the second script as a
subprocess, so the suite is not blind to the page's wiring.

The second executes `app.py`'s `build()` against a recording shim and checks the
button really is bound to `core.compare`, and that the output components carry
`core.OUTPUT_FIELDS`' names **element by element** — by name, not by count. That
distinction is the point: transposing the `base_*`/`tuned_*` pair in either list
keeps the count identical while routing the tuned model's answer under the BASE
heading. Both scripts fail on that transposition; `check_app_wiring.py` performs
it on itself as one of its negative cases and requires `build()` to raise. The
input components are checked the same way, against `core.compare`'s own
parameter names, because Gradio passes `inputs` positionally: a transposed pair
there would send a visitor's constraints as the task.

Neither exercises Gradio itself. Gradio is not installed in this venv. See
"What is still unproven" at the bottom.

---

## 2. Assemble the Space repo

```bash
bash space/build_space.sh /private/tmp/slmforge-space
```

Observed: `pins complete: 13 markers present`, 29 files, exit `0`. The file count
tracks whatever `constraints/` and `scenarios/` currently contain, so it moves
when those packages do. Count it on a clean build: running anything from inside
the directory leaves `__pycache__` behind and inflates it.

It writes `app.py`, `core.py`, `_repo.py`, `requirements.txt`, `README.md`, a
`.gitignore`, and `pins/` — a mirror of the project modules the Space imports
(`eval.py`, `harness.py`, `judge.py`, `constraints/`, `scenarios/`,
`scripts/demo.py`, `configs/models.json`, `configs/train.json`, the shipped
constraint library) plus `pins/PINS_MANIFEST.txt` recording the source commit and
a sha256 per file. The completeness check is derived from `_repo.MARKERS`, so a
mirror missing a file fails the build instead of shipping a Space that cannot
boot.

It writes **only** to the output directory. It does not touch the slmforge tree
and runs no git index operation.

### Confirm the assembled Space boots from `pins/` alone

```bash
cp space/test_space_app.py space/check_app_wiring.py /private/tmp/slmforge-space/
env -u SLMFORGE_ROOT .venv/bin/python /private/tmp/slmforge-space/test_space_app.py
env -u SLMFORGE_ROOT .venv/bin/python /private/tmp/slmforge-space/check_app_wiring.py
rm -f /private/tmp/slmforge-space/test_space_app.py \
      /private/tmp/slmforge-space/check_app_wiring.py
find /private/tmp/slmforge-space -type d -name __pycache__ -prune -exec rm -rf {} +
```

Observed on the assembled copy: `29 passed, 2 skipped, 0 failed (31 cases)`,
exit `0` — 29 executed, plus two `SKIP` lines for `verify_deploy.py`, a
deploy-time script that deliberately stays out of the upload. Wiring OK exit `0`,
and `pins rendered from /private/tmp/slmforge-space/pins/configs/models.json` —
i.e. it resolved the mirror, not the checkout. Delete the two test files again
before uploading; they are not part of the Space.

Both files in the `cp` above are required: `check_app_wiring.py` is the only case
covering the page's wiring, so `test_space_app.py` FAILS rather than skips when it
is not beside it, in every layout. Measured: with it absent and the `base_*`/
`tuned_*` pair transposed in `core.OUTPUT_FIELDS`, the suite exits `1` with
`FAIL test_app_wiring_holds_including_its_own_negative_cases`.

The `find` matters and a plain `rm -rf .../__pycache__` is not enough: importing
the mirrored modules writes caches into `pins/`, `pins/constraints/`,
`pins/scenarios/` and `pins/scripts/` too — 10 `.pyc` files here, which is the
whole difference between a 29-file build and a 39-file one. `--exclude "*.pyc"`
in step 4 keeps them off the Hub even if this is skipped, and `verify_deploy.py`
ignores caches on both sides, but the build directory should match what is
uploaded.

### Confirm the mirror is byte-identical to the repo

The manifest prints its own re-verification command in its header, with the
manifest's absolute path and no placeholder in it. Run it from the repo root:

```bash
shasum -a 256 -c <(sed 's| pins/| |' /private/tmp/slmforge-space/pins/PINS_MANIFEST.txt | grep -v '^#')
```

Observed: 22 lines of `OK`, exit `0`. Every mirrored file is the same bytes as
the file it was copied from. A `FAILED` line names a pin that drifted.

A deliberately broken mirror (two files removed) was also tried: import fails
with `_repo.RepoRootNotFound`, naming both missing files, exit `1`. The Space
refuses to start rather than substituting its own checkers.

---

## 3. Create the Space — **NOT RUN**

```bash
set -a && . ./.env && set +a
.venv/bin/hf repos create HankH18/forbidden-constraint-holder \
    --type space --space-sdk gradio --public
```

`--public` is required by the deliverable. `hf repos create` is the current
spelling; `hf repo create` still works but prints a deprecation warning.

If the Space already exists this is a no-op and the flags are ignored — check
visibility on the settings page rather than assuming.

---

## 4. Upload — **NOT RUN**

```bash
set -a && . ./.env && set +a
.venv/bin/hf upload HankH18/forbidden-constraint-holder \
    /private/tmp/slmforge-space . \
    --type space \
    --exclude "__pycache__/*" --exclude "*.pyc" \
    --commit-message "Live base-vs-tuned constraint demo, pinned modules mirrored"
```

Uploading the directory as one commit keeps `pins/` and the app in the same
revision, which matters: a Space whose app and pinned modules came from different
commits is exactly the drift `pins/PINS_MANIFEST.txt` exists to make visible.

---

## 5. Read it back — **NOT RUN against a live Space**

Do not conclude anything from step 4's output. Ask the Hub what is there:

```bash
set -a && . ./.env && set +a
.venv/bin/python space/verify_deploy.py HankH18/forbidden-constraint-holder \
    --built /private/tmp/slmforge-space
```

Exit `0` only when the Space exists, is **public**, its SDK is `gradio`, its
runtime stage is `RUNNING`, and every file matches the local build's sha256 in
**both** directions — a file on the Hub that the build did not produce is
reported too. It downloads only the small text files; no weights.

Right after a push the Space is still building. Use `--no-runtime` for the first
few minutes, then run it again without the flag; a `BUILD_ERROR` Space still has
all its files, so the file diff alone is not evidence that the demo is up.

Verified about this script: `--help` exits `0`; its Hub path was exercised
against a non-existent Space and reported
`FAIL: cannot read space ...: RepositoryNotFoundError: 404`, exit `1`; and its
manifest comparison was unit-tested in both directions. Its **success** path has
never run, because nothing has been pushed.

---

## 6. Check it in the browser

Open `https://huggingface.co/spaces/HankH18/forbidden-constraint-holder` and
confirm, in this order:

1. The header prints two commit-pinned model specs. They must equal
   `configs/models.json`'s `base` and `tuned`. They are read at startup, not
   compiled in.
2. The hardware banner names the tier and says the run takes minutes.
3. Type a task and at least one constraint, press Run. The first run downloads
   ~3.4 GB per model, so the first press is much slower than later ones.
4. Under each answer: a per-constraint verdict table, and a `spec_pass` line
   reading **UNDECIDED**, not a pass. If it ever reads PASS, something is wrong —
   no judge runs in this Space.
5. The footer prints the 147-scenario measurement including the negative
   headline, rendered by `scripts/demo.py:format_footer()`.

For the recording that satisfies the spec's live-comparison requirement, take
the prompt and the constraints from the grader on camera, paste them in, and let
both models run. Nothing is cached and nothing is pre-selected, so the run is
reproducible by them afterwards with the same inputs — or on the CLI:

```bash
set -a && . ./.env && set +a
.venv/bin/python scripts/demo.py --prompt "<their prompt>" --constraint "<their constraint>"
```

The Space and that CLI call the same functions, so their verdicts must agree.

That command line is checked as far as parsing: substituting a constraint type
that does not exist stops it inside `constraints/registry.py` with exit `2` and
the syntax help, before any weight is touched. A real generation from this
machine was not run.

---

## 7. Updating it later

Re-run steps 2, 4 and 5. `build_space.sh` deletes and rebuilds the output
directory, so the mirror is never partially stale, and `verify_deploy.py` will
flag any file left behind on the Hub that the new build does not produce.

---

## Troubleshooting

**`sdk_version` rejected at build time.** This is the first thing to check,
because it is the most likely way a correct upload still fails to boot.
`README.md`'s front matter names `5.49.1`. That release **exists**:
`GET https://pypi.org/pypi/gradio/5.49.1/json` returns `200` with
`version: 5.49.1`, `requires_python: >=3.10`. What is **not** verified is that
the Hub's Space builder accepts it as an `sdk_version` — that list is the Hub's,
not PyPI's, no Space was created from this machine, and Gradio is deliberately
not installed here. If the build log rejects it, it names the versions it does
accept: put one of those in `README.md` line 7 and re-upload. Nothing else
changes; the whole Space is one YAML line away from working in that case.

**Build succeeds, page 500s on load.** Almost always `_repo.RepoRootNotFound` —
`pins/` did not upload completely. Check the Space's Files tab against
`pins/PINS_MANIFEST.txt` and re-run `verify_deploy.py`.

**A request appears to hang.** It is generating. 512 tokens on a free CPU box is
minutes per model and there are two models. The status line updates between
models; elapsed seconds are printed under each answer. Lower `max_new_tokens` if
a grader will not wait — the page states plainly when the budget is below the
evaluated 512 and warns that a truncated answer can fail a constraint it would
otherwise have held.

**Out of memory.** The two models are already loaded and freed one at a time and
the queue is limited to one job. If it still OOMs, the free tier is not enough
and the Space needs upgraded hardware; do not "fix" it by holding both models
resident.

---

## What is still unproven

* **Gradio itself.** Not installed in this venv and deliberately not installed.
  The component tree in `app.py` executes against a recording shim that accepts
  any keyword argument, so nothing here proves real Gradio accepts
  `gr.Textbox(show_copy_button=True)`, or `elem_id=` on every output component,
  or `(label, value)` choice tuples in `gr.Dropdown`, or that a generator `fn`
  streams as intended. First boot is the first real test of that.
* **`sdk_version: 5.49.1`.** The release exists on PyPI (checked: HTTP `200`).
  Whether the Hub's Space builder accepts that exact string is not checked;
  see the first troubleshooting entry.
* **The real models.** Every local check uses a stub generator. The
  model-loading seam is `scripts/demo.py:generate` → `eval.make_generator`, the
  same path `scripts/demo.py` and `eval.py` use, but no weight has been loaded by
  anything in `space/`.
* **The push and the live read-back.** Steps 3, 4 and 5's success path.
* **Latency.** The figure in the UI is labelled an estimate and is not a
  measurement on Space hardware. The app prints its own real elapsed seconds per
  model, which replaces the estimate the first time anyone presses Run.
