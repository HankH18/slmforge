#!/usr/bin/env python3
"""Replay the partner's 20 confirmed breaks against a model, before and after.

This is the harness that produces the number the assignment is graded on:
*how many of the partner's jailbreaks still break the model*. Run it once
against the base checkpoint and once against the retrained one and the two run
JSONs are the before and the after.

WHAT IT SENDS
-------------
The stored prompt, **verbatim**. Not re-rendered, not wrapped, not decorated.
That distinction is the whole point of this file: ``eval.build_prompt`` puts
every scenario through ``scenarios.render.render_user_message``, which appends a
canonical "Requirements -- your response must satisfy every one of these:"
block. The partner's prompts state their rule in prose instead, and re-rendering
them would measure a different attack from the one that succeeded. So the prompt
dict is built here -- ``{"system": None, "turns": [...]}``, the exact shape
``eval.build_prompt`` returns -- and handed straight to ``eval.HFGenerator``:

* the 15 single-turn rows go as ONE user turn;
* the 5 multi-turn rows go as the partner's own three messages
  (user -> assistant -> user), because the attack IS the follow-up.

Everything else about generation is imported, not re-implemented:
``eval.make_generator`` builds it, ``eval.HFGenerator.__call__`` does the
``apply_chat_template(add_generation_prompt=True, enable_thinking=False)`` and
the ``max_new_tokens=512, do_sample=False`` decode, and
``scripts.demo.release_generator`` frees the weights afterwards. No system
prompt is ever added.

HOW IT SCORES
-------------
By minting an ad-hoc constraint from the row's ``(family, arg)`` through
``constraints.registry.build_constraint`` and calling
``constraints.checkers.check`` -- the pattern ``scripts/demo.py`` uses, and the
same pinned entry point every published number in this repo is re-derived from.

**No stored verdict is ever read.** Not the partner's ``violated`` flag, not a
previous run's ``broke`` field. Every verdict in every mode of this script is
re-derived from ``(constraint, output)``. ``--check`` re-derives them a second
time from the committed run JSONs and fails if a stored verdict disagrees.

QUARANTINE, AND WHY IT IS LOUD
------------------------------
A row whose verdict cannot be derived -- the generator raised, the checker
raised, or the model returned nothing -- is QUARANTINED. It is never counted as
held and never counted as broken, it is named in the run JSON and in the
generated block, and the denominators say so. A model that errors on 6 of 20
must not be able to read as "14 fixed". Empty output is quarantined too, and
deliberately: ``constraints.checkers`` passes empty text vacuously (ruling A7),
so a model that returns nothing would otherwise score a clean sweep.

COVERAGE
--------
Exactly the 20 exam ids, each exactly once, nothing extra. Checked by
:func:`coverage_check` here rather than by ``scripts/gate_check.py``'s
``score_cell``: that function hard-refuses any cell below
``MIN_SCENARIOS_PER_CELL = 30`` (gate_check.py:835) and *returns None* rather
than raising, so a 20-row exam routed through it would produce no table at all
and an empty table reads exactly like a clean run.

THE SPLITS THAT MAKE THE HEADLINE HONEST
----------------------------------------
"20" is four different claims. The block reports all of them:

* **15 single-turn vs 5 multi-turn.** The multi-turn erosion attacks are out of
  distribution for anything trained on single-turn rows.
* **17 trainable-type vs 3 held-out-type.** ``edge-punct-1``,
  ``ans-punct-decimal`` and ``erode2-punct-list`` are ``banned_punctuation``,
  whose ``constraints.registry.split_of`` is ``"heldout_type"``;
  ``datagen/audit.py`` fails any dataset that uses a held-out type. Those three
  can be measured forever and trained on never, so a fix there is generalisation
  and a fix elsewhere may not be.

USAGE
-----
::

    python scripts/replay_breaks.py --model hf:Qwen/Qwen3-1.7B@<sha> \
                                    --out runs/breaks/base.json
    python scripts/replay_breaks.py --emit-block runs/breaks/base.json \
                                                 runs/breaks/tuned.json
    python scripts/replay_breaks.py --check docs/SOME_REPORT.md \
                                    runs/breaks/base.json runs/breaks/tuned.json

``--emit-block`` prints a marker-fenced generated block; ``--check`` re-derives
that block from the run JSONs and refuses the document unless the two are
byte-identical, in the style of ``scripts/check_mvp.py``'s GENERATED METRICS
fence and ``scripts/adversarial_report.py``'s document binding.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import difflib
import gc
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import eval as eval_mod                                     # noqa: E402
from constraints import checkers                            # noqa: E402
from scripts import demo as demo_mod                        # noqa: E402
from scripts import freeze_breaks                           # noqa: E402
from scripts.freeze_breaks import (                         # noqa: E402
    EXPECTED_ROWS, EXPECTED_SHA256, FreezeError, constraint_for, turns_of)

#: Schema tag written into every run JSON, so a consumer can refuse a shape it
#: does not understand instead of silently reading fields that moved.
SCHEMA = "partner_breaks_replay/1"

DEFAULT_FROZEN = freeze_breaks.DEFAULT_OUT
DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "breaks"

#: The fence around the generated block. Same convention as check_mvp.py's
#: "<!-- BEGIN GENERATED METRICS -->": everything between the markers is
#: generated output, and `--check` compares it byte for byte.
BEGIN_BLOCK = "<!-- BEGIN GENERATED BREAKS -->"
END_BLOCK = "<!-- END GENERATED BREAKS -->"

#: Why a row has no verdict. Each value is printed verbatim in the block, so a
#: reader sees WHICH failure produced the missing number.
Q_GENERATOR = "generator raised"
Q_CHECKER = "checker raised"
Q_EMPTY = "empty output"
Q_NO_OUTPUT = "no output recorded"


class ReplayError(Exception):
    """Anything that would make the produced number a lie."""


# --------------------------------------------------------------------------
# The exam
# --------------------------------------------------------------------------

def load_exam(path: Path = DEFAULT_FROZEN, *, require_pin: bool = True
              ) -> List[Dict[str, Any]]:
    """The frozen exam, re-validated and pin-checked.

    Goes through ``freeze_breaks.load_frozen``, so a committed artifact that
    would not survive freezing again cannot be replayed. The sha256 pin is
    checked here rather than trusted: the before-run and the after-run must
    provably face the same 20 prompts, and that is only true if the file has not
    moved between them.
    """
    rows = freeze_breaks.load_frozen(path)
    digest = freeze_breaks.frozen_sha256(path)
    if require_pin and digest != EXPECTED_SHA256:
        raise ReplayError(
            f"{path} sha256 {digest} != the pin in scripts/freeze_breaks.py "
            f"({EXPECTED_SHA256}). The exam changed under the runs that cite it; "
            f"re-freeze deliberately or restore the artifact.")
    return rows


def prompt_for(row: Mapping[str, Any]) -> Dict[str, Any]:
    """The row's stored prompt as the prompt dict ``eval.HFGenerator`` consumes.

    ``{"system": None, "turns": [...]}`` -- the exact shape ``eval.build_prompt``
    returns, with ``system`` always None because the partner's attacks landed
    with no system prompt and adding one would measure a different model.
    The turns come from ``freeze_breaks.turns_of``, the ONE place the string and
    message-list shapes are reconciled, so what is validated is what is sent.
    """
    turns = turns_of(row["prompt"])
    stored = row["prompt"]
    # Verbatim, asserted rather than assumed: nothing in this file may rewrite
    # the attack on its way to the model.
    if isinstance(stored, str):
        if len(turns) != 1 or turns[0]["content"] != stored:
            raise ReplayError(f"{row['id']}: single-turn prompt was not sent verbatim")
    else:
        if [t["content"] for t in turns] != [m["content"] for m in stored]:
            raise ReplayError(f"{row['id']}: multi-turn prompt was not sent verbatim")
    return {"system": None, "turns": turns}


def scenario_for(row: Mapping[str, Any]) -> Dict[str, Any]:
    """A minimal scenario stub for the generator's signature.

    ``eval.HFGenerator.__call__(scenario, prompt, constraint_lookup)`` ignores
    ``scenario`` entirely (eval.py:224) -- it reads the prompt. It is passed
    anyway so the call is the evaluated one, and carries the row id so a
    generator that DOES log it logs something true.
    """
    return {"scenario_id": row["id"]}


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def score_output(constraint: Mapping[str, Any], output: Any) -> Dict[str, Any]:
    """(constraint, output) -> the verdict. Never reads a stored one.

    Returns ``{"status", "evidence", "broke", "quarantined", "quarantine_reason"}``.
    ``broke`` is ``None`` -- never ``False`` -- whenever the verdict could not be
    derived. A missing verdict is not a constraint that held.
    """
    if output is None:
        return _quarantine(Q_NO_OUTPUT, "generator returned None")
    if not isinstance(output, str):
        return _quarantine(Q_NO_OUTPUT, f"output is {type(output).__name__}, not str")
    if not output.strip():
        # checkers ruling A7: empty text passes every checker vacuously. Scoring
        # that as "held" would let a mute model sweep the exam.
        return _quarantine(Q_EMPTY, "model returned no text; checkers pass empty "
                                    "text vacuously (constraints/checkers.py ruling A7)")
    try:
        verdict = checkers.check(dict(constraint), output)
    except Exception as exc:  # noqa: BLE001 - reported, never swallowed
        return _quarantine(Q_CHECKER, f"{type(exc).__name__}: {exc}")
    broke = not verdict["pass"]
    return {"status": "BROKE" if broke else "HELD",
            "evidence": str(verdict["evidence"]),
            "broke": broke, "quarantined": False, "quarantine_reason": None}


def _quarantine(kind: str, detail: str) -> Dict[str, Any]:
    return {"status": "QUARANTINED", "evidence": detail, "broke": None,
            "quarantined": True, "quarantine_reason": kind}


def coverage_check(exam: Sequence[Mapping[str, Any]],
                   rows: Sequence[Mapping[str, Any]]) -> None:
    """Exactly the exam's ids, once each, nothing extra. Raises, never returns None."""
    want = [r["id"] for r in exam]
    got = [r["id"] for r in rows]
    dupes = sorted({i for i in got if got.count(i) > 1})
    missing = sorted(set(want) - set(got))
    extra = sorted(set(got) - set(want))
    problems = []
    if len(want) != EXPECTED_ROWS:
        problems.append(f"exam has {len(want)} rows, expected {EXPECTED_ROWS}")
    if missing:
        problems.append(f"missing {len(missing)}: {missing}")
    if extra:
        problems.append(f"not in the exam: {extra}")
    if dupes:
        problems.append(f"scored more than once: {dupes}")
    if problems:
        raise ReplayError("coverage: " + "; ".join(problems))


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------

def replay(spec: str, exam: Sequence[Mapping[str, Any]], *,
           generator_factory: Optional[Callable[[str], Any]] = None,
           frozen_path: Path = DEFAULT_FROZEN,
           label: Optional[str] = None,
           progress: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """Send all 20 prompts to one model and score every output. Returns the run.

    The generator is built ONCE for the whole exam and released in a ``finally``,
    because ``eval.HFGenerator``'s docstring records a measured 19.7GB peak for a
    single 1.7B model on this machine and two models resident at once killed two
    ablation rows.
    """
    factory = generator_factory or eval_mod.make_generator
    say = progress or (lambda _m: None)
    gen = factory(spec)
    rows: List[Dict[str, Any]] = []
    try:
        for i, item in enumerate(exam):
            constraint = constraint_for(item, i)
            prompt = prompt_for(item)
            try:
                output = gen(scenario_for(item), prompt, {constraint["id"]: constraint})
            except Exception as exc:  # noqa: BLE001 - becomes a quarantine, not a pass
                verdict = _quarantine(Q_GENERATOR, f"{type(exc).__name__}: {exc}")
                output = None
            else:
                verdict = score_output(constraint, output)
            rows.append(_row(item, constraint, prompt, output, verdict))
            say(f"[{i + 1:2d}/{len(exam)}] {item['id']:<22} {verdict['status']}")
    finally:
        demo_mod.release_generator(gen)
        del gen
        gc.collect()

    coverage_check(exam, rows)
    return {
        "schema": SCHEMA,
        "label": label or spec,
        "model": spec,
        "exam": _repo_rel(frozen_path),
        "exam_sha256": freeze_breaks.frozen_sha256(frozen_path),
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "n_rows": len(rows),
        "rows": rows,
        "summary": summarize(rows),
    }


def _row(item: Mapping[str, Any], constraint: Mapping[str, Any],
         prompt: Mapping[str, Any], output: Any,
         verdict: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "id": item["id"],
        "category": item["category"],
        "family": item["family"],
        "constraint_type": constraint["type"],
        "constraint_id": constraint["id"],
        "arg": item["arg"],
        "split": item["split"],
        "trainable": item["trainable"],
        "multiturn": item["multiturn"],
        "turns_sent": list(prompt["turns"]),
        "system_prompt": prompt["system"],
        "output": output,
        "verdict": {"status": verdict["status"], "evidence": verdict["evidence"]},
        "broke": verdict["broke"],
        "quarantined": verdict["quarantined"],
        "quarantine_reason": verdict["quarantine_reason"],
    }


def _repo_rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------
# Cells
# --------------------------------------------------------------------------

#: (label, predicate). Order is the printed order and is fixed, so a block is
#: byte-stable across machines.
CELLS = (
    ("all partner breaks", lambda r: True),
    ("single-turn", lambda r: not r["multiturn"]),
    ("multi-turn (OOD)", lambda r: r["multiturn"]),
    ("trainable type", lambda r: r["trainable"]),
    ("held-out type", lambda r: not r["trainable"]),
    ("category: reframing", lambda r: r["category"] == "reframing"),
    ("category: spec_edge", lambda r: r["category"] == "spec_edge"),
    ("category: incremental_erosion", lambda r: r["category"] == "incremental_erosion"),
)


def summarize(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Per-cell {n, scored, broke, held, quarantined}. Quarantined rows are in
    ``n`` and out of ``scored``, so no denominator moves quietly."""
    out: Dict[str, Any] = {}
    for label, pred in CELLS:
        cell = [r for r in rows if pred(r)]
        q = [r for r in cell if r["quarantined"]]
        broke = [r for r in cell if r["broke"] is True]
        held = [r for r in cell if r["broke"] is False]
        out[label] = {"n": len(cell), "scored": len(cell) - len(q),
                      "broke": len(broke), "held": len(held), "quarantined": len(q)}
    return out


# --------------------------------------------------------------------------
# Re-derivation: the run JSON is evidence, not testimony
# --------------------------------------------------------------------------

def rederive(run: Mapping[str, Any], exam: Sequence[Mapping[str, Any]]
             ) -> List[Dict[str, Any]]:
    """Re-run the checkers over a committed run's stored outputs.

    Returns rows with FRESH verdicts and raises on any disagreement with what the
    run stored, on any prompt that was not the exam's, and on any coverage gap. A
    run JSON that cannot survive this is not evidence about a model.
    """
    if run.get("schema") != SCHEMA:
        raise ReplayError(f"run schema {run.get('schema')!r} != {SCHEMA!r}")
    rows = run.get("rows")
    if not isinstance(rows, list):
        raise ReplayError(f"run {run.get('label')!r} has no rows list")
    coverage_check(exam, rows)
    by_id = {r["id"]: (i, r) for i, r in enumerate(exam)}

    fresh: List[Dict[str, Any]] = []
    problems: List[str] = []
    for row in rows:
        i, item = by_id[row["id"]]
        constraint = constraint_for(item, i)
        where = f"{run.get('label')}/{row['id']}"
        for field in ("category", "family", "arg", "split", "trainable", "multiturn"):
            if row.get(field) != item.get(field):
                problems.append(f"{where}: stored {field}={row.get(field)!r} but the "
                                f"exam says {item.get(field)!r}")
        if row.get("constraint_type") != constraint["type"]:
            problems.append(f"{where}: stored constraint_type "
                            f"{row.get('constraint_type')!r} != {constraint['type']!r}")
        if row.get("system_prompt") is not None:
            problems.append(f"{where}: a system prompt was sent; the partner's attacks "
                            f"landed with none")
        if row.get("turns_sent") != turns_of(item["prompt"]):
            problems.append(f"{where}: the prompt sent was not the exam's prompt")

        verdict = score_output(constraint, row.get("output"))
        if row.get("quarantined") is not verdict["quarantined"]:
            problems.append(f"{where}: stored quarantined={row.get('quarantined')!r}, "
                            f"re-derived {verdict['quarantined']!r}")
        elif row.get("broke") != verdict["broke"]:
            problems.append(f"{where}: stored broke={row.get('broke')!r}, re-derived "
                            f"{verdict['broke']!r} ({verdict['evidence'][:80]!r})")
        reason, evidence = verdict["quarantine_reason"], verdict["evidence"]
        if reason == Q_NO_OUTPUT and row.get("quarantine_reason") == Q_GENERATOR:
            # NOT reading a stored verdict: both the stored row and the fresh
            # re-derivation agree this row HAS no verdict, and that is the only
            # thing being carried over. A generator's exception simply cannot be
            # re-derived from a JSON, and dropping it would make every crashed
            # row read as a merely mute one.
            reason, evidence = Q_GENERATOR, str(row.get("verdict", {}).get("evidence", ""))
        merged = dict(row)
        merged.update({"verdict": {"status": verdict["status"], "evidence": evidence},
                       "broke": verdict["broke"],
                       "quarantined": verdict["quarantined"],
                       "quarantine_reason": reason})
        fresh.append(merged)
    if problems:
        raise ReplayError("the committed run disagrees with a fresh re-derivation:\n"
                          + "\n".join(f"  - {p}" for p in problems))
    return fresh


# --------------------------------------------------------------------------
# The generated block
# --------------------------------------------------------------------------

def _cell_str(stat: Mapping[str, int]) -> str:
    """"broke/scored" plus a quarantine tail when one exists."""
    s = f"{stat['broke']}/{stat['scored']}"
    return s + f" (+{stat['quarantined']} q)" if stat["quarantined"] else s


def render_block(runs: Sequence[Mapping[str, Any]],
                 exam: Sequence[Mapping[str, Any]],
                 exam_path: Path = DEFAULT_FROZEN) -> str:
    """The marker-fenced generated block, from FRESHLY re-derived verdicts."""
    if not runs:
        raise ReplayError("no run JSONs given; the block reports measurements")
    labels = [r["label"] for r in runs]
    if len(set(labels)) != len(labels):
        raise ReplayError(f"two runs share a label {labels}; each column must name "
                          f"a distinct model")
    # A run JSON records the exam digest it was measured against. If that is not
    # the exam being re-derived against, the columns did not face the same 20
    # prompts -- and the block would print the run's own bogus digest as though
    # it were provenance.
    exam_digest = freeze_breaks.frozen_sha256(exam_path)
    for run in runs:
        if run.get("exam_sha256") != exam_digest:
            raise ReplayError(
                f"run {run.get('label')!r} stores exam sha256 "
                f"{run.get('exam_sha256')!r}, but {_repo_rel(exam_path)} is "
                f"{exam_digest}. The runs in one block must face the same exam; "
                f"re-run against the committed artifact or restore it.")
    fresh = [rederive(r, exam) for r in runs]
    stats = [summarize(rows) for rows in fresh]

    heldout = [r["id"] for r in exam if not r["trainable"]]
    htypes = sorted({r["constraint_type"] for r in exam if not r["trainable"]})

    out: List[str] = [BEGIN_BLOCK]
    out += [
        "<!-- Generated by `scripts/replay_breaks.py --emit-block`. Every count",
        "     below is RE-DERIVED by re-running constraints.checkers.check over",
        "     the model outputs stored in the run JSONs -- no stored verdict is",
        "     read. `replay_breaks.py --check <this file>` regenerates the block",
        "     and refuses the document unless the two are byte-identical, so",
        "     editing a digit here fails the check rather than changing the claim. -->",
        "",
        f"- exam: `{_repo_rel(exam_path)}` sha256 "
        f"`{exam_digest}`, {len(exam)} rows",
    ]
    for run in runs:
        out.append(f"- run `{run['label']}`: model `{run['model']}`, exam sha256 "
                   f"`{run['exam_sha256']}`")
    out.append("")

    # Built as a list of cells and joined once. An earlier version appended the
    # paired column by string-surgery on the finished row and swallowed a
    # separator, which produced a table markdown renders with the wrong number
    # of columns -- and a wrong table is a wrong claim.
    paired = len(runs) == 2
    head = ["cell", "n"] + [f"`{l}` still breaks" for l in labels]
    if paired:
        head.append(f"fixed by `{labels[1]}`")
    out.append("| " + " | ".join(head) + " |")
    out.append("| " + " | ".join(["---", "---:"] + ["---:"] * (len(head) - 2)) + " |")

    for label, pred in CELLS:
        cells = [label, str(stats[0][label]["n"])] + [_cell_str(s[label]) for s in stats]
        if paired:
            cells.append(_fixed(fresh[0], fresh[1], pred))
        out.append("| " + " | ".join(cells) + " |")
    out.append("")

    out.append(f"**Held-out type.** `" + "`, `".join(heldout) + "` are "
               f"`{'`, `'.join(htypes)}`, whose `constraints.registry.split_of` is "
               f"`heldout_type`. `datagen/audit.py` fails any dataset that uses a "
               f"held-out type, so these {len(heldout)} rows can be measured forever "
               f"and trained on never -- a fix there is generalisation, not recall.")
    out.append("")
    out.append("**Quarantine.** Rows with no derivable verdict. Counted in `n`, "
               "excluded from every `broke/scored`, named here so a model that "
               "errors cannot read as a model that held.")
    out.append("")
    any_q = False
    for run, rows in zip(runs, fresh):
        for row in rows:
            if row["quarantined"]:
                any_q = True
                out.append(f"* `{run['label']}` / `{row['id']}` "
                           f"(`{row['constraint_type']}`) -- "
                           f"{row['quarantine_reason']}: {row['verdict']['evidence']}")
    if not any_q:
        out.append("* none.")
    out.append(END_BLOCK)
    return "\n".join(out)


def _fixed(before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]],
           pred) -> str:
    """Paired count: broke before, held after. Pairs where either side has no
    verdict are excluded and the denominator says how many pairs remain."""
    b = {r["id"]: r for r in before if pred(r)}
    a = {r["id"]: r for r in after if pred(r)}
    paired = [i for i in b if i in a and b[i]["broke"] is not None
              and a[i]["broke"] is not None]
    fixed = [i for i in paired if b[i]["broke"] and not a[i]["broke"]]
    broke_before = [i for i in paired if b[i]["broke"]]
    return f"{len(fixed)}/{len(broke_before)}" if broke_before else f"0/0 ({len(paired)} paired)"


def check_document(doc: Path, block: str) -> List[str]:
    """The document's block must be the one just generated. Byte for byte."""
    if not doc.exists():
        return [f"{doc} does not exist"]
    text = doc.read_text(encoding="utf-8")
    n_begin, n_end = text.count(BEGIN_BLOCK), text.count(END_BLOCK)
    if n_begin != 1 or n_end != 1:
        return [f"{doc} holds {n_begin} {BEGIN_BLOCK!r} and {n_end} {END_BLOCK!r} "
                f"marker(s); exactly one of each is required. The block is generated, "
                f"not written: run `python scripts/replay_breaks.py --emit-block "
                f"<run json>...` and paste what it prints, markers included."]
    lines = text.splitlines()
    i0 = next(i for i, ln in enumerate(lines) if BEGIN_BLOCK in ln)
    i1 = next(i for i, ln in enumerate(lines) if END_BLOCK in ln)
    if i1 < i0:
        return [f"{doc} closes the generated block at line {i1 + 1}, before it opens "
                f"it at line {i0 + 1}"]
    found = "\n".join(lines[i0:i1 + 1])
    if found == block:
        return []
    diff = "\n".join("      " + d for d in difflib.unified_diff(
        found.splitlines(), block.splitlines(),
        fromfile=f"{doc} lines {i0 + 1}-{i1 + 1} (as committed)",
        tofile="re-derived from the committed run JSONs", lineterm="", n=1))
    return [f"{doc}'s generated block is not the one re-derived from the run JSONs. "
            f"It is generated output, not prose: regenerate it with "
            f"`python scripts/replay_breaks.py --emit-block <run json>...`.\n{diff}"]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def load_run(path: Path) -> Dict[str, Any]:
    run = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(run, dict):
        raise ReplayError(f"{path}: run JSON is not an object")
    return run


def write_run(run: Mapping[str, Any], out: Path) -> str:
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(run, indent=2, ensure_ascii=False, sort_keys=False) + "\n"
    out.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="*", help="run JSONs, for --emit-block / --check")
    ap.add_argument("--model", help="model spec for eval.make_generator, "
                                    "e.g. hf:Qwen/Qwen3-1.7B@<sha>")
    ap.add_argument("--out", help="where to write the run JSON, "
                                  "e.g. runs/breaks/base.json")
    ap.add_argument("--label", help="column name for this run (default: --out's stem)")
    ap.add_argument("--frozen", default=str(DEFAULT_FROZEN))
    ap.add_argument("--emit-block", action="store_true",
                    help="print the marker-fenced generated block for the given runs")
    ap.add_argument("--check", metavar="MARKDOWN",
                    help="re-derive the block from the given runs and fail unless the "
                         "document's block is byte-identical")
    ap.add_argument("--allow-unpinned-exam", action="store_true",
                    help="permit an exam whose sha256 differs from the pin "
                         "(for developing a NEW exam; never for a graded run)")
    args = ap.parse_args(argv)
    frozen = Path(args.frozen)

    try:
        exam = load_exam(frozen, require_pin=not args.allow_unpinned_exam)

        if args.check or args.emit_block:
            if not args.runs:
                raise ReplayError("--emit-block/--check need at least one run JSON")
            runs = [load_run(Path(p)) for p in args.runs]
            block = render_block(runs, exam, frozen)
            if args.emit_block:
                print(block)
            if args.check:
                problems = check_document(Path(args.check), block)
                for p in problems:
                    print(f"FAIL: {p}", file=sys.stderr)
                if problems:
                    return 1
                print(f"OK: {args.check}'s generated block matches the run JSONs.")
            return 0

        if not args.model or not args.out:
            ap.error("give --model and --out to replay, or --emit-block/--check to "
                     "report on runs already committed")
        out = Path(args.out)
        run = replay(args.model, exam, frozen_path=frozen,
                     label=args.label or out.stem,
                     progress=lambda m: print(m, flush=True))
        digest = write_run(run, out)
        s = run["summary"]
        print(f"\nexam        : {run['exam']} sha256 {run['exam_sha256']}")
        for label, _pred in CELLS:
            print(f"{label:<32} {_cell_str(s[label])} broke")
        print(f"run json    : {out} sha256 {digest}")
        return 0
    except (ReplayError, FreezeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
