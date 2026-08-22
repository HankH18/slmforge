"""Bind `scripts/adversarial_report.py`, its stored artefact, and `docs/ADVERSARIAL_EVAL.md`.

The script already carries a real gate: `--check` re-derives every figure from the
committed transcripts and exits 1 when `runs/adversarial_v2/adversarial_report.json`
disagrees, and — since the script grew `document_bindings()` — when a figure in the
Results section of `docs/ADVERSARIAL_EVAL.md` disagrees too. That gate was reachable
only by someone remembering to type it, which this repository does not count as a gate.
These tests run it from the ordinary suite.

What is asserted, and what each assertion would catch:

* `--check` as a subprocess, exit 0 — a hand-edited digit in the stored JSON, a
  transcript that no longer re-derives to it, or a figure in the document's Results
  section that the transcripts do not produce.
* An INDEPENDENT comparison of the same document against the same script: every table
  the script prints that also appears in the document, cell by cell. This is not a
  duplicate of `--check`. That gate compares a hand-maintained list of bindings, so a
  table row nobody added to `document_bindings()` is bound by nothing; this test binds
  whole tables by shape, and it keeps binding them if `--check` is ever weakened.
* Non-vacuity, three ways: the script's tables must parse to a non-empty set of rows;
  every row of every matched table must actually be visited (the count is derived from
  the script side, never typed here); and a positive control mutates one digit and
  asserts the comparator reports it. A comparison that silently matched nothing would
  pass the first two and fail the third.

NO STATISTIC IS WRITTEN INTO THIS FILE. Both sides of every comparison are derived at
run time, so a figure that is mid-correction moves on both sides at once and these
tests stay green; only a DISAGREEMENT between script, artefact and document reddens.

Network-free and model-free: the script reads committed transcripts, and nothing here
runs a model or touches MPS.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "adversarial_report.py"
DOC = REPO_ROOT / "docs" / "ADVERSARIAL_EVAL.md"
ARTIFACT = REPO_ROOT / "runs" / "adversarial_v2" / "adversarial_report.json"

#: Structural floors for the doc/script comparison. These are counts of TABLES and
#: ROWS -- shape, not results -- and they exist so that a comparison which matched
#: nothing cannot pass. They are deliberately below the current shape (4 tables /
#: 20 rows) so that a document edit which adds or drops one table does not redden a
#: test about digits. The exact-coverage assertion below is the sharp one.
MIN_MATCHED_TABLES = 3
MIN_MATCHED_ROWS = 12


# ------------------------------------------------------------------ markdown parsing
def _norm(cell: str) -> str:
    """Compare cells for their content, not their typography.

    The document and the script differ in ways that are not claims: `95 % CI` against
    `95% CI`, U+2212 MINUS SIGN against ASCII `-`, backticks and bold around a value.
    Digits are never touched, so no mutation of a number can survive this.
    """
    s = cell.replace("−", "-").replace("`", "").replace("*", "")
    return "".join(s.split()).lower()


def _is_rule(cells: List[str]) -> bool:
    """The `|---|---|` separator under a markdown header."""
    return all(set(c.strip()) <= set("-: ") and c.strip() for c in cells)


def parse_tables(text: str) -> List[Tuple[List[str], List[List[str]]]]:
    """Every pipe table in `text`, as (header cells, data rows), normalised."""
    blocks: List[List[List[str]]] = []
    cur: List[List[str]] = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("|") and s.endswith("|") and len(s) > 1:
            cur.append([c.strip() for c in s[1:-1].split("|")])
        elif cur:
            blocks.append(cur)
            cur = []
    if cur:
        blocks.append(cur)

    out = []
    for b in blocks:
        if len(b) < 3 or not _is_rule(b[1]):
            continue
        header = [_norm(c) for c in b[0]]
        rows = [[_norm(c) for c in r] for r in b[2:] if not _is_rule(r)]
        if rows:
            out.append((header, rows))
    return out


def table_key(header: List[str], rows: List[List[str]]) -> Tuple:
    """Identify a table by its shape, not by any value it reports.

    The row labels (first column) plus the column count. Two of the script's tables
    share their row labels -- the per-family targeted table and the attacked-versus-
    control table are both the six target-carrying families plus `pooled` -- and the
    column count separates them. Nothing here is a statistic, so a corrected figure
    does not move a key.
    """
    return (tuple(sorted(r[0] for r in rows)), len(header))


def index_unique(tables) -> Dict[Tuple, Tuple[List[str], List[List[str]]]]:
    """Tables by key, dropping any key that is not unique in this document."""
    seen: Dict[Tuple, int] = {}
    idx: Dict[Tuple, Tuple] = {}
    for header, rows in tables:
        k = table_key(header, rows)
        seen[k] = seen.get(k, 0) + 1
        idx[k] = (header, rows)
    return {k: v for k, v in idx.items() if seen[k] == 1}


def compare(script_tables, doc_text: str):
    """Compare every script table that the document also carries.

    Returns (mismatches, matched_table_count, compared_row_count, compared_cell_count).
    A mismatch is (key, row label, column index, doc cell, script cell).
    """
    doc_idx = index_unique(parse_tables(doc_text))
    script_idx = index_unique(script_tables)
    mismatches = []
    tables = rows_seen = cells = 0
    for key, (s_header, s_rows) in script_idx.items():
        if key not in doc_idx:
            continue
        d_header, d_rows = doc_idx[key]
        tables += 1
        d_by_label = {r[0]: r for r in d_rows}
        for s_row in s_rows:
            d_row = d_by_label.get(s_row[0])
            if d_row is None:            # key matched on the label SET, so unreachable
                mismatches.append((key, s_row[0], -1, "<missing row>", "|".join(s_row)))
                continue
            rows_seen += 1
            if len(d_row) != len(s_row):
                mismatches.append((key, s_row[0], -1,
                                   f"{len(d_row)} cells", f"{len(s_row)} cells"))
                continue
            for i, (d_cell, s_cell) in enumerate(zip(d_row, s_row)):
                cells += 1
                if d_cell != s_cell:
                    col = d_header[i] if i < len(d_header) else f"col{i}"
                    mismatches.append((key, s_row[0], col, d_cell, s_cell))
    return mismatches, tables, rows_seen, cells


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def check_proc():
    """The gate on the stored artefact AND on the document's Results section, run on
    its own so its verdict is attributable to it alone."""
    return subprocess.run([sys.executable, str(SCRIPT), "--check"],
                          cwd=str(REPO_ROOT), capture_output=True, text=True,
                          timeout=900)


@pytest.fixture(scope="module")
def markdown_proc(tmp_path_factory):
    """The tables, derived from the transcripts, with the write redirected.

    Deliberately NOT `--check --markdown`: a stale artefact or document would then
    suppress the tables, and the comparisons below would fail for `--check`'s reason
    instead of their own, which is the opposite of an independent check.

    `--markdown` is documented as read-only, so `--out` is belt and braces: it points
    the artefact path at a scratch file, so if printing the report ever regains the
    write side effect it once had, this test still cannot regenerate — and so repair —
    the artefact it exists to gate.
    """
    scratch = tmp_path_factory.mktemp("adv") / "report.json"
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--markdown", "--out", str(scratch)],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=900)


# ------------------------------------------------------------------ the gate
def test_check_re_derives_the_stored_report(check_proc):
    """THE ONE THAT MATTERS. `--check` re-runs every checker from (constraint, output),
    refuses `adversarial_report.json` if the transcripts do not produce it byte for
    byte, and refuses `docs/ADVERSARIAL_EVAL.md` if a figure in its Results section is
    not what those transcripts re-derive to. Running it here is what turns a command
    someone remembers into a gate the suite enforces."""
    proc = check_proc
    assert proc.returncode == 0, (
        f"adversarial_report.py --check exited {proc.returncode}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")


def test_stored_report_exists_and_is_not_empty():
    """Non-vacuity for the artefact itself: `--check` compares text, and comparing two
    empty files also succeeds. Assert the report actually carries scored rows."""
    assert ARTIFACT.exists(), f"{ARTIFACT} is missing"
    rep = json.loads(ARTIFACT.read_text())
    assert set(rep["aggregate"]) >= {"base", "tuned"}
    for cell, agg in rep["aggregate"].items():
        assert agg["n_scored"] > 0, f"{cell} scored no rows"
    assert rep["targeted_constraint"]["pooled"]["n"] > 0, (
        "the targeted-constraint measure covers no rows")
    assert rep["spec_pass_by_family"], "no family carries a spec_pass cell"


# ------------------------------------------------------------------ doc vs script
def test_document_tables_match_the_script(markdown_proc):
    """THE ONE THAT MATTERS for the document. Every Results table in
    `docs/ADVERSARIAL_EVAL.md` that the script also prints is compared cell by cell
    against the script's own output, so a digit retyped by hand reddens here instead
    of shipping. Both sides are derived at run time; nothing is hardcoded."""
    proc = markdown_proc
    assert proc.returncode == 0, f"the script failed before the tables:\n{proc.stderr}"

    script_tables = parse_tables(proc.stdout)
    assert script_tables, "the script printed no markdown tables to compare against"

    mismatches, tables, rows, cells = compare(script_tables, DOC.read_text())

    detail = "\n".join(
        f"  table {k[0][:3]}... col {c}: doc {d!r} vs script {s!r} (row {lbl!r})"
        for k, lbl, c, d, s in mismatches[:20])
    assert not mismatches, (
        f"{DOC.name} disagrees with `adversarial_report.py --markdown` in "
        f"{len(mismatches)} cell(s). The script is the derivation; the document is a "
        f"transcription of it. Re-run `python scripts/adversarial_report.py "
        f"--markdown` and paste, or explain why the rows changed.\n{detail}")

    # Non-vacuity. A comparison that matched no table would report zero mismatches.
    assert tables >= MIN_MATCHED_TABLES, (
        f"only {tables} of the script's tables were found in {DOC.name}; the "
        f"comparison is not covering the Results section")
    assert rows >= MIN_MATCHED_ROWS, f"only {rows} rows were compared"
    assert cells > rows, "each row must contribute more than its label"


def test_the_headline_tables_are_the_ones_being_compared(markdown_proc):
    """Non-vacuity, sharpened: name the two tables the document's result rests on and
    require that they are among the ones compared, identified by their SHAPE (the
    columns they carry) rather than by any value in them. Without this, a document
    that dropped R1 and R2 and kept three incidental tables would still clear the
    floor above."""
    proc = markdown_proc
    assert proc.returncode == 0, (
        f"adversarial_report.py --markdown exited {proc.returncode}, so there are no "
        f"tables to match:\n{proc.stderr}")
    script_idx = index_unique(parse_tables(proc.stdout))
    doc_idx = index_unique(parse_tables(DOC.read_text()))
    matched = [(h, r) for k, (h, r) in script_idx.items() if k in doc_idx]
    assert matched, "no table of the script's appears in the document at all"

    headers = [h for h, _ in matched]
    assert any("spec_adherence" in h for h in headers), (
        "the aggregate table (R1) is not among the tables compared against the "
        "document, so its numbers are bound to nothing")
    assert any("baseheld" in h and "tunedheld" in h for h in headers), (
        "the targeted-constraint hold table (R2) -- the document's one resolving "
        "result -- is not among the tables compared against the document")


def test_every_row_of_every_matched_table_is_visited(markdown_proc):
    """The non-vacuity guard proper. A comparison loop that passes over zero rows is
    the failure this test is designed against, so assert the visited-row count equals
    the number of rows the SCRIPT emitted for exactly the tables that matched -- a
    number derived from the script, never typed into this file."""
    proc = markdown_proc
    assert proc.returncode == 0, (
        f"adversarial_report.py --markdown exited {proc.returncode}, so there is "
        f"nothing to count:\n{proc.stderr}")
    script_tables = parse_tables(proc.stdout)
    script_idx = index_unique(script_tables)
    doc_idx = index_unique(parse_tables(DOC.read_text()))

    expected_rows = sum(len(rows) for k, (_, rows) in script_idx.items() if k in doc_idx)
    _, tables, rows, _ = compare(script_tables, DOC.read_text())

    assert expected_rows > 0, "no matched table carries any row"
    assert rows == expected_rows, (
        f"the comparison visited {rows} rows but the {tables} matched tables carry "
        f"{expected_rows}; rows are being skipped")


def test_the_comparison_reddens_on_a_changed_digit(markdown_proc):
    """POSITIVE CONTROL. The three tests above prove the document agrees with the
    script. They do NOT prove the comparator can tell disagreement from agreement --
    a parser that matched nothing would report the same silence. So feed the script's
    own output back in as a synthetic document (which must compare clean), change one
    digit in it, and require that exact cell to be reported."""
    proc = markdown_proc
    assert proc.returncode == 0, (
        f"adversarial_report.py --markdown exited {proc.returncode}, so the control "
        f"has no tables to work on:\n{proc.stderr}")
    script_tables = parse_tables(proc.stdout)

    clean, tables, rows, cells = compare(script_tables, proc.stdout)
    assert not clean, "the script's own output does not compare clean against itself"
    assert tables > 0 and rows > 0 and cells > 0, (
        "the comparator matched nothing even against an identical document, so its "
        "silence on the real document means nothing")

    mutated, changed = _bump_one_digit(proc.stdout)
    assert changed is not None, "found no digit to mutate in the script's tables"
    found, _, _, _ = compare(script_tables, mutated)
    assert found, (
        f"changing {changed} in a table cell produced no mismatch; the comparison "
        f"is not binding the document's digits")


def _bump_one_digit(text: str):
    """Change one digit in the first VALUE cell of the first table, leaving row
    labels alone so the table still matches and the mutation lands in a compared cell.

    Returns (mutated text, a description of what moved) so the failure message can
    name the mutation rather than asserting on a silence.
    """
    lines = text.splitlines()
    in_table = False
    for i, line in enumerate(lines):
        s = line.strip()
        if not (s.startswith("|") and s.endswith("|") and len(s) > 1):
            in_table = False
            continue
        cells = [c.strip() for c in s[1:-1].split("|")]
        if _is_rule(cells):
            in_table = True
            continue
        if not in_table:
            continue
        for ci in range(1, len(cells)):           # never the row label
            for j, ch in enumerate(cells[ci]):
                if ch.isdigit():
                    new = str((int(ch) + 1) % 10)
                    before = cells[ci]
                    cells[ci] = before[:j] + new + before[j + 1:]
                    lines[i] = "| " + " | ".join(cells) + " |"
                    return ("\n".join(lines),
                            f"{before!r} -> {cells[ci]!r} on line {i + 1}")
    return text, None
