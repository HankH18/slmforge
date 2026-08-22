"""Bind `scripts/composition_report.py` to `docs/COMPOSITION.md` from the ordinary suite.

The script carries two real guards. `--selftest` checks every statistic it computes
against a closed form, and `--check DOC` re-derives the report from the committed
`runs/final` transcripts and exits 1 if the document disagrees -- its generated block,
and every figure its prose states. Neither guard was reachable from `pytest`, so a
green suite said nothing about either: a digit typed by hand into a table between the
markers landed, and a statistic that had drifted from its closed form landed with it.

What is asserted:

* `--selftest` as a subprocess, exit 0 — the Wilson interval, the Cochran-Armitage
  permutation test, the Holm adjustment and McNemar against their known answers.
* `--check docs/COMPOSITION.md` as a subprocess, exit 0 — the document's generated
  block, and the figures its prose states, against the transcripts.
* An INDEPENDENT comparison of the same two things, parsed here rather than inside the
  script. `--check` is the script grading its own transcription; if that comparison
  were ever weakened, this test still reddens on a changed digit.
* Non-vacuity: the block must exist and carry tables, every row of every table must be
  visited (the count derived from the script's output, never typed here), and a
  positive control mutates one digit and requires the comparator to report it.

NO STATISTIC IS WRITTEN INTO THIS FILE. Both sides of every comparison are re-derived
at run time, so a figure that is mid-correction moves on both sides at once; only a
DISAGREEMENT between script and document reddens.

Network-free and model-free: the script reads committed transcripts under `runs/final`,
and nothing here runs a model.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "composition_report.py"
DOC = REPO_ROOT / "docs" / "COMPOSITION.md"
MARKER_BEGIN = "<!-- BEGIN composition_report -->"
MARKER_END = "<!-- END composition_report -->"

#: Structural floors, so a comparison that matched nothing cannot pass. Counts of
#: tables and rows -- shape, not results -- set below the current shape (14 tables,
#: 144 rows) so that adding or dropping one table does not redden a test about digits.
MIN_TABLES = 10
MIN_ROWS = 120


# ------------------------------------------------------------------ markdown parsing
def _norm(cell: str) -> str:
    """Compare cells for content, not typography: backticks, bold and whitespace go,
    U+2212 MINUS SIGN folds to ASCII. Digits are never touched."""
    s = cell.replace("−", "-").replace("`", "").replace("*", "")
    return "".join(s.split()).lower()


def _is_rule(cells: List[str]) -> bool:
    return all(set(c.strip()) <= set("-: ") and c.strip() for c in cells)


def parse_tables(text: str) -> List[Tuple[List[str], List[List[str]]]]:
    """Every pipe table in `text`, as (header, data rows), normalised."""
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
        rows = [[_norm(c) for c in r] for r in b[2:] if not _is_rule(r)]
        if rows:
            out.append(([_norm(c) for c in b[0]], rows))
    return out


def compare(script_tables, doc_tables):
    """Positional comparison: the block is emitted verbatim by `markdown_tables`, so
    the i-th table in the document is the i-th table the script printed.

    Returns (mismatches, rows visited, cells compared). A structural difference (a
    table count, a header, a row count) is itself a mismatch rather than a reason to
    stop comparing quietly.
    """
    mismatches = []
    rows_seen = cells = 0
    if len(script_tables) != len(doc_tables):
        mismatches.append(
            f"the document's block has {len(doc_tables)} tables and the script "
            f"printed {len(script_tables)}")
    for t, (s, d) in enumerate(zip(script_tables, doc_tables)):
        s_header, s_rows = s
        d_header, d_rows = d
        if s_header != d_header:
            mismatches.append(f"table {t}: header {d_header} vs script {s_header}")
            continue
        if len(s_rows) != len(d_rows):
            mismatches.append(
                f"table {t} ({'|'.join(s_header)}): {len(d_rows)} rows in the "
                f"document, {len(s_rows)} from the script")
            continue
        for r, (s_row, d_row) in enumerate(zip(s_rows, d_rows)):
            rows_seen += 1
            if len(s_row) != len(d_row):
                mismatches.append(f"table {t} row {r}: cell count differs")
                continue
            for i, (s_cell, d_cell) in enumerate(zip(s_row, d_row)):
                cells += 1
                if s_cell != d_cell:
                    col = s_header[i] if i < len(s_header) else f"col{i}"
                    mismatches.append(
                        f"table {t} row {r} ({d_row[0]}) column {col}: document "
                        f"{d_cell!r} vs script {s_cell!r}")
    return mismatches, rows_seen, cells


def generated_block(doc_text: str) -> str:
    body = doc_text.split(MARKER_BEGIN, 1)[1]
    return body.split(MARKER_END, 1)[0]


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def markdown_run():
    """One derivation of the tables from the committed transcripts. Module-scoped
    because the witness pass makes it the slow part; the script writes nothing, so
    running it cannot disturb what it is being used to gate."""
    return subprocess.run([sys.executable, str(SCRIPT), "--markdown"],
                          cwd=str(REPO_ROOT), capture_output=True, text=True,
                          timeout=1800)


# ------------------------------------------------------------------ the script's guards
def test_selftest_passes():
    """`--selftest` checks the Wilson interval, the Cochran-Armitage trend statistic
    against its exact permutation distribution, Holm, and McNemar against closed
    forms. It needs no data, so it is the cheap half of this file's job -- and it was
    reachable only by hand until now."""
    proc = subprocess.run([sys.executable, str(SCRIPT), "--selftest"],
                          cwd=str(REPO_ROOT), capture_output=True, text=True,
                          timeout=600)
    assert proc.returncode == 0, (
        f"composition_report.py --selftest exited {proc.returncode}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")


def test_check_gates_the_document():
    """THE ONE THAT MATTERS. `--check` re-derives the report from `runs/final` and
    exits 1 if `docs/COMPOSITION.md` is not what the transcripts produce -- the
    generated block, and every figure the prose around it states. Running it here is
    what makes a hand-edited digit a red suite rather than a command someone might
    have run."""
    proc = subprocess.run([sys.executable, str(SCRIPT), "--check", str(DOC)],
                          cwd=str(REPO_ROOT), capture_output=True, text=True,
                          timeout=1800)
    assert proc.returncode == 0, (
        f"composition_report.py --check {DOC.name} exited {proc.returncode}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")


# ------------------------------------------------------------------ the block itself
def test_the_document_has_a_generated_block():
    """Non-vacuity for the markers. `--check` fails loudly when they are absent, but
    a block present and EMPTY would leave the document's prose figures bound to
    nothing while every gate stayed green."""
    text = DOC.read_text()
    assert MARKER_BEGIN in text and MARKER_END in text, (
        f"{DOC.name} has no {MARKER_BEGIN} / {MARKER_END} block")
    tables = parse_tables(generated_block(text))
    assert len(tables) >= MIN_TABLES, (
        f"the generated block holds {len(tables)} tables; the document's results are "
        f"not being carried by it")
    assert all(rows for _, rows in tables), "a table in the block has no rows"


def test_block_matches_the_script_independently(markdown_run):
    """The same comparison `--check` makes, parsed here instead of inside the script,
    so the binding survives `--check` itself being weakened. Both sides derived at run
    time; no figure is written into this file."""
    proc = markdown_run
    assert proc.returncode == 0, (
        f"composition_report.py --markdown exited {proc.returncode}\n{proc.stderr}")

    script_tables = parse_tables(proc.stdout)
    doc_tables = parse_tables(generated_block(DOC.read_text()))
    assert script_tables, "the script printed no markdown tables"

    mismatches, rows, cells = compare(script_tables, doc_tables)
    assert not mismatches, (
        f"{DOC.name}'s generated block disagrees with `composition_report.py "
        f"--markdown` in {len(mismatches)} place(s). Re-run the script and paste the "
        f"result between the markers, or explain why the rows changed.\n  "
        + "\n  ".join(mismatches[:20]))

    assert rows >= MIN_ROWS, f"only {rows} rows were compared"
    assert cells > rows, "each row must contribute more than its label"


def test_every_row_is_visited(markdown_run):
    """The non-vacuity guard proper. A comparison loop that passes over zero rows is
    the failure this file is designed against, so require the visited-row count to
    equal the number of rows the SCRIPT emitted -- derived, never typed here."""
    proc = markdown_run
    assert proc.returncode == 0, (
        f"composition_report.py --markdown exited {proc.returncode}, so there is "
        f"nothing to count:\n{proc.stderr}")
    script_tables = parse_tables(proc.stdout)
    expected = sum(len(rows) for _, rows in script_tables)
    _, rows, _ = compare(script_tables, parse_tables(generated_block(DOC.read_text())))
    assert expected > 0, "the script emitted no rows at all"
    assert rows == expected, (
        f"the comparison visited {rows} rows but the script emitted {expected}; rows "
        f"are being skipped")


def test_the_comparison_reddens_on_a_changed_digit(markdown_run):
    """POSITIVE CONTROL. The tests above prove agreement. They do not prove the
    comparator can tell agreement from a parser that matched nothing. So compare the
    script's output against itself (must be clean, over a non-zero row count), change
    one digit, and require that cell to be named."""
    proc = markdown_run
    assert proc.returncode == 0, (
        f"composition_report.py --markdown exited {proc.returncode}, so the control "
        f"has no tables to work on:\n{proc.stderr}")
    script_tables = parse_tables(proc.stdout)

    clean, rows, cells = compare(script_tables, script_tables)
    assert not clean, "the script's own tables do not compare clean against themselves"
    assert rows > 0 and cells > 0, (
        "the comparator visited nothing even against identical input, so its silence "
        "on the real document means nothing")

    mutated, changed = _bump_one_digit(proc.stdout)
    assert changed is not None, "found no digit to mutate in the script's tables"
    found, _, _ = compare(script_tables, parse_tables(mutated))
    assert found, (
        f"changing {changed} in a table cell produced no mismatch; the comparison is "
        f"not binding the block's digits")


def _bump_one_digit(text: str):
    """Change one digit in the first VALUE cell of the first table, leaving the row
    label alone. Returns (mutated text, what moved)."""
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
        for ci in range(1, len(cells)):
            for j, ch in enumerate(cells[ci]):
                if ch.isdigit():
                    new = str((int(ch) + 1) % 10)
                    before = cells[ci]
                    cells[ci] = before[:j] + new + before[j + 1:]
                    lines[i] = "| " + " | ".join(cells) + " |"
                    return ("\n".join(lines),
                            f"{before!r} -> {cells[ci]!r} on line {i + 1}")
    return text, None
