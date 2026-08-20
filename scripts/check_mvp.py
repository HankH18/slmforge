#!/usr/bin/env python3
"""
P6 verify command -- decides whether the MVP submission package is real.

    python scripts/check_mvp.py

Offline: no API keys, no GPU, no network. Nothing is taken on trust.

WHAT THE MVP PACKAGE IS (pivot: Forbidden-Constraint Holder)
------------------------------------------------------------
    runs/mvp/                       both models, one eval set, one judge
        <tuned>__zero_shot.results.json      + .transcripts.jsonl
        <base>__zero_shot.results.json       + .transcripts.jsonl
    docs/MVP_REPORT.md              the base-vs-tuned write-up
    configs/models.json             pins 'tuned' (repo@sha) and 'base'

WHY THIS FILE HAD TO BE REWRITTEN
---------------------------------
It was the abstainer harness's gate: it imported ``check_ablation`` (retired to
archive/abstainer/, so the script could not even load) and spoke that project's
rubric -- rubric_hash, gold_label, pred_class, parse_output, entailment. None of
those exist in the pivot. The pivot's ground truth is programmatic: a constraint
either holds in the output or it does not, and the judge scores task quality
only.

NOTHING IS TRUSTED, AND NOTHING IS REIMPLEMENTED
------------------------------------------------
The re-derivation is scripts/gate_check.py's, IMPORTED:

  * ``gate_check.score_cell`` re-runs ``constraints.checkers.check`` over every
    (constraint, output) pair from the row's own raw output, re-derives
    ``spec_pass`` from THOSE verdicts plus the row's quality_score, requires the
    rows to BE the eval set (no flattering subset, no duplicates), and compares
    the stored aggregate against the recomputed one. A transcript edited to
    flatter the model fails even when it is internally self-consistent.
  * QUARANTINE is honoured exactly as gate_check honours it: a row carrying
    ``judge_error``/``generator_error`` has ``spec_pass=None``, is excluded from
    every rate, is counted, and is capped at
    ``gate_check.MAX_QUARANTINE_SHARE``. It is also reported out loud, because a
    package that quarantined a quarter of the eval set is not a measurement even
    when it is under a cap.

There is deliberately ONE implementation of "re-derive a cell from its rows" in
this repo, and it is gate_check's. A second copy here would be a second thing to
keep in sync, and the two would eventually disagree about what passing means.

This module does not import eval.py: it needs no scoring code, and results files
are located by the ``model`` field inside them rather than by a filename stem, so
a renamed file cannot smuggle in a cell (the name is cosmetic; the header is the
claim).

THE REPORT'S NUMBERS ARE CHECKED, NOT ITS VOCABULARY
-----------------------------------------------------
An earlier version only grepped docs/MVP_REPORT.md for the metric NAMES, so the
headline base-vs-tuned figures could be arbitrarily inflated while every gate
stayed green. The version after that validated numbers found under RECOGNISED
headings -- and an adversarial verifier walked a prose claim, two renamed tables
and an extra column straight past it, because a table matching no recognised
heading was never examined at all.

The rule is therefore INVERTED, and the default carries no whitelist: EVERY
percentage, ratio or printed fraction in the report must either be a table cell
this gate matched -- to the precision it is printed at -- against the value
RE-DERIVED from the transcripts, or sit on a line the report explicitly declares
non-metric with a stated reason (`<!-- non-metric: <why> -->`), and such a line
may not also speak metric vocabulary. check_report_metrics spells out exactly
what that admits. A renamed column loses because its numbers are unaccounted
for, not because its name is on a list.

ROBUSTNESS IS NOT OPTIONAL
--------------------------
Robustness is the pivot's headline metric and it is defined on the ADVERSARIAL
subset of the eval set. A package whose eval set holds no adversarial scenarios
produces robustness=None, prints "n/a", and has measured nothing -- so it is not
an MVP package. This gate requires a non-empty adversarial subset and a real
robustness value in BOTH cells.

VERIFICATION REQUIREMENTS (project spec; PIVOT_SPEC R9)
-------------------------------------------------------
  * public HF repo id + exact commit hash for the tuned checkpoint
  * the exact eval-code commit hash
  * a one-command repro line, byte-consistent with README.md
  * raw judge transcripts as JSONL, referenced by the report
  * every path the report cites exists on disk
Whether the HF repo is actually PUBLIC cannot be established offline; this gate
checks the pinned form and that the report states it.

Exit 0 means a grader could pull the pinned model and reproduce the numbers from
the committed artifacts.
"""
import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
_SCRIPTS = str(Path(__file__).resolve().parent)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import judge as judge_mod  # noqa: E402  (rubric hash, re-derived not pinned)
# The G1 gate's re-derivation, imported whole. See the module docstring: this
# file deliberately owns no second copy of it.
from gate_check import (  # noqa: E402
    DEFAULT_INSTANCES_PATH,
    MAX_QUARANTINE_SHARE,
    MIN_SCENARIOS_PER_CELL,
    GateError,
    check_constraints_resolve,
    load_constraint_instances,
    load_eval_set,
    read_json,
    score_cell,
    sha256_of,
)

REQUIRED_STRATEGY = "zero_shot"

#: A public HF checkpoint pinned to a commit: "<owner>/<name>@<40-hex>".
TUNED_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*@[0-9a-f]{40}$")
SHA40_RE = re.compile(r"\b[0-9a-f]{40}\b")
TABLE_SEP_RE = re.compile(r"^\|[\s:|-]+\|$")

#: The metrics gate_check re-derives from the rows, and therefore the only
#: numbers docs/MVP_REPORT.md may print in a metric table.
METRIC_KEYS = ("spec_adherence", "robustness", "fail_rate")
#: A column/row label that reads like a metric but is not one this harness
#: emits: printing it means printing a number no artifact can back.
METRIC_LOOKALIKE_RE = re.compile(
    r"adherence|robust|_rate$|^rate|accuracy|precision|recall|^f1|score|"
    r"violation|compliance|pass_rate")
DELTA_LABEL_RE = re.compile(r"\b(delta|difference|change|improvement|gain|lift)\b")
# first number in a cell, with the unit that says whether it is a percentage
NUM_RE = re.compile(r"([-+]?\d+(?:\.\d+)?)\s*(%|pp|percentage[ _]points?)?", re.I)
NA_CELLS = {"", "-", "--", "—", "–", "n/a", "na", "none", "null", "n\\a"}

#: A cell that is NOTHING BUT a number (optionally a percentage) -- i.e. a value
#: the report is asserting, as opposed to a label that merely contains digits.
FULL_NUM_RE = re.compile(
    r"^[-+]?\d+(?:\.\d+)?[ \t]*(?:%|pp|percentage[ _-]?points?)?$", re.I)

#: The shape a PERFORMANCE claim takes anywhere in the report: a percentage, a
#: ratio in [0, 1] (how this harness stores every rate), or a printed fraction.
#: The lookbehinds keep identifiers out of it -- the `1.7` of `Qwen3-1.7B`, the
#: `1.0` of a version string -- since those are not rates in any reading.
PERF_NUM_RE = re.compile(
    r"[-+]?\d+(?:\.\d+)?[ \t]*(?:%|pp\b|percentage[ _-]?points?\b)"
    r"|(?<![\w./-])[-+]?[01]\.\d+(?!\d)"
    r"|(?<![\w./-])\d+[ \t]*/[ \t]*\d+(?![\w./])",
    re.I)

#: How a report DECLARES that a number is not a performance claim. Line-scoped
#: and block-scoped; both must state a reason.
NON_METRIC_LINE_RE = re.compile(r"<!--\s*non-metric\s*:\s*([^>]*?)\s*-->", re.I)
NON_METRIC_OPEN_RE = re.compile(r"<!--\s*non-metric-block\s*:\s*([^>]*?)\s*-->",
                                re.I)
NON_METRIC_CLOSE_RE = re.compile(r"<!--\s*/\s*non-metric-block\s*-->", re.I)

#: Vocabulary that makes a line a claim ABOUT model performance. A line carrying
#: both this and a number may not be declared non-metric.
METRIC_WORDS_RE = re.compile(
    r"adherence|robust|fail_rate|pass_rate|accuracy|precision|recall|\bf1\b|"
    r"violation|compliance|spec_pass|win|score|lift|improv")

KNOWN_DIRS = {"runs", "data", "docs", "configs", "scripts", "training",
              "datagen", "tests", "constraints", "scenarios"}
PATHY_EXTS = (".jsonl", ".json", ".md", ".py", ".txt", ".yaml", ".yml", ".csv")
STRIP_CHARS = "`()[]{}<>,;:.!\"'*|"
# a path-like run: one or more "<segment>/" then a final segment. Anchored on the
# characters legal in this repo's paths, so it lifts "runs/mvp/x.jsonl" out of a
# markdown link or a table cell without dragging punctuation along.
PATH_RE = re.compile(r"(?:[A-Za-z0-9_.@-]+/)+[A-Za-z0-9_.@*?-]+")


class Gate:
    def __init__(self, title):
        self.title = title
        self.failures = []
        self.notes = []

    def require(self, cond, msg):
        if not cond:
            self.failures.append(msg)
        return bool(cond)

    def fail(self, msg):
        self.failures.append(msg)

    def note(self, msg):
        self.notes.append(msg)

    def finish(self):
        for n in self.notes:
            print(f"  ok   {n}")
        if self.failures:
            print(f"\nFAIL {self.title} — {len(self.failures)} problem(s):")
            for f in self.failures:
                print(f"  - {f}")
            return 1
        print(f"\nPASS {self.title}")
        return 0


def logical_commands(text):
    """Shell-ish lines with backslash continuations joined and whitespace
    collapsed.

    "byte-consistent" is checked after exactly this normalization, so a repro
    command may be wrapped across lines in the report and inline in the README
    and still count as the same command -- but any changed flag or value fails.
    """
    out, buf = [], None
    for raw in text.splitlines():
        ln = raw.strip()
        if ln.startswith("```"):
            continue
        buf = ln if buf is None else f"{buf} {ln}"
        if buf.endswith("\\"):
            buf = buf[:-1].rstrip()
            continue
        out.append(" ".join(buf.split()))
        buf = None
    if buf is not None:
        out.append(" ".join(buf.split()))
    return [c for c in out if c]


def referenced_paths(text):
    """Path-like substrings in the report: must contain '/' and either start in
    a known repo directory or end in a data/doc extension. HF repo ids
    ('owner/name', no extension), URLs and globs are deliberately excluded, and
    markdown-link punctuation is stripped."""
    found = set()
    for raw in PATH_RE.findall(re.sub(r"\w+://\S+", " ", text)):
        tok = raw.strip(STRIP_CHARS).rstrip("/")
        if not tok or "*" in tok or "?" in tok or "<" in tok:
            continue
        head = tok.split("/", 1)[0]
        if head in KNOWN_DIRS or tok.endswith(PATHY_EXTS):
            found.add(tok)
    return sorted(found)


# ------------------------------------------------------ report metric values
def split_row_spans(line):
    """[(cell_text, start, end), ...] for one markdown table row.

    Offsets are into `line`, so a cell this gate has validated can be excluded
    from the whole-document number sweep BY POSITION rather than by re-matching
    its text (two identical numbers elsewhere in the report must not inherit one
    cell's clean bill of health).
    """
    s = line.rstrip()
    bars = [i for i, ch in enumerate(s) if ch == "|"]
    if not bars:
        return []
    spans = [(a + 1, b) for a, b in zip(bars, bars[1:])]
    if bars[-1] + 1 < len(s):                    # a row with no trailing pipe
        spans.append((bars[-1] + 1, len(s)))
    return [(s[a:b].strip(), a, b) for a, b in spans]


def split_row(line):
    return [cell for cell, _, _ in split_row_spans(line)]


def norm_key(cell):
    """A markdown cell -> comparable snake_case token ('Spec-Adherence' -> ...)."""
    t = re.sub(r"[`*_]+", " ", str(cell)).strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", t).strip("_")


def parse_tables(text):
    """[(header_cells, [(line_no, [(cell, start, end), ...]), ...]), ...]."""
    lines = text.splitlines()
    tables, i = [], 0
    while i < len(lines) - 1:
        if lines[i].strip().startswith("|") and TABLE_SEP_RE.match(lines[i + 1].strip()):
            header = split_row(lines[i])
            body, j = [], i + 2
            while j < len(lines) and lines[j].strip().startswith("|"):
                body.append((j, split_row_spans(lines[j])))
                j += 1
            tables.append((header, body))
            i = j
        else:
            i += 1
    return tables


def parse_number_cell(cell):
    """-> (kind, value_as_fraction, tolerance) with kind in num | na | bad.

    A cell printed to k decimals can only be trusted to half a unit in the last
    place, so the tolerance is derived from the printed precision rather than
    being a loose constant: '90.6%' pins the true value to +-0.05pp.
    """
    t = re.sub(r"[`*]", "", str(cell)).strip()
    if t.lower() in NA_CELLS:
        return "na", None, None
    m = NUM_RE.search(t)
    if not m:
        return "bad", None, None
    digits = m.group(1)
    decimals = len(digits.split(".")[1]) if "." in digits else 0
    half_ulp = 0.5 * (10.0 ** -decimals)
    value, tol = float(digits), half_ulp * 1.05
    if m.group(2):                      # printed in percent / percentage points
        value, tol = value / 100.0, tol / 100.0
    return "num", value, tol + 1e-12


def value_cell_kind(cell):
    """'' if the cell is prose or a label; 'num'/'na' if it is a REPORTED VALUE.

    A cell that is nothing but a number is a value the report is asserting. A
    cell that merely contains digits ("base `Qwen/Qwen3-1.7B@70d2...`") is a
    label and is left to the whole-document sweep.
    """
    t = re.sub(r"[`*]", "", str(cell)).strip()
    if not t:
        return ""
    if t.lower() in NA_CELLS:
        return "na"
    return "num" if FULL_NUM_RE.match(t) else ""


def role_of(label, tuned_pin, base_repo):
    """Which committed run a table label refers to: tuned | base | delta | None."""
    low = label.lower()
    tuned_repo = tuned_pin.split("@", 1)[0].lower()
    base_name = base_repo.split("@", 1)[0].lower()
    if tuned_pin.lower() in low or tuned_repo in low:
        return "tuned"
    if base_repo.lower() in low or base_name in low:
        return "base"
    if DELTA_LABEL_RE.search(low) or "\u0394" in label or "\u03b4" in low:
        return "delta"
    if re.search(r"\btuned\b", low):
        return "tuned"
    if re.search(r"\bbase(line)?\b", low):
        return "base"
    return None


def declared_non_metric_lines(gate, text):
    """{line_index: reason} for lines the report EXPLICITLY declares carry no
    performance number.

    This is the ONLY escape from the rule below, and it is deliberately loud: it
    lives in the report itself, next to the number, and it must state a reason.
    A grader reading the report sees every place the author claimed a number is
    not a result.
    """
    exempt, open_at, why = {}, None, ""
    for i, line in enumerate(text.splitlines()):
        if NON_METRIC_CLOSE_RE.search(line):
            if open_at is None:
                gate.fail(f"docs/MVP_REPORT.md line {i + 1} closes a non-metric "
                          f"block that was never opened")
            exempt[i], open_at, why = why, None, ""
            continue
        m = NON_METRIC_OPEN_RE.search(line)
        if m:
            if open_at is not None:
                gate.fail(f"docs/MVP_REPORT.md line {i + 1} opens a non-metric "
                          f"block inside the one opened at line {open_at + 1}")
            open_at, why = i, m.group(1).strip()
            exempt[i] = why
            if not why:
                gate.fail(f"docs/MVP_REPORT.md line {i + 1}: a non-metric block "
                          f"must state WHY the numbers in it are not results "
                          f"(`<!-- non-metric-block: <reason> -->`)")
            continue
        if open_at is not None:
            exempt[i] = why
            continue
        m = NON_METRIC_LINE_RE.search(line)
        if m:
            exempt[i] = m.group(1).strip()
            if not exempt[i]:
                gate.fail(f"docs/MVP_REPORT.md line {i + 1}: a non-metric "
                          f"declaration must state WHY the number is not a result "
                          f"(`<!-- non-metric: <reason> -->`)")
    if open_at is not None:
        gate.fail(f"docs/MVP_REPORT.md opens a non-metric block at line "
                  f"{open_at + 1} and never closes it -- an unterminated block "
                  f"would exempt the whole rest of the report")
    return exempt


def check_report_metrics(gate, text, expected, tuned_pin, base_repo):
    """Every performance number in docs/MVP_REPORT.md must be one this gate has
    matched against the value RE-DERIVED from the committed transcripts -- or be
    explicitly declared not to be a performance number at all.

    THE RULE IS INVERTED ON PURPOSE. The previous version validated numbers found
    under RECOGNISED headings: a table only got inspected if a column header or a
    row label normalised exactly to spec_adherence / robustness / fail_rate, and
    the lookalike guard lived inside those branches. A table matching neither was
    never examined, so with the honest table left intact an adversarial verifier
    walked all of these past the gate:

      a prose claim ("lifts spec adherence from 12.0% to 99.4%");
      a table headed `metric | base | tuned | improvement` with a row
        "constraint adherence";
      a table headed `model | overall adherence | adversarial`;
      an extra "constraint_wins" column inside the real metric table.

    A whitelist of recognised headings will always lose to a renamed column, so
    there is no whitelist in the default any more. The rule is:

      1. In any table this gate CAN orient (it has a recognised metric column or
         a recognised metric row), every body cell that is nothing but a number
         must resolve to a (role, metric) pair and equal the re-derived value.
         An unrecognised column or row label carrying a value is a failure --
         "constraint_wins" is caught by having numbers, not by being on a list.
      2. Every percentage, ratio or printed fraction ANYWHERE in the report --
         prose, headings, unrecognisable tables, captions -- must either sit
         inside a cell rule 1 examined, or on a line the report explicitly
         declares non-metric with a stated reason.
      3. A declared line may not also speak metric vocabulary. The escape hatch
         is for config values, counts and dates; it may not be used to exempt the
         numbers the MVP claim rests on.

    WHERE THE BOUNDARY IS, STATED HONESTLY. A "performance number" is a
    percentage, a ratio in [0, 1] or a printed fraction -- the three shapes every
    rate this harness produces can be printed in. A claim phrased with no such
    number in it ("adherence rises to 99 points") is outside the rule, as is a
    bare count. What the rule guarantees is narrower and worth more than a
    vocabulary scan: the report cannot print a RATE that no committed artifact
    backs, whatever it calls the column it sits in.
    """
    lines = text.splitlines()
    checked, attempted = set(), set()
    accounted = {}                      # line_no -> [(start, end), ...]

    def account(line_no, a, b):
        accounted.setdefault(line_no, []).append((a, b))

    def compare(role, metric, cell, where):
        attempted.add((role, metric))
        kind, val, tol = parse_number_cell(cell)
        exp = expected.get(role, {}).get(metric)
        if kind == "bad":
            gate.fail(f"docs/MVP_REPORT.md: {where} reads {cell!r}, which is not a "
                      f"number that can be checked against the committed "
                      f"transcripts")
            return
        if kind == "na":
            if exp is not None:
                gate.fail(f"docs/MVP_REPORT.md: {where} says {cell!r} but the "
                          f"committed transcripts give {exp!r}")
            else:
                checked.add((role, metric))
            return
        if exp is None:
            gate.fail(f"docs/MVP_REPORT.md: {where} reports {cell!r} but nothing "
                      f"committed has a {metric} value for the {role} run -- the "
                      f"report states a number no artifact backs")
            return
        if abs(val - exp) > tol:
            gate.fail(f"docs/MVP_REPORT.md: {where} reports {cell!r} but the "
                      f"committed transcripts give {exp!r} "
                      f"(|{val:.6f} - {exp:.6f}| > {tol:.6f}) -- the report does "
                      f"not match the artifact it claims to summarize")
            return
        checked.add((role, metric))

    # --- rule 1: every reported value in an orientable table -------------
    for header, body in parse_tables(text):
        hkeys = [norm_key(h) for h in header]
        metric_cols = {i: k for i, k in enumerate(hkeys) if i > 0 and k in METRIC_KEYS}
        metric_rows = [cells for _, cells in body
                       if cells and norm_key(cells[0][0]) in METRIC_KEYS]

        if metric_cols:                                   # rows are models
            # A renamed metric column is a claim even before it carries a value,
            # so a lookalike heading fails on sight.
            for i, key in enumerate(hkeys):
                if i > 0 and key not in METRIC_KEYS and METRIC_LOOKALIKE_RE.search(key):
                    gate.fail(f"docs/MVP_REPORT.md reports a metric column "
                              f"{header[i]!r} that this harness never emits, so no "
                              f"artifact can back it (known metrics: "
                              f"{list(METRIC_KEYS)})")
            for line_no, cells in body:
                if not cells:
                    continue
                role = role_of(cells[0][0], tuned_pin, base_repo)
                for i, (cell, a, b) in enumerate(cells):
                    if i == 0 or not value_cell_kind(cell):
                        continue
                    account(line_no, a, b)
                    metric = metric_cols.get(i)
                    if metric is None:
                        col = header[i] if i < len(header) else f"column {i}"
                        gate.fail(f"docs/MVP_REPORT.md prints the value {cell!r} "
                                  f"under {col!r}, which this harness never emits "
                                  f"-- no artifact can back it (known metrics: "
                                  f"{list(METRIC_KEYS)})")
                        continue
                    if role is None:
                        gate.fail(f"docs/MVP_REPORT.md prints metric numbers on the "
                                  f"row {cells[0][0]!r}, which names neither the "
                                  f"pinned tuned model, the base model, nor a delta "
                                  f"-- there is nothing to check it against")
                        continue
                    compare(role, metric, cell, f"the {role} {metric} cell")

        elif metric_rows:                                 # rows are metrics
            col_roles = {i: role_of(h, tuned_pin, base_repo)
                         for i, h in enumerate(header) if i > 0}
            for line_no, cells in body:
                if not cells:
                    continue
                key = norm_key(cells[0][0])
                for i, (cell, a, b) in enumerate(cells):
                    if i == 0 or not value_cell_kind(cell):
                        continue
                    account(line_no, a, b)
                    if key not in METRIC_KEYS:
                        gate.fail(f"docs/MVP_REPORT.md reports a metric "
                                  f"{cells[0][0]!r} that this harness never emits, "
                                  f"so no artifact can back it (known metrics: "
                                  f"{list(METRIC_KEYS)})")
                        continue
                    role = col_roles.get(i)
                    if role is None:
                        col = header[i] if i < len(header) else f"column {i}"
                        gate.fail(f"docs/MVP_REPORT.md prints a {key} number "
                                  f"under the column {col!r}, which names "
                                  f"neither the pinned tuned model, the base "
                                  f"model, nor a delta")
                        continue
                    compare(role, key, cell, f"the {role} {key} cell")

    # --- rule 2 + 3: every rate-shaped number in the whole document ------
    exempt = declared_non_metric_lines(gate, text)
    n_declared = 0
    for line_no, line in enumerate(lines):
        hits = list(PERF_NUM_RE.finditer(line))
        if not hits:
            continue
        if line_no in exempt:
            if METRIC_WORDS_RE.search(norm_key(line)):
                gate.fail(
                    f"docs/MVP_REPORT.md line {line_no + 1} is declared non-metric "
                    f"({exempt[line_no]!r}) but states a number in metric "
                    f"vocabulary: {line.strip()!r}. The declaration is for config "
                    f"values, counts and dates; a number the MVP claim rests on "
                    f"has to be checkable against the transcripts.")
            else:
                n_declared += len(hits)
            continue
        spans = accounted.get(line_no, ())
        for m in hits:
            if any(a <= m.start() and m.end() <= b for a, b in spans):
                continue
            gate.fail(
                f"docs/MVP_REPORT.md line {line_no + 1} states {m.group(0)!r}, a "
                f"performance number this gate cannot check: it is not a metric "
                f"cell matched against the committed transcripts, and the line is "
                f"not declared non-metric. Every percentage or ratio in the report "
                f"must be one or the other -- put it in a table whose column or row "
                f"is one of {list(METRIC_KEYS)}, or mark the line "
                f"`<!-- non-metric: <why> -->`. Line: {line.strip()!r}")

    # --- and the base-vs-tuned claim must actually be printed ------------
    for role in ("base", "tuned"):
        for metric in ("spec_adherence", "robustness"):
            if (role, metric) not in attempted:
                gate.fail(
                    f"docs/MVP_REPORT.md never prints a checkable {role} {metric} "
                    f"value in a markdown table. The MVP claim IS base-vs-tuned on "
                    f"{metric}, and a number that cannot be matched to the {role} "
                    f"transcripts is not evidence")
    if checked:
        gate.note(f"{len(checked)} reported metric value(s) match the numbers "
                  f"re-derived from the transcripts, to the precision they are "
                  f"printed at; every other percentage or ratio in the report is "
                  f"accounted for")
    if n_declared:
        gate.note(f"{n_declared} number(s) explicitly declared non-metric: "
                  f"{sorted({w for w in exempt.values() if w})}")


# ------------------------------------------------------------------ package
def load_models_json(gate, root):
    path = root / "configs" / "models.json"
    if not gate.require(path.exists(), "configs/models.json missing"):
        return None
    try:
        cfg = read_json(path)
    except GateError as exc:
        gate.fail(str(exc))
        return None
    if not isinstance(cfg, dict):
        gate.fail("configs/models.json is not a JSON object")
        return None
    tuned = cfg.get("tuned")
    if not isinstance(tuned, str) or not TUNED_RE.match(tuned):
        gate.fail(f"configs/models.json['tuned'] == {tuned!r}; expected "
                  f"'<owner>/<name>@<40-hex-sha>' -- the spec's Verification "
                  f"Requirements need the public HF repo id AND the exact commit")
        return None
    base = cfg.get("base")
    if not isinstance(base, str) or not base.startswith("hf:"):
        gate.fail(f"configs/models.json['base'] == {base!r}; expected "
                  f"'hf:<repo>@<revision>'")
        return None
    return cfg


def discover_cells(gate, run_dir, wanted):
    """{model_spec: (results_path, results_dict)} for the pinned models.

    Cells are matched on the ``model`` field INSIDE each results JSON, never on
    the filename: the name is cosmetic, the header is the claim. Anything in
    runs/mvp that is neither pinned model is a failure -- an MVP package with a
    third, unexplained cell is not the comparison it says it is.
    """
    found, extra = {}, []
    for rp in sorted(run_dir.glob("*.results.json")):
        try:
            res = read_json(rp)
        except GateError as exc:
            gate.fail(str(exc))
            continue
        if not isinstance(res, dict):
            gate.fail(f"{rp.name}: results is not a JSON object")
            continue
        spec = res.get("model")
        if not isinstance(spec, str) or not spec.strip():
            gate.fail(f"{rp.name}: names no model")
            continue
        if spec not in wanted:
            extra.append(f"{rp.name} (model {spec!r})")
            continue
        if spec in found:
            gate.fail(f"{rp.name}: a second results JSON for {spec!r} (first was "
                      f"{found[spec][0].name}) -- which one the report summarizes "
                      f"is ambiguous")
            continue
        found[spec] = (rp, res)
    if extra:
        gate.fail(f"{run_dir} holds results that are neither the pinned tuned nor "
                  f"the pinned base model: {extra}")
    return found


def resolve_eval_set(gate, root, cells):
    """The one eval set both models were scored on, as a path under `root`."""
    declared = {}
    for spec, (rp, res) in cells.items():
        raw = res.get("eval_set")
        if not isinstance(raw, str) or not raw.strip():
            gate.fail(f"{rp.name}: names no 'eval_set' -- without it there is no "
                      f"proof the two models sat the same exam, and the rows "
                      f"cannot be joined to their scenarios")
            return None
        declared[spec] = raw.strip()
    if len(set(declared.values())) > 1:
        gate.fail(f"tuned and base were scored on DIFFERENT eval sets "
                  f"({sorted(set(declared.values()))}) -- the base-vs-tuned delta "
                  f"is then a comparison of two different exams")
        return None
    rel = next(iter(declared.values()))
    path = (root / rel).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        gate.fail(f"eval_set {rel!r} resolves outside {root} ({path}) -- it is not "
                  f"a committed artifact")
        return None
    if not path.exists():
        gate.fail(f"eval_set {rel!r} does not exist at {path}")
        return None
    return path


def check_headers(gate, cells, roles):
    """The metadata both cells must agree on, plus the rubric this repo actually
    ships. The rubric hash is RE-DERIVED from judge.py rather than compared to a
    literal pinned in this file: a constant here would have to be updated by hand
    and could be updated to whatever made the gate pass."""
    live_rubric = judge_mod.rubric_hash()
    strategies = {}
    for role, spec in roles.items():
        rp, res = cells[spec]
        strategy = res.get("strategy")
        if strategy != REQUIRED_STRATEGY:
            gate.fail(f"{rp.name}: strategy {strategy!r} != {REQUIRED_STRATEGY!r} "
                      f"-- the MVP comparison must run the deployment (zero-shot) "
                      f"condition the model was trained for")
        strategies[role] = strategy
        if res.get("judge_rubric_hash") != live_rubric:
            gate.fail(f"{rp.name}: judge_rubric_hash "
                      f"{res.get('judge_rubric_hash')!r} != this repo's "
                      f"judge.rubric_hash() {live_rubric!r} -- the numbers were "
                      f"scored under a rubric that is no longer the shipped one")
        if not res.get("judge_model"):
            gate.fail(f"{rp.name}: no judge_model recorded")

    tuned_res = cells[roles["tuned"]][1]
    base_res = cells[roles["base"]][1]
    if tuned_res.get("judge_model") != base_res.get("judge_model"):
        gate.fail(f"tuned judge {tuned_res.get('judge_model')!r} != base judge "
                  f"{base_res.get('judge_model')!r} -- one rubric means one judge")
    if tuned_res.get("strategy") != base_res.get("strategy"):
        gate.fail("tuned and base were run under different strategies")
    return live_rubric


def score_role(gate, role, spec, cells, eval_index, instances):
    """gate_check.score_cell over one committed cell. Returns its stats or None."""
    rp, res = cells[spec]
    tp = rp.with_name(rp.name.replace(".results.json", ".transcripts.jsonl"))
    if not tp.exists():
        gate.fail(f"{tp.name}: missing -- the spec requires the full per-example "
                  f"judge transcript JSONL for the {role} model, not just the "
                  f"aggregate table")
        return None
    errors = []
    try:
        stats = score_cell(rp, tp, spec, res.get("strategy"), eval_index,
                           instances, errors)
    except GateError as exc:
        gate.fail(f"{role}: {exc}")
        return None
    for err in errors:
        gate.fail(f"{role}: {err}")
    if stats is None:
        if not errors:
            gate.fail(f"{role}: {tp.name} could not be re-derived")
        return None

    # ROBUSTNESS IS THE PIVOT'S HEADLINE METRIC, and it is defined on the
    # ADVERSARIAL subset. gate_check returns robustness=None when that subset is
    # empty, the report then prints "n/a", and the n/a branch used to accept it:
    # a package that never measured robustness passed as an MVP package. An eval
    # set with no adversarial pressure is not a harder exam the model happened to
    # ace, it is an exam that was never set.
    if not stats.get("n_adversarial"):
        gate.fail(f"{role} ({spec}): the eval set has NO adversarial scenarios, so "
                  f"robustness -- the pivot's headline metric, defined on the "
                  f"adversarial subset -- was never measured. An MVP package must "
                  f"score both models on a non-empty adversarial subset; a package "
                  f"that reports robustness as 'n/a' has measured nothing.")
        return None
    if stats.get("robustness") is None:
        gate.fail(f"{role} ({spec}): {stats['n_adversarial']} adversarial "
                  f"scenario(s) in the eval set but no robustness value could be "
                  f"re-derived -- every adversarial row was quarantined, so the "
                  f"headline metric rests on nothing.")
        return None

    quarantined = len(eval_index) - stats["n"]
    share = quarantined / max(len(eval_index), 1)
    gate.note(f"{role} ({spec}): {stats['n']} of {len(eval_index)} eval-set rows "
              f"scored, {quarantined} quarantined ({share:.1%}, cap "
              f"{MAX_QUARANTINE_SHARE:.0%}); every checker verdict re-run from "
              f"(constraint, output); spec_adherence "
              f"{stats['spec_adherence']:.4f}, robustness {stats['robustness']!r}")
    return stats


# ------------------------------------------------------------------- report
def check_report(gate, root, cfg, base_spec, run_dir, expected, rubric_hash,
                 eval_rel):
    report_path = root / "docs" / "MVP_REPORT.md"
    if not gate.require(report_path.exists(), "docs/MVP_REPORT.md missing"):
        return
    text = report_path.read_text()
    if not text.strip():
        gate.fail("docs/MVP_REPORT.md is empty")
        return

    tuned = cfg["tuned"]
    repo_id, _, model_sha = tuned.partition("@")

    # --- pinned versions: public repo id + exact commit ------------------
    if tuned not in text:
        gate.fail(f"docs/MVP_REPORT.md does not contain the pinned tuned model "
                  f"{tuned!r} from configs/models.json['tuned'] -- the spec's "
                  f"Verification Requirements need the exact HF repo id + commit "
                  f"hash")
    else:
        gate.note(f"tuned checkpoint pinned in the report and matching "
                  f"models.json: {tuned}")
    for other in SHA40_RE.findall(text):
        if other != model_sha and f"{repo_id}@{other}" in text:
            gate.fail(f"docs/MVP_REPORT.md pins {repo_id}@{other} but "
                      f"configs/models.json['tuned'] says {tuned}")

    base_repo = base_spec.split(":", 1)[1]
    if base_repo not in text and base_repo.split("@", 1)[0] not in text:
        gate.fail(f"docs/MVP_REPORT.md never mentions the base model "
                  f"{base_repo!r} -- a base-vs-tuned report must name both")

    # --- the eval-code commit hash --------------------------------------
    eval_shas = set()
    for ln in text.splitlines():
        m = SHA40_RE.search(ln)
        if m and "eval" in ln.lower() and m.group(0) != model_sha:
            eval_shas.add(m.group(0))
    if not eval_shas:
        gate.fail("docs/MVP_REPORT.md has no eval-code commit hash: the spec "
                  "requires the exact eval-code commit alongside the model "
                  "commit. Write it on a line that says 'eval', e.g. "
                  "`| eval-code commit | <40-hex-sha> |`")
    else:
        gate.note(f"eval-code commit recorded: {sorted(eval_shas)[0]}")

    if rubric_hash not in text:
        gate.fail(f"docs/MVP_REPORT.md does not state the judge rubric hash "
                  f"{rubric_hash!r} the numbers were scored under")
    if eval_rel not in text:
        gate.fail(f"docs/MVP_REPORT.md does not name the eval set {eval_rel!r} "
                  f"both models were scored on")

    # --- both tables -----------------------------------------------------
    n_tables = sum(1 for ln in text.splitlines() if TABLE_SEP_RE.match(ln.strip()))
    if n_tables < 2:
        gate.fail(f"docs/MVP_REPORT.md contains {n_tables} markdown table(s); the "
                  f"MVP report requires both the base-vs-tuned results table and "
                  f"the delta")
    normalized = text.lower().replace("-", "_").replace(" ", "_")
    for metric in ("spec_adherence", "robustness"):
        if metric not in normalized:
            gate.fail(f"docs/MVP_REPORT.md never reports {metric}")

    # --- and the numbers in those tables are the re-derived ones ---------
    check_report_metrics(gate, text, expected, tuned, base_repo)

    # --- one-command repro line ------------------------------------------
    report_cmds = logical_commands(text)
    repro = [c for c in report_cmds
             if "eval.py" in c and "--model" in c and "--eval-set" in c]
    pinned_repro = [c for c in repro if tuned in c]
    if not repro:
        gate.fail("docs/MVP_REPORT.md has no one-command repro line "
                  "(`eval.py --model ... --eval-set ...`)")
    elif not pinned_repro:
        gate.fail(f"the repro line in docs/MVP_REPORT.md does not pin "
                  f"{tuned!r}: {repro[0]!r}")
    else:
        readme_path = root / "README.md"
        if not readme_path.exists():
            gate.fail("README.md missing -- the repro line must be mirrored there")
        else:
            readme_cmds = set(logical_commands(readme_path.read_text()))
            for cmd in pinned_repro:
                if cmd not in readme_cmds:
                    gate.fail(f"the repro line in docs/MVP_REPORT.md is not "
                              f"byte-consistent with README.md.\n"
                              f"      report: {cmd}\n"
                              f"      README has: "
                              f"{[c for c in readme_cmds if 'eval.py' in c] or 'no eval.py command'}")
                else:
                    gate.note(f"repro line mirrored byte-for-byte in README.md: {cmd}")

    # --- raw transcripts, pointed at ------------------------------------
    transcripts = sorted(run_dir.glob("*.transcripts.jsonl"))
    if not transcripts:
        gate.fail(f"no transcripts JSONL in {run_dir}")
    for tp in transcripts:
        rel = tp.relative_to(root).as_posix()
        if rel not in text:
            gate.fail(f"docs/MVP_REPORT.md does not reference the transcripts "
                      f"file {rel} -- the spec requires the raw judge transcripts "
                      f"to be pointed at, not just produced")

    # --- every path it cites exists --------------------------------------
    for rel in referenced_paths(text):
        if not (root / rel).exists():
            gate.fail(f"docs/MVP_REPORT.md references {rel} which does not exist "
                      f"on disk")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Re-check the committed MVP submission package (P6)")
    ap.add_argument("--root", default=str(REPO_ROOT),
                    help="repo root holding runs/mvp, docs/, configs/, README.md")
    ap.add_argument("--run-dir", default=None, help="override runs/mvp")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    run_dir = Path(args.run_dir).resolve() if args.run_dir else root / "runs" / "mvp"
    gate = Gate(f"MVP submission check: {root}")
    print(f"checking {root} against judge rubric {judge_mod.rubric_hash()}")

    if not root.is_dir():
        gate.fail(f"{root} does not exist")
        return gate.finish()
    if not run_dir.is_dir() or not any(run_dir.iterdir()):
        gate.fail(f"{run_dir} is missing or empty -- no MVP run artifacts "
                  f"committed. The package is runs/mvp/ holding results + "
                  f"transcripts for BOTH the tuned and the base model.")
        return gate.finish()

    cfg = load_models_json(gate, root)
    if cfg is None:
        return gate.finish()
    tuned_spec = f"hf:{cfg['tuned']}"
    base_spec = cfg["base"]
    if tuned_spec == base_spec:
        gate.fail("configs/models.json pins the same repo@commit for 'tuned' and "
                  "'base' -- there is no comparison to make")
        return gate.finish()
    roles = {"tuned": tuned_spec, "base": base_spec}

    cells = discover_cells(gate, run_dir, set(roles.values()))
    for role, spec in roles.items():
        if spec not in cells:
            gate.fail(f"no {role} results JSON naming {spec!r} in {run_dir} -- the "
                      f"MVP package must hold BOTH models, scored on the same eval "
                      f"set by the same judge")
    if any(spec not in cells for spec in roles.values()):
        return gate.finish()

    rubric_hash = check_headers(gate, cells, roles)

    eval_path = resolve_eval_set(gate, root, cells)
    if eval_path is None:
        return gate.finish()
    eval_rel = eval_path.relative_to(root).as_posix()
    instances_path = root / DEFAULT_INSTANCES_PATH
    try:
        eval_index = load_eval_set(eval_path)
        instances = load_constraint_instances(instances_path)
        check_constraints_resolve(eval_index, instances, eval_path, instances_path)
    except GateError as exc:
        gate.fail(str(exc))
        return gate.finish()
    gate.note(f"eval set {eval_rel} sha256:{sha256_of(eval_path)[:16]}… "
              f"({len(eval_index)} scenarios, >= {MIN_SCENARIOS_PER_CELL} required "
              f"per cell), constraints resolved against {DEFAULT_INSTANCES_PATH}")

    stats = {role: score_role(gate, role, spec, cells, eval_index, instances)
             for role, spec in roles.items()}
    if any(v is None for v in stats.values()):
        return gate.finish()

    expected = {
        "tuned": {k: stats["tuned"].get(k) for k in METRIC_KEYS},
        "base": {k: stats["base"].get(k) for k in METRIC_KEYS},
    }
    expected["delta"] = {
        k: (None if expected["tuned"][k] is None or expected["base"][k] is None
            else expected["tuned"][k] - expected["base"][k])
        for k in METRIC_KEYS
    }
    gate.note(f"base-vs-tuned on {len(eval_index)} shared scenarios, "
              f"spec_adherence {expected['base']['spec_adherence']:.4f} -> "
              f"{expected['tuned']['spec_adherence']:.4f}")

    check_report(gate, root, cfg, base_spec, run_dir, expected, rubric_hash,
                 eval_rel)
    return gate.finish()


if __name__ == "__main__":
    sys.exit(main())
