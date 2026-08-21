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

THE REPORT'S NUMBERS ARE GENERATED, NOT PARSED
-----------------------------------------------
An earlier version only grepped docs/MVP_REPORT.md for the metric NAMES, so the
headline base-vs-tuned figures could be arbitrarily inflated while every gate
stayed green. The version after that validated numbers found under RECOGNISED
headings -- and an adversarial verifier walked a prose claim, two renamed tables
and an extra column straight past it, because a table matching no recognised
heading was never examined at all. The version after THAT inverted the rule --
every percentage, ratio or printed fraction had to be a matched table cell or a
declared non-metric line -- and an audit ran 18 adversarial report bodies through
it with the honest table left intact. ALL 18 PASSED: the "percentage" it
recognised was `%`, `pp` and nothing else (not `percent`, not `pct`, not `％`,
not `&#37;`, not `**99.4**%`, not a claim split across a newline); an
unorientable table's bare numbers were shapeless to it; a body row with no
leading pipe made the whole table parse as empty; and the 15-word vocabulary
policing the escape hatch could be stepped around by rewording the claim.

Every one of those lost the same way: it tried to PARSE a claim back out of prose
the author is free to rewrite. So the report does not state the numbers any more.
``render_metric_block`` GENERATES them -- from the stats
``gate_check.score_cell`` re-derives from the transcripts, never from a results
JSON's stored ``metrics`` field -- between two literal markers, and this gate
requires that block to appear in the report VERBATIM, byte for byte, printing a
unified diff when it does not. ``check_mvp.py --emit-metrics`` prints the block
for the author to paste; every other line of the report stays hand-written.

That inverts the remaining work. The block is the only accounted region and it is
contiguous, so what is left is ONE sweep of everything outside it: NFKC-normalize
(fullwidth ％ and fullwidth digits die here), blank every character whose Unicode
CATEGORY renders as nothing (Cf/Cc/Cn/Mn/… -- by category, because the enumeration
this replaced missed U+2066 and an audit put one between '99' and '%'), decode HTML
entities and blank again (`&#8288;` decodes to a word joiner), strip tags, JOIN the
surviving lines (a claim split across a newline is still one claim), mask the spans
that are never rates (paths, shas, ISO dates, code-span identifiers, any token with
a letter touching a digit -- but never the rate-shaped number glued to one), and
refuse what is left if it is rate-shaped:

  A  a number carrying a rate unit, in either order -- %, ‰, pct, per cent,
     percent, percentage point(s), pp, p.p., bps, basis point(s), x/× multiplier
  C  a printed fraction (146/147, 146⁄147)
  O  an of-form ratio (146 of 147, 146 of the 147, 99 times out of 100, 6 in 1000)
  B  a bare fractional number in prose (.994, 0,994, 99.4)
  D  a bare number alone in a table cell, unless the row is key -> ONE value with
     an identifier-shaped key (`| max_seq_length | 2048 |`)
  E  metric vocabulary on the same line as any number that survived masking

A and C are NEVER declarable: '%' and a printed '/' are rate notation and nothing
else, so there is no honest reason for one to sit outside the block. O, B, D and E
may sit on a line the report declares with a stated reason
(`<!-- non-metric: <why> -->` or its honest synonym
`<!-- not-a-gate-metric: <why> -->`), and only if that line speaks no metric
vocabulary. Dates, counts (n=147, 881 rows), version numbers, model sizes, commit
hashes and hyperparameter tables stay writable anywhere in the report, undeclared.

A DECLARATION HAS TO BE VISIBLE
-------------------------------
The hatch used to be an HTML comment and nothing else, which renders as NOTHING --
so it cost an author zero: an audit wrapped an entire fabricated base-vs-tuned
table (`| letter ban held | 0.6120 | 0.9940 |`) in
`<!-- non-metric-block: bookkeeping counts -->` and the gate called the report
clean, while a grader read an unqualified results table. Policing the hatch by
VOCABULARY is what failed the design before this one, and rewording is free.

So a declaration that actually suppresses a number must also say so where the
reader SEES it: somewhere in the declared region (or the few lines above it, where
a table caption goes) the rendered text has to say these numbers are not measured
results. That is a boundary, not a proof -- it cannot stop an author writing a
false number under a visible disclaimer. What it stops is a number appearing in
the rendered report as if it were this package's measurement while the only thing
marking it otherwise is invisible.

WHAT THIS RULE STILL CANNOT DO -- documented, not hidden
--------------------------------------------------------
  1. A claim with no number in it ("the tuned model is dramatically better") is
     outside the rule and always will be.
  2. A percentage may NEVER sit outside the block, declared or not. That is a
     deliberate trade: an audit showed the off-vocabulary claim "the tuned model
     obeys the forbidden letter in 99.4% of outputs <!-- non-metric: illustrative
     -->" is indistinguishable, to any lexical rule, from the honest corpus fact
     "character constraints are 59.8% of train". Only one of the two can be
     writable, and letting the percentage through is the error that ships a false
     headline. The cost is real and is paid by honest content: corpus composition
     has to be written as a decimal (0.598) on a declared, disclaimed line.
  3. `| tuned | 146 |` -- a bare integer, no denominator, no unit, a one-word
     label -- passes undeclared, because it is the same shape as `| seed | 42 |`.
     Adding the denominator that would make it a rate (`146 of 147`) is tier O.
  4. A model pin with a one-decimal version written WITHOUT backticks
     (openai:gpt-5.6-sol in prose) reads as the bare fractional number 5.6 and is
     refused. Pre-existing, and the repo's convention (`code spans for pins`)
     already avoids it; the alternative -- masking any letter+digit token whole --
     is exactly the bug that let `0.994-v1` erase a real rate.
  5. This gate refuses rate SHAPES. It does not know English, and it never
     compares a number in the prose against the re-derived truth, because parsing
     a claim back out of prose the author is free to rewrite is the failure mode
     this whole design replaced.

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
import difflib
import re
import sys
import unicodedata
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
    QUALITY_SCORE_MIN,
    GateError,
    check_constraints_resolve,
    load_constraint_instances,
    load_eval_set,
    read_json,
    read_jsonl,
    rederive_checker_results,
    score_cell,
    sha256_of,
)

REQUIRED_STRATEGY = "zero_shot"

#: A public HF checkpoint pinned to a commit: "<owner>/<name>@<40-hex>".
TUNED_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*@[0-9a-f]{40}$")
SHA40_RE = re.compile(r"\b[0-9a-f]{40}\b")

#: The metrics gate_check re-derives from the rows, and therefore the only
#: numbers docs/MVP_REPORT.md prints.
METRIC_KEYS = ("spec_adherence", "robustness", "fail_rate")


#: The literal fence around the GENERATED block. Everything between these two
#: markers is written by render_metric_block and checked byte for byte;
#: everything outside them is swept for rate-shaped numbers.
BEGIN_METRICS = "<!-- BEGIN GENERATED METRICS -->"
END_METRICS = "<!-- END GENERATED METRICS -->"
#: rows of the generated table, in printed order: the three rates gate_check
#: re-derives, then the three counts that say what they were derived FROM.
BLOCK_RATES = METRIC_KEYS
BLOCK_COUNTS = ("n_scored", "n_adversarial", "n_quarantined")

#: How a report DECLARES that a number is not one this gate re-derived.
#: Line-scoped and block-scoped; both must state a reason.
#:
#: `non-metric` and `not-a-gate-metric` are the SAME declaration under two
#: names. The second exists because the first is a lie for one honest case: the
#: G1 ablation fail rates in runs/g1 ARE measurements -- they are the prior-run
#: evidence that authorised this fine-tune -- they are simply not numbers THIS
#: gate re-derived from THIS package's transcripts. A report should not have to
#: misdescribe its own strongest evidence to print it.
_DECL = r"(?:non-metric|not-a-gate-metric)"
NON_METRIC_LINE_RE = re.compile(rf"<!--\s*{_DECL}\s*:\s*([^>]*?)\s*-->", re.I)
NON_METRIC_OPEN_RE = re.compile(rf"<!--\s*{_DECL}-block\s*:\s*([^>]*?)\s*-->",
                                re.I)
NON_METRIC_CLOSE_RE = re.compile(rf"<!--\s*/\s*{_DECL}-block\s*-->", re.I)

#: WHY A DECLARATION MUST ALSO BE VISIBLE.
#: An HTML comment renders as nothing. So the previous hatch cost an author
#: nothing at all: an audit wrapped an entire fabricated base-vs-tuned results
#: table (`| letter ban held | 0.6120 | 0.9940 |`) in
#: `<!-- non-metric-block: bookkeeping counts -->` and the gate reported the
#: report clean, while a grader reading the rendered markdown saw an unqualified
#: results table. Policing the hatch with a word list is what failed before (the
#: docstring records it), and rewording is free.
#:
#: So the hatch is no longer policed lexically at all. It is policed by
#: VISIBILITY: a declaration that actually exempts a number must also say so in
#: text the reader SEES. The gate still cannot tell a true sentence from a false
#: one -- but it can guarantee that a number a grader could mistake for this
#: package's result is either generated by this gate from the transcripts, or
#: carries a disclaimer in the rendered report next to it.
DISCLAIMER_RE = re.compile(
    r"\bnot\s+(?:\w+[\s'’-]+){0,3}"
    r"(?:measurement|measured|result|metric|number|figure)s?\b", re.I)
#: how far above a declaration the visible caption may sit -- a caption above a
#: table is where a person writes one, and markdown renders it right there.
DISCLAIMER_LOOKBACK = 4

# ---------------------------------------------------------------- rate shapes
#: Units that turn a number into a RATE. Spelled out because the version this
#: replaced knew only '%' and 'pp', and "12 percent to 99 percent", "99.4 pct",
#: "87.4 p.p.", "8740 bps" and "8.3x" all walked past it.
PCT_UNIT = (r"%|‰|٪|pct\b|per[ _-]?cents?\b|percents?\b|"
            r"percentage[ _-]?points?\b|pp\b|p\.p\.|bps\b|basis[ _-]?points?\b|"
            r"[x×](?![\w])")
#: markdown emphasis and spacing a number can hide behind: **99.4**%, 99.4`%`
_SEP = r"[\s\*_`~]*"
#: everything a number can be GLUED to its unit with. Wider than _SEP because
#: tier A is matched against the raw line: an audit put U+2066 (an invisible
#: bidi isolate, not in the hand-written blanking list this file used to carry)
#: between the digits and the '%' and walked '99%' straight past the gate, and
#: '99-v1%' does the same with a visible tag. The invisible half is fixed at the
#: source -- normalize_report_text now blanks by Unicode CATEGORY -- and the
#: visible half is fixed here: a hyphen-tag, a slash or a dash between a number
#: and a percent sign does not stop a reader seeing a percentage.
_GLUE = r"[\s\*_`~/–—-]*(?:[-–—/][A-Za-z][\w.]*)?[\s\*_`~/–—-]*"
_NUM = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:[.,]\d+)?"
#: '1,000' / '12,345' -- a thousands-separated INTEGER, not the European decimal
#: '0,994'. Tier B reads ',' as a decimal comma on purpose; this is the one
#: shape where that reading is wrong, and 881,000 rows is not a rate.
THOUSANDS_RE = re.compile(r"^[-+]?[1-9]\d{0,2}(?:,\d{3})+$")

#: the same claim with the number spelled out ("ninety-nine percent"). It only
#: fires next to an explicit percent word, so it cannot swallow ordinary prose.
_WORD_NUM = (r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
             r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
             r"nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
             r"hundred|half)")

TIER_A = re.compile(rf"(?<![\w.])({_NUM}){_GLUE}(?:{PCT_UNIT})", re.I)
TIER_A2 = re.compile(rf"(?:{PCT_UNIT}){_SEP}({_NUM})(?![\w.])", re.I)
TIER_A3 = re.compile(
    rf"\b(?:{_WORD_NUM}[\s-]+)*{_WORD_NUM}{_SEP}"
    rf"(?:percents?\b|per[ -]?cents?\b|pct\b|percentage[ -]?points?\b)", re.I)
TIER_C = re.compile(r"(?<![\w./])\d+\s*[/⁄∕]\s*\d+(?!\.?\d)(?!\w)")

#: THE OF-FORM RATIO, and why it is not a two-integer regex any more.
#: It used to be r"\b\d+\s*(?:of|out\s+of)\s*\d+\b" -- the two integers had to be
#: ADJACENT. An audit added one article and walked four fabricated claims past
#: the gate in prose and in a table: '146 of the 147', '47 out of the 48',
#: '99 times out of 100', '6 in 1000'. A bare integer is fatal nowhere else
#: outside a table cell, so nothing else caught them.
#:
#: So the filler between the two numbers is allowed (bounded, and it may hold no
#: digits of its own), and the connector set covers the ratio idioms English
#: actually uses. Unlike tiers A and C this one CAN be declared away: an of-form
#: is ordinary English whose shape collides with honest sentences ('3 epochs in
#: 500 steps'), where a '%' or a printed '146/147' is rate notation and nothing
#: else.
#: 'of' / 'out of' take filler on both sides ('99 TIMES out of THE 100'); 'in'
#: and 'per' take only 'every', because they are also ordinary prepositions and
#: allowing filler made "3 epochs in 500 steps" a false positive on a line
#: carrying no claim at all. '6 in 1000' and '6 in every 1000' are the ratio
#: idiom; '<count> <plural noun> in <count> <plural noun>' is not.
_OF_FILL = r"(?:[\s\*_`~]+(?:[A-Za-z][\w'’-]*)){0,3}[\s\*_`~]*"
TIER_C_OF = re.compile(
    rf"(?<![\w.])(\d[\d,]*)"
    rf"(?:{_OF_FILL}\b(?:out\s+of|of)\b{_OF_FILL}"
    rf"|[\s\*_`~]+(?:in|per)(?:\s+every)?[\s\*_`~]+)"
    rf"(\d[\d,]*)(?![\w.])", re.I)

TIER_B = re.compile(r"(?<![\w.,])[-+]?\d*[.,]\d+(?![\w])")
#: a markdown table row, split into cells. Tier D reads the CELLS rather than
#: matching a pipe-number-pipe span: see table_cell_hits for why the row's shape
#: is what decides whether a bare integer in it is a claim.
TIER_D_CELL = re.compile(rf"^{_SEP}({_NUM}){_SEP}$")
ANY_NUM = re.compile(r"[-+]?\d+(?:[.,]\d+)*")

#: Vocabulary that makes a line a claim ABOUT model performance. A line carrying
#: both this and a number may not be declared non-metric, and outside the block a
#: line carrying both is refused whatever shape the number takes -- "adherence
#: rises to 99 points" is a claim even though "99 points" is not a rate.
METRIC_WORDS = re.compile(
    r"adherence|robust|fail[_ ]?rate|pass[_ ]?rate|accurac|precision|recall|"
    r"\bf1\b|violation|complian|spec[_ ]?pass|\bwins?\b|\bscor\w*|\blift\w*|"
    r"improv|outperform|\bbeat\w*|\bbetter\b|\bworse\b", re.I)

KNOWN_DIRS = {"runs", "data", "docs", "configs", "scripts", "training",
              "datagen", "tests", "constraints", "scenarios"}
PATHY_EXTS = (".jsonl", ".json", ".md", ".py", ".txt", ".yaml", ".yml", ".csv")
STRIP_CHARS = "`()[]{}<>,;:.!\"'*|"
# a path-like run: one or more "<segment>/" then a final segment. Anchored on the
# characters legal in this repo's paths, so it lifts "runs/mvp/x.jsonl" out of a
# markdown link or a table cell without dragging punctuation along.
PATH_RE = re.compile(r"(?:[A-Za-z0-9_.@-]+/)+[A-Za-z0-9_.@*?-]+")

#: spans that are never rates, blanked before the bare-number tiers run
SHA_RE = re.compile(r"\b[0-9a-f]{7,64}\b")
ISO_RE = re.compile(r"\d{4}-\d{2}-\d{2}"
                    r"(?:[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?"
                    r"|\d{4}/\d{2}/\d{2}")
#: a token with a letter TOUCHING a digit: Qwen3-1.7B, v1, r7-hifreq, k=4
ADJ_RE = re.compile(r"[\w.@+-]*(?:[A-Za-z]\d|\d[A-Za-z])[\w.@+-]*")
#: a dotted version with three or more parts (transformers 4.57.2, torch 2.12.1).
#: Two parts stays visible on purpose: '2.12' and a rate are the same shape, and
#: guessing wrong in that direction is how a rate gets waved through.
VERSION_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+){2,}(?![\w])")   # trl 0.26.0.
#: a number wearing a unit that is never a rate -- a size or a duration. Tier A
#: is matched BEFORE any masking, so this can never hide a percentage.
NONRATE_UNIT_RE = re.compile(
    r"(?<![\w.])[-+]?\d+(?:[.,]\d+)?\s*"
    r"(?:[KMGT]i?B|kB|bytes?|GPU-?h(?:ours?)?|hours?|hrs?|minutes?|mins?|"
    r"seconds?|secs?|ms)\b", re.I)
CODE_SPAN_RE = re.compile(r"`([^`]+)`")
#: a number printed to enough precision to BE a rate -- what stops a code span
#: like `spec_adherence-0.994` from being masked as an identifier
RATE_PRECISION_RE = re.compile(r"(?<!\w)[-+]?[01]?[.,]\d{2,}")


class Gate:
    def __init__(self, title, stream=None):
        self.title = title
        self.failures = []
        self.notes = []
        # --emit-metrics writes the pasteable block to stdout and nothing else,
        # so under that flag the running commentary goes to stderr instead.
        self.stream = stream if stream is not None else sys.stdout

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
            print(f"  ok   {n}", file=self.stream)
        if self.failures:
            print(f"\nFAIL {self.title} — {len(self.failures)} problem(s):",
                  file=self.stream)
            for f in self.failures:
                print(f"  - {f}", file=self.stream)
            return 1
        print(f"\nPASS {self.title}", file=self.stream)
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


# ------------------------------------------------- the generated metric block
def _rate(value):
    return "n/a" if value is None else f"{value:.4f}"


def _rate_delta(tuned, base):
    return "n/a" if tuned is None or base is None else f"{tuned - base:+.4f}"


def derive_breakdown(stats, eval_index, instances):
    """Per-split and per-constraint-type counts, RE-DERIVED from the transcripts.

    These are the numbers the analysis actually turns on -- a headline that moved
    two points is not a finding, and "which constraint types moved, and were they
    the ones training saw" is. Hand-typing them into the report would leave the
    report's most load-bearing table as the one thing nothing checks, which is
    the hole the generated block exists to close for the headline.

    So they are generated on the same terms: the checkers are re-run here from
    (constraint, output) via gate_check.rederive_checker_results, spec_pass is
    recomputed from THOSE verdicts plus the row's quality score, and quarantined
    rows are excluded exactly as score_cell excludes them (a row carrying a
    judge_error or generator_error has no verdict to count).
    """
    # read_jsonl yields (line_number, object) pairs; only the object matters here.
    rows = [obj for _lineno, obj in read_jsonl(Path(stats["transcripts_jsonl_path"]))]
    splits, types, quality_ok = {}, {}, [0]
    for row in rows:
        scen = eval_index.get(row.get("scenario_id"))
        if scen is None:
            continue
        if row.get("judge_error") is not None or row.get("generator_error") is not None:
            continue
        errors = []
        derived = rederive_checker_results(
            scen, instances, row.get("output"), "breakdown", errors)
        if derived is None or errors:
            continue
        checkers_pass = all(p for p, _ in derived.values())
        spec_pass = (checkers_pass
                     and (row.get("quality_score") or 0) >= QUALITY_SCORE_MIN)
        tally = splits.setdefault(scen["split"], [0, 0, 0])
        tally[0] += 1
        tally[1] += 1 if checkers_pass else 0
        tally[2] += 1 if spec_pass else 0
        if (row.get("quality_score") or 0) >= QUALITY_SCORE_MIN:
            quality_ok[0] += 1
        for cid, (passed, _evidence) in derived.items():
            inst = instances.get(cid)
            if inst is None:
                continue
            t = types.setdefault((inst["type"], inst.get("split", "")), [0, 0])
            t[0] += 1 if passed else 0
            t[1] += 1
    return {"splits": splits, "types": types,
            "n_quality_at_least_min": quality_ok[0],
            "n_fail_from_quality_only": stats.get("n_fail_from_quality_only", 0)}


def render_metric_block(base_stats, tuned_stats, *, n_eval, eval_rel,
                        eval_sha256, rubric_hash, judge_model,
                        base_breakdown=None, tuned_breakdown=None):
    """The report's numbers, GENERATED rather than parsed back out of prose.

    `base_stats` and `tuned_stats` are gate_check.score_cell's return values --
    the stats it RE-DERIVES by re-running constraints.checkers.check over every
    (constraint, output) pair in the transcripts. Nothing here reads a results
    JSON's stored ``metrics`` field: that field is a claim, and a claim is the
    thing this block exists to replace. (score_cell has already refused the cell
    if the stored aggregate disagrees with the rows, so a tampered ``metrics``
    never reaches this function -- but if the two ever parted company, the block
    would print the transcripts' answer, not the file's.)

    Determinism is the whole contract: fixed row order, fixed column order, rates
    at four decimals, deltas signed. The same artifacts must render the same
    bytes on any machine, which is why nothing absolute (a path under /Users, a
    timestamp) is allowed in here.
    """
    def counts(stats):
        return {"n_scored": stats["n"],
                "n_adversarial": stats["n_adversarial"],
                "n_quarantined": n_eval - stats["n"]}

    b_counts, t_counts = counts(base_stats), counts(tuned_stats)
    rows = [
        f"| {k} | {_rate(base_stats.get(k))} | {_rate(tuned_stats.get(k))} | "
        f"{_rate_delta(tuned_stats.get(k), base_stats.get(k))} |"
        for k in BLOCK_RATES
    ] + [
        f"| {k} | {b_counts[k]} | {t_counts[k]} | "
        f"{t_counts[k] - b_counts[k]:+d} |"
        for k in BLOCK_COUNTS
    ]
    return "\n".join([
        BEGIN_METRICS,
        # NB: no repo-relative path in here. Every path a report cites has to
        # exist on disk (check_report), and a block that named
        # scripts/check_mvp.py would force every package to ship this gate.
        "<!-- Generated by `check_mvp.py --emit-metrics`. Every value below is",
        "     RE-DERIVED from the committed transcripts by",
        "     gate_check.score_cell, not read from any results JSON's stored",
        "     `metrics` field. check_mvp.py regenerates this block and refuses",
        "     the report unless the two are byte-identical, so editing a digit",
        "     here fails the gate rather than changing the claim. -->",
        "",
        "| metric | base | tuned | delta |",
        "| --- | --- | --- | --- |",
        *rows,
        "",
        f"- eval set: `{eval_rel}` sha256 `{eval_sha256}`",
        f"- judge rubric hash: `{rubric_hash}`",
        f"- judge model: `{judge_model}`",
        f"- base results JSON: `{base_stats['results_json']}` "
        f"sha256 `{base_stats['results_json_sha256']}`",
        f"- tuned results JSON: `{tuned_stats['results_json']}` "
        f"sha256 `{tuned_stats['results_json_sha256']}`",
        *_breakdown_lines(base_breakdown, tuned_breakdown),
        END_METRICS,
    ])


def _breakdown_lines(base_breakdown, tuned_breakdown):
    """The per-split and per-type tables, in a fixed order so byte-equality is
    stable across machines. Empty when no breakdown was supplied, which keeps the
    block exactly as it was for callers that do not pass one."""
    if not base_breakdown or not tuned_breakdown:
        return []
    def _row(key):
        b, t = base_breakdown.get(key, 0), tuned_breakdown.get(key, 0)
        return f"| {key} | {b} | {t} | {t - b:+d} |"

    out = ["",
           "| judged quality | base | tuned | delta |",
           "| --- | --- | --- | --- |",
           _row("n_quality_at_least_min"),
           _row("n_fail_from_quality_only"),
           "",
           "| split | model | scored | all_checkers_passed | spec_pass |",
           "| --- | --- | --- | --- | --- |"]
    for split in sorted(set(base_breakdown["splits"]) | set(tuned_breakdown["splits"])):
        for role, bd in (("base", base_breakdown), ("tuned", tuned_breakdown)):
            scored, all_ok, sp = bd["splits"].get(split, [0, 0, 0])
            out.append(f"| {split} | {role} | {scored} | {all_ok} | {sp} |")
    out += ["",
            "| constraint type | type split | base passed | base of | "
            "tuned passed | tuned of |",
            "| --- | --- | --- | --- | --- | --- |"]
    for key in sorted(set(base_breakdown["types"]) | set(tuned_breakdown["types"])):
        ctype, csplit = key
        b_ok, b_n = base_breakdown["types"].get(key, [0, 0])
        t_ok, t_n = tuned_breakdown["types"].get(key, [0, 0])
        out.append(f"| {ctype} | {csplit} | {b_ok} | {b_n} | {t_ok} | {t_n} |")
    return out


# ------------------------------------------------------- the outside-the-block
# sweep. Every helper below is LENGTH-PRESERVING: masking blanks a span instead
# of deleting it, so an offset into a normalized line still points at the same
# character, and a hit found in the joined document can be traced back to the
# line it came from.
def _blank(m):
    return " " * len(m.group(0))


def _chr(m, base):
    """One numeric character reference, or the reference itself if it names no
    character. A report is untrusted input: `&#9999999;` must not crash the gate
    that is refusing it."""
    try:
        return chr(int(m.group(1), base))
    except ValueError:
        return m.group(0)


#: Unicode general categories that render as NOTHING. Blanked by CATEGORY, not
#: by code point: the hand-written list this replaced covered
#: [\u00ad\u180e\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff] and missed
#: U+2066..U+2069 (bidi isolates), U+061C, U+206A..U+206F, the U+E0020 tag block
#: and U+034F -- an audit put U+2066 between '99' and '%' and the gate saw no
#: percentage, while every reader saw '99%'. An enumeration of an open set is
#: not a rule; the CATEGORY is the rule, and it covers characters Unicode has
#: not assigned yet (Cn) too. Mn/Me (combining marks) go with them: NFKC has
#: already composed the accents that belong to real letters, so a mark left
#: standing between a digit and its unit is there to hide the unit.
INVISIBLE_CATS = frozenset({"Cc", "Cf", "Cs", "Co", "Cn", "Mn", "Me", "Zl", "Zp"})


def _blank_invisible(s):
    """Blank every character that renders as nothing, length-preserving."""
    if s.isascii() and s.isprintable():
        return s
    return "".join(" " if (ch not in "\t\n"
                           and unicodedata.category(ch) in INVISIBLE_CATS)
                   else ch for ch in s)


def normalize_report_text(s):
    """One report line, reduced to the characters a rate can hide behind.

    NFKC folds fullwidth '\uff05' to '%', fullwidth digits to ASCII and NBSP to a
    space; the entity decode catches `&#37;`/`&nbsp;`/`&#x25;`; stripping tags
    removes `<br>`, `<!-- comments -->` and any HTML an author wraps a digit in.

    Invisible characters are blanked rather than deleted so offsets survive, and
    they are blanked TWICE: once on the raw text and once after the entity
    decode, because `&#8288;` decodes to a word joiner and a single pass before
    the decode would have let the entity form of the same attack through.
    """
    s = _blank_invisible(unicodedata.normalize("NFKC", s))
    s = re.sub(r"&nbsp;?|&#160;|&#xa0;", " ", s, flags=re.I)
    s = re.sub(r"&percnt;", "%", s, flags=re.I)
    s = re.sub(r"&#(\d{1,7});", lambda m: _chr(m, 10), s)
    s = re.sub(r"&#x([0-9a-f]{1,6});", lambda m: _chr(m, 16), s, flags=re.I)
    # entities can decode to \uff05/\u066a, or to an invisible format character
    s = _blank_invisible(unicodedata.normalize("NFKC", s))
    return re.sub(r"<[^>]*>", _blank, s)


def mask_code_spans(s):
    """Blank `backticked identifiers` -- model pins, hashes, paths, filenames.

    A code span is only an identifier if it holds no rate-shaped number, so
    `spec_adherence-0.994` and `tuned/0.994` stay visible to the tiers below.
    """
    for m in list(CODE_SPAN_RE.finditer(s)):
        body = m.group(1)
        if (re.match(r"^[\w@][\w./@+:-]*$", body)
                and re.search(r"[A-Za-z]", body)
                and not RATE_PRECISION_RE.search(body)):
            s = s[:m.start()] + " " * len(m.group(0)) + s[m.end():]
    return s


def mask_paths(s):
    for m in list(PATH_RE.finditer(s)):
        tok = m.group(0)
        if tok.split("/", 1)[0] in KNOWN_DIRS or tok.endswith(PATHY_EXTS):
            s = s[:m.start()] + " " * len(tok) + s[m.end():]
    return s


def _blank_adj(m):
    """Blank a letter-touching-digit token -- but NEVER the rate-shaped number
    glued to it.

    ADJ_RE blanks the whole token on purpose: `Qwen3-1.7B`, `llama-3.1-8b` and
    `qwen-2.5-72b` are identifiers, and their '1.7' / '3.1' / '2.5' would
    otherwise read as bare fractional numbers. But blanking the whole token also
    erased a real rate an author appended a tag to: an audit wrote
    "spec_adherence improves from 0.612-v1 to 0.994-v1" and mask_structural
    returned nothing but spaces, so EVERY tier that reads the masked line -- B, D
    and E among them -- had no number to find, on a line naming the metric in
    metric vocabulary.

    A rate-shaped number (a decimal point and two or more digits after it, at the
    start of a token) is never part of an identifier, so it survives while the
    rest of the token is blanked. '1.7B' has one digit after the point and stays
    masked; 'x0.994' is preceded by a letter and stays masked.
    """
    tok = m.group(0)
    out = [" "] * len(tok)
    for r in RATE_PRECISION_RE.finditer(tok):
        out[r.start():r.end()] = tok[r.start():r.end()]
    return "".join(out)


def mask_structural(s):
    """Blank the spans that are never rates, so the bare-number tiers can be
    strict without taxing honest content: repo paths, commit hashes, ISO dates,
    code-span identifiers, and any token with a letter touching a digit
    (Qwen3-1.7B, r7-hifreq, transformers 4.57.2, n=147 stays visible)."""
    s = mask_code_spans(s)
    s = mask_paths(s)
    for rx in (SHA_RE, ISO_RE, VERSION_RE, NONRATE_UNIT_RE):
        s = rx.sub(_blank, s)
    return ADJ_RE.sub(_blank_adj, s)


def table_cell_hits(masked):
    """Bare numbers standing ALONE in a markdown table cell, as tier-D hits.

    A cell holding nothing but a number is the shape a grader reads as a result,
    and tier B only sees the FRACTIONAL ones -- so a table of bare integers
    (`| tuned | 146 | 47 |`) needs a rule of its own.

    But `| r | 16 |`, `| seed | 42 |`, `| max_seq_length | 2048 |`,
    `| num_train_epochs | 3 |` are the whole LoRA hyperparameter block from
    configs/train.json, and refusing them is a false positive on content carrying
    no performance claim of any kind -- the docstring's own promise is that
    integers (n=147, 881 rows, seed 1234) need nothing.

    The discriminator is the ROW's shape, not the number's, and it is two
    conditions rather than one, because relaxing it to "a row with one number" on
    its own would have let `| letter ban held | 146 |` through:

      * a config row is key -> ONE value. A comparison row carries two or more
        numbers, because a comparison is what it is.
      * a config key is an IDENTIFIER -- one token, no spaces (`max_seq_length`,
        `lora_alpha`, `r`). A claim is a phrase ('letter ban held', 'held under
        pressure'), because a claim has to say what it is claiming.

    A fractional number alone in a cell is a claim on its own, whatever the row
    looks like: no LoRA setting needs four decimal places except the learning
    rate, and that one is exactly what the declaration hatch is for.

    KNOWN GAP, stated rather than hidden: `| tuned | 146 |` satisfies both
    conditions -- one number, an identifier-shaped label -- and passes
    undeclared. A bare integer with no denominator, no unit and a one-word label
    is genuinely the same shape as a config value, and 146 says nothing at all
    without the 147 that would make it a rate (which tier O then refuses).
    """
    if masked.count("|") < 2:
        return []
    leading = masked.strip().startswith("|")
    cells = masked.split("|")[1:-1] if leading else masked.split("|")
    pos = masked.index("|") + 1 if leading else 0
    numeric, labels = [], []
    for cell in cells:
        m = TIER_D_CELL.match(cell)
        if m:
            numeric.append((m.group(1), pos + m.start(1), pos + m.end(1)))
        elif cell.strip():
            labels.append(cell.strip())
        pos += len(cell) + 1
    config_row = (len(numeric) == 1
                  and all(re.fullmatch(r"[\w.@/+-]+", lb) for lb in labels))
    return [("D", ev, a, b) for ev, a, b in numeric
            if not config_row or re.search(r"[.,]\d", ev)]


def rate_hits(line, declared=False):
    """(fatal, declared_numbers) for ONE normalized line.

    fatal: [(tier, evidence, start, end)] the gate refuses.
    declared_numbers: the bare numbers a declaration legitimately covers, so the
    gate can say out loud how many numbers the author claimed are not results.

    Tiers A (unit-bearing) and C (a PRINTED fraction, 146/147) are fatal whether
    or not the line is declared: '%' and '/' are rate notation and nothing else,
    so a declaration may not cover one. Tiers O (an of-form ratio in prose), B
    and D (bare numbers) and E (metric vocabulary beside a surviving number) are
    fatal only on an UNDECLARED line -- except that a declared line which also
    speaks metric vocabulary loses its declaration (tier V), because that is the
    hatch being used for exactly what it may not cover.
    """
    fatal = [("A", m.group(0).strip(), m.start(), m.end())
             for rx in (TIER_A, TIER_A2, TIER_A3) for m in rx.finditer(line)]
    masked = mask_structural(line)
    fatal += [("C", m.group(0), m.start(), m.end())
              for m in TIER_C.finditer(masked)]
    bare = [("O", " ".join(m.group(0).split()), m.start(), m.end())
            for m in TIER_C_OF.finditer(masked)]
    bare += [("B", m.group(0), m.start(), m.end())
             for m in TIER_B.finditer(masked)
             if not THOUSANDS_RE.match(m.group(0))]
    bare += table_cell_hits(masked)
    bare.sort(key=lambda h: (h[2], h[3]))
    vocab = bool(METRIC_WORDS.search(line))
    if declared:
        if vocab and bare:
            tier, ev, a, b = bare[0]
            fatal.append(("V", ev, a, b))
        return _dedupe(fatal), _dedupe(bare)
    fatal += bare
    if vocab:
        left = ANY_NUM.search(masked)
        if left:
            fatal.append(("E", left.group(0), left.start(), left.end()))
    return _dedupe(fatal), []


def _dedupe(hits):
    """One number reported once: TIER_B and the table-cell rule match the same span of
    '| 0.0002 |', and a reader does not need to be told twice."""
    out, seen = [], set()
    for tier, ev, a, b in hits:
        if (a, b) in seen:
            continue
        seen.add((a, b))
        out.append((tier, ev, a, b))
    return out


def join_lines(kept):
    """The non-block report as ONE string, plus (start, end, line_no) spans.

    A claim split across a newline ('Adherence reaches 99.4\\n% on the subset')
    is one claim; joining is what makes it visible. The spans map a hit in the
    joined text back to the line(s) it straddles.
    """
    parts, spans, pos = [], [], 0
    for line_no, norm in kept:
        parts.append(norm)
        spans.append((pos, pos + len(norm), line_no))
        pos += len(norm) + 1
    return " ".join(parts), spans


def _paragraph(text, i):
    """The markdown PARAGRAPH holding line `i` -- the contiguous run of non-blank
    lines around it -- or just [i] if that run holds a table row.

    A source line is not a line to the reader: prose wrapped at 80 columns is one
    sentence, and a line-scoped declaration on a wrapped sentence covered only the
    fragment that happened to carry the comment. A report that cannot declare
    "character constraints are 0.598 of train — corpus composition, not a
    measurement" because the sentence wrapped is a report nobody can write.

    A TABLE row is different: it IS a line to the reader, and letting one
    declaration at the bottom of a table exempt every row above it would make the
    explicit `<!-- non-metric-block: ... -->` form pointless. Tables keep the
    block form, which names its own boundaries.
    """
    lines = text.splitlines()

    def stop(j):
        ln = lines[j].strip()
        return (not ln or BEGIN_METRICS in lines[j] or END_METRICS in lines[j])

    lo = hi = i
    while lo > 0 and not stop(lo - 1):
        lo -= 1
    while hi + 1 < len(lines) and not stop(hi + 1):
        hi += 1
    run = list(range(lo, hi + 1))
    if any(lines[j].strip().startswith("|") for j in run):
        return [i]
    return run


def declared_non_metric_lines(gate, text):
    """(exempt, regions) for the lines the report EXPLICITLY declares carry no
    number this gate re-derived.

    exempt  {line_index: reason}
    regions {region_id: (declared_line_indices, reason, opened_at)} -- one entry
            per declaration, so the visible-disclaimer rule below can ask about
            the DECLARATION rather than about each line under it.

    This is the ONLY escape from the rule below, and it is deliberately loud: it
    lives in the report itself, next to the number, and it must state a reason.
    A grader reading the report sees every place the author claimed a number is
    not a result.
    """
    exempt, regions, open_at, why = {}, {}, None, ""
    for i, line in enumerate(text.splitlines()):
        if NON_METRIC_CLOSE_RE.search(line):
            if open_at is None:
                gate.fail(f"docs/MVP_REPORT.md line {i + 1} closes a non-metric "
                          f"block that was never opened")
            else:
                regions[open_at][0].append(i)
            exempt[i], open_at, why = why, None, ""
            continue
        m = NON_METRIC_OPEN_RE.search(line)
        if m:
            if open_at is not None:
                gate.fail(f"docs/MVP_REPORT.md line {i + 1} opens a non-metric "
                          f"block inside the one opened at line {open_at + 1}")
            open_at, why = i, m.group(1).strip()
            exempt[i] = why
            regions[i] = ([i], why, i)
            if not why:
                gate.fail(f"docs/MVP_REPORT.md line {i + 1}: a non-metric block "
                          f"must state WHY the numbers in it are not results "
                          f"(`<!-- non-metric-block: <reason> -->`)")
            continue
        if open_at is not None:
            exempt[i] = why
            regions[open_at][0].append(i)
            continue
        m = NON_METRIC_LINE_RE.search(line)
        if m:
            exempt[i] = m.group(1).strip()
            regions[i] = (_paragraph(text, i), exempt[i], i)
            for j in regions[i][0]:
                exempt.setdefault(j, exempt[i])
            if not exempt[i]:
                gate.fail(f"docs/MVP_REPORT.md line {i + 1}: a non-metric "
                          f"declaration must state WHY the number is not a result "
                          f"(`<!-- non-metric: <reason> -->`)")
    if open_at is not None:
        gate.fail(f"docs/MVP_REPORT.md opens a non-metric block at line "
                  f"{open_at + 1} and never closes it -- an unterminated block "
                  f"would exempt the whole rest of the report")
    return exempt, regions


def check_declarations_are_visible(gate, regions, normalized, covered):
    """A declaration that actually exempts a number must SAY SO IN THE REPORT.

    `<!-- non-metric: ... -->` renders as nothing, so the hatch used to cost an
    author nothing: an audit wrapped a whole fabricated base-vs-tuned table in
    one and the gate called the report clean while every reader saw an
    unqualified results table. Policing the wording of the hatch is what failed
    before it (a 19-word vocabulary, worded around in one line), so the hatch is
    policed by VISIBILITY instead: somewhere in the declared region, in text that
    RENDERS, the report has to say these numbers are not measured results.

    That is a boundary, not a proof: it cannot stop an author writing a false
    number under a visible disclaimer. What it does stop is a number appearing in
    the rendered report as if it were this package's measurement while the only
    thing marking it otherwise is an invisible comment.

    `covered` is the set of line indices where a declaration actually suppressed
    something -- a declaration over prose with no numbers in it needs no
    disclaimer, and demanding one would just teach authors to paste it blindly.

    The disclaimer may sit in the declared region OR in the few lines
    immediately above it, because a caption above a table is where a person
    actually writes one and markdown renders it right there.
    """
    for lines, why, opened_at in regions.values():
        if not (covered & set(lines)):
            continue
        window = list(lines) + [i for i in range(max(0, opened_at - DISCLAIMER_LOOKBACK),
                                                 opened_at)]
        if any(DISCLAIMER_RE.search(normalized.get(i, "")) for i in window):
            continue
        gate.fail(
            f"docs/MVP_REPORT.md line {opened_at + 1} declares "
            f"{why!r} non-metric, and that declaration is an HTML comment: it "
            f"renders as NOTHING, so a grader reading the report sees the "
            f"numbers under it with no qualification at all. Say it where the "
            f"reader can see it -- put a phrase like 'not a measurement' or "
            f"'not a measured result' (and what these numbers actually are) in "
            f"the visible text of the declared region, not only in the comment.")


def check_report_metrics(gate, text, block):
    """docs/MVP_REPORT.md must CONTAIN the generated block, byte for byte, and
    state no rate-shaped number outside it.

    WHY THIS IS NOT A PARSER ANY MORE. The rule this replaced was already the
    inverted one -- every percentage, ratio or printed fraction had to be either a
    table cell matched against the re-derived value or a line explicitly declared
    non-metric. An audit ran 18 adversarial report bodies through it with the
    honest table left intact and ALL 18 PASSED, in three families:

      * the "percentage" it recognised was `%`, `pp` and `percentage point` and
        nothing else. "12 percent to 99 percent", "99.4 pct", "87.4 p.p.",
        "8740 bps", "8.3x", ".994", "0,994", "99.4 %", "**99.4**%", "99.4&#37;",
        "99.4％", "99.4٪", "146/147." and "146 of 147" all walked past it, as did
        a claim split across a newline and "spec_adherence-0.994" (its lookbehind
        excluded any ratio preceded by punctuation).
      * only tables it could ORIENT were examined, so an unorientable table of
        bare numbers was never looked at -- and a body row with no leading pipe,
        or a separator row with no trailing pipe, made the whole table parse with
        an EMPTY BODY, so nothing in it was checked at all.
      * the escape hatch was policed by a 15-word vocabulary, so an inflated
        claim phrased outside that vocabulary could be declared away.

    So the report no longer states the numbers. `block` is regenerated here from
    the transcripts by render_metric_block, and this function requires it to
    appear VERBATIM. That leaves exactly three things to check:

      (a) exactly one BEGIN and one END marker, and the region between them equal
          to the regenerated block byte for byte (a unified diff on mismatch --
          "mismatch" alone tells the author nothing);
      (b) the block is not sitting inside a non-metric declaration -- the one way
          left to make the gate's own evidence disappear;
      (c) every line OUTSIDE the block, and the joined text of all of them, is
          free of rate-shaped numbers unless the line is declared non-metric with
          a stated reason, speaks no metric vocabulary, and carries a VISIBLE
          disclaimer -- see check_declarations_are_visible: an HTML comment
          renders as nothing, so a hatch made only of one costs a dishonest
          author nothing at all.

    WHERE THE BOUNDARY IS, STATED HONESTLY. (c) refuses a rate; it does not
    understand English. A claim with no number in it at all ("the tuned model is
    dramatically better") is outside the rule, and always will be. What the rule
    guarantees is that a NUMBER a grader could mistake for a result cannot appear
    in this report unless this gate generated it from the committed transcripts.
    """
    lines = text.splitlines()
    block_lines, block_ok = set(), False

    # --- (a) the block, byte for byte ------------------------------------
    n_begin, n_end = text.count(BEGIN_METRICS), text.count(END_METRICS)
    if n_begin != 1 or n_end != 1:
        gate.fail(
            f"docs/MVP_REPORT.md holds {n_begin} {BEGIN_METRICS!r} marker(s) and "
            f"{n_end} {END_METRICS!r} marker(s); exactly one of each is required. "
            f"The report's numbers are generated, not written: run "
            f"`python scripts/check_mvp.py --emit-metrics` and paste the block it "
            f"prints (markers included) into the report.")
    else:
        i0 = next(i for i, ln in enumerate(lines) if BEGIN_METRICS in ln)
        i1 = next(i for i, ln in enumerate(lines) if END_METRICS in ln)
        if i1 < i0:
            gate.fail(f"docs/MVP_REPORT.md closes the generated metrics block at "
                      f"line {i1 + 1}, before it opens it at line {i0 + 1}")
        else:
            block_lines = set(range(i0, i1 + 1))
            found = "\n".join(lines[i0:i1 + 1])
            if found == block:
                block_ok = True
            else:
                diff = "\n".join(
                    "      " + d for d in difflib.unified_diff(
                        found.splitlines(), block.splitlines(),
                        fromfile="docs/MVP_REPORT.md lines "
                                 f"{i0 + 1}-{i1 + 1} (as committed)",
                        tofile="re-derived from the committed transcripts",
                        lineterm="", n=1))
                gate.fail(
                    f"docs/MVP_REPORT.md's generated metrics block is not the one "
                    f"this gate re-derives from the transcripts. It is generated "
                    f"output, not prose: regenerate it with "
                    f"`python scripts/check_mvp.py --emit-metrics` rather than "
                    f"editing it.\n{diff}")

    # --- (b) the block may not be declared away ---------------------------
    exempt, regions = declared_non_metric_lines(gate, text)
    covered = sorted(block_lines & set(exempt))
    if covered:
        gate.fail(
            f"docs/MVP_REPORT.md line {covered[0] + 1}: the generated metrics "
            f"block sits inside a non-metric declaration "
            f"({exempt[covered[0]]!r}). The block IS the measurement -- the "
            f"declaration is for config values, counts and dates, and a report "
            f"that declares its own results non-metric has declared away the "
            f"only numbers this gate can vouch for.")

    # --- (c) everything outside it ----------------------------------------
    where = (f"it is outside the generated metrics block ({BEGIN_METRICS} … "
             f"{END_METRICS})")
    kept, n_declared, swept = [], 0, 0
    normalized, exempted_numbers = {}, set()
    for line_no, raw in enumerate(lines):
        if line_no in block_lines:
            continue
        norm = normalize_report_text(raw)
        normalized[line_no] = norm
        kept.append((line_no, norm))
        declared = line_no in exempt
        fatal, allowed = rate_hits(norm, declared=declared)
        n_declared += len(allowed)
        if allowed:
            exempted_numbers.add(line_no)
        if declared:
            for tier, ev, _, _ in fatal:
                if tier in ("V", "E") or METRIC_WORDS.search(norm):
                    gate.fail(
                        f"docs/MVP_REPORT.md line {line_no + 1} is declared "
                        f"non-metric ({exempt[line_no]!r}) but states a number in "
                        f"metric vocabulary: {ev!r} in {raw.strip()!r}. The "
                        f"declaration is for config values, counts and dates; a "
                        f"number the MVP claim rests on has to be one this gate "
                        f"generated from the transcripts.")
                else:
                    gate.fail(
                        f"docs/MVP_REPORT.md line {line_no + 1} is declared "
                        f"non-metric ({exempt[line_no]!r}) but states {ev!r}, and "
                        f"a percentage, a printed fraction or a number carrying a "
                        f"rate unit can never be declared away -- there is no "
                        f"honest reason for one to sit outside the generated "
                        f"metrics block. Line: {raw.strip()!r}")
            continue
        swept += 1
        for tier, ev, _, _ in fatal:
            if tier == "E":
                gate.fail(
                    f"docs/MVP_REPORT.md line {line_no + 1} speaks metric "
                    f"vocabulary beside {ev!r}, a performance number this gate "
                    f"cannot check: {where}. Let the block carry the claim, or -- "
                    f"if {ev!r} really is a count, a seed or a config value -- "
                    f"mark the line `<!-- non-metric: <why> -->` and word it "
                    f"without metric vocabulary. Line: {raw.strip()!r}")
            else:
                gate.fail(
                    f"docs/MVP_REPORT.md line {line_no + 1} states {ev!r}, a "
                    f"performance number this gate cannot check: {where}, and the "
                    f"line is not declared non-metric. Regenerate the block with "
                    f"`python scripts/check_mvp.py --emit-metrics` and let it "
                    f"carry the number, or mark the line "
                    f"`<!-- non-metric: <why> -->` if it is a config value, a "
                    f"count or a date. Line: {raw.strip()!r}")

    # A declaration that actually hid a number must be visible to the reader.
    check_declarations_are_visible(gate, regions, normalized, exempted_numbers)

    # A rate split across a newline is still one rate. The joined text is read
    # only where a hit STRADDLES a line boundary -- everything inside one line
    # was already reported above -- and a straddle every one of whose lines is
    # declared is treated exactly as the same hit would be on one line.
    joined, spans = join_lines(kept)
    masked_join = mask_structural(joined)
    for rx, src, declarable in ((TIER_A, joined, False),
                                (TIER_A2, joined, False),
                                (TIER_A3, joined, False),
                                (TIER_C, masked_join, False),
                                (TIER_C_OF, masked_join, True)):
        for m in rx.finditer(src):
            touched = [s for s in spans if s[0] < m.end() and m.start() < s[1]]
            if len(touched) < 2:
                continue
            if declarable and all(s[2] in exempt for s in touched):
                continue
            first, last = touched[0][2], touched[-1][2]
            gate.fail(
                f"docs/MVP_REPORT.md lines {first + 1}-{last + 1} state "
                f"{' '.join(m.group(0).split())!r} across a line break, a "
                f"performance number this gate cannot check: {where}. Splitting a "
                f"claim over two lines does not make it a different claim.")

    if block_ok:
        gate.note(f"the generated metrics block in docs/MVP_REPORT.md is "
                  f"byte-identical to the one re-derived from the committed "
                  f"transcripts ({len(BLOCK_RATES) + len(BLOCK_COUNTS)} rows x "
                  f"base|tuned|delta, {len(block.splitlines())} lines)")
    gate.note(f"{swept} line(s) outside the generated block swept for "
              f"percentages, ratios, printed fractions, unit-bearing rates and "
              f"metric vocabulary")
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
def check_report(gate, root, cfg, base_spec, run_dir, block, rubric_hash,
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

    # --- the report states both headline metrics -------------------------
    # (the "at least two markdown tables" count this replaced was a proxy for
    # "results table + delta table". The generated block IS both -- six metrics
    # against base | tuned | delta -- and it is required below byte for byte, so
    # counting separator rows no longer proves anything the block does not.)
    normalized = text.lower().replace("-", "_").replace(" ", "_")
    for metric in ("spec_adherence", "robustness"):
        if metric not in normalized:
            gate.fail(f"docs/MVP_REPORT.md never reports {metric}")

    # --- the numbers are the GENERATED ones, and nothing else is a number --
    check_report_metrics(gate, text, block)

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
    ap.add_argument("--emit-metrics", action="store_true",
                    help="print the generated metrics block for "
                         "docs/MVP_REPORT.md on stdout and exit 0. The report is "
                         "hand-written everywhere else; this is the one region "
                         "the author pastes rather than types, and the gate "
                         "requires it back byte for byte.")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    run_dir = Path(args.run_dir).resolve() if args.run_dir else root / "runs" / "mvp"
    emit = args.emit_metrics
    log = sys.stderr if emit else sys.stdout
    gate = Gate(f"MVP submission check: {root}", stream=log)
    print(f"checking {root} against judge rubric {judge_mod.rubric_hash()}",
          file=log)

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

    gate.note(f"base-vs-tuned on {len(eval_index)} shared scenarios, "
              f"spec_adherence {stats['base']['spec_adherence']:.4f} -> "
              f"{stats['tuned']['spec_adherence']:.4f}")

    # The report's numbers, re-derived. This is the ONLY place they come from:
    # the block is rendered from score_cell's stats here, printed for the author
    # under --emit-metrics, and required back from the report byte for byte.
    block = render_metric_block(
        stats["base"], stats["tuned"], n_eval=len(eval_index),
        eval_rel=eval_rel, eval_sha256=sha256_of(eval_path),
        rubric_hash=rubric_hash,
        judge_model=cells[roles["tuned"]][1].get("judge_model"),
        base_breakdown=derive_breakdown(stats["base"], eval_index, instances),
        tuned_breakdown=derive_breakdown(stats["tuned"], eval_index, instances))

    if emit:
        # A package that does not re-derive has no honest block to emit.
        if gate.failures:
            return gate.finish()
        print(block)
        return 0

    check_report(gate, root, cfg, base_spec, run_dir, block, rubric_hash,
                 eval_rel)
    return gate.finish()


if __name__ == "__main__":
    sys.exit(main())
