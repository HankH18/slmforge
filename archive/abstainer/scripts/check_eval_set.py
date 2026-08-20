#!/usr/bin/env python3
"""Preflight validator for an eval scenario set — READ-ONLY.

Run this on ANY scenario JSONL (especially a staff held-out set you have never
seen) BEFORE running eval.py on it. It answers one question: will the frozen
harness score this file correctly?

Why this exists
---------------
`eval.py` is frozen and does no input validation. Two failure modes follow:

  crash    — it subscripts `scenario["id"]`, `["gold_label"]`, `["question"]`,
             `["passage"]` directly (eval.py:118, 149-156, 174, 179), so a
             missing key is a KeyError mid-run.

  SILENT MISSCORE — worse. `score_example` branches on
             `if scenario["gold_label"] == "unanswerable":` (eval.py:158) with
             no else-validation. EVERY other value falls through to the
             "gold answerable" branch. A held-out set that spells the label
             "not_answerable", "Unanswerable", "no_answer" or `false` is scored
             as if every unanswerable row were answerable: a perfectly
             abstaining model reports 0.0% spec_adherence with
             failure=false_refusal, and false_answer_rate reports "n/a"
             because the unanswerable bucket is empty. Exit code 0, clean
             table, completely wrong numbers.

Exit codes
----------
  0  the file is safe to score
  1  the file would crash or be misscored (details printed)
  2  usage / IO error

This script never writes anything: it does not modify eval.py and it does not
rewrite, normalize, or "fix" the input file. Validate and report only.

Usage
-----
  python scripts/check_eval_set.py data/eval/scenarios_v1.jsonl
  python scripts/check_eval_set.py <held-out>.jsonl --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

VALID_LABELS = ("answerable", "unanswerable")

# Keys eval.py subscripts directly (KeyError if absent) for every row.
HARD_KEYS = ("id", "gold_label", "question", "passage")
# Keys eval.py subscripts only on gold=answerable rows, and only through the
# mock:oracle generator (eval.py:290-293) — the sanity double the whole suite
# and run_ablation lean on. Absent => KeyError/IndexError.
ORACLE_KEYS = ("gold_answer", "gold_spans")
# Keys read with .get() — never crash, but silently change the metrics.
SOFT_KEYS = ("type", "adversarial")

MIN_SCENARIOS = 30  # project spec: >= 30 scenarios per model x strategy cell


class Report:
    """Collects findings. `errors` mean the file would crash or be misscored."""

    def __init__(self) -> None:
        self.errors: list[tuple[str, str]] = []
        self.warnings: list[tuple[str, str]] = []

    def error(self, check: str, msg: str) -> None:
        self.errors.append((check, msg))

    def warn(self, check: str, msg: str) -> None:
        self.warnings.append((check, msg))

    @property
    def ok(self) -> bool:
        return not self.errors


def _where(lineno: int, row: object) -> str:
    rid = row.get("id") if isinstance(row, dict) else None
    return f"line {lineno} (id={rid!r})" if rid is not None else f"line {lineno}"


def check_import_of_frozen_harness(rep: Report) -> None:
    """The single most common environment trip-up: eval.py needs Python 3.11+."""
    if sys.version_info < (3, 11):
        rep.error(
            "python_version",
            f"running on Python {sys.version_info.major}.{sys.version_info.minor}; "
            "eval.py uses PEP 604 annotations (`list | None`) evaluated at import "
            "time and cannot be imported below 3.11. Use the project venv: "
            ".venv/bin/python",
        )
        return
    sys.path.insert(0, str(REPO_ROOT))
    try:
        import eval as E  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        rep.error("harness_import", f"could not import the frozen eval.py: {exc!r}")


def check_rows(rows: list[tuple[int, object]], rep: Report) -> None:
    seen_ids: dict[str, int] = {}
    n_adversarial = 0
    has_adversarial_key = 0

    for lineno, row in rows:
        where = _where(lineno, row)

        if not isinstance(row, dict):
            rep.error("row_type", f"{where}: row is {type(row).__name__}, not a JSON object")
            continue

        # ---- hard-required keys: eval.py subscripts these, KeyError if absent
        missing = [k for k in HARD_KEYS if k not in row]
        if missing:
            rep.error(
                "missing_required_key",
                f"{where}: missing {missing} — eval.py subscripts these directly "
                "(build_prompt eval.py:118, score_example eval.py:149-158), so the "
                "run dies with KeyError partway through",
            )

        # ---- id: uniqueness. Duplicates do not crash; they silently collide in
        #      the transcripts and make per-scenario failure analysis wrong.
        rid = row.get("id")
        if "id" in row:
            if not isinstance(rid, str) or not rid.strip():
                rep.error("bad_id", f"{where}: id is {rid!r}; expected a non-empty string")
            else:
                if rid in seen_ids:
                    rep.error(
                        "duplicate_id",
                        f"{where}: id {rid!r} already used on line {seen_ids[rid]} — "
                        "transcripts key on scenario_id, so the duplicate rows are "
                        "indistinguishable after the run",
                    )
                else:
                    seen_ids[rid] = lineno

        # ---- gold_label: THE silent-misscore vector
        label = row.get("gold_label")
        if "gold_label" in row and label not in VALID_LABELS:
            rep.error(
                "bad_gold_label",
                f"{where}: gold_label={label!r} is not one of {list(VALID_LABELS)}. "
                "score_example (eval.py:158) tests `== \"unanswerable\"` and treats "
                "EVERY other value as answerable — this row is SILENTLY MISSCORED, "
                "not rejected. A correct refusal on it is recorded as false_refusal.",
            )

        # ---- question / passage
        for key in ("question", "passage"):
            if key in row:
                val = row[key]
                if not isinstance(val, str):
                    rep.error("bad_text_field", f"{where}: {key} is {type(val).__name__}, expected string")
                elif not val.strip():
                    rep.error("bad_text_field", f"{where}: {key} is empty")

        passage = row.get("passage")

        # ---- double quotes in the passage break Citation round-tripping
        if isinstance(passage, str) and '"' in passage:
            rep.error(
                "quote_in_passage",
                f'{where}: passage contains a double-quote character. The output '
                'format is `Citation: "<span>"` (CITATION_RE, eval.py:68), so a '
                "cited span containing a quote is only scored correctly if the "
                "model reproduces the quote raw; JSON-escaped (\\\"), curly (“ ”) "
                "or single-quoted renderings all parse but fail the verbatim-"
                "substring check (eval.py:174) as nonverbatim_citation. The score "
                "on this row depends on the model's quoting convention, not on the "
                "behavior under test.",
            )

        # ---- adversarial: soft key, but its absence silently empties robustness
        if "adversarial" not in row:
            rep.error(
                "missing_adversarial",
                f"{where}: no 'adversarial' key. score_example stores "
                "bool(scenario.get(\"adversarial\")) (eval.py:152), so the row "
                "silently counts as non-adversarial; if no row has it, aggregate "
                "reports robustness=None and the metric prints as 'n/a' with no "
                "error (eval.py:203).",
            )
        else:
            has_adversarial_key += 1
            adv = row["adversarial"]
            if not isinstance(adv, bool):
                rep.error(
                    "non_boolean_adversarial",
                    f"{where}: adversarial={adv!r} ({type(adv).__name__}), expected "
                    "a JSON boolean. eval.py coerces with bool(), so the string "
                    '"false" and the string "no" both become True and inflate the '
                    "robustness denominator.",
                )
            elif adv:
                n_adversarial += 1

        # ---- gold-side fields, needed on answerable rows
        if label == "answerable":
            for key in ORACLE_KEYS:
                if key not in row or row[key] is None:
                    rep.error(
                        "missing_gold_field",
                        f"{where}: gold_label=answerable but {key} is "
                        f"{'absent' if key not in row else 'null'} — the mock:oracle "
                        "sanity generator subscripts scenario['gold_answer'] and "
                        "scenario['gold_spans'][0] (eval.py:290-293) and dies with "
                        "KeyError/IndexError.",
                    )
            spans = row.get("gold_spans")
            if spans is not None:
                if not isinstance(spans, list):
                    rep.error(
                        "bad_gold_spans",
                        f"{where}: gold_spans is {type(spans).__name__}, expected a list of strings",
                    )
                elif not spans:
                    rep.error(
                        "empty_gold_spans",
                        f"{where}: gold_label=answerable with an empty gold_spans list — "
                        "mock:oracle raises IndexError on gold_spans[0] (eval.py:293)",
                    )
                elif isinstance(passage, str):
                    for i, span in enumerate(spans):
                        if not isinstance(span, str):
                            rep.error(
                                "bad_gold_spans",
                                f"{where}: gold_spans[{i}] is {type(span).__name__}, expected a string",
                            )
                        elif span not in passage:
                            rep.error(
                                "nonverbatim_gold_span",
                                f"{where}: gold_spans[{i}] is not a verbatim substring of "
                                f"the passage: {span[:70]!r}. A model that cites it exactly "
                                "is scored nonverbatim_citation (eval.py:174) — the row is "
                                "unpassable, which drags spec_adherence down for a defect "
                                "in the eval set, not in the model.",
                            )
        elif label == "unanswerable":
            if row.get("gold_spans"):
                rep.warn(
                    "unanswerable_with_spans",
                    f"{where}: gold_label=unanswerable but gold_spans is non-empty. "
                    "Harmless to scoring (the harness never reads it on this branch) "
                    "but it usually means the row's label is wrong.",
                )

    # ---- file-level checks
    n = len(rows)
    if has_adversarial_key and n_adversarial == 0:
        rep.warn(
            "no_adversarial_rows",
            "every row has adversarial=false, so aggregate reports robustness=None "
            "and the table prints 'n/a' for it (eval.py:203). Expected if the set is "
            "deliberately non-adversarial; a red flag otherwise.",
        )
    if 0 < n < MIN_SCENARIOS:
        rep.warn(
            "small_eval_set",
            f"{n} scenarios; the project spec asks for >= {MIN_SCENARIOS} per "
            "model x strategy cell.",
        )


def load_rows(path: Path, rep: Report) -> list[tuple[int, object]]:
    """Read the JSONL. Read-only: opened 'r', never written back."""
    rows: list[tuple[int, object]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            if not raw.strip():
                continue
            try:
                rows.append((lineno, json.loads(raw)))
            except json.JSONDecodeError as exc:
                rep.error(
                    "bad_json",
                    f"line {lineno}: not valid JSON ({exc.msg}) — eval.load_jsonl "
                    "(eval.py:402-404) raises before a single scenario is scored",
                )
    if not rows and not rep.errors:
        rep.error("empty_file", f"{path} contains no scenario rows")
    return rows


def summarize(rows: list[tuple[int, object]]) -> dict:
    dicts = [r for _, r in rows if isinstance(r, dict)]
    labels: dict[str, int] = {}
    for r in dicts:
        raw = r.get("gold_label")
        key = raw if isinstance(raw, str) else repr(raw)
        labels[key] = labels.get(key, 0) + 1
    return {
        "n": len(rows),
        "gold_label_counts": labels,
        "adversarial_true": sum(1 for r in dicts if r.get("adversarial") is True),
        "abstention_floor": (
            round(sum(1 for r in dicts if r.get("gold_label") == "unanswerable") / len(dicts), 4)
            if dicts else None
        ),
    }


def print_report(path: Path, stats: dict, rep: Report) -> None:
    print(f"eval-set preflight: {path}")
    print(f"  scenarios          {stats['n']}")
    print(f"  gold_label counts  {stats['gold_label_counts']}")
    print(f"  adversarial=true   {stats['adversarial_true']}")
    print(f"  abstention floor   {stats['abstention_floor']}"
          "   (score for refusing everything)")
    print()

    for title, items in (("ERROR", rep.errors), ("WARN", rep.warnings)):
        if not items:
            continue
        print(f"{title} ({len(items)}):")
        for check, msg in items:
            print(f"  [{check}] {msg}")
        print()

    if rep.ok:
        print("PASS — the frozen harness will score this file correctly.")
    else:
        print(f"FAIL — {len(rep.errors)} problem(s) would crash the run or misscore it. "
              "Do NOT trust numbers produced from this file until they are resolved.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Read-only preflight check that eval.py will score a scenario JSONL correctly.")
    ap.add_argument("eval_set", help="path to a scenario JSONL")
    ap.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = ap.parse_args(argv)

    path = Path(args.eval_set)
    if not path.is_file():
        print(f"error: no such file: {path}", file=sys.stderr)
        return 2

    rep = Report()
    check_import_of_frozen_harness(rep)
    try:
        rows = load_rows(path, rep)
    except OSError as exc:
        print(f"error: cannot read {path}: {exc}", file=sys.stderr)
        return 2
    check_rows(rows, rep)
    stats = summarize(rows)

    if args.json:
        print(json.dumps({
            "eval_set": str(path),
            "ok": rep.ok,
            "stats": stats,
            "errors": [{"check": c, "message": m} for c, m in rep.errors],
            "warnings": [{"check": c, "message": m} for c, m in rep.warnings],
        }, indent=2))
    else:
        print_report(path, stats, rep)

    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
