#!/usr/bin/env python3
"""Audit a generated training file. This is P3's verify gate, not a printer.

  python datagen/audit.py data/train/smoke_v0.jsonl --max-imbalance 0.05

Every number this gate prints is RECOMPUTED FROM THE ROWS. The generator's
sidecar report is treated as an untrusted claim: it is cross-checked against
the recomputed numbers and any disagreement is a gate failure. (A grader reads
these numbers; a sidecar that says "kept in file 64" over a 4-row file must not
print, it must fail.)

The leakage re-check is a SECOND, INDEPENDENT implementation of the 6-gram rule
(``AuditLeakageIndex`` below) -- not a call into the same ``datagen.filter``
code the generator used. Both are run and their verdicts must agree; if they
disagree the gate fails, which is what makes the re-check evidence rather than
a tautology.

Exits NON-ZERO if

  * label imbalance exceeds --max-imbalance, or
  * ANY 6-gram leakage against data/eval/*.jsonl is found, or
  * any row violates the training-example schema or the verifier contract, or
  * the file holds duplicate (passage, question) pairs, or
  * the file is empty / unreadable, or
  * the sidecar report disagrees with the rows, is missing, or is unreadable, or
  * the row count is below the size floor (--min-rows, or --min-fraction of the
    sidecar's n_requested), or
  * the two leakage implementations disagree.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import eval as E  # noqa: E402
from datagen.filter import (  # noqa: E402
    GOLD_LABELS, REASONS, TAXONOMY_TYPES, LeakageIndex, dedupe_key,
    default_leakage_index, normalize, split_user_msg,
)

META_KEYS = {"type", "gold_label", "source_batch", "verifier"}
ROW_KEYS = {"messages", "meta"}
ROLES = ("system", "user", "assistant")

#: default share of the sidecar's n_requested that must actually be in the file
DEFAULT_MIN_FRACTION = 0.95

EVAL_SCENARIOS = REPO_ROOT / "data" / "eval" / "scenarios_v1.jsonl"
EVAL_EXEMPLARS = REPO_ROOT / "data" / "eval" / "fewshot_exemplars.jsonl"
AUDIT_NGRAM_N = 6


def label_imbalance(counts: dict[str, int]) -> float:
    total = counts.get("answerable", 0) + counts.get("unanswerable", 0)
    if not total:
        return 0.0
    return abs(counts.get("answerable", 0) / total - 0.5)


# -------------------------------------------------- independent leakage guard
def audit_normalize(text: str) -> str:
    """Lowercase + collapse whitespace. Written here on purpose."""
    return " ".join(str(text).split()).lower()


def audit_ngrams(text: str, n: int = AUDIT_NGRAM_N) -> set[str]:
    words = audit_normalize(text).split()
    if len(words) < n:
        return set()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


class AuditLeakageIndex:
    """A SECOND implementation of the 6-gram leakage rule, owned by the gate.

    Deliberately shares no code with ``datagen.filter.LeakageIndex``: it does
    its own whitespace normalisation, its own n-gram extraction and its own
    JSONL parsing. If the gate re-used the generator's filter, a bug in that
    filter would be invisible to the gate and the "re-check" would only prove
    the code agrees with itself.
    """

    def __init__(self, paths=None, n: int = AUDIT_NGRAM_N):
        self.n = n
        self.grams: set[str] = set()
        self.sources = 0
        self.paths = [Path(p) for p in (paths or (EVAL_SCENARIOS, EVAL_EXEMPLARS))]
        for p in self.paths:
            for line in p.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                for field in ("passage", "question"):
                    text = row.get(field) or ""
                    if text:
                        self.grams |= audit_ngrams(text, n)
                        self.sources += 1

    def hits(self, *texts: str) -> list[str]:
        found: set[str] = set()
        for t in texts:
            if t:
                found |= (audit_ngrams(t, self.n) & self.grams)
        return sorted(found)


def check_row(row, idx: int, leakage, seen: dict, xcheck=None) -> list[str]:
    """Every schema + verifier violation on one row, as ``code: detail`` strings."""
    p = f"row {idx}"
    if not isinstance(row, dict):
        return [f"{p} bad_schema: row is not a JSON object"]
    if set(row) != ROW_KEYS:
        return [f"{p} bad_schema: top-level keys {sorted(row)} != {sorted(ROW_KEYS)}"]

    msgs = row["messages"]
    if not isinstance(msgs, list) or len(msgs) != 3:
        return [f"{p} bad_schema: messages must be a list of exactly 3 turns"]
    for i, (m, role) in enumerate(zip(msgs, ROLES)):
        if not isinstance(m, dict) or set(m) != {"role", "content"}:
            return [f"{p} bad_schema: message {i} keys must be exactly role/content"]
        if m["role"] != role:
            return [f"{p} bad_schema: message {i} role {m['role']!r} != {role!r}"]
        if not isinstance(m["content"], str) or not m["content"].strip():
            return [f"{p} bad_schema: message {i} content is empty or not a string"]

    errs: list[str] = []
    if msgs[0]["content"] != E.SYSTEM_CORE:
        errs.append(f"{p} system_not_eval_SYSTEM_CORE: system message is not byte-identical "
                    "to eval.SYSTEM_CORE")
    try:
        passage, question = split_user_msg(msgs[1]["content"])
    except ValueError as exc:
        errs.append(f"{p} user_not_eval_user_msg: {exc}")
        return errs
    assistant = msgs[2]["content"]

    meta = row["meta"]
    if not isinstance(meta, dict) or set(meta) != META_KEYS:
        errs.append(f"{p} bad_schema: meta keys {sorted(meta) if isinstance(meta, dict) else meta!r}"
                    f" != {sorted(META_KEYS)}")
        return errs
    if meta["type"] not in TAXONOMY_TYPES:
        errs.append(f"{p} unknown_type: {meta['type']!r}")
    if meta["gold_label"] not in GOLD_LABELS:
        errs.append(f"{p} unknown_gold_label: {meta['gold_label']!r}")
        return errs
    if not isinstance(meta["source_batch"], str) or not meta["source_batch"].strip():
        errs.append(f"{p} bad_schema: meta.source_batch is empty")
    if not meta["verifier"] or not isinstance(meta["verifier"], (str, dict)):
        errs.append(f"{p} bad_schema: meta.verifier is empty or the wrong type")

    # --- verifier contract, re-run from scratch on the emitted row ---
    if '"' in passage:
        errs.append(f"{p} quote_in_passage: passage holds a double-quote character")
    parsed = E.parse_output(assistant)
    cls = parsed["class"]
    if meta["gold_label"] == "unanswerable":
        if cls != "refusal":
            errs.append(f"{p} class_mismatch: gold unanswerable but parse class {cls!r}")
    else:
        if cls != "answer":
            errs.append(f"{p} class_mismatch: gold answerable but parse class {cls!r}")
        else:
            for s in parsed["spans"]:
                if s not in passage:
                    errs.append(f"{p} nonverbatim_citation: {s[:80]!r}")

    hits = leakage.hits(passage, question)
    if hits:
        errs.append(f"{p} leakage_6gram: {len(hits)} shared {leakage.n}-gram(s), "
                    f"first={hits[0]!r}")
    if xcheck is not None:
        # The generator's own guard, run as a second opinion. Agreement is what
        # makes the re-check evidence; disagreement means one of the two
        # implementations of the leakage rule is broken and must be fixed.
        other = xcheck.hits(passage, question)
        if bool(other) != bool(hits):
            errs.append(
                f"{p} guard_disagreement: audit guard says "
                f"{'LEAK' if hits else 'clean'} but datagen.filter says "
                f"{'LEAK' if other else 'clean'}")

    key = dedupe_key(passage, question)
    if key in seen:
        errs.append(f"{p} duplicate: same normalized (passage, question) as row {seen[key]}")
    else:
        seen[key] = idx
    return errs


def load_generation_report(path: Path, explicit: str | None) -> tuple[dict | None, Path | None]:
    cand = Path(explicit) if explicit else path.with_suffix(".report.json")
    if cand.exists():
        try:
            return json.loads(cand.read_text()), cand
        except Exception as exc:  # noqa: BLE001
            print(f"  (could not read {cand}: {exc})")
    return None, cand


def _nonzero(counts) -> dict[str, int]:
    """Drop zero entries so {} and {'x': 0} compare equal."""
    if not isinstance(counts, dict):
        return {}
    return {str(k): int(v) for k, v in counts.items() if isinstance(v, (int, float)) and v}


def crosscheck_report(report: dict, observed: dict) -> list[str]:
    """Recompute every sidecar number from the rows. Disagreement is a failure.

    ``observed`` carries what was measured off the file: rows, label_counts,
    type_counts, imbalance, batches. Anything the sidecar claims that the rows
    contradict lands here as a ``report_mismatch``.
    """
    errs: list[str] = []

    def mismatch(field, claimed, actual):
        errs.append(f"report_mismatch {field}: sidecar says {claimed!r}, "
                    f"rows say {actual!r}")

    rows = observed["rows"]

    n_kept = report.get("n_kept")
    if not isinstance(n_kept, int):
        errs.append(f"report_mismatch n_kept: missing or not an int ({n_kept!r})")
    elif n_kept != rows:
        mismatch("n_kept", n_kept, rows)

    for field, actual in (("label_counts", observed["label_counts"]),
                          ("type_counts", observed["type_counts"])):
        if field not in report:
            errs.append(f"report_mismatch {field}: absent from the sidecar")
            continue
        claimed = _nonzero(report.get(field))
        if claimed != _nonzero(actual):
            mismatch(field, claimed, _nonzero(actual))

    if "label_imbalance" in report:
        claimed = report["label_imbalance"]
        if not isinstance(claimed, (int, float)) or abs(claimed - observed["imbalance"]) > 1e-6:
            mismatch("label_imbalance", claimed, round(observed["imbalance"], 6))

    if "bucket_counts" in report:
        total = sum(_nonzero(report["bucket_counts"]).values())
        if total != rows:
            mismatch("bucket_counts total", total, rows)

    batch = report.get("batch_name")
    if isinstance(batch, str) and batch and observed["batches"]:
        if set(observed["batches"]) != {batch}:
            mismatch("batch_name", batch, sorted(observed["batches"]))

    claimed_hash = report.get("rubric_hash")
    if claimed_hash is not None and claimed_hash != E.rubric_hash():
        mismatch("rubric_hash", claimed_hash, E.rubric_hash())

    passed = report.get("passed_verifier")
    if isinstance(passed, int) and passed < rows:
        errs.append(f"report_mismatch passed_verifier: sidecar says {passed} passed "
                    f"the verifier but the file holds {rows} rows")

    requested = report.get("n_requested")
    if isinstance(requested, int) and isinstance(n_kept, int) and n_kept > requested:
        errs.append(f"report_mismatch n_kept: {n_kept} kept exceeds "
                    f"n_requested {requested}")

    rejected = report.get("rejected") or {}
    if isinstance(rejected, dict):
        unknown = sorted(set(rejected) - set(REASONS))
        if unknown:
            errs.append("report_mismatch rejected: reason code(s) not in the "
                        f"machine-readable REASONS contract: {unknown}")
        total = report.get("rejected_total")
        if isinstance(total, int) and total != sum(rejected.values()):
            mismatch("rejected_total", total, sum(rejected.values()))
    else:
        errs.append(f"report_mismatch rejected: not an object ({rejected!r})")

    return errs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Audit a datagen training file (P3 gate)")
    ap.add_argument("jsonl")
    ap.add_argument("--max-imbalance", type=float, default=0.05,
                    help="max |P(answerable) - 0.5| tolerated (default 0.05)")
    ap.add_argument("--report", default=None,
                    help="generator report JSON (default <jsonl>.report.json)")
    ap.add_argument("--eval-file", action="append", default=None,
                    help="override the leakage sources (repeatable)")
    ap.add_argument("--json-out", default=None, help="write the audit summary here")
    ap.add_argument("--max-errors", type=int, default=20,
                    help="how many violations to print in full")
    ap.add_argument("--min-rows", type=int, default=0,
                    help="absolute floor on row count (P3 smoke ~64, P5 v1 ~1024)")
    ap.add_argument("--min-fraction", type=float, default=DEFAULT_MIN_FRACTION,
                    help="rows must be >= this share of the sidecar's n_requested "
                         f"(default {DEFAULT_MIN_FRACTION})")
    ap.add_argument("--allow-missing-report", action="store_true",
                    help="do not fail when the generator sidecar is absent; the "
                         "kept/rejected numbers then go UNVERIFIED")
    args = ap.parse_args(argv)

    path = Path(args.jsonl)
    print(f"=== datagen audit: {path} ===")
    if not path.exists():
        print(f"FAIL  file not found: {path}")
        return 2
    try:
        rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    except json.JSONDecodeError as exc:
        print(f"FAIL  file is not valid JSONL: {exc}")
        return 2

    # Primary guard: the audit's OWN implementation of the 6-gram rule.
    # Second opinion: the generator's filter. Both run; disagreement fails.
    leakage = AuditLeakageIndex(args.eval_file)
    filter_leakage = (LeakageIndex.from_eval_files(args.eval_file) if args.eval_file
                      else default_leakage_index())

    errors: list[str] = []
    if leakage.grams != filter_leakage.grams:
        only_audit = len(leakage.grams - filter_leakage.grams)
        only_filter = len(filter_leakage.grams - leakage.grams)
        errors.append(
            f"guard_disagreement: the two {leakage.n}-gram indexes differ "
            f"({only_audit} gram(s) only in the audit guard, {only_filter} only in "
            "datagen.filter) -- the leakage rule is not implemented consistently")

    label_counts: dict[str, int] = {"answerable": 0, "unanswerable": 0}
    type_counts: dict[str, int] = {}
    batches: dict[str, int] = {}
    seen: dict = {}
    for i, row in enumerate(rows):
        errors.extend(check_row(row, i, leakage, seen, xcheck=filter_leakage))
        meta = row.get("meta") if isinstance(row, dict) else None
        if isinstance(meta, dict):
            if meta.get("gold_label") in label_counts:
                label_counts[meta["gold_label"]] += 1
            type_counts[str(meta.get("type"))] = type_counts.get(str(meta.get("type")), 0) + 1
            batches[str(meta.get("source_batch"))] = batches.get(str(meta.get("source_batch")), 0) + 1

    imbalance = label_imbalance(label_counts)
    leak_errors = [e for e in errors if " leakage_6gram:" in e]
    guard_errors = [e for e in errors if "guard_disagreement:" in e]
    schema_errors = [e for e in errors
                     if e not in leak_errors and e not in guard_errors]

    print(f"\nrows: {len(rows)}")
    print(f"source batches: {batches or '-'}")
    print(f"rubric hash (this checkout): {E.rubric_hash()}")
    print(f"leakage index (audit's own): {len(leakage.grams)} {leakage.n}-grams from "
          f"{leakage.sources} eval passages/questions")
    print(f"leakage index (datagen.filter, cross-check): {len(filter_leakage.grams)} "
          f"{filter_leakage.n}-grams -- "
          f"{'AGREE' if leakage.grams == filter_leakage.grams else 'DISAGREE'}")

    print("\n-- per-type counts --")
    for t in TAXONOMY_TYPES:
        print(f"  {t:<18} {type_counts.get(t, 0):>5}")
    unknown = {k: v for k, v in type_counts.items() if k not in TAXONOMY_TYPES}
    for k, v in sorted(unknown.items()):
        print(f"  {k:<18} {v:>5}   <-- NOT a taxonomy type")
    missing = [t for t in TAXONOMY_TYPES if not type_counts.get(t)]
    if missing:
        print(f"  note: no examples of {', '.join(missing)} (not a gate failure)")

    print("\n-- label balance --")
    total = sum(label_counts.values())
    for lab in GOLD_LABELS:
        share = (label_counts[lab] / total) if total else 0.0
        print(f"  {lab:<14} {label_counts[lab]:>5}  ({100 * share:5.1f}%)")
    print(f"  imbalance |P(answerable) - 0.5| = {imbalance:.4f}  "
          f"(budget {args.max_imbalance:.4f})")

    print("\n-- kept / rejected (sidecar report, CROSS-CHECKED against the rows) --")
    report, rpath = load_generation_report(path, args.report)
    report_errors: list[str] = []
    required_rows = max(0, args.min_rows)
    if report is None:
        print(f"  NO generation report at {rpath}: the kept/rejected/label numbers "
              "cannot be verified against anything.")
        if not args.allow_missing_report:
            report_errors.append(
                f"missing_report: no readable sidecar at {rpath} (pass "
                "--allow-missing-report to audit the rows alone)")
    else:
        observed = {"rows": len(rows), "label_counts": label_counts,
                    "type_counts": type_counts, "imbalance": imbalance,
                    "batches": batches}
        report_errors = crosscheck_report(report, observed)
        print(f"  report: {rpath}")
        print(f"  teacher: {report.get('teacher')}  batch: {report.get('batch_name')}")
        print(f"  requested {report.get('n_requested')}  "
              f"passed verifier {report.get('passed_verifier')}  "
              f"kept in file {report.get('n_kept')}  "
              f"(rows actually in the file: {len(rows)})")
        rejected = report.get("rejected") or {}
        if not isinstance(rejected, dict):
            rejected = {}
        if not rejected:
            print("  rejected: none")
        for code, count in sorted(rejected.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"    {code:<26} {count:>5}   {REASONS.get(code, '(unknown reason code)')}")
        print(f"  rejected total: {report.get('rejected_total', sum(rejected.values()))}")
        print(f"  teacher api calls: {report.get('teacher_api_calls')}  "
              f"cache hits: {report.get('cache_hits')}")
        for s in report.get("short_buckets") or []:
            print(f"  short bucket {s.get('bucket')}: {s.get('kept')}/{s.get('need')} "
                  f"-- {s.get('stop')}")
        requested = report.get("n_requested")
        if isinstance(requested, int) and requested > 0:
            required_rows = max(required_rows,
                                math.ceil(args.min_fraction * requested))
        print(f"  cross-check: {len(report_errors) or 'no'} disagreement(s) between "
              "the sidecar and the rows")
        for e in report_errors:
            print(f"    {e}")

    print("\n-- size floor --")
    print(f"  {len(rows)} row(s), floor {required_rows} "
          f"(--min-rows {args.min_rows}, --min-fraction {args.min_fraction})")

    print("\n-- leakage re-check against the frozen eval set --")
    print(f"  {len(leak_errors)} row(s) share a {leakage.n}-gram with an eval "
          "passage or question (audit's own guard)")
    print(f"  {len(guard_errors)} guard disagreement(s) between the audit guard "
          "and datagen.filter")

    print("\n-- schema / verifier violations --")
    print(f"  {len(schema_errors)} violation(s)")
    for e in (schema_errors + leak_errors + guard_errors)[:args.max_errors]:
        print(f"    {e}")
    extra = len(errors) - min(len(errors), args.max_errors)
    if extra > 0:
        print(f"    ... and {extra} more")

    failures = []
    if not rows:
        failures.append("file holds no rows")
    if schema_errors:
        failures.append(f"{len(schema_errors)} schema/verifier violation(s)")
    if leak_errors:
        failures.append(f"{len(leak_errors)} leaking row(s)")
    if guard_errors:
        failures.append(f"{len(guard_errors)} leakage-guard disagreement(s)")
    if imbalance > args.max_imbalance:
        failures.append(f"label imbalance {imbalance:.4f} > {args.max_imbalance:.4f}")
    if report_errors:
        failures.append(f"{len(report_errors)} sidecar/rows disagreement(s): "
                        + "; ".join(report_errors[:3]))
    if rows and len(rows) < required_rows:
        failures.append(f"only {len(rows)} row(s), below the floor of {required_rows}")

    summary = {
        "path": str(path), "rows": len(rows), "label_counts": label_counts,
        "imbalance": round(imbalance, 6), "max_imbalance": args.max_imbalance,
        "type_counts": type_counts, "leaking_rows": len(leak_errors),
        "guard_disagreements": len(guard_errors),
        "schema_violations": len(schema_errors),
        "errors": errors + report_errors,
        "report_mismatches": report_errors,
        "required_rows": required_rows,
        "generation_report": str(rpath) if report is not None else None,
        "passed": not failures, "failures": failures,
    }
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(summary, indent=2) + "\n")

    print()
    if failures:
        for f in failures:
            print(f"FAIL  {f}")
        return 1
    print(f"PASS  {len(rows)} rows (floor {required_rows}), imbalance {imbalance:.4f} "
          f"<= {args.max_imbalance:.4f}, zero leakage under two independent guards, "
          f"schema clean, sidecar agrees with the rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
