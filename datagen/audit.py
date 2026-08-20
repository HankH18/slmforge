#!/usr/bin/env python3
"""Dataset gate: RECOMPUTE every number from the rows, or exit non-zero.

This is a gate, not a printer. Everything it reports it derived from the file it
was given, one row at a time. It reads no sidecar and believes no manifest -- an
earlier version of this audit printed a hand-authored sidecar's counts as fact
while the file it was auditing held four rows, which is exactly the failure a
gate exists to make impossible.

What it re-derives
------------------
* row schema: exactly ``[user, assistant]``, no system turn, a non-empty
  completion, a constraint list, a provenance object with teacher + attempt, and
  the ``scenario`` the prompt can be re-derived from.
* PROMPT CONTRACT: every row's user message re-rendered through
  :func:`scenarios.render.render_user_message` and compared BYTE FOR BYTE. This
  is the same check ``scripts/check_loop.py`` and ``training/modal_app.py`` make
  before a GPU-second is spent; failing it here costs a second instead.
* COMPLIANCE: every constraint re-run through :func:`harness.run_checkers`, on
  the exact completion stored on disk.
* SPLIT: zero constraints of a held-out type.
* DISJOINTNESS: zero (task, constraint-set) pairs shared with the eval corpus,
  and zero duplicate pairs within the file.
* COMPOSITION: every set has 2-4 constraints with at least one negative, k spans
  2/3/4 over the file, and no constraint type exceeds ``--max-type-share`` of the
  constraint slots.

Exit codes: ``0`` clean, ``1`` at least one violation, ``2`` the file could not
be read at all.

Usage
-----
  python datagen/audit.py data/train/smoke_v0.jsonl
  python datagen/audit.py data/train/v1.jsonl --max-type-share 0.25 --min-rows 4096
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from constraints import registry                                  # noqa: E402
from harness import run_checkers                                  # noqa: E402
from scenarios.paths import load_corpus, load_instances, load_smoke  # noqa: E402
from scenarios.render import render_user_message                  # noqa: E402

from datagen.sampling import (HELDOUT_TYPES, NEGATIVE_TYPES,      # noqa: E402
                              eval_pair_keys, normalize_text, pair_key,
                              type_shares)

__all__ = ["audit", "AuditResult", "main"]

MAX_REPORTED_PER_CHECK = 5


class AuditResult(dict):
    """Findings plus the recomputed numbers. Truthy when the dataset is clean."""

    @property
    def ok(self) -> bool:
        return not self["violations"]

    def __bool__(self) -> bool:  # pragma: no cover - convenience only
        return self.ok


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from None
    return rows


def _schema_error(row: Mapping[str, Any], idx: int) -> Optional[str]:
    msgs = row.get("messages")
    if not isinstance(msgs, list) or not all(isinstance(m, dict) for m in msgs):
        return f"row {idx}: 'messages' is not a list of turn objects"
    roles = [m.get("role") for m in msgs]
    if "system" in roles:
        return (f"row {idx}: carries a system turn; the pinned schema has NO system "
                f"message (roles {roles!r})")
    if roles != ["user", "assistant"]:
        return f"row {idx}: turn roles are {roles!r}, expected ['user', 'assistant']"
    for turn, role in zip(msgs, roles):
        if not isinstance(turn.get("content"), str):
            return f"row {idx}: {role} content is not a string"
    if not msgs[1]["content"].strip():
        return (f"row {idx}: the completion is empty. An empty response passes every "
                f"checker vacuously (ruling A7); training on it teaches that silence "
                f"is compliance")
    cids = row.get("constraints")
    if not isinstance(cids, list) or not cids:
        return f"row {idx}: 'constraints' is missing or empty"
    if not all(isinstance(c, str) and c for c in cids):
        return f"row {idx}: 'constraints' must be a list of instance ids"
    prov = row.get("provenance")
    if not isinstance(prov, dict):
        return f"row {idx}: 'provenance' is missing"
    if not isinstance(prov.get("teacher"), str) or not prov["teacher"].strip():
        return f"row {idx}: provenance.teacher is missing or empty"
    attempt = prov.get("attempt")
    if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 0:
        return f"row {idx}: provenance.attempt is not a non-negative int ({attempt!r})"
    scenario = row.get("scenario")
    if not isinstance(scenario, dict):
        return (f"row {idx}: carries no 'scenario' object, so its prompt can only be "
                f"self-attested, never re-derived")
    if not isinstance(scenario.get("task"), str) or not scenario["task"].strip():
        return f"row {idx}: the row's scenario has no 'task' text to re-render from"
    if list(scenario.get("constraints") or []) != list(cids):
        return (f"row {idx}: the row's constraints do not match its scenario's -- the "
                f"row was not built from this scenario")
    return None


def audit(path: Path,
          max_type_share: float = 0.25,
          min_rows: int = 1,
          instances: Optional[Mapping[str, Mapping[str, Any]]] = None,
          corpus_rows: Optional[Sequence[Mapping[str, Any]]] = None) -> AuditResult:
    """Re-derive everything about ``path``. Returns findings + recomputed stats."""
    instances = instances if instances is not None else load_instances()
    if corpus_rows is None:
        corpus_rows = list(load_corpus()) + list(load_smoke())
    forbidden = eval_pair_keys(corpus_rows)
    forbidden_tasks = {normalize_text(r["task"]) for r in corpus_rows}

    rows = _load_rows(Path(path))
    violations: List[str] = []
    detail: Dict[str, List[str]] = {}

    def fail(check: str, message: str) -> None:
        bucket = detail.setdefault(check, [])
        if len(bucket) < MAX_REPORTED_PER_CHECK:
            bucket.append(message)
        if check not in violations:
            violations.append(check)

    if len(rows) < min_rows:
        fail("min_rows", f"{len(rows)} rows, expected at least {min_rows}")

    valid: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        err = _schema_error(row, idx)
        if err:
            fail("schema", err)
        else:
            valid.append(row)

    seen_pairs: Dict[Any, int] = {}
    row_types: List[List[str]] = []
    ks: List[Any] = []
    pressures: List[Any] = []
    tasks: set = set()

    for idx, row in enumerate(valid):
        scenario = row["scenario"]
        cids = list(row["constraints"])

        unknown = [c for c in cids if c not in instances]
        if unknown:
            fail("unknown_constraints",
                 f"row {idx}: constraint ids {unknown} are not in the instance library")
            continue

        types = [instances[c]["type"] for c in cids]
        row_types.append(types)
        ks.append(scenario.get("k"))
        pressures.append(scenario.get("pressure"))
        tasks.add(scenario["task"])

        held = sorted({t for t in types if registry.split_of(t) != "train"})
        if held:
            fail("heldout_types",
                 f"row {idx}: uses held-out type(s) {held}. Those types are the "
                 f"generalization measurement; training on them destroys it")

        if not 2 <= len(cids) <= 4:
            fail("k_range", f"row {idx}: {len(cids)} constraints, the spec says 2-4")
        if scenario.get("k") != len(cids):
            fail("k_field", f"row {idx}: scenario k={scenario.get('k')!r} but "
                            f"{len(cids)} constraints")
        if not any(t in NEGATIVE_TYPES for t in types):
            fail("negative_constraint",
                 f"row {idx}: no negative constraint; types were {types}")

        try:
            rendered = render_user_message(scenario, instances)
        except Exception as exc:  # noqa: BLE001
            fail("prompt_contract", f"row {idx}: scenario will not render: {exc}")
            continue
        if rendered != row["messages"][0]["content"]:
            fail("prompt_contract",
                 f"row {idx}: the stored user message is NOT what the shared renderer "
                 f"produces for this scenario.\n    stored:   "
                 f"{row['messages'][0]['content'][:180]!r}\n    rendered: "
                 f"{rendered[:180]!r}")

        completion = row["messages"][1]["content"]
        results = run_checkers(scenario, instances, completion)
        bad = [r for r in results if not r["pass"]]
        if bad:
            fail("compliance",
                 f"row {idx}: FAILS "
                 f"{[(r['constraint_id'], r['evidence'][:60]) for r in bad]}")

        key = pair_key(scenario["task"], cids)
        if key in forbidden:
            fail("eval_overlap",
                 f"row {idx}: its (task, constraint-set) pair is in the eval corpus")
        # TASK-level too, which is the standard datagen/task_pool.py states and the
        # auditor previously did not enforce: "training on its tasks under different
        # constraint sets is still a model that has seen the exact prose it will be
        # graded on." Pair-level alone certified verbatim eval prose as clean.
        if normalize_text(scenario["task"]) in forbidden_tasks:
            fail("eval_task_overlap",
                 f"row {idx}: its task text appears verbatim in the eval corpus, "
                 f"under any constraint set")
        if key in seen_pairs:
            fail("duplicate_pairs",
                 f"row {idx}: duplicates row {seen_pairs[key]} (same task and "
                 f"constraint set)")
        else:
            seen_pairs[key] = idx

    shares = type_shares(row_types)
    over = {t: round(v, 4) for t, v in shares.items() if v > max_type_share}
    if over:
        fail("type_share",
             f"types over the {max_type_share:.0%} cap: {over}")

    k_counts = Counter(ks)
    missing_k = [k for k in (2, 3, 4) if k_counts.get(k, 0) == 0]
    if valid and missing_k:
        fail("k_span", f"no examples with k in {missing_k}; the eval spans 2-4")

    stats = {
        "path": str(path),
        "rows": len(rows),
        "rows_valid": len(valid),
        "unique_tasks": len(tasks),
        "unique_pairs": len(seen_pairs),
        "k_distribution": dict(sorted(k_counts.items(), key=lambda kv: str(kv[0]))),
        "pressure_distribution": dict(Counter(pressures)),
        "type_counts": dict(Counter(t for types in row_types for t in types)),
        "type_shares": {t: round(v, 4) for t, v in shares.items()},
        "max_type_share": round(max(shares.values()), 4) if shares else 0.0,
        "constraint_slots": sum(len(t) for t in row_types),
        "heldout_types_present": sorted(
            {t for types in row_types for t in types if registry.split_of(t) != "train"}),
        "teachers": dict(Counter(r["provenance"]["teacher"] for r in valid)),
        "attempt_histogram": dict(Counter(r["provenance"]["attempt"] for r in valid)),
        "eval_corpus_pairs_compared": len(forbidden),
    }
    return AuditResult({"violations": violations, "detail": detail, "stats": stats,
                        "max_type_share": max_type_share, "min_rows": min_rows})


def _print_report(result: AuditResult) -> None:
    stats = result["stats"]
    print(f"audit: {stats['path']}")
    print(f"  rows                  {stats['rows']} "
          f"(schema-valid {stats['rows_valid']})")
    print(f"  unique tasks          {stats['unique_tasks']}")
    print(f"  unique (task,set)     {stats['unique_pairs']}")
    print(f"  constraint slots      {stats['constraint_slots']}")
    print(f"  k distribution        {stats['k_distribution']}")
    print(f"  pressure              {stats['pressure_distribution']}")
    print(f"  type counts           {stats['type_counts']}")
    print(f"  type shares           {stats['type_shares']}")
    print(f"  max type share        {stats['max_type_share']:.1%} "
          f"(cap {result['max_type_share']:.0%})")
    print(f"  held-out types        {stats['heldout_types_present'] or 'none'} "
          f"(forbidden: {', '.join(HELDOUT_TYPES)})")
    print(f"  teachers              {stats['teachers']}")
    print(f"  attempts              {stats['attempt_histogram']}")
    print(f"  eval pairs compared   {stats['eval_corpus_pairs_compared']}")
    if result.ok:
        print("\nPASS - every row re-verified: prompt re-rendered byte-identical, "
              "every checker re-run, zero held-out types, zero eval overlap.")
        return
    print(f"\nFAIL - {len(result['violations'])} check(s) failed: "
          f"{', '.join(result['violations'])}")
    for check, messages in result["detail"].items():
        print(f"\n  [{check}]")
        for message in messages:
            print(f"    {message}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("jsonl", help="dataset to audit")
    ap.add_argument("--max-type-share", type=float, default=0.25)
    ap.add_argument("--min-rows", type=int, default=1)
    args = ap.parse_args(argv)

    path = Path(args.jsonl)
    if not path.exists():
        print(f"{path} does not exist", file=sys.stderr)
        return 2
    try:
        result = audit(path, max_type_share=args.max_type_share, min_rows=args.min_rows)
    except ValueError as exc:
        print(f"{exc}", file=sys.stderr)
        return 2
    _print_report(result)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
