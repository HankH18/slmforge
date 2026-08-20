#!/usr/bin/env python3
"""Teacher-distillation pipeline: draft, check, retry, reject, write.

Modelled on :mod:`scenarios.make_exemplars`, which does this correctly at
six-example scale: the teacher drafts, the pinned checkers verify, a violating
draft is retried, and nothing unverified ships. This is the same discipline with
a sampler in front of it, a rejection ledger behind it, and a disk cache so a
rerun costs nothing.

What a run does, per example
----------------------------
1. :mod:`datagen.sampling` pairs a FRESH task with a wide train-split constraint
   set (never a held-out type, always at least one negative constraint).
2. The prompt is :func:`scenarios.render.render_user_message` -- imported, the
   one definition, shared with eval. There is no second renderer here.
3. The teacher drafts. The response is cached on disk by request hash.
4. :func:`datagen.filter.decide` keeps or rejects it on the checkers' verdict.
5. A rejected draft is retried up to ``--attempts`` times, each retry carrying
   the failed draft and the checker's own evidence back to the teacher.
6. Every rejection is recorded with a machine-readable reason code. The fluent,
   on-task, single-violation ones are additionally written to a DPO-negatives
   file as candidate preference pairs.

Before anything is written, every row is re-verified from the values that will
actually be on disk: prompt re-rendered and compared byte for byte, checkers
re-run on the exact stored completion, held-out types re-checked, and the
(task, constraint-set) pair checked against the eval corpus. A row that fails
any of those is a bug in this file, and the run dies rather than shipping it.

Exit codes
----------
``0`` kept >= --min-kept. ``2`` a material shortfall. ``3`` zero rows kept, in
which case the dataset file is NOT created -- a previous incarnation of this
pipeline exited 0 having written a 0-byte dataset, which is a success signal
that means nothing.

Usage
-----
  set -a && . ./.env && set +a
  python datagen/generate.py --n 64 --out data/train/smoke_v0.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from constraints import registry                                  # noqa: E402
from harness import run_checkers                                  # noqa: E402
from scenarios.paths import load_corpus, load_instances, load_smoke  # noqa: E402
from scenarios.render import render_user_message                  # noqa: E402

from datagen import DatagenError                                  # noqa: E402
from datagen.filter import decide, fluency_report                 # noqa: E402
from datagen.sampling import (assert_has_negative,                # noqa: E402
                              assert_train_split_only, build_scenarios,
                              eval_pair_keys, pair_key, type_shares)
from datagen.task_pool import assert_disjoint_from_corpus         # noqa: E402
from datagen.teacher import (CACHE_DIR, TeacherCache, call_teacher,  # noqa: E402
                             repair_turns, resolve_teacher)

__all__ = ["generate", "write_outputs", "sidecar_paths", "summarise", "main"]

#: Fraction of --n that must survive before the batch counts as usable.
DEFAULT_KEEP_FRACTION = 0.75

VERIFIED_BY = "constraints.checkers.check"


def sidecar_paths(out: Path) -> Dict[str, Path]:
    """The three files that travel with a dataset, derived from its path."""
    out = Path(out)
    stem = out.name[: -len(out.suffix)] if out.suffix else out.name
    return {"dataset": out,
            "rejected": out.with_name(f"{stem}.rejected.jsonl"),
            "dpo_negatives": out.with_name(f"{stem}.dpo_negatives.jsonl"),
            "report": out.with_name(f"{stem}.report.json")}


def _row_for(sample, draft: str, attempt: int, teacher: str, reason: str,
             seed: int, lookup: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    scenario = sample.scenario
    return {
        "messages": [
            {"role": "user", "content": render_user_message(scenario, lookup)},
            {"role": "assistant", "content": draft},
        ],
        "constraints": list(scenario["constraints"]),
        "provenance": {
            "teacher": teacher,
            "attempt": attempt,
            "teacher_resolved_by": reason,
            "task_id": sample.task_id,
            "lexicon": sample.lexicon,
            "nudge": sample.nudge_id,
            "distractor": sample.distractor,
            "seed": seed,
            "verified_by": VERIFIED_BY,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
        },
        "scenario": dict(scenario),
    }


def _verify_before_write(row: Mapping[str, Any],
                         lookup: Mapping[str, Mapping[str, Any]],
                         forbidden_pairs: set) -> None:
    """Everything the auditor will check, checked here first, on the exact values
    that are about to be written."""
    scenario = row["scenario"]
    cids = list(scenario["constraints"])
    if list(row["constraints"]) != cids:
        raise DatagenError(f"{scenario['scenario_id']}: row constraints disagree with "
                           f"its scenario's")
    assert_train_split_only(cids, lookup)
    assert_has_negative(cids, lookup)

    rendered = render_user_message(scenario, lookup)
    if row["messages"][0]["content"] != rendered:
        raise DatagenError(
            f"{scenario['scenario_id']}: the stored user message is not what the "
            f"shared renderer produces. Training on it would measure the model "
            f"under a prompt it never saw.")

    completion = row["messages"][1]["content"]
    if not isinstance(completion, str) or not completion.strip():
        raise DatagenError(f"{scenario['scenario_id']}: empty completion")
    results = run_checkers(scenario, lookup, completion)
    bad = [r for r in results if not r["pass"]]
    if bad:
        raise DatagenError(
            f"{scenario['scenario_id']}: about to write a row that FAILS "
            f"{[r['constraint_id'] for r in bad]} on the exact stored text")

    if pair_key(scenario["task"], cids) in forbidden_pairs:
        raise DatagenError(
            f"{scenario['scenario_id']}: its (task, constraint-set) pair is in the "
            f"eval corpus -- that is contamination")


def generate(samples: Sequence[Any],
             generator,
             teacher: str,
             teacher_reason: str,
             lookup: Mapping[str, Mapping[str, Any]],
             forbidden_pairs: set,
             attempts: int = 4,
             cache: Optional[TeacherCache] = None,
             seed: int = 0,
             workers: int = 8,
             log=print) -> Dict[str, Any]:
    """Run the pipeline over pre-built samples. Returns rows + ledgers + stats.

    Takes the generator as an argument rather than building one, so the whole
    pipeline is exercisable with a scripted teacher and no network.
    """
    if attempts < 1:
        raise DatagenError(f"--attempts must be at least 1, got {attempts}")
    import eval as E  # the live harness, for its prompt plumbing only

    cache = cache if cache is not None else TeacherCache()
    if not getattr(generator, "thread_safe", False):
        workers = 1
    workers = max(1, min(int(workers), len(samples) or 1))

    lock = threading.Lock()
    kept: List[Optional[Dict[str, Any]]] = [None] * len(samples)
    rejections: List[Dict[str, Any]] = []
    negatives: List[Dict[str, Any]] = []
    give_ups: List[Dict[str, Any]] = []

    def one(index_sample):
        index, sample = index_sample
        scenario = sample.scenario
        prompt = E.build_prompt("zero_shot", scenario, lookup, None)
        base_turns = list(prompt["turns"])
        turns = base_turns
        local_rejections: List[Dict[str, Any]] = []
        local_negatives: List[Dict[str, Any]] = []
        row = None
        kept_attempt = None

        for attempt in range(1, attempts + 1):
            request = {"system": prompt["system"], "turns": turns}
            try:
                raw = call_teacher(generator, scenario, request, lookup, teacher, cache)
            except Exception as exc:  # noqa: BLE001 - recorded, never swallowed
                local_rejections.append({
                    "scenario_id": scenario["scenario_id"], "task_id": sample.task_id,
                    "attempt": attempt, "reason": "teacher_error",
                    "detail": f"{type(exc).__name__}: {exc}",
                    "constraints": list(scenario["constraints"]),
                    "failures": [], "draft": None})
                break

            draft = raw.strip()
            verdict = decide(draft, scenario, lookup)
            if verdict.keep:
                row = _row_for(sample, draft, attempt, teacher, teacher_reason,
                               seed, lookup)
                kept_attempt = attempt
                break

            report = fluency_report(draft, scenario["task"], verdict.results)
            local_rejections.append({
                "scenario_id": scenario["scenario_id"], "task_id": sample.task_id,
                "attempt": attempt, "reason": verdict.reason,
                "constraints": list(scenario["constraints"]),
                "failures": verdict.failures,
                "draft": draft,
                "fluency": report.as_dict()})
            if report.fluent:
                local_negatives.append({
                    "scenario_id": scenario["scenario_id"],
                    "prompt": render_user_message(scenario, lookup),
                    "rejected": draft,
                    "chosen": None,          # filled in below if a later attempt won
                    "violated": verdict.failures,
                    "constraints": list(scenario["constraints"]),
                    "scenario": dict(scenario),
                    "fluency": report.as_dict(),
                    "provenance": {"teacher": teacher, "attempt": attempt,
                                   "task_id": sample.task_id, "seed": seed}})
            if verdict.reason == "constraint_violation":
                turns = repair_turns(base_turns, draft, verdict.failures)
            else:
                # Nothing to quote back at the teacher for an empty or degenerate
                # draft, so ask again from the top rather than send it its own
                # silence.
                turns = base_turns

        with lock:
            kept[index] = row
            rejections.extend(local_rejections)
            if row is not None:
                for neg in local_negatives:
                    neg["chosen"] = row["messages"][1]["content"]
            negatives.extend(local_negatives)
            if row is None:
                give_ups.append({
                    "scenario_id": scenario["scenario_id"],
                    "task_id": sample.task_id,
                    "constraints": list(scenario["constraints"]),
                    "attempts": attempts,
                    "last_reason": (local_rejections[-1]["reason"]
                                    if local_rejections else "no_attempt")})
            done = sum(1 for r in kept if r is not None) + len(give_ups)
        if done % 16 == 0 or done == len(samples):
            log(f"  {done}/{len(samples)} examples resolved "
                f"({sum(1 for r in kept if r is not None)} kept)")

    if workers == 1:
        for pair in enumerate(samples):
            one(pair)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(one, list(enumerate(samples))))

    rows = [r for r in kept if r is not None]
    for row in rows:
        _verify_before_write(row, lookup, forbidden_pairs)

    return {"rows": rows, "rejections": rejections, "negatives": negatives,
            "give_ups": give_ups,
            "cache": {"hits": cache.hits, "misses": cache.misses}}


def summarise(rows: Sequence[Mapping[str, Any]],
              rejections: Sequence[Mapping[str, Any]],
              negatives: Sequence[Mapping[str, Any]],
              give_ups: Sequence[Mapping[str, Any]],
              lookup: Mapping[str, Mapping[str, Any]],
              requested: int) -> Dict[str, Any]:
    """Every reported number, computed from the rows themselves."""
    row_types = [[lookup[c]["type"] for c in row["constraints"]] for row in rows]
    shares = type_shares(row_types)
    type_counts = Counter(t for types in row_types for t in types)
    return {
        "requested": requested,
        "kept": len(rows),
        "accept_rate": (len(rows) / requested) if requested else 0.0,
        "gave_up": len(give_ups),
        "rejections": len(rejections),
        "rejection_reasons": dict(Counter(r["reason"] for r in rejections)),
        "rejected_constraint_types": dict(Counter(
            f["type"] for r in rejections for f in r.get("failures", []))),
        "dpo_negatives": len(negatives),
        "attempt_histogram": dict(Counter(row["provenance"]["attempt"] for row in rows)),
        "k_distribution": dict(Counter(row["scenario"]["k"] for row in rows)),
        "pressure_distribution": dict(Counter(row["scenario"]["pressure"] for row in rows)),
        "type_counts": dict(type_counts),
        "type_shares": {t: round(v, 4) for t, v in shares.items()},
        "max_type_share": round(max(shares.values()), 4) if shares else 0.0,
        "distractor_examples": sum(1 for row in rows
                                   if row["provenance"].get("distractor")),
        "unique_tasks": len({row["provenance"]["task_id"] for row in rows}),
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_outputs(out: Path, result: Mapping[str, Any],
                  report: Mapping[str, Any]) -> Dict[str, Path]:
    """Write the dataset and its ledgers. Refuses to create an empty dataset."""
    paths = sidecar_paths(Path(out))
    if not result["rows"]:
        raise DatagenError(
            "zero rows survived the filter; refusing to write an empty dataset "
            "file. An empty file that exists is worse than no file: it looks like "
            "a successful run to everything downstream.")
    _write_jsonl(paths["dataset"], result["rows"])
    _write_jsonl(paths["rejected"], result["rejections"])
    _write_jsonl(paths["dpo_negatives"], result["negatives"])
    paths["report"].write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return paths


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, required=True, help="examples to attempt")
    ap.add_argument("--out", required=True, help="dataset JSONL path")
    ap.add_argument("--teacher", default=None,
                    help="model spec; default resolves from configs/models.json")
    ap.add_argument("--attempts", type=int, default=4,
                    help="teacher attempts per example before giving up")
    ap.add_argument("--min-kept", type=int, default=None,
                    help=f"minimum kept rows; default ceil({DEFAULT_KEEP_FRACTION:.2f} * n)")
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-cache", action="store_true",
                    help="bypass datagen/.cache (this re-bills the teacher)")
    args = ap.parse_args(argv)

    if args.n <= 0:
        print(f"--n must be positive, got {args.n}", file=sys.stderr)
        return 2
    min_kept = (args.min_kept if args.min_kept is not None
                else math.ceil(DEFAULT_KEEP_FRACTION * args.n))

    instances = load_instances()
    corpus = list(load_corpus()) + list(load_smoke())
    assert_disjoint_from_corpus(corpus)
    forbidden = eval_pair_keys(corpus)

    teacher, reason = resolve_teacher(args.teacher)
    print(f"teacher: {teacher}  ({reason})")
    print(f"held-out types never sampled: "
          f"{', '.join(registry.types_by_split('heldout_type'))}")

    samples = build_scenarios(args.n, instances, corpus, seed=args.seed)
    print(f"sampled {len(samples)} scenarios from {len({s.task_id for s in samples})} "
          f"fresh tasks")

    import eval as E
    generator = E.make_generator(teacher)
    cache = TeacherCache(CACHE_DIR, enabled=not args.no_cache)

    result = generate(samples, generator, teacher, reason, instances, forbidden,
                      attempts=args.attempts, cache=cache, seed=args.seed,
                      workers=args.workers)

    report = summarise(result["rows"], result["rejections"], result["negatives"],
                       result["give_ups"], instances, args.n)
    report.update({"teacher": teacher, "teacher_resolved_by": reason,
                   "attempts": args.attempts, "seed": args.seed,
                   "min_kept": min_kept, "cache": result["cache"],
                   "generated_utc": datetime.now(timezone.utc).isoformat()})

    if not result["rows"]:
        print("\nZERO rows kept. No dataset written.", file=sys.stderr)
        print(f"rejection reasons: {report['rejection_reasons']}", file=sys.stderr)
        return 3

    paths = write_outputs(Path(args.out), result, report)

    print("\n--- realised distribution -------------------------------------")
    print(f"kept {report['kept']}/{report['requested']} "
          f"(accept rate {report['accept_rate']:.1%}), gave up on {report['gave_up']}")
    print(f"k: {report['k_distribution']}   pressure: {report['pressure_distribution']}")
    print(f"type shares: {report['type_shares']}  (max {report['max_type_share']:.1%})")
    print(f"attempts: {report['attempt_histogram']}")
    print(f"rejections: {report['rejections']} {report['rejection_reasons']}")
    print(f"candidate DPO negatives: {report['dpo_negatives']}")
    print(f"cache: {report['cache']}")
    for name, path in paths.items():
        print(f"  {name}: {path}")

    if report["kept"] < min_kept:
        print(f"\nSHORTFALL: kept {report['kept']} < --min-kept {min_kept}",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
