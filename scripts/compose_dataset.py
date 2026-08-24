#!/usr/bin/env python3
"""Compose v4 = v3 + the attack block, and the drafts file the block is built from.

Two subcommands, because the v4 corpus is built in two steps and each one has a
property that has to be PROVED rather than assumed.

``drafts``
    Three subagent teachers each drafted 60 completions for one attack shape,
    and each of them numbered its rows from ``atk-a-00001`` because
    :func:`datagen.attacks.build_attacks` restarts the ordinal at 1 for whatever
    subset of shapes it is asked for. Concatenating the three files therefore
    produces THREE rows called ``atk-a-00001`` and
    :class:`datagen.generate.DraftFile` refuses the file outright ("two records
    for scenario ...; which one is the teacher's answer is then undecidable").

    The fix is not an offset table copied out of somebody's notes. Every draft
    record carries the full ``scenario`` object it was written against, so this
    rekeys by CONTENT: build the pool at the same seed with all three shapes,
    index the result by ``(task, constraint ids)``, and look each draft up. A
    draft whose scenario is not in that build is a hard error -- it would mean
    the prose was written against constraints the pipeline will not apply, and
    the row would ship as a violation or a lie about its own provenance.

    The measured mapping at pool A / seed 20260824 is a constant per-shape
    offset (format +0, subject +140, erosion +735), which is what you would
    expect from an ordinal counter that runs across shapes in
    :data:`datagen.attacks.ATTACK_SHAPES` order. That is an OUTPUT of this
    script, printed and asserted to be constant per shape, not an input to it.

``compose``
    ``data/train/v4.jsonl = data/train/v3.jsonl (verbatim) + the attack block``.

    The data-efficiency curve is only interpretable if the larger corpus
    CONTAINS the smaller one: otherwise two points differ by which rows they
    drew as well as by how many, and the difference cannot be attributed to
    size. So v3 is copied through byte-for-byte, in order, and the containment
    is re-checked from the written file before this exits -- the same standard
    ``tests/test_shipped_dataset.py::test_v3_contains_every_v2_row`` holds the
    previous rung to.

Neither subcommand makes a network call and neither writes outside
``data/train/``.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datagen import attacks  # noqa: E402

TRAIN_DIR = REPO_ROOT / "data" / "train"

#: The per-shape draft files the three teacher subagents wrote, in
#: :data:`datagen.attacks.ATTACK_SHAPES` order.
SHAPE_DRAFTS: Dict[str, Path] = {
    "format_compulsion": TRAIN_DIR / "v4_format_compulsion.draft.jsonl",
    "subject_compulsion": TRAIN_DIR / "v4_subject_compulsion.draft.jsonl",
    "erosion_recap": TRAIN_DIR / "v4_erosion_recap.draft.jsonl",
}

DEFAULT_MERGED_DRAFTS = TRAIN_DIR / "v4_attacks.draft.jsonl"


class ComposeError(RuntimeError):
    """Refusing to write a corpus whose stated property does not hold."""


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ComposeError(f"{path}:{number} is not valid JSON: {exc}")
    if not rows:
        raise ComposeError(f"{path} holds no rows")
    return rows


def scenario_index(pool: str, seed: int) -> Dict[Tuple[str, Tuple[str, ...]], Tuple[str, str]]:
    """``(task, constraint ids) -> (scenario_id, shape)`` for an ALL-SHAPES build.

    All shapes, always: the ids a merged drafts file needs are the ids the
    pipeline will assign when it is asked for every shape at once, and those are
    the only ids under which the three blocks do not collide.
    """
    lib = attacks.instance_library()
    rows, quarantined = attacks.build_attacks(
        pool=pool, seed=seed, shapes=attacks.ATTACK_SHAPES, lib=lib)
    index: Dict[Tuple[str, Tuple[str, ...]], Tuple[str, str]] = {}
    for row in rows:
        scen = row.scenario
        key = (scen["task"], tuple(scen["constraints"]))
        if key in index:
            raise ComposeError(
                f"the pool built two scenarios with the same (task, constraint "
                f"set): {index[key][0]} and {scen['scenario_id']}. Rekeying by "
                f"content is then ambiguous.")
        index[key] = (scen["scenario_id"], row.shape)
    print(f"built {len(rows)} attack scenarios from pool {pool} at seed {seed} "
          f"({len(quarantined)} quarantined); {len(index)} distinct (task, set) keys")
    return index


def build_merged_drafts(pool: str, seed: int) -> List[Dict[str, Any]]:
    """The three per-shape draft files, rekeyed to all-shapes scenario ids."""
    index = scenario_index(pool, seed)
    merged: List[Dict[str, Any]] = []
    offsets: Dict[str, set] = defaultdict(set)
    for shape, path in SHAPE_DRAFTS.items():
        drafts = read_jsonl(path)
        for record in drafts:
            scen = record.get("scenario")
            if not isinstance(scen, Mapping) or "task" not in scen \
                    or "constraints" not in scen:
                raise ComposeError(
                    f"{path}: draft {record.get('scenario_id')!r} carries no "
                    f"'scenario' object, so there is nothing to rekey it BY. A "
                    f"bare ordinal is not evidence about which prompt the prose "
                    f"was written for.")
            key = (scen["task"], tuple(scen["constraints"]))
            hit = index.get(key)
            if hit is None:
                raise ComposeError(
                    f"{path}: draft {record.get('scenario_id')!r} was written "
                    f"against a (task, constraint set) that pool {pool} at seed "
                    f"{seed} does not build. The prose would be graded against "
                    f"constraints it was never written for.")
            scenario_id, built_shape = hit
            if built_shape != shape:
                raise ComposeError(
                    f"{path}: draft {record.get('scenario_id')!r} resolves to a "
                    f"{built_shape} row, not {shape}")
            old = str(record["scenario_id"])
            offsets[shape].add(int(scenario_id.rsplit("-", 1)[1])
                               - int(old.rsplit("-", 1)[1]))
            out = dict(record)
            out["scenario_id"] = scenario_id
            out["scenario"] = dict(scen, scenario_id=scenario_id)
            out["shape"] = shape
            out["source_file"] = path.name
            out["source_scenario_id"] = old
            merged.append(out)
        print(f"  {shape}: {len(drafts)} drafts from {path.name}, "
              f"id offset(s) {sorted(offsets[shape])}")
    for shape, seen in offsets.items():
        if len(seen) != 1:
            raise ComposeError(
                f"{shape} drafts do not share a single ordinal offset ({sorted(seen)}); "
                f"the per-shape blocks are not contiguous and the rekey is suspect")
    ids = [record["scenario_id"] for record in merged]
    duplicates = [sid for sid, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise ComposeError(
            f"{len(duplicates)} scenario ids appear twice after rekeying "
            f"({duplicates[:5]}); DraftFile would refuse this file")
    merged.sort(key=lambda record: record["scenario_id"])
    return merged


def check_covers_the_block(merged: Sequence[Mapping[str, Any]], n: int,
                           pool: str, seed: int) -> None:
    """Every scenario the sampler will pick at ``--n n`` has a draft here.

    Without this the run still "succeeds": the pipeline records a
    ``teacher_error`` for each scenario it cannot find a draft for and ships a
    shorter block than the report's ``requested`` field claims was attempted.
    """
    from scenarios.paths import load_corpus, load_instances, load_smoke
    from datagen.sampling import build_attack_samples

    instances = load_instances()
    corpus = list(load_corpus()) + list(load_smoke())
    samples = build_attack_samples(n, instances, corpus, seed=seed, pool=pool)
    wanted = [s.scenario["scenario_id"] for s in samples]
    have = {record["scenario_id"] for record in merged}
    missing = [sid for sid in wanted if sid not in have]
    if missing:
        raise ComposeError(
            f"the sampler picks {len(wanted)} scenarios at --n {n} but "
            f"{len(missing)} of them have no draft ({missing[:5]}). The pipeline "
            f"would record a teacher_error for each and ship a short block.")
    extra = sorted(have - set(wanted))
    print(f"the sampler's {len(wanted)} scenarios at --n {n} are all drafted "
          f"({len(extra)} drafts unused)")


def cmd_drafts(args: argparse.Namespace) -> int:
    merged = build_merged_drafts(args.pool, args.seed)
    check_covers_the_block(merged, args.n, args.pool, args.seed)
    out = Path(args.out)
    out.write_text("".join(json.dumps(record) + "\n" for record in merged))
    print(f"wrote {len(merged)} rekeyed drafts to {out}")
    return 0


# ------------------------------------------------------------------ compose
def row_key(row: Mapping[str, Any]) -> str:
    """Content identity of a training row, ignoring the wall-clock stamp.

    The same identity ``tests/test_shipped_dataset.py`` uses, so "v4 contains
    v3" means here exactly what the test will mean by it.
    """
    import copy
    row = copy.deepcopy(dict(row))
    provenance = row.get("provenance")
    if isinstance(provenance, dict):
        provenance.pop("generated_utc", None)
    return json.dumps(row, sort_keys=True)


def cmd_compose(args: argparse.Namespace) -> int:
    base_path, block_path, out = Path(args.base), Path(args.block), Path(args.out)
    base_lines = [l for l in base_path.read_text().splitlines() if l.strip()]
    block_lines = [l for l in block_path.read_text().splitlines() if l.strip()]
    if not base_lines or not block_lines:
        raise ComposeError(f"{base_path} or {block_path} is empty")

    # Byte-for-byte passthrough of the base, in order, then the block.
    out.write_text("\n".join(base_lines + block_lines) + "\n")

    written = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    if len(written) != len(base_lines) + len(block_lines):
        raise ComposeError(
            f"{out} has {len(written)} rows, expected "
            f"{len(base_lines) + len(block_lines)}")

    # Re-read BOTH files from disk and check containment on the written artefact,
    # never on the in-memory lists that produced it.
    base_keys = {row_key(json.loads(l)) for l in base_lines}
    out_keys = {row_key(row) for row in written}
    missing = base_keys - out_keys
    if missing:
        raise ComposeError(
            f"{len(missing)} of {len(base_keys)} {base_path.name} rows are absent "
            f"from {out.name}; the curve's nested family is broken")
    if len(out_keys) <= len(base_keys):
        raise ComposeError(
            f"{out.name} is not strictly larger than {base_path.name}")
    print(f"{out}: {len(written)} rows = {len(base_lines)} from {base_path.name} "
          f"+ {len(block_lines)} from {block_path.name}")
    print(f"containment re-checked from the written file: all {len(base_keys)} "
          f"distinct {base_path.name} rows present")

    # The composition report is DERIVED from the written corpus, never from the
    # arguments -- the same standard datagen/audit.py holds itself to. It reads
    # the block's own pipeline report only for the numbers the pipeline alone
    # knows (rejections, evasive drafts), and says which file each came from.
    from datagen.audit import audit
    result = audit(str(out), min_rows=1)
    block_report_path = block_path.with_suffix("").with_suffix(".report.json")
    block_report: Dict[str, Any] = {}
    if block_report_path.exists():
        block_report = json.loads(block_report_path.read_text())
    teachers = Counter(row.get("provenance", {}).get("teacher") for row in written)
    shapes = Counter(row.get("provenance", {}).get("attack", {}).get("shape")
                     for row in written
                     if isinstance(row.get("provenance", {}).get("attack"), dict))
    report = {
        "corpus": out.name,
        "rows": len(written),
        "composed_from": {base_path.name: len(base_lines),
                          block_path.name: len(block_lines)},
        "contains_every_row_of": {
            base_path.name: {"distinct_rows": len(base_keys), "all_present": True}},
        "teachers": dict(teachers),
        "attack_shapes": dict(shapes),
        "audit": {"violations": list(result["violations"]),
                  "passed": not result["violations"],
                  "stats": result.get("stats", {})},
        # Pre-registered in docs/RETRAIN_PREREGISTRATION.md sec.4 (falsifier F5):
        # the count of drafts that passed every checker by DECLINING the job.
        "evasive_drafts": block_report.get("evasive_drafts"),
        "evasion_rules": block_report.get("evasion_rules"),
        "block_report": block_report_path.name if block_report else None,
    }
    report_path = out.with_suffix("").with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"audit of {out.name}: "
          f"{'PASS' if not result['violations'] else result['violations']}")
    print(f"evasive drafts rejected while building {block_path.name}: "
          f"{report['evasive_drafts']}")
    print(f"wrote {report_path}")
    if result["violations"]:
        raise ComposeError(
            f"{out.name} fails its own audit: {list(result['violations'])}. The "
            f"corpus is written but it is NOT clean; fix the data.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    d = sub.add_parser("drafts", help="rekey the per-shape draft files into one")
    d.add_argument("--pool", default="A")
    d.add_argument("--seed", type=int, default=20260824)
    d.add_argument("--n", type=int, default=180,
                   help="the --n the block will be generated at; every scenario "
                        "the sampler picks at this n must have a draft")
    d.add_argument("--out", default=str(DEFAULT_MERGED_DRAFTS))
    d.set_defaults(func=cmd_drafts)

    c = sub.add_parser("compose", help="base + block -> corpus")
    c.add_argument("--base", default=str(TRAIN_DIR / "v3.jsonl"))
    c.add_argument("--block", default=str(TRAIN_DIR / "v4_attacks.jsonl"))
    c.add_argument("--out", default=str(TRAIN_DIR / "v4.jsonl"))
    c.set_defaults(func=cmd_compose)

    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except ComposeError as exc:
        print(f"REFUSING TO WRITE: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
