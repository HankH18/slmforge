#!/usr/bin/env python3
"""Deterministic NESTED subsets of a training set, for the data-efficiency curve.

The curve asks one question: how much of this dataset do you actually need? A
point on it is only interpretable if the only thing that changed is SIZE.

Two properties make that true, and this script exists to guarantee both rather
than leave them to whoever runs the sweep:

NESTED. The rows of the N/8 subset are a subset of N/4, which is a subset of
N/2, which is a subset of N. One seeded shuffle is taken and each subset is a
PREFIX of it. Drawing each point independently would let a small point win or
lose on which rows it happened to get, and the resulting wiggle would be read as
a property of dataset size.

DISTRIBUTION-PRESERVING. A prefix of a shuffle keeps every composition of the
parent in expectation -- constraint types, k, pressure, and (the one that
matters most here) which letter a banned_letter row bans. A subset that happened
to drop the hard letters would post a flattering number for a small N and
misplace the minimum viable size, which is the single conclusion this curve is
for. The realised composition of every subset is printed and written to the
manifest so the claim is auditable rather than asserted.

    python scripts/make_subset.py data/train/v2.jsonl --sizes 110 220 440 \\
        --out-dir data/train/subsets

Writes `<out-dir>/<stem>.n<N>.jsonl` per size plus one `<stem>.subsets.json`
manifest recording each subset's sha256 and composition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_SEED = 20260821


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def letter_of(row, instances):
    """The banned letter this row carries, or None. Rows reference constraints by
    id; the letter lives in the instance library, not in the row."""
    for cid in row.get("constraints", []):
        inst = instances.get(cid) if isinstance(cid, str) else None
        if inst and inst.get("type") == "banned_letter":
            return inst["params"]["letter"]
    return None


def composition(rows, instances):
    """What a subset is made of -- the numbers a reader needs to believe the
    subset is a fair miniature of its parent."""
    types = Counter()
    for row in rows:
        for cid in row.get("constraints", []):
            inst = instances.get(cid) if isinstance(cid, str) else None
            if inst:
                types[inst["type"]] += 1
    letters = Counter(
        L for L in (letter_of(r, instances) for r in rows) if L is not None)
    total_slots = sum(types.values()) or 1
    return {
        "rows": len(rows),
        "type_counts": dict(sorted(types.items())),
        "type_shares": {t: round(c / total_slots, 4) for t, c in sorted(types.items())},
        "banned_letter_rows": sum(letters.values()),
        "letter_counts": dict(sorted(letters.items(), key=lambda kv: -kv[1])),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dataset", help="training JSONL to subset")
    ap.add_argument("--sizes", type=int, nargs="+", required=True,
                    help="subset sizes, e.g. --sizes 110 220 440")
    ap.add_argument("--out-dir", default=None,
                    help="directory for subsets (default: alongside the dataset)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--instances", default="data/scenarios/constraint_instances.jsonl")
    args = ap.parse_args(argv)

    src = Path(args.dataset)
    if not src.exists():
        print(f"dataset not found: {src}", file=sys.stderr)
        return 2
    rows = read_jsonl(src)
    instances = {r["id"]: r for r in read_jsonl(Path(args.instances))}

    sizes = sorted(set(args.sizes))
    too_big = [n for n in sizes if n > len(rows)]
    if too_big:
        print(f"cannot take subsets {too_big} from a {len(rows)}-row dataset",
              file=sys.stderr)
        return 2
    if any(n <= 0 for n in sizes):
        print("subset sizes must be positive", file=sys.stderr)
        return 2

    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)

    out_dir = Path(args.out_dir) if args.out_dir else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = src.name[: -len(".jsonl")] if src.name.endswith(".jsonl") else src.name

    manifest = {"source": str(src), "source_rows": len(rows), "seed": args.seed,
                "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
                "nested": True, "subsets": {}}
    print(f"{src} -> {len(rows)} rows, seed {args.seed}")
    parent_ids = None
    for n in sizes:
        picked = [rows[i] for i in order[:n]]
        out = out_dir / f"{stem}.n{n}.jsonl"
        payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in picked)
        out.write_text(payload)
        comp = composition(picked, instances)
        comp["path"] = str(out)
        comp["sha256"] = hashlib.sha256(payload.encode()).hexdigest()
        manifest["subsets"][str(n)] = comp

        # Nesting is the property the curve rests on, so verify it here rather
        # than trust that a prefix of one shuffle stayed a prefix.
        ids = {id(r) for r in picked}
        if parent_ids is not None and not parent_ids <= ids:
            raise AssertionError(f"subset n={n} does not contain its smaller sibling")
        parent_ids = ids

        print(f"  n={n:5d}  {out.name}  letters={comp['letter_counts']}")
    (out_dir / f"{stem}.subsets.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(sizes)} subsets + {stem}.subsets.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
