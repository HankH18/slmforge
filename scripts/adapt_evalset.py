#!/usr/bin/env python3
"""Make a foreign scenario set scoreable, and publish exactly what was changed.

The staff held-out set arrives with its own constraint library, and this harness
refuses it: constraint ids are pinned to ``^c-(?:lex|chr|str|fmt)-\\d+$``, each
constraint's ``family`` and ``split`` must equal the registry's pin for its type,
and any unrecognised key raises. A file that is semantically perfectly gradeable
therefore dies at scoring time with ``ConstraintError``.

Refusing to grade it would be wrong; quietly grading the subset that happens to
fit would be worse. So this rewrites the *metadata* the registry pins and leaves
the *semantics* untouched:

* ``type`` and ``params`` are copied byte for byte. They are the entire
  behavioural content of a constraint -- what the checker reads.
* ``id`` is replaced with a registry-shaped one, ``family`` and ``split`` are set
  to the registry's own values for that type, and any other key is dropped.
* Every rename and every dropped key is recorded in a mapping file, so the
  transformation is auditable rather than asserted, and the numbers can be
  reported alongside it.

It **refuses** rather than adapts when the meaning would change:

* a constraint type this project does not implement -- there is no checker for
  it, and scoring the rest while dropping it would report a number for a
  different eval set than the one supplied;
* two constraints that differ only in a dropped key, which would silently
  collapse into one.

    python scripts/adapt_evalset.py --scenarios staff.jsonl \\
        --constraints staff_constraints.jsonl --out-dir runs/staff_adapted
    python scripts/preflight_evalset.py runs/staff_adapted/scenarios.jsonl \\
        runs/staff_adapted/constraint_instances.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from constraints import registry  # noqa: E402

#: Keys a constraint instance may carry. Anything else is dropped and recorded.
KEPT_KEYS = ("id", "type", "params", "family", "split")


class AdaptError(Exception):
    """A set that cannot be adapted without changing what it measures."""


def read_jsonl(path: Path) -> List[Mapping[str, Any]]:
    if not path.is_file():
        raise AdaptError(f"{path} does not exist")
    rows = []
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise AdaptError(f"{path}:{lineno}: not valid JSON ({exc})") from exc
    return rows


def adapt_constraints(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Rewrite ids/family/split; keep type and params exactly."""
    known = set(registry.list_types())
    unknown = sorted({r.get("type") for r in rows if r.get("type") not in known})
    if unknown:
        raise AdaptError(
            f"constraint type(s) {unknown} are not implemented by this harness "
            f"(it implements {sorted(known)}). There is no checker for them, so "
            f"the set cannot be scored. Dropping them and scoring the rest would "
            f"report a number for a different eval set than the one supplied -- "
            f"say so instead.")

    seq: Counter = Counter()
    mapping: Dict[str, str] = {}
    dropped: Dict[str, List[str]] = {}
    out: List[Dict[str, Any]] = []
    semantic_seen: Dict[str, str] = {}

    for row in rows:
        old_id = row.get("id")
        if not isinstance(old_id, str) or not old_id:
            raise AdaptError(f"a constraint has no usable 'id': {row!r}")
        if old_id in mapping:
            raise AdaptError(f"duplicate constraint id {old_id!r} in the source")

        ctype = row["type"]
        family = registry.family_of(ctype)
        # Already the full leading segment, e.g. "c-lex-" -- not the bare "lex".
        prefix = registry.ID_PREFIX_BY_FAMILY[family]
        seq[prefix] += 1
        new_id = f"{prefix}{seq[prefix]:03d}"

        # Two source constraints that differ ONLY in a key we are about to drop
        # would become the same constraint. That is a silent merge, and it
        # re-weights every rate the pair appears in.
        semantic = json.dumps({"type": ctype, "params": row.get("params")},
                              sort_keys=True)
        if semantic in semantic_seen:
            raise AdaptError(
                f"{old_id!r} and {semantic_seen[semantic]!r} are identical once "
                f"the non-registry keys are dropped. Adapting them would merge "
                f"two distinct constraints into one and change what the set "
                f"measures.")
        semantic_seen[semantic] = old_id

        extra = sorted(k for k in row if k not in KEPT_KEYS)
        if extra:
            dropped[old_id] = extra

        mapping[old_id] = new_id
        out.append({"id": new_id, "type": ctype, "params": row.get("params"),
                    "family": family, "split": registry.split_of(ctype)})

    return {"constraints": out, "mapping": mapping, "dropped_keys": dropped}


def adapt_scenarios(rows: Sequence[Mapping[str, Any]],
                    mapping: Mapping[str, str]) -> Dict[str, Any]:
    """Rewrite each scenario's constraint ids. Everything else is untouched."""
    out: List[Dict[str, Any]] = []
    missing_fields: Counter = Counter()
    for i, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise AdaptError(f"scenario row {i} is not a JSON object")
        ids = row.get("constraints")
        if not isinstance(ids, list) or not ids:
            raise AdaptError(f"scenario row {i} has no usable 'constraints' list")
        unknown = [c for c in ids if c not in mapping]
        if unknown:
            raise AdaptError(
                f"scenario row {i} references constraint id(s) {unknown} that are "
                f"not in the supplied constraints file")
        new = dict(row)
        new["constraints"] = [mapping[c] for c in ids]
        for field in ("pressure", "split", "k"):
            if field not in row:
                missing_fields[field] += 1
        out.append(new)
    return {"scenarios": out, "missing_fields": dict(missing_fields)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenarios", required=True)
    ap.add_argument("--constraints", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)

    try:
        cons = adapt_constraints(read_jsonl(Path(args.constraints)))
        scen = adapt_scenarios(read_jsonl(Path(args.scenarios)), cons["mapping"])
    except AdaptError as exc:
        print(f"FAIL adapt_evalset: {exc}", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "constraint_instances.jsonl").write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cons["constraints"]))
    (out_dir / "scenarios.jsonl").write_text(
        "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in scen["scenarios"]))
    (out_dir / "adaptation.json").write_text(json.dumps({
        "_note": ("Metadata-only adaptation. 'type' and 'params' -- the entire "
                  "behavioural content of a constraint -- were copied unchanged; "
                  "only the registry-pinned id, family and split were rewritten. "
                  "Publish this file with any numbers produced from the adapted "
                  "set."),
        "source_scenarios": args.scenarios,
        "source_constraints": args.constraints,
        "id_mapping": cons["mapping"],
        "dropped_keys": cons["dropped_keys"],
        "scenarios_missing_fields": scen["missing_fields"],
    }, indent=2, ensure_ascii=False) + "\n")

    print(f"adapted {len(cons['constraints'])} constraint(s) and "
          f"{len(scen['scenarios'])} scenario(s) into {out_dir}")
    if cons["dropped_keys"]:
        print(f"  note {len(cons['dropped_keys'])} constraint(s) carried keys the "
              f"registry does not accept; they are recorded in adaptation.json")
    if scen["missing_fields"]:
        print(f"  WARNING scenarios are missing {scen['missing_fields']} -- the "
              f"harness needs pressure, split and k on every row and this script "
              f"will not invent them. Run scripts/preflight_evalset.py next.",
              file=sys.stderr)
    print(f"\nnext: python scripts/preflight_evalset.py "
          f"{out_dir / 'scenarios.jsonl'} {out_dir / 'constraint_instances.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
