#!/usr/bin/env python3
"""Emit the scenario artefacts under ``data/scenarios/``.

    .venv/bin/python -m scenarios.build            # write
    .venv/bin/python -m scenarios.build --check     # verify on-disk == regenerated

Deterministic: no RNG anywhere in the pipeline, so a rebuild that changes a
byte means an input changed. ``--check`` is what makes that useful -- it fails
if the shipped JSONL has drifted from the authored source, which is how a
hand-edited corpus row gets caught.

Files written:

``constraint_instances.jsonl``  the instance library (Deliverable 1)
``corpus.jsonl``                the scenario corpus (Deliverable 2)
``smoke.jsonl``                 5 corpus scenarios for T-3's end-to-end smoke
                                test (Deliverable 3) -- a strict SUBSET of the
                                corpus, same ids, so it needs no separate
                                instance file and cannot drift from it
``manifest.json``               per-scenario provenance: which distractor
                                mechanism fires with what evidence, which
                                embedded-pressure sentence was appended, and
                                the reachability witness the checkers passed
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

from scenarios.generator import generate, summarise
from scenarios.instances import INSTANCE_META, build_instances
from scenarios.paths import (CORPUS_PATH, DATA_DIR, INSTANCES_PATH, MANIFEST_PATH,
                             SMOKE_PATH)

#: The 5 smoke scenarios, named by task id so the choice is explicit rather
#: than "the first five". Mixed k (2, 3, 4, 3, 4), mixed pressure, both splits.
SMOKE_TASK_IDS = (
    "a04-one-pan-sauce",        # train, adversarial, k=2
    "a06-three-am-note",        # train, adversarial, k=3
    "c12-first-chapter",        # train, clean,       k=4
    "e09-towpath-morning",      # heldout_type, clean,       k=3
    "d17-harvest-supper",       # heldout_type, adversarial, k=4
)


def _dumps(rows: Sequence[Dict[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n"
                   for row in rows)


def render() -> Dict[Path, str]:
    """Everything that belongs on disk, as ``path -> text``. Pure."""
    scenarios, manifests = generate()
    by_task = {m["task_key"]: m["scenario_id"] for m in manifests}
    missing = [t for t in SMOKE_TASK_IDS if t not in by_task]
    if missing:
        raise SystemExit(f"smoke set names unknown tasks: {missing}")
    smoke_ids = {by_task[t] for t in SMOKE_TASK_IDS}
    smoke = [s for s in scenarios if s["scenario_id"] in smoke_ids]

    manifest = {
        "generator": "scenarios.build",
        "summary": summarise(scenarios),
        "smoke_scenarios": sorted(smoke_ids),
        "instance_notes": {cid: {"lexicon": meta.lexicon, "note": meta.note}
                           for cid, meta in INSTANCE_META.items()},
        "scenarios": manifests,
    }
    return {
        INSTANCES_PATH: _dumps(build_instances()),
        CORPUS_PATH: _dumps(scenarios),
        SMOKE_PATH: _dumps(smoke),
        MANIFEST_PATH: json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="fail if the files on disk differ from a fresh build")
    args = parser.parse_args(argv)

    rendered = render()
    if args.check:
        drifted = []
        for path, text in rendered.items():
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                drifted.append(str(path))
        if drifted:
            print("DRIFT: on-disk artefacts differ from a fresh build:", file=sys.stderr)
            for path in drifted:
                print(f"  {path}", file=sys.stderr)
            return 1
        print("ok: on-disk artefacts match a fresh build")
        return 0

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for path, text in rendered.items():
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path}")
    scenarios, _ = generate()
    print(json.dumps(summarise(scenarios), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
