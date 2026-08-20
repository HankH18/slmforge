#!/usr/bin/env python3
"""Build the few-shot exemplars for the ablation's few_shot strategy.

Every exemplar is a (rendered prompt, compliant response) pair. "Compliant" is
not asserted, it is PROVEN: the teacher's draft is run through the same
constraints.checkers.check the eval scores with, and a draft that violates
anything is retried, never shipped. An exemplar that violated its own
constraints would teach the failure it is meant to prevent.

Two deliberate restrictions:

* Exemplar tasks are FRESH -- none appears in data/scenarios/corpus.jsonl. A
  worked example of a scenario the model is then graded on is leakage.
* Exemplar constraints use TRAIN-SPLIT TYPES ONLY. The held-out types
  (banned_punctuation, sentence_initial_letter) exist to measure generalization
  to constraint types never seen; demonstrating one in-context would hand the
  model exactly the help that measurement is designed to withhold.

Usage:
  set -a && . ./.env && set +a
  python -m scenarios.make_exemplars --teacher openai:gpt-5.6-terra
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from constraints import registry                      # noqa: E402
from harness import build_constraint_lookup, run_checkers  # noqa: E402
from scenarios.paths import load_corpus, load_instances    # noqa: E402
from scenarios.render import render_user_message           # noqa: E402

OUT_PATH = REPO_ROOT / "data" / "scenarios" / "fewshot_exemplars.jsonl"

# Fresh tasks, deliberately unlike the corpus in subject matter. k spans 2-4 to
# match the eval distribution, and every set carries >=1 negative constraint.
EXEMPLAR_SPECS: List[Dict[str, Any]] = [
    {"id": "fx-ex-01", "k": 2,
     "task": "Write a short paragraph explaining to a new volunteer how the tool "
             "library lends out power drills and what happens if one comes back damaged."},
    {"id": "fx-ex-02", "k": 3,
     "task": "Draft a brief notice for a climbing gym telling members that the "
             "bouldering wall will be resurfaced next month and the lead wall stays open."},
    {"id": "fx-ex-03", "k": 2,
     "task": "Write a compact summary of why a small bakery might switch from bagged "
             "flour to bulk delivery, aimed at the owner rather than an accountant."},
    {"id": "fx-ex-04", "k": 4,
     "task": "Explain, for a community newsletter, how the new kerbside compost "
             "collection works and what residents should put in the caddy."},
    {"id": "fx-ex-05", "k": 3,
     "task": "Write a paragraph for a museum label describing a restored printing "
             "press and what visitors can watch it do on demonstration days."},
    {"id": "fx-ex-06", "k": 2,
     "task": "Draft a short note to a running club explaining that Tuesday sessions "
             "move to the athletics track while the towpath is being repaired."},
]


def pick_constraints(instances: Dict[str, Dict[str, Any]], k: int, seed: int) -> List[str]:
    """Deterministic spread over TRAIN-split instances only."""
    train = sorted(cid for cid, c in instances.items()
                   if registry.split_of(c["type"]) == "train")
    if len(train) < k:
        raise SystemExit(f"only {len(train)} train-split instances; need {k}")
    picked, seen_types = [], set()
    i = seed
    while len(picked) < k and i < seed + 10 * len(train):
        cid = train[i % len(train)]
        ctype = instances[cid]["type"]
        if ctype not in seen_types:          # one instance per type: k distinct demands
            picked.append(cid)
            seen_types.add(ctype)
        i += 1
    if len(picked) < k:
        raise SystemExit(f"could not find {k} distinct train-split types from seed {seed}")
    return picked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="openai:gpt-5.6-terra")
    ap.add_argument("--attempts", type=int, default=5)
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    import eval as E  # the live harness, for its generator plumbing only

    instances = load_instances()
    lookup = build_constraint_lookup(instances.values())
    corpus_tasks = {s["task"].strip() for s in load_corpus()}

    gen = E.make_generator(args.teacher)
    rows, failed = [], []

    for n, spec in enumerate(EXEMPLAR_SPECS):
        if spec["task"].strip() in corpus_tasks:
            raise SystemExit(f"{spec['id']}: task also appears in the corpus — leakage")
        scenario = {
            "scenario_id": spec["id"], "task": spec["task"],
            "constraints": pick_constraints(instances, spec["k"], seed=n * 3),
            "k": spec["k"], "pressure": "clean", "split": "train",
        }
        rendered = render_user_message(scenario, lookup)
        prompt = E.build_prompt("zero_shot", scenario, lookup, None)

        for attempt in range(1, args.attempts + 1):
            out = gen(scenario, prompt, lookup)
            results = run_checkers(scenario, lookup, out)
            bad = [r for r in results if not r["pass"]]
            if not bad:
                rows.append({
                    "exemplar_id": spec["id"],
                    "rendered_prompt": rendered,
                    "target_output": out.strip(),
                    "constraints": scenario["constraints"],
                    "k": spec["k"],
                    "teacher": args.teacher,
                    "attempts": attempt,
                    "verified_by": "constraints.checkers.check",
                })
                print(f"  {spec['id']}  k={spec['k']}  attempt {attempt}  COMPLIANT")
                break
            print(f"  {spec['id']}  attempt {attempt} violated: "
                  f"{[(r['constraint_id'], r['evidence'][:40]) for r in bad]}")
        else:
            failed.append(spec["id"])

    if failed:
        print(f"\nFAILED to get a compliant exemplar for: {failed}")
        print("Shipping fewer exemplars rather than an exemplar that breaks its own "
              "constraints. Rerun, raise --attempts, or simplify those constraint sets.")

    if not rows:
        print("no compliant exemplars produced — refusing to write an empty file")
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # Re-verify from disk: what ships is what was checked.
    reread = [json.loads(l) for l in open(out_path) if l.strip()]
    for r in reread:
        scen = {"scenario_id": r["exemplar_id"], "task": "x",
                "constraints": r["constraints"], "k": r["k"],
                "pressure": "clean", "split": "train"}
        res = run_checkers(scen, lookup, r["target_output"])
        assert all(x["pass"] for x in res), f"{r['exemplar_id']} fails on re-read"

    print(f"\nwrote {len(rows)} verified exemplars -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
