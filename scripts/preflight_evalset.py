#!/usr/bin/env python3
"""Check a scenario file the harness has never seen — before spending anything.

The staff held-out eval set is by definition a file this project did not write.
`eval.py` now refuses an unscoreable set up front (`harness.validate_eval_set`),
but "run the real command and see if it complains" still loads a 1.7B model
first. This does the same validation with no model, no network and no cost, and
then prints what the set actually contains, so a file can be inspected before it
is trusted.

    python scripts/preflight_evalset.py <scenarios.jsonl> [constraints.jsonl]

Exit 0 = `eval.py` will accept this file. Exit 1 = it will not, and every reason
is listed.

Known limits, worth reading before the harness meets an unfamiliar file:

* Constraint ids are regex-pinned to `^c-(?:lex|chr|str|fmt)-\\d+$` and every
  constraint's `family` and `split` must match this project's registry
  (`constraints/registry.py`). A staff file bringing its own constraint library
  under different ids cannot be scored as-is; the ids have to be mapped first.
* Only the seven registered constraint types can be checked. There is no
  extension point, and a set introducing an eighth is not gradeable by this
  harness — a fact to disclose, not to paper over.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from constraints.checkers import check  # noqa: E402
from harness import HarnessError, build_constraint_lookup, validate_eval_set  # noqa: E402
from scenarios.render import render_user_message  # noqa: E402

DEFAULT_CONSTRAINTS = REPO_ROOT / "data" / "scenarios" / "constraint_instances.jsonl"


def read_jsonl(path: Path) -> List[Mapping[str, Any]]:
    rows = []
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"FAIL {path}:{lineno}: not valid JSON ({exc})")
    return rows


def describe(scenarios: Sequence[Mapping[str, Any]],
             lookup: Mapping[str, Mapping[str, Any]]) -> None:
    print(f"  {len(scenarios)} scenarios, {len(lookup)} constraint instances")
    for field in ("split", "pressure", "k"):
        counts = Counter(str(s.get(field)) for s in scenarios)
        print(f"  by {field:<9} {dict(sorted(counts.items()))}")
    types = Counter(lookup[c]["type"]
                    for s in scenarios for c in s.get("constraints", [])
                    if c in lookup)
    print(f"  constraint checks by type {dict(sorted(types.items()))}")
    n_adv = sum(1 for s in scenarios if s.get("pressure") == "adversarial")
    note = ("robustness is defined" if n_adv
            else "NO adversarial rows -- robustness will be null")
    print(f"  adversarial subset: {n_adv} ({note})")


def check_every_constraint_runs(scenarios: Sequence[Mapping[str, Any]],
                                lookup: Mapping[str, Mapping[str, Any]]) -> List[str]:
    """Run each distinct constraint against a probe string.

    validate_eval_set proves the ids resolve and the rows render. It does not
    prove the constraint OBJECTS are ones the pinned checkers accept -- a
    foreign id, an unexpected key or a family/split that disagrees with the
    registry all raise at scoring time, which is after the money again.
    """
    problems: List[str] = []
    used = {c for s in scenarios for c in s.get("constraints", []) if isinstance(c, str)}
    for cid in sorted(used):
        con = lookup.get(cid)
        if con is None:
            continue
        try:
            check(con, "A short probe sentence, with nothing unusual about it.")
        except Exception as exc:  # noqa: BLE001 - any checker refusal is fatal
            problems.append(f"constraint {cid!r}: the pinned checker refuses it "
                            f"({type(exc).__name__}: {exc})")
    return problems


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("eval_set")
    ap.add_argument("constraints", nargs="?", default=str(DEFAULT_CONSTRAINTS))
    args = ap.parse_args(argv)

    eval_path, con_path = Path(args.eval_set), Path(args.constraints)
    for p in (eval_path, con_path):
        if not p.is_file():
            print(f"FAIL {p} does not exist", file=sys.stderr)
            return 1

    scenarios = read_jsonl(eval_path)
    try:
        lookup = build_constraint_lookup(read_jsonl(con_path))
    except HarnessError as exc:
        print(f"FAIL {con_path}: {exc}", file=sys.stderr)
        return 1

    print(f"preflight {eval_path}")
    failed = False
    try:
        validate_eval_set(scenarios, lookup, render_user_message)
        print("  ok   every row has the fields the run needs, resolves its "
              "constraints, and renders into a prompt")
    except HarnessError as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        failed = True

    problems = check_every_constraint_runs(scenarios, lookup)
    if problems:
        failed = True
        print(f"FAIL {len(problems)} constraint(s) the pinned checkers refuse:",
              file=sys.stderr)
        for p in problems[:20]:
            print(f"  - {p}", file=sys.stderr)
    else:
        print("  ok   every constraint referenced is one the pinned checkers accept")

    if not failed:
        describe(scenarios, lookup)
        print("\nPREFLIGHT PASS -- eval.py will accept this file")
        return 0
    print("\nPREFLIGHT FAIL -- eval.py will refuse this file; nothing would be "
          "spent, but nothing would be scored either", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
