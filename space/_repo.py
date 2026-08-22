"""Put the PINNED slmforge modules on ``sys.path`` -- or fail loudly.

The Space is not allowed to contain a second copy of the prompt renderer, the
checkers, or the model pins. It imports the real ones. That means the Space
process has to find a slmforge tree, and there are exactly two situations:

    running from the repo    ``space/`` sits inside the slmforge checkout, and
                             its parent IS the tree.
    running on the Hub       the Space repo carries a partial mirror of the
                             tree under ``pins/``, written by
                             ``build_space.sh`` at deploy time.

A root is accepted only when EVERY marker below resolves. A half-populated
mirror is rejected with the list of what is missing rather than imported and
discovered three imports later, because the failure mode this guards against --
a Space that silently falls back to something that is not the pinned code -- is
the one that would make every verdict it prints meaningless.

There is deliberately no "best effort" branch and no bundled fallback copy of
any pinned module: if the pins are not present, this raises. A Space that
cannot find the checkers must not start.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Tuple

HERE = Path(__file__).resolve().parent

#: Every file the Space's import graph actually needs, relative to a root.
#: Derived from the imports in core.py: scripts/demo.py -> eval.py -> harness.py
#: + judge.py + scenarios.render + constraints.registry, plus the two configs
#: read at runtime and the shipped constraint library --constraint-id reuses.
MARKERS: Tuple[str, ...] = (
    "eval.py",
    "harness.py",
    "judge.py",
    "constraints/__init__.py",
    "constraints/registry.py",
    "constraints/checkers.py",
    "scenarios/__init__.py",
    "scenarios/paths.py",
    "scenarios/render.py",
    "scripts/demo.py",
    "configs/models.json",
    "configs/train.json",
    "data/scenarios/constraint_instances.jsonl",
)

#: Set ``SLMFORGE_ROOT`` to override the search entirely (used by the local
#: verification harness and by anyone running the app from an odd cwd).
ROOT_ENV = "SLMFORGE_ROOT"


class RepoRootNotFound(RuntimeError):
    """No candidate directory carried the pinned modules."""


def candidates() -> List[Path]:
    """Search order, most explicit first.

    The repo checkout is preferred over ``pins/`` so that a developer running
    from the tree exercises the LIVE pinned modules, not a stale mirror; on the
    Hub only ``pins/`` exists, so the same order resolves there without a flag.
    """
    out: List[Path] = []
    env = os.environ.get(ROOT_ENV)
    if env:
        out.append(Path(env).expanduser().resolve())
    out.append(HERE.parent)     # slmforge checkout, when space/ lives inside it
    out.append(HERE / "pins")   # the deployed mirror
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def missing_markers(root: Path) -> List[str]:
    return [m for m in MARKERS if not (root / m).exists()]


def find_root() -> Path:
    """First candidate carrying every marker. Raises with the full diagnosis."""
    report = []
    for root in candidates():
        missing = missing_markers(root)
        if not missing:
            return root
        report.append(f"  {root}\n      missing: {', '.join(missing)}")
    raise RepoRootNotFound(
        "could not find the pinned slmforge modules. This Space imports the "
        "project's real renderer, checkers and model pins and will not "
        "substitute its own.\n"
        "Looked in:\n" + "\n".join(report) + "\n"
        f"Fix: re-assemble the Space with space/build_space.sh (it writes pins/ "
        f"and refuses to finish if a marker is missing), or set {ROOT_ENV} to a "
        f"slmforge checkout.")


def install() -> Path:
    """Find the root, put it (and ``scripts/``) on ``sys.path``, return it.

    Idempotent: repeated calls do not stack duplicate entries.
    ``scripts/`` is added because the repo has no ``scripts/__init__.py`` --
    ``scripts/demo.py`` is imported as the top-level module ``demo``, exactly as
    the repo's own tests import it.
    """
    root = find_root()
    for path in (str(root / "scripts"), str(root)):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)
    return root
