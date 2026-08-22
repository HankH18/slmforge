#!/usr/bin/env python3
"""Read the deployed Space back off the Hub and compare it with what was built.

    set -a && . ./.env && set +a
    .venv/bin/python space/verify_deploy.py HankH18/forbidden-constraint-holder \
        --built /private/tmp/slmforge-space

Exit 0 only when ALL of the following hold:

  * the Space exists, is PUBLIC, and its SDK is gradio;
  * its runtime stage is RUNNING (a Space stuck in BUILD_ERROR still has files);
  * every file in the local build directory exists in the Space with the SAME
    sha256, and the Space carries no extra Python file the build did not put
    there.

The point is that "I pushed it" is not evidence. This asks the Hub what is
actually there and diffs it, which is the only version of that claim worth
making. It downloads only the small text files -- no weights.

``--no-runtime`` skips the RUNNING check for the minutes right after a push
while the Space is still building.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Optional

#: Files the Space is expected to carry, beyond the pins mirror. Kept explicit
#: so an unexpected extra .py file on the Hub is reported rather than ignored.
EXPECTED_TOP_LEVEL = ("app.py", "core.py", "_repo.py", "README.md",
                      "requirements.txt", ".gitignore")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_tree(root: Path) -> Dict[str, str]:
    """relative path -> sha256, skipping caches and git metadata."""
    out: Dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if rel.startswith(".git/") or "__pycache__" in rel or rel.endswith(".pyc"):
            continue
        out[rel] = sha256(p)
    return out


def compare_manifest(built: Dict[str, str], deployed: Dict[str, str]) -> List[str]:
    """Every difference, as human-readable lines. Empty list means identical.

    Both directions are reported. A file present on the Hub but not in the build
    is just as much a problem as a missing one -- it means the Space is running
    code that no local artefact accounts for.
    """
    problems: List[str] = []
    for rel, want in sorted(built.items()):
        got = deployed.get(rel)
        if got is None:
            problems.append(f"MISSING on the Hub: {rel}")
        elif got != want:
            problems.append(f"DIFFERENT bytes: {rel}\n"
                            f"    built    {want}\n"
                            f"    deployed {got}")
    for rel in sorted(set(deployed) - set(built)):
        problems.append(f"EXTRA on the Hub (not in the build): {rel}")
    return problems


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="space/verify_deploy.py",
                                 description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo_id", help="e.g. HankH18/forbidden-constraint-holder")
    ap.add_argument("--built", required=True,
                    help="the directory space/build_space.sh wrote")
    ap.add_argument("--no-runtime", action="store_true",
                    help="skip the RUNNING check (use while it is still building)")
    args = ap.parse_args(argv)

    built_dir = Path(args.built).expanduser().resolve()
    if not built_dir.is_dir():
        print(f"error: --built {built_dir} is not a directory", file=sys.stderr)
        return 2

    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    try:
        info = api.space_info(args.repo_id)
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: cannot read space {args.repo_id}: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    problems: List[str] = []
    sdk = getattr(info, "sdk", None)
    private = getattr(info, "private", None)
    runtime = getattr(info, "runtime", None)
    stage = getattr(runtime, "stage", None) if runtime else None
    hardware = getattr(runtime, "hardware", None) if runtime else None

    print(f"space     : {args.repo_id}")
    print(f"url       : https://huggingface.co/spaces/{args.repo_id}")
    print(f"sdk       : {sdk}")
    print(f"private   : {private}")
    print(f"stage     : {stage}")
    print(f"hardware  : {hardware}")

    if sdk != "gradio":
        problems.append(f"sdk is {sdk!r}, expected 'gradio'")
    if private:
        problems.append("the Space is PRIVATE; the deliverable requires it public")
    if not args.no_runtime and stage != "RUNNING":
        problems.append(f"runtime stage is {stage!r}, not 'RUNNING' -- the demo is "
                        f"not up, whatever the files say")

    local = snapshot_download(repo_id=args.repo_id, repo_type="space")
    deployed = digest_tree(Path(local))
    built = digest_tree(built_dir)
    print(f"files     : {len(built)} built, {len(deployed)} on the Hub")

    for name in EXPECTED_TOP_LEVEL:
        if name not in deployed:
            problems.append(f"the Space does not carry {name}")
    problems += compare_manifest(built, deployed)

    if problems:
        print("\nFAIL:")
        for p in problems:
            print(f"  {p}")
        return 1
    if args.no_runtime:
        # --no-runtime suppresses a check; the success line must say so rather
        # than report the same OK a full run earns.
        print(f"\nOK, RUNTIME CHECK SKIPPED: the Space is public, its SDK is "
              f"gradio, and every file matches the local build byte for byte. "
              f"Its stage is {stage!r} and was not required to be RUNNING, so "
              f"this is NOT evidence the demo is up. Re-run without "
              f"--no-runtime for that.")
    else:
        print("\nOK: the Space is public, running gradio, stage RUNNING, and "
              "every file matches the local build byte for byte.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
