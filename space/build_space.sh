#!/usr/bin/env bash
# Assemble the complete Hugging Face Space repo in OUTDIR.
#
#   space/build_space.sh /path/to/outdir
#
# The Space must import this project's PINNED renderer, checkers and model pins
# rather than carry a second copy of them, so the deploy step mirrors the real
# modules into OUTDIR/pins/ and _repo.py finds them there at boot. Every file
# copied is listed with its sha256 in pins/PINS_MANIFEST.txt, so what the Space
# is running can be diffed against the repo at any time.
#
# This script writes only to OUTDIR. It never touches the slmforge working tree
# and never runs a git index operation.
set -euo pipefail

SPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SPACE_DIR/.." && pwd)"
OUT="${1:-${TMPDIR:-/tmp}/slmforge-space}"

PY="$REPO_ROOT/.venv/bin/python"
[ -x "$PY" ] || { echo "error: $PY not found (this repo requires .venv/bin/python 3.11)" >&2; exit 1; }

echo "repo   : $REPO_ROOT"

rm -rf "$OUT"
mkdir -p "$OUT/pins"
# Absolute from here on. The manifest header prints this path inside a command a
# reader is told to run from the repo root, so a relative OUT would print a
# command that only works from whatever directory the build was launched in.
OUT="$(cd "$OUT" && pwd)"
echo "output : $OUT"

# ---- the Space's own files -------------------------------------------------
for f in app.py core.py _repo.py requirements.txt README.md; do
  cp "$SPACE_DIR/$f" "$OUT/$f"
done

cat > "$OUT/.gitignore" <<'EOF'
__pycache__/
*.pyc
EOF

# ---- the pinned mirror -----------------------------------------------------
mkdir -p "$OUT/pins/configs" "$OUT/pins/data/scenarios" "$OUT/pins/scripts"
cp "$REPO_ROOT/eval.py" "$REPO_ROOT/harness.py" "$REPO_ROOT/judge.py" "$OUT/pins/"
cp "$REPO_ROOT/scripts/demo.py" "$OUT/pins/scripts/"
cp "$REPO_ROOT/configs/models.json" "$REPO_ROOT/configs/train.json" "$OUT/pins/configs/"
cp "$REPO_ROOT/data/scenarios/constraint_instances.jsonl" "$OUT/pins/data/scenarios/"
for pkg in constraints scenarios; do
  mkdir -p "$OUT/pins/$pkg"
  # .py only: __pycache__ and any data files are not part of the import graph
  find "$REPO_ROOT/$pkg" -maxdepth 1 -name '*.py' -exec cp {} "$OUT/pins/$pkg/" \;
done

# ---- prove the mirror is complete, from _repo.py's own marker list ----------
# The list is not retyped here. If _repo.MARKERS grows a file this script does
# not copy, the build fails rather than shipping a Space that cannot boot.
"$PY" - "$SPACE_DIR" "$OUT/pins" <<'PYEOF'
import sys, pathlib
sys.path.insert(0, sys.argv[1])
import _repo
root = pathlib.Path(sys.argv[2])
missing = _repo.missing_markers(root)
if missing:
    sys.exit("build_space.sh did not copy: " + ", ".join(missing))
print(f"pins complete: {len(_repo.MARKERS)} markers present")
PYEOF

# ---- manifest --------------------------------------------------------------
COMMIT="$(cd "$REPO_ROOT" && git rev-parse HEAD 2>/dev/null || echo unknown)"
{
  echo "# Pinned slmforge modules mirrored into this Space."
  echo "# source repo commit: $COMMIT"
  echo "# built (UTC): $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "#"
  # Printed with the manifest's real absolute path and no placeholder, so the
  # line below runs as printed. The sed strips the "pins/" prefix, leaving paths
  # relative to the repo root -- which is the one directory it must be run from.
  echo "# Re-verify the mirror against the repo -- run this from the slmforge repo root:"
  echo "# shasum -a 256 -c <(sed 's| pins/| |' $OUT/pins/PINS_MANIFEST.txt | grep -v '^#')"
  cd "$OUT"
  find pins -name '*.pyc' -prune -o -type f -print | LC_ALL=C sort | xargs shasum -a 256
} > "$OUT/.PINS_MANIFEST.txt.tmp"
# The temp file is written OUTSIDE pins/ on purpose. Written inside it, the
# `find pins` above lists the half-written temp file itself, and the manifest
# ships a line for a path that does not exist after the mv -- which makes the
# re-verify command printed in its own header exit 1 on a correct build.
mv "$OUT/.PINS_MANIFEST.txt.tmp" "$OUT/pins/PINS_MANIFEST.txt"

echo
echo "Space assembled. Contents:"
( cd "$OUT" && find . -type f -not -path './.git/*' | LC_ALL=C sort | sed 's|^|  |' )
echo
echo "Next: see space/DEPLOY.md"
