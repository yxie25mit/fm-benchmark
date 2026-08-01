#!/usr/bin/env bash
# Assemble a clean, relocatable pharma release from this pipeline.
# Usage:  bash make_pharma_release.sh [DEST_DIR]   (default: ../pharma_release)
# Produces code + benchmark data + splits + docs, WITHOUT results/, logs, caches,
# manuscript files, or transient user_*/e2e_* datasets. See PACKAGING.md for the
# conda envs, fork repos, and env-vars needed to actually run it on a new machine.
set -euo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)"
DEST="${1:-$SRC/../pharma_release}"

echo ">> assembling clean release"
echo "   from: $SRC"
echo "   to:   $DEST"

rsync -aL \
  --exclude='results' \
  --exclude='logs' \
  --exclude='cache' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='jcim_*' \
  --exclude='ext_raw' \
  --exclude='pharma_run.py' \
  --exclude='foundation_model_*' \
  --exclude='*.pdf' --exclude='*.tex' --exclude='*.docx' \
  --exclude='*.tar.gz' --exclude='*.html' \
  --exclude='cleaned/user_*' --exclude='cleaned/e2e_*' \
  --exclude='splits/user_*'  --exclude='splits/e2e_*' \
  --exclude='.git' \
  "$SRC/" "$DEST/"

mkdir -p "$DEST/results"   # fresh, writable output dir (the source results/ is a symlink we skipped)

cd "$DEST"
if command -v git >/dev/null 2>&1; then
  git init -q
  git add -A
  git commit -q -m "Initial pharma release (code + benchmark data + splits)" || true
  echo ">> git repo initialized at $DEST"
fi

echo ">> done. Next (see PACKAGING.md):"
echo "   1. export the 5 conda envs into $DEST/envs/*.yml"
echo "   2. vendor the fork repos + checkpoints, set MOLFCL_DIR/MOTIL_DIR/MOLFORMER_DIR/CHEMELEON_HOME"
echo "   3. on the new machine: recreate envs, source config/env.sh, then follow README_PHARMA.md"
