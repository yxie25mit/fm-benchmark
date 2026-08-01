#!/usr/bin/env bash
# Fetch the large pretrained checkpoints that are NOT in git (too big).
# Fill in each CANONICAL source URL below — from the method's original repo / HuggingFace /
# Zenodo — or re-host them on Zenodo and point here. The small MolCLR GIN already ships in-repo
# (methods/molclr/ckpt), so it is not downloaded.
#
# Run AFTER `source config/env.sh` (needs MOLFCL_DIR / MOTIL_DIR / MOLFORMER_DIR / CHEMELEON_HOME).
set -e
: "${MOLFCL_DIR:?set MOLFCL_DIR — source config/env.sh first}"
: "${MOTIL_DIR:?set MOTIL_DIR}"
: "${MOLFORMER_DIR:?set MOLFORMER_DIR}"
: "${CHEMELEON_HOME:?set CHEMELEON_HOME}"

fetch () {  # fetch <url> <dest>
  local url="$1" dest="$2"
  if [ -f "$dest" ]; then echo "  have  $dest"; return; fi
  if [ "$url" = "TODO" ]; then echo "  !! set the URL for: $dest"; return; fi
  mkdir -p "$(dirname "$dest")"
  echo "  fetch $url -> $dest"
  curl -fL "$url" -o "$dest"
}

# CheMeleon foundation (~34 MB) — chemprop CheMeleon release asset
fetch "TODO" "$CHEMELEON_HOME/.chemprop/chemeleon_mp.pt"

# MolFCL pretrained (~12 MB) — MolFCL repo
fetch "TODO" "$MOLFCL_DIR/ckpt/original_MoleculeModel.pkl"

# MotiL pretrained (~7 MB) — MotiL repo
fetch "TODO" "$MOTIL_DIR/dumped/pre-train/1-model/original_CMPN_0707_0800_12000th_epoch.pkl"

# MoLFormer pretrained (~562 MB, this ONE checkpoint — not the full checkpoints dir) — IBM MoLFormer release / HuggingFace
fetch "TODO" "$MOLFORMER_DIR/Pretrained MoLFormer/checkpoints/N-Step-Checkpoint_3_30000.ckpt"

echo "done — any 'set the URL' lines above still need the canonical source filled in."
