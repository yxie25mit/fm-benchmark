# Setup — get running in 5 steps

This benchmark runs 8 molecular property-prediction methods on MoleculeNet + TDC (or your own
data). The git repo holds the pipeline code + benchmark data; the conda envs, method fork repos,
and large pretrained checkpoints are pulled in separately (they're too big for git). Below is the
whole setup.

---

## Step 1 — Clone (with the method forks)
```bash
git clone --recursive https://github.com/yxie25mit/fm-benchmark.git
cd fm-benchmark
# if you forgot --recursive:  git submodule update --init
```
The method code (MolFCL, MotiL, MoLFormer) comes in as submodules under `forks/`.

## Step 2 — Create the conda environments
The methods have **conflicting** deps (some pin torch 1.7.1 / CUDA 11.0), so there are 5 envs —
one lightweight orchestrator + one per method family:
```bash
for e in orchestrator chemprop2 molclr molfcl molformer; do
  conda env create -f envs/$e.yml
done
```
- `orchestrator` (light: numpy/pandas/rdkit/sklearn/scipy/matplotlib) runs the pipeline + prepares
  data + collects results. **No GPU/torch.**
- `chemprop2` runs chemprop2, chemprop2_nofp, chemeleon, chemeleon_nofp.
- `molclr`, `molfcl` (also motil), `molformer` — one each.
- **cu110 note:** `molclr` and `molformer` need torch 1.7.1 + CUDA 11.0. `molclr.yml` already
  includes the `--find-links` for the cu110 wheels; if `molformer`'s conda `pytorch=1.7.1` build is
  unavailable, install it with `pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html`.

## Step 3 — Get the pretrained checkpoints (not in git)
Small one (MolCLR GIN) ships in the repo. The large ones download from their original sources:
```bash
bash download_checkpoints.sh        # pulls MoLFormer / CheMeleon / MolFCL / MotiL weights
```
(If a source is offline, the script prints where each file must go.)

## Step 4 — Point the pipeline at your paths
```bash
cp config/env.sh.example config/env.sh
source config/env.sh
```
**You normally edit nothing** — all paths auto-detect (conda env location via `conda info --base`;
fork + CheMeleon paths from the repo itself). Only if `source` says `conda: command not found`,
open `config/env.sh` and set the single `[EDIT]` line `PIPELINE_CONDA_ENVS` to your conda envs
folder (e.g. `/home/you/miniconda3/envs`).

## Step 5 — Run  (see README_PHARMA.md for all options)
```bash
# our methods on the benchmark datasets:
python -m orchestrator.run_benchmark --methods chemeleon molclr --datasets bace esol \
  --protocols v2_astartes --phases hp_search hp_final --gpus 0,1,2,3 --jobs-per-gpu 2

# your own data (see README_PHARMA.md):
python prepare_user_data.py --name acme --train-csv train.csv --val-csv val.csv --test-csv test.csv \
  --smiles-col SMILES --target-cols activity --task cls
python -m orchestrator.run_benchmark --methods chemeleon --datasets acme --protocols custom \
  --phases default --gpus 0,1,2,3

# collect results into a table (+ learning-curve plot):
python collect_results.py --dataset acme --protocol custom --phase default
```
Run the top-level commands with the **orchestrator** env's python (`PIPELINE_PYTHON`).

---

# For the maintainer — publishing the method forks

The 3 method repos are *modified* third-party code, so they're published as your own repos and
linked as submodules (keeps the main repo lean + preserves attribution).

**1. Publish each fork** (run inside each fork directory — `MolFCL`, `MotiL`, `molformer`):
```bash
cd /data/rbg/users/yxie25/molclr_chemprop/MolFCL
git init 2>/dev/null; git add -A
git commit -m "MolFCL fork for fm-benchmark (modified from <UPSTREAM_URL>)"
gh repo create yxie25mit/MolFCL-fork --public --source=. --remote=origin
git push -u origin main
```
Repeat for `MotiL` → `yxie25mit/MotiL-fork` and `molformer` → `yxie25mit/molformer-fork`.
Add an attribution line to each fork's README crediting the original repo.
**Do NOT commit the big checkpoints into the fork git** (molformer's is 3.7 GB) — keep them in
`download_checkpoints.sh` (from the original source) or git-LFS.

**2. Link them as submodules** in the benchmark repo:
```bash
cd fm-benchmark
git submodule add https://github.com/yxie25mit/MolFCL-fork.git   forks/MolFCL
git submodule add https://github.com/yxie25mit/MotiL-fork.git    forks/MotiL
git submodule add https://github.com/yxie25mit/molformer-fork.git forks/molformer
git commit -m "Add method forks as submodules"
git push
```

**3. Make the pipeline point at the submodules** — in `config/env.sh`:
```bash
export MOLFCL_DIR=$PWD/forks/MolFCL
export MOTIL_DIR=$PWD/forks/MotiL/MotiL_micromolecule
export MOLFORMER_DIR=$PWD/forks/molformer
```
That's it — users then get everything with a single `git clone --recursive`.
