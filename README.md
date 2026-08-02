# Molecular property benchmark

Run 7 molecular property-prediction methods — `chemprop2`, `chemprop2_nofp`, `chemeleon`,
`molclr`, `molfcl`, `motil`, `molformer` — on MoleculeNet + TDC, or on **your own
data / splits / fine-tuned checkpoints**.

Two parts: **[Setup](#setup)** (once) → **[Run](#run)**.

---

# Setup

### 1. Clone (with the method forks)
```bash
git clone --recursive https://github.com/yxie25mit/fm-benchmark.git
cd fm-benchmark
# forgot --recursive?   git submodule update --init
```
The modified method repos (MolFCL, MotiL, MoLFormer) come in as submodules under `forks/`.

### 2. Create the conda environments
There are **5** — one lightweight orchestrator + one per method family (they pin conflicting
torch/CUDA, so one env can't cover all):
```bash
for e in orchestrator chemprop2 molclr molfcl molformer; do conda env create -f envs/$e.yml; done
```
| env | runs | notes |
|-----|------|-------|
| `orchestrator` | the pipeline + data prep + result collection (no GPU/torch) | light: numpy/pandas/rdkit/sklearn/scipy/matplotlib |
| `chemprop2` | chemprop2, chemprop2_nofp, chemeleon | chemprop v2 |
| `molclr` | molclr | torch 1.7.1+cu110 (yml has the cu110 `--find-links`) |
| `molfcl` | molfcl, motil | chemprop v1 fork |
| `molformer` | molformer | torch 1.7.1, pytorch-lightning |

### 3. Download the pretrained checkpoints
```bash
bash download_checkpoints.sh
```
Auto-fetches **CheMeleon** (Zenodo) and **MoLFormer** (HuggingFace). MolCLR's ships in the repo;
MolFCL/MotiL's ride inside their fork submodules.

### 4. Point the pipeline at your paths
```bash
cp config/env.sh.example config/env.sh
source config/env.sh
```
**You normally edit nothing** — all paths auto-detect (conda env location via `conda info --base`;
fork + CheMeleon paths from the repo). Only if `source` prints `conda: command not found`, open
`config/env.sh` and set the single `[EDIT]` line `PIPELINE_CONDA_ENVS` to your conda envs folder
(e.g. `/home/you/miniconda3/envs`).

Run all top-level commands with the **orchestrator** env's python (that's `$PIPELINE_PYTHON`, set
by `config/env.sh`).

---

# Run

## Your data format
A CSV with a **SMILES** column (any name) + one or more **label** columns (classification 0/1, or
regression floats; blank = missing). You provide train/val/test as separate CSVs (or a folder of
folds). No split-index or mask files needed.
```
# train.csv (classification)        # train.csv (regression, 1+ targets)
SMILES,activity                     smiles,logD,solubility
CCO,1                               CCO,-0.31,1.10
c1ccccc1,0                          c1ccccc1,2.13,-2.00
```

## Adapt your data (converts CSVs → the pipeline's format; never re-splits)
Single split:
```bash
python prepare_user_data.py --name acme \
  --train-csv train.csv --val-csv val.csv --test-csv test.csv \
  --smiles-col SMILES --target-cols activity --task cls
```
Many folds (→ mean±std across them) — a directory of fold subdirs, each with `train/val/test.csv`:
```bash
python prepare_user_data.py --name acme --splits-dir folds/ --target-cols Y --task reg
```
It drops RDKit-invalid SMILES (reported) and warns on train/test leakage, but **preserves your
split exactly** — nothing is reshuffled (safe for time splits).

## The run command = `--protocols` (which split) + `--phases` (default vs tuned HPs)

**Protocols:**
- **`custom`** — your prepared splits (from the step above).
- **`v1_preshuffle`** / **`v2_astartes`** — our two split schemes for the benchmark datasets. **Both
  are Bemis–Murcko scaffold splits** (neither is random): `v1_preshuffle` sorts scaffold buckets
  largest→smallest (equal-sized shuffled) then fills train→val→test; `v2_astartes` assigns whole
  scaffold clusters to train/val/test via astartes. (`tdc` = the 5 fixed ADMET splits.)

**Phases:**
- **`--phases default`** — each method's default HPs, one 5-model ensemble per fold (fast).
- **`--phases hp_search hp_final`** — tunes the grid: every config runs on **all** folds, best is
  picked by **mean validation across folds** (TDC-style), then trained on every fold.

### Commands
Our methods on **your** split (default HPs):
```bash
python -m orchestrator.run_benchmark --methods chemeleon chemprop2 molclr molformer \
  --datasets acme --protocols custom --phases default --gpus 0,1,2,3 --jobs-per-gpu 1
```
Our methods on **your** split (tuned):
```bash
python -m orchestrator.run_benchmark --methods chemeleon molclr motil \
  --datasets acme --protocols custom --phases hp_search hp_final --gpus 0,1,2,3 --jobs-per-gpu 2
```
Our methods on **our** benchmark datasets (both scaffold protocols):
```bash
python -m orchestrator.run_benchmark --methods chemeleon chemprop2 molclr \
  --datasets bace esol --protocols v1_preshuffle v2_astartes \
  --phases hp_search hp_final --gpus 0,1,2,3 --jobs-per-gpu 2
```
(`molformer` must use `--jobs-per-gpu 1`; others tolerate 2–4.)

### Use YOUR OWN fine-tuned checkpoint
Prefix the method's env var — no extra flag, works on your split or ours:
```bash
CHEMELEON_CKPT=/path/your_chemeleon.pt \
python -m orchestrator.run_benchmark --methods chemeleon --datasets acme \
  --protocols custom --phases default --gpus 0,1,2,3
```
| method | env var | expects |
|--------|---------|---------|
| chemeleon | `CHEMELEON_CKPT` | a chemprop-saved model `.pt` (a *fine-tuned* CheMeleon from a prior chemprop run — **not** the raw foundation file) |
| molclr | `MOLCLR_CKPT_DIR` | a dir holding `checkpoints/model.pth` |
| molfcl | `MOLFCL_CKPT` | a `.pkl` |
| motil | `MOTIL_CKPT` | a `.pkl` |
| molformer | `MOLFORMER_CKPT` | a `.ckpt` |
| chemprop2 / chemprop2_nofp | — | trains from scratch |

## Output
```
results/<method>/<name>/<protocol>/<phase>/
  seed0_em0/  metrics.json  pred_test.npy  labels_test.npy  done.flag   # one per (fold × ensemble member)
  ...
  _summary.json                                                         # mean ± std across folds
```
- **Per run** (`metrics.json`): that model's `test_metric`/`val_metric`, per-target scores, HPs used.
- **Per method** (`_summary.json`): the pipeline already ensemble-averages the 5 members and reports
  **mean ± std across folds** — no manual aggregation.
- **Across methods** — one comparison table:
  ```bash
  python collect_results.py --dataset acme --protocol custom --phase default \
    --methods chemeleon molclr molformer --out acme.csv
  ```

## Learning curves (low-data regime)
Add subsample sizes at prepare time (class-stratified; val/test held fixed; repeats give error bars):
```bash
python prepare_user_data.py --name acme --train-csv train.csv --val-csv val.csv --test-csv test.csv \
  --smiles-col SMILES --target-cols activity --task cls \
  --learning-curve-sizes 100 200 500 1000 --lc-repeats 3
python runners/run_learning_curve.py --methods chemeleon molclr --datasets acme \
  --protocols custom --fractions 100 200 500 1000 --seeds 0 1 2 --gpus 0,1,2,3
python collect_results.py --dataset acme --protocol custom --learning-curve \
  --methods chemeleon molclr --out acme_curve.csv --plot acme_curve.png
```

---

## Method credits (originals these forks/checkpoints derive from)
MolCLR (github.com/yuyangw/MolCLR) · MolFCL (github.com/tangxiangcsu/MolFCL) ·
MotiL (github.com/Young0222/MotiL) · MoLFormer (github.com/IBM/molformer) ·
chemprop (github.com/chemprop/chemprop) · CheMeleon (github.com/JacksonBurns/chemeleon).
