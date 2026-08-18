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

### 3. Point the pipeline at your paths
```bash
conda activate orchestrator
cp config/env.sh.example config/env.sh
source config/env.sh
```
**You normally edit nothing** — all paths auto-detect (conda env location via `conda info --base`;
fork + CheMeleon paths from the repo). Only if `source` prints `conda: command not found`, open
`config/env.sh` and set the single `[EDIT]` line `PIPELINE_CONDA_ENVS` to your conda envs folder
(e.g. `/home/you/miniconda3/envs`).

Run all top-level commands from the activated **orchestrator** env (equivalently, `$PIPELINE_PYTHON`,
which `config/env.sh` sets). You do **not** activate the per-method envs by hand — the pipeline calls
each method's env python by absolute path.

### 4. Download the pretrained checkpoints
```bash
bash download_checkpoints.sh
```
Auto-fetches **CheMeleon** (Zenodo) and **MoLFormer** (HuggingFace). MolCLR's ships in the repo;
MolFCL/MotiL's ride inside their fork submodules. (Reads the env vars set in step 3, so run it after.)

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
Many folds (→ mean±std across them) — a directory of fold subdirs, each with `train/val/test.csv`
(`val` may also be named `valid`; files may be `.csv` (comma) or `.tsv` (tab))
(subdirs are read in **sorted name order** → `custom_seed0, custom_seed1, …`):
```bash
python prepare_user_data.py --name acme --splits-dir folds/ --target-cols Y --task reg
```
It drops RDKit-invalid SMILES (reported) and warns on train/test leakage, but **preserves your
split exactly** — nothing is reshuffled (safe for time splits). At run time `--protocols custom`
runs **every** fold and reports **mean±std**; tuned runs pick the best HP by **mean validation over
all folds** (TDC-style).

**Flags:**
- `--target-cols` — one or more label columns (pass several for multitask); blanks = missing (NaN-aware).
- `--metric` — optional; if omitted it's derived from `--task` (cls→ROC-AUC, reg→RMSE). Case-insensitive,
  `-`/`_` interchangeable. Supported: **cls** `roc_auc`(`auc`), `pr_auc`(`auprc`) — maximized; **reg**
  `rmse`, `mae`, `mse`, `scaled_mae` — minimized · `spearman`, `pearson` — maximized. The optimize
  direction (used for best-val HP selection) is inferred from the metric.
- `--learning-curve-sizes` / `--lc-repeats` — see [Learning curves](#learning-curves-low-data-regime) below.

> Have just **one CSV** and want us to scaffold-split it the way we split the benchmark datasets?
> See [Reproducing our scaffold splits on your own data](#reproducing-our-scaffold-splits-on-your-own-data) at the end.

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

### Quick check (does every method run? ~10 min)
Trains all methods on the smallest dataset with a tiny split and only 2 epochs — for confirming the
install works, **not** for real numbers:
```bash
# 6 fast methods together
python -m orchestrator.run_benchmark \
  --methods chemprop2 chemprop2_nofp chemeleon molclr molfcl motil \
  --datasets freesolv --protocols v1_det --phases default \
  --epochs-default 2 --gpus 0,1,2,3 --jobs-per-gpu 2
# molformer on its own (needs --jobs-per-gpu 1)
python -m orchestrator.run_benchmark \
  --methods molformer --datasets freesolv --protocols v1_det --phases default \
  --epochs-default 2 --gpus 0,1,2,3 --jobs-per-gpu 1
# see per-method results (one row each = it worked)
python collect_results.py --dataset freesolv --protocol v1_det --phase default \
  --methods chemprop2 chemprop2_nofp chemeleon molclr molfcl motil molformer --out smoke.csv
```

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

## Notes on molformer (how it differs from upstream)
molformer's finetuning here departs from the upstream IBM MoLFormer repo in ways that matter when
you read its numbers:

- **Multi-target (multitask) support is our extension.** Upstream MoLFormer trains **one model per
  property** for regression — e.g. its QM9 paper number is the *average of 12 single-property models*
  — and only supports multitask *classification* for a few hardcoded benchmark datasets. To keep
  molformer consistent with the other 6 methods, here it trains **one shared multitask model** (N
  output heads) for both classification and regression, and works on your **own** multi-target data. If your dataset has one endpoint (one
  property), molformer uses the original upstream trainer. The multitask paths are code we added.
- **Per-target standardization.** For multitask regression each target column is standardized (train
  mean/std) before training and predictions are de-standardized for the reported MAE — necessary
  because properties can be on very different scales.
- **Same as upstream:** single-target classification/regression, and the multitask-classification
  model structure.

## Method credits (originals these forks/checkpoints derive from)
MolCLR (github.com/yuyangw/MolCLR) · MolFCL (github.com/tangxiangcsu/MolFCL) ·
MotiL (github.com/Young0222/MotiL) · MoLFormer (github.com/IBM/molformer) ·
chemprop (github.com/chemprop/chemprop) · CheMeleon (github.com/JacksonBurns/chemeleon).

---

## Reproducing our scaffold splits on your own data
The most realistic evaluation is usually a **time split** — train on molecules discovered before a
cutoff date, test on ones after — because it mirrors real prospective use. If you have reliable
timestamps, prefer that (feed it via the [`--train/val/test-csv` / `--splits-dir` path](#adapt-your-data-converts-csvs--the-pipelines-format-never-re-splits) above,
which preserves your split exactly under `--protocols custom`).

If instead you want to evaluate the **same way we do** — our two Bemis–Murcko scaffold-split styles,
applied to *your* molecules with the identical algorithm — hand us **one CSV** and we generate both.
(Scaffold splits are a well-established proxy for time splits: chemprop's own comparisons show they
approximate prospective performance reasonably well.) This runs the *same* `v1_preshuffle` +
`v2_astartes` code we run on the benchmark datasets (verified to reproduce them byte-for-byte), at
seeds 0/1/2 (eval) + 3 (HP-only) each, so your numbers line up with ours.
```bash
# 1) SPLIT: one CSV -> splits/mydata/{v1_preshuffle,v2_astartes}_seed{0,1,2,3}/ + cleaned/mydata.csv
python prepare_user_data.py --name mydata --csv mydata.csv \
  --smiles-col SMILES --target-cols activity --task cls   # --metric optional; --task reg for regression

# 2) RUN — downstream is IDENTICAL to a built-in dataset. Run ONE of the two phases (or both):

# 2a) DEFAULT HPs (fast: each method's default HPs, one 5-model ensemble per fold)
python -m orchestrator.run_benchmark --methods chemeleon chemprop2 molclr \
  --datasets mydata --protocols v1_preshuffle v2_astartes --phases default \
  --gpus 0,1,2,3 --jobs-per-gpu 1

# 2b) TUNED HPs (grid search; best config by mean validation across the 3 eval seeds, then final ensembles)
python -m orchestrator.run_benchmark --methods chemeleon chemprop2 molclr \
  --datasets mydata --protocols v1_preshuffle v2_astartes --phases hp_search hp_final \
  --gpus 0,1,2,3 --jobs-per-gpu 1

# 3) COLLECT — one table per split style (mean ± std over the 3 eval seeds). Match --phase to the run.
#    after 2a (default):
python collect_results.py --dataset mydata --protocol v1_preshuffle --phase default \
  --methods chemeleon chemprop2 molclr --out mydata_v1_default.csv
python collect_results.py --dataset mydata --protocol v2_astartes  --phase default \
  --methods chemeleon chemprop2 molclr --out mydata_v2_default.csv
#    after 2b (tuned):
python collect_results.py --dataset mydata --protocol v1_preshuffle --phase hp_final \
  --methods chemeleon chemprop2 molclr --out mydata_v1_hpfinal.csv
python collect_results.py --dataset mydata --protocol v2_astartes  --phase hp_final \
  --methods chemeleon chemprop2 molclr --out mydata_v2_hpfinal.csv
```
- `molformer` must use `--jobs-per-gpu 1`; the others tolerate 2–4.
- Same flags as the adapter above: multitask (several `--target-cols`), `--metric`, and
  `--learning-curve-sizes` / `--lc-repeats` (subsets are generated per style; then
  `run_learning_curve.py --protocols v1_preshuffle v2_astartes --fractions <sizes>`).
- Invalid SMILES dropped and exact duplicates merged (both reported); everything else is scaffold-split.
- One style only: add `--split-styles v1_preshuffle` at split time and drop the other from `--protocols`.
- Requires `astartes` in the orchestrator env (in `envs/orchestrator.yml`; `pip install 'astartes[molecules]'`
  if you built that env before this was added).

---

## Time splits (chronological) on your own data
If you have reliable timestamps, a **time split** is the most realistic (prospective) evaluation.
Give us **one time-sorted CSV** (or any CSV plus a `--date-col`) and we generate two designs, matching
how chemprop reports time-split results:

- **`<name>_chrono`** — a single **80/10/10 chronological split** (earliest → train, latest → test).
  One fold; the headline prospective number.
- **`<name>_sliding`** — **constant-train sliding windows** for error bars (default: 12 chunks,
  train 8 / val 1 / test 1, sliding by 1 → 3 folds testing the newest chunks). Train size is held
  **constant** across folds (unlike an expanding window), so the fold-to-fold spread is interpretable.

**Per-fold HP tuning (the anti-leakage flag, on by default for sliding).** For the sliding dataset,
each fold is tuned on **its own validation only** — never the mean validation across folds. This is
essential for time splits: a later fold's validation is an earlier fold's *test*, so pooling validation
across folds (or reusing the 80/10/10 HPs) would select hyperparameters on data you later test on.
Per-fold tuning avoids that — every fold's validation strictly precedes its own test. `--time-split`
stamps `hp_per_fold` on `<name>_sliding`, so `run_phase` does this automatically; you can also force it
anywhere with `--hp-per-fold` on `run_benchmark`. (Without it, multi-fold `custom` datasets pick one HP
by mean validation, TDC-style — correct for scaffold folds, **leaky** for time folds.)

```bash
# 1) SPLIT: one time-sorted CSV -> mydata_chrono (1 fold) + mydata_sliding (3 folds, hp_per_fold=True)
python prepare_user_data.py --name mydata --csv mydata.csv \
  --smiles-col SMILES --target-cols activity --task cls \
  --time-split --date-col assay_date          # omit --date-col if the CSV is already time-sorted

# 2) RUN both datasets together, in ONE phase (or both). run_benchmark applies per-fold HP to
#    mydata_sliding automatically (its meta) and normal selection to mydata_chrono.

# 2a) DEFAULT HPs (fast; no tuning, so the per-fold-HP question doesn't arise)
python -m orchestrator.run_benchmark --methods chemprop2 molclr \
  --datasets mydata_chrono mydata_sliding --protocols custom --phases default \
  --gpus 0,1,2,3 --jobs-per-gpu 1

# 2b) TUNED HPs (chrono tuned on its own val; sliding tuned PER FOLD on each fold's own val, automatic)
python -m orchestrator.run_benchmark --methods chemprop2 molclr \
  --datasets mydata_chrono mydata_sliding --protocols custom --phases hp_search hp_final \
  --gpus 0,1,2,3 --jobs-per-gpu 1

# 3) COMBINED TABLE (chrono headline + sliding mean±std). Match --phase to the run:
python collect_time_split.py --name mydata --phase default  --methods chemprop2 molclr   # after 2a
python collect_time_split.py --name mydata --phase hp_final --methods chemprop2 molclr   # after 2b
```
- `molformer` must use `--jobs-per-gpu 1`; the others tolerate 2–4.
- **Same data flags as the other modes** (all valid with `--time-split`): `--task cls|reg` (required),
  `--target-cols` (one column, or several for a multitask model), `--metric` (optional; else ROC-AUC for
  cls / RMSE for reg — case-insensitive, e.g. `roc_auc`/`pr_auc`/`rmse`/`mae`/`spearman`/`pearson`),
  `--smiles-col` (default `smiles`). Blank label cells = missing (NaN-aware). Both emitted datasets
  (`_chrono`, `_sliding`) inherit these. (`--split-styles` is scaffold-only. `--learning-curve-sizes` **is**
  supported — see the learning-curve block below; `--lc-repeats` is ignored for time splits because the
  3 sliding folds already provide the replicates.)
- **Window geometry** (defaults shown): `--tw-chunks 12` cuts the time-sorted rows into 12 equal-count
  chunks; `--tw-train-chunks 8` = each fold trains on 8 consecutive chunks (**constant** window);
  `--tw-folds 3` = number of sliding folds. With the defaults that gives:
  fold 0 = train chunks **1–8**, val **9**, test **10**; fold 1 = train **2–9**, val **10**, test **11**;
  fold 2 = train **3–10**, val **11**, test **12**. (Needs `tw-chunks ≥ tw-train-chunks + tw-folds + 1`.)
- `--date-col` sorts the CSV by that column; without it, rows are assumed already oldest→newest.
- `--phases default` has no HP tuning, so no leakage concern; `--hp-per-fold` only matters for the tuned
  path (and is already automatic for `mydata_sliding`). No `astartes` needed here (time order comes from
  your data, not scaffolds).

### Learning curve (how performance scales with training-set size)
Add `--learning-curve-sizes` at split time. This builds **recency-nested** training subsets on the
**sliding folds** — at each size, each fold trains on its most-recent *N* rows (nested suffixes:
last-100 ⊂ last-200 ⊂ …) with that fold's val/test fixed — so the curve answers "how much recent
history do I need." The **error bars come from the 3 folds** (the analog of the 3 seeds on MoleculeNet),
and sizes are capped at each fold's train-window size. Learning curves use **default HPs** (no per-size
tuning), so there is no leakage concern.
```bash
# 1) split, adding sizes (generates mydata_sliding/custom__<size>_seed{0,1,2}, recency-nested)
python prepare_user_data.py --name mydata --csv mydata.csv \
  --smiles-col SMILES --target-cols activity --task cls \
  --time-split --date-col assay_date --learning-curve-sizes 100 200 500 1000

# 2) run the curve on the sliding folds (5-model ensemble per point; error bars across the 3 folds)
python runners/run_learning_curve.py --methods chemprop2 molclr --datasets mydata_sliding \
  --protocols custom --fractions 100 200 500 1000 --seeds 0 1 2 --ensemble 5 --gpus 0,1,2,3

# 3) collect (mean±std over the 3 folds at each size) + plot
python collect_results.py --dataset mydata_sliding --protocol custom --learning-curve \
  --methods chemprop2 molclr --out mydata_curve.csv --plot mydata_curve.png
```
- **`--fractions` must match the `--learning-curve-sizes` you split with**, and **`--protocols custom`**
  (the sliding dataset's splits live under `custom__<size>_seed<fold>`).
- **`--seeds 0 1 2` are the 3 sliding folds** — they must match `--tw-folds`. Passing fewer (e.g. the
  default `--seeds 0`) runs only some folds and you lose the error bars. `--ensemble 5` matches the
  5-model ensembles used elsewhere (use fewer only for a quick preview).
- Sizes are **capped at each fold's train-window size** (~`tw-train-chunks/tw-chunks` of your data, so
  ~67% by default). To anchor the curve at the full window, include a size ≥ that (it caps to 100%).
- Run it on **`mydata_sliding`** (not `_chrono`): the curve's error bars come from the 3 folds. `molformer`
  needs `--jobs-per-gpu 1` here too.
