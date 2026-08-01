# Running the molecular property benchmark on your own data

Run any of the 8 methods on **your data, your train/val/test splits, and (optionally) your own
fine-tuned checkpoints**.

Methods: `chemprop2`, `chemprop2_nofp`, `chemeleon`, `chemeleon_nofp`, `molclr`, `molfcl`,
`motil`, `molformer`.

It's two steps: **(1)** adapt your CSVs → **(2)** run. Your splits are used exactly as given —
nothing is re-split.

---

## What you provide

**A SMILES column + one or more label columns.** Column names are up to you.

**Option 1 — a single split:** three files `train.csv`, `val.csv`, `test.csv`.
```
# train.csv  (classification)          # train.csv  (regression, 1+ targets)
SMILES,activity                        smiles,logD,solubility
CCO,1                                  CCO,-0.31,1.10
c1ccccc1,0                             c1ccccc1,2.13,-2.00
CC(=O)Oc1ccccc1C(=O)O,1               CCN,0.48,0.30
```
Labels may be blank for missing values.

**Option 2 — multiple splits** (e.g. rolling time folds → you'll get mean±std across them): a
directory of fold subdirs, each with its own `train.csv`/`val.csv`/`test.csv`:
```
folds/
  2019/        # subdir names are free (fold0, 2019, cv1, …); sorted alphabetically
    train.csv  # -> becomes custom_seed0
    val.csv
    test.csv
  2020/        # -> custom_seed1
    train.csv
    val.csv
    test.csv
```
Rule: each subdir must contain exactly `train.csv`, `val.csv`, `test.csv`; subdirs are sorted by
name and mapped to fold 0, 1, 2, … in that order.

---

## Step 1 — Adapt your data

**Single split:**
```bash
python prepare_user_data.py --name acme \
  --train-csv train.csv --val-csv val.csv --test-csv test.csv \
  --smiles-col SMILES --target-cols activity --task cls
```
**Directory of folds** (run on multiple splits, results reported as mean±std):
```bash
python prepare_user_data.py --name acme \
  --splits-dir folds/ --smiles-col SMILES --target-cols activity --task cls
```
Either creates a dataset named exactly **`acme`** and prints a QC report (invalid SMILES dropped,
per-split sizes, any leakage). Run once per dataset.

<details>
<summary><b>What this step does (and why it does NOT re-split)</b></summary>

Our methods read a dataset internally as **one CSV + three row-index arrays**. You give separate
CSVs. This step is a **format adapter**: it concatenates your splits into one table and records
which rows came from which file.

- **Your split membership is preserved exactly, 1:1** — no molecule is moved between train/val/test,
  nothing is reshuffled. Critical for **time splits**.
- It **drops SMILES RDKit can't parse** (they'd crash training) and reports the count. Nothing else
  is removed; it never re-splits.
- It **reports duplicates/leakage** (same molecule in train and test of a fold) as a warning, left
  in place for you to fix upstream if unintended.

Writes: `cleaned/acme.csv`, `cleaned/acme.meta.json`, `splits/acme/custom_seed<i>/{train,val,test}_idx.npy`.
</details>

<details>
<summary><b>Flags for <code>prepare_user_data.py</code></b></summary>

| Flag | Required | Meaning |
|------|----------|---------|
| `--name` | yes | Dataset name (e.g. `acme`); used as `--datasets acme` in step 2. Can't be a built-in name. |
| `--train-csv`/`--val-csv`/`--test-csv` | one split | Your three split files. |
| `--splits-dir` | many folds | Instead of the three above: a directory of fold subdirs (see Option 2). |
| `--smiles-col` | no (default `smiles`) | Name of the SMILES column in your CSVs. |
| `--target-cols` | yes | Which column(s) are the labels, space-separated (e.g. `activity`, or `logD solubility`). |
| `--task` | yes | `cls` (0/1 labels) or `reg` (numeric). Sets the loss + default metric. |
| `--metric` | no | Force a metric: `roc-auc`/`pr-auc`/`mae`/`rmse`/`spearman`. Omit to derive (roc-auc for cls, rmse for reg). |
</details>

---

## Step 2 — Run

A run is controlled by two choices: **`--protocols`** (which split) and **`--phases`** (default vs
tuned hyperparameters).

### Protocols — which split

- **`custom`** — your prepared splits from step 1. Use this for your own data.
- **`v1_preshuffle`** and **`v2_astartes`** — our two split schemes for the benchmark datasets.
  **Both are Bemis–Murcko scaffold splits** (molecules are grouped by scaffold and whole scaffold
  groups go to train/val/test, so the test set contains scaffolds unseen in training — **neither is
  a random split**):
  - **`v1_preshuffle`** — scaffold buckets sorted **largest → smallest**, equal-sized buckets
    shuffled by the seed, then filled greedily into train → val → test.
  - **`v2_astartes`** — scaffold clusters sorted by size and assigned as **whole clusters** to
    train/val/test via astartes extrapolative sampling (large clusters grouped together, small
    together).
  Each runs 3 seeds × 5-model ensemble. (`tdc` ADMET datasets use their 5 fixed splits.)

### Phases — default vs tuned hyperparameters

- **`--phases default`** — trains each method's **default hyperparameters**: one 5-model ensemble
  per fold (5 models with different seeds, averaged). Fast — use it to just get a number, or with a
  checkpoint.
- **`--phases hp_search hp_final`** — **tunes** the grid: every config is run on **all** folds and
  scored by its **mean validation across folds** (TDC-style; a single split just uses that one val),
  the best config is picked, then trained as a 5-model ensemble on **every** fold.

In both, a "run" is one (fold × ensemble-member); multiple folds → **per-fold results + mean±std
across folds**, automatically.

### Commands

**Our methods on YOUR split — default HP:**
```bash
python -m orchestrator.run_benchmark --methods chemeleon chemprop2 molclr molformer \
  --datasets acme --protocols custom --phases default --gpus 0,1,2,3 --jobs-per-gpu 1
```
**Our methods on YOUR split — tuned (HP search → final):**
```bash
python -m orchestrator.run_benchmark --methods chemeleon molclr motil \
  --datasets acme --protocols custom --phases hp_search hp_final --gpus 0,1,2,3 --jobs-per-gpu 2
```
**Our methods on OUR benchmark datasets** (e.g. `bace esol tox21`), both scaffold protocols:
```bash
python -m orchestrator.run_benchmark --methods chemeleon chemprop2 molclr \
  --datasets bace esol --protocols v1_preshuffle v2_astartes \
  --phases hp_search hp_final --gpus 0,1,2,3 --jobs-per-gpu 2
```
(`molformer` needs `--jobs-per-gpu 1`; other methods can use 2–4.)

---

## Output — what you get and where

```
results/<method>/<name>/<protocol>/<phase>/
  seed0_em0/   metrics.json  pred_test.npy  labels_test.npy  pred_val.npy  done.flag
  seed0_em1/   ...            # the 5 ensemble members for fold 0
  ...
  seed1_em0/   ...            # fold 1 (only if you gave multiple splits)
  _summary.json               # mean ± std across folds
```

- **Per run** (`seed<fold>_em<member>/metrics.json`): that model's `test_metric`, `val_metric`,
  per-target scores, and the exact HPs used. Raw `pred_test.npy` + `labels_test.npy` are alongside
  it if you want to compute your own metrics.
- **Per method** (`_summary.json`): the pipeline **already ensemble-averages the 5 members per fold
  and reports mean ± std across folds** (`agg_am: {mean, std, n}`). For a single method you do **not**
  aggregate anything yourself.
- **Across methods** — one comparison table (each method writes its own `_summary.json`; this gathers
  them):
  ```
  python collect_results.py --dataset acme --protocol custom --phase default \
    --methods chemeleon molclr molformer --out acme.csv
  ```
  prints (and optionally writes the CSV):
  ```
  method          test_metric     std   n_folds
  chemeleon           0.8421    0.0111     3
  molclr              0.8090    0.0203     3
  ```
  Higher is better for AUC (cls); lower for RMSE/MAE (reg). Use `--phase hp_final` for tuned runs.

---

## Learning curves (low-data regime)

To see how each method scales with training-set size, add `--learning-curve-sizes` at **prepare**
time. It makes **class-stratified** (cls) or range-covering (reg) subsamples of your training set at
each size — validation/test held fixed — with `--lc-repeats` random repeats per size (for error bars):
```
python prepare_user_data.py --name acme \
  --train-csv train.csv --val-csv val.csv --test-csv test.csv \
  --smiles-col SMILES --target-cols activity --task cls \
  --learning-curve-sizes 100 200 500 1000 --lc-repeats 3
```
This creates the subsampled splits under `splits/acme/custom__<size>_seed<repeat>/`. Then train each
size and collect the curve — `run_learning_curve.py` reuses exactly those splits:
```
python runners/run_learning_curve.py --methods chemeleon molclr --datasets acme \
  --protocols custom --fractions 100 200 500 1000 --seeds 0 1 2 --gpus 0,1,2,3

python collect_results.py --dataset acme --protocol custom --learning-curve \
  --methods chemeleon molclr --out acme_curve.csv --plot acme_curve.png
```
- `--fractions` are just the sizes you prepared (`100 200 500 1000`).
- `--seeds` are the **repeat indices** (`0 1 2` = the 3 `--lc-repeats`).
- The collector prints `method × train_size → mean ± std`, writes the CSV, and `--plot` saves a
  learning-curve figure.

---

## Using your own fine-tuned checkpoint

Load your weights instead of the default pretrained by **prefixing the method's env var** — no
extra flag, no code change. Works on **your split or ours**, in any run above.

On **your** split:
```bash
CHEMELEON_CKPT=/path/your_chemeleon.pt \
python -m orchestrator.run_benchmark --methods chemeleon \
  --datasets acme --protocols custom --phases default --gpus 0,1,2,3
```
On **our** split (e.g. score your model against the benchmark):
```bash
CHEMELEON_CKPT=/path/your_chemeleon.pt \
python -m orchestrator.run_benchmark --methods chemeleon \
  --datasets bace --protocols v2_astartes --phases default --gpus 0,1,2,3
```

| Method | env var | expects |
|--------|---------|---------|
| chemeleon / chemeleon_nofp | `CHEMELEON_CKPT` | a `.pt` file |
| molclr | `MOLCLR_CKPT_DIR` | a **directory** holding `checkpoints/model.pth` |
| molfcl | `MOLFCL_CKPT` | a `.pkl` |
| motil | `MOTIL_CKPT` | a `.pkl` |
| molformer | `MOLFORMER_CKPT` | a `.ckpt` (PyTorch-Lightning) |
| chemprop2 / chemprop2_nofp | — | trains from scratch; no pretrained to swap |

> **Limitation:** the pipeline is *train → predict → discard* — it does not currently save the
> fine-tuned model. "Fine-tune once, reuse on other splits later" needs a small change to persist
> the checkpoint; ask if you need the trained model back.

---

## GPU distribution

`--gpus 0,1,2,3 --jobs-per-gpu N` runs **N jobs per GPU** = `len(gpus) × N` concurrently. Each job
(one fold × ensemble-member) grabs a free GPU, trains, and releases it — no GPU is over-packed.

- **`molformer` must use `--jobs-per-gpu 1`** (two molformer processes on one GPU wedge). Others
  tolerate 2–4.
- Long-SMILES molformer runs may need `MOLFORMER_MICRO_BATCH=16` if you hit out-of-memory.
