# Custom-protocol wiring — STATUS: ✅ APPLIED & VERIFIED

This patch has been applied to `configs.py`, `run_phase.py`, and `run_benchmark.py` and verified
(imports clean, existing `v2_astartes`/`tdc` protocols unchanged, custom job-expansion produces
the right per-fold seeds). Kept below as a changelog / re-apply reference — **do not re-apply.**

Goal: after this, user data prepared by `prepare_user_data.py` runs through the **normal**
`run_benchmark`/`run_phase` with `--protocols custom`. No separate runner, no duplicated
GPU/HP logic. Checkpoints need **no code change** (env vars already propagate — see §4).

Total: 4 tiny edits. All additive/backward-compatible (existing protocols unchanged).

---

## Edit 1 — `methods/configs.py`: add `custom` to the seed maps

In `EVAL_SEEDS`, add a line:
```python
    "tdc":            [1, 2, 3, 4, 5],
    "custom":         [0],          # overridden per-fold by run_phase (globs custom_seed*)
}
```
In `HP_SEED`, add a line:
```python
    "tdc":            [1, 2, 3, 4, 5],
    "custom":         0,            # fallback; run_phase runs custom HP on ALL folds (TDC-style mean-val)
}
```

## Edit 2 — `runners/run_phase.py`: derive eval seeds per user fold

Add this helper just above `build_default_jobs` (~line 85):
```python
def _eval_seeds(dataset, protocol):
    """Eval seeds for a protocol. For 'custom', one per user fold (splits/<ds>/custom_seed*/)."""
    if protocol == "custom":
        folds = sorted(int(p.name.replace("custom_seed", ""))
                       for p in (PIPELINE / "splits" / dataset).glob("custom_seed*"))
        return folds or [0]
    return EVAL_SEEDS[protocol]
```
Then in **`build_default_jobs`** change:
```python
    seeds = EVAL_SEEDS[protocol]          # OLD
    seeds = _eval_seeds(dataset, protocol)  # NEW
```
and the identical line in **`build_hp_final_jobs`**:
```python
    seeds = EVAL_SEEDS[protocol]          # OLD
    seeds = _eval_seeds(dataset, protocol)  # NEW
```
(`build_hp_search_jobs` needs no change — it reads `HP_SEED["custom"] = 0` from Edit 1.)

## Edit 3 — `runners/run_phase.py`: allow `custom` on the CLI

In the argparse for `--protocols`:
```python
    choices=["v1_det", "v1_preshuffle", "v2_astartes", "tdc"],            # OLD
    choices=["v1_det", "v1_preshuffle", "v2_astartes", "tdc", "custom"],  # NEW
```

## Edit 4 — `orchestrator/run_benchmark.py`: allow `custom` on the CLI

```python
    choices=PROTOCOLS + ["tdc"],            # OLD
    choices=PROTOCOLS + ["tdc", "custom"],  # NEW
```

---

## §4 — Checkpoints need NO code change

`run_phase.py::run_job` builds each training subprocess env with `os.environ.copy()`, and
`run_benchmark` launches `run_phase` inheriting the environment. So any checkpoint env var you
export propagates all the way to `train_one.py`. Just prefix the command:

```bash
CHEMELEON_CKPT=/path/their_chemeleon.pt \
python -m orchestrator.run_benchmark --methods chemeleon \
  --datasets user_acme --protocols custom --phases default --gpus 0,1,2,3
```
(Per-method env vars: `CHEMELEON_CKPT`, `MOLCLR_CKPT_DIR` [a dir], `MOLFCL_CKPT`, `MOTIL_CKPT`,
`MOLFORMER_CKPT`. chemprop2 trains from scratch.)

---

## After applying — verify (must print cleanly)

```bash
cd /data/rbg/users/yxie25/molclr_chemprop/clean_pipeline_v1
python -c "import sys; sys.path.insert(0,'methods'); import configs; print('configs OK', configs.HP_SEED['custom'])"
python -c "import ast; ast.parse(open('runners/run_phase.py').read()); print('run_phase OK')"
python -c "import ast; ast.parse(open('orchestrator/run_benchmark.py').read()); print('run_benchmark OK')"
# end-to-end on a tiny prepared dataset (1 model, 2 epochs):
python prepare_user_data.py --name smoke --train-csv t.csv --val-csv v.csv --test-csv te.csv \
  --target-cols activity --task cls
python -m orchestrator.run_benchmark --methods chemeleon --datasets user_smoke \
  --protocols custom --phases default --gpus 0 --epochs-default 2
```

## Usage after patch (this replaces `pharma_run.py`)

```bash
# 1. adapt the user's data (no re-split):
python prepare_user_data.py --name acme --train-csv train.csv --val-csv val.csv \
  --test-csv test.csv --smiles-col SMILES --target-cols activity --task cls

# 2a. run with default HPs (fast) — or with their checkpoint via env prefix:
python -m orchestrator.run_benchmark --methods chemeleon --datasets user_acme \
  --protocols custom --phases default --gpus 0,1,2,3

# 2b. full HP search + final on their split:
python -m orchestrator.run_benchmark --methods chemeleon --datasets user_acme \
  --protocols custom --phases hp_search hp_final --gpus 0,1,2,3
```
`--phases default` = one run per fold × ensemble with the method's default HPs.
`--phases hp_search hp_final` = run every config on ALL folds, pick best by mean validation
(TDC-style), then eval the winner on every fold.
molformer still needs `--jobs-per-gpu 1`.
