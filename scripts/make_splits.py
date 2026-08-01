"""
Generate scaffold splits for the cleaned MoleculeNet datasets under 3 protocols.

Protocols
---------
A) "v1_det"        - chemprop v1.6.1 scaffold_split(balanced=False).
                     Sort scaffold buckets by size DESCENDING, fill train -> val -> test.
                     Fully deterministic (seed silently ignored). One split per dataset.

B) "v1_preshuffle" - SAME chemprop v1.6.1 algorithm as (A), but with the input rows
                     pre-shuffled by `seed` BEFORE running. Stable sort means only ties
                     (same-size buckets, mostly singletons) reorder. Three seeds: 0, 1, 2.

C) "v2_astartes"   - chemprop v2 scaffold split, which delegates to
                     astartes.molecules.train_val_test_split_molecules(sampler="scaffold",
                     random_state=seed). Sorts ascending, shuffles small clusters
                     (<= max_shufflable_size = min(n_val, n_test)), fills test -> val -> train.
                     Three seeds: 0, 1, 2.

Sources used directly (no re-implementation of scaffold computation)
--------------------------------------------------------------------
- chemprop v1.6.1: clean_pipeline_v1 imports `generate_scaffold` from
    /data/rbg/users/yxie25/molclr_chemprop/chemprop_v1_backup/data/scaffold.py
  (this is the authoritative v1 implementation, copied from chemprop 1.6.1 source).
  The bucket-sort + greedy-fill loop here is byte-equivalent to lines 91-118 of that file.
- astartes 1.3.3: `from astartes.molecules import train_val_test_split_molecules`.
  Internally calls the Scaffold sampler in astartes/samplers/extrapolation/scaffold.py and
  the extrapolative-fill in astartes/main.py:_extrapolative_sampling.

Outputs
-------
clean_pipeline_v1/splits/<dataset>/<protocol>_seed<S>/{train_idx.npy, val_idx.npy, test_idx.npy}
clean_pipeline_v1/splits/_stats.json   (machine-readable summary)
"""

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from random import Random

import numpy as np
import pandas as pd

# --- Add chemprop v1 backup to path so we can import its scaffold helper ----
CHEMPROP_V1_DIR = "/data/rbg/users/yxie25/molclr_chemprop/chemprop_v1_backup"
sys.path.insert(0, os.path.dirname(CHEMPROP_V1_DIR))
# We bypass the heavy `MoleculeDataset` interface in chemprop v1's `scaffold_split`
# and call its `generate_scaffold` directly. The downstream bucket sort + greedy
# fill loop below is identical to chemprop v1.6.1 source (verified line-for-line).
try:
    from chemprop_v1_backup.data.scaffold import generate_scaffold as _v1_generate_scaffold  # type: ignore
except Exception:
    # Fallback: replicate the exact one-liner that chemprop v1.6.1 uses
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    def _v1_generate_scaffold(smiles_or_mol, include_chirality: bool = False) -> str:
        if isinstance(smiles_or_mol, str):
            mol = Chem.MolFromSmiles(smiles_or_mol)
        else:
            mol = smiles_or_mol
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)

# astartes (chemprop v2 scaffold split delegates to this)
from astartes.molecules import train_val_test_split_molecules

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
ROOT = Path("/data/rbg/users/yxie25/molclr_chemprop/clean_pipeline_v1")
CLEANED = ROOT / "cleaned"
SPLITS = ROOT / "splits"
SPLITS.mkdir(parents=True, exist_ok=True)

DATASETS = [
    ("bbbp", "cls"), ("bace", "cls"), ("hiv", "cls"),
    ("clintox", "cls"), ("tox21", "cls"), ("sider", "cls"),
    ("esol", "reg"), ("freesolv", "reg"), ("lipo", "reg"),
    ("qm7", "reg"), ("qm8", "reg"), ("qm9", "reg"),
]
SEEDS = [0, 1, 2]               # eval seeds for final reporting
HP_SEED = 3                      # HP-only split (Option B2: decouple HP val from eval test)
ALL_SEEDS = SEEDS + [HP_SEED]
SIZES = (0.8, 0.1, 0.1)


# ---------------------------------------------------------------------------
# Splitters
# ---------------------------------------------------------------------------
def chemprop_v1_balanced_false(smiles, sizes=SIZES):
    """
    Verbatim of chemprop v1.6.1 `scaffold_split(balanced=False)`, decoupled from
    MoleculeDataset. Fully deterministic. The seed argument of the v1 function is
    only used for `balanced=True` and is silently unused here.
    """
    n_total = len(smiles)
    n_train, n_val, n_test = sizes[0] * n_total, sizes[1] * n_total, sizes[2] * n_total
    scaffold_to_indices = defaultdict(set)
    for i, s in enumerate(smiles):
        scaffold_to_indices[_v1_generate_scaffold(s)].add(i)
    # chemprop v1 line 105-107: sort largest -> smallest
    index_sets = sorted([list(v) for v in scaffold_to_indices.values()],
                        key=lambda x: len(x), reverse=True)
    train, val, test = [], [], []
    for s in index_sets:
        if len(train) + len(s) <= n_train:
            train += s
        elif len(val) + len(s) <= n_val:
            val += s
        else:
            test += s
    return np.array(train, dtype=np.int64), np.array(val, dtype=np.int64), np.array(test, dtype=np.int64)


def chemprop_v1_preshuffle(smiles, seed, sizes=SIZES):
    """
    Same algorithm as chemprop_v1_balanced_false, but with input rows pre-shuffled
    by `seed`. Returns indices in the ORIGINAL (unshuffled) numbering.

    Only ties in `sorted(..., reverse=True)` are affected (`sorted` is stable).
    Distinct-size buckets keep their position; same-size buckets reorder.
    """
    n = len(smiles)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)  # shuffled[i] -> original perm[i]
    smiles_shuffled = [smiles[i] for i in perm]
    tr_s, va_s, te_s = chemprop_v1_balanced_false(smiles_shuffled, sizes=sizes)
    # map shuffled positions back to original row indices
    return perm[tr_s], perm[va_s], perm[te_s]


def chemprop_v2_astartes(smiles, seed, sizes=SIZES):
    """
    Chemprop v2 scaffold split = `astartes.molecules.train_val_test_split_molecules`
    with sampler="scaffold". Returns indices in the original numbering.
    """
    out = train_val_test_split_molecules(
        np.array(smiles, dtype=object),
        train_size=sizes[0], val_size=sizes[1], test_size=sizes[2],
        sampler="scaffold",
        random_state=seed,
        return_indices=True,
    )
    # 9-tuple when y is None: (X_tr, X_val, X_te, ?, ?, ?, idx_tr, idx_val, idx_te)
    tr, va, te = np.array(out[-3]), np.array(out[-2]), np.array(out[-1])
    return tr.astype(np.int64), va.astype(np.int64), te.astype(np.int64)


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------
def compute_scaffold_index(smiles):
    scaf_to_idx = defaultdict(list)
    for i, s in enumerate(smiles):
        scaf_to_idx[_v1_generate_scaffold(s)].append(i)
    idx_to_scaf = {}
    for sc, idxs in scaf_to_idx.items():
        for i in idxs:
            idx_to_scaf[i] = sc
    return scaf_to_idx, idx_to_scaf


def class_stats_per_target(y, mask, idxs):
    """For each target column, return (n_pos, n_neg, n_nan) restricted to `idxs`."""
    if len(idxs) == 0:
        return []
    yi = y[idxs]
    mi = mask[idxs]
    out = []
    for t in range(y.shape[1]):
        col = yi[:, t]
        m = mi[:, t]
        n_valid = int(m.sum())
        n_pos = int(((col == 1) & m).sum())
        n_neg = int(((col == 0) & m).sum())
        out.append({"n_valid": n_valid, "n_pos": n_pos, "n_neg": n_neg,
                    "pos_frac": (n_pos / n_valid) if n_valid > 0 else None})
    return out


def reg_stats_per_target(y, mask, idxs):
    if len(idxs) == 0:
        return []
    yi = y[idxs]
    mi = mask[idxs]
    out = []
    for t in range(y.shape[1]):
        col = yi[:, t][mi[:, t].astype(bool)]
        if len(col) == 0:
            out.append(None)
            continue
        out.append({"n_valid": int(len(col)),
                    "mean": float(col.mean()), "std": float(col.std()),
                    "min": float(col.min()), "max": float(col.max())})
    return out


def split_stats(name, tr, va, te, smiles, scaf_to_idx, idx_to_scaf, y, mask, task_type):
    tr_scafs = {idx_to_scaf[i] for i in tr}
    va_scafs = {idx_to_scaf[i] for i in va}
    te_scafs = {idx_to_scaf[i] for i in te}
    te_scaf_sizes = [len(scaf_to_idx[idx_to_scaf[i]]) for i in te]
    tr_scaf_sizes = [len(scaf_to_idx[idx_to_scaf[i]]) for i in tr]
    s = {
        "name": name,
        "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
        "n_scaffolds_train": len(tr_scafs), "n_scaffolds_val": len(va_scafs),
        "n_scaffolds_test": len(te_scafs),
        "scaffold_overlap_train_test": len(tr_scafs & te_scafs),
        "scaffold_overlap_train_val": len(tr_scafs & va_scafs),
        "test_singleton_count": int(sum(1 for x in te_scaf_sizes if x == 1)),
        "test_singleton_frac": float(sum(1 for x in te_scaf_sizes if x == 1) / max(len(te), 1)),
        "test_scaf_median": float(np.median(te_scaf_sizes)) if te_scaf_sizes else None,
        "test_scaf_max": int(max(te_scaf_sizes)) if te_scaf_sizes else None,
        "train_scaf_max": int(max(tr_scaf_sizes)) if tr_scaf_sizes else None,
    }
    if task_type == "cls":
        s["test_target_stats"] = class_stats_per_target(y, mask, te)
    else:
        s["test_target_stats"] = reg_stats_per_target(y, mask, te)
    return s


def jaccard(a, b):
    sa, sb = set(np.asarray(a).tolist()), set(np.asarray(b).tolist())
    if not (sa | sb):
        return 1.0
    return len(sa & sb) / len(sa | sb)


def scaffold_jaccard(idx_a, idx_b, idx_to_scaf):
    sa = {idx_to_scaf[i] for i in idx_a}
    sb = {idx_to_scaf[i] for i in idx_b}
    if not (sa | sb):
        return 1.0
    return len(sa & sb) / len(sa | sb)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    all_stats = {}
    for ds_name, task_type in DATASETS:
        t0 = time.time()
        csv_path = CLEANED / f"{ds_name}.csv"
        mask_path = CLEANED / f"{ds_name}_mask.npy"
        meta_path = CLEANED / f"{ds_name}.meta.json"

        df = pd.read_csv(csv_path)
        with open(meta_path) as f:
            meta = json.load(f)
        target_cols = meta["target_columns"]
        smiles = df["smiles"].tolist()
        y = df[target_cols].to_numpy(dtype=np.float64)
        mask = np.load(mask_path).astype(bool)
        N = len(df)

        scaf_to_idx, idx_to_scaf = compute_scaffold_index(smiles)
        n_singletons = sum(1 for v in scaf_to_idx.values() if len(v) == 1)
        max_cluster = max(len(v) for v in scaf_to_idx.values())

        ds_stats = {
            "n_total": N, "n_targets": len(target_cols),
            "task_type": task_type,
            "n_scaffolds": len(scaf_to_idx),
            "n_singletons": n_singletons,
            "singleton_frac": n_singletons / len(scaf_to_idx),
            "max_cluster_size": max_cluster,
            "splits": {},
        }

        # --- Protocol A: v1 deterministic (1 split, no seed) ---------------
        tr, va, te = chemprop_v1_balanced_false(smiles)
        outdir = SPLITS / ds_name / "v1_det_seed0"
        outdir.mkdir(parents=True, exist_ok=True)
        np.save(outdir / "train_idx.npy", tr)
        np.save(outdir / "val_idx.npy", va)
        np.save(outdir / "test_idx.npy", te)
        ds_stats["splits"]["v1_det_seed0"] = split_stats(
            "v1_det_seed0", tr, va, te, smiles, scaf_to_idx, idx_to_scaf, y, mask, task_type
        )

        # --- Protocol B: v1 + pre-shuffle, seeds 0/1/2 + HP-only seed3 -----
        v1ps_test = {}
        for seed in ALL_SEEDS:
            tr, va, te = chemprop_v1_preshuffle(smiles, seed=seed)
            outdir = SPLITS / ds_name / f"v1_preshuffle_seed{seed}"
            outdir.mkdir(parents=True, exist_ok=True)
            np.save(outdir / "train_idx.npy", tr)
            np.save(outdir / "val_idx.npy", va)
            np.save(outdir / "test_idx.npy", te)
            ds_stats["splits"][f"v1_preshuffle_seed{seed}"] = split_stats(
                f"v1_preshuffle_seed{seed}", tr, va, te, smiles, scaf_to_idx, idx_to_scaf, y, mask, task_type
            )
            v1ps_test[seed] = te

        # --- Protocol C: astartes v2, seeds 0/1/2 + HP-only seed3 ----------
        v2_test = {}
        for seed in ALL_SEEDS:
            tr, va, te = chemprop_v2_astartes(smiles, seed=seed)
            outdir = SPLITS / ds_name / f"v2_astartes_seed{seed}"
            outdir.mkdir(parents=True, exist_ok=True)
            np.save(outdir / "train_idx.npy", tr)
            np.save(outdir / "val_idx.npy", va)
            np.save(outdir / "test_idx.npy", te)
            ds_stats["splits"][f"v2_astartes_seed{seed}"] = split_stats(
                f"v2_astartes_seed{seed}", tr, va, te, smiles, scaf_to_idx, idx_to_scaf, y, mask, task_type
            )
            v2_test[seed] = te

        # --- Pairwise Jaccard across seeds ---------------------------------
        ds_stats["jaccard_test_idx"] = {
            "v1_preshuffle_01": jaccard(v1ps_test[0], v1ps_test[1]),
            "v1_preshuffle_02": jaccard(v1ps_test[0], v1ps_test[2]),
            "v1_preshuffle_12": jaccard(v1ps_test[1], v1ps_test[2]),
            "v2_astartes_01":   jaccard(v2_test[0], v2_test[1]),
            "v2_astartes_02":   jaccard(v2_test[0], v2_test[2]),
            "v2_astartes_12":   jaccard(v2_test[1], v2_test[2]),
            # HP-test coupling sanity (lower = HP val less correlated with eval test)
            "v1_preshuffle_03": jaccard(v1ps_test[0], v1ps_test[3]),
            "v1_preshuffle_13": jaccard(v1ps_test[1], v1ps_test[3]),
            "v1_preshuffle_23": jaccard(v1ps_test[2], v1ps_test[3]),
            "v2_astartes_03":   jaccard(v2_test[0], v2_test[3]),
            "v2_astartes_13":   jaccard(v2_test[1], v2_test[3]),
            "v2_astartes_23":   jaccard(v2_test[2], v2_test[3]),
        }
        ds_stats["jaccard_test_scaffolds"] = {
            "v1_preshuffle_01": scaffold_jaccard(v1ps_test[0], v1ps_test[1], idx_to_scaf),
            "v1_preshuffle_02": scaffold_jaccard(v1ps_test[0], v1ps_test[2], idx_to_scaf),
            "v1_preshuffle_12": scaffold_jaccard(v1ps_test[1], v1ps_test[2], idx_to_scaf),
            "v2_astartes_01":   scaffold_jaccard(v2_test[0], v2_test[1], idx_to_scaf),
            "v2_astartes_02":   scaffold_jaccard(v2_test[0], v2_test[2], idx_to_scaf),
            "v2_astartes_12":   scaffold_jaccard(v2_test[1], v2_test[2], idx_to_scaf),
            "v1_preshuffle_03": scaffold_jaccard(v1ps_test[0], v1ps_test[3], idx_to_scaf),
            "v1_preshuffle_13": scaffold_jaccard(v1ps_test[1], v1ps_test[3], idx_to_scaf),
            "v1_preshuffle_23": scaffold_jaccard(v1ps_test[2], v1ps_test[3], idx_to_scaf),
            "v2_astartes_03":   scaffold_jaccard(v2_test[0], v2_test[3], idx_to_scaf),
            "v2_astartes_13":   scaffold_jaccard(v2_test[1], v2_test[3], idx_to_scaf),
            "v2_astartes_23":   scaffold_jaccard(v2_test[2], v2_test[3], idx_to_scaf),
        }
        ds_stats["elapsed_sec"] = round(time.time() - t0, 1)
        all_stats[ds_name] = ds_stats
        print(f"[{ds_name}] N={N}, scaffolds={len(scaf_to_idx)} (sing {n_singletons}), elapsed {ds_stats['elapsed_sec']}s")

    with open(SPLITS / "_stats.json", "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nWrote {SPLITS}/_stats.json")


if __name__ == "__main__":
    main()
