"""
prepare_user_data.py — turn user-supplied train/val/test CSVs into the pipeline's
internal format so the EXISTING orchestrator can run on them.

This is a FORMAT ADAPTER, not a splitter. Our methods read one CSV plus three
index arrays; you give three CSVs. This records which rows are train/val/test and
writes them out. Your split membership is preserved 1:1 — nothing is reshuffled or
moved between train/val/test. The only optional change is dropping unparseable
SMILES (reported), which would otherwise crash training.

It writes (where <name> is exactly what you pass to --name):
  cleaned/<name>.csv           # molecules + labels (one pooled table)
  cleaned/<name>.meta.json     # task type, target columns, metric
  splits/<name>/custom_seed<i>/{train,val,test}_idx.npy   # one per fold

Then run the normal pipeline with the `custom` protocol (see README_PHARMA.md):
  python -m orchestrator.run_benchmark --methods chemeleon \
    --datasets <name> --protocols custom --gpus 0,1,2,3
  # (to load your own checkpoint, prefix e.g. CHEMELEON_CKPT=/path/model.pt)

Inputs — one split:
  --train-csv --val-csv --test-csv
Inputs — many folds (e.g. rolling time splits):
  --splits-dir DIR   where DIR/<fold>/{train,val,test}.csv
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PIPELINE = Path(__file__).resolve().parent


def _canonicalize_and_flag(smiles_list):
    """Return (canonical-or-None list, n_invalid). Unparseable / 0-atom -> None."""
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    out, n_invalid = [], 0
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        if mol is None or mol.GetNumHeavyAtoms() == 0:
            out.append(None); n_invalid += 1
        else:
            out.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
    return out, n_invalid


def _read_split_csvs(args):
    """Return [(fold_name, {'train':df,'val':df,'test':df})]; renames the smiles column."""
    def load(path):
        df = pd.read_csv(path)
        if args.smiles_col not in df.columns:
            sys.exit(f"[prep] '{args.smiles_col}' not in {path}; columns: {list(df.columns)}")
        return df.rename(columns={args.smiles_col: "smiles"})

    if args.splits_dir:
        folds = []
        for sub in sorted(p for p in Path(args.splits_dir).iterdir() if p.is_dir()):
            need = {s: sub / f"{s}.csv" for s in ("train", "val", "test")}
            missing = [str(p) for p in need.values() if not p.exists()]
            if missing:
                print(f"[prep] skip fold '{sub.name}': missing {missing}"); continue
            folds.append((sub.name, {s: load(p) for s, p in need.items()}))
        if not folds:
            sys.exit(f"[prep] no fold subdirs with train/val/test.csv under {args.splits_dir}")
        return folds
    for role in ("train_csv", "val_csv", "test_csv"):
        if not getattr(args, role):
            sys.exit("[prep] provide --train-csv/--val-csv/--test-csv OR --splits-dir")
    return [("fold0", {"train": load(args.train_csv),
                       "val": load(args.val_csv), "test": load(args.test_csv)})]


def _stratified_subsample(train_idx, labels, n, task, rng):
    """~n indices from train_idx, keeping class balance (cls) or target-range coverage (reg)."""
    train_idx = np.asarray(train_idx)
    if n >= len(train_idx):
        return train_idx.copy()
    labels = np.asarray(labels, dtype=float)
    if task == "cls":
        groups = [train_idx[labels == c] for c in np.unique(labels[~np.isnan(labels)])]
    else:
        order = train_idx[np.argsort(labels)]
        groups = [g for g in np.array_split(order, min(10, len(order))) if len(g)]
    picked = []
    for g in groups:
        k = max(1, int(round(n * len(g) / len(train_idx))))
        picked.extend(rng.choice(g, size=min(k, len(g)), replace=False))
    picked = np.array(picked, dtype=np.int64)
    if len(picked) > n:
        picked = rng.choice(picked, size=n, replace=False)
    return picked


def prepare(args):
    dataset = args.name
    meta_path = PIPELINE / "cleaned" / f"{dataset}.meta.json"
    if meta_path.exists() and not json.loads(meta_path.read_text()).get("user_created"):
        sys.exit(f"[prep] '{dataset}' is a built-in benchmark dataset — pick a different --name.")
    targets = args.target_cols
    raw_folds = _read_split_csvs(args)

    # One shared canonical-SMILES -> row pool (the CSV store), like our own datasets:
    # one CSV, many split subdirs pointing into it. Identical molecules across folds
    # map to the same row. Split MEMBERSHIP per fold is preserved exactly.
    pool_rows, pool_index = [], {}
    fold_idx, n_invalid_total = [], 0
    report_folds = []

    for name, dfs in raw_folds:
        idx_of = {"train": [], "val": [], "test": []}
        seen_role, overlap = {}, 0
        for role in ("train", "val", "test"):
            df = dfs[role]
            for col in targets:
                if col not in df.columns:
                    sys.exit(f"[prep] target '{col}' missing from fold '{name}' {role}.csv")
            canon, n_bad = _canonicalize_and_flag(df["smiles"].tolist())
            n_invalid_total += n_bad
            for row_i, smiles in enumerate(canon):
                if smiles is None:
                    continue
                if smiles not in pool_index:
                    pool_index[smiles] = len(pool_rows)
                    row = {"smiles": smiles}
                    row.update({t: df.iloc[row_i][t] for t in targets})
                    pool_rows.append(row)
                rid = pool_index[smiles]
                if smiles in seen_role and seen_role[smiles] != role:
                    overlap += 1
                seen_role[smiles] = role
                idx_of[role].append(rid)
        fold_idx.append((name, idx_of))
        report_folds.append((name, len(idx_of["train"]), len(idx_of["val"]),
                             len(idx_of["test"]), overlap))
        if overlap:
            print(f"[prep] WARNING fold '{name}': {overlap} molecule(s) appear in more than "
                  f"one of train/val/test (possible leakage) — kept as-is; fix upstream if unintended.")

    cleaned_df = pd.DataFrame(pool_rows, columns=["smiles"] + targets)
    (PIPELINE / "cleaned").mkdir(exist_ok=True)
    cleaned_df.to_csv(PIPELINE / "cleaned" / f"{dataset}.csv", index=False)

    meta = {"dataset": dataset, "task_type": args.task, "target_columns": targets,
            "n_targets": len(targets), "n_after": len(cleaned_df), "user_created": True}
    if args.metric:
        meta["metric"] = args.metric
        meta["source"] = "tdc"   # tells scoring to use meta['metric'] verbatim
    (PIPELINE / "cleaned" / f"{dataset}.meta.json").write_text(json.dumps(meta, indent=2))

    for i, (name, idx_of) in enumerate(fold_idx):
        sub = PIPELINE / "splits" / dataset / f"custom_seed{i}"
        sub.mkdir(parents=True, exist_ok=True)
        for role, fname in (("train", "train_idx"), ("val", "val_idx"), ("test", "test_idx")):
            np.save(sub / f"{fname}.npy", np.array(idx_of[role], dtype=np.int64))

    # Optional learning-curve subsets: nested-ish stratified samples of fold 0's train at each
    # requested absolute size (val/test held fixed), as pseudo-protocols custom__n<size>_seed<r>.
    if args.learning_curve_sizes:
        fold0 = fold_idx[0][1]
        y = cleaned_df.iloc[fold0["train"]][targets[0]].to_numpy()
        made = []
        for size in sorted(set(args.learning_curve_sizes)):
            tag = str(size)   # plain size, so run_learning_curve --fractions 100 200 ... matches
            for rep in range(args.lc_repeats):
                rng = np.random.default_rng(size * 1000 + rep)
                sub_train = _stratified_subsample(fold0["train"], y, size, args.task, rng)
                d = PIPELINE / "splits" / dataset / f"custom__{tag}_seed{rep}"
                d.mkdir(parents=True, exist_ok=True)
                np.save(d / "train_idx.npy", np.asarray(sub_train, dtype=np.int64))
                np.save(d / "val_idx.npy", np.array(fold0["val"], dtype=np.int64))
                np.save(d / "test_idx.npy", np.array(fold0["test"], dtype=np.int64))
            made.append(f"{tag}={min(size, len(fold0['train']))}")
        print(f"[prep] learning-curve subsets ({args.lc_repeats} repeats each): " + ", ".join(made))

    print(f"\n[prep] dataset '{dataset}': {len(cleaned_df)} unique molecules, "
          f"{n_invalid_total} invalid SMILES dropped, {len(fold_idx)} fold(s).")
    for name, ntr, nva, nte, ov in report_folds:
        print(f"       fold {name} -> custom_seed{[n for n,_,_,_,_ in report_folds].index(name)}: "
              f"train {ntr} / val {nva} / test {nte}" + (f"  [leakage {ov}]" if ov else ""))
    print(f"\n[prep] next, run the normal pipeline on it:")
    print(f"       python -m orchestrator.run_benchmark --methods <METHOD> \\")
    print(f"         --datasets {dataset} --protocols custom [--checkpoint <FILE>] --gpus 0,1,2,3")
    return dataset


def main():
    p = argparse.ArgumentParser(description="Adapt user CSVs into the pipeline data format (no re-splitting).")
    p.add_argument("--name", required=True,
                   help="dataset tag; writes cleaned/<name>.csv + splits/<name>/ (must not collide with a built-in dataset).")
    p.add_argument("--train-csv"); p.add_argument("--val-csv"); p.add_argument("--test-csv")
    p.add_argument("--splits-dir", help="dir of fold subdirs each with train/val/test.csv")
    p.add_argument("--smiles-col", default="smiles")
    p.add_argument("--target-cols", nargs="+", required=True)
    p.add_argument("--task", required=True, choices=["cls", "reg"])
    p.add_argument("--metric", default=None, help="opt: roc-auc/pr-auc/mae/rmse/spearman; else derived")
    p.add_argument("--learning-curve-sizes", nargs="+", type=int, default=None,
                   help="opt: absolute train sizes for a learning curve, e.g. 100 200 500 1000")
    p.add_argument("--lc-repeats", type=int, default=3, help="random subsample repeats per LC size")
    args = p.parse_args()
    prepare(args)


if __name__ == "__main__":
    main()
