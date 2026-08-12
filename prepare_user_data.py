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
Input — a SINGLE csv, we scaffold-split it into BOTH our published styles:
  --csv DATA.csv     writes splits/<name>/{v1_preshuffle,v2_astartes}_seed{0,1,2,3}/
                     Then run with `--protocols v1_preshuffle v2_astartes` exactly like a
                     built-in benchmark dataset (same downstream commands).
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


def _write_learning_curve(dataset, base_tag, train_idx, val_idx, test_idx, y_train, sizes, repeats, task):
    """Nested-ish stratified subsets of `train_idx` at each absolute size (val/test held
    fixed), written as pseudo-protocol dirs `<base_tag>__<size>_seed<rep>` so
    run_learning_curve.py loads them via --protocols <base_tag> --fractions <size>."""
    made = []
    for size in sorted(set(sizes)):
        for rep in range(repeats):
            rng = np.random.default_rng(size * 1000 + rep)
            sub_train = _stratified_subsample(train_idx, y_train, size, task, rng)
            d = PIPELINE / "splits" / dataset / f"{base_tag}__{size}_seed{rep}"
            d.mkdir(parents=True, exist_ok=True)
            np.save(d / "train_idx.npy", np.asarray(sub_train, dtype=np.int64))
            np.save(d / "val_idx.npy", np.asarray(val_idx, dtype=np.int64))
            np.save(d / "test_idx.npy", np.asarray(test_idx, dtype=np.int64))
        made.append(f"{size}={min(size, len(train_idx))}")
    print(f"[prep] {base_tag} learning-curve subsets ({repeats} repeats each): " + ", ".join(made))


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
        _write_learning_curve(dataset, "custom", fold0["train"], fold0["val"], fold0["test"],
                              y, args.learning_curve_sizes, args.lc_repeats, args.task)

    print(f"\n[prep] dataset '{dataset}': {len(cleaned_df)} unique molecules, "
          f"{n_invalid_total} invalid SMILES dropped, {len(fold_idx)} fold(s).")
    for name, ntr, nva, nte, ov in report_folds:
        print(f"       fold {name} -> custom_seed{[n for n,_,_,_,_ in report_folds].index(name)}: "
              f"train {ntr} / val {nva} / test {nte}" + (f"  [leakage {ov}]" if ov else ""))
    print(f"\n[prep] next, run the normal pipeline on it:")
    print(f"       python -m orchestrator.run_benchmark --methods <METHOD> \\")
    print(f"         --datasets {dataset} --protocols custom [--checkpoint <FILE>] --gpus 0,1,2,3")
    return dataset


def prepare_from_single_csv(args):
    """One CSV in -> we scaffold-split it into our two published styles (v1_preshuffle,
    v2_astartes), each at seeds 0/1/2 (eval) + 3 (HP-only), writing the SAME layout a
    built-in benchmark dataset has. Downstream `--protocols v1_preshuffle v2_astartes`
    then behaves identically to a built-in dataset."""
    from scaffold_splits import SPLITTERS, DEFAULT_SEEDS

    if "v2_astartes" in args.split_styles:   # fail fast, before writing anything
        try:
            import astartes  # noqa: F401
        except ImportError:
            sys.exit("[prep] the v2_astartes split needs the 'astartes' package in this (orchestrator) env.\n"
                     "       Install it:   pip install 'astartes[molecules]'\n"
                     "       (or generate only the v1 style with:  --split-styles v1_preshuffle)")

    dataset = args.name
    targets = args.target_cols
    meta_path = PIPELINE / "cleaned" / f"{dataset}.meta.json"
    if meta_path.exists() and not json.loads(meta_path.read_text()).get("user_created"):
        sys.exit(f"[prep] '{dataset}' is a built-in benchmark dataset — pick a different --name.")

    df = pd.read_csv(args.csv)
    if args.smiles_col not in df.columns:
        sys.exit(f"[prep] '{args.smiles_col}' not in {args.csv}; columns: {list(df.columns)}")
    df = df.rename(columns={args.smiles_col: "smiles"})
    for col in targets:
        if col not in df.columns:
            sys.exit(f"[prep] target '{col}' missing from {args.csv}; columns: {list(df.columns)}")

    canon, n_invalid = _canonicalize_and_flag(df["smiles"].tolist())
    pool_rows, pool_index = [], {}
    for row_i, smiles in enumerate(canon):
        if smiles is None or smiles in pool_index:   # drop invalid + dedup (keep first)
            continue
        pool_index[smiles] = len(pool_rows)
        row = {"smiles": smiles}
        row.update({t: df.iloc[row_i][t] for t in targets})
        pool_rows.append(row)
    n_dupes = len(canon) - n_invalid - len(pool_rows)

    cleaned_df = pd.DataFrame(pool_rows, columns=["smiles"] + targets)
    (PIPELINE / "cleaned").mkdir(exist_ok=True)
    cleaned_df.to_csv(PIPELINE / "cleaned" / f"{dataset}.csv", index=False)
    meta = {"dataset": dataset, "task_type": args.task, "target_columns": targets,
            "n_targets": len(targets), "n_after": len(cleaned_df), "user_created": True}
    if args.metric:
        meta["metric"] = args.metric
        meta["source"] = "tdc"
    meta_path.write_text(json.dumps(meta, indent=2))

    smiles = cleaned_df["smiles"].tolist()
    print(f"\n[prep] dataset '{dataset}': {len(cleaned_df)} unique molecules "
          f"({n_invalid} invalid SMILES dropped, {n_dupes} duplicates merged).")
    for style in args.split_styles:
        splitter = SPLITTERS[style]
        for seed in DEFAULT_SEEDS:
            tr, va, te = splitter(smiles, seed=seed)
            sub = PIPELINE / "splits" / dataset / f"{style}_seed{seed}"
            sub.mkdir(parents=True, exist_ok=True)
            np.save(sub / "train_idx.npy", tr)
            np.save(sub / "val_idx.npy", va)
            np.save(sub / "test_idx.npy", te)
        s0 = PIPELINE / "splits" / dataset / f"{style}_seed0"
        ntr, nva, nte = (len(np.load(s0 / f"{r}_idx.npy")) for r in ("train", "val", "test"))
        print(f"       {style}: seeds {DEFAULT_SEEDS} written (seed0 train {ntr} / val {nva} / test {nte})")

    if args.learning_curve_sizes:   # per style, subsample that style's seed0 train (val/test fixed)
        for style in args.split_styles:
            s0 = PIPELINE / "splits" / dataset / f"{style}_seed0"
            tr = np.load(s0 / "train_idx.npy")
            va = np.load(s0 / "val_idx.npy")
            te = np.load(s0 / "test_idx.npy")
            y = cleaned_df.iloc[tr][targets[0]].to_numpy()
            _write_learning_curve(dataset, style, tr, va, te, y,
                                  args.learning_curve_sizes, args.lc_repeats, args.task)

    print(f"\n[prep] next, run BOTH scaffold styles exactly like a built-in dataset:")
    print(f"       python -m orchestrator.run_benchmark --methods <METHOD> \\")
    print(f"         --datasets {dataset} --protocols {' '.join(args.split_styles)} --gpus 0,1,2,3")
    return dataset


def _write_single_fold_dataset(sub_name, pool_df, targets, task, metric, train, val, test):
    """Write one single-fold `custom` dataset: cleaned/<sub>.csv + meta + one custom_seed0
    split. Single fold => the pipeline selects HP on THIS fold's own validation (no pooling),
    which is what makes the sliding-window folds leakage-free."""
    (PIPELINE / "cleaned").mkdir(exist_ok=True)
    pool_df.to_csv(PIPELINE / "cleaned" / f"{sub_name}.csv", index=False)
    meta = {"dataset": sub_name, "task_type": task, "target_columns": targets,
            "n_targets": len(targets), "n_after": len(pool_df), "user_created": True}
    if metric:
        meta["metric"] = metric
        meta["source"] = "tdc"
    (PIPELINE / "cleaned" / f"{sub_name}.meta.json").write_text(json.dumps(meta, indent=2))
    d = PIPELINE / "splits" / sub_name / "custom_seed0"
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "train_idx.npy", np.asarray(train, dtype=np.int64))
    np.save(d / "val_idx.npy", np.asarray(val, dtype=np.int64))
    np.save(d / "test_idx.npy", np.asarray(test, dtype=np.int64))


def _write_multifold_dataset(sub_name, pool_df, targets, task, metric, folds, hp_per_fold):
    """Write one multi-fold `custom` dataset (custom_seed0..K-1). hp_per_fold=True stamps the
    meta so run_phase tunes each fold on its OWN validation (not the mean across folds)."""
    (PIPELINE / "cleaned").mkdir(exist_ok=True)
    pool_df.to_csv(PIPELINE / "cleaned" / f"{sub_name}.csv", index=False)
    meta = {"dataset": sub_name, "task_type": task, "target_columns": targets,
            "n_targets": len(targets), "n_after": len(pool_df), "user_created": True}
    if metric:
        meta["metric"] = metric
        meta["source"] = "tdc"
    if hp_per_fold:
        meta["hp_per_fold"] = True
    (PIPELINE / "cleaned" / f"{sub_name}.meta.json").write_text(json.dumps(meta, indent=2))
    for i, (train, val, test) in enumerate(folds):
        d = PIPELINE / "splits" / sub_name / f"custom_seed{i}"
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "train_idx.npy", np.asarray(train, dtype=np.int64))
        np.save(d / "val_idx.npy", np.asarray(val, dtype=np.int64))
        np.save(d / "test_idx.npy", np.asarray(test, dtype=np.int64))


def _write_time_lc(sub_name, folds, sizes):
    """Recency-nested learning-curve subsets per sliding fold: train = the most-recent `size` rows
    of that fold's (time-sorted) train window — nested suffixes (last-k ⊂ last-2k ⊂ …) — with the
    fold's val/test held fixed. Written as custom__<size>_seed<fold> so:
      run_learning_curve.py --datasets <sub_name> --protocols custom --fractions <sizes> --seeds 0..K-1
    loads them and collect_results.py --learning-curve averages mean±std across the folds at each size."""
    for fold_i, (train, val, test) in enumerate(folds):
        train = np.asarray(train, dtype=np.int64)   # ascending == time order; most-recent = tail
        for size in sorted(set(sizes)):
            k = min(size, len(train))
            d = PIPELINE / "splits" / sub_name / f"custom__{size}_seed{fold_i}"
            d.mkdir(parents=True, exist_ok=True)
            np.save(d / "train_idx.npy", train[-k:])
            np.save(d / "val_idx.npy", np.asarray(val, dtype=np.int64))
            np.save(d / "test_idx.npy", np.asarray(test, dtype=np.int64))
    full = len(folds[0][0])
    caps = ", ".join(f"{s}(->{min(s, full)})" for s in sorted(set(sizes)))
    print(f"[prep] {sub_name} learning curve (recency-nested, {len(folds)} folds, capped at train={full}): {caps}")


def prepare_time_split(args):
    """One time-sorted CSV in -> a chronological 80/10/10 split (<name>_chrono) plus
    constant-train sliding windows (<name>_tw0..K), each emitted as its own single-fold
    dataset so the normal hp_search/hp_final tunes each on its OWN validation (leakage-free).
    Then aggregate the _tw* test scores for the sliding-window mean±std."""
    from time_splits import chrono_split, sliding_windows

    dataset = args.name
    targets = args.target_cols
    for sub in (f"{dataset}_chrono", f"{dataset}_sliding"):
        mp = PIPELINE / "cleaned" / f"{sub}.meta.json"
        if mp.exists() and not json.loads(mp.read_text()).get("user_created"):
            sys.exit(f"[prep] '{sub}' collides with a built-in dataset — pick a different --name.")

    df = pd.read_csv(args.csv)
    if args.smiles_col not in df.columns:
        sys.exit(f"[prep] '{args.smiles_col}' not in {args.csv}; columns: {list(df.columns)}")
    df = df.rename(columns={args.smiles_col: "smiles"})
    for col in targets:
        if col not in df.columns:
            sys.exit(f"[prep] target '{col}' missing from {args.csv}; columns: {list(df.columns)}")
    if args.date_col:
        if args.date_col not in df.columns:
            sys.exit(f"[prep] --date-col '{args.date_col}' not in {args.csv}; columns: {list(df.columns)}")
        df = df.sort_values(args.date_col, kind="stable").reset_index(drop=True)

    canon, n_invalid = _canonicalize_and_flag(df["smiles"].tolist())
    pool_rows, seen = [], set()
    for row_i, smiles in enumerate(canon):
        if smiles is None or smiles in seen:   # drop invalid + dedup, keeping EARLIEST occurrence
            continue
        seen.add(smiles)
        row = {"smiles": smiles}
        row.update({t: df.iloc[row_i][t] for t in targets})
        pool_rows.append(row)
    n_dupes = len(canon) - n_invalid - len(pool_rows)
    pool_df = pd.DataFrame(pool_rows, columns=["smiles"] + targets)
    n = len(pool_df)

    sort_note = f"sorted by '{args.date_col}'" if args.date_col else "using input row order (assumed time-sorted)"
    print(f"\n[prep] time-split '{dataset}': {n} unique molecules {sort_note} "
          f"({n_invalid} invalid dropped, {n_dupes} duplicates merged, earliest kept).")

    tr, va, te = chrono_split(n)
    _write_single_fold_dataset(f"{dataset}_chrono", pool_df, targets, args.task, args.metric, tr, va, te)
    print(f"       {dataset}_chrono: 80/10/10 chronological (train {len(tr)} / val {len(va)} / test {len(te)})")

    folds = sliding_windows(n, n_chunks=args.tw_chunks, train_chunks=args.tw_train_chunks, n_folds=args.tw_folds)
    _write_multifold_dataset(f"{dataset}_sliding", pool_df, targets, args.task, args.metric, folds, hp_per_fold=True)
    for i, (tr, va, te) in enumerate(folds):
        print(f"       {dataset}_sliding fold {i}: train {len(tr)} / val {len(va)} / test {len(te)} "
              f"(tests newest chunk {args.tw_train_chunks + i + 2}/{args.tw_chunks})")
    print(f"       -> {dataset}_sliding meta sets hp_per_fold=True (each fold tuned on its OWN val).")
    if args.learning_curve_sizes:
        _write_time_lc(f"{dataset}_sliding", folds, args.learning_curve_sizes)

    print(f"\n[prep] next — tune + evaluate (each fold tuned on its OWN validation, leakage-free):")
    print(f"       # headline chronological number (single 80/10/10 fold):")
    print(f"       python -m orchestrator.run_benchmark --methods <METHOD> \\")
    print(f"         --datasets {dataset}_chrono --protocols custom --phases hp_search hp_final --gpus 0,1,2,3")
    print(f"       # sliding-window error bars ({len(folds)} folds, per-fold HP is automatic):")
    print(f"       python -m orchestrator.run_benchmark --methods <METHOD> \\")
    print(f"         --datasets {dataset}_sliding --protocols custom --phases hp_search hp_final --gpus 0,1,2,3")
    print(f"       # combined table (chrono headline + sliding mean±std):")
    print(f"       python collect_time_split.py --name {dataset} --phase hp_final --methods <METHOD>")
    if args.learning_curve_sizes:
        sizes = " ".join(str(s) for s in sorted(set(args.learning_curve_sizes)))
        print(f"       # learning curve on the sliding folds (recency-nested; error bars across {len(folds)} folds):")
        print(f"       python runners/run_learning_curve.py --methods <METHOD> --datasets {dataset}_sliding \\")
        print(f"         --protocols custom --fractions {sizes} --seeds 0 1 2 --ensemble 5 --gpus 0,1,2,3")
        print(f"       python collect_results.py --dataset {dataset}_sliding --protocol custom --learning-curve \\")
        print(f"         --methods <METHOD> --out {dataset}_curve.csv --plot {dataset}_curve.png")
    return dataset


def main():
    p = argparse.ArgumentParser(description="Adapt user CSVs into the pipeline data format (no re-splitting).")
    p.add_argument("--name", required=True,
                   help="dataset tag; writes cleaned/<name>.csv + splits/<name>/ (must not collide with a built-in dataset).")
    p.add_argument("--train-csv"); p.add_argument("--val-csv"); p.add_argument("--test-csv")
    p.add_argument("--splits-dir", help="dir of fold subdirs each with train/val/test.csv")
    p.add_argument("--csv", help="a SINGLE csv; we scaffold-split it into both published styles")
    p.add_argument("--split-styles", nargs="+", default=["v1_preshuffle", "v2_astartes"],
                   choices=["v1_preshuffle", "v2_astartes"],
                   help="which scaffold styles to generate from --csv (default: both)")
    p.add_argument("--time-split", action="store_true",
                   help="with --csv: chronological instead of scaffold. Emits <name>_chrono "
                        "(80/10/10) + <name>_tw0..K (constant-train sliding windows).")
    p.add_argument("--date-col", help="time-split: column to sort by (else input row order is used)")
    p.add_argument("--tw-chunks", type=int, default=12,
                   help="time-split sliding: cut time-sorted rows into this many equal-count chunks (default 12)")
    p.add_argument("--tw-train-chunks", type=int, default=8,
                   help="time-split sliding: each fold trains on this many consecutive chunks, constant window (default 8)")
    p.add_argument("--tw-folds", type=int, default=3,
                   help="time-split sliding: number of sliding folds; needs tw-chunks >= tw-train-chunks + tw-folds + 1 (default 3)")
    p.add_argument("--smiles-col", default="smiles")
    p.add_argument("--target-cols", nargs="+", required=True)
    p.add_argument("--task", required=True, choices=["cls", "reg"])
    p.add_argument("--metric", default=None, help="opt: roc-auc/pr-auc/mae/rmse/spearman; else derived")
    p.add_argument("--learning-curve-sizes", nargs="+", type=int, default=None,
                   help="opt: absolute train sizes for a learning curve, e.g. 100 200 500 1000")
    p.add_argument("--lc-repeats", type=int, default=3, help="random subsample repeats per LC size")
    args = p.parse_args()
    if args.csv:
        if args.train_csv or args.val_csv or args.test_csv or args.splits_dir:
            sys.exit("[prep] --csv (we split) is mutually exclusive with --train/val/test-csv / --splits-dir (you split).")
        if args.time_split:
            prepare_time_split(args)
        else:
            prepare_from_single_csv(args)
    else:
        prepare(args)


if __name__ == "__main__":
    main()
