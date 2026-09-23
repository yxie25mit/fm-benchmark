"""Recover the row order of OLD (pre-fix, token-budget-on) molformer outputs by replaying the real eval loader.

Old molformer wrote pred/labels in eval-loader order = np.argsort(token_lengths, kind="stable") over the test
dataset (forks/molformer/finetune/token_budget.py). That order depends only on the test molecules and the real
MolFormer tokenizer — NOT on the token-budget value — so it can be replayed exactly. This uses the pipeline's own
code (materialize_data_root, the finetune script's DataModule, token_budget._lengths); it does NOT re-implement
tokenization (that is where hand-written replays go wrong).

Must run with the MOLFORMER env python:
  <molformer-env>/bin/python replay_molformer_ids.py --root <repo> --dataset X --protocol P --seed S [--out ids.npy]
Prints / saves the global cleaned-CSV index of each old test (and val) output row, in loader order.
"""
import argparse
import importlib
import importlib.util
import os
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np

ROOT = Path.cwd()   # overridden by --root


def load_train_one():
    global ROOT
    spec = importlib.util.spec_from_file_location("mf_train_one", ROOT / "methods" / "molformer" / "train_one.py")
    mto = importlib.util.module_from_spec(spec)
    cwd = os.getcwd()
    spec.loader.exec_module(mto)          # inserts finetune/ on sys.path and chdirs there (needs bert_vocab.txt)
    return mto, cwd


def dispatch(mto, dataset):
    import json
    if dataset in mto.TASK_DISPATCH:
        return mto.TASK_DISPATCH[dataset]
    meta = json.loads((mto.PIPELINE / "cleaned" / f"{dataset}.meta.json").read_text())
    cols = meta.get("target_columns") or ["Y"]
    if len(cols) == 1:
        return (("finetune_pubchem_light_classification", "classification", cols[0], None) if meta["task_type"] == "cls"
                else ("finetune_pubchem_light", "regression", None, None))
    os.environ["MOLFORMER_MEASURE_NAMES"] = json.dumps(cols)
    return (("finetune_pubchem_light_classification_multitask", "classification", None, dataset) if meta["task_type"] == "cls"
            else ("finetune_pubchem_light_regression_multitask", "regression", None, dataset))


def fork_args(**overrides):
    """Full hparams Namespace from the fork's own parser defaults (so every field a finetune script's DataModule
    reads is present), with the data fields overridden."""
    import args as fork_args_module                       # forks/molformer/finetune/args.py (on sys.path)
    margs = Namespace(**{a.dest: a.default for a in fork_args_module.get_parser()._actions if a.dest != 'help'})
    for key, value in overrides.items():
        setattr(margs, key, value)
    return margs


def replay_ids(dataset, protocol, seed):
    import json
    mto, _ = load_train_one()
    import token_budget as tb
    script, ds_type, target_col, multi_name = dispatch(mto, dataset)
    meta = json.loads((mto.PIPELINE / "cleaned" / f"{dataset}.meta.json").read_text())
    target_cols = meta["target_columns"]
    work = Path(tempfile.mkdtemp(prefix="mf_replay_"))
    _, splits = mto.materialize_data_root(dataset, protocol, seed, work, dataset_name=(multi_name or dataset),
                                          target_cols=target_cols, standardize=(ds_type == "regression"))
    te = np.asarray(splits["test"], np.int64)
    va = np.asarray(splits["val"], np.int64)
    mod = importlib.import_module(script)
    margs = fork_args(data_root=str(work), dataset_name=(multi_name or dataset),
                      measure_name=(target_col or target_cols[0]), batch_size=128, num_workers=0,
                      train_dataset_length=None, eval_dataset_length=None, aug=False, seed=0)
    dm = mod.PropertyPredictionDataModule(margs)
    dm.prepare_data()
    def loader_ids(dataset_obj, split_idx):
        lengths = tb._lengths(dataset_obj, dm.tokenizer)
        order = np.argsort(np.asarray(lengths), kind="stable")    # exactly the eval sampler's shuffle=False order
        kept = dataset_obj.df.index.to_numpy()                    # rows of the split csv that survived dropna, in order
        if len(kept) != len(lengths):
            raise RuntimeError("dataset rows != df rows; cannot map dataset index -> split row")
        return split_idx[kept[order]]                             # global cleaned index per old output row
    return loader_ids(dm.val_ds[1], te), loader_ids(dm.val_ds[0], va)   # val_ds = [val, test]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", default=None, help="save test-row ids here")
    ap.add_argument("--out-val", default=None, help="save val-row ids here")
    ap.add_argument("--root", default=".", help="repo root (has methods/, cleaned/, splits/)")
    a = ap.parse_args()
    global ROOT
    ROOT = Path(a.root).resolve()
    out = str(Path(a.out).resolve()) if a.out else None   # loading train_one chdirs into finetune/
    out_val = str(Path(a.out_val).resolve()) if a.out_val else None
    ids, val_ids = replay_ids(a.dataset, a.protocol, a.seed)
    if out:
        np.save(out, ids)
    if out_val:
        np.save(out_val, val_ids)
    print(f"replayed {len(ids)} test / {len(val_ids)} val molformer row ids for {a.dataset}/{a.protocol}/seed{a.seed}")


if __name__ == "__main__":
    main()
