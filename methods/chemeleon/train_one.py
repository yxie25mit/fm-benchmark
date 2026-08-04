"""
Chemeleon train_one — wraps `chemprop train --from-foundation CheMeleon`.
The D-MPNN backbone is locked by the foundation; only FFN/training-side knobs
are tunable. Identical CLI contract to other methods' train_one.py.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# RDKit-2D descriptor features are ON by default; set NO_RDKIT2D=1 to ablate them
# (train + val-predict must drop them together, else feature dims mismatch).
USE_RDKIT2D = os.environ.get("NO_RDKIT2D") != "1"
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

PIPELINE = Path(__file__).resolve().parents[2]   # clean_pipeline_v1/ (relocatable)
CHEMPROP = str(Path(sys.executable).parent / "chemprop")  # binary lives beside this env's python

# appended, not prepended: the per-method dirs must keep priority for their own modules
sys.path.append(str(Path(__file__).resolve().parents[1]))
import _chemprop_cli  # noqa: E402
from _descriptors import descriptors_for_csv, write_descriptor_subset  # noqa: E402
from _scratch import make_scratch_root  # noqa: E402
from _heartbeat import start_heartbeat  # noqa: E402


# Default HPs — Chemeleon-tunable knobs only (FFN + training; backbone locked).
DEFAULT_HP = {
    "max_lr":            1e-4,
    "ffn_num_layers":    2,
    "ffn_hidden_dim":    600,
    "dropout":           0.1,
    "batch_size":        64,
}


def build_combined_csv(dataset, protocol, seed, out_dir):
    csv = PIPELINE / "cleaned" / f"{dataset}.csv"
    df = pd.read_csv(csv)
    sub = "v1_det_seed0" if protocol == "v1_det" else f"{protocol}_seed{seed}"
    splits_dir = PIPELINE / "splits" / dataset / sub
    train_idx = np.load(splits_dir / "train_idx.npy")
    val_idx = np.load(splits_dir / "val_idx.npy")
    test_idx = np.load(splits_dir / "test_idx.npy")
    splits = np.array(["train"] * len(df))
    splits[val_idx] = "val"
    splits[test_idx] = "test"
    df = df.copy()
    df["split"] = splits
    # Restrict train to train_idx exactly by dropping rows in none of train/val/test.
    # Full splits: train_idx == complement of val/test, so nothing drops (unchanged).
    # Learning-curve fraction splits: this correctly shrinks the training set.
    keep = np.zeros(len(df), dtype=bool)
    keep[np.concatenate([train_idx, val_idx, test_idx])] = True
    df = df[keep]
    combined = out_dir / "combined_data.csv"
    df.to_csv(combined, index=False)
    return combined, keep


def per_target_metric(preds_csv, test_idx, full_df, targets, task_type):
    """Compute per-target metric on test rows. Returns (per_target_list, agg)."""
    pred_df = pd.read_csv(preds_csv)
    # chemprop v2 outputs predictions for ALL rows in the input (not just test);
    # restrict to test_idx and join on smiles for safety.
    pred_df = pred_df.set_index("smiles")
    test_df = full_df.iloc[test_idx].set_index("smiles")
    per = []
    for t in targets:
        if t not in pred_df.columns or t not in test_df.columns:
            per.append(None); continue
        # align on smiles; chemprop's output index may not match test_df row order
        common = test_df.index.intersection(pred_df.index)
        y_true = test_df.loc[common, t].values
        y_pred = pred_df.loc[common, t].values
        valid = ~(np.isnan(y_true) | np.isnan(y_pred))
        if valid.sum() == 0:
            per.append(None); continue
        yt, yp = y_true[valid], y_pred[valid]
        try:
            if task_type == "cls":
                if yt.min() == yt.max():
                    per.append(None); continue
                per.append(float(roc_auc_score(yt, yp)))
            else:
                # MAE for QM, RMSE for other reg (matches chemprop reporting)
                if any(k in str(t).lower() for k in []):  # placeholder
                    pass
                # decide by dataset: caller provides task_type only; use RMSE here, override
                # to MAE for QM datasets in caller. (This simple version uses RMSE; for QM the
                # caller will set --metric-name mae and we adjust below.)
                per.append(float(np.sqrt(mean_squared_error(yt, yp))))
        except Exception:
            per.append(None)
    valid_vals = [v for v in per if v is not None]
    agg = float(np.mean(valid_vals)) if valid_vals else None
    return per, agg


def compute_val_metric(out, scratch_root, dataset, protocol, seed, targets, task_type,
                       qm_dataset, feat_args):
    """Predict the VAL split with the trained checkpoint and score it with the
    SAME per-target NaN-aware macro aggregation used for test, so HP selection
    uses val instead of leaking via test. Returns the val metric, or None."""
    ckpt = next((scratch_root / "cp_out").rglob("best.pt"), None)
    if ckpt is None:
        return None
    sub = "v1_det_seed0" if protocol == "v1_det" else f"{protocol}_seed{seed}"
    val_idx = np.load(PIPELINE / "splits" / dataset / sub / "val_idx.npy")
    full = pd.read_csv(PIPELINE / "cleaned" / f"{dataset}.csv")
    val_csv = scratch_root / "_val_input.csv"
    val_preds = scratch_root / "_val_preds.csv"
    full.iloc[val_idx].to_csv(val_csv, index=False)
    ok = _chemprop_cli.run_predict([
        "-i", str(val_csv), "--model-path", str(ckpt),
        "-o", str(val_preds), "-s", "smiles",
        *feat_args,
    ])
    if not ok or not val_preds.exists():
        return None
    pred_df = pd.read_csv(val_preds).set_index("smiles")
    val_df = full.iloc[val_idx].set_index("smiles")
    common = val_df.index.intersection(pred_df.index)
    per = []
    for t in targets:
        yt = val_df.loc[common, t].to_numpy(dtype=np.float64)
        yp = pred_df.loc[common, t].to_numpy(dtype=np.float64)
        valid = ~(np.isnan(yt) | np.isnan(yp))
        if valid.sum() == 0:
            per.append(None); continue
        ytv, ypv = yt[valid], yp[valid]
        try:
            if task_type == "cls":
                if ytv.min() == ytv.max():
                    per.append(None); continue
                per.append(float(roc_auc_score(ytv, ypv)))
            else:
                per.append(float(mean_absolute_error(ytv, ypv) if qm_dataset
                                 else np.sqrt(mean_squared_error(ytv, ypv))))
        except Exception:
            per.append(None)
    # Save aligned (N,T) val preds+labels so pick_best_hp can re-score with the
    # dataset's prescribed metric (TDC). Harmless for MoleculeNet (unused there).
    try:
        np.save(out / "labels_val.npy", val_df.loc[common, list(targets)].to_numpy(dtype=np.float64))
        np.save(out / "pred_val.npy", pred_df.loc[common, list(targets)].to_numpy(dtype=np.float64))
    except Exception:
        pass
    vals = [v for v in per if v is not None]
    return float(np.mean(vals)) if vals else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--protocol", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--hp-config", default="default")
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--ensemble-member", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--gpu", default="0")    # chemprop CLI uses CUDA_VISIBLE_DEVICES
    args = p.parse_args()

    out = Path(args.out_dir)
    start_heartbeat(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "done.flag").exists():
        print(f"[chemeleon] already done: {out}")
        return

    hp = DEFAULT_HP.copy()
    if args.hp_config != "default":
        with open(args.hp_config) as f:
            hp.update(json.load(f))

    meta = json.loads((PIPELINE / "cleaned" / f"{args.dataset}.meta.json").read_text())
    targets = meta["target_columns"]
    task_type = meta["task_type"]                            # "cls" or "reg"
    cli_task = "classification" if task_type == "cls" else "regression"
    qm_dataset = args.dataset in ("qm7", "qm8", "qm9")
    metric_for_cli = "roc" if task_type == "cls" else ("mae" if qm_dataset else "rmse")

    # Training I/O on the shared NFS results volume is the throughput bottleneck at
    # concurrency: the input CSV is read and every epoch checkpoint / TB event is
    # written over NFS. Stage both on node-local disk; results/ still receives only
    # the final artifacts.
    scratch_root = make_scratch_root("CHEMELEON_SCRATCH", "chemeleon")

    combined, keep_mask = build_combined_csv(args.dataset, args.protocol, args.seed, scratch_root)
    full_df = pd.read_csv(PIPELINE / "cleaned" / f"{args.dataset}.csv")
    split_dir = (PIPELINE / "splits" / args.dataset
                 / ("v1_det_seed0" if args.protocol == "v1_det"
                    else f"{args.protocol}_seed{args.seed}"))
    test_idx = np.load(split_dir / "test_idx.npy")

    # Cached RDKit-2D descriptors, sliced to the rows each CLI call sees, instead of
    # recomputing the whole matrix in the train process and again in the val process.
    if USE_RDKIT2D:
        descriptors = descriptors_for_csv(PIPELINE / "cleaned" / f"{args.dataset}.csv")
        train_feat_args = ["--descriptors-path", str(write_descriptor_subset(
            descriptors, keep_mask, scratch_root / "train_descriptors.npz")),
            "--no-descriptor-scaling"]
        val_feat_args = ["--descriptors-path", str(write_descriptor_subset(
            descriptors, np.load(split_dir / "val_idx.npy"),
            scratch_root / "val_descriptors.npz")),
            "--no-descriptor-scaling"]
    else:
        train_feat_args, val_feat_args = [], []

    cmd = [
        CHEMPROP, "train",
        # Default loads the public CheMeleon foundation; CHEMELEON_CKPT lets a user
        # point at their own fine-tuned CheMeleon .pt (chemprop accepts a local path here).
        "--from-foundation", os.environ.get("CHEMELEON_CKPT", "CheMeleon"),
        "-i", str(combined),
        "-t", cli_task,
        "-o", str(scratch_root / "cp_out"),
        "-s", "smiles",
        "--target-columns", *targets,
        "--splits-column", "split",
        "--ffn-num-layers", str(hp["ffn_num_layers"]),
        "--ffn-hidden-dim", str(hp["ffn_hidden_dim"]),
        "--dropout", str(hp["dropout"]),
        "--batch-size", str(hp["batch_size"]),
        "--max-lr", str(hp["max_lr"]),
        *train_feat_args,
        "--epochs", str(args.epochs),
        "--warmup-epochs", str(min(2, args.epochs - 1)),
        "--num-workers", "4",
        "--ensemble-size", "1",
        "--metrics", metric_for_cli,
        "--pytorch-seed", str(args.ensemble_member),
        "-q",
    ]
    # Set on this process, not just the child env, so the in-process val prediction
    # below sees the same GPU masking and the same ~/.chemprop redirect (AFS HOME is
    # read-only; the chemeleon foundation .pt is cached at this HOME from prior runs).
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["HOME"] = os.environ.get("CHEMELEON_HOME", str(Path(__file__).resolve().parents[2] / "foundations" / "chemeleon"))
    env = os.environ.copy()

    print(f"[chemeleon] launching: {' '.join(cmd[:10])}...")
    _chemprop_cli.start_background_import()
    t0 = time.time()
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    elapsed = time.time() - t0
    if result.returncode != 0:
        (out / "stderr.log").write_text(result.stderr)
        (out / "stdout.log").write_text(result.stdout)
        print(f"[chemeleon] FAILED returncode={result.returncode}; see {out}/stderr.log")
        return

    # Find predictions
    pred_files = list((scratch_root / "cp_out").rglob("test_predictions.csv"))
    if not pred_files:
        pred_files = list((scratch_root / "cp_out").rglob("preds_*.csv"))
    if not pred_files:
        print(f"[chemeleon] no predictions found under {scratch_root}/cp_out")
        return
    preds_csv = pred_files[0]

    # Build pred_test.npy and labels_test.npy aligned to test_idx of full CSV.
    pred_df = pd.read_csv(preds_csv).set_index("smiles")
    full = pd.read_csv(PIPELINE / "cleaned" / f"{args.dataset}.csv")
    test_df = full.iloc[test_idx].set_index("smiles")
    common = test_df.index.intersection(pred_df.index)
    pred_mat = pred_df.loc[common, targets].to_numpy(dtype=np.float64)
    label_mat = test_df.loc[common, targets].to_numpy(dtype=np.float64)
    np.save(out / "pred_test.npy", pred_mat)
    np.save(out / "labels_test.npy", label_mat)

    # Per-target metric (NaN-aware, arithmetic mean across targets).
    per = []
    for ti in range(len(targets)):
        yt = label_mat[:, ti]
        yp = pred_mat[:, ti]
        valid = ~(np.isnan(yt) | np.isnan(yp))
        if valid.sum() == 0:
            per.append(None); continue
        yt_v, yp_v = yt[valid], yp[valid]
        try:
            if task_type == "cls":
                if yt_v.min() == yt_v.max():
                    per.append(None); continue
                per.append(float(roc_auc_score(yt_v, yp_v)))
            else:
                per.append(float(mean_absolute_error(yt_v, yp_v) if qm_dataset
                                 else np.sqrt(mean_squared_error(yt_v, yp_v))))
        except Exception:
            per.append(None)
    valid_vals = [v for v in per if v is not None]
    agg_am = float(np.mean(valid_vals)) if valid_vals else None
    agg_gm = (float(np.exp(np.mean(np.log(valid_vals))))
              if valid_vals and all(v > 0 for v in valid_vals) else None)

    val_metric = compute_val_metric(out, scratch_root, args.dataset, args.protocol,
                                     args.seed, targets, task_type, qm_dataset, val_feat_args)

    metrics = {
        "method": "chemeleon",
        "dataset": args.dataset, "protocol": args.protocol, "seed": args.seed,
        "ensemble_member": args.ensemble_member, "epochs": args.epochs,
        "test_metric": agg_am, "test_metric_am": agg_am, "test_metric_gm": agg_gm,
        "val_metric": val_metric, "val_metric_source": "predict_val_split",
        "test_per_target": per,
        "n_targets": len(targets), "task_type": task_type, "hp": hp,
        "elapsed_sec": round(elapsed, 1),
    }
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    (out / "done.flag").touch()
    shutil.rmtree(scratch_root, ignore_errors=True)
    print(f"[chemeleon] DONE: {out}  AM={agg_am} GM={agg_gm}  ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
