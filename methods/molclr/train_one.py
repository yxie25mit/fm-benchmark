"""
MolCLR train_one worker — single training trial, identical CLI across all 6 methods.

Inputs (CLI):
  --dataset             one of bbbp/bace/clintox/sider/tox21/hiv/freesolv/esol/lipo/qm7/qm8/qm9
  --protocol            v1_det | v1_preshuffle | v2_astartes
  --seed                split seed (0 for v1_det; 0/1/2 for eval; 3 for hp-search)
  --hp-config           path to JSON with HPs (use "default" for method defaults)
  --epochs              int
  --ensemble-member     int 0..4 (controls torch random seed for init/init variance)
  --out-dir             where to write artifacts

Outputs (in --out-dir):
  pred_test.npy         (n_test, n_targets) post-sigmoid (cls) or denormalized (reg)
  labels_test.npy       (n_test, n_targets) with NaN for missing
  metrics.json          {val_metric, test_metric, test_per_target, hp, ...}
  done.flag             empty marker for resume
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

# torch_sparse (via torch_geometric) needs libcusparse.so.11, which torch 1.7.1+cu110
# bundles only partially. Some hosts lack it on the system library path. The
# nvidia-cusparse-cu11 pip package provides it; ensure its dir is on LD_LIBRARY_PATH
# BEFORE importing torch. Setting os.environ mid-process is too late for the dynamic
# loader, so re-exec once with the corrected environment.
import sysconfig
_CUSPARSE_DIR = os.path.join(sysconfig.get_paths()["purelib"], "nvidia", "cusparse", "lib")
if os.path.isdir(_CUSPARSE_DIR) and _CUSPARSE_DIR not in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = _CUSPARSE_DIR + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    os.execv(sys.executable, [sys.executable] + sys.argv)

import numpy as np
import torch

PIPELINE = Path(__file__).resolve().parents[2]   # clean_pipeline_v1/ (relocatable)
sys.path.insert(0, str(Path(__file__).parent))
# appended, not prepended: the per-method dirs must keep priority for their own modules
sys.path.append(str(Path(__file__).resolve().parents[1]))

from _scratch import make_scratch_root  # noqa: E402
from _heartbeat import start_heartbeat  # noqa: E402


# Default HPs — MolCLR's published config_finetune.yaml values.
DEFAULT_HP = {
    "init_lr":        0.0005,
    # 0.0005 matches prior hyperopt_v2_r7 Bayesian-tuned best across all
    # datasets and was also empirically the only LR pair that brought freesolv
    # RMSE to ~2.6. Paper config_finetune.yaml shipped 0.0001 but their saved
    # .runs_parallel configs didn't expose this.
    "init_base_lr":   0.0005,
    "weight_decay":   "1e-6",       # passed as string (finetune.py does eval())
    "batch_size":     32,
    # MolCLR's shipped config_finetune.yaml says 0.1, but every per-dataset
    # config in molclr/figs/.runs_parallel/<DS>/config_finetune.yaml (their actual
    # experiment artifacts) uses 0.3. 0.3 is the paper-actual value.
    "drop_ratio":     0.3,
    "num_layer":      5,
    "emb_dim":        300,
    "feat_dim":       512,
    "pred_n_layer":   2,
    "pred_act":       "softplus",
    "pool":           "mean",
}


def build_config(args, hp, log_dir):
    csv = PIPELINE / "cleaned" / f"{args.dataset}.csv"
    meta = json.loads((PIPELINE / "cleaned" / f"{args.dataset}.meta.json").read_text())
    sub = "v1_det_seed0" if args.protocol == "v1_det" else f"{args.protocol}_seed{args.seed}"
    splits_dir = PIPELINE / "splits" / args.dataset / sub
    train_idx = np.load(splits_dir / "train_idx.npy").tolist()
    val_idx = np.load(splits_dir / "val_idx.npy").tolist()
    test_idx = np.load(splits_dir / "test_idx.npy").tolist()
    targets = meta["target_columns"]
    task = "classification" if meta["task_type"] == "cls" else "regression"

    return {
        "task_name": args.dataset,
        "model_type": "gin",
        "epochs": args.epochs,
        # Absolute path so molclr's _save_config_file works regardless of CWD.
        "config_path": str(Path(__file__).parent / "config_finetune.yaml"),
        "init_lr": hp["init_lr"],
        "init_base_lr": hp["init_base_lr"],
        "weight_decay": str(hp["weight_decay"]),
        "log_every_n_steps": 50,
        "eval_every_n_epochs": 1,
        "fp16_precision": False,
        "fine_tune_from": os.environ.get("MOLCLR_CKPT_DIR", "pretrained_gin"),
        "gpu": args.gpu,
        "log_dir": str(log_dir),
        "batch_size": int(hp["batch_size"]),
        "model": {
            "num_layer":   int(hp["num_layer"]),
            "emb_dim":     int(hp["emb_dim"]),
            "feat_dim":    int(hp["feat_dim"]),
            "drop_ratio":  float(hp["drop_ratio"]),
            "pool":        hp["pool"],
            "pred_n_layer": int(hp["pred_n_layer"]),
            "pred_act":    hp["pred_act"],
        },
        "dataset": {
            "task": task,
            "target": targets,
            "data_path": str(csv),
            "splitting": "external",
            "external_split": (train_idx, val_idx, test_idx),
            "valid_size": 0.1, "test_size": 0.1,
            # Graphs are precomputed into the dataset cache, so worker fork+IPC (12 forks
            # per epoch of a multi-GB process) costs far more than it saves.
            "num_workers": 0,
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--protocol", required=True)  # accepts learning-curve pseudo-protocols (v1_preshuffle__f020)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--hp-config", default="default")
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--ensemble-member", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--gpu", default="cuda:0",
                   help="cuda device. Numeric '0' is also accepted (mapped to 'cuda:0' since CUDA_VISIBLE_DEVICES masks other GPUs).")
    args = p.parse_args()
    # Map plain int strings ("0") to "cuda:0" — run_phase.py passes "0" because
    # CUDA_VISIBLE_DEVICES already masks to a single visible device.
    if isinstance(args.gpu, str) and args.gpu.isdigit():
        args.gpu = f"cuda:{int(args.gpu)}"

    out = Path(args.out_dir)
    start_heartbeat(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "done.flag").exists():
        print(f"[molclr] already done: {out}")
        return

    hp = DEFAULT_HP.copy()
    if args.hp_config != "default":
        with open(args.hp_config) as f:
            user_hp = json.load(f)
        hp.update(user_hp)

    # Set torch random seed for ensemble-member init variance.
    torch.manual_seed(args.ensemble_member)
    np.random.seed(args.ensemble_member)

    # The tensorboard writer and the val-best model.pth both live under log_dir and are
    # rewritten throughout training; on the shared NFS results volume that is the
    # throughput bottleneck at concurrency. Stage them on node-local disk instead —
    # results/ keeps only the final artifacts. (The dataset itself is read from the
    # CSV once into memory, so it needs no staging.)
    scratch_root = make_scratch_root("MOLCLR_SCRATCH", "molclr")

    cfg = build_config(args, hp, scratch_root / "_tb")

    # Import after sys.path so the patched finetune.py is found.
    from finetune import FineTune
    from dataset.dataset_test import MolTestDatasetWrapper

    dataset = MolTestDatasetWrapper(cfg["batch_size"], **cfg["dataset"])
    fine_tune = FineTune(dataset, cfg)
    t0 = time.time()
    fine_tune.train()
    elapsed = time.time() - t0

    # Save artifacts.
    meta = json.loads((PIPELINE / "cleaned" / f"{args.dataset}.meta.json").read_text())
    test_pred = fine_tune.test_predictions
    if meta.get("source") == "tdc" and meta["task_type"] == "cls" and meta.get("n_targets", 1) == 1:
        # Single-target cls only: (N,2) softmax -> positive-class score (N,1) so _tdc_metrics
        # scores roc-auc/pr-auc correctly. For multitask (n_targets>1) the columns are the
        # tasks, not the 2 classes, so keep all columns.
        tp = np.asarray(test_pred)
        if tp.ndim == 2 and tp.shape[1] == 2:
            test_pred = tp[:, 1].reshape(-1, 1)
    np.save(out / "pred_test.npy", test_pred)
    np.save(out / "labels_test.npy", fine_tune.test_labels)
    if getattr(fine_tune, "val_predictions", None) is not None:
        np.save(out / "pred_val.npy", fine_tune.val_predictions)
        np.save(out / "labels_val.npy", fine_tune.val_labels)
    # Compute both AM (canonical) and GM aggregations from per-target list.
    valid_vals = [v for v in fine_tune.test_per_target if v is not None]
    agg_am = float(np.mean(valid_vals)) if valid_vals else None
    agg_gm = (float(np.exp(np.mean(np.log(valid_vals))))
              if valid_vals and all(v > 0 for v in valid_vals) else None)
    metrics = {
        "method": "molclr",
        "dataset": args.dataset,
        "protocol": args.protocol,
        "seed": args.seed,
        "ensemble_member": args.ensemble_member,
        "epochs": args.epochs,
        "test_metric": agg_am,
        "test_metric_am": agg_am,
        "test_metric_gm": agg_gm,
        "val_metric": getattr(fine_tune, "best_val_metric", None),
        "test_per_target": fine_tune.test_per_target,
        "n_targets": len(cfg["dataset"]["target"]),
        "task_type": "cls" if cfg["dataset"]["task"] == "classification" else "reg",
        "hp": hp,
        "elapsed_sec": round(elapsed, 1),
    }
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    (out / "done.flag").touch()
    shutil.rmtree(scratch_root, ignore_errors=True)
    print(f"[molclr] DONE: {out}  AM={agg_am} GM={agg_gm}  ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
