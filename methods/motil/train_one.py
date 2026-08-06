"""
MotiL train_one — single training trial. Approach C: imports MotiL's chemprop
fork directly via sys.path. CLI matches the unified contract.

Per-target metric recomputed from saved pred/label matrices (NaN-aware,
arithmetic mean across targets) for cross-method consistency.
"""
import argparse
import importlib
import json
import os
import random
import shutil
import sys
import time
import warnings
from argparse import Namespace
from pathlib import Path

warnings.filterwarnings("ignore")

# libcusparse.so.11 (torch_sparse SpMM) ships inside the env (nvidia-cusparse-cu11) but is
# not on the default loader path; a clean machine has no system CUDA to fall back on. Setting
# os.environ mid-process is too late for the dynamic loader, so re-exec once with it fixed
# (same approach as methods/molclr/train_one.py).
import sysconfig
_cusparse_dirs = [os.path.join(sysconfig.get_paths()["purelib"], "nvidia", "cusparse", "lib"),
                  os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(sys.executable))), "lib")]
_add = [d for d in _cusparse_dirs if os.path.isdir(d) and d not in os.environ.get("LD_LIBRARY_PATH", "")]
if _add:
    os.environ["LD_LIBRARY_PATH"] = ":".join(_add) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    os.execv(sys.executable, [sys.executable] + sys.argv)

import numpy as np
import torch
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

PIPELINE = Path(__file__).resolve().parents[2]   # clean_pipeline_v1/ (relocatable)
MOTIL_DIR = Path(os.environ.get("MOTIL_DIR", str(PIPELINE / "forks" / "MotiL" / "MotiL_micromolecule")))

# Redirect HOME so DGL can write to ~/.dgl/ (AFS HOME is read-only).
import tempfile
_dgl_home = os.environ.get("DGL_HOME_DIR", os.path.join(tempfile.gettempdir(), "molnet_dgl_home"))
os.makedirs(_dgl_home, exist_ok=True)
os.environ["HOME"] = _dgl_home
os.environ.setdefault("DGLBACKEND", "pytorch")  # use pytorch backend directly; skip ~/.dgl/config.json (avoids concurrent first-run race)

os.chdir(MOTIL_DIR)
sys.path.insert(0, str(MOTIL_DIR))
# appended, not prepended: MOTIL_DIR must keep priority for its own modules
sys.path.append(str(Path(__file__).resolve().parents[1]))

from _scratch import make_scratch_root  # noqa: E402
from _heartbeat import start_heartbeat  # noqa: E402
from rdkit import RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

from chemprop.parsing import modify_train_args, parse_train_args  # noqa: E402
from chemprop.torchlight import initialize_exp  # noqa: E402
from chemprop.utils import makedirs  # noqa: E402

_evaluate_module = importlib.import_module("chemprop.train.evaluate")
_predict_module = importlib.import_module("chemprop.train.predict")
_run_training_module = importlib.import_module("chemprop.train.run_training")
from chemprop.train.run_training import run_training  # noqa: E402


# Paper-faithful schema: HP search varies (epochs, init_lr, final_lr, batch_size, wd)
# per the MotiL paper supplement. max_lr is not a tuned dim — the NoamLR scheduler
# still consults it, so we set max_lr=init_lr in main() (warmup_epochs=0).
# dropout, ffn_num_layers, ffn_hidden_size are NOT tuned (paper doesn't search them);
# they stay at the CMPNN-fork defaults.
DEFAULT_HP = {
    "init_lr":         1e-4,
    "final_lr":        1e-6,
    "batch_size":      64,
    "wd":              1e-5,
    "dropout":         0.1,
    "ffn_num_layers":  2,
    "ffn_hidden_size": 300,
}
# Default lives on NFS; set MOTIL_CKPT to a node-local copy before launching at
# high concurrency.
MOTIL_CKPT = Path(os.environ.get(
    "MOTIL_CKPT",
    str(MOTIL_DIR / "dumped/pre-train/1-model/original_CMPN_0707_0800_12000th_epoch.pkl")))


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def base_args(exp_id):
    # MotiL's parse_train_args requires --exp_name and --exp_id even at startup.
    saved = sys.argv
    sys.argv = ["dummy", "--exp_name", "trial", "--exp_id", exp_id]
    try:
        return parse_train_args()
    finally:
        sys.argv = saved


def protocol_seed_to_runs_id(protocol, seed):
    """Unique runs-id per (protocol, seed) — prevents file-write races during
    cross-cell concurrent scheduling."""
    if protocol == "v1_det":
        return 0
    if protocol == "v1_preshuffle":
        return 10 + int(seed)
    if protocol == "v2_astartes":
        return 20 + int(seed)
    if protocol == "tdc":
        return 30 + int(seed)
    # learning-curve pseudo-protocols (e.g. "v1_preshuffle__f005"): unique, collision-free id
    import zlib
    return 1000 + zlib.crc32(f"{protocol}_{seed}".encode()) % 1_000_000


def materialize_split(dataset, protocol, seed, root_dir):
    """Written under this process's own root_dir, which also removes the write race
    between the ensemble members of one (protocol, seed) — they share a runs-id."""
    runs_id = protocol_seed_to_runs_id(protocol, seed)
    sub = "v1_det_seed0" if protocol == "v1_det" else f"{protocol}_seed{seed}"
    splits_dir = PIPELINE / "splits" / dataset / sub
    tr = np.load(splits_dir / "train_idx.npy")
    va = np.load(splits_dir / "val_idx.npy")
    te = np.load(splits_dir / "test_idx.npy")
    target_dir = root_dir / dataset
    target_dir.mkdir(parents=True, exist_ok=True)
    np.save(target_dir / f"{dataset}-scaffold-{runs_id}.npy",
            np.array([tr, va, te], dtype=object), allow_pickle=True)
    return runs_id


def per_target_metric(preds, targets, task_type, qm_dataset):
    per = []
    for ti in range(targets.shape[1]):
        yt, yp = targets[:, ti], preds[:, ti]
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
    return per, agg_am, agg_gm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--protocol", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--hp-config", default="default")
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--ensemble-member", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--gpu", type=int, default=0)
    cli = p.parse_args()

    out = Path(cli.out_dir)
    start_heartbeat(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "done.flag").exists():
        print(f"[motil] already done: {out}")
        return

    hp = DEFAULT_HP.copy()
    if cli.hp_config != "default":
        with open(cli.hp_config) as f:
            hp.update(json.load(f))

    meta = json.loads((PIPELINE / "cleaned" / f"{cli.dataset}.meta.json").read_text())
    dataset_type = "classification" if meta["task_type"] == "cls" else "regression"
    qm_dataset = cli.dataset in ("qm7", "qm8", "qm9")
    if meta.get("source") == "tdc":
        # TDC selects HP by the dataset's PRESCRIBED metric (map to the fork's names).
        _mnorm = meta["metric"].strip().lower().replace("_", "-")
        metric = {"roc-auc": "auc", "auc": "auc",
                  "pr-auc": "prc-auc", "auprc": "prc-auc", "average-precision": "prc-auc",
                  "mae": "mae", "rmse": "rmse", "mse": "mse",
                  "spearman": "spearman", "spearmanr": "spearman",
                  "pearson": "spearman", "pearsonr": "spearman",
                  }.get(_mnorm, "auc" if meta["task_type"] == "cls" else "rmse")
    else:
        metric = "auc" if meta["task_type"] == "cls" else ("mae" if qm_dataset else "rmse")

    # The split file, the per-epoch val-best model.pt, the TB events and the training
    # log all sit on the shared NFS results volume today, which is the throughput
    # bottleneck at concurrency. Stage them on node-local disk; results/ keeps only the
    # final artifacts. (The dataset CSV is read once into memory, so it needs no staging.)
    scratch_root = make_scratch_root("MOTIL_SCRATCH", "motil")
    data_root = scratch_root / "data"

    runs_id = materialize_split(cli.dataset, cli.protocol, cli.seed, data_root)

    save_dir = scratch_root / "_motil_save"
    save_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cli.ensemble_member)

    args = base_args(cli.dataset)
    args.data_path = str(PIPELINE / "cleaned" / f"{cli.dataset}.csv")
    args.dataset = cli.dataset
    args.root_path = str(data_root)
    args.metric = metric
    args.dataset_type = dataset_type
    args.split_type = "scaffold_balanced"
    args.runs = runs_id
    args.exp_name = "trial"
    args.exp_id = cli.dataset
    args.checkpoint_path = str(MOTIL_CKPT)
    args.gpu = cli.gpu
    # epochs is fixed by the phase for every method (100 at hp_final) so MotiL gets the
    # same budget as the baselines. Stale best_hp.json from an older grid may still carry
    # an "epochs" key; it is deliberately ignored.
    args.epochs = cli.epochs
    args.atom_messages = False
    args.step = "finetune"
    args.dump_path = str(save_dir)
    args.save_dir = str(save_dir)
    args.ensemble_size = 1
    args.seed = cli.ensemble_member
    # The fork's per-epoch train-set evaluation is a second full forward pass over the
    # train split whose only consumer is a TB scalar in the scratch dir deleted below.
    args.skip_train_eval = True
    # Same for the per-epoch test inference: the reported predictions come from the
    # post-loop pass over the best checkpoint, which still runs.
    args.skip_epoch_test_eval = True
    args.batch_size = int(hp["batch_size"])
    args.warmup_epochs = 0
    args.hidden_size = 300
    args.depth = 3
    args.encoder_name = "CMPNN"
    # NoamLR with warmup=0: schedule starts at init_lr (=max_lr) and decays to final_lr.
    args.init_lr = float(hp["init_lr"])
    args.max_lr = float(hp["init_lr"])
    args.final_lr = float(hp["final_lr"])
    args.dropout = float(hp["dropout"])
    args.ffn_num_layers = int(hp["ffn_num_layers"])
    args.ffn_hidden_size = int(hp["ffn_hidden_size"])
    args.wd = float(hp["wd"])

    modify_train_args(args)
    makedirs(args.save_dir)
    logger, args.save_dir = initialize_exp(Namespace(**args.__dict__))

    captured = {"preds": None, "targets": None, "best_val": None}
    orig_info = logger.info

    def patched_info(msg, *a, **kw):
        if isinstance(msg, str) and "best validation" in msg:
            try:
                v = float(msg.split("=", 1)[1].split("on epoch")[0].strip())
                if captured["best_val"] is None:
                    captured["best_val"] = v
                else:
                    captured["best_val"] = (min if metric in ("rmse", "mae", "mse") else max)(captured["best_val"], v)
            except Exception:
                pass
        return orig_info(msg, *a, **kw)
    logger.info = patched_info

    orig_predict = _predict_module.predict
    orig_eval = _evaluate_module.evaluate_predictions

    def patched_predict(*a, **kw):
        out_ = orig_predict(*a, **kw)
        try:
            captured["preds"] = np.array(out_)
        except Exception:
            pass
        return out_

    def patched_eval(preds, targets, *a, **kw):
        try:
            captured["targets"] = targets
        except Exception:
            pass
        return orig_eval(preds, targets, *a, **kw)

    _predict_module.predict = patched_predict
    _run_training_module.predict = patched_predict
    _evaluate_module.evaluate_predictions = patched_eval
    _run_training_module.evaluate_predictions = patched_eval

    t0 = time.time()
    try:
        run_training(args, logger)
    finally:
        _predict_module.predict = orig_predict
        _run_training_module.predict = orig_predict
        _evaluate_module.evaluate_predictions = orig_eval
        _run_training_module.evaluate_predictions = orig_eval
    elapsed = time.time() - t0

    if captured["preds"] is None or captured["targets"] is None:
        print("[motil] missing preds/targets; aborting")
        return

    preds = np.asarray(captured["preds"], dtype=np.float64)
    if preds.ndim == 1:
        preds = preds[:, None]
    tgt = captured["targets"]
    targets = (np.array([list(t) for t in tgt], dtype=np.float64)
               if isinstance(tgt, list) else np.asarray(tgt, dtype=np.float64))
    if targets.ndim == 1:
        targets = targets[:, None]

    np.save(out / "pred_test.npy", preds)
    np.save(out / "labels_test.npy", targets)

    per, agg_am, agg_gm = per_target_metric(preds, targets, meta["task_type"], qm_dataset)

    metrics = {
        "method": "motil",
        "dataset": cli.dataset, "protocol": cli.protocol, "seed": cli.seed,
        "ensemble_member": cli.ensemble_member, "epochs": cli.epochs,
        "test_metric": agg_am, "test_metric_am": agg_am, "test_metric_gm": agg_gm,
        "test_per_target": per,
        "val_metric": captured["best_val"],
        "n_targets": targets.shape[1], "task_type": meta["task_type"], "hp": hp,
        "elapsed_sec": round(elapsed, 1),
    }
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    (out / "done.flag").touch()
    shutil.rmtree(scratch_root, ignore_errors=True)
    print(f"[motil] DONE: {out}  AM={agg_am} GM={agg_gm}  ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
