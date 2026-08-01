"""
MolFCL train_one — single training trial. Approach C: imports MolFCL's
chemprop fork directly via sys.path (MolFCL has no setup.py / pip install).

CLI matches the unified contract:
  --dataset --protocol --seed --hp-config --epochs --ensemble-member --out-dir

Per-target metric is recomputed from saved pred/label matrices (NaN-aware,
arithmetic mean across targets) so all 6 methods report identically-aggregated
metrics regardless of each upstream's internal convention.
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

import numpy as np
import torch
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

PIPELINE = Path(__file__).resolve().parents[2]   # clean_pipeline_v1/ (relocatable)
MOLFCL_DIR = Path(os.environ.get("MOLFCL_DIR", "/data/rbg/users/yxie25/molclr_chemprop/MolFCL"))

# Redirect HOME so DGL/RDKit can write to ~/.dgl/, etc. (AFS HOME is read-only).
os.environ.setdefault("HOME", "/data/rbg/users/yxie25/dgl_home")
os.environ["HOME"] = "/data/rbg/users/yxie25/dgl_home"

# MolFCL is a research repo (no setup.py). Add to sys.path AND chdir into it
# (it uses relative paths like ./data, ./ckpt internally).
os.chdir(MOLFCL_DIR)
sys.path.insert(0, str(MOLFCL_DIR))
# appended, not prepended: MOLFCL_DIR must keep priority for its own modules
sys.path.append(str(Path(__file__).resolve().parents[1]))

from _scratch import make_scratch_root  # noqa: E402
from _heartbeat import start_heartbeat  # noqa: E402
from rdkit import RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

# Pretrained MolFCL checkpoint. Default lives on NFS; set MOLFCL_CKPT to a node-local
# copy before launching at high concurrency.
MOLFCL_CKPT = Path(os.environ.get(
    "MOLFCL_CKPT", str(MOLFCL_DIR / "ckpt" / "original_MoleculeModel.pkl")))

from chemprop.parsing import modify_train_args, parse_train_args  # noqa: E402
from chemprop.torchlight import initialize_exp  # noqa: E402
from chemprop.utils import makedirs  # noqa: E402

# chemprop/train/__init__.py shadows submodule names; pull modules via importlib
# so we can monkey-patch predict/evaluate to capture preds/targets.
_evaluate_module = importlib.import_module("chemprop.train.evaluate")
_predict_module = importlib.import_module("chemprop.train.predict")
_run_training_module = importlib.import_module("chemprop.train.run_training")
from chemprop.train.run_training import run_training  # noqa: E402


DEFAULT_HP = {
    "max_lr":          1e-3,
    "dropout":         0.1,
    "ffn_num_layers":  2,
    "ffn_hidden_size": 300,
}


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def base_args():
    saved = sys.argv
    sys.argv = ["dummy"]
    try:
        return parse_train_args()
    finally:
        sys.argv = saved


def protocol_seed_to_runs_id(protocol, seed):
    """Encode (protocol, seed) into a unique runs-id so materialized split files
    DON'T collide across concurrent cross-cell jobs.
        v1_det          → 0
        v1_preshuffle s → 10 + s
        v2_astartes  s  → 20 + s
    """
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
    """Write the split file MolFCL's load_data() reads, under this process's own
    root_dir. Returns the runs-id to pass to args.runs. Keeping it per-process also
    removes the write race between the ensemble members of one (protocol, seed),
    which all encode to the same runs-id."""
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
    """NaN-aware per-target metric (AUC for cls; MAE for QM, RMSE for other reg).
    Aggregation = arithmetic mean across non-None targets."""
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
        print(f"[molfcl] already done: {out}")
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
        metric = {"roc-auc": "auc", "pr-auc": "prc-auc",
                  "mae": "mae", "spearman": "spearman"}[meta["metric"]]
    else:
        metric = "auc" if meta["task_type"] == "cls" else ("mae" if qm_dataset else "rmse")

    # The split file, the per-epoch val-best model.pt and the training log all sit on
    # the shared NFS results volume today, which is the throughput bottleneck at
    # concurrency. Stage them on node-local disk; results/ keeps only the final
    # artifacts. (The dataset CSV is read once into memory, so it needs no staging.)
    scratch_root = make_scratch_root("MOLFCL_SCRATCH", "molfcl")
    data_root = scratch_root / "data"

    runs_id = materialize_split(cli.dataset, cli.protocol, cli.seed, data_root)

    save_dir = scratch_root / "_molfcl_save"
    save_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cli.ensemble_member)

    args = base_args()
    args.data_path = str(PIPELINE / "cleaned" / f"{cli.dataset}.csv")
    args.dataset = cli.dataset
    args.root_path = str(data_root)
    args.metric = metric
    args.dataset_type = dataset_type
    args.split_type = "scaffold_balanced"
    args.runs = runs_id
    args.encoder = False
    args.exp_name = "trial"
    args.exp_id = cli.dataset
    args.checkpoint_path = str(MOLFCL_CKPT)
    args.gpu = cli.gpu
    args.epochs = cli.epochs
    args.pretrain = False
    args.atom_messages = True
    args.increase_parm = 1
    args.add_step = "concat_mol_frag_attention"
    args.step = "func_prompt"
    args.add_reactive = False
    args.early_stop = True
    args.patience = 15
    args.last_early_stop = 0
    # Paper-faithful: gamma (L2 penalty on the FG-embedding table) and the FG-prompt
    # self-attention layer count are the SI's key tuned knobs; read them from the HP
    # (defaults = SI optima 0.1 / 7) instead of the old fixed 0 / 2.
    args.l2_norm = float(hp.get("l2_norm", 0.1))
    args.num_attention = int(hp.get("num_attention", 7))
    args.dump_path = str(save_dir)
    args.save_dir = str(save_dir)
    args.ensemble_size = 1
    args.seed = cli.ensemble_member
    # Paper uses batch 256 for MoleculeNet fine-tuning, 64 for TDC (Table S5). Clamp to
    # n_train so small learning-curve fractions still yield >=1 batch (drop_last would
    # otherwise give 0 batches -> ZeroDivisionError in training).
    _sub = "v1_det_seed0" if cli.protocol == "v1_det" else f"{cli.protocol}_seed{cli.seed}"
    _n_train = int(len(np.load(PIPELINE / "splits" / cli.dataset / _sub / "train_idx.npy")))
    _paper_batch = 64 if cli.protocol == "tdc" else 256
    args.batch_size = min(_paper_batch, max(1, _n_train))
    args.warmup_epochs = 2
    args.hidden_size = 300
    args.depth = 3
    args.encoder_name = "CMPNN"
    args.init_lr = 1e-4
    args.final_lr = 1e-4
    args.max_lr = float(hp["max_lr"])
    args.dropout = float(hp["dropout"])
    # Paper fixes the predictor at 2 layers / hidden 300 for MoleculeNet fine-tuning.
    args.ffn_num_layers = int(hp.get("ffn_num_layers", 2))
    args.ffn_hidden_size = int(hp.get("ffn_hidden_size", 300))

    modify_train_args(args)
    makedirs(args.save_dir)
    logger, args.save_dir = initialize_exp(Namespace(**args.__dict__))

    # Capture preds/targets/best-val by monkey-patching MolFCL's predict/evaluate.
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
        run_training(args, args.pretrain, logger)
    finally:
        _predict_module.predict = orig_predict
        _run_training_module.predict = orig_predict
        _evaluate_module.evaluate_predictions = orig_eval
        _run_training_module.evaluate_predictions = orig_eval
    elapsed = time.time() - t0

    if captured["preds"] is None or captured["targets"] is None:
        print("[molfcl] missing preds/targets; aborting")
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
        "method": "molfcl",
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
    print(f"[molfcl] DONE: {out}  AM={agg_am} GM={agg_gm}  ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
