"""
MolFormer train_one — single training trial. Approach C: imports MolFormer's
finetune modules directly via sys.path; materializes train/valid/test.csv from
our split indices into a per-trial work dir.

CLI matches the unified contract.
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
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

PIPELINE = Path(__file__).resolve().parents[2]   # clean_pipeline_v1/ (relocatable)
MOLFORMER_DIR = Path(os.environ.get("MOLFORMER_DIR", str(Path(__file__).resolve().parents[2] / "forks" / "molformer")))
FINETUNE_DIR = MOLFORMER_DIR / "finetune"
sys.path.insert(0, str(FINETUNE_DIR))
os.chdir(str(FINETUNE_DIR))

from rdkit import RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

# Pretrained MolFormer checkpoint. Default lives on NFS; many workers reading it
# concurrently deadlocks on NFS RPC, so set MOLFORMER_CKPT to a node-local copy
# (e.g. /scratch/...) before launching at concurrency.
MOLFORMER_CKPT = Path(os.environ.get(
    "MOLFORMER_CKPT",
    str(MOLFORMER_DIR / "Pretrained MoLFormer" / "checkpoints" / "N-Step-Checkpoint_3_30000.ckpt"),
))

# task -> (script_module, dataset_type, single_target_col, multi_dataset_name)
TASK_DISPATCH = {
    "bbbp":     ("finetune_pubchem_light_classification",            "classification", "p_np",  None),
    "bace":     ("finetune_pubchem_light_classification",            "classification", "Class", None),
    "esol":     ("finetune_pubchem_light",                           "regression",     None,    None),
    "freesolv": ("finetune_pubchem_light",                           "regression",     None,    None),
    "lipo":     ("finetune_pubchem_light",                           "regression",     None,    None),
    "qm7":      ("finetune_pubchem_light",                           "regression",     None,    None),
    "tox21":    ("finetune_pubchem_light_classification_multitask",  "classification", None,    "tox21"),
    "sider":    ("finetune_pubchem_light_classification_multitask",  "classification", None,    "sider"),
    "clintox":  ("finetune_pubchem_light_classification_multitask",  "classification", None,    "clintox"),
    # hiv is single-task; the multitask script's 1-task path has a head-shape bug
    # (BCE input/target mismatch on all configs), so route it to the single-task
    # classification script like bbbp/bace.
    "hiv":      ("finetune_pubchem_light_classification",            "classification", "HIV_active", None),
    # qm8/qm9 multi-target reg not supported by MolFormer upstream.
}

DEFAULT_HP = {
    "max_lr":          3e-5,
    "dropout":         0.1,
    "batch_size":      128,
    "ffn_hidden_dims": [768, 768, 768],
}

# The multitask script hardcodes the per-task column order of its prediction tensor
# (finetune_pubchem_light_classification_multitask.main, the `measure_names` block).
# Our meta target_columns can differ from this order (e.g. clintox), so the saved
# preds must be reordered to our column order before scoring against labels. This
# mirrors that block exactly — keep in sync if the upstream order ever changes.
MULTITASK_MEASURE_ORDER = {
    "tox21": ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
              'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'],
    "clintox": ['FDA_APPROVED', 'CT_TOX'],
    "sider": [
        'Hepatobiliary disorders', 'Metabolism and nutrition disorders',
        'Product issues', 'Eye disorders', 'Investigations',
        'Musculoskeletal and connective tissue disorders',
        'Gastrointestinal disorders', 'Social circumstances',
        'Immune system disorders', 'Reproductive system and breast disorders',
        'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
        'General disorders and administration site conditions',
        'Endocrine disorders', 'Surgical and medical procedures',
        'Vascular disorders', 'Blood and lymphatic system disorders',
        'Skin and subcutaneous tissue disorders',
        'Congenital, familial and genetic disorders', 'Infections and infestations',
        'Respiratory, thoracic and mediastinal disorders', 'Psychiatric disorders',
        'Renal and urinary disorders',
        'Pregnancy, puerperium and perinatal conditions',
        'Ear and labyrinth disorders', 'Cardiac disorders',
        'Nervous system disorders', 'Injury, poisoning and procedural complications'],
    "hiv": ["HIV_active"],
}


def molformer_pred_to_scores(preds, task_type, multi_name, target_cols):
    """Map raw saved MolFormer predictions to per-target positive-class scores aligned
    to target_cols (so roc_auc(label[:,t], score[:,t]) is correct).

    - Single-task cls: net output is 2-class logits, saved as (N, 2) = [neg, pos].
      Positive-class score = softmax(.,axis=1)[:, 1] (matches the upstream
      validation_epoch_end which scores F.softmax(preds)[:, 1]). -> (N, 1).
    - Multitask cls: net output is per-task sigmoid prob, saved as (N, T) but ordered
      by the script's hardcoded measure_names, NOT our meta order. Reorder columns to
      target_cols so they line up with the saved labels. No negation: the values are
      already positive-class probabilities.
    - Regression: returned unchanged.
    """
    if task_type != "cls":
        return preds
    if multi_name is None:
        # single-task two-column logits -> positive-class prob, shape (N, 1)
        logits = preds - preds.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs[:, 1:2]
    src_order = MULTITASK_MEASURE_ORDER[multi_name]
    col_for = {name: i for i, name in enumerate(src_order)}
    reorder = [col_for[name] for name in target_cols]
    return preds[:, reorder]


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def materialize_data_root(dataset, protocol, seed, work_dir, dataset_name,
                          target_cols=None, standardize=False):
    """MolFormer expects files named <dataset_name>_{train,valid,test}.csv (single-target
    regression scripts) AND/OR plain {train,valid,test}.csv (multitask scripts).
    Write both naming variants so all task scripts work.

    For regression, standardize target columns using TRAIN mean/std before writing.
    MolFormer's regression head is zero-init with no internal target scaling, so on
    large-magnitude targets (e.g. qm7 atomization energy, mean ~-1545) it predicts ~0
    and never learns the offset (observed MAE ~1458). Standardizing fixes this and is
    harmless for near-zero-mean targets. Returns {col: (mean, std)} so the caller can
    de-standardize predictions back to original units for the reported metric.
    """
    src = pd.read_csv(PIPELINE / "cleaned" / f"{dataset}.csv")
    sub = "v1_det_seed0" if protocol == "v1_det" else f"{protocol}_seed{seed}"
    splits_dir = PIPELINE / "splits" / dataset / sub
    tr = np.load(splits_dir / "train_idx.npy")
    va = np.load(splits_dir / "val_idx.npy")
    te = np.load(splits_dir / "test_idx.npy")
    work_dir.mkdir(parents=True, exist_ok=True)

    stats = {}
    if standardize and target_cols:
        train_df = src.iloc[tr]
        for c in target_cols:
            mu = float(train_df[c].mean())
            sd = float(train_df[c].std())
            if not np.isfinite(sd) or sd == 0:
                sd = 1.0
            stats[c] = (mu, sd)

    for split, idx in [("train", tr), ("valid", va), ("test", te)]:
        sub_df = src.iloc[idx].copy()
        for c, (mu, sd) in stats.items():
            sub_df[c] = (sub_df[c] - mu) / sd
        sub_df.to_csv(work_dir / f"{split}.csv", index=False)
        sub_df.to_csv(work_dir / f"{dataset_name}_{split}.csv", index=False)
    return stats


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
    # Split the effective batch into micro-batches of this size with matching gradient
    # accumulation. 0 = off (default) -> byte-identical behaviour to before.
    p.add_argument("--micro-batch-size", type=int,
                   default=int(os.environ.get("MOLFORMER_MICRO_BATCH", "0")))
    cli = p.parse_args()

    out = Path(cli.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "done.flag").exists():
        print(f"[molformer] already done: {out}")
        return

    # Any dataset not in the built-in table (TDC or a user's custom split): derive the
    # script from its meta.json. Single-target cls -> single-task classification, single-
    # target reg -> regression (same as esol/freesolv). Multitask has no upstream script.
    if cli.dataset not in TASK_DISPATCH:
        _m = json.loads((PIPELINE / "cleaned" / f"{cli.dataset}.meta.json").read_text())
        _target_cols = _m.get("target_columns") or ["Y"]
        if len(_target_cols) == 1:
            if _m["task_type"] == "cls":
                TASK_DISPATCH[cli.dataset] = ("finetune_pubchem_light_classification", "classification", _target_cols[0], None)
            else:
                TASK_DISPATCH[cli.dataset] = ("finetune_pubchem_light", "regression", None, None)
        else:
            # Multitask: one shared model with N task heads (same paradigm as the other
            # methods). Pass the columns in the user's order so the head trains and emits
            # in that order -> identity realignment. Classification uses the upstream
            # multitask script; regression uses our multitask-regression variant of it.
            os.environ["MOLFORMER_MEASURE_NAMES"] = json.dumps(_target_cols)
            MULTITASK_MEASURE_ORDER[cli.dataset] = _target_cols
            if _m["task_type"] == "cls":
                TASK_DISPATCH[cli.dataset] = ("finetune_pubchem_light_classification_multitask", "classification", None, cli.dataset)
            else:
                TASK_DISPATCH[cli.dataset] = ("finetune_pubchem_light_regression_multitask", "regression", None, cli.dataset)

    if cli.dataset not in TASK_DISPATCH:
        rec = {"method": "molformer", "dataset": cli.dataset,
               "error": "task not supported by MolFormer"}
        (out / "metrics.json").write_text(json.dumps(rec, indent=2))
        (out / "done.flag").touch()
        print(f"[molformer] SKIP: {cli.dataset} not supported")
        return

    hp = DEFAULT_HP.copy()
    if cli.hp_config != "default":
        with open(cli.hp_config) as f:
            hp.update(json.load(f))

    # MolFormer's collate pads every batch to its longest SMILES, so one 1300-token
    # outlier (sider has 24 molecules >500 chars) inflates the whole 128-batch and OOMs.
    # Micro-batching caps peak activations at the micro-batch while gradient
    # accumulation preserves the effective batch. Off unless explicitly requested.
    effective_batch = int(hp["batch_size"])
    micro_batch_size = int(cli.micro_batch_size or hp.get("micro_batch_size") or 0)
    accumulate_grad_batches = 1
    step_batch_size = effective_batch
    if micro_batch_size and micro_batch_size < effective_batch:
        if effective_batch % micro_batch_size != 0:
            raise SystemExit(
                f"[molformer] micro_batch_size={micro_batch_size} must divide "
                f"batch_size={effective_batch} so the effective batch is preserved")
        step_batch_size = micro_batch_size
        accumulate_grad_batches = effective_batch // micro_batch_size
        print(f"[molformer] micro-batching: {step_batch_size} x "
              f"{accumulate_grad_batches} = effective batch {effective_batch}")

    meta = json.loads((PIPELINE / "cleaned" / f"{cli.dataset}.meta.json").read_text())
    task_type = meta["task_type"]
    target_cols = meta["target_columns"]
    qm_dataset = cli.dataset in ("qm7", "qm8", "qm9")

    script_name, ds_type, target_col, multi_name = TASK_DISPATCH[cli.dataset]
    set_seed(cli.ensemble_member)

    # Training I/O on the shared NFS results volume is the throughput bottleneck at
    # concurrency: every batch is read and every epoch checkpoint written over NFS.
    # Stage both on node-local disk; results/ still receives only the final artifacts.
    scratch_root = (Path(os.environ.get("MOLFORMER_SCRATCH", "/var/tmp"))
                    / f"molformer_{os.getpid()}").absolute()
    work_data = (scratch_root / "_data").absolute()
    reg_stats = materialize_data_root(cli.dataset, cli.protocol, cli.seed, work_data,
                          dataset_name=(multi_name or cli.dataset),
                          target_cols=target_cols, standardize=(ds_type == "regression"))
    cp_folder = (scratch_root / "_cp").absolute()
    cp_folder.mkdir(parents=True, exist_ok=True)

    # Workers only pay off now that the data is node-local; the original silent hang
    # was fork + NFS. Small sets stay in-process.
    loader_workers = 4 if int(meta.get("n_after", 0)) >= 5000 else 0

    n_out = 1 if multi_name is None else len(target_cols)
    ffn_dims = list(hp["ffn_hidden_dims"]) + [n_out]

    argv = [
        f"finetune/{script_name}.py",
        "--device", "cuda",
        "--batch_size", str(step_batch_size),
        "--accumulate_grad_batches", str(accumulate_grad_batches),
        "--n_head", "12",
        "--n_layer", "12",
        "--n_embd", "768",
        "--d_dropout", f"{float(hp['dropout']):.6g}",
        "--dropout", f"{float(hp['dropout']):.6g}",
        "--lr_start", f"{float(hp['max_lr']):.6g}",
        "--num_workers", str(loader_workers),
        "--max_epochs", str(int(cli.epochs)),
        "--num_feats", "32",
        "--seed_path", str(MOLFORMER_CKPT),
        "--data_root", str(work_data),
        "--dims", *[str(d) for d in ffn_dims],
        "--checkpoints_folder", str(cp_folder),
        # Per-ensemble-member seed (default is a fixed 12345). Combined with the
        # on_load_checkpoint no-op below, this makes ensemble members genuinely
        # differ (dropout/shuffle), matching the 5-ensemble the other methods use.
        "--seed", str(12345 + cli.ensemble_member),
    ]
    if multi_name is not None:
        argv += ["--dataset_name", multi_name, "--num_classes", "2"]
    else:
        argv += ["--measure_name", target_col or target_cols[0], "--dataset_name", cli.dataset]
        if ds_type == "classification":
            argv += ["--num_classes", "2"]

    # The multitask script names its per-epoch results CSV with os.environ["LSB_JOBID"]
    # (an LSF cluster var). Off-cluster that KeyErrors; set a per-process unique id.
    os.environ.setdefault("LSB_JOBID", str(os.getpid()))

    mod = importlib.import_module(script_name)
    # single-task scripts name the class LightningModule; the multitask script
    # names it MultitaskModel. Same validation output keys (val_loss, pred).
    lm_class = getattr(mod, "LightningModule", None) or getattr(mod, "MultitaskModel", None)
    # on_load_checkpoint restores the checkpoint's saved RNG state, which forces every
    # ensemble member to identical training (degenerate ensemble). No-op it so the
    # per-member --seed takes effect and members genuinely differ.
    lm_class.on_load_checkpoint = lambda self, checkpoint: None
    # PL writes a 536MB last.ckpt every epoch, but nothing ever reads it: training never
    # resumes, best-epoch predictions are captured in memory below, and the scratch dir is
    # deleted at the end. On a spinning node-local disk that write dominates wall clock.
    import pytorch_lightning as pl
    pl.callbacks.ModelCheckpoint.save_checkpoint = lambda self, trainer, pl_module: None
    orig_val_epoch_end = lm_class.validation_epoch_end

    captor = {"best_val_loss": float("inf"), "best_test_preds": None,
              "best_val_preds": None, "best_epoch": -1}

    def patched_val_epoch_end(self, outputs):
        out_ = orig_val_epoch_end(self, outputs)
        try:
            # outputs[0] = val split, outputs[1] = test split (see PropertyPredictionDataModule)
            val_outs = outputs[0]
            val_loss = float(torch.stack([x["val_loss"] for x in val_outs]).mean().item())
            if val_loss < captor["best_val_loss"] and len(outputs) > 1:
                val_preds = torch.cat([x["pred"] for x in val_outs]).detach().cpu().numpy()
                test_preds = torch.cat([x["pred"] for x in outputs[1]]).detach().cpu().numpy()
                captor["best_val_loss"] = val_loss
                captor["best_val_preds"] = val_preds
                captor["best_test_preds"] = test_preds
                captor["best_epoch"] = int(self.current_epoch)
        except Exception as e:
            print(f"WARN: capture failed at epoch {getattr(self, 'current_epoch', '?')}: {e}")
        # Per-epoch heartbeat on the shared results volume: lets a monitor see live
        # progress mid-cell (molformer otherwise writes nothing here until completion).
        try:
            beat = out / "heartbeat.json"
            tmp = out / "heartbeat.json.tmp"
            tmp.write_text(json.dumps({"epoch": int(self.current_epoch),
                                       "total_epochs": int(cli.epochs),
                                       "ts": int(time.time())}))
            os.replace(tmp, beat)
        except Exception:
            pass
        return out_

    lm_class.validation_epoch_end = patched_val_epoch_end

    saved_argv = sys.argv
    sys.argv = argv
    t0 = time.time()
    try:
        mod.main()
    finally:
        sys.argv = saved_argv
        lm_class.validation_epoch_end = orig_val_epoch_end
    elapsed = time.time() - t0

    if captor["best_test_preds"] is None:
        print("[molformer] no test predictions captured; aborting")
        return

    preds = captor["best_test_preds"]
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
    # Map raw net outputs to per-target positive-class scores aligned to target_cols.
    preds = molformer_pred_to_scores(preds.astype(np.float64), task_type, multi_name, target_cols)
    test_csv = pd.read_csv(work_data / "test.csv")
    targets = test_csv[target_cols].to_numpy(dtype=np.float64)
    if targets.ndim == 1:
        targets = targets[:, None]
    # Regression targets were standardized for training; de-standardize preds AND the
    # (also-standardized) labels back to original units so the metric matches other methods.
    preds = preds.astype(np.float64)
    if reg_stats:
        for t, c in enumerate(target_cols):
            mu, sd = reg_stats[c]
            preds[:, t] = preds[:, t] * sd + mu
            targets[:, t] = targets[:, t] * sd + mu

    if targets.shape[0] != preds.shape[0]:
        print(f"[molformer] WARN: target rows {targets.shape[0]} != pred rows {preds.shape[0]}")

    np.save(out / "pred_test.npy", preds.astype(np.float64))
    np.save(out / "labels_test.npy", targets)

    per, agg_am, agg_gm = per_target_metric(preds.astype(np.float64), targets, task_type, qm_dataset)

    # val_metric: same per-target aggregation as test, on the captured val preds at the
    # best-val-loss epoch. Required by pick_best_hp (which errors if absent).
    val_metric = None
    val_preds = captor["best_val_preds"]
    if val_preds is not None:
        if val_preds.ndim == 1:
            val_preds = val_preds.reshape(-1, 1)
        val_preds = molformer_pred_to_scores(val_preds.astype(np.float64), task_type, multi_name, target_cols)
        valid_csv = pd.read_csv(work_data / "valid.csv")
        val_targets = valid_csv[target_cols].to_numpy(dtype=np.float64)
        if val_targets.ndim == 1:
            val_targets = val_targets[:, None]
        val_preds = val_preds.astype(np.float64)
        if reg_stats:
            for t, c in enumerate(target_cols):
                mu, sd = reg_stats[c]
                val_preds[:, t] = val_preds[:, t] * sd + mu
                val_targets[:, t] = val_targets[:, t] * sd + mu
        if val_targets.shape[0] == val_preds.shape[0]:
            # Save val preds+labels so pick_best_hp can re-score with the prescribed
            # TDC metric (harmless for MoleculeNet, where it is unused).
            try:
                np.save(out / "pred_val.npy", val_preds)
                np.save(out / "labels_val.npy", val_targets)
            except Exception:
                pass
            _, val_metric, _ = per_target_metric(val_preds, val_targets, task_type, qm_dataset)
        else:
            print(f"[molformer] WARN: val target rows {val_targets.shape[0]} != val pred rows {val_preds.shape[0]}")

    metrics = {
        "method": "molformer",
        "dataset": cli.dataset, "protocol": cli.protocol, "seed": cli.seed,
        "ensemble_member": cli.ensemble_member, "epochs": cli.epochs,
        "test_metric": agg_am, "test_metric_am": agg_am, "test_metric_gm": agg_gm,
        "test_per_target": per,
        "val_metric": val_metric,
        "val_metric_source": "captured_val_split_best_val_loss",
        "val_loss": (captor["best_val_loss"] if captor["best_val_loss"] != float("inf") else None),
        "best_epoch": captor["best_epoch"] if captor["best_epoch"] >= 0 else None,
        "n_targets": targets.shape[1], "task_type": task_type, "hp": hp,
        "elapsed_sec": round(elapsed, 1),
    }
    if accumulate_grad_batches > 1:
        metrics["micro_batch_size"] = step_batch_size
        metrics["accumulate_grad_batches"] = accumulate_grad_batches
    try:
        metrics["peak_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    except Exception:
        pass
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    (out / "done.flag").touch()
    # Predictions/metrics are saved, so the transient training checkpoints in _cp/ are
    # dead weight. molformer (unlike chemprop) otherwise never removes them, and they
    # accumulate into hundreds of GB and exhaust the shared-results quota.
    shutil.rmtree(scratch_root, ignore_errors=True)
    print(f"[molformer] DONE: {out}  AM={agg_am} GM={agg_gm}  ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
