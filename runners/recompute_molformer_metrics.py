"""
Recompute molformer test_metric_am/_gm + val_metric for ALREADY-COMPLETED runs from
saved predictions, WITHOUT retraining. Fixes the two metric-computation bugs:

  BUG 1 (inverted classification scores):
    - single-task cls (bace/bbbp/hiv): pred_test.npy is (N, 2) class logits; the
      positive-class score is softmax(.,axis=1)[:, 1], not column 0.
    - multitask cls (clintox/tox21/sider): pred_test.npy is (N, T) sigmoid probs but
      ordered by the upstream script's hardcoded measure_names, which can differ from
      our meta target_columns (e.g. clintox swaps CT_TOX / FDA_APPROVED). Reorder
      columns to match labels; no negation needed.

  BUG 2 (no val_metric): old runs only wrote val_loss. We can only recompute a true
    val_metric here if a saved val prediction exists (pred_val.npy + labels_val.npy);
    old runs predate the val-capture fix, so val_metric is left as-is unless those
    files are present. For old runs, the practical path to a real val_metric is to
    re-run train_one.py (now fixed) — this script only repairs test_metric in place.

Usage (dry-run prints, does NOT write unless --write):
  python runners/recompute_molformer_metrics.py --glob 'results/molformer/*/*/*/*'
  python runners/recompute_molformer_metrics.py --glob 'results/molformer/bace/v1_det/default/*' --write

DO NOT run broadly without coordinating: it mutates metrics.json / pred_test.npy in
results/. Default is dry-run.
"""
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score

PIPELINE = Path("/data/rbg/users/yxie25/molclr_chemprop/clean_pipeline_v1")

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

# dataset -> multi_name (None = single-task two-column logits)
MULTI_NAME = {
    "bbbp": None, "bace": None,
    "tox21": "tox21", "sider": "sider", "clintox": "clintox", "hiv": "hiv",
}


def pred_to_scores(preds, task_type, multi_name, target_cols):
    if task_type != "cls":
        return preds
    if multi_name is None:
        logits = preds - preds.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs[:, 1:2]
    src_order = MULTITASK_MEASURE_ORDER[multi_name]
    col_for = {name: i for i, name in enumerate(src_order)}
    return preds[:, [col_for[name] for name in target_cols]]


def per_target_metric(preds, targets, task_type, qm_dataset):
    per = []
    for ti in range(targets.shape[1]):
        yt, yp = targets[:, ti], preds[:, ti]
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
    vals = [v for v in per if v is not None]
    am = float(np.mean(vals)) if vals else None
    gm = (float(np.exp(np.mean(np.log(vals)))) if vals and all(v > 0 for v in vals) else None)
    return per, am, gm


def recompute_run(run_dir, write):
    run_dir = Path(run_dir)
    mfile = run_dir / "metrics.json"
    pfile = run_dir / "pred_test.npy"
    lfile = run_dir / "labels_test.npy"
    if not (mfile.exists() and pfile.exists() and lfile.exists()):
        return None
    m = json.loads(mfile.read_text())
    if m.get("error"):
        return None
    dataset = m["dataset"]
    if dataset not in MULTI_NAME:
        return None
    meta = json.loads((PIPELINE / "cleaned" / f"{dataset}.meta.json").read_text())
    task_type = meta["task_type"]
    target_cols = meta["target_columns"]
    qm_dataset = dataset in ("qm7", "qm8", "qm9")
    multi_name = MULTI_NAME[dataset]

    preds = np.load(pfile).astype(np.float64)
    labels = np.load(lfile).astype(np.float64)
    if preds.ndim == 1:
        preds = preds[:, None]
    if labels.ndim == 1:
        labels = labels[:, None]
    # Idempotency: only single-task raw saves are (N,2); once mapped they are (N,1).
    already_mapped = (task_type == "cls" and multi_name is None and preds.shape[1] == 1)
    scores = preds if already_mapped else pred_to_scores(preds, task_type, multi_name, target_cols)
    per, am, gm = per_target_metric(scores, labels, task_type, qm_dataset)
    old_am = m.get("test_metric_am")
    print(f"{run_dir}: dataset={dataset} old_AM={old_am} -> new_AM={am}")
    if write:
        m["test_metric"] = am
        m["test_metric_am"] = am
        m["test_metric_gm"] = gm
        m["test_per_target"] = per
        m["test_metric_recompute_source"] = "recompute_molformer_metrics"
        np.save(pfile, scores)
        mfile.write_text(json.dumps(m, indent=2, default=str))
    return am


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--glob", required=True,
                   help="glob of run dirs, e.g. 'results/molformer/*/*/*/*'")
    p.add_argument("--write", action="store_true",
                   help="actually overwrite metrics.json/pred_test.npy (default: dry-run)")
    args = p.parse_args()
    dirs = sorted({os.path.dirname(p) if os.path.isfile(p) else p
                   for p in glob.glob(args.glob)})
    n = 0
    for d in dirs:
        if recompute_run(d, args.write) is not None:
            n += 1
    print(f"\n{'WROTE' if args.write else 'DRY-RUN'} {n} runs.")


if __name__ == "__main__":
    main()
