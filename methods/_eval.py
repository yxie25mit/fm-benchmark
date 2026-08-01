"""
Shared per-target metric + AM/GM aggregation helpers.

Used by:
  - runners/run_phase.py for ensemble-averaged metric (mean over 5 ensemble preds).
  - (train_ones keep their inline copies because each runs in its own conda env.)

Dependencies: numpy + sklearn — both available in every method's env.
"""
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)


def per_target_metric(preds, labels, task_type, qm_dataset):
    """Per-target metric, NaN-aware.
       task_type: "cls" -> AUC.   "reg" -> MAE for QM, RMSE elsewhere.
       Returns list of length n_targets with None for unscorable targets."""
    if preds.ndim == 1:
        preds = preds[:, None]
    if labels.ndim == 1:
        labels = labels[:, None]
    per = []
    for ti in range(labels.shape[1]):
        yt, yp = labels[:, ti], preds[:, ti]
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
    return per


def aggregate_am_gm(per_target):
    """Arithmetic + geometric means across non-None entries."""
    valid_vals = [v for v in per_target if v is not None]
    if not valid_vals:
        return None, None
    am = float(np.mean(valid_vals))
    gm = (float(np.exp(np.mean(np.log(valid_vals))))
          if all(v > 0 for v in valid_vals) else None)
    return am, gm


def ensemble_metric_for_seed(seed_dirs, task_type, qm_dataset, metric=None):
    """Average pred matrices across ensemble members for one seed, then score.

    If `metric` is given (TDC: the dataset's prescribed metric id), score the averaged
    preds with the single prescribed metric via _tdc_metrics (am == gm == that value).
    Otherwise keep the MoleculeNet behavior (per-target macro AUC / RMSE / MAE)."""
    pred_list, labels = [], None
    for d in seed_dirs:
        pred_path = d / "pred_test.npy"
        if not pred_path.exists():
            continue
        pred_list.append(np.load(pred_path))
        if labels is None:
            labels = np.load(d / "labels_test.npy")
    if not pred_list:
        return None, None, None
    pred_avg = np.mean(np.stack(pred_list, axis=0), axis=0)
    if metric is not None:
        from _tdc_metrics import score as _tdc_score
        val = _tdc_score(pred_avg, labels, metric)
        return val, val, [val]
    per = per_target_metric(pred_avg, labels, task_type, qm_dataset)
    am, gm = aggregate_am_gm(per)
    return am, gm, per


def cross_seed_summary(per_seed_vals):
    vals = [v for v in per_seed_vals if v is not None]
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=0)), "n": len(vals)}
