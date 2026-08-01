"""Single source of truth for scoring, incl. TDC-prescribed metrics.

Used by run_phase (val selection + test aggregation) so val and test are scored
by the IDENTICAL function with the dataset's prescribed metric (meta["metric"]).

Supported metric ids (case-insensitive, '-' and '_' interchangeable):
  reg, minimize : mae, rmse, scaled_mae
  reg, MAXIMIZE : spearman(r), pearson(r)
  cls, MAXIMIZE : roc_auc / auc, pr_auc / auprc / average_precision
"""
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    mean_absolute_error, mean_squared_error,
)
from scipy.stats import spearmanr, pearsonr

# metrics where HIGHER is better
_MAXIMIZE = {"roc_auc", "auc", "pr_auc", "auprc", "average_precision",
             "spearman", "spearmanr", "pearson", "pearsonr"}
_MINIMIZE = {"mae", "rmse", "mse", "scaled_mae"}


def _norm(metric: str) -> str:
    return metric.strip().lower().replace("-", "_")


def maximize(metric: str) -> bool:
    """True if a higher value of `metric` is better (so HP selection maximizes it)."""
    m = _norm(metric)
    if m in _MAXIMIZE:
        return True
    if m in _MINIMIZE:
        return False
    raise ValueError(f"unknown metric {metric!r} — add it to _tdc_metrics")


def _score_one(y, p, m):
    """Scalar metric on one target's non-NaN entries (y, p already 1-D, finite mask applied)."""
    if m in ("mae",):
        return float(mean_absolute_error(y, p))
    if m in ("rmse",):
        return float(np.sqrt(mean_squared_error(y, p)))
    if m in ("mse",):
        return float(mean_squared_error(y, p))
    if m in ("scaled_mae",):
        s = float(np.std(y))
        return float(mean_absolute_error(y, p) / s) if s > 0 else float("nan")
    if m in ("spearman", "spearmanr"):
        return float(spearmanr(y, p).correlation)
    if m in ("pearson", "pearsonr"):
        return float(pearsonr(y, p)[0])
    if m in ("roc_auc", "auc"):
        return float(roc_auc_score(y, p))
    if m in ("pr_auc", "auprc", "average_precision"):
        return float(average_precision_score(y, p))
    raise ValueError(f"unknown metric {m!r}")


def score(pred, labels, metric):
    """NaN-aware, multi-target-aware scalar score. pred/labels shape (N,) or (N,T).
    Per target: drop NaN labels; for cls metrics also skip single-class targets.
    Returns the macro-mean over scorable targets (== the single value for T=1)."""
    m = _norm(metric)
    pred = np.asarray(pred, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if pred.ndim == 1:
        pred = pred[:, None]
    if labels.ndim == 1:
        labels = labels[:, None]
    is_cls = m in ("roc_auc", "auc", "pr_auc", "auprc", "average_precision")
    vals = []
    for t in range(labels.shape[1]):
        y = labels[:, t]
        p = pred[:, t]
        ok = ~np.isnan(y) & ~np.isnan(p)
        y, p = y[ok], p[ok]
        if len(y) == 0:
            continue
        if is_cls and len(np.unique(y)) < 2:
            continue
        vals.append(_score_one(y, p, m))
    return float(np.mean(vals)) if vals else float("nan")
