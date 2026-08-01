"""
Assemble learning-curve points from run_learning_curve outputs.

For each (method, dataset, protocol, fraction): ensemble-average the test predictions
across ensemble members (per seed), compute the test metric, then mean/std across outer
seeds. The f=1.00 point is pulled from the existing phases (default HP -> `default`
phase; tuned HP -> `hp_final`), NOT recomputed.

Emits a tidy JSON: rows of {method, dataset, protocol, frac, n_train, metric_mean,
metric_std, n_seeds}. Metric direction is per dataset (cls: ROC-AUC up; reg: RMSE/MAE down).
"""
import json
import numpy as np
from pathlib import Path

PIPELINE = Path("/data/rbg/users/yxie25/molclr_chemprop/clean_pipeline_v1")
RES = PIPELINE / "results"
SPLITS = PIPELINE / "splits"
CLEANED = PIPELINE / "cleaned"

FRAC_TAGS = ["f005", "f010", "f020", "f040", "f070"]
FRAC_VAL = {"f005": 0.05, "f010": 0.10, "f020": 0.20, "f040": 0.40, "f070": 0.70, "f100": 1.0}


def n_train_for(dataset, base, ftag, seed):
    if ftag == "f100":
        d = SPLITS / dataset / f"{base}_seed{seed}"
    else:
        d = SPLITS / dataset / f"{base}__{ftag}_seed{seed}"
    p = d / "train_idx.npy"
    return int(len(np.load(p))) if p.exists() else None


def cell_seeds_members(method, dataset, base, ftag):
    """Return {seed: [member_dirs...]} for completed cells of one fraction point."""
    root = RES / method / dataset / "learning_curve" / base / ftag
    out = {}
    if not root.exists():
        return out
    for cell in sorted(root.glob("seed*_em*")):
        if not (cell / "done.flag").exists():
            continue
        name = cell.name  # seedS_emE
        seed = int(name.split("seed")[1].split("_")[0])
        out.setdefault(seed, []).append(cell)
    return out


def metric_from_members(member_dirs):
    """Ensemble-average pred_test across members, else fall back to single metric.json.
    Returns the test metric using each cell's own recorded value when only one member."""
    if len(member_dirs) == 1:
        m = json.loads((member_dirs[0] / "metrics.json").read_text())
        return m.get("test_metric")
    preds = []
    labels = None
    task_type = None
    for d in member_dirs:
        pv = d / "pred_test.npy"
        lv = d / "labels_test.npy"
        if not pv.exists():
            continue
        preds.append(np.load(pv))
        labels = np.load(lv)
        task_type = json.loads((d / "metrics.json").read_text()).get("task_type")
    if not preds:
        return None
    mean_pred = np.mean(preds, axis=0)
    return score(mean_pred, labels, task_type)


def score(pred, labels, task_type):
    from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
    pred = np.asarray(pred).ravel()
    labels = np.asarray(labels).ravel()
    mask = ~np.isnan(labels)
    if task_type == "cls":
        return float(roc_auc_score(labels[mask], pred[mask]))
    return float(np.sqrt(mean_squared_error(labels[mask], pred[mask])))


def f100_point(method, dataset, base):
    """f=1.00 default-HP point from the existing `default` phase summary (mean over seeds)."""
    summ = RES / method / dataset / base / "default" / "_summary.json"
    if not summ.exists():
        return None
    d = json.loads(summ.read_text())
    agg = d.get("agg_am") or {}
    return agg.get("mean"), agg.get("std"), agg.get("n")


def main():
    methods = ["chemprop2", "molclr", "chemeleon", "molfcl", "motil", "molformer"]
    datasets = ["bace", "esol"]
    protocols = ["v1_preshuffle", "v2_astartes"]
    meta_cache = {}
    rows = []
    for method in methods:
        for dataset in datasets:
            tt = meta_cache.setdefault(
                dataset, json.loads((CLEANED / f"{dataset}.meta.json").read_text())["task_type"])
            for base in protocols:
                for ftag in FRAC_TAGS:
                    by_seed = cell_seeds_members(method, dataset, base, ftag)
                    seed_metrics, n_train = [], None
                    for seed, members in by_seed.items():
                        val = metric_from_members(members)
                        if val is not None:
                            seed_metrics.append(val)
                            n_train = n_train or n_train_for(dataset, base, ftag, seed)
                    if not seed_metrics:
                        continue
                    rows.append({
                        "method": method, "dataset": dataset, "protocol": base,
                        "frac": FRAC_VAL[ftag], "ftag": ftag, "n_train": n_train,
                        "task": tt, "metric_mean": float(np.mean(seed_metrics)),
                        "metric_std": float(np.std(seed_metrics)), "n_seeds": len(seed_metrics),
                        "source": "learning_curve",
                    })
                # f100 from existing default phase
                pt = f100_point(method, dataset, base)
                if pt and pt[0] is not None:
                    rows.append({
                        "method": method, "dataset": dataset, "protocol": base,
                        "frac": 1.0, "ftag": "f100",
                        "n_train": n_train_for(dataset, base, "f100", 0),
                        "task": tt, "metric_mean": float(pt[0]),
                        "metric_std": float(pt[1] or 0.0), "n_seeds": int(pt[2] or 0),
                        "source": "default_phase",
                    })
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
