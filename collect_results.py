"""
collect_results.py — gather results across methods into one table.

The pipeline writes per-method summaries (mean±std across folds/seeds) at
  results/<method>/<dataset>/<protocol>/<phase>/_summary.json
This script collects them for one dataset+protocol+phase into a table (printed + CSV),
so you don't have to open each method's summary by hand.

Example:
  python collect_results.py --dataset acme --protocol custom --phase default
  python collect_results.py --dataset acme --protocol custom --phase hp_final \
    --methods chemeleon molclr molformer --out acme_results.csv
"""
import argparse
import json
import statistics
from pathlib import Path

PIPELINE = Path(__file__).resolve().parent
ALL_METHODS = ["chemprop2", "chemprop2_nofp", "chemeleon", "chemeleon_nofp",
               "molclr", "molfcl", "motil", "molformer"]


def _metric_name(task_type, qm, prescribed):
    if prescribed:
        return prescribed
    return "mae" if (task_type != "cls" and qm) else ("roc_auc" if task_type == "cls" else "rmse")


def learning_curve(args):
    """Per (method, train-size): the ENSEMBLE test metric per fold, then mean±std across folds.

    Ensembling is metric-of-mean (average the members' pred_test.npy, score once) via the SAME
    ensemble_metric_for_seed the main pipeline/default table uses — NOT mean-of-members — so the
    curve reports true ensemble performance and its full-size point matches the headline table.
    Error bars are the spread across folds (--seeds = the sliding folds)."""
    import sys
    sys.path.insert(0, str(PIPELINE / "methods"))
    from _eval import ensemble_metric_for_seed  # noqa: E402

    meta_path = PIPELINE / "cleaned" / f"{args.dataset}.meta.json"
    if not meta_path.exists():
        print(f"No meta.json for {args.dataset} (looked in cleaned/). Was the dataset prepared?")
        return
    meta = json.loads(meta_path.read_text())
    task_type = meta["task_type"]
    qm = args.dataset in ("qm7", "qm8", "qm9")
    prescribed_metric = meta["metric"] if meta.get("source") == "tdc" else None
    metric_name = _metric_name(task_type, qm, prescribed_metric)

    rows = []       # (method, size, mean, std, n_folds) — for the table/CSV/plot
    detail = {}     # method -> size -> {fold_seed: ensemble_metric}
    for method in args.methods:
        lc_root = PIPELINE / "results" / method / args.dataset / "learning_curve" / args.protocol
        if not lc_root.exists():
            continue
        for size_dir in sorted(p for p in lc_root.iterdir() if p.is_dir()):
            digits = "".join(c for c in size_dir.name if c.isdigit())
            if not digits:
                continue
            size = int(digits)
            # group ensemble-member dirs by fold (seed), then metric-of-mean within each fold
            by_fold = {}
            for cell in size_dir.glob("seed*_em*"):
                if cell.is_dir():
                    by_fold.setdefault(cell.name.split("_em")[0], []).append(cell)
            fold_vals = {}
            for seed, member_dirs in by_fold.items():
                am, _, _ = ensemble_metric_for_seed(sorted(member_dirs), task_type, qm,
                                                    metric=prescribed_metric)
                if am is not None:
                    fold_vals[seed] = am
            if fold_vals:
                vals = list(fold_vals.values())
                std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
                rows.append((method, size, statistics.mean(vals), std, len(vals)))
                detail.setdefault(method, {})[size] = fold_vals
    if not rows:
        print(f"No learning-curve results under results/<method>/{args.dataset}/learning_curve/"
              f"{args.protocol}/. Has run_learning_curve.py finished?")
        return
    rows.sort(key=lambda r: (r[0], r[1]))
    print(f"\nLearning curve  |  {args.dataset}  protocol={args.protocol}  metric={metric_name} "
          f"(ensemble = mean of member predictions, scored once)\n")
    print(f"{'method':<14}{'train_size':>11}{metric_name[:11]:>13}{'std':>9}{'n_folds':>9}   per-fold")
    print("-" * 74)
    for method, size, mean, std, n in rows:
        pf = detail[method][size]
        pf_str = " ".join(f"{s}={v:.4f}" for s, v in sorted(pf.items()))
        print(f"{method:<14}{size:>11}{mean:>13.4f}{std:>9.4f}{n:>9}   {pf_str}")
    if args.out:
        import csv
        with open(args.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["method", "train_size", "metric", "test_metric_mean",
                        "test_metric_std", "n_folds", "per_fold"])
            for method, size, mean, std, n in rows:
                pf = detail[method][size]
                pf_str = ";".join(f"{s}={v}" for s, v in sorted(pf.items()))
                w.writerow([method, size, metric_name, mean, std, n, pf_str])
        print(f"\nwrote {args.out}")
        # companion JSON with the full per-fold breakdown for programmatic use / sharing
        jpath = str(args.out).rsplit(".", 1)[0] + ".folds.json"
        payload = {"dataset": args.dataset, "protocol": args.protocol, "metric": metric_name,
                   "ensembling": "metric-of-mean (avg member predictions, score once)",
                   "curves": {m: {str(sz): {"mean": statistics.mean(fv.values()),
                                            "std": statistics.pstdev(list(fv.values())) if len(fv) > 1 else 0.0,
                                            "n_folds": len(fv), "per_fold": fv}
                                  for sz, fv in sizes.items()}
                              for m, sizes in detail.items()}}
        with open(jpath, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"wrote {jpath}")
    if args.plot:
        _plot_curve(rows, args)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--protocol", default="custom")
    p.add_argument("--phase", default="default", choices=["default", "hp_final"])
    p.add_argument("--methods", nargs="+", default=ALL_METHODS)
    p.add_argument("--out", default=None, help="opt: write the table to this CSV")
    p.add_argument("--learning-curve", action="store_true",
                   help="collect a train-size learning curve instead of a single-phase table")
    p.add_argument("--plot", default=None, help="learning-curve only: save a PNG plot to this path")
    args = p.parse_args()

    if args.learning_curve:
        learning_curve(args)
        return

    rows = []
    for method in args.methods:
        summ = PIPELINE / "results" / method / args.dataset / args.protocol / args.phase / "_summary.json"
        if not summ.exists():
            continue
        agg = json.loads(summ.read_text()).get("agg_am", {})
        rows.append((method, agg.get("mean"), agg.get("std"), agg.get("n")))

    if not rows:
        print(f"No _summary.json found for {args.dataset}/{args.protocol}/{args.phase}. "
              f"Has the run finished?")
        return

    rows.sort(key=lambda r: (r[1] is None, -(r[1] or 0)))  # best mean first
    print(f"\n{args.dataset}  |  protocol={args.protocol}  phase={args.phase}\n")
    print(f"{'method':<16}{'test_metric':>14}{'std':>10}{'n_folds':>9}")
    print("-" * 49)
    for method, mean, std, n in rows:
        mean_s = f"{mean:.4f}" if mean is not None else "n/a"
        std_s = f"{std:.4f}" if std is not None else "n/a"
        print(f"{method:<16}{mean_s:>14}{std_s:>10}{str(n):>9}")

    if args.out:
        import csv
        with open(args.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["method", "test_metric_mean", "test_metric_std", "n_folds"])
            w.writerows(rows)
        print(f"\nwrote {args.out}")


def _plot_curve(rows, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not installed; skipping plot.")
        return
    from collections import defaultdict
    by_method = defaultdict(list)
    for method, size, mean, std, n in rows:
        by_method[method].append((size, mean, std))
    fig, ax = plt.subplots(figsize=(6, 4))
    for method, pts in sorted(by_method.items()):
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        es = [p[2] for p in pts]
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=method)
    ax.set_xscale("log")
    ax.set_xlabel("train size")
    ax.set_ylabel("test metric")
    ax.set_title(f"Learning curve — {args.dataset}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.plot, dpi=120)
    print(f"[plot] wrote {args.plot}")


if __name__ == "__main__":
    main()
