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


def learning_curve(args):
    """Per (method, train-size): mean±std test metric across subsample repeats -> a curve."""
    rows = []
    for method in args.methods:
        lc_root = PIPELINE / "results" / method / args.dataset / "learning_curve" / args.protocol
        if not lc_root.exists():
            continue
        for size_dir in sorted(p for p in lc_root.iterdir() if p.is_dir()):
            digits = "".join(c for c in size_dir.name if c.isdigit())
            if not digits:
                continue
            size = int(digits)
            by_seed = {}   # each repeat = one point; average ensemble members within it
            for cell in size_dir.glob("seed*_em*/metrics.json"):
                seed = cell.parent.name.split("_em")[0]
                v = json.loads(cell.read_text()).get("test_metric")
                if v is not None:
                    by_seed.setdefault(seed, []).append(v)
            pts = [statistics.mean(vs) for vs in by_seed.values() if vs]
            if pts:
                std = statistics.pstdev(pts) if len(pts) > 1 else 0.0
                rows.append((method, size, statistics.mean(pts), std, len(pts)))
    if not rows:
        print(f"No learning-curve results under results/<method>/{args.dataset}/learning_curve/"
              f"{args.protocol}/. Has run_learning_curve.py finished?")
        return
    rows.sort(key=lambda r: (r[0], r[1]))
    print(f"\nLearning curve  |  {args.dataset}  protocol={args.protocol}\n")
    print(f"{'method':<14}{'train_size':>11}{'test_metric':>13}{'std':>9}{'repeats':>9}")
    print("-" * 56)
    for method, size, mean, std, n in rows:
        print(f"{method:<14}{size:>11}{mean:>13.4f}{std:>9.4f}{n:>9}")
    if args.out:
        import csv
        with open(args.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["method", "train_size", "test_metric_mean", "test_metric_std", "repeats"])
            w.writerows(rows)
        print(f"\nwrote {args.out}")
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
