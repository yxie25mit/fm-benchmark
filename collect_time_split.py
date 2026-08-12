"""
Aggregate a time-split experiment produced by `prepare_user_data.py --csv --time-split`.

Reads the per-dataset _summary.json the pipeline already writes for:
  <name>_chrono    -> the single 80/10/10 chronological headline number
  <name>_sliding   -> the K sliding-window folds (multi-fold; each tuned on its OWN validation
                      via hp_per_fold, so its agg_am is already the mean±std across folds)

  python collect_time_split.py --name mydata --phase hp_final --methods chemprop2 molclr
"""
import argparse
import csv
import json
import math
from pathlib import Path

PIPELINE = Path(__file__).resolve().parent


def _summary(method, dataset, phase):
    f = PIPELINE / "results" / method / dataset / "custom" / phase / "_summary.json"
    return json.loads(f.read_text()) if f.exists() else None


def _finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def main():
    p = argparse.ArgumentParser(description="Aggregate a --time-split experiment (chrono + sliding).")
    p.add_argument("--name", required=True, help="base name passed to prepare_user_data --time-split")
    p.add_argument("--phase", default="hp_final", choices=["default", "hp_final"])
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--out", default=None, help="opt: write the table to this CSV")
    args = p.parse_args()

    rows = []
    for method in args.methods:
        ch = _summary(method, f"{args.name}_chrono", args.phase)
        sl = _summary(method, f"{args.name}_sliding", args.phase)
        chrono = ch.get("agg_am", {}).get("mean") if ch else None
        sw_mean = sw_std = None
        sw_n = 0
        if sl:
            per_seed = sl.get("per_seed_am", [])
            finite = [v for v in per_seed if _finite(v)]
            sw_n = len(finite)
            if finite and len(finite) == len(per_seed):
                # no bad folds -> reuse the pipeline's own agg_am so this table matches _summary.json
                agg = sl.get("agg_am", {})
                sw_mean, sw_std = agg.get("mean"), agg.get("std")
            elif finite:
                # a fold was non-finite (e.g. class-degenerate test chunk in a tiny smoke run):
                # drop it and recompute so one bad fold doesn't sink the whole aggregate.
                sw_mean = sum(finite) / sw_n
                sw_std = (sum((v - sw_mean) ** 2 for v in finite) / sw_n) ** 0.5
        if chrono is None and sw_mean is None:
            print(f"[warn] no results for {method} (looked under results/{method}/{args.name}_*/custom/{args.phase}/)")
            continue
        rows.append((method, chrono, sw_mean, sw_std, sw_n))

    if not rows:
        print("No time-split results found. Did the run_benchmark step finish?")
        return

    print(f"\nTime split  |  {args.name}  phase={args.phase}\n")
    print(f"{'method':<16}{'chrono(80/10/10)':>18}{'sliding_mean':>14}{'std':>9}{'n_folds':>9}")
    print("-" * 66)
    for method, chrono, sw_mean, sw_std, n in rows:
        cs = f"{chrono:.4f}" if _finite(chrono) else "n/a"
        ms = f"{sw_mean:.4f}" if _finite(sw_mean) else "n/a"
        ss = f"{sw_std:.4f}" if _finite(sw_std) else "n/a"
        print(f"{method:<16}{cs:>18}{ms:>14}{ss:>9}{n:>9}")

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["method", "chrono", "sliding_mean", "sliding_std", "n_folds"])
            w.writerows(rows)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
