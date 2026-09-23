#!/usr/bin/env python3
"""
Chemprop-style paired Wilcoxon signed-rank test for model comparison.

Compares a candidate model against a baseline model on the SAME test molecules,
using the prediction-level paired test described in Yang et al. (2019),
"Analyzing Learned Molecular Representations for Property Prediction",
J. Chem. Inf. Model. 59(8), 3370-3388.

WHAT IT TESTS
-------------
H0: the candidate's per-unit error distribution is the same as the baseline's.
H1 (one-sided): the candidate's error is SMALLER than the baseline's,
    i.e. the candidate is better.

The "unit" depends on the endpoint's metric:

  * MAE / RMSE regression   -> one unit per test molecule.
                               Molecule-level errors are directly comparable
                               because the metric decomposes over molecules.
  * Spearman regression     -> one unit per 1/30th of the test set.
  * ROC-AUC / PR-AUC        -> one unit per 1/30th of the test set.
                               A rank metric is NOT defined for a single
                               molecule, so the test set is partitioned into
                               30 equal parts, the metric is computed within
                               each part, and the test runs on those 30 values
                               (exactly the Chemprop paper's procedure).

INPUT FORMAT
------------
One CSV (or Parquet) per model per fold, or a single long-format table.
Long format is easiest -- one row per (fold, molecule):

    fold,smiles,y_true,y_pred
    0,CCO,0.42,0.51
    0,c1ccccc1,1.10,0.98
    ...

Predictions must be ENSEMBLE-AVERAGED before they reach this script if you
ensemble: average the member prediction vectors, then pass the average.
Do not average per-member metrics.

Rows are matched between the two models on (fold, smiles). Order does not
matter; the script aligns them. Duplicate SMILES within a fold are matched
positionally in file order. If both files also have an `id` column (a stable
per-row identifier), rows are matched on (fold, id) instead -- exact even for
repeated SMILES -- and the rank-metric partition no longer depends on row order.

USAGE
-----
    # regression, MAE endpoint
    python wilcoxon_paired.py \
        --baseline baseline_preds.csv \
        --candidate candidate_preds.csv \
        --metric mae \
        --name "Sol. FaSSIF / CheMeleon"

    # classification, ROC-AUC endpoint
    python wilcoxon_paired.py \
        --baseline base.csv --candidate cand.csv \
        --metric roc-auc

    # many comparisons at once, with multiplicity correction
    python wilcoxon_paired.py --manifest comparisons.csv --out results.csv

The manifest is a CSV with columns:
    name,metric,baseline,candidate
and the script applies Benjamini-Hochberg across all its rows.

OUTPUT
------
  n_units        number of paired units entering the test
  median_delta   median of (baseline_error - candidate_error); POSITIVE means
                 the candidate is better
  statistic      Wilcoxon signed-rank statistic
  p_wilcoxon     one-sided p-value for "candidate better than baseline"
  q_bh           Benjamini-Hochberg adjusted p-value (manifest mode only)

DETERMINISM
-----------
The 30-part partition is pseudo-random. The seed is derived from the
comparison name, so a given comparison always yields the same partition
regardless of how many other comparisons are run or in what order. Pass
--seed to override.
"""
import argparse
import hashlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score

N_PARTS = 30           # Chemprop paper: "divide all the test molecules into 30 equal parts"
MIN_UNITS = 10         # below this the signed-rank test has no useful power

METRICS = ("mae", "rmse", "spearman", "roc-auc", "pr-auc")
# Metrics where a larger value is better. Errors are always sign-flipped so
# that SMALLER is better, which is what the one-sided test assumes.
MAXIMIZE = {"mae": False, "rmse": False, "spearman": True,
            "roc-auc": True, "pr-auc": True}


def _seed_from(name, override=None):
    """Stable per-comparison seed, so results never depend on call order."""
    if override is not None:
        return int(override)
    h = hashlib.sha256(name.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big")


def load_preds(path):
    """Read a long-format prediction table and validate it."""
    df = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
    need = {"smiles", "y_true", "y_pred"}
    missing = need - set(df.columns)
    if missing:
        sys.exit(f"{path}: missing required column(s): {sorted(missing)}\n"
                 f"  found: {list(df.columns)}")
    if "fold" not in df.columns:
        df = df.assign(fold=0)
    keep = ["fold", "smiles", "y_true", "y_pred"] + (["id"] if "id" in df.columns else [])
    df = df[keep].copy()
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")
    if "id" in df.columns and df.duplicated(["fold", "id"]).any():
        sys.exit(f"{path}: (fold, id) is not unique -- each test row must appear once per fold")
    # positional index within (fold, smiles) so duplicate SMILES align 1:1 (legacy pairing, no id column)
    df["_rep"] = df.groupby(["fold", "smiles"]).cumcount()
    return df


def align(base, cand):
    """Pair the two tables row-for-row.

    If BOTH files carry an `id` column (a stable per-row identifier, e.g. the cleaned-CSV row index written by
    align_per_molecule.py), rows are paired on (fold, id) -- exact even when a SMILES repeats -- and the merged
    table is sorted by (fold, id), so the 30-part partition used for rank metrics does not depend on the order
    rows happen to appear in the files. Without `id`, the original behaviour is kept unchanged: pair on
    (fold, smiles, replicate index), duplicate SMILES matched positionally in file order."""
    if "id" in base.columns and "id" in cand.columns:
        m = base.drop(columns=["_rep"]).merge(cand.drop(columns=["_rep", "smiles"]), on=["fold", "id"],
                                              suffixes=("_b", "_c"))
        m = m.sort_values(["fold", "id"], kind="mergesort").reset_index(drop=True)
        key = "(fold, id)"
    else:
        if ("id" in base.columns) != ("id" in cand.columns):
            print("warning: only one file has an `id` column; pairing on (fold, smiles) instead", file=sys.stderr)
        m = base.merge(cand, on=["fold", "smiles", "_rep"], suffixes=("_b", "_c"))
        key = "(fold, smiles)"
    if m.empty:
        sys.exit(f"no {key} pairs in common between the two files")
    dropped_b, dropped_c = len(base) - len(m), len(cand) - len(m)
    if dropped_b or dropped_c:
        print(f"warning: {dropped_b} baseline and {dropped_c} candidate rows have no partner on {key} and are "
              f"excluded -- check both files cover the same test set", file=sys.stderr)
    bad = (m.y_true_b - m.y_true_c).abs() > 1e-8
    if bad.any():
        sys.exit(f"y_true disagrees on {int(bad.sum())} matched rows -- "
                 "the two files describe different labels")
    ok = m[["y_true_b", "y_pred_b", "y_pred_c"]].notna().all(axis=1)
    return m[ok]


def unit_errors(y, pred, metric, rng):
    """
    Per-unit 'error' values, sign-flipped so that lower is always better.

    Returns (errors, unit_name). For decomposable metrics there is one value
    per molecule; for rank metrics there are N_PARTS values per fold.
    """
    if metric in ("mae", "rmse"):
        e = np.abs(y - pred) if metric == "mae" else (y - pred) ** 2
        return e, "molecule"

    # Rank metrics: score within each of 30 partitions of this fold.
    parts = np.array_split(rng.permutation(len(y)), N_PARTS)
    if metric == "spearman":
        vals = [spearmanr(y[p], pred[p]).statistic if len(p) > 1 else np.nan
                for p in parts]
    else:
        score = roc_auc_score if metric == "roc-auc" else average_precision_score
        vals = []
        for p in parts:
            # a partition with only one class present has no defined AUC
            vals.append(score(y[p], pred[p]) if len(np.unique(y[p])) >= 2 else np.nan)
    v = np.asarray(vals, dtype=float)
    return (-v if MAXIMIZE[metric] else v), f"1/{N_PARTS} test-set chunk"


def compare(base_path, cand_path, metric, name, seed=None):
    if metric not in METRICS:
        sys.exit(f"unknown --metric {metric!r}; choose from {list(METRICS)}")
    m = align(load_preds(base_path), load_preds(cand_path))
    rng = np.random.default_rng(_seed_from(name, seed))

    eb_all, ec_all, unit = [], [], None
    for fold, g in m.groupby("fold", sort=True):
        y = g.y_true_b.to_numpy(float)
        # Both models share one partition per fold, so the pairing is preserved.
        rng_fold = np.random.default_rng(_seed_from(f"{name}|fold={fold}", seed))
        eb, unit = unit_errors(y, g.y_pred_b.to_numpy(float), metric, rng_fold)
        rng_fold = np.random.default_rng(_seed_from(f"{name}|fold={fold}", seed))
        ec, _ = unit_errors(y, g.y_pred_c.to_numpy(float), metric, rng_fold)
        eb_all.append(eb)
        ec_all.append(ec)

    eb = np.concatenate(eb_all)
    ec = np.concatenate(ec_all)
    keep = np.isfinite(eb) & np.isfinite(ec)
    eb, ec = eb[keep], ec[keep]

    row = {"name": name, "metric": metric, "unit": unit,
           "n_folds": int(m.fold.nunique()), "n_units": int(len(eb))}
    if len(eb) < MIN_UNITS:
        row.update(statistic=np.nan, p_wilcoxon=np.nan, median_delta=np.nan,
                   note=f"only {len(eb)} usable units (< {MIN_UNITS})")
        return row
    if np.allclose(eb, ec):
        row.update(statistic=np.nan, p_wilcoxon=1.0, median_delta=0.0,
                   note="identical predictions")
        return row

    # alternative="less": candidate error stochastically smaller => candidate better
    stat, p = wilcoxon(ec, eb, alternative="less", zero_method="wilcox")
    row.update(statistic=float(stat), p_wilcoxon=float(p),
               median_delta=float(np.median(eb - ec)), note="")
    return row


def bh(pvals):
    """Benjamini-Hochberg adjusted p-values, NaN-safe."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    idx = np.flatnonzero(np.isfinite(p))
    if idx.size == 0:
        return out
    q = p[idx]
    order = np.argsort(q)
    n = q.size
    adj = q[order] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    res = np.empty(n)
    res[order] = np.clip(adj, 0, 1)
    out[idx] = res
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Chemprop-style paired Wilcoxon signed-rank test.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", help="baseline prediction CSV/Parquet")
    ap.add_argument("--candidate", help="candidate prediction CSV/Parquet")
    ap.add_argument("--metric", help=f"one of {list(METRICS)}")
    ap.add_argument("--name", default="comparison", help="label for this comparison")
    ap.add_argument("--manifest", help="CSV with columns name,metric,baseline,candidate")
    ap.add_argument("--out", help="write results here as CSV")
    ap.add_argument("--seed", type=int, default=None,
                    help="override the partition seed (default: derived from --name)")
    a = ap.parse_args()

    if a.manifest:
        man = pd.read_csv(a.manifest)
        need = {"name", "metric", "baseline", "candidate"}
        if not need <= set(man.columns):
            sys.exit(f"manifest needs columns {sorted(need)}; found {list(man.columns)}")
        rows = [compare(r.baseline, r.candidate, r.metric, r["name"], a.seed)
                for _, r in man.iterrows()]
        res = pd.DataFrame(rows)
        res["q_bh"] = bh(res.p_wilcoxon)
        res["significant"] = res.q_bh < 0.05
    else:
        if not (a.baseline and a.candidate and a.metric):
            sys.exit("need --baseline, --candidate and --metric (or --manifest)")
        res = pd.DataFrame([compare(a.baseline, a.candidate, a.metric, a.name, a.seed)])

    cols = ["name", "metric", "unit", "n_folds", "n_units",
            "median_delta", "statistic", "p_wilcoxon"]
    cols += [c for c in ("q_bh", "significant", "note") if c in res.columns]
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(res[cols].to_string(index=False))
    if a.out:
        res.to_csv(a.out, index=False)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
