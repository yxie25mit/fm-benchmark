#!/usr/bin/env python
"""wilcoxon_from_results.py — one command from pipeline results to the paired Wilcoxon table.

For ONE split protocol and its datasets, it:
  1. runs align_per_molecule.py on each dataset, which recovers which molecule every saved prediction row is
     (for runs made before or after the ids fix, or a mix), averages ensemble members by molecule, and writes
     per-method files  fold,id,smiles,y_true,y_pred ;
  2. stops if the raw files do not reproduce the pipeline's own _summary.json, or if a recovered metric differs
     from it without an explained correction (corrected cells — bug-2 molformer, mixed-version members — are
     listed: the pipeline's reported number for them was computed on mispaired rows);
  3. picks the Chemprop baseline per dataset — whichever of chemprop2 / chemprop2_nofp scores better on the
     VALIDATION folds of these same results — and builds one manifest comparing it against every foundation
     model, per dataset (and per target for multitask), using each dataset's headline metric;
  4. runs wilcoxon_paired.py on it once, so the Benjamini-Hochberg correction spans the whole protocol.
     If a molformer cell's row order cannot be decided from its predictions, the comparison is run under both
     possible orders and the LARGER p is reported (flagged order_ambiguous; BH uses the conservative p).

Only the results CSV is meant to leave your side; the per-molecule files stay in --workdir.

Run from the repo root with the chemprop2 env python (needed so the chemprop row-order replay matches):
  <chemprop2-env>/bin/python scripts/wilcoxon_from_results.py --protocol custom \
      --datasets caco2_time_sliding clint_hum_app_time_sliding --phase default \
      --molformer-python <molformer-env>/bin/python --out wilcoxon_results_time_sliding.csv

One call per protocol (e.g. time sliding, scaffold v1, scaffold v2). Output columns per comparison: the one-sided
test "foundation model better" (p_wilcoxon, q_bh, significant — the headline), the reverse test "Chemprop better"
(p_baseline_better, q_bh_baseline_better), which baseline was used and its validation scores, and for
classification a Brier-score test (brier_*: calibration + ranking, reported next to the ROC-AUC/PR-AUC test).
Molecules shared by several folds (scaffold splits) count once (see wilcoxon_paired.py).
Override the baseline with --baseline chemprop2 (all datasets) or --baseline-for DATASET=chemprop2_nofp.
"""
import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CHEMPROP_VARIANTS = {"chemprop2", "chemprop2_nofp"}


def load_module(filename):
    spec = importlib.util.spec_from_file_location(filename[:-3], HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--phase", default="default", help="default or hp_final")
    ap.add_argument("--config", default=None, help="config dir name if a phase dir holds several")
    ap.add_argument("--root", default=".", help="repo root (has cleaned/ splits/ results/)")
    ap.add_argument("--methods", nargs="+", default=None, help="restrict to these methods (default: all found)")
    ap.add_argument("--baseline", default="auto",
                    help="'auto' (default): per dataset, the Chemprop variant with the better validation score; "
                         "or a method name to use for every dataset")
    ap.add_argument("--baseline-for", nargs="*", default=[], metavar="DATASET=METHOD",
                    help="per-dataset baseline override, e.g. caco2_time_sliding=chemprop2_nofp")
    ap.add_argument("--metric", default=None, help="override the headline metric for all datasets")
    ap.add_argument("--molformer-python", default=None, help="molformer env python (needed for molformer rows)")
    ap.add_argument("--workdir", default="wilcoxon_run", help="per-molecule files go here (keep them private)")
    ap.add_argument("--out", default="wilcoxon_results.csv")
    ap.add_argument("--no-brier", action="store_true", help="skip the Brier-score test for classification datasets")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    align = load_module("align_per_molecule.py")
    baseline_for = dict(item.split("=", 1) for item in args.baseline_for)

    manifest, problems, excluded, corrected, alternatives = [], [], [], [], {}
    brier_manifest, baseline_info = [], {}
    for dataset in args.datasets:
        inputs = workdir / "inputs" / dataset
        for stale in inputs.glob("*.csv"):     # never let files from an earlier run into this manifest
            stale.unlink()
        cmd = [sys.executable, str(HERE / "align_per_molecule.py"), "--root", str(root), "--dataset", dataset,
               "--protocol", args.protocol, "--phase", args.phase,
               "--out", str(workdir / "aligned" / dataset), "--wilcoxon-dir", str(inputs)]
        if args.config:
            cmd += ["--config", args.config]
        if args.methods:
            cmd += ["--methods", *args.methods]
        if args.molformer_python:
            cmd += ["--molformer-python", args.molformer_python]
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(f"\n######## {dataset}\n{result.stdout.strip()}")
        if result.returncode != 0:
            problems.append(f"{dataset}: align_per_molecule.py failed:\n{result.stderr.strip()[-800:]}")
            continue
        for marker in ("DATA MISMATCH", "UNEXPLAINED"):
            if marker in result.stdout:
                problems.append(f"{dataset}: {marker} against the pipeline's _summary.json (see above)")
        for line in result.stdout.splitlines():
            if "CORRECTED folds" in line:
                corrected.append(f"{dataset}: {line.strip()}")
        incomplete = set()                   # any fold not aligned -> drop the whole method (no partial-fold tests)
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) > 3 and parts[2:4] == ["NOT", "ALIGNED"]:
                incomplete.add(parts[0])
                excluded.append(f"{dataset}: {parts[0]} fold {parts[1]} — {' '.join(parts[4:])[:140]}")

        meta = json.loads((root / "cleaned" / f"{dataset}.meta.json").read_text())
        metric = args.metric or align.headline_metric(meta)
        by_target = {}
        for f in sorted(inputs.glob("*.csv")):
            method, _, target = f.stem.partition("__")
            by_target.setdefault(target, {})[method] = f
        usable = sorted(v for v in CHEMPROP_VARIANTS
                        if v not in incomplete and by_target and all(v in files for files in by_target.values()))
        if dataset in baseline_for:
            baseline, choice = baseline_for[dataset], {"baseline_rule": "--baseline-for"}
        elif args.baseline != "auto":
            baseline, choice = args.baseline, {"baseline_rule": "--baseline"}
        else:
            baseline, choice = choose_baseline(root, dataset, args.protocol, args.phase, args.config, metric, align, usable)
        choice["baseline"] = baseline
        scores = (f"; validation chemprop2={choice['val_chemprop2']}, chemprop2_nofp={choice['val_chemprop2_nofp']}"
                  if "val_chemprop2" in choice else "")
        print(f"  baseline for {dataset}: {baseline}  ({choice['baseline_rule']}{scores})")
        is_classification = meta["task_type"] == "cls"
        for target, files in by_target.items():
            label = dataset + (f" / {target}" if target else "")
            if baseline not in files or baseline in incomplete:
                problems.append(f"{label}: baseline {baseline} is not aligned on every fold — cannot compare")
                continue
            for method, path in files.items():
                if method in CHEMPROP_VARIANTS or method in incomplete:
                    continue
                name = f"{label} / {method}"
                row = {"name": name, "metric": metric, "baseline": str(files[baseline]), "candidate": str(path)}
                manifest.append(row)
                baseline_info[name] = choice
                if is_classification and not args.no_brier:
                    brier_manifest.append({**row, "metric": "brier"})
                alt = inputs / "alt_order" / path.name
                if alt.exists():
                    alternatives[name] = {**row, "candidate": str(alt)}

    if problems:
        print("\nSTOPPED — fix these before running the test:\n  " + "\n  ".join(problems))
        sys.exit(1)
    if not manifest:
        print("\nno comparisons to run")
        sys.exit(1)
    wp = load_module("wilcoxon_paired.py")
    out = Path(args.out).resolve()
    print(f"\n######## Wilcoxon: {len(manifest)} comparisons (BH across all of them)")
    table = run_family(manifest, alternatives, workdir, "manifest", wp)
    for col in ("baseline", "baseline_rule", "val_metric", "val_chemprop2", "val_chemprop2_nofp"):
        table[col] = table["name"].map(lambda n: baseline_info[n].get(col))
    if brier_manifest:
        brier_alt = {n: {**r, "metric": "brier"} for n, r in alternatives.items() if n in {b["name"] for b in brier_manifest}}
        brier = run_family(brier_manifest, brier_alt, workdir, "manifest_brier", wp).set_index("name")
        for src, dst in (("n_units", "brier_n_units"), ("median_delta", "brier_median_delta"), ("p_wilcoxon", "brier_p"),
                         ("q_bh", "brier_q"), ("p_baseline_better", "brier_p_baseline_better"),
                         ("q_bh_baseline_better", "brier_q_baseline_better")):
            table[dst] = table["name"].map(brier[src])
    table.to_csv(out, index=False)
    cols = [c for c in ["name", "metric", "baseline", "n_folds", "n_distinct", "n_units", "median_delta", "p_wilcoxon",
                        "q_bh", "significant", "p_baseline_better", "q_bh_baseline_better", "brier_p", "brier_q",
                        "order_ambiguous", "note"] if c in table.columns]
    with pd.option_context("display.width", 260, "display.max_columns", 40):
        print(table[cols].to_string(index=False))
    print(f"\nwrote {out}")
    if alternatives:
        print("\norder_ambiguous rows: molformer's row order could not be decided from its predictions, so each test "
              "was run under both possible orders and the LARGER p is reported (p_split_order, p_loader_order, "
              "conclusion_agrees).")
    if corrected:
        print("\nCorrected cells (the pipeline's reported metric for these was computed on mispaired rows; the "
              "recovered one is right):\n  " + "\n  ".join(corrected))
    if excluded:
        print("\nExcluded (could not be aligned — re-run these cells with the current code):\n  " + "\n  ".join(excluded))


def run_family(rows, alternatives, workdir, tag, wp):
    """Run one BH family of comparisons. For undecidable molformer rows the same comparison is also run under the
    other possible row order and the LARGER p is kept for each direction (conservative); BH is recomputed."""
    manifest_path = workdir / f"{tag}.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    out = workdir / f"{tag}_results.csv"
    res = subprocess.run([sys.executable, str(HERE / "wilcoxon_paired.py"), "--manifest", str(manifest_path),
                          "--out", str(out)], capture_output=True, text=True)
    if res.stderr.strip():
        print(res.stderr.strip())
    if res.returncode != 0:
        print(res.stdout.strip()); sys.exit(res.returncode)
    table = pd.read_csv(out)
    table["order_ambiguous"] = table["name"].isin(alternatives)
    if alternatives:
        alt_path, alt_out = workdir / f"{tag}_alt_order.csv", workdir / f"{tag}_alt_order_results.csv"
        pd.DataFrame(list(alternatives.values())).to_csv(alt_path, index=False)
        res = subprocess.run([sys.executable, str(HERE / "wilcoxon_paired.py"), "--manifest", str(alt_path),
                              "--out", str(alt_out)], capture_output=True, text=True)
        if res.returncode != 0:
            print(res.stdout + res.stderr); sys.exit(res.returncode)
        alt = pd.read_csv(alt_out).set_index("name")
        for i, row in table[table.order_ambiguous].iterrows():
            other = alt.loc[row["name"]]
            table.loc[i, "p_split_order"], table.loc[i, "p_loader_order"] = row.p_wilcoxon, other.p_wilcoxon
            table.loc[i, "conclusion_agrees"] = (row.p_wilcoxon < 0.05) == (other.p_wilcoxon < 0.05) and \
                (row.p_baseline_better < 0.05) == (other.p_baseline_better < 0.05)
            if other.p_wilcoxon > row.p_wilcoxon:
                for col in ("n_units", "median_delta", "statistic", "p_wilcoxon"):
                    table.loc[i, col] = other[col]
            table.loc[i, "p_baseline_better"] = max(row.p_baseline_better, other.p_baseline_better)
        table["q_bh"] = wp.bh(table.p_wilcoxon)
        table["significant"] = table.q_bh < 0.05
        table["q_bh_baseline_better"] = wp.bh(table.p_baseline_better)
        table["baseline_significantly_better"] = table.q_bh_baseline_better < 0.05
    return table


MAXIMIZED = {"roc-auc", "pr-auc", "spearman"}


def validation_score(root, method, dataset, protocol, phase, config, metric, align):
    """Mean over folds of the members' validation score for this method's results, from the saved validation
    predictions (pred_val.npy vs labels_val.npy, written side by side, so no alignment is needed), falling back to
    each member's metrics.json val_metric. Returns (score_from_predictions, score_from_val_metric)."""
    phase_root = root / "results" / method / dataset / protocol
    base = phase_root / phase
    if not base.exists():
        return None, None
    mapping, problem = align.seed_config_dirs(base, phase_root, phase, config)
    if problem or not mapping:
        return None, None
    from_preds, from_json = [], []
    for seed, cfg in sorted(mapping.items()):
        fold_preds, fold_json = [], []
        for member in sorted(cfg.glob(f"seed{seed}_em*")):
            pv, lv = member / "pred_val.npy", member / "labels_val.npy"
            if pv.exists() and lv.exists():
                pred, lab = align.as2d(np.load(pv)).astype(float), align.as2d(np.load(lv)).astype(float)
                if pred.shape[1] == 2 and lab.shape[1] == 1:
                    pred = pred[:, 1:2]
                if pred.shape == lab.shape:
                    fold_preds.append(align.cell_metric(metric, lab, pred))
            if (member / "metrics.json").exists():
                value = json.loads((member / "metrics.json").read_text()).get("val_metric")
                if value is not None:
                    fold_json.append(float(value))
        if fold_preds:
            from_preds.append(float(np.nanmean(fold_preds)))
        if fold_json:
            from_json.append(float(np.nanmean(fold_json)))
    n_folds = len(mapping)
    return (float(np.mean(from_preds)) if len(from_preds) == n_folds else None,
            float(np.mean(from_json)) if len(from_json) == n_folds else None)


def choose_baseline(root, dataset, protocol, phase, config, metric, align, usable):
    """chemprop2 vs chemprop2_nofp: of the variants aligned on every fold (`usable`), the one with the better
    validation score on these results. Both are scored from the same source (saved validation predictions if
    both have them, else metrics.json val_metric)."""
    if len(usable) < 2:
        return (usable[0] if usable else "chemprop2"), {
            "baseline_rule": "only aligned Chemprop variant" if usable else "no aligned Chemprop variant"}
    scores = {v: validation_score(root, v, dataset, protocol, phase, config, metric, align) for v in usable}
    source = 0 if all(s[0] is not None for s in scores.values()) else (1 if all(s[1] is not None for s in scores.values()) else None)
    if source is None:
        have = [v for v, s in scores.items() if s[0] is not None or s[1] is not None]
        chosen = have[0] if len(have) == 1 else "chemprop2"
        return chosen, {"baseline_rule": ("only variant with validation scores" if len(have) == 1
                                          else "no validation scores found — defaulted to chemprop2")}
    values = {v: s[source] for v, s in scores.items()}
    better = max if metric in MAXIMIZED else min
    best = better(values.values())
    chosen = "chemprop2" if values["chemprop2"] == best else "chemprop2_nofp"   # tie -> chemprop2
    return chosen, {"baseline_rule": "better validation " + ("(saved val predictions)" if source == 0 else "(val_metric)"),
                    "val_metric": metric if source == 0 else "val_metric",
                    "val_chemprop2": round(values["chemprop2"], 6), "val_chemprop2_nofp": round(values["chemprop2_nofp"], 6)}


if __name__ == "__main__":
    main()
