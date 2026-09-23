#!/usr/bin/env python
"""wilcoxon_from_results.py — one command from pipeline results to the paired Wilcoxon table.

For ONE split protocol and its datasets, it:
  1. runs align_per_molecule.py on each dataset, which recovers which molecule every saved prediction row is
     (for runs made before or after the ids fix, or a mix), averages ensemble members by molecule, and writes
     per-method files  fold,id,smiles,y_true,y_pred ;
  2. stops if the raw files do not reproduce the pipeline's own _summary.json, or if a recovered metric differs
     from it without an explained correction (corrected cells — bug-2 molformer, mixed-version members — are
     listed: the pipeline's reported number for them was computed on mispaired rows);
  3. builds one manifest comparing the Chemprop baseline against every foundation model, per dataset (and per
     target for multitask), using each dataset's headline metric;
  4. runs wilcoxon_paired.py on it once, so the Benjamini-Hochberg correction spans the whole protocol.
     If a molformer cell's row order cannot be decided from its predictions, the comparison is run under both
     possible orders and the LARGER p is reported (flagged order_ambiguous; BH uses the conservative p).

Only the results CSV is meant to leave your side; the per-molecule files stay in --workdir.

Run from the repo root with the chemprop2 env python (needed so the chemprop row-order replay matches):
  <chemprop2-env>/bin/python scripts/wilcoxon_from_results.py --protocol custom \
      --datasets caco2_time_sliding clint_hum_app_time_sliding --phase default \
      --molformer-python <molformer-env>/bin/python --out wilcoxon_results_time_sliding.csv

One call per protocol (e.g. time sliding, scaffold v1, scaffold v2). The Chemprop baseline defaults to chemprop2;
pass --baseline-for DATASET=chemprop2_nofp where the no-descriptor variant scored better on validation.
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
    ap.add_argument("--baseline", default="chemprop2", help="Chemprop baseline for every dataset")
    ap.add_argument("--baseline-for", nargs="*", default=[], metavar="DATASET=METHOD",
                    help="per-dataset baseline override, e.g. caco2_time_sliding=chemprop2_nofp")
    ap.add_argument("--metric", default=None, help="override the headline metric for all datasets")
    ap.add_argument("--molformer-python", default=None, help="molformer env python (needed for molformer rows)")
    ap.add_argument("--workdir", default="wilcoxon_run", help="per-molecule files go here (keep them private)")
    ap.add_argument("--out", default="wilcoxon_results.csv")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    align = load_module("align_per_molecule.py")
    baseline_for = dict(item.split("=", 1) for item in args.baseline_for)

    manifest, problems, excluded, corrected, alternatives = [], [], [], [], {}
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
        baseline = baseline_for.get(dataset, args.baseline)
        by_target = {}
        for f in sorted(inputs.glob("*.csv")):
            method, _, target = f.stem.partition("__")
            by_target.setdefault(target, {})[method] = f
        for target, files in by_target.items():
            label = dataset + (f" / {target}" if target else "")
            if baseline not in files or baseline in incomplete:
                problems.append(f"{label}: baseline {baseline} is not aligned on every fold — cannot compare")
                continue
            for method, path in files.items():
                if method in CHEMPROP_VARIANTS or method in incomplete:
                    continue
                name = f"{label} / {method}"
                manifest.append({"name": name, "metric": metric,
                                 "baseline": str(files[baseline]), "candidate": str(path)})
                alt = inputs / "alt_order" / path.name
                if alt.exists():
                    alternatives[name] = {"name": name, "metric": metric,
                                          "baseline": str(files[baseline]), "candidate": str(alt)}

    if problems:
        print("\nSTOPPED — fix these before running the test:\n  " + "\n  ".join(problems))
        sys.exit(1)
    if not manifest:
        print("\nno comparisons to run")
        sys.exit(1)
    manifest_path = workdir / "manifest.csv"
    pd.DataFrame(manifest).to_csv(manifest_path, index=False)
    out = Path(args.out).resolve()
    print(f"\n######## Wilcoxon: {len(manifest)} comparisons (BH across all of them)")
    result = subprocess.run([sys.executable, str(HERE / "wilcoxon_paired.py"), "--manifest", str(manifest_path),
                             "--out", str(out)], capture_output=True, text=True)
    if result.stderr.strip():
        print(result.stderr.strip())
    if result.returncode != 0:
        print(result.stdout.strip())
        sys.exit(result.returncode)
    table = pd.read_csv(out)
    table["order_ambiguous"] = table["name"].isin(alternatives)
    if alternatives:
        # undecidable molformer order: same comparison under the other possible order; report the larger p
        alt_manifest = workdir / "manifest_alt_order.csv"
        pd.DataFrame(list(alternatives.values())).to_csv(alt_manifest, index=False)
        alt_out = workdir / "wilcoxon_alt_order.csv"
        res = subprocess.run([sys.executable, str(HERE / "wilcoxon_paired.py"), "--manifest", str(alt_manifest),
                              "--out", str(alt_out)], capture_output=True, text=True)
        if res.returncode != 0:
            print(res.stdout + res.stderr); sys.exit(res.returncode)
        alt = pd.read_csv(alt_out).set_index("name")
        wp = load_module("wilcoxon_paired.py")
        for i, row in table[table.order_ambiguous].iterrows():
            other = alt.loc[row["name"]]
            table.loc[i, "p_split_order"], table.loc[i, "p_loader_order"] = row.p_wilcoxon, other.p_wilcoxon
            table.loc[i, "conclusion_agrees"] = (row.p_wilcoxon < 0.05) == (other.p_wilcoxon < 0.05) and \
                np.sign(row.median_delta) == np.sign(other.median_delta)
            if other.p_wilcoxon > row.p_wilcoxon:
                for col in ("n_units", "median_delta", "statistic", "p_wilcoxon"):
                    table.loc[i, col] = other[col]
        table["q_bh"] = wp.bh(table.p_wilcoxon)
        table["significant"] = table.q_bh < 0.05
    table.to_csv(out, index=False)
    cols = [c for c in ["name", "metric", "unit", "n_folds", "n_units", "median_delta", "statistic", "p_wilcoxon",
                        "q_bh", "significant", "order_ambiguous", "note"] if c in table.columns]
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(table[cols].to_string(index=False))
    print(f"\nwrote {out}")
    if alternatives:
        print("\norder_ambiguous rows: molformer's row order could not be decided from its predictions, so the test "
              "was run under both possible orders and the LARGER p is reported (columns p_split_order, "
              "p_loader_order, conclusion_agrees).")
    if corrected:
        print("\nCorrected cells (the pipeline's reported metric for these was computed on mispaired rows; the "
              "recovered one is right):\n  " + "\n  ".join(corrected))
    if excluded:
        print("\nExcluded (could not be aligned — re-run these cells with the current code):\n  " + "\n  ".join(excluded))


if __name__ == "__main__":
    main()
