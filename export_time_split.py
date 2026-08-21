"""
Bundle a benchmark run into ONE shareable JSON of scalars — safe to send back (no raw
predictions, labels, or SMILES ever leave). Works for BOTH:

  time split :  --name mydata                         (-> mydata_chrono, mydata_sliding @ custom)
  scaffold   :  --datasets mydata --protocols v1_preshuffle v2_astartes
  any        :  --datasets A B --protocols custom v1_preshuffle ...

Per (dataset @ protocol) x method x phase it collects:
  - sizes + metric + metric_direction + per-fold train/val/test counts + (cls) per-fold test class balance
  - _summary.json      : per-fold ensembled score(s) + agg mean/std  (+ per-target for multitask)
  - best_hp.json       : chosen HPs (single, or per-fold for the sliding dataset)
  - completeness       : ensemble members found vs expected (flags partial/crashed cells)
  - val_summary        : per-fold mean member val_metric + agg (overfitting signal)
  - members            : every seed<S>_em<E>/metrics.json, ONLY scalars (test/val metric, hp, elapsed)
  - provenance         : git commit + UTC timestamp + methods/phases

  python export_time_split.py --name mydata --methods chemprop2 molclr --phases default hp_final \
      --out mydata_share.json
"""
import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PIPELINE = Path(__file__).resolve().parent
ENSEMBLE_SIZE = 5   # matches methods/configs.py ENSEMBLE_SIZE
MAXIMIZE = {"roc_auc", "auc", "pr_auc", "auprc", "average_precision",
            "spearman", "spearmanr", "pearson", "pearsonr"}


def _norm(m):
    return (m or "").lower().replace("-", "_")


def _meta(ds):
    p = PIPELINE / "cleaned" / f"{ds}.meta.json"
    return json.loads(p.read_text()) if p.exists() else None


def _eff_metric(meta):
    return meta.get("metric") or ("roc_auc" if meta.get("task_type") == "cls" else "rmse")


def _split_dir(ds, protocol, seed):
    name = "custom_seed{}".format(seed) if protocol == "custom" else (
        "v1_det_seed0" if protocol == "v1_det" else "{}_seed{}".format(protocol, seed))
    return PIPELINE / "splits" / ds / name


def _seeds_from_results(method, ds, protocol, phases):
    seeds = set()
    for phase in phases:
        for mj in (PIPELINE / "results" / method / ds / protocol).glob(f"{phase}/**/seed*_em*/metrics.json"):
            try:
                seeds.add(int(mj.parent.name.split("_em")[0].replace("seed", "")))
            except ValueError:
                pass
    return sorted(seeds)


def dataset_info(ds, protocol, seeds):
    meta = _meta(ds)
    if meta is None:
        return None
    targets = meta.get("target_columns", [])
    metric = _eff_metric(meta)
    info = {"n_total": meta.get("n_after"), "n_targets": meta.get("n_targets", len(targets)),
            "task": meta.get("task_type"), "metric": metric, "metric_direction": _direction(metric),
            "targets": targets, "folds": []}
    csv = PIPELINE / "cleaned" / f"{ds}.csv"
    df = pd.read_csv(csv) if (meta.get("task_type") == "cls" and csv.exists()) else None
    for s in seeds:
        d = _split_dir(ds, protocol, s)
        if not d.exists():
            continue
        fold = {"seed": s}
        for r in ("train", "val", "test"):
            fp = d / f"{r}_idx.npy"
            if fp.exists():
                fold[r] = int(len(np.load(fp)))
        if df is not None and (d / "test_idx.npy").exists():   # per-fold test class balance (counts only)
            te = np.load(d / "test_idx.npy")
            fold["test_class_balance"] = {t: {"pos": int((df.iloc[te][t] == 1).sum()),
                                              "neg": int((df.iloc[te][t] == 0).sum()),
                                              "missing": int(df.iloc[te][t].isna().sum())}
                                          for t in targets if t in df.columns}
        info["folds"].append(fold)
    return info


def _direction(metric):
    return "maximize" if _norm(metric) in MAXIMIZE else "minimize"


def _member(mj_path):
    d = json.loads(mj_path.read_text())
    parent = mj_path.parent.parent.name
    return {"seed": d.get("seed"), "ensemble_member": d.get("ensemble_member"),
            "hp_id": parent if parent != d.get("dataset") else "default",
            "test_metric": d.get("test_metric"), "val_metric": d.get("val_metric"),
            "test_per_target": d.get("test_per_target"), "elapsed_sec": d.get("elapsed_sec"),
            "hp": d.get("hp")}


def phase_bundle(method, ds, protocol, phase, seeds):
    pdir = PIPELINE / "results" / method / ds / protocol / phase
    if not pdir.exists():
        return None
    out = {}
    summ = pdir / "_summary.json"
    if summ.exists():
        s = json.loads(summ.read_text())
        out["summary"] = {"per_fold_ensembled": s.get("per_seed_am"), "agg": s.get("agg_am"),
                          "per_fold_hp": s.get("per_fold_hp")}
        if s.get("agg_per_target") is not None:
            out["summary"]["per_target"] = {"target_columns": s.get("target_columns"),
                                            "per_fold_ensembled": s.get("per_seed_per_target"),
                                            "agg": s.get("agg_per_target")}
    bh = PIPELINE / "results" / method / ds / protocol / "best_hp.json"
    if bh.exists() and phase == "hp_final":
        out["best_hp"] = json.loads(bh.read_text())
    members = [_member(p) for p in sorted(pdir.glob("**/seed*_em*/metrics.json"))]
    out["members"] = members
    # completeness: members found vs expected (n_folds x ENSEMBLE_SIZE)
    per_fold_found = {s: sum(1 for m in members if m["seed"] == s) for s in seeds}
    out["completeness"] = {"members_found": len(members),
                           "members_expected": len(seeds) * ENSEMBLE_SIZE,
                           "per_fold_found": per_fold_found,
                           "complete": all(per_fold_found.get(s, 0) == ENSEMBLE_SIZE for s in seeds)}
    # val summary (mean member val_metric per fold; overfitting signal vs test). Approximate:
    # it's the mean of members' own val, not an ensembled val (which needs raw val preds).
    val_per_fold = []
    for s in seeds:
        vs = [m["val_metric"] for m in members if m["seed"] == s and m["val_metric"] is not None]
        val_per_fold.append(float(np.mean(vs)) if vs else None)
    fin = [v for v in val_per_fold if v is not None]
    out["val_summary"] = {"per_fold_mean_member": val_per_fold, "note": "mean of members' own val (not ensembled)",
                          "agg": {"mean": float(np.mean(fin)), "std": float(np.std(fin)), "n": len(fin)} if fin else None}
    return out


def _git_commit():
    try:
        return subprocess.run(["git", "-C", str(PIPELINE), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or None
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser(description="Export a benchmark run (time or scaffold) as one shareable scalar JSON.")
    p.add_argument("--name", help="time-split convenience: exports <name>_chrono + <name>_sliding @ custom")
    p.add_argument("--datasets", nargs="+", help="explicit datasets (for scaffold or any protocol)")
    p.add_argument("--protocols", nargs="+", default=["custom"], help="protocols for --datasets (default custom)")
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--phases", nargs="+", default=["default", "hp_final"], choices=["default", "hp_final"])
    p.add_argument("--diversity", action="store_true",
                   help="also compute dataset diversity + per-fold train->test novelty (needs rdkit)")
    p.add_argument("--cluster-threshold", type=float, default=0.8, help="Tanimoto threshold for --diversity clusters")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    if args.name:
        cells = [(f"{args.name}_chrono", "custom"), (f"{args.name}_sliding", "custom")]
    elif args.datasets:
        cells = [(ds, proto) for ds in args.datasets for proto in args.protocols]
    else:
        p.error("provide --name (time split) or --datasets [+ --protocols]")

    bundle = {"provenance": {"git_commit": _git_commit(),
                             "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                             "methods": args.methods, "phases": args.phases},
              "note": "scalars only; no predictions/labels/SMILES included",
              "dataset_info": {}, "results": {}}
    for method in args.methods:
        bundle["results"][method] = {}
    for ds, proto in cells:
        key = f"{ds}@{proto}"
        any_seeds = sorted(set().union(*[set(_seeds_from_results(m, ds, proto, args.phases)) for m in args.methods])) if args.methods else []
        di = dataset_info(ds, proto, any_seeds)
        if di:
            bundle["dataset_info"][key] = di
            if args.diversity:   # embed diversity + per-fold train->test novelty into the same bundle
                from dataset_diversity import analyze as _div
                dv = _div(ds, proto, args.cluster_threshold)
                if dv:
                    di["scaffold_diversity"] = dv["scaffold_diversity"]
                    di["fingerprint_diversity"] = dv["fingerprint_diversity"]
                    di["clusters"] = dv["clusters"]
                    nov = {f["seed"]: f for f in dv["folds"]}
                    for fold in di["folds"]:
                        f = nov.get(fold["seed"])
                        if f:
                            fold["train_test_novelty"] = f["train_test_novelty"]
                            fold["scaffold_novelty"] = f["scaffold_novelty"]
        for method in args.methods:
            seeds = _seeds_from_results(method, ds, proto, args.phases)
            per_phase = {}
            for phase in args.phases:
                b = phase_bundle(method, ds, proto, phase, seeds)
                if b is not None:
                    per_phase[phase] = b
            if per_phase:
                bundle["results"][method][key] = per_phase

    Path(args.out).write_text(json.dumps(bundle, indent=2))
    print(f"\nwrote {args.out}  (git {(_git_commit() or '?')[:8]})\n")
    print(f"{'method':<12}{'dataset@proto':<26}{'phase':<10}{'per-fold':<28}{'mean±std':<20}{'complete':<9}")
    print("-" * 105)
    for method, cellmap in bundle["results"].items():
        for key, phases in cellmap.items():
            for phase, b in phases.items():
                s = b.get("summary", {})
                pf = s.get("per_fold_ensembled") or []
                agg = s.get("agg") or {}
                pf_s = "[" + ", ".join(f"{v:.4f}" if isinstance(v, (int, float)) else "NA" for v in pf) + "]"
                ms = f"{agg.get('mean'):.4f} ± {agg.get('std'):.4f}" if agg.get("mean") is not None else "n/a"
                comp = "yes" if b.get("completeness", {}).get("complete") else f"{b['completeness']['members_found']}/{b['completeness']['members_expected']}"
                print(f"{method:<12}{key:<26}{phase:<10}{pf_s:<28}{ms:<20}{comp:<9}")


if __name__ == "__main__":
    main()
