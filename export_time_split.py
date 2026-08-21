"""
Bundle a time-split run into ONE shareable JSON of scalars — safe to send back (no raw
predictions, labels, or SMILES ever leave). Collects, per method x dataset x phase:

  - dataset sizes (N, per-fold train/val/test counts, task/metric/#targets, class balance)
  - _summary.json         : per-fold ensembled scores (per_seed_am) + agg mean/std
  - best_hp.json          : chosen HPs (single, or per-fold for the sliding dataset)
  - every seed<S>_em<E>/metrics.json, but ONLY its scalars (test_metric, val_metric,
                            test_per_target, hp, elapsed) — never pred_test/labels_test.

Datasets are <name>_chrono (1 fold) and <name>_sliding (K folds); phases are default and/or
hp_final. Single-target: test_per_target is a 1-element list. Multi-target: one entry per target.

  python export_time_split.py --name mydata --methods chemprop2 molclr molformer \
      --phases default hp_final --out mydata_share.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PIPELINE = Path(__file__).resolve().parent


def _meta(ds):
    p = PIPELINE / "cleaned" / f"{ds}.meta.json"
    return json.loads(p.read_text()) if p.exists() else None


def dataset_info(ds):
    meta = _meta(ds)
    if meta is None:
        return None
    targets = meta.get("target_columns", [])
    info = {"n_total": meta.get("n_after"), "n_targets": meta.get("n_targets", len(targets)),
            "task": meta.get("task_type"), "metric": meta.get("metric"), "targets": targets, "folds": []}
    for sd in sorted((PIPELINE / "splits" / ds).glob("custom_seed*")):
        seed = int(sd.name.replace("custom_seed", ""))
        sizes = {r: int(len(np.load(sd / f"{r}_idx.npy"))) for r in ("train", "val", "test")}
        info["folds"].append({"seed": seed, **sizes})
    # class balance (cls only) — counts per target, from the local cleaned CSV (only counts are exported)
    csv = PIPELINE / "cleaned" / f"{ds}.csv"
    if meta.get("task_type") == "cls" and csv.exists() and targets:
        df = pd.read_csv(csv)
        info["class_balance"] = {t: {"pos": int((df[t] == 1).sum()), "neg": int((df[t] == 0).sum()),
                                     "missing": int(df[t].isna().sum())} for t in targets if t in df.columns}
    return info


def _member(mj_path):
    d = json.loads(mj_path.read_text())
    return {"seed": d.get("seed"), "ensemble_member": d.get("ensemble_member"),
            "hp_id": mj_path.parent.parent.name if mj_path.parent.parent.name not in ("default",) else "default",
            "test_metric": d.get("test_metric"), "val_metric": d.get("val_metric"),
            "test_per_target": d.get("test_per_target"), "elapsed_sec": d.get("elapsed_sec"),
            "hp": d.get("hp")}


def phase_bundle(method, ds, phase):
    pdir = PIPELINE / "results" / method / ds / "custom" / phase
    if not pdir.exists():
        return None
    out = {}
    summ = pdir / "_summary.json"
    if summ.exists():
        s = json.loads(summ.read_text())
        out["summary"] = {"per_fold_ensembled": s.get("per_seed_am"), "agg": s.get("agg_am"),
                          "per_fold_hp": s.get("per_fold_hp")}
        if s.get("agg_per_target") is not None:   # multitask: ensembled per-target
            out["summary"]["per_target"] = {
                "target_columns": s.get("target_columns"),
                "per_fold_ensembled": s.get("per_seed_per_target"),  # [fold][target]
                "agg": s.get("agg_per_target"),                      # [target] -> {mean,std,n} over folds
            }
    bh = PIPELINE / "results" / method / ds / "custom" / "best_hp.json"
    if bh.exists() and phase == "hp_final":
        out["best_hp"] = json.loads(bh.read_text())
    out["members"] = [_member(p) for p in sorted(pdir.glob("**/seed*_em*/metrics.json"))]
    return out


def main():
    p = argparse.ArgumentParser(description="Export a time-split run as one shareable scalar JSON.")
    p.add_argument("--name", required=True, help="base name from prepare_user_data --time-split")
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--phases", nargs="+", default=["default", "hp_final"], choices=["default", "hp_final"])
    p.add_argument("--out", required=True, help="output JSON path (this is what you send back)")
    args = p.parse_args()

    datasets = [f"{args.name}_chrono", f"{args.name}_sliding"]
    bundle = {"name": args.name, "note": "scalars only; no predictions/labels/SMILES included",
              "dataset_info": {}, "results": {}}
    for ds in datasets:
        di = dataset_info(ds)
        if di:
            bundle["dataset_info"][ds] = di
    for method in args.methods:
        bundle["results"][method] = {}
        for ds in datasets:
            per_phase = {}
            for phase in args.phases:
                b = phase_bundle(method, ds, phase)
                if b is not None:
                    per_phase[phase] = b
            if per_phase:
                bundle["results"][method][ds] = per_phase

    Path(args.out).write_text(json.dumps(bundle, indent=2))
    # human-readable summary
    print(f"\nwrote {args.out}\n")
    print(f"{'method':<12}{'dataset':<10}{'phase':<10}{'per-fold ensembled':<34}{'mean±std':<16}")
    print("-" * 82)
    for method, dss in bundle["results"].items():
        for ds, phases in dss.items():
            tag = "chrono" if ds.endswith("_chrono") else "sliding"
            for phase, b in phases.items():
                s = b.get("summary", {})
                pf = s.get("per_fold_ensembled") or []
                agg = s.get("agg") or {}
                pf_s = "[" + ", ".join(f"{v:.4f}" if isinstance(v, (int, float)) else str(v) for v in pf) + "]"
                ms = (f"{agg.get('mean'):.4f} ± {agg.get('std'):.4f} (n={agg.get('n')})"
                      if agg.get("mean") is not None else "n/a")
                print(f"{method:<12}{tag:<10}{phase:<10}{pf_s:<34}{ms:<16}")


if __name__ == "__main__":
    main()
