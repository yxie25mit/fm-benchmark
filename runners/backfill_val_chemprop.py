"""
Backfill val_metric into chemprop2/chemeleon hp_search metrics.json.

These wrappers never wrote val_metric, so pick_best_hp silently fell back to
test_metric_am (test-set leakage in HP selection). chemprop logs the val metric
it early-stopped on to TensorBoard:
  cls -> val/roc   reg -> val/rmse   qm -> val/mae
We take the best epoch's value (max for roc, min for rmse/mae) and write it as
val_metric. Idempotent: skips configs that already have a non-null val_metric.

Run with a python whose env HAS tensorboard (e.g. the molfcl env):
  .../envs/molfcl/bin/python runners/backfill_val_chemprop.py \
      --methods chemprop2 chemeleon --protocols v1_det v1_preshuffle v2_astartes
"""
import argparse
import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

PIPELINE = Path("/data/rbg/users/yxie25/molclr_chemprop/clean_pipeline_v1")
DATASETS = ["freesolv", "esol", "sider", "clintox", "bace", "bbbp",
            "lipo", "qm7", "tox21", "qm8", "hiv", "qm9"]
QM = {"qm7", "qm8", "qm9"}


def val_tag_and_best(dataset):
    """Return (tensorboard scalar tag, reducer) for this dataset."""
    meta = json.loads((PIPELINE / "cleaned" / f"{dataset}.meta.json").read_text())
    if meta["task_type"] == "cls":
        return "val/roc", max
    return ("val/mae" if dataset in QM else "val/rmse"), min


def best_val_from_tb(events_files, tag, reducer):
    """Reduce the tag across ALL events files for this run. Interrupted/resumed
    runs leave an empty version_0 stub plus the real version_1 — scanning only the
    first (alphabetically version_0) would miss the metric, so scan them all."""
    vals = []
    for ev in events_files:
        ea = EventAccumulator(str(ev))
        ea.Reload()
        if tag in ea.Tags().get("scalars", []):
            vals.extend(s.value for s in ea.Scalars(tag))
    return float(reducer(vals)) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["chemprop2", "chemeleon"])
    ap.add_argument("--protocols", nargs="+",
                    default=["v1_det", "v1_preshuffle", "v2_astartes"])
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    filled = skipped = missing_tb = already = 0
    for method in args.methods:
        for proto in args.protocols:
            for ds in args.datasets:
                tag, reducer = val_tag_and_best(ds)
                hp_root = PIPELINE / "results" / method / ds / proto / "hp_search"
                if not hp_root.is_dir():
                    continue
                for cfg_dir in hp_root.iterdir():
                    if not cfg_dir.is_dir() or not (cfg_dir / "_hp.json").exists():
                        continue
                    seed_dir = next(cfg_dir.glob("seed*_em0"), None)
                    if seed_dir is None:
                        continue
                    mfile = seed_dir / "metrics.json"
                    if not mfile.exists():
                        continue
                    m = json.loads(mfile.read_text())
                    if m.get("val_metric") is not None:
                        already += 1
                        continue
                    evs = list((seed_dir / "cp_out").rglob("events.out.tfevents.*"))
                    if not evs:
                        missing_tb += 1
                        continue
                    v = best_val_from_tb(evs, tag, reducer)
                    if v is None:
                        missing_tb += 1
                        continue
                    if args.dry_run:
                        skipped += 1
                        continue
                    m["val_metric"] = v
                    m["val_metric_source"] = f"tb:{tag}"
                    mfile.write_text(json.dumps(m, indent=2, default=str))
                    filled += 1
                print(f"  {method}/{ds}/{proto}: cumulative filled={filled} "
                      f"already={already} missing_tb={missing_tb}")
    print(f"\nDONE: filled={filled} already_had={already} "
          f"missing_tb={missing_tb} dry_run_would_fill={skipped}")


if __name__ == "__main__":
    main()
