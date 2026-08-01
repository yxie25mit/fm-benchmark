"""
Re-run pick_best_hp (val-based) and rewrite best_hp.json + hp_search/_summary.json
for the given methods/datasets/protocols, AFTER val_metric has been backfilled.

This does NO training — it only re-scans existing hp_search metrics.json and
re-selects the best config by val_metric (pick_best_hp now refuses test fallback).

  .../envs/foldeverything/bin/python runners/repick_best_hp.py \
      --methods molclr chemprop2 chemeleon --protocols v1_det v1_preshuffle v2_astartes
"""
import argparse
import json
import sys
from pathlib import Path

PIPELINE = Path("/data/rbg/users/yxie25/molclr_chemprop/clean_pipeline_v1")
sys.path.insert(0, str(PIPELINE / "runners"))
sys.path.insert(0, str(PIPELINE / "methods"))

from run_phase import pick_best_hp  # noqa: E402

ALL_DS = ["freesolv", "esol", "sider", "clintox", "bace", "bbbp",
          "lipo", "qm7", "tox21", "qm8", "hiv", "qm9"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--protocols", nargs="+",
                    default=["v1_det", "v1_preshuffle", "v2_astartes"])
    ap.add_argument("--datasets", nargs="+", default=ALL_DS)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    changed = unchanged = errored = 0
    for method in args.methods:
        for proto in args.protocols:
            for ds in args.datasets:
                phase_dir = PIPELINE / "results" / method / ds / proto / "hp_search"
                if not phase_dir.is_dir():
                    continue
                best_id, best_hp, best_score = pick_best_hp(phase_dir, ds)
                if best_id is None:
                    print(f"  ERROR {method}/{ds}/{proto}: no val_metric — skipped")
                    errored += 1
                    continue
                bh_path = phase_dir.parent / "best_hp.json"
                old = json.loads(bh_path.read_text()) if bh_path.exists() else {}
                new = {"hp": best_hp, "id": best_id, "val_score": best_score}
                tag = "UNCHANGED" if old.get("id") == best_id else \
                      f"CHANGED {old.get('id')}->{best_id}"
                if tag != "UNCHANGED":
                    changed += 1
                else:
                    unchanged += 1
                print(f"  {method}/{ds}/{proto}: id={best_id} val={best_score:.4g}  [{tag}]")
                if not args.dry_run:
                    bh_path.write_text(json.dumps(new, indent=2))
                    summ = {"best_id": best_id, "best_hp": best_hp,
                            "best_val_score": best_score}
                    (phase_dir / "_summary.json").write_text(json.dumps(summ, indent=2, default=str))
    print(f"\nDONE: changed={changed} unchanged={unchanged} errored={errored}"
          + ("  (dry-run, nothing written)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
