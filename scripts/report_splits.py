"""Print detailed stats for the 3 split protocols x 12 datasets."""
import json
from pathlib import Path

ROOT = Path("/data/rbg/users/yxie25/molclr_chemprop/clean_pipeline_v1")
stats = json.loads((ROOT / "splits" / "_stats.json").read_text())

# ---------------------------------------------------------------------------
# Section 1: per-dataset scaffold landscape
# ---------------------------------------------------------------------------
print("="*120)
print("Section 1. Scaffold landscape per dataset (cleaned)")
print("="*120)
print(f"{'dataset':<10}{'N':>8}{'task':>5}{'#tgt':>5}{'#scaf':>8}{'#sing':>8}{'sing%':>8}{'max_clust':>11}")
print("-"*120)
for ds, s in stats.items():
    print(f"{ds:<10}{s['n_total']:>8}{s['task_type']:>5}{s['n_targets']:>5}"
          f"{s['n_scaffolds']:>8}{s['n_singletons']:>8}"
          f"{100*s['singleton_frac']:>7.1f}%{s['max_cluster_size']:>11}")

# ---------------------------------------------------------------------------
# Section 2: split shapes & test composition
# ---------------------------------------------------------------------------
print("\n"+"="*150)
print("Section 2. Split shapes + test scaffold composition (one row per split)")
print("="*150)
print(f"{'dataset':<10}{'split':<22}{'|tr|':>7}{'|va|':>7}{'|te|':>7}{'tr_scaf':>9}{'te_scaf':>9}"
      f"{'tr∩te':>7}{'te_sing':>9}{'te_sing%':>10}{'te_med':>8}{'te_max':>8}{'tr_max':>8}")
print("-"*150)
for ds, s in stats.items():
    for sp_name in ["v1_det_seed0",
                    "v1_preshuffle_seed0", "v1_preshuffle_seed1", "v1_preshuffle_seed2",
                    "v2_astartes_seed0",   "v2_astartes_seed1",   "v2_astartes_seed2"]:
        sp = s["splits"][sp_name]
        print(f"{ds:<10}{sp_name:<22}{sp['n_train']:>7}{sp['n_val']:>7}{sp['n_test']:>7}"
              f"{sp['n_scaffolds_train']:>9}{sp['n_scaffolds_test']:>9}"
              f"{sp['scaffold_overlap_train_test']:>7}{sp['test_singleton_count']:>9}"
              f"{100*sp['test_singleton_frac']:>9.1f}%{sp['test_scaf_median']:>8.1f}"
              f"{sp['test_scaf_max']:>8}{sp['train_scaf_max']:>8}")
    print()

# ---------------------------------------------------------------------------
# Section 3: Jaccard across seeds (test indices and test scaffolds)
# ---------------------------------------------------------------------------
print("="*130)
print("Section 3. Test-set Jaccard across seeds (higher = seeds give more similar tests = less variance)")
print("="*130)
print(f"{'dataset':<10}|{'v1_preshuffle (test idx)':^32}|{'v2_astartes (test idx)':^32}|"
      f"{'v1_preshuffle (test scaffolds)':^36}|{'v2_astartes (test scaffolds)':^32}")
print(f"{'':<10}|{'01':>8}{'02':>8}{'12':>8}{'mean':>8}|{'01':>8}{'02':>8}{'12':>8}{'mean':>8}|"
      f"{'01':>10}{'02':>10}{'12':>10}{'mean':>6}|{'01':>8}{'02':>8}{'12':>8}{'mean':>8}")
print("-"*144)
for ds, s in stats.items():
    j = s["jaccard_test_idx"]; sj = s["jaccard_test_scaffolds"]
    j_mean = (j["v1_preshuffle_01"] + j["v1_preshuffle_02"] + j["v1_preshuffle_12"]) / 3
    j2_mean = (j["v2_astartes_01"] + j["v2_astartes_02"] + j["v2_astartes_12"]) / 3
    sj_mean = (sj["v1_preshuffle_01"] + sj["v1_preshuffle_02"] + sj["v1_preshuffle_12"]) / 3
    sj2_mean = (sj["v2_astartes_01"] + sj["v2_astartes_02"] + sj["v2_astartes_12"]) / 3
    print(f"{ds:<10}|{j['v1_preshuffle_01']:>8.3f}{j['v1_preshuffle_02']:>8.3f}{j['v1_preshuffle_12']:>8.3f}{j_mean:>8.3f}|"
          f"{j['v2_astartes_01']:>8.3f}{j['v2_astartes_02']:>8.3f}{j['v2_astartes_12']:>8.3f}{j2_mean:>8.3f}|"
          f"{sj['v1_preshuffle_01']:>10.3f}{sj['v1_preshuffle_02']:>10.3f}{sj['v1_preshuffle_12']:>10.3f}{sj_mean:>6.3f}|"
          f"{sj['v2_astartes_01']:>8.3f}{sj['v2_astartes_02']:>8.3f}{sj['v2_astartes_12']:>8.3f}{sj2_mean:>8.3f}")

# ---------------------------------------------------------------------------
# Section 4: per-target test composition (cls: pos%, reg: target distribution shift)
# ---------------------------------------------------------------------------
print("\n"+"="*120)
print("Section 4. Test target distribution (first 3 targets shown for multi-target sets)")
print("="*120)
for ds, s in stats.items():
    task = s["task_type"]
    print(f"\n[{ds}] task={task}, n_targets={s['n_targets']}")
    if task == "cls":
        print(f"  {'split':<22} | per-target test pos% (n_valid pos/neg)")
        for sp_name in ["v1_det_seed0",
                        "v1_preshuffle_seed0","v1_preshuffle_seed1","v1_preshuffle_seed2",
                        "v2_astartes_seed0","v2_astartes_seed1","v2_astartes_seed2"]:
            sp = s["splits"][sp_name]
            tgts = sp["test_target_stats"][:3]
            cells = []
            for t in tgts:
                if t is None or t["n_valid"] == 0:
                    cells.append("N/A")
                else:
                    cells.append(f"{100*t['pos_frac']:5.1f}% ({t['n_pos']}+/{t['n_neg']}-)")
            print(f"  {sp_name:<22} | " + "   ".join(cells))
    else:
        print(f"  {'split':<22} | per-target test mean / std / [min,max]")
        for sp_name in ["v1_det_seed0",
                        "v1_preshuffle_seed0","v1_preshuffle_seed1","v1_preshuffle_seed2",
                        "v2_astartes_seed0","v2_astartes_seed1","v2_astartes_seed2"]:
            sp = s["splits"][sp_name]
            tgts = sp["test_target_stats"][:3]
            cells = []
            for t in tgts:
                if t is None:
                    cells.append("N/A")
                else:
                    cells.append(f"{t['mean']:7.3f}±{t['std']:6.3f} [{t['min']:7.3f},{t['max']:7.3f}]")
            print(f"  {sp_name:<22} | " + "  |  ".join(cells))
