"""Status/heal helper for v2_astartes hp_search + hp_final watchdog.

Subcommands:
  count   print total remaining jobs (hp_search incomplete configs +
          hp_final incomplete ems for cells that already have best_hp.json)
  heal    delete _summary.json for incomplete cells so the orchestrator
          re-runs them instead of skipping (cell_done() only checks the summary)
  report  per-method/dataset breakdown
"""
import sys, glob, os, json
from pathlib import Path

PIPELINE = Path("/data/rbg/users/yxie25/molclr_chemprop/clean_pipeline_v1")
sys.path.insert(0, str(PIPELINE / "methods"))
from configs import all_hp_configs  # noqa: E402

METHODS = ["molclr", "chemprop2", "chemeleon", "molfcl", "motil"]
DATASETS = ["freesolv", "esol", "sider", "clintox", "bace", "bbbp",
            "lipo", "qm7", "tox21", "qm8", "hiv"]
PROTO = "v2_astartes"
HP_FINAL_NEEDED = 15  # 3 eval seeds x 5 ensemble members


def hp_search_done(m, d):
    base = PIPELINE / "results" / m / d / PROTO / "hp_search"
    return len(glob.glob(str(base / "*" / "seed*_em0" / "done.flag")))


def best_hp_id(m, d):
    f = PIPELINE / "results" / m / d / PROTO / "best_hp.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text()).get("id")
    except Exception:
        return None


def hp_final_done(m, d, best_id):
    base = PIPELINE / "results" / m / d / PROTO / "hp_final" / str(best_id)
    return len(glob.glob(str(base / "seed*_em*" / "done.flag")))


def remaining_for_cell(m, d):
    ncfg = len(all_hp_configs(m))
    hs = hp_search_done(m, d)
    rem = max(0, ncfg - hs)
    bid = best_hp_id(m, d)
    if bid is not None:
        rem += max(0, HP_FINAL_NEEDED - hp_final_done(m, d, bid))
    # if best_hp not yet produced, hp_final work is blocked (not counted here)
    return rem, ncfg, hs, bid


def cmd_count():
    total = 0
    for m in METHODS:
        for d in DATASETS:
            total += remaining_for_cell(m, d)[0]
    print(total)


def cmd_heal():
    for m in METHODS:
        ncfg = len(all_hp_configs(m))
        for d in DATASETS:
            # hp_search summary stale if not all configs done
            hs_summary = PIPELINE / "results" / m / d / PROTO / "hp_search" / "_summary.json"
            if hs_summary.exists() and hp_search_done(m, d) < ncfg:
                hs_summary.unlink()
                print(f"  heal: removed {hs_summary} ({hp_search_done(m,d)}/{ncfg})")
            # hp_final summary stale if best_hp exists and <15 done
            bid = best_hp_id(m, d)
            hf_summary = PIPELINE / "results" / m / d / PROTO / "hp_final" / "_summary.json"
            if bid is not None and hf_summary.exists() and hp_final_done(m, d, bid) < HP_FINAL_NEEDED:
                hf_summary.unlink()
                print(f"  heal: removed {hf_summary} ({hp_final_done(m,d,bid)}/{HP_FINAL_NEEDED})")


def cmd_report():
    grand_hs = grand_hs_need = grand_hf = grand_hf_need = 0
    for m in METHODS:
        ncfg = len(all_hp_configs(m))
        for d in DATASETS:
            rem, ncfg, hs, bid = remaining_for_cell(m, d)
            hf = hp_final_done(m, d, bid) if bid else 0
            hf_need = HP_FINAL_NEEDED if bid else 0
            grand_hs += hs; grand_hs_need += ncfg
            grand_hf += hf; grand_hf_need += hf_need
            flag = "" if rem == 0 else f"  <-- {rem} remaining"
            print(f"  {m:10s}/{d:9s} hp_search {hs:3d}/{ncfg}  best_hp={'Y' if bid else '-'}  hp_final {hf:2d}/{hf_need}{flag}")
    print(f"\nTOTAL hp_search {grand_hs}/{grand_hs_need}   hp_final {grand_hf}/{grand_hf_need}")


if __name__ == "__main__":
    {"count": cmd_count, "heal": cmd_heal, "report": cmd_report}[sys.argv[1]]()
