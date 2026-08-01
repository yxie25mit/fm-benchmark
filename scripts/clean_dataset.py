#!/usr/bin/env python
"""Step 1 of the clean MoleculeNet comparison pipeline: data cleaning.

For each of the 12 MoleculeNet datasets we use ours_*.csv as the raw input
(those are the same files used by every recent run: MolFCL HP, MolFormer,
chemprop reproduction). They are MolCLR raw with invalid SMILES already
pre-dropped, so the molecule set matches prior runs exactly.

Pipeline (one rule, no per-dataset overrides):

  1. Read raw CSV.
  2. Parse with rdkit; drop rows where MolFromSmiles is None or 0 heavy atoms.
  3. Canonicalize: Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True).
  4. Deduplicate by canonical SMILES:
       classification target: same -> keep / disagree -> set that target NaN.
       regression target: per-dataset tolerance (abs_tol, rel_tol). A duplicate
                          group is averaged if range(values) <= abs_tol OR
                          range/|mean| <= rel_tol; else that target -> NaN.
                          Tolerances are picked from the property's typical
                          measurement repeatability (experimental targets) or
                          float precision (computational quantum-chemistry).
       (single-target datasets fall through naturally: NaN target -> all-NaN row -> dropped in step 5.)
  5. Drop rows where every target is NaN.
  6. Write:
       cleaned/{dataset}.csv         columns = [smiles, target_0, target_1, ...]
       cleaned/{dataset}_mask.npy    bool array (n_rows, n_targets), True = valid label
       cleaned/{dataset}.meta.json   provenance + counts

Run:
  python scripts/clean_dataset.py            # all 12 datasets
  python scripts/clean_dataset.py --tasks freesolv,bace
"""

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

# silence rdkit's "WARNING: not removing hydrogen ..." spam during MolFromSmiles
RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------------------
# Configuration: where to read raw data from and which columns are targets.
# ---------------------------------------------------------------------------
RAW_DIR = Path("/data/rbg/users/yxie25/molclr_chemprop/MolFCL/data")
OUT_DIR = Path("/data/rbg/users/yxie25/molclr_chemprop/clean_pipeline_v1/cleaned")

# Per-dataset config: (raw_filename, task_type, target_columns, reg_tol).
# task_type: "cls" = binary classification, "reg" = regression.
# reg_tol (only used for "reg"): {"abs": float, "rel": float}.
#   For each duplicate-group's values for a target, average if either
#     (max - min) <= abs   OR   (max - min) / |mean| <= rel
#   else set that target to NaN.
#
# Tolerance choice rationale:
#   - Experimental targets (esol/freesolv/lipo): abs threshold ~ typical
#     measurement repeatability for that property. ESOL/FreeSolv duplicates
#     observed to differ by <0.25 log-units, occasionally up to ~1 log-unit
#     for very-hydrophilic molecules (sorbitol-type) where saturation is hard
#     to measure -- those should be dropped, not averaged.
#   - Computational quantum-chemistry targets (qm7/8/9): values are DFT
#     outputs; same molecule should give identical values. Any disagreement
#     beyond float precision is a data integrity issue. Use rel < 1e-3 only.
DATASETS = {
    # single-target classification
    "bbbp":     ("ours_bbbp.csv",        "cls", ["p_np"],         None),
    "bace":     ("ours_bace.csv",        "cls", ["Class"],        None),
    "hiv":      ("ours_hiv.csv",         "cls", ["HIV_active"],   None),
    # multi-target classification
    "clintox":  ("ours_clintox.csv",     "cls", ["CT_TOX", "FDA_APPROVED"], None),
    "tox21":    ("ours_tox21.csv",       "cls", [
        "NR-AR","NR-AR-LBD","NR-AhR","NR-Aromatase","NR-ER","NR-ER-LBD",
        "NR-PPAR-gamma","SR-ARE","SR-ATAD5","SR-HSE","SR-MMP","SR-p53",
    ], None),
    "sider":    ("ours_sider.csv",       "cls", None, None),  # 27 targets, fill in dynamically
    # single-target regression (experimental: tolerate small measurement noise)
    "esol":     ("ours_esol.csv",        "reg", ["measured log solubility in mols per litre"],
                                                              {"abs": 0.5, "rel": 0.0}),
    "freesolv": ("ours_freesolv.csv",    "reg", ["expt"],     {"abs": 0.5, "rel": 0.0}),
    "lipo":     ("ours_lipo.csv",        "reg", ["exp"],      {"abs": 0.3, "rel": 0.0}),
    # qm7 is calc atomization energy: treat as computational (exact-only)
    "qm7":      ("ours_qm7.csv",         "reg", ["u0_atom"],  {"abs": 0.0, "rel": 1e-3}),
    # multi-target regression (computational: only float-precision noise tolerated)
    "qm8":      ("ours_qm8.csv",         "reg", [
        "E1-CC2","E2-CC2","f1-CC2","f2-CC2",
        "E1-PBE0","E2-PBE0","f1-PBE0","f2-PBE0",
        "E1-CAM","E2-CAM","f1-CAM","f2-CAM",
    ], {"abs": 0.0, "rel": 1e-3}),
    "qm9":      ("ours_qm9_seed0.csv",   "reg", ["mu","alpha","homo","lumo","gap","r2","zpve","cv"],
                                                              {"abs": 0.0, "rel": 1e-3}),
}


# ---------------------------------------------------------------------------
# Per-step helpers, one function per pipeline step so the comments stay tight.
# ---------------------------------------------------------------------------

def step2_parse_and_filter_invalid(df, smiles_col):
    """Step 2. Parse SMILES with rdkit; drop rows where MolFromSmiles is None
    or the molecule has zero heavy atoms (single-atom ions, empty strings)."""
    mols = [Chem.MolFromSmiles(s) if isinstance(s, str) else None for s in df[smiles_col]]
    valid = [m is not None and m.GetNumHeavyAtoms() > 0 for m in mols]
    n_invalid = (~np.array(valid)).sum()
    return df.loc[valid].reset_index(drop=True), [m for m, v in zip(mols, valid) if v], int(n_invalid)


def step3_canonicalize(mols):
    """Step 3. Canonicalize each rdkit Mol with stereochemistry preserved.
    Stereochemistry matters for BACE (enantiomer-resolved IC50) and qm8/qm9
    (geometry-derived properties), so we keep isomericSmiles=True."""
    return [Chem.MolToSmiles(m, canonical=True, isomericSmiles=True) for m in mols]


def step4_dedupe(df, target_cols, task_type, canon_smiles, reg_tol):
    """Step 4. Group by canonical SMILES; merge each group into one row.

    For each target column independently:
        classification: keep value if all duplicates agree, NaN if any disagree.
        regression:     average if range <= reg_tol["abs"] OR
                        range / |mean| <= reg_tol["rel"]; else NaN.

    Returns:
        merged_df:        one row per unique canonical SMILES, columns = [smiles] + target_cols
        n_unique:         number of unique canonical SMILES
        n_dup_groups:     number of canonical SMILES that had >=2 source rows
        per_target_stats: dict[target] -> {"groups_same": k, "groups_conflict": k}
                          "same" = kept/averaged; "conflict" = set NaN.
                          Counted ONLY over groups with >=2 non-NaN values.
    """
    df = df.copy()
    df["__canon"] = canon_smiles

    # Group rows by canonical SMILES
    groups = df.groupby("__canon", sort=False)
    n_unique = groups.ngroups
    n_dup_groups = int((groups.size() >= 2).sum())

    per_target_stats = {t: {"groups_same": 0, "groups_conflict": 0} for t in target_cols}

    out_rows = []
    for canon, gdf in groups:
        merged = {"smiles": canon}
        for t in target_cols:
            vals = gdf[t].dropna().astype(float).values
            if len(vals) == 0:
                merged[t] = np.nan
                continue
            if len(vals) == 1:
                merged[t] = float(vals[0])
                continue
            # >=2 non-NaN values for this target in this canonical-SMILES group
            if task_type == "cls":
                # Same value across all duplicates? (binary 0/1, but use ==)
                if np.all(vals == vals[0]):
                    merged[t] = float(vals[0])
                    per_target_stats[t]["groups_same"] += 1
                else:
                    merged[t] = np.nan
                    per_target_stats[t]["groups_conflict"] += 1
            else:  # regression
                rng = float(vals.max() - vals.min())
                mean = float(vals.mean())
                abs_ok = rng <= reg_tol["abs"]
                # rel_ok: skip if mean ~0 to avoid division blowup; if mean=0 and range>abs_tol -> NaN
                rel_ok = (abs(mean) > 0) and (rng <= reg_tol["rel"] * abs(mean))
                if abs_ok or rel_ok:
                    merged[t] = mean
                    per_target_stats[t]["groups_same"] += 1
                else:
                    merged[t] = np.nan
                    per_target_stats[t]["groups_conflict"] += 1
        out_rows.append(merged)

    merged_df = pd.DataFrame(out_rows, columns=["smiles"] + target_cols)
    return merged_df, n_unique, n_dup_groups, per_target_stats


def step5_drop_all_nan(merged_df, target_cols):
    """Step 5. Drop rows where every target is NaN (no usable label left)."""
    valid_mask = merged_df[target_cols].notna().any(axis=1)
    n_dropped = int((~valid_mask).sum())
    return merged_df.loc[valid_mask].reset_index(drop=True), n_dropped


def step7_write_outputs(name, cleaned_df, target_cols, meta):
    """Step 7. Save cleaned CSV, mask npy, meta json. Every downstream method
    script reads from this trio; no method ever reparses raw data."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"{name}.csv"
    mask_path = OUT_DIR / f"{name}_mask.npy"
    meta_path = OUT_DIR / f"{name}.meta.json"

    cleaned_df.to_csv(csv_path, index=False)
    mask = cleaned_df[target_cols].notna().to_numpy()  # (n_rows, n_targets) bool
    np.save(mask_path, mask)
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    return csv_path, mask_path, meta_path


# ---------------------------------------------------------------------------
# Per-dataset driver
# ---------------------------------------------------------------------------

def get_git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent.parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def clean_one(name, raw_filename, task_type, target_cols, reg_tol, dry_run=False):
    raw_path = RAW_DIR / raw_filename
    df_raw = pd.read_csv(raw_path)

    # SIDER targets are everything except 'smiles'
    if target_cols is None:
        target_cols = [c for c in df_raw.columns if c != "smiles"]

    smiles_col = "smiles"
    if smiles_col not in df_raw.columns:
        raise KeyError(f"[{name}] no 'smiles' column in {raw_path}; cols={list(df_raw.columns)[:5]}...")

    n_raw = len(df_raw)

    # --- Step 2: parse and drop invalid SMILES
    df, mols, n_invalid = step2_parse_and_filter_invalid(df_raw, smiles_col)

    # --- Step 3: canonicalize
    canon = step3_canonicalize(mols)

    # --- Step 4: dedupe by canonical SMILES
    merged, n_unique, n_dup_groups, per_target_stats = step4_dedupe(
        df, target_cols, task_type, canon, reg_tol
    )

    # --- Step 5: drop rows with all-NaN targets
    cleaned, n_dropped_all_nan = step5_drop_all_nan(merged, target_cols)

    # --- Aggregate stats for the user-facing summary
    total_dup_rows = int(n_raw - n_invalid - n_unique)  # rows that collapsed into existing canonical SMILES

    meta = {
        "dataset": name,
        "task_type": task_type,
        "raw_path": str(raw_path),
        "raw_rows": n_raw,
        "n_targets": len(target_cols),
        "target_columns": target_cols,
        "n_dropped_invalid_smiles": n_invalid,
        "n_unique_canonical_smiles": n_unique,
        "n_duplicate_groups": n_dup_groups,                 # canonical SMILES with >=2 source rows
        "n_duplicate_source_rows_collapsed": total_dup_rows,  # extra rows that merged
        "per_target_dedup": per_target_stats,                 # only counts groups with >=2 vals/target
        "regression_dup_tolerance": reg_tol if task_type == "reg" else None,
        "n_dropped_all_nan_after_dedup": n_dropped_all_nan,
        "n_after": len(cleaned),
        "rdkit_version": Chem.__version__ if hasattr(Chem, "__version__") else "n/a",
        "git_sha": get_git_sha(),
    }

    if not dry_run:
        # --- Step 7: write artifacts
        step7_write_outputs(name, cleaned, target_cols, meta)

    return meta


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _fmt_dup_summary(meta):
    """Build the 'X agreed / Y conflicting' string aggregated across targets."""
    n_same = sum(s["groups_same"] for s in meta["per_target_dedup"].values())
    n_conf = sum(s["groups_conflict"] for s in meta["per_target_dedup"].values())
    return n_same, n_conf


def print_summary(metas):
    print()
    print("=" * 110)
    print(f"{'dataset':<10} {'type':<4} {'n_raw':>7} {'invalid':>8} {'uniq':>7} "
          f"{'dup_grp':>8} {'agree':>7} {'conflict':>9} {'all_nan':>8} {'final':>7}")
    print("-" * 110)
    for m in metas:
        n_same, n_conf = _fmt_dup_summary(m)
        print(f"{m['dataset']:<10} {m['task_type']:<4} {m['raw_rows']:>7} "
              f"{m['n_dropped_invalid_smiles']:>8} {m['n_unique_canonical_smiles']:>7} "
              f"{m['n_duplicate_groups']:>8} {n_same:>7} {n_conf:>9} "
              f"{m['n_dropped_all_nan_after_dedup']:>8} {m['n_after']:>7}")
    print("=" * 110)
    print("Legend:")
    print("  invalid   = rows dropped because MolFromSmiles failed or 0 heavy atoms")
    print("  uniq      = unique canonical SMILES (after invalid filter)")
    print("  dup_grp   = canonical SMILES that came from >=2 source rows")
    print("  agree     = (target,group) pairs where >=2 source values existed and were merged")
    print("              (cls: same value;  reg: range<=abs_tol OR range/|mean|<=rel_tol)")
    print("  conflict  = (target,group) pairs with disagreement -> set NaN for that target")
    print("  all_nan   = rows dropped after dedup because every target ended up NaN")
    print("  final     = rows in cleaned/{dataset}.csv")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default="all", help="comma-separated, e.g. freesolv,bace")
    p.add_argument("--dry-run", action="store_true", help="don't write any files; just print stats")
    cli = p.parse_args()

    names = list(DATASETS) if cli.tasks == "all" else [t.strip() for t in cli.tasks.split(",")]
    metas = []
    for name in names:
        if name not in DATASETS:
            print(f"[{name}] unknown dataset; skipping")
            continue
        raw_filename, task_type, target_cols, reg_tol = DATASETS[name]
        print(f"[{name}] cleaning ({task_type}) from {raw_filename} ...")
        meta = clean_one(name, raw_filename, task_type, target_cols, reg_tol, dry_run=cli.dry_run)
        metas.append(meta)
        n_same, n_conf = _fmt_dup_summary(meta)
        print(f"  raw={meta['raw_rows']}  invalid={meta['n_dropped_invalid_smiles']}  "
              f"unique={meta['n_unique_canonical_smiles']}  dup_groups={meta['n_duplicate_groups']}  "
              f"agree={n_same}  conflict={n_conf}  "
              f"all_nan_dropped={meta['n_dropped_all_nan_after_dedup']}  final={meta['n_after']}")

    print_summary(metas)
    print()
    print(f"Outputs: {OUT_DIR}/{{<dataset>.csv, <dataset>_mask.npy, <dataset>.meta.json}}")


if __name__ == "__main__":
    main()
