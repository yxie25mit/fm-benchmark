"""
Dataset diversity + train->test novelty, as ONE shareable JSON of aggregate scalars — safe to
send back (no molecules/SMILES leave). Explains the "FMs win more in low-data" effect: small
pharma datasets are often clustered (few scaffolds), and time-split test sets can contain new
chemistry far from train — both favor pretrained foundation models over train-from-scratch.

Per dataset (full set):
  scaffold_diversity   : #unique Bemis-Murcko scaffolds, scaffolds/molecule, frac in largest /
                         top-5 / top-10 scaffolds, singleton-scaffold fraction
  fingerprint_diversity: median/mean pairwise Morgan-Tanimoto (sampled), internal diversity (1-mean)
  clusters             : #effective clusters at a Tanimoto threshold (0.8), clusters/molecule

Per fold (the key model-comparison signal):
  train_test_novelty   : distribution of s(x)=max_{z in train} Tanimoto(x,z) over test molecules x
                         (median/mean/p10/p25/p75, and frac of test with s(x)<0.4 / <0.5)
  scaffold_novelty     : fraction of test scaffolds not present in train

  python dataset_diversity.py --name mydata --out mydata_diversity.json           # time split
  python dataset_diversity.py --datasets mydata --protocols v1_preshuffle v2_astartes --out d.json
"""
import argparse
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
PIPELINE = Path(__file__).resolve().parent


def _fp(smiles, radius=2, nbits=2048):
    m = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    return AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits) if m is not None else None


def _scaffold(smiles):
    try:
        m = Chem.MolFromSmiles(smiles)
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False) if m else None
    except Exception:
        return None


def scaffold_diversity(scaffolds):
    valid = [s for s in scaffolds if s]
    n = len(valid)
    if n == 0:
        return None
    counts = Counter(valid)
    sizes = sorted(counts.values(), reverse=True)
    return {"n_unique_scaffolds": len(counts),
            "scaffolds_per_molecule": round(len(counts) / n, 4),
            "frac_in_largest_scaffold": round(sizes[0] / n, 4),
            "frac_in_top5_scaffolds": round(sum(sizes[:5]) / n, 4),
            "frac_in_top10_scaffolds": round(sum(sizes[:10]) / n, 4),
            "frac_singleton_scaffolds": round(sum(1 for v in counts.values() if v == 1) / len(counts), 4)}


def fingerprint_diversity(fps, sample=1500, seed=0):
    fps = [f for f in fps if f is not None]
    if len(fps) < 2:
        return None
    rng = np.random.default_rng(seed)
    if len(fps) > sample:
        fps = [fps[i] for i in rng.choice(len(fps), sample, replace=False)]
    sims = []
    for i in range(len(fps) - 1):
        sims.extend(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:]))
    sims = np.array(sims)
    return {"median_pairwise_tanimoto": round(float(np.median(sims)), 4),
            "mean_pairwise_tanimoto": round(float(sims.mean()), 4),
            "internal_diversity": round(float(1 - sims.mean()), 4),
            "n_molecules_used": len(fps)}


def effective_clusters(fps, threshold=0.8, butina_cap=3000, leader_cap=8000, seed=0):
    """#clusters at a Tanimoto threshold. Butina (deterministic) for small N; leader/
    sphere-exclusion (sampled) for large N."""
    fps = [f for f in fps if f is not None]
    n = len(fps)
    if n == 0:
        return None
    if n <= butina_cap:
        from rdkit.ML.Cluster import Butina
        dists = []
        for i in range(1, n):
            dists.extend([1 - s for s in DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])])
        clusters = Butina.ClusterData(dists, n, 1 - threshold, isDistData=True)
        return {"n_clusters": len(clusters), "clusters_per_molecule": round(len(clusters) / n, 4),
                "tanimoto_threshold": threshold, "method": "butina", "n_used": n}
    rng = np.random.default_rng(seed)
    used = [fps[i] for i in rng.choice(n, min(n, leader_cap), replace=False)]
    leaders = []
    for f in used:
        if not leaders or max(DataStructs.BulkTanimotoSimilarity(f, leaders)) < threshold:
            leaders.append(f)
    return {"n_clusters": len(leaders), "clusters_per_molecule": round(len(leaders) / len(used), 4),
            "tanimoto_threshold": threshold, "method": "leader_sampled", "n_used": len(used)}


def train_test_novelty(train_fps, test_fps, train_cap=6000, seed=0):
    tr = [f for f in train_fps if f is not None]
    rng = np.random.default_rng(seed)
    if len(tr) > train_cap:   # cap the train reference for speed; NN-max is stable under sampling
        tr = [tr[i] for i in rng.choice(len(tr), train_cap, replace=False)]
    s = np.array([max(DataStructs.BulkTanimotoSimilarity(f, tr)) for f in test_fps if f is not None])
    if len(s) == 0:
        return None
    return {"n_test": int(len(s)),
            "nearest_train_tanimoto": {"median": round(float(np.median(s)), 4), "mean": round(float(s.mean()), 4),
                                       "p10": round(float(np.percentile(s, 10)), 4),
                                       "p25": round(float(np.percentile(s, 25)), 4),
                                       "p75": round(float(np.percentile(s, 75)), 4),
                                       "min": round(float(s.min()), 4), "max": round(float(s.max()), 4)},
            "frac_test_below_0.4": round(float((s < 0.4).mean()), 4),
            "frac_test_below_0.5": round(float((s < 0.5).mean()), 4)}


def scaffold_novelty(train_scaf, test_scaf):
    tr = {s for s in train_scaf if s}
    te = [s for s in test_scaf if s]
    if not te:
        return None
    return {"frac_test_scaffolds_unseen_in_train": round(sum(1 for s in te if s not in tr) / len(te), 4)}


def _split_dir(ds, protocol, seed):
    name = f"custom_seed{seed}" if protocol == "custom" else (
        "v1_det_seed0" if protocol == "v1_det" else f"{protocol}_seed{seed}")
    return PIPELINE / "splits" / ds / name


def _fold_seeds(ds, protocol):
    base = PIPELINE / "splits" / ds
    if protocol == "custom":
        return sorted(int(p.name.replace("custom_seed", "")) for p in base.glob("custom_seed*"))
    if protocol == "v1_det":
        return [0]
    return [s for s in (0, 1, 2) if (base / f"{protocol}_seed{s}").exists()]   # scaffold eval folds


def analyze(ds, protocol, cluster_threshold):
    csv = PIPELINE / "cleaned" / f"{ds}.csv"
    if not csv.exists():
        return None
    smiles = pd.read_csv(csv)["smiles"].tolist()
    fps = [_fp(s) for s in smiles]
    scaffolds = [_scaffold(s) for s in smiles]
    out = {"n_molecules": len(smiles),
           "scaffold_diversity": scaffold_diversity(scaffolds),
           "fingerprint_diversity": fingerprint_diversity(fps),
           "clusters": effective_clusters(fps, threshold=cluster_threshold), "folds": []}
    for s in _fold_seeds(ds, protocol):
        d = _split_dir(ds, protocol, s)
        tr_i, te_i = d / "train_idx.npy", d / "test_idx.npy"
        if not (tr_i.exists() and te_i.exists()):
            continue
        tri, tei = np.load(tr_i), np.load(te_i)
        out["folds"].append({
            "seed": int(s), "n_train": int(len(tri)), "n_test": int(len(tei)),
            "train_test_novelty": train_test_novelty([fps[i] for i in tri], [fps[i] for i in tei]),
            "scaffold_novelty": scaffold_novelty([scaffolds[i] for i in tri], [scaffolds[i] for i in tei])})
    return out


def _git():
    try:
        return subprocess.run(["git", "-C", str(PIPELINE), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or None
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser(description="Dataset diversity + train->test novelty (shareable scalars).")
    p.add_argument("--name", help="time-split: analyzes <name>_chrono + <name>_sliding @ custom")
    p.add_argument("--datasets", nargs="+")
    p.add_argument("--protocols", nargs="+", default=["custom"])
    p.add_argument("--cluster-threshold", type=float, default=0.8)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    if args.name:
        cells = [(f"{args.name}_chrono", "custom"), (f"{args.name}_sliding", "custom")]
    elif args.datasets:
        cells = [(ds, proto) for ds in args.datasets for proto in args.protocols]
    else:
        p.error("provide --name or --datasets [+ --protocols]")

    bundle = {"provenance": {"git_commit": _git(),
                             "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                             "cluster_threshold": args.cluster_threshold},
              "note": "aggregate scalars only; no molecules/SMILES included", "datasets": {}}
    for ds, proto in cells:
        a = analyze(ds, proto, args.cluster_threshold)
        if a:
            bundle[f"datasets"][f"{ds}@{proto}"] = a

    Path(args.out).write_text(json.dumps(bundle, indent=2))
    print(f"\nwrote {args.out}\n")
    print(f"{'dataset@proto':<24}{'N':>6}{'#scaf':>7}{'top10%':>8}{'medTani':>9}{'#clust':>8}"
          f"{'fold':>6}{'med NN-train':>13}{'%test<0.4':>11}")
    print("-" * 92)
    for key, a in bundle["datasets"].items():
        sd, fd, cl = a["scaffold_diversity"] or {}, a["fingerprint_diversity"] or {}, a["clusters"] or {}
        base = (f"{key:<24}{a['n_molecules']:>6}{sd.get('n_unique_scaffolds',''):>7}"
                f"{sd.get('frac_in_top10_scaffolds',''):>8}{fd.get('median_pairwise_tanimoto',''):>9}"
                f"{cl.get('n_clusters',''):>8}")
        if not a["folds"]:
            print(base)
        for f in a["folds"]:
            nn = (f.get("train_test_novelty") or {}).get("nearest_train_tanimoto", {})
            print(f"{base}{f['seed']:>6}{nn.get('median',''):>13}"
                  f"{(f.get('train_test_novelty') or {}).get('frac_test_below_0.4',''):>11}")
            base = " " * 46


if __name__ == "__main__":
    main()
