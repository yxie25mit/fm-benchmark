"""Cached RDKit-2D-normalized descriptors, shared by the chemprop-CLI workers.

`--molecule-featurizers v1_rdkit_2d_normalized` recomputes a 200-dim descriptor row
per molecule at every process start (~35 ms/mol), once for `chemprop train` and again
for the `chemprop predict` val pass. The values depend only on the SMILES, so they are
computed once per dataset and reused.

The cache key contains the SHA-256 of the CSV, so an edited CSV can never hit a stale
entry — it simply misses and recomputes.
"""
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

FEATURIZER = "v1_rdkit_2d_normalized"
N_WORKERS = int(os.environ.get("MOLNET_DESC_WORKERS", "4"))
CACHE_DIR = Path(os.environ.get(
    "MOLNET_DESC_CACHE",
    str(Path(__file__).resolve().parents[1] / "cache" / "rdkit2d")))


def _content_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def descriptors_for_csv(csv_path, smiles_col="smiles"):
    """(N, 200) float64 descriptor matrix row-aligned to csv_path, cached on disk.

    Reproduces chemprop's own pipeline exactly: the CLI defaults
    (keep_h/add_h/ignore_stereo/reorder_atoms all False) feed make_mol, then the
    registered featurizer; make_datapoints vstacks those same per-molecule rows.
    """
    csv_path = Path(csv_path)
    cache_path = CACHE_DIR / f"{csv_path.stem}.{FEATURIZER}.{_content_hash(csv_path)}.npz"
    if cache_path.exists():
        try:
            return np.load(cache_path)["arr_0"]
        except Exception:
            pass

    from chemprop.cli.utils.parsing import make_mol
    from chemprop.featurizers import MoleculeFeaturizerRegistry
    from chemprop.utils.utils import create_and_call_object, parallel_execute

    featurizer_cls = MoleculeFeaturizerRegistry[FEATURIZER]
    smiles = pd.read_csv(csv_path)[smiles_col].tolist()
    mols = parallel_execute(
        make_mol, [(s, False, False, False, False) for s in smiles], n_workers=N_WORKERS)
    matrix = np.vstack(parallel_execute(
        create_and_call_object, [(featurizer_cls, (mol,)) for mol in mols],
        n_workers=N_WORKERS))

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(cache_path.name + f".{os.getpid()}.tmp.npz")
    np.savez(tmp_path, matrix)
    os.replace(tmp_path, cache_path)
    return matrix


def write_descriptor_subset(matrix, row_selector, out_path):
    """Save the rows of `matrix` selected by `row_selector` in the npz layout chemprop's
    --descriptors-path expects (a single 'arr_0' array)."""
    np.savez(out_path, matrix[row_selector])
    return out_path
