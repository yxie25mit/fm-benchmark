"""
Scaffold splitters that reproduce our two published split styles from a SMILES list.
Same algorithms as scripts/make_splits.py (chemprop v1.6.1 + chemprop v2/astartes),
but path-free and importable so prepare_user_data.py can split a raw CSV into the
IDENTICAL protocols we use — guaranteeing comparability with published numbers.

  v1_preshuffle : chemprop v1.6.1 scaffold_split(balanced=False), rows pre-shuffled by seed.
  v2_astartes   : chemprop v2 scaffold split via astartes.train_val_test_split_molecules.

Both return (train_idx, val_idx, test_idx) as np.int64 arrays into the input order.
"""
from collections import defaultdict

import numpy as np

# chemprop v1.6.1's generate_scaffold is exactly this Murcko call (verified line-for-line
# against chemprop_v1_backup/data/scaffold.py) — so the rdkit path is authoritative here.
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

DEFAULT_SIZES = (0.8, 0.1, 0.1)
DEFAULT_SEEDS = [0, 1, 2, 3]   # 0/1/2 = eval seeds; 3 = HP-only seed (decouples HP val from eval test)


def _generate_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else smiles
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)


def _v1_balanced_false(smiles, sizes=DEFAULT_SIZES):
    """Verbatim chemprop v1.6.1 scaffold_split(balanced=False): bucket by scaffold,
    sort largest->smallest, greedily fill train -> val -> test."""
    n_total = len(smiles)
    n_train, n_val = sizes[0] * n_total, sizes[1] * n_total
    scaffold_to_indices = defaultdict(set)
    for i, s in enumerate(smiles):
        scaffold_to_indices[_generate_scaffold(s)].add(i)
    index_sets = sorted([list(v) for v in scaffold_to_indices.values()],
                        key=lambda x: len(x), reverse=True)
    train, val, test = [], [], []
    for s in index_sets:
        if len(train) + len(s) <= n_train:
            train += s
        elif len(val) + len(s) <= n_val:
            val += s
        else:
            test += s
    return (np.array(train, dtype=np.int64), np.array(val, dtype=np.int64),
            np.array(test, dtype=np.int64))


def v1_preshuffle_split(smiles, seed, sizes=DEFAULT_SIZES):
    """chemprop v1.6.1 scaffold_split with input rows pre-shuffled by `seed`
    (only same-size scaffold buckets reorder, since sort is stable)."""
    n = len(smiles)
    perm = np.random.default_rng(seed).permutation(n)
    tr, va, te = _v1_balanced_false([smiles[i] for i in perm], sizes=sizes)
    return perm[tr], perm[va], perm[te]


def v2_astartes_split(smiles, seed, sizes=DEFAULT_SIZES):
    """chemprop v2 scaffold split = astartes scaffold sampler (random_state=seed)."""
    from astartes.molecules import train_val_test_split_molecules
    out = train_val_test_split_molecules(
        np.array(smiles, dtype=object),
        train_size=sizes[0], val_size=sizes[1], test_size=sizes[2],
        sampler="scaffold", random_state=seed, return_indices=True,
    )
    return (np.array(out[-3], dtype=np.int64), np.array(out[-2], dtype=np.int64),
            np.array(out[-1], dtype=np.int64))


SPLITTERS = {"v1_preshuffle": v1_preshuffle_split, "v2_astartes": v2_astartes_split}
