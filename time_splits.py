"""
Chronological (time-based) splitters for `prepare_user_data.py --csv --time-split`.
Input rows are assumed time-sorted oldest->newest (pass --date-col to sort explicitly).

Two designs, mirroring the chemprop time-split protocol:

  chrono_split    : one 80/10/10 chronological split (earliest -> train, latest -> test).
                    The headline "prospective" number.

  sliding_windows : fixed-size sliding windows with CONSTANT train size (not expanding),
                    e.g. 12 chunks, train=8 / val=1 / test=1, sliding by 1 -> 3 folds that
                    test the newest chunks. Each fold is meant to be tuned on its OWN
                    validation only (never pooled across folds), which is leakage-free:
                    a fold's validation always precedes its own test, and no single fold's
                    reported number ever selects HP on the chunk it tests.

All functions take n (row count) and return (train_idx, val_idx, test_idx) as int64 arrays
into the time-sorted row order.
"""
import numpy as np


def chrono_split(n, fractions=(0.8, 0.1, 0.1)):
    idx = np.arange(n, dtype=np.int64)
    n_train = int(round(fractions[0] * n))
    n_val = int(round(fractions[1] * n))
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


def sliding_windows(n, n_chunks=12, train_chunks=8, n_folds=3):
    """Constant-train sliding windows. Fold i: train = chunks[i : i+train_chunks],
    val = chunk[i+train_chunks], test = chunk[i+train_chunks+1]. Tests slide to the newest
    chunks; train size stays fixed. Returns a list of (train, val, test)."""
    if n_chunks < train_chunks + 2:
        raise ValueError(f"need n_chunks >= train_chunks+2 (got {n_chunks} < {train_chunks}+2)")
    max_folds = n_chunks - (train_chunks + 1)
    if n_folds > max_folds:
        raise ValueError(f"{n_folds} folds need {n_folds + train_chunks + 1} chunks; "
                         f"n_chunks={n_chunks} only allows {max_folds} folds")
    chunks = np.array_split(np.arange(n, dtype=np.int64), n_chunks)
    folds = []
    for i in range(n_folds):
        train = np.concatenate(chunks[i:i + train_chunks]).astype(np.int64)
        val = chunks[i + train_chunks].astype(np.int64)
        test = chunks[i + train_chunks + 1].astype(np.int64)
        folds.append((train, val, test))
    return folds
