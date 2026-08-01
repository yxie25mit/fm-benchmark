"""
Nested learning-curve training subsets, implemented as pseudo-protocol split dirs.

For each (dataset, base_protocol, outer_seed) we take the existing outer split's
train_idx and build nested prefixes T5% ⊂ T10% ⊂ T20% ⊂ T40% ⊂ T70% (⊂ T100%=train_idx),
sharing the split's val_idx/test_idx unchanged. Written as:

  splits/<ds>/<base>__f<FFF>_seed<S>/{train_idx.npy, val_idx.npy, test_idx.npy}

so the existing trainers load them via --protocol <base>__f<FFF> with NO code change
(train_one builds the path as f"{protocol}_seed{seed}"). f100 == the existing split.

Classification: class-stratified prefixes (stable class balance, nested).
Regression:     decile-binned prefixes (target range coverage, nested).
"""
import json
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path

PIPELINE = Path(__file__).resolve().parent.parent
SPLITS = PIPELINE / "splits"
CLEANED = PIPELINE / "cleaned"

FRACTIONS = [0.05, 0.10, 0.20, 0.40, 0.70]   # f100 reuses the existing split
DATASETS = ["bace", "esol"]
BASE_PROTOCOLS = ["v1_preshuffle", "v2_astartes"]
SEEDS = [0, 1, 2]
MIN_PER_CLASS = 4
N_REG_BINS = 10


def frac_tag(frac):
    return f"f{int(round(frac * 100)):03d}"


def stable_seed(dataset, base, outer_seed):
    key = f"{dataset}|{base}|{outer_seed}".encode()
    return int(hashlib.md5(key).hexdigest()[:8], 16)


def nested_cls(train_idx, y_train, rng):
    pos = train_idx[y_train == 1]
    neg = train_idx[y_train == 0]
    pos = rng.permutation(pos)
    neg = rng.permutation(neg)
    subsets = {}
    for frac in FRACTIONS:
        n_pos = max(MIN_PER_CLASS, int(np.ceil(frac * len(pos))))
        n_neg = max(MIN_PER_CLASS, int(np.ceil(frac * len(neg))))
        n_pos = min(n_pos, len(pos))
        n_neg = min(n_neg, len(neg))
        sub = np.sort(np.concatenate([pos[:n_pos], neg[:n_neg]]))
        subsets[frac] = (sub, {"n_train": int(len(sub)), "n_pos": int(n_pos), "n_neg": int(n_neg)})
    return subsets


def nested_reg(train_idx, y_train, rng):
    edges = np.quantile(y_train, np.linspace(0, 1, N_REG_BINS + 1)[1:-1])
    bin_of = np.digitize(y_train, edges)
    order = {}
    for b in range(N_REG_BINS):
        members = train_idx[bin_of == b]
        order[b] = rng.permutation(members)
    subsets = {}
    for frac in FRACTIONS:
        parts = []
        for b in range(N_REG_BINS):
            k = int(np.ceil(frac * len(order[b])))
            parts.append(order[b][:k])
        sub = np.sort(np.concatenate(parts)) if parts else np.array([], dtype=np.int64)
        subsets[frac] = (sub, {"n_train": int(len(sub)), "n_bins": N_REG_BINS})
    return subsets


def check_nested(subsets):
    ordered = [subsets[f][0] for f in FRACTIONS]
    for smaller, larger in zip(ordered, ordered[1:]):
        assert set(smaller.tolist()).issubset(set(larger.tolist())), "nesting violated"


def main():
    for dataset in DATASETS:
        meta = json.loads((CLEANED / f"{dataset}.meta.json").read_text())
        task_type = meta["task_type"]
        target = meta["target_columns"][0]
        df = pd.read_csv(CLEANED / f"{dataset}.csv")
        y_full = df[target].to_numpy()
        for base in BASE_PROTOCOLS:
            for seed in SEEDS:
                src = SPLITS / dataset / f"{base}_seed{seed}"
                if not src.exists():
                    print(f"  SKIP missing outer split: {src}")
                    continue
                train_idx = np.load(src / "train_idx.npy")
                val_idx = np.load(src / "val_idx.npy")
                test_idx = np.load(src / "test_idx.npy")
                y_train = y_full[train_idx]
                rng = np.random.default_rng(stable_seed(dataset, base, seed))
                if task_type == "cls":
                    subsets = nested_cls(train_idx, y_train, rng)
                else:
                    subsets = nested_reg(train_idx, y_train, rng)
                check_nested(subsets)

                meta_out = {"dataset": dataset, "base_protocol": base, "outer_seed": seed,
                            "task_type": task_type, "n_train_full": int(len(train_idx)),
                            "fractions": {}}
                for frac in FRACTIONS:
                    sub, info = subsets[frac]
                    dst = SPLITS / dataset / f"{base}__{frac_tag(frac)}_seed{seed}"
                    dst.mkdir(parents=True, exist_ok=True)
                    np.save(dst / "train_idx.npy", sub)
                    np.save(dst / "val_idx.npy", val_idx)
                    np.save(dst / "test_idx.npy", test_idx)
                    meta_out["fractions"][frac_tag(frac)] = {"frac": frac, **info}
                meta_out["fractions"]["f100"] = {"frac": 1.0, "n_train": int(len(train_idx)),
                                                 "reuse_split": f"{base}_seed{seed}"}
                (SPLITS / dataset / f"_lc_meta_{base}_seed{seed}.json").write_text(
                    json.dumps(meta_out, indent=2))
                sizes = " ".join(f"{frac_tag(f)}={subsets[f][1]['n_train']}" for f in FRACTIONS)
                print(f"[{dataset}/{base}/seed{seed}] full={len(train_idx)} {sizes} f100={len(train_idx)}")


# ---------------------------------------------------------------------------
# Absolute-n mode for larger / contrast datasets (5% would not be "low data").
# HIV: single-task, imbalanced -> class-stratified. tox21/sider: multi-task -> random nested.
ABS_N = {
    "hiv":   [100, 250, 500, 1000, 2500, 5000, 10000, 20000],
    "tox21": [100, 250, 500, 1000, 2500, 5000],
    "sider": [100, 250, 500, 1000],
}


def n_tag(size):
    return f"n{size:05d}"


def nested_cls_single_abs(train_idx, y_train, sizes, rng):
    pos = rng.permutation(train_idx[y_train == 1])
    neg = rng.permutation(train_idx[y_train == 0])
    pos_frac = len(pos) / len(train_idx)
    out = {}
    for size in sizes:
        if size >= len(train_idx):
            continue
        n_pos = min(len(pos), max(MIN_PER_CLASS, int(round(size * pos_frac))))
        n_neg = min(len(neg), max(MIN_PER_CLASS, size - n_pos))
        sub = np.sort(np.concatenate([pos[:n_pos], neg[:n_neg]]))
        out[size] = (sub, {"n_train": int(len(sub)), "n_pos": int(n_pos), "n_neg": int(n_neg)})
    return out


def nested_random_abs(train_idx, sizes, rng):
    perm = rng.permutation(train_idx)
    out = {}
    for size in sizes:
        if size >= len(train_idx):
            continue
        sub = np.sort(perm[:size])
        out[size] = (sub, {"n_train": int(len(sub))})
    return out


def process_abs_datasets():
    for dataset, sizes in ABS_N.items():
        meta = json.loads((CLEANED / f"{dataset}.meta.json").read_text())
        task_type = meta["task_type"]
        n_targets = meta.get("n_targets", 1)
        target = meta["target_columns"][0]
        df = pd.read_csv(CLEANED / f"{dataset}.csv")
        for base in BASE_PROTOCOLS:
            for seed in SEEDS:
                src = SPLITS / dataset / f"{base}_seed{seed}"
                if not src.exists():
                    print(f"  SKIP missing outer split: {src}")
                    continue
                train_idx = np.load(src / "train_idx.npy")
                val_idx = np.load(src / "val_idx.npy")
                test_idx = np.load(src / "test_idx.npy")
                rng = np.random.default_rng(stable_seed(dataset, base, seed))
                if task_type == "cls" and n_targets == 1:
                    subsets = nested_cls_single_abs(train_idx, df[target].to_numpy()[train_idx], sizes, rng)
                else:
                    subsets = nested_random_abs(train_idx, sizes, rng)  # multi-task cls / reg
                ordered = [subsets[s][0] for s in sorted(subsets)]
                for smaller, larger in zip(ordered, ordered[1:]):
                    assert set(smaller.tolist()).issubset(set(larger.tolist())), "abs nesting violated"
                meta_out = {"dataset": dataset, "base_protocol": base, "outer_seed": seed,
                            "mode": "absolute_n", "task_type": task_type, "n_targets": n_targets,
                            "n_train_full": int(len(train_idx)), "points": {}}
                for size in sorted(subsets):
                    sub, info = subsets[size]
                    dst = SPLITS / dataset / f"{base}__{n_tag(size)}_seed{seed}"
                    dst.mkdir(parents=True, exist_ok=True)
                    np.save(dst / "train_idx.npy", sub)
                    np.save(dst / "val_idx.npy", val_idx)
                    np.save(dst / "test_idx.npy", test_idx)
                    meta_out["points"][n_tag(size)] = {"target_n": size, **info}
                meta_out["points"]["n_full"] = {"target_n": int(len(train_idx)),
                                                "n_train": int(len(train_idx)),
                                                "reuse_split": f"{base}_seed{seed}"}
                (SPLITS / dataset / f"_lc_absn_meta_{base}_seed{seed}.json").write_text(
                    json.dumps(meta_out, indent=2))
                got = " ".join(f"{n_tag(s)}={subsets[s][1]['n_train']}" for s in sorted(subsets))
                print(f"[{dataset}/{base}/seed{seed}] full={len(train_idx)} {got}")


if __name__ == "__main__":
    main()
    process_abs_datasets()
