"""
Method registry: env, worker path, default HPs, HP grid.

HP grids are the EXACT Cartesian products previously decided per method
(no random sampling) — copied from hyperopt_v2_r7/run_*_hp.py.

Each entry consumed by runners/run_phase.py to launch subprocess training.
"""
import itertools
import os
import sys
from pathlib import Path

PIPELINE = Path(__file__).resolve().parents[1]   # clean_pipeline_v1/ (relocatable)
METHODS_DIR = PIPELINE / "methods"
ENVS = os.environ.get("PIPELINE_CONDA_ENVS", str(Path(sys.executable).resolve().parents[2]))


METHODS = {
    "molclr": {
        "env": "molclr",
        "python": f"{ENVS}/molclr/bin/python",
        "worker": str(METHODS_DIR / "molclr" / "train_one.py"),
        # drop_ratio=0.3 + init_base_lr=5e-4 matches MolCLR's paper-actual
        # `.runs_parallel/*/config_finetune.yaml`; verified on freesolv smoke
        # → ensemble RMSE 2.71 (paper-tuned target 2.62).
        "default_hp": {
            "init_lr": 0.0005, "init_base_lr": 0.0005, "weight_decay": "1e-6",
            "batch_size": 32, "drop_ratio": 0.3,
            "num_layer": 5, "emb_dim": 300, "feat_dim": 512,
            "pred_n_layer": 2, "pred_act": "softplus", "pool": "mean",
        },
        # Prior grid (hyperopt_v2_r7/run_molclr_grid.py): 4 * 4 * 3 * 2 = 96.
        "hp_grid": {
            "init_base_lr": [5e-5, 1e-4, 2e-4, 5e-4],
            "drop_ratio":   [0.0, 0.1, 0.3, 0.5],
            "batch_size":   [32, 128, 256],
            "init_lr":      [5e-4, 1e-3],
        },
    },
    "chemprop2": {
        "env": "chemprop2",
        "python": f"{ENVS}/chemprop2/bin/python",
        "worker": str(METHODS_DIR / "chemprop2" / "train_one.py"),
        "default_hp": {
            "depth": 3, "message_hidden_dim": 300,
            "ffn_num_layers": 2, "ffn_hidden_dim": 300,
            "dropout": 0.0, "batch_size": 64, "max_lr": 1e-3,
            "aggregation": "norm",
        },
        # Prior chemprop2 used Bayesian search over 8 dims.
        # We use a fixed Cartesian (4 * 3 * 3 * 3 = 108) over the 4 highest-impact dims.
        "hp_grid": {
            "depth":              [3, 4, 5, 6],
            "message_hidden_dim": [300, 600, 900],
            "ffn_num_layers":     [1, 2, 3],
            "dropout":            [0.0, 0.1, 0.2],
        },
    },
    "chemeleon": {
        "env": "chemprop2",
        "python": f"{ENVS}/chemprop2/bin/python",
        "worker": str(METHODS_DIR / "chemeleon" / "train_one.py"),
        "default_hp": {
            "max_lr": 1e-4, "ffn_num_layers": 2, "ffn_hidden_dim": 600,
            "dropout": 0.1, "batch_size": 64,
        },
        # Prior grid (hyperopt_v2_r7/run_chemmeleon_hp.py): 4 * 3 * 3 * 3 = 108.
        "hp_grid": {
            "max_lr":         [5e-5, 1e-4, 5e-4, 1e-3],
            "ffn_num_layers": [1, 2, 3],
            "ffn_hidden_dim": [300, 600, 900],
            "dropout":        [0.0, 0.1, 0.2],
        },
    },
    "chemprop2_nofp": {
        "env": "chemprop2",
        "python": f"{ENVS}/chemprop2/bin/python",
        "worker": str(METHODS_DIR / "chemprop2_nofp" / "train_one.py"),
        "default_hp": {
            "depth": 3, "message_hidden_dim": 300,
            "ffn_num_layers": 2, "ffn_hidden_dim": 300,
            "dropout": 0.0, "batch_size": 64, "max_lr": 1e-3,
            "aggregation": "norm",
        },
        "hp_grid": {
            "depth":              [3, 4, 5, 6],
            "message_hidden_dim": [300, 600, 900],
            "ffn_num_layers":     [1, 2, 3],
            "dropout":            [0.0, 0.1, 0.2],
        },
    },
    "chemeleon_nofp": {
        "env": "chemprop2",
        "python": f"{ENVS}/chemprop2/bin/python",
        "worker": str(METHODS_DIR / "chemeleon_nofp" / "train_one.py"),
        "default_hp": {
            "max_lr": 1e-4, "ffn_num_layers": 2, "ffn_hidden_dim": 600,
            "dropout": 0.1, "batch_size": 64,
        },
        "hp_grid": {
            "max_lr":         [5e-5, 1e-4, 5e-4, 1e-3],
            "ffn_num_layers": [1, 2, 3],
            "ffn_hidden_dim": [300, 600, 900],
            "dropout":        [0.0, 0.1, 0.2],
        },
    },
    "molfcl": {
        "env": "molfcl",
        "python": f"{ENVS}/molfcl/bin/python",
        "worker": str(METHODS_DIR / "molfcl" / "train_one.py"),
        # Paper-faithful: MolFCL SI fixes predictor at 2 layers / hidden 300 and tunes
        # the chemistry-guided knobs (FG-prompt self-attention layers `num_attention`,
        # L2 penalty gamma `l2_norm`, fine-tune lr). Defaults = the SI's reported optima
        # (num_attention=7, l2_norm/gamma=0.1, lr=1e-3). Predictor dims are fixed in the
        # worker, not tuned.
        "default_hp": {
            "max_lr": 1e-3, "dropout": 0.1,
            "num_attention": 7, "l2_norm": 0.1,
        },
        # Grid spans the union of the paper's MoleculeNet and TDC fine-tune ranges:
        # lr covers TDC {3e-5,3e-4} + MolNet {1e-4,1e-3} (SI best 1e-3); gamma {0.1..100}
        # and dropout {0,0.1} are the full SI ranges; self-attn {2,4,7} covers the SI
        # best (7). Batch size and patience are suite-conditional in the worker.
        "hp_grid": {
            "max_lr":        [3e-5, 1e-4, 3e-4, 1e-3],
            "num_attention": [2, 4, 7],
            "l2_norm":       [0.1, 1.0, 10.0, 50.0, 100.0],
            "dropout":       [0.0, 0.1],
        },  # 4 * 3 * 5 * 2 = 120 configs
    },
    "motil": {
        "env": "molfcl",            # shares chemprop fork env
        "python": f"{ENVS}/molfcl/bin/python",
        "worker": str(METHODS_DIR / "motil" / "train_one.py"),
        "default_hp": {
            "init_lr": 1e-4, "final_lr": 1e-6,
            "batch_size": 64, "wd": 1e-5,
            "dropout": 0.1, "ffn_num_layers": 2, "ffn_hidden_size": 300,
        },
        # Paper-faithful HP grid (MotiL Supplementary Table 7): init_lr, final_lr,
        # batch_size, wd. `epochs` is intentionally NOT tuned — hp_final fixes epochs
        # at 100 for ALL methods (train_one falls back to the phase --epochs), so MotiL
        # stays fair; the freed budget tunes wd (paper, was fixed 1e-5) + broader batch.
        # max_lr is not a tuned dim — train_one sets max_lr=init_lr.
        "hp_grid": {
            "init_lr":    [1e-5, 1e-4, 1e-3],
            "final_lr":   [1e-7, 1e-6, 1e-5],
            "batch_size": [16, 64, 128, 256],
            "wd":         [1e-5, 5e-5, 1e-4],
        },  # 3 * 3 * 4 * 3 = 108, no epochs (fixed 100/30 like all methods)
    },
    "molformer": {
        "env": "molformer",
        "python": f"{ENVS}/molformer/bin/python",
        "worker": str(METHODS_DIR / "molformer" / "train_one.py"),
        # Head is paper-fixed at 2 FC layers / hidden 768 / GELU by the upstream finetune
        # scripts (the `dims` arg is ignored), so `head_depth` was a phantom HP — it
        # changed nothing and tripled the grid into duplicates. Dropped.
        "default_hp": {
            "max_lr": 3e-5, "dropout": 0.1, "batch_size": 128,
        },
        "hp_grid": {
            # TDC run: dropped max_lr=1e-5 (never selected best across 29 MoleculeNet
            # cells; winners were 3e-4 >> 1e-4, 3e-5 kept as the paper's stated optimum).
            # batch 32 and dropout are KEPT — on small datasets our MoleculeNet HP search
            # picks batch 32 (62%) and varies dropout, contradicting the paper's QM9 optima.
            # Epochs stay at 30: MoleculeNet best-val epoch is median 27 (only 24% by ep 15).
            "max_lr":     [3e-5, 1e-4, 3e-4],
            "dropout":    [0.0, 0.1, 0.2],
            "batch_size": [32, 64, 128],
        },  # 3 * 3 * 3 = 27 configs (TDC)
    },
}


# Datasets in size order (smallest first) for orchestration.
DATASETS_SIZE_ORDERED = [
    "freesolv",  # 640
    "esol",      # 1100
    "sider",     # 1400
    "clintox",   # 1500
    "bace",      # 1500
    "bbbp",      # 2000
    "lipo",      # 4200
    "qm7",       # 7000
    "tox21",     # 7800
    "qm8",       # 22000
    "hiv",       # 41000
    "qm9",       # 134000
]

PROTOCOLS = ["v1_det", "v1_preshuffle", "v2_astartes"]

EVAL_SEEDS = {
    "v1_det":         [0],
    "v1_preshuffle":  [0, 1, 2],
    "v2_astartes":    [0, 1, 2],
    # TDC benchmark-group: 5 train/val splits over a FIXED test set (tdc_seed1..5).
    "tdc":            [1, 2, 3, 4, 5],
    "custom":         [0],   # user folds; run_phase._eval_seeds overrides per fold
}
HP_SEED = {
    "v1_det":         0,    # HP on own val (no separate seed3 for v1_det)
    "v1_preshuffle":  3,
    "v2_astartes":    3,
    # TDC selects HP by the MEAN validation over all 5 train/val splits (a LIST here);
    # build_hp_search_jobs runs every config on each seed, pick_best_hp averages the vals.
    "tdc":            [1, 2, 3, 4, 5],
    "custom":         0,     # fallback only; run_phase runs custom HP on ALL folds (TDC-style mean-val)
}

ENSEMBLE_SIZE = 5


def all_hp_configs(method: str):
    """Full Cartesian product of the method's grid (no random sampling)."""
    cfg = METHODS[method]
    keys = list(cfg["hp_grid"].keys())
    values = [cfg["hp_grid"][k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def hp_config_id(hp: dict) -> str:
    """Stable short id for an HP config (used in dir names)."""
    import hashlib
    blob = "|".join(f"{k}={hp[k]}" for k in sorted(hp.keys()))
    return hashlib.md5(blob.encode()).hexdigest()[:10]


def expand_molformer_head(hp: dict) -> dict:
    """molformer worker takes ffn_hidden_dims (list); the grid uses head_depth (int).
    Translate at job-construction time."""
    hp = dict(hp)
    if "head_depth" in hp:
        d = hp.pop("head_depth")
        hp["ffn_hidden_dims"] = [768] * d
    return hp
