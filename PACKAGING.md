# Packaging for a different machine

This pipeline runs across **several** conda environments (the methods have conflicting
dependencies — see §1) and loads a few external fork repos + pretrained checkpoints. To run on a
new machine you need: (1) this repo, (2) the per-method conda envs, (3) the fork repos + weights,
(4) a few env vars pointing at (2)/(3). **All external paths are env-var-overridable** with the
current cluster paths as defaults, so nothing breaks in place.

## 1. Conda environments — you need SEVERAL, not one

A single env will **not** work: `molclr` and `molformer` pin torch 1.7.1 / CUDA 11.0, which is
incompatible with the modern chemprop-v2 env. (The orchestration runs under the lightweight
`orchestrator` env, but each method is launched with its own env's python — that's why one env only appears
to work for the chemprop-family methods.)

| env | used by | key deps |
|-----|---------|----------|
| `chemprop2` | chemprop2, chemprop2_nofp, chemeleon, chemeleon_nofp | chemprop v2 |
| `molclr` | molclr | torch 1.7.1+cu110, torch_geometric 1.6.3 |
| `molfcl` | molfcl, motil | chemprop v1 fork |
| `molformer` | molformer | torch 1.7.1, pytorch-lightning, apex |
| `orchestrator` | orchestration + prepare_user_data + collect_results (no GPU/torch) | rdkit, pandas, numpy, sklearn, scipy, matplotlib |

Export each on this machine, recreate on the new one:
```bash
mkdir -p envs
for e in chemprop2 molclr molfcl molformer; do conda env export -n $e > envs/$e.yml; done  # orchestrator.yml is hand-maintained
# on the new machine:
for e in orchestrator chemprop2 molclr molfcl molformer; do conda env create -f envs/$e.yml; done
export PIPELINE_CONDA_ENVS=/path/to/your/conda/envs   # if not ~/anaconda3/envs
```
(cu110 wheels for molclr/molformer may need `--no-deps` + the matching pip wheels; keep the
`nvidia-cusparse-cu11` pip package for molclr.)

## 2. Fork repos + checkpoints + foundation — YES, ship these

The method wrappers load code/weights from external repos. Copy them and set the matching env var
(defaults shown are our paths):

| what | env var | contents |
|------|---------|----------|
| MolFCL fork | `MOLFCL_DIR` | code + `ckpt/original_MoleculeModel.pkl` |
| MotiL fork | `MOTIL_DIR` | code + `dumped/pre-train/.../*.pkl` |
| MoLFormer | `MOLFORMER_DIR` | code + `Pretrained MoLFormer/checkpoints/*.ckpt` |
| CheMeleon foundation | `CHEMELEON_HOME` | `.chemprop/chemeleon_mp.pt` (34 MB) |
| MolCLR pretrained GIN | — (in-repo) | `methods/molclr/ckpt/pretrained_gin/` — already included |

These are large binaries — **vendor them next to the repo (or git-LFS), not in plain git history.**
`chemprop_v1_backup` is only needed to *regenerate* splits (`scripts/make_splits.py`); pharma using
the shipped `splits/` does not need it.

## 3. Env config the user sources before running
```bash
# config/env.sh  (example — edit paths for the new machine)
export PIPELINE_CONDA_ENVS=/path/to/conda/envs
export MOLFCL_DIR=/path/to/MolFCL
export MOTIL_DIR=/path/to/MotiL/MotiL_micromolecule
export MOLFORMER_DIR=/path/to/molformer
export CHEMELEON_HOME=/path/to/chemmeleon_home
# optional per-user checkpoints:
# export CHEMELEON_CKPT=...  MOLCLR_CKPT_DIR=...  MOLFCL_CKPT=...  MOTIL_CKPT=...  MOLFORMER_CKPT=...
```
The pipeline `PIPELINE` path itself is now derived from the code location (relocatable) — no edit
needed when you move the repo.

## 4. Assemble the clean repo
```bash
bash make_pharma_release.sh                 # -> ../pharma_release (code+data+docs; no results/junk)
# then drop in envs/*.yml and the vendored forks, and it's a self-contained handoff.
```

## Included vs excluded
- **Included:** `methods/ orchestrator/ runners/ scripts/ cleaned/ splits/ prepare_user_data.py
  collect_results.py README_PHARMA.md CUSTOM_PROTOCOL_PATCH.md .gitignore`.
- **Excluded:** `results/` (runtime output), `logs/`, `cache/`, `__pycache__/`, `jcim_*/`,
  `*.pdf/.tex/.docx/.html/.tar.gz`, and transient `user_*`/`e2e_*` datasets.
