#!/usr/bin/env bash
# Run on the SOURCE machine to export the 5 conda envs to YAML, so the new machine
# can recreate them.  (Conda envs are NOT file-copyable across machines — they must be
# rebuilt from these YAMLs.)  Output: envs/<name>.yml
set -e
mkdir -p envs
for e in chemprop2 molclr molfcl molformer foldeverything; do
  echo ">> exporting $e"
  conda env export -n "$e" > "envs/$e.yml"
done
echo ">> wrote envs/*.yml"
echo "   On the new machine:  for e in chemprop2 molclr molfcl molformer foldeverything; do conda env create -f envs/\$e.yml; done"
echo "   NOTE: the cu110 wheels for molclr/molformer (torch 1.7.1) may need manual pip install of the"
echo "   matching wheels (+ nvidia-cusparse-cu11 for molclr) — a plain 'conda env create' can miss them."
