"""chemprop2 WITHOUT RDKit-2D descriptor features (ablation).
Sets NO_RDKIT2D=1 and re-execs the real chemprop2 worker with the same args."""
import os
import sys
from pathlib import Path

os.environ["NO_RDKIT2D"] = "1"
ORIG = str(Path(__file__).resolve().parents[1] / "chemprop2" / "train_one.py")
os.execv(sys.executable, [sys.executable, ORIG, *sys.argv[1:]])
