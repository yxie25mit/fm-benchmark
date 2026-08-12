"""
Top-level benchmark orchestrator.

Default order (`phase_first`):
  for phase in [default, hp_search, hp_final]:
      for protocol in [v1_det, v1_preshuffle, v2_astartes]:
          for method in selected_methods (fast→slow):
              run_phase.py --method M --protocols P --datasets ALL
              (run_phase pools all datasets × ensemble jobs for this method/protocol/phase)

Each (method, protocol, phase) cell-bundle is delegated to runners/run_phase.py
via subprocess. Already-completed cells are skipped (each phase has _summary.json).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PIPELINE = Path(__file__).resolve().parents[1]   # repo root (relocatable)
sys.path.insert(0, str(PIPELINE / "methods"))

from configs import METHODS, DATASETS_SIZE_ORDERED, PROTOCOLS  # noqa: E402

PYTHON = os.environ.get("PIPELINE_PYTHON", sys.executable)   # the env launching run_benchmark (foldeverything)
RUN_PHASE = PIPELINE / "runners" / "run_phase.py"


def cell_done(method, dataset, protocol, phase):
    """A phase cell is "done" when its _summary.json exists."""
    summary = PIPELINE / "results" / method / dataset / protocol / phase / "_summary.json"
    return summary.exists()


def run_method_phase(method, datasets, protocols, phase, gpus, jobs_per_gpu, epochs,
                     max_configs=None, hp_per_fold=False):
    """Run one (method, phase, protocols) bundle with cross-cell pooling.
    run_phase.py pools jobs from ALL listed (dataset × protocol) cells into a
    single ThreadPoolExecutor of size len(gpus)*jobs_per_gpu."""
    cmd = [
        PYTHON, str(RUN_PHASE),
        "--method", method,
        "--datasets", *datasets,
        "--protocols", *protocols,
        "--phase", phase,
        "--gpus", gpus,
        "--jobs-per-gpu", str(jobs_per_gpu),
    ]
    if epochs:
        cmd += ["--epochs", str(epochs)]
    if max_configs and phase == "hp_search":
        cmd += ["--max-configs", str(max_configs)]
    if hp_per_fold:
        cmd += ["--hp-per-fold"]
    print(f"\n>>> {method} / {phase} / {len(datasets)} datasets × {len(protocols)} protocols  "
          f"(gpus={gpus} jobs/gpu={jobs_per_gpu})")
    return subprocess.run(cmd).returncode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", default=list(METHODS.keys()))
    p.add_argument("--datasets", nargs="+", default=DATASETS_SIZE_ORDERED)
    p.add_argument("--protocols", nargs="+", default=PROTOCOLS,
                   choices=PROTOCOLS + ["tdc", "custom"])
    p.add_argument("--phases", nargs="+", default=["default", "hp_search", "hp_final"],
                   choices=["default", "hp_search", "hp_final"])
    p.add_argument("--max-configs", type=int, default=None,
                   help="(hp_search only) cap HP configs — for smoke/scale tests.")
    p.add_argument("--hp-per-fold", action="store_true",
                   help="Select HPs per fold on each fold's own validation (not mean across folds). "
                        "Auto-on for --time-split sliding datasets; prevents temporal HP leakage.")
    p.add_argument("--gpus", default="0", help="comma-separated, e.g. '0,1,2,3'")
    p.add_argument("--jobs-per-gpu", type=int, default=2,
                   help="Concurrent jobs per GPU. Total workers = len(gpus) × jobs_per_gpu.")
    p.add_argument("--epochs-default", type=int, default=None)
    p.add_argument("--epochs-hp-search", type=int, default=None)
    p.add_argument("--epochs-hp-final", type=int, default=None)
    p.add_argument("--order", default="phase_first",
                   choices=["phase_first", "protocol_first", "method_first"],
                   help="phase_first (default): phase → protocol → method, so "
                        "all v1_det defaults finish before v1_preshuffle starts. "
                        "protocol_first: protocol → phase → method, so v1_det "
                        "default+hp_search+hp_final finishes before v1_preshuffle "
                        "starts. method_first: method → phase → protocol.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    epochs_map = {
        "default": args.epochs_default,
        "hp_search": args.epochs_hp_search,
        "hp_final": args.epochs_hp_final,
    }

    # Cell-bundle scheduling: each plan entry = (method, phase, protocol).
    # Each invocation of run_phase pools all (datasets × ensemble seeds) jobs
    # for ONE (method, phase, protocol) cell-bundle into a single ThreadPool.
    plan = []
    if args.order == "phase_first":
        # phase → protocol → method: finish all v1_det defaults, then v1_preshuffle, etc.
        for phase in args.phases:
            for protocol in args.protocols:
                for method in args.methods:
                    plan.append((method, phase, protocol))
    elif args.order == "method_first":
        # method → phase → protocol: original
        for method in args.methods:
            for phase in args.phases:
                for protocol in args.protocols:
                    plan.append((method, phase, protocol))
    else:  # protocol_first
        for protocol in args.protocols:
            for phase in args.phases:
                for method in args.methods:
                    plan.append((method, phase, protocol))

    print(f"=== orchestrator: {len(plan)} (method, phase, protocol) cell-bundles, "
          f"each spanning {len(args.datasets)} datasets ===")
    if args.dry_run:
        for m, ph, p in plan:
            done_count = sum(cell_done(m, d, p, ph) for d in args.datasets)
            total_count = len(args.datasets)
            print(f"  {m}/{ph}/{p}: {done_count}/{total_count} cells done")
        return

    ran, failed = 0, 0
    t0 = time.time()
    for method, phase, protocol in plan:
        # Skip the cell-bundle entirely if ALL its datasets are done.
        all_done = all(cell_done(method, d, protocol, phase) for d in args.datasets)
        if all_done:
            print(f"\n>>> SKIP {method}/{phase}/{protocol}: all cells already done")
            continue
        rc = run_method_phase(method, args.datasets, [protocol], phase,
                              args.gpus, args.jobs_per_gpu, epochs_map[phase],
                              max_configs=args.max_configs, hp_per_fold=args.hp_per_fold)
        ran += 1
        if rc != 0:
            failed += 1
    elapsed = (time.time() - t0) / 60
    print(f"\n=== orchestrator DONE: ran={ran}, failed={failed}, "
          f"elapsed={elapsed:.1f} min ===")


if __name__ == "__main__":
    main()
