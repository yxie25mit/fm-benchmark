"""
Unified phase runner: handles ONE (method × dataset × protocol × phase) cell.

Phases
------
default    : default HP × eval seeds × 5 ensemble (no search)
hp_search  : ALL ~108 HPs × HP seed × 1 ensemble (epochs=30 by default), pick best by val
hp_final   : best HP × eval seeds × 5 ensemble (epochs=100 by default)

Layout under results/<method>/<dataset>/<protocol>/:
  default/<config_id>/seed<S>_em<E>/{pred_test.npy,labels_test.npy,metrics.json,done.flag}
  hp_search/<config_id>/{_hp.json, seed<HP_SEED>_em0/...}
  hp_final/<config_id>/seed<S>_em<E>/...
  best_hp.json   (after hp_search)
  <phase>/_summary.json
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue

PIPELINE = Path("/data/rbg/users/yxie25/molclr_chemprop/clean_pipeline_v1")
sys.path.insert(0, str(PIPELINE / "methods"))

from configs import (  # noqa: E402
    METHODS, EVAL_SEEDS, HP_SEED, ENSEMBLE_SIZE,
    all_hp_configs, hp_config_id, expand_molformer_head,
)
from _eval import (  # noqa: E402
    ensemble_metric_for_seed, cross_seed_summary,
)


# ---------------------------------------------------------------------------
class Job:
    def __init__(self, method, dataset, protocol, seed, hp, hp_id,
                 ensemble_member, epochs, out_dir, gpu):
        self.method, self.dataset, self.protocol = method, dataset, protocol
        self.seed = seed
        self.hp = hp                       # dict OR "default"
        self.hp_id = hp_id                 # "default" OR md5 short
        self.ensemble_member = ensemble_member
        self.epochs = epochs
        self.out_dir = Path(out_dir)
        self.gpu = gpu

    def cmd(self):
        cfg = METHODS[self.method]
        # CUDA_VISIBLE_DEVICES is set in run_job() to mask all other GPUs; pass
        # "--gpu 0" (which inside subprocess refers to the only visible GPU).
        cmd = [
            cfg["python"], cfg["worker"],
            "--dataset", self.dataset,
            "--protocol", self.protocol,
            "--seed", str(self.seed),
            "--epochs", str(self.epochs),
            "--ensemble-member", str(self.ensemble_member),
            "--out-dir", str(self.out_dir),
            "--gpu", "0",
        ]
        if self.hp == "default":
            cmd += ["--hp-config", "default"]
        else:
            hp_path = self.out_dir.parent / "_hp.json"
            self.out_dir.parent.mkdir(parents=True, exist_ok=True)
            if not hp_path.exists():
                hp_to_dump = expand_molformer_head(self.hp) if self.method == "molformer" else self.hp
                with open(hp_path, "w") as f:
                    json.dump(hp_to_dump, f)
            cmd += ["--hp-config", str(hp_path)]
        return cmd

    def __repr__(self):
        return (f"Job({self.method}/{self.dataset}/{self.protocol}/{self.hp_id} "
                f"seed={self.seed} em={self.ensemble_member} gpu={self.gpu})")


# ---------------------------------------------------------------------------
def _eval_seeds(dataset, protocol):
    """Eval seeds for a protocol. For 'custom', one per user fold (splits/<ds>/custom_seed*/)."""
    if protocol == "custom":
        folds = sorted(int(p.name.replace("custom_seed", ""))
                       for p in (PIPELINE / "splits" / dataset).glob("custom_seed*"))
        return folds or [0]
    return EVAL_SEEDS[protocol]


def build_default_jobs(method, dataset, protocol, phase_dir, epochs, gpus):
    seeds = _eval_seeds(dataset, protocol)
    jobs = []
    i = 0
    for s in seeds:
        for em in range(ENSEMBLE_SIZE):
            out = phase_dir / "default" / f"seed{s}_em{em}"
            jobs.append(Job(method, dataset, protocol, s,
                            "default", "default", em, epochs, out, gpus[i % len(gpus)]))
            i += 1
    return jobs


def build_hp_search_jobs(method, dataset, protocol, phase_dir, epochs, gpus, max_configs=None):
    # HP_SEED may be an int (MoleculeNet: one HP seed) or a list (TDC: run every config
    # on all 5 train/val splits and select by the MEAN validation — pick_best_hp averages).
    if protocol == "custom":                        # like TDC: run every config on ALL user folds
        hp_seeds = _eval_seeds(dataset, protocol)   # and pick best by MEAN validation across folds
    else:
        hp_seeds = HP_SEED[protocol]
        if isinstance(hp_seeds, int):
            hp_seeds = [hp_seeds]
    configs_list = all_hp_configs(method)
    if max_configs is not None:
        configs_list = configs_list[:max_configs]
    jobs = []
    n = 0
    for hp in configs_list:
        hid = hp_config_id(hp)
        for hp_seed in hp_seeds:
            out = phase_dir / hid / f"seed{hp_seed}_em0"
            jobs.append(Job(method, dataset, protocol, hp_seed, hp, hid,
                            0, epochs, out, gpus[n % len(gpus)]))
            n += 1
    return jobs


def build_hp_final_jobs(method, dataset, protocol, phase_dir, epochs, gpus, best_hp, hp_id):
    seeds = _eval_seeds(dataset, protocol)
    jobs = []
    i = 0
    for s in seeds:
        for em in range(ENSEMBLE_SIZE):
            out = phase_dir / hp_id / f"seed{s}_em{em}"
            jobs.append(Job(method, dataset, protocol, s, best_hp, hp_id,
                            em, epochs, out, gpus[i % len(gpus)]))
            i += 1
    return jobs


# ---------------------------------------------------------------------------
def _safe_write(path, text):
    """Best-effort write that won't crash the thread on NFS / missing dir issues."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text or "")
    except Exception as e:
        # Last-resort: log to stdout. Don't propagate — never kill the thread pool.
        print(f"  [warn] could not write {path}: {e}")


def run_job(job: Job, gpu_queue: Queue):
    try:
        if (job.out_dir / "done.flag").exists():
            return ("done", job, 0.0)
        # Acquire a physical GPU from the shared pool at RUN time (not statically by
        # job index): the pool holds exactly jobs_per_gpu slots per GPU, so no GPU
        # ever runs more than jobs_per_gpu concurrent jobs even when jobs finish at
        # different rates. Static index-pinning + a shared thread pool let a freed
        # worker start a job pinned to a still-busy GPU -> collisions -> OOM.
        gpu = gpu_queue.get()
        job.gpu = gpu
        try:
            job.out_dir.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            t0 = time.time()
            try:
                result = subprocess.run(job.cmd(), env=env, capture_output=True,
                                        text=True, timeout=108000)  # 30h: headroom for the largest TDC cells; still self-kills a true wedge
            except subprocess.TimeoutExpired:
                _safe_write(job.out_dir / "timeout.flag", "")
                return ("timeout", job, time.time() - t0)
            elapsed = time.time() - t0
            if result.returncode != 0:
                _safe_write(job.out_dir / "stderr.log", result.stderr)
                _safe_write(job.out_dir / "stdout.log", result.stdout)
                is_oom = "out of memory" in (result.stderr or "").lower()
                return ("oom" if is_oom else "fail", job, elapsed)
            return ("ok", job, elapsed)
        finally:
            gpu_queue.put(gpu)
    except Exception as e:
        # Catch ANY exception so the thread pool never dies. Worst case: surface as fail.
        print(f"  [worker-error] {job}: {type(e).__name__}: {e}")
        return ("fail", job, 0.0)


def run_jobs_with_retry(jobs, gpus, jobs_per_gpu, max_oom_retries=2):
    results = {"ok": [], "oom": [], "fail": [], "timeout": [], "done": []}
    pending = list(jobs)
    workers = len(gpus) * jobs_per_gpu
    retries = 0
    while pending and retries <= max_oom_retries:
        # Slot pool: `workers` slots spread round-robin across GPUs so each GPU holds
        # at most ceil(workers/len(gpus)) concurrent jobs (== jobs_per_gpu at full
        # width, fewer after a halving retry). Rebuilt each retry to match `workers`.
        gpu_queue = Queue()
        for i in range(workers):
            gpu_queue.put(gpus[i % len(gpus)])
        cur_results = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(run_job, j, gpu_queue) for j in pending]
            for f in as_completed(futs):
                try:
                    status, job, elapsed = f.result()
                except Exception as e:
                    # Belt-and-suspenders: if a worker still raises (shouldn't with try/except above),
                    # log it and continue rather than crashing the pool.
                    print(f"  [future-error] {type(e).__name__}: {e}")
                    continue
                cur_results.append((status, job, elapsed))
                results[status].append((job, elapsed))
                print(f"  [{status:7s}] {job}  ({elapsed:.0f}s)")
        oom_jobs = [job for status, job, _ in cur_results if status == "oom"]
        if not oom_jobs:
            break
        pending = oom_jobs
        workers = max(1, workers // 2)
        retries += 1
        print(f"  [retry {retries}] {len(oom_jobs)} OOMs, halving workers to {workers}")
    return results


# ---------------------------------------------------------------------------
def aggregate_phase(method, dataset, protocol, phase_dir, hp_id):
    """Ensemble-average preds per seed, compute metric, summarize across seeds.

    phase_dir is the directory containing the seed_em dirs (or hp_id subdirs).
    For default phase: phase_dir = .../default, hp_id = "default" — jobs at phase_dir/seed_em.
    For hp_final:      phase_dir = .../hp_final, hp_id = best_id  — jobs at phase_dir/<id>/seed_em.
    """
    meta = json.loads((PIPELINE / "cleaned" / f"{dataset}.meta.json").read_text())
    task_type, qm = meta["task_type"], dataset in ("qm7", "qm8", "qm9")
    # TDC: score the fixed test with the dataset's prescribed metric (mae/spearman/roc-auc/
    # pr-auc). MoleculeNet: prescribed_metric=None -> macro AUC / RMSE / MAE as before.
    prescribed_metric = meta["metric"] if meta.get("source") == "tdc" else None
    seeds = _eval_seeds(dataset, protocol)
    base = phase_dir if hp_id == "default" else phase_dir / hp_id
    per_seed_am, per_seed_gm = [], []
    for s in seeds:
        seed_dirs = sorted(base.glob(f"seed{s}_em*"))
        am, gm, _ = ensemble_metric_for_seed(seed_dirs, task_type, qm, metric=prescribed_metric)
        per_seed_am.append(am); per_seed_gm.append(gm)
    return {
        "per_seed_am": per_seed_am, "agg_am": cross_seed_summary(per_seed_am),
        "per_seed_gm": per_seed_gm, "agg_gm": cross_seed_summary(per_seed_gm),
    }


def pick_best_hp(phase_dir, dataset):
    """Pick the HP config with the best VALIDATION score. Never touches test (no leakage).

    MoleculeNet: one HP seed per config -> read that seed's worker-written val_metric.
    TDC: every config was run on all 5 train/val splits -> re-score each split's saved
    val predictions with the dataset's PRESCRIBED metric (meta["metric"]) via _tdc_metrics,
    then rank by the MEAN over the 5 splits. Direction (min/max) is metric-driven
    (mae/rmse minimize; roc_auc/pr_auc/spearman/pearson maximize)."""
    meta = json.loads((PIPELINE / "cleaned" / f"{dataset}.meta.json").read_text())
    is_tdc = meta.get("source") == "tdc"
    metric_name = meta.get("metric")
    if is_tdc:
        from _tdc_metrics import maximize as _tdc_max, score as _tdc_score
        minimize = not _tdc_max(metric_name)
    else:
        minimize = meta["task_type"] == "reg"

    def config_val_score(cfg_dir):
        """Mean validation score over this config's seed_em0 dirs (val-only, no test)."""
        vals = []
        for sd in sorted(cfg_dir.glob("seed*_em0")):
            pv, lv = sd / "pred_val.npy", sd / "labels_val.npy"
            if is_tdc and pv.exists() and lv.exists():
                vals.append(_tdc_score(np.load(pv), np.load(lv), metric_name))
            elif (sd / "metrics.json").exists():
                v = json.loads((sd / "metrics.json").read_text()).get("val_metric")
                if v is not None:
                    vals.append(v)
        return float(np.mean(vals)) if vals else None

    best_id, best_score, best_hp = None, None, None
    for cfg_dir in phase_dir.iterdir():
        if not cfg_dir.is_dir():
            continue
        hp_json = cfg_dir / "_hp.json"
        if not hp_json.exists():
            continue
        score = config_val_score(cfg_dir)
        if score is None:
            continue
        if best_score is None or (minimize and score < best_score) or (not minimize and score > best_score):
            best_score, best_id, best_hp = score, cfg_dir.name, json.loads(hp_json.read_text())
    if best_id is None:
        print(f"  ERROR: no config has a validation score in {phase_dir} — refusing to pick HP on test.")
    return best_id, best_hp, best_score


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=list(METHODS.keys()))
    # Multi-cell scheduling: accept lists for cross-cell job pooling.
    p.add_argument("--datasets", nargs="+", required=True,
                   help="One or more datasets. Multiple → cross-cell pooling.")
    p.add_argument("--protocols", nargs="+", required=True,
                   choices=["v1_det", "v1_preshuffle", "v2_astartes", "tdc", "custom"],
                   help="One or more protocols. Multiple → cross-cell pooling.")
    p.add_argument("--phase", required=True, choices=["default", "hp_search", "hp_final"])
    p.add_argument("--epochs", type=int, default=None,
                   help="Override default epochs (default 100, hp_search 30).")
    p.add_argument("--gpus", type=str, default="0",
                   help="Comma-separated GPU ids, e.g., '0,1,2,3'.")
    p.add_argument("--jobs-per-gpu", type=int, default=1,
                   help="Jobs per GPU (cross-cell parallelism). max_workers = len(gpus)*jobs_per_gpu.")
    p.add_argument("--max-configs", type=int, default=None,
                   help="(hp_search only) cap number of HP configs (smoke testing).")
    args = p.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]
    max_workers = len(gpus) * args.jobs_per_gpu
    epochs = args.epochs or (30 if args.phase == "hp_search" else 100)

    # Build the global job list across all (dataset, protocol) cells.
    all_jobs = []
    for protocol in args.protocols:
        for dataset in args.datasets:
            phase_root = PIPELINE / "results" / args.method / dataset / protocol
            phase_dir = phase_root / args.phase
            phase_dir.mkdir(parents=True, exist_ok=True)
            if args.phase == "default":
                cell_jobs = build_default_jobs(args.method, dataset, protocol,
                                               phase_root, epochs, gpus)
            elif args.phase == "hp_search":
                cell_jobs = build_hp_search_jobs(args.method, dataset, protocol,
                                                 phase_dir, epochs, gpus,
                                                 max_configs=args.max_configs)
            elif args.phase == "hp_final":
                best_hp_path = phase_root / "best_hp.json"
                if not best_hp_path.exists():
                    print(f"  WARN: skipping hp_final {dataset}/{protocol}: best_hp.json missing")
                    continue
                rec = json.loads(best_hp_path.read_text())
                cell_jobs = build_hp_final_jobs(args.method, dataset, protocol,
                                                phase_dir, epochs, gpus,
                                                rec["hp"], rec["id"])
            all_jobs.extend(cell_jobs)

    # GPU is assigned dynamically at run time from a slot pool (see run_job /
    # run_jobs_with_retry), so no static per-job pinning here.
    print(f"=== run_phase scheduling {len(all_jobs)} jobs across {len(args.datasets)} datasets, "
          f"{len(args.protocols)} protocols ({args.method}/{args.phase}) ===")
    print(f"max_workers = {len(gpus)} GPUs × {args.jobs_per_gpu} jobs/gpu = {max_workers}")

    run_jobs_with_retry(all_jobs, gpus, args.jobs_per_gpu)

    # Aggregate per cell after global pool finishes.
    print("\n=== per-cell summaries ===")
    for protocol in args.protocols:
        for dataset in args.datasets:
            phase_root = PIPELINE / "results" / args.method / dataset / protocol
            phase_dir = phase_root / args.phase
            if args.phase == "default":
                summary = aggregate_phase(args.method, dataset, protocol,
                                          phase_dir, "default")
            elif args.phase == "hp_search":
                best_id, best_hp, best_score = pick_best_hp(phase_dir, dataset)
                summary = {"best_id": best_id, "best_hp": best_hp, "best_val_score": best_score}
                if best_hp is not None:
                    (phase_root / "best_hp.json").write_text(json.dumps(
                        {"hp": best_hp, "id": best_id, "val_score": best_score}, indent=2))
            elif args.phase == "hp_final":
                best_hp_path = phase_root / "best_hp.json"
                if not best_hp_path.exists():
                    continue
                rec = json.loads(best_hp_path.read_text())
                summary = aggregate_phase(args.method, dataset, protocol,
                                          phase_dir, rec["id"])
            (phase_dir / "_summary.json").write_text(json.dumps(summary, indent=2, default=str))
            am = summary.get("agg_am", {}).get("mean") if isinstance(summary, dict) else None
            print(f"  {args.method}/{dataset}/{protocol}/{args.phase}: agg_am={am}")


if __name__ == "__main__":
    main()
