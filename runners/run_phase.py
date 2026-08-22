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

PIPELINE = Path(__file__).resolve().parents[1]   # repo root (relocatable)
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


def run_jobs_with_retry(jobs, gpus, jobs_per_gpu, max_oom_retries=None):
    """On OOM, step concurrency DOWN to settle at the highest jobs-per-gpu that still fits — so
    memory stays near-maxed rather than being halved away. First 3 retries decrement by 1
    (fine-grained: reclaims most of the card, ideal after an `auto` near-miss); if it's still
    OOMing after that (a badly-off estimate), it halves to collapse fast — all the way to serial
    (1 job/gpu). If a job still OOMs at serial, that single job exceeds the GPU (reduce batch /
    set MOLFORMER_TOKEN_BUDGET). max_oom_retries kept for API compat; the step-down self-terminates."""
    results = {"ok": [], "oom": [], "fail": [], "timeout": [], "done": []}
    pending = list(jobs)
    jpg = jobs_per_gpu
    step = 0
    while pending:
        workers = len(gpus) * jpg
        # Slot pool: `workers` slots round-robin across GPUs so each GPU holds at most `jpg`
        # concurrent jobs. Rebuilt each round to match the current width.
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
                    print(f"  [future-error] {type(e).__name__}: {e}")
                    continue
                cur_results.append((status, job, elapsed))
                results[status].append((job, elapsed))
                print(f"  [{status:7s}] {job}  ({elapsed:.0f}s)")
        oom_jobs = [job for status, job, _ in cur_results if status == "oom"]
        if not oom_jobs:
            break
        pending = oom_jobs
        if jpg <= 1:   # already serial and STILL OOM -> a single job exceeds GPU memory
            print(f"  [oom] {len(oom_jobs)} job(s) still OOM at 1 job/gpu (serial): a single job "
                  f"exceeds GPU memory — reduce batch / set MOLFORMER_TOKEN_BUDGET; can't reduce further.")
            break
        step += 1
        jpg = (jpg - 1) if step <= 3 else max(1, jpg // 2)   # gentle first (max memory), then halve
        print(f"  [retry {step}] {len(oom_jobs)} OOM(s), reducing to {jpg} job(s)/gpu"
              f"{' (serial)' if jpg == 1 else ''}")
    return results


# ---------------------------------------------------------------------------
def calibrate_jobs_per_gpu(job, gpu, cap=8, safety=0.85, stable_seconds=90, probe_max=300):
    """Launch ONE job on `gpu` and poll its GPU memory; as soon as the peak STABILIZES (no new max
    for `stable_seconds` after training actually starts) or `probe_max` elapses, KILL the probe and
    size the pool: jobs_per_gpu = clamp(floor(total*safety / peak_per_job), 1, cap).

    This measures peak in ~1-3 min instead of running the (possibly hours-long) job to completion —
    so it's fast even on huge datasets. The probe leaves no done.flag, so it simply re-runs in the
    pool. The OOM step-down in run_jobs_with_retry is the backstop if the early peak under-estimates."""
    import signal

    def q(field):
        try:
            r = subprocess.run(["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits",
                                "-i", str(gpu)], capture_output=True, text=True, timeout=10)
            return int(r.stdout.strip().split("\n")[0])
        except Exception:
            return None

    total = q("memory.total")
    base = q("memory.used") or 0
    peak = base
    last_new = time.time()
    started = False
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    job.gpu = gpu
    job.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    proc = subprocess.Popen(job.cmd(), env=env, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        while proc.poll() is None:
            m = q("memory.used")
            if m is not None and m > peak:
                peak = m
                last_new = time.time()
            if not started and peak > base + 500:   # GPU memory rose -> training actually started
                started = True
            now = time.time()
            if (started and now - last_new > stable_seconds) or (now - t0 > probe_max):
                break
            time.sleep(1.0)
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=10)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
    per_job = max(peak - base, 300)
    jobs = max(1, min(cap, int((total * safety) // per_job))) if total else 1
    why = "peak stabilized" if started else "probe timeout"
    print(f"[auto jobs-per-gpu] {why}: peak/job ~{per_job} MiB (base {base}, GPU total {total} MiB) "
          f"in {time.time() - t0:.0f}s -> jobs-per-gpu = {jobs} (probe killed; re-runs in the pool)", flush=True)
    return jobs


def _agg_per_target(per_seed_target):
    """per_seed_target: per-fold list of per-target score lists (aligned by target index, None ok).
    Returns a per-target list of {mean,std,n} across folds. None if nothing multitask to report."""
    lists = [p for p in per_seed_target if p]
    if not lists:
        return None
    n_t = max(len(p) for p in lists)
    out = []
    for t in range(n_t):
        vals = [p[t] for p in lists if t < len(p) and p[t] is not None]
        out.append(cross_seed_summary(vals))
    return out


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
    per_seed_am, per_seed_gm, per_seed_target = [], [], []
    for s in seeds:
        seed_dirs = sorted(base.glob(f"seed{s}_em*"))
        am, gm, per = ensemble_metric_for_seed(seed_dirs, task_type, qm, metric=prescribed_metric)
        per_seed_am.append(am); per_seed_gm.append(gm); per_seed_target.append(per)
    summary = {
        "per_seed_am": per_seed_am, "agg_am": cross_seed_summary(per_seed_am),
        "per_seed_gm": per_seed_gm, "agg_gm": cross_seed_summary(per_seed_gm),
    }
    if meta.get("n_targets", 1) > 1:   # ensembled per-target (multitask only)
        summary["per_seed_per_target"] = per_seed_target
        summary["agg_per_target"] = _agg_per_target(per_seed_target)
        summary["target_columns"] = meta.get("target_columns")
    return summary


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


def _hp_per_fold_enabled(dataset, cli_flag):
    """Per-fold HP selection is on if the CLI flag is set OR the dataset's meta declares it
    (prepare_user_data --time-split sets hp_per_fold on the sliding dataset)."""
    if cli_flag:
        return True
    meta = json.loads((PIPELINE / "cleaned" / f"{dataset}.meta.json").read_text())
    return bool(meta.get("hp_per_fold"))


def _hp_val_scorer(dataset):
    """Return (score_fn(seed_em0_dir)->val|None, minimize), matching pick_best_hp's metric
    handling (TDC prescribed metric, else task-type default)."""
    meta = json.loads((PIPELINE / "cleaned" / f"{dataset}.meta.json").read_text())
    is_tdc = meta.get("source") == "tdc"
    metric_name = meta.get("metric")
    if is_tdc:
        from _tdc_metrics import maximize as _tdc_max, score as _tdc_score
        minimize = not _tdc_max(metric_name)

        def score(seed_dir):
            pv, lv = seed_dir / "pred_val.npy", seed_dir / "labels_val.npy"
            if pv.exists() and lv.exists():
                return _tdc_score(np.load(pv), np.load(lv), metric_name)
            # molfcl/motil (and some molformer modes) don't dump val preds — fall back to their
            # scalar val_metric, exactly like pick_best_hp does. Its direction matches the task
            # (reg: lower rmse/mae; cls: higher auc), consistent with `minimize` above.
            mj = seed_dir / "metrics.json"
            return json.loads(mj.read_text()).get("val_metric") if mj.exists() else None
    else:
        minimize = meta["task_type"] == "reg"

        def score(seed_dir):
            mj = seed_dir / "metrics.json"
            return json.loads(mj.read_text()).get("val_metric") if mj.exists() else None
    return score, minimize


def pick_best_hp_per_fold(phase_dir, dataset, protocol):
    """For EACH fold (eval seed), pick the config best on THAT fold's OWN seed{seed}_em0
    validation — never a mean across folds. This is the anti-leakage rule for time-sliding
    folds: a later fold's validation is an earlier fold's test, so pooling validation would
    tune on data you later test. Returns {str(seed): {id, hp, val_score}}."""
    score_fn, minimize = _hp_val_scorer(dataset)
    winners = {}
    for s in _eval_seeds(dataset, protocol):
        best = None
        for cfg_dir in phase_dir.iterdir():
            if not cfg_dir.is_dir() or not (cfg_dir / "_hp.json").exists():
                continue
            v = score_fn(cfg_dir / f"seed{s}_em0")
            if v is None:
                continue
            if best is None or (minimize and v < best[0]) or (not minimize and v > best[0]):
                best = (v, cfg_dir.name, json.loads((cfg_dir / "_hp.json").read_text()))
        if best is None:
            print(f"  ERROR: no config has a validation score for fold seed{s} in {phase_dir}.")
        else:
            winners[str(s)] = {"id": best[1], "hp": best[2], "val_score": best[0]}
    return winners


def build_hp_final_jobs_per_fold(method, dataset, protocol, phase_dir, epochs, gpus, folds):
    """Each fold trains ITS OWN winning config (folds may differ) at phase_dir/<id>/seed{s}_em*."""
    jobs, i = [], 0
    for s in _eval_seeds(dataset, protocol):
        w = folds.get(str(s))
        if w is None:
            continue
        for em in range(ENSEMBLE_SIZE):
            out = phase_dir / w["id"] / f"seed{s}_em{em}"
            jobs.append(Job(method, dataset, protocol, s, w["hp"], w["id"],
                            em, epochs, out, gpus[i % len(gpus)]))
            i += 1
    return jobs


def aggregate_phase_per_fold(method, dataset, protocol, phase_dir, folds):
    """Like aggregate_phase, but each fold's seeds live under its OWN winner's id dir."""
    meta = json.loads((PIPELINE / "cleaned" / f"{dataset}.meta.json").read_text())
    task_type, qm = meta["task_type"], dataset in ("qm7", "qm8", "qm9")
    prescribed_metric = meta["metric"] if meta.get("source") == "tdc" else None
    per_seed_am, per_seed_gm, per_seed_target, used = [], [], [], {}
    for s in _eval_seeds(dataset, protocol):
        w = folds.get(str(s))
        if w is None:
            continue
        seed_dirs = sorted((phase_dir / w["id"]).glob(f"seed{s}_em*"))
        am, gm, per = ensemble_metric_for_seed(seed_dirs, task_type, qm, metric=prescribed_metric)
        per_seed_am.append(am); per_seed_gm.append(gm); per_seed_target.append(per); used[str(s)] = w["id"]
    summary = {
        "per_seed_am": per_seed_am, "agg_am": cross_seed_summary(per_seed_am),
        "per_seed_gm": per_seed_gm, "agg_gm": cross_seed_summary(per_seed_gm),
        "per_fold_hp": used,
    }
    if meta.get("n_targets", 1) > 1:
        summary["per_seed_per_target"] = per_seed_target
        summary["agg_per_target"] = _agg_per_target(per_seed_target)
        summary["target_columns"] = meta.get("target_columns")
    return summary


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
    p.add_argument("--jobs-per-gpu", default="1",
                   help="Jobs per GPU, or 'auto' to measure one job's peak GPU memory and fit as many "
                        "as the card holds (safety 0.85, OOM-halving backstop). Adapts to 48/80 GB cards.")
    p.add_argument("--max-configs", type=int, default=None,
                   help="(hp_search only) cap number of HP configs (smoke testing).")
    p.add_argument("--hp-per-fold", action="store_true",
                   help="Select HPs per fold on each fold's OWN validation instead of the mean "
                        "across folds. Auto-on for datasets whose meta sets hp_per_fold (the "
                        "--time-split sliding dataset). Prevents temporal HP leakage across folds.")
    args = p.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]
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
                if rec.get("per_fold"):
                    cell_jobs = build_hp_final_jobs_per_fold(args.method, dataset, protocol,
                                                             phase_dir, epochs, gpus, rec["folds"])
                else:
                    cell_jobs = build_hp_final_jobs(args.method, dataset, protocol,
                                                    phase_dir, epochs, gpus,
                                                    rec["hp"], rec["id"])
            all_jobs.extend(cell_jobs)

    # Resolve jobs-per-gpu: 'auto' measures one job's GPU memory and fits as many as the card holds.
    if str(args.jobs_per_gpu) == "auto":
        todo = [j for j in all_jobs if not (j.out_dir / "done.flag").exists()]
        jobs_per_gpu = calibrate_jobs_per_gpu(todo[0], gpus[0]) if todo else 1
    else:
        jobs_per_gpu = int(args.jobs_per_gpu)
    max_workers = len(gpus) * jobs_per_gpu

    # GPU is assigned dynamically at run time from a slot pool (see run_job /
    # run_jobs_with_retry), so no static per-job pinning here.
    print(f"=== run_phase scheduling {len(all_jobs)} jobs across {len(args.datasets)} datasets, "
          f"{len(args.protocols)} protocols ({args.method}/{args.phase}) ===")
    print(f"max_workers = {len(gpus)} GPUs × {jobs_per_gpu} jobs/gpu = {max_workers}")

    run_jobs_with_retry(all_jobs, gpus, jobs_per_gpu)

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
                if _hp_per_fold_enabled(dataset, args.hp_per_fold):
                    folds = pick_best_hp_per_fold(phase_dir, dataset, protocol)
                    summary = {"per_fold": True, "folds": folds}
                    if folds:
                        (phase_root / "best_hp.json").write_text(
                            json.dumps({"per_fold": True, "folds": folds}, indent=2))
                else:
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
                if rec.get("per_fold"):
                    summary = aggregate_phase_per_fold(args.method, dataset, protocol,
                                                       phase_dir, rec["folds"])
                else:
                    summary = aggregate_phase(args.method, dataset, protocol,
                                              phase_dir, rec["id"])
            (phase_dir / "_summary.json").write_text(json.dumps(summary, indent=2, default=str))
            am = summary.get("agg_am", {}).get("mean") if isinstance(summary, dict) else None
            print(f"  {args.method}/{dataset}/{protocol}/{args.phase}: agg_am={am}")


if __name__ == "__main__":
    main()
