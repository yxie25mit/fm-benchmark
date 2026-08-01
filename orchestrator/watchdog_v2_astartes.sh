#!/bin/bash
# Watchdog: restart orchestrator if it dies, downgrade jobs-per-gpu after repeated failures.
set -u
cd /data/rbg/users/yxie25/molclr_chemprop/clean_pipeline_v1
PY=/data/rbg/users/yxie25/anaconda3/envs/foldeverything/bin/python
LOGDIR=orchestrator_logs
mkdir -p "$LOGDIR"

JOBS_PER_GPU=6
ATTEMPTS=0
LAST_REMAINING=999999

count_remaining() {
  $PY -c "
import glob
need={'molclr':165,'chemprop2':165,'chemeleon':165,'molfcl':165,'motil':165}
remaining = 0
for m, need_m in need.items():
    d = len(glob.glob(f'results/{m}/*/v2_astartes/default/seed*_em*/done.flag'))
    remaining += max(0, need_m - d)
print(remaining)
"
}

# Delete _summary.json for any cell with <15 done.flags. A stale summary makes
# the orchestrator's cell_done() skip an incomplete cell forever (infinite
# restart loop). Run before every orchestrator launch so skips can't strand jobs.
auto_heal_summaries() {
  $PY -c "
import glob, os
for f in glob.glob('results/*/*/v2_astartes/default/_summary.json'):
    cell = os.path.dirname(f)
    n = len(glob.glob(os.path.join(cell, 'seed*_em*/done.flag')))
    if n < 15:
        os.remove(f)
        print('  auto-heal: removed stale', f, f'({n}/15)')
"
}

while true; do
  ATTEMPTS=$((ATTEMPTS+1))
  REMAINING=$(count_remaining)
  echo "==================== watchdog attempt $ATTEMPTS  remaining=$REMAINING  jobs-per-gpu=$JOBS_PER_GPU  $(date) ===================="
  if [ "$REMAINING" -le 0 ]; then
    echo "ALL DONE — exiting watchdog"
    break
  fi
  LOG="$LOGDIR/v2_astartes_attempt${ATTEMPTS}_$(date +%Y%m%d_%H%M%S).log"
  echo "log: $LOG"
  auto_heal_summaries
  $PY -u orchestrator/run_benchmark.py \
    --methods molclr chemprop2 chemeleon molfcl motil \
    --datasets freesolv esol sider clintox bace bbbp lipo qm7 tox21 qm8 hiv \
    --phases default --protocols v2_astartes \
    --gpus 0,1,2,3,4,5,6 --jobs-per-gpu "$JOBS_PER_GPU" --order phase_first \
    > "$LOG" 2>&1
  RC=$?
  NEW_REMAINING=$(count_remaining)
  PROGRESS=$((LAST_REMAINING - NEW_REMAINING))
  echo "=== exit=$RC remaining: $REMAINING -> $NEW_REMAINING (progress=$PROGRESS) at $(date) ==="

  if [ "$NEW_REMAINING" -le 0 ]; then
    echo "ALL DONE — exiting watchdog"
    break
  fi

  # If we restarted but made <10 jobs of progress, the run is likely
  # crashing on the same OOM/contention. Drop jobs-per-gpu.
  if [ "$PROGRESS" -lt 10 ] && [ "$ATTEMPTS" -ge 2 ] && [ "$JOBS_PER_GPU" -gt 3 ]; then
    JOBS_PER_GPU=$((JOBS_PER_GPU - 1))
    echo "=== little progress — reducing jobs-per-gpu to $JOBS_PER_GPU ==="
  fi
  LAST_REMAINING=$NEW_REMAINING
  sleep 30
done
