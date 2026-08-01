#!/bin/bash
# Watchdog for v2_astartes hp_search + hp_final.
# - Restarts orchestrator if it dies.
# - Heals stale _summary.json (incomplete cells) so they re-run instead of being skipped.
# - Downgrades jobs-per-gpu on no-progress; HARD STOPS after repeated stalls at min
#   concurrency (prevents infinite loops on permanently-failing configs).
set -u
cd /data/rbg/users/yxie25/molclr_chemprop/clean_pipeline_v1
PY=/data/rbg/users/yxie25/anaconda3/envs/foldeverything/bin/python
STATUS=orchestrator/hp_v2_status.py
LOGDIR=orchestrator_logs
mkdir -p "$LOGDIR"

JOBS_PER_GPU=6
MIN_JPG=3
ATTEMPTS=0
LAST_REMAINING=$($PY $STATUS count)
STALLS=0

while true; do
  ATTEMPTS=$((ATTEMPTS+1))
  REMAINING=$($PY $STATUS count)
  echo "==================== hp-v2 watchdog attempt $ATTEMPTS  remaining=$REMAINING  jobs-per-gpu=$JOBS_PER_GPU  stalls=$STALLS  $(date) ===================="
  if [ "$REMAINING" -le 0 ]; then
    echo "ALL DONE — exiting watchdog"
    break
  fi
  echo "--- heal stale summaries ---"
  $PY $STATUS heal
  LOG="$LOGDIR/hp_v2_attempt${ATTEMPTS}_$(date +%Y%m%d_%H%M%S).log"
  echo "log: $LOG"
  $PY -u orchestrator/run_benchmark.py \
    --methods molclr chemprop2 chemeleon molfcl motil \
    --datasets freesolv esol sider clintox bace bbbp lipo qm7 tox21 qm8 hiv \
    --phases hp_search hp_final --protocols v2_astartes \
    --gpus 0,1,2,3,4,5,6 --jobs-per-gpu "$JOBS_PER_GPU" --order phase_first \
    > "$LOG" 2>&1
  RC=$?
  NEW_REMAINING=$($PY $STATUS count)
  PROGRESS=$((REMAINING - NEW_REMAINING))
  echo "=== exit=$RC remaining: $REMAINING -> $NEW_REMAINING (progress=$PROGRESS) at $(date) ==="

  if [ "$NEW_REMAINING" -le 0 ]; then
    echo "ALL DONE — exiting watchdog"
    break
  fi

  if [ "$PROGRESS" -lt 5 ]; then
    STALLS=$((STALLS+1))
    if [ "$JOBS_PER_GPU" -gt "$MIN_JPG" ]; then
      JOBS_PER_GPU=$((JOBS_PER_GPU-1))
      echo "=== stall #$STALLS — reducing jobs-per-gpu to $JOBS_PER_GPU ==="
    elif [ "$STALLS" -ge 4 ]; then
      echo "=== HARD STOP: $STALLS stalls at min concurrency, $NEW_REMAINING jobs unrecoverable. Investigate. ==="
      $PY $STATUS report
      break
    else
      echo "=== stall #$STALLS at min concurrency ==="
    fi
  else
    STALLS=0
  fi
  LAST_REMAINING=$NEW_REMAINING
  sleep 30
done
