#!/usr/bin/env bash
# Spot-preemption experiment on the real GR00T trainer.
#
# For one backend it runs:
#   ref : MAX_STEPS uninterrupted
#   pre : same config, SIGTERM to the trainer once it passes KILL_AT_STEP, then relaunch with
#         --resume_from_checkpoint and train to MAX_STEPS
# and writes a timeline (events.jsonl) next to the per-step bench metrics, so loss after the
# resume can be compared step-by-step against the uninterrupted reference.
#
#   BACKEND=lance  DATASET=/data/datasets/libero_10_lance OUT=/data/runs/exp2/lance  bash scripts/preempt_experiment.sh
#   BACKEND=native DATASET=/data/datasets/libero_10       OUT=/data/runs/exp2/native bash scripts/preempt_experiment.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="${BACKEND:?}"
DATASET="${DATASET:?}"
OUT="${OUT:?}"
MAX_STEPS="${MAX_STEPS:-200}"
SAVE_STEPS="${SAVE_STEPS:-100}"
KILL_AT_STEP="${KILL_AT_STEP:-130}"
SKIP_REF="${SKIP_REF:-0}"
mkdir -p "$OUT"
EV="$OUT/events.jsonl"
ev() { printf '{"t": %s, "event": "%s"%s}\n' "$(date +%s.%N)" "$1" "${2:+, $2}" >> "$EV"; }

steps_done() { grep -c '"event": "step"' "$1/bench_metrics.jsonl" 2>/dev/null || echo 0; }
last_step() { grep '"event": "step"' "$1/bench_metrics.jsonl" 2>/dev/null | tail -1 | sed -E 's/.*"global_step": ([0-9]+).*/\1/'; }

if [ "$SKIP_REF" != 1 ]; then
  ev ref_start "\"backend\": \"$BACKEND\""
  BACKEND=$BACKEND DATASET=$DATASET OUT=$OUT/ref MAX_STEPS=$MAX_STEPS SAVE_STEPS=$SAVE_STEPS \
    bash "$HERE/run_gr00t_finetune.sh" > "$OUT/ref.log" 2>&1
  ev ref_exit "\"code\": $?"
fi

ev pre_start "\"backend\": \"$BACKEND\""
BACKEND=$BACKEND DATASET=$DATASET OUT=$OUT/pre MAX_STEPS=$MAX_STEPS SAVE_STEPS=$SAVE_STEPS \
  bash "$HERE/run_gr00t_finetune.sh" > "$OUT/pre_1.log" 2>&1 &
LAUNCHER=$!
while kill -0 $LAUNCHER 2>/dev/null; do
  s=$(last_step "$OUT/pre"); s=${s:-0}
  if [ "$s" -ge "$KILL_AT_STEP" ]; then
    PID=$(pgrep -f "[.]venv/bin/python .*launch_finetune_lance.py.*$OUT/pre" | head -1)
    ev sigterm "\"step\": $s, \"pid\": ${PID:-0}"
    [ -n "$PID" ] && kill -TERM "$PID"
    break
  fi
  sleep 0.5
done
wait $LAUNCHER; CODE=$?
ev pre_exit "\"code\": $CODE, \"last_step\": $(last_step "$OUT/pre")"
ls -d "$OUT"/pre/checkpoint-* 2>/dev/null | sed 's/^/  checkpoint: /'

ev resume_start
BACKEND=$BACKEND DATASET=$DATASET OUT=$OUT/pre MAX_STEPS=$MAX_STEPS SAVE_STEPS=$SAVE_STEPS \
  bash "$HERE/run_gr00t_finetune.sh" --resume_from_checkpoint > "$OUT/pre_2.log" 2>&1
ev resume_exit "\"code\": $?, \"last_step\": $(last_step "$OUT/pre")"
echo "done: $OUT"
