#!/usr/bin/env bash
# Flywheel fast validation on GR00T N1.7 + LIBERO-10 (one seed, G0 vs G3 sharing 70% of training).
# See docs/flywheel-fast-validation.md. Run one stage at a time:
#
#   STAGE=smoke    bash scripts/flywheel/fast_validation.sh   # 200 steps + 1 episode/task, timings
#   STAGE=embed    ...                                        # kinematic embedding column (CPU)
#   STAGE=g0       ...                                        # baseline, 20k steps, keeps 5k/10k/14k/20k
#   STAGE=curve    ...                                        # 100-episode eval of 5k/10k/20k
#   STAGE=mine     ...                                        # 500-episode eval of 14k + failure mining
#   STAGE=g3       ...                                        # fork from G0@14k, weighted 6k steps
#   STAGE=final    ...                                        # 500-episode eval of G0@20k and G3@20k + paired report
#
# Required: DATASET (libero_10 Lance dir), RUNS (output root). Optional overrides below.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
STAGE="${STAGE:?set STAGE=smoke|embed|g0|curve|mine|g3|final}"
DATASET="${DATASET:?set DATASET to the libero_10 Lance dataset}"
RUNS="${RUNS:?set RUNS to the output root}"
GR00T_ROOT="${GR00T_ROOT:-/data/Isaac-GR00T}"
SIM_PY="${SIM_PY:-$GR00T_ROOT/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python}"
PROJECT_PY="${PROJECT_PY:-python}"          # env with gr00t-lancedb + lancedb (for embed/mine/report)
MAX_STEPS="${MAX_STEPS:-20000}"
FORK_STEP="${FORK_STEP:-14000}"
CURVE_STEPS="${CURVE_STEPS:-5000,10000}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"        # per task -> 500 for libero_10
CURVE_EPISODES="${CURVE_EPISODES:-10}"      # per task -> 100
N_ENVS="${N_ENVS:-10}"
EVAL_SEED="${EVAL_SEED:-7}"
PORT="${PORT:-5555}"
WEIGHT_COLUMN="${WEIGHT_COLUMN:-flywheel_weight}"
G0="$RUNS/g0"
G3="$RUNS/g3"
mkdir -p "$RUNS"

train() {  # train <out> [extra launch_finetune args...]
  local out="$1"; shift
  BACKEND=lance DATASET="$DATASET" OUT="$out" MAX_STEPS="$MAX_STEPS" SAVE_STEPS="$SAVE_STEPS" \
    GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}" \
    bash "$REPO/scripts/run_gr00t_finetune.sh" "$@"
}

serve_and_record() {  # serve_and_record <ckpt> <out_dir> <episodes_per_task>
  local ckpt="$1" out="$2" eps="$3"
  [ -f "$out/summary.jsonl" ] && [ "${RERECORD:-0}" != 1 ] && { echo "exists: $out (RERECORD=1 to redo)"; return; }
  ( cd "$GR00T_ROOT" && exec uv run --no-sync python gr00t/eval/run_gr00t_server.py \
      --model-path "$ckpt" --embodiment-tag LIBERO_PANDA --use-sim-policy-wrapper --port "$PORT" ) \
      > "$out.server.log" 2>&1 &
  local pid=$!
  trap "kill $pid 2>/dev/null || true" RETURN
  for _ in $(seq 1 180); do
    (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null && break
    kill -0 "$pid" 2>/dev/null || { echo "policy server died, see $out.server.log"; exit 1; }
    sleep 2
  done
  ( cd "$GR00T_ROOT" && "$SIM_PY" "$HERE/record_libero_rollouts.py" --out "$out" \
      --episodes-per-task "$eps" --n-envs "$N_ENVS" --base-seed "$EVAL_SEED" \
      --policy-client-host 127.0.0.1 --policy-client-port "$PORT" --checkpoint "$ckpt" --resume )
  kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true
}

case "$STAGE" in
  smoke)
    MAX_STEPS=200 SAVE_STEPS=200 GR00T_KEEP_CKPT_STEPS=200 train "$RUNS/smoke"
    serve_and_record "$RUNS/smoke/keep/checkpoint-200" "$RUNS/smoke/eval_200" 1
    "$PROJECT_PY" -m gr00t_lancedb.flywheel.cli success --rollouts "$RUNS/smoke/eval_200"
    "$PROJECT_PY" - "$RUNS/smoke/bench_metrics.jsonl" <<'PY'
import json, sys
recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
dts = [r["step_s"] for r in recs if r.get("event") == "step" and r.get("step_s")][20:]
if dts:
    dts.sort(); print(f"steady s/step (median, steps 20+): {dts[len(dts)//2]:.3f}")
PY
    ;;
  embed)
    "$PROJECT_PY" -c "from gr00t_lancedb.vector_curate import main_vector_curate as m; m()" \
      embed --root "$DATASET" --provider kinematic --horizon 16 --out-json "$RUNS/embed_report.json"
    ;;
  g0)
    GR00T_KEEP_CKPT_STEPS="$FORK_STEP" GR00T_KEEP_CKPT_WEIGHTS_ONLY="$CURVE_STEPS,$MAX_STEPS" train "$G0"
    ;;
  curve)
    for s in ${CURVE_STEPS//,/ } "$MAX_STEPS"; do
      serve_and_record "$G0/keep/checkpoint-$s" "$G0/eval_curve_$s" "$CURVE_EPISODES"
      "$PROJECT_PY" -m gr00t_lancedb.flywheel.cli success --rollouts "$G0/eval_curve_$s" | tee "$G0/eval_curve_$s.json"
    done
    ;;
  mine)
    serve_and_record "$G0/keep/checkpoint-$FORK_STEP" "$G0/eval_$FORK_STEP" "$EVAL_EPISODES"
    "$PROJECT_PY" -m gr00t_lancedb.flywheel.cli mine --lance-root "$DATASET" --rollouts "$G0/eval_$FORK_STEP" \
      --weight-column "$WEIGHT_COLUMN" --out-json "$RUNS/mining_report.json" ${MINE_ARGS:-}
    ;;
  g3)
    [ -d "$G0/keep/checkpoint-$FORK_STEP" ] || { echo "missing $G0/keep/checkpoint-$FORK_STEP"; exit 1; }
    mkdir -p "$G3"
    if [ ! -d "$G3/checkpoint-$FORK_STEP" ]; then
      # Hard links: the 24 GB fork point is not copied. HF resumes from the newest checkpoint-*.
      cp -al "$G0/keep/checkpoint-$FORK_STEP" "$G3/checkpoint-$FORK_STEP"
      rm -f "$G3/checkpoint-$FORK_STEP/keep_manifest.json"
    fi
    VERSION="$("$PROJECT_PY" -c "import json;print(json.load(open('$RUNS/mining_report.json'))['write']['frames_version'])")"
    GR00T_LANCE_WEIGHT_COLUMN="$WEIGHT_COLUMN" GR00T_LANCE_WEIGHT_VERSION="$VERSION" \
      GR00T_KEEP_CKPT_WEIGHTS_ONLY="$MAX_STEPS" train "$G3" --resume_from_checkpoint
    ;;
  final)
    serve_and_record "$G0/keep/checkpoint-$MAX_STEPS" "$G0/eval_$MAX_STEPS" "$EVAL_EPISODES"
    serve_and_record "$G3/keep/checkpoint-$MAX_STEPS" "$G3/eval_$MAX_STEPS" "$EVAL_EPISODES"
    "$PROJECT_PY" -m gr00t_lancedb.flywheel.cli compare --baseline "$G0/eval_$MAX_STEPS" \
      --candidate "$G3/eval_$MAX_STEPS" --out-json "$RUNS/paired_eval.json" --out-md "$RUNS/paired_eval.md"
    ;;
  *) echo "unknown STAGE=$STAGE"; exit 2 ;;
esac
