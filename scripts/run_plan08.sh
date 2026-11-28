#!/usr/bin/env bash
# docs/plans/08 (24 GB full batch) on the GPU box, one step per call:
#
#   STEP=probe  bash scripts/run_plan08.sh           Phase 1: optional A100 FlashAdamW microbench
#                                                    + 50-step probes under a 22.1 GiB cap + gate G2
#   STEP=train  MODE=flash24_b32 bash scripts/run_plan08.sh   3000 steps under the same cap
#   STEP=eval   MODE=flash24_b32 bash scripts/run_plan08.sh   100 seeded episodes (pairs with falsify_07)
#   STEP=gate   MODE=flash24_b32 bash scripts/run_plan08.sh   Gate 2A vs B0 (+ VS="name=eval.json ...")
#
# FlashAdamW modes need flashoptim importable: FLASH_PYTHONPATH (a --target dir holding only
# flashoptim) is prepended to PYTHONPATH for them and for the microbench, nothing else.
set -euo pipefail

[ -f "${GR00T_ENV_FILE:-/data/env.sh}" ] && source "${GR00T_ENV_FILE:-/data/env.sh}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$HERE/.." && pwd)}"
GR00T_ROOT="${GR00T_ROOT:-/data/Isaac-GR00T}"
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.7-3B}"
STEP="${STEP:?set STEP=probe|train|eval|gate}"
PLAN_ROOT="${PLAN_ROOT:-/data/runs/plan08}"
TRAIN_ROOT="${TRAIN_ROOT:-/data/runs/online_turbo_3000s}"
MAX_STEPS="${MAX_STEPS:-3000}"
MEM_FRACTION="${MEM_FRACTION:-0.56}"         # 22.1 GiB of an A100-40GB (a real L4 has 22.03 GiB)
PROBE_MODES="${PROBE_MODES:-flash24_b32 mn2_b32_flash flash24_b32:compile lowmem_only}"
PROBE_STEPS="${PROBE_STEPS:-50}"
OPTIM_BENCH="${OPTIM_BENCH:-1}"
FLASH_PYTHONPATH="${FLASH_PYTHONPATH:-}"
B0_EVAL="${B0_EVAL:-/data/runs/falsify_07/eval/b0.json}"
N_EPISODES="${N_EPISODES:-10}"
N_ENVS="${N_ENVS:-5}"
EVAL_SEED_BASE="${EVAL_SEED_BASE:-20260}"
PORT="${PORT:-5571}"
VS="${VS:-}"
PROBE_DIR="$PLAN_ROOT/probe"
EVAL_DIR="$PLAN_ROOT/eval"
mkdir -p "$PLAN_ROOT"

pythonpath_for() {  # flashoptim only where a FlashAdamW optimizer is built (never an empty entry)
  local parts=()
  case "$1" in *flash*|mn2_ga2) [ -n "$FLASH_PYTHONPATH" ] && parts+=("$FLASH_PYTHONPATH") ;; esac
  [ -n "${PYTHONPATH:-}" ] && parts+=("$PYTHONPATH")
  local IFS=:
  echo "${parts[*]:-}"
}

run_mode() {  # MODE first, then KEY=VALUE overrides for run_online_turbo.sh
  local mode="$1"; shift
  env REPO_ROOT="$REPO_ROOT" MODE="$mode" PYTHONPATH="$(pythonpath_for "$mode")" "$@" \
    bash "$REPO_ROOT/scripts/run_online_turbo.sh"
}

probe() {  # probe MODE TAG [KEY=VALUE ...]
  local mode="$1" tag="$2"; shift 2
  run_mode "$mode" BENCH_STEPS="$PROBE_STEPS" BENCH_TAG="$tag" GR00T_CUDA_MEM_FRACTION="$MEM_FRACTION" \
    SKIP_TRITON_TEST=1 EVAL=0 OUT_ROOT="$PROBE_DIR" "$@" || echo "[plan08] $mode$tag ended with a non-OOM error"
  NAMES+=("$mode$tag")
}

status_of() {
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status"))' "$PROBE_DIR/bench_$1/bench_summary.json" 2>/dev/null || echo missing
}

case "$STEP" in
  probe)
    mkdir -p "$PROBE_DIR"
    nvidia-smi | tee "$PROBE_DIR/nvidia_smi.txt"
    if [ "$OPTIM_BENCH" = "1" ]; then
      echo "================ [Phase 1A] FlashAdamW vs LowMemAdamW microbench (537 real tensors) ================"
      (cd "$GR00T_ROOT" && PYTHONPATH="$(pythonpath_for flash)" uv run --no-sync python "$REPO_ROOT/scripts/bench_lowmem_adamw.py" \
        --base-model-path "$BASE_MODEL" --gr00t-root "$GR00T_ROOT" --kinds flash_adamw_24,flash_adamw_32,lowmem_bf16 \
        --gate-g1 --out-json "$PROBE_DIR/optim_bench_a100.json") 2>&1 | tee "$PROBE_DIR/optim_bench_a100.txt" \
        || echo "[plan08] microbench ended with an error"
    fi
    echo "================ [Phase 1A'] LowMemAdamW Gate 0 ================"
    (cd "$GR00T_ROOT" && PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
      uv run --no-sync python -m pytest "$REPO_ROOT/tests/test_turbo_lowmem_adamw.py" -k triton -q)
    NAMES=()
    for spec in $PROBE_MODES; do
      mode="${spec%%:*}"
      if [ "$spec" != "$mode" ] && [ "${spec#*:}" = "compile" ]; then
        probe "$mode" _cap22_compile COMPILE_HEAD=1
      else
        probe "$mode" _cap22
      fi
    done
    # Fallback of the plan: 16 x 2 x grad-accum 2 only when 32 x 1 does not fit.
    if [ "$(status_of flash24_b32_cap22)" != "ok" ]; then
      probe mn2_ga2 _cap22
    fi
    GPU_NAME="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
    python3 "$REPO_ROOT/scripts/turbo_run_checks.py" table "$PROBE_DIR" "${NAMES[@]}" \
      --gpu "$GPU_NAME" --out-json "$PROBE_DIR/proof_fullbatch.json" | tee "$PROBE_DIR/proof_fullbatch.md"
    python3 "$REPO_ROOT/scripts/plan08_gates.py" g2 --proof "$PROBE_DIR/proof_fullbatch.json" \
      --out-json "$PROBE_DIR/gate_g2.json"
    ;;

  train)
    MODE="${MODE:?set MODE}"
    FREE_GB=$(( $(df -Pk "$TRAIN_ROOT" | awk 'NR==2 {print $4}') / 1024 / 1024 ))
    if [ "$FREE_GB" -lt "${NEED_GB:-20}" ]; then
      echo "FAIL: $FREE_GB GB free under $TRAIN_ROOT, need ~${NEED_GB:-20} GB for the checkpoint" >&2
      exit 2
    fi
    # Trained under the same cap the probes used: a run that fits only on a 40 GB card fails here.
    run_mode "$MODE" MAX_STEPS="$MAX_STEPS" EVAL=0 OUT_ROOT="$TRAIN_ROOT" GR00T_SKIP_FINAL_MODEL_SAVE=1 \
      GR00T_CUDA_MEM_FRACTION="${TRAIN_MEM_FRACTION:-$MEM_FRACTION}"
    test -d "$TRAIN_ROOT/$MODE/checkpoint-$MAX_STEPS"
    ;;

  eval)
    MODE="${MODE:?set MODE}"
    CKPT="${CKPT:-$TRAIN_ROOT/$MODE/checkpoint-$MAX_STEPS}"
    [ -d "$CKPT" ] || { echo "FAIL: checkpoint $CKPT not found" >&2; exit 2; }
    mkdir -p "$EVAL_DIR"
    (cd "$GR00T_ROOT" && uv run --no-sync python "$REPO_ROOT/scripts/eval_libero10_checkpoint.py" \
      --gr00t-root "$GR00T_ROOT" --model-path "$CKPT" --n-episodes "$N_EPISODES" --n-envs "$N_ENVS" \
      --seed-base "$EVAL_SEED_BASE" --port "$PORT" --out-json "$EVAL_DIR/$MODE.json") 2>&1 | tee "$EVAL_DIR/$MODE.log"
    ;;

  gate)
    MODE="${MODE:?set MODE}"
    VS_ARGS=()
    for spec in $VS; do VS_ARGS+=(--vs "$spec"); done
    python3 "$REPO_ROOT/scripts/plan08_gates.py" run --name "$MODE" --eval "$EVAL_DIR/$MODE.json" --b0 "$B0_EVAL" \
      --metrics "$TRAIN_ROOT/$MODE/bench_metrics.jsonl" --max-steps "$MAX_STEPS" ${VS_ARGS[@]+"${VS_ARGS[@]}"} \
      --out-json "$PLAN_ROOT/gate_$MODE.json" --out-md "$PLAN_ROOT/gate_$MODE.md"
    ;;

  *)
    echo "Unknown STEP=$STEP (expected probe | train | eval | gate)" >&2
    exit 2
    ;;
esac
