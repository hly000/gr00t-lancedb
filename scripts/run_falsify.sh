#!/usr/bin/env bash
# docs/plans/07 (Exp-Falsify) on the GPU box, one step per call:
#
#   STEP=1 bash scripts/run_falsify.sh        optimizer microbench + 50-step memory matrix (24 GiB cap)
#   STEP=2 bash scripts/run_falsify.sh        train G_b16_lowmem  (MODE=b16_k1, 3000 steps)
#   STEP=3 bash scripts/run_falsify.sh        train G_b16_fp32    (MODE=b16_k1_fp32, 3000 steps)
#   STEP=4 bash scripts/run_falsify.sh        paired seeded LIBERO-10 evals of B0, G_b16_fp32, G_b16_lowmem, mn2
#   STEP=verdict bash scripts/run_falsify.sh  apply the pre-registered rules (no GPU)
#
# Two code trees, so the new models are trained by exactly the code that trained B0 and mn2:
#   REPO_ROOT  src/ and tests/ byte-identical to feat/online-turbo @ 93abd3a, scripts/ current.
#              Every training run (Steps 1-3) uses it, with the per-tensor LowMemAdamW launch.
#   WIP_ROOT   the current tree (fused LowMemAdamW, bitsandbytes optimizer). Used only by
#              Step 1 measurements that need it: the optimizer microbench and MODE=bnb8bit.
set -euo pipefail

[ -f "${GR00T_ENV_FILE:-/data/env.sh}" ] && source "${GR00T_ENV_FILE:-/data/env.sh}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$HERE/.." && pwd)}"
WIP_ROOT="${WIP_ROOT:-}"
GR00T_ROOT="${GR00T_ROOT:-/data/Isaac-GR00T}"
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.7-3B}"
STEP="${STEP:?set STEP=1|2|3|4|verdict}"
FALSIFY_ROOT="${FALSIFY_ROOT:-/data/runs/falsify_07}"
TRAIN_ROOT="${TRAIN_ROOT:-/data/runs/online_turbo_3000s}"
MAX_STEPS="${MAX_STEPS:-3000}"
B0_CKPT="${B0_CKPT:-/data/runs/e2e_ablation_3000s/group1_native_online/checkpoint-3000}"
MN2_CKPT="${MN2_CKPT:-$TRAIN_ROOT/mn2/checkpoint-$MAX_STEPS}"
BENCH_STEPS="${BENCH_STEPS:-50}"
MEM_FRACTION="${MEM_FRACTION:-0.6}"
CAP_MODES="${CAP_MODES:-stock stock_ga2 b16_k1_fp32 b16_k1 mn2}"
N_EPISODES="${N_EPISODES:-10}"
N_ENVS="${N_ENVS:-5}"
EVAL_SEED_BASE="${EVAL_SEED_BASE:-20260}"
EVAL_PARALLEL="${EVAL_PARALLEL:-2}"
EVAL_BASE_PORT="${EVAL_BASE_PORT:-5561}"
EVAL_ONLY="${EVAL_ONLY:-}"   # e.g. "b0 mn2": evaluate a subset in Step 4

EVAL_DIR="$FALSIFY_ROOT/eval"
STEP1_DIR="$FALSIFY_ROOT/step1"
mkdir -p "$FALSIFY_ROOT"

# The rules compare against the code that trained mn2; refuse to train from a tree whose
# optimizer code is not that one (the fused launch only exists after 93abd3a).
assert_frozen_tree() {
  if grep -q "lowmem_fused" "$REPO_ROOT/src/gr00t_lancedb/turbo/online_turbo.py"; then
    echo "FAIL: $REPO_ROOT/src is not the frozen 93abd3a training code (has lowmem_fused)." >&2
    echo "      Training for Exp-Falsify must use src/ and tests/ from 93abd3a." >&2
    exit 2
  fi
}

run_mode() {  # run_online_turbo.sh from a given tree, everything else from the environment
  local root="$1"; shift
  env REPO_ROOT="$root" LOWMEM_FUSED=off COMPILE_HEAD=0 "$@" bash "$root/scripts/run_online_turbo.sh"
}

case "$STEP" in
  1)
    assert_frozen_tree
    mkdir -p "$STEP1_DIR"
    nvidia-smi | tee "$STEP1_DIR/nvidia_smi.txt"
    if [ -n "$WIP_ROOT" ]; then
      echo "================ [Step 1a] fused LowMemAdamW parity on GPU (WIP tree; informational) ================"
      (cd "$GR00T_ROOT" && PYTHONPATH="$WIP_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
        uv run --no-sync python -m pytest "$WIP_ROOT/tests/test_turbo_lowmem_adamw.py" -k triton -q) \
        2>&1 | tee "$STEP1_DIR/wip_gate0.txt" || echo "[falsify] WIP Gate 0 failed (does not affect the frozen runs)"
      echo "================ [Step 1b] optimizer microbench, 980 real action-head tensors ================"
      # BENCH_PYTHONPATH: e.g. a --target dir holding torchao, kept out of the training runs.
      (cd "$GR00T_ROOT" && PYTHONPATH="${BENCH_PYTHONPATH:-}${PYTHONPATH:+:$PYTHONPATH}" \
        uv run --no-sync python "$WIP_ROOT/scripts/bench_lowmem_adamw.py" \
        --base-model-path "$BASE_MODEL" --gr00t-root "$GR00T_ROOT" --out-json "$STEP1_DIR/optim_bench.json") \
        2>&1 | tee "$STEP1_DIR/optim_bench.txt" || echo "[falsify] microbench ended with an error"
    else
      echo "[falsify] WIP_ROOT unset: skipping the optimizer microbench and MODE=bnb8bit"
    fi

    echo "================ [Step 1c] Gate 0 on the frozen tree ================"
    (cd "$GR00T_ROOT" && PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
      uv run --no-sync python -m pytest "$REPO_ROOT/tests/test_turbo_lowmem_adamw.py" -k triton -q)

    NAMES=()
    for MODE in $CAP_MODES; do
      run_mode "$REPO_ROOT" MODE="$MODE" BENCH_STEPS="$BENCH_STEPS" BENCH_TAG=_cap24 \
        GR00T_CUDA_MEM_FRACTION="$MEM_FRACTION" SKIP_TRITON_TEST=1 EVAL=0 OUT_ROOT="$STEP1_DIR" \
        || echo "[falsify] $MODE (capped) ended with a non-OOM error"
      NAMES+=("${MODE}_cap24")
    done
    if [ -n "$WIP_ROOT" ]; then
      run_mode "$WIP_ROOT" MODE=bnb8bit BENCH_STEPS="$BENCH_STEPS" BENCH_TAG=_cap24 \
        GR00T_CUDA_MEM_FRACTION="$MEM_FRACTION" SKIP_TRITON_TEST=1 EVAL=0 OUT_ROOT="$STEP1_DIR" \
        || echo "[falsify] bnb8bit (capped) ended with a non-OOM error"
      NAMES+=(bnb8bit_cap24)
    fi
    # Whatever ran out of memory under the cap: once more without it, for its real peak.
    for NAME in "${NAMES[@]}"; do
      MODE="${NAME%_cap24}"
      STATUS="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status"))' \
        "$STEP1_DIR/bench_${NAME}/bench_summary.json" 2>/dev/null || echo missing)"
      if [ "$STATUS" = "oom" ]; then
        ROOT="$REPO_ROOT"; [ "$MODE" = "bnb8bit" ] && ROOT="$WIP_ROOT"
        run_mode "$ROOT" MODE="$MODE" BENCH_STEPS="$BENCH_STEPS" BENCH_TAG=_nocap \
          SKIP_TRITON_TEST=1 EVAL=0 OUT_ROOT="$STEP1_DIR" || echo "[falsify] $MODE (uncapped) ended with an error"
        NAMES+=("${MODE}_nocap")
      fi
    done
    GPU_NAME="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
    python3 "$REPO_ROOT/scripts/turbo_run_checks.py" table "$STEP1_DIR" "${NAMES[@]}" \
      --gpu "$GPU_NAME" --out-json "$STEP1_DIR/proof_24gb.json" | tee "$STEP1_DIR/proof_24gb.md"
    ;;

  2|3)
    assert_frozen_tree
    MODE=b16_k1; NEED_GB=20
    [ "$STEP" = "3" ] && { MODE=b16_k1_fp32; NEED_GB=32; }  # fp32 checkpoint: weights + AdamW state
    FREE_GB=$(( $(df -Pk "$TRAIN_ROOT" 2>/dev/null | awk 'NR==2 {print $4}' || echo 0) / 1024 / 1024 ))
    if [ "$FREE_GB" -lt "$NEED_GB" ]; then
      echo "FAIL: $FREE_GB GB free under $TRAIN_ROOT, $MODE needs ~$NEED_GB GB for its checkpoint" >&2
      exit 2
    fi
    # The eval reads checkpoint-$MAX_STEPS; the final save_model() copy is skipped to save disk.
    run_mode "$REPO_ROOT" MODE="$MODE" MAX_STEPS="$MAX_STEPS" EVAL=0 OUT_ROOT="$TRAIN_ROOT" \
      GR00T_SKIP_FINAL_MODEL_SAVE=1
    test -d "$TRAIN_ROOT/$MODE/checkpoint-$MAX_STEPS"
    ;;

  4)
    mkdir -p "$EVAL_DIR"
    EVALS=(
      "b0=$B0_CKPT"
      "mn2=$MN2_CKPT"
      "b16_lowmem=$TRAIN_ROOT/b16_k1/checkpoint-$MAX_STEPS"
      "b16_fp32=$TRAIN_ROOT/b16_k1_fp32/checkpoint-$MAX_STEPS"
    )
    for spec in "${EVALS[@]}"; do
      [ -d "${spec#*=}" ] || { echo "FAIL: checkpoint ${spec#*=} not found" >&2; exit 2; }
    done
    ( while true; do echo "$(date +%T) $(cut -d' ' -f1-3 /proc/loadavg) $(nproc)cpu" >> "$EVAL_DIR/load.log"; sleep 60; done ) &
    LOAD_PID=$!
    trap 'kill $LOAD_PID 2>/dev/null || true' EXIT
    run_eval() {
      local name="$1" ckpt="$2" port="$3"
      if [ -f "$EVAL_DIR/$name.json" ] && [ "${RERUN_EVAL:-0}" != "1" ]; then
        echo "[falsify] $EVAL_DIR/$name.json exists, skipping (RERUN_EVAL=1 to redo)"; return 0
      fi
      echo "[falsify] eval $name ($ckpt) on port $port"
      (cd "$GR00T_ROOT" && uv run --no-sync python "$REPO_ROOT/scripts/eval_libero10_checkpoint.py" \
        --gr00t-root "$GR00T_ROOT" --model-path "$ckpt" --n-episodes "$N_EPISODES" --n-envs "$N_ENVS" \
        --seed-base "$EVAL_SEED_BASE" --port "$port" --out-json "$EVAL_DIR/$name.json") \
        > "$EVAL_DIR/$name.log" 2>&1
    }
    # FIFO: a port is reused only after the eval that held it has finished.
    PIDS=(); NAMES=(); i=0; FAILED=()
    for spec in "${EVALS[@]}"; do
      name="${spec%%=*}"
      if [ -n "$EVAL_ONLY" ] && [[ " $EVAL_ONLY " != *" $name "* ]]; then continue; fi
      run_eval "$name" "${spec#*=}" $((EVAL_BASE_PORT + i % EVAL_PARALLEL)) &
      PIDS+=($!); NAMES+=("$name"); i=$((i + 1))
      if [ "${#PIDS[@]}" -ge "$EVAL_PARALLEL" ]; then
        wait "${PIDS[0]}" || FAILED+=("${NAMES[0]}")
        PIDS=("${PIDS[@]:1}"); NAMES=("${NAMES[@]:1}")
      fi
    done
    for j in ${PIDS[@]+"${!PIDS[@]}"}; do wait "${PIDS[$j]}" || FAILED+=("${NAMES[$j]}"); done
    [ "${#FAILED[@]}" -eq 0 ] || { echo "[falsify] evals that did not complete: ${FAILED[*]} (see $EVAL_DIR/<name>.log)"; exit 3; }
    ;;

  verdict)
    PROOF=()
    [ -f "$STEP1_DIR/proof_24gb.json" ] && PROOF=(--proof-json "$STEP1_DIR/proof_24gb.json")
    python3 "$REPO_ROOT/scripts/falsify_verdict.py" \
      --b0 "$EVAL_DIR/b0.json" --b16-fp32 "$EVAL_DIR/b16_fp32.json" \
      --b16-lowmem "$EVAL_DIR/b16_lowmem.json" --mn2 "$EVAL_DIR/mn2.json" \
      ${PROOF[@]+"${PROOF[@]}"} --out-json "$FALSIFY_ROOT/verdict.json" --out-md "$FALSIFY_ROOT/verdict.md"
    ;;

  *)
    echo "Unknown STEP=$STEP (expected 1 | 2 | 3 | 4 | verdict)" >&2
    exit 2
    ;;
esac
