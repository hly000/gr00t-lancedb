#!/usr/bin/env bash
# One GPU session that separates the candidate causes of the cached-training gap.
# Every config trains the action head from the SAME existing VL cache (no re-materialisation)
# and logs a held-out probe-loss curve; closed-loop eval is optional because it is the slow part.
#
#   CONFIGS      space-separated list, default "fp32_adamw bf16_lowmem"
#                  fp32_adamw     fp32 weights + torch fused AdamW      (accuracy reference, 16 B/param)
#                  bf16_lowmem    bf16 weights + LowMemAdamW            (the 24 GB candidate)
#                  fp32_lowmem    fp32 weights + LowMemAdamW            (isolates the 8-bit moments)
#                  bf16_adamw     bf16 weights + torch fused AdamW      (the first run's failure mode)
#                  bf16_lowmem_mn4  as bf16_lowmem with 8 frames x 4 noise samples per step
#   EVAL=1       also run LIBERO-10 closed-loop for each config (N_EPISODES per task, default 10)
#   NATIVE_CKPT  if set, its probe loss is added to the comparison (e.g. group1 checkpoint-3000)
#
# Budget on an A100 at 3,000 steps: ~9 min of training per compiled config (eager fp32 is slower); closed-loop adds ~16 min per 50 episodes.
# Run first:  pytest tests/test_turbo_lowmem_adamw.py -k triton -q
set -euo pipefail

[ -f /data/env.sh ] && source /data/env.sh

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$HERE/.." && pwd)}"
GR00T_ROOT="${GR00T_ROOT:-/data/Isaac-GR00T}"
DATASET="${DATASET:-/data/datasets/libero_10_lance}"
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.7-3B}"
CACHE_DIR="${CACHE_DIR:-/data/tmp/libero10_vl_cache_full}"
MAX_STEPS="${MAX_STEPS:-3000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
OUT_ROOT="${OUT_ROOT:-/data/runs/cache_ablation_${MAX_STEPS}s}"
CONFIGS="${CONFIGS:-fp32_adamw bf16_lowmem}"
EVAL="${EVAL:-0}"
N_EPISODES="${N_EPISODES:-10}"
NATIVE_CKPT="${NATIVE_CKPT:-}"
SEED="${SEED:-42}"

mkdir -p "$OUT_ROOT"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MUJOCO_GL="${MUJOCO_GL_BACKEND:-egl}" PYOPENGL_PLATFORM="${MUJOCO_GL_BACKEND:-egl}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$GR00T_ROOT"

flags_for() {
  case "$1" in
    # fp32 weights + bf16 autocast crashes PyTorch 2.9's Inductor at step 1, so those run eager.
    # (The trainer also falls back to eager by itself if a compiled dry run fails.)
    fp32_adamw)      echo "--fp32-master-weights --optimizer adamw" ;;
    bf16_lowmem)     echo "--no-fp32-master-weights --optimizer lowmem --compile" ;;
    fp32_lowmem)     echo "--fp32-master-weights --optimizer lowmem" ;;
    bf16_adamw)      echo "--no-fp32-master-weights --optimizer adamw --compile" ;;
    bf16_lowmem_mn4) echo "--no-fp32-master-weights --optimizer lowmem --multi-noise 4 --compile" ;;
    *) echo "unknown config $1" >&2; exit 2 ;;
  esac
}

PROBE_ARGS=()
[ -n "$NATIVE_CKPT" ] && PROBE_ARGS+=(--ckpt "native=$NATIVE_CKPT")

port=5560
for cfg in $CONFIGS; do
  out="$OUT_ROOT/$cfg"
  if [ -f "$out/train_summary.json" ] && [ "${RERUN:-0}" != 1 ]; then
    echo "[ablation] $cfg: train_summary.json exists, skipping (RERUN=1 to redo)"
  else
    echo "================ [ablation] train $cfg ================"
    # shellcheck disable=SC2046
    uv run --no-sync python "$REPO_ROOT/scripts/train_vl_cache_e2e.py" \
      --gr00t-root "$GR00T_ROOT" --base-model-path "$BASE_MODEL" --dataset-path "$DATASET" \
      --embodiment-tag LIBERO_PANDA --cache-dir "$CACHE_DIR" --output-dir "$out" \
      --batch-size "$BATCH_SIZE" --max-steps "$MAX_STEPS" --num-workers 4 --seed "$SEED" \
      $(flags_for "$cfg") 2>&1 | tee "$out.train.log"
  fi
  PROBE_ARGS+=(--ckpt "$cfg=$out/merged_gr00t_ckpt")
  if [ "$EVAL" = 1 ] && { [ ! -f "$out/eval.json" ] || [ "${RERUN:-0}" = 1 ]; }; then
    echo "================ [ablation] closed-loop $cfg ================"
    uv run --no-sync python "$REPO_ROOT/scripts/eval_libero10_checkpoint.py" \
      --gr00t-root "$GR00T_ROOT" --model-path "$out/merged_gr00t_ckpt" \
      --n-episodes "$N_EPISODES" --n-envs "$N_EPISODES" --port "$port" --out-json "$out/eval.json"
  fi
  port=$((port + 1))
done

echo "================ [ablation] paired probe comparison ================"
uv run --no-sync python "$REPO_ROOT/scripts/eval_cached_fm_loss.py" \
  --gr00t-root "$GR00T_ROOT" --cache-dir "$CACHE_DIR" "${PROBE_ARGS[@]}" --out-json "$OUT_ROOT/probe_compare.json"

python3 "$REPO_ROOT/scripts/summarize_cache_ablation.py" "$OUT_ROOT" $CONFIGS | tee "$OUT_ROOT/summary.md"
