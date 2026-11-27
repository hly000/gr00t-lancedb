#!/usr/bin/env bash
# End-to-end 2-group closed-loop comparison on LIBERO-10:
#   Group 1: Native Online GR00T N1.7 (official launch_finetune_lance.py)
#   Group 2: Lance VL-Cache (per_token_int8 -> ActionHead-Only + Fused AdamW + torch.compile
#            + FeatureSpaceAugmenter -> lossless export to standard GR00T checkpoint)
#   Eval:    Both checkpoints evaluated on all 10 LIBERO-10 tasks in closed-loop MuJoCo sim!
set -euo pipefail

[ -f /data/env.sh ] && source /data/env.sh

GR00T_ROOT="${GR00T_ROOT:-/data/Isaac-GR00T}"
REPO_ROOT="${REPO_ROOT:-/data/branch-vl-cache}"
DATASET="${DATASET:-/data/datasets/libero_10_lance}"
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.7-3B}"
MAX_STEPS="${MAX_STEPS:-3000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
N_EPISODES="${N_EPISODES:-5}"
OUT_ROOT="${OUT_ROOT:-/data/runs/e2e_ablation_${MAX_STEPS}s}"
CACHE_DIR="${CACHE_DIR:-/data/tmp/libero10_vl_cache_full}"
MUJOCO_GL_BACKEND="${MUJOCO_GL_BACKEND:-egl}"

mkdir -p "$OUT_ROOT"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MUJOCO_GL="$MUJOCO_GL_BACKEND"
export PYOPENGL_PLATFORM="$MUJOCO_GL_BACKEND"

cd "$GR00T_ROOT"

echo "================================================================"
echo "[E2E] Group 2: Materialize + Lance VL-Cache ActionHead-Only Train"
echo "================================================================"
T_G2_START=$(date +%s)
PYTHONPATH="$REPO_ROOT/src" uv run --no-sync python "$REPO_ROOT/scripts/train_vl_cache_e2e.py" \
  --gr00t-root "$GR00T_ROOT" \
  --base-model-path "$BASE_MODEL" \
  --dataset-path "$DATASET" \
  --embodiment-tag LIBERO_PANDA \
  --cache-dir "$CACHE_DIR" \
  --output-dir "$OUT_ROOT/group2_vl_cache" \
  --codec per_token_int8 \
  --mat-batch-size 64 \
  --batch-size "$BATCH_SIZE" \
  --multi-noise 1 \
  --max-steps "$MAX_STEPS" \
  --token-dropout 0.05 \
  --gaussian-jitter 0.01 \
  --compile \
  --num-workers 4 \
  --seed 42
T_G2_END=$(date +%s)
echo "[E2E] Group 2 total wall: $((T_G2_END - T_G2_START))s"

echo "================================================================"
echo "[E2E] Eval Group 2 (Lance VL-Cache) on LIBERO-10 (${N_EPISODES} ep/task)"
echo "================================================================"
PYTHONPATH="$REPO_ROOT/src" uv run --no-sync python "$REPO_ROOT/scripts/eval_libero10_checkpoint.py" \
  --gr00t-root "$GR00T_ROOT" \
  --model-path "$OUT_ROOT/group2_vl_cache/merged_gr00t_ckpt" \
  --n-episodes "$N_EPISODES" \
  --n-envs "$N_EPISODES" \
  --port 5556 \
  --out-json "$OUT_ROOT/eval_group2_vl_cache.json"

echo "================================================================"
echo "[E2E] Group 1: Native Online GR00T N1.7 Baseline (${MAX_STEPS} steps)"
echo "================================================================"
T_G1_START=$(date +%s)
BACKEND=lance DATASET="$DATASET" OUT="$OUT_ROOT/group1_native_online" \
  MAX_STEPS="$MAX_STEPS" SAVE_STEPS="$MAX_STEPS" GLOBAL_BATCH_SIZE="$BATCH_SIZE" \
  NUM_WORKERS=4 SAVE_TOTAL_LIMIT=1 \
  bash "$REPO_ROOT/scripts/run_gr00t_finetune.sh"
T_G1_END=$(date +%s)
echo "[E2E] Group 1 total wall: $((T_G1_END - T_G1_START))s"

G1_CKPT="$OUT_ROOT/group1_native_online/checkpoint-${MAX_STEPS}"
if [ ! -d "$G1_CKPT" ]; then
  G1_CKPT="$OUT_ROOT/group1_native_online"
fi

echo "================================================================"
echo "[E2E] Eval Group 1 (Native Online) on LIBERO-10 (${N_EPISODES} ep/task)"
echo "================================================================"
PYTHONPATH="$REPO_ROOT/src" uv run --no-sync python "$REPO_ROOT/scripts/eval_libero10_checkpoint.py" \
  --gr00t-root "$GR00T_ROOT" \
  --model-path "$G1_CKPT" \
  --n-episodes "$N_EPISODES" \
  --n-envs "$N_EPISODES" \
  --port 5557 \
  --out-json "$OUT_ROOT/eval_group1_native.json"

echo "================================================================"
echo "[E2E] FINAL COMPARISON SUMMARY"
echo "================================================================"
python3 - "$OUT_ROOT" "$((T_G1_END - T_G1_START))" "$((T_G2_END - T_G2_START))" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
g1_wall, g2_wall = int(sys.argv[2]), int(sys.argv[3])
g2_tr = json.loads((root / "group2_vl_cache/train_summary.json").read_text())
e1 = json.loads((root / "eval_group1_native.json").read_text())
e2 = json.loads((root / "eval_group2_vl_cache.json").read_text())

summary = {
    "group1_native_online": {
        "train_wall_s": g1_wall,
        "libero10_mean_success_rate": e1["mean_success_rate"],
        "eval_episodes": e1["total_episodes"],
    },
    "group2_vl_cache": {
        "materialize_s": g2_tr["materialize_s"],
        "train_loop_s": g2_tr["train_loop_s"],
        "total_wall_s": g2_wall,
        "median_step_ms": g2_tr["median_step_ms"],
        "peak_reserved_gb": g2_tr["peak_reserved_gb"],
        "libero10_mean_success_rate": e2["mean_success_rate"],
        "eval_episodes": e2["total_episodes"],
    },
    "speedup_train_only_x": round(g1_wall / max(1.0, g2_tr["train_loop_s"]), 2),
}
out = root / "final_e2e_summary.json"
out.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY
