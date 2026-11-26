#!/usr/bin/env bash
# Single-GPU GR00T N1.7 finetune with the same flags as Isaac-GR00T's examples/finetune.sh,
# switchable between GR00T's native data loader and the gr00t-lancedb backend.
#
#   BACKEND=native DATASET=/data/datasets/libero_10      OUT=/data/runs/native bash scripts/run_gr00t_finetune.sh
#   BACKEND=lance  DATASET=/data/datasets/libero_10_lance OUT=/data/runs/lance  bash scripts/run_gr00t_finetune.sh
#
# Extra launch_finetune.py flags can be appended after the script name.
set -euo pipefail

GR00T_ROOT="${GR00T_ROOT:-/data/Isaac-GR00T}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="${BACKEND:-native}"
DATASET="${DATASET:?set DATASET}"
OUT="${OUT:?set OUT}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-LIBERO_PANDA}"
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.7-3B}"
MAX_STEPS="${MAX_STEPS:-200}"
SAVE_STEPS="${SAVE_STEPS:-100}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED_ARGS=()

# Box-local cache/env settings (HF_HOME, UV_CACHE_DIR, ...), if present.
[ -f "${GR00T_ENV_FILE:-/data/env.sh}" ] && source "${GR00T_ENV_FILE:-/data/env.sh}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export GR00T_DATA_BACKEND="$BACKEND"
# GR00T calls the Hub API (model_info) while building the backbone, so offline mode is not
# usable; both backends pay the same small Hub round-trips at startup.

cd "$GR00T_ROOT"
exec uv run --no-sync python "$HERE/launch_finetune_lance.py" --gr00t-root "$GR00T_ROOT" -- \
    --base_model_path "$BASE_MODEL" \
    --dataset_path "$DATASET" \
    --embodiment_tag "$EMBODIMENT_TAG" \
    --num_gpus 1 \
    --output_dir "$OUT" \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit 5 \
    --max_steps "$MAX_STEPS" \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate 1e-4 \
    --global_batch_size "$GLOBAL_BATCH_SIZE" \
    --dataloader_num_workers "$NUM_WORKERS" \
    --shard_size 1024 \
    --num_shards_per_epoch 100000 \
    --episode_sampling_rate 0.1 \
    --state_dropout_prob 0.2 \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    "${SEED_ARGS[@]}" "$@"
