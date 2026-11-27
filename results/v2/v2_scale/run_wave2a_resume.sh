#!/usr/bin/env bash
set -euo pipefail
source /data/env.sh
set -a; source /data/config/secrets/gcs-hmac.env; set +a
export AWS_ENDPOINT="${AWS_ENDPOINT:-$AWS_ENDPOINT_URL}"
export PYTHONPATH="/data/branch-v2/src"
export GR00T_BENCH_SKIP_FINAL_SAVE=1
export RCLONE_CONFIG_GCSS3_TYPE=s3
export RCLONE_CONFIG_GCSS3_PROVIDER=GCS
export RCLONE_CONFIG_GCSS3_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID"
export RCLONE_CONFIG_GCSS3_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY"
export RCLONE_CONFIG_GCSS3_ENDPOINT="$AWS_ENDPOINT_URL"
export RCLONE_BIN=/data/bin/rclone
export SUDO="sudo -E"

S5CMD=/data/Isaac-GR00T/.venv/bin/s5cmd
OUT=/data/runs/v2_scale
cd /data/Isaac-GR00T

echo "[$(date +%H:%M:%S)] === [3/5] P5 Curation benchmark on x10 (6.4 GB) ==="
rm -rf /data/tmp/curation_x10_work
CUDA_VISIBLE_DEVICES="" uv run --no-sync python /data/branch-v2/scripts/bench_curation_e6.py \
  --v2-dir /data/datasets/x_tier/x10/v21 \
  --lance-dir /data/datasets/x_tier/x10/lance_copy \
  --work-dir /data/tmp/curation_x10_work \
  --drop-frac 0.05 --drop-seed 42 --idle-eps 1e-3 \
  --out-json "$OUT/p5_curation_x10.json" --out-md "$OUT/p5_curation_x10.md"
rm -rf /data/tmp/curation_x10_work

echo "[$(date +%H:%M:%S)] === [4/5] P2 Cold-Start on x10 (6.4 GB): A / B / C / D (25 steps each) ==="
rm -rf /data/stage_x10 "$OUT/x10_A_native" "$OUT/x10_B_rclone" "$OUT/x10_C_v2_gcs" "$OUT/x10_D_v2_local"
mkdir -p /data/stage_x10/gen-1
T0=$(date +%s.%N)
$S5CMD --endpoint-url "$AWS_ENDPOINT_URL" --numworkers 128 --log error cp "s3://$BUCKET/x_tier/x10/v21/*" /data/stage_x10/gen-1/
T1=$(date +%s.%N)
DL_BYTES=$(du -sb /data/stage_x10/gen-1 | cut -f1)
python3 -c "import json; json.dump({'data_prep_s': round($T1 - $T0, 3), 'bytes': int($DL_BYTES)}, open('$OUT/x10_A_prep.json', 'w'))"
echo "x10 A downloaded $DL_BYTES bytes in $(python3 -c "print(round($T1-$T0, 2))")s"
BACKEND=native DATASET=/data/stage_x10/gen-1 OUT="$OUT/x10_A_native" MAX_STEPS=25 SAVE_STEPS=1000 \
  bash /data/branch-v2/scripts/run_gr00t_finetune.sh > "$OUT/x10_A_native.log" 2>&1
rm -rf /data/stage_x10

export B_REMOTE="gcss3:$BUCKET/x_tier/x10"
export B_DATASET_SUBDIR="v21"
export B_BASE_DIR=/data/groupb_x10
export B_READ_ONLY=0
export RCLONE_EXTRA_ARGS="--vfs-write-back 1h"
bash /data/branch-v2/scripts/run_group_b_rclone.sh unmount 2>/dev/null || true
sudo rm -rf "$B_BASE_DIR"
T0=$(date +%s.%N)
B_PATH=$(bash /data/branch-v2/scripts/run_group_b_rclone.sh reset)
T1=$(date +%s.%N)
python3 -c "import json; json.dump({'data_prep_s': round($T1 - $T0, 3), 'path': '$B_PATH'}, open('$OUT/x10_B_prep.json', 'w'))"
BACKEND=native DATASET="$B_PATH" OUT="$OUT/x10_B_rclone" MAX_STEPS=25 SAVE_STEPS=1000 \
  bash /data/branch-v2/scripts/run_gr00t_finetune.sh > "$OUT/x10_B_rclone.log" 2>&1
bash /data/branch-v2/scripts/run_group_b_rclone.sh unmount || true
sudo rm -rf "$B_BASE_DIR"

rm -rf "${GR00T_LANCE_CACHE_DIR:-$HOME/.cache/gr00t_lancedb}" /data/tmp/gop_cache_x10 && mkdir -p /data/tmp/gop_cache_x10
S3_ENV_FILE=/data/config/secrets/gcs-hmac.env \
  BACKEND=lance DATASET="s3://$BUCKET/x_tier/x10/lance_copy" \
  GR00T_LANCE_VIDEO_CACHE_GB=20 GR00T_LANCE_VIDEO_CACHE_DIR=/data/tmp/gop_cache_x10 \
  OUT="$OUT/x10_C_v2_gcs" MAX_STEPS=25 SAVE_STEPS=1000 \
  bash /data/branch-v2/scripts/run_gr00t_finetune.sh > "$OUT/x10_C_v2_gcs.log" 2>&1
rm -rf /data/tmp/gop_cache_x10

BACKEND=lance DATASET=/data/datasets/x_tier/x10/lance_copy \
  OUT="$OUT/x10_D_v2_local" MAX_STEPS=25 SAVE_STEPS=1000 \
  bash /data/branch-v2/scripts/run_gr00t_finetune.sh > "$OUT/x10_D_v2_local.log" 2>&1

echo "[$(date +%H:%M:%S)] === [5/5] Build & upload x50 (32 GB) v21 + lance_copy ==="
rm -rf /data/datasets/x_tier/x50
CUDA_VISIBLE_DEVICES="" uv run --no-sync python /data/branch-v2/scripts/build_scale_tier_x.py \
  --src /data/datasets/libero_10 --out-root /data/datasets/x_tier \
  --factors 50 --lance copy --workers 8 --overwrite --manifest /data/datasets/x_tier/manifest_x50.json

echo "[$(date +%H:%M:%S)] Uploading x50 v21 (32 GB) + lance_copy (32 GB) to GCS..."
$RCLONE_BIN copy /data/datasets/x_tier/x50/v21 "gcss3:$BUCKET/x_tier/x50/v21" --transfers 96 --checkers 96 --s3-upload-concurrency 8 --stats 30s
$RCLONE_BIN copy /data/datasets/x_tier/x50/lance_copy "gcss3:$BUCKET/x_tier/x50/lance_copy" --transfers 32 --checkers 32 --s3-upload-concurrency 8 --stats 30s

echo "[$(date +%H:%M:%S)] === WAVE2A_ALL_DONE ==="
df -h /data
