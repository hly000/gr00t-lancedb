#!/usr/bin/env bash
set -euo pipefail
source /data/env.sh
set -a; source /data/config/secrets/gcs-hmac.env; set +a
export AWS_ENDPOINT="${AWS_ENDPOINT:-$AWS_ENDPOINT_URL}"
export PYTHONPATH="/data/branch-v2/src"
export GR00T_BENCH_SKIP_FINAL_SAVE=1
export GR00T_SAMPLE_FINGERPRINTS=1

cd /data/Isaac-GR00T
OUT_BASE=/data/runs/v2_libero10

echo "[$(date +%H:%M:%S)] === [3/7] group_b_rclone_s (B: rclone mount cold VFS + native GR00T, 60 steps) ==="
export RCLONE_CONFIG_GCSS3_TYPE=s3
export RCLONE_CONFIG_GCSS3_PROVIDER=GCS
export RCLONE_CONFIG_GCSS3_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID"
export RCLONE_CONFIG_GCSS3_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY"
export RCLONE_CONFIG_GCSS3_ENDPOINT="$AWS_ENDPOINT_URL"
export RCLONE_BIN=/data/bin/rclone
export SUDO="sudo -E"
export B_REMOTE="gcss3:$BUCKET"
export B_DATASET_SUBDIR="libero_10"
export B_BASE_DIR=/data/groupb_s
export B_READ_ONLY=0
export RCLONE_EXTRA_ARGS="--vfs-write-back 1h"
sudo umount -l /data/groupb_s/mnt/gen-1 2>/dev/null || true
sudo pkill -f "rclone mount.*groupb_s" 2>/dev/null || true
sleep 1
sudo rm -rf "$B_BASE_DIR"
T_MNT0=$(date +%s.%N)
B_PATH=$(bash /data/branch-v2/scripts/run_group_b_rclone.sh reset)
T_MNT1=$(date +%s.%N)
python3 -c "import json; json.dump({'mount_s': round($T_MNT1 - $T_MNT0, 3), 'path': '$B_PATH'}, open('$OUT_BASE/group_b_rclone_s_mount.json', 'w'))"
echo "Mounted B at $B_PATH"
BACKEND=native DATASET="$B_PATH" \
  OUT="$OUT_BASE/group_b_rclone_s" MAX_STEPS=60 SAVE_STEPS=1000 \
  bash /data/branch-v2/scripts/run_gr00t_finetune.sh > "$OUT_BASE/group_b_rclone_s.log" 2>&1
bash /data/branch-v2/scripts/run_group_b_rclone.sh unmount || true

echo "[$(date +%H:%M:%S)] === [4/7] v2_filtered (P5: curated Lance + where=\"is_valid = true\" + fork, 60 steps) ==="
rm -rf /data/home/.cache/gr00t_lancedb/where
BACKEND=lance DATASET=/data/datasets/libero_10_lance_curated \
  GR00T_LANCE_WHERE="is_valid = true" \
  OUT="$OUT_BASE/v2_filtered" MAX_STEPS=60 SAVE_STEPS=1000 \
  bash /data/branch-v2/scripts/run_gr00t_finetune.sh > "$OUT_BASE/v2_filtered.log" 2>&1

echo "[$(date +%H:%M:%S)] === [5/7] P4a v2_preempt_skip (80 steps, save@30, SIGTERM@45: calibrated cost-gate + os._exit) ==="
BACKEND=lance DATASET=/data/datasets/libero_10_lance \
  GR00T_PREEMPT_CKPT_DTYPE=auto GR00T_PREEMPT_FAST_EXIT=1 \
  OUT="$OUT_BASE/v2_preempt_skip" MAX_STEPS=80 SAVE_STEPS=30 KILL_AT_STEP=45 SKIP_REF=1 \
  bash /data/branch-v2/scripts/preempt_experiment.sh > "$OUT_BASE/v2_preempt_skip.log" 2>&1

echo "[$(date +%H:%M:%S)] === [6/7] P4a v2_preempt_auto_save (80 steps, save@30, SIGTERM@45: DTYPE=auto <=25s + payload release + os._exit) ==="
BACKEND=lance DATASET=/data/datasets/libero_10_lance \
  GR00T_PREEMPT_COST_GATE=0 GR00T_PREEMPT_CKPT_DTYPE=auto GR00T_PREEMPT_DEADLINE_S=25 GR00T_PREEMPT_FAST_EXIT=1 \
  OUT="$OUT_BASE/v2_preempt_auto_save" MAX_STEPS=80 SAVE_STEPS=30 KILL_AT_STEP=45 SKIP_REF=1 \
  bash /data/branch-v2/scripts/preempt_experiment.sh > "$OUT_BASE/v2_preempt_auto_save.log" 2>&1

echo "[$(date +%H:%M:%S)] === [7/7] P1 CPU-only multi-epoch GCS loader I/O bench (v1-like vs v2 GOP cache) ==="
rm -rf /data/tmp/p1_gop_cache && mkdir -p /data/tmp/p1_gop_cache
CUDA_VISIBLE_DEVICES="" uv run --no-sync python /data/branch-v2/scripts/bench_loader_io.py \
  --dataset "s3://$BUCKET/libero_10_lance" \
  --storage-option "aws_endpoint=$AWS_ENDPOINT_URL" \
  --storage-option "aws_access_key_id=$AWS_ACCESS_KEY_ID" \
  --storage-option "aws_secret_access_key=$AWS_SECRET_ACCESS_KEY" \
  --storage-option "aws_region=${AWS_REGION:-us-central1}" \
  --embodiment-tag libero_sim --epochs 2 --batches-per-epoch 250 --batch-size 32 --num-workers 4 \
  --first-decode-batch 32 --steady-decode-batch 32 --decoder-cache 16 --video-cache-gb 0 \
  --out "$OUT_BASE/p1_io_v1_nocache.json" > "$OUT_BASE/p1_io_v1_nocache.log" 2>&1

CUDA_VISIBLE_DEVICES="" uv run --no-sync python /data/branch-v2/scripts/bench_loader_io.py \
  --dataset "s3://$BUCKET/libero_10_lance" \
  --storage-option "aws_endpoint=$AWS_ENDPOINT_URL" \
  --storage-option "aws_access_key_id=$AWS_ACCESS_KEY_ID" \
  --storage-option "aws_secret_access_key=$AWS_SECRET_ACCESS_KEY" \
  --storage-option "aws_region=${AWS_REGION:-us-central1}" \
  --embodiment-tag libero_sim --epochs 2 --batches-per-epoch 250 --batch-size 32 --num-workers 4 \
  --first-decode-batch 32 --steady-decode-batch 128 --decoder-cache 64 \
  --video-cache-gb 20 --video-cache-dir /data/tmp/p1_gop_cache \
  --out "$OUT_BASE/p1_io_v2_gopcache.json" > "$OUT_BASE/p1_io_v2_gopcache.log" 2>&1

echo "[$(date +%H:%M:%S)] === WAVE1_ALL_DONE ==="
