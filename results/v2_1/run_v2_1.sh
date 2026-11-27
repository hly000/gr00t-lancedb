#!/usr/bin/env bash
# v2.1 rerun on the A100 box: same configurations as results/v2, with stock GR00T rerun in
# the same session (v2 reused an older native number for the 0.64 GB tier).
#
#   WAVES="s x10 preempt" bash run_v2_1.sh        # default
#   WAVES="x50" bash run_v2_1.sh                  # 32 GB cold start (uploads x50 first)
#
# Every run is a fresh process. Lance "cold" runs delete the local meta/ cache, the where
# cache and the GOP cache first. Checkpoints are deleted after each preemption run (24 GB each).
set -euo pipefail
source /data/env.sh
set -a; source /data/config/secrets/gcs-hmac.env; set +a
export AWS_ENDPOINT="${AWS_ENDPOINT:-$AWS_ENDPOINT_URL}"
REPO="${REPO:-/data/branch-v2_1}"
export PYTHONPATH="$REPO/src"
export GR00T_BENCH_SKIP_FINAL_SAVE=1
export GR00T_SAMPLE_FINGERPRINTS=1
export RCLONE_CONFIG_GCSS3_TYPE=s3
export RCLONE_CONFIG_GCSS3_PROVIDER=GCS
export RCLONE_CONFIG_GCSS3_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID"
export RCLONE_CONFIG_GCSS3_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY"
export RCLONE_CONFIG_GCSS3_ENDPOINT="$AWS_ENDPOINT_URL"
export RCLONE_BIN=/data/bin/rclone
export SUDO="sudo -E"
export B_READ_ONLY=0
export RCLONE_EXTRA_ARGS="--vfs-write-back 1h"

S5CMD=/data/Isaac-GR00T/.venv/bin/s5cmd
OUT="${OUT:-/data/runs/v2_1}"
WAVES="${WAVES:-s x10 preempt}"
META_CACHE=/data/cache/hf/lerobot/remote
mkdir -p "$OUT"
cd /data/Isaac-GR00T

say() { echo "[$(date -u +%H:%M:%S)] === $* ==="; }
now() { date +%s.%N; }
finetune() { bash "$REPO/scripts/run_gr00t_finetune.sh"; }
# Wall time of the whole launcher process, including interpreter teardown after the last step.
timed_run() {  # timed_run <name>   (env carries the run configuration)
  local name="$1" t0 t1
  t0=$(now)
  OUT="$OUT/$name" finetune > "$OUT/$name.log" 2>&1 || echo "run $name exited with $?"
  t1=$(now)
  python3 -c "import json; json.dump({'process_wall_s': round($t1 - $t0, 3), 't_start': $t0, 't_end': $t1}, open('$OUT/${name}_wall.json', 'w'))"
}
prep_json() { python3 -c "import json,sys; json.dump({'data_prep_s': round(float(sys.argv[2]) - float(sys.argv[1]), 3), **dict(a.split('=', 1) for a in sys.argv[4:])}, open(sys.argv[3], 'w'))" "$@"; }
cold_lance() {  # cold_lance <meta-cache-glob> <gop-dir>
  rm -rf "$META_CACHE"/$1 "${GR00T_LANCE_CACHE_DIR:-$HOME/.cache/gr00t_lancedb}" "$2"
  mkdir -p "$2"
}
download() {  # download <bucket-prefix> <dest> <prep-json>
  rm -rf "$2" && mkdir -p "$2"
  local t0 t1; t0=$(now)
  $S5CMD --endpoint-url "$AWS_ENDPOINT_URL" --numworkers 128 --log error cp "s3://$BUCKET/$1/*" "$2/"
  t1=$(now)
  prep_json "$t0" "$t1" "$3" "bytes=$(du -sb "$2" | cut -f1)"
}
mount_b() {  # mount_b <prep-json>; needs B_REMOTE / B_DATASET_SUBDIR / B_BASE_DIR; prints the dataset path
  bash "$REPO/scripts/run_group_b_rclone.sh" unmount >/dev/null 2>&1 || true
  sudo rm -rf "$B_BASE_DIR"
  local t0 t1 path; t0=$(now)
  path=$(bash "$REPO/scripts/run_group_b_rclone.sh" reset)
  t1=$(now)
  prep_json "$t0" "$t1" "$1" "path=$path"
  echo "$path"
}
unmount_b() { bash "$REPO/scripts/run_group_b_rclone.sh" unmount >/dev/null 2>&1 || true; sudo rm -rf "$B_BASE_DIR"; }

# Four ways to read the same data, each from a "new machine" state.
tier() {  # tier <tag> <steps> <bucket-v21> <bucket-lance> <local-lance> <meta-cache-glob> <b-remote-suffix> <b-subdir>
  local tag="$1" steps="$2" v21="$3" lance="$4" local_lance="$5" glob="$6" bsuf="$7" bsub="$8" path
  say "$tag A: s5cmd download + stock GR00T ($steps steps)"
  download "$v21" "/data/stage_$tag/gen-1" "$OUT/${tag}_A_prep.json"
  BACKEND=native DATASET="/data/stage_$tag/gen-1" MAX_STEPS=$steps SAVE_STEPS=100000 timed_run "${tag}_A_native"
  rm -rf "/data/stage_$tag"

  say "$tag B: rclone cold mount + stock GR00T ($steps steps)"
  # Exported here, not inside mount_b: that runs in a subshell and unmount_b needs them too.
  export B_REMOTE="gcss3:$BUCKET$bsuf" B_DATASET_SUBDIR="$bsub" B_BASE_DIR="/data/groupb_$tag"
  path=$(mount_b "$OUT/${tag}_B_prep.json")
  BACKEND=native DATASET="$path" MAX_STEPS=$steps SAVE_STEPS=100000 timed_run "${tag}_B_rclone"
  unmount_b

  say "$tag C: Lance read straight from the bucket, cold caches ($steps steps)"
  cold_lance "$glob" "/data/tmp/gop_cache_$tag"
  S3_ENV_FILE=/data/config/secrets/gcs-hmac.env BACKEND=lance DATASET="s3://$BUCKET/$lance" \
    GR00T_LANCE_VIDEO_CACHE_GB=20 GR00T_LANCE_VIDEO_CACHE_DIR="/data/tmp/gop_cache_$tag" \
    MAX_STEPS=$steps SAVE_STEPS=100000 timed_run "${tag}_C_lance_gcs"
  rm -rf "/data/tmp/gop_cache_$tag"

  say "$tag D: Lance on local disk ($steps steps)"
  BACKEND=lance DATASET="$local_lance" MAX_STEPS=$steps SAVE_STEPS=100000 timed_run "${tag}_D_lance_local"
}

preempt() {  # preempt <name> <backend> <dataset>   (extra env carries the policy)
  local name="$1"
  say "preempt $name (80 steps, save@30, SIGTERM@45)"
  BACKEND="$2" DATASET="$3" OUT="$OUT/$name" MAX_STEPS=80 SAVE_STEPS=30 KILL_AT_STEP=45 SKIP_REF=1 \
    bash "$REPO/scripts/preempt_experiment.sh" > "$OUT/$name.log" 2>&1 || echo "preempt $name exited with $?"
  du -sh "$OUT/$name"/pre/checkpoint-* 2>/dev/null | sed 's/^/  kept until now: /' || true
  rm -rf "$OUT/$name"/pre/checkpoint-* "$OUT/$name"/pre/.tmp-checkpoint-*
}

for wave in $WAVES; do
  case "$wave" in
    s)
      tier s 60 libero_10 libero_10_lance /data/datasets/libero_10_lance 's3_*_libero_10_lance*' "" libero_10
      say "s C warm: same run again with meta/ and GOP caches kept"
      mkdir -p /data/tmp/gop_cache_s_warm
      S3_ENV_FILE=/data/config/secrets/gcs-hmac.env BACKEND=lance DATASET="s3://$BUCKET/libero_10_lance" \
        GR00T_LANCE_VIDEO_CACHE_GB=20 GR00T_LANCE_VIDEO_CACHE_DIR=/data/tmp/gop_cache_s_warm \
        MAX_STEPS=60 SAVE_STEPS=100000 timed_run s_C_lance_gcs_warm
      rm -rf /data/tmp/gop_cache_s_warm
      say "s D without background import (A/B for the import contention)"
      BACKEND=lance DATASET=/data/datasets/libero_10_lance GR00T_LANCE_BG_IMPORT=0 GR00T_LANCE_PRELOAD_BEFORE_FORK=0 \
        MAX_STEPS=60 SAVE_STEPS=100000 timed_run s_D_lance_local_nobg
      say "s filtered: curated Lance + where (cold where cache)"
      rm -rf "${GR00T_LANCE_CACHE_DIR:-$HOME/.cache/gr00t_lancedb}/where"
      BACKEND=lance DATASET=/data/datasets/libero_10_lance_curated GR00T_LANCE_WHERE="is_valid = true" \
        MAX_STEPS=60 SAVE_STEPS=100000 timed_run s_filtered
      ;;
    x10)
      tier x10 25 x_tier/x10/v21 x_tier/x10/lance_copy /data/datasets/x_tier/x10/lance_copy \
        's3_*_x_tier_x10_lance_copy*' /x_tier/x10 v21
      ;;
    x50)
      say "x50: upload v21 + lance_copy (rclone skips what is already there)"
      $RCLONE_BIN copy /data/datasets/x_tier/x50/v21 "gcss3:$BUCKET/x_tier/x50/v21" --transfers 96 --checkers 96 --s3-upload-concurrency 8 --stats 60s
      $RCLONE_BIN copy /data/datasets/x_tier/x50/lance_copy "gcss3:$BUCKET/x_tier/x50/lance_copy" --transfers 32 --checkers 32 --s3-upload-concurrency 8 --stats 60s
      tier x50 25 x_tier/x50/v21 x_tier/x50/lance_copy /data/datasets/x_tier/x50/lance_copy \
        's3_*_x_tier_x50_lance_copy*' /x_tier/x50 v21
      ;;
    preempt)
      preempt native_pre native /data/datasets/libero_10
      GR00T_PREEMPT_CKPT_DTYPE=auto GR00T_PREEMPT_FAST_EXIT=1 \
        preempt lance_skip lance /data/datasets/libero_10_lance
      GR00T_PREEMPT_COST_GATE=0 GR00T_PREEMPT_CKPT_DTYPE=auto GR00T_PREEMPT_DEADLINE_S=25 GR00T_PREEMPT_FAST_EXIT=1 \
        preempt lance_auto_save lance /data/datasets/libero_10_lance
      ;;
    *) echo "unknown wave: $wave" >&2; exit 2 ;;
  esac
done
say "V2_1_DONE ($WAVES)"
df -h /data | tail -1
