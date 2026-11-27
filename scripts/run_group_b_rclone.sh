#!/usr/bin/env bash
# Group B (mount + local cache): rclone mount of the bucket with a full VFS cache, read by
# GR00T's native loader. Equivalent to SkyPilot MOUNT_CACHED.
#
#   run_group_b_rclone.sh mount     mount a fresh generation, print the dataset path (stdout)
#   run_group_b_rclone.sh reset     "new instance": unmount, delete the VFS cache, mount again
#                                   under a NEW directory name, print the new dataset path
#   run_group_b_rclone.sh unmount   unmount the current generation (VFS cache is kept)
#   run_group_b_rclone.sh status    show generation, mount state, cache size
#
# Verified recipe (gVisor, rclone v1.68.2, docs/experiment-design-v1.md section 2): mount as
# root with --allow-other --uid --gid; a non-root mount fails ("not allowed on 9p"). The page
# cache cannot be dropped in gVisor, so every reset uses new mount + cache directory names
# (gen-<n>); files read after a reset have never been in any cache.
#
# Only the dataset path goes to stdout, so callers can do
#   DATASET=$(bash scripts/run_group_b_rclone.sh mount)
# A stable symlink $B_BASE_DIR/current -> the current mount (+ subdir) is also maintained, so
# a fixed --dataset-path survives resets (handy for preempt_schedule_runner.py).
#
# Config (environment):
#   B_REMOTE            rclone remote, e.g. "gcs:my-bucket/datasets" (required for mount/reset)
#   B_DATASET_SUBDIR    path inside the remote to hand to GR00T, e.g. "libero_10" (default "")
#   B_BASE_DIR          root for mount dirs + state (default /data/groupb)
#   B_CACHE_ROOT        root for per-generation VFS caches (default $B_BASE_DIR/vfs-cache)
#   B_VFS_CACHE_MAX     --vfs-cache-max-size (default 150G)
#   B_UID / B_GID       owner shown inside the mount (default: $SUDO_UID/$SUDO_GID or caller)
#   B_MOUNT_TIMEOUT     seconds to wait for the mount to appear (default 60)
#   RCLONE_BIN          rclone binary (default rclone); RCLONE_CONFIG is passed through
#   RCLONE_EXTRA_ARGS   extra flags for "rclone mount" (word-split)
#   B_REUSE_CACHE=1     on mount, move the previous generation's VFS cache to the new one
#                       (warm-cache remount); reset always deletes the cache
#   SUDO                prefix for privileged commands (default "sudo" unless already root)
#   DRY_RUN=1           print the commands (to stderr) instead of running them; state is
#                       read but never written
set -euo pipefail

CMD="${1:-}"
B_BASE_DIR="${B_BASE_DIR:-/data/groupb}"
B_CACHE_ROOT="${B_CACHE_ROOT:-$B_BASE_DIR/vfs-cache}"
B_DATASET_SUBDIR="${B_DATASET_SUBDIR:-}"
B_VFS_CACHE_MAX="${B_VFS_CACHE_MAX:-150G}"
B_MOUNT_TIMEOUT="${B_MOUNT_TIMEOUT:-60}"
RCLONE_BIN="${RCLONE_BIN:-rclone}"
RCLONE_EXTRA_ARGS="${RCLONE_EXTRA_ARGS:-}"
DRY_RUN="${DRY_RUN:-0}"
if [ -z "${SUDO+x}" ]; then
  if [ "$(id -u)" = "0" ]; then SUDO=""; else SUDO="sudo"; fi
fi
B_UID="${B_UID:-${SUDO_UID:-$(id -u)}}"
B_GID="${B_GID:-${SUDO_GID:-$(id -g)}}"
STATE_DIR="$B_BASE_DIR/state"
GEN_FILE="$STATE_DIR/gen"

log() { echo "[group-b] $*" >&2; }
die() { echo "[group-b] error: $*" >&2; exit 1; }

# Run (or, with DRY_RUN=1, print) a command.
run() {
  if [ "$DRY_RUN" = "1" ]; then
    echo "+ $*" >&2
  else
    "$@"
  fi
}

cur_gen() {
  if [ -f "$GEN_FILE" ]; then
    tr -dc '0-9' < "$GEN_FILE"
  else
    echo ""
  fi
}
mnt_dir() { echo "$B_BASE_DIR/mnt/gen-$1"; }
cache_dir() { echo "$B_CACHE_ROOT/gen-$1"; }
pid_file() { echo "$STATE_DIR/gen-$1.pid"; }
log_file() { echo "$B_BASE_DIR/logs/rclone-gen-$1.log"; }
dataset_path() {
  if [ -n "$B_DATASET_SUBDIR" ]; then echo "$(mnt_dir "$1")/$B_DATASET_SUBDIR"; else mnt_dir "$1"; fi
}

is_mounted() {
  # /proc/mounts works in gVisor and on regular Linux; `mount` is the fallback elsewhere.
  if [ -r /proc/mounts ]; then
    grep -qs " $1 " /proc/mounts
  else
    mount 2>/dev/null | grep -q " on $1 "
  fi
}

write_state() {
  # $1 = gen. Skipped in dry-run so DRY_RUN never changes anything on disk.
  [ "$DRY_RUN" = "1" ] && return 0
  mkdir -p "$STATE_DIR"
  echo "$1" > "$GEN_FILE"
  ln -sfn "$(dataset_path "$1")" "$B_BASE_DIR/current"
}

do_mount() {
  [ -n "${B_REMOTE:-}" ] || die "set B_REMOTE (e.g. gcs:my-bucket/datasets)"
  local gen cur mnt cache
  cur="$(cur_gen)"
  if [ -n "$cur" ] && [ "$DRY_RUN" != "1" ] && is_mounted "$(mnt_dir "$cur")"; then
    log "generation $cur already mounted"
    dataset_path "$cur"
    return 0
  fi
  if [ -n "$cur" ]; then gen=$((cur + 1)); else gen=1; fi
  mnt="$(mnt_dir "$gen")"
  cache="$(cache_dir "$gen")"
  if [ "${B_REUSE_CACHE:-0}" = "1" ] && [ -n "$cur" ] && [ -d "$(cache_dir "$cur")" ]; then
    # Warm-cache remount (E2 "cache warm"): carry the previous generation's cache over.
    run $SUDO mv "$(cache_dir "$cur")" "$cache"
  fi
  run mkdir -p "$B_BASE_DIR" "$STATE_DIR" "$B_BASE_DIR/logs"
  run $SUDO mkdir -p "$mnt" "$cache"
  local ro_flag=""
  if [ "${B_READ_ONLY:-0}" = "1" ]; then ro_flag="--read-only"; fi
  local cmd="$SUDO $RCLONE_BIN mount $B_REMOTE $mnt $ro_flag --allow-other --uid $B_UID --gid $B_GID --vfs-cache-mode full --cache-dir $cache --vfs-cache-max-size $B_VFS_CACHE_MAX --log-file $(log_file "$gen") --log-level INFO $RCLONE_EXTRA_ARGS"
  if [ "$DRY_RUN" = "1" ]; then
    echo "+ $cmd &" >&2
  else
    # shellcheck disable=SC2086
    nohup $cmd > /dev/null 2>&1 &
    local pid=$!
    mkdir -p "$STATE_DIR"
    echo "$pid" > "$(pid_file "$gen")"
    local t0 waited
    t0=$(date +%s)
    while ! is_mounted "$mnt"; do
      kill -0 "$pid" 2>/dev/null || die "rclone exited before mounting; see $(log_file "$gen")"
      waited=$(( $(date +%s) - t0 ))
      [ "$waited" -lt "$B_MOUNT_TIMEOUT" ] || die "mount did not appear within ${B_MOUNT_TIMEOUT}s"
      sleep 0.2
    done
    log "mounted gen $gen at $mnt in $(( $(date +%s) - t0 ))s (pid $pid)"
  fi
  write_state "$gen"
  dataset_path "$gen"
}

do_unmount() {
  # $1 = gen to unmount
  local gen="$1" mnt pidf pid i
  mnt="$(mnt_dir "$gen")"
  pidf="$(pid_file "$gen")"
  if [ "$DRY_RUN" = "1" ] || is_mounted "$mnt"; then
    if [ "$DRY_RUN" = "1" ]; then
      echo "+ $SUDO umount $mnt || $SUDO umount -l $mnt" >&2
    else
      $SUDO umount "$mnt" 2>/dev/null || $SUDO umount -l "$mnt"
    fi
  fi
  if [ "$DRY_RUN" != "1" ] && [ -f "$pidf" ]; then
    pid="$(cat "$pidf")"
    for i in $(seq 1 50); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.2
    done
    if kill -0 "$pid" 2>/dev/null; then
      log "rclone pid $pid still alive after unmount; killing"
      $SUDO kill "$pid" 2>/dev/null || true
    fi
    rm -f "$pidf"
  fi
  if [ "$DRY_RUN" = "1" ]; then
    echo "+ $SUDO rmdir $mnt" >&2
  else
    $SUDO rmdir "$mnt" 2>/dev/null || true
  fi
}

cmd_unmount() {
  local cur
  cur="$(cur_gen)"
  [ -n "$cur" ] || { log "nothing mounted (no state in $STATE_DIR)"; return 0; }
  do_unmount "$cur"
  log "unmounted gen $cur (cache kept at $(cache_dir "$cur"))"
}

cmd_reset() {
  local cur t0
  t0=$(date +%s)
  cur="$(cur_gen)"
  if [ -n "$cur" ]; then
    do_unmount "$cur"
    run $SUDO rm -rf "$(cache_dir "$cur")"
    log "deleted VFS cache of gen $cur"
  fi
  if [ "${B_RESET_REMOUNT:-1}" = "1" ]; then
    do_mount
  fi
  log "reset done in $(( $(date +%s) - t0 ))s"
}

cmd_status() {
  local cur mnt cache
  cur="$(cur_gen)"
  if [ -z "$cur" ]; then
    echo "generation: none"
    return 0
  fi
  mnt="$(mnt_dir "$cur")"
  cache="$(cache_dir "$cur")"
  echo "generation: $cur"
  echo "dataset_path: $(dataset_path "$cur")"
  if is_mounted "$mnt"; then echo "mounted: yes"; else echo "mounted: no"; fi
  if [ -f "$(pid_file "$cur")" ]; then echo "rclone_pid: $(cat "$(pid_file "$cur")")"; fi
  if [ -d "$cache" ]; then echo "cache: $cache ($(du -sh "$cache" 2>/dev/null | cut -f1))"; else echo "cache: $cache (absent)"; fi
}

case "$CMD" in
  mount) do_mount ;;
  reset) cmd_reset ;;
  unmount|umount) cmd_unmount ;;
  status) cmd_status ;;
  *)
    sed -n '2,13p' "$0" >&2
    exit 2
    ;;
esac
