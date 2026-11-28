#!/usr/bin/env bash
# Fail-fast online acceleration benchmark for GR00T N1.7 on LIBERO-10 (NO offline VL-cache).
#
# Preserves 100 % of GR00T's native online pipeline (per-step RandomCrop + ColorJitter, real
# Qwen3-VL padding states, fp32 flow-matching targets/loss) while layering:
#   - Online Stratified Multi-Noise Flow Matching (VLM runs once per B frames, DiT runs K times)
#   - LowMemAdamW (8-bit moments + Kahan bf16 weights) + parameter-only bf16 casting
#     (keeps FP32 RoPE inv_freq intact; targets < 24 GB VRAM for RTX 4090 / single-GPU runs)
#
# Modes (MODE=...):
#   mn2          16 frames x 2 noise = 32 FM samples + LowMemAdamW bf16  [default, safest]
#   mn4           8 frames x 4 noise = 32 FM samples + LowMemAdamW bf16  [fastest VLM cut]
#   lowmem_only  32 frames x 1 noise = 32 FM samples + LowMemAdamW bf16  [isolates 24 GB opt]
#   mn2_fp32     16 frames x 2 noise = 32 FM samples + stock fp32 AdamW  [isolates multi-noise]
# Falsification matrix (docs/plans/07, single-variable steps from B0 to mn2):
#   b16_k1_fp32  16 frames x 1 noise, stock fp32 AdamW, nothing from the turbo path
#                [= stock at batch 16: B0 -> this isolates the batch size]
#   b16_k1       16 frames x 1 noise + LowMemAdamW bf16 (mn2 without multi-noise)
#                [b16_k1_fp32 -> this isolates the optimizer; this -> mn2 isolates K=2]
# Baselines for the comparison table (meant for BENCH_STEPS runs):
#   stock        32 frames, stock fp32 AdamW, nothing from this repo's turbo path          [B0]
#   stock_ga2    16 frames x grad-accum 2, stock fp32 AdamW, frozen VLM cast to bf16
#                ["just lower the batch size and cast the backbone": B1]
#   bnb8bit      16 frames x 2 noise + bitsandbytes AdamW8bit on fp32 weights          [B2]
# FlashAdamW (flashoptim, master_weight_bits=24, keep_norm_fp32=1; flashoptim must be importable):
#   b16_k1_flash   16 frames x 1 noise                    [L4 runs of 2026-11-28, results/l4_24g]
#   mn2_flash      16 frames x 2 noise                    [L4 runs of 2026-11-28]
# Full batch, 32 frames per optimizer step like B0 (docs/plans/08):
#   flash24_b32    32 frames x 1 noise + FlashAdamW-24    [Phase 2A]
#   mn2_b32_flash  32 frames x 2 noise + FlashAdamW-24    [Phase 2B]
#   lowmem_only    32 frames x 1 noise + LowMemAdamW      (listed above; no external dependency)
#   mn2_b32        32 frames x 2 noise + LowMemAdamW
#   mn2_ga2        16 frames x 2 noise x grad-accum 2 = 32 frames/step + FlashAdamW-24
#                  (OPTIMIZER=lowmem for the LowMemAdamW variant) [fallback if 32 x 2 OOMs]
#
# Optimizer-side switches (defaults keep LowMemAdamW runs numerically equal to falsify_07):
#   KEEP_NORM_FP32=0|1  norm parameters stay fp32 under BF16_WEIGHTS (default 1 for FlashAdamW
#                       modes, 0 otherwise)
#   FUSED_CLIP=0|1      clip through LowMemAdamW.clip_scale() (default 0)
#   GRAD_CKPT_HEAD=0|1  activation checkpointing of the DiT blocks (default 0; for 32 x 2 on 24 GB)
#   FLASHOPTIM_BITS     for OPTIMIZER=flashoptim (flash24 / flash32 pin it)
#   TRAIN_SEED          GR00T's data / init / noise seed (config.data.seed); unset = GR00T's 42,
#                       as in every earlier run. SEED only seeds LowMemAdamW's stochastic rounding.
#
# Usage:
#   bash scripts/run_online_turbo.sh
#   MODE=mn4 COMPILE_HEAD=1 bash scripts/run_online_turbo.sh
#   MODE=mn2 BASELINE_EVAL=/data/runs/b0/eval.json bash scripts/run_online_turbo.sh
#   MODE=stock_ga2 BENCH_STEPS=50 bash scripts/run_online_turbo.sh   # speed/memory only
#
# BENCH_STEPS=N  run N steps, save nothing, skip the eval, and write bench_summary.json with
#                the steady step time and peak GPU memory. A CUDA out-of-memory error is
#                recorded as the result ("status": "oom") and the script exits 0.
#                BENCH_TAG=_cap24 appends to the run name (bench_<MODE><TAG>), so a run under
#                GR00T_CUDA_MEM_FRACTION=0.6 (24 GiB of an A100-40GB) and an uncapped run of
#                the same mode do not overwrite each other.
# Eval: N_EPISODES per task (default 20 = 200 episodes), one seeded simulator process per
#       episode, N_ENVS of them in parallel. Use the same EVAL_SEED_BASE for every
#       checkpoint you want to compare; BASELINE_EVAL=<eval.json> prints the paired
#       comparison (difference, 95% CI, exact McNemar) at the end.
set -euo pipefail

[ -f "${GR00T_ENV_FILE:-/data/env.sh}" ] && source "${GR00T_ENV_FILE:-/data/env.sh}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$HERE/.." && pwd)}"
GR00T_ROOT="${GR00T_ROOT:-/data/Isaac-GR00T}"
DATASET="${DATASET:-/data/datasets/libero_10_lance}"
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.7-3B}"
BACKEND="${BACKEND:-lance}"
MODE="${MODE:-mn2}"
MAX_STEPS="${MAX_STEPS:-3000}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OUT_ROOT="${OUT_ROOT:-/data/runs/online_turbo_${MAX_STEPS}s}"
EVAL="${EVAL:-1}"
N_EPISODES="${N_EPISODES:-20}"
N_ENVS="${N_ENVS:-5}"
EVAL_SEED_BASE="${EVAL_SEED_BASE:-20260}"
BASELINE_EVAL="${BASELINE_EVAL:-}"
PORT="${PORT:-5560}"
COMPILE_HEAD="${COMPILE_HEAD:-0}"
SEED="${SEED:-42}"
TRAIN_SEED="${TRAIN_SEED:-}"
FUSED_CLIP="${FUSED_CLIP:-0}"
GRAD_CKPT_HEAD="${GRAD_CKPT_HEAD:-0}"
BENCH_STEPS="${BENCH_STEPS:-0}"
# A bench run is itself short and may be expected to run out of memory: no smoke gate.
if [ "$BENCH_STEPS" != "0" ]; then SMOKE="${SMOKE:-0}"; else SMOKE="${SMOKE:-1}"; fi
ALLOW_PURE_BF16=0
EXTRA_ARGS=()

case "$MODE" in
  mn2)
    GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
    MULTI_NOISE="${MULTI_NOISE:-2}"
    OPTIMIZER="${OPTIMIZER:-lowmem}"
    BF16_WEIGHTS="${BF16_WEIGHTS:-1}"
    BF16_BACKBONE="${BF16_BACKBONE:-1}"
    ;;
  mn4)
    GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
    MULTI_NOISE="${MULTI_NOISE:-4}"
    OPTIMIZER="${OPTIMIZER:-lowmem}"
    BF16_WEIGHTS="${BF16_WEIGHTS:-1}"
    BF16_BACKBONE="${BF16_BACKBONE:-1}"
    ;;
  lowmem_only)
    GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
    MULTI_NOISE="${MULTI_NOISE:-1}"
    OPTIMIZER="${OPTIMIZER:-lowmem}"
    BF16_WEIGHTS="${BF16_WEIGHTS:-1}"
    BF16_BACKBONE="${BF16_BACKBONE:-1}"
    ;;
  mn2_fp32)
    GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
    MULTI_NOISE="${MULTI_NOISE:-2}"
    OPTIMIZER="${OPTIMIZER:-native}"
    BF16_WEIGHTS="${BF16_WEIGHTS:-0}"
    BF16_BACKBONE="${BF16_BACKBONE:-0}"
    ;;
  b16_k1)
    GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
    MULTI_NOISE="${MULTI_NOISE:-1}"
    OPTIMIZER="${OPTIMIZER:-lowmem}"
    BF16_WEIGHTS="${BF16_WEIGHTS:-1}"
    BF16_BACKBONE="${BF16_BACKBONE:-1}"
    ;;
  b16_k1_fp32)
    GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
    MULTI_NOISE="${MULTI_NOISE:-1}"
    OPTIMIZER="${OPTIMIZER:-native}"
    BF16_WEIGHTS="${BF16_WEIGHTS:-0}"
    BF16_BACKBONE="${BF16_BACKBONE:-0}"
    COMPILE_HEAD=0
    ;;
  stock)
    GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
    MULTI_NOISE="${MULTI_NOISE:-1}"
    OPTIMIZER="${OPTIMIZER:-native}"
    BF16_WEIGHTS="${BF16_WEIGHTS:-0}"
    BF16_BACKBONE="${BF16_BACKBONE:-0}"
    COMPILE_HEAD=0
    ;;
  stock_ga2)
    GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
    MULTI_NOISE="${MULTI_NOISE:-1}"
    OPTIMIZER="${OPTIMIZER:-native}"
    BF16_WEIGHTS="${BF16_WEIGHTS:-0}"
    BF16_BACKBONE="${BF16_BACKBONE:-1}"
    EXTRA_ARGS=(--gradient_accumulation_steps "${GRAD_ACCUM:-2}")
    ;;
  bnb8bit)
    GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
    MULTI_NOISE="${MULTI_NOISE:-2}"
    OPTIMIZER="${OPTIMIZER:-bnb8bit}"
    # fp32 weights: what a user gets from optim=adamw_8bit. BF16_WEIGHTS=1 measures the
    # smaller (and non-converging: no carry for sub-ulp updates) pure-bf16 variant.
    BF16_WEIGHTS="${BF16_WEIGHTS:-0}"
    BF16_BACKBONE="${BF16_BACKBONE:-1}"
    [ "$BF16_WEIGHTS" = "1" ] && ALLOW_PURE_BF16=1
    ;;
  b16_k1_flash|mn2_flash|flash24_b32|mn2_b32_flash|mn2_b32|mn2_ga2)
    case "$MODE" in
      b16_k1_flash) B=16; K=1; OPT=flash24 ;;
      mn2_flash) B=16; K=2; OPT=flash24 ;;
      flash24_b32) B=32; K=1; OPT=flash24 ;;
      mn2_b32_flash) B=32; K=2; OPT=flash24 ;;
      mn2_b32) B=32; K=2; OPT=lowmem ;;
      mn2_ga2) B=16; K=2; OPT=flash24; EXTRA_ARGS=(--gradient_accumulation_steps "${GRAD_ACCUM:-2}") ;;
    esac
    GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$B}"
    MULTI_NOISE="${MULTI_NOISE:-$K}"
    OPTIMIZER="${OPTIMIZER:-$OPT}"
    BF16_WEIGHTS="${BF16_WEIGHTS:-1}"
    BF16_BACKBONE="${BF16_BACKBONE:-1}"
    ;;
  *)
    echo "Unknown MODE=$MODE (expected mn2 | mn4 | lowmem_only | mn2_fp32 | b16_k1 | b16_k1_fp32 | stock | stock_ga2 | bnb8bit | b16_k1_flash | mn2_flash | flash24_b32 | mn2_b32_flash | mn2_b32 | mn2_ga2)" >&2
    exit 2
    ;;
esac

case "$OPTIMIZER" in flash*) KEEP_NORM_FP32="${KEEP_NORM_FP32:-1}" ;; *) KEEP_NORM_FP32="${KEEP_NORM_FP32:-0}" ;; esac
GRAD_ACCUM_STEPS=1
if [ "${#EXTRA_ARGS[@]}" -ge 2 ] && [ "${EXTRA_ARGS[0]}" = "--gradient_accumulation_steps" ]; then GRAD_ACCUM_STEPS="${EXTRA_ARGS[1]}"; fi

mkdir -p "$OUT_ROOT"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MUJOCO_GL="${MUJOCO_GL_BACKEND:-egl}"
export PYOPENGL_PLATFORM="${MUJOCO_GL_BACKEND:-egl}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# ---------------------------------------------------------------------------
# Gate 0: Triton kernel parity check (~30 s) BEFORE burning any training GPU
# ---------------------------------------------------------------------------
if [ "$OPTIMIZER" = "lowmem" ] && [ "${SKIP_TRITON_TEST:-0}" != "1" ]; then
  echo "================ [Gate 0] LowMemAdamW Triton parity test ================"
  cd "$GR00T_ROOT"
  uv run --no-sync python -m pytest "$REPO_ROOT/tests/test_turbo_lowmem_adamw.py" -k triton -q
fi

export GR00T_TURBO_MULTI_NOISE="$MULTI_NOISE"
export GR00T_TURBO_STRATIFIED=1
export GR00T_TURBO_OPTIMIZER="$OPTIMIZER"
export GR00T_TURBO_BF16_WEIGHTS="$BF16_WEIGHTS"
export GR00T_TURBO_BF16_BACKBONE="$BF16_BACKBONE"
export GR00T_TURBO_COMPILE_HEAD="$COMPILE_HEAD"
export GR00T_TURBO_KEEP_NORM_FP32="$KEEP_NORM_FP32"
export GR00T_TURBO_FUSED_CLIP="$FUSED_CLIP"
export GR00T_TURBO_GRAD_CKPT_HEAD="$GRAD_CKPT_HEAD"
export GR00T_TURBO_ALLOW_PURE_BF16="$ALLOW_PURE_BF16"
export GR00T_TURBO_SEED="$SEED"
[ -n "${FLASHOPTIM_BITS:-}" ] && export GR00T_TURBO_FLASHOPTIM_BITS="$FLASHOPTIM_BITS"
if [ -n "$TRAIN_SEED" ]; then export GR00T_TRAIN_SEED="$TRAIN_SEED"; else unset GR00T_TRAIN_SEED; fi

CHECK_FLAGS=()
case "$MODE" in stock|b16_k1_fp32) CHECK_FLAGS+=(--no-turbo) ;; esac
[ "$COMPILE_HEAD" = "1" ] && CHECK_FLAGS+=(--expect-compile)

# ---------------------------------------------------------------------------
# Gate 1: 5-step smoke test + checkpoint key & finite-loss verification
# ---------------------------------------------------------------------------
if [ "$SMOKE" = "1" ]; then
  SMOKE_OUT="$OUT_ROOT/_smoke_${MODE}"
  rm -rf "$SMOKE_OUT"
  echo "================ [Gate 1] 5-step smoke test ($MODE: B=$GLOBAL_BATCH_SIZE x K=$MULTI_NOISE, opt=$OPTIMIZER, bf16=$BF16_WEIGHTS) ================"
  BACKEND="$BACKEND" DATASET="$DATASET" OUT="$SMOKE_OUT" BASE_MODEL="$BASE_MODEL" \
    MAX_STEPS=5 SAVE_STEPS=5 SAVE_TOTAL_LIMIT=1 \
    GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" NUM_WORKERS="$NUM_WORKERS" \
    bash "$REPO_ROOT/scripts/run_gr00t_finetune.sh" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} 2>&1 | tee "$SMOKE_OUT.log"

  # Also fails when COMPILE_HEAD=1 did not actually compile (it would silently time the
  # uncompiled head).
  (
    cd "$GR00T_ROOT"
    uv run --no-sync python "$REPO_ROOT/scripts/turbo_run_checks.py" smoke "$SMOKE_OUT" "$SMOKE_OUT.log" \
      ${CHECK_FLAGS[@]+"${CHECK_FLAGS[@]}"}
  )
  rm -rf "$SMOKE_OUT" "$SMOKE_OUT.log"
fi

# ---------------------------------------------------------------------------
# Bench mode: N steps for step time + peak memory, nothing saved, no eval
# ---------------------------------------------------------------------------
if [ "$BENCH_STEPS" != "0" ]; then
  RUN_OUT="$OUT_ROOT/bench_${MODE}${BENCH_TAG:-}"
  rm -rf "$RUN_OUT"
  echo "================ [Bench] $MODE${BENCH_TAG:-}: $BENCH_STEPS steps (B=$GLOBAL_BATCH_SIZE x K=$MULTI_NOISE, opt=$OPTIMIZER, compile=$COMPILE_HEAD, mem_fraction=${GR00T_CUDA_MEM_FRACTION:-none}) ================"
  set +e
  BACKEND="$BACKEND" DATASET="$DATASET" OUT="$RUN_OUT" BASE_MODEL="$BASE_MODEL" \
    MAX_STEPS="$BENCH_STEPS" SAVE_STEPS=1000000 SAVE_TOTAL_LIMIT=1 GR00T_BENCH_SKIP_FINAL_SAVE=1 \
    GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" NUM_WORKERS="$NUM_WORKERS" \
    bash "$REPO_ROOT/scripts/run_gr00t_finetune.sh" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} 2>&1 | tee "$RUN_OUT.train.log"
  set -e
  mkdir -p "$RUN_OUT"
  python3 "$REPO_ROOT/scripts/turbo_run_checks.py" bench "$RUN_OUT" "$RUN_OUT.train.log" \
    --mode "$MODE${BENCH_TAG:-}" --steps "$BENCH_STEPS" --out-json "$RUN_OUT/bench_summary.json"
  exit $?
fi

# ---------------------------------------------------------------------------
# Main Run: full online training (no offline cache)
# ---------------------------------------------------------------------------
RUN_OUT="$OUT_ROOT/$MODE"
CKPT="$RUN_OUT/checkpoint-${MAX_STEPS}"
if [ -d "$CKPT" ] && [ "${RERUN:-0}" != "1" ]; then
  echo "[online-turbo] $CKPT already exists, skipping training (RERUN=1 to redo)"
else
  echo "================ [Main] Online Turbo $MODE ($MAX_STEPS steps, B=$GLOBAL_BATCH_SIZE x K=$MULTI_NOISE = $((GLOBAL_BATCH_SIZE * MULTI_NOISE)) FM samples/step) ================"
  T0=$(date +%s)
  BACKEND="$BACKEND" DATASET="$DATASET" OUT="$RUN_OUT" BASE_MODEL="$BASE_MODEL" \
    MAX_STEPS="$MAX_STEPS" SAVE_STEPS="$MAX_STEPS" SAVE_TOTAL_LIMIT=1 \
    GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" NUM_WORKERS="$NUM_WORKERS" \
    bash "$REPO_ROOT/scripts/run_gr00t_finetune.sh" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} 2>&1 | tee "$RUN_OUT.train.log"
  T1=$(date +%s)
  echo "{\"mode\":\"$MODE\",\"train_wall_s\":$((T1 - T0)),\"global_batch_size\":$GLOBAL_BATCH_SIZE,\"grad_accum\":$GRAD_ACCUM_STEPS,\"multi_noise\":$MULTI_NOISE,\"optimizer\":\"$OPTIMIZER\",\"bf16_weights\":$BF16_WEIGHTS,\"keep_norm_fp32\":$KEEP_NORM_FP32,\"fused_clip\":$FUSED_CLIP,\"grad_ckpt_head\":$GRAD_CKPT_HEAD,\"compile_head\":$COMPILE_HEAD,\"train_seed\":${TRAIN_SEED:-null}}" \
    > "$RUN_OUT/online_turbo_run.json"
fi
[ -d "$CKPT" ] || CKPT="$RUN_OUT"

# ---------------------------------------------------------------------------
# Closed-loop LIBERO-10 Eval (with capped --n-envs to avoid EGL OOM)
# ---------------------------------------------------------------------------
if [ "$EVAL" = "1" ]; then
  echo "================ [Eval] LIBERO-10 closed-loop ($MODE, ${N_EPISODES} ep/task, n_envs=${N_ENVS}) ================"
  cd "$GR00T_ROOT"
  uv run --no-sync python "$REPO_ROOT/scripts/eval_libero10_checkpoint.py" \
    --gr00t-root "$GR00T_ROOT" \
    --model-path "$CKPT" \
    --n-episodes "$N_EPISODES" \
    --n-envs "$N_ENVS" \
    --seed-base "$EVAL_SEED_BASE" \
    --port "$PORT" \
    --out-json "$RUN_OUT/eval.json"

  python3 - "$RUN_OUT" <<'PY'
import json, sys
from pathlib import Path
run_dir = Path(sys.argv[1])
ev = json.loads((run_dir / "eval.json").read_text())
meta = json.loads((run_dir / "online_turbo_run.json").read_text()) if (run_dir / "online_turbo_run.json").is_file() else {}
sr = ev.get("mean_success_rate")
sr_s = f"{100.0 * sr:.2f}%" if sr is not None else "FAILED (null)"
wall_m = f"{meta['train_wall_s'] / 60.0:.1f} min" if "train_wall_s" in meta else "n/a"
ci = ev.get("ci95")
n_s = f" ({ev.get('successes')}/{ev.get('valid_episodes')}, 95% CI {ci})" if ci else ""
print(f"\n[SUMMARY] mode={meta.get('mode', run_dir.name)} | train={wall_m} | LIBERO-10={sr_s}{n_s}")
PY

  if [ -n "$BASELINE_EVAL" ]; then
    python3 "$REPO_ROOT/scripts/compare_libero_evals.py" --baseline "$BASELINE_EVAL" \
      "$MODE=$RUN_OUT/eval.json" --out-json "$RUN_OUT/eval_vs_baseline.json"
  fi
fi
