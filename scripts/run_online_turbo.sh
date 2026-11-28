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
#
# Usage:
#   bash scripts/run_online_turbo.sh
#   MODE=mn4 EVAL=1 N_EPISODES=10 bash scripts/run_online_turbo.sh
#   MODE=mn2_fp32 COMPILE_HEAD=0 bash scripts/run_online_turbo.sh
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
SMOKE="${SMOKE:-1}"
EVAL="${EVAL:-1}"
N_EPISODES="${N_EPISODES:-5}"
N_ENVS="${N_ENVS:-5}"
PORT="${PORT:-5560}"
COMPILE_HEAD="${COMPILE_HEAD:-0}"
SEED="${SEED:-42}"

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
  *)
    echo "Unknown MODE=$MODE (expected mn2 | mn4 | lowmem_only | mn2_fp32)" >&2
    exit 2
    ;;
esac

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
export GR00T_TURBO_SEED="$SEED"

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
    bash "$REPO_ROOT/scripts/run_gr00t_finetune.sh"

  (
    cd "$GR00T_ROOT"
    uv run --no-sync python - "$SMOKE_OUT" <<'PY'
import json, math, sys
from pathlib import Path
from safetensors import safe_open

smoke_dir = Path(sys.argv[1])
metrics_path = smoke_dir / "bench_metrics.jsonl"
if not metrics_path.is_file():
    raise SystemExit(f"FAIL: missing {metrics_path}")
rows = [json.loads(l) for l in metrics_path.read_text().splitlines() if l.strip()]
step_rows = [r for r in rows if r.get("event") == "step"]
if len(step_rows) < 5:
    raise SystemExit(f"FAIL: expected 5 step rows, got {len(step_rows)}")
for r in step_rows:
    loss = r.get("step_loss")
    if loss is None or not math.isfinite(float(loss)):
        raise SystemExit(f"FAIL: non-finite loss at step {r.get('global_step')}: {loss}")
turbo_records = [r["online_turbo"] for r in rows if isinstance(r.get("online_turbo"), dict)]
if not turbo_records or not turbo_records[-1].get("enabled"):
    raise SystemExit(f"FAIL: online_turbo was not active in smoke run: {turbo_records}")
ckpt = smoke_dir / "checkpoint-5"
if not ckpt.is_dir():
    ckpt = smoke_dir
st_files = sorted(ckpt.glob("*.safetensors"))
if not st_files:
    raise SystemExit(f"FAIL: no safetensors found in {ckpt}")
bad_keys = []
for sf in st_files:
    with safe_open(str(sf), framework="pt", device="cpu") as fh:
        bad_keys.extend(k for k in fh.keys() if "_orig_mod" in k)
if bad_keys:
    raise SystemExit(f"FAIL: compiled _orig_mod keys leaked into checkpoint: {bad_keys[:5]}")
losses = [round(float(r["step_loss"]), 4) for r in step_rows]
mems = [r.get("gpu_mem_mb") for r in step_rows if r.get("gpu_mem_mb") is not None]
peak_mb = max(mems) if mems else None
fits_24g = (f"{peak_mb:.0f} MB ({'<=24GB OK' if peak_mb <= 24576 else '>24GB'})") if peak_mb is not None else "n/a"
print(f"[Gate 1 PASS] losses={losses} peak_gpu={fits_24g} safetensors={len(st_files)} (0 _orig_mod keys) turbo={turbo_records[-1]}")
PY
  )
  rm -rf "$SMOKE_OUT"
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
    bash "$REPO_ROOT/scripts/run_gr00t_finetune.sh" 2>&1 | tee "$RUN_OUT.train.log"
  T1=$(date +%s)
  echo "{\"mode\":\"$MODE\",\"train_wall_s\":$((T1 - T0)),\"global_batch_size\":$GLOBAL_BATCH_SIZE,\"multi_noise\":$MULTI_NOISE,\"optimizer\":\"$OPTIMIZER\",\"bf16_weights\":$BF16_WEIGHTS,\"compile_head\":$COMPILE_HEAD}" \
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
print(f"\n[SUMMARY] mode={meta.get('mode', run_dir.name)} | train={wall_m} | LIBERO-10={sr_s} (vs Native Group 1: 38.3 min, 37.0 GB, 81.67%)")
PY
fi
