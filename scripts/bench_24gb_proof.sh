#!/usr/bin/env bash
# "Does it fit a 24 GB card?" in one run: short training runs of the stock configurations and
# the Online Turbo modes on THIS machine's GPU, with the outcome of each (runs / CUDA OOM),
# peak GPU memory and steady step time collected into one table.
#
#   bash scripts/bench_24gb_proof.sh                       # mn2 mn4 stock stock_ga2, 60 steps each
#   MODES="mn2 stock" STEPS=100 bash scripts/bench_24gb_proof.sh
#
# Meant for a 24 GB card (L4, L40 24 GB slice, RTX 3090/4090). On a larger card every mode
# runs and the table shows the peak memory each one needs instead of an OOM. An out-of-memory
# error is the expected result for the stock modes on 24 GB and does not fail this script; any
# other error does.
#
# Writes $OUT_ROOT/proof_24gb.{md,json}, nvidia_smi.txt and one bench_<mode>.train.log per mode
# (the OOM traceback is in there: that is the screenshot).
set -euo pipefail

[ -f "${GR00T_ENV_FILE:-/data/env.sh}" ] && source "${GR00T_ENV_FILE:-/data/env.sh}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$HERE/.." && pwd)}"
GR00T_ROOT="${GR00T_ROOT:-/data/Isaac-GR00T}"
STEPS="${STEPS:-60}"
MODES="${MODES:-mn2 mn4 stock stock_ga2}"
OUT_ROOT="${OUT_ROOT:-/data/runs/proof_24gb}"
COMPILE_HEAD="${COMPILE_HEAD:-1}"
export OUT_ROOT

mkdir -p "$OUT_ROOT"
nvidia-smi | tee "$OUT_ROOT/nvidia_smi.txt"
GPU_NAME="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
GPU_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)"
if [ "$GPU_MB" -gt 26000 ]; then
  echo "[proof] NOTE: this GPU has ${GPU_MB} MB, not 24 GB: nothing will OOM here; the table shows peak memory only."
fi

# One kernel parity check up front instead of one per mode.
if [ "${SKIP_TRITON_TEST:-0}" != "1" ]; then
  echo "================ [Gate 0] LowMemAdamW Triton parity test ================"
  (cd "$GR00T_ROOT" && PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    uv run --no-sync python -m pytest "$REPO_ROOT/tests/test_turbo_lowmem_adamw.py" -k triton -q)
fi

for MODE in $MODES; do
  # Stock modes never compile (run_online_turbo.sh forces it off for `stock`).
  case "$MODE" in stock*|bnb8bit|b16_k1_fp32) CH=0 ;; *) CH="$COMPILE_HEAD" ;; esac
  MODE="$MODE" BENCH_STEPS="$STEPS" COMPILE_HEAD="$CH" SKIP_TRITON_TEST=1 EVAL=0 \
    bash "$REPO_ROOT/scripts/run_online_turbo.sh" || echo "[proof] $MODE ended with a non-OOM error (see $OUT_ROOT/bench_${MODE}.train.log)"
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv >> "$OUT_ROOT/nvidia_smi.txt"
done

echo "================ [Proof] ${GPU_NAME}, ${STEPS} steps per mode ================"
# shellcheck disable=SC2086
python3 "$REPO_ROOT/scripts/turbo_run_checks.py" table "$OUT_ROOT" $MODES --gpu "$GPU_NAME" \
  --out-json "$OUT_ROOT/proof_24gb.json" | tee "$OUT_ROOT/proof_24gb.md"
