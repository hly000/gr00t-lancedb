# `feat/train-perf-v1` Real A100 Benchmark Summary

- **Hardware**: 1× NVIDIA A100-SXM4-40GB, 11 vCPU, 70 GB RAM (Instabox `groot`, GCP `us-central1`)
- **Model / Dataset**: `nvidia/GR00T-N1.7-3B` fine-tuned on LIBERO-10 (`379` episodes, `101,469` frames), `global_batch_size=32`, `dataloader_num_workers=4`, `seed=42`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- **Cloud streaming bucket**: `s3://alignd-7fb6a-groot-stream/libero_10_lance` (GCS S3-compatible HMAC endpoint, zero local download)

---

## 1. Cold-Start, Steady-State, GCS Streaming & Fingerprint Parity (60 steps)

| Configuration | `first_batch_wait` | `first_step` | `launch → first_step` | `steady_step` (median) | `data_wait` (median) | `peak_sys_used` | Mean GPU Util | Batch FP vs `lance_v0` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `native` (local LeRobot v2.1) | 7.35 s | 11.85 s | 46.93 s | 0.665 s | 0.50 ms (0.07%) | 18.85 GB | 68.8% | — |
| `lance_v0_like` (`forkserver` + eager) | 18.49 s | 21.63 s | 61.21 s | 0.656 s | 0.51 ms (0.08%) | 18.34 GB | 69.1% | 60/60 (ref) |
| `lance_v1_local` (`fork` + lazy decode) | **6.09 s** (3.0× faster than v0) | 10.00 s | 47.48 s (-13.7 s vs v0) | 0.653 s | 0.48 ms (0.07%) | **16.85 GB** (-2.00 GB vs native) | **75.4%** | **60/60** |
| `lance_v1_prefetch` (`+ GR00T_LANCE_ASYNC_PREFETCH=1`) | **4.83 s** (3.8× vs v0 / 34% faster than native) | **8.94 s** | 48.00 s | **0.651 s** | 0.53 ms (0.08%) | **16.77 GB** (-2.08 GB vs native) | 72.3% | **60/60** |
| `lance_v1_gcs` (GCS direct, zero download) | **7.04 s** (faster than local native) | 10.88 s | 48.52 s | 0.655 s | 0.48 ms (0.07%) | **16.98 GB** (-1.87 GB vs native) | 71.7% | **60/60** |

> Raw per-step metrics: `results/v1/cold_steady/*/bench_metrics.jsonl`

---

## 2. Spot Preemption & Exact Resume Comparison (80 steps, `save_steps=30`, `SIGTERM@step=45`)

### Uninterrupted 80-step References (`save_steps=1000`)

| Reference Run | `launch → end` | `first_batch_wait` | `steady_step` (median) | `peak_sys_used` | Step-80 Loss |
|---|---:|---:|---:|---:|---:|
| `lance_v1_ref` | **102.07 s** | **5.50 s** | **0.652 s** | **16.87 GB** | 1.1523 |
| `native_ref` | 109.74 s | 7.25 s | 0.658 s | 19.04 GB | 1.1469 |

### Preempted Runs (`SIGTERM` at step 45 → resume to step 80)

| Run | Mode / Policy | `SIGTERM → exit` | Resumed From | Replayed Steps | `resume launch → 1st step` | `resume 1st batch` | Total E2E Wall (`pre_start → resume_exit`) | Post-Resume Batch FP vs Ref | `peak_sys_used` | Step-80 Loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `lance_v1_light` | Trainable-only emergency ckpt (`19 GB` vs `24 GB` full) | 61.39 s | step 46 | **0** | **68.57 s** | 6.83 s | 379.72 s | **34/34 (100% bit-exact)** | 37.34 GB | 1.1496 |
| `lance_v1_skip` | Cost-gated skip (`16 × 0.69s = 11.1s ≤ 30s`) | **22.57 s** (fits 30 s GCP window) | step 30 | 16 | **68.90 s** | **5.25 s** | **355.20 s** (-24.5 s vs light) | **50/50 (100% bit-exact)** | **20.88 GB** | 1.1531 |
| `native_pre` | Stock GR00T (no SIGTERM handler; reseeds `seed + step`) | 0.20 s | step 30 | 15 | 86.29 s (+17.4 s slower) | 7.06 s | 304.87 s | **0/50 (0% — scrambled)** | 23.16 GB | 1.1397 |

### Key Takeaways & Next-Step Opportunities

1. **Data-order exactness**: Both `lance_v1_light` (`46/46` + `34/34`) and `lance_v1_skip` (`46/46` + `50/50`) reproduce **100% bit-exact** batch sequences relative to the uninterrupted `lance_v1_ref`, while stock `native_pre` gets **0/50** after resume due to `seed + global_step` reseeding.
2. **Faster resume startup**: Both Lance v1 modes reach the first resumed step **~17.4 s faster** than `native_pre` (`68.6–68.9 s` vs `86.3 s`).
3. **Cost-gating wins when recent checkpoint exists**: Skipping an emergency save when recompute cost (`11.1 s`) is lower than save cost (`~39 s` for `19 GB`) finishes **24.5 s faster end-to-end** (`355.2 s` vs `379.7 s`), avoids the CPU RAM spike (`20.88 GB` vs `37.34 GB`), and exits inside GCP's 30 s Spot grace window (`22.57 s`).
4. **Two clear v2 follow-ups** spotted directly from these logs:
   - Use immediate exit (`os._exit(143)` after flushing metrics) on cost-gated skip to drop `SIGTERM → exit` from `22.57 s` to `<0.2 s` (eliminating ~22 s of Python/DataLoader teardown).
   - Store AdamW moments in `bf16`/8-bit inside `emergency_trainable.pt` to cut emergency checkpoint size from `19 GB` (`fp32` trainable weights + `fp32` AdamW `m`/`v`) down to `~5–6 GB` (`<10 s` write, fitting inside a 30 s grace window even when `save_steps` is large).

> Raw metrics & timelines: `results/v1/preempt/*/{events.jsonl,bench_metrics.jsonl,pre/bench_metrics.jsonl}`
