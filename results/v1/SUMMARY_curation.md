# `feat/lance-curation` Real A100 Benchmark Summary

- **Hardware**: 1× NVIDIA A100-SXM4-40GB, 11 vCPU, 70 GB RAM (Instabox `groot`)
- **Dataset**: LIBERO-10 (`379` episodes, `101,469` frames)

## 1. E6 Curation & Repair Benchmark (Native LeRobot v2.1 vs Lance)

Workload: drop `5.0%` bad episodes (`seed=42`) + filter idle frames (`idle_eps=1e-3`) + recompute GR00T `q01`/`q99` stats.

| Metric | Native LeRobot v2.1 (curated copy) | Lance (zero-copy column evolution) | Ratio / Delta |
|---|---:|---:|---:|
| Wall time (s) | 1.716 s | **0.296 s** | **5.8× faster** |
| Total bytes written | 604.03 MB | **0.90 MB** | **674× less I/O** |
| Video bytes written | 594.19 MB (720 MP4s) | **0.00 MB (0 files)** | **Zero video rewrite** |
| Tabular / frames bytes written | 8.85 MB | 0.87 MB | — |
| Metadata bytes written | 0.99 MB | 0.02 MB | — |
| Files / objects written | 1,087 | **10 (`frames.lance`) + meta** | — |
| Resulting disk footprint | 604.03 MB | 635.12 MB (`+0.89 MB`) | — |
| Mid-episode idle filtering | Not supported without breaking horizons / re-encoding MP4s | **Supported (`where="is_valid = true"`)** | **5.2% anchors filtered** |
| Filtered dataset init time | N/A | **68.5 ms** (`90,751 / 95,784` anchors) | Lance `v3 → v6` |
| Unmodified `lerobot-lancedb` compat | N/A | **PASS** | Zero breaking changes |

## 2. Filtered GR00T-N1.7-3B A100 Finetuning Verification

- Command env: `GR00T_LANCE_WHERE="is_valid = true"`, 60 steps on curated `/data/datasets/libero_10_lance_curated`
- Result: **PASS** — `60/60` steps completed, median steady step `0.675 s/step`, mean GPU utilization `73.2%`, peak `sys_used` `18.33 GB`.

> Raw files: `results/v1/e6_libero10.json`, `results/v1/e6_libero10.md`, `results/v1/filtered_train/bench_metrics.jsonl`
