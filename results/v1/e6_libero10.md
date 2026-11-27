### E6 Curation & Repair Benchmark (Native LeRobot v2.1 vs Lance)

- **Dataset**: `/data/datasets/libero_10` (379 episodes, 101469 frames)
- **Workload**: drop `5.0%` bad episodes (`seed=42`) + filter idle frames (`idle_eps=0.001`) + recompute GR00T `q01`/`q99` stats

| Metric | Native LeRobot v2.1 (curated copy) | Lance (zero-copy column evolution) | Ratio / Delta |
|---|---:|---:|---:|
| Wall time (s) | 1.716 s | **0.296 s** | **5.8x faster** |
| Total bytes written | 604.03 MB | **0.90 MB** | **674x less I/O** |
| Video bytes written | 594.19 MB (720 MP4s) | **0.00 MB (0 files)** | **0 video rewrite** |
| Tabular / frames bytes written | 8.85 MB | 0.87 MB | — |
| Metadata bytes written | 0.99 MB | 0.02 MB | — |
| Files / objects written | 1087 | **10 (frames) + meta** | — |
| Resulting disk footprint | 604.03 MB | 635.12 MB (+0.89 MB) | — |
| Mid-episode idle filtering | Not supported without breaking horizons / re-encoding MP4s | **Supported (`where="is_valid = true"`)** | **5.2% anchors filtered** |
| Filtered dataset init time | N/A | **68.5 ms** (90,751 / 95,784 anchors) | Lance v3 -> v6 |
| Unmodified `lerobot-lancedb` compat | N/A | **PASS** | Zero breaking changes |
