### E6 Curation & Repair Benchmark (Native LeRobot v2.1 vs Lance)

- **Dataset**: `/data/datasets/x_tier/x10/v21` (3790 episodes, 1014690 frames)
- **Workload**: drop `5.0%` bad episodes (`seed=42`) + filter idle frames (`idle_eps=0.001`) + recompute GR00T `q01`/`q99` stats

| Metric | Native LeRobot v2.1 (curated copy) | Lance (zero-copy column evolution) | Ratio / Delta |
|---|---:|---:|---:|
| Wall time (s) | 18.749 s | **3.158 s** | **5.9x faster** |
| Total bytes written | 6038.44 MB | **8.22 MB** | **734x less I/O** |
| Video bytes written | 5942.37 MB (7200 MP4s) | **0.00 MB (0 files)** | **0 video rewrite** |
| Tabular / frames bytes written | 86.24 MB | 8.17 MB | — |
| Metadata bytes written | 9.83 MB | 0.05 MB | — |
| Files / objects written | 10807 | **10 (frames) + meta** | — |
| Resulting disk footprint | 6038.44 MB | 6342.26 MB (+8.22 MB) | — |
| Mid-episode idle filtering | Not supported without breaking horizons / re-encoding MP4s | **Supported (`where="is_valid = true"`)** | **5.2% anchors filtered** |
| Filtered dataset init time | N/A | **13652.7 ms** (907,880 / 957,840 anchors) | Lance v3 -> v6 |
| Unmodified `lerobot-lancedb` compat | N/A | **PASS** | Zero breaking changes |
