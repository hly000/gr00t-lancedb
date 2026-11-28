| group | LIBERO-10 success | episodes | 95% CI |
|---|---:|---:|---|
| B0 (32x1, fp32 AdamW) | 75.0% | 75/100 | [65.7%, 82.5%] |
| G_b16_fp32 (16x1, fp32 AdamW) | 36.0% | 36/100 | [27.3%, 45.8%] |
| G_b16_lowmem (16x1, LowMemAdamW bf16) | 46.0% | 46/100 | [36.6%, 55.7%] |
| G_b16_mn2 (16x2, LowMemAdamW bf16) | 61.0% | 61/100 | [51.2%, 70.0%] |

| paired comparison | diff (pts) | 95% CI | discordant (cand-only / base-only) | McNemar p |
|---|---:|---|---:|---:|
| `b16_fp32_vs_b0` | -39.0 | [-50.7, -27.3] | 6 / 45 | 0.0 |
| `b16_lowmem_vs_b0` | -29.0 | [-41.5, -16.5] | 10 / 39 | 4e-05 |
| `mn2_vs_b0` | -14.0 | [-26.1, -1.9] | 13 / 27 | 0.03848 |
| `b16_lowmem_vs_b16_fp32` | +10.0 | [-3.1, +23.2] | 28 / 18 | 0.18392 |
| `mn2_vs_b16_fp32` | +25.0 | [+12.5, +37.5] | 36 / 11 | 0.00035 |
| `mn2_vs_b16_lowmem` | +15.0 | [+2.2, +27.8] | 30 / 15 | 0.0357 |

- **Verdict A, accuracy** (G_b16_lowmem vs G_b16_fp32): **PASS**, diff +10.0 pts, p=0.18392
- **Verdict A, memory** (b16_k1 under the 24 GiB cap, peak <= 20 GB): **PASS**, peak 16.92 GB
- **Verdict B, multi-noise** (G_b16_mn2 vs G_b16_lowmem): **KEEP (preliminary: needs a second training seed)**, diff +15.0 pts, p=0.0357
- Batch 32 -> 16 (G_b16_fp32 vs B0), for information: diff -39.0 pts, p=0.0
- **Route**: Both hold: reproduce mn2 vs b16_k1 with a second training seed, then the final matrix.
