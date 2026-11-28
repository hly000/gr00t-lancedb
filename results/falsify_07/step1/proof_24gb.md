| mode (NVIDIA A100-SXM4-40GB, 40960 MiB) | result | peak GPU memory | ms/step | vs first finished | torch max_alloc | allocator cap |
|---|---|---:|---:|---:|---:|---:|
| `stock_cap24` | CUDA OOM at step 1 | n/a | - | - | - | 23.7 GiB |
| `stock_ga2_cap24` | CUDA OOM at step 1 | n/a | - | - | - | 23.7 GiB |
| `b16_k1_fp32_cap24` | CUDA OOM at step 1 | n/a | - | - | - | 23.7 GiB |
| `b16_k1_cap24` | runs | 16.92 GB | 395.31 | 1.00x | 15.70 GB | 23.7 GiB |
| `mn2_cap24` | runs | 18.55 GB | 398.39 | 0.99x | 17.69 GB | 23.7 GiB |
| `bnb8bit_cap24` | runs | 22.52 GB | 488.32 | 0.81x | 20.06 GB | 23.7 GiB |
| `stock_nocap` | runs | 37.00 GB | 649.49 | 0.61x | 35.87 GB | none |
| `stock_ga2_nocap` | runs | 34.26 GB | 838.9 | 0.47x | 33.04 GB | none |
| `b16_k1_fp32_nocap` | runs | 37.08 GB | 469.12 | 0.84x | 35.87 GB | none |
