# Online Turbo Fine-Tuning for GR00T N1.7 (`feat/online-turbo`)

**零离线特征缓存、100% 保留原生在线图像增强**的单卡提速与降显存分支。

## 为什么要做这个分支（把离线缓存这个元凶摘掉）

离线 VL-Cache 带来了三类硬问题：44 分钟物化反亏总耗时、剥夺每步在线 `RandomCrop + ColorJitter`、把 forward 从中间劈开引入左 padding 补零与 dtype 漂移。

本分支彻底去掉离线缓存，只保留在**原生在线训练**上 100% 独立成立的两项核心优化，直接挂载到官方 `launch_finetune.py`（通过 `scripts/launch_finetune_lance.py` 的 opt-in seam）：

1. **在线 Stratified Multi-Noise Flow Matching**（`GR00T_TURBO_MULTI_NOISE=2|4`）
   - 2B VLM Backbone 每步只对 `B` 张全新在线增强图像跑一次；
   - 轻量 DiT 动作头复用这份**原生 Qwen3-VL 输出**（含真实 `<pad>` 状态，无补零 Bug）跑 `K` 组分层 Beta 时间步 + 独立高斯噪声，保持等效流匹配批大小 `B × K = 32`；
   - 0 额外磁盘、0 离线物化、单次微调即净赚速度与显存。
2. **24 GB 显存方案：`LowMemAdamW` + Parameter-Only BF16**（`GR00T_TURBO_OPTIMIZER=lowmem` + `GR00T_TURBO_BF16_WEIGHTS=1`）
   - 只将 `nn.Parameter` 转为 `bfloat16`（冻结 Backbone 省 ~4.0 GB，动作头权重+梯度+状态从 25.9 GB 压到 ~13.1 GB），**绝不动非参数 buffer（保护 Qwen3-VL FP32 RoPE `inv_freq`）**；
   - `LowMemAdamW` 用 Kahan bf16 补偿 + 8-bit 随机舍入动量避免纯 bf16 丢小更新；
   - 如果开了 `GR00T_TURBO_COMPILE_HEAD=1`，保存 checkpoint 时自动临时 unwrap `_orig_mod`，保证导出权重 100% 标准兼容 `run_gr00t_server.py`。

## 四档预设模式

| `MODE` | 帧/步 (`B`) × 噪声 (`K`) | 优化器与权重 | 作用 |
|---|---:|---|---|
| **`mn2`**（默认，最稳） | `16 × 2 = 32` | `LowMemAdamW` + bf16 | VLM 计算与激活减半、显存压进 24 GB，同时保留 16 张不同增强帧/步 |
| `mn4`（最快） | `8 × 4 = 32` | `LowMemAdamW` + bf16 | VLM 计算砍到 1/4，验证激进提速下的闭环成功率 |
| `lowmem_only` | `32 × 1 = 32` | `LowMemAdamW` + bf16 | 与原生 Group 1 数据流完全一致，单独验证 24 GB 优化器精度与速度 |
| `mn2_fp32` | `16 × 2 = 32` | 原生 fp32 `AdamW` | 单独验证在线 Multi-Noise 的提速与闭环精度 |

## 上 GPU 运行（自带 2 道防浪费快速门禁）

`scripts/run_online_turbo.sh` 默认先跑 **Gate 0（30 秒 Triton 对拍）** 和 **Gate 1（5 步冒烟 + 有限 loss + 检查 0 个 `_orig_mod` 键）**，任一门禁不通过直接退出，绝不浪费 GPU 跑 3,000 步：

```bash
# 推荐首跑：默认 MODE=mn2（16x2 + LowMemAdamW bf16），含门禁 + 3000 步训练 + 50 回合闭环
bash scripts/run_online_turbo.sh

# 跑完如果稳在 ~80%，再测更快档位 mn4 或把评测加到 10 ep/task（--n-envs 自动上限为 5 防 EGL 爆显存）
MODE=mn4 N_EPISODES=10 N_ENVS=5 bash scripts/run_online_turbo.sh
```

参考基线（原生 Group 1，`32 × 1`，fp32 AdamW）：**38.3 分钟 / 37.0 GB / LIBERO-10 81.67%**。
