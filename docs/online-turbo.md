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

基线对照模式（配合 `BENCH_STEPS` 只测速度和显存）：

| `MODE` | 配置 | 对应主表 |
|---|---|---|
| `stock` | `32 × 1`，原生 fp32 AdamW，turbo 全关 | `B0` |
| `stock_ga2` | `16` × 梯度累积 2，原生 fp32 AdamW，只把冻结 VLM 转 bf16 | `B1` |
| `bnb8bit` | `16 × 2` + `bitsandbytes.AdamW8bit`（fp32 权重） | `B2` |

## 上 GPU 运行（门禁在前，贵的在后）

`scripts/run_online_turbo.sh` 先跑 **Gate 0（Triton 对拍，含多张量融合内核）** 和 **Gate 1（5 步冒烟）**，任一不过直接退出。Gate 1 现在检查五件事：loss 有限、turbo 生效、0 个 `_orig_mod` 键、`COMPILE_HEAD=1` 时 `torch.compile` **真的生效**（fp32 权重会被静默跳过）、`LowMemAdamW` **真的在用融合发射**（没有退回逐张量）。后两条是为了不让"悄悄走慢路径"的 3,000 步结果进主表。

```bash
# Phase 2 第 1 步（约 3 分钟）：优化器微基准 + 融合门禁
pytest tests/test_turbo_lowmem_adamw.py -k triton -q
python scripts/bench_lowmem_adamw.py --base-model-path nvidia/GR00T-N1.7-3B --require-fused --out-json bench_adamw.json

# 第 2 步：B1 / B2 各 50 步，只要步时和峰值显存（OOM 记为结果，不算失败）
MODE=stock_ga2 BENCH_STEPS=50 bash scripts/run_online_turbo.sh
MODE=bnb8bit   BENCH_STEPS=50 bash scripts/run_online_turbo.sh

# 第 3 步：三档各 3,000 步（先不评测）
for m in mn2 mn4 lowmem_only; do MODE=$m COMPILE_HEAD=1 EVAL=0 bash scripts/run_online_turbo.sh; done

# 第 4 步：评测。先给 B0 的已有 checkpoint 补一份同种子评测，再评各档并自动配对比较
python scripts/eval_libero10_checkpoint.py --model-path /data/runs/e2e_ablation_3000s/group1_native_online/checkpoint-3000 \
  --n-episodes 20 --out-json /data/runs/b0_eval.json
MODE=mn2 COMPILE_HEAD=1 BASELINE_EVAL=/data/runs/b0_eval.json bash scripts/run_online_turbo.sh   # 已有 checkpoint 会跳过训练
```

## 评测口径（Phase 1 改过，旧数字不能直接比）

旧评测每个任务起一个并行向量环境，`rollout_policy.py` 在"至少完成 n 回合"时停。这有两个问题：

1. 同一步可能有多个环境同时结束，要 5 回合有时报 6 回合，这就是归档结果里 `66.7%` / `50%` 的来源。
2. **偏差**：先结束的环境会立刻开下一回合。成功的回合提前结束，失败的回合跑满 720 步，所以"最先完成的 n 个"里成功被多算。按这个机制，报出来的成功率偏高，越弱的 checkpoint 偏得越多；偏多少没有量过。归档的 `81.67%` 和 `85.00%` 都是用这个口径跑的。

新评测每回合一个进程、一个仿真器、一个显式种子（`seed_base + 1000 × 任务号 + 回合号`），`N_ENVS` 个进程并行。每任务精确 `N_EPISODES` 回合，JSON 里逐回合记录种子和胜负。代价是每回合多一次仿真器启动（约 10–15 秒）。

两个 checkpoint 用同一个 `EVAL_SEED_BASE` 评测后：

```bash
python scripts/compare_libero_evals.py --baseline b0_eval.json mn2=mn2/eval.json mn4=mn4/eval.json
```

输出成功率、差值、95% 置信区间、精确 McNemar p 值。配对控制的是**初始场景相同**；策略自己的采样噪声不受控，单回合不是确定性的。

**200 回合能说明什么**：两个 checkpoint 真实水平相同、约 15% 的回合结果不一致时，差值的 95% 区间约 ±5 个点。所以 200 回合能下的结论是"没有显著差异 / 不比基线差超过约 5 个点"，**下不了"差值在 ±2% 以内"**这个结论；后者约需 1,400 个配对回合。计划里"±2%"这条门禁要么改成"McNemar 不显著且区间下界 > −5 点"，要么把回合数加上去。

## 24 GB 实拍

```bash
bash scripts/bench_24gb_proof.sh        # mn2 mn4 stock stock_ga2 各 60 步
```

在 24 GB 卡上输出一张表：每档是"能跑 + 峰值显存 + 步时"还是"CUDA OOM"，以及 `nvidia_smi.txt` 和每档的训练日志（OOM 的 traceback 在里面）。非 OOM 的报错会让脚本失败，不会被当成 OOM 写进表。
