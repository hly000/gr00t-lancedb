# 特征缓存管线：物化提速、留出集探针、一次跑完的消融（`feat/vl-cache-fast`）

这个分支叠在 `feat/turbo-kernels` 上，包含那边的全部内容（见 `docs/turbo-kernels.md`）。

## 改了什么

### 1. 物化不再让 GPU 等 CPU

首版物化 38 帧/秒，95,784 帧用了 42.5 分钟。原来每个 batch 把 fp32 特征整块搬回 CPU，再在主线程里逐行做 numpy 量化、求池化向量、攒 Arrow 数组，GPU 这段时间空等。

- 量化和池化移到 GPU（`encode_features_torch`、`pooled_visual_emb_torch`），过 PCIe 的是 int8，不是 fp32。
- 写 Lance 放到后台线程（`AsyncSink`），队列满了会反压，写失败会在主线程抛出。
- collate 移进 DataLoader worker，主线程只负责喂 GPU。
- 默认 `--mat-batch-size 128`、`--mat-workers 8`。

GPU 端编码和原来的 numpy 编码**逐字节一致**（有测试），指纹不变，已有的 15 GB 缓存直接复用，不需要重新物化。

实际能快多少没测过。新的 `*_summary.json` 里有 `stage_s`（等解码、骨干加编码、写盘线程各占多少秒），哪一项最大，下一步就该修哪一项。如果最大的是"等解码"，瓶颈在视频解码和预处理，再优化 GPU 端没有用。

### 2. 留出集探针 loss

50 回合的闭环评测要 16 分钟，95% 置信区间大约是 ±11 个点；训练循环打印的 loss 是每个样本随机抽一组 `(t, 噪声)`、在 bf16 里求和的结果。两者都分不清差几个百分点的两个配置。

探针做三件事：

- 按整条 episode 留出 3%（`--holdout-frac`），训练不采这些帧。
- 每个检查点用同一批帧、同一组 `(t, 噪声)`，eval 模式，loss 在 fp32 里算。
- 训练中每 500 步算一次（约 2 秒），曲线写进 `train_summary.json` 的 `probe_curve`。

`scripts/eval_cached_fm_loss.py` 可以对任意检查点算探针 loss，包括原生训练出来的，并给出配对差值和标准误。

**它是代理指标**。这里没有验证过"探针 loss 更低就等于成功率更高"，用它淘汰明显差的配置、发现退化，最后仍要用闭环评测确认。原生检查点训练时见过这 3% 的 episode，和缓存训练的检查点比时略占便宜。

### 3. 输入精度对齐原生

状态、动作、掩码以 fp32 送进动作头，和原生训练一致（原生靠 `autocast` 在各层内部转 bf16）。这样流匹配的时间 `t`、加噪轨迹和速度目标都在 fp32 里构造。新缓存的标量按 fp32 落盘；旧缓存里 float16 的标量读入时转成 fp32。

bf16 权重配 fp32 输入这条路径没在真实模型上跑过；另外已知 fp32 权重配 `torch.compile` 在 PyTorch 2.9 上第 1 步会崩。所以训练开始时先做一次试跑：一次带反向的训练前向，加一次探针前向。失败就依次退回"不编译"、"bf16 输入"，并打印警告。实际用的设置记在 `train_summary.json` 的 `compile` 和 `input_dtype`，对比两组的每步耗时前先看这两项是否一致。

## 一次 GPU 会话该怎么跑

```bash
# 0. 约 30 秒。Triton 内核对拍，不过就只跑 fp32_adamw
pytest tests/test_turbo_lowmem_adamw.py -k triton -q

# 1. 约 1 分钟。三个已有检查点在同一批留出帧上的 loss
python scripts/eval_cached_fm_loss.py --cache-dir /data/tmp/libero10_vl_cache_full \
  --ckpt native=/data/runs/e2e_ablation_3000s/group1_native_online/checkpoint-3000 \
  --ckpt cache_v1=/data/runs/e2e_ablation_3000s/group2_vl_cache/merged_gr00t_ckpt \
  --ckpt base=nvidia/GR00T-N1.7-3B

# 2. 默认两组：fp32_adamw（精度基准，不编译，更慢）和 bf16_lowmem（24 GB 候选，编译，约 9 分钟）
NATIVE_CKPT=/data/runs/e2e_ablation_3000s/group1_native_online/checkpoint-3000 \
  bash scripts/run_cache_ablation.sh

# 3. 探针 loss 接近的配置再跑闭环。每任务 10 回合约 32 分钟一组
EVAL=1 N_EPISODES=10 bash scripts/run_cache_ablation.sh    # 已训练的配置会跳过训练
```

第 1 步的三个数怎么读：

| 结果 | 含义 |
|---|---|
| `native` 明显低于 `cache_v1` | 缓存训练出来的动作头在同样输入上确实更差，是优化问题（比如纯 bf16 权重） |
| `native` 和 `base` 差不多 | 原生训练的动作头读不懂 int8 特征，问题在缓存本身（量化、padding、token 顺序） |
| `native` 低、`cache_v1` 也低，但闭环差很多 | 问题在导出或推理侧（处理器、统计量），不在训练 |

可选配置：`CONFIGS="fp32_adamw bf16_lowmem fp32_lowmem bf16_adamw bf16_lowmem_mn4"`。`fp32_lowmem` 单独检验 8-bit 动量，`bf16_adamw` 复现首版的失败方式。

## 关于首版 27% 的几点判断

- 3,000 步 × 32 = 96,000 个样本，数据集有 95,784 帧，等于**每帧只看了一遍**。单视角、没有图像增强造成的过拟合要靠重复看同一帧才会出现，所以它解释不了这次掉点，Hybrid 80/20 在这个训练量下也补不了什么。训练多个 epoch 时这个问题才会出现。
- float16 的状态和动作在 ±1 范围内误差不超过 2.5e-4，是 bf16 舍入误差的八分之一，而模型内部本来就在 bf16 里算，这一项可以排除。
- 纯 bf16 权重是最可能的主因：`lr=1e-4` 时大部分更新不到半个 ulp，被直接舍掉。
- 特征加噪（5% token 丢弃 + 1% 抖动）和 int8 量化的影响没有单独量过；`profile_kernels.py` 的 `features` 一节会给出 int8 的误差和对 loss 的影响。

## 没做的

- **Hybrid 80/20 和多视角缓存**。按上面第一条，3,000 步的预算下它们不解决任何已观察到的问题。等有了多 epoch 的掉点数据再做。
- **Triton 融合反量化**。现在的反量化是一次 `int8 → bf16` 乘法，在每步里占比极小。
- 评测脚本仍是每任务 5 回合、从输出文本里解析成功率，实际回合数不一定正好是 5。做结论前把 `N_EPISODES` 提到 10 以上。
