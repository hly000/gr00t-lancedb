# 低显存优化器内核与热点剖析（`feat/turbo-kernels`）

这个分支只做两件事，都不依赖特征缓存：

1. `LowMemAdamW`：优化器状态每参数 2 字节的 AdamW，权重可以是 bf16 而不丢小更新。
2. `scripts/profile_kernels.py`：一次跑出单步训练的时间、显存、算子热点和特征量化统计。

## 为什么需要这个优化器

动作头有 16.2 亿可训练参数。三种现成做法各有一个硬伤：

| 做法 | 每参数字节（权重+梯度+状态） | 16.2 亿参数 | 问题 |
|---|---:|---:|---|
| fp32 权重 + `torch.AdamW` | 4 + 4 + 8 = 16 | 25.9 GB | 还没算激活就超过 24 GB |
| bf16 权重 + `torch.AdamW` | 2 + 2 + 4 = 8 | 13.0 GB | 小于半个 ulp 的更新被舍掉 |
| `bitsandbytes.AdamW8bit` | 2 + 2 + 2 = 6 | 9.7 GB | 实测每步 98.5 ms（fused AdamW 是 19.8 ms），且同样是 bf16 权重 |
| **`LowMemAdamW`，bf16 权重** | 2 + 2 + 4.06 = 8.06 | 13.1 GB | 见下面"代价" |
| `LowMemAdamW`，fp32 权重 | 4 + 4 + 2.06 = 10.06 | 16.3 GB | 同上 |

bf16 只有 8 位有效数字。权重 0.05 的一个 ulp 是 2.4e-4，`lr=1e-4` 时绝大多数更新不到半个 ulp，直接被舍成 0。CPU 上的合成实验里，纯 bf16 的 AdamW 只走了 fp32 轨迹的 15%–20%。

## 做法

- **权重 bf16 + 补偿项 bf16**：补偿项存"真实权重减去 bf16 权重"的余数，两者合起来约 16 位有效数字（fp32 是 24 位）。前向读到的就是 bf16 权重，和 fp32 主权重在 `autocast(bf16)` 下送进矩阵乘法的值相同。
- **一阶动量存成 `m/√v` 的 int8**，每 128 个元素一个缩放系数。分辨率相对每个元素自己的梯度尺度，而不是相对块内最大值。
- **二阶动量存成 uint8 的"迷你浮点"**（5 位指数 + 3 位尾数），块内可表示 31 个倍频程，相对误差有界。线性量化会把小的 `v` 舍成 0，更新 `m/√v` 会爆。
- **两个动量都用随机舍入**。`β2=0.999` 时 `v` 每步只变 0.1%，量化级距是 6%–12%，四舍五入会让它永远不动；随机舍入让期望值正确。
- 当步的更新用未量化的动量算，量化只影响带到下一步的状态。
- 全部编解码都是 int32 位运算，CPU 参考实现和 Triton 内核按同一套算术写。

## 代价（如实）

这些数字来自 `tests/test_turbo_lowmem_adamw.py` 的合成问题，**不是真实模型**：

- 带到下一步的 `v` 围绕精确值有约 20% 的均方根抖动，折算到每个元素的步长约 10%。
- `E[1/√v̂] > 1/√v`，等效学习率高约 1.4%。
- 在"梯度与权重无关、误差只累积不纠正"的最坏场景下，最终权重和 fp32 AdamW 的差距，不到"换一个 minibatch 顺序"造成差距的一半。
- 补偿后的权重是 16 位精度，不是 fp32 的 24 位。

速度没有测过。优化器在仅动作头的一步里只占约 12%（171 ms 里的 20 ms），所以它的价值是显存，不是速度。另外内核按张量逐个发射，动作头有几百个参数张量，Python 发射开销可能有十几到几十毫秒；`bench_lowmem_adamw.py` 里的 `lowmem_bf16_1t` 一项专门用来量这个开销。

## 多张量融合发射（Phase 1 新增）

A100 实测（1.62B 参数）：逐张量发射 980 个张量 56.34 ms，同样大小合成 1 个张量 16.89 ms。差的 ~40 ms 全是 Python 发射开销（每次约 40 µs），不是计算。

现在同一参数组、同一 dtype 的所有张量用**一次**内核发射更新：

- 主机每步传一张小表（每张量一行 int64：各数组相对本批第一个张量的元素偏移、元素个数、起始 program、随机种子）和一个"program → 张量编号"的 int32 数组；每个 program 先查表拿到自己的指针，之后的算术和逐张量内核逐行相同。
- **不拷贝、不拼接**任何权重 / 梯度 / 状态，额外显存只有那张表（每 1024 个元素 4 字节，1.62B 参数约 6 MB）。
- 种子和元素偏移仍按张量算，所以结果和逐张量发射一致；`state_dict` 格式不变，旧 checkpoint 可以直接续训。
- 默认 `fused=None`（自动）：融合内核编译失败时打一条 `LOWMEM_FUSED_FALLBACK` 警告并退回逐张量发射（结果相同，只是慢）。`run_online_turbo.sh` 的 Gate 1 看到这条警告会直接判失败，不会让你带着慢路径跑 3,000 步。`GR00T_TURBO_LOWMEM_FUSED=off` 可手动关闭。

**没验证的部分**：融合内核同样没在 GPU 上跑过。CPU 上验证了的是主机侧全部逻辑（分批、表、种子、偏移，用"基地址 + 表内偏移"真实解引用内存，结果与逐张量逐位相同）以及两份内核的算术文本逐行相同。Triton 能否编译"指针 + 表里读出的 int64 偏移"这种寻址、以及实际能快到多少，要等 Gate 0 和 `bench_lowmem_adamw.py --require-fused`。目标 ≤ 25 ms 是目标，不是测量值。

## 上 GPU 的顺序

Triton 内核**没有在任何 GPU 上跑过**（本机没有 CUDA）。第一步必须是对拍测试：

```bash
# 1. 约 30 秒：Triton 内核和 CPU 参考实现逐元素对拍。不过就不要往下跑
pytest tests/test_turbo_lowmem_adamw.py -k triton -q

# 2. 约 1 分钟：动作头真实形状下的每步耗时和显存；--require-fused 是门禁
#    （融合发射必须生效，且 lowmem_bf16 ≤ 25 ms、峰值 ≤ 12.5 GB，否则退出码非 0）
python scripts/bench_lowmem_adamw.py --base-model-path nvidia/GR00T-N1.7-3B --require-fused --out-json bench_adamw.json

# 3. 约 5 分钟：原生口径（fp32 权重 + bf16 autocast）的热点剖析
python scripts/profile_kernels.py --out-dir /data/runs/profile_kernels
```

训练时用：`train_vl_cache_e2e.py --optimizer lowmem --no-fp32-master-weights`。

## 怎么读 `profile_kernels.json`

- `phases`：各阶段耗时。
- `memory.analytic` 对 `memory.measured_peak_alloc_gb_by_phase`：静态占用和各阶段峰值，直接回答"fp32 动作头到底要多少显存"。
- `ops_head.gpu_busy_frac`：动作头这一段里内核执行时间占墙钟的比例。**这个数决定要不要写融合算子**：接近 1 说明是矩阵乘法在算，融合逐元素算子没有收益；明显低于 1（比如 0.6）才说明时间耗在内核发射上，`torch.compile`、CUDA Graph 或融合算子才有用。
- `ops_head.by_category_pct`：矩阵乘、注意力、归一化、逐元素各占多少。
- `features`：真实 VLM 特征的峰值因子、离群通道、int8 误差，以及同一组噪声下精确特征和 int8 特征的动作头 loss（fp32 计算）。之前的对齐检查比的是两个 bf16 loss，分辨不出 0.4% 以下的差异。

## 没做的

融合 AdaLN、融合流匹配 loss 这两个 Triton 算子没写。它们值不值得写取决于上面的 `gpu_busy_frac` 和 `by_category_pct`，在拿到这两个数之前写是盲写。
