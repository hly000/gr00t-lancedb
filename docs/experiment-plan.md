# GR00T × Lance 训练数据层：实验方案与 v0 基线结果

> **版本**：Lance v0 基线（`gr00t-lancedb` commit `7f722aa` 及之后的评测脚本），2026-11-26
> **一句话现状**：正确性已全部过关：数据逐字节/逐像素一致，抢占续训数据序列精确一致。性能上，小数据集稳态持平，冷启动反而落后约 12 秒。"几百 GB"规模的 H1/H2 还没测，RoboCasa 4 任务（约 32 GB）正在准备。
> 图例：✅ 已完成　🟡 部分完成　⏳ 进行中　⬜ 未开始　❌ 结论不成立

---

## 0. 要回答的问题与当前证据

| # | 假设 | 当前证据（v0） | 状态 |
|---|---|---|---|
| H1 | 冷启动从"几十分钟到小时级"降到"分钟级" | 只测了 LIBERO-10（0.64 GB）。原生先下载再训 **51.6 s**，Lance 直读 GCS **64.0 s**，**Lance 慢 12 s**，因为下载只要 2.8 s，而 Lance 首个 batch 慢。大数据集没测；按实测 230 MB/s 外推，353 GB 下载约 26 分钟，且本机 196 GB 磁盘放不下 | 🟡 小数据集不成立，大数据集待测 |
| H2 | 被抢占后 Lance 恢复更快，浪费的 GPU 时间更少 | 数据正确性：**Lance 精确续训**（300/300 步批次指纹一致，0 步重做）；原生回滚 50 步，续训后 0/200 步一致。时间：本地数据场景下 **Lance 总耗时反而多 18%**（571 s vs 483 s） | 🟡 正确性成立，时间不成立（v0） |
| H3 | 稳态吞吐不输本地盘，等数据占比 < 5% | 每步耗时 0.692 s（GCS）对 0.686 s（本地），GPU 利用率 71–72%；等数据占比 6.1% 对 5.3%，**两边都略高于 5%** | ✅ 持平（< 5% 的目标两边都没达到） |
| H4 | 内存与数据集规模无关，不会爆 | 系统已用内存峰值：Lance 19.8–20.1 GB，原生 22.7–22.8 GB（**低 11–13%**）。还没做"随规模变化"的实验 | 🟡 |
| H5 | 清洗/修数据更便宜，存储不变大（核实"省 40%"） | 字节数 636.4 MB → 634.2 MB（−0.3%），**"省 40%"不成立**；对象数 1,145 → 31（−97%）；转换耗时 22.9 s。修数据成本还没测 | 🟡 存储节省 ❌，对象数 ✅ |

---

## 1. 对照组

| 组 | 数据路径 | v0 实现 | 已跑 |
|---|---|---|---|
| **A 现状** | GR00T 原生 loader，先把 parquet+mp4 从桶下载到本地（`s5cmd`） | `BACKEND=native`，下载由实验脚本完成 | ✅ LIBERO |
| **B 强基线** | GR00T 原生 loader + 挂载对象存储（SkyPilot `MOUNT_CACHED` / rclone VFS cache） | 本机有 `/dev/fuse`，rclone 还没装 | ⬜ **最需要补** |
| **C 我们** | Lance 直读对象存储 + modality 适配层，不下载 | `BACKEND=lance`，`--dataset_path s3://…` | ✅ LIBERO |
| **D 上限** | Lance 读本地盘 | `BACKEND=lance`，本地目录 | ✅ LIBERO |
| C′（新增） | 原生 GR00T 数据 + 我们的精确续训 loader（不用 Lance） | `BACKEND=native-exact` | ✅ LIBERO |
| E 可选 | LeRobot 原生 Hub 流式 | — | ⬜ |

**C′ 是新加的一组**，作用是把"精确续训"的收益和"Lance 存储格式"的收益分开。它也是最有可能被 NVIDIA 单独接受的部分，因为不依赖 Lance。

**统一条件（已落实）**
- 同一台机器、同一份代码，GR00T 官方 `launch_finetune.py` 只替换两个接口（数据集工厂、Trainer）。模型、处理器、超参全部来自 GR00T。
- 采样顺序：**所有组都用 GR00T 自己的 fast loader 混合/洗牌逻辑**。同一种子下样本流逐比特一致，已用批次指纹在 300 步真实训练里验证。这解决了方案里提到的"打乱方式不同"的问题。

---

## 2. 实验环境（含限制）

| 项 | 值 |
|---|---|
| GPU | 1× A100-SXM4-40GB，驱动 580，torch 2.9.0+cu128，flash-attn 2.8.3 |
| 主机 | 11 vCPU、70 GB 内存；instabox 沙箱，GCP us-central1（Iowa） |
| 对象存储 | GCS 桶 `alignd-7fb6a-groot-stream`（us-central1），通过 S3 兼容接口（HMAC）访问 |
| GR00T | Isaac-GR00T `708617f`（main，仅 N1.7），基座 `nvidia/GR00T-N1.7-3B`，`LIBERO_PANDA` |
| 训练配置 | 与 `examples/finetune.sh` 相同：global batch 32、4 个 dataloader worker、lr 1e-4、warmup 0.05、state dropout 0.2、color jitter、shard 1024、episode_sampling_rate 0.1、seed 42 |
| Lance | lancedb 0.42.0b36、lerobot-lancedb 0.3.1（`4cd7a3e`）、lerobot 0.7.0、PyAV 15.1 |

**环境限制，影响结论的解读：**
1. **gVisor 沙箱不允许 `drop_caches`**：所有"本地盘"冷启动都是热页缓存下测的，对原生本地组有利。对象存储组不受影响。
2. **gVisor 下 PSS 不可靠**：测出的 PSS 总和大于系统已用内存。内存一律报系统已用内存（`MemTotal − MemAvailable`），本机只跑训练，这个数是可信的。PSS 要等换到普通 VM 再测。
3. `/data` 是 9p 文件系统，不是 NVMe：热缓存读约 1 GB/s。D 组只能算"本地盘"，不能算"本地 NVMe"上限。
4. 没有 SkyPilot、没有真正的实例终止：抢占用 SIGTERM 模拟进程级中断，本地盘不清空。
5. 没有 `tc`：gVisor 网络栈也基本不支持 netem，P4b 要换方法（见第 6 节）。
6. 单卡：方案里的 8 卡对照组做不了。

---

## 3. 实验阶段与 v0 结果

### P0 正确性 ✅（前提已满足）

| 检查 | 方法 | 结果 |
|---|---|---|
| P0.1 转换保真 | `scripts/check_blob_bytes.py`：`videos.lance` 里每个 MP4 与源文件比较 SHA-256 | ✅ **758/758 个视频逐字节一致**（625.6 MB），帧数 101,469 = 101,469，episode 379 = 379 |
| P0.2 逐样本（本地） | `scripts/parity_vs_gr00t.py`：用 GR00T 自己的 `LeRobotEpisodeLoader` + `extract_step_data` 对比 | ✅ 40 episode / 240 样本 / 480 帧：state、action 误差 0，像素差 0，文本一致，统计量一致 |
| P0.2 逐样本（GCS 直读） | 同上，Lance 读 `s3://` | ✅ 10 episode / 40 样本 / 80 帧，全部一致 |
| P0.2 一万个随机样本 | 同上，全部 379 个 episode，每个随机抽 27 步 | ✅ **10,233 个样本 / 20,466 帧**：state、action 误差 0，像素差 0，文本一致 |
| P0.2b 样本流 | `scripts/check_exact_resume.py`：GR00T 原版 fast loader vs 精确续训 loader（原生数据）vs 精确续训 loader（Lance），2 worker | ✅ 三者 24 个 batch / 192 个样本逐比特同序 |
| P0.3 训练一致性 | 真实训练 300 步，每步记录批次指纹 | ✅ **Lance 组与 C′ 组 300/300 步批次完全相同** |

**P0.3 的修正**：方案要求"两条 loss 曲线重合（浮点噪声内）"，实际做不到。两次配置完全相同的运行，前 3 步 loss 逐比特一致（1.24173 / 1.25547 / 1.24727），第 4 步起 GPU 非确定性带来约 1e-5 的差异，在 loss 快速下降阶段（第 90–200 步）放大到约 0.1。最后 50 步的平均 loss：Lance 0.293，C′ 0.256，原生 0.231。前两组训练用的批次完全相同，所以这 0.04 的差别全部来自 GPU 非确定性。
**建议改成**：数据一致性用批次指纹证明；训练效果用多种子 loss 区间和仿真成功率评估，或者开确定性模式跑。

### P1 只测数据加载 🟡

| 配置 | 首个 batch | 稳态吞吐 |
|---|---|---|
| 0 worker | 1.7 s | 52 样本/s |
| 4 worker，forkserver，未预热 | 65.2 s | 142 样本/s |
| 4 worker，forkserver，**预热**（v0 默认） | 16.7 s | 145 样本/s |

单卡训练只需要约 47 样本/s，4 个 worker 足够。方案里的完整扫描（`num_workers` ∈ {2,4,8,16} × batch ∈ {32,64} × 3 次）、CPU 占用、每样本读取字节数还没做。

### P2 冷启动 🟡（LIBERO-10，0.64 GB，各 1 次）

| 段（秒） | A 先下载（GCS→本地） | C Lance 直读 GCS | D Lance 本地 | 原生本地（数据已在盘上） |
|---|---|---|---|---|
| 数据准备（下载） | 2.8（3 次：3.0 / 2.6 / 2.8） | 0 | 0 | 0 |
| 启动 → 开始训练（加载模型、建数据集） | 36.6 | 39.2 | 39.4 | 36.5 |
| 首个 batch 等待 | 7.4 | 22.2 | 19.2 | 7.5 |
| **启动 → 第一步合计** | **51.6** | **64.0** | **61.6** | **48.8** |

- 下载实测 **230 MB/s**（1,145 个文件，同区域）。按这个速度外推：RoboCasa 全量 353 GB 约 26 分钟，DROID 约 27 分钟，而且都需要同样大小的本地盘。
- Lance 慢在"首个 batch"：forkserver 起 worker、每个解码线程各自打开表、首次填满洗牌缓冲区（约 1 个 shard，即 1,024 个样本）。这是 v0 最大的短板，见第 5 节。

### P3 稳态训练 ✅（300 步/组，不是方案要求的 2 万步）

| 指标 | 原生本地 | C′ 原生数据+精确续训 | D Lance 本地 | C Lance 直读 GCS |
|---|---|---|---|---|
| 每步耗时（均值 / p95，秒） | 0.690 / 0.752 | 0.694 / 0.755 | 0.690 / 0.777 | 0.692 / 0.781 |
| **等数据占比** | 5.3% | 5.3% | 6.2% | 6.1% |
| GPU 利用率（nvidia-smi） | 71.8% | 70.9% | 72.5% | 71.3% |
| 系统已用内存峰值 | 22.7 GB | 22.7 GB | 19.8 GB | 20.1 GB |
| 显存峰值 | 37.9 GB | 37.9 GB | 37.9 GB | 37.9 GB |

- GPU 是瓶颈，所有组稳态持平。GPU 利用率用的是 nvidia-smi，方案要求的 DCGM SM Active 还没接。
- **读放大**：C 组 300 步（9,600 个样本）期间，网卡收了 746 MB；原生组训练期间（数据已在本地）收了 152 MB，是同期的其他流量。粗估 Lance 每个样本读约 60 KB，因为解码一帧要取到它所在的 GOP。长时间训练会反复读同一份数据，同区域流量免费，但 GCS/S3 按请求次数计费，必须进成本模型。

### P4a 抢占 🟡（单次确定性 SIGTERM，第 150 步，数据在本地，save_steps=100）

| 指标 | Lance（精确续训 + SIGTERM 保存检查点） | 原生 GR00T |
|---|---|---|
| SIGTERM → 进程退出 | 55.0 s（写 24 GB 检查点） | 0.2 s（没有处理，直接被杀） |
| 续训从第几步开始 | 152（不丢步） | 101（回滚到 checkpoint-100） |
| 重做的步数 | **0** | **50** |
| 续训启动 → 第一步 | 131.3 s | 71.2 s |
| 续训首个 batch 等待 | 25.5 s（含重放 151 个 batch 的洗牌状态） | 7.4 s |
| 跑完 300 步总墙钟 | 571 s | 483 s |
| **数据顺序和不中断运行一致的步数** | **300/300** | **100/300**（续训后 0/200） |
| 续训方式 | 重放到原位置 | `Resetting seed to 142`，数据顺序变了 |

- **正确性**是 Lance/C′ 的硬优势：原生 GR00T 每次续训都换一套随机顺序。之前用纯数据仿真测过：1 万个样本、3 次抢占，原生的重复率和漏采率都是 30.8%。
- **时间上 v0 输了**，原因有两个：①SIGTERM 时写 24 GB 检查点要 55 s，比原生重做 50 步（约 35 s）还贵；②续训冷启动慢 60 s。真实场景（每 1,000 步存一次、单步更慢、本地盘被清空要重新下载）下结论可能反转，需要 P4a 完整版来验证。
- 关于方案里提到的隐藏开销：GR00T 设置了 `ignore_data_skip=True`，不会把跳过的 batch 读出来再丢掉，所以 A/B 组没有这项开销。代价是它改成了重新播种，数据顺序会变。

### P4b 网络抖动 ⬜
本机没有 `tc`，gVisor 网络栈也不支持 netem。替代方案：在 S3 端点前放 toxiproxy，注入延迟、丢包、限速和 503 SlowDown。

### P5 清洗与存储 🟡

| 指标 | 原始 LeRobot v2.1 | Lance（copy 模式） |
|---|---|---|
| 字节数 | 636.4 MB | 634.2 MB（−0.3%） |
| 对象数 | 1,145 | **31（−97%）** |
| 转换耗时 | — | 22.9 s（379 episode） |

- "省 40%"不成立：视频原样存储，不重新编码。另一种 external 模式（表里只存索引，MP4 留在原处）只有几十 MB，但那是"不复制视频"，不算压缩。
- `lerobot-lance-doctor` 检查、删除 5% 坏 episode 的修复成本，都还没做。

---

## 4. 工程实现状态

| 方案项 | 状态 | 说明 |
|---|---|---|
| modality 适配层 | ✅ | `LanceShardedSingleStepDataset` / `LanceEpisodeLoader` 实现 GR00T 的 `ShardedDataset` 接口（N1.7 已用这套接口替代 `LeRobotSingleDataset`），支持关节组切片、相机键映射、语言标注、统计量 |
| 转换器 | ✅ | `gr00t-lance-convert`：v2.1（本地或 s3）→ Lance 三表，保留 modality/stats。方案里提的风险（lerobot-lancedb 面向 v3）已解决：转换器直接读 v2.1；运行时与 GR00T 的依赖冲突（huggingface_hub<1.0、numpy 1.26）通过 `_hub_compat` 兼容层解决 |
| 可恢复 sampler | ✅ | `ExactResumeFastShardedMixtureDataset`：在 GR00T fast loader 之上按"已消费 batch 数"重放洗牌缓冲区，不需要额外状态文件，用 HF checkpoint 里的 `global_step` 即可。同时按样本位置播种增强随机数，用逻辑 worker 编号保证轮询顺序一致 |
| 埋点 | 🟡 | 每步写一行 JSONL：步时、等数据时间、内存、GPU 利用率/显存、逐步 loss、批次指纹。还没有：S3 读取字节/请求数、冷启动更细的分段、W&B 上报 |

---

## 5. v0 → v1 优化清单（按收益排序）

| # | 问题（v0 实测） | 方向 | 预期 |
|---|---|---|---|
| 1 | 首个 batch 19–22 s（原生 7.5 s） | ①在模型加载**期间**就启动 DataLoader 迭代器并预取首批；②预热 forkserver 时顺便打开表；③首次填充用 `min_fill` 小缓冲先出数，同时保持顺序确定 | 冷启动追平或超过原生 |
| 2 | 续训冷启动 131 s（原生 71 s） | 拆分各段耗时（加载检查点、forkserver、重放、表打开）；怀疑 forkserver 预加载和加载检查点抢 CPU/IO | −40~60 s |
| 3 | SIGTERM 写 24 GB 检查点要 55 s | 用代价模型决定保不保存（距上次存档的步数 × 单步时间 > 保存耗时才存）；只存模型和优化器；异步上传 S3 | 抢占总耗时追平或超过原生 |
| 4 | 读放大约 60 KB/样本 | 按 GOP 合并同一视频段的样本；本地盘做 LRU 缓存（对象存储做后备）；`sample_index` 粒度调优 | 请求数/字节数 −50% 以上 |
| 5 | 每个解码线程各开一个 reader，各带 16 线程解码池 | 按 worker 共享解码池，控制总线程数 | CPU 占用下降 |
| 6 | forkserver 里 lancedb 报"fork 支持是实验性的"警告 | 预加载时不导入 lancedb，或者改用 spawn 加预热 | 消除隐患 |
| 7 | 内存 PSS 在 gVisor 下测不准 | 换普通 VM 再测 PSS，加"数据集规模 × 内存"曲线 | H4 证据补全 |

---

## 6. 剩余实验排期

> 详细设计（对照组的复位动作、抢占时间表、规模阶梯、埋点、消融、决策点）见 **[第二阶段实验设计（v1）](experiment-design-v1.md)**。下表是摘要。

| 优先级 | 实验 | 数据集 | 需要什么 | 估时 |
|---|---|---|---|---|
| 1 ⏳ | RoboCasa 4 任务（约 32 GB）：P0 抽检 + P2 + P3（A/C/D） | `single_panda_gripper.{CoffeePressButton, TurnSinkSpout, TurnOffStove, CloseDrawer}` | 正在从 HF 下载（免费账号每 5 分钟 1,000 次请求，每个任务约 1.2 万个文件） | 下载约 30 分钟 + 实验 1 小时 |
| 2 | **B 组 rclone 挂载**（VFS cache） | LIBERO + RoboCasa 4 任务 | 装 rclone，验证 gVisor 下 FUSE 能不能用 | 半天 |
| 3 | v1 优化 #1–#3，然后重跑 P2/P4a | 同上 | — | 1–2 天 |
| 4 | P4a 完整版：泊松抢占时间表（每组 3 个种子），**每次抢占都清空本地数据目录**（模拟新实例），检查点存到桶里 | RoboCasa 4 任务 | GPU 约 10–20 小时 | 2–3 天 |
| 5 | 规模点：RoboCasa 全量 353 GB（24 任务混合，GR00T 官方基准）或 DROID | 全量 | 桶存储约 $7/月；A 组本机放不下，需要 ≥ 400 GB 盘的机器；HF 限流下下载 6 小时以上 | 2–3 天 |
| 6 | P1 完整扫描、P4b（toxiproxy）、P5 doctor/修复成本 | LIBERO / RoboCasa | 只要 CPU | 1–2 天 |
| 7 | 仿真成功率评估（LIBERO），作为训练效果的最终证明 | LIBERO-10 | 需要配置 LIBERO 仿真环境；单卡 2 万步需要 GPU 约 4 小时 | 1 天 |

---

## 7. 报告三张图：数据准备情况

| 图 | 需要的数据 | 现在有 |
|---|---|---|
| 冷启动分段条形图 | P2 各组分段计时 × 3 个数据集 | 只有 LIBERO（单次） |
| 抢占频率 × 完成 N 步总成本 | P4a 完整版 | 只有一个点（单次 SIGTERM） |
| 稳态等数据占比 + 内存峰值 | P3 | 只有 LIBERO |

---

## 附录：复现

```bash
# 转换（先用 GR00T 生成统计量，统计量会一起写进 meta.lance）
uv run python -c "from gr00t.data.stats import generate_stats; generate_stats('/data/datasets/libero_10')"
uv run gr00t-lance-convert --src /data/datasets/libero_10 --out /data/datasets/libero_10_lance --video-mode copy

# 正确性
uv run python scripts/check_blob_bytes.py   --v2-dir … --lance-dir …
uv run python scripts/parity_vs_gr00t.py    --v2-dir … --lance-dir … --embodiment-tag libero_sim
uv run python scripts/check_exact_resume.py --v2-dir … --lance-dir …

# 训练（BACKEND=native | native-exact | lance；DATASET 可以是本地目录或 s3://）
BACKEND=lance DATASET=s3://bucket/libero_10_lance S3_ENV_FILE=… OUT=/data/runs/x \
  MAX_STEPS=300 GR00T_BENCH_SKIP_FINAL_SAVE=1 bash scripts/run_gr00t_finetune.sh

# 抢占
BACKEND=lance DATASET=… OUT=… MAX_STEPS=300 SAVE_STEPS=100 KILL_AT_STEP=150 SKIP_REF=1 \
  bash scripts/preempt_experiment.sh
python scripts/compare_fingerprints.py REF_RUN PREEMPTED_RUN/pre
python scripts/summarize_runs.py a=RUN_A b=RUN_B
```

原始结果已存进仓库 [`results/v0/`](../results/v0)：`exp4/`（本地，含抢占）、`exp5/`（GCS）的逐步指标 `bench_metrics.jsonl` 和事件时间线 `events.jsonl`，以及 `results/*.json`（正确性检查）。
