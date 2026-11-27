# v1 下一批实验：代码改动说明（分支 `feat/v1-next`）

这份文档说明 `feat/v1-next` 相对 `main` 改了什么、每项改动对应哪个实验/Issue、怎么开关，以及本地（无 GPU）已经验证到什么程度。所有开关都是环境变量，默认值就是推荐配置；做 E8 消融时逐项关掉即可。

## 0. 分支构成

- 合并了 `feat/train-perf-v1`（懒解码洗牌缓冲、fork 安全的共享 reader、轻量应急存档）和 `feat/lance-curation`（表格清洗 + SQL 过滤）。
- 在合并结果上完成计划里的 Batch 1、Batch 2，以及 Batch 3（实验脚手架脚本）、Batch 4（向量去重）。

## 1. Batch 1：合并底座 + 抢占进 30 秒

### 1.1 `--lance-where` 不再破坏 fork（Issue #1）

| 以前 | 现在 |
|---|---|
| 主进程里 `lancedb.connect()` 查 `frames.lance`，启动了 Tokio 运行时，只能退回 forkserver | 查询放进一个**一次性子进程**（`python -m gr00t_lancedb._isolated`），只把 `(episode_index, frame_index)` 两列整数数组传回来；主进程从头到尾不导入 `lancedb` |
| 主进程顶层就导入 `lerobot_lancedb`（本机实测 1.8–2.8 秒） | 包改成按需导入；数据集构建阶段完全不碰 lerobot / lancedb / av |

- 子进程开销：本地数据集或 `s3://`/`gs://` 只导入 `lancedb`，本机约 **0.7 秒**；`hf://` 才走 lerobot（约 2.7 秒）。
- 远端数据集第一次启动（本地还没有 `meta/` 缓存）也走子进程，所以冷启动同样可以用 fork。
- 指定了版本号的 `where` 结果会缓存在 `~/.cache/gr00t_lancedb/where/`（`GR00T_LANCE_CACHE_DIR` 可改）：抢占后续训**完全跳过查询**。
- fork 前钩子：第一次 fork 前自动等待后台导入完成（或同步导入一次），worker 直接继承已导入的模块；顺带解决了 macOS 上首次在子进程初始化 ObjC 导致崩溃的问题。
- 启动器默认在 GR00T 加载模型时**后台线程**导入 lerobot-lancedb（`GR00T_LANCE_BG_IMPORT=1`）。

本机 CPU 模拟（合成小数据集，2 个 fork worker，用 3 秒 sleep 代表加载模型）：

| | 关键路径（不含模型加载） |
|---|---|
| `feat/train-perf-v1` | 1.96–2.62 s |
| `feat/v1-next`（后台导入） | 0.37–0.41 s |

注意 sleep 会释放 GIL，真实的模型加载只能部分重叠，A100 上的收益要用 E1 的 `startup_summary` 实测（预期能抹掉 `lance_v1` 比 `native` 多出的约 2.4 秒）。

### 1.2 清洗版本和统计量一一对应（Issue #2）

- 每次 `gr00t-lance-curate` 都在 `meta.lance`（和本地 `meta/`）里新增：
  - `gr00t_stats_v{N}.json`、`stats_v{N}.json`、`curation_report_v{N}.json`（N = 这次清洗后 `frames.lance` 的版本号）
  - `curation_versions.json`：每条记录覆盖一段版本区间 `[frames_version_from, frames_version]`；第一次清洗时会把**原始数据的统计量**也存一份。
- `meta.lance` 改为只写有变化的行（`merge_insert`），历史版本都还能 `checkout`。
- 训练时 `GR00T_LANCE_VERSION=N`（或 `where` 实际查询时的版本）会同时锁定：锚点、读到的行（reader 也 checkout 到 N）、q01/q99 统计量。
- 远端数据集如果本地 `meta/` 缓存过期，会通过子进程从最新 `meta.lance` 取对应版本的统计量。
- `GR00T_LANCE_AUTO_PIN=1`（默认）：首次启动时把 `where` 实际用的版本写进 `<output_dir>/lance_pins.json`，续训自动沿用，中途有人再清洗也不影响正在跑的实验。

### 1.3 应急存档压缩 + 自校准代价门控（Issue #6）

`GR00T_PREEMPT_CKPT_DTYPE`（默认 `auto`）：

| 档位 | 每个可训练参数字节数 | GR00T-N1.7（1.62B 可训练）估算 | 是否精确 |
|---|---|---|---|
| `fp32` | 12 | 约 19.4 GB | 逐比特一致 |
| `bf16` | 6 | 约 9.7 GB | 有损（约 3 位有效数字） |
| `int8` | 4 | 约 6.5 GB | 有损 |
| `auto` | — | 选**能在截止时间内写完的最精确档位** | — |

要点（和之前方案描述不同的地方，如实说明）：

- **bf16 / int8 都是有损的**，恢复时会还原成 fp32 类型，但数值已经被舍入，续训和不中断的运行不再逐比特一致。存档里带 `lossless` 标记，加载时会打警告；E4 分析时要区分。
- `int8` 档：权重 bf16；一阶矩分块 int8；二阶矩在开方后量化为 uint8 并**向上取整**，保证还原出的二阶矩只会偏大，Adam 的更新量只会变小、不会因为被舍成 0 而爆掉。块缩放因子用 fp32（fp16 会把很小的矩下溢成 0）。
- `GR00T_PREEMPT_EXACT_WEIGHTS=1`：任何档位都保留 fp32 权重，只压缩优化器状态。
- 编码按块在 GPU 上做，单步临时显存不超过约 64 MB；写完先 `fsync` 再原子改名（防止被停机的 Spot 实例丢页缓存）。

代价门控不再写死 10 秒：

- 预计保存时间 = 固定开销（`GR00T_PREEMPT_SAVE_OVERHEAD_S`，默认 1 秒）+ 存档字节数 ÷ 实测写盘带宽。
- 带宽来源按可信度：真实应急存档 > 常规 HF 存档（`on_save` 时按目录大小 ÷ 耗时）> 训练开始时后台 `fsync` 探测写（`GR00T_PREEMPT_PROBE_MB`，默认 64 MB）> 先验 400 MB/s（`GR00T_PREEMPT_PRIOR_WRITE_MBPS`）。结果存在 `<output_dir>/preempt_calibration.json`，续训后直接沿用。
- 决策规则：有上一个存档且"重做浪费的时间 ≤ 预计保存时间"→ 不存；预计超过 `GR00T_PREEMPT_DEADLINE_S`（默认 25 秒）→ 不存（写一半被 SIGKILL 纯属浪费）；一个存档都没有时总是尝试。
- 每次决策写一行到 `<output_dir>/preempt_events.jsonl`（候选档位的预计耗时、实际字节数和耗时），E3 可以直接用来画图。

### 1.4 异步预热时机、7 段启动时间、单样本指纹

- `GR00T_LANCE_ASYNC_PREFETCH=auto`（默认）：**只在续训时**提前启动数据加载器，让首批解码和加载 10–24 GB 检查点并行；新开训练不再白白多一个线程（上次实测这项让冷启动慢了 0.5 秒）。`1` / `0` 可强制。
- `startup_summary` 现在有：`imports_s`、`model_load_s`、`dataset_build_s`、`trainer_init_s`、`ckpt_load_s`（再细分 `ckpt_weights_s` / `ckpt_optim_s` / `ckpt_rng_s`）、`dataloader_create_s`、`first_batch_wait_s`、`first_step_s`，另有 `lance_isolated_s`（子进程耗时）、`lance_preload_s`、`lance_where_cache_hit`、`lance_frames_version`。
- `GR00T_SAMPLE_FINGERPRINTS=1`：每步记录 batch 里每个样本的哈希（`sample_fps`）。
- `scripts/analyze_e4_samples.py --ref <不中断> --test <被抢占>`：按"每个 global_step 最后一次执行"重建有效样本流，输出顺序一致率、重复率、漏采率、第一次分叉的步数、被重做的步数。

## 2. Batch 2：多 epoch 读放大

### 2.1 冷热分级解码批

- `GR00T_LANCE_FIRST_DECODE_BATCH=32`（每次启动/恢复后的第一块），`GR00T_LANCE_STEADY_DECODE_BATCH=128`（稳态）。旧变量 `GR00T_LANCE_DECODE_BATCH_SIZE` 仍可用作稳态值。
- 分块只改变"多少锚点一起解码"，锚点仍按同一个随机数序列抽取，**样本顺序逐比特不变**（有单测覆盖不中断和中途恢复）。
- 跨 episode 的批量读取本来就按 `(episode, frame)` 排序去重后再按视频文件合并字节区间，稳态批变大后同一个视频/GOP 被合并的机会更多。
- 每个 worker 的视频解码器缓存默认从 16 提到 64（`GR00T_LANCE_VIDEO_DECODER_CACHE`）。**内存要在 A100 上复测**（上次 16 时 pss 约 17.25 GB）。

### 2.2 本地盘 GOP 缓存（Issue #3）

- `GR00T_LANCE_VIDEO_CACHE_GB=20`、`GR00T_LANCE_VIDEO_CACHE_DIR=/local/nvme/gop`；默认只对对象存储数据集生效（本地 Lance 本来就在本地盘，`GR00T_LANCE_VIDEO_CACHE_FORCE=1` 可强制，用于测试）。
- 缓存单位是 **GOP**：用 `videos.lance` 里已有的关键帧字节位置 `kf_positions` 切分（过长的 GOP 按 1 MB 再切），只缓存训练真正读到的 GOP，不像 rclone 那样缓存整个 MP4。
- 一次请求里缺失的相邻 GOP 合并成一个远端区间，仍是一次 `fetch_blob_ranges` 调用。
- 多 worker 安全：临时文件 + `os.replace` 原子写；`flock` 保护的字节账本控制总量，超额时按最近使用时间淘汰到 90%。
- 返回的字节和远端完全一样，解码结果逐比特一致（本地和 moto S3 都测过）；第二个 epoch 远端 blob 读取次数为 0。

### 2.3 对象存储请求与读放大埋点

- 每步 `step.io`（累计值）：`lance_blob_calls` / `lance_blob_ranges` / `lance_blob_bytes`（缓存之后真正发给 Lance 的）、`requested_ranges` / `requested_bytes`（reader 想要的）、`gop_cache_hits/misses` 及字节数、`frames_take_calls/rows`、`net_rx_bytes`（`/proc/net/dev`，仅 Linux）。
- `scripts/summarize_runs.py` 会算出每步请求数、每步 MB、缓存命中率，以及按 `--usd-per-1k-get`（默认 0.0004 美元/千次）折算的**每 1000 步请求费**。
- 说明：`lance_blob_ranges` 是交给 Lance 的区间数，Lance 内部还会合并/拆分，和对象存储账单上的 GET 次数接近但不完全相等。需要精确值时打开 `RUST_LOG=object_store=debug` 另行统计。

## 3. Batch 3 / Batch 4

- Batch 3 脚本说明见 `docs/v1-scripts.md`。
- Batch 4 向量去重见 `docs/vector-curation.md`。

## 4. 本地验证范围

- 本机（macOS，CPU，Python 3.12，lancedb 0.42.0b36，torch 2.15）全部单测通过；没有 GPU、没有真实 GR00T。
- 下面这些**只能在 A100 上验证**：真实冷启动各段耗时、19 GB / 9.7 GB / 6.5 GB 存档的实际写盘时间、解码器缓存 64 的内存占用、稳态批 128 的吞吐、GCS 上多 epoch 的请求数下降幅度。
