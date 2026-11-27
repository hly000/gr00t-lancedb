# 相关工作与竞品调研

> **日期**：2026-11-27
> **一句话结论**："不下载、直接从对象存储训练"在 LeRobot 上已经由 Hugging Face 和 LanceDB 做成了；NVIDIA 也刚解决了 GR00T 本地训练的数据加载瓶颈。还没有人做的是 **GR00T 的精确续训**和**给 GR00T 用的 Lance 数据层**，这是本项目剩下的差异点。
> 标注：✅ 读过原文核实　⚠️ 厂商自报数字　❔ 没有直接来源，是推断

---

## 1. 直接重叠：同样的问题，在 LeRobot 上已经有官方实现

### 1.1 LeRobot v0.7.0（Hugging Face，2026-11-05）✅

- `--dataset.streaming=true`：新的 reader 只取需要的视频字节，数据来自 Hub 或 HF Storage Buckets，不落盘。
- 官方博客原话："Every frame is visited exactly once per epoch, and runs resume deterministically"。也就是说，LeRobot 的数据集和训练流程已经能从远端流式训练，并且能确定性续训（博客没有列出具体模型）。
- 原生读取 Lance 表（MP4 原样存为 blob，不重新编码）：`LeRobotDataset("lance-format/pusht-lance")`。

### 1.2 LanceDB ✅ ⚠️

[官方文档](https://docs.lancedb.com/training/lerobot)称 LeRobot 可以直接从对象存储（S3、GCS、Hub、Storage Buckets）上的 Lance 训练。下面的基准数字是厂商自报：

| 场景 | Lance 直读对象存储 | 先下载再训练 |
|---|---|---|
| SmolVLA，DROID，10,000 步，8×H100 | 1 小时 27 分；495 样本/s；等数据占比 1.7% | 下载 384 GB 后训练：2 小时 00 分；361 样本/s；等数据占比 37.4% |

两边最终 loss 都是 0.2380。文档里**没有** GR00T、Isaac-GR00T、`modality.json`，也没有抢占后的精确续训。

**和本项目 v0 对比时要注意**：上表"先下载、等数据占比 37.4%"用的是 LeRobot 自己的加载器。本项目 v0 测的 GR00T 原生加载器，等数据占比只有 5.3%（见 [experiment-plan.md](experiment-plan.md) P3）。LanceDB 的数字不能直接搬来讲 GR00T。

---

## 2. GR00T 这边

### 2.1 NVIDIA Isaac-GR00T 官方 ✅

- **新的默认数据加载器（2026-11-08，commit `7aa0bca`）**：直接读 LeRobot 数据集，只解码每个 shard 采样到的帧，并限制每个 worker 的 FFmpeg 线程数。BridgeV2、8×H100 上，等数据占比从 74% 降到 0.1% 以下，训练快约 5 倍，收敛不变。`--data_loader stock` 可以切回旧加载器。
- **只读本地路径 ❔**：用 GitHub 代码搜索查 Isaac-GR00T，除了 `uv.lock` 外搜不到 `s3://` 或 `fsspec`（2026-11-27）。官方没有明说不支持远端。
- **本项目 v0 的基线已经包含它**：v0 用的 `708617f`（2026-11-17）比 `7aa0bca` 晚 6 个 commit。所以 v0 是在跟 NVIDIA 最新的加载器比，对比是公平的。
- **续训仍然会重新播种**：`gr00t/experiment/trainer.py`（main）里设置了 `ignore_data_skip = True`，然后 `new_seed = seed + global_step`，并打印 "this will make the experiment non-reproducible"。本项目的精确续训解决的就是这个问题，截至调研日主线没有改。
- **[issue #800](https://github.com/NVIDIA/Isaac-GR00T/issues/800) 透露的态度**：一个第三方闭源 fork（knonik）声称能让 GR00T N1.7 训练快 3 倍。维护者拒绝的核心理由是：转换由闭源工具完成，**没法验证转换后的数据和原始数据一致**（另外还有专有许可、需要登录）。NVIDIA 选择自己写一个直接读 LeRobot 的加载器。所以往 NVIDIA 推需要转格式的方案时，必须能拿出转换前后一致的证据。

### 2.2 Backblaze：`aistore-b2-gr00t-robotics-training` ✅

- 用 GR00T N1.7 微调。B2 是数据的正本，NVIDIA AIStore 做读缓存，数据先暂存到节点本地目录，GR00T 读本地盘。
- 仓库自己写着 "This repository publishes no numbers"、"Training is not faster"。会把 SIGINT/SIGTERM 转给训练进程，只同步完整的检查点，但没有描述续训。
- 思路相当于本项目的 A/B 组（先暂存或缓存到本地）。

---

## 3. 通用组件：能替代其中一部分

| 项目 | 能做什么 | 和本项目的差别 |
|---|---|---|
| [MosaicML Streaming](https://docs.mosaicml.com/projects/streaming/en/latest/distributed_training/fast_resumption.html)（Databricks）✅ | 从对象存储流式读取；在 epoch 中间确定性续训，不用重放已经读过的样本 | 要先把数据转成它自己的格式；没有 GR00T、`modality.json` 适配，也不保证和 GR00T 加载器的采样顺序一致 |
| [torchdata StatefulDataLoader](https://pytorch.org/data/beta/torchdata.stateful_dataloader.html) ✅ | 保存和恢复数据加载器的状态；默认按已产出的 batch 数快进 | 只处理同一进程里的多个 worker，跨 rank 的状态要自己管 |
| [SkyPilot](https://docs.skypilot.co/en/stable/examples/managed-jobs.html) 托管 Spot 任务 + `MOUNT_CACHED` ✅ | 抢占后自动恢复任务；桶挂载加本地缓存 | 就是本项目的 B 组；没有找到 GR00T 的现成配置 |
| [Backblaze `lerobot-b2-streaming`](https://github.com/backblaze-b2-samples/lerobot-b2-streaming) ✅ | 用 S3 Range 请求从 B2 流式读取 LeRobot v3 数据 | 只支持 LeRobot 0.4.x（`>=0.4.4,<0.5`）；LeRobot 0.7 加 Lance 已经覆盖了这个缺口 |

---

## 4. 对本项目的影响和建议

1. **"零下载 / 冷启动快"不再是差异点。** LeRobot 加 LanceDB 已经占了这个位置；NVIDIA 也解决了 GR00T 在本地的稳态瓶颈。建议**提前采用** [experiment-design-v1.md](experiment-design-v1.md) 第 7 节 E1 决策点的结论，把报告主线改成"精确续训 + 不爆内存 + 修数据便宜"。依据是外部竞品的情况，不是 E1 的数据（E1 还没跑），冷启动只作为大数据集上的补充证据。
2. **把真正没人做的三件事放到前面：**
   - GR00T 的精确续训：主线仍然会重新播种。确定性续训本身早有通用实现（第 3 节的 MosaicML Streaming、torchdata），LeRobot 0.7 也只管它自己的训练流程；本项目的不同在于直接接在 GR00T 的加载器上。
   - 和 NVIDIA 新加载器的样本流逐比特一致：v0 的 P0.2b 已经验证（LIBERO，2 个 worker，24 个 batch / 192 个样本）。
   - 同一份 Lance 数据，LeRobot 和 GR00T 都能直接训练。
3. **把精确续训单独提给 NVIDIA。** C′ 组（原生数据 + 精确续训，不用 Lance）不涉及转格式，避开了 issue #800 里"转换没法验证"的问题，改动也小，是最可能被接受的一块。以后推 Lance 时，拿 v0 的 P0 正确性检查（视频逐字节一致、1 万个样本逐样本一致）回应同一个顾虑。
4. **和 LanceDB 合作，不要和他们竞争。** 他们的生态里正好缺 GR00T。可以提出让本项目作为 GR00T 后端，进入他们的文档或 `lerobot-lancedb`，并一起做 GR00T 上的基准测试。
5. **依赖风险（待确认）**：[huggingface/lerobot#4567](https://github.com/huggingface/lerobot/pull/4567)（2026-09-20 合并）✅ 写道 `LeRobotLanceDataset` 读数据已经不需要了，`lerobot-lancedb` 以后只用来转换格式。本项目的 `LanceEpisodeLoader` 依赖的是另一个类 `lerobot_lancedb.LanceDBDataset`，它会不会也停止维护还没查。如果会，要迁移到 LeRobot 原生的 Lance reader。
6. **要盯的两件事**：LanceDB 自己加 GR00T 支持；NVIDIA 修好续训时的重新播种。这两件事技术门槛都不高，任何一件发生，对应的差异点就没了。

---

## 来源

- [LeRobot v0.7.0 发布博客](https://huggingface.co/blog/lerobot/lerobot-release-v070)
- [LanceDB：用 Lance 训练 LeRobot](https://docs.lancedb.com/training/lerobot)
- [Isaac-GR00T issue #800](https://github.com/NVIDIA/Isaac-GR00T/issues/800)
- [Isaac-GR00T `gr00t/experiment/trainer.py`](https://github.com/NVIDIA/Isaac-GR00T/blob/main/gr00t/experiment/trainer.py)
- [backblaze-b2-samples/aistore-b2-gr00t-robotics-training](https://github.com/backblaze-b2-samples/aistore-b2-gr00t-robotics-training)
- [backblaze-b2-samples/lerobot-b2-streaming](https://github.com/backblaze-b2-samples/lerobot-b2-streaming)
- [huggingface/lerobot#4567](https://github.com/huggingface/lerobot/pull/4567)
- [MosaicML Streaming：Fast Resumption](https://docs.mosaicml.com/projects/streaming/en/latest/distributed_training/fast_resumption.html)
- [torchdata StatefulDataLoader](https://pytorch.org/data/beta/torchdata.stateful_dataloader.html)
- [SkyPilot Managed Jobs](https://docs.skypilot.co/en/stable/examples/managed-jobs.html)
