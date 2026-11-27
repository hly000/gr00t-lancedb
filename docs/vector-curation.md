# 向量整理：嵌入列、向量索引、语义去重和相似检索

对应 issue #4。代码在 `src/gr00t_lancedb/vector_curate.py`，命令行入口是
`gr00t-lance-vector-curate`，基准脚本是 `scripts/bench_vector_curation.py`。

## 它做什么

在训练用的那张 `frames.lance` 表上，直接加几列：

| 列 | 类型 | 含义 |
| --- | --- | --- |
| `embedding` | `FixedSizeList<float32, D>`，可为空 | 每帧一个向量（已归一化，按余弦距离比较） |
| `is_semantic_dup` | `bool` | 这一帧是不是某个"代表帧"的近似重复 |
| `semantic_cluster_id` | `int64` | 所属重复簇的代表帧的全局 `index`；代表帧和独有帧指向自己；没参与去重的帧是 `-1` |
| `semantic_nn_dist` | `float32`，可为空 | 到最近邻（范围内）的精确余弦距离；值越大，这一帧越"少见" |

然后可以：

1. **语义去重**：训练时加 `where="NOT is_semantic_dup"`，或者和已有的整理列一起用：
   `where="is_valid AND NOT is_semantic_dup"`。
2. **难例挖掘**：按 `semantic_nn_dist` 从大到小取"和其他示范都不像"的帧。
3. **场景切片**：给一个回合，找出轨迹最像它的 k 个回合，直接得到
   `episode_index IN (...)` 这样的过滤条件，交给 `LanceShardedSingleStepDataset(where=...)`。
4. **混合检索**：向量查询 + SQL 预过滤一次完成，例如"和第 2 回合第 20 帧最像、而且
   `is_valid`、而且不是重复帧"的 10 帧。

整个过程 **不写 `videos.lance` 一个字节**（每一步都按表统计写入字节，`bytes_written_videos`
必须是 0），也不写 `meta.lance`。

过滤只决定"哪些帧可以做锚点"，加载器仍然读取完整回合，所以动作窗口（action horizon）和视频
`delta_indices` 都不会被截断。比如第 5 回合第 10 帧保留、第 11～27 帧被标成重复，那么锚点
(5, 10) 取出的 4 步动作仍然是第 10～13 帧的原始数据，和不过滤时逐字节相同（有测试覆盖）。

## 为什么放在同一张 Lance 表里

常见做法是：把嵌入导出到单独的向量库（FAISS / Milvus / Qdrant 等），在那里做近邻搜索，再写一个
旁路的掩码文件（parquet / json），训练时再写代码把掩码和数据对上。问题是：

- **要同步两份东西**：数据改了（重新整理、追加回合），向量库和掩码文件就过期了，没人会提醒你。
- **要写连接代码**：GR00T 的加载器不认识旁路文件，要么改加载器，要么再生成一份新数据。
- **没有版本**：掩码对应的是哪一版数据，只能靠文件名约定。
- **过滤条件分散**：质量标签在一处，向量在另一处，"有效且不重复且和某回合相似"要在两个系统里拼。

放进同一张 Lance 表以后：

- **存储、向量检索、SQL 过滤、版本、训练读取是同一个东西**。`add_columns` 是零拷贝的列追加，
  旧列和视频都不动；向量索引是这张表的一部分；SQL 条件（`is_valid`、`task_index = 3`、
  `NOT is_semantic_dup`）可以直接作为向量查询的预过滤。
- **每一步都是一个新的表版本**。报告里有 `frames_table_version`，训练时传
  `lance_version=...` 就能固定"用的是哪一次去重结果"，以后重跑去重也不会影响已经开始的实验。
- **训练侧零改动**：结果就是一个 `where` 字符串。
- 原版 `lerobot_lancedb.LanceDBDataset` 照常能打开（行数和视频都没变）。

## 设计说明

### 嵌入列：每帧一行，间隔抽帧时其余为空

`add_columns` 要求新列和表的行一一对应，所以嵌入列是"每帧一行"。`every_n > 1` 时只对每个回合
中每隔 N 帧的关键帧（按回合内位置，第 0 帧一定是关键帧）计算嵌入，其余行存空值：

- 空值只占一个有效位，几乎不占空间；向量检索自动跳过空值；`embedding IS NOT NULL` 就能选出关键帧。
- 不需要单独的"回合表"或"关键帧表"，也就不需要连接，检索结果直接是 `(episode_index, frame_index)`。
- 去重时，两个关键帧之间的帧继承前一个关键帧的标签（`propagated_frames` 记录继承了多少帧）。

之所以没有选"回合级的单独表"：回合级向量会把一个回合里不同阶段的内容平均掉，无法做帧级去重和
帧级检索；需要回合级结论时，用下面的场景切片从帧级结果汇总即可。

嵌入的来源（提供者名字、参数、维度、`every_n`、距离类型）写在这一列的 Arrow 字段元数据里，
键是 `gr00t_lancedb.embedding`。

### 默认嵌入：运动学嵌入（不需要下载模型，CPU 即可）

对回合中的第 t 帧：

```
[ z(state[t]),  z(action[t : t+k]) / sqrt(k),  1 ]
```

- `z` 是全数据集逐维度的标准化（只用有限值的行计算均值和标准差）；
- 动作片段取接下来 k 帧（默认 16，和 GR00T 动作窗口一致），回合末尾不够时重复最后一个动作；
- 末尾的常数 1 让"刚好在均值附近"的帧在余弦距离下也有确定的方向；
- 原始维度超过 `dim`（默认 64）时，用固定种子的高斯随机投影降维；最后做 L2 归一化。
- 任何输入是 NaN/Inf 的帧，嵌入存为空值，不参与检索和去重。

它刻画的是"机器人此刻在什么状态、接下来要做什么"，非常适合找重复录制的示范和长时间静止的片段。

### 可选：视觉嵌入

`VisualEmbeddingProvider(camera, image_embed_fn=... | model_path=...)`：

- 通过 `lerobot_lancedb.LanceDBDataset` 解码关键帧（只读 `videos.lance`）；
- `image_embed_fn` 是你自己的函数：输入一组 `HWC uint8` 图像，输出 `float32[N, D]`；
- 或者给一个**本地** SigLIP / CLIP / DINOv2 目录（`model_path`），用 `transformers` 加载，
  `local_files_only=True`，**不会联网下载**。SigLIP/CLIP 用 `get_image_features`，DINOv2 用
  `pooler_output` 或 CLS 向量。没装 `transformers` 时会给出清楚的报错。
- 解码和模型推理是主要耗时，建议配合 `every_n`（比如 5 或 10）使用。

测试里用一个"图像平均颜色"的小函数验证了整条视觉路径（解码、间隔抽帧、空值、0 字节写视频），
没有下载任何权重；真实 SigLIP/DINOv2 推理没有自动化测试。

也可以传任意函数 `fn(episode_index[], frame_index[], tabular) -> float32[N, D]`
（`tabular` 里有 `index`、`state`、`action`），会被自动包装成提供者。

### 向量索引：IVF-PQ（余弦），小表退回精确搜索

`build_vector_index` 根据向量个数 N 和维度 D 选参数：

- 分区数 ≈ √N，但不超过 N/256（保证每个分区有足够的训练样本）；
- 子向量数让每段约 8 维（D ≥ 256 时约 16 维），并且能整除 D；例如 D=64 → 8 段，D=768 → 48 段。
- 非空向量少于 `min_rows`（默认 256，PQ 的 8 位码本至少需要 256 个训练样本）时，不建索引，
  使用精确的暴力搜索；如果 Lance 建出来的索引一行都没收录，同样退回。报告里的 `search_mode`
  （`ivf_pq` / `flat`）和 `fallback_reason` 记录了实际用的是哪一种。

### 去重算法

候选帧 = 嵌入非空 且 满足 `where` 的帧。建议先跑表格整理（`gr00t-lance-curate`），再用
`--where is_valid` 去重，这样每个重复簇的"代表帧"一定是有效帧。

三种范围（`scope`）：

- `cross_episode`（默认）：每帧找**其他回合**里最近的 k 帧（默认 10），余弦距离不超过阈值
  （默认 0.02）的算作近似重复。走向量索引：按整回合分组（约 `query_batch_size` 帧一组），
  每组一次批量向量查询，预过滤 `episode_index NOT IN (本组回合)`；组内不同回合之间用内存精确比较。
  两部分合并后用**精确距离**重新打分、取前 k，所以结果和分组方式无关（测试验证了"一组包含所有
  回合、0 次 Lance 查询"与"每组约 1～2 个回合"结果完全一致）。
- `within_episode`：回合内的**连续静止段**——从某帧开始，后面每一帧都和这帧距离不超过阈值，
  就属于同一段；只保留段首。这是精确计算，不用索引，代价和回合长度成正比。注意它不处理
  "隔了很久又回到同一个姿态"的情况。
- `all`：两者都做。

保留谁：按 `(episode_index, frame_index)` 从小到大遍历，还没被认领的帧成为代表，并认领它所有
还没被认领的近邻。所以：

- 每个簇**恰好保留一帧**，而且顺序固定，结果可复现；
- 每个重复帧都**直接**和它的代表在阈值以内，不会出现 A 像 B、B 像 C、结果 A 和 C 被串成一簇的
  "链式传递"。

`semantic_nn_dist`：`cross_episode` / `all` 下是到其他回合最近帧的精确距离；`within_episode`
下是到相邻帧的距离。

阈值和数据有关。报告里的 `nn_dist_quantiles`（p01/p05/p25/p50）是所有候选帧最近邻距离的分位数，
先用默认值跑一次，看分位数再调。

### 场景切片

`slice_episodes_by_similarity(root, episode, k)`：从查询回合里均匀取最多 64 个关键帧，一次批量
向量查询（预过滤：用户的 `where` 且不是查询回合本身）。对每个候选回合，分数 = 各查询帧到该回合的
最近距离的平均值（某查询帧的结果里没出现这个回合时，按该查询结果里最远的距离计）。分数越小越像；
`coverage` 是"有多少比例的查询帧在这个回合里找到了近邻"。

## 用法

### 命令行

```bash
# 1. 计算嵌入（默认运动学嵌入：动作片段 16 帧，降到 64 维）
gr00t-lance-vector-curate embed --root /data/libero_10.lance --horizon 16 --dim 64 --out-json embed.json

#    视觉嵌入（本地模型，不下载；每 5 帧取 1 帧）
gr00t-lance-vector-curate embed --root /data/libero_10.lance --provider visual \
    --camera front --model-path /models/siglip-base-patch16-224 --every-n 5

# 2. 建索引（小表会自动记录为精确搜索）
gr00t-lance-vector-curate index --root /data/libero_10.lance --out-json index.json

# 3. 先做表格整理，再在有效帧里去重
gr00t-lance-curate --root /data/libero_10.lance
gr00t-lance-vector-curate dedup --root /data/libero_10.lance --threshold 0.02 \
    --scope cross_episode --where is_valid --out-json dedup.json

# 4. 检索
#    和第 2 回合最像的 5 个回合（输出里有 where_predicate）
gr00t-lance-vector-curate search --root /data/libero_10.lance --episode 2 --level episode --k 5
#    和第 2 回合第 20 帧最像的 10 帧，只看有效且不重复的帧
gr00t-lance-vector-curate search --root /data/libero_10.lance --episode 2 --frame 20 --k 10 \
    --where "is_valid AND NOT is_semantic_dup"
#    最"少见"的 50 帧（难例）
gr00t-lance-vector-curate search --root /data/libero_10.lance --level rare --k 50
```

每个子命令都把 JSON 报告打印到标准输出，`--out-json` 另存一份。远程数据集用
`--root s3://... --storage-option KEY=VALUE`。

### Python

```python
from gr00t_lancedb import LanceShardedSingleStepDataset
from gr00t_lancedb.vector_curate import (
    KinematicEmbeddingProvider, add_embedding_column, build_vector_index,
    tag_semantic_duplicates, slice_episodes_by_similarity, hybrid_search,
)

root = "/data/libero_10.lance"
add_embedding_column(root, KinematicEmbeddingProvider(horizon=16, dim=64))
build_vector_index(root)
rep = tag_semantic_duplicates(root, threshold=0.02, where="is_valid")
print(rep.dup_frames, rep.duplicate_episodes, rep.nn_dist_quantiles)

# 去重后训练（固定这次去重对应的表版本）
ds = LanceShardedSingleStepDataset(
    root, embodiment_tag, modality_configs,
    where="is_valid AND NOT is_semantic_dup",
    lance_version=rep.frames_table_version,
)

# 场景切片 -> 训练子集
sl = slice_episodes_by_similarity(root, episode=2, k=20)
ds_scene = LanceShardedSingleStepDataset(root, embodiment_tag, modality_configs, where=sl.to_where())

# 帧级检索 -> 逐回合锚点
hits = hybrid_search(root, episode=2, frame=20, k=50, where="NOT is_semantic_dup")
hits.anchors_by_episode()   # {episode_index: np.ndarray[frame_index]}
hits.to_where("frame")      # "(episode_index = 7 AND frame_index IN (...)) OR ..."
```

注意：这些函数会在当前进程里启动 lancedb（Tokio 运行时不能安全 fork）。请在训练开始前单独跑
（命令行或独立脚本），把得到的 `where` 字符串交给训练，不要在训练主进程里、DataLoader 启动前调用。

## 基准

`scripts/bench_vector_curation.py` 跑完整流程（嵌入 → 索引 → 去重 → 场景切片 → 训练过滤查询），
并和"外部工作流"对比：把嵌入导出成 `.npy` + `.parquet`，用 numpy 分块矩阵乘做暴力余弦 kNN
（同样的去重规则），写旁路掩码 parquet，再读回来得到训练锚点。两边共用同一份嵌入，只比较
检索/去重/写入/接入训练的成本，并报告两边重复帧集合的 Jaccard 重合度。

```bash
python scripts/bench_vector_curation.py --v2-dir /data/libero_10_v2 --out-json bench.json
python scripts/bench_vector_curation.py --lance-root /data/libero_10.lance --threshold 0.01
```

默认在临时目录里的副本上运行，不改动原数据（`--in-place` 才会直接改）。

在一台 Apple M5 笔记本（只用 CPU）、合成数据（随机游走轨迹，8 维状态/动作，注入 5 个带 1e-3 噪声的
重复回合，`--threshold 0.01`，默认 `--horizon 16 --dim 64`）上的一次结果：

| 帧数 | 索引 | Lance：建索引 + 去重 | 外部：导出 + 暴力 kNN + 写掩码 | 重复帧（两边） | Jaccard | 写入 `videos.lance` |
| --- | --- | --- | --- | --- | --- | --- |
| 2 万 | IVF-PQ，78 分区，8 段 | 1.57 s | 1.58 s | 1000 / 1000 | 1.0 | 0 字节 |
| 10 万 | IVF-PQ，316 分区，8 段 | 7.96 s | 34.06 s | 1015 / 1015 | 1.0 | 0 字节 |

暴力 kNN 的代价随帧数平方增长，向量索引接近线性；数据越大差距越大。训练侧的差别在于：Lance 是
一个 `where` 字符串（10 万帧锚点查询约 3 ms），外部方案需要额外的连接代码和一个要手动保持同步的文件。

## 限制

- **阈值和数据相关**。默认 0.02 只是起点，请看 `nn_dist_quantiles` 调整；阈值太大会把正常的
  慢速运动也当成重复。
- **k 要大于预期的副本数**。同一帧如果有 20 份副本、k=10，近邻列表放不下所有副本，可能会多保留
  几个代表。
- **IVF-PQ 是近似搜索**。去重对候选会用精确距离重新打分，所以标签只受召回率影响；Lance 的 k-means
  初始化没有种子，重建索引后召回可能有极小差异。小表（或 `--mode flat`）是精确的、完全确定的。
  测试和基准里近似结果与精确结果一致。
- **内存**：去重会把嵌入整列读进内存（N × D × 4 字节，100 万帧 × 64 维约 256 MB），运动学嵌入也会
  读入全部状态/动作列。
- **`within_episode` 只识别连续静止段**，不识别隔一段时间后回到同一状态。
- **重新计算嵌入会删除去重列**（避免用到过期标签），之后要重新跑 `dedup`；依赖
  `NOT is_semantic_dup` 的训练配置在这之间会因为列不存在而报错（这是故意的）。
- **`(episode_index, frame_index)` 必须唯一**（LeRobot 数据本来如此），否则报错。
- **远程数据集（s3:// 等）无法在本地统计写入字节**，报告里 `bytes_measured=false`；本模块从不
  打开 `videos.lance` 写入，这一点和本地一样。
- **视觉嵌入很慢**（解码 + 模型推理），且需要本地模型目录；真实模型推理没有自动化测试。
- 不要在别的进程同时写同一张 `frames.lance`（和表格整理一样，没有做跨进程写锁）。
