# 数据飞轮快速验证：GR00T N1.7 + LIBERO-10

这份文档说明怎么用一个种子、约 10 个 A100 小时，回答一个问题：**把失败回合检索到的训练帧写回 `frames.lance`、按权重重采样后，成功率涨不涨。**

## 设计

- **G0**：基线。均匀采样训满 N 步（默认 2 万步，批 32）。第 0.7N 步（14k）存一个完整检查点（带优化器、调度器、RNG），另外存 5k、10k、20k 的权重。
- **G3**：从 G0 的 14k 检查点接着训剩下的 0.3N 步。学习率曲线不变，唯一的区别是采样按 `flywheel_weight` 列加权。
- 两组前 70% 的训练完全相同，差异只来自加权这一个变量。G3 只多花 6k 步。
- 清洗和去重（G1、G2）不在这一步做，单独看飞轮本身。

## 代码在哪

| 模块 | 作用 |
|---|---|
| `gr00t_lancedb/flywheel/weights.py` | 把权重列映射到训练锚点，生成按权重有放回抽样的确定性流 |
| `gr00t_lancedb/exact_resume.py` | `ExactResumeMixin(weighted_anchors=...)`：加权流照样精确续训，跳过已训样本只重放整数抽样 |
| `scripts/launch_finetune_lance.py` | `GR00T_LANCE_WEIGHT_COLUMN` / `GR00T_LANCE_WEIGHT_VERSION` 打开加权；`GR00T_KEEP_CKPT_STEPS` / `GR00T_KEEP_CKPT_WEIGHTS_ONLY` 把指定检查点硬链接到 `keep/`，不被 `save_total_limit` 轮转删掉 |
| `scripts/flywheel/record_libero_rollouts.py` | 在 LIBERO 环境里录回合：每个回合一个固定种子，两个检查点用同一个 `--base-seed` 就能逐回合配对；记录每次决策的状态、动作块和失败回合的缩略图 |
| `gr00t_lancedb/flywheel/mining.py` | 失败挖掘与写回（见下） |
| `gr00t_lancedb/flywheel/paired_eval.py` | 逐回合配对比较、McNemar 精确检验、最难 3 个任务、判定 |
| `scripts/flywheel/fast_validation.sh` | 按阶段跑完整流程 |

### 失败挖掘怎么做

1. 失败回合的每个决策点，用和 `frames.lance` 嵌入列相同的运动学嵌入（同一组 z-score 参数、同一个随机投影）编码。
2. 每个决策点和同任务训练帧的最大余弦相似度，构成一条相似度曲线。
3. **分叉点**：平滑后的相似度第一次比此前最高值低 `divergence_drop`（默认 0.15）的位置。它通常比超时早很多，避免拿"卡住发呆"的末尾画面去检索。曲线从没掉这么多时，取覆盖最差的点。
4. 分叉点前 2 个、后 1 个决策点去检索，每个取 top-8 训练帧。相似度不低于 `min_match_sim` 的帧记分 `sim × exp(-|k-k*|/τ)`，再按三角核扩散到该示范前后 8 帧。
5. 相似度都低于 `min_match_sim` 的失败记为**覆盖盲区**：数据里没有类似状态，重采样救不回来，只报告，不加权。
6. 所有失败的分数按帧求和并归一到 [0, 1]，权重为 `1 + α·score`（α 默认 3），截断到 [0.25, 4]，再归一到均值 1。
7. 写回两列 float32：`flywheel_fail_score` 和 `flywheel_weight`。用 Lance 的零拷贝加列，`videos.lance` 写入字节为 0，报告里有每张表的写入量。

## 运行

前提：Isaac-GR00T 在 `$GR00T_ROOT`，已经跑过 `gr00t/eval/sim/LIBERO/setup_libero.sh`；`libero_10` 已转成 Lance（`gr00t-lance-convert --video-mode external`）。

```bash
export DATASET=/data/datasets/libero_10_lance RUNS=/data/runs/flywheel GR00T_ROOT=/data/Isaac-GR00T

STAGE=smoke bash scripts/flywheel/fast_validation.sh   # 约 1 h：200 步 + 每任务 1 回合，打印每步耗时
STAGE=embed bash scripts/flywheel/fast_validation.sh   # CPU，几分钟：写运动学嵌入列
STAGE=g0    bash scripts/flywheel/fast_validation.sh   # 约 3.6 h
STAGE=curve bash scripts/flywheel/fast_validation.sh   # 约 0.6 h：5k/10k/20k 各 100 回合
STAGE=mine  bash scripts/flywheel/fast_validation.sh   # 约 1 h：14k 评 500 回合，然后挖掘、写回
STAGE=g3    bash scripts/flywheel/fast_validation.sh   # 约 1.1 h：从 14k 加权续训到 20k
STAGE=final bash scripts/flywheel/fast_validation.sh   # 约 2 h：两组 20k 各 500 回合，出配对报告
```

产物：`$RUNS/mining_report.json`（每个失败回合的分叉点、相似度曲线、覆盖盲区、权重分布）和 `$RUNS/paired_eval.md`。

常用覆盖项：`MAX_STEPS`、`FORK_STEP`、`SAVE_STEPS`（必须整除 5k/10k/14k/20k）、`EVAL_EPISODES`、`N_ENVS`、`EVAL_SEED`、`MINE_ARGS`（传给 `gr00t-lance-flywheel mine`，例如 `--min-match-sim 0.6 --alpha 2`）。

磁盘：默认 `SAVE_TOTAL_LIMIT=2`，`keep/` 里是硬链接。峰值大约是 3 个完整检查点加 3 个只含权重的检查点。

## 判定

| 结果 | 下一步 |
|---|---|
| 校准时 G0@20k 超过 90% | 提升空间太小：把 `MAX_STEPS` 降到 1 万步，或换扰动评测 |
| 校准时 G0@20k 不到 30% | 训练太短：`MAX_STEPS` 加到 3–4 万步 |
| G3 比 G0 高 5 个点以上 | 信号明确：补 2 个种子，再补 G1、G2 |
| 高 2–5 个点 | 单个种子说明不了问题，先补 1 个种子 |
| 不到 2 个点 | 先别加 GPU，调挖掘参数（`--min-match-sim`、`--top-k`、`--w-max`） |

`paired_eval.md` 里的 McNemar p 值只覆盖"评测抽到了哪些回合"的随机性，不覆盖训练种子的随机性。后者要多个种子才能估计。

## 已验证和未验证

已在 CPU 测试中验证（`tests/test_flywheel_*.py`）：

- 加权流在 0、2、3 个 worker 下确定、可精确续训；从均匀训练的检查点分叉后，续训流和从头开始的加权流一致。
- 零权重的锚点从不被抽到；抽样频率和权重一致。
- 仿真回合如果就是某条训练示范，嵌入后和存储的嵌入列一致（误差 < 2e-5）。
- 合成的失败回合：分叉点落在真实分叉位置附近；只有对应示范在分叉点附近的帧被加权；覆盖盲区被识别且不加权；`videos.lance` 写入 0 字节。
- 训练启动器的权重加载走隔离子进程，训练主进程不打开 Lance。
- 录制脚本的波次循环（用假环境）。

**还没在真实环境上跑过**：LIBERO 渲染、GR00T 策略服务端和录制脚本的对接、`--resume_from_checkpoint` 从硬链接检查点续训。`STAGE=smoke` 会先暴露这些问题。另外两个参数要按真实数据调：`min_match_sim`（默认 0.5）和 `divergence_drop`（默认 0.15）都依赖 LIBERO 运动学嵌入的相似度分布，第一次挖掘后先看 `mining_report.json` 里的相似度曲线再定。
