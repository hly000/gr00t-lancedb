# v1 实验脚本使用说明

> 配合 [第二阶段实验设计（v1）](experiment-design-v1.md) 使用。这里只讲"怎么跑、产出什么"。
> 所有命令都在仓库根目录执行。`BUCKET`、价格、路径换成你自己的。

## 一览

| 脚本 | 用在哪个实验 | 做什么 | 主要产出 |
|---|---|---|---|
| `scripts/build_scale_tier_x.py` | E1、E2（X 档） | 把一个 LeRobot v2.1 数据集复制成 1/10/50/200 份，可顺带转成 Lance | 每档一个数据目录 + `manifest.json` |
| `scripts/run_group_b_rclone.sh` | B 组（E1/E2/E3） | rclone 挂载桶 + 本地缓存；"换新机器"时卸载、删缓存、换新目录名重挂 | 标准输出打印数据集路径 |
| `scripts/preempt_schedule_runner.py` | E3（也可用于 E1/E2） | 按时间表杀训练进程 → 执行复位 → 续训，直到跑完；统计浪费和成本 | `events.jsonl`、`preempt_summary.json` |
| `scripts/bench_lerobot_groot_baseline.py` | issue #5 | 用 LeRobot 自带的 GR00T 策略在 Lance 上跑同样的训练并计时；生成功能对比表 | 结果 JSON、对比表 Markdown |
| `scripts/plot_v1_report.py` | 汇总 | 读 `results/v1/**`，画三张主图并出表 | `fig1/2/3*.png`、`report.md` |
| `scripts/v1_common.py` | （公共） | 读 `bench_metrics.jsonl` 的共用函数，兼容旧日志 | — |

## 结果目录约定

画图脚本按路径识别每次运行属于哪个组、哪一档，所以请按这个格式放：

```
results/v1/<实验>/group=<组>/tier=<档>/rep=<第几次>/      # E1、E2
results/v1/e3/group=<组>/rate=<每小时抢占次数>/seed=<种子>/  # E3
```

例如 `results/v1/e1/group=B/tier=x50/rep=3/`。目录里放 `bench_metrics.jsonl`（训练日志）和/或
`preempt_summary.json`（监督脚本的汇总）。还可以放一个 `run_meta.json` 补充信息，例如
`{"dataset_gb": 32.1, "data_prep_s": 85.0}`。路径里的 `键=值` 和 `run_meta.json` 都会被读进来，后者优先。

**最省事的做法**：E1、E2、E3 都用 `preempt_schedule_runner.py` 启动训练，把 `--runner-dir` 指到上面的目录。
它会把训练日志复制过去，还会记下"数据准备"（复位命令）的耗时。

---

## 1. `build_scale_tier_x.py`：生成 X 档数据

**做什么**：同一份数据复制 N 份，只改编号，不改内容。第 r 份里的第 i 个 episode 编号变成 `r × 原 episode 数 + i`。
会改写的地方：

- 每个 parquet 里的 `episode_index` 和全局 `index`（新的连续编号），文件放到新编号对应的 chunk 目录下；
- 视频：内容不变，换成新编号的文件名。默认用**硬链接**（本地几乎不占额外空间），也可以 `--link-mode symlink` 或 `--copy-videos`（真复制）；
- `meta/episodes.jsonl`、`meta/episodes_stats.jsonl`（编号，以及 `index`/`episode_index` 两列的统计）、`meta/info.json`（总数、chunk 数、splits）、`meta/stats.json` 里 `index`/`episode_index` 两项。
- state/action 的统计量不用改：原样复制 N 份，分位数、均值都不变。其他 meta 文件原样复制。

加 `--lance copy,external` 会再用 `convert_v2_to_lance` 转成 Lance：`copy` 把视频存进 `videos.lance`，
`external` 只记视频在哪（默认指向本地 v21 目录，用 `--external-base-uri` 改成桶里的位置）。
`modality.json` 由转换函数自己嵌进去（取自 `meta/` 或 `--modality-json`），**不需要再跑 `attach_gr00t_meta`**。

**怎么跑**（先 `--dry-run` 看计划，不写任何文件）：

```bash
python scripts/build_scale_tier_x.py --src /data/datasets/libero_10 --out-root /data/datasets/x_tier \
    --factors 1,10,50,200 --lance copy --dry-run

# 真正生成。200 份的 Lance copy 约 128 GB，本机盘放不下，直接写到桶里：
python scripts/build_scale_tier_x.py --src /data/datasets/libero_10 --out-root /data/datasets/x_tier \
    --factors 1,10,50,200 --lance copy \
    --lance-uri-template 's3://BUCKET/x_tier/x{factor}/lance_{mode}'
#   源数据没有 meta/modality.json 时加 --modality-json <LIBERO 的 modality.json>

# 原始格式上传到桶（A、B 组用）：硬链接会被当成普通文件完整上传，正好是我们要的"真实体积"
rclone copy /data/datasets/x_tier/x50/v21 gcs:BUCKET/x_tier/x50/v21 --transfers 64
#   如果用了 --link-mode symlink，要加 --copy-links
```

写桶用的凭证从环境变量读（`AWS_ACCESS_KEY_ID`、`AWS_SECRET_ACCESS_KEY`、`AWS_ENDPOINT` 等，和训练时一样）。
目标位置必须是空的。

**产出**：

- `<out-root>/x<F>/v21/`：复制后的 LeRobot v2.1 数据集；
- `<out-root>/x<F>/lance_copy/`、`lance_external/`（或模板指定的桶路径）；
- `<out-root>/manifest.json`：每档的 episode 数、帧数、视频数，每个版本的 `files`（文件数）、
  `logical_bytes`（上传时实际传的字节数）、`unique_disk_bytes`（本地真实占用，硬链接只算一次）、`build_s`（生成耗时）。
  写到桶里的版本字节数记为 `null`，请用 `rclone size` 量。画图时可以用 `--manifest` 把它传给画图脚本，自动得到每档体积。

---

## 2. `run_group_b_rclone.sh`：B 组挂载

**做什么**：按第 2 节验证过的方式挂载：root 挂载，加 `--allow-other --uid --gid`，`--vfs-cache-mode full`，缓存放本地盘。
gVisor 里清不掉页缓存，所以**每次复位都换新的目录名**（`mnt/gen-<n>`、`vfs-cache/gen-<n>`），保证复位后读到的文件从没进过任何缓存。

```bash
export B_REMOTE=gcs:BUCKET/x_tier/x50     # rclone 远端
export B_DATASET_SUBDIR=v21               # 远端里交给 GR00T 的子目录
export B_BASE_DIR=/data/groupb            # 默认就是这个

DATASET=$(bash scripts/run_group_b_rclone.sh mount)   # 挂载，标准输出只有数据集路径
bash scripts/run_group_b_rclone.sh status             # 第几代、是否挂着、缓存多大
bash scripts/run_group_b_rclone.sh reset              # "换新机器"：卸载 + 删缓存 + 新目录名重挂，打印新路径
bash scripts/run_group_b_rclone.sh unmount            # 只卸载，缓存保留
DRY_RUN=1 bash scripts/run_group_b_rclone.sh reset    # 只打印要执行的命令，不动任何东西
```

另外它维护一个固定的软链接 `$B_BASE_DIR/current`，指向当前这一代的数据集路径。
训练命令里写 `--dataset-path /data/groupb/current`，复位之后不用改命令。

其他环境变量：`B_CACHE_ROOT`（缓存根目录）、`B_VFS_CACHE_MAX`（默认 150G）、`B_UID`/`B_GID`、`B_MOUNT_TIMEOUT`、
`RCLONE_BIN`、`RCLONE_EXTRA_ARGS`（额外的 rclone 参数）、`SUDO`（非 root 时默认 `sudo`）、
`B_REUSE_CACHE=1`（重挂时沿用上一代缓存，用来测"缓存热"）、`B_RESET_REMOUNT=0`（reset 只清理不重挂）。

**产出**：`$B_BASE_DIR/state/gen`（当前第几代）、`state/gen-<n>.pid`、`logs/rclone-gen-<n>.log`、`current` 软链接。

---

## 3. `preempt_schedule_runner.py`：抢占监督脚本（E3 核心）

**做什么**：模拟 SkyPilot 托管的 spot 任务。

1. （默认）先执行一次复位命令，记作"数据准备"耗时；
2. 在独立的进程组里启动 `--` 后面的训练命令；
3. 到了时间表上的时间点就"抢占"：`--kill-mode kill` 直接 `SIGKILL` 整组进程；`--kill-mode term` 先发 `SIGTERM`，
   等 `--grace-s`（默认 30 秒，对应 GCP）后还没退出就 `SIGKILL`；
4. 执行该组的"新机器复位"命令（`--reset-cmd`）；
5. 如果输出目录里有完整的检查点（`checkpoint-<N>/` 里有 `trainer_state.json`），就在命令后面加 `--resume_from_checkpoint` 重启；
   没有就从头开始。不完整的检查点目录会先改名挪开，免得训练程序误读；
6. 重复，直到训练正常结束（退出码 0 且到了最大步数，或日志里有到达最大步数的 `train_end`）。

**时间表**（JSON 文件）三种写法：

```json
{"mode": "since_launch", "times_s": [412.0, 95.3]}   // 第 i 次抢占发生在第 i 次启动后 t_i 秒（机器寿命）
{"mode": "wall", "times_s": [412.0, 1630.2]}         // 从第一次启动算起的累计秒数
{"mode": "at_step", "times_s": [130, 260]}           // 日志里训练到第 N 步时抢占（E8 短版用，可精确复现）
```

`wall` 模式下，如果某个时间点落在复位/重启期间（没有训练进程可杀），就在下次启动后立即抢占；
同一段空档里落了好几个点，只算一次（`collapsed` 记被合并的个数）。

**E3 用法**：

```bash
# 1) 先生成时间表文件，所有组共用同一份。平均间隔 5/10/20 分钟 = 每小时 12/6/3 次
for r in 3 6 12; do for s in 1 2; do
  python scripts/preempt_schedule_runner.py --gen-poisson --rate-per-hour $r --horizon-h 6 --seed $s \
      --write-schedule results/v1/e3/schedules/r${r}_s${s}.json
done; done

# 2) 每个组 × 频率 × 种子跑一次。下面是 B 组、每小时 6 次、种子 1、硬杀：
python scripts/preempt_schedule_runner.py \
    --schedule results/v1/e3/schedules/r6_s1.json --kill-mode kill \
    --output-dir /data/runs/e3/B_r6_s1 --runner-dir results/v1/e3/group=B/rate=6/seed=1 \
    --reset-cmd 'B_REMOTE=gcs:BUCKET/datasets B_DATASET_SUBDIR=robocasa bash scripts/run_group_b_rclone.sh reset' \
    --gpu-usd-per-hour 1.20 --boot-overhead-s 120 \
    --meta group=B --meta rate_per_hour=6 --meta seed=1 -- \
    env BACKEND=native DATASET=/data/groupb/current OUT=/data/runs/e3/B_r6_s1 \
        MAX_STEPS=2000 SAVE_STEPS=250 bash scripts/run_gr00t_finetune.sh
```

三个组的复位命令（`--reset-cmd`）参考：

| 组 | 复位命令 | 训练命令里的数据路径 |
|---|---|---|
| A 先下载 | `old=$(readlink /data/groupa/current); [ -n "$old" ] && rm -rf "$old"; d=/data/groupa/gen-$(date +%s); s5cmd --numworkers 256 cp "s3://BUCKET/datasets/robocasa/*" "$d/" && ln -sfn "$d" /data/groupa/current` | `DATASET=/data/groupa/current` |
| B 挂载 | `bash scripts/run_group_b_rclone.sh reset` | `DATASET=/data/groupb/current` |
| C / C1 Lance | `rm -rf "${GR00T_LANCE_CACHE_DIR:-$HOME/.cache/gr00t_lancedb}"`（若设了 `GR00T_LANCE_VIDEO_CACHE_DIR` 也一起删） | `DATASET=s3://BUCKET/.../lance_copy`，`BACKEND=lance` |

A 组同样每次用新目录名（`gen-<时间戳>`），原因同 B 组。s5cmd 访问 GCS 时要按你的环境加 `--endpoint-url`。

其他常用参数：`--max-steps`（不写就从命令里的 `--max_steps N` 或 `MAX_STEPS=N` 读）、`--signal-target group|leader`
（默认 `group`：整组进程一起收信号，像整台机器关机；`leader` 只发给启动的那个进程）、`--no-reset-before-first`、
`--resume-args`（默认 `--resume_from_checkpoint`）、`--max-launches`、`--max-crash-restarts`（训练自己崩溃时最多重试几次，默认 0）、
`--dry-run`（打印时间表和首次/续训两条命令，不执行）。

**E1 / E2 也可以用它**（不抢占）：`--no-preempt`。E1 每次都要先复位（默认就会），复位耗时就是"数据准备"那一段；
建议 `MAX_STEPS=21 GR00T_BENCH_SKIP_FINAL_SAVE=1`（第一步之后再跑 20 步，不存 24 GB 的最终检查点）。
E2 测"缓存热"时加 `--no-reset-before-first`。

**产出**（都在 `--runner-dir`，默认 `<output-dir>/runner`）：

- `events.jsonl`：每行一个事件，`t` 是 Unix 时间，可以和训练日志对齐。事件有 `start`、`reset_start`/`reset_end`（含 `duration_s`、退出码）、
  `launch`（第几次、是否续训、从哪一步续、完整命令）、`preempt`（信号、计划时间、实际在启动后多少秒、当时训练到第几步）、
  `escalate_sigkill`（宽限期内没退出被强杀）、`exit`（退出码、运行时长、最后一步、当前最新检查点、从发信号到退出的秒数）、
  `quarantine_ckpt`、`crash`、`done`。
- `preempt_summary.json`：
  - `total_wall_s` 总墙钟（含最开始的复位）、`initial_reset_s`、`launches` 启动次数、`interruptions` 抢占次数、`crashes`；
  - `steps_reexecuted` 重做的步数（每次中断：被杀前最后一步 − 续训起点）；`steps_reexecuted_from_records` 用训练日志里的 step 记录交叉核对，两者应相等；
  - `recovery_s_total`/`recovery_s_median`：每次恢复耗时 = 发出抢占信号 → 新进程的第一个训练步结束；
  - `lost_work_s_total`：最后一次落盘（`save` 记录，或检查点对应那一步）之后、到被杀为止白干的时间；
  - `wasted_gpu_s` = 白干时间 + 恢复耗时；`wasted_gpu_s_doc_formula` = 恢复耗时 + 重做步数 × 稳态单步时间（设计文档的写法），两者应接近；
  - `billable_s` = 总墙钟 + 抢占次数 × `--boot-overhead-s`（开机时间不在本机实测，只进成本模型）；`total_cost_usd` = `billable_s` × 每小时价格；`wasted_cost_usd`；
  - `interruption_details`：每次中断一行（信号、是否被强杀、中断时第几步、从哪一步续、是否在收到 SIGTERM 后写了检查点、丢了几步、各段耗时）；
  - `launch_details`：每次启动一行（退出码、时长、启动各段耗时 `startup`）；
  - `steady`：稳态单步时间、等数据占比、内存峰值、每 1k 步的读请求；`meta`：`--meta` 传进来的标签。
- `launch-<k>.log` / `reset-<k>.log`：每次启动、每次复位的输出；
- `bench_metrics.jsonl`：训练日志的副本（画图用）。

---

## 4. `bench_lerobot_groot_baseline.py`：issue #5 对照

**做什么**：回答"为什么不直接用 LeRobot 的 GR00T 策略读 Lance"。两个子命令：

- `run`：包一层 `lerobot-train --policy.type=groot`，用和 GR00T 那次相同的步数、batch、种子、worker 数、基础模型（N1.7）。
  `--log_freq=1` 让每步打一行日志，脚本给每行打时间戳，同时每 2 秒采一次进程树内存。
- `table`：生成三方功能对比表。C 列（LeRobot）会去**本机安装的 lerobot 源码**里逐项核对（不 import，只读源码）；
  核对不上就标"待验证"。对不上的典型原因是装的版本和写表时看的 0.7.0 不一样。

```bash
# 先看要执行的完整命令
python scripts/bench_lerobot_groot_baseline.py run --dry-run \
    --dataset-root /data/datasets/libero_10_lance --steps 300 --batch-size 32 --seed 42 \
    --output-dir /data/runs/issue5/lerobot
# 真跑（需要 GPU，以及装了 lerobot 的环境；注意 lerobot 和 GR00T 依赖冲突，要分开的环境）
python scripts/bench_lerobot_groot_baseline.py run \
    --dataset-root /data/datasets/libero_10_lance --steps 300 --batch-size 32 --seed 42 \
    --output-dir /data/runs/issue5/lerobot --results results/v1/issue5/lerobot_results.json
# 对比表 + 实测对比（GR00T+Lance 那次的 bench_metrics.jsonl 用同样的步数/batch/种子跑）
python scripts/bench_lerobot_groot_baseline.py table --out results/v1/issue5/gap_table.md \
    --results results/v1/issue5/lerobot_results.json \
    --gr00t-metrics /data/runs/issue5/gr00t_lance/bench_metrics.jsonl
```

`--` 后面可以追加任意 `lerobot-train` 参数（如 `--policy.embodiment_tag=...`）。

**没有对齐的部分**（结果 JSON 的 `config.not_matched` 里也写了）：优化器和学习率预设、图像增强、动作长度、state dropout、
数据加载预取。所以只比时间和内存，不比 loss。

**产出**：

- 结果 JSON：`launch_to_first_batch_s`（启动 → 第一个 batch 到手，≈ 第一步日志时间 − 第一步的更新耗时）、
  `launch_to_first_step_s`、`first_step_data_s`、`steady_step_s_wall_median`（稳态每步墙钟，按相邻日志行间隔）、
  `steady_step_s_logged_median`（日志里 `updt_s + data_s`）、`steady_data_s_mean`、`peak_pss_tree_mb`/`peak_rss_tree_mb`/`peak_sys_used_mb`、
  loss 首尾、完整命令、lerobot 版本；同名 `.log` 是带时间戳的完整输出。
- 对比表 Markdown：7 项功能 × 3 种方案，每格写"支持情况 + 依据"；有实测时附一张时间/内存对比表。

---

## 5. `plot_v1_report.py`：三张主图

```bash
python scripts/plot_v1_report.py --root results/v1 --manifest /data/datasets/x_tier/manifest.json \
    --fig1-exp e1 --fig2-exp e3 --fig3-exp e2 --gpu-usd-per-hour 1.20
```

- **图 1** `fig1_cold_start.png`：左边是每个"组 × 档"的冷启动堆叠条（中位数）。8 段依次是：数据准备、进程启动+导入
  （含日志里没覆盖到的时间）、加载模型、建数据集、训练器初始化+加载检查点、创建数据加载器、首个 batch、首个训练步。
  右边是"冷启动（含数据准备）vs 数据集体积"，每组一条线，误差线是最小–最大。
- **图 2** `fig2_preempt_cost.png`：横轴每小时抢占次数，纵轴跑完的总成本（所有运行都有价格时用美元，否则用 GPU 小时），
  每组一条线；硬杀和带通知各一个子图。
- **图 3** `fig3_steady.png`：稳态单步时间（斜线部分是等数据）、内存峰值（实心 = 训练进程树 PSS，斜线 = 整机已用）、
  每 1k 步真正发给 Lance 的读调用次数（来自 `io.lance_blob_calls`，原生组没有这个计数，显示 n/a）。
- `report.md`：三张图对应的表。图 3 的表还有读区间数、读取 MB、网卡接收 MB、GOP 缓存命中率、按每千次请求 $0.0004 估算的请求费
  （`--usd-per-1k-requests` 可改）。

颜色固定跟组走（A 蓝、B 橙、C 青绿、C1 黄……），三张图里同一个组颜色一致；用的是色弱友好的配色。
不想出图只要表，加 `--no-png`（不需要装 matplotlib）。

---

## 哪些已经在本地测过，哪些要上 GPU / 云

测试文件：`tests/test_v1_scripts.py`（`PYTHONPATH=src:tests pytest tests/test_v1_scripts.py`）。

| 内容 | 本地已测 | 还需要在 GPU 机 / 云上确认 |
|---|---|---|
| X 档复制 | 合成数据集复制 3 份：每个样本用参考加载器（和 GR00T 读法一致）读出来与原 episode 逐值、逐像素相同；Lance copy/external 两种都转换成功、行数正确；写到（本地模拟的）S3 也成功 | LIBERO-10 真数据的 200 份耗时和体积；上传到真实桶 |
| B 组挂载 | 只测了 `DRY_RUN=1` 打印的命令（rclone 参数、换代、删缓存） | 真正的 rclone 挂载/卸载（本机没有 rclone 和 FUSE） |
| 抢占监督 | 用假训练程序端到端测了：SIGTERM 时写检查点、退出 143、续训 0 步重做；SIGKILL 时从上一个检查点续、重做步数和日志一致；复位命令次数；成本公式；不完整检查点挪开；`wall` 模式合并 | 真实 GR00T 进程树（`uv run` 下的信号传递、数据加载 worker 同时收到 SIGTERM 时紧急存档是否还能成功） |
| issue #5 | `--dry-run` 命令、日志解析、用假的 `lerobot-train` 跑通 `run`、对比表生成和"版本对不上就标待验证" | 真实 `lerobot-train --policy.type=groot` 能否直接读这份 Lance、实测时间和内存 |
| 画图 | 合成日志（含旧格式日志、截断的最后一行）出三张图和表，数值逐项核对 | 真实结果的图面是否需要调整 |

## 需要注意的地方

- **SIGTERM 发给谁**：默认发给整个进程组（像整台机器关机，数据加载 worker 也会同时收到）。上 GPU 先用 `--kill-mode term`
  跑一次短的，看 `launch-<k>.log` 里有没有 "Preempted at step … exiting (143)"。如果 worker 先死导致紧急存档失败，
  这本身是要报告的结果；想对比 v0 的做法（只发给训练进程）可以用 `--signal-target leader`，但要先确认 `uv run` 会把信号转给 Python。
- **检查点完整的判断**：`checkpoint-<N>/trainer_state.json` 存在就算完整。HF 的定期检查点在最后才写这个文件；本仓库的轻量紧急检查点是整个目录写完后一次性改名出来的。
- **输出目录可以重复用**：监督脚本只复制本次运行新写的日志行，不会把上一次的记录混进来。不过续训会读目录里已有的检查点，所以 E1/E3 每次运行最好用新的 `--output-dir`。
- **E1 里的"数据准备"**：取复位命令的耗时（`initial_reset_s`）或 `run_meta.json` 里的 `data_prep_s`。C 组复位只是删本地缓存，所以这一段几乎为 0，
  它拉元数据的时间算在"建数据集"里。
- **复制的数据只用来画趋势**：内容完全重复，压缩、缓存命中可能比真实数据好看，结论还要靠 M、L 档。
