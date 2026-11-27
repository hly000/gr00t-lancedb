#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Issue #5 baseline: "why not just use LeRobot's GR00T policy on a Lance dataset?"

Two subcommands:

``run``    wraps ``lerobot-train --policy.type=groot`` on a Lance dataset with the same steps,
           batch size and seed as the GR00T run, timestamps every log line, samples host RAM
           of the process tree, and writes a results JSON (time to first batch / first step,
           steady step time, data wait, peak RAM). ``--dry-run`` prints the exact command.
``table``  writes the Markdown feature-gap table for
           (A) native Isaac-GR00T + LeRobot v2.1 files,
           (B) native Isaac-GR00T + this repo's Lance loader,
           (C) LeRobot GR00T policy + Lance.
           Column C is checked against the *installed* lerobot source when it is importable
           (``--no-probe`` to skip); anything not confirmed is marked "待验证". With
           ``--results`` / ``--gr00t-metrics`` a measured comparison table is appended.

Examples::

    python scripts/bench_lerobot_groot_baseline.py run --dry-run \
        --dataset-root /data/datasets/libero_10_lance --steps 300 --batch-size 32 --seed 42 \
        --output-dir /data/runs/issue5/lerobot --results results/v1/issue5/lerobot_results.json
    python scripts/bench_lerobot_groot_baseline.py table --out docs/issue5-gap-table.md \
        --results results/v1/issue5/lerobot_results.json \
        --gr00t-metrics /data/runs/issue5/gr00t_lance/bench_metrics.jsonl

Matched: steps, batch size, seed, dataloader workers, base model (N1.7). Not matched (recorded
in the results JSON): optimizer / LR schedule presets, augmentation, action horizon, which
LeRobot sets per policy; compare step *time*, not loss.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import statistics as st
import subprocess
import sys
import threading
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v1_common import md_table, read_jsonl, split_segments, startup_breakdown, steady_stats  # noqa: E402

# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

_STEP_RE = re.compile(r"\bstep:(\d+(?:\.\d+)?)([KMBT]?)\b")
_FLOAT_RE = {k: re.compile(rf"\b{k}:([0-9.]+(?:e-?\d+)?)") for k in ("loss", "updt_s", "data_s")}
_SUFFIX = {"": 1, "K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}


def build_lerobot_cmd(args: argparse.Namespace) -> list[str]:
    repo_id = args.repo_id or f"local/{Path(str(args.dataset_root).rstrip('/')).name}"
    cmd = [
        *shlex.split(args.lerobot_train),
        f"--dataset.repo_id={repo_id}",
        f"--dataset.root={args.dataset_root}",
        "--policy.type=groot",
        f"--policy.base_model_path={args.base_model_path}",
        "--policy.push_to_hub=false",
        f"--policy.device={args.device}",
        f"--batch_size={args.batch_size}",
        f"--steps={args.steps}",
        f"--seed={args.seed}",
        f"--num_workers={args.num_workers}",
        "--log_freq=1",  # one log line per step -> per-step wall timestamps
        f"--save_checkpoint={'true' if args.save_checkpoint else 'false'}",
        f"--output_dir={args.output_dir}",
        f"--job_name={args.job_name}",
    ]
    if args.embodiment_tag:
        cmd.append(f"--policy.embodiment_tag={args.embodiment_tag}")
    return [*cmd, *args.extra]


def parse_log_line(line: str) -> dict[str, float] | None:
    """Parse a LeRobot ``MetricsTracker`` line (``step:12 smpl:384 ... updt_s:0.31 data_s:0.02``)."""
    m = _STEP_RE.search(line)
    if not m or "updt_s:" not in line:
        return None
    out = {"step": float(m.group(1)) * _SUFFIX[m.group(2)]}
    for k, rx in _FLOAT_RE.items():
        mm = rx.search(line)
        if mm:
            out[k] = float(mm.group(1))
    return out


def _tree_mem_mb(pid: int) -> dict[str, float]:
    try:
        import psutil
    except ImportError:
        return {}
    out: dict[str, float] = {}
    try:
        vm = psutil.virtual_memory()
        out["sys_used_mb"] = (vm.total - vm.available) / 2**20
        root = psutil.Process(pid)
        procs = [root, *root.children(recursive=True)]
        rss = pss = 0
        have_pss = True
        for p in procs:
            try:
                info = p.memory_full_info()
                rss += info.rss
                if hasattr(info, "pss"):
                    pss += info.pss
                else:
                    have_pss = False
            except (psutil.Error, AttributeError):
                have_pss = False
        out["rss_tree_mb"] = rss / 2**20
        if have_pss:
            out["pss_tree_mb"] = pss / 2**20
        out["num_procs"] = float(len(procs))
    except Exception:  # noqa: BLE001 - sampling must never break the run
        pass
    return out


def summarize_lerobot_run(
    t_launch: float, lines: list[tuple[float, str]], mem: list[dict[str, float]], warmup: int = 10
) -> dict[str, Any]:
    """Turn timestamped log lines + memory samples into the results metrics."""
    steps = []
    for t, line in lines:
        rec = parse_log_line(line)
        if rec is not None:
            rec["t"] = t
            steps.append(rec)
    out: dict[str, Any] = {"step_lines": len(steps)}
    if steps:
        first = steps[0]
        out["launch_to_first_step_s"] = first["t"] - t_launch
        out["first_step_data_s"] = first.get("data_s")
        out["first_step_updt_s"] = first.get("updt_s")
        if first.get("updt_s") is not None:
            # The step line is printed right after the update, so the batch arrived ~updt_s earlier.
            out["launch_to_first_batch_s"] = out["launch_to_first_step_s"] - first["updt_s"]
        steady = steps[min(warmup, max(0, len(steps) - 2)):]
        # --log_freq=1: consecutive step lines are consecutive steps (the printed step number
        # is rounded to K/M above 999, so the line gap is used instead of the number).
        gaps = [b["t"] - a["t"] for a, b in zip(steady, steady[1:])]
        if gaps:
            out["steady_step_s_wall_median"] = st.median(gaps)
        upd = [s["updt_s"] + s.get("data_s", 0.0) for s in steady if "updt_s" in s]
        if upd:
            out["steady_step_s_logged_median"] = st.median(upd)
        data = [s["data_s"] for s in steady if "data_s" in s]
        if data:
            out["steady_data_s_mean"] = st.mean(data)
        losses = [s["loss"] for s in steps if "loss" in s]
        if losses:
            out["loss_first"] = losses[0]
            out["loss_last10_mean"] = st.mean(losses[-10:])
        out["last_step"] = steps[-1]["step"]
    for key in ("pss_tree_mb", "rss_tree_mb", "sys_used_mb"):
        vals = [m[key] for m in mem if key in m]
        if vals:
            out[f"peak_{key}"] = max(vals)
    return out


def cmd_run(args: argparse.Namespace) -> int:
    cmd = build_lerobot_cmd(args)
    config = {
        "dataset_root": str(args.dataset_root),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "base_model_path": args.base_model_path,
        "not_matched": ["optimizer/scheduler preset", "image augmentation", "action horizon / chunk size",
                        "state dropout", "dataloader prefetch settings"],
    }
    if args.dry_run:
        print(shlex.join(cmd))
        print(json.dumps({"cmd": cmd, "config": config}, indent=2))
        return 0
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.results.with_suffix(".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[tuple[float, str]] = []
    mem: list[dict[str, float]] = []
    t_launch = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                            env={**os.environ, "PYTHONUNBUFFERED": "1"})
    stop = threading.Event()

    def sampler() -> None:
        while not stop.is_set():
            m = _tree_mem_mb(proc.pid)
            if m:
                m["t"] = time.time()
                mem.append(m)
            stop.wait(args.mem_interval_s)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    assert proc.stdout is not None
    with open(log_path, "w", encoding="utf-8") as log:
        for line in proc.stdout:
            t = time.time()
            lines.append((t, line.rstrip("\n")))
            log.write(f"{t:.3f} {line}")
    rc = proc.wait()
    stop.set()
    th.join(timeout=5)
    t_end = time.time()
    results = {
        "system": "lerobot_groot_lance",
        "returncode": rc,
        "cmd": cmd,
        "config": config,
        "wall_s": t_end - t_launch,
        "lerobot_version": _lerobot_version(),
        "metrics": summarize_lerobot_run(t_launch, lines, mem),
        "log": str(log_path),
    }
    args.results.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results["metrics"], indent=2))
    return 0 if rc == 0 else rc


# ---------------------------------------------------------------------------
# table
# ---------------------------------------------------------------------------

YES, NO, PART, TBD = "✅", "❌", "⚠️ 部分", "待验证"
NA = "➖ 不适用"

FEATURES: list[tuple[str, str]] = [
    ("n17", "GR00T N1.7"),
    ("mixture", "多数据集混合 + 权重"),
    ("q01q99", "GR00T q01/q99 统计量"),
    ("where", "帧级 `where` 锚点筛选"),
    ("version", "Lance 表版本固定（MVCC）"),
    ("exact_resume", "抢占后精确续训"),
    ("lazy_shuffle", "只存索引、延迟解码的洗牌缓冲区"),
]

# Columns A and B: facts about Isaac-GR00T and this repo, with where they were established.
STATIC: dict[str, dict[str, tuple[str, str]]] = {
    "A": {
        "n17": (YES, "Isaac-GR00T main 只支持 N1.7；v0 在 A100 上实跑（`BACKEND=native`）"),
        "mixture": (YES, "GR00T 的 `DatasetFactory` 按 `mix_ratio` / `ds_weights_alpha` 加权混合"
                         "（本仓库 `exact_resume.py` 照搬同一逻辑）"),
        "q01q99": (YES, "GR00T 读取 `meta/stats.json` 的 q01/q99"),
        "where": (TBD, "未见按帧筛选参数，预期只能改文件或 episode 列表"),
        "version": (NO, "普通文件没有表版本；只能靠桶的对象版本或另存一份目录"),
        "exact_resume": (NO, "续训用 `seed + global_step` 重新播种：v0 实测续训后 0/200 步与不中断一致，"
                             "纯数据仿真重复率/漏采率 30.8%"),
        "lazy_shuffle": (NO, "fast loader 的洗牌缓冲区存放已解码的样本（`load_split` 整段解码；"
                             "见 `exact_resume.py` 文档）"),
    },
    "B": {
        "n17": (YES, "`run_gr00t_finetune.sh` + `BACKEND=lance`，v0 实跑"),
        "mixture": (YES, "沿用 GR00T 的加权混合（`exact_resume.py`），另有 `resumable_mixture.py`"),
        "q01q99": (YES, "`convert.py` 保留或从 `frames.lance` 计算 q01/q99"),
        "where": (YES, "`GR00T_LANCE_WHERE`（SQL 谓词下推到 `frames.lance`）"),
        "version": (YES, "`GR00T_LANCE_VERSION`（训练固定在某个表版本）"),
        "exact_resume": (YES, "v0 实测 300/300 步批次指纹一致，0 步重做；SIGTERM 时按代价决定是否存档"),
        "lazy_shuffle": (YES, "`GR00T_LANCE_LAZY_DECODE=1`（lance 后端默认）"),
    },
}

# Column C: (status, evidence, probe key, expected probe result for the status to hold).
LEROBOT_FACTS: dict[str, tuple[str, str, str | None, bool | None]] = {
    "n17": (YES, "`policies/groot/configuration_groot.py`：只支持 n1.7（N1.5 已移除）", "n17", True),
    "mixture": (NO, "`datasets/factory.py`：多数据集时抛 `NotImplementedError`（MultiLeRobotDataset 未启用）",
                "multi_dataset_disabled", True),
    "q01q99": (PART, "`policies/groot/utils.py` 支持百分位归一化（q01/q99 替代 min/max）；"
                     "与 GR00T 原生统计量逐值一致待验证", "q01q99", True),
    "where": (NO, "只有 episode 级筛选（`--dataset.episodes` / `--dataset.exclude_episodes`），"
                  "lance 后端源码里没有帧级谓词", "lance_frame_filter", False),
    "version": (NO, "只有 Hub 的 `--dataset.revision`；lance 后端源码里没有表版本参数", "lance_version_pin", False),
    "exact_resume": (PART, "`--resume` 时按 step 推出采样器位置，从上一个定期存档起样本精确；"
                           "训练脚本里没有 SIGTERM 处理，中断时不存档（GPU 上待验证）", "sampler_resume", True),
    "lazy_shuffle": (NA, "map 式随机读取（EpisodeAwareSampler），没有洗牌缓冲区，每个样本单独解码",
                     "episode_aware_sampler", True),
}


def _lerobot_version() -> str | None:
    try:
        return importlib.metadata.version("lerobot")
    except importlib.metadata.PackageNotFoundError:
        return None


def probe_lerobot() -> dict[str, Any]:
    """Grep the installed lerobot source (without importing it) for the facts in column C."""
    version = _lerobot_version()
    spec = importlib.util.find_spec("lerobot") if version else None
    if not spec or not spec.origin:
        return {"installed": False}
    root = Path(spec.origin).parent

    def text(rel: str) -> str:
        try:
            return (root / rel).read_text(encoding="utf-8")
        except OSError:
            return ""

    lance = text("datasets/lance_backend.py")
    train = text("scripts/lerobot_train.py")
    return {
        "installed": True,
        "version": version,
        "path": str(root),
        "n17": "GROOT_N1_7" in text("policies/groot/configuration_groot.py"),
        "multi_dataset_disabled": "MultiLeRobotDataset isn't supported" in text("datasets/factory.py"),
        "q01q99": "q99" in text("policies/groot/utils.py"),
        "lance_backend": bool(lance),
        "lance_frame_filter": bool(re.search(r"\.where\(|filter=|prefilter", lance)),
        "lance_version_pin": bool(re.search(r"\bversion\b", lance)),
        "sampler_resume": "compute_sampler_state" in train,
        "sigterm_handler": "SIGTERM" in train or "ProcessSignalHandler" in train,
        "episode_aware_sampler": "EpisodeAwareSampler" in train,
    }


def column_c(probe: dict[str, Any] | None) -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    for key, (status, evidence, pkey, expected) in LEROBOT_FACTS.items():
        if probe and probe.get("installed") and pkey is not None:
            if pkey not in probe or probe[pkey] != expected or not probe.get("lance_backend", False):
                status = TBD
                evidence = f"{evidence}（已安装版本 {probe.get('version')} 与 0.7.0 源码不符，需重新核对）"
            elif key == "exact_resume" and probe.get("sigterm_handler"):
                status, evidence = TBD, f"{evidence}（已安装版本出现 SIGTERM 处理，需重新核对）"
        out[key] = (status, evidence)
    return out


def gap_table_markdown(probe: dict[str, Any] | None, results: dict[str, Any] | None = None,
                       gr00t_metrics: Path | None = None) -> str:
    c = column_c(probe)
    if probe and probe.get("installed"):
        src = f"本机安装的 lerobot {probe['version']}（`{probe['path']}`）逐项核对"
    else:
        src = "lerobot 0.7.0 源码（未在本机找到 lerobot，未重新核对）"
    lines = [
        "# 为什么不直接用 LeRobot 的 GR00T 策略读 Lance？（issue #5）",
        "",
        "- (A) Isaac-GR00T 原生训练 + LeRobot v2.1 文件",
        "- (B) Isaac-GR00T 原生训练 + 本仓库的 Lance 加载器",
        f"- (C) LeRobot 的 GR00T 策略（`lerobot-train --policy.type=groot`）+ Lance；C 列依据：{src}",
        "",
        "✅ 支持　❌ 不支持　⚠️ 部分支持　➖ 不适用　待验证 = 还没有证据，不下结论",
        "",
        "| 功能 | (A) GR00T + v2.1 | (B) GR00T + 本仓库 Lance | (C) LeRobot GR00T + Lance |",
        "|---|---|---|---|",
    ]
    for key, label in FEATURES:
        cells = [STATIC["A"][key], STATIC["B"][key], c[key]]
        lines.append(f"| {label} | " + " | ".join(f"{s}<br>{e}" for s, e in cells) + " |")
    measured = measured_comparison(results, gr00t_metrics)
    if measured:
        lines += ["", "## 实测对比（同步数、同 batch、同种子）", "", measured]
    return "\n".join(lines) + "\n"


def measured_comparison(results: dict[str, Any] | None, gr00t_metrics: Path | None) -> str:
    rows: list[list[Any]] = []
    if gr00t_metrics is not None:
        recs = read_jsonl(gr00t_metrics)
        if recs:
            seg = split_segments(recs)[0]
            sb = startup_breakdown(seg)
            stt = steady_stats(seg)
            first_batch = None
            if sb.get("launch_to_first_step_s") is not None and sb.get("first_step_s") is not None:
                first_batch = sb["launch_to_first_step_s"] - sb["first_step_s"]
            rows.append(["(B) GR00T + Lance", first_batch, sb.get("launch_to_first_step_s"),
                         stt.get("steady_step_s_median"), stt.get("steady_data_wait_s_mean"),
                         stt.get("peak_pss_tree_mb"), stt.get("peak_sys_used_mb")])
    if results:
        m = results.get("metrics", {})
        rows.append(["(C) LeRobot GR00T + Lance", m.get("launch_to_first_batch_s"), m.get("launch_to_first_step_s"),
                     m.get("steady_step_s_wall_median"), m.get("steady_data_s_mean"),
                     m.get("peak_pss_tree_mb") or m.get("peak_rss_tree_mb"), m.get("peak_sys_used_mb")])
    if not rows:
        return ""
    return md_table(["系统", "启动→首个 batch (s)", "启动→首步 (s)", "稳态单步 (s)", "稳态等数据 (s)",
                     "进程树内存峰值 (MB)", "系统已用内存峰值 (MB)"], rows, nd=2)


def cmd_table(args: argparse.Namespace) -> int:
    probe = None if args.no_probe else probe_lerobot()
    results = json.loads(args.results.read_text(encoding="utf-8")) if args.results else None
    md = gap_table_markdown(probe, results, args.gr00t_metrics)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md, encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(md)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="Run lerobot-train --policy.type=groot on a Lance dataset")
    r.add_argument("--dataset-root", required=True, help="Lance dataset dir (or URI lerobot accepts)")
    r.add_argument("--repo-id", default=None, help="--dataset.repo_id (default local/<dir name>)")
    r.add_argument("--steps", type=int, default=300)
    r.add_argument("--batch-size", type=int, default=32)
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--num-workers", type=int, default=4)
    r.add_argument("--base-model-path", default="nvidia/GR00T-N1.7-3B")
    r.add_argument("--embodiment-tag", default=None, help="--policy.embodiment_tag (default: lerobot's)")
    r.add_argument("--device", default="cuda")
    r.add_argument("--output-dir", type=Path, required=True)
    r.add_argument("--job-name", default="issue5_lerobot_groot_lance")
    r.add_argument("--save-checkpoint", action="store_true", help="Also time LeRobot's checkpointing")
    r.add_argument("--lerobot-train", default="lerobot-train", help="Command to run (shlex-split)")
    r.add_argument("--results", type=Path, default=Path("results/v1/issue5/lerobot_results.json"))
    r.add_argument("--mem-interval-s", type=float, default=2.0)
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("extra", nargs=argparse.REMAINDER, help="-- extra lerobot-train flags")
    t = sub.add_parser("table", help="Write the Markdown feature-gap table")
    t.add_argument("--out", type=Path, default=None, help="Markdown file (default: stdout)")
    t.add_argument("--results", type=Path, default=None, help="results JSON from `run`")
    t.add_argument("--gr00t-metrics", type=Path, default=None, help="GR00T+Lance bench_metrics.jsonl to compare")
    t.add_argument("--no-probe", action="store_true", help="Do not inspect the installed lerobot")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        args.extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra
        return cmd_run(args)
    return cmd_table(args)


if __name__ == "__main__":
    sys.exit(main())
