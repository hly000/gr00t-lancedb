#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Summarize ``bench_metrics.jsonl`` files written by ``launch_finetune_lance.py``.

    python scripts/summarize_runs.py native=/data/runs/exp1/native lance=/data/runs/exp1/lance
"""

from __future__ import annotations

import json
from pathlib import Path
import statistics as st
import sys


def load(run_dir: Path) -> list[dict]:
    path = run_dir / "bench_metrics.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(recs: list[dict], warmup: int = 10) -> dict:
    begins = [r for r in recs if r["event"] == "train_begin"]
    steps = [r for r in recs if r["event"] == "step"]
    logs = [r for r in recs if r["event"] == "log"]
    out: dict = {"segments": len(begins)}
    if not steps:
        return out
    first = steps[0]
    steady = [s for s in steps[warmup:]] or steps
    out.update(
        {
            "launch_to_train_begin_s": round(begins[0]["since_launch_s"], 1),
            "first_step_s": round(first["step_s"], 2),
            "first_batch_wait_s": round(first["data_wait_s"], 2),
            "launch_to_first_step_s": round(first["since_launch_s"], 1),
            "steps": len(steps),
            "last_step": steps[-1]["global_step"],
            "steady_step_s_mean": round(st.mean(s["step_s"] for s in steady), 3),
            "steady_step_s_p95": round(sorted(s["step_s"] for s in steady)[int(0.95 * (len(steady) - 1))], 3),
            "steady_data_wait_s_mean": round(st.mean(s["data_wait_s"] for s in steady), 4),
            "data_wait_frac": round(sum(s["data_wait_s"] for s in steady) / sum(s["step_s"] for s in steady), 4),
        }
    )
    utils = [s["gpu_util"] for s in steady if s.get("gpu_util") is not None]
    if utils:
        out["gpu_util_mean"] = round(st.mean(utils), 1)
    for key in ("sys_used_mb", "main_rss_mb", "pss_tree_mb", "gpu_mem_mb"):
        vals = [s[key] for s in steps if s.get(key)]
        if vals:
            out[f"peak_{key}"] = round(max(vals))
    losses = [s["step_loss"] for s in steps if "step_loss" in s]
    if losses:
        out["loss_first10_mean"] = round(st.mean(losses[:10]), 4)
        out["loss_last50_mean"] = round(st.mean(losses[-50:]), 4)
    elif logs:
        out["logged_loss_last"] = logs[-1].get("loss")
    return out


def main() -> None:
    rows = {}
    for arg in sys.argv[1:]:
        name, _, path = arg.partition("=")
        rows[name] = summarize(load(Path(path)))
    keys = sorted({k for r in rows.values() for k in r}, key=lambda k: list(next(iter(rows.values())).keys()).index(k) if k in next(iter(rows.values())) else 99)
    width = max(len(k) for k in keys)
    print("metric".ljust(width), *[n.rjust(14) for n in rows])
    for k in keys:
        print(k.ljust(width), *[str(rows[n].get(k, "-")).rjust(14) for n in rows])
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
