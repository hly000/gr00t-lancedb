#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Summarize ``bench_metrics.jsonl`` files written by ``launch_finetune_lance.py``.

    python scripts/summarize_runs.py native=/data/runs/exp1/native lance=/data/runs/exp1/lance

Besides step-time / memory it reports the startup segments of ``startup_summary`` and,
when ``step.io`` is present, Lance blob range requests (~ object-store GETs), bytes and GOP
cache hit rate per step, plus an object-store request bill per 1k steps
(``--usd-per-1k-get``, default 0.0004 = S3 Standard GET / GCS Standard class B).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import statistics as st
import sys

STARTUP_KEYS = (
    "imports_s",
    "model_load_s",
    "dataset_build_s",
    "trainer_init_s",
    "ckpt_load_s",
    "dataloader_create_s",
    "first_batch_wait_s",
    "first_step_s",
    "lance_isolated_s",
    "lance_preload_s",
)
USD_PER_1K_GET = float(os.environ.get("USD_PER_1K_GET", "0.0004"))


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
    summaries = [r for r in recs if r["event"] == "startup_summary"]
    if summaries:
        for k in STARTUP_KEYS:
            if k in summaries[0]:
                out[f"startup.{k}"] = round(float(summaries[0][k]), 2)
        if len(summaries) > 1:  # resumed segments: cold start of each relaunch
            out["resume_first_step_since_launch_s"] = [round(r["since_launch_s"], 1) for r in summaries[1:]]
    out.update(io_summary(steps, warmup))
    losses = [s["step_loss"] for s in steps if "step_loss" in s]
    if losses:
        out["loss_first10_mean"] = round(st.mean(losses[:10]), 4)
        out["loss_last50_mean"] = round(st.mean(losses[-50:]), 4)
    elif logs:
        out["logged_loss_last"] = logs[-1].get("loss")
    return out


def io_summary(steps: list[dict], warmup: int = 10) -> dict:
    """Per-step object-store traffic from the cumulative ``io`` counters of one launch."""
    with_io = [s for s in steps if isinstance(s.get("io"), dict)]
    if len(with_io) < 2:
        return {}
    # Counters restart with every launch (new stats dir); use the longest run of
    # monotonically increasing request counts at the end (the last launch).
    tail = [with_io[-1]]
    for rec in reversed(with_io[:-1]):
        if rec["io"].get("requested_ranges", 0) <= tail[0]["io"].get("requested_ranges", 0):
            tail.insert(0, rec)
        else:
            break
    if len(tail) > warmup + 1:
        tail = tail[warmup:]
    a, b = tail[0]["io"], tail[-1]["io"]
    n = max(1, int(tail[-1]["global_step"]) - int(tail[0]["global_step"]))

    def d(k: str) -> float:
        return float(b.get(k, 0)) - float(a.get(k, 0))

    out = {
        "io.lance_ranges_per_step": round(d("lance_blob_ranges") / n, 2),
        "io.lance_calls_per_step": round(d("lance_blob_calls") / n, 2),
        "io.lance_mb_per_step": round(d("lance_blob_bytes") / n / 2**20, 3),
        "io.requested_mb_per_step": round(d("requested_bytes") / n / 2**20, 3),
        "io.frames_takes_per_step": round(d("frames_take_calls") / n, 2),
        "io.est_get_usd_per_1k_steps": round((d("lance_blob_ranges") + d("frames_take_calls")) / n * USD_PER_1K_GET, 5),
    }
    hits, misses = d("gop_cache_hits"), d("gop_cache_misses")
    if hits + misses > 0:
        out["io.gop_cache_hit_rate"] = round(hits / (hits + misses), 4)
    if "net_rx_bytes" in b and "net_rx_bytes" in a:
        out["io.net_rx_mb_per_step"] = round(d("net_rx_bytes") / n / 2**20, 3)
    return out


def main() -> None:
    global USD_PER_1K_GET
    argv = sys.argv[1:]
    if "--usd-per-1k-get" in argv:
        i = argv.index("--usd-per-1k-get")
        USD_PER_1K_GET = float(argv[i + 1])
        argv = argv[:i] + argv[i + 2 :]
    rows = {}
    for arg in argv:
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
