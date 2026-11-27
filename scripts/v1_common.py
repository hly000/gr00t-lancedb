# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the v1 experiment scripts (pure Python, stdlib only).

Parses the ``bench_metrics.jsonl`` files written by ``scripts/launch_finetune_lance.py``.
Every reader here tolerates older logs: missing fields become ``None`` and a truncated last
line (process killed mid-write) is skipped.

Record kinds (``event`` field): ``phase``, ``train_begin``, ``step``, ``startup_summary``,
``save``, ``log``, ``train_end``. Each launch of the trainer appends to the same file and
restarts its ``since_launch_s`` clock, which is how launches ("segments") are told apart.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import json
import math
from pathlib import Path
import statistics as st
from typing import Any

# Startup segments, in the order they happen (keys of a ``startup_summary`` record).
STARTUP_KEYS: tuple[str, ...] = (
    "imports_s",
    "model_load_s",
    "dataset_build_s",
    "trainer_init_s",
    "ckpt_load_s",
    "dataloader_create_s",
    "first_batch_wait_s",
    "first_step_s",
)

# ``phase`` record name -> startup key (fallback when an old log has no startup_summary).
_PHASE_TO_KEY = {
    "imports": "imports_s",
    "model_load": "model_load_s",
    "dataset_build": "dataset_build_s",
    "trainer_init": "trainer_init_s",
    "ckpt_load": "ckpt_load_s",
    "dataloader_create": "dataloader_create_s",
    "first_batch": "first_batch_wait_s",
}

# Cumulative counters that ``step`` records may carry in an ``io`` dict.
IO_KEYS: tuple[str, ...] = (
    "lance_blob_calls",
    "lance_blob_ranges",
    "lance_blob_bytes",
    "gop_cache_hits",
    "gop_cache_misses",
    "gop_cache_hit_bytes",
    "gop_cache_miss_bytes",
    "net_rx_bytes",
)


def read_jsonl(path: str | Path, offset: int = 0) -> list[dict[str, Any]]:
    """Read JSON lines starting at byte ``offset``; skip blank and unparseable lines."""
    p = Path(path)
    if not p.is_file():
        return []
    with open(p, "rb") as fh:
        fh.seek(offset)
        data = fh.read()
    out: list[dict[str, Any]] = []
    for raw in data.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            rec = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def num(x: Any) -> float | None:
    """``float(x)`` if ``x`` is a finite number, else ``None``."""
    if isinstance(x, bool) or x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def split_segments(records: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Split one ``bench_metrics.jsonl`` into launches.

    A new launch starts when ``since_launch_s`` goes backwards. Records without that field
    (``log``) stay with the current launch.
    """
    segs: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    prev: float | None = None
    for rec in records:
        s = num(rec.get("since_launch_s"))
        if s is not None and prev is not None and s + 1e-6 < prev and cur:
            segs.append(cur)
            cur = []
        if s is not None:
            prev = s
        cur.append(rec)
    if cur:
        segs.append(cur)
    return segs


def steps_of(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in records if r.get("event") == "step" and num(r.get("global_step")) is not None]


def last_step(records: Iterable[dict[str, Any]]) -> int | None:
    vals = [int(num(r.get("global_step")) or 0) for r in steps_of(records)]
    return max(vals) if vals else None


def startup_breakdown(segment: Sequence[dict[str, Any]]) -> dict[str, float | None]:
    """Startup durations of one launch, in seconds.

    Uses the ``startup_summary`` record when present, filling gaps from ``phase`` records.
    ``launch_to_first_step_s`` is the wall time from process start to the end of the first
    training step; ``other_s`` is whatever the named segments do not explain (>= 0).
    """
    out: dict[str, float | None] = {k: None for k in STARTUP_KEYS}
    summary = next((r for r in segment if r.get("event") == "startup_summary"), None)
    if summary is not None:
        for k in STARTUP_KEYS:
            out[k] = num(summary.get(k))
    for r in segment:
        if r.get("event") == "phase":
            key = _PHASE_TO_KEY.get(str(r.get("phase")))
            if key and out.get(key) is None:
                out[key] = num(r.get("duration_s"))
    steps = steps_of(segment)
    first = steps[0] if steps else None
    if out["first_step_s"] is None and first is not None:
        out["first_step_s"] = num(first.get("step_s"))
    if out["first_batch_wait_s"] is None and first is not None:
        out["first_batch_wait_s"] = num(first.get("data_wait_s"))
    total = None
    if summary is not None:
        total = num(summary.get("since_launch_s"))
    if total is None and first is not None:
        total = num(first.get("since_launch_s"))
    out["launch_to_first_step_s"] = total
    if total is not None:
        known = sum(v for k in STARTUP_KEYS if (v := out.get(k)) is not None)
        out["other_s"] = max(0.0, total - known)
    else:
        out["other_s"] = None
    return out


def percentile(vals: Sequence[float], q: float) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    pos = q * (len(s) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def io_rates_per_1k_steps(segments: Sequence[Sequence[dict[str, Any]]]) -> dict[str, float]:
    """Object-store / cache counters per 1,000 steps, from cumulative ``io`` dicts.

    Counters are per process, so each launch is differenced on its own (first step with an
    ``io`` dict is the baseline, which leaves out the first-batch fetch) and the launches
    are pooled weighted by steps.
    """
    tot: dict[str, float] = {}
    steps_total = 0
    for seg in segments:
        with_io = [r for r in steps_of(seg) if isinstance(r.get("io"), dict)]
        if len(with_io) < 2:
            continue
        a, b = with_io[0], with_io[-1]
        dsteps = int(num(b.get("global_step")) or 0) - int(num(a.get("global_step")) or 0)
        if dsteps <= 0:
            continue
        steps_total += dsteps
        for k in set(a["io"]) | set(b["io"]):
            va, vb = num(a["io"].get(k)), num(b["io"].get(k))
            if va is not None and vb is not None:
                tot[k] = tot.get(k, 0.0) + (vb - va)
    if steps_total == 0:
        return {}
    return {k: v * 1000.0 / steps_total for k, v in sorted(tot.items())}


def steady_stats(records: Sequence[dict[str, Any]], warmup: int = 10) -> dict[str, Any]:
    """Steady-state step time, data wait, host memory and I/O rates over all launches.

    The first ``warmup`` steps of every launch are dropped (fewer if the launch is short).
    """
    segs = split_segments(records)
    steady: list[dict[str, Any]] = []
    all_steps: list[dict[str, Any]] = []
    for seg in segs:
        ss = steps_of(seg)
        all_steps.extend(ss)
        w = min(warmup, max(0, len(ss) - 1))
        steady.extend(ss[w:])
    out: dict[str, Any] = {"launches": len(segs), "step_records": len(all_steps)}
    step_s = [v for r in steady if (v := num(r.get("step_s"))) is not None]
    wait_s = [v for r in steady if (v := num(r.get("data_wait_s"))) is not None]
    if step_s:
        out["steady_step_s_median"] = st.median(step_s)
        out["steady_step_s_mean"] = st.mean(step_s)
        out["steady_step_s_p95"] = percentile(step_s, 0.95)
    if wait_s:
        out["steady_data_wait_s_mean"] = st.mean(wait_s)
        if step_s and sum(step_s) > 0:
            out["data_wait_frac"] = sum(wait_s) / sum(step_s)
    for key in ("pss_tree_mb", "sys_used_mb", "main_rss_mb", "gpu_mem_mb"):
        vals = [v for r in all_steps if (v := num(r.get(key))) is not None and v > 0]
        if vals:
            out[f"peak_{key}"] = max(vals)
    utils = [v for r in steady if (v := num(r.get("gpu_util"))) is not None]
    if utils:
        out["gpu_util_mean"] = st.mean(utils)
    io = io_rates_per_1k_steps(segs)
    if io:
        out["io_per_1k_steps"] = io
    return out


def median_or_none(vals: Iterable[float | None]) -> float | None:
    v = [x for x in vals if x is not None]
    return st.median(v) if v else None


def fmt(v: Any, nd: int = 1) -> str:
    """Compact table cell: numbers rounded, ``None`` as an en dash."""
    if v is None:
        return "–"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if abs(v) >= 1e6:
            return f"{v:.3g}"
        return f"{v:.{nd}f}"
    return str(v)


def md_table(headers: Sequence[str], rows: Iterable[Sequence[Any]], nd: int = 1) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(c, nd) for c in row) + " |")
    return "\n".join(lines)
