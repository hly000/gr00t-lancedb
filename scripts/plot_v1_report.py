#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Figures + Markdown tables for the v1 report from ``results/v1/**``.

Inputs (found recursively under ``--root``):

- ``bench_metrics.jsonl`` (from ``launch_finetune_lance.py``, or the copy the runner keeps);
- ``preempt_summary.json`` (from ``preempt_schedule_runner.py``);
- optional ``run_meta.json`` next to them, e.g. ``{"group": "B", "tier": "x50", "dataset_gb": 32.1}``.

Run labels come from, in increasing priority: ``key=value`` path components
(``results/v1/e1/group=B/tier=x50/rep=2/``), a path component like ``e1`` (experiment),
the summary's ``meta``, then ``run_meta.json``. ``dataset_gb`` can also come from the
``--manifest`` written by ``build_scale_tier_x.py`` (tier ``x<F>``).

Outputs in ``--out-dir`` (default ``<root>/report``):

- ``fig1_cold_start.png``: (a) cold-start stacked bars per group x tier, one segment per
  startup timestamp; (b) cold start vs dataset size per group (when >= 2 sizes);
- ``fig2_preempt_cost.png``: total cost (or GPU hours) to finish vs preemption rate, one
  line per group (one panel per kill mode);
- ``fig3_steady.png``: steady step time (with data-wait share), host RAM peaks
  (``pss_tree_mb`` / ``sys_used_mb``), object-store requests per 1k steps (from ``io``);
- ``report.md``: the same numbers as tables, plus the figure links.

Medians over repeats; whiskers / ranges are min-max. matplotlib is only imported when PNGs
are written (``--no-png`` for tables only).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import statistics as st
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v1_common import md_table, num, read_jsonl, split_segments, startup_breakdown, steady_stats  # noqa: E402

# Validated categorical palette (dataviz reference palette, light mode), fixed order.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GROUP_ORDER = ["A", "B", "C", "C1", "D", "C'", "Cp", "C′"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

# Cold-start stack (8 segments = 8 palette slots), in the order they happen.
SEGMENTS: list[tuple[str, str]] = [
    ("data_prep_s", "data prep (download / mount / meta)"),
    ("proc_s", "process start + imports"),
    ("model_load_s", "model load"),
    ("dataset_build_s", "dataset build (+ stats)"),
    ("trainer_ckpt_s", "trainer init + ckpt load"),
    ("dataloader_create_s", "dataloader create"),
    ("first_batch_wait_s", "first batch"),
    ("first_step_s", "first step"),
]


@dataclass
class Run:
    path: Path
    meta: dict[str, Any] = field(default_factory=dict)
    records: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] | None = None

    @property
    def group(self) -> str:
        return str(self.meta.get("group", self.path.name))

    @property
    def tier(self) -> str | None:
        t = self.meta.get("tier")
        return None if t is None else str(t)

    def label(self) -> str:
        return f"{self.group} · {self.tier}" if self.tier else self.group


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _path_meta(path: Path, root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts
    for part in parts:
        k, sep, v = part.partition("=")
        if sep:
            out[k] = num(v) if num(v) is not None and k not in ("group", "tier") else v
        elif re.fullmatch(r"[eE]\d+", part):
            out["experiment"] = part.lower()
    return out


def discover(root: Path) -> list[Run]:
    dirs = {p.parent for p in root.rglob("bench_metrics.jsonl")} | {p.parent for p in root.rglob("preempt_summary.json")}
    runs: list[Run] = []
    for d in sorted(dirs):
        run = Run(path=d, meta=_path_meta(d, root))
        if (d / "bench_metrics.jsonl").is_file():
            run.records = read_jsonl(d / "bench_metrics.jsonl")
        if (d / "preempt_summary.json").is_file():
            try:
                run.summary = json.loads((d / "preempt_summary.json").read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                run.summary = None
            if run.summary and isinstance(run.summary.get("meta"), dict):
                run.meta.update(run.summary["meta"])
        if (d / "run_meta.json").is_file():
            try:
                run.meta.update(json.loads((d / "run_meta.json").read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                pass
        runs.append(run)
    return runs


def _manifest_gb(manifest: dict[str, Any] | None, run: Run) -> float | None:
    if not manifest or not run.tier:
        return None
    m = re.fullmatch(r"x(\d+)", run.tier)
    if not m:
        return None
    tier = next((t for t in manifest.get("tiers", []) if int(t.get("factor", -1)) == int(m.group(1))), None)
    if not tier:
        return None
    variants = tier.get("variants", {})
    want = run.meta.get("variant") or ("lance_external" if run.group.upper().startswith(("C", "D")) else "v21")
    for name in (want, "lance_copy", "lance_external", "v21"):
        v = variants.get(name)
        if v and num(v.get("logical_bytes") or v.get("est_logical_bytes")) is not None:
            return float(v.get("logical_bytes") or v.get("est_logical_bytes")) / 1e9
    return None


def _filter(runs: list[Run], experiments: list[str] | None) -> list[Run]:
    if not experiments:
        return runs
    want = {e.lower() for e in experiments}
    return [r for r in runs if str(r.meta.get("experiment", "")).lower() in want]


def _group_key(g: str) -> tuple[int, str]:
    return (GROUP_ORDER.index(g) if g in GROUP_ORDER else len(GROUP_ORDER), g)


def group_colors(groups: list[str]) -> dict[str, str]:
    """Color follows the entity: known groups have fixed slots, others take the next free ones."""
    out: dict[str, str] = {}
    fixed = {"A": 0, "B": 1, "C": 2, "C1": 3, "D": 4, "C'": 5, "Cp": 5, "C′": 5}
    used = set()
    for g in groups:
        if g in fixed:
            out[g] = PALETTE[fixed[g]]
            used.add(fixed[g])
    free = [i for i in range(len(PALETTE)) if i not in used]
    for g in sorted(groups, key=_group_key):
        if g not in out:
            out[g] = PALETTE[free.pop(0)] if free else "#8a8984"
    return out


def _tier_key(t: str | None) -> tuple[float, str]:
    if t is None:
        return (0.0, "")
    m = re.search(r"(\d+(?:\.\d+)?)", t)
    return (float(m.group(1)) if m else 1e9, t)


def _stats(vals: list[float]) -> tuple[float | None, float | None, float | None]:
    v = [x for x in vals if x is not None]
    if not v:
        return None, None, None
    return st.median(v), min(v), max(v)


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


def cold_start_rows(runs: list[Run], manifest: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """One row per run: the startup segments of its FIRST launch (cold start)."""
    rows: list[dict[str, Any]] = []
    for run in runs:
        if not run.records:
            continue
        seg = split_segments(run.records)[0]
        sb = startup_breakdown(seg)
        if sb.get("launch_to_first_step_s") is None:
            continue
        prep = num(run.meta.get("data_prep_s"))
        if prep is None and run.summary:
            prep = num(run.summary.get("initial_reset_s"))
        row = {
            "run": str(run.path),
            "group": run.group,
            "tier": run.tier,
            "dataset_gb": num(run.meta.get("dataset_gb")) or _manifest_gb(manifest, run),
            "data_prep_s": prep,
            "proc_s": sum(sb.get(k) or 0.0 for k in ("imports_s", "other_s")),
            "model_load_s": sb.get("model_load_s"),
            "dataset_build_s": sb.get("dataset_build_s"),
            "trainer_ckpt_s": (sb.get("trainer_init_s") or 0.0) + (sb.get("ckpt_load_s") or 0.0),
            "dataloader_create_s": sb.get("dataloader_create_s"),
            "first_batch_wait_s": sb.get("first_batch_wait_s"),
            "first_step_s": sb.get("first_step_s"),
            "launch_to_first_step_s": sb["launch_to_first_step_s"],
        }
        row["cold_start_s"] = row["launch_to_first_step_s"] + (prep or 0.0)
        rows.append(row)
    return rows


def aggregate_cold(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by: dict[tuple[str, str | None], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[(r["group"], r["tier"])].append(r)
    out = []
    for (g, t), rs in sorted(by.items(), key=lambda kv: (_tier_key(kv[0][1]), _group_key(kv[0][0]))):
        agg: dict[str, Any] = {"group": g, "tier": t, "n": len(rs)}
        for k, _ in SEGMENTS:
            agg[k] = _stats([r.get(k) for r in rs])[0]
        agg["cold_start_s"], agg["cold_start_min"], agg["cold_start_max"] = _stats([r["cold_start_s"] for r in rs])
        agg["dataset_gb"] = _stats([r.get("dataset_gb") for r in rs])[0]
        out.append(agg)
    return out


def preempt_rows(runs: list[Run], usd_per_hour: float | None = None) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        s = run.summary
        if not s or "total_wall_s" not in s:
            continue
        rate = num(run.meta.get("rate_per_hour"))
        if rate is None:
            rate = num(((s.get("schedule") or {}).get("generator") or {}).get("rate_per_hour"))
        if rate is None and not (s.get("schedule") or {}).get("times_s"):
            rate = 0.0
        billable = num(s.get("billable_s")) or num(s.get("total_wall_s")) or 0.0
        price = usd_per_hour if usd_per_hour is not None else num(s.get("gpu_usd_per_hour"))
        rows.append({
            "run": str(run.path),
            "group": run.group,
            "kill_mode": s.get("kill_mode") or "term",
            "rate_per_hour": rate,
            "seed": run.meta.get("seed"),
            "completed": bool(s.get("completed")),
            "total_wall_s": num(s.get("total_wall_s")),
            "gpu_hours": billable / 3600.0,
            "cost_usd": billable / 3600.0 * price if price else None,
            "interruptions": num(s.get("interruptions")),
            "steps_reexecuted": num(s.get("steps_reexecuted")),
            "wasted_gpu_s": num(s.get("wasted_gpu_s")),
            "recovery_s_median": num(s.get("recovery_s_median")),
        })
    return rows


def aggregate_preempt(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by: dict[tuple[str, str, float | None], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[(r["group"], r["kill_mode"], r["rate_per_hour"])].append(r)
    out = []
    for (g, km, rate), rs in sorted(by.items(), key=lambda kv: (kv[0][1], _group_key(kv[0][0]), kv[0][2] or 0.0)):
        agg: dict[str, Any] = {"group": g, "kill_mode": km, "rate_per_hour": rate, "n": len(rs),
                               "completed": sum(r["completed"] for r in rs)}
        for k in ("cost_usd", "gpu_hours", "total_wall_s", "interruptions", "steps_reexecuted", "wasted_gpu_s",
                  "recovery_s_median"):
            med, lo, hi = _stats([r.get(k) for r in rs])
            agg[k], agg[f"{k}_min"], agg[f"{k}_max"] = med, lo, hi
        out.append(agg)
    return out


def steady_rows(runs: list[Run], warmup: int = 10) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        if not run.records:
            continue
        s = steady_stats(run.records, warmup=warmup)
        if "steady_step_s_median" not in s:
            continue
        io = s.get("io_per_1k_steps", {})
        rows.append({
            "run": str(run.path), "group": run.group, "tier": run.tier,
            "step_s": s.get("steady_step_s_median"), "data_wait_s": s.get("steady_data_wait_s_mean"),
            "data_wait_frac": s.get("data_wait_frac"),
            "pss_tree_mb": s.get("peak_pss_tree_mb"), "sys_used_mb": s.get("peak_sys_used_mb"),
            "requests_per_1k": io.get("lance_blob_calls", io.get("lance_blob_ranges")),
            "ranges_per_1k": io.get("lance_blob_ranges"),
            "mb_per_1k": (io["lance_blob_bytes"] / 1e6) if "lance_blob_bytes" in io else None,
            "net_rx_mb_per_1k": (io["net_rx_bytes"] / 1e6) if "net_rx_bytes" in io else None,
            "gop_hit_rate": (io["gop_cache_hits"] / (io["gop_cache_hits"] + io["gop_cache_misses"]))
            if io.get("gop_cache_hits") is not None and io.get("gop_cache_misses") is not None
            and (io["gop_cache_hits"] + io["gop_cache_misses"]) > 0 else None,
        })
    return rows


def aggregate_steady(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by: dict[tuple[str, str | None], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[(r["group"], r["tier"])].append(r)
    out = []
    for (g, t), rs in sorted(by.items(), key=lambda kv: (_tier_key(kv[0][1]), _group_key(kv[0][0]))):
        agg: dict[str, Any] = {"group": g, "tier": t, "n": len(rs)}
        for k in ("step_s", "data_wait_s", "data_wait_frac", "pss_tree_mb", "sys_used_mb", "requests_per_1k",
                  "ranges_per_1k", "mb_per_1k", "net_rx_mb_per_1k", "gop_hit_rate"):
            agg[k] = _stats([r.get(k) for r in rs])[0]
        out.append(agg)
    return out


# ---------------------------------------------------------------------------
# Figures (matplotlib imported lazily)
# ---------------------------------------------------------------------------


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2,
        "ytick.color": INK_2, "axes.titlecolor": INK, "axes.titlesize": 10, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white",
    })
    return plt


def _style_axis(ax) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_fig1(cold: list[dict[str, Any]], out: Path) -> Path | None:
    if not cold:
        return None
    plt = _plt()
    sizes = defaultdict(set)
    for a in cold:
        if a.get("dataset_gb"):
            sizes[a["group"]].add(round(a["dataset_gb"], 6))
    with_line = any(len(v) >= 2 for v in sizes.values())
    fig, axes = plt.subplots(1, 2 if with_line else 1, figsize=(12 if with_line else 8, 4.6),
                             gridspec_kw={"width_ratios": [1.5, 1]} if with_line else None)
    ax = axes[0] if with_line else axes
    labels = [f"{a['group']} · {a['tier']}" if a["tier"] else a["group"] for a in cold]
    x = list(range(len(cold)))
    bottom = [0.0] * len(cold)
    for i, (key, name) in enumerate(SEGMENTS):
        vals = [a.get(key) or 0.0 for a in cold]
        if not any(vals):
            continue
        ax.bar(x, vals, bottom=bottom, width=0.6, color=PALETTE[i], edgecolor="white", linewidth=1.0, label=name)
        bottom = [b + v for b, v in zip(bottom, vals)]
    for xi, a, b in zip(x, cold, bottom):
        ax.annotate(f"{a['cold_start_s']:.0f}", (xi, b), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x, labels, rotation=30 if len(cold) > 6 else 0, ha="right" if len(cold) > 6 else "center")
    ax.set_ylabel("time to first training step (s)")
    ax.set_title("(a) Cold start by startup segment (median of repeats)", loc="left")
    _style_axis(ax)
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.2 if len(cold) > 6 else -0.1), ncol=4)
    if with_line:
        ax2 = axes[1]
        colors = group_colors(sorted({a["group"] for a in cold}, key=_group_key))
        for g in sorted(sizes, key=_group_key):
            pts = sorted((a["dataset_gb"], a) for a in cold if a["group"] == g and a.get("dataset_gb"))
            xs = [p[0] for p in pts]
            ys = [p[1]["cold_start_s"] for p in pts]
            lo = [y - p[1]["cold_start_min"] for y, p in zip(ys, pts)]
            hi = [p[1]["cold_start_max"] - y for y, p in zip(ys, pts)]
            ax2.errorbar(xs, ys, yerr=[lo, hi], color=colors[g], linewidth=2, marker="o", markersize=6,
                         capsize=3, label=g)
            ax2.annotate(g, (xs[-1], ys[-1]), xytext=(5, 0), textcoords="offset points", va="center",
                         fontsize=8, color=INK)
        ax2.set_xscale("log")
        ax2.set_xlabel("dataset size (GB, log scale)")
        ax2.set_ylabel("cold start incl. data prep (s)")
        ax2.set_title("(b) Cold start vs dataset size", loc="left")
        _style_axis(ax2)
        ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_fig2(pre: list[dict[str, Any]], out: Path, metric: str) -> Path | None:
    pre = [p for p in pre if p.get(metric) is not None and p.get("rate_per_hour") is not None]
    if not pre:
        return None
    plt = _plt()
    modes = sorted({p["kill_mode"] for p in pre})
    fig, axes = plt.subplots(1, len(modes), figsize=(5.5 * len(modes), 4.2), squeeze=False, sharey=True)
    colors = group_colors(sorted({p["group"] for p in pre}, key=_group_key))
    ylabel = "total cost to finish (USD)" if metric == "cost_usd" else "GPU time to finish (h)"
    for ax, mode in zip(axes[0], modes):
        for g in sorted({p["group"] for p in pre if p["kill_mode"] == mode}, key=_group_key):
            pts = sorted((p["rate_per_hour"], p) for p in pre if p["kill_mode"] == mode and p["group"] == g)
            xs = [r for r, _ in pts]
            ys = [p[metric] for _, p in pts]
            lo = [y - (p[f"{metric}_min"] if p[f"{metric}_min"] is not None else y) for y, (_, p) in zip(ys, pts)]
            hi = [(p[f"{metric}_max"] if p[f"{metric}_max"] is not None else y) - y for y, (_, p) in zip(ys, pts)]
            ax.errorbar(xs, ys, yerr=[lo, hi], color=colors[g], linewidth=2, marker="o", markersize=6, capsize=3,
                        label=g)
            ax.annotate(g, (xs[-1], ys[-1]), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8,
                        color=INK)
        name = {"kill": "hard kill (SIGKILL)", "term": "SIGTERM + grace"}.get(mode, mode)
        ax.set_title(f"Cost vs preemption rate — {name}", loc="left")
        ax.set_xlabel("preemptions per hour of instance time")
        _style_axis(ax)
        ax.legend(fontsize=8)
    axes[0][0].set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _hatched_bar(ax, x, heights, colors, bottom=None, width=0.6):
    """Bars with white fill and a group-colored hatch (the second measure of a group)."""
    for xi, h, c, b in zip(x, heights, colors, bottom or [0.0] * len(heights)):
        kw = {"hatch": "////", "facecolor": "white", "linewidth": 1.0}
        try:
            ax.bar(xi, h, bottom=b, width=width, edgecolor=c, **kw)
        except (AttributeError, TypeError):  # pragma: no cover - very old matplotlib
            ax.bar(xi, h, bottom=b, width=width, color=c)


def _legend_proxies(ax, entries: list[tuple[str, bool]], ncol: int = 2) -> None:
    from matplotlib.patches import Patch

    handles = [Patch(facecolor="white", edgecolor=INK_2, hatch="////", label=lbl) if hatched
               else Patch(facecolor="#8a8984", edgecolor="white", label=lbl) for lbl, hatched in entries]
    ax.legend(handles=handles, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=ncol)


def plot_fig3(steady: list[dict[str, Any]], out: Path) -> Path | None:
    if not steady:
        return None
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    labels = [f"{a['group']} · {a['tier']}" if a["tier"] else a["group"] for a in steady]
    gcol = group_colors(sorted({a["group"] for a in steady}, key=_group_key))
    colors = [gcol[a["group"]] for a in steady]  # color follows the group, as in fig 1 and 2
    x = list(range(len(steady)))
    rot = {"rotation": 30, "ha": "right"} if len(steady) > 5 else {}

    ax = axes[0]
    wait = [a.get("data_wait_s") or 0.0 for a in steady]
    rest = [max(0.0, (a.get("step_s") or 0.0) - w) for a, w in zip(steady, wait)]
    ax.bar(x, rest, width=0.6, color=colors, edgecolor="white", linewidth=1.0)
    _hatched_bar(ax, x, wait, colors, bottom=rest)
    for xi, a in zip(x, steady):
        if a.get("data_wait_frac") is not None:
            ax.annotate(f"{a['data_wait_frac'] * 100:.1f}% wait", (xi, a.get("step_s") or 0.0), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(x, labels, **rot)
    ax.set_ylabel("steady step time (s, median)")
    ax.set_title("(a) Steady step time", loc="left")
    _legend_proxies(ax, [("compute (rest of step)", False), ("data wait", True)])
    _style_axis(ax)

    ax = axes[1]
    w = 0.38
    pss = [(a.get("pss_tree_mb") or 0.0) / 1024 for a in steady]
    sysm = [(a.get("sys_used_mb") or 0.0) / 1024 for a in steady]
    ax.bar([i - w / 2 for i in x], pss, width=w, color=colors, edgecolor="white", linewidth=1.0)
    _hatched_bar(ax, [i + w / 2 for i in x], sysm, colors, width=w)
    ax.set_xticks(x, labels, **rot)
    ax.set_ylabel("peak host RAM (GiB)")
    ax.set_title("(b) Host memory peak", loc="left")
    _legend_proxies(ax, [("training process tree (PSS)", False), ("whole machine (used)", True)])
    _style_axis(ax)

    ax = axes[2]
    req = [a.get("requests_per_1k") for a in steady]
    if any(v is not None for v in req):
        ax.bar(x, [v or 0.0 for v in req], width=0.6, color=colors, edgecolor="white", linewidth=1.0)
        for xi, v in zip(x, req):
            ax.annotate("n/a" if v is None else f"{v:,.0f}", (xi, v or 0.0), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=7.5, color=INK)
        ax.set_ylabel("Lance blob read calls per 1k steps (count)")
    else:
        ax.text(0.5, 0.5, "no `io` counters in these logs", transform=ax.transAxes, ha="center", color=INK_2)
        ax.set_yticks([])
    ax.set_xticks(x, labels, **rot)
    ax.set_title("(c) Object-store reads after local cache", loc="left")
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def build_report(root: Path, out_dir: Path, *, manifest: dict[str, Any] | None = None, png: bool = True,
                 usd_per_hour: float | None = None, usd_per_1k_requests: float = 0.0004, warmup: int = 10,
                 exp: dict[str, list[str] | None] | None = None) -> dict[str, Any]:
    exp = exp or {}
    runs = discover(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    cold = aggregate_cold(cold_start_rows(_filter(runs, exp.get("fig1")), manifest))
    pre = aggregate_preempt(preempt_rows(_filter(runs, exp.get("fig2")), usd_per_hour))
    steady = aggregate_steady(steady_rows(_filter(runs, exp.get("fig3")), warmup))
    metric = "cost_usd" if pre and all(p.get("cost_usd") is not None for p in pre) else "gpu_hours"
    figs: dict[str, Path | None] = {}
    if png:
        figs["fig1"] = plot_fig1(cold, out_dir / "fig1_cold_start.png")
        figs["fig2"] = plot_fig2(pre, out_dir / "fig2_preempt_cost.png", metric)
        figs["fig3"] = plot_fig3(steady, out_dir / "fig3_steady.png")

    md = ["# v1 实验报告（自动生成）", "", f"数据目录：`{root}`，共 {len(runs)} 个运行目录。中位数；范围为最小–最大。", ""]
    md += ["## 图 1 冷启动（首个训练步之前的各段耗时，单位 s）", ""]
    if figs.get("fig1"):
        md += ["![fig1](fig1_cold_start.png)", ""]
    if cold:
        md.append(md_table(["组", "档", "次数", "数据集 GB", *[name for _, name in SEGMENTS], "合计 (min–max)"],
                           [[a["group"], a["tier"], a["n"], a["dataset_gb"], *[a.get(k) for k, _ in SEGMENTS],
                             f"{a['cold_start_s']:.1f} ({a['cold_start_min']:.1f}–{a['cold_start_max']:.1f})"]
                            for a in cold]))
    else:
        md.append("（没有带 step 记录的 bench_metrics.jsonl）")
    md += ["", "## 图 2 抢占：完成训练的总成本", ""]
    if figs.get("fig2"):
        md += ["![fig2](fig2_preempt_cost.png)", ""]
    if pre:
        md.append(md_table(["组", "抢占方式", "次/小时", "次数", "完成", "总成本 USD", "GPU 小时", "总墙钟 s",
                            "中断次数", "重做步数", "浪费 GPU s", "恢复耗时中位 s"],
                           [[p["group"], p["kill_mode"], p["rate_per_hour"], p["n"], p["completed"], p["cost_usd"],
                             p["gpu_hours"], p["total_wall_s"], p["interruptions"], p["steps_reexecuted"],
                             p["wasted_gpu_s"], p["recovery_s_median"]] for p in pre], nd=2))
    else:
        md.append("（没有 preempt_summary.json）")
    md += ["", "## 图 3 稳态：单步时间、内存、对象存储请求", ""]
    if figs.get("fig3"):
        md += ["![fig3](fig3_steady.png)", ""]
    if steady:
        md.append(md_table(["组", "档", "次数", "单步 s", "等数据 s", "等数据占比", "进程树 PSS 峰值 MB",
                            "系统已用峰值 MB", "Lance 读调用/1k 步", "读区间/1k 步", "读取 MB/1k 步", "网卡收 MB/1k 步",
                            "GOP 缓存命中率", "请求费 USD/1k 步"],
                           [[a["group"], a["tier"], a["n"], a["step_s"], a["data_wait_s"], a["data_wait_frac"],
                             a["pss_tree_mb"], a["sys_used_mb"], a["requests_per_1k"], a["ranges_per_1k"],
                             a["mb_per_1k"], a["net_rx_mb_per_1k"], a["gop_hit_rate"],
                             (a["requests_per_1k"] / 1000.0 * usd_per_1k_requests)
                             if a["requests_per_1k"] is not None else None] for a in steady], nd=3))
        md.append("")
        md.append(f"“Lance 读调用”是穿过本地 GOP 缓存、真正发给 Lance 的读取次数（`io.lance_blob_calls`），"
                  f"只是对象存储 GET 次数的近似；请求费按每 1,000 次 ${usd_per_1k_requests} 估算"
                  "（`--usd-per-1k-requests` 可改）。")
    else:
        md.append("（没有可用的稳态 step 记录）")
    report = out_dir / "report.md"
    report.write_text("\n".join(md) + "\n", encoding="utf-8")
    return {"runs": len(runs), "report": str(report), "figures": {k: str(v) for k, v in figs.items() if v},
            "cold": cold, "preempt": pre, "steady": steady, "fig2_metric": metric}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path("results/v1"))
    p.add_argument("--out-dir", type=Path, default=None, help="Default <root>/report")
    p.add_argument("--manifest", type=Path, default=None, help="manifest.json from build_scale_tier_x.py")
    p.add_argument("--gpu-usd-per-hour", type=float, default=None, help="Re-price every preemption run")
    p.add_argument("--usd-per-1k-requests", type=float, default=0.0004)
    p.add_argument("--warmup", type=int, default=10, help="Steps dropped at the start of every launch")
    p.add_argument("--fig1-exp", nargs="*", default=None, help="Only runs of these experiments (e.g. e1)")
    p.add_argument("--fig2-exp", nargs="*", default=None)
    p.add_argument("--fig3-exp", nargs="*", default=None)
    p.add_argument("--no-png", action="store_true")
    args = p.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8")) if args.manifest else None
    res = build_report(args.root, args.out_dir or (args.root / "report"), manifest=manifest, png=not args.no_png,
                       usd_per_hour=args.gpu_usd_per_hour, usd_per_1k_requests=args.usd_per_1k_requests,
                       warmup=args.warmup, exp={"fig1": args.fig1_exp, "fig2": args.fig2_exp, "fig3": args.fig3_exp})
    print(json.dumps({k: res[k] for k in ("runs", "report", "figures", "fig2_metric")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
