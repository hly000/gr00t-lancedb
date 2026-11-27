#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Markdown table for ``run_cache_ablation.sh``: ``summarize_cache_ablation.py <out_root> <config>...``."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson interval for ``k`` successes in ``n`` trials."""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


def _load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def build_table(root: Path, configs: list[str]) -> str:
    probe = _load(root / "probe_compare.json") or {}
    ckpts = probe.get("checkpoints", {})
    lines = [
        "| config | optimizer | weights | step ms | peak GB | opt state GB | probe loss | closed-loop |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    if "native" in ckpts:
        n = ckpts["native"]
        lines.append(f"| native (online) | | | | | | {n['probe_loss']:.4f} ± {n['probe_sem']:.4f} | |")
    for cfg in configs:
        tr = _load(root / cfg / "train_summary.json")
        if tr is None:
            lines.append(f"| {cfg} | not run | | | | | | |")
            continue
        pr = ckpts.get(cfg)
        probe_s = f"{pr['probe_loss']:.4f} ± {pr['probe_sem']:.4f}" if pr else "n/a"
        ev = _load(root / cfg / "eval.json")
        if ev:
            n = int(ev["total_episodes"])
            rate = float(ev["mean_success_rate"])
            lo, hi = wilson(rate * n, n)
            ev_s = f"{100 * rate:.1f}% (n≈{n}, 95% CI {100 * lo:.0f}–{100 * hi:.0f}%)"
        else:
            ev_s = "not run"
        state = tr.get("optimizer_state_gb")
        lines.append(
            f"| {cfg} | {tr['optimizer']} | {tr.get('param_dtype', '?')} | {tr['median_step_ms']} | "
            f"{tr['peak_reserved_gb']} | {state if state is not None else ''} | {probe_s} | {ev_s} |"
        )
    out = ["\n".join(lines)]
    paired = probe.get("paired_vs_first", {})
    if paired:
        out.append("\nPaired probe differences (same frames, same noise):\n")
        out += [f"- {k}: {v['diff']:+.4f} ± {v['sem']:.4f} (z = {v['z']:+.1f})" for k, v in paired.items()]
    out.append(
        "\nThe closed-loop interval treats the mean of per-task rates as one binomial sample of "
        "about n episodes; it is a rough guide to the noise, not an exact test."
    )
    return "\n".join(out)


if __name__ == "__main__":
    print(build_table(Path(sys.argv[1]), sys.argv[2:]))
