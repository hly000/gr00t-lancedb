#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pass/fail checks and summaries for ``run_online_turbo.sh`` (standard library only).

    turbo_run_checks.py smoke RUN_DIR LOG [--expect-compile] [--no-turbo]
        5-step smoke gate: finite losses, turbo active (or off for stock runs), torch.compile
        really applied when it was requested, no ``_orig_mod`` checkpoint keys.

    turbo_run_checks.py bench RUN_DIR LOG --mode NAME --steps N [--out-json PATH]
        Short speed/memory run: steady step time, peak GPU memory, and whether the run ended
        in a CUDA out-of-memory error (a result, not a failure: exit code 0).

    turbo_run_checks.py table OUT_ROOT MODE [MODE ...] [--gpu NAME] [--out-json PATH]
        Collect ``OUT_ROOT/bench_<MODE>/bench_summary.json`` into one Markdown table.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

GB24_MB = 24 * 1024
OOM_PATTERNS = ("out of memory", "outofmemoryerror")


def load_rows(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "bench_metrics.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _step_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("event") == "step"]


def _turbo_record(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    recs = [r["online_turbo"] for r in rows if isinstance(r.get("online_turbo"), dict)]
    return recs[-1] if recs else None


def smoke_problems(
    rows: list[dict[str, Any]],
    log_text: str,
    *,
    n_steps: int = 5,
    expect_turbo: bool = True,
    expect_compile: bool = False,
    orig_mod_keys: list[str] | None = None,
) -> list[str]:
    """Everything wrong with a smoke run (empty list: the gate passes)."""
    problems: list[str] = []
    steps = _step_rows(rows)
    if len(steps) < n_steps:
        problems.append(f"expected {n_steps} step rows, got {len(steps)}")
    for r in steps:
        loss = r.get("step_loss")
        if loss is None or not math.isfinite(float(loss)):
            problems.append(f"non-finite loss at step {r.get('global_step')}: {loss}")
    turbo = _turbo_record(rows)
    if not expect_turbo:
        if turbo and turbo.get("enabled"):
            problems.append(f"stock run has online_turbo enabled: {turbo}")
    elif not turbo or not turbo.get("enabled"):
        problems.append(f"online_turbo was not active: {turbo}")
    elif expect_compile and not turbo.get("compile_applied"):
        problems.append(
            "COMPILE_HEAD=1 but torch.compile was not applied "
            f"({turbo.get('compile_skipped_reason') or 'no reason recorded'}); the run would be timed uncompiled"
        )
    if orig_mod_keys:
        problems.append(f"compiled _orig_mod keys leaked into the checkpoint: {orig_mod_keys[:5]}")
    return problems


def bench_summary(
    rows: list[dict[str, Any]], log_text: str, *, mode: str, steps: int, warmup: int = 10
) -> dict[str, Any]:
    """Steady-state numbers of a short run. ``status`` is ``ok``, ``oom`` or ``error``."""
    step_rows = _step_rows(rows)
    low = log_text.lower()
    if len(step_rows) >= steps:
        status = "ok"
    elif any(pat in low for pat in OOM_PATTERNS):
        status = "oom"
    else:
        status = "error"
    out: dict[str, Any] = {"mode": mode, "status": status, "steps_requested": steps, "steps_done": len(step_rows)}
    mems = [r["gpu_mem_mb"] for r in step_rows if r.get("gpu_mem_mb") is not None]
    if mems:
        out["peak_gpu_mem_gb"] = round(max(mems) / 1024, 2)
        out["fits_24gb"] = max(mems) <= GB24_MB
    # nvidia-smi (above) is device-wide and includes the CUDA context; these are the
    # process's own allocator peaks, i.e. what torch.cuda.max_memory_allocated() reports.
    for key, name in (("torch_max_alloc_mb", "peak_torch_alloc_gb"), ("torch_max_reserved_mb", "peak_torch_reserved_gb")):
        vals = [r[key] for r in step_rows if r.get(key) is not None]
        if vals:
            out[name] = round(max(vals) / 1024, 2)
    limit = next((r for r in rows if r.get("event") == "cuda_mem_limit"), None)
    if limit:
        out["cuda_mem_fraction"] = limit.get("fraction")
        out["cuda_mem_limit_gib"] = limit.get("limit_gib")
    # torch.compile and cuDNN autotune make the first steps slow; a short run that cannot
    # spare `warmup` steps still reports, on its second half.
    steady = step_rows[warmup:] if len(step_rows) >= 2 * warmup else step_rows[len(step_rows) // 2 :]
    if status == "ok" and steady:
        ms = [1e3 * float(r["step_s"]) for r in steady]
        out["median_step_ms"] = round(statistics.median(ms), 2)
        out["mean_step_ms"] = round(statistics.fmean(ms), 2)
        out["steady_steps"] = len(ms)
    if status == "oom":
        out["oom_line"] = next((ln.strip()[:300] for ln in log_text.splitlines() if "out of memory" in ln.lower()), "")
    turbo = _turbo_record(rows)
    if turbo:
        out["online_turbo"] = turbo
    return out


def bench_table(summaries: list[dict[str, Any]], gpu: str = "") -> str:
    """Markdown table of bench summaries, speed-up relative to the first run that finished."""
    ref = next((s["median_step_ms"] for s in summaries if s.get("median_step_ms")), None)
    lines = [
        f"| mode ({gpu or 'GPU'}) | result | peak GPU memory | ms/step | vs first finished "
        "| torch max_alloc | allocator cap |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        ms = s.get("median_step_ms")
        peak = f"{s['peak_gpu_mem_gb']:.2f} GB" if s.get("peak_gpu_mem_gb") is not None else "n/a"
        if s["status"] == "ok":
            result = "runs"
        elif s["status"] == "oom":
            result = f"CUDA OOM at step {s['steps_done'] + 1}"
            peak = f">= {peak}" if peak != "n/a" else peak
        else:
            result = "ERROR (not OOM)"
        speed = f"{ref / ms:.2f}x" if (ms and ref) else "-"
        alloc = f"{s['peak_torch_alloc_gb']:.2f} GB" if s.get("peak_torch_alloc_gb") is not None else "-"
        cap = f"{s['cuda_mem_limit_gib']:.1f} GiB" if s.get("cuda_mem_limit_gib") is not None else "none"
        lines.append(f"| `{s['mode']}` | {result} | {peak} | {ms if ms else '-'} | {speed} | {alloc} | {cap} |")
    return "\n".join(lines)


def _orig_mod_keys(run_dir: Path, n_steps: int) -> tuple[list[str], int]:
    from safetensors import safe_open

    ckpt = run_dir / f"checkpoint-{n_steps}"
    if not ckpt.is_dir():
        ckpt = run_dir
    files = sorted(ckpt.glob("*.safetensors"))
    bad: list[str] = []
    for sf in files:
        with safe_open(str(sf), framework="pt", device="cpu") as fh:
            bad.extend(k for k in fh.keys() if "_orig_mod" in k)
    return bad, len(files)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sm = sub.add_parser("smoke")
    sm.add_argument("run_dir")
    sm.add_argument("log")
    sm.add_argument("--steps", type=int, default=5)
    sm.add_argument("--no-turbo", action="store_true", help="stock baseline: turbo must be off")
    sm.add_argument("--expect-compile", action="store_true")
    be = sub.add_parser("bench")
    be.add_argument("run_dir")
    be.add_argument("log")
    be.add_argument("--mode", required=True)
    be.add_argument("--steps", type=int, required=True)
    be.add_argument("--warmup", type=int, default=10)
    be.add_argument("--out-json", default=None)
    tb = sub.add_parser("table")
    tb.add_argument("out_root")
    tb.add_argument("modes", nargs="+")
    tb.add_argument("--gpu", default="")
    tb.add_argument("--out-json", default=None)
    args = ap.parse_args(argv)

    if args.cmd == "table":
        summaries = []
        for mode in args.modes:
            path = Path(args.out_root) / f"bench_{mode}" / "bench_summary.json"
            summaries.append(
                json.loads(path.read_text())
                if path.is_file()
                else {"mode": mode, "status": "error", "steps_done": 0}
            )
        print(bench_table(summaries, args.gpu))
        if args.out_json:
            Path(args.out_json).write_text(json.dumps({"gpu": args.gpu, "runs": summaries}, indent=2))
        return 0 if all(s["status"] in ("ok", "oom") for s in summaries) else 1

    run_dir = Path(args.run_dir)
    rows = load_rows(run_dir)
    log_text = Path(args.log).read_text(errors="replace") if Path(args.log).is_file() else ""

    if args.cmd == "smoke":
        bad, n_files = _orig_mod_keys(run_dir, args.steps)
        problems = smoke_problems(
            rows,
            log_text,
            n_steps=args.steps,
            expect_turbo=not args.no_turbo,
            expect_compile=args.expect_compile,
            orig_mod_keys=bad,
        )
        if n_files == 0:
            problems.append(f"no safetensors checkpoint found under {run_dir}")
        if problems:
            print("[Gate 1 FAIL] " + " | ".join(problems))
            return 1
        steps = _step_rows(rows)
        mems = [r["gpu_mem_mb"] for r in steps if r.get("gpu_mem_mb") is not None]
        peak = f"{max(mems):.0f} MB ({'<=24GB OK' if max(mems) <= GB24_MB else '>24GB'})" if mems else "n/a"
        print(
            f"[Gate 1 PASS] losses={[round(float(r['step_loss']), 4) for r in steps]} peak_gpu={peak} "
            f"safetensors={n_files} (0 _orig_mod keys) compile={'applied' if args.expect_compile else 'off'}"
        )
        return 0

    summary = bench_summary(rows, log_text, mode=args.mode, steps=args.steps, warmup=args.warmup)
    out = Path(args.out_json) if args.out_json else run_dir / "bench_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    shown = {k: v for k, v in summary.items() if k != "online_turbo"}
    print(f"[bench] {json.dumps(shown)}")
    return 0 if summary["status"] in ("ok", "oom") else 1


if __name__ == "__main__":
    sys.exit(main())
