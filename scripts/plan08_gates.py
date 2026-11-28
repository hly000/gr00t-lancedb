#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pre-registered gates of docs/plans/08 (24 GB full-batch), applied mechanically.

    plan08_gates.py g2 --proof probe/proof_fullbatch.json [--out-json g2.json]
    plan08_gates.py run --eval eval/flash24_b32.json --b0 falsify_07/eval/b0.json \
        --metrics <train run>/bench_metrics.jsonl [--vs NAME=eval.json ...] [--out-json ..] [--out-md ..]

Gate G2 (Phase 1, 50-step probes under the 22.1 GiB allocator cap = GR00T_CUDA_MEM_FRACTION=0.56):
    Phase 2A  flash24_b32 (32x1) runs, peak <= 21.0 GB
    Phase 2B  mn2_b32_flash (32x2) runs, peak <= 21.8 GB; if it OOMs: DiT activation
              checkpointing (WP2 2.3, GRAD_CKPT_HEAD) is needed first.

Gate 2A (Phase 2, a 3000-step run trained under the same cap, 100 paired episodes vs B0):
    memory    trained under the cap, all steps done, peak <= 21.5 GB
    speed     median step < 649.5 ms (B0's 50-step median on the A100)
    accuracy  paired diff vs B0 >= -4 pts and exact McNemar p > 0.05
              (fixed here before the data: significantly *better* than B0, p <= 0.05 with
              diff > 0, counts as PASS and is flagged; the rule exists to catch a loss)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from libero_eval_stats import compare_evals  # noqa: E402

G2_B32_MAX_GB = 21.0
G2_B32_K2_MAX_GB = 21.8
G2A_MAX_GB = 21.5
G2A_MAX_STEP_MS = 649.5
G2A_MIN_DIFF = -0.04
ALPHA = 0.05
CAP_TAG = "_cap22"


def _run(proof: dict[str, Any], mode: str) -> dict[str, Any] | None:
    runs = {r["mode"]: r for r in proof.get("runs", [])}
    return runs.get(mode + CAP_TAG)


def gate_g2(proof: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    b32 = _run(proof, "flash24_b32")
    if b32 is None:
        out["phase_2a"] = {"result": "NOT RUN"}
    else:
        ok = b32.get("status") == "ok" and b32.get("peak_gpu_mem_gb") is not None and b32["peak_gpu_mem_gb"] <= G2_B32_MAX_GB
        out["phase_2a"] = {
            "result": "LOCK flash24_b32" if ok else "FAIL",
            "status": b32.get("status"),
            "peak_gpu_mem_gb": b32.get("peak_gpu_mem_gb"),
            "median_step_ms": b32.get("median_step_ms"),
            "limit_gb": G2_B32_MAX_GB,
        }
    k2 = _run(proof, "mn2_b32_flash")
    if k2 is None:
        out["phase_2b"] = {"result": "NOT RUN"}
    else:
        status, peak = k2.get("status"), k2.get("peak_gpu_mem_gb")
        if status == "ok" and peak is not None and peak <= G2_B32_K2_MAX_GB:
            result = "LOCK mn2_b32_flash"
        elif status == "oom":
            result = "OOM: needs DiT activation checkpointing (WP2 2.3) before Phase 2B"
        else:
            result = "FAIL"
        out["phase_2b"] = {"result": result, "status": status, "peak_gpu_mem_gb": peak,
                           "median_step_ms": k2.get("median_step_ms"), "limit_gb": G2_B32_K2_MAX_GB}
    return out


def train_stats(metrics_rows: list[dict[str, Any]], warmup: int = 10) -> dict[str, Any]:
    steps = [r for r in metrics_rows if r.get("event") == "step"]
    limit = next((r for r in metrics_rows if r.get("event") == "cuda_mem_limit"), None)
    seed = next((r for r in metrics_rows if r.get("event") == "train_seed"), None)
    mem = [r["gpu_mem_mb"] for r in steps if r.get("gpu_mem_mb") is not None]
    alloc = [r["torch_max_alloc_mb"] for r in steps if r.get("torch_max_alloc_mb") is not None]
    ms = [1e3 * float(r["step_s"]) for r in steps[warmup:] if r.get("step_s") is not None]
    return {
        "steps": len(steps),
        "last_step": steps[-1].get("global_step") if steps else None,
        "peak_gpu_mem_gb": round(max(mem) / 1024, 2) if mem else None,
        "peak_torch_alloc_gb": round(max(alloc) / 1024, 2) if alloc else None,
        "median_step_ms": round(statistics.median(ms), 2) if ms else None,
        "cuda_mem_limit_gib": limit.get("limit_gib") if limit else None,
        "train_seed": seed.get("seed") if seed else None,
    }


def gate_2a(cand: dict[str, Any], b0: dict[str, Any], stats: dict[str, Any], max_steps: int) -> dict[str, Any]:
    cmp = compare_evals(b0, cand, margin=-G2A_MIN_DIFF)
    if not cmp.get("paired"):
        raise SystemExit(f"Gate 2A needs paired evals ({cmp.get('why_unpaired')})")
    peak = stats["peak_gpu_mem_gb"]
    mem_ok = stats["cuda_mem_limit_gib"] is not None and stats["steps"] >= max_steps and peak is not None and peak <= G2A_MAX_GB
    speed_ok = stats["median_step_ms"] is not None and stats["median_step_ms"] < G2A_MAX_STEP_MS
    diff, p = cmp["diff"], cmp["mcnemar_p"]
    acc_ok = diff >= G2A_MIN_DIFF and (p > ALPHA or diff > 0)
    note = "significantly better than B0 (p <= 0.05); counted as PASS" if (acc_ok and p <= ALPHA) else None
    return {
        "memory": {"result": "PASS" if mem_ok else "FAIL", "peak_gpu_mem_gb": peak, "limit_gb": G2A_MAX_GB,
                   "trained_under_cap_gib": stats["cuda_mem_limit_gib"], "steps": stats["steps"]},
        "speed": {"result": "PASS" if speed_ok else "FAIL", "median_step_ms": stats["median_step_ms"], "limit_ms": G2A_MAX_STEP_MS},
        "accuracy": {"result": "PASS" if acc_ok else "FAIL", "diff": diff, "diff_ci95": cmp["diff_ci95"],
                     "mcnemar_p": p, "candidate": cmp["candidate"], "b0": cmp["baseline"], "note": note},
        "overall": "PASS" if (mem_ok and speed_ok and acc_ok) else "FAIL",
        "pair": cmp,
    }


def to_markdown(name: str, g: dict[str, Any], others: dict[str, dict[str, Any]]) -> str:
    def pts(x: float) -> str:
        return f"{100 * x:+.1f}"

    a, m, s = g["accuracy"], g["memory"], g["speed"]
    lines = [
        f"### Gate 2A: `{name}` vs B0 -> **{g['overall']}**",
        "",
        f"- memory: **{m['result']}** peak {m['peak_gpu_mem_gb']} GB (limit {m['limit_gb']}), trained under "
        f"{m['trained_under_cap_gib']} GiB cap, {m['steps']} steps",
        f"- speed: **{s['result']}** median {s['median_step_ms']} ms/step (limit {s['limit_ms']})",
        f"- accuracy: **{a['result']}** {100 * a['candidate']['success_rate']:.1f}% vs B0 {100 * a['b0']['success_rate']:.1f}%, "
        f"diff {pts(a['diff'])} pts, 95% CI [{pts(a['diff_ci95'][0])}, {pts(a['diff_ci95'][1])}], McNemar p={a['mcnemar_p']}"
        + (f" ({a['note']})" if a.get("note") else ""),
    ]
    if others:
        lines += ["", "| other pairing | diff (pts) | 95% CI | McNemar p |", "|---|---:|---|---:|"]
        for other, c in others.items():
            lo, hi = c["diff_ci95"]
            lines.append(f"| `{name}` vs `{other}` | {pts(c['diff'])} | [{pts(lo)}, {pts(hi)}] | {c['mcnemar_p']} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g2 = sub.add_parser("g2")
    g2.add_argument("--proof", required=True)
    g2.add_argument("--out-json", default=None)
    run = sub.add_parser("run")
    run.add_argument("--eval", required=True)
    run.add_argument("--b0", required=True)
    run.add_argument("--metrics", required=True)
    run.add_argument("--name", default=None)
    run.add_argument("--max-steps", type=int, default=3000)
    run.add_argument("--vs", action="append", default=[], help="NAME=eval.json: extra paired comparison")
    run.add_argument("--out-json", default=None)
    run.add_argument("--out-md", default=None)
    args = ap.parse_args(argv)

    if args.cmd == "g2":
        out = gate_g2(json.loads(Path(args.proof).read_text()))
        print(json.dumps(out, indent=2))
    else:
        cand = json.loads(Path(args.eval).read_text())
        rows = [json.loads(ln) for ln in Path(args.metrics).read_text().splitlines() if ln.strip()]
        stats = train_stats(rows)
        out = gate_2a(cand, json.loads(Path(args.b0).read_text()), stats, args.max_steps)
        out["train"] = stats
        others = {}
        for spec in args.vs:
            other, _, path = spec.partition("=")
            others[other] = compare_evals(json.loads(Path(path).read_text()), cand)
        out["others"] = others
        name = args.name or Path(args.eval).stem
        md = to_markdown(name, out, others)
        print(md)
        if args.out_md:
            Path(args.out_md).write_text(md + "\n", encoding="utf-8")
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
