#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Optimizer micro-benchmark: step time and memory at action-head scale, no model needed.

Compares, on tensors with the real GR00T N1.7 action-head shapes (or a synthetic stand-in):

    adamw_fp32      torch.optim.AdamW(fused) on fp32 weights      16 B/param
    adamw_bf16      torch.optim.AdamW(fused) on bf16 weights       8 B/param (drops small updates)
    bnb8bit_bf16    bitsandbytes.AdamW8bit on bf16 weights
    bnb8bit_fp32    bitsandbytes.AdamW8bit on fp32 weights (what optim=adamw_8bit gives)
    torchao_8bit_bf16sr  torchao AdamW8bit on bf16 weights, bf16 stochastic rounding
    torchao_8bit_fp32    torchao AdamW8bit on fp32 weights
    lowmem_bf16     LowMemAdamW (Triton, one fused launch), Kahan bf16 weights    8.06 B/param
    lowmem_bf16_pt  same, one launch PER TENSOR (the v0 behaviour; fused=False)
    lowmem_fp32     LowMemAdamW (Triton, one fused launch), fp32 weights         10.06 B/param
    lowmem_bf16_1t  same total size as ONE tensor: the floor without any launch overhead

Every row except ``*_1t`` steps the same tensor list (980 tensors with the real action-head
shapes); ``*_1t`` is a bandwidth floor and is printed apart from them, never as a rival.
A row whose library is not installed (bitsandbytes, torchao) is reported as skipped.

Each LowMemAdamW row reports ``fused`` (active / off / fallback: reason) and the launches
issued per step. ``--require-fused`` exits non-zero when the fused launch is not active or
``lowmem_bf16`` is slower than ``--max-fused-ms`` (the Phase-2 gate: 25 ms on an A100).

Takes ~1 minute of GPU. Run the kernel parity test first:

    pytest tests/test_turbo_lowmem_adamw.py -k triton -q
    python scripts/bench_lowmem_adamw.py --base-model-path nvidia/GR00T-N1.7-3B --out-json bench.json
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch

_REPO_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)


def synthetic_shapes(total_params: int) -> list[tuple[int, ...]]:
    """A DiT-like mix: per layer 6 square matrices, an 4x MLP pair, and a dozen bias/norm vectors."""
    d = 1536
    layer = [(d, d)] * 6 + [(4 * d, d), (d, 4 * d)] + [(d,)] * 10 + [(4 * d,)] * 2
    per_layer = sum(int(torch.tensor(s).prod()) for s in layer)
    return layer * max(1, total_params // per_layer)


def action_head_shapes(base_model_path: str, gr00t_root: str | None) -> list[tuple[int, ...]]:
    if gr00t_root and gr00t_root not in sys.path:
        sys.path.insert(0, gr00t_root)
    from gr00t_lancedb.vl_cache import load_action_head_only

    ah, _, _ = load_action_head_only(base_model_path, device="cpu", dtype=torch.float32)
    return [tuple(p.shape) for p in ah.parameters() if p.requires_grad]


OPTIONAL_LIBS = ("bitsandbytes", "torchao")


def _torchao_adamw8bit() -> Any:
    try:
        from torchao.optim import AdamW8bit  # torchao >= 0.11
    except ImportError:
        from torchao.prototype.low_bit_optim import AdamW8bit
    return AdamW8bit


def _make_optimizer(kind: str, params: list[torch.nn.Parameter]) -> Any:
    kw = dict(lr=1e-4, betas=(0.95, 0.999), weight_decay=1e-5)
    if kind.startswith("adamw"):
        return torch.optim.AdamW(params, fused=True, **kw)
    if kind.startswith("bnb8bit"):
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(params, **kw)
    if kind.startswith("torchao_8bit"):
        extra = {"bf16_stochastic_round": True} if kind.endswith("bf16sr") else {}
        return _torchao_adamw8bit()(params, **kw, **extra)
    from gr00t_lancedb.turbo import LowMemAdamW

    return LowMemAdamW(params, backend="triton", fused=False if kind.endswith("_pt") else None, **kw)


def bench(kind: str, shapes: list[tuple[int, ...]], steps: int, warmup: int) -> dict[str, Any]:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    dtype = torch.float32 if "fp32" in kind else torch.bfloat16
    if kind.endswith("_1t"):
        shapes = [(sum(int(torch.tensor(s).prod()) for s in shapes),)]
    params = [torch.nn.Parameter(torch.randn(s, device="cuda", dtype=dtype) * 0.05) for s in shapes]
    n = sum(p.numel() for p in params)
    opt = _make_optimizer(kind, params)
    times: list[float] = []
    for step in range(warmup + steps):
        for p in params:  # fresh gradient tensors every step, like autograd
            p.grad = torch.randn_like(p) * 0.01
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        e0.record()
        opt.step()
        e1.record()
        torch.cuda.synchronize()
        if step >= warmup:
            times.append((e0.elapsed_time(e1), (time.perf_counter() - t0) * 1e3))
        opt.zero_grad(set_to_none=True)
    times.sort()
    med = times[len(times) // 2]
    held = torch.cuda.memory_allocated() - base  # weights + optimizer state, gradients freed
    rec = {
        "tensors": len(params),
        "params": n,
        "step_ms": round(med[0], 2),
        "step_wall_ms": round(med[1], 2),
        "weights_plus_state_gb": round(held / 1024**3, 3),
        "weights_plus_state_bytes_per_param": round(held / n, 3),
        "with_grad_bytes_per_param": round(held / n + params[0].element_size(), 3),
        "peak_alloc_gb": round((torch.cuda.max_memory_allocated() - base) / 1024**3, 3),
    }
    if hasattr(opt, "fused_status"):
        rec["fused"] = opt.fused_status
        rec["launches_per_step"] = opt.last_step_stats["fused_launches"] + opt.last_step_stats["per_tensor_launches"]
    del opt, params
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model-path", default=None, help="use the real action-head shapes (CPU load, ~15 s)")
    ap.add_argument("--gr00t-root", default="/data/Isaac-GR00T")
    ap.add_argument("--total-params", type=float, default=1.62e9, help="synthetic size when no model is given")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument(
        "--kinds",
        default=(
            "adamw_bf16,lowmem_bf16,lowmem_bf16_pt,lowmem_bf16_1t,lowmem_fp32,bnb8bit_bf16,bnb8bit_fp32,"
            "torchao_8bit_bf16sr,torchao_8bit_fp32,adamw_fp32"
        ),
        help="comma list; adamw_fp32 last because it needs 26 GB at action-head scale",
    )
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--require-fused", action="store_true", help="gate: fused launch active and fast enough")
    ap.add_argument("--max-fused-ms", type=float, default=25.0)
    ap.add_argument("--max-fused-peak-gb", type=float, default=12.5)
    args = ap.parse_args()

    if args.base_model_path:
        shapes = action_head_shapes(args.base_model_path, args.gr00t_root)
    else:
        shapes = synthetic_shapes(int(args.total_params))
    report: dict[str, Any] = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "shapes_from": args.base_model_path or "synthetic",
        "results": {},
    }
    try:
        import triton

        report["triton"] = triton.__version__
    except Exception as exc:  # noqa: BLE001
        report["triton"] = f"unavailable: {exc}"
    for kind in args.kinds.split(","):
        try:
            report["results"][kind] = bench(kind, shapes, args.steps, args.warmup)
        except ModuleNotFoundError as exc:
            if (exc.name or "").split(".")[0] in OPTIONAL_LIBS:
                report["results"][kind] = {"skipped": f"{exc.name} not installed"}
            else:
                report["results"][kind] = {"error": f"{type(exc).__name__}: {exc}"}
        except Exception as exc:  # noqa: BLE001 - one failing config must not lose the others
            report["results"][kind] = {"error": f"{type(exc).__name__}: {exc}"}
            gc.collect()
            torch.cuda.empty_cache()
        print(f"[bench] {kind}: {report['results'][kind]}", flush=True)
    print(results_table(report["results"]), flush=True)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.require_fused:
        problems = fused_gate(report["results"], args.max_fused_ms, args.max_fused_peak_gb)
        if problems:
            raise SystemExit("[bench] FUSED GATE FAILED: " + "; ".join(problems))
        print(f"[bench] fused gate passed (<= {args.max_fused_ms} ms, <= {args.max_fused_peak_gb} GB)", flush=True)


def results_table(results: dict[str, Any]) -> str:
    """Markdown: rows that step the same tensor list first, the single-tensor floor apart."""

    def row(kind: str, rec: dict[str, Any]) -> str:
        if "skipped" in rec or "error" in rec:
            return f"| `{kind}` | {rec.get('skipped') or rec.get('error')} | | | | |"
        return (
            f"| `{kind}` | {rec['tensors']} | {rec['step_ms']} | {rec['weights_plus_state_bytes_per_param']} "
            f"| {rec['peak_alloc_gb']} | {rec.get('fused', '-')} |"
        )

    head = [
        "| optimizer | tensors | step ms (median) | weights+state B/param | peak alloc GB | fused |",
        "|---|---:|---:|---:|---:|---|",
    ]
    same = [row(k, r) for k, r in results.items() if not k.endswith("_1t")]
    floor = [row(k, r) for k, r in results.items() if k.endswith("_1t")]
    out = head + same
    if floor:
        out += ["", "Single-tensor floor (not comparable with the rows above):", "", *head, *floor]
    return "\n".join(out)


def fused_gate(results: dict[str, Any], max_ms: float, max_peak_gb: float) -> list[str]:
    """Why ``lowmem_bf16`` does not meet the fused-launch gate (empty list: it does)."""
    rec = results.get("lowmem_bf16")
    if not rec or "error" in rec:
        return [f"lowmem_bf16 did not run: {rec}"]
    problems = []
    if rec.get("fused") != "active":
        problems.append(f"fused launch not active ({rec.get('fused')})")
    if rec["step_ms"] > max_ms:
        problems.append(f"step {rec['step_ms']} ms > {max_ms} ms")
    if rec["peak_alloc_gb"] > max_peak_gb:
        problems.append(f"peak {rec['peak_alloc_gb']} GB > {max_peak_gb} GB")
    return problems


if __name__ == "__main__":
    main()
