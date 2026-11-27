#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Optimizer micro-benchmark: step time and memory at action-head scale, no model needed.

Compares, on tensors with the real GR00T N1.7 action-head shapes (or a synthetic stand-in):

    adamw_fp32      torch.optim.AdamW(fused) on fp32 weights      16 B/param
    adamw_bf16      torch.optim.AdamW(fused) on bf16 weights       8 B/param (drops small updates)
    bnb8bit_bf16    bitsandbytes.AdamW8bit on bf16 weights
    lowmem_bf16     LowMemAdamW (Triton), Kahan bf16 weights    8.06 B/param
    lowmem_fp32     LowMemAdamW (Triton), fp32 weights         10.06 B/param
    lowmem_bf16_1t  same total size as ONE tensor: isolates per-tensor launch overhead

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


def _make_optimizer(kind: str, params: list[torch.nn.Parameter]) -> Any:
    kw = dict(lr=1e-4, betas=(0.95, 0.999), weight_decay=1e-5)
    if kind.startswith("adamw"):
        return torch.optim.AdamW(params, fused=True, **kw)
    if kind.startswith("bnb8bit"):
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(params, **kw)
    from gr00t_lancedb.turbo import LowMemAdamW

    return LowMemAdamW(params, backend="triton", **kw)


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
        default="adamw_bf16,lowmem_bf16,lowmem_bf16_1t,lowmem_fp32,bnb8bit_bf16,adamw_fp32",
        help="comma list; adamw_fp32 last because it needs 26 GB at action-head scale",
    )
    ap.add_argument("--out-json", default=None)
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
        except Exception as exc:  # noqa: BLE001 - one failing config must not lose the others
            report["results"][kind] = {"error": f"{type(exc).__name__}: {exc}"}
            gc.collect()
            torch.cuda.empty_cache()
        print(f"[bench] {kind}: {report['results'][kind]}", flush=True)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
