#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Optimizer micro-benchmark: step time and memory at action-head scale, no model needed.

Compares, on tensors with the real GR00T N1.7 action-head shapes (``--base-model-path``;
537 trainable tensors, 1.62 B parameters) or a synthetic stand-in:

    adamw_fp32           torch.optim.AdamW(fused) on fp32 weights
    adamw_bf16           torch.optim.AdamW(fused) on bf16 weights (drops sub-ulp updates)
    lowmem_bf16          LowMemAdamW (Triton, one launch per tensor), Kahan bf16 weights
    lowmem_fp32          LowMemAdamW (Triton, one launch per tensor), fp32 weights
    lowmem_bf16_1t       lowmem_bf16 on ONE tensor of the same total size: launch-free floor
    flash_adamw_24       flashoptim FlashAdamW, bf16 weights, master_weight_bits=24
    flash_adamw_32       flashoptim FlashAdamW, bf16 weights, master_weight_bits=32
    bnb8bit_bf16 / _fp32 bitsandbytes AdamW8bit on bf16 / fp32 weights
    torchao_8bit_bf16sr  torchao AdamW8bit, bf16 weights + bf16 stochastic rounding
    torchao_8bit_fp32    torchao AdamW8bit on fp32 weights

Every row except ``*_1t`` steps the same tensor list; ``*_1t`` is printed apart from them.
A row whose library is not installed (flashoptim, bitsandbytes, torchao) is reported as
skipped. ``effective_bw_gb_s`` divides an estimate of the minimum bytes each optimizer must
read and write per parameter (``TRAFFIC_BYTES_PER_PARAM``) by the step time: a comparison
against the card's HBM bandwidth, not a measurement of traffic.

``--gate-g1`` prints the docs/plans/08 gate G1 (which optimizer route WP3 takes).

Takes a few minutes of GPU (torchao compiles per shape). Run the kernel parity test first:

    pytest tests/test_turbo_lowmem_adamw.py -k triton -q
    python scripts/bench_lowmem_adamw.py --base-model-path nvidia/GR00T-N1.7-3B --out-json bench.json --gate-g1
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

OPTIONAL_LIBS = ("bitsandbytes", "torchao", "flashoptim")

# Minimum HBM bytes per parameter per step (reads + writes; quantisation scales ignored):
# weights (+ compensation) read and written, gradient read, both moments read and written.
TRAFFIC_BYTES_PER_PARAM = {
    "adamw_fp32": 4 + 4 + 2 * 4 + (4 + 2 * 4),  # w, g, m, v read; w, m, v written
    "adamw_bf16": 2 + 2 + 2 * 2 + (2 + 2 * 2),
    "lowmem_bf16": 2 + 2 + 2 + 2 * 1 + (2 + 2 + 2 * 1),  # w, kahan c, g, 8-bit m/v
    "lowmem_fp32": 4 + 4 + 2 * 1 + (4 + 2 * 1),
    "flash_adamw_24": 2 + 1 + 2 + 2 * 1 + (2 + 1 + 2 * 1),  # bf16 w, 8-bit correction, 8-bit m/v
    "flash_adamw_32": 2 + 2 + 2 + 2 * 1 + (2 + 2 + 2 * 1),  # bf16 w, 16-bit correction
    "bnb8bit_bf16": 2 + 2 + 2 * 1 + (2 + 2 * 1),
    "bnb8bit_fp32": 4 + 4 + 2 * 1 + (4 + 2 * 1),
    "torchao_8bit_bf16sr": 2 + 2 + 2 * 1 + (2 + 2 * 1),
    "torchao_8bit_fp32": 4 + 4 + 2 * 1 + (4 + 2 * 1),
}

# docs/plans/08 gate G1 (fixed before the measurement)
G1_FLASH_MAX_MS = 22.0
G1_FLASH_MAX_BYTES_WITH_GRAD = 7.2
G1_TIE = 1.05  # LowMemAdamW within 5 % of FlashAdamW-24 counts as "ties"


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


def _torchao_adamw8bit() -> Any:
    try:
        from torchao.optim import AdamW8bit  # torchao >= 0.11
    except ImportError:
        from torchao.prototype.low_bit_optim import AdamW8bit
    return AdamW8bit


def make_flash_adamw(params: Any, *, master_weight_bits: int, **kw: Any) -> Any:
    """flashoptim.FlashAdamW; 0.1.0 named the argument ``master_bytewidth`` (3 / 4 bytes)."""
    from flashoptim import FlashAdamW

    try:
        return FlashAdamW(params, master_weight_bits=master_weight_bits, **kw)
    except TypeError as exc:
        if "master_weight_bits" not in str(exc):
            raise
        return FlashAdamW(params, master_bytewidth=master_weight_bits // 8, **kw)


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
    if kind.startswith("flash_adamw_"):
        return make_flash_adamw(params, master_weight_bits=int(kind.rsplit("_", 1)[1]), **kw)
    from gr00t_lancedb.turbo import LowMemAdamW

    return LowMemAdamW(params, backend="triton", **kw)


def _param_dtype(kind: str) -> torch.dtype:
    return torch.float32 if "fp32" in kind else torch.bfloat16


def bench(kind: str, shapes: list[tuple[int, ...]], steps: int, warmup: int) -> dict[str, Any]:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    dtype = _param_dtype(kind)
    if kind.endswith("_1t"):
        shapes = [(sum(int(torch.tensor(s).prod()) for s in shapes),)]
    params = [torch.nn.Parameter(torch.randn(s, device="cuda", dtype=dtype) * 0.05) for s in shapes]
    n = sum(p.numel() for p in params)
    opt = _make_optimizer(kind, params)
    times: list[tuple[float, float]] = []
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
    rec: dict[str, Any] = {
        "tensors": len(params),
        "params": n,
        "step_ms": round(med[0], 2),
        "step_wall_ms": round(med[1], 2),
        "weights_plus_state_gb": round(held / 1024**3, 3),
        "weights_plus_state_bytes_per_param": round(held / n, 3),
        "with_grad_bytes_per_param": round(held / n + params[0].element_size(), 3),
        "peak_alloc_gb": round((torch.cuda.max_memory_allocated() - base) / 1024**3, 3),
    }
    traffic = TRAFFIC_BYTES_PER_PARAM.get(kind.removesuffix("_1t"))
    if traffic is not None and med[0] > 0:
        rec["est_traffic_bytes_per_param"] = traffic
        rec["effective_bw_gb_s"] = round(traffic * n / (med[0] / 1e3) / 1e9, 1)
    del opt, params
    return rec


def gate_g1(results: dict[str, Any]) -> dict[str, Any]:
    """docs/plans/08 gate G1: A = stop building optimizer kernels, B = LowMemAdamW v2."""
    flash, lowmem = results.get("flash_adamw_24", {}), results.get("lowmem_bf16", {})
    if "step_ms" not in flash or "step_ms" not in lowmem:
        return {"branch": "UNDECIDED", "why": f"missing rows: flash_adamw_24={flash} lowmem_bf16={lowmem}"}
    f_ms, f_b, l_ms = flash["step_ms"], flash["with_grad_bytes_per_param"], lowmem["step_ms"]
    out = {"flash24_ms": f_ms, "flash24_bytes_with_grad": f_b, "lowmem_ms": l_ms,
           "lowmem_bytes_with_grad": lowmem["with_grad_bytes_per_param"]}
    if f_ms <= G1_FLASH_MAX_MS and f_b <= G1_FLASH_MAX_BYTES_WITH_GRAD:
        out.update(branch="A", why=f"FlashAdamW-24 {f_ms} ms <= {G1_FLASH_MAX_MS} and {f_b} B/param <= {G1_FLASH_MAX_BYTES_WITH_GRAD}")
    elif l_ms <= G1_TIE * f_ms:
        out.update(branch="B", why=f"LowMemAdamW {l_ms} ms ties or beats FlashAdamW-24 {f_ms} ms")
    else:
        out.update(branch="UNDECIDED", why=f"FlashAdamW-24 is faster ({f_ms} vs {l_ms} ms) but misses the A thresholds")
    return out


def results_table(results: dict[str, Any]) -> str:
    """Markdown: rows that step the same tensor list first, the single-tensor floor apart."""

    def row(kind: str, rec: dict[str, Any]) -> str:
        if "skipped" in rec or "error" in rec:
            return f"| `{kind}` | {rec.get('skipped') or rec.get('error')} | | | | | |"
        return (
            f"| `{kind}` | {rec['tensors']} | {rec['step_ms']} | {rec['weights_plus_state_bytes_per_param']} "
            f"| {rec['with_grad_bytes_per_param']} | {rec['peak_alloc_gb']} | {rec.get('effective_bw_gb_s', '-')} |"
        )

    head = [
        "| optimizer | tensors | step ms (median) | weights+state B/param | with grad B/param | peak alloc GB "
        "| est. effective GB/s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    same = [row(k, r) for k, r in results.items() if not k.endswith("_1t")]
    floor = [row(k, r) for k, r in results.items() if k.endswith("_1t")]
    out = head + same
    if floor:
        out += ["", "Single-tensor floor (not comparable with the rows above):", "", *head, *floor]
    return "\n".join(out)


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
            "adamw_bf16,lowmem_bf16,lowmem_bf16_1t,lowmem_fp32,flash_adamw_24,flash_adamw_32,"
            "bnb8bit_bf16,bnb8bit_fp32,torchao_8bit_fp32,adamw_fp32"
        ),
        help="comma list; adamw_fp32 last because it needs 24 GB at action-head scale",
    )
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--gate-g1", action="store_true", help="print docs/plans/08 gate G1")
    args = ap.parse_args()

    if args.base_model_path:
        shapes = action_head_shapes(args.base_model_path, args.gr00t_root)
    else:
        shapes = synthetic_shapes(int(args.total_params))
    n_params = sum(int(torch.tensor(s).prod()) for s in shapes)
    print(f"[bench] {len(shapes)} tensors, {n_params / 1e9:.3f} B parameters ({args.base_model_path or 'synthetic'})", flush=True)
    report: dict[str, Any] = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "shapes_from": args.base_model_path or "synthetic",
        "tensors": len(shapes),
        "params": n_params,
        "results": {},
    }
    for lib in ("triton", *OPTIONAL_LIBS):
        try:
            report[lib] = __import__(lib).__version__
        except Exception as exc:  # noqa: BLE001
            report[lib] = f"unavailable: {type(exc).__name__}"
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
    if args.gate_g1:
        report["gate_g1"] = gate_g1(report["results"])
        print(f"[bench] gate G1: branch {report['gate_g1']['branch']} ({report['gate_g1']['why']})", flush=True)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
