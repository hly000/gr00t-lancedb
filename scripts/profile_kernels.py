#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Where do the milliseconds and gigabytes of one GR00T N1.7 fine-tuning step go?

One run (~5 minutes of GPU, mostly model load) writes ``<out-dir>/profile_kernels.json`` with:

1. ``phases``        wall time of prepare / backbone / head forward / backward / optimizer.
2. ``memory``        analytic ledger (weights, gradients, optimizer state per module) next to
                     the measured peak after each phase.
3. ``ops_full``      torch.profiler top ops by self CUDA time for the whole step.
4. ``ops_head``      the same for the action head alone (backbone output precomputed), plus
                     ``gpu_busy_frac``: kernel time / wall time. Low means the head is bound by
                     kernel launches and Python, not by arithmetic - the case fused kernels,
                     ``torch.compile`` or CUDA graphs can help. High means it is GEMM-bound and
                     fusing elementwise ops will not move the needle.
5. ``features``      statistics of real backbone features that decide how they may be
                     quantised: per-token crest factor, per-channel outliers, int8 error, and
                     the action-head loss on exact vs int8 features in fp32 (the earlier parity
                     check compared two bf16 losses, which cannot resolve differences below 0.4 %).

``--precision native`` reproduces what HF Trainer does (fp32 weights, bf16 autocast, fused
AdamW; ~37 GB at batch 32). Use ``--precision bf16`` or a smaller batch on a 24 GB card.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CATEGORIES: list[tuple[str, tuple[str, ...]]] = [
    ("optimizer", ("_fused_adamw_", "addcdiv_", "addcmul_", "lerp_", "optimizer.step")),
    ("attention", ("scaled_dot_product_attention", "_scaled_dot_product", "softmax", "_softmax", "baddbmm")),
    ("gemm", ("linear", "addmm", "matmul", "bmm", "mm", "einsum", "conv1d", "conv2d", "convolution")),
    ("norm", ("layer_norm", "native_layer_norm", "group_norm", "rms_norm", "batch_norm")),
    ("copy_cast", ("copy_", "to", "_to_copy", "clone", "contiguous", "cat", "stack", "repeat", "repeat_interleave",
                   "index", "index_select", "gather", "embedding", "masked_fill", "masked_fill_")),
    ("elementwise", ("mul", "mul_", "add", "add_", "sub", "div", "gelu", "silu", "sigmoid", "tanh", "where", "neg",
                     "pow", "sqrt", "rsqrt", "exp", "sin", "cos", "relu", "dropout", "native_dropout")),
    ("reduction", ("sum", "mean", "mse_loss", "norm", "linalg_vector_norm", "max", "min", "amax", "any", "all")),
]
_BACKWARD_ALIASES = {"mm": "gemm", "addmm": "gemm", "bmm": "gemm", "nativelayernorm": "norm", "gelu": "elementwise",
                     "silu": "elementwise", "mul": "elementwise", "add": "elementwise", "softmax": "attention",
                     "scaleddotproductattention": "attention", "scaleddotproductflashattention": "attention"}


def categorize(op_name: str) -> str:
    """Bucket a profiler row. ``aten::linear`` -> gemm, ``AddmmBackward0`` -> gemm, unknown -> other."""
    name = op_name.strip()
    low = name.lower()
    if low.startswith("aten::"):
        base = low[len("aten::"):]
        for cat, names in CATEGORIES:
            if base in names:
                return cat
        return "other"
    if "optimizer" in low or "adamw" in low:
        return "optimizer"
    if "backward" in low:
        base = low.split("backward")[0].replace("autograd::engine::evaluate_function: ", "").replace("_", "")
        return _BACKWARD_ALIASES.get(base, "other")
    return "other"


def _self_time_us(evt: Any, on_gpu: bool) -> float:
    """Self time in microseconds; the GPU attribute was renamed in torch 2.4."""
    names = ("self_device_time_total", "self_cuda_time_total") if on_gpu else ("self_cpu_time_total",)
    for n in names:
        if hasattr(evt, n):
            return float(getattr(evt, n))
    return 0.0


def run_profile(step_fn: Any, steps: int, trace_path: Path | None, on_gpu: bool | None = None) -> dict[str, Any]:
    """Profile ``steps`` calls of ``step_fn``. ``on_gpu=False`` ranks by CPU self time (tests)."""
    from torch.profiler import ProfilerActivity, profile

    if on_gpu is None:
        on_gpu = torch.cuda.is_available()
    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if on_gpu else [])
    sync = torch.cuda.synchronize if on_gpu else (lambda: None)
    sync()
    t0 = time.perf_counter()
    with profile(activities=activities) as prof:
        for _ in range(steps):
            step_fn()
        sync()
    wall_ms = (time.perf_counter() - t0) * 1e3

    ops = [(e.key, _self_time_us(e, on_gpu) / 1e3, int(e.count)) for e in prof.key_averages()]
    ops = [o for o in ops if o[1] > 0]
    ops.sort(key=lambda o: -o[1])
    total = sum(o[1] for o in ops) or 1.0
    by_cat: dict[str, float] = defaultdict(float)
    for name, ms, _ in ops:
        by_cat[categorize(name)] += ms
    out: dict[str, Any] = {
        "profiled_steps": steps,
        "wall_ms_per_step_with_profiler": round(wall_ms / steps, 2),
        "gpu_ms_per_step": round(total / steps, 2),
        "top_ops": [
            {"op": n, "ms_per_step": round(ms / steps, 3), "pct": round(100 * ms / total, 1), "calls_per_step": c // steps}
            for n, ms, c in ops[:25]
        ],
        "by_category_pct": {k: round(100 * v / total, 1) for k, v in sorted(by_cat.items(), key=lambda kv: -kv[1])},
    }
    if not on_gpu:
        return out
    try:  # raw CUDA kernels: names tell GEMM from elementwise; durations give the busy fraction
        kern: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
        for e in prof.events():
            if "cuda" in str(getattr(e, "device_type", "")).lower():
                dur = float(getattr(e, "device_time", getattr(e, "cuda_time", 0.0)))
                kern[e.name][0] += dur / 1e3
                kern[e.name][1] += 1
        ktotal = sum(v[0] for v in kern.values())
        top = sorted(kern.items(), key=lambda kv: -kv[1][0])[:20]
        out["top_kernels"] = [
            {"kernel": k[:160], "ms_per_step": round(v[0] / steps, 3), "launches_per_step": v[1] // steps} for k, v in top
        ]
        out["kernel_launches_per_step"] = sum(v[1] for v in kern.values()) // steps
        out["gpu_busy_frac"] = round(ktotal / wall_ms, 3)
    except Exception as exc:  # noqa: BLE001
        out["top_kernels_error"] = f"{type(exc).__name__}: {exc}"
    if trace_path is not None:
        try:
            prof.export_chrome_trace(str(trace_path))
            out["chrome_trace"] = str(trace_path)
        except Exception as exc:  # noqa: BLE001
            out["chrome_trace_error"] = f"{type(exc).__name__}: {exc}"
    return out


def _gb(n_bytes: float) -> float:
    return round(n_bytes / 1024**3, 3)


def memory_ledger(model: Any, opt: Any) -> dict[str, Any]:
    def bytes_of(params: Any) -> tuple[int, int]:
        ps = list(params)
        return sum(p.numel() * p.element_size() for p in ps), sum(
            p.numel() * p.element_size() for p in ps if p.requires_grad
        )

    bb_all, bb_train = bytes_of(model.backbone.parameters())
    ah_all, ah_train = bytes_of(model.action_head.parameters())
    opt_state = sum(
        v.numel() * v.element_size() for st in opt.state.values() for v in st.values() if isinstance(v, torch.Tensor)
    )
    return {
        "backbone_weights_gb": _gb(bb_all),
        "action_head_weights_gb": _gb(ah_all),
        "gradients_gb": _gb(bb_train + ah_train),
        "optimizer_state_gb": _gb(opt_state),
        "static_total_gb": _gb(bb_all + ah_all + bb_train + ah_train + opt_state),
    }


def int8_per_token(x: torch.Tensor) -> torch.Tensor:
    """Same arithmetic as ``vl_cache.codec.encode_features`` + dequantise, on GPU in fp32."""
    scale = (x.abs().amax(dim=-1, keepdim=True) / 127.0).clamp_min(1e-6).to(torch.float16).to(torch.float32)
    return torch.clamp(torch.round(x / scale), -127, 127) * scale


def feature_stats(model: Any, bb_outs: list[Any], act_ins: list[Any], autocast: Any) -> dict[str, Any]:
    from transformers import BatchFeature

    feats, masks = [], []
    for bb in bb_outs:
        f = bb["backbone_features"].float()
        m = bb["backbone_attention_mask"].bool()
        feats.append(f[m])
        masks.append(m)
    x = torch.cat(feats)  # [tokens, D]
    q = int8_per_token(x)
    rms_tok = x.pow(2).mean(-1).sqrt()
    crest = x.abs().amax(-1) / rms_tok.clamp_min(1e-12)
    rms_ch = x.pow(2).mean(0).sqrt()
    err_ch = (q - x).pow(2).mean(0).sqrt() / rms_ch.clamp_min(1e-12)
    cos = torch.nn.functional.cosine_similarity(x, q, dim=-1)
    qs = torch.tensor([0.5, 0.9, 0.99, 1.0], device=x.device)
    out: dict[str, Any] = {
        "tokens": int(x.shape[0]),
        "hidden_dim": int(x.shape[1]),
        "crest_factor_p50_p90_p99_max": [round(float(v), 2) for v in torch.quantile(crest, qs)],
        "channel_rms_max_over_median": round(float(rms_ch.max() / rms_ch.median()), 2),
        "top_outlier_channels": [int(i) for i in torch.topk(rms_ch, 8).indices],
        "int8_rel_l2_error": round(float((q - x).norm() / x.norm()), 5),
        "int8_channel_rel_error_p50_p90_p99_max": [round(float(v), 4) for v in torch.quantile(err_ch, qs)],
        "int8_frac_nonzero_values_rounded_to_zero": round(float(((q == 0) & (x != 0)).float().mean()), 4),
        "int8_token_cosine_min_p01_p50": [
            round(float(v), 6) for v in (cos.min(), torch.quantile(cos, 0.01), torch.quantile(cos, 0.5))
        ],
        "note": "crest factor = per-token |max| / rms; a per-token int8 code has 127/crest levels per rms",
    }

    # Functional check in fp32: same (t, noise, state dropout) for exact and int8 features.
    exact, quant, per_sample = [], [], []
    ah = model.action_head
    with torch.no_grad():
        for seed, (bb, act) in enumerate(zip(bb_outs, act_ins, strict=True)):
            f = bb["backbone_features"]
            m = bb["backbone_attention_mask"].bool().unsqueeze(-1)
            fq = torch.where(m, int8_per_token(f.float()), f.float()).to(f.dtype)
            losses = []
            for variant in (f, fq):
                torch.manual_seed(1000 + seed)
                torch.cuda.manual_seed_all(1000 + seed)
                with autocast():
                    o = ah(BatchFeature(data={**dict(bb), "backbone_features": variant}), act)
                num = (o["action_loss"].float()).flatten(1).sum(1)
                den = o["action_mask"].float().flatten(1).sum(1).clamp_min(1)
                losses.append(num / den)
            exact.append(float(losses[0].mean()))
            quant.append(float(losses[1].mean()))
            per_sample.append(((losses[1] - losses[0]).abs() / losses[0].clamp_min(1e-6)))
    rel = torch.cat(per_sample)
    out["head_loss_exact_features"] = [round(v, 6) for v in exact]
    out["head_loss_int8_features"] = [round(v, 6) for v in quant]
    out["head_loss_per_sample_rel_diff_mean_max"] = [round(float(rel.mean()), 5), round(float(rel.max()), 5)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gr00t-root", default="/data/Isaac-GR00T")
    ap.add_argument("--base-model-path", default="nvidia/GR00T-N1.7-3B")
    ap.add_argument("--dataset-path", default="/data/datasets/libero_10_lance")
    ap.add_argument("--embodiment-tag", default="LIBERO_PANDA")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-batches", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--steps", type=int, default=10, help="timed steps without the profiler")
    ap.add_argument("--profile-steps", type=int, default=3)
    ap.add_argument("--precision", choices=["native", "bf16"], default="native")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    if args.gr00t_root not in sys.path:
        sys.path.insert(0, args.gr00t_root)
    from bench_vl_cache_profile import _load_real_batches
    from transformers import AutoModel

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "precision": args.precision,
        "batch_size": args.batch_size,
    }

    def save() -> None:  # after every section: a late failure keeps the earlier results
        (out_dir / "profile_kernels.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    batches, _cfg = _load_real_batches(
        args.dataset_path, args.base_model_path, args.embodiment_tag, args.batch_size, args.num_batches
    )
    weight_dtype = torch.float32 if args.precision == "native" else torch.bfloat16
    model = AutoModel.from_pretrained(args.base_model_path, trust_remote_code=True, torch_dtype=weight_dtype)
    model.backbone.set_trainable_parameters(tune_llm=False, tune_visual=False, tune_top_llm_layers=0)
    model.action_head.set_trainable_parameters(tune_projector=True, tune_diffusion_model=True, tune_vlln=True)
    model = model.to(device="cuda", dtype=weight_dtype).train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4, betas=(0.95, 0.999), weight_decay=1e-5, fused=True)

    def autocast() -> Any:
        return torch.autocast("cuda", dtype=torch.bfloat16)

    state = {"i": 0}
    bb_cache: list[Any] = []
    act_cache: list[Any] = []

    def full_step(events: list[Any] | None = None, peaks: dict[str, int] | None = None) -> None:
        raw = batches[state["i"] % len(batches)]
        state["i"] += 1

        def mark(name: str) -> None:
            if events is not None:
                e = torch.cuda.Event(enable_timing=True)
                e.record()
                events.append(e)
            if peaks is not None:
                torch.cuda.synchronize()
                peaks[name] = max(peaks.get(name, 0), torch.cuda.max_memory_allocated())
                torch.cuda.reset_peak_memory_stats()

        opt.zero_grad(set_to_none=True)
        mark("start")
        with autocast():
            bb_in, act_in = model.prepare_input(dict(raw))
            mark("prepare")
            with torch.no_grad():
                bb_out = model.backbone(bb_in)
            mark("backbone_fwd")
            if len(bb_cache) < len(batches):
                bb_cache.append(bb_out)
                act_cache.append(act_in)
            loss = model.action_head(bb_out, act_in)["loss"]
        mark("head_fwd")
        loss.backward()
        mark("head_bwd")
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        mark("clip_and_optimizer")

    def head_step() -> None:
        k = state["i"] % len(bb_cache)
        state["i"] += 1
        opt.zero_grad(set_to_none=True)
        with autocast():
            loss = model.action_head(bb_cache[k], act_cache[k])["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

    for _ in range(args.warmup):
        full_step()
    torch.cuda.synchronize()

    # 1. phase timings
    names = ["prepare", "backbone_fwd", "head_fwd", "head_bwd", "clip_and_optimizer"]
    acc: dict[str, list[float]] = {n: [] for n in names}
    for _ in range(args.steps):
        ev: list[Any] = []
        full_step(events=ev)
        torch.cuda.synchronize()
        for n, a, b in zip(names, ev[:-1], ev[1:], strict=True):
            acc[n].append(a.elapsed_time(b))
    med = {n: sorted(v)[len(v) // 2] for n, v in acc.items()}
    total = sum(med.values())
    report["phases"] = {n: {"ms": round(v, 2), "pct": round(100 * v / total, 1)} for n, v in med.items()}
    report["phases"]["total_ms"] = round(total, 2)
    save()

    # 2. memory
    peaks: dict[str, int] = {}
    torch.cuda.reset_peak_memory_stats()
    for _ in range(2):
        full_step(peaks=peaks)
    report["memory"] = {
        "analytic": memory_ledger(model, opt),
        "measured_peak_alloc_gb_by_phase": {k: _gb(v) for k, v in peaks.items() if k != "start"},
        "reserved_gb": _gb(torch.cuda.memory_reserved()),
        "seq_len": int(bb_cache[0]["backbone_features"].shape[1]),
    }
    save()

    # 3 + 4. op / kernel profiles
    for key, fn in (("ops_full", full_step), ("ops_head", head_step)):
        try:
            report[key] = run_profile(fn, args.profile_steps, out_dir / f"trace_{key}.json")
        except Exception as exc:  # noqa: BLE001
            report[key] = {"error": f"{type(exc).__name__}: {exc}"}
        save()

    # 5. feature statistics
    try:
        report["features"] = feature_stats(model, bb_cache, act_cache, autocast)
    except Exception as exc:  # noqa: BLE001
        report["features"] = {"error": f"{type(exc).__name__}: {exc}"}
    save()
    print(json.dumps({k: v for k, v in report.items() if k not in ("ops_full", "ops_head")}, indent=2))
    for key in ("ops_full", "ops_head"):
        r = report.get(key, {})
        print(f"\n[{key}] gpu_busy_frac={r.get('gpu_busy_frac')} by_category={r.get('by_category_pct')}")
        for row in r.get("top_ops", [])[:20]:
            print(f"  {row['pct']:5.1f}%  {row['ms_per_step']:8.3f} ms  x{row['calls_per_step']:<5d} {row['op']}")


if __name__ == "__main__":
    main()
