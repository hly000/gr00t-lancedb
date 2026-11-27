# SPDX-License-Identifier: Apache-2.0
"""Compact encodings for the SIGTERM emergency checkpoint (issue #6, plan item 1.3).

GR00T-N1.7-3B fine-tuning trains 1.62 B parameters. An fp32 emergency checkpoint holds the
weights (4 B/param) plus AdamW ``exp_avg`` / ``exp_avg_sq`` (8 B/param): ~19 GB, which
does not reliably finish inside a 30 s spot grace period on a ~0.5 GB/s disk.

Presets (bytes per trainable parameter, weights + both moments):

``fp32``  12 B  exact (bit-identical resume)
``bf16``   6 B  weights, exp_avg, exp_avg_sq stored as bfloat16
``int8``   4 B  weights bfloat16; exp_avg blockwise absmax int8; exp_avg_sq as
                ``sqrt`` quantized to uint8 **rounding up**
``auto``       the most precise preset whose estimated write time fits the deadline

Lossy presets are **not** lossless: tensors come back as fp32 but rounded (bf16 keeps
~3 significant digits). ``GR00T_PREEMPT_CKPT_EXACT_WEIGHTS=1`` keeps weights fp32 under
any preset (8 / 6 B/param). The second moment is quantized in the ``sqrt`` domain with
ceil rounding so a decoded ``v`` is never smaller than the true one: Adam's update
``m / (sqrt(v) + eps)`` can only shrink, never blow up because ``v`` rounded to 0.
Small tensors (< ``MIN_NUMEL`` elements, e.g. the ``step`` counter) stay exact.
"""

from __future__ import annotations

from typing import Any

import torch

FORMAT_KEY = "__gr00t_codec__"
PRESETS: dict[str, dict[str, str]] = {
    "fp32": {"weights": "fp32", "moment": "fp32", "second_moment": "fp32"},
    "bf16": {"weights": "bf16", "moment": "bf16", "second_moment": "bf16"},
    "int8": {"weights": "bf16", "moment": "int8", "second_moment": "sqrt_u8"},
}
AUTO_ORDER = ("fp32", "bf16", "int8")
BYTES_PER_ELEMENT = {"fp32": 4.0, "bf16": 2.0, "int8": 1.0, "sqrt_u8": 1.0}
BLOCK = 2048
MIN_NUMEL = 1024
SECOND_MOMENT_KEYS = frozenset({"exp_avg_sq", "max_exp_avg_sq", "exp_avg_sq_row", "exp_avg_sq_col"})


def resolve_preset(name: str, *, exact_weights: bool = False) -> dict[str, str]:
    if name not in PRESETS:
        raise ValueError(f"Unknown emergency checkpoint dtype {name!r}; expected {sorted(PRESETS)} or 'auto'")
    spec = dict(PRESETS[name])
    if exact_weights:
        spec["weights"] = "fp32"
    spec["name"] = name
    return spec


def is_lossless(spec: dict[str, str]) -> bool:
    return all(spec[k] == "fp32" for k in ("weights", "moment", "second_moment"))


# ---------------------------------------------------------------------------
# tensor codecs
# ---------------------------------------------------------------------------


CHUNK = BLOCK * 8192  # 16.8 M elements: bounds device temporaries to ~64 MB per step


def _encode_chunk(x: torch.Tensor, codec: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 1-D fp32 chunk whose length is a multiple of ``BLOCK`` (except padding)."""
    pad = (-x.numel()) % BLOCK
    if pad:
        x = torch.cat([x, x.new_zeros(pad)])
    blocks = x.view(-1, BLOCK)
    if codec == "int8":
        scale = blocks.abs().amax(dim=1) / 127.0
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        q = torch.round(blocks / scale[:, None]).clamp_(-127, 127).to(torch.int8)
    elif codec == "sqrt_u8":
        blocks = blocks.clamp(min=0).sqrt()
        scale = blocks.amax(dim=1) / 255.0
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        # ceil: decoded sqrt(v) >= true sqrt(v)  ->  Adam updates never grow
        q = torch.ceil(blocks / scale[:, None]).clamp_(0, 255).to(torch.uint8)
    else:
        raise ValueError(f"Unknown codec {codec!r}")
    return q.reshape(-1).cpu(), scale.float().cpu()


def _encode_on(src: torch.Tensor, codec: str) -> dict[str, Any]:
    flat = src.reshape(-1)
    n = flat.numel()
    if codec == "bf16":
        parts = [flat[i : i + CHUNK].to(torch.bfloat16).cpu() for i in range(0, n, CHUNK)]
        return {"data": torch.cat(parts) if len(parts) > 1 else parts[0]}
    qs, scales = [], []
    for i in range(0, n, CHUNK):
        q, sc = _encode_chunk(flat[i : i + CHUNK].float(), codec)
        qs.append(q)
        scales.append(sc)
    return {"q": torch.cat(qs).view(-1, BLOCK), "scale": torch.cat(scales), "numel": n}


def encode_tensor(t: torch.Tensor, codec: str) -> Any:
    """Encode a floating tensor; returns the tensor itself (on CPU) when kept exact."""
    if codec == "fp32" or not torch.is_floating_point(t) or t.numel() < MIN_NUMEL:
        return t.detach().to("cpu", copy=True)
    if codec not in BYTES_PER_ELEMENT:
        raise ValueError(f"Unknown codec {codec!r}")
    src = t.detach()
    meta: dict[str, Any] = {FORMAT_KEY: codec, "shape": tuple(src.shape), "dtype": str(src.dtype).replace("torch.", "")}
    try:
        meta.update(_encode_on(src, codec))
    except torch.cuda.OutOfMemoryError:  # pragma: no cover - only on a full GPU
        meta.update(_encode_on(src.cpu(), codec))
    return meta


def decode_tensor(obj: Any) -> Any:
    if not (isinstance(obj, dict) and FORMAT_KEY in obj):
        return obj
    codec = obj[FORMAT_KEY]
    dtype = getattr(torch, obj.get("dtype", "float32"))
    if codec == "bf16":
        return obj["data"].to(dtype).reshape(obj["shape"])
    q, scale, n = obj["q"], obj["scale"], int(obj["numel"])
    if codec == "int8":
        x = q.float() * scale[:, None]
    elif codec == "sqrt_u8":
        x = (q.float() * scale[:, None]).square()
    else:
        raise ValueError(f"Unknown codec {codec!r}")
    return x.reshape(-1)[:n].to(dtype).reshape(obj["shape"])


def decode_tree(obj: Any) -> Any:
    """Recursively decode every encoded tensor inside dicts / lists / tuples."""
    if isinstance(obj, dict):
        if FORMAT_KEY in obj:
            return decode_tensor(obj)
        return {k: decode_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [decode_tree(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(decode_tree(v) for v in obj)
    return obj


# ---------------------------------------------------------------------------
# state-dict level helpers
# ---------------------------------------------------------------------------


def encode_trainable_state(named: dict[str, torch.Tensor], spec: dict[str, str]) -> dict[str, Any]:
    return {k: encode_tensor(v, spec["weights"]) for k, v in named.items()}


def encode_optimizer_state(sd: dict[str, Any] | None, spec: dict[str, str]) -> dict[str, Any] | None:
    """Encode ``optimizer.state_dict()``: moments compressed, counters / param_groups exact."""
    if sd is None:
        return None
    state = {}
    for pid, pstate in sd.get("state", {}).items():
        enc = {}
        for k, v in pstate.items():
            if isinstance(v, torch.Tensor):
                enc[k] = encode_tensor(v, spec["second_moment"] if k in SECOND_MOMENT_KEYS else spec["moment"])
            else:
                enc[k] = v
        state[pid] = enc
    out = dict(sd)
    out["state"] = state
    return out


def estimate_payload_bytes(model: Any, optimizer: Any, spec: dict[str, str]) -> int:
    """Bytes the emergency checkpoint will occupy for ``spec`` (tensor payload only)."""
    total = 0.0
    if model is not None:
        while hasattr(model, "module"):
            model = model.module
        for p in model.parameters():
            if p.requires_grad:
                total += _tensor_bytes(p, spec["weights"])
    if optimizer is not None:
        for pstate in getattr(optimizer, "state", {}).values():
            for k, v in pstate.items():
                if isinstance(v, torch.Tensor):
                    total += _tensor_bytes(v, spec["second_moment"] if k in SECOND_MOMENT_KEYS else spec["moment"])
    return int(total)


def _tensor_bytes(t: torch.Tensor, codec: str) -> float:
    n = t.numel()
    if codec == "fp32" or not torch.is_floating_point(t) or n < MIN_NUMEL:
        return float(n * t.element_size())
    per = BYTES_PER_ELEMENT[codec]
    if codec in ("int8", "sqrt_u8"):
        return n * per + 4.0 * ((n + BLOCK - 1) // BLOCK)
    return n * per


__all__ = [
    "AUTO_ORDER",
    "FORMAT_KEY",
    "PRESETS",
    "decode_tensor",
    "decode_tree",
    "encode_optimizer_state",
    "encode_tensor",
    "encode_trainable_state",
    "estimate_payload_bytes",
    "is_lossless",
    "resolve_preset",
]
