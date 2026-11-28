# SPDX-License-Identifier: Apache-2.0
"""AdamW with 2 bytes/param of optimizer state and bf16 weights that still accumulate
sub-ulp updates.

Why: the GR00T N1.7 action head has 1.62 B trainable parameters. ``torch.optim.AdamW``
on fp32 weights holds weight + grad + two moments = 16 bytes/param (25.9 GB) before a single
activation is allocated; casting everything to bf16 halves that but silently drops every
update smaller than half a bf16 ulp (0.4 % of the weight), which at ``lr=1e-4`` is most of
them. This optimizer keeps, per parameter:

====================  =========  ==============================================
weight                bf16       what the forward pass reads
compensation          bf16       ``true weight - bf16(weight)`` (Kahan carry)
normalised momentum   int8       ``m / sqrt(v)``, per-128-element block scale
second moment         uint8      5-bit exponent + 3-bit mantissa "minifloat",
                                 per-block scale
block scales          2 x fp32   per 128 elements (0.06 bytes/param)
====================  =========  ==============================================

Total 6.06 bytes/param + a bf16 gradient = 8.06 bytes/param (13.1 GB for the action head).
fp32 parameters are also accepted (no compensation tensor; 6.06 + 4 bytes/param).

Design notes

* **Weight + compensation** carries ~16 significant bits (fp32 has 24). The forward pass
  sees ``bf16(true weight)``, which is exactly what ``autocast(bf16)`` feeds a matmul from
  an fp32 master weight.
* **Both moments use stochastic rounding.** With round-to-nearest an EMA whose per-step
  change is smaller than half a quantisation level never moves (``beta2=0.999`` changes
  ``v`` by 0.1 %/step; the level spacing is 6-12 %). Stochastic rounding makes the stored
  value an unbiased estimate, so the EMA is exact in expectation.
* **The second moment is quantised in the value domain with log-spaced levels** (relative
  error bounded, 31 octaves of range inside a block), not linearly: a linear code rounds
  small ``v`` to zero and the update ``m / sqrt(v)`` explodes. Values below the range are
  clamped *up* (smaller steps, the safe direction).
* **Momentum is stored as ``m / sqrt(v_hat)``** so the int8 resolution is relative to each
  element's own gradient scale instead of the largest element of its block.
* The update applied at step *t* uses the exact (unquantised) moments of step *t*;
  quantisation only affects the state carried to step *t+1*.

What it costs (measured by ``tests/test_turbo_lowmem_adamw.py`` on a synthetic problem, not
on a real model): unbiased is not noise-free. The carried second moment wanders ~20 % rms
around the exact EMA (~10 % on the per-element step size) and ``E[1/sqrt(v_hat)]`` exceeds
``1/sqrt(v)``, so the effective learning rate is ~1.4 % higher. With gradients that ignore
the weights (errors integrate, nothing corrects them) the final weights differ from fp32
AdamW by under half of what a different minibatch order causes.

Two interchangeable backends compute the same arithmetic:

* ``torch``: chunked reference implementation, runs anywhere (CPU tests use it).
* ``triton``: one fused pass over ``(w, c, g, q_r, q_v)`` per tensor. All codec steps are
  integer bit operations so both backends agree except for last-bit differences where the
  GPU contracts ``a*b+c`` into an FMA.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

import torch

BLOCK = 128
_KV_BIAS = (127 - 31) << 3  # minifloat code 0 <-> 2**-31
_X_MIN = 2.0**-31
_X_MAX = 1.875
_MASK32 = 0xFFFFFFFF


# ---------------------------------------------------------------------------
# Bit-level helpers shared (by construction) with the Triton kernel
# ---------------------------------------------------------------------------


def _i32(x: int) -> int:
    """Python int -> two's-complement int32 value."""
    x &= _MASK32
    return x - (1 << 32) if x >= (1 << 31) else x


_C1 = _i32(0x7FEB352D)
_C2 = _i32(0x846CA68B)
_GOLDEN = _i32(0x9E3779B9)


def mix32(x: torch.Tensor) -> torch.Tensor:
    """lowbias32 integer hash on int32 tensors (wrapping multiply, logical shifts)."""
    x = x ^ ((x >> 16) & 0xFFFF)
    x = x * _C1
    x = x ^ ((x >> 15) & 0x1FFFF)
    x = x * _C2
    x = x ^ ((x >> 16) & 0xFFFF)
    return x


def mix32_py(x: int) -> int:
    """Pure-Python lowbias32, the ground truth for :func:`mix32`."""
    x &= _MASK32
    x ^= x >> 16
    x = (x * 0x7FEB352D) & _MASK32
    x ^= x >> 15
    x = (x * 0x846CA68B) & _MASK32
    x ^= x >> 16
    return _i32(x)


def step_seed(base_seed: int, step: int, tensor_index: int) -> int:
    """Well-mixed int32 seed for one (step, tensor) kernel launch."""
    h = mix32_py(base_seed ^ 0x51ED270B)
    h = mix32_py(h ^ (step & _MASK32))
    h = mix32_py(h ^ ((tensor_index * 0x9E3779B1) & _MASK32))
    return h


def bf16_bits_to_f32(bits16: torch.Tensor) -> torch.Tensor:
    """int16 bf16 bit pattern -> float32 (exact)."""
    return (bits16.to(torch.int32) << 16).view(torch.float32)


def f32_to_bf16_bits(x: torch.Tensor) -> torch.Tensor:
    """float32 -> int16 bf16 bit pattern, round-to-nearest-even."""
    b = x.contiguous().view(torch.int32)
    return ((b + 0x7FFF + ((b >> 16) & 1)) >> 16).to(torch.int16)


def decode_v(code: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """uint8 minifloat code + per-block scale -> float32 second moment."""
    return ((code.to(torch.int32) + _KV_BIAS) << 20).view(torch.float32) * scale


def encode_v(v: torch.Tensor, rand20: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``v [rows, BLOCK]`` -> ``(code int32 in [0, 255], scale [rows, 1])``, stochastic rounding."""
    scale = v.amax(dim=1, keepdim=True) * (1.0 / _X_MAX)
    x = torch.where(scale > 0, v / scale, torch.zeros_like(v)).clamp(_X_MIN, _X_MAX)
    bits = x.view(torch.int32)
    code = (bits >> 20) - _KV_BIAS + (rand20 < (bits & 0xFFFFF)).to(torch.int32)
    return code.clamp_(0, 255), scale


def encode_r(r: torch.Tensor, u24: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``r [rows, BLOCK]`` -> ``(code int32 in [-127, 127], scale [rows, 1])``, stochastic rounding."""
    scale = r.abs().amax(dim=1, keepdim=True)
    y = torch.where(scale > 0, r / scale, torch.zeros_like(r)) * 127.0 + 127.0
    fl = y.to(torch.int32)
    code = fl + (u24 < (y - fl.to(torch.float32))).to(torch.int32) - 127
    return code.clamp_(-127, 127), scale


def _randoms(offsets: torch.Tensor, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    h1 = mix32(offsets ^ seed)
    h2 = mix32(h1 + _GOLDEN)
    rand20 = (h1 >> 12) & 0xFFFFF
    u24 = ((h2 >> 8) & 0xFFFFFF).to(torch.float32) * (1.0 / 16777216.0)
    return rand20, u24


# ---------------------------------------------------------------------------
# Reference backend
# ---------------------------------------------------------------------------


def _update_tensor_torch(
    w: torch.Tensor,
    comp: torch.Tensor | None,
    grad: torch.Tensor,
    qr: torch.Tensor,
    qv: torch.Tensor,
    sr: torch.Tensor,
    sv: torch.Tensor,
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    inv_bc1: float,
    inv_bc2: float,
    grad_scale: float,
    seed: int,
    chunk_blocks: int,
) -> None:
    """One AdamW step for one tensor, ``chunk_blocks`` blocks at a time (bounded temporaries)."""
    kahan = comp is not None
    n = w.numel()
    nb = sr.numel()
    w_flat = w.view(torch.int16).reshape(-1) if kahan else w.reshape(-1)
    c_flat = comp.view(torch.int16).reshape(-1) if kahan else None
    g_flat = grad.view(torch.int16).reshape(-1) if grad.dtype == torch.bfloat16 else grad.reshape(-1)

    for b0 in range(0, nb, chunk_blocks):
        b1 = min(nb, b0 + chunk_blocks)
        s, e = b0 * BLOCK, min(n, b1 * BLOCK)
        rows, cnt = b1 - b0, e - s
        pad = rows * BLOCK - cnt

        def _load(flat: torch.Tensor) -> torch.Tensor:
            x = flat[s:e]
            if pad:
                x = torch.cat([x, x.new_zeros(pad)])
            return x.reshape(rows, BLOCK)

        g_raw = _load(g_flat)
        g = (bf16_bits_to_f32(g_raw) if g_raw.dtype == torch.int16 else g_raw.to(torch.float32)) * grad_scale
        sr_old = sr[b0:b1].unsqueeze(1)
        sv_old = sv[b0:b1].unsqueeze(1)

        v_old = decode_v(_load(qv), sv_old)
        m_old = _load(qr).to(torch.float32) * (sr_old * (1.0 / 127.0)) * torch.sqrt(v_old)
        m = beta1 * m_old + (1.0 - beta1) * g
        v = beta2 * v_old + (1.0 - beta2) * g * g
        upd = (m * inv_bc1) / (torch.sqrt(v * inv_bc2) + eps)

        if kahan:
            eff = bf16_bits_to_f32(_load(w_flat)) + bf16_bits_to_f32(_load(c_flat))
        else:
            eff = _load(w_flat).to(torch.float32)
        new = eff - lr * (upd + weight_decay * eff)

        if pad:
            v[-1, BLOCK - pad :] = 0.0
        offsets = torch.arange(s, s + rows * BLOCK, dtype=torch.int32, device=w.device).reshape(rows, BLOCK)
        rand20, u24 = _randoms(offsets, seed)
        kv, sv_new = encode_v(v, rand20)
        v_hat = decode_v(kv, sv_new)
        r = torch.where(v_hat > 0, m / torch.sqrt(v_hat), torch.zeros_like(m))
        if pad:
            r[-1, BLOCK - pad :] = 0.0
        kr, sr_new = encode_r(r, u24)

        if kahan:
            w_bits = f32_to_bf16_bits(new)
            c_bits = f32_to_bf16_bits(new - bf16_bits_to_f32(w_bits))
            w_flat[s:e] = w_bits.reshape(-1)[:cnt]
            c_flat[s:e] = c_bits.reshape(-1)[:cnt]
        else:
            w_flat[s:e] = new.reshape(-1)[:cnt]
        qv[s:e] = kv.reshape(-1)[:cnt].to(torch.uint8)
        qr[s:e] = kr.reshape(-1)[:cnt].to(torch.int8)
        sv[b0:b1] = sv_new.squeeze(1)
        sr[b0:b1] = sr_new.squeeze(1)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


def triton_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401
    except Exception:
        return False
    return True


class LowMemAdamW(torch.optim.Optimizer):
    """AdamW with int8/uint8 moments and Kahan-compensated bf16 (or plain fp32) weights.

    Args:
        backend: ``"auto"`` picks the Triton kernel for CUDA parameters when Triton imports,
            otherwise the chunked torch implementation. ``"triton"`` / ``"torch"`` force one.
        seed: stochastic-rounding stream; a run is reproducible given (seed, step count).
        chunk_blocks: torch backend only, blocks processed per temporary allocation.

    ``step(grad_scale=...)`` multiplies every gradient inside the update (use it with
    :meth:`clip_scale` instead of rewriting bf16 gradients in place).
    """

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        *,
        backend: str = "auto",
        seed: int = 0,
        chunk_blocks: int = 1 << 13,
    ) -> None:
        if lr < 0 or eps < 0 or weight_decay < 0:
            raise ValueError("lr, eps and weight_decay must be non-negative")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"invalid betas {betas}")
        if backend not in ("auto", "torch", "triton"):
            raise ValueError(f"unknown backend {backend!r}")
        if backend == "triton" and not triton_available():
            raise RuntimeError("backend='triton' needs CUDA and an importable triton")
        super().__init__(params, {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay})
        self.backend = backend
        self.seed = int(seed)
        self.chunk_blocks = int(chunk_blocks)
        self._kernel: Callable[..., None] | None = None

    # -- state ---------------------------------------------------------------

    def _init_state(self, p: torch.Tensor, index: int) -> dict[str, Any]:
        if p.dtype not in (torch.bfloat16, torch.float32):
            raise TypeError(f"LowMemAdamW supports bfloat16 and float32 parameters, got {p.dtype}")
        if not p.is_contiguous():
            raise ValueError("LowMemAdamW needs contiguous parameters")
        n = p.numel()
        nb = (n + BLOCK - 1) // BLOCK
        st = self.state[p]
        st["index"] = index
        st["step"] = 0
        st["qr"] = torch.zeros(n, dtype=torch.int8, device=p.device)
        st["qv"] = torch.zeros(n, dtype=torch.uint8, device=p.device)
        st["sr"] = torch.zeros(nb, dtype=torch.float32, device=p.device)
        st["sv"] = torch.zeros(nb, dtype=torch.float32, device=p.device)
        if p.dtype == torch.bfloat16:
            st["comp"] = torch.zeros_like(p)
        return st

    def state_bytes(self) -> int:
        """Bytes held in optimizer state tensors (excludes weights and gradients)."""
        total = 0
        for st in self.state.values():
            for v in st.values():
                if isinstance(v, torch.Tensor):
                    total += v.numel() * v.element_size()
        return total

    @torch.no_grad()
    def clip_scale(self, max_norm: float) -> tuple[float, float]:
        """Global gradient L2 norm (accumulated in fp32) and the matching clip multiplier.

        Returns ``(norm, scale)``; pass ``scale`` to ``step(grad_scale=scale)``.
        """
        grads = [
            p.grad
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None and p.grad.numel() > 0
        ]
        if not grads:
            return 0.0, 1.0
        by_key: dict[tuple[torch.device, torch.dtype], list[torch.Tensor]] = {}
        for g in grads:
            by_key.setdefault((g.device, g.dtype), []).append(g)
        norms: list[torch.Tensor] = []
        for (_, dtype), bucket in by_key.items():
            # _foreach_norm avoids ~300 per-tensor kernel launches on the GR00T action head.
            # For fp32 grads it accumulates in fp32 directly; for bf16 we keep fp32 accumulation
            # via vector_norm only if _foreach_norm is unavailable, or upcast small norms to fp32.
            if hasattr(torch, "_foreach_norm") and dtype == torch.float32:
                norms.extend(n.to(torch.float32) for n in torch._foreach_norm(bucket, 2.0))
            else:
                norms.extend(torch.linalg.vector_norm(g, dtype=torch.float32) for g in bucket)
        norm = float(torch.linalg.vector_norm(torch.stack([n.to(norms[0].device) for n in norms])))
        return norm, min(1.0, float(max_norm) / (norm + 1e-6))

    @torch.no_grad()
    def true_weight(self, p: torch.Tensor) -> torch.Tensor:
        """fp32 ``weight + compensation`` (what the optimizer is actually integrating)."""
        st = self.state.get(p)
        if st is None or "comp" not in st:
            return p.detach().to(torch.float32)
        return p.detach().to(torch.float32) + st["comp"].to(torch.float32)

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # torch casts every state tensor to the parameter's dtype on load, which would turn
        # the int8/uint8 codes and fp32 block scales into bf16. Restore them verbatim.
        super().load_state_dict(state_dict)
        saved_ids = [i for g in state_dict["param_groups"] for i in g["params"]]
        params = [p for g in self.param_groups for p in g["params"]]
        for pid, p in zip(saved_ids, params, strict=True):
            saved = state_dict["state"].get(pid)
            if saved is None:
                continue
            self.state[p] = {
                k: (v.detach().clone().to(p.device) if isinstance(v, torch.Tensor) else v) for k, v in saved.items()
            }

    # -- step ----------------------------------------------------------------

    def _use_triton(self, p: torch.Tensor) -> bool:
        if self.backend == "torch" or not p.is_cuda:
            return False
        if self.backend == "triton":
            return True
        return triton_available()

    @torch.no_grad()
    def step(self, closure: Callable[[], Any] | None = None, *, grad_scale: float = 1.0) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        index = 0
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                index += 1
                if p.grad is None or p.numel() == 0:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("LowMemAdamW does not support sparse gradients")
                if grad.dtype != p.dtype or not grad.is_contiguous():
                    grad = grad.to(p.dtype).contiguous()
                st = self.state[p]
                if not st:
                    st = self._init_state(p, index)
                st["step"] += 1
                t = st["step"]
                kw = dict(
                    lr=float(group["lr"]),
                    beta1=float(beta1),
                    beta2=float(beta2),
                    eps=float(group["eps"]),
                    weight_decay=float(group["weight_decay"]),
                    inv_bc1=1.0 / (1.0 - beta1**t),
                    inv_bc2=1.0 / (1.0 - beta2**t),
                    grad_scale=float(grad_scale),
                    seed=step_seed(self.seed, t, st["index"]),
                )
                args = (p.data, st.get("comp"), grad, st["qr"], st["qv"], st["sr"], st["sv"])
                if self._use_triton(p):
                    if self._kernel is None:
                        from ._triton_adamw import update_tensor_triton

                        self._kernel = update_tensor_triton
                    self._kernel(*args, **kw)
                else:
                    _update_tensor_torch(*args, chunk_blocks=self.chunk_blocks, **kw)
        return loss
