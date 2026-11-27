# SPDX-License-Identifier: Apache-2.0
"""Fused single-pass Triton kernel for :class:`gr00t_lancedb.turbo.LowMemAdamW`.

One program handles ``ROWS`` quantisation blocks of ``BLOCK`` elements: it reads the weight,
compensation, gradient and both 8-bit moments once, writes all of them once, and reduces the
two per-block scales in registers. The arithmetic mirrors ``_update_tensor_torch`` line by
line; every codec step is an int32 bit operation (bf16 tensors are passed as int16 views) so
the result does not depend on how a Triton version lowers bf16 casts.

Import this module only on a CUDA machine with Triton installed.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .lowmem_adamw import BLOCK, _C1, _C2, _GOLDEN, _KV_BIAS, _X_MAX, _X_MIN

ROWS = 8
NUM_WARPS = 4


@triton.jit
def _mix32(x):
    x = x ^ ((x >> 16) & 0xFFFF)
    x = x * 2146121005  # 0x7FEB352D
    x = x ^ ((x >> 15) & 0x1FFFF)
    x = x * (-2073254261)  # 0x846CA68B as int32
    x = x ^ ((x >> 16) & 0xFFFF)
    return x


@triton.jit
def _bf16_to_f32(bits16):
    return (bits16.to(tl.int32) << 16).to(tl.float32, bitcast=True)


@triton.jit
def _f32_to_bf16(x):
    b = x.to(tl.int32, bitcast=True)
    return (b + 0x7FFF + ((b >> 16) & 1)) >> 16


@triton.jit
def _lowmem_adamw_kernel(
    w_ptr,
    c_ptr,
    g_ptr,
    qr_ptr,
    qv_ptr,
    sr_ptr,
    sv_ptr,
    n,
    lr,
    beta1,
    beta2,
    one_minus_beta1,
    one_minus_beta2,
    eps,
    weight_decay,
    inv_bc1,
    inv_bc2,
    grad_scale,
    seed,
    KAHAN: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
    KV_BIAS: tl.constexpr,
    X_MIN: tl.constexpr,
    X_MAX: tl.constexpr,
    GOLDEN: tl.constexpr,
):
    pid = tl.program_id(0)
    blk = pid * ROWS + tl.arange(0, ROWS)
    offs = blk[:, None] * BLOCK + tl.arange(0, BLOCK)[None, :]
    mask = offs < n
    blk_mask = (blk * BLOCK) < n

    if KAHAN:
        g = _bf16_to_f32(tl.load(g_ptr + offs, mask=mask, other=0)) * grad_scale
    else:
        g = tl.load(g_ptr + offs, mask=mask, other=0.0) * grad_scale

    sr_old = tl.load(sr_ptr + blk, mask=blk_mask, other=0.0)
    sv_old = tl.load(sv_ptr + blk, mask=blk_mask, other=0.0)
    kv_old = tl.load(qv_ptr + offs, mask=mask, other=0).to(tl.int32)
    kr_old = tl.load(qr_ptr + offs, mask=mask, other=0).to(tl.float32)

    v_old = ((kv_old + KV_BIAS) << 20).to(tl.float32, bitcast=True) * sv_old[:, None]
    m_old = kr_old * (sr_old[:, None] * (1.0 / 127.0)) * tl.sqrt(v_old)
    m = beta1 * m_old + one_minus_beta1 * g
    v = beta2 * v_old + one_minus_beta2 * g * g
    upd = (m * inv_bc1) / (tl.sqrt(v * inv_bc2) + eps)

    if KAHAN:
        eff = _bf16_to_f32(tl.load(w_ptr + offs, mask=mask, other=0)) + _bf16_to_f32(
            tl.load(c_ptr + offs, mask=mask, other=0)
        )
    else:
        eff = tl.load(w_ptr + offs, mask=mask, other=0.0)
    new = eff - lr * (upd + weight_decay * eff)

    h1 = _mix32(offs ^ seed)
    h2 = _mix32(h1 + GOLDEN)
    rand20 = (h1 >> 12) & 0xFFFFF
    u24 = ((h2 >> 8) & 0xFFFFFF).to(tl.float32) * (1.0 / 16777216.0)

    # second moment -> minifloat, stochastic rounding on the 20 dropped mantissa bits
    sv_new = tl.max(v, axis=1) * (1.0 / X_MAX)
    sv_b = sv_new[:, None]
    x = tl.where(sv_b > 0.0, v / sv_b, 0.0)
    x = tl.minimum(tl.maximum(x, X_MIN), X_MAX)
    xb = x.to(tl.int32, bitcast=True)
    kv = (xb >> 20) - KV_BIAS + (rand20 < (xb & 0xFFFFF)).to(tl.int32)
    kv = tl.minimum(tl.maximum(kv, 0), 255)
    v_hat = ((kv + KV_BIAS) << 20).to(tl.float32, bitcast=True) * sv_b

    # normalised momentum -> int8, stochastic rounding
    r = tl.where(v_hat > 0.0, m / tl.sqrt(v_hat), 0.0)
    sr_new = tl.max(tl.abs(r), axis=1)
    sr_b = sr_new[:, None]
    y = tl.where(sr_b > 0.0, r / sr_b, 0.0) * 127.0 + 127.0
    fl = y.to(tl.int32)
    kr = fl + (u24 < (y - fl.to(tl.float32))).to(tl.int32) - 127
    kr = tl.minimum(tl.maximum(kr, -127), 127)

    if KAHAN:
        wb = _f32_to_bf16(new)
        cb = _f32_to_bf16(new - (wb << 16).to(tl.float32, bitcast=True))
        tl.store(w_ptr + offs, wb.to(tl.int16), mask=mask)
        tl.store(c_ptr + offs, cb.to(tl.int16), mask=mask)
    else:
        tl.store(w_ptr + offs, new, mask=mask)
    tl.store(qv_ptr + offs, kv.to(tl.uint8), mask=mask)
    tl.store(qr_ptr + offs, kr.to(tl.int8), mask=mask)
    tl.store(sv_ptr + blk, sv_new, mask=blk_mask)
    tl.store(sr_ptr + blk, sr_new, mask=blk_mask)


def update_tensor_triton(
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
) -> None:
    kahan = comp is not None
    n = w.numel()
    if kahan:
        w_arg, c_arg, g_arg = w.view(torch.int16), comp.view(torch.int16), grad.view(torch.int16)
    else:
        w_arg, c_arg, g_arg = w, w, grad  # c_ptr is never dereferenced when KAHAN is False
    grid = (triton.cdiv(sr.numel(), ROWS),)
    _lowmem_adamw_kernel[grid](
        w_arg,
        c_arg,
        g_arg,
        qr,
        qv,
        sr,
        sv,
        n,
        lr,
        beta1,
        beta2,
        1.0 - beta1,
        1.0 - beta2,
        eps,
        weight_decay,
        inv_bc1,
        inv_bc2,
        grad_scale,
        seed,
        KAHAN=kahan,
        ROWS=ROWS,
        BLOCK=BLOCK,
        KV_BIAS=_KV_BIAS,
        X_MIN=_X_MIN,
        X_MAX=_X_MAX,
        GOLDEN=_GOLDEN,
        num_warps=NUM_WARPS,
    )


assert _C1 == 2146121005 and _C2 == -2073254261, "hash constants drifted from lowmem_adamw.py"
