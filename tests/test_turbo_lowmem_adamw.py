# SPDX-License-Identifier: Apache-2.0
"""LowMemAdamW: codec correctness, fidelity to fp32 AdamW, resume, and Triton/torch parity."""

from __future__ import annotations

import copy
import math

import pytest

torch = pytest.importorskip("torch")

from gr00t_lancedb.turbo import BLOCK, LowMemAdamW, triton_available
from gr00t_lancedb.turbo import lowmem_adamw as L

BETAS = (0.95, 0.999)  # GR00T action-head defaults


def test_mix32_matches_pure_python():
    xs = [0, 1, 2, 12345, -5, 2**31 - 1, -(2**31), 0x12345678]
    got = L.mix32(torch.tensor(xs, dtype=torch.int32)).tolist()
    assert got == [L.mix32_py(x) for x in xs]
    # int32 multiply must wrap on this platform; a saturating multiply would collapse the hash
    assert len(set(L.mix32(torch.arange(100000, dtype=torch.int32)).tolist())) == 100000


def test_bf16_bit_codec_is_round_to_nearest_even():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(200000, generator=g) * torch.logspace(-8, 6, 200000)
    bits = L.f32_to_bf16_bits(x)
    assert torch.equal(bits, x.to(torch.bfloat16).view(torch.int16))
    assert torch.equal(L.bf16_bits_to_f32(bits), x.to(torch.bfloat16).to(torch.float32))


def _rand(shape, seed):
    offs = torch.arange(shape[0] * shape[1], dtype=torch.int32).reshape(shape)
    return L._randoms(offs, seed)


def test_second_moment_codec_is_unbiased_with_bounded_relative_error():
    g = torch.Generator().manual_seed(1)
    v = torch.exp(torch.randn(64, BLOCK, generator=g) * 3.0) * 1e-6  # ~26 octaves inside a block
    acc = torch.zeros_like(v, dtype=torch.float64)
    n_seeds = 400
    for seed in range(n_seeds):
        rand20, _ = _rand(v.shape, L.step_seed(0, seed, 1))
        code, scale = L.encode_v(v, rand20)
        assert int(code.min()) >= 0 and int(code.max()) <= 255
        v_hat = L.decode_v(code.to(torch.uint8), scale)
        ratio = v_hat / v
        assert float(ratio.min()) > 1 / 1.126 and float(ratio.max()) < 1.126
        acc += v_hat.double()
    mean_ratio = (acc / n_seeds) / v.double()
    assert float((mean_ratio - 1).abs().max()) < 0.02
    assert abs(float(mean_ratio.mean()) - 1) < 5e-4


def test_second_moment_codec_edge_cases():
    v = torch.zeros(3, BLOCK)
    v[1, 0] = 1.0  # one live element; the rest are exactly zero
    v[2] = 1e-30
    v[2, 5] = 1.0  # 100 octaves below the block max -> clamped up, never zero
    rand20, _ = _rand(v.shape, 7)
    code, scale = L.encode_v(v, rand20)
    v_hat = L.decode_v(code.to(torch.uint8), scale)
    assert torch.all(v_hat[0] == 0) and float(scale[0]) == 0
    assert float(v_hat[1, 0]) == pytest.approx(1.0)
    assert float(v_hat[2, 5]) == pytest.approx(1.0)
    assert 0 < float(v_hat[2, 0]) < 1e-8 and float(v_hat[2, 0]) >= 1e-30


def test_momentum_codec_is_unbiased():
    g = torch.Generator().manual_seed(2)
    r = torch.randn(32, BLOCK, generator=g)
    acc = torch.zeros_like(r, dtype=torch.float64)
    n_seeds = 400
    for seed in range(n_seeds):
        _, u24 = _rand(r.shape, L.step_seed(0, seed, 2))
        code, scale = L.encode_r(r, u24)
        assert int(code.abs().max()) <= 127
        acc += (code.to(torch.float32) * scale / 127.0).double()
    step = r.abs().amax(dim=1, keepdim=True) / 127.0
    assert float(((acc / n_seeds - r.double()).abs() / step).max()) < 0.12


def _train(kind, *, steps, n=20000, lr=1e-4, closed_loop=True, seed=0, noise_seed=123, backend="torch", device="cpu"):
    """Ill-conditioned synthetic problem (gradient scales span 4 decades), cosine LR to zero."""
    g = torch.Generator().manual_seed(11)
    w0 = torch.randn(n, generator=g) * 0.05
    target = w0 + torch.randn(n, generator=g) * 0.02
    scale = torch.logspace(-3, 1, n)[torch.randperm(n, generator=g)]
    drift = torch.randn(n, generator=g)
    dtype = torch.bfloat16 if kind in ("bf16", "kahan") else torch.float32
    p = torch.nn.Parameter(w0.clone().to(dtype).to(device))
    if kind in ("fp32", "bf16"):
        opt = torch.optim.AdamW([p], lr=lr, betas=BETAS, weight_decay=1e-5)
    else:
        opt = LowMemAdamW([p], lr=lr, betas=BETAS, weight_decay=1e-5, backend=backend, seed=seed)
    gn = torch.Generator().manual_seed(noise_seed)
    for t in range(steps):
        for group in opt.param_groups:
            group["lr"] = lr * 0.5 * (1 + math.cos(math.pi * t / steps))
        noise = torch.randn(n, generator=gn)
        if closed_loop:
            grad = scale * ((p.detach().float().cpu() - target) + 0.02 * noise)
        else:  # gradient independent of the weights: errors are integrated, never corrected
            grad = scale * (0.3 * drift + noise)
        p.grad = grad.to(dtype).to(device)
        opt.step()
    true_w = opt.true_weight(p).cpu() if isinstance(opt, LowMemAdamW) else p.detach().float().cpu()
    return w0, p.detach().float().cpu(), true_w


def test_tracks_fp32_adamw_where_pure_bf16_stalls():
    steps = 1500
    w0, ref, _ = _train("fp32", steps=steps)
    moved = (ref - w0).norm()
    rounding_floor = float((ref.to(torch.bfloat16).float() - ref).norm() / moved)

    _, bf16, _ = _train("bf16", steps=steps)
    assert float((bf16 - ref).norm() / moved) > 0.3  # the failure mode this optimizer exists for

    _, q8, _ = _train("q8", steps=steps)
    assert float((q8 - ref).norm() / moved) < 0.02

    _, fwd, true_w = _train("kahan", steps=steps)
    assert float((true_w - ref).norm() / moved) < 0.02
    # the bf16 forward weight cannot beat rounding the fp32 answer to bf16
    assert float((fwd - ref).norm() / moved) < rounding_floor + 0.02


def test_open_loop_error_is_small_next_to_minibatch_noise():
    """Gradients that ignore the weights integrate every error. Stochastic rounding of the
    moments must then look like a little extra gradient noise, not like a different optimizer."""
    steps = 1500
    w0, ref, _ = _train("fp32", steps=steps, closed_loop=False)
    d_ref = ref - w0
    _, other, _ = _train("fp32", steps=steps, closed_loop=False, noise_seed=999)
    minibatch_dev = float((other - ref).norm() / d_ref.norm())

    _, bf16, _ = _train("bf16", steps=steps, closed_loop=False)
    assert float(((bf16 - w0) * d_ref).sum() / (d_ref * d_ref).sum()) < 0.5  # moves a fraction of the way

    for kind in ("q8", "kahan"):
        _, _, true_w = _train(kind, steps=steps, closed_loop=False)
        d = true_w - w0
        gain = float((d * d_ref).sum() / (d_ref * d_ref).sum())
        assert 0.98 < gain < 1.03  # E[1/sqrt(v_hat)] > 1/sqrt(v): ~1.4 % larger effective step
        assert float((true_w - ref).norm() / d_ref.norm()) < 0.5 * minibatch_dev


def test_carried_moments_match_exact_adam():
    n, steps = 8 * BLOCK, 300
    g = torch.Generator().manual_seed(5)
    scale = torch.logspace(-4, 0, n)[torch.randperm(n, generator=g)]
    p = torch.nn.Parameter(torch.zeros(n))
    opt = LowMemAdamW([p], lr=0.0, betas=BETAS, backend="torch")
    m = torch.zeros(n, dtype=torch.float64)
    v = torch.zeros(n, dtype=torch.float64)
    for _ in range(steps):
        grad = scale * (0.5 + torch.randn(n, generator=g))
        p.grad = grad.clone()
        opt.step()
        m = BETAS[0] * m + (1 - BETAS[0]) * grad.double()
        v = BETAS[1] * v + (1 - BETAS[1]) * grad.double() ** 2
    st = opt.state[p]
    v_hat = L.decode_v(st["qv"].reshape(-1, BLOCK), st["sv"].unsqueeze(1)).reshape(-1).double()
    m_hat = (st["qr"].reshape(-1, BLOCK).float() * (st["sr"].unsqueeze(1) / 127.0)).reshape(-1).double() * v_hat.sqrt()
    # Stochastic rounding keeps the EMA unbiased but not noise-free: the carried v wanders
    # ~20 % rms around the exact value (~10 % on the step size, far below minibatch noise).
    v_err = (v_hat / v - 1).abs()
    assert float(v_err.mean()) < 0.25 and float(v_err.max()) < 1.5
    assert abs(float((v_hat / v).mean()) - 1) < 0.03  # no systematic drift of the EMA
    m_err = (m_hat - m) / v.sqrt()
    assert float(m_err.abs().mean()) < 0.08 and abs(float(m_err.mean())) < 0.01


def test_state_dict_resume_is_bit_exact():
    def make():
        torch.manual_seed(0)
        return torch.nn.Parameter((torch.randn(3, 301) * 0.05).to(torch.bfloat16))

    def grads(k):
        g = torch.Generator().manual_seed(100 + k)
        return (torch.randn(3, 301, generator=g) * 0.1).to(torch.bfloat16)

    p_ref = make()
    opt_ref = LowMemAdamW([p_ref], lr=1e-3, betas=BETAS, weight_decay=0.01, backend="torch", seed=3)
    for k in range(12):
        p_ref.grad = grads(k)
        opt_ref.step()

    p = make()
    opt = LowMemAdamW([p], lr=1e-3, betas=BETAS, weight_decay=0.01, backend="torch", seed=3)
    for k in range(5):
        p.grad = grads(k)
        opt.step()
    saved_w, saved_opt = p.detach().clone(), copy.deepcopy(opt.state_dict())

    p2 = torch.nn.Parameter(saved_w)
    opt2 = LowMemAdamW([p2], lr=1e-3, betas=BETAS, weight_decay=0.01, backend="torch", seed=3)
    opt2.load_state_dict(saved_opt)
    for k in range(5, 12):
        p2.grad = grads(k)
        opt2.step()
    assert torch.equal(p2.view(torch.int16), p_ref.view(torch.int16))
    assert torch.equal(opt2.state[p2]["comp"].view(torch.int16), opt_ref.state[p_ref]["comp"].view(torch.int16))
    assert torch.equal(opt2.state[p2]["qr"], opt_ref.state[p_ref]["qr"])


def test_grad_scale_and_clip_scale():
    def run(scale_in_grad):
        torch.manual_seed(0)
        p = torch.nn.Parameter(torch.randn(700) * 0.05)
        q = torch.nn.Parameter(torch.randn(5, 7) * 0.05)
        frozen = torch.nn.Parameter(torch.randn(9))
        opt = LowMemAdamW([p, q, frozen], lr=1e-3, backend="torch")
        g = torch.Generator().manual_seed(1)
        for _ in range(4):
            gp, gq = torch.randn(700, generator=g) * 4, torch.randn(5, 7, generator=g) * 4
            p.grad, q.grad = gp.clone(), gq.clone()
            norm, scale = opt.clip_scale(1.0)
            assert norm == pytest.approx(float(torch.cat([gp, gq.reshape(-1)]).norm()), rel=1e-5)
            assert scale == pytest.approx(1.0 / norm, rel=1e-4)
            if scale_in_grad:
                p.grad, q.grad = gp * scale, gq * scale
                opt.step()
            else:
                opt.step(grad_scale=scale)
        assert frozen not in opt.state  # no gradient -> untouched, no state
        return p.detach().clone(), q.detach().clone()

    (p1, q1), (p2, q2) = run(True), run(False)
    assert torch.allclose(p1, p2, atol=2e-6) and torch.allclose(q1, q2, atol=2e-6)


def test_weight_decay_only_and_state_size():
    p = torch.nn.Parameter(torch.full((1000,), 0.5, dtype=torch.bfloat16))
    opt = LowMemAdamW([p], lr=0.01, weight_decay=0.1, backend="torch")
    for _ in range(100):
        p.grad = torch.zeros_like(p)
        opt.step()
    assert torch.allclose(opt.true_weight(p), torch.full((1000,), 0.5 * (1 - 1e-3) ** 100), rtol=2e-4)
    # comp (2 B) + qr (1 B) + qv (1 B) per element, two fp32 scales per block
    assert opt.state_bytes() == 1000 * 4 + math.ceil(1000 / BLOCK) * 8


def test_rejects_unsupported_dtype():
    p = torch.nn.Parameter(torch.zeros(4, dtype=torch.float64))
    p.grad = torch.zeros_like(p)
    with pytest.raises(TypeError):
        LowMemAdamW([p], backend="torch").step()


@pytest.mark.skipif(not triton_available(), reason="needs CUDA + triton")
@pytest.mark.parametrize("dtype", ["bf16", "fp32"])
@pytest.mark.parametrize("n", [BLOCK * 8 * 5, 100003, 7])
def test_triton_kernel_matches_torch_reference(dtype, n):
    """Run this first on the GPU box: it is the only check the Triton path gets off-GPU."""
    dt = torch.bfloat16 if dtype == "bf16" else torch.float32
    g = torch.Generator().manual_seed(0)
    w0 = (torch.randn(n, generator=g) * 0.05).to(dt)
    scale = torch.logspace(-3, 1, n)
    ps, opts = [], []
    for backend in ("torch", "triton"):
        p = torch.nn.Parameter(w0.clone().cuda())
        ps.append(p)
        opts.append(LowMemAdamW([p], lr=1e-3, betas=BETAS, weight_decay=0.01, backend=backend, seed=9))
    for k in range(20):
        grad = (scale * torch.randn(n, generator=g)).to(dt).cuda()
        if k % 5 == 0:  # resync so one-ulp FMA differences cannot compound across steps
            ps[1].data.copy_(ps[0].data)
            for key, val in opts[0].state[ps[0]].items():
                if isinstance(val, torch.Tensor):
                    opts[1].state.setdefault(ps[1], {})
                    opts[1].state[ps[1]][key] = val.clone()
                else:
                    opts[1].state[ps[1]][key] = val
        for p, opt in zip(ps, opts, strict=True):
            p.grad = grad.clone()
            opt.step(grad_scale=0.7)
        if k == 15:
            # Single-step exact arithmetic check immediately after resync (no multi-step SR carry):
            # weights and sv must match tightly; sr may shift on a rare 1-ULP kv flip (<= 1/16).
            a1, b1 = opts[0].state[ps[0]], opts[1].state[ps[1]]
            w1a, w1b = opts[0].true_weight(ps[0]), opts[1].true_weight(ps[1])
            assert float((w1a - w1b).abs().max()) < 2e-6
            assert torch.allclose(a1["sv"], b1["sv"], rtol=1e-5)
            sr1_rel = (a1["sr"] - b1["sr"]).abs() / b1["sr"].abs().clamp_min(1e-8)
            assert float((sr1_rel <= 1e-4).float().mean()) >= 0.99
            assert float(sr1_rel.max()) < 0.08
    a, b = opts[0].state[ps[0]], opts[1].state[ps[1]]
    assert float((a["qv"] != b["qv"]).float().mean()) < 0.02
    assert float((a["qr"] != b["qr"]).float().mean()) < 0.02
    assert int((a["qv"].int() - b["qv"].int()).abs().max()) <= 1
    assert torch.allclose(a["sv"], b["sv"], rtol=1e-5)
    # Across 5 unsynced steps on 100k elements, a 1-ULP FMA/div difference can flip an int8 qr
    # stochastic-rounding coin on a few elements; if that element is a block argmax, its block's
    # sr shifts by ~beta1^3/127 (~0.67%, or <= ~6% for a kv flip). Check both bulk and worst block.
    sr_rel = (a["sr"] - b["sr"]).abs() / b["sr"].abs().clamp_min(1e-8)
    assert float((sr_rel <= 1e-4).float().mean()) >= 0.98
    assert float(sr_rel.max()) < 0.08
    wa, wb = opts[0].true_weight(ps[0]), opts[1].true_weight(ps[1])
    assert float((wa - wb).abs().max()) < 2e-5
