# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the GPU-only scripts: everything that does not need the real model."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gr00t_lancedb.vl_cache import build_action_head_optimizer  # noqa: E402
from gr00t_lancedb.vl_cache.codec import decode_features_numpy, encode_features  # noqa: E402

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pk = _load("profile_kernels")
bench = _load("bench_lowmem_adamw")


def test_categorize():
    assert pk.categorize("aten::linear") == "gemm"
    assert pk.categorize("aten::addmm") == "gemm"
    assert pk.categorize("AddmmBackward0") == "gemm"
    assert pk.categorize("autograd::engine::evaluate_function: MmBackward0") == "gemm"
    assert pk.categorize("aten::native_layer_norm") == "norm"
    assert pk.categorize("aten::scaled_dot_product_attention") == "attention"
    assert pk.categorize("aten::_fused_adamw_") == "optimizer"
    assert pk.categorize("Optimizer.step#AdamW.step") == "optimizer"
    assert pk.categorize("aten::to") == "copy_cast"
    assert pk.categorize("aten::topk") == "other"  # no substring matching
    assert pk.categorize("aten::mul") == "elementwise"


def test_run_profile_on_cpu(tmp_path: Path):
    model = torch.nn.Sequential(torch.nn.Linear(64, 256), torch.nn.GELU(), torch.nn.Linear(256, 8))
    opt = torch.optim.AdamW(model.parameters())
    x = torch.randn(32, 64)

    def step():
        opt.zero_grad()
        model(x).pow(2).mean().backward()
        opt.step()

    rep = pk.run_profile(step, 3, tmp_path / "trace.json", on_gpu=False)
    assert rep["profiled_steps"] == 3 and rep["top_ops"]
    assert "gemm" in rep["by_category_pct"]
    assert abs(sum(rep["by_category_pct"].values()) - 100) < 1.0
    assert all(r["calls_per_step"] >= 0 for r in rep["top_ops"])


def test_int8_per_token_matches_cache_codec():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((37, 64)) * rng.lognormal(0, 1.5, (37, 1))).astype(np.float32)
    x[:, 5] *= 40  # an outlier channel
    scale, payload = encode_features(x, codec="per_token_int8")
    ref = decode_features_numpy(scale, payload, 37, 64, codec="per_token_int8")
    got = pk.int8_per_token(torch.from_numpy(x)).numpy()
    assert np.array_equal(got, ref)


class _FakeHead:
    """Loss depends on the features, the noise drawn after seeding, and nothing else."""

    def __call__(self, bb, act):
        feats = bb["backbone_features"].float()
        noise = torch.randn(act["action"].shape)
        pred = feats.mean(dim=(1, 2), keepdim=False)[:, None, None] + noise
        return {"action_loss": (pred - act["action"]) ** 2, "action_mask": torch.ones_like(act["action"])}


def test_feature_stats_and_memory_ledger():
    from contextlib import nullcontext

    from transformers import BatchFeature

    g = torch.Generator().manual_seed(0)
    bbs, acts = [], []
    for _ in range(2):
        f = torch.randn(4, 12, 32, generator=g)
        f[..., 3] *= 25
        m = torch.ones(4, 12, dtype=torch.bool)
        m[:, :3] = False
        bbs.append(BatchFeature(data={"backbone_features": f, "backbone_attention_mask": m}))
        acts.append({"action": torch.randn(4, 16, 7, generator=g)})
    model = types.SimpleNamespace(action_head=_FakeHead())
    st = pk.feature_stats(model, bbs, acts, nullcontext)
    assert st["tokens"] == 2 * 4 * 9 and st["hidden_dim"] == 32
    assert st["top_outlier_channels"][0] == 3
    assert 0 < st["int8_rel_l2_error"] < 0.05
    assert st["int8_channel_rel_error_p50_p90_p99_max"][0] > st["int8_rel_l2_error"]  # small channels pay
    assert len(st["head_loss_exact_features"]) == 2
    assert 0 <= st["head_loss_per_sample_rel_diff_mean_max"][0] < 0.05  # same noise for both variants

    backbone = torch.nn.Linear(10, 10).requires_grad_(False)
    head = torch.nn.Linear(10, 20)
    opt = torch.optim.AdamW(head.parameters())
    head(torch.randn(2, 10)).sum().backward()
    opt.step()
    led = pk.memory_ledger(types.SimpleNamespace(backbone=backbone, action_head=head), opt)
    n_head = 10 * 20 + 20
    assert led["static_total_gb"] == pytest.approx(((110 + n_head) * 4 + n_head * 4 + n_head * 8) / 1024**3, abs=1e-3)


def test_synthetic_shapes_hit_requested_size():
    shapes = bench.synthetic_shapes(int(1.62e9))
    total = sum(int(np.prod(s)) for s in shapes)
    assert 0.95 * 1.62e9 < total <= 1.62e9
    assert len(shapes) > 300  # per-tensor launch overhead is part of what the bench measures


def test_optimizer_builder_kinds():
    lin = torch.nn.Linear(8, 8)
    opt, name = build_action_head_optimizer(lin, kind="lowmem", lr=1e-3)
    assert name == "LowMemAdamW(torch)" and type(opt).__name__ == "LowMemAdamW"
    _, name = build_action_head_optimizer(lin, kind="adamw")
    assert name.startswith("torch.AdamW")
    with pytest.raises(ValueError):
        build_action_head_optimizer(lin, kind="nope")
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError):  # an explicit request must not silently fall back
            build_action_head_optimizer(lin, kind="bnb8bit")
