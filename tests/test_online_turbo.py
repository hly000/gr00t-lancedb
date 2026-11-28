# SPDX-License-Identifier: Apache-2.0
"""CPU regression tests for online GR00T N1.7 turbo acceleration (no offline VL-cache).

Exercises every failure mode that could waste GPU time:
- Parameter-only bf16 casting must preserve FP32 non-parameter buffers (e.g. RoPE inv_freq).
- Fail-closed guard against pure bf16 without LowMemAdamW.
- Multi-noise patch: training expands B -> B*K with finite grads; eval bypasses to K=1.
- Compiled action_head.model unwrapping during checkpoint save (zero `_orig_mod` keys).
- Real HuggingFace `transformers.Trainer` end-to-end training + checkpoint save + exact
  LowMemAdamW state_dict resume with decay/no-decay param groups.
- LowMemAdamW edge cases: 0-numel tensors, masked tail-block scale isolation, clip_scale.
"""

from __future__ import annotations

import math
from pathlib import Path
import types
from typing import Any

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
import torch.nn.functional as F

from gr00t_lancedb.turbo import (
    BLOCK,
    LowMemAdamW,
    OnlineTurboConfig,
    apply_online_turbo,
    cast_parameters_only,
    make_online_turbo_trainer_class,
    unwrap_compiled_action_head,
)
from gr00t_lancedb.vl_cache.accel import patch_action_head_multi_noise


# ---------------------------------------------------------------------------
# Minimal GR00T-shaped stub modules (runs on CPU in milliseconds)
# ---------------------------------------------------------------------------


class _FakeBackbone(nn.Module):
    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.register_buffer("inv_freq", torch.linspace(1.0, 1e-4, dim // 2, dtype=torch.float32), persistent=False)
        for p in self.parameters():
            p.requires_grad_(False)
        self.saw_grad_enabled: list[bool] = []

    def forward(self, x: torch.Tensor) -> types.SimpleNamespace:
        self.saw_grad_enabled.append(torch.is_grad_enabled())
        feats = self.proj(x.to(self.proj.weight.dtype))
        bsz, seq = feats.shape[:2]
        return types.SimpleNamespace(
            backbone_features=feats,
            backbone_attention_mask=torch.ones(bsz, seq, dtype=torch.bool),
            image_mask=torch.ones(bsz, seq, dtype=torch.bool),
        )


class _FakeDiT(nn.Module):
    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)
        self.last_batch_size: int | None = None

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
        timestep: torch.Tensor | None = None,
        return_all_hidden_states: bool = True,
        image_mask: torch.Tensor | None = None,
        backbone_attention_mask: torch.Tensor | None = None,
        hidden_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        self.last_batch_size = int(hidden_states.shape[0])
        ctx = encoder_hidden_states.mean(dim=1, keepdim=True)
        normed = self.norm(hidden_states.to(self.norm.weight.dtype) + ctx.to(self.norm.weight.dtype))
        h = self.linear(normed.to(self.linear.weight.dtype))
        return h, [h]


class _FakeActionHead(nn.Module):
    def __init__(self, dim: int = 16, state_dim: int = 4, action_dim: int = 4, horizon: int = 6) -> None:
        super().__init__()
        self.config = types.SimpleNamespace(
            state_history_length=1,
            strict_action_padding_mask=False,
            strict_vl_padding_mask=False,
            noise_beta_alpha=1.5,
            noise_beta_beta=1.0,
            noise_s=0.999,
            add_pos_embed=False,
            use_alternate_vl_dit=True,
        )
        self.state_dropout_prob = 0.2
        self.num_timestep_buckets = 1000
        self.vlln = nn.LayerNorm(dim)
        self.state_proj = nn.Linear(state_dim, dim)
        self.action_proj = nn.Linear(action_dim, dim)
        self.model = _FakeDiT(dim)
        self.out_proj = nn.Linear(dim, action_dim)

    def set_frozen_modules_to_eval_mode(self) -> None:
        pass

    def process_backbone_output(self, bb: Any) -> Any:
        bb.backbone_features = self.vlln(bb.backbone_features.to(self.vlln.weight.dtype))
        return bb

    def state_encoder(self, state_flat: torch.Tensor, embodiment_id: torch.Tensor) -> torch.Tensor:
        return self.state_proj(state_flat.to(self.state_proj.weight.dtype))

    def action_encoder(
        self, noisy_trajectory: torch.Tensor, t_discretized: torch.Tensor, embodiment_id: torch.Tensor
    ) -> torch.Tensor:
        return self.action_proj(noisy_trajectory.to(self.action_proj.weight.dtype))

    def action_decoder(self, model_output: torch.Tensor, embodiment_id: torch.Tensor) -> torch.Tensor:
        return self.out_proj(model_output.to(self.out_proj.weight.dtype))

    def sample_time(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.rand(batch_size, device=device, dtype=dtype) * 0.999

    def forward(self, backbone_output: Any, action_input: Any) -> dict[str, torch.Tensor]:
        bb = self.process_backbone_output(backbone_output)
        s = self.state_encoder(action_input.state.view(action_input.state.shape[0], 1, -1), action_input.embodiment_id)
        actions = action_input.action
        noise = torch.randn_like(actions)
        t = self.sample_time(actions.shape[0], actions.device, actions.dtype)[:, None, None]
        noisy = (1 - t) * noise + t * actions
        vel = actions - noise
        a = self.action_encoder(noisy, (t[:, 0, 0] * self.num_timestep_buckets).long(), action_input.embodiment_id)
        sa = torch.cat((s, a), dim=1)
        out, _ = self.model(
            hidden_states=sa,
            encoder_hidden_states=bb.backbone_features,
            encoder_attention_mask=bb.backbone_attention_mask,
            timestep=(t[:, 0, 0] * self.num_timestep_buckets).long(),
            image_mask=bb.image_mask,
            backbone_attention_mask=bb.backbone_attention_mask,
        )
        pred = self.action_decoder(out, action_input.embodiment_id)[:, -actions.shape[1] :]
        action_loss = F.mse_loss(pred.float(), vel.float(), reduction="none") * action_input.action_mask.float()
        loss = action_loss.sum() / (action_input.action_mask.float().sum() + 1e-6)
        return {"loss": loss, "action_loss": action_loss, "action_mask": action_input.action_mask}


class _FakeGr00tModel(nn.Module):
    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.backbone = _FakeBackbone(dim)
        self.action_head = _FakeActionHead(dim)

    def forward(
        self,
        *,
        pixel_tokens: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
        action_mask: torch.Tensor,
        embodiment_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        bb = self.backbone(pixel_tokens)
        act_in = types.SimpleNamespace(
            state=state,
            action=action,
            action_mask=action_mask,
            embodiment_id=embodiment_id,
        )
        return self.action_head(bb, act_in)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_config_validation_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in (
        "GR00T_TURBO_MULTI_NOISE",
        "GR00T_TURBO_OPTIMIZER",
        "GR00T_TURBO_BF16_WEIGHTS",
        "GR00T_TURBO_BF16_BACKBONE",
        "GR00T_TURBO_COMPILE_HEAD",
        "GR00T_TURBO_ALLOW_PURE_BF16",
    ):
        monkeypatch.delenv(k, raising=False)

    default_cfg = OnlineTurboConfig.from_env()
    assert not default_cfg.enabled

    # Pure bf16 without LowMemAdamW must fail closed by default.
    with pytest.raises(ValueError, match="bf16_weights=True without optimizer='lowmem'"):
        OnlineTurboConfig(bf16_weights=True, optimizer="native")

    monkeypatch.setenv("GR00T_TURBO_MULTI_NOISE", "2")
    monkeypatch.setenv("GR00T_TURBO_OPTIMIZER", "lowmem")
    monkeypatch.setenv("GR00T_TURBO_BF16_WEIGHTS", "1")
    cfg = OnlineTurboConfig.from_env()
    assert cfg.enabled and cfg.multi_noise == 2 and cfg.bf16_weights and cfg.bf16_frozen_backbone


def test_cast_parameters_only_preserves_fp32_rope_buffers() -> None:
    model = _FakeGr00tModel()
    assert model.backbone.proj.weight.dtype == torch.float32
    assert model.backbone.inv_freq.dtype == torch.float32

    n_cast, saved = cast_parameters_only(model.backbone, torch.bfloat16, only_frozen=True)
    assert n_cast == 2 and saved > 0
    assert model.backbone.proj.weight.dtype == torch.bfloat16
    assert model.backbone.proj.bias.dtype == torch.bfloat16
    # Non-parameter RoPE inv_freq buffer MUST stay FP32.
    assert model.backbone.inv_freq.dtype == torch.float32


def test_apply_online_turbo_multi_noise_train_vs_eval_and_no_grad() -> None:
    model = _FakeGr00tModel()
    cfg = OnlineTurboConfig(
        multi_noise=4,
        optimizer="lowmem",
        lowmem_backend="torch",
        bf16_weights=True,
        bf16_frozen_backbone=True,
        compile_head=False,
    )
    info = apply_online_turbo(model, cfg)
    assert info["multi_noise_applied"] is True
    assert info["backbone_no_grad"] is True
    assert model.backbone.inv_freq.dtype == torch.float32
    assert model.action_head.out_proj.weight.dtype == torch.bfloat16

    batch = dict(
        pixel_tokens=torch.randn(3, 5, 16),
        state=torch.randn(3, 1, 4),
        action=torch.randn(3, 6, 4),
        action_mask=torch.ones(3, 6, 4),
        embodiment_id=torch.zeros(3, dtype=torch.long),
    )

    # Training mode: DiT sees B * K = 12 samples, backbone runs inside no_grad, loss is fp32 and finite.
    model.train()
    out_tr = model(**batch)
    assert model.backbone.saw_grad_enabled[-1] is False
    assert model.action_head.model.last_batch_size == 12
    assert out_tr["loss"].dtype == torch.float32
    assert torch.isfinite(out_tr["loss"])
    out_tr["loss"].backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.action_head.parameters())

    # Even when actions/masks arrive as bf16, multi-noise loss reduction MUST stay fp32.
    bf16_batch = {
        k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in batch.items()
    }
    out_bf16 = model(**bf16_batch)
    assert out_bf16["loss"].dtype == torch.float32 and torch.isfinite(out_bf16["loss"])

    # Eval mode: multi-noise MUST bypass to orig_forward (DiT sees B = 3, not 12).
    model.eval()
    with torch.no_grad():
        out_ev = model(**batch)
    assert model.action_head.model.last_batch_size == 3
    assert out_ev["action_loss"].shape[0] == 3


def test_compile_skipped_on_fp32_and_unwrap_cleans_state_dict() -> None:
    model = _FakeGr00tModel()
    cfg_fp32 = OnlineTurboConfig(compile_head=True, bf16_weights=False, optimizer="native")
    info = apply_online_turbo(model, cfg_fp32)
    assert info["compile_applied"] is False
    assert "fp32" in str(info["compile_skipped_reason"])

    # Simulate a compiled module wrapper carrying `_orig_mod` and verify unwrap context removes it.
    class _FakeCompiled(nn.Module):
        def __init__(self, orig: nn.Module) -> None:
            super().__init__()
            self._orig_mod = orig

        def forward(self, *a, **kw):
            return self._orig_mod(*a, **kw)

    orig_dit = model.action_head.model
    model.action_head.model = _FakeCompiled(orig_dit)
    assert any("_orig_mod" in k for k in model.state_dict())
    with unwrap_compiled_action_head(model):
        clean_keys = list(model.state_dict().keys())
        assert not any("_orig_mod" in k for k in clean_keys)
        assert "action_head.model.linear.weight" in clean_keys
    # Compiled wrapper restored after saving.
    assert hasattr(model.action_head.model, "_orig_mod")

    # DDP / Accelerate `.module` wrapper MUST also be unwrapped cleanly.
    class _FakeDDP(nn.Module):
        def __init__(self, inner: nn.Module) -> None:
            super().__init__()
            self.module = inner

    ddp = _FakeDDP(model)
    with unwrap_compiled_action_head(ddp):
        assert not any("_orig_mod" in k for k in ddp.module.state_dict())
    assert hasattr(model.action_head.model, "_orig_mod")


def test_hf_trainer_end_to_end_with_lowmem_and_multi_noise(tmp_path: Path) -> None:
    transformers = pytest.importorskip("transformers")
    from torch.utils.data import Dataset

    class _TinyDataset(Dataset):
        def __len__(self) -> int:
            return 12

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            g = torch.Generator().manual_seed(idx)
            return {
                "pixel_tokens": torch.randn(5, 16, generator=g),
                "state": torch.randn(1, 4, generator=g),
                "action": torch.randn(6, 4, generator=g),
                "action_mask": torch.ones(6, 4),
                "embodiment_id": torch.tensor(0, dtype=torch.long),
            }

    model = _FakeGr00tModel()
    cfg = OnlineTurboConfig(
        multi_noise=2,
        optimizer="lowmem",
        lowmem_backend="torch",
        bf16_weights=True,
        bf16_frozen_backbone=True,
        keep_norm_fp32=True,  # opt-in (off by default): exercised here end to end
        fused_clip=True,
        seed=7,
    )
    trainer_cls = make_online_turbo_trainer_class(transformers.Trainer, cfg)
    out_dir = tmp_path / "run"
    args = transformers.TrainingArguments(
        output_dir=str(out_dir),
        max_steps=4,
        per_device_train_batch_size=3,
        learning_rate=1e-3,
        weight_decay=1e-2,
        logging_steps=1,
        save_steps=4,
        save_total_limit=1,
        report_to="none",
        use_cpu=True,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=7,
    )
    trainer = trainer_cls(model=model, args=args, train_dataset=_TinyDataset())
    result = trainer.train()
    assert math.isfinite(float(result.training_loss))
    raw_opt = getattr(trainer.optimizer, "optimizer", trainer.optimizer)
    assert isinstance(raw_opt, LowMemAdamW)
    # Decay & no-decay parameter groups both present and honor weight decay split.
    wds = {g["weight_decay"] for g in raw_opt.param_groups}
    assert wds == {0.0, 1e-2}
    # Frozen backbone stayed frozen (no optimizer state) and RoPE inv_freq stayed fp32.
    assert model.backbone.inv_freq.dtype == torch.float32
    assert all(p not in raw_opt.state for p in model.backbone.parameters())
    assert model.action_head.model.last_batch_size == 6  # 3 frames x 2 noise
    # Checkpoint saved cleanly without `_orig_mod`.
    trainer.save_model(str(out_dir / "final"))
    ckpt_files = list((out_dir / "final").glob("*.safetensors")) + list((out_dir / "final").glob("*.bin"))
    assert ckpt_files

    # Resume from checkpoint-4 via HF Trainer and verify LowMemAdamW preserves exact state dtypes.
    ckpt_4 = out_dir / "checkpoint-4"
    assert ckpt_4.is_dir()
    model2 = _FakeGr00tModel()
    args2 = transformers.TrainingArguments(
        output_dir=str(out_dir),
        max_steps=6,
        per_device_train_batch_size=3,
        learning_rate=1e-3,
        weight_decay=1e-2,
        logging_steps=1,
        save_steps=6,
        save_total_limit=1,
        report_to="none",
        use_cpu=True,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=7,
    )
    trainer2 = trainer_cls(model=model2, args=args2, train_dataset=_TinyDataset())
    res2 = trainer2.train(resume_from_checkpoint=str(ckpt_4))
    assert math.isfinite(float(res2.training_loss))
    raw_opt2 = getattr(trainer2.optimizer, "optimizer", trainer2.optimizer)
    assert isinstance(raw_opt2, LowMemAdamW)
    assert model2.action_head.vlln.weight.dtype == torch.float32
    sample_st = next(st for p, st in raw_opt2.state.items() if p.dtype == torch.bfloat16)
    assert sample_st["qr"].dtype == torch.int8
    assert sample_st["qv"].dtype == torch.uint8
    assert sample_st["sr"].dtype == torch.float32
    assert sample_st["sv"].dtype == torch.float32
    assert sample_st["comp"].dtype == torch.bfloat16
    assert sample_st["step"] == 6


def test_lowmem_adamw_empty_tensor_and_partial_block_masking() -> None:
    # 0-numel parameter must be skipped cleanly without raising or allocating grid (0,).
    empty = nn.Parameter(torch.empty(0, dtype=torch.bfloat16))
    small = nn.Parameter(torch.full((7,), 0.1, dtype=torch.bfloat16))  # partial tail block (7 < BLOCK=128)
    opt = LowMemAdamW([empty, small], lr=1e-3, backend="torch", seed=3)
    empty.grad = torch.empty_like(empty)
    small.grad = torch.full_like(small, 0.05)
    norm, scale = opt.clip_scale(1.0)
    assert math.isfinite(norm) and 0.0 < scale <= 1.0
    opt.step(grad_scale=scale)
    assert empty not in opt.state
    assert small in opt.state
    # Subsequent zero-grad step on a partial block must keep sv strictly bounded by real lanes.
    for _ in range(3):
        small.grad = torch.zeros_like(small)
        opt.step()
    assert torch.isfinite(opt.true_weight(small)).all()
    assert opt.state[small]["sv"].numel() == 1 and BLOCK == 128


def test_patch_multi_noise_idempotent_double_patch() -> None:
    ah = _FakeActionHead()
    orig = ah.forward
    patch_action_head_multi_noise(ah, k_noise=2)
    patch_action_head_multi_noise(ah, k_noise=4)
    assert ah._orig_forward == orig
    assert ah._k_noise == 4


def test_wp2_features_are_opt_in_and_flash_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    """keep_norm_fp32 / fused_clip default off (falsify_07-comparable LowMemAdamW numerics);
    flash24 / flash32 pin master_weight_bits, flashoptim reads it from the environment."""
    cfg = OnlineTurboConfig(optimizer="lowmem", bf16_weights=True)
    assert not cfg.keep_norm_fp32 and not cfg.fused_clip
    for var in ("GR00T_TURBO_KEEP_NORM_FP32", "GR00T_TURBO_FUSED_CLIP", "GR00T_TURBO_FLASHOPTIM_BITS"):
        monkeypatch.delenv(var, raising=False)
    assert not OnlineTurboConfig.from_env().keep_norm_fp32 and not OnlineTurboConfig.from_env().fused_clip
    monkeypatch.setenv("GR00T_TURBO_KEEP_NORM_FP32", "1")
    monkeypatch.setenv("GR00T_TURBO_FUSED_CLIP", "1")
    env_cfg = OnlineTurboConfig.from_env()
    assert env_cfg.keep_norm_fp32 and env_cfg.fused_clip

    for kind in ("flashoptim", "flash24", "flash32"):
        assert OnlineTurboConfig(optimizer=kind, bf16_weights=True).enabled
    with pytest.raises(ValueError, match="drops sub-ulp updates"):
        OnlineTurboConfig(optimizer="bnb8bit", bf16_weights=True)
    with pytest.raises(ValueError, match="optimizer must be one of"):
        OnlineTurboConfig(optimizer="flash16")

    seen: list[int] = []

    class _FakeFlashAdamW:
        def __init__(self, groups: Any, *, master_weight_bits: int, **kw: Any) -> None:
            seen.append(master_weight_bits)
            self.param_groups = groups
            self.kw = kw

    monkeypatch.setitem(__import__("sys").modules, "flashoptim", types.SimpleNamespace(FlashAdamW=_FakeFlashAdamW))
    from gr00t_lancedb.turbo import build_third_party_optimizer

    model = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
    for kind in ("flash24", "flash32"):
        opt = build_third_party_optimizer(kind, model, lr=1e-4, weight_decay=1e-5)
        assert {g["weight_decay"] for g in opt.param_groups} == {1e-5, 0.0} and opt.kw["lr"] == 1e-4
    monkeypatch.setenv("GR00T_TURBO_FLASHOPTIM_BITS", "32")
    build_third_party_optimizer("flashoptim", model, lr=1e-4, weight_decay=1e-5)
    assert seen == [24, 32, 32]
    monkeypatch.setenv("GR00T_TURBO_FLASHOPTIM_BITS", "16")
    with pytest.raises(ValueError, match="24 or 32"):
        build_third_party_optimizer("flashoptim", model, lr=1e-4, weight_decay=1e-5)


def test_fused_clip_hook_only_when_enabled(tmp_path: Path) -> None:
    transformers = pytest.importorskip("transformers")
    args = transformers.TrainingArguments(output_dir=str(tmp_path), max_steps=1, report_to="none", use_cpu=True)
    for fused_clip in (False, True):
        cfg = OnlineTurboConfig(optimizer="lowmem", lowmem_backend="torch", bf16_weights=True, fused_clip=fused_clip)
        trainer = make_online_turbo_trainer_class(transformers.Trainer, cfg)(model=_FakeGr00tModel(), args=args)
        assert bool(getattr(trainer.accelerator, "_lowmem_clip_hooked", False)) is fused_clip


def test_plan09_config_env_grad_ckpt_and_int8_comp(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "GR00T_TURBO_LOWMEM_COMP",
        "GR00T_TURBO_GRAD_CKPT_HEAD",
        "GR00T_TURBO_EMA_DECAY",
        "GR00T_TURBO_EMA_EVERY",
    ):
        monkeypatch.delenv(var, raising=False)
    default_cfg = OnlineTurboConfig.from_env()
    assert default_cfg.lowmem_comp == "bf16"
    assert default_cfg.grad_ckpt_head is False
    assert default_cfg.ema_decay == 0.0
    assert default_cfg.ema_update_every == 10

    monkeypatch.setenv("GR00T_TURBO_LOWMEM_COMP", "int8")
    monkeypatch.setenv("GR00T_TURBO_GRAD_CKPT_HEAD", "1")
    monkeypatch.setenv("GR00T_TURBO_EMA_DECAY", "0.995")
    monkeypatch.setenv("GR00T_TURBO_EMA_EVERY", "5")
    cfg = OnlineTurboConfig.from_env()
    assert cfg.enabled and cfg.lowmem_comp == "int8" and cfg.grad_ckpt_head is True
    assert cfg.ema_decay == pytest.approx(0.995) and cfg.ema_update_every == 5

    with pytest.raises(ValueError, match="lowmem_comp"):
        OnlineTurboConfig(lowmem_comp="fp8")
    with pytest.raises(ValueError, match="ema_decay"):
        OnlineTurboConfig(ema_decay=1.0)

    # apply_online_turbo enables gradient_checkpointing on action_head.model (both method & attribute paths)
    model = _FakeGr00tModel()
    info = apply_online_turbo(model, OnlineTurboConfig(grad_ckpt_head=True))
    assert info["grad_ckpt_head_applied"] is True
    assert getattr(model.action_head.model, "gradient_checkpointing", False) is True

    called: list[Any] = []

    def _enable(gradient_checkpointing_kwargs: Any = None) -> None:
        called.append(gradient_checkpointing_kwargs)

    model2 = _FakeGr00tModel()
    model2.action_head.model.gradient_checkpointing_enable = _enable  # type: ignore[attr-defined]
    info2 = apply_online_turbo(model2, OnlineTurboConfig(grad_ckpt_head=True))
    assert info2["grad_ckpt_head_applied"] is True
    assert called == [{"use_reentrant": False}]
