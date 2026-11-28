# SPDX-License-Identifier: Apache-2.0
"""CPU unit & integration tests for Zero-VRAM CpuShadowEMA."""

from __future__ import annotations

import math
from pathlib import Path
import types

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
import torch.nn.functional as F

from gr00t_lancedb.turbo import (
    CpuShadowEMA,
    LowMemAdamW,
    OnlineTurboConfig,
    make_online_turbo_trainer_class,
)


class _ToyGr00tModel(nn.Module):
    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.backbone = nn.Linear(dim, dim)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.action_head = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            h = self.backbone(x.to(self.backbone.weight.dtype))
        pred = self.action_head[0](h.to(self.action_head[0].weight.dtype))
        pred = self.action_head[1](pred.to(self.action_head[1].weight.dtype))
        pred = self.action_head[2](pred.to(self.action_head[2].weight.dtype))
        return {"loss": F.mse_loss(pred.float(), y.float())}


def test_cpu_shadow_ema_reads_true_weight_warmup_and_bit_exact_restore() -> None:
    torch.manual_seed(0)
    model = _ToyGr00tModel(dim=32)
    for p in model.action_head.parameters():
        p.data = p.data.to(torch.bfloat16)

    opt = LowMemAdamW(model.action_head.parameters(), lr=1e-4, backend="torch", comp_dtype="int8", seed=5)
    ema = CpuShadowEMA(model, decay=0.995, update_every=2, warmup_steps=10)

    # Frozen backbone is excluded; shadow lives on CPU in fp32
    assert len(ema._params) == len(list(model.action_head.parameters()))
    assert all(t.device.type == "cpu" and t.dtype == torch.float32 for t in ema._shadow.values())
    assert ema.shadow_bytes() == sum(p.numel() * 4 for p in model.action_head.parameters())

    # Warmup decay formula check
    expected_step2 = min(0.995, (1.0 + 2.0) / (10.0 + 2.0)) ** 2
    assert ema.effective_decay(2) == pytest.approx(expected_step2, rel=1e-6)
    assert ema.effective_decay(10000) == pytest.approx(0.995**2, rel=1e-6)

    # Step 1 should not trigger update (update_every=2)
    x = torch.randn(4, 32)
    y = torch.randn(4, 32)
    for step in range(1, 5):
        opt.zero_grad()
        model(x, y)["loss"].backward()
        opt.step()
        updated = ema.update(step, optimizer=opt)
        assert updated is (step % 2 == 0)

    assert ema.num_updates == 2 and ema.last_step == 4

    # Snapshot raw weights bit-for-bit before swap_into_model
    raw_bits = {name: p.detach().view(torch.int16).clone() for name, p in ema._params}
    with ema.swap_into_model():
        # Inside context, live model weights equal rounded EMA shadow
        for name, p in ema._params:
            expected_bf16 = ema._shadow[name].to(dtype=p.dtype)
            assert torch.equal(p.detach().view(torch.int16), expected_bf16.view(torch.int16))
    # Outside context, live weights are bit-exact restored
    for name, p in ema._params:
        assert torch.equal(p.detach().view(torch.int16), raw_bits[name])


def test_hf_trainer_saves_ema_checkpoint_with_int8_lowmem(tmp_path: Path) -> None:
    transformers = pytest.importorskip("transformers")
    from torch.utils.data import Dataset

    class _DS(Dataset):
        def __len__(self) -> int:
            return 12

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            g = torch.Generator().manual_seed(idx)
            return {"x": torch.randn(16, generator=g), "y": torch.randn(16, generator=g)}

    model = _ToyGr00tModel(dim=16)
    cfg = OnlineTurboConfig(
        optimizer="lowmem",
        lowmem_backend="torch",
        lowmem_comp="int8",
        bf16_weights=True,
        bf16_frozen_backbone=True,
        keep_norm_fp32=True,
        fused_clip=True,
        ema_decay=0.99,
        ema_update_every=2,
        seed=11,
    )
    trainer_cls = make_online_turbo_trainer_class(transformers.Trainer, cfg)
    out_dir = tmp_path / "ema_run"
    args = transformers.TrainingArguments(
        output_dir=str(out_dir),
        max_steps=4,
        per_device_train_batch_size=3,
        learning_rate=1e-2,
        weight_decay=1e-2,
        logging_steps=1,
        save_steps=4,
        save_total_limit=1,
        report_to="none",
        use_cpu=True,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=11,
    )
    trainer = trainer_cls(model=model, args=args, train_dataset=_DS())
    res = trainer.train()
    assert math.isfinite(float(res.training_loss))

    raw_opt = getattr(trainer.optimizer, "optimizer", trainer.optimizer)
    assert isinstance(raw_opt, LowMemAdamW) and raw_opt.comp_dtype == "int8"
    bf16_state = next(st for p, st in raw_opt.state.items() if p.dtype == torch.bfloat16)
    assert bf16_state["comp"].dtype == torch.int8

    assert trainer._cpu_ema is not None and trainer._cpu_ema.num_updates == 2
    raw_ckpt = out_dir / "checkpoint-4"
    ema_ckpt = out_dir / "checkpoint-4-ema"
    assert raw_ckpt.is_dir() and ema_ckpt.is_dir()
    ema_weights = list(ema_ckpt.glob("*.safetensors")) + list(ema_ckpt.glob("*.bin"))
    assert ema_weights
    # Optimizer state must NOT be duplicated into the EMA checkpoint
    assert not (ema_ckpt / "optimizer.pt").exists()
