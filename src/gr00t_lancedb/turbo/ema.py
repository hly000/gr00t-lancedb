# SPDX-License-Identifier: Apache-2.0
"""Zero-VRAM CPU shadow EMA for single-GPU VLA fine-tuning.

Maintains an ``fp32`` exponential moving average of trainable action-head parameters in
CPU RAM (0.00 GB GPU VRAM). When paired with :class:`LowMemAdamW`, updates read
``optimizer.true_weight(p)`` (exact compensated weight) sequentially one tensor at a time
to avoid peak GPU allocation spikes, and :meth:`swap_into_model` backs up live weights to
CPU in their native dtype (``bf16``) to keep host RAM overhead minimal.
"""

from __future__ import annotations

from contextlib import contextmanager
import logging
from typing import Any, Iterator

import torch
import torch.nn as nn

logger = logging.getLogger("gr00t_lancedb.turbo.ema")


def _unwrap_model(model: nn.Module) -> nn.Module:
    base = model
    while hasattr(base, "module") and isinstance(base.module, nn.Module):
        base = base.module
    return base


class CpuShadowEMA:
    """CPU-resident ``fp32`` EMA shadow weights for trainable model parameters."""

    def __init__(
        self,
        model: nn.Module,
        *,
        decay: float = 0.995,
        update_every: int = 10,
        warmup_steps: int = 10,
        target_submodule: str | None = "action_head",
    ) -> None:
        if not (0.0 < float(decay) < 1.0):
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        if int(update_every) < 1:
            raise ValueError(f"update_every must be >= 1, got {update_every}")
        if int(warmup_steps) < 1:
            raise ValueError(f"warmup_steps must be >= 1, got {warmup_steps}")

        self.decay = float(decay)
        self.update_every = int(update_every)
        self.warmup_steps = int(warmup_steps)
        self.num_updates = 0
        self.last_step = 0

        base = _unwrap_model(model)
        root = getattr(base, target_submodule, None) if target_submodule else None
        if not isinstance(root, nn.Module):
            root = base

        self._params: list[tuple[str, nn.Parameter]] = []
        self._shadow: dict[str, torch.Tensor] = {}
        seen: set[int] = set()
        for name, p in root.named_parameters():
            pid = id(p)
            if pid in seen or not p.requires_grad or not p.is_floating_point():
                continue
            seen.add(pid)
            self._params.append((name, p))
            self._shadow[name] = p.detach().to(device="cpu", dtype=torch.float32).clone().contiguous()

        shadow_gb = self.shadow_bytes() / 1024**3
        logger.info(
            "[cpu-ema] initialized %d tensors (%.2f GiB CPU fp32), decay=%.5f, update_every=%d",
            len(self._params),
            shadow_gb,
            self.decay,
            self.update_every,
        )
        try:
            import psutil

            avail_gb = psutil.virtual_memory().available / 1024**3
            if avail_gb < 12.0:
                logger.warning(
                    "[cpu-ema] low available host RAM after shadow init: %.2f GiB (< 12 GiB)",
                    avail_gb,
                )
        except Exception:
            pass

    def shadow_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self._shadow.values())

    def effective_decay(self, step: int) -> float:
        d = min(self.decay, (1.0 + float(step)) / (float(self.warmup_steps) + float(step)))
        return float(d ** self.update_every)

    @torch.no_grad()
    def update(self, step: int, optimizer: Any = None) -> bool:
        """Update CPU shadow weights when ``step > 0`` and ``step % update_every == 0``."""
        step_i = int(step)
        if step_i <= 0 or (step_i % self.update_every) != 0:
            return False
        raw_opt = getattr(optimizer, "optimizer", optimizer) if optimizer is not None else None
        true_weight_fn = getattr(raw_opt, "true_weight", None)
        use_true_weight = callable(true_weight_fn)

        alpha = self.effective_decay(step_i)
        one_minus_alpha = 1.0 - alpha

        for name, p in self._params:
            w_src = true_weight_fn(p) if use_true_weight else p.detach()
            w_cpu = w_src.detach().to(device="cpu", dtype=torch.float32)
            self._shadow[name].lerp_(w_cpu, one_minus_alpha)
            del w_cpu

        self.num_updates += 1
        self.last_step = step_i
        return True

    @contextmanager
    def swap_into_model(self) -> Iterator[None]:
        """Temporarily load EMA shadow weights into the live model (native-dtype CPU backup)."""
        backup: dict[str, torch.Tensor] = {}
        try:
            with torch.no_grad():
                for name, p in self._params:
                    backup[name] = p.detach().to(device="cpu").clone()
                    p.data.copy_(self._shadow[name].to(device=p.device, dtype=p.dtype))
            yield
        finally:
            with torch.no_grad():
                for name, p in self._params:
                    if name in backup:
                        p.data.copy_(backup[name].to(device=p.device, dtype=p.dtype))
            backup.clear()
