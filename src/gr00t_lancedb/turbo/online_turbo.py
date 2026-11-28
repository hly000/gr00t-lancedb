# SPDX-License-Identifier: Apache-2.0
"""Online-only single-GPU acceleration for GR00T N1.7 fine-tuning (no offline feature cache).

Combines two orthogonal, zero-disk-overhead wins while preserving 100 % of GR00T's native
online pipeline (per-step RandomCrop + ColorJitter, real Qwen3-VL padding states, fp32
flow-matching targets and loss):

1. **Online Stratified Multi-Noise Flow Matching** (``multi_noise=K``):
   Runs the frozen 2B VLM backbone ONCE on ``B`` freshly augmented images, then evaluates
   the lightweight DiT action head on ``K`` stratified Beta timesteps + independent Gaussian
   noises per observation (``effective_fm_batch = B * K``).
2. **24 GB Memory Budget via ``LowMemAdamW`` + Parameter-Only BF16 Casting**:
   Casts frozen backbone parameters (4 GB saved) and trainable action-head parameters
   (3.2 GB weights + 3.2 GB grads + 6.5 GB optimizer states saved) to ``bfloat16`` while
   *never* touching non-parameter buffers (such as Qwen3-VL's analytic FP32 RoPE
   ``inv_freq``). Kahan compensation in :class:`LowMemAdamW` keeps sub-ulp updates intact.

Everything is opt-in via ``GR00T_TURBO_*`` environment variables; when disabled, training
is byte-identical to stock GR00T.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import logging
import os
from typing import Any, Iterator

import torch
import torch.nn as nn

from .lowmem_adamw import LowMemAdamW, triton_available

logger = logging.getLogger("gr00t_lancedb.turbo.online")

_NO_DECAY_SUBSTRINGS = ("bias", "layernorm", "layer_norm", "rmsnorm", "rms_norm", "vlln", ".norm")


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    val = raw.strip().lower()
    if val in ("1", "true", "yes", "on"):
        return True
    if val in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be boolean (0/1/true/false), got {raw!r}")


@dataclass(frozen=True)
class OnlineTurboConfig:
    """Configuration for online GR00T N1.7 acceleration (no offline VL-cache)."""

    multi_noise: int = 1
    stratified: bool = True
    optimizer: str = "native"  # "native" | "lowmem" | "bnb8bit" (baseline: bitsandbytes AdamW8bit)
    lowmem_backend: str = "auto"  # "auto" | "triton" | "torch"
    lowmem_fused: str = "auto"  # "auto" | "on" | "off": one kernel launch for all tensors
    bf16_weights: bool = False
    bf16_frozen_backbone: bool = False
    compile_head: bool = False
    frozen_backbone_no_grad: bool = True
    allow_pure_bf16: bool = False
    seed: int = 42

    def __post_init__(self) -> None:
        if self.multi_noise < 1:
            raise ValueError(f"multi_noise must be >= 1, got {self.multi_noise}")
        if self.optimizer not in ("native", "lowmem", "bnb8bit"):
            raise ValueError(f"optimizer must be 'native', 'lowmem' or 'bnb8bit', got {self.optimizer!r}")
        if self.lowmem_fused not in ("auto", "on", "off"):
            raise ValueError(f"lowmem_fused must be 'auto'|'on'|'off', got {self.lowmem_fused!r}")
        if self.lowmem_backend not in ("auto", "triton", "torch"):
            raise ValueError(f"lowmem_backend must be 'auto'|'triton'|'torch', got {self.lowmem_backend!r}")
        if self.bf16_weights and self.optimizer != "lowmem" and not self.allow_pure_bf16:
            raise ValueError(
                "bf16_weights=True without optimizer='lowmem' drops sub-ulp updates in pure bf16 "
                "(known to stall GR00T training). Set GR00T_TURBO_OPTIMIZER=lowmem, or pass "
                "GR00T_TURBO_ALLOW_PURE_BF16=1 only for intentional ablation."
            )

    @property
    def enabled(self) -> bool:
        return (
            self.multi_noise > 1
            or self.optimizer != "native"
            or self.bf16_weights
            or self.bf16_frozen_backbone
            or self.compile_head
        )

    @classmethod
    def from_env(cls, *, default_seed: int = 42) -> "OnlineTurboConfig":
        bf16_weights = _env_bool("GR00T_TURBO_BF16_WEIGHTS", False)
        bf16_backbone_raw = os.environ.get("GR00T_TURBO_BF16_BACKBONE")
        bf16_backbone = bf16_weights if bf16_backbone_raw is None else _env_bool("GR00T_TURBO_BF16_BACKBONE", False)
        return cls(
            multi_noise=int(os.environ.get("GR00T_TURBO_MULTI_NOISE", "1")),
            stratified=_env_bool("GR00T_TURBO_STRATIFIED", True),
            optimizer=os.environ.get("GR00T_TURBO_OPTIMIZER", "native").strip().lower(),
            lowmem_backend=os.environ.get("GR00T_TURBO_LOWMEM_BACKEND", "auto").strip().lower(),
            lowmem_fused=os.environ.get("GR00T_TURBO_LOWMEM_FUSED", "auto").strip().lower(),
            bf16_weights=bf16_weights,
            bf16_frozen_backbone=bf16_backbone,
            compile_head=_env_bool("GR00T_TURBO_COMPILE_HEAD", False),
            frozen_backbone_no_grad=_env_bool("GR00T_TURBO_FROZEN_BACKBONE_NO_GRAD", True),
            allow_pure_bf16=_env_bool("GR00T_TURBO_ALLOW_PURE_BF16", False),
            seed=int(os.environ.get("GR00T_TURBO_SEED", str(default_seed))),
        )


def cast_parameters_only(
    module: nn.Module,
    dtype: torch.dtype,
    *,
    only_frozen: bool = False,
    only_trainable: bool = False,
) -> tuple[int, int]:
    """Cast floating-point ``nn.Parameter`` tensors in ``module`` to ``dtype`` in place.

    Crucially does NOT touch non-parameter buffers (e.g. Qwen3-VL's non-persistent FP32
    RoPE ``inv_freq`` buffers), avoiding silent positional-frequency truncation.
    Returns ``(tensors_cast, bytes_saved)``.
    """
    if only_frozen and only_trainable:
        raise ValueError("only_frozen and only_trainable are mutually exclusive")
    cast_count = 0
    bytes_saved = 0
    seen: set[int] = set()
    for p in module.parameters():
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)
        if not p.is_floating_point() or p.dtype == dtype:
            continue
        if only_frozen and p.requires_grad:
            continue
        if only_trainable and not p.requires_grad:
            continue
        old_bytes = p.numel() * p.element_size()
        p.data = p.data.to(dtype=dtype)
        new_bytes = p.numel() * p.element_size()
        cast_count += 1
        bytes_saved += old_bytes - new_bytes
    return cast_count, bytes_saved


def _wrap_frozen_backbone_no_grad(backbone: nn.Module) -> bool:
    """Wrap ``backbone.forward`` in ``torch.no_grad()`` when all backbone params are frozen."""
    if any(p.requires_grad for p in backbone.parameters()):
        return False
    if getattr(backbone, "_turbo_no_grad_wrapped", False):
        return True
    orig_forward = backbone.forward

    def _no_grad_forward(*args: Any, **kwargs: Any) -> Any:
        with torch.no_grad():
            return orig_forward(*args, **kwargs)

    backbone._turbo_orig_forward = orig_forward
    backbone._turbo_no_grad_wrapped = True
    backbone.forward = _no_grad_forward  # type: ignore[method-assign]
    return True


def _is_no_decay_param(name: str, param: nn.Parameter) -> bool:
    if param.ndim < 2:
        return True
    low = name.lower()
    return any(tok in low for tok in _NO_DECAY_SUBSTRINGS)


def _decay_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """HF-Trainer-equivalent decay / no-decay param groups over the trainable parameters."""
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if _is_no_decay_param(name, p):
            no_decay.append(p)
        else:
            decay.append(p)
    groups: list[dict[str, Any]] = []
    if decay:
        groups.append({"params": decay, "weight_decay": float(weight_decay)})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    if not groups:
        raise ValueError("model has no trainable parameters (requires_grad=True)")
    return groups


def build_lowmem_optimizer(
    model: nn.Module,
    *,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    backend: str = "auto",
    seed: int = 0,
    fused: bool | None = None,
) -> LowMemAdamW:
    """Build :class:`LowMemAdamW` with HF-Trainer-equivalent decay / no-decay param groups."""
    return LowMemAdamW(
        _decay_groups(model, weight_decay),
        lr=float(lr),
        betas=betas,
        eps=float(eps),
        weight_decay=float(weight_decay),
        backend=backend,
        seed=seed,
        fused=fused,
    )


def build_bnb8bit_optimizer(
    model: nn.Module,
    *,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> Any:
    """Baseline for comparisons: ``bitsandbytes.optim.AdamW8bit`` on the same param groups.

    An explicit request never falls back: without bitsandbytes this raises ImportError.
    """
    import bitsandbytes as bnb

    return bnb.optim.AdamW8bit(
        _decay_groups(model, weight_decay), lr=float(lr), betas=betas, eps=float(eps), weight_decay=float(weight_decay)
    )


def apply_online_turbo(model: nn.Module, cfg: OnlineTurboConfig) -> dict[str, Any]:
    """Apply online acceleration patches in place on a GR00T N1.7 model.

    Returns a JSON-serializable summary of what was applied. Safe to call when ``cfg.enabled``
    is ``False`` (no-op).
    """
    info: dict[str, Any] = {
        **asdict(cfg),
        "enabled": cfg.enabled,
        "backbone_bf16_tensors": 0,
        "backbone_bytes_saved_gb": 0.0,
        "backbone_no_grad": False,
        "action_head_bf16_tensors": 0,
        "action_head_bytes_saved_gb": 0.0,
        "multi_noise_applied": False,
        "compile_applied": False,
        "compile_skipped_reason": None,
        "triton_available": triton_available(),
    }
    if not cfg.enabled:
        return info

    backbone = getattr(model, "backbone", None)
    action_head = getattr(model, "action_head", None)

    if backbone is not None:
        if cfg.bf16_frozen_backbone:
            n_cast, saved = cast_parameters_only(backbone, torch.bfloat16, only_frozen=True)
            info["backbone_bf16_tensors"] = n_cast
            info["backbone_bytes_saved_gb"] = round(saved / 1024**3, 3)
        if cfg.frozen_backbone_no_grad:
            info["backbone_no_grad"] = _wrap_frozen_backbone_no_grad(backbone)

    if action_head is not None:
        if cfg.bf16_weights:
            n_cast, saved = cast_parameters_only(action_head, torch.bfloat16, only_trainable=True)
            info["action_head_bf16_tensors"] = n_cast
            info["action_head_bytes_saved_gb"] = round(saved / 1024**3, 3)

        if cfg.multi_noise > 1:
            from gr00t_lancedb.vl_cache.accel import patch_action_head_multi_noise

            patch_action_head_multi_noise(action_head, k_noise=cfg.multi_noise, stratified=cfg.stratified)
            info["multi_noise_applied"] = True

        if cfg.compile_head:
            dit = getattr(action_head, "model", None)
            if dit is None:
                info["compile_skipped_reason"] = "action_head has no .model attribute"
            elif any(p.requires_grad and p.dtype == torch.float32 for p in dit.parameters()):
                # Known PyTorch 2.9 Inductor crash on fp32 weights + bf16 autocast in GR00T DiT.
                reason = "skipped torch.compile on fp32 action_head.model weights (known Inductor bf16 autocast crash)"
                logger.warning("[online-turbo] %s", reason)
                info["compile_skipped_reason"] = reason
            elif not hasattr(dit, "_orig_mod"):
                action_head.model = torch.compile(dit, mode="default")
                info["compile_applied"] = True
            else:
                info["compile_applied"] = True

    logger.info("[online-turbo] applied: %s", info)
    return info


@contextmanager
def unwrap_compiled_action_head(model: nn.Module) -> Iterator[None]:
    """Temporarily restore uncompiled ``action_head.model`` so saved checkpoints never contain
    ``_orig_mod.`` keys (which would break ``Gr00tPolicy.from_pretrained`` at eval time).

    Unwraps DDP / Accelerate ``.module`` wrappers before locating ``action_head``.
    """
    base = model
    while hasattr(base, "module") and isinstance(base.module, nn.Module):
        base = base.module
    ah = getattr(base, "action_head", None)
    compiled = getattr(ah, "model", None) if ah is not None else None
    if ah is None or compiled is None or not hasattr(compiled, "_orig_mod"):
        yield
        return
    ah.model = compiled._orig_mod
    try:
        yield
    finally:
        ah.model = compiled


def make_online_turbo_trainer_class(base_trainer_cls: type, cfg: OnlineTurboConfig) -> type:
    """Wrap a HuggingFace/GR00T ``Trainer`` subclass to apply :func:`apply_online_turbo`,
    optionally build :class:`LowMemAdamW`, and unwrap compiled submodules on checkpoint save."""
    if not cfg.enabled:
        return base_trainer_cls

    class OnlineTurboTrainer(base_trainer_cls):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            model = kwargs.get("model") if "model" in kwargs else (args[0] if args else None)
            self._online_turbo_cfg = cfg
            self._online_turbo_info = apply_online_turbo(model, cfg) if model is not None else {}
            super().__init__(*args, **kwargs)

        def create_optimizer(self) -> Any:
            if self._online_turbo_cfg.optimizer == "native":
                return super().create_optimizer()
            if self.optimizer is None:
                betas = (
                    float(getattr(self.args, "adam_beta1", 0.9)),
                    float(getattr(self.args, "adam_beta2", 0.999)),
                )
                eps = float(getattr(self.args, "adam_epsilon", 1e-8))
                if self._online_turbo_cfg.optimizer == "bnb8bit":
                    self.optimizer = build_bnb8bit_optimizer(
                        self.model,
                        lr=float(self.args.learning_rate),
                        weight_decay=float(self.args.weight_decay),
                        betas=betas,
                        eps=eps,
                    )
                    logger.info("[online-turbo] created bitsandbytes AdamW8bit (baseline, betas=%s)", betas)
                    return self.optimizer
                self.optimizer = build_lowmem_optimizer(
                    self.model,
                    lr=float(self.args.learning_rate),
                    weight_decay=float(self.args.weight_decay),
                    betas=betas,
                    eps=eps,
                    backend=self._online_turbo_cfg.lowmem_backend,
                    seed=self._online_turbo_cfg.seed,
                    fused={"auto": None, "on": True, "off": False}[self._online_turbo_cfg.lowmem_fused],
                )
                logger.info(
                    "[online-turbo] created LowMemAdamW (backend=%s, fused=%s, betas=%s, lr=%g, wd=%g)",
                    self.optimizer.backend,
                    self._online_turbo_cfg.lowmem_fused,
                    betas,
                    self.args.learning_rate,
                    self.args.weight_decay,
                )
            return self.optimizer

        def _save(self, output_dir: str | None = None, state_dict: dict[str, Any] | None = None) -> None:
            with unwrap_compiled_action_head(self.model):
                return super()._save(output_dir=output_dir, state_dict=state_dict)

        def save_model(self, output_dir: str | None = None, _internal_call: bool = False) -> None:
            with unwrap_compiled_action_head(self.model):
                return super().save_model(output_dir=output_dir, _internal_call=_internal_call)

    OnlineTurboTrainer.__name__ = f"OnlineTurbo{base_trainer_cls.__name__}"
    return OnlineTurboTrainer
