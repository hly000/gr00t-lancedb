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
from pathlib import Path
import shutil
from typing import Any, Iterator

import torch
import torch.nn as nn

from .ema import CpuShadowEMA
from .lowmem_adamw import LowMemAdamW, triton_available

logger = logging.getLogger("gr00t_lancedb.turbo.online")

_NO_DECAY_SUBSTRINGS = ("bias", "layernorm", "layer_norm", "rmsnorm", "rms_norm", "vlln", ".norm")
_NORM_SUBSTRINGS = ("layernorm", "layer_norm", "rmsnorm", "rms_norm", "vlln", ".norm")
_NORM_TYPES = (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)

OPTIMIZERS = ("native", "lowmem", "flashoptim", "flash24", "flash32", "bnb8bit", "torchao8bit")
# Keep a sub-ulp correction of bf16 weights (Kahan term / FlashOptim error bits).
COMPENSATED_BF16_OPTIMIZERS = ("lowmem", "flashoptim", "flash24", "flash32")
_FLASH_BITS = {"flash24": 24, "flash32": 32}


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
    # "native" | "lowmem" | baselines: "flashoptim" (GR00T_TURBO_FLASHOPTIM_BITS, default 24),
    # "flash24" | "flash32" (explicit master_weight_bits), "bnb8bit", "torchao8bit"
    optimizer: str = "native"
    lowmem_backend: str = "auto"  # "auto" | "triton" | "torch"
    lowmem_comp: str = "bf16"  # "bf16" | "int8" (int8 ULP-relative saves 1 B/param -> 7.06 B/param)
    bf16_weights: bool = False
    bf16_frozen_backbone: bool = False
    # Off by default so LowMemAdamW runs stay numerically comparable with falsify_07 (93abd3a);
    # the FlashAdamW and lowmem24_* modes of run_online_turbo.sh turn it on.
    keep_norm_fp32: bool = False
    # Clip through LowMemAdamW.clip_scale() (gradient scale applied inside the update) instead
    # of HF's in-place clip of bf16 gradients. Off by default for the same reason.
    fused_clip: bool = False
    # Activation checkpointing of the DiT transformer blocks (docs/plans/08 WP2 2.3): recompute
    # each block in backward instead of keeping its activations. Off by default.
    grad_ckpt_head: bool = False
    ema_decay: float = 0.0
    ema_update_every: int = 10
    compile_head: bool = False
    frozen_backbone_no_grad: bool = True
    allow_pure_bf16: bool = False
    seed: int = 42

    def __post_init__(self) -> None:
        if self.multi_noise < 1:
            raise ValueError(f"multi_noise must be >= 1, got {self.multi_noise}")
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {OPTIMIZERS}, got {self.optimizer!r}")
        if self.lowmem_backend not in ("auto", "triton", "torch"):
            raise ValueError(f"lowmem_backend must be 'auto'|'triton'|'torch', got {self.lowmem_backend!r}")
        if self.lowmem_comp not in ("bf16", "int8"):
            raise ValueError(f"lowmem_comp must be 'bf16' or 'int8', got {self.lowmem_comp!r}")
        if not (0.0 <= self.ema_decay < 1.0):
            raise ValueError(f"ema_decay must be in [0, 1), got {self.ema_decay}")
        if self.ema_update_every < 1:
            raise ValueError(f"ema_update_every must be >= 1, got {self.ema_update_every}")
        if self.bf16_weights and self.optimizer not in COMPENSATED_BF16_OPTIMIZERS and not self.allow_pure_bf16:
            raise ValueError(
                "bf16_weights=True without optimizer='lowmem' (or 'flashoptim') drops sub-ulp updates in pure bf16 "
                "(known to stall GR00T training). Set GR00T_TURBO_OPTIMIZER=lowmem|flashoptim, or pass "
                "GR00T_TURBO_ALLOW_PURE_BF16=1 only for intentional ablation."
            )

    @property
    def enabled(self) -> bool:
        return (
            self.multi_noise > 1
            or self.optimizer != "native"
            or self.bf16_weights
            or self.bf16_frozen_backbone
            or self.grad_ckpt_head
            or self.ema_decay > 0.0
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
            lowmem_comp=os.environ.get("GR00T_TURBO_LOWMEM_COMP", "bf16").strip().lower(),
            bf16_weights=bf16_weights,
            bf16_frozen_backbone=bf16_backbone,
            keep_norm_fp32=_env_bool("GR00T_TURBO_KEEP_NORM_FP32", False),
            fused_clip=_env_bool("GR00T_TURBO_FUSED_CLIP", False),
            grad_ckpt_head=_env_bool("GR00T_TURBO_GRAD_CKPT_HEAD", False),
            ema_decay=float(os.environ.get("GR00T_TURBO_EMA_DECAY", "0.0")),
            ema_update_every=int(os.environ.get("GR00T_TURBO_EMA_EVERY", "10")),
            compile_head=_env_bool("GR00T_TURBO_COMPILE_HEAD", False),
            frozen_backbone_no_grad=_env_bool("GR00T_TURBO_FROZEN_BACKBONE_NO_GRAD", True),
            allow_pure_bf16=_env_bool("GR00T_TURBO_ALLOW_PURE_BF16", False),
            seed=int(os.environ.get("GR00T_TURBO_SEED", str(default_seed))),
        )


def _norm_param_ids(module: nn.Module) -> set[int]:
    ids: set[int] = set()
    for sub in module.modules():
        if isinstance(sub, _NORM_TYPES) or "norm" in type(sub).__name__.lower():
            for p in sub.parameters(recurse=False):
                ids.add(id(p))
    for name, p in module.named_parameters():
        low = name.lower()
        if any(tok in low for tok in _NORM_SUBSTRINGS):
            ids.add(id(p))
    return ids


def cast_parameters_only(
    module: nn.Module,
    dtype: torch.dtype,
    *,
    only_frozen: bool = False,
    only_trainable: bool = False,
    keep_norm_fp32: bool = False,
) -> tuple[int, int]:
    """Cast floating-point ``nn.Parameter`` tensors in ``module`` to ``dtype`` in place.

    Crucially does NOT touch non-parameter buffers (e.g. Qwen3-VL's non-persistent FP32
    RoPE ``inv_freq`` buffers), avoiding silent positional-frequency truncation.
    When ``keep_norm_fp32=True``, normalization-layer parameters remain in ``float32``.
    Returns ``(tensors_cast, bytes_saved)``.
    """
    if only_frozen and only_trainable:
        raise ValueError("only_frozen and only_trainable are mutually exclusive")
    norm_ids = _norm_param_ids(module) if keep_norm_fp32 else set()
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
        if pid in norm_ids and p.dtype == torch.float32:
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


def _split_decay_param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
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
    comp_dtype: str = "bf16",
    seed: int = 0,
) -> LowMemAdamW:
    """Build :class:`LowMemAdamW` with HF-Trainer-equivalent decay / no-decay param groups."""
    groups = _split_decay_param_groups(model, weight_decay)
    return LowMemAdamW(
        groups,
        lr=float(lr),
        betas=betas,
        eps=float(eps),
        weight_decay=float(weight_decay),
        backend=backend,
        comp_dtype=comp_dtype,
        seed=seed,
    )


def build_third_party_optimizer(
    kind: str,
    model: nn.Module,
    *,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> torch.optim.Optimizer:
    """Build an opt-in comparison optimizer.

    ``flashoptim`` reads ``GR00T_TURBO_FLASHOPTIM_BITS`` (default 24); ``flash24`` / ``flash32``
    fix ``master_weight_bits``. A missing package raises ImportError (no silent fallback).
    """
    groups = _split_decay_param_groups(model, weight_decay)
    kw = dict(lr=float(lr), betas=betas, eps=float(eps), weight_decay=float(weight_decay))
    if kind in ("flashoptim", *_FLASH_BITS):
        from flashoptim import FlashAdamW

        bits = _FLASH_BITS.get(kind) or int(os.environ.get("GR00T_TURBO_FLASHOPTIM_BITS", "24"))
        if bits not in (24, 32):
            raise ValueError(f"FlashAdamW master_weight_bits must be 24 or 32, got {bits}")
        return FlashAdamW(groups, master_weight_bits=bits, **kw)
    if kind == "bnb8bit":
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(groups, **kw)
    if kind == "torchao8bit":
        try:
            from torchao.optim import AdamW8bit as TorchaoAdamW8bit
        except ImportError:
            from torchao.prototype.low_bit_optim import AdamW8bit as TorchaoAdamW8bit

        return TorchaoAdamW8bit(groups, **kw)
    raise ValueError(f"Unsupported third-party optimizer kind={kind!r}")


def checkpoint_dit_blocks(action_head: nn.Module) -> int:
    """Wrap every ``action_head.model.transformer_blocks[i]`` in non-reentrant activation
    checkpointing. Active only while the block is training and autograd is on, so evaluation
    and ``no_grad`` passes are unchanged. Idempotent; returns the number of blocks wrapped."""
    from torch.utils.checkpoint import checkpoint

    blocks = getattr(getattr(action_head, "model", None), "transformer_blocks", None)
    if blocks is None:
        return 0
    wrapped = 0
    for block in blocks:
        if getattr(block, "_turbo_grad_ckpt", False):
            continue
        original = block.forward

        def _forward(*args: Any, _orig: Any = original, _mod: nn.Module = block, **kwargs: Any) -> Any:
            if _mod.training and torch.is_grad_enabled():
                return checkpoint(_orig, *args, use_reentrant=False, **kwargs)
            return _orig(*args, **kwargs)

        block.forward = _forward  # type: ignore[method-assign]
        block._turbo_grad_ckpt = True
        wrapped += 1
    return wrapped


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
        "grad_ckpt_head_applied": False,
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
            n_cast, saved = cast_parameters_only(
                action_head,
                torch.bfloat16,
                only_trainable=True,
                keep_norm_fp32=cfg.keep_norm_fp32,
            )
            info["action_head_bf16_tensors"] = n_cast
            info["action_head_bytes_saved_gb"] = round(saved / 1024**3, 3)
            if cfg.keep_norm_fp32:
                info["action_head_norm_fp32_kept"] = sum(
                    1 for p in action_head.parameters() if p.requires_grad and p.dtype == torch.float32
                )

        if cfg.multi_noise > 1:
            from gr00t_lancedb.vl_cache.accel import patch_action_head_multi_noise

            patch_action_head_multi_noise(action_head, k_noise=cfg.multi_noise, stratified=cfg.stratified)
            info["multi_noise_applied"] = True

        if cfg.grad_ckpt_head:
            # GR00T's DiT has a gradient_checkpointing flag that its forward never reads, so the
            # flag alone (or diffusers' enable_gradient_checkpointing) would change nothing: wrap
            # the transformer blocks themselves.
            n_blocks = checkpoint_dit_blocks(action_head)
            info["grad_ckpt_dit_blocks"] = n_blocks
            info["grad_ckpt_head_applied"] = n_blocks > 0
            if n_blocks == 0:
                logger.warning("[online-turbo] grad_ckpt_head=True but action_head.model has no transformer_blocks")
            if cfg.compile_head:
                logger.warning(
                    "[online-turbo] grad_ckpt_head=True with compile_head=True may conflict or slow down compilation"
                )

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
            self._cpu_ema: CpuShadowEMA | None = None
            super().__init__(*args, **kwargs)
            if cfg.fused_clip:
                self._install_lowmem_clip_hook()
            if cfg.ema_decay > 0.0:
                self._cpu_ema = CpuShadowEMA(
                    self.model,
                    decay=cfg.ema_decay,
                    update_every=cfg.ema_update_every,
                )
                self._install_cpu_ema_callback()

        def _install_cpu_ema_callback(self) -> None:
            from transformers import TrainerCallback

            trainer_self = self
            skip_names = {
                "optimizer.pt",
                "scheduler.pt",
                "rng_state.pth",
                "trainer_state.json",
                "training_args.bin",
                "scaler.pt",
            }

            class _CpuEmaCallback(TrainerCallback):
                def __init__(self) -> None:
                    self._saving_ema = False

                def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
                    ema = trainer_self._cpu_ema
                    if ema is not None:
                        ema.update(int(state.global_step), optimizer=getattr(trainer_self, "optimizer", None))

                def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
                    ema = trainer_self._cpu_ema
                    if ema is None or self._saving_ema or not trainer_self.is_world_process_zero():
                        return
                    self._saving_ema = True
                    try:
                        step = int(state.global_step)
                        raw_dir = Path(args.output_dir) / f"checkpoint-{step}"
                        ema_dir = Path(args.output_dir) / f"checkpoint-{step}-ema"
                        with ema.swap_into_model():
                            trainer_self.save_model(str(ema_dir))
                        if ema_dir.is_dir():
                            # Copy GR00T sidecar metadata from raw_dir AND from output_dir/{processor,experiment_cfg}
                            # (GR00T's own CheckpointCallback may copy processor files into raw_dir after this callback).
                            out_root = Path(args.output_dir)
                            for src_dir in (raw_dir, out_root / "processor"):
                                if not src_dir.is_dir():
                                    continue
                                for item in src_dir.iterdir():
                                    dst = ema_dir / item.name
                                    if dst.exists() or item.name in skip_names:
                                        continue
                                    if item.suffix in (".safetensors", ".bin") or item.name.endswith(".index.json"):
                                        continue
                                    if item.is_dir():
                                        shutil.copytree(item, dst, dirs_exist_ok=True)
                                    elif item.is_file():
                                        shutil.copy2(item, dst)
                            exp_cfg = out_root / "experiment_cfg"
                            if exp_cfg.is_dir() and not (ema_dir / "experiment_cfg").exists():
                                shutil.copytree(exp_cfg, ema_dir / "experiment_cfg", dirs_exist_ok=True)
                        logger.info("[cpu-ema] saved EMA checkpoint -> %s (updates=%d)", ema_dir, ema.num_updates)
                    finally:
                        self._saving_ema = False

            self.add_callback(_CpuEmaCallback())

        def _install_lowmem_clip_hook(self) -> None:
            accel = getattr(self, "accelerator", None)
            if accel is None or getattr(accel, "_lowmem_clip_hooked", False):
                return
            orig_clip = accel.clip_grad_norm_
            trainer_self = self

            def _lowmem_clip_grad_norm(parameters: Any, max_norm: float, norm_type: int = 2) -> Any:
                opt = getattr(trainer_self, "optimizer", None)
                raw_opt = getattr(opt, "optimizer", opt)
                if isinstance(raw_opt, LowMemAdamW) and norm_type == 2:
                    norm, scale = raw_opt.clip_scale(float(max_norm))
                    raw_opt._pending_grad_scale = scale
                    return torch.tensor(norm)
                return orig_clip(parameters, max_norm, norm_type=norm_type)

            accel.clip_grad_norm_ = _lowmem_clip_grad_norm
            accel._lowmem_clip_hooked = True

        def create_optimizer(self) -> Any:
            opt_kind = self._online_turbo_cfg.optimizer
            if opt_kind == "native":
                return super().create_optimizer()
            if self.optimizer is None:
                betas = (
                    float(getattr(self.args, "adam_beta1", 0.9)),
                    float(getattr(self.args, "adam_beta2", 0.999)),
                )
                eps = float(getattr(self.args, "adam_epsilon", 1e-8))
                if opt_kind == "lowmem":
                    self.optimizer = build_lowmem_optimizer(
                        self.model,
                        lr=float(self.args.learning_rate),
                        weight_decay=float(self.args.weight_decay),
                        betas=betas,
                        eps=eps,
                        backend=self._online_turbo_cfg.lowmem_backend,
                        comp_dtype=self._online_turbo_cfg.lowmem_comp,
                        seed=self._online_turbo_cfg.seed,
                    )
                    logger.info(
                        "[online-turbo] created LowMemAdamW (backend=%s, comp=%s, betas=%s, lr=%g, wd=%g)",
                        self.optimizer.backend,
                        self.optimizer.comp_dtype,
                        betas,
                        self.args.learning_rate,
                        self.args.weight_decay,
                    )
                else:
                    self.optimizer = build_third_party_optimizer(
                        opt_kind,
                        self.model,
                        lr=float(self.args.learning_rate),
                        weight_decay=float(self.args.weight_decay),
                        betas=betas,
                        eps=eps,
                    )
                    logger.info(
                        "[online-turbo] created %s (%s, betas=%s, lr=%g, wd=%g)",
                        opt_kind,
                        type(self.optimizer).__name__,
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
