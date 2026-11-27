# SPDX-License-Identifier: Apache-2.0
"""ActionHead-Only lazy loader, Stratified Multi-Noise Flow Matching, feature-space
augmentation, 8-bit AdamW + unused-embodiment pruning, lossless checkpoint export,
and cost-model routing.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import types
from typing import Any, Iterable, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Stratified Multi-Noise Flow Matching
# ---------------------------------------------------------------------------


def sample_stratified_beta_time(
    batch_size: int,
    k_noise: int,
    *,
    alpha: float = 1.5,
    beta: float = 1.0,
    noise_s: float = 0.999,
    device: Any = "cpu",
    dtype: Any = None,
) -> Any:
    """Sample ``[batch_size * k_noise]`` flow-matching timesteps via stratified Beta sampling.

    For each sample in ``batch_size``, partitions ``u in [0, 1)`` into ``k_noise`` strata
    ``[j/K, (j+1)/K)``, draws one uniform point per stratum, and maps through the inverse
    Beta(alpha, beta) CDF (using exact closed form for the GR00T default ``beta=1.0``
    and scipy/Beta fallback otherwise). Guarantees every sample covers early, mid, and
    late denoising times in a single step, cutting gradient variance.
    """
    import torch

    if dtype is None:
        dtype = torch.float32
    if k_noise <= 1:
        dist = torch.distributions.Beta(
            torch.tensor(float(alpha), dtype=torch.float32),
            torch.tensor(float(beta), dtype=torch.float32),
        )
        t = dist.sample((batch_size,))
        return ((1.0 - t) * float(noise_s)).to(device=device, dtype=dtype)

    # Stratified uniform draws in [0, 1): shape [B, K]
    strata = torch.arange(k_noise, dtype=torch.float32).unsqueeze(0)  # [1, K]
    u = (strata + torch.rand(batch_size, k_noise, dtype=torch.float32)) / float(k_noise)
    u = u.clamp(1e-6, 1.0 - 1e-6)

    # Fast exact inverse CDF when beta == 1.0 (GR00T N1.7 default: alpha=1.5, beta=1.0):
    # CDF(x) = x^alpha => x = u^(1/alpha)
    if abs(beta - 1.0) < 1e-7:
        x = u.pow(1.0 / float(alpha))
    elif abs(alpha - 1.0) < 1e-7:
        x = 1.0 - (1.0 - u).pow(1.0 / float(beta))
    else:
        from scipy.stats import beta as scipy_beta

        x = torch.from_numpy(scipy_beta.ppf(u.numpy(), alpha, beta).astype(np.float32))

    # Match Gr00tN1d7ActionHead.sample_time exactly: t = (1 - sample) * noise_s
    t = (1.0 - x) * float(noise_s)
    return t.reshape(batch_size * k_noise).to(device=device, dtype=dtype)


def patch_action_head_multi_noise(
    action_head: Any,
    k_noise: int = 4,
    *,
    stratified: bool = True,
) -> Any:
    """Patch ``Gr00tN1d7ActionHead.forward`` in-place to reuse one VL forward across ``k_noise``
    independent ``(t, noise)`` flow-matching samples.

    ``process_backbone_output`` (vlln + vl_self_attention) and ``state_encoder`` run ONCE per
    observation, then tensors are expanded by ``k_noise`` for the lightweight 41-token DiT.
    """
    import torch
    import torch.nn.functional as F

    if k_noise <= 1:
        return action_head

    orig_forward = action_head.forward

    def _forward_multi_noise(self: Any, backbone_output: Any, action_input: Any) -> dict[str, Any]:
        self.set_frozen_modules_to_eval_mode()
        backbone_output = self.process_backbone_output(backbone_output)

        vl_embeds = backbone_output.backbone_features
        device = vl_embeds.device
        embodiment_id = action_input.embodiment_id

        state = action_input.state
        assert state.shape[1] == self.config.state_history_length
        state_flat = state.view(state.shape[0], 1, -1)
        state_features = self.state_encoder(state_flat, embodiment_id)

        bsz = vl_embeds.shape[0]
        vl_attn_mask = backbone_output.backbone_attention_mask
        image_mask = getattr(backbone_output, "image_mask", None)

        # Expand across K noise samples
        vl_embeds_k = vl_embeds.repeat_interleave(k_noise, dim=0)
        vl_attn_mask_k = vl_attn_mask.repeat_interleave(k_noise, dim=0) if vl_attn_mask is not None else None
        image_mask_k = image_mask.repeat_interleave(k_noise, dim=0) if image_mask is not None else None
        state_features_k = state_features.repeat_interleave(k_noise, dim=0)
        embodiment_id_k = embodiment_id.repeat_interleave(k_noise, dim=0)

        # Independent state dropout per noise copy
        if self.training and self.state_dropout_prob > 0:
            do_dropout = torch.rand(state_features_k.shape[0], device=device) < self.state_dropout_prob
            state_features_k = state_features_k * (1 - do_dropout[:, None, None].to(dtype=state_features_k.dtype))

        actions = action_input.action.repeat_interleave(k_noise, dim=0)
        raw_mask = action_input.action_mask
        if raw_mask is None:
            raise ValueError("action_mask is required during training")
        if raw_mask.ndim == 2:
            raw_mask = raw_mask.unsqueeze(1).expand_as(action_input.action)
        action_mask = raw_mask.repeat_interleave(k_noise, dim=0)

        strict_action_mask = action_mask if self.config.strict_action_padding_mask else None
        if strict_action_mask is not None:
            actions = actions * strict_action_mask

        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        if strict_action_mask is not None:
            noise = noise * strict_action_mask

        if stratified:
            t_1d = sample_stratified_beta_time(
                bsz,
                k_noise,
                alpha=float(self.config.noise_beta_alpha),
                beta=float(self.config.noise_beta_beta),
                noise_s=float(self.config.noise_s),
                device=actions.device,
                dtype=actions.dtype,
            )
        else:
            t_1d = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t_1d[:, None, None]

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise
        action_token_mask = None
        if strict_action_mask is not None:
            noisy_trajectory = noisy_trajectory * strict_action_mask
            velocity = velocity * strict_action_mask
            action_token_mask = strict_action_mask.any(dim=-1)

        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id_k)

        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)
        if action_token_mask is not None:
            action_features = action_features * action_token_mask.unsqueeze(-1).to(dtype=action_features.dtype)

        sa_embs = torch.cat((state_features_k, action_features), dim=1)
        hidden_attention_mask = None
        if action_token_mask is not None:
            hidden_attention_mask = torch.cat(
                (torch.ones(state_features_k.shape[:2], device=device, dtype=torch.bool), action_token_mask),
                dim=1,
            )

        if self.config.use_alternate_vl_dit:
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds_k,
                encoder_attention_mask=vl_attn_mask_k,
                timestep=t_discretized,
                return_all_hidden_states=True,
                image_mask=image_mask_k,
                backbone_attention_mask=vl_attn_mask_k,
                hidden_attention_mask=hidden_attention_mask,
            )
        else:
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds_k,
                encoder_attention_mask=vl_attn_mask_k if self.config.strict_vl_padding_mask else None,
                timestep=t_discretized,
                return_all_hidden_states=True,
                hidden_attention_mask=hidden_attention_mask,
            )

        pred = self.action_decoder(model_output, embodiment_id_k)
        pred_actions = pred[:, -actions.shape[1] :]
        if strict_action_mask is not None:
            pred_actions = pred_actions * strict_action_mask

        action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        loss = action_loss.sum() / (action_mask.sum() + 1e-6)
        return {
            "loss": loss,
            "action_loss": action_loss,
            "action_mask": action_mask,
            "backbone_features": vl_embeds,
            "state_features": state_features,
        }

    action_head._orig_forward = orig_forward
    action_head._k_noise = int(k_noise)
    action_head.forward = types.MethodType(_forward_multi_noise, action_head)
    return action_head


# ---------------------------------------------------------------------------
# Feature-Space Augmentation (0 extra storage)
# ---------------------------------------------------------------------------


class FeatureSpaceAugmenter:
    """Regularizes cached VLM visual tokens on GPU without extra disk storage.

    Applies visual-token-only random dropout (zeroing a small fraction of image tokens)
    and scale-adaptive Gaussian jitter. Text tokens (``image_mask == False``) are untouched.
    """

    def __init__(self, token_dropout: float = 0.08, gaussian_jitter: float = 0.02) -> None:
        self.token_dropout = float(token_dropout)
        self.gaussian_jitter = float(gaussian_jitter)

    def __call__(self, backbone_output: Any, training: bool = True) -> Any:
        if not training or (self.token_dropout <= 0.0 and self.gaussian_jitter <= 0.0):
            return backbone_output
        import torch

        feats = backbone_output["backbone_features"]
        img_mask = backbone_output.get("image_mask")
        if img_mask is None:
            return backbone_output

        out = feats
        img_mask_f = img_mask.unsqueeze(-1).to(dtype=out.dtype)  # [B, T, 1]

        if self.gaussian_jitter > 0.0:
            # Per-token std for scale-adaptive perturbation
            std = out.detach().float().std(dim=-1, keepdim=True).clamp(min=1e-4).to(dtype=out.dtype)
            noise = torch.randn_like(out) * (self.gaussian_jitter * std)
            out = out + noise * img_mask_f

        if self.token_dropout > 0.0:
            keep = (torch.rand(img_mask.shape, device=out.device) >= self.token_dropout).unsqueeze(-1)
            # Only drop visual tokens; keep all text tokens
            effective_keep = torch.where(img_mask.unsqueeze(-1), keep, torch.ones_like(keep))
            out = out * effective_keep.to(dtype=out.dtype)

        backbone_output["backbone_features"] = out
        return backbone_output


# ---------------------------------------------------------------------------
# ActionHead-Only Lazy Loader & Lossless Checkpoint Export
# ---------------------------------------------------------------------------


def _resolve_model_dir(base_model_path: str | Path, revision: str | None = None) -> Path:
    p = Path(base_model_path)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=str(base_model_path),
            revision=revision,
            allow_patterns=["config.json", "*.safetensors", "*.safetensors.index.json"],
        )
    )


def _load_prefix_state_dict(model_dir: Path, prefix: str = "action_head.") -> dict[str, Any]:
    """Lazily read only tensors starting with ``prefix`` from safetensors shards."""
    from safetensors import safe_open

    index_path = model_dir / "model.safetensors.index.json"
    state_dict: dict[str, Any] = {}
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map: dict[str, str] = index["weight_map"]
        by_shard: dict[str, list[str]] = {}
        for k, shard in weight_map.items():
            if k.startswith(prefix):
                by_shard.setdefault(shard, []).append(k)
        for shard, keys in by_shard.items():
            with safe_open(str(model_dir / shard), framework="pt", device="cpu") as fh:
                for k in keys:
                    state_dict[k[len(prefix) :]] = fh.get_tensor(k)
        return state_dict

    single = model_dir / "model.safetensors"
    if not single.is_file():
        raise FileNotFoundError(f"No safetensors weights found in {model_dir}")
    with safe_open(str(single), framework="pt", device="cpu") as fh:
        for k in fh.keys():
            if k.startswith(prefix):
                state_dict[k[len(prefix) :]] = fh.get_tensor(k)
    return state_dict


def load_action_head_only(
    base_model_path: str | Path,
    *,
    revision: str | None = None,
    device: str = "cuda",
    dtype: Any | None = None,
    tune_projector: bool = True,
    tune_diffusion_model: bool = True,
    tune_vlln: bool = True,
) -> tuple[Any, Any, Path]:
    """Instantiate ONLY ``Gr00tN1d7ActionHead`` and lazily load ``action_head.*`` weights.

    Never instantiates the 2B VLM backbone on CPU or GPU.
    Returns ``(action_head, config, resolved_model_dir)``.
    """
    import torch
    from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead

    model_dir = _resolve_model_dir(base_model_path, revision=revision)
    config = Gr00tN1d7Config.from_pretrained(str(model_dir))
    config.tune_llm = False
    config.tune_visual = False
    config.tune_top_llm_layers = 0
    config.tune_projector = tune_projector
    config.tune_diffusion_model = tune_diffusion_model
    config.tune_vlln = tune_vlln

    action_head = Gr00tN1d7ActionHead(config)
    sd = _load_prefix_state_dict(model_dir, prefix="action_head.")
    missing, unexpected = action_head.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys when loading action_head: {missing[:10]}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys when loading action_head: {unexpected[:10]}")

    if dtype is None:
        dtype = torch.bfloat16 if getattr(config, "model_dtype", "bfloat16") == "bfloat16" else torch.float32
    action_head = action_head.to(device=device, dtype=dtype)
    action_head.set_trainable_parameters(tune_projector, tune_diffusion_model, tune_vlln)
    return action_head, config, model_dir


def build_action_head_optimizer(
    action_head: Any,
    *,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    use_8bit: bool = True,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> tuple[Any, str]:
    """Build AdamW (preferring 8-bit AdamW when requested and available) for trainable params."""
    import torch

    params = [p for p in action_head.parameters() if p.requires_grad]
    if use_8bit:
        try:
            import bitsandbytes as bnb

            return (
                bnb.optim.AdamW8bit(params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps),
                "bitsandbytes.AdamW8bit",
            )
        except ImportError:
            pass
    try:
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps, fused=True)
        return opt, "torch.AdamW(fused)"
    except Exception:
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
        return opt, "torch.AdamW"


def save_action_head_checkpoint(
    action_head: Any,
    output_dir: str | Path,
    *,
    base_model_path: str,
    base_model_revision: str | None = None,
    cache_fingerprint: str | None = None,
    step: int | None = None,
    save_dtype: str = "bfloat16",
) -> Path:
    """Save lightweight ActionHead-Only checkpoint (~3.2 GB bf16 instead of full 6.5+ GB model)."""
    import torch
    from safetensors.torch import save_file

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    target_dt = torch.bfloat16 if save_dtype == "bfloat16" else torch.float32
    sd = {
        k: (v.detach().to(device="cpu", dtype=target_dt).contiguous() if v.is_floating_point() else v.detach().cpu())
        for k, v in action_head.state_dict().items()
    }
    weights_path = out / "action_head.safetensors"
    save_file(sd, str(weights_path))
    meta = {
        "format": "robodata.action_head_ckpt/v1",
        "base_model_path": base_model_path,
        "base_model_revision": base_model_revision,
        "cache_fingerprint": cache_fingerprint,
        "step": step,
        "save_dtype": save_dtype,
        "num_tensors": len(sd),
    }
    (out / "action_head_adapter.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def export_merged_gr00t_checkpoint(
    action_head_ckpt_dir: str | Path,
    output_dir: str | Path,
    *,
    base_model_path: str | Path | None = None,
) -> Path:
    """Merge frozen ``backbone.*`` from base model with trained ``action_head.*`` into a
    standard HuggingFace / Isaac-GR00T checkpoint directory (100% inference & TRT compatible).
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    ckpt_dir = Path(action_head_ckpt_dir)
    adapter_meta = json.loads((ckpt_dir / "action_head_adapter.json").read_text(encoding="utf-8"))
    resolved_base = _resolve_model_dir(
        base_model_path or adapter_meta["base_model_path"],
        revision=adapter_meta.get("base_model_revision"),
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Copy all non-weight config/processor files from base model
    for item in resolved_base.iterdir():
        if item.is_file() and not item.name.endswith(".safetensors") and not item.name.endswith(".safetensors.index.json"):
            shutil.copy2(item, out / item.name)

    # Load trained action_head tensors
    ah_tensors: dict[str, Any] = {}
    with safe_open(str(ckpt_dir / "action_head.safetensors"), framework="pt", device="cpu") as fh:
        for k in fh.keys():
            ah_tensors[f"action_head.{k}"] = fh.get_tensor(k)

    # Stream base shards, replacing action_head.* keys in-place
    index_path = resolved_base / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map: dict[str, str] = index["weight_map"]
        shards = sorted(set(weight_map.values()))
        for shard in shards:
            merged: dict[str, Any] = {}
            with safe_open(str(resolved_base / shard), framework="pt", device="cpu") as fh:
                for k in fh.keys():
                    merged[k] = ah_tensors[k] if k in ah_tensors else fh.get_tensor(k)
            save_file(merged, str(out / shard))
        shutil.copy2(index_path, out / "model.safetensors.index.json")
    else:
        merged = {}
        with safe_open(str(resolved_base / "model.safetensors"), framework="pt", device="cpu") as fh:
            for k in fh.keys():
                merged[k] = ah_tensors[k] if k in ah_tensors else fh.get_tensor(k)
        save_file(merged, str(out / "model.safetensors"))

    return out


# ---------------------------------------------------------------------------
# Auto Cost-Model Router
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostRouteDecision:
    route: str  # "vl_cache" | "online_multi_noise"
    reason: str
    reuse_factor: float
    estimated_cache_gb: float
    estimated_online_s: float
    estimated_cached_s: float
    estimated_speedup: float


def estimate_cost_route(
    *,
    num_frames: int,
    total_steps: int,
    batch_size: int,
    num_runs: int = 1,
    num_views: int = 1,
    tune_backbone: bool = False,
    available_disk_gb: float = 200.0,
    bytes_per_frame: int = 398_000,  # ~389 KiB per-token INT8 @ 2 cams
    t_backbone_ms: float = 360.0,
    t_head_ms: float = 270.0,
    t_cache_io_ms: float = 10.0,
    infer_speedup_factor: float = 1.25,
) -> CostRouteDecision:
    """Decide whether to materialize a Lance VL cache or stay on online Multi-Noise."""
    est_cache_gb = (num_frames * num_views * bytes_per_frame) / (1024.0**3)
    total_samples = num_runs * total_steps * batch_size
    reuse = total_samples / max(1, num_frames * num_views)

    online_s = num_runs * total_steps * (t_backbone_ms + t_head_ms) / 1000.0
    materialize_s = (num_frames * num_views / max(1, batch_size)) * (t_backbone_ms / infer_speedup_factor) / 1000.0
    cached_s = materialize_s + num_runs * total_steps * (t_cache_io_ms + t_head_ms) / 1000.0
    speedup = online_s / max(1e-6, cached_s)

    if tune_backbone:
        return CostRouteDecision(
            route="online_multi_noise",
            reason="Backbone is unfrozen (tune_llm/tune_visual=True); caching is invalid.",
            reuse_factor=reuse,
            estimated_cache_gb=est_cache_gb,
            estimated_online_s=online_s,
            estimated_cached_s=cached_s,
            estimated_speedup=1.0,
        )
    if est_cache_gb > available_disk_gb * 0.85:
        return CostRouteDecision(
            route="online_multi_noise",
            reason=f"Estimated cache size ({est_cache_gb:.1f} GB) exceeds 85% of available disk ({available_disk_gb:.1f} GB).",
            reuse_factor=reuse,
            estimated_cache_gb=est_cache_gb,
            estimated_online_s=online_s,
            estimated_cached_s=cached_s,
            estimated_speedup=1.0,
        )
    if speedup < 1.15:
        return CostRouteDecision(
            route="online_multi_noise",
            reason=f"Reuse factor ({reuse:.2f}x) too low to amortize materialization (est speedup {speedup:.2f}x).",
            reuse_factor=reuse,
            estimated_cache_gb=est_cache_gb,
            estimated_online_s=online_s,
            estimated_cached_s=cached_s,
            estimated_speedup=speedup,
        )
    return CostRouteDecision(
        route="vl_cache",
        reason=f"Frozen backbone with {reuse:.2f}x reuse; est {speedup:.2f}x end-to-end speedup ({est_cache_gb:.1f} GB cache).",
        reuse_factor=reuse,
        estimated_cache_gb=est_cache_gb,
        estimated_online_s=online_s,
        estimated_cached_s=cached_s,
        estimated_speedup=speedup,
    )
