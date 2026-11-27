# SPDX-License-Identifier: Apache-2.0
"""Feature compression codecs, padding helpers, and cache fingerprinting (Protocol 5)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Literal

import numpy as np

VL_CACHE_FORMAT = "robodata.vl_cache/v1"
CACHE_META_KEY = b"robodata.vl_cache_meta"

QuantCodec = Literal["per_token_int8", "bf16_raw"]
PadSide = Literal["left", "right"]


@dataclass(frozen=True)
class CacheFingerprint:
    """Deterministic identity for a frozen VLM feature cache table.

    Any mismatch between materialized cache metadata and the active training config
    fails closed rather than silently feeding mismatched representations.
    """

    base_model_path: str
    select_layer: int
    hidden_dim: int
    image_target_size: tuple[int, int] | None = (256, 256)
    image_crop_size: tuple[int, int] | None = (230, 230)
    augmentation_signature: str = "none"
    quant_codec: QuantCodec = "per_token_int8"
    base_model_revision: str | None = None
    backbone_name: str = "nvidia/Cosmos-Reason2-2B"
    num_views: int = 1
    source_frames_version: int | None = None
    pad_side: PadSide = "left"

    def canonical_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Exclude mutable/bookkeeping fields from the compatibility hash so
        # num_views growth or frames_version metadata can be validated separately.
        hash_fields = {
            "base_model_path": d["base_model_path"],
            "base_model_revision": d["base_model_revision"],
            "backbone_name": d["backbone_name"],
            "select_layer": int(d["select_layer"]),
            "hidden_dim": int(d["hidden_dim"]),
            "image_target_size": list(d["image_target_size"]) if d["image_target_size"] else None,
            "image_crop_size": list(d["image_crop_size"]) if d["image_crop_size"] else None,
            "augmentation_signature": d["augmentation_signature"],
            "quant_codec": d["quant_codec"],
        }
        return hash_fields

    def sha256(self) -> str:
        payload = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def short_id(self) -> str:
        return self.sha256()[:8]

    def table_name(self) -> str:
        return f"vl_cache_{self.short_id()}"

    def to_metadata_json(self) -> str:
        meta = asdict(self)
        meta["format"] = VL_CACHE_FORMAT
        meta["cache_fingerprint"] = self.sha256()
        return json.dumps(meta, sort_keys=True)

    @classmethod
    def from_metadata_json(cls, raw: str | bytes) -> tuple["CacheFingerprint", str]:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        data = json.loads(raw)
        if data.get("format") != VL_CACHE_FORMAT:
            raise ValueError(f"Unsupported vl_cache format: {data.get('format')!r}")
        stored_fp = str(data.get("cache_fingerprint", ""))
        fp = cls(
            base_model_path=data["base_model_path"],
            select_layer=int(data["select_layer"]),
            hidden_dim=int(data["hidden_dim"]),
            image_target_size=tuple(data["image_target_size"]) if data.get("image_target_size") else None,
            image_crop_size=tuple(data["image_crop_size"]) if data.get("image_crop_size") else None,
            augmentation_signature=str(data.get("augmentation_signature", "none")),
            quant_codec=data.get("quant_codec", "per_token_int8"),
            base_model_revision=data.get("base_model_revision"),
            backbone_name=str(data.get("backbone_name", "nvidia/Cosmos-Reason2-2B")),
            num_views=int(data.get("num_views", 1)),
            source_frames_version=data.get("source_frames_version"),
            pad_side=data.get("pad_side", "left"),
        )
        return fp, stored_fp


def detect_pad_side(attention_mask: np.ndarray) -> PadSide:
    """Infer whether a 1D boolean attention mask is left-padded or right-padded."""
    mask = np.asarray(attention_mask, dtype=bool)
    if mask.ndim != 1 or mask.size == 0 or mask.all():
        return "left"
    # If the first element is False and the last is True -> left-padded.
    if not mask[0] and mask[-1]:
        return "left"
    if mask[0] and not mask[-1]:
        return "right"
    return "left"


def strip_padding(
    features: np.ndarray,
    attention_mask: np.ndarray,
    image_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, PadSide]:
    """Strip padded tokens from a single sample ``[T, D]`` using ``attention_mask``.

    Works for both left-padded (Qwen3-VL default) and right-padded sequences.
    Returns ``(valid_features [L, D], valid_image_mask [L], pad_side)``.
    """
    mask = np.asarray(attention_mask, dtype=bool)
    img_mask = np.asarray(image_mask, dtype=bool)
    if features.ndim != 2 or mask.ndim != 1 or img_mask.ndim != 1:
        raise ValueError(
            f"Expected features [T, D], mask [T], image_mask [T]; got "
            f"{features.shape}, {mask.shape}, {img_mask.shape}"
        )
    if features.shape[0] != mask.shape[0] or mask.shape[0] != img_mask.shape[0]:
        raise ValueError("Sequence length mismatch across features, attention_mask, image_mask")
    pad_side = detect_pad_side(mask)
    valid_feat = features[mask]
    valid_img = img_mask[mask]
    return valid_feat, valid_img, pad_side


def pack_bool_mask(mask: np.ndarray) -> bytes:
    """Bit-pack a 1D boolean array into compact bytes (``ceil(L / 8)`` bytes)."""
    return np.packbits(np.asarray(mask, dtype=np.uint8)).tobytes()


def unpack_bool_mask(data: bytes, seq_len: int) -> np.ndarray:
    """Unpack bit-packed bytes back into a 1D boolean array of length ``seq_len``."""
    arr = np.frombuffer(data, dtype=np.uint8)
    bits = np.unpackbits(arr)[:seq_len]
    return bits.astype(bool)


def encode_features(
    features: np.ndarray,
    codec: QuantCodec = "per_token_int8",
) -> tuple[bytes, bytes]:
    """Encode ``[L, D]`` float32/float16/bf16 features into ``(scale_bytes, payload_bytes)``.

    - ``per_token_int8``: symmetric per-token int8 with fp16 scales (2x smaller than bf16).
    - ``bf16_raw``: lossless raw uint16-viewed bfloat16 (empty scale_bytes).
    """
    if features.ndim != 2:
        raise ValueError(f"Expected 2D [L, D] features, got {features.shape}")

    if codec == "per_token_int8":
        f32 = np.asarray(features, dtype=np.float32)
        amax = np.max(np.abs(f32), axis=-1, keepdims=True)
        scale_f32 = np.maximum(amax / 127.0, 1e-6)
        scale_f16 = scale_f32.astype(np.float16)
        # Use fp16 scale during quantization so encode/decode share exact scale values.
        q = np.clip(np.rint(f32 / scale_f16.astype(np.float32)), -127, 127).astype(np.int8)
        return scale_f16.reshape(-1).tobytes(), np.ascontiguousarray(q).tobytes()

    if codec == "bf16_raw":
        # Represent bf16 as top 16 bits of IEEE-754 float32 (round-to-nearest-even).
        f32 = np.ascontiguousarray(features, dtype=np.float32)
        u32 = f32.view(np.uint32)
        # Round-to-nearest-even to bfloat16
        lsb = (u32 >> 16) & np.uint32(1)
        rounding_bias = np.uint32(0x7FFF) + lsb
        bf16_u16 = ((u32 + rounding_bias) >> 16).astype(np.uint16)
        return b"", np.ascontiguousarray(bf16_u16).tobytes()

    raise ValueError(f"Unknown quant_codec: {codec!r}")


def encode_features_torch(features: Any, codec: QuantCodec = "per_token_int8") -> tuple[Any, Any | None]:
    """Encode ``[..., T, D]`` features on their own device: ``(payload, scale | None)``.

    Bit-identical to :func:`encode_features` (same fp32 arithmetic, fp16 scales, round-half-even)
    so a cache written through this path has the same fingerprint and the same bytes. Doing it
    on the GPU moves int8 instead of fp32 across PCIe and takes the per-row numpy work off the
    materialisation loop.
    """
    import torch

    if codec == "per_token_int8":
        f32 = features.to(torch.float32)
        scale16 = (f32.abs().amax(dim=-1, keepdim=True) / 127.0).clamp_min(1e-6).to(torch.float16)
        q = torch.round(f32 / scale16.to(torch.float32)).clamp_(-127, 127).to(torch.int8)
        return q, scale16.squeeze(-1)
    if codec == "bf16_raw":
        return features.to(torch.bfloat16).contiguous().view(torch.int16), None
    raise ValueError(f"Unknown quant_codec: {codec!r}")


def pooled_visual_emb_torch(features: Any, attention_mask: Any, image_mask: Any) -> Any:
    """Batched :func:`compute_pooled_visual_emb`: ``[B, T, D]`` -> L2-normalised ``[B, D]`` fp32."""
    import torch

    attn = attention_mask.bool()
    sel = attn & image_mask.bool()
    sel = torch.where(sel.any(dim=1, keepdim=True), sel, attn)
    w = sel.unsqueeze(-1).to(torch.float32)
    pooled = (features.to(torch.float32) * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)
    norm = pooled.norm(dim=-1, keepdim=True)
    return torch.where(norm > 1e-8, pooled / norm.clamp_min(1e-30), pooled)


def decode_features_numpy(
    scale_bytes: bytes,
    payload_bytes: bytes,
    seq_len: int,
    hidden_dim: int,
    codec: QuantCodec = "per_token_int8",
) -> np.ndarray:
    """Decode bytes back to ``float32`` numpy array ``[seq_len, hidden_dim]``."""
    if codec == "per_token_int8":
        scale = np.frombuffer(scale_bytes, dtype=np.float16).reshape(seq_len, 1).astype(np.float32)
        q = np.frombuffer(payload_bytes, dtype=np.int8).reshape(seq_len, hidden_dim)
        return q.astype(np.float32) * scale

    if codec == "bf16_raw":
        bf16_u16 = np.frombuffer(payload_bytes, dtype=np.uint16).reshape(seq_len, hidden_dim)
        u32 = bf16_u16.astype(np.uint32) << np.uint32(16)
        return u32.view(np.float32)

    raise ValueError(f"Unknown quant_codec: {codec!r}")


def dequantize_features_torch(
    payload_tensor: Any,
    scale_tensor: Any | None,
    codec: QuantCodec = "per_token_int8",
    target_dtype: Any | None = None,
) -> Any:
    """Fast GPU-side dequantization of a collated batch ``[B, T, D]``.

    Keeping ``payload_tensor`` in ``int8`` during host->device transfer cuts PCIe traffic
    in half and expands to ``bf16`` in a single fused elementwise op on GPU.
    """
    import torch

    if target_dtype is None:
        target_dtype = torch.bfloat16

    if codec == "per_token_int8":
        if scale_tensor is None:
            raise ValueError("scale_tensor is required for per_token_int8")
        return payload_tensor.to(dtype=target_dtype) * scale_tensor.to(dtype=target_dtype).unsqueeze(-1)

    if codec == "bf16_raw":
        if payload_tensor.dtype == torch.int16 or payload_tensor.dtype == torch.uint16:
            return payload_tensor.view(torch.bfloat16).to(dtype=target_dtype)
        return payload_tensor.to(dtype=target_dtype)

    raise ValueError(f"Unknown quant_codec: {codec!r}")


def compute_pooled_visual_emb(
    features: np.ndarray,
    image_mask: np.ndarray,
) -> np.ndarray:
    """Compute an L2-normalized mean-pooled visual embedding ``[D]`` (float32).

    Reuses already-computed VLM backbone features for downstream semantic dedup and
    failure retrieval without running a separate vision encoder.
    """
    f32 = np.asarray(features, dtype=np.float32)
    mask = np.asarray(image_mask, dtype=bool)
    selected = f32[mask] if mask.any() else f32
    pooled = selected.mean(axis=0)
    norm = float(np.linalg.norm(pooled))
    if norm > 1e-8:
        pooled = pooled / norm
    return pooled.astype(np.float32)
