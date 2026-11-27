# SPDX-License-Identifier: Apache-2.0
"""Frozen VLM feature caching, Multi-Noise Flow Matching, and low-VRAM ActionHead training.

Designed for NVIDIA Isaac-GR00T N1.7 on top of LanceDB:
- ``codec``: padding stripping, Per-Token INT8 / BF16 codecs, bit-packed masks, fingerprinting.
- ``store``: ``vl_cache_<fp>.lance`` writer and fast random-access ``take()`` reader.
- ``accel``: ActionHead-Only lazy loader, Stratified Multi-Noise Flow Matching, feature-space
  augmentation, 8-bit AdamW + unused-embodiment pruning, lossless checkpoint export, and
  cost-model router.
"""

from __future__ import annotations

from .accel import (
    CostRouteDecision,
    FeatureSpaceAugmenter,
    build_action_head_optimizer,
    estimate_cost_route,
    export_merged_gr00t_checkpoint,
    load_action_head_only,
    patch_action_head_multi_noise,
    sample_stratified_beta_time,
    save_action_head_checkpoint,
)
from .codec import (
    CACHE_META_KEY,
    VL_CACHE_FORMAT,
    CacheFingerprint,
    compute_pooled_visual_emb,
    decode_features_numpy,
    dequantize_features_torch,
    encode_features,
    encode_features_torch,
    pack_bool_mask,
    pooled_visual_emb_torch,
    strip_padding,
    unpack_bool_mask,
)
from .probe import build_probe_batches, head_inputs_from_cached, probe_fm_loss, split_holdout_rows
from .store import AsyncSink, VLCacheReader, VLCacheWriter, collate_cached_vl_batch

__all__ = [
    "AsyncSink",
    "build_probe_batches",
    "encode_features_torch",
    "head_inputs_from_cached",
    "pooled_visual_emb_torch",
    "probe_fm_loss",
    "split_holdout_rows",
    "CACHE_META_KEY",
    "VL_CACHE_FORMAT",
    "CacheFingerprint",
    "CostRouteDecision",
    "FeatureSpaceAugmenter",
    "VLCacheReader",
    "VLCacheWriter",
    "build_action_head_optimizer",
    "collate_cached_vl_batch",
    "compute_pooled_visual_emb",
    "decode_features_numpy",
    "dequantize_features_torch",
    "encode_features",
    "estimate_cost_route",
    "export_merged_gr00t_checkpoint",
    "load_action_head_only",
    "pack_bool_mask",
    "patch_action_head_multi_noise",
    "sample_stratified_beta_time",
    "save_action_head_checkpoint",
    "strip_padding",
    "unpack_bool_mask",
]
