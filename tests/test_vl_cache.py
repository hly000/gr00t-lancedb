# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for gr00t_lancedb.vl_cache (codec, store, fingerprint, multi-noise, router)."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from gr00t_lancedb.vl_cache.codec import (
    CacheFingerprint,
    compute_pooled_visual_emb,
    decode_features_numpy,
    encode_features,
    pack_bool_mask,
    strip_padding,
    unpack_bool_mask,
)
from gr00t_lancedb.vl_cache.accel import estimate_cost_route


def test_strip_padding_left_and_right() -> None:
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((12, 16)).astype(np.float32)

    # Left-padded (4 pad + 8 valid)
    attn_left = np.array([False] * 4 + [True] * 8)
    img_left = np.array([False] * 4 + [True] * 5 + [False] * 3)
    vf, vim, side = strip_padding(feats, attn_left, img_left)
    assert side == "left"
    assert vf.shape == (8, 16)
    assert np.array_equal(vf, feats[4:])
    assert np.array_equal(vim, img_left[4:])

    # Right-padded (7 valid + 5 pad)
    attn_right = np.array([True] * 7 + [False] * 5)
    img_right = np.array([True] * 4 + [False] * 8)
    vf2, vim2, side2 = strip_padding(feats, attn_right, img_right)
    assert side2 == "right"
    assert vf2.shape == (7, 16)
    assert np.array_equal(vf2, feats[:7])
    assert np.array_equal(vim2, img_right[:7])


def test_per_token_int8_and_bf16_roundtrip() -> None:
    rng = np.random.default_rng(7)
    feats = (rng.standard_normal((190, 256)) * 2.5).astype(np.float32)

    # Per-token INT8
    sc_b, pl_b = encode_features(feats, codec="per_token_int8")
    assert len(sc_b) == 190 * 2
    assert len(pl_b) == 190 * 256
    recon = decode_features_numpy(sc_b, pl_b, 190, 256, codec="per_token_int8")
    rel_l2 = np.linalg.norm(recon - feats) / np.linalg.norm(feats)
    assert rel_l2 < 0.015  # Typically ~0.008 for Gaussian per-token int8

    # BF16 raw
    sc_b2, pl_b2 = encode_features(feats, codec="bf16_raw")
    assert sc_b2 == b""
    assert len(pl_b2) == 190 * 256 * 2
    recon2 = decode_features_numpy(sc_b2, pl_b2, 190, 256, codec="bf16_raw")
    rel_l2_bf16 = np.linalg.norm(recon2 - feats) / np.linalg.norm(feats)
    assert rel_l2_bf16 < 0.003


def test_pack_unpack_bool_mask_and_pooled_emb() -> None:
    mask = np.array([True, False, True, True, False, False, True, False, True, True], dtype=bool)
    packed = pack_bool_mask(mask)
    assert len(packed) == 2
    unpacked = unpack_bool_mask(packed, len(mask))
    assert np.array_equal(unpacked, mask)

    feats = np.ones((10, 8), dtype=np.float32) * 3.0
    emb = compute_pooled_visual_emb(feats, mask)
    assert emb.shape == (8,)
    assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-5)


def test_fingerprint_determinism_and_sensitivity() -> None:
    fp1 = CacheFingerprint(base_model_path="nvidia/GR00T-N1.7-3B", select_layer=16, hidden_dim=2048)
    fp2 = CacheFingerprint(base_model_path="nvidia/GR00T-N1.7-3B", select_layer=16, hidden_dim=2048)
    fp3 = CacheFingerprint(base_model_path="nvidia/GR00T-N1.7-3B", select_layer=12, hidden_dim=2048)
    assert fp1.sha256() == fp2.sha256()
    assert fp1.sha256() != fp3.sha256()

    parsed, sha = CacheFingerprint.from_metadata_json(fp1.to_metadata_json())
    assert sha == fp1.sha256()
    assert parsed.select_layer == 16


def test_cost_router_decisions() -> None:
    # LIBERO-10 (100k frames), 20k steps @ B=32 (reuse = 6.4x) -> vl_cache
    d1 = estimate_cost_route(num_frames=100_000, total_steps=20_000, batch_size=32, available_disk_gb=200.0)
    assert d1.route == "vl_cache"
    assert d1.estimated_speedup > 1.5

    # Unfrozen backbone -> online_multi_noise
    d2 = estimate_cost_route(num_frames=100_000, total_steps=20_000, batch_size=32, tune_backbone=True)
    assert d2.route == "online_multi_noise"

    # 300 GB tier (50M frames) -> disk overflow -> online_multi_noise
    d3 = estimate_cost_route(num_frames=50_000_000, total_steps=20_000, batch_size=32, available_disk_gb=500.0)
    assert d3.route == "online_multi_noise"


def test_lance_store_write_read_and_fail_closed(tmp_path: Path) -> None:
    lancedb = pytest.importorskip("lancedb")
    _ = lancedb
    from gr00t_lancedb.vl_cache.store import VLCacheReader, VLCacheWriter

    fp = CacheFingerprint(
        base_model_path="test/model",
        select_layer=16,
        hidden_dim=32,
        quant_codec="per_token_int8",
    )
    writer = VLCacheWriter(tmp_path, fp, flush_rows=4)
    rng = np.random.default_rng(123)
    for i in range(10):
        L = 12 + (i % 5)
        feats = rng.standard_normal((16, 32)).astype(np.float32)
        attn = np.array([False] * (16 - L) + [True] * L)
        img = np.array([False] * (16 - L) + [True] * 6 + [False] * (L - 6))
        writer.add_sample(
            row_key=i,
            episode_index=i // 5,
            frame_index=i % 5,
            view_id=0,
            features=feats,
            attention_mask=attn,
            image_mask=img,
        )
    summary = writer.close()
    assert summary["rows_written"] == 10

    reader = VLCacheReader(tmp_path, fp.table_name(), expected_fingerprint=fp)
    assert reader.num_rows == 10
    rows = reader.take_rows([9, 0, 4])
    assert [r.seq_len for r in rows] == [12 + (9 % 5), 12, 12 + 4]

    # Fail-closed on mismatched fingerprint
    bad_fp = CacheFingerprint(
        base_model_path="test/model",
        select_layer=12,
        hidden_dim=32,
        quant_codec="per_token_int8",
    )
    bad_reader = VLCacheReader(tmp_path, fp.table_name(), expected_fingerprint=bad_fp)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        _ = bad_reader.num_rows
