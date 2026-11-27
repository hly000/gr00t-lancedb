# SPDX-License-Identifier: Apache-2.0
"""LanceDB storage table writer/reader for compressed frozen VLM features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyarrow as pa

from .codec import (
    CACHE_META_KEY,
    CacheFingerprint,
    PadSide,
    QuantCodec,
    compute_pooled_visual_emb,
    dequantize_features_torch,
    encode_features,
    pack_bool_mask,
    strip_padding,
    unpack_bool_mask,
)


def make_vl_cache_schema(
    hidden_dim: int,
    fingerprint: CacheFingerprint,
    include_pooled_emb: bool = True,
) -> pa.Schema:
    """Build the Arrow schema for ``vl_cache_<fp8>.lance`` with embedded fingerprint metadata."""
    fields: list[pa.Field] = [
        pa.field("row_key", pa.int64(), nullable=False),
        pa.field("episode_index", pa.int32(), nullable=False),
        pa.field("frame_index", pa.int32(), nullable=False),
        pa.field("view_id", pa.int8(), nullable=False),
        pa.field("seq_len", pa.int16(), nullable=False),
        pa.field("image_mask", pa.binary(), nullable=False),
        pa.field("feat_scale", pa.binary(), nullable=False),
        pa.field("feat_payload", pa.binary(), nullable=False),
    ]
    if include_pooled_emb:
        fields.append(pa.field("pooled_visual_emb", pa.list_(pa.float32(), hidden_dim), nullable=True))
    return pa.schema(fields, metadata={CACHE_META_KEY: fingerprint.to_metadata_json().encode("utf-8")})


class VLCacheWriter:
    """Streaming writer that materializes frozen VLM outputs into a Lance table.

    Rows are buffered in batches and flushed to ``<db_uri>/<vl_cache_fp8>.lance`` so memory
    stays bounded even for multi-million-frame datasets.
    """

    def __init__(
        self,
        db_uri: str | Path,
        fingerprint: CacheFingerprint,
        *,
        table_name: str | None = None,
        include_pooled_emb: bool = True,
        flush_rows: int = 1024,
        overwrite: bool = True,
        storage_options: dict[str, str] | None = None,
    ) -> None:
        self.db_uri = str(db_uri)
        self.fingerprint = fingerprint
        self.table_name = table_name or fingerprint.table_name()
        self.include_pooled_emb = include_pooled_emb
        self.flush_rows = max(1, int(flush_rows))
        self.overwrite = overwrite
        self.storage_options = storage_options
        self.schema = make_vl_cache_schema(
            hidden_dim=fingerprint.hidden_dim,
            fingerprint=fingerprint,
            include_pooled_emb=include_pooled_emb,
        )

        self._db: Any = None
        self._table: Any = None
        self._buf: dict[str, list[Any]] = self._empty_buffer()
        self.rows_written: int = 0
        self.payload_bytes_written: int = 0

    def _empty_buffer(self) -> dict[str, list[Any]]:
        buf: dict[str, list[Any]] = {
            "row_key": [],
            "episode_index": [],
            "frame_index": [],
            "view_id": [],
            "seq_len": [],
            "image_mask": [],
            "feat_scale": [],
            "feat_payload": [],
        }
        if self.include_pooled_emb:
            buf["pooled_visual_emb"] = []
        return buf

    def _connect(self) -> Any:
        if self._db is None:
            import lancedb

            kwargs: dict[str, Any] = {}
            if self.storage_options:
                kwargs["storage_options"] = self.storage_options
            self._db = lancedb.connect(self.db_uri, **kwargs)
        return self._db

    def add_sample(
        self,
        *,
        row_key: int,
        episode_index: int,
        frame_index: int,
        view_id: int,
        features: np.ndarray,
        attention_mask: np.ndarray,
        image_mask: np.ndarray,
    ) -> None:
        """Strip padding, compress, and buffer one frame's VLM output."""
        valid_feat, valid_img, _ = strip_padding(features, attention_mask, image_mask)
        seq_len = int(valid_feat.shape[0])
        scale_bytes, payload_bytes = encode_features(valid_feat, codec=self.fingerprint.quant_codec)
        img_bytes = pack_bool_mask(valid_img)

        self._buf["row_key"].append(int(row_key))
        self._buf["episode_index"].append(int(episode_index))
        self._buf["frame_index"].append(int(frame_index))
        self._buf["view_id"].append(int(view_id))
        self._buf["seq_len"].append(seq_len)
        self._buf["image_mask"].append(img_bytes)
        self._buf["feat_scale"].append(scale_bytes)
        self._buf["feat_payload"].append(payload_bytes)
        if self.include_pooled_emb:
            self._buf["pooled_visual_emb"].append(compute_pooled_visual_emb(valid_feat, valid_img))

        self.payload_bytes_written += len(payload_bytes) + len(scale_bytes) + len(img_bytes)
        if len(self._buf["row_key"]) >= self.flush_rows:
            self.flush()

    def add_batch(
        self,
        *,
        row_keys: Sequence[int],
        episode_indices: Sequence[int],
        frame_indices: Sequence[int],
        view_ids: Sequence[int],
        features: np.ndarray,
        attention_mask: np.ndarray,
        image_mask: np.ndarray,
    ) -> None:
        """Add a batch ``[B, T, D]`` of VLM outputs."""
        bsz = len(row_keys)
        for i in range(bsz):
            self.add_sample(
                row_key=int(row_keys[i]),
                episode_index=int(episode_indices[i]),
                frame_index=int(frame_indices[i]),
                view_id=int(view_ids[i]),
                features=features[i],
                attention_mask=attention_mask[i],
                image_mask=image_mask[i],
            )

    def flush(self) -> None:
        n = len(self._buf["row_key"])
        if n == 0:
            return
        db = self._connect()
        arrays: list[pa.Array] = [
            pa.array(self._buf["row_key"], type=pa.int64()),
            pa.array(self._buf["episode_index"], type=pa.int32()),
            pa.array(self._buf["frame_index"], type=pa.int32()),
            pa.array(self._buf["view_id"], type=pa.int8()),
            pa.array(self._buf["seq_len"], type=pa.int16()),
            pa.array(self._buf["image_mask"], type=pa.binary()),
            pa.array(self._buf["feat_scale"], type=pa.binary()),
            pa.array(self._buf["feat_payload"], type=pa.binary()),
        ]
        if self.include_pooled_emb:
            stacked = np.stack(self._buf["pooled_visual_emb"], axis=0).astype(np.float32, copy=False)
            flat = pa.array(stacked.reshape(-1), type=pa.float32())
            arrays.append(pa.FixedSizeListArray.from_arrays(flat, self.fingerprint.hidden_dim))

        pa_table = pa.Table.from_arrays(arrays, schema=self.schema)
        if self._table is None:
            mode = "overwrite" if self.overwrite else "append"
            self._table = db.create_table(self.table_name, data=pa_table, schema=self.schema, mode=mode)
        else:
            self._table.add(pa_table)
        self.rows_written += n
        self._buf = self._empty_buffer()

    def close(self) -> dict[str, Any]:
        self.flush()
        return {
            "db_uri": self.db_uri,
            "table_name": self.table_name,
            "fingerprint": self.fingerprint.sha256(),
            "rows_written": self.rows_written,
            "payload_bytes_written": self.payload_bytes_written,
            "avg_bytes_per_row": (self.payload_bytes_written / self.rows_written) if self.rows_written else 0.0,
        }


@dataclass
class EncodedVLRow:
    """Raw compact row fetched from Lance before GPU dequantization."""

    seq_len: int
    image_mask: np.ndarray  # [L] bool
    feat_scale: np.ndarray | None  # [L] fp16 or None
    feat_payload: np.ndarray  # [L, D] int8 or uint16(bf16)


class VLCacheReader:
    """Fast random-access reader for ``vl_cache_<fp8>.lance`` using Lance ``take()``.

    Fork-safe: opens the underlying Lance dataset lazily on first read inside each
    DataLoader worker. Validates ``cache_fingerprint`` fail-closed on open.
    """

    def __init__(
        self,
        db_uri: str | Path,
        table_name: str,
        *,
        expected_fingerprint: CacheFingerprint | None = None,
        storage_options: dict[str, str] | None = None,
    ) -> None:
        self.db_uri = str(db_uri)
        self.table_name = table_name
        self.expected_fingerprint = expected_fingerprint
        self.storage_options = storage_options

        self._table: Any = None
        self._lance_ds: Any = None
        self.fingerprint: CacheFingerprint | None = None
        self._key_to_row: dict[tuple[int, int, int], int] | None = None

    def _ensure_open(self) -> None:
        if self._lance_ds is not None:
            return
        import lancedb

        kwargs: dict[str, Any] = {}
        if self.storage_options:
            kwargs["storage_options"] = self.storage_options
        db = lancedb.connect(self.db_uri, **kwargs)
        self._table = db.open_table(self.table_name)
        self._lance_ds = self._table.to_lance()

        meta = self._lance_ds.schema.metadata or {}
        if CACHE_META_KEY not in meta:
            raise ValueError(
                f"Table {self.table_name!r} in {self.db_uri!r} is missing {CACHE_META_KEY!r} metadata."
            )
        fp, stored_sha = CacheFingerprint.from_metadata_json(meta[CACHE_META_KEY])
        if fp.sha256() != stored_sha:
            raise ValueError(
                f"Corrupted vl_cache metadata in {self.table_name!r}: "
                f"computed {fp.sha256()} != stored {stored_sha}"
            )
        if self.expected_fingerprint is not None:
            exp_sha = self.expected_fingerprint.sha256()
            if exp_sha != stored_sha:
                raise ValueError(
                    f"vl_cache fingerprint mismatch! Expected {exp_sha} "
                    f"({self.expected_fingerprint.canonical_dict()}), "
                    f"got {stored_sha} ({fp.canonical_dict()})."
                )
        self.fingerprint = fp

    @property
    def num_rows(self) -> int:
        self._ensure_open()
        return int(self._lance_ds.count_rows())

    def build_key_index(self) -> dict[tuple[int, int, int], int]:
        """Build an in-memory ``(episode_index, frame_index, view_id) -> physical_row_idx`` map."""
        self._ensure_open()
        if self._key_to_row is None:
            tbl = self._lance_ds.to_table(columns=["episode_index", "frame_index", "view_id"])
            eps = tbl["episode_index"].to_numpy(zero_copy_only=False)
            frs = tbl["frame_index"].to_numpy(zero_copy_only=False)
            vws = tbl["view_id"].to_numpy(zero_copy_only=False)
            self._key_to_row = {
                (int(e), int(f), int(v)): idx for idx, (e, f, v) in enumerate(zip(eps, frs, vws))
            }
        return self._key_to_row

    def take_rows(self, row_indices: Sequence[int]) -> list[EncodedVLRow]:
        """Fetch a batch of rows by physical row index using Lance's O(1) random ``take()``."""
        self._ensure_open()
        assert self.fingerprint is not None
        cols = ["seq_len", "image_mask", "feat_scale", "feat_payload"]
        pa_tbl = self._lance_ds.take(list(row_indices), columns=cols)

        seq_lens = pa_tbl["seq_len"].to_pylist()
        img_masks = pa_tbl["image_mask"].to_pylist()
        scales = pa_tbl["feat_scale"].to_pylist()
        payloads = pa_tbl["feat_payload"].to_pylist()

        codec = self.fingerprint.quant_codec
        hidden_dim = self.fingerprint.hidden_dim
        rows: list[EncodedVLRow] = []
        for L, im_b, sc_b, pl_b in zip(seq_lens, img_masks, scales, payloads):
            L = int(L)
            im = unpack_bool_mask(im_b, L)
            if codec == "per_token_int8":
                sc = np.frombuffer(sc_b, dtype=np.float16).copy()
                pl = np.frombuffer(pl_b, dtype=np.int8).reshape(L, hidden_dim).copy()
            elif codec == "bf16_raw":
                sc = None
                pl = np.frombuffer(pl_b, dtype=np.uint16).reshape(L, hidden_dim).copy()
            else:
                raise ValueError(f"Unsupported codec {codec!r}")
            rows.append(EncodedVLRow(seq_len=L, image_mask=im, feat_scale=sc, feat_payload=pl))
        return rows

    def take_by_keys(self, keys: Sequence[tuple[int, int, int]]) -> list[EncodedVLRow]:
        """Fetch a batch of rows by ``(episode_index, frame_index, view_id)``."""
        index = self.build_key_index()
        row_ids = [index[k] for k in keys]
        return self.take_rows(row_ids)


def collate_cached_vl_batch(
    rows: Sequence[EncodedVLRow],
    *,
    codec: QuantCodec = "per_token_int8",
    pad_side: PadSide = "left",
    dequantize_on_cpu: bool = False,
    device: Any | None = None,
    target_dtype: Any | None = None,
) -> dict[str, Any]:
    """Dynamically pad variable-length ``EncodedVLRow`` items into a batch dict.

    If ``dequantize_on_cpu=False`` and ``device`` is provided, transfers compact ``int8``
    payloads to GPU and dequantizes directly into ``target_dtype`` (default ``bf16``).
    """
    import torch

    bsz = len(rows)
    if bsz == 0:
        raise ValueError("Cannot collate an empty batch")
    max_len = max(r.seq_len for r in rows)
    hidden_dim = int(rows[0].feat_payload.shape[1])

    attn_mask = np.zeros((bsz, max_len), dtype=bool)
    img_mask = np.zeros((bsz, max_len), dtype=bool)

    if codec == "per_token_int8":
        payload_np = np.zeros((bsz, max_len, hidden_dim), dtype=np.int8)
        scale_np = np.zeros((bsz, max_len), dtype=np.float16)
    elif codec == "bf16_raw":
        payload_np = np.zeros((bsz, max_len, hidden_dim), dtype=np.uint16)
        scale_np = None
    else:
        raise ValueError(f"Unsupported codec {codec!r}")

    for i, r in enumerate(rows):
        L = r.seq_len
        if pad_side == "left":
            sl = slice(max_len - L, max_len)
        else:
            sl = slice(0, L)
        attn_mask[i, sl] = True
        img_mask[i, sl] = r.image_mask
        payload_np[i, sl, :] = r.feat_payload
        if scale_np is not None and r.feat_scale is not None:
            scale_np[i, sl] = r.feat_scale

    if codec == "bf16_raw":
        payload_t = torch.from_numpy(payload_np.view(np.int16))
    else:
        payload_t = torch.from_numpy(payload_np)
    scale_t = torch.from_numpy(scale_np) if scale_np is not None else None
    attn_t = torch.from_numpy(attn_mask)
    img_t = torch.from_numpy(img_mask)

    if device is not None:
        payload_t = payload_t.to(device, non_blocking=True)
        if scale_t is not None:
            scale_t = scale_t.to(device, non_blocking=True)
        attn_t = attn_t.to(device, non_blocking=True)
        img_t = img_t.to(device, non_blocking=True)
        feats_t = dequantize_features_torch(payload_t, scale_t, codec=codec, target_dtype=target_dtype)
        return {
            "backbone_features": feats_t,
            "backbone_attention_mask": attn_t,
            "image_mask": img_t,
        }

    if dequantize_on_cpu:
        feats_t = dequantize_features_torch(payload_t, scale_t, codec=codec, target_dtype=target_dtype)
        return {
            "backbone_features": feats_t,
            "backbone_attention_mask": attn_t,
            "image_mask": img_t,
        }

    return {
        "feat_payload": payload_t,
        "feat_scale": scale_t,
        "backbone_attention_mask": attn_t,
        "image_mask": img_t,
        "quant_codec": codec,
    }
