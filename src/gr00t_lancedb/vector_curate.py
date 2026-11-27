# SPDX-License-Identifier: Apache-2.0
"""Embedding columns, IVF-PQ vector index, semantic dedup and hybrid search on ``frames.lance``.

Everything lives in the *same* Lance table GR00T trains from:

* :func:`add_embedding_column` appends a ``FixedSizeList<float32, D>`` column (one row per frame,
  ``NULL`` for frames that were not embedded) via LanceDB zero-copy column evolution
  (``add_columns``). Only tabular columns are read for the default kinematic embedding, and
  ``videos.lance`` is never written (bytes written per table are measured and reported).
* :func:`build_vector_index` builds an IVF-PQ (cosine) index sized from row count and dimension,
  falling back to exact flat search for tiny tables (and recording which one is in use).
* :func:`tag_semantic_duplicates` finds near-duplicate frames (cross-episode kNN through the
  vector index and/or temporally static runs inside an episode), keeps exactly one
  deterministic representative per cluster, and writes ``is_semantic_dup`` /
  ``semantic_cluster_id`` / ``semantic_nn_dist`` columns that compose with curation columns in
  SQL, e.g. ``where="is_valid AND NOT is_semantic_dup"``.
* :func:`hybrid_search` / :func:`slice_episodes_by_similarity` / :func:`mine_rare_frames` run
  vector queries with an SQL prefilter and return ``(episode_index, frame_index)`` anchors that
  convert into a ``where`` predicate for ``LanceShardedSingleStepDataset(where=...)``.

Because the filter only selects *anchor* frames, the GR00T loader still reads the full episode
around each anchor, so action horizons and video ``delta_indices`` are never cut.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
import inspect
import json
import logging
import math
from pathlib import Path
import time
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
import pyarrow as pa

from lerobot_lancedb import FRAMES_TABLE, META_TABLE, VIDEOS_TABLE, to_lance_column
from lerobot_lancedb._source import is_remote_uri

from .metadata import DEFAULT_COLUMN_NAMES, LEROBOT_MODALITY_FILENAME

logger = logging.getLogger(__name__)

EMBEDDING_COLUMN = "embedding"
EMBEDDING_FIELD_METADATA_KEY = b"gr00t_lancedb.embedding"
EMBEDDING_FORMAT = "gr00t_lancedb.embedding/v1"
SEMANTIC_DUP_COLUMNS: tuple[str, ...] = ("is_semantic_dup", "semantic_cluster_id", "semantic_nn_dist")
DISTANCE_TYPE = "cosine"
NEVER_TRUE_PREDICATE = "1 = 0"

DedupScope = Literal["cross_episode", "within_episode", "all"]
SearchMode = Literal["auto", "index", "flat"]


# ---------------------------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------------------------
def _connect(lance_root: str | Path, storage_options: Mapping[str, str] | None) -> Any:
    import lancedb

    uri = str(lance_root)
    if storage_options:
        return lancedb.connect(uri, storage_options=dict(storage_options))
    return lancedb.connect(uri)


def _open_frames(
    lance_root: str | Path,
    storage_options: Mapping[str, str] | None = None,
    lance_version: int | None = None,
) -> tuple[Any, Any]:
    db = _connect(lance_root, storage_options)
    kw = {"storage_options": dict(storage_options)} if storage_options else {}
    table = db.open_table(FRAMES_TABLE, **kw)
    if lance_version is not None:
        table.checkout(int(lance_version))
    return db, table


def _local_root(lance_root: str | Path) -> Path | None:
    if is_remote_uri(lance_root):
        return None
    return Path(lance_root).expanduser().resolve()


def _snapshot_tree(root: Path) -> dict[str, tuple[int, int]]:
    """``{relative_posix_path: (size, mtime_ns)}`` for every file under ``root``."""
    if not root.exists():
        return {}
    out: dict[str, tuple[int, int]] = {}
    for f in root.rglob("*"):
        if f.is_file():
            st = f.stat()
            out[f.relative_to(root).as_posix()] = (st.st_size, st.st_mtime_ns)
    return out


class _WriteMeter:
    """Measure bytes written per top-level entry of a local dataset root (``frames.lance``, ...).

    Remote roots cannot be measured from here; :meth:`stop` then reports zeros with
    ``bytes_measured=False`` (this module never opens ``videos.lance`` for writing either way).
    """

    def __init__(self, lance_root: str | Path) -> None:
        self.root = _local_root(lance_root)
        self._before = _snapshot_tree(self.root) if self.root is not None else {}

    def stop(self) -> dict[str, Any]:
        by_entry: dict[str, int] = {}
        files_by_entry: dict[str, int] = {}
        if self.root is not None:
            after = _snapshot_tree(self.root)
            for rel, sig in after.items():
                if self._before.get(rel) != sig:
                    top = rel.split("/", 1)[0]
                    by_entry[top] = by_entry.get(top, 0) + sig[0]
                    files_by_entry[top] = files_by_entry.get(top, 0) + 1
        frames_key, videos_key, meta_key = f"{FRAMES_TABLE}.lance", f"{VIDEOS_TABLE}.lance", f"{META_TABLE}.lance"
        known = {frames_key, videos_key, meta_key}
        return {
            "bytes_measured": self.root is not None,
            "bytes_written_frames": int(by_entry.get(frames_key, 0)),
            "bytes_written_videos": int(by_entry.get(videos_key, 0)),
            "bytes_written_meta": int(by_entry.get(meta_key, 0)),
            "bytes_written_other": int(sum(v for k, v in by_entry.items() if k not in known)),
            "files_written_frames": int(files_by_entry.get(frames_key, 0)),
            "files_written_videos": int(files_by_entry.get(videos_key, 0)),
            "bytes_written_by_table": dict(sorted(by_entry.items())),
        }


def _col_to_numpy(tbl: pa.Table, name: str, dtype: Any) -> np.ndarray:
    return tbl.column(name).combine_chunks().to_numpy(zero_copy_only=False).astype(dtype, copy=False)


def _arrow_col_to_2d_float64(col: pa.ChunkedArray | pa.Array) -> np.ndarray:
    """FixedSizeList / List / scalar numeric Arrow column -> contiguous ``(N, W)`` float64."""
    arr = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
    n = len(arr)
    if n == 0:
        return np.empty((0, 0), dtype=np.float64)
    if pa.types.is_fixed_size_list(arr.type):
        width = arr.type.list_size
        vals = arr.values.to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
        if arr.offset or len(vals) != n * width:
            vals = arr.flatten().to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
        return vals.reshape(n, width)
    if pa.types.is_list(arr.type) or pa.types.is_large_list(arr.type):
        flat = arr.flatten().to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
        if len(flat) % n == 0:
            return flat.reshape(n, len(flat) // n)
        return np.vstack([np.asarray(x, dtype=np.float64).reshape(-1) for x in arr.to_pylist()])
    return arr.to_numpy(zero_copy_only=False).astype(np.float64, copy=False).reshape(n, 1)


def _fsl_to_float32(col: pa.ChunkedArray | pa.Array) -> tuple[np.ndarray, np.ndarray]:
    """FixedSizeList<float32, D> (nullable) -> ``(values[N, D] float32, valid[N] bool)``."""
    arr = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
    if not pa.types.is_fixed_size_list(arr.type):
        raise TypeError(f"Embedding column must be FixedSizeList<float32, D>, got {arr.type}")
    n, dim = len(arr), arr.type.list_size
    valid = ~np.asarray(arr.is_null().to_numpy(zero_copy_only=False), dtype=bool)
    # ``.values`` ignores the parent offset; the child slice for a FixedSizeList is offset*dim.
    child = arr.values.slice(arr.offset * dim, n * dim)
    vals = np.asarray(child.to_numpy(zero_copy_only=False), dtype=np.float32).reshape(n, dim)
    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
    vals[~valid] = 0.0
    return np.ascontiguousarray(vals), valid


def _matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``a @ b`` without numpy>=2 + Apple Accelerate's spurious FP warnings (results are exact)."""
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return a @ b


def _l2_normalize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row-normalize; returns ``(normalized, ok_mask)`` where non-finite / zero rows are not ok."""
    x = np.asarray(x, dtype=np.float32)
    finite = np.all(np.isfinite(x), axis=1) if x.size else np.zeros(len(x), dtype=bool)
    norms = np.linalg.norm(np.where(finite[:, None], x, 0.0), axis=1) if x.size else np.zeros(len(x))
    ok = finite & (norms > 1e-12)
    out = np.zeros_like(x, dtype=np.float32)
    out[ok] = x[ok] / norms[ok, None]
    return out, ok


def _episode_order(episode_index: np.ndarray, frame_index: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(order, starts, ends)``: rows sorted by (episode, frame) and per-episode spans."""
    order = np.lexsort((frame_index, episode_index))
    if len(order) == 0:
        empty = np.empty(0, dtype=np.int64)
        return order, empty, empty
    ep_sorted = episode_index[order]
    _, starts = np.unique(ep_sorted, return_index=True)
    ends = np.append(starts[1:], len(order))
    return order, starts.astype(np.int64), ends.astype(np.int64)


def _position_in_episode(episode_index: np.ndarray, frame_index: np.ndarray) -> np.ndarray:
    """0-based position of each row inside its episode (sorted by ``frame_index``)."""
    order, starts, ends = _episode_order(episode_index, frame_index)
    pos = np.empty(len(order), dtype=np.int64)
    for s, e in zip(starts, ends, strict=True):
        pos[order[s:e]] = np.arange(e - s, dtype=np.int64)
    return pos


def _frame_keys(episode_index: np.ndarray, frame_index: np.ndarray, stride: int) -> np.ndarray:
    return episode_index.astype(np.int64) * int(stride) + frame_index.astype(np.int64)


def _and(*preds: str | None) -> str | None:
    parts = [p.strip() for p in preds if p and p.strip()]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return " AND ".join(f"({p})" for p in parts)


def combine_where(*predicates: str | None) -> str | None:
    """AND together SQL predicates (``None`` / empty are ignored)."""
    return _and(*predicates)


def episodes_to_where(episodes: Sequence[int]) -> str:
    """``episode_index IN (...)`` predicate (never-true predicate for an empty list)."""
    eps = sorted({int(e) for e in episodes})
    if not eps:
        return NEVER_TRUE_PREDICATE
    return f"episode_index IN ({', '.join(str(e) for e in eps)})"


def anchors_to_where(anchors: Mapping[int, Sequence[int] | np.ndarray]) -> str:
    """Frame-level predicate selecting exactly the given ``{episode_index: frame_indices}``."""
    parts: list[str] = []
    for ep in sorted(int(e) for e in anchors):
        frames = sorted({int(f) for f in np.asarray(anchors[ep]).reshape(-1)})
        if frames:
            parts.append(f"(episode_index = {ep} AND frame_index IN ({', '.join(str(f) for f in frames)}))")
    if not parts:
        return NEVER_TRUE_PREDICATE
    return " OR ".join(parts)


def _read_modality(db: Any, lance_root: str | Path, storage_options: Mapping[str, str] | None) -> dict[str, Any]:
    kw = {"storage_options": dict(storage_options)} if storage_options else {}
    try:
        meta = db.open_table(META_TABLE, **kw)
        tbl = (
            meta.search()
            .where(f"path = '{LEROBOT_MODALITY_FILENAME}'")
            .select(["path", "data"])
            .limit(None)
            .to_arrow()
        )
        if tbl.num_rows:
            return json.loads(bytes(tbl.column("data")[0].as_py()).decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - fall back to the local meta/ directory
        logger.debug("Could not read %s from %s.lance: %s", LEROBOT_MODALITY_FILENAME, META_TABLE, exc)
    local = _local_root(lance_root)
    if local is not None and (local / "meta" / LEROBOT_MODALITY_FILENAME).exists():
        return json.loads((local / "meta" / LEROBOT_MODALITY_FILENAME).read_text(encoding="utf-8"))
    raise FileNotFoundError(f"No {LEROBOT_MODALITY_FILENAME} in {META_TABLE}.lance or meta/ at {lance_root}")


def _modality_original_keys(modality: Mapping[str, Any], section: str) -> list[str]:
    keys: list[str] = []
    for group in modality.get(section, {}).values():
        k = group.get("original_key") or DEFAULT_COLUMN_NAMES[section]
        if k not in keys:
            keys.append(k)
    return keys


def _embedding_field_info(table: Any, column: str) -> dict[str, Any]:
    if column not in table.schema.names:
        raise KeyError(
            f"Column {column!r} not found on {FRAMES_TABLE}.lance; run add_embedding_column() first."
        )
    fld = table.schema.field(column)
    if not pa.types.is_fixed_size_list(fld.type):
        raise TypeError(f"Column {column!r} must be FixedSizeList<float32, D>, got {fld.type}")
    info: dict[str, Any] = {}
    md = fld.metadata or {}
    if EMBEDDING_FIELD_METADATA_KEY in md:
        try:
            info = json.loads(md[EMBEDDING_FIELD_METADATA_KEY].decode("utf-8"))
        except Exception:  # noqa: BLE001
            info = {}
    info.setdefault("dim", int(fld.type.list_size))
    info.setdefault("every_n", 1)
    return info


def _indices_on(table: Any, column: str) -> list[Any]:
    out = []
    try:
        for idx in table.list_indices():
            if list(getattr(idx, "columns", [])) == [column]:
                out.append(idx)
    except Exception as exc:  # noqa: BLE001
        logger.debug("list_indices failed: %s", exc)
    return out


def _drop_columns_and_indices(table: Any, columns: Sequence[str]) -> list[str]:
    present = [c for c in columns if c in table.schema.names]
    for c in present:
        for idx in _indices_on(table, c):
            try:
                table.drop_index(idx.name)
            except Exception as exc:  # noqa: BLE001 - dropping the column removes it anyway
                logger.debug("drop_index(%s) failed: %s", idx.name, exc)
    if present:
        table.drop_columns(present)
    return present


def _vector_index_state(table: Any, column: str) -> dict[str, Any] | None:
    """Return ``{name, index_type, num_indexed_rows, ...}`` for a usable vector index, else ``None``."""
    for idx in _indices_on(table, column):
        itype = str(getattr(idx, "index_type", ""))
        if not itype.lower().startswith(("ivf", "hnsw")):
            continue
        try:
            st = table.index_stats(idx.name)
        except Exception:  # noqa: BLE001
            st = None
        n_idx = int(getattr(st, "num_indexed_rows", 0) or 0) if st is not None else 0
        if n_idx > 0:
            return {
                "name": idx.name,
                "index_type": getattr(st, "index_type", itype),
                "distance_type": getattr(st, "distance_type", None),
                "num_indexed_rows": n_idx,
                "num_unindexed_rows": int(getattr(st, "num_unindexed_rows", 0) or 0),
            }
    return None


def _resolve_search_mode(table: Any, column: str, mode: SearchMode) -> tuple[str, dict[str, Any] | None]:
    state = _vector_index_state(table, column)
    if mode == "flat":
        return "flat", state
    if mode == "index":
        if state is None:
            raise RuntimeError(f"mode='index' but no populated vector index exists on {column!r}")
        return "ivf_pq", state
    return ("ivf_pq" if state is not None else "flat"), state


def _knn_query(
    table: Any,
    column: str,
    queries: np.ndarray,
    *,
    where: str | None,
    k: int,
    mode: str,
    nprobes: int,
    refine_factor: int | None,
    select: Sequence[str],
) -> pa.Table:
    """Batched cosine kNN through LanceDB (one call for all ``queries``); adds ``query_index``."""
    q = np.ascontiguousarray(np.asarray(queries, dtype=np.float32).reshape(len(queries), -1))
    builder = (
        table.search(q, vector_column_name=column)
        .distance_type(DISTANCE_TYPE)
        .limit(int(k))
        .select([*select, "_distance"])
    )
    if where:
        builder = builder.where(where, prefilter=True)
    if mode == "flat":
        builder = builder.bypass_vector_index()
    else:
        builder = builder.nprobes(int(nprobes))
        if refine_factor:
            builder = builder.refine_factor(int(refine_factor))
    res = builder.to_arrow()
    if "query_index" not in res.schema.names:
        res = res.append_column("query_index", pa.array(np.zeros(res.num_rows, dtype=np.int32)))
    return res


# ---------------------------------------------------------------------------------------------
# Embedding providers
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FrameBatch:
    """Tabular view of ``frames.lance`` handed to an :class:`EmbeddingProvider`.

    All arrays are aligned with the table's physical row order. ``rows`` lists the positions the
    provider must embed (all frames for ``every_n=1``, keyframes otherwise); the full arrays are
    passed so providers can look at temporal context (e.g. the next ``k`` actions).
    """

    lance_root: str
    storage_options: dict[str, str] | None
    index: np.ndarray
    episode_index: np.ndarray
    frame_index: np.ndarray
    state: np.ndarray
    action: np.ndarray
    rows: np.ndarray
    modality_meta: dict[str, Any] = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return int(len(self.index))


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Anything that maps a :class:`FrameBatch` to ``float32[len(batch.rows), D]``.

    Rows with non-finite or all-zero output are stored as ``NULL`` (excluded from search/dedup).
    """

    name: str

    def __call__(self, batch: FrameBatch) -> np.ndarray: ...


class FunctionEmbeddingProvider:
    """Adapter for a plain ``fn(episode_index[], frame_index[], tabular) -> float32[N, D]``.

    ``tabular`` holds ``index``, ``state`` and ``action`` for the rows being embedded.
    """

    def __init__(self, fn: Callable[[np.ndarray, np.ndarray, dict[str, np.ndarray]], np.ndarray], name: str | None = None) -> None:
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "function")

    def __call__(self, batch: FrameBatch) -> np.ndarray:
        r = batch.rows
        tabular = {"index": batch.index[r], "state": batch.state[r], "action": batch.action[r]}
        return np.asarray(self.fn(batch.episode_index[r], batch.frame_index[r], tabular), dtype=np.float32)

    def config(self) -> dict[str, Any]:
        return {"function": self.name}


def _as_provider(obj: Any) -> Any:
    if obj is None:
        return KinematicEmbeddingProvider()
    if isinstance(obj, (KinematicEmbeddingProvider, VisualEmbeddingProvider, FunctionEmbeddingProvider)):
        return obj
    if callable(obj):
        try:
            n_params = len(inspect.signature(obj).parameters)
        except (TypeError, ValueError):
            n_params = 1
        if n_params >= 3:
            return FunctionEmbeddingProvider(obj)
        return obj
    raise TypeError(f"provider must be callable, got {type(obj)!r}")


@dataclass
class KinematicEmbeddingProvider:
    """CPU-only embedding of *what the robot is doing*: current state + the next ``horizon`` actions.

    Per frame ``t`` of an episode the raw feature is::

        [ w_s * z(state[t]),  w_a * z(action[t : t + horizon]) / sqrt(horizon),  1.0 ]

    where ``z`` is a dataset-wide per-dimension z-score (finite rows only), the action chunk is
    padded with the episode's last action (same as GR00T ``allow_padding``), and the constant
    last component keeps frames at the dataset mean well defined under cosine distance. If the
    raw size exceeds ``dim`` a fixed Gaussian random projection (seeded) reduces it. Output rows
    are L2-normalized. Frames with any non-finite input in the window are returned as NaN and
    end up ``NULL``.
    """

    horizon: int = 16
    dim: int | None = 64
    seed: int = 0
    state_weight: float = 1.0
    action_weight: float = 1.0
    name: str = "kinematic"

    def config(self) -> dict[str, Any]:
        return {
            "horizon": int(self.horizon),
            "dim": None if self.dim is None else int(self.dim),
            "seed": int(self.seed),
            "state_weight": float(self.state_weight),
            "action_weight": float(self.action_weight),
        }

    @staticmethod
    def zscore_params(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-dimension ``(mean, std)`` over finite rows, as used by ``__call__``."""
        x = np.asarray(x, dtype=np.float64)
        if x.shape[1] == 0:
            return np.zeros(0), np.ones(0)
        finite_rows = np.all(np.isfinite(x), axis=1)
        ref = x[finite_rows] if np.any(finite_rows) else np.zeros((1, x.shape[1]))
        mu = ref.mean(axis=0)
        sd = ref.std(axis=0)
        sd = np.where(sd > 1e-8, sd, 1.0)
        return mu, sd

    @classmethod
    def _zscore(cls, x: np.ndarray) -> np.ndarray:
        if x.shape[1] == 0:
            return x
        mu, sd = cls.zscore_params(x)
        return (x - mu) / sd

    def _finish(self, parts: list[np.ndarray]) -> np.ndarray:
        feat = np.concatenate(parts, axis=1)
        if self.dim is not None and 0 < int(self.dim) < feat.shape[1]:
            rng = np.random.default_rng(self.seed)
            proj = rng.standard_normal((feat.shape[1], int(self.dim))) / math.sqrt(int(self.dim))
            bad = ~np.all(np.isfinite(feat), axis=1)
            feat = _matmul(np.where(bad[:, None], 0.0, feat), proj)
            feat[bad] = np.nan
        out, ok = _l2_normalize(feat.astype(np.float32))
        out[~ok] = np.nan
        return out

    def embed_windows(
        self,
        state: np.ndarray,
        action_window: np.ndarray,
        *,
        state_ref: tuple[np.ndarray, np.ndarray],
        action_ref: tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray:
        """Embed frames that are not rows of the table (e.g. simulator rollouts).

        ``state`` is ``(R, Ds)`` and ``action_window`` is ``(R, horizon, Da)`` (already padded);
        ``state_ref`` / ``action_ref`` are the table's :meth:`zscore_params`, so the result is in
        the same space as the stored embedding column.
        """
        state = np.asarray(state, dtype=np.float64)
        action_window = np.asarray(action_window, dtype=np.float64)
        if action_window.ndim != 3 or action_window.shape[1] != self.horizon:
            raise ValueError(f"action_window must be (R, {self.horizon}, Da), got {action_window.shape}")
        r = len(state)
        parts = []
        if state.shape[1]:
            parts.append(self.state_weight * (state - state_ref[0]) / state_ref[1])
        if action_window.shape[2]:
            za = (action_window - action_ref[0]) / action_ref[1]
            parts.append(self.action_weight * za.reshape(r, -1) / math.sqrt(self.horizon))
        parts.append(np.ones((r, 1), dtype=np.float64))
        return self._finish(parts)

    def __call__(self, batch: FrameBatch) -> np.ndarray:
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")
        state = np.asarray(batch.state, dtype=np.float64)
        action = np.asarray(batch.action, dtype=np.float64)
        if state.shape[1] + action.shape[1] == 0:
            raise ValueError("Kinematic embedding needs state and/or action columns (see modality.json)")
        rows = np.asarray(batch.rows, dtype=np.int64)
        zs = self._zscore(state)
        za = self._zscore(action)

        order, starts, ends = _episode_order(batch.episode_index, batch.frame_index)
        inv = np.empty(len(order), dtype=np.int64)
        inv[order] = np.arange(len(order), dtype=np.int64)
        ep_end = np.empty(len(order), dtype=np.int64)
        for s, e in zip(starts, ends, strict=True):
            ep_end[s:e] = e
        sorted_pos = inv[rows]
        last = ep_end[sorted_pos] - 1
        steps = np.arange(self.horizon, dtype=np.int64)[None, :]
        chunk_sorted = np.minimum(sorted_pos[:, None] + steps, last[:, None])
        chunk_rows = order[chunk_sorted]  # (R, horizon) physical rows

        parts = []
        if zs.shape[1]:
            parts.append(self.state_weight * zs[rows])
        if za.shape[1]:
            chunk = za[chunk_rows].reshape(len(rows), -1)
            parts.append(self.action_weight * chunk / math.sqrt(self.horizon))
        parts.append(np.ones((len(rows), 1), dtype=np.float64))
        return self._finish(parts)


def load_hf_image_encoder(model_path: str | Path, *, device: str = "cpu") -> Callable[[list[np.ndarray]], np.ndarray]:
    """Load a *local* SigLIP / CLIP / DINOv2 checkpoint with ``transformers`` (never downloads).

    Returns ``fn(list[HWC uint8]) -> float32[N, D]``. SigLIP/CLIP use ``get_image_features``;
    other vision backbones (DINOv2) use ``pooler_output`` or the CLS token.
    """
    path = Path(model_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Local model directory not found: {path} (downloads are disabled on purpose)")
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError("Visual embeddings need `transformers` (pip install transformers)") from exc

    processor = AutoImageProcessor.from_pretrained(str(path), local_files_only=True)
    model = AutoModel.from_pretrained(str(path), local_files_only=True).to(device).eval()

    def _encode(images: list[np.ndarray]) -> np.ndarray:  # pragma: no cover - needs weights
        inputs = processor(images=list(images), return_tensors="pt").to(device)
        with torch.inference_mode():
            if hasattr(model, "get_image_features"):
                feats = model.get_image_features(pixel_values=inputs["pixel_values"])
            else:
                out = model(**inputs)
                pooled = getattr(out, "pooler_output", None)
                feats = pooled if pooled is not None else out.last_hidden_state[:, 0]
        return feats.float().cpu().numpy()

    return _encode


class VisualEmbeddingProvider:
    """Embed decoded camera frames (keyframes) with a user callable or a local HF vision model.

    Frames are decoded through ``lerobot_lancedb.LanceDBDataset`` (read-only on ``videos.lance``).
    Use ``every_n`` in :func:`add_embedding_column` to embed one frame in ``N`` — decoding and the
    vision model dominate cost, not storage.

    Args:
        camera: modality.json video key (``"front"``) or original feature key
            (``"observation.images.front_cam"``).
        image_embed_fn: ``fn(list[HWC uint8 ndarray]) -> float32[N, D]``. Takes precedence.
        model_path: local SigLIP/DINOv2 directory, loaded via :func:`load_hf_image_encoder`.
    """

    def __init__(
        self,
        camera: str,
        *,
        image_embed_fn: Callable[[list[np.ndarray]], np.ndarray] | None = None,
        model_path: str | Path | None = None,
        device: str = "cpu",
        decode_batch_size: int = 32,
        name: str = "visual",
    ) -> None:
        if image_embed_fn is None and model_path is None:
            raise ValueError("VisualEmbeddingProvider needs image_embed_fn or a local model_path")
        self.camera = camera
        self.image_embed_fn = image_embed_fn
        self.model_path = str(model_path) if model_path is not None else None
        self.device = device
        self.decode_batch_size = max(1, int(decode_batch_size))
        self.name = name

    def config(self) -> dict[str, Any]:
        return {
            "camera": self.camera,
            "model_path": self.model_path,
            "image_embed_fn": getattr(self.image_embed_fn, "__name__", None) if self.image_embed_fn else None,
        }

    def _camera_feature(self, modality: Mapping[str, Any]) -> str:
        video = modality.get("video", {})
        if self.camera in video:
            return video[self.camera].get("original_key") or f"observation.images.{self.camera}"
        return self.camera

    def __call__(self, batch: FrameBatch) -> np.ndarray:
        from lerobot_lancedb import LanceDBDataset

        cam = self._camera_feature(batch.modality_meta)
        encode = self.image_embed_fn or load_hf_image_encoder(self.model_path or "", device=self.device)
        ds = LanceDBDataset(
            root=batch.lance_root,
            columns=[cam],
            return_uint8=True,
            storage_options=batch.storage_options,
        )
        try:
            abs_idx = [int(i) for i in batch.index[batch.rows]]
            outs: list[np.ndarray] = []
            for s in range(0, len(abs_idx), self.decode_batch_size):
                items = ds.__getitems__(abs_idx[s : s + self.decode_batch_size])
                images = [_chw_to_hwc_uint8(it[cam]) for it in items]
                outs.append(np.asarray(encode(images), dtype=np.float32).reshape(len(images), -1))
        finally:
            pool = getattr(ds, "_prefetch_pool", None)
            if pool is not None:
                pool.shutdown(wait=False)
        if not outs:
            return np.empty((0, 0), dtype=np.float32)
        out, ok = _l2_normalize(np.concatenate(outs, axis=0))
        out[~ok] = np.nan
        return out


def _chw_to_hwc_uint8(frame: Any) -> np.ndarray:
    arr = frame.detach().cpu().numpy() if hasattr(frame, "detach") else np.asarray(frame)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = np.clip(arr * (255.0 if arr.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


# ---------------------------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------------------------
@dataclass
class _Report:
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EmbeddingReport(_Report):
    lance_root: str
    column: str
    provider: str
    provider_config: dict[str, Any]
    dim: int
    every_n: int
    total_frames: int
    keyframes: int
    embedded_frames: int
    null_frames: int
    invalid_keyframes: int
    embedding_payload_bytes: int
    frames_table_version_before: int
    frames_table_version: int
    dropped_columns: list[str]
    embed_wall_s: float
    write_wall_s: float
    wall_time_s: float
    bytes_measured: bool
    bytes_written_frames: int
    bytes_written_videos: int
    bytes_written_meta: int
    bytes_written_other: int
    files_written_frames: int
    files_written_videos: int
    bytes_written_by_table: dict[str, int]


@dataclass
class VectorIndexReport(_Report):
    lance_root: str
    column: str
    search_mode: str
    index_type: str
    index_name: str | None
    distance_type: str
    dim: int
    num_vectors: int
    num_partitions: int | None
    num_sub_vectors: int | None
    num_indexed_rows: int
    num_unindexed_rows: int
    fallback_reason: str | None
    frames_table_version_before: int
    frames_table_version: int
    build_wall_s: float
    bytes_measured: bool
    bytes_written_frames: int
    bytes_written_videos: int
    bytes_written_meta: int
    bytes_written_other: int
    files_written_frames: int
    files_written_videos: int
    bytes_written_by_table: dict[str, int]


@dataclass
class SemanticDedupReport(_Report):
    lance_root: str
    column: str
    threshold: float
    scope: str
    k: int
    where: str | None
    search_mode: str
    total_frames: int
    candidate_frames: int
    dup_frames: int
    dup_pct: float
    kept_candidate_frames: int
    propagated_frames: int
    clusters_with_dups: int
    largest_cluster: int
    dup_frames_by_episode: dict[str, int]
    duplicate_episodes: list[dict[str, Any]]
    nn_dist_quantiles: dict[str, float | None]
    lance_queries: int
    frames_table_version_before: int
    frames_table_version: int
    scalar_indexes_built: list[str]
    search_wall_s: float
    assign_wall_s: float
    write_wall_s: float
    wall_time_s: float
    bytes_measured: bool
    bytes_written_frames: int
    bytes_written_videos: int
    bytes_written_meta: int
    bytes_written_other: int
    files_written_frames: int
    files_written_videos: int
    bytes_written_by_table: dict[str, int]


@dataclass(frozen=True)
class SearchHit:
    episode_index: int
    frame_index: int
    index: int
    distance: float


@dataclass
class SearchResult(_Report):
    """Frame-level hits, best first. Convert with :meth:`to_where` or :meth:`anchors_by_episode`."""

    hits: list[SearchHit]
    query: dict[str, Any]
    where: str | None
    search_mode: str
    k: int
    wall_time_s: float

    def episodes(self) -> list[int]:
        """Distinct episode indices ordered by their best hit."""
        seen: dict[int, None] = {}
        for h in self.hits:
            seen.setdefault(h.episode_index, None)
        return list(seen)

    def anchors_by_episode(self) -> dict[int, np.ndarray]:
        out: dict[int, list[int]] = {}
        for h in self.hits:
            out.setdefault(h.episode_index, []).append(h.frame_index)
        return {e: np.unique(np.asarray(v, dtype=np.int64)) for e, v in sorted(out.items())}

    def to_where(self, level: Literal["episode", "frame"] = "frame") -> str:
        if level == "episode":
            return episodes_to_where(self.episodes())
        return anchors_to_where(self.anchors_by_episode())

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["episodes"] = self.episodes()
        d["where_episode_predicate"] = self.to_where("episode")
        return d


@dataclass(frozen=True)
class EpisodeScore:
    episode_index: int
    score: float
    coverage: float
    best_distance: float


@dataclass
class EpisodeSliceResult(_Report):
    """Episodes ranked by trajectory similarity to ``query_episode`` (lower score = more similar)."""

    query_episode: int
    ranked: list[EpisodeScore]
    query_frames: int
    where: str | None
    search_mode: str
    wall_time_s: float

    def episodes(self) -> list[int]:
        return [r.episode_index for r in self.ranked]

    def to_where(self, include_query: bool = False) -> str:
        eps = self.episodes() + ([self.query_episode] if include_query else [])
        return episodes_to_where(eps)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["episodes"] = self.episodes()
        d["where_predicate"] = self.to_where()
        return d


# ---------------------------------------------------------------------------------------------
# 1. Embedding column
# ---------------------------------------------------------------------------------------------
def add_embedding_column(
    lance_root: str | Path,
    provider: Any = None,
    *,
    column: str = EMBEDDING_COLUMN,
    every_n: int = 1,
    storage_options: Mapping[str, str] | None = None,
    drop_dependent_columns: bool = True,
) -> EmbeddingReport:
    """Compute per-frame embeddings and append them to ``frames.lance`` as ``FixedSizeList<float32, D>``.

    * One value per frame row (required by zero-copy ``add_columns``); with ``every_n > 1`` only
      every N-th frame of each episode (by position, frame 0 always included) is embedded and the
      rest are ``NULL`` — nulls cost a validity bit, vector search skips them, and
      ``tag_semantic_duplicates`` propagates a keyframe's tag to the frames it covers.
    * Re-running drops and re-adds the column (and its vector index). With
      ``drop_dependent_columns`` the semantic-dedup columns derived from the old embedding are
      dropped too, so stale tags can never be silently used.
    * Provenance (provider, config, dim, every_n, metric) is stored in the Arrow field metadata.
    """
    if every_n < 1:
        raise ValueError(f"every_n must be >= 1, got {every_n}")
    prov = _as_provider(provider)
    t0 = time.perf_counter()
    meter = _WriteMeter(lance_root)
    db, table = _open_frames(lance_root, storage_options)
    version_before = int(table.version)
    modality = _read_modality(db, lance_root, storage_options)

    available = set(table.schema.names)
    state_cols = [to_lance_column(k) for k in _modality_original_keys(modality, "state")]
    action_cols = [to_lance_column(k) for k in _modality_original_keys(modality, "action")]
    state_cols = [c for c in state_cols if c in available]
    action_cols = [c for c in action_cols if c in available]
    select = ["index", "episode_index", "frame_index", *dict.fromkeys([*state_cols, *action_cols])]
    tab = table.search().select(select).limit(None).to_arrow()  # physical row order, tabular only
    n = tab.num_rows
    ep = _col_to_numpy(tab, "episode_index", np.int64)
    fr = _col_to_numpy(tab, "frame_index", np.int64)
    gidx = _col_to_numpy(tab, "index", np.int64)

    def _stack(cols: list[str]) -> np.ndarray:
        blocks = [_arrow_col_to_2d_float64(tab.column(c)) for c in cols]
        blocks = [b for b in blocks if b.shape[1] > 0]
        return np.concatenate(blocks, axis=1) if blocks else np.empty((n, 0), dtype=np.float64)

    pos = _position_in_episode(ep, fr)
    rows = np.flatnonzero(pos % int(every_n) == 0).astype(np.int64)
    batch = FrameBatch(
        lance_root=str(lance_root),
        storage_options=dict(storage_options) if storage_options else None,
        index=gidx,
        episode_index=ep,
        frame_index=fr,
        state=_stack(state_cols),
        action=_stack(action_cols),
        rows=rows,
        modality_meta=modality,
    )

    t_embed = time.perf_counter()
    raw = np.asarray(prov(batch), dtype=np.float32)
    embed_wall = time.perf_counter() - t_embed
    if raw.ndim != 2 or raw.shape[0] != len(rows) or (len(rows) and raw.shape[1] == 0):
        raise ValueError(f"provider {getattr(prov, 'name', prov)!r} returned shape {raw.shape}, expected ({len(rows)}, D)")
    dim = int(raw.shape[1]) if raw.ndim == 2 and raw.shape[1] else 1
    normed, ok = _l2_normalize(raw) if len(rows) else (np.zeros((0, dim), np.float32), np.zeros(0, bool))

    values = np.zeros((n, dim), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    values[rows[ok]] = normed[ok]
    valid[rows[ok]] = True

    prov_name = str(getattr(prov, "name", type(prov).__name__))
    prov_cfg = prov.config() if hasattr(prov, "config") else {}
    info = {
        "format": EMBEDDING_FORMAT,
        "provider": prov_name,
        "config": prov_cfg,
        "dim": dim,
        "every_n": int(every_n),
        "metric": DISTANCE_TYPE,
        "normalized": True,
    }
    fld = pa.field(
        column,
        pa.list_(pa.float32(), dim),
        nullable=True,
        metadata={EMBEDDING_FIELD_METADATA_KEY: json.dumps(info, sort_keys=True).encode("utf-8")},
    )
    fsl = pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1), type=pa.float32()), dim, mask=pa.array(~valid))
    new_cols = pa.Table.from_arrays([fsl], schema=pa.schema([fld]))

    t_write = time.perf_counter()
    to_drop = [column, *(SEMANTIC_DUP_COLUMNS if drop_dependent_columns else ())]
    dropped = _drop_columns_and_indices(table, to_drop)
    table.add_columns(new_cols)
    write_wall = time.perf_counter() - t_write
    version_after = int(table.version)
    io = meter.stop()

    return EmbeddingReport(
        lance_root=str(lance_root),
        column=column,
        provider=prov_name,
        provider_config=prov_cfg,
        dim=dim,
        every_n=int(every_n),
        total_frames=int(n),
        keyframes=int(len(rows)),
        embedded_frames=int(valid.sum()),
        null_frames=int(n - valid.sum()),
        invalid_keyframes=int(len(rows) - ok.sum()),
        embedding_payload_bytes=int(valid.sum()) * dim * 4,
        frames_table_version_before=version_before,
        frames_table_version=version_after,
        dropped_columns=dropped,
        embed_wall_s=round(embed_wall, 4),
        write_wall_s=round(write_wall, 4),
        wall_time_s=round(time.perf_counter() - t0, 4),
        **io,
    )


# ---------------------------------------------------------------------------------------------
# 2. Vector index
# ---------------------------------------------------------------------------------------------
def suggest_ivf_pq_params(num_vectors: int, dim: int) -> tuple[int, int]:
    """``(num_partitions, num_sub_vectors)`` for an IVF-PQ index.

    * partitions ~ sqrt(N), but at most N/256 so every k-means cell gets enough training rows;
    * sub-vectors so each sub-vector has ~8 dims (16 for dim >= 256), and divides ``dim``.
    """
    n = max(1, int(num_vectors))
    num_partitions = max(1, min(int(round(math.sqrt(n))), n // 256))
    target = 16 if dim >= 256 else 8
    divisors = [d for d in range(1, dim + 1) if dim % d == 0]
    num_sub_vectors = min(divisors, key=lambda d: (abs(dim / d - target), -d))
    return num_partitions, int(num_sub_vectors)


def build_vector_index(
    lance_root: str | Path,
    *,
    column: str = EMBEDDING_COLUMN,
    num_partitions: int | None = None,
    num_sub_vectors: int | None = None,
    min_rows: int = 256,
    force_flat: bool = False,
    sample_rate: int = 256,
    max_iterations: int = 50,
    storage_options: Mapping[str, str] | None = None,
) -> VectorIndexReport:
    """Build an IVF-PQ (cosine) index on ``column``, or record a flat-search fallback.

    Tables with fewer than ``min_rows`` non-null vectors (PQ needs 256 training rows for 8-bit
    codebooks) use exact flat search; so does any build that ends up indexing zero rows. The
    returned ``search_mode`` (``"ivf_pq"`` / ``"flat"``) is what searches will use in ``auto``.
    Note: Lance's k-means initialisation is not seeded, so recall may vary slightly between
    rebuilds; dedup re-scores candidates with exact distances, so tags only depend on recall.
    """
    from lancedb.index import IvfPq

    t0 = time.perf_counter()
    meter = _WriteMeter(lance_root)
    _, table = _open_frames(lance_root, storage_options)
    version_before = int(table.version)
    info = _embedding_field_info(table, column)
    dim = int(info["dim"])
    n_vec = int(table.count_rows(f"{column} IS NOT NULL"))
    index_name = f"{column}_idx"

    fallback: str | None = None
    parts = subs = None
    if force_flat:
        fallback = "force_flat requested"
    elif n_vec < int(min_rows):
        fallback = f"only {n_vec} vectors (< min_rows={min_rows}); exact flat search is used"
    if fallback is not None:
        for idx in _indices_on(table, column):
            table.drop_index(idx.name)
    else:
        sp, ss = suggest_ivf_pq_params(n_vec, dim)
        parts = int(num_partitions) if num_partitions else sp
        subs = int(num_sub_vectors) if num_sub_vectors else ss
        if dim % subs != 0:
            raise ValueError(f"num_sub_vectors={subs} must divide dim={dim}")
        table.create_index(
            column,
            config=IvfPq(
                distance_type=DISTANCE_TYPE,
                num_partitions=parts,
                num_sub_vectors=subs,
                sample_rate=int(sample_rate),
                max_iterations=int(max_iterations),
            ),
            name=index_name,
            replace=True,
        )
        if _vector_index_state(table, column) is None:
            fallback = "IVF-PQ build indexed 0 rows (too few vectors); exact flat search is used"
            for idx in _indices_on(table, column):
                table.drop_index(idx.name)

    state = _vector_index_state(table, column)
    version_after = int(table.version)
    io = meter.stop()
    return VectorIndexReport(
        lance_root=str(lance_root),
        column=column,
        search_mode="ivf_pq" if state is not None else "flat",
        index_type=str(state["index_type"]) if state is not None else "FLAT",
        index_name=state["name"] if state is not None else None,
        distance_type=DISTANCE_TYPE,
        dim=dim,
        num_vectors=n_vec,
        num_partitions=parts if state is not None else None,
        num_sub_vectors=subs if state is not None else None,
        num_indexed_rows=int(state["num_indexed_rows"]) if state is not None else 0,
        num_unindexed_rows=int(state["num_unindexed_rows"]) if state is not None else n_vec,
        fallback_reason=fallback,
        frames_table_version_before=version_before,
        frames_table_version=version_after,
        build_wall_s=round(time.perf_counter() - t0, 4),
        **io,
    )


# ---------------------------------------------------------------------------------------------
# 3. Semantic dedup
# ---------------------------------------------------------------------------------------------
def greedy_duplicate_assignment(
    num_nodes: int,
    src: np.ndarray,
    dst: np.ndarray,
    order: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic leader clustering over an (undirected) near-duplicate edge list.

    Nodes are visited in ``order``; an unassigned node becomes a representative and claims every
    unassigned neighbour as its duplicate. Each duplicate is therefore *directly* within the
    threshold of its representative (no transitive chaining), and exactly one node per cluster
    is kept. Returns ``(is_dup[bool], rep[int])`` with ``rep[i] == i`` for representatives.
    """
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    a = np.concatenate([src, dst])
    b = np.concatenate([dst, src])
    srt = np.argsort(a, kind="stable")
    a, b = a[srt], b[srt]
    indptr = np.searchsorted(a, np.arange(num_nodes + 1, dtype=np.int64), side="left")
    rep = np.full(num_nodes, -1, dtype=np.int64)
    is_dup = np.zeros(num_nodes, dtype=bool)
    for i in np.asarray(order, dtype=np.int64):
        if rep[i] >= 0:
            continue
        rep[i] = i
        nbrs = b[indptr[i] : indptr[i + 1]]
        if len(nbrs):
            free = nbrs[rep[nbrs] < 0]
            rep[free] = i
            is_dup[free] = True
    return is_dup, rep


def _static_run_edges(
    emb: np.ndarray, ep: np.ndarray, fr: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Within-episode *temporally static runs*: each run's first frame + every following frame that
    stays within ``threshold`` of it. Returns ``(src, dst, adjacent_nn_dist)`` over local positions."""
    order, starts, ends = _episode_order(ep, fr)
    src: list[int] = []
    dst: list[int] = []
    nn = np.full(len(ep), np.nan, dtype=np.float64)
    for s, e in zip(starts, ends, strict=True):
        p = order[s:e]
        if len(p) < 2:
            continue
        e_ep = emb[p]
        adj = np.clip(1.0 - np.sum(e_ep[1:] * e_ep[:-1], axis=1, dtype=np.float64), 0.0, 2.0)
        left = np.concatenate([[np.inf], adj])
        right = np.concatenate([adj, [np.inf]])
        nn[p] = np.minimum(left, right)
        leader = 0
        for j in range(1, len(p)):
            d = 1.0 - float(np.dot(e_ep[leader], e_ep[j]))
            if d <= threshold:
                src.append(int(p[leader]))
                dst.append(int(p[j]))
            else:
                leader = j
    return np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64), nn


def _cross_episode_neighbors(
    table: Any,
    column: str,
    emb: np.ndarray,
    ep: np.ndarray,
    keys: np.ndarray,
    stride: int,
    *,
    base_where: str,
    k: int,
    mode: str,
    nprobes: int,
    refine_factor: int | None,
    query_batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Top-``k`` cross-episode neighbours of every candidate (exact distances).

    Candidates are processed in groups of whole episodes (~``query_batch_size`` frames): one
    batched LanceDB vector query per group against all *other* episodes
    (``episode_index NOT IN group`` prefilter, IVF-PQ + exact refine, or flat), plus an exact
    in-memory comparison between the group's own episodes. The union is re-scored exactly and cut
    to the ``k`` nearest per frame, so results don't depend on how episodes were grouped.

    Returns ``(q, nbr, dist, nn_dist, lance_queries)`` over candidate-local positions.
    """
    m = len(ep)
    sort_keys = np.argsort(keys, kind="stable")
    keys_sorted = keys[sort_keys]
    order, starts, ends = _episode_order(ep, np.arange(m, dtype=np.int64))
    ep_ids = ep[order[starts]] if len(starts) else np.empty(0, dtype=np.int64)
    nn_dist = np.full(m, np.nan, dtype=np.float64)
    out_q: list[np.ndarray] = []
    out_n: list[np.ndarray] = []
    out_d: list[np.ndarray] = []
    lance_queries = 0

    groups: list[list[int]] = []
    cur: list[int] = []
    cur_n = 0
    for gi, (s, e) in enumerate(zip(starts, ends, strict=True)):
        cur.append(gi)
        cur_n += int(e - s)
        if cur_n >= query_batch_size:
            groups.append(cur)
            cur, cur_n = [], 0
    if cur:
        groups.append(cur)

    for g in groups:
        pos = np.concatenate([order[starts[gi] : ends[gi]] for gi in g])
        g_eps = ep_ids[g]
        q_emb = emb[pos]
        cand_q: list[np.ndarray] = []
        cand_n: list[np.ndarray] = []

        # (a) LanceDB vector search against every other episode
        if len(g) < len(ep_ids):
            res = _knn_query(
                table,
                column,
                q_emb,
                where=_and(base_where, f"episode_index NOT IN ({', '.join(str(int(x)) for x in g_eps)})"),
                k=k,
                mode=mode,
                nprobes=nprobes,
                refine_factor=refine_factor,
                select=["episode_index", "frame_index"],
            )
            lance_queries += 1
            if res.num_rows:
                qi = _col_to_numpy(res, "query_index", np.int64)
                rk = _frame_keys(_col_to_numpy(res, "episode_index", np.int64), _col_to_numpy(res, "frame_index", np.int64), stride)
                loc = np.searchsorted(keys_sorted, rk)
                loc = np.clip(loc, 0, max(0, m - 1))
                found = keys_sorted[loc] == rk
                cand_q.append(pos[qi[found]])
                cand_n.append(sort_keys[loc[found]])

        # (b) exact comparison between different episodes inside this group (row-chunked)
        if len(g) > 1:
            g_ep = ep[pos]
            kk = min(k, len(pos))
            for r0 in range(0, len(pos), 1024):
                r1 = min(len(pos), r0 + 1024)
                d = 1.0 - _matmul(q_emb[r0:r1], q_emb.T)
                d[g_ep[r0:r1, None] == g_ep[None, :]] = np.inf
                if kk < len(pos):
                    part = np.argpartition(d, kk - 1, axis=1)[:, :kk]
                else:
                    part = np.tile(np.arange(len(pos)), (r1 - r0, 1))
                rows_i = np.repeat(np.arange(r1 - r0), part.shape[1])
                cols_j = part.reshape(-1)
                keep = np.isfinite(d[rows_i, cols_j])
                cand_q.append(pos[r0 + rows_i[keep]])
                cand_n.append(pos[cols_j[keep]])

        if not cand_q:
            continue
        cq = np.concatenate(cand_q)
        cn = np.concatenate(cand_n)
        if len(cq) == 0:
            continue
        # exact re-score, dedupe pairs, keep k nearest per query (ties broken by neighbour position)
        dist = np.clip(1.0 - np.sum(emb[cq] * emb[cn], axis=1, dtype=np.float64), 0.0, 2.0)
        srt = np.lexsort((cn, dist, cq))
        cq, cn, dist = cq[srt], cn[srt], dist[srt]
        uniq = np.ones(len(cq), dtype=bool)
        uniq[1:] = (cq[1:] != cq[:-1]) | (cn[1:] != cn[:-1])
        cq, cn, dist = cq[uniq], cn[uniq], dist[uniq]
        # duplicates removed but order by (cq, dist, cn) is preserved -> rank within each query
        first = np.ones(len(cq), dtype=bool)
        first[1:] = cq[1:] != cq[:-1]
        grp_start = np.maximum.accumulate(np.where(first, np.arange(len(cq)), 0))
        rank = np.arange(len(cq)) - grp_start
        topk = rank < k
        cq, cn, dist = cq[topk], cn[topk], dist[topk]
        nn_dist[cq[first[topk]]] = dist[first[topk]]
        out_q.append(cq)
        out_n.append(cn)
        out_d.append(dist)

    if not out_q:
        e64 = np.empty(0, dtype=np.int64)
        return e64, e64, np.empty(0, dtype=np.float64), nn_dist, lance_queries
    return np.concatenate(out_q), np.concatenate(out_n), np.concatenate(out_d), nn_dist, lance_queries


def tag_semantic_duplicates(
    lance_root: str | Path,
    *,
    column: str = EMBEDDING_COLUMN,
    threshold: float = 0.02,
    scope: DedupScope = "cross_episode",
    k: int = 10,
    where: str | None = None,
    mode: SearchMode = "auto",
    nprobes: int = 20,
    refine_factor: int | None = 10,
    query_batch_size: int = 1024,
    propagate_keyframes: bool = True,
    build_bitmap_index: bool = True,
    storage_options: Mapping[str, str] | None = None,
) -> SemanticDedupReport:
    """Tag near-duplicate frames and write ``is_semantic_dup`` / ``semantic_cluster_id`` / ``semantic_nn_dist``.

    Candidates are frames with a non-null embedding that satisfy ``where`` (pass
    ``where="is_valid"`` after tabular curation so representatives are always valid frames).

    * ``scope="cross_episode"`` (default): edges between a frame and its top-``k`` nearest frames
      of *other* episodes with cosine distance ``<= threshold`` (vector index when available).
    * ``scope="within_episode"``: temporally static runs — inside an episode, frames that stay
      within ``threshold`` of the run's first frame.
    * ``scope="all"``: both.

    Frames are visited in ``(episode_index, frame_index)`` order and :func:`greedy_duplicate_assignment`
    keeps the earliest frame of each cluster; ``semantic_cluster_id`` is the representative's
    global ``index`` (``-1`` for non-candidates). ``semantic_nn_dist`` is the exact cosine
    distance to the nearest frame of another episode (``cross_episode``/``all``; large = rare frame,
    useful for hard-example mining) or to the adjacent frame (``within_episode``).
    With ``every_n > 1`` embeddings, frames between keyframes inherit their keyframe's tags.
    """
    if scope not in ("cross_episode", "within_episode", "all"):
        raise ValueError(f"scope must be cross_episode|within_episode|all, got {scope!r}")
    if not (0.0 <= threshold <= 2.0):
        raise ValueError(f"cosine-distance threshold must be in [0, 2], got {threshold}")
    t0 = time.perf_counter()
    meter = _WriteMeter(lance_root)
    _, table = _open_frames(lance_root, storage_options)
    version_before = int(table.version)
    info = _embedding_field_info(table, column)
    every_n = int(info.get("every_n", 1) or 1)
    search_mode, _ = _resolve_search_mode(table, column, mode)

    full = table.search().select(["index", "episode_index", "frame_index", column]).limit(None).to_arrow()
    n = full.num_rows
    ep_all = _col_to_numpy(full, "episode_index", np.int64)
    fr_all = _col_to_numpy(full, "frame_index", np.int64)
    gidx_all = _col_to_numpy(full, "index", np.int64)
    emb_all, has_emb = _fsl_to_float32(full.column(column))
    del full
    stride = int(fr_all.max()) + 1 if n else 1
    keys_all = _frame_keys(ep_all, fr_all, stride)
    if len(np.unique(keys_all)) != n:
        raise ValueError("(episode_index, frame_index) pairs must be unique in frames.lance")

    if where:
        wt = table.search().where(where).select(["episode_index", "frame_index"]).limit(None).to_arrow()
        where_mask = np.isin(keys_all, _frame_keys(_col_to_numpy(wt, "episode_index", np.int64), _col_to_numpy(wt, "frame_index", np.int64), stride))
    else:
        where_mask = np.ones(n, dtype=bool)
    cand = has_emb & where_mask
    cpos = np.flatnonzero(cand)
    m = len(cpos)
    c_ep, c_fr, c_emb, c_keys = ep_all[cpos], fr_all[cpos], emb_all[cpos], keys_all[cpos]

    t_search = time.perf_counter()
    src_l: list[np.ndarray] = []
    dst_l: list[np.ndarray] = []
    nn = np.full(m, np.nan, dtype=np.float64)
    lance_queries = 0
    if m and scope in ("cross_episode", "all"):
        q, nb, dist, nn_cross, lance_queries = _cross_episode_neighbors(
            table,
            column,
            c_emb,
            c_ep,
            c_keys,
            stride,
            base_where=_and(f"{column} IS NOT NULL", where) or "",
            k=max(1, int(k)),
            mode=search_mode,
            nprobes=nprobes,
            refine_factor=refine_factor,
            query_batch_size=max(1, int(query_batch_size)),
        )
        hit = dist <= threshold
        src_l.append(q[hit])
        dst_l.append(nb[hit])
        nn = np.fmin(nn, nn_cross)
    if m and scope in ("within_episode", "all"):
        s_src, s_dst, nn_static = _static_run_edges(c_emb, c_ep, c_fr, float(threshold))
        src_l.append(s_src)
        dst_l.append(s_dst)
        nn = np.fmin(nn, nn_static) if scope == "all" else nn_static
    search_wall = time.perf_counter() - t_search

    t_assign = time.perf_counter()
    src = np.concatenate(src_l) if src_l else np.empty(0, dtype=np.int64)
    dst = np.concatenate(dst_l) if dst_l else np.empty(0, dtype=np.int64)
    c_order = np.lexsort((c_fr, c_ep))
    c_dup, c_rep = greedy_duplicate_assignment(m, src, dst, c_order)

    is_dup = np.zeros(n, dtype=bool)
    cluster = np.full(n, -1, dtype=np.int64)
    nn_full = np.full(n, np.nan, dtype=np.float64)
    is_dup[cpos] = c_dup
    cluster[cpos] = gidx_all[cpos[c_rep]] if m else cluster[cpos]
    nn_full[cpos] = nn

    propagated = 0
    if every_n > 1 and propagate_keyframes and m:
        pos_in_ep = _position_in_episode(ep_all, fr_all)
        order, starts, ends = _episode_order(ep_all, fr_all)
        kf_of = np.empty(n, dtype=np.int64)
        for s, e in zip(starts, ends, strict=True):
            p = order[s:e]
            kf_of[p] = p[(np.arange(e - s) // every_n) * every_n]
        target = (~has_emb) & where_mask & (pos_in_ep % every_n != 0) & cand[kf_of]
        is_dup[target] = is_dup[kf_of[target]]
        cluster[target] = cluster[kf_of[target]]
        propagated = int(target.sum())
    assign_wall = time.perf_counter() - t_assign

    t_write = time.perf_counter()
    nn_arr = pa.array(nn_full.astype(np.float32), type=pa.float32(), mask=~np.isfinite(nn_full))
    new_cols = pa.table(
        {
            "is_semantic_dup": pa.array(is_dup, type=pa.bool_()),
            "semantic_cluster_id": pa.array(cluster, type=pa.int64()),
            "semantic_nn_dist": nn_arr,
        }
    )
    _drop_columns_and_indices(table, SEMANTIC_DUP_COLUMNS)
    table.add_columns(new_cols)
    built: list[str] = []
    if build_bitmap_index:
        try:
            from lancedb.index import Bitmap

            table.create_index("is_semantic_dup", config=Bitmap(), replace=True)
            built.append("is_semantic_dup")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not build BITMAP index on is_semantic_dup: %s", exc)
    write_wall = time.perf_counter() - t_write
    version_after = int(table.version)
    io = meter.stop()

    # -- report ---------------------------------------------------------------------------
    dup_by_ep: dict[str, int] = {}
    dup_eps, dup_counts = np.unique(ep_all[is_dup], return_counts=True)
    for e, c in zip(dup_eps, dup_counts, strict=True):
        dup_by_ep[str(int(e))] = int(c)
    gsort = np.argsort(gidx_all, kind="stable")
    gsorted = gidx_all[gsort]
    duplicate_episodes: list[dict[str, Any]] = []
    for e in dup_eps:
        in_ep = (ep_all == e) & (cluster >= 0)
        denom = int(in_ep.sum())
        mask = (ep_all == e) & is_dup
        rep_eps = ep_all[gsort[np.searchsorted(gsorted, cluster[mask])]]
        if len(rep_eps) == 0 or denom == 0:
            continue
        vals, cnts = np.unique(rep_eps, return_counts=True)
        best = int(np.argmax(cnts))
        frac = float(mask.sum()) / denom
        if frac >= 0.5:
            duplicate_episodes.append(
                {
                    "episode_index": int(e),
                    "duplicate_of_episode": int(vals[best]),
                    "dup_fraction": round(frac, 4),
                    "frames_from_main_source": int(cnts[best]),
                }
            )
    sizes = np.bincount(np.searchsorted(np.unique(cluster[cluster >= 0]), cluster[cluster >= 0])) if np.any(cluster >= 0) else np.zeros(0, int)
    finite_nn = nn_full[np.isfinite(nn_full)]
    quantiles: dict[str, float | None] = {}
    for qn in (0.01, 0.05, 0.25, 0.5):
        quantiles[f"p{int(qn * 100):02d}"] = round(float(np.quantile(finite_nn, qn)), 6) if len(finite_nn) else None
    dup_frames = int(is_dup.sum())
    n_cand_total = int(m + propagated)
    return SemanticDedupReport(
        lance_root=str(lance_root),
        column=column,
        threshold=float(threshold),
        scope=str(scope),
        k=int(k),
        where=where,
        search_mode=search_mode if scope != "within_episode" else "exact_local",
        total_frames=int(n),
        candidate_frames=n_cand_total,
        dup_frames=dup_frames,
        dup_pct=round(100.0 * dup_frames / n, 4) if n else 0.0,
        kept_candidate_frames=n_cand_total - dup_frames,
        propagated_frames=propagated,
        clusters_with_dups=int(np.sum(sizes >= 2)),
        largest_cluster=int(sizes.max()) if len(sizes) else 0,
        dup_frames_by_episode=dup_by_ep,
        duplicate_episodes=duplicate_episodes,
        nn_dist_quantiles=quantiles,
        lance_queries=int(lance_queries),
        frames_table_version_before=version_before,
        frames_table_version=version_after,
        scalar_indexes_built=built,
        search_wall_s=round(search_wall, 4),
        assign_wall_s=round(assign_wall, 4),
        write_wall_s=round(write_wall, 4),
        wall_time_s=round(time.perf_counter() - t0, 4),
        **io,
    )


# ---------------------------------------------------------------------------------------------
# 4. Hybrid search, scene slicing, hard-example mining
# ---------------------------------------------------------------------------------------------
def _episode_embeddings(table: Any, column: str, episode: int, max_frame: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    pred = f"episode_index = {int(episode)} AND {column} IS NOT NULL"
    if max_frame is not None:
        pred += f" AND frame_index <= {int(max_frame)}"
    tbl = table.search().where(pred).select(["frame_index", column]).limit(None).to_arrow()
    fr = _col_to_numpy(tbl, "frame_index", np.int64)
    emb, _ = _fsl_to_float32(tbl.column(column)) if tbl.num_rows else (np.empty((0, 0), np.float32), None)
    order = np.argsort(fr, kind="stable")
    return fr[order], emb[order]


def _resolve_query_vector(
    table: Any,
    column: str,
    *,
    query_vector: np.ndarray | Sequence[float] | None,
    episode: int | None,
    frame: int | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if query_vector is not None:
        v, ok = _l2_normalize(np.asarray(query_vector, dtype=np.float32).reshape(1, -1))
        if not ok[0]:
            raise ValueError("query_vector must be finite and non-zero")
        return v[0], {"type": "vector"}
    if episode is None:
        raise ValueError("Provide query_vector, or episode (and optionally frame)")
    if frame is not None:
        fr, emb = _episode_embeddings(table, column, int(episode), max_frame=int(frame))
        if len(fr) == 0:
            raise ValueError(f"No embedded frame <= {frame} in episode {episode}")
        return emb[-1], {"type": "frame", "episode_index": int(episode), "frame_index": int(frame), "embedded_frame_used": int(fr[-1])}
    fr, emb = _episode_embeddings(table, column, int(episode))
    if len(fr) == 0:
        raise ValueError(f"Episode {episode} has no embedded frames")
    v, ok = _l2_normalize(emb.mean(axis=0, keepdims=True))
    if not ok[0]:
        raise ValueError(f"Mean embedding of episode {episode} is degenerate")
    return v[0], {"type": "episode_mean", "episode_index": int(episode), "frames": int(len(fr))}


def hybrid_search(
    lance_root: str | Path,
    *,
    query_vector: np.ndarray | Sequence[float] | None = None,
    episode: int | None = None,
    frame: int | None = None,
    where: str | None = None,
    k: int = 10,
    column: str = EMBEDDING_COLUMN,
    exclude_query_episode: bool = True,
    mode: SearchMode = "auto",
    nprobes: int = 20,
    refine_factor: int | None = 10,
    storage_options: Mapping[str, str] | None = None,
    lance_version: int | None = None,
) -> SearchResult:
    """Vector query + SQL prefilter in one LanceDB call.

    The query is ``query_vector``, the embedding of ``(episode, frame)`` (nearest embedded frame
    at or before ``frame``), or the mean embedding of ``episode``. ``where`` is any SQL predicate
    over ``frames.lance`` (``"is_valid AND NOT is_semantic_dup"``, ``"task_index = 3"``, ...) and
    is applied *before* the vector search (prefilter), so ``k`` hits always satisfy it.
    """
    t0 = time.perf_counter()
    _, table = _open_frames(lance_root, storage_options, lance_version)
    _embedding_field_info(table, column)
    vec, qdesc = _resolve_query_vector(table, column, query_vector=query_vector, episode=episode, frame=frame)
    search_mode, _ = _resolve_search_mode(table, column, mode)
    excl = f"episode_index != {int(episode)}" if (episode is not None and exclude_query_episode) else None
    pred = _and(f"{column} IS NOT NULL", where, excl)
    res = _knn_query(
        table,
        column,
        vec[None, :],
        where=pred,
        k=k,
        mode=search_mode,
        nprobes=nprobes,
        refine_factor=refine_factor,
        select=["index", "episode_index", "frame_index"],
    )
    eps = _col_to_numpy(res, "episode_index", np.int64)
    frs = _col_to_numpy(res, "frame_index", np.int64)
    gid = _col_to_numpy(res, "index", np.int64)
    dist = _col_to_numpy(res, "_distance", np.float64)
    order = np.lexsort((frs, eps, dist))
    hits = [SearchHit(int(eps[i]), int(frs[i]), int(gid[i]), round(float(max(dist[i], 0.0)), 6)) for i in order]
    return SearchResult(
        hits=hits,
        query=qdesc,
        where=pred,
        search_mode=search_mode,
        k=int(k),
        wall_time_s=round(time.perf_counter() - t0, 4),
    )


def slice_episodes_by_similarity(
    lance_root: str | Path,
    episode: int,
    *,
    k: int = 5,
    where: str | None = None,
    column: str = EMBEDDING_COLUMN,
    max_query_frames: int = 64,
    per_query_k: int = 32,
    mode: SearchMode = "auto",
    nprobes: int = 20,
    refine_factor: int | None = 10,
    storage_options: Mapping[str, str] | None = None,
    lance_version: int | None = None,
) -> EpisodeSliceResult:
    """Scene slicing: the ``k`` episodes whose trajectories are closest to ``episode``.

    Up to ``max_query_frames`` evenly spaced embedded frames of the query episode are sent as one
    batched vector query (prefilter: ``where`` and ``episode_index != episode``). For each other
    episode, score = mean over query frames of its best distance (a missing episode counts as that
    query's worst returned distance), i.e. a one-sided Chamfer distance; ``coverage`` = fraction
    of query frames for which the episode appeared. Lower score = more similar.
    """
    t0 = time.perf_counter()
    _, table = _open_frames(lance_root, storage_options, lance_version)
    _embedding_field_info(table, column)
    fr, emb = _episode_embeddings(table, column, int(episode))
    if len(fr) == 0:
        raise ValueError(f"Episode {episode} has no embedded frames")
    if len(fr) > max_query_frames:
        pick = np.unique(np.linspace(0, len(fr) - 1, int(max_query_frames)).round().astype(np.int64))
        emb = emb[pick]
    search_mode, _ = _resolve_search_mode(table, column, mode)
    pred = _and(f"{column} IS NOT NULL", where, f"episode_index != {int(episode)}")
    res = _knn_query(
        table,
        column,
        emb,
        where=pred,
        k=per_query_k,
        mode=search_mode,
        nprobes=nprobes,
        refine_factor=refine_factor,
        select=["episode_index"],
    )
    nq = len(emb)
    if res.num_rows == 0:
        return EpisodeSliceResult(int(episode), [], nq, pred, search_mode, round(time.perf_counter() - t0, 4))
    qi = _col_to_numpy(res, "query_index", np.int64)
    eps = _col_to_numpy(res, "episode_index", np.int64)
    dist = np.maximum(_col_to_numpy(res, "_distance", np.float64), 0.0)
    worst = np.full(nq, np.nan)
    for q in range(nq):
        sel = dist[qi == q]
        worst[q] = sel.max() if len(sel) else np.nan
    fill = float(np.nanmax(worst)) if np.any(np.isfinite(worst)) else 2.0
    worst = np.where(np.isfinite(worst), worst, fill)
    cand_eps = np.unique(eps)
    best = np.tile(worst[:, None], (1, len(cand_eps)))
    seen = np.zeros((nq, len(cand_eps)), dtype=bool)
    col = np.searchsorted(cand_eps, eps)
    for q, c, d in zip(qi, col, dist, strict=True):
        if not seen[q, c] or d < best[q, c]:
            best[q, c] = d
            seen[q, c] = True
    scores = best.mean(axis=0)
    coverage = seen.mean(axis=0)
    mins = np.where(seen, best, np.inf).min(axis=0)
    order = np.lexsort((cand_eps, -coverage, scores))[: int(k)]
    ranked = [
        EpisodeScore(int(cand_eps[i]), round(float(scores[i]), 6), round(float(coverage[i]), 4), round(float(mins[i]), 6))
        for i in order
    ]
    return EpisodeSliceResult(int(episode), ranked, nq, pred, search_mode, round(time.perf_counter() - t0, 4))


def mine_rare_frames(
    lance_root: str | Path,
    *,
    top_k: int = 100,
    where: str | None = None,
    min_nn_dist: float | None = None,
    storage_options: Mapping[str, str] | None = None,
    lance_version: int | None = None,
) -> SearchResult:
    """Hard-example mining: frames farthest from any other episode (largest ``semantic_nn_dist``).

    Requires :func:`tag_semantic_duplicates` (``scope`` ``cross_episode`` or ``all``).
    """
    t0 = time.perf_counter()
    _, table = _open_frames(lance_root, storage_options, lance_version)
    if "semantic_nn_dist" not in table.schema.names:
        raise KeyError("semantic_nn_dist missing; run tag_semantic_duplicates() first")
    pred = _and("semantic_nn_dist IS NOT NULL", where, f"semantic_nn_dist >= {float(min_nn_dist)}" if min_nn_dist is not None else None)
    tbl = table.search().where(pred).select(["index", "episode_index", "frame_index", "semantic_nn_dist"]).limit(None).to_arrow()
    eps = _col_to_numpy(tbl, "episode_index", np.int64)
    frs = _col_to_numpy(tbl, "frame_index", np.int64)
    gid = _col_to_numpy(tbl, "index", np.int64)
    d = _col_to_numpy(tbl, "semantic_nn_dist", np.float64)
    order = np.lexsort((frs, eps, -d))[: int(top_k)]
    hits = [SearchHit(int(eps[i]), int(frs[i]), int(gid[i]), round(float(d[i]), 6)) for i in order]
    return SearchResult(hits, {"type": "rare_frames", "top_k": int(top_k)}, pred, "scalar_scan", int(top_k), round(time.perf_counter() - t0, 4))


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------
def _parse_kv(items: list[str] | None) -> dict[str, str] | None:
    if not items:
        return None
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE storage option, got {item!r}")
        key, val = item.split("=", 1)
        out[key] = val
    return out


def _emit(payload: dict[str, Any], out_json: str | None) -> None:
    text = json.dumps(payload, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o)) + "\n"
    if out_json:
        p = Path(out_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    print(text, end="")


def main_vector_curate(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="gr00t-lance-vector-curate",
        description=(
            "Embedding column + IVF-PQ vector index on frames.lance: semantic dedup, hybrid "
            "(vector + SQL) search and scene slicing. Never writes videos.lance."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--root", required=True, help="Lance dataset directory or s3:// URI.")
        p.add_argument("--column", default=EMBEDDING_COLUMN, help="Embedding column (default: embedding).")
        p.add_argument("--storage-option", action="append", default=None, help="Storage KEY=VALUE (repeatable).")
        p.add_argument("--out-json", default=None, help="Also write the JSON report to this path.")

    p_embed = sub.add_parser("embed", help="Compute embeddings and append them to frames.lance.")
    common(p_embed)
    p_embed.add_argument("--provider", choices=("kinematic", "visual"), default="kinematic")
    p_embed.add_argument("--horizon", type=int, default=16, help="Kinematic: action-chunk length.")
    p_embed.add_argument("--dim", type=int, default=64, help="Kinematic: random-projection dim (0 = none).")
    p_embed.add_argument("--seed", type=int, default=0)
    p_embed.add_argument("--every-n", type=int, default=1, help="Embed every N-th frame per episode (others NULL).")
    p_embed.add_argument("--camera", default=None, help="Visual: modality video key or feature key.")
    p_embed.add_argument("--model-path", default=None, help="Visual: LOCAL SigLIP/DINOv2 directory (no downloads).")
    p_embed.add_argument("--device", default="cpu")

    p_index = sub.add_parser("index", help="Build the IVF-PQ (cosine) index, or record flat fallback.")
    common(p_index)
    p_index.add_argument("--num-partitions", type=int, default=None)
    p_index.add_argument("--num-sub-vectors", type=int, default=None)
    p_index.add_argument("--min-rows", type=int, default=256)
    p_index.add_argument("--force-flat", action="store_true")

    p_dedup = sub.add_parser("dedup", help="Tag semantic near-duplicates (is_semantic_dup, ...).")
    common(p_dedup)
    p_dedup.add_argument("--threshold", type=float, default=0.02, help="Cosine distance threshold (default 0.02).")
    p_dedup.add_argument("--scope", choices=("cross_episode", "within_episode", "all"), default="cross_episode")
    p_dedup.add_argument("--k", type=int, default=10)
    p_dedup.add_argument("--where", default=None, help="Candidate SQL filter, e.g. 'is_valid'.")
    p_dedup.add_argument("--mode", choices=("auto", "index", "flat"), default="auto")
    p_dedup.add_argument("--nprobes", type=int, default=20)
    p_dedup.add_argument("--refine-factor", type=int, default=10)
    p_dedup.add_argument("--query-batch-size", type=int, default=1024)

    p_search = sub.add_parser("search", help="Hybrid vector + SQL search / scene slicing / rare frames.")
    common(p_search)
    p_search.add_argument("--episode", type=int, default=None)
    p_search.add_argument("--frame", type=int, default=None)
    p_search.add_argument("--vector-json", default=None, help="Query vector as a JSON list.")
    p_search.add_argument("--where", default=None)
    p_search.add_argument("--k", type=int, default=10)
    p_search.add_argument("--level", choices=("frame", "episode", "rare"), default="frame",
                          help="frame: kNN frames; episode: rank episodes similar to --episode; rare: hard examples.")
    p_search.add_argument("--include-query-episode", action="store_true")
    p_search.add_argument("--mode", choices=("auto", "index", "flat"), default="auto")
    p_search.add_argument("--lance-version", type=int, default=None)

    args = parser.parse_args(argv)
    opts = _parse_kv(args.storage_option)

    if args.cmd == "embed":
        if args.provider == "visual":
            if not args.camera or not args.model_path:
                parser.error("--provider visual needs --camera and --model-path")
            provider: Any = VisualEmbeddingProvider(args.camera, model_path=args.model_path, device=args.device)
        else:
            provider = KinematicEmbeddingProvider(horizon=args.horizon, dim=args.dim or None, seed=args.seed)
        rep: _Report = add_embedding_column(args.root, provider, column=args.column, every_n=args.every_n, storage_options=opts)
        _emit(rep.to_dict(), args.out_json)
    elif args.cmd == "index":
        rep = build_vector_index(
            args.root,
            column=args.column,
            num_partitions=args.num_partitions,
            num_sub_vectors=args.num_sub_vectors,
            min_rows=args.min_rows,
            force_flat=args.force_flat,
            storage_options=opts,
        )
        _emit(rep.to_dict(), args.out_json)
    elif args.cmd == "dedup":
        rep = tag_semantic_duplicates(
            args.root,
            column=args.column,
            threshold=args.threshold,
            scope=args.scope,
            k=args.k,
            where=args.where,
            mode=args.mode,
            nprobes=args.nprobes,
            refine_factor=args.refine_factor or None,
            query_batch_size=args.query_batch_size,
            storage_options=opts,
        )
        _emit(rep.to_dict(), args.out_json)
    else:
        if args.level == "rare":
            res: _Report = mine_rare_frames(args.root, top_k=args.k, where=args.where, storage_options=opts, lance_version=args.lance_version)
        elif args.level == "episode":
            if args.episode is None:
                parser.error("--level episode needs --episode")
            res = slice_episodes_by_similarity(
                args.root, args.episode, k=args.k, where=args.where, column=args.column, mode=args.mode,
                storage_options=opts, lance_version=args.lance_version,
            )
        else:
            vec = json.loads(args.vector_json) if args.vector_json else None
            if vec is None and args.episode is None:
                parser.error("search needs --episode [--frame] or --vector-json")
            res = hybrid_search(
                args.root, query_vector=vec, episode=args.episode, frame=args.frame, where=args.where, k=args.k,
                column=args.column, exclude_query_episode=not args.include_query_episode, mode=args.mode,
                storage_options=opts, lance_version=args.lance_version,
            )
        _emit(res.to_dict(), args.out_json)


__all__ = [
    "EMBEDDING_COLUMN",
    "SEMANTIC_DUP_COLUMNS",
    "EmbeddingProvider",
    "EmbeddingReport",
    "EpisodeScore",
    "EpisodeSliceResult",
    "FrameBatch",
    "FunctionEmbeddingProvider",
    "KinematicEmbeddingProvider",
    "SearchHit",
    "SearchResult",
    "SemanticDedupReport",
    "VectorIndexReport",
    "VisualEmbeddingProvider",
    "add_embedding_column",
    "anchors_to_where",
    "build_vector_index",
    "combine_where",
    "episodes_to_where",
    "greedy_duplicate_assignment",
    "hybrid_search",
    "load_hf_image_encoder",
    "main_vector_curate",
    "mine_rare_frames",
    "slice_episodes_by_similarity",
    "suggest_ivf_pq_params",
    "tag_semantic_duplicates",
]
