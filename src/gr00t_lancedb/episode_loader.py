# SPDX-License-Identifier: Apache-2.0
"""LanceDB-backed drop-in replacement for GR00T's ``LeRobotEpisodeLoader``.

Streams tabular and video frames directly from local or object-store (``s3://``,
``gs://``, ``hf://``) Lance tables via ``lerobot_lancedb.LanceDBDataset`` while
preserving 100% of GR00T's ``modality.json`` slicing, camera key mapping,
language annotation resolution, and ``VLAStepData`` output contract.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import re
import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch

from ._gr00t_compat import EmbodimentTag, ModalityConfig, VLAStepData
from .metadata import (
    CURATION_VERSIONS_FILENAME,
    DEFAULT_COLUMN_NAMES,
    LANG_KEYS,
    extract_dataset_statistics,
    load_gr00t_meta_bundle,
    parse_and_validate_modality_configs,
    resolve_video_original_key,
    select_curation_entry,
)

if TYPE_CHECKING:  # imported lazily at runtime: lerobot + lancedb + av cost ~2-3 s
    from lerobot_lancedb import LanceDBDataset

# lancedb registers its blob Arrow extension type lazily on first table open; two threads
# opening at once race on that registration ("lance.blob.v2 is already registered").
_OPEN_LOCK = threading.Lock()

# Same value as ``lerobot_lancedb.FRAMES_TABLE``; duplicated so that importing this module
# never imports lerobot_lancedb (and through it lerobot, lancedb, av) in the main process.
FRAMES_TABLE = "frames"
DEFAULT_VIDEO_DECODER_CACHE_SIZE = 64

logger = logging.getLogger(__name__)


def is_remote_uri(path: str | Path) -> bool:
    """``s3://``, ``gs://``, ``hf://`` ... (``file://`` counts as local)."""
    raw = str(path)
    return "://" in raw and not raw.startswith("file://")


def _user_cache_dir() -> Path:
    base = os.environ.get("GR00T_LANCE_CACHE_DIR")
    if base:
        return Path(base)
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "gr00t_lancedb"


def _remote_meta_cache_root(
    root: str, storage_options: dict | None, revision: str | None, token: str | bool | None
) -> Path | None:
    """Mirror of ``lerobot_lancedb.reader.resolve_lance_root``'s cache path (no Lance I/O).

    Returns ``None`` if lerobot-lancedb's internals changed, in which case the caller falls
    back to the in-process resolution.
    """
    try:
        from lerobot_lancedb import reader as _r

        db_uri = str(root).rstrip("/")
        options = _r._storage_options(db_uri, storage_options, revision, token)
        cache_key = db_uri
        routing = {k.lower(): v for k, v in options.items() if k.lower() in _r._METADATA_ROUTING_OPTIONS}
        if db_uri.startswith("hf://"):
            ref = options.get("hf_revision", options.get("revision"))
            if ref:
                routing["revision"] = ref
        if routing:
            identity = json.dumps(routing, sort_keys=True, separators=(",", ":"))
            cache_key += "-" + hashlib.sha256(identity.encode()).hexdigest()[:16]
        return Path(_r.HF_LEROBOT_HOME) / "remote" / re.sub(r"[^A-Za-z0-9._-]+", "_", cache_key)
    except Exception:  # noqa: BLE001
        return None


_FORK_PRELOAD = {"armed": False}


def _preload_before_fork() -> None:
    """``os.register_at_fork(before=...)``: make forked workers inherit the Lance stack.

    The main process deliberately does not import lerobot-lancedb at dataset build time.
    Right before the first ``fork()`` (DataLoader workers) we join a running background
    preload, or import synchronously once, so (1) no fork happens mid-import, (2) workers
    do not each pay the import on the first batch, and (3) macOS' ObjC fork-safety check
    is not tripped by first-time framework initialization inside a child.
    """
    import sys

    if not _FORK_PRELOAD["armed"] or os.environ.get("GR00T_LANCE_PRELOAD_BEFORE_FORK", "1") != "1":
        return
    try:
        import gr00t_lancedb as _pkg

        if "lerobot_lancedb" in sys.modules and _pkg._PRELOAD["thread"] is None:
            return
        if _pkg._PRELOAD["thread"] is None:
            _pkg.preload_lance_stack(background=False)
        _pkg.join_preload()
    except Exception as exc:  # noqa: BLE001 - never block a fork
        logger.debug("pre-fork Lance preload failed: %s", exc)
    finally:
        _FORK_PRELOAD["armed"] = False


if hasattr(os, "register_at_fork"):
    os.register_at_fork(before=_preload_before_fork)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    try:
        if path.read_bytes() == data:
            return
    except OSError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _tensor_chw_to_hwc_uint8(frame: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert a decoded CHW uint8 tensor (or HWC array) to contiguous HWC uint8 ndarray."""
    if isinstance(frame, torch.Tensor):
        if frame.ndim == 3 and frame.shape[0] in (1, 3, 4):
            arr = frame.permute(1, 2, 0).cpu().numpy()
        else:
            arr = frame.cpu().numpy()
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.0:
                arr = np.rint(arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
        return np.ascontiguousarray(arr)
    arr = np.asarray(frame)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    return np.ascontiguousarray(arr.astype(np.uint8, copy=False))


def _resolve_step_indices(
    step_indices: np.ndarray,
    delta_indices: Sequence[int],
    length: int,
    allow_padding: bool,
) -> np.ndarray:
    """Resolve ``step_indices[:, None] + delta_indices[None, :]`` matching GR00T exactly."""
    deltas = np.asarray(delta_indices, dtype=np.int64)
    rows = np.asarray(step_indices, dtype=np.int64)[:, None] + deltas[None, :]
    if allow_padding:
        return np.clip(rows, 0, length - 1)
    # Match pandas .iloc negative indexing semantics (mirrors FastShardedSingleStepDataset).
    return np.where(rows < 0, rows + length, rows)


class LanceEpisodeLoader:
    """Episode- and step-level loader for GR00T backed by ``lerobot-lancedb``.

    Unlike GR00T's local-disk ``LeRobotEpisodeLoader``, ``dataset_path`` may be:
    - a local Lance dataset directory (copy or Hybrid external mode), or
    - a remote object-store URI (``s3://...``, ``gs://...``, ``hf://...``) where
      ``meta/`` is materialized once from ``meta.lance`` and frames/MP4 byte ranges
      are streamed on demand.
    """

    def __init__(
        self,
        dataset_path: str | Path,
        modality_configs: dict[str, ModalityConfig],
        *,
        repo_id: str | None = None,
        modality_json_path: str | Path | None = None,
        stats_json_path: str | Path | None = None,
        storage_options: dict[str, str] | None = None,
        revision: str | None = None,
        token: str | bool | None = None,
        use_mini_mp4: bool = True,
        video_decoder_cache_size: int = DEFAULT_VIDEO_DECODER_CACHE_SIZE,
        decoder_kwargs: dict[str, Any] | None = None,
        where: str | None = None,
        lance_version: int | None = None,
        video_cache_dir: str | Path | None = None,
        video_cache_gb: float | None = None,
        video_cache_force_local: bool | None = None,
    ) -> None:
        self.dataset_path = str(dataset_path)
        self.repo_id = repo_id
        self.modality_json_path = str(modality_json_path) if modality_json_path is not None else None
        self.stats_json_path = str(stats_json_path) if stats_json_path is not None else None
        self.storage_options = dict(storage_options) if storage_options else None
        self.revision = revision
        self.token = token
        self.use_mini_mp4 = use_mini_mp4
        self.video_decoder_cache_size = video_decoder_cache_size
        self.decoder_kwargs = decoder_kwargs
        self.where = where
        self.lance_version = int(lance_version) if lance_version is not None else None
        # Effective MVCC snapshot of frames.lance for anchors, rows and stats: the explicit
        # ``lance_version`` or, with ``where``, the version the predicate was evaluated at.
        self.frames_version: int | None = self.lance_version
        self.stats_snapshot: dict[str, Any] | None = None
        # Startup accounting for benchmarks (time spent in isolated Lance child processes ...).
        self.startup_info: dict[str, Any] = {"isolated_calls": 0, "isolated_wall_s": 0.0}

        from .gop_cache import resolve_cache_config

        self._video_cache_config = resolve_cache_config(
            self.dataset_path,
            cache_dir=video_cache_dir,
            cache_gb=video_cache_gb,
            force_local=video_cache_force_local,
        )

        _FORK_PRELOAD["armed"] = True
        # Resolve the local meta/ root without starting Lance/Tokio in this process.
        self._meta_opened_lance_pid: int | None = None
        self.local_meta_root: Path = self._resolve_meta_root()
        meta_dir = self.local_meta_root / "meta"

        # Frame-level SQL pushdown + version-pinned stats, computed out of process.
        where_frames: tuple[np.ndarray, np.ndarray] | None = None
        versioned_meta: dict[str, Any] | None = None
        if self.where is not None:
            where_frames, versioned_meta = self._query_where(self.where, self.lance_version)
        effective_stats_path = self.stats_json_path
        if effective_stats_path is None and self.frames_version is not None:
            effective_stats_path = self._resolve_versioned_stats(meta_dir, self.frames_version, versioned_meta)

        bundle = load_gr00t_meta_bundle(
            meta_dir,
            modality_json_path=self.modality_json_path,
            stats_json_path=effective_stats_path,
        )
        self.info_meta = bundle.info_meta
        self.modality_meta = bundle.modality_meta
        self.episodes_metadata = bundle.episodes_metadata
        self.tasks_map = bundle.tasks_map
        self.stats = bundle.stats
        self._initial_actions = bundle.initial_actions
        self.feature_config = bundle.feature_config
        self.chunk_size = bundle.chunk_size
        self.fps = bundle.fps

        self.modality_configs, self._video_key_mapping = parse_and_validate_modality_configs(
            modality_configs, self.modality_meta
        )

        # Compute exact minimal column projection for LanceDBDataset so unselected
        # cameras and unused tabular columns are never read from S3.
        self._video_orig_keys: dict[str, str] = {}
        projected_columns: list[str] = []
        available_features = set(self.feature_config.keys())

        for modality_type in ("state", "action"):
            if modality_type not in self.modality_configs:
                continue
            mod_meta = self.modality_meta.get(modality_type, {})
            for group_name in self.modality_configs[modality_type].modality_keys:
                if group_name in mod_meta:
                    orig_key = mod_meta[group_name].get("original_key", DEFAULT_COLUMN_NAMES[modality_type])
                    if orig_key not in projected_columns and orig_key in available_features:
                        projected_columns.append(orig_key)

        if "language" in self.modality_configs:
            for key in self.modality_configs["language"].modality_keys:
                if key in LANG_KEYS:
                    continue
                if not key.startswith("annotation."):
                    raise AssertionError(f"Unsupported language key {key!r}; expected 'task', 'sub_task', or 'annotation.*'")
                subkey = key.replace("annotation.", "", 1)
                if subkey not in self.modality_meta.get("annotation", {}):
                    raise AssertionError(f"Key {subkey!r} not found in modality.json annotation section")
                orig_key = self.modality_meta["annotation"][subkey].get("original_key", key)
                if orig_key not in projected_columns and orig_key in available_features:
                    projected_columns.append(orig_key)

        if "video" in self.modality_configs:
            for img_key in self.modality_configs["video"].modality_keys:
                orig_key = resolve_video_original_key(img_key, self.modality_meta, self._video_key_mapping)
                if orig_key not in available_features:
                    raise AssertionError(f"Original video key {orig_key!r} not found in feature config")
                self._video_orig_keys[img_key] = orig_key
                if orig_key not in projected_columns:
                    projected_columns.append(orig_key)

        if not projected_columns:
            # Ensure at least one valid column is projected for metadata-only tests.
            projected_columns = [next(iter(available_features))]

        self._projected_columns = projected_columns
        self._reader_kwargs = dict(
            repo_id=repo_id,
            root=self.dataset_path,
            return_uint8=True,
            storage_options=self.storage_options,
            revision=self.revision,
            token=self.token,
            columns=self._projected_columns,
            use_mini_mp4=self.use_mini_mp4,
            video_decoder_cache_size=self.video_decoder_cache_size,
        )
        # Opened lazily on first read inside each worker so the main process never starts
        # Lance/Tokio before fork(), and shared across decode threads via _io_lock (avoiding
        # per-thread reader cloning & 16-thread decode-pool explosion).
        self._reader0: LanceDBDataset | None = None
        self._opened_pid: int | None = None
        self._ep_bounds_cache: dict[int, tuple[int, int]] = {}
        self._init_lock()

        # Map GR00T episode positional idx -> Lance episode_index & bounds.
        self.episode_lengths = [int(ep["length"]) for ep in self.episodes_metadata]
        self._ep_id_to_pos = {
            int(ep["episode_index"]): pos for pos, ep in enumerate(self.episodes_metadata)
        }

        # Optional frame-level SQL predicate pushdown against frames.lance (zero video reads).
        self.valid_steps_by_episode: dict[int, np.ndarray] | None = None
        if where_frames is not None:
            self.valid_steps_by_episode = self._group_valid_steps(*where_frames)

    # -- metadata root / where / versioned stats (no Lance in this process) -------------
    def _note_isolated(self, res: dict[str, Any]) -> None:
        self.startup_info["isolated_calls"] += 1
        self.startup_info["isolated_wall_s"] += float(res.get("_isolated_wall_s", 0.0))

    def _isolated_kwargs(self) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "repo_id": self.repo_id,
            "storage_options": self.storage_options,
            "revision": self.revision,
            "token": self.token,
        }

    def _resolve_meta_root(self) -> Path:
        path = self.dataset_path
        if "://" not in path and self.repo_id is None:
            root = Path(path)
            if (root / f"{FRAMES_TABLE}.lance").exists():
                return root
            raise FileNotFoundError(f"No '{FRAMES_TABLE}.lance' table under {root}.")
        if "://" in path:
            cached = _remote_meta_cache_root(path, self.storage_options, self.revision, self.token)
            if cached is not None and (cached / "meta").exists():
                return cached
            if cached is not None:
                # Cold remote meta: materialize meta.lance -> meta/ in a child process so the
                # main process stays Tokio-free and DataLoader workers can still use fork.
                from ._isolated import run_isolated

                res = run_isolated("gr00t_lancedb.lance_query:materialize_meta", self._isolated_kwargs())
                self._note_isolated(res)
                return Path(res["local_root"])
        # hf:// repo ids (or unexpected lerobot-lancedb internals): resolve in process.
        from lerobot_lancedb.reader import resolve_lance_root

        t_before_ns = time.time_ns()
        _db_uri, resolved_root = resolve_lance_root(
            repo_id=self.repo_id,
            root=self.dataset_path,
            storage_options=self.storage_options,
            revision=self.revision,
            token=self.token,
        )
        root = Path(resolved_root)
        try:
            if (root / "meta").stat().st_mtime_ns >= t_before_ns:
                self._meta_opened_lance_pid = os.getpid()
        except OSError:
            self._meta_opened_lance_pid = os.getpid()
        return root

    def _where_cache_path(self, where: str, version: int) -> Path:
        ident = json.dumps([self.dataset_path, self.repo_id, where, int(version)])
        return _user_cache_dir() / "where" / f"{hashlib.sha256(ident.encode()).hexdigest()[:24]}.npz"

    def _query_where(
        self, where: str, lance_version: int | None
    ) -> tuple[tuple[np.ndarray, np.ndarray], dict[str, Any] | None]:
        """Matching ``(episode_index, frame_index)`` pairs for ``where`` at one version.

        With an explicit version the result is immutable (MVCC), so it is cached on disk:
        a resumed run after preemption skips the query entirely.
        """
        if lance_version is not None:
            cache = self._where_cache_path(where, lance_version)
            try:
                with np.load(cache) as z:
                    self.startup_info["where_cache_hit"] = True
                    return (z["episode_index"], z["frame_index"]), None
            except (OSError, KeyError, ValueError):
                pass
        from ._isolated import run_isolated

        res = run_isolated(
            "gr00t_lancedb.lance_query:query_where_anchors",
            {**self._isolated_kwargs(), "where": where, "lance_version": lance_version},
        )
        self._note_isolated(res)
        self.startup_info["where_cache_hit"] = False
        self.frames_version = int(res["frames_version"])
        ep, fr = np.asarray(res["episode_index"], np.int64), np.asarray(res["frame_index"], np.int64)
        try:
            cache = self._where_cache_path(where, self.frames_version)
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_name(f".{cache.stem}.{os.getpid()}.tmp.npz")
            np.savez(tmp, episode_index=ep, frame_index=fr)
            os.replace(tmp, cache)
        except OSError:
            pass
        return (ep, fr), res.get("versioned_meta")

    def _resolve_versioned_stats(
        self, meta_dir: Path, frames_version: int, versioned_meta: dict[str, Any] | None
    ) -> str | None:
        """Path of the ``gr00t_stats.json`` snapshot in force at ``frames_version`` (issue #2).

        Source of truth is ``curation_versions.json`` in the latest ``meta.lance``. Local
        datasets keep ``meta/`` in sync with it; for object-store datasets a possibly stale
        local cache is refreshed from ``meta.lance`` through an isolated child.
        """
        local_manifest_path = meta_dir / CURATION_VERSIONS_FILENAME

        def local_entry() -> dict[str, Any] | None:
            try:
                manifest = json.loads(local_manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return None
            entry = select_curation_entry(manifest, frames_version)
            if entry and (meta_dir / entry["gr00t_stats"]).exists():
                return entry
            return None

        entry = None
        if versioned_meta is None:
            entry = local_entry()
            if entry is None and is_remote_uri(self.dataset_path):
                from ._isolated import run_isolated

                versioned_meta = run_isolated(
                    "gr00t_lancedb.lance_query:fetch_versioned_meta",
                    {**self._isolated_kwargs(), "frames_version": frames_version},
                )
                self._note_isolated(versioned_meta)
        if entry is None and versioned_meta is not None:
            if versioned_meta.get("manifest"):
                _atomic_write_bytes(local_manifest_path, versioned_meta["manifest"])
            for name, data in (versioned_meta.get("files") or {}).items():
                if data is not None:
                    _atomic_write_bytes(meta_dir / name, data)
            entry = local_entry()
        if entry is None:
            logger.info(
                "No curation stats snapshot for frames.lance v%s in %s; using default stats.",
                frames_version,
                self.dataset_path,
            )
            return None
        self.stats_snapshot = dict(entry)
        if int(entry.get("frames_version", frames_version)) < frames_version:
            logger.info(
                "frames.lance v%s was not produced by curation; using the newest snapshot (v%s).",
                frames_version,
                entry.get("frames_version"),
            )
        return str(meta_dir / entry["gr00t_stats"])

    def _group_valid_steps(self, ep_arr: np.ndarray, frame_arr: np.ndarray) -> dict[int, np.ndarray]:
        """Per positional episode: sorted unique matching anchor frame indices."""
        valid_by_pos: dict[int, np.ndarray] = {
            pos: np.empty(0, dtype=np.int64) for pos in range(len(self.episodes_metadata))
        }
        if len(ep_arr) == 0:
            return valid_by_pos
        order = np.lexsort((frame_arr, ep_arr))
        ep_sorted = ep_arr[order]
        frame_sorted = frame_arr[order]
        unique_eps, split_starts = np.unique(ep_sorted, return_index=True)
        split_ends = np.append(split_starts[1:], len(ep_sorted))
        for ep_id_np, lo_i, hi_i in zip(unique_eps, split_starts, split_ends, strict=True):
            pos = self._ep_id_to_pos.get(int(ep_id_np))
            if pos is None:
                continue
            ep_frames = np.unique(frame_sorted[lo_i:hi_i])
            ep_len = self.episode_lengths[pos]
            lo = int(np.searchsorted(ep_frames, 0, side="left"))
            hi = int(np.searchsorted(ep_frames, ep_len, side="left"))
            valid_by_pos[pos] = np.ascontiguousarray(ep_frames[lo:hi], dtype=np.int64)
        return valid_by_pos

    def get_valid_steps_for_episode(
        self,
        ep_idx: int,
        *,
        max_step_exclusive: int | None = None,
    ) -> np.ndarray:
        """Return sorted valid anchor ``frame_index`` values for positional episode ``ep_idx``."""
        limit = self.episode_lengths[ep_idx]
        if max_step_exclusive is not None:
            limit = min(limit, int(max_step_exclusive))
        if limit <= 0:
            return np.empty(0, dtype=np.int64)
        if self.valid_steps_by_episode is None:
            return np.arange(0, limit, dtype=np.int64)
        steps = self.valid_steps_by_episode.get(ep_idx)
        if steps is None or len(steps) == 0:
            return np.empty(0, dtype=np.int64)
        cutoff = int(np.searchsorted(steps, limit, side="left"))
        return steps[:cutoff]

    # -- lazy worker-shared reader & fork safety ------------------------------------
    def _init_lock(self) -> None:
        self._io_lock = threading.Lock()

    @staticmethod
    def _reader_is_open(r: LanceDBDataset | None) -> bool:
        if r is None:
            return False
        return any(
            getattr(r, attr, None) is not None
            for attr in ("_frames_perm", "_videos_table", "_video_row_ids", "_prefetch_pool", "_decode_pool")
        )

    @property
    def is_open(self) -> bool:
        """Whether this loader has opened any Lance table handle or thread pool."""
        return self._reader_is_open(self._reader0)

    def assert_unopened(self) -> None:
        """Raise ``RuntimeError`` if Lance handles or Tokio runtime were opened in this process."""
        if self.is_open:
            raise RuntimeError(
                f"LanceEpisodeLoader for {self.dataset_path!r} already opened Lance handles/pools "
                f"in pid {os.getpid()}; cannot safely fork() after opening Lance/Tokio."
            )
        if self._meta_opened_lance_pid == os.getpid():
            raise RuntimeError(
                f"LanceEpisodeLoader for {self.dataset_path!r} touched Lance/Tokio while "
                f"materializing remote meta.lance in pid {os.getpid()}; use forkserver/spawn."
            )

    def _ensure_reader_locked(self) -> LanceDBDataset:
        """Return an open ``LanceDBDataset``; caller MUST hold ``self._io_lock``."""
        r = self._reader0
        if r is None:
            from lerobot_lancedb import LanceDBDataset

            r = LanceDBDataset(**self._reader_kwargs)
            self._reader0 = r
        needs_open = getattr(r, "_frames_perm", None) is None or (
            bool(getattr(r, "_video_keys", None))
            and (getattr(r, "_prefetch_pool", None) is None or getattr(r, "_decode_pool", None) is None)
        )
        if needs_open:
            with _OPEN_LOCK:
                # If pools were closed on a previously opened reader, reset _frames_perm so
                # _ensure_open() recreates pools and handles cleanly.
                if getattr(r, "_frames_perm", None) is not None:
                    r._frames_perm = None
                r._ensure_open()
                if self.frames_version is not None:
                    self._pin_frames_version(r, self.frames_version)
                if getattr(r, "_videos_table", None) is not None:
                    from .gop_cache import wrap_videos_table

                    wrap_videos_table(
                        r,
                        dataset_id=str(getattr(r, "_db_uri", self.dataset_path)),
                        cache_config=self._video_cache_config,
                    )
            self._opened_pid = os.getpid()
        return r

    @staticmethod
    def _pin_frames_version(r: LanceDBDataset, version: int) -> None:
        """Point the reader's row permutation at one MVCC snapshot of ``frames.lance``."""
        from lancedb.permutation import Permutation
        from lerobot_lancedb.reader import _connect, _open_table

        options = getattr(r, "_storage_options", None)
        table = _open_table(_connect(r._db_uri, options), FRAMES_TABLE, options or None)
        table.checkout(int(version))
        n_rows = table.count_rows()
        if n_rows != r.meta.total_frames:
            raise ValueError(
                f"frames.lance v{version} has {n_rows} rows but meta declares "
                f"{r.meta.total_frames}; cannot pin a version with a different row count."
            )
        r._frames_perm = Permutation.identity(table).select_columns(r._fetch_columns).with_format("arrow")

    def _episode_bounds(self, episode_id: int) -> tuple[int, int]:
        bounds = self._ep_bounds_cache.get(episode_id)
        if bounds is None:
            with self._io_lock:
                bounds = self._ep_bounds_cache.get(episode_id)
                if bounds is None:
                    bounds = self._ensure_reader_locked()._episode_bounds(episode_id)
                    self._ep_bounds_cache[episode_id] = bounds
        return bounds

    @property
    def reader(self) -> LanceDBDataset:
        r = self._reader0
        if r is not None and self._reader_is_open(r):
            return r
        with self._io_lock:
            return self._ensure_reader_locked()

    @reader.setter
    def reader(self, value: LanceDBDataset | None) -> None:
        """Replace the base reader (e.g. a per-worker copy); resets lock and bounds cache."""
        self._reader0 = value
        self._opened_pid = os.getpid() if self._reader_is_open(value) else None
        self._ep_bounds_cache = {}
        self._init_lock()

    def __copy__(self) -> LanceEpisodeLoader:
        cls = self.__class__
        dup = cls.__new__(cls)
        dup.__dict__.update(self.__dict__)
        dup._ep_bounds_cache = dict(self._ep_bounds_cache)
        dup._init_lock()
        return dup

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("_io_lock", None)
        # Never pickle an open Lance reader across worker processes; workers open lazily.
        state["_reader0"] = None
        state["_opened_pid"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._init_lock()

    def __len__(self) -> int:
        return len(self.episodes_metadata)

    def get_episode_lengths(self) -> list[int]:
        return list(self.episode_lengths)

    def get_episode_length(self, idx: int) -> int:
        return self.episode_lengths[idx]

    def get_dataset_statistics(self) -> dict[str, Any]:
        return extract_dataset_statistics(self.modality_configs, self.modality_meta, self.stats)

    def get_initial_actions(self) -> list[dict[str, Any]]:
        return list(self._initial_actions)

    def create_language_from_meta(
        self,
        episode_meta: dict[str, Any],
        nframes: int,
        lang_key: str,
        *,
        rng: np.random.Generator | None = None,
    ) -> list[str]:
        """Mirror ``LeRobotEpisodeLoader.create_language_from_meta`` with optional deterministic RNG."""
        if lang_key == "task":
            tasks = episode_meta.get("tasks") or [""]
            if len(tasks) == 1:
                chosen = tasks[0]
            elif rng is not None:
                chosen = tasks[int(rng.integers(len(tasks)))]
            else:
                chosen = random.choice(tasks)
            return [chosen] * nframes
        if lang_key == "sub_task":
            action_delta_indices = self.modality_configs["action"].delta_indices
            action_horizon = max(action_delta_indices) - min(action_delta_indices) + 1
            new_languages: list[list[str]] = [[] for _ in range(nframes)]
            sub_tasks = episode_meta.get("sub_tasks", [])
            horizon = action_horizon // 2
            for sub_task in sub_tasks:
                start_idx = int(sub_task["start"])
                end_idx = int(sub_task["end"])
                sub_text = str(sub_task["text"])
                for i in range(max(0, start_idx - horizon), min(nframes, end_idx)):
                    new_languages[i].append(sub_text)
            result: list[str] = []
            for choices in new_languages:
                if not choices:
                    result.append("")
                elif len(choices) == 1:
                    result.append(choices[0])
                elif rng is not None:
                    result.append(choices[int(rng.integers(len(choices)))])
                else:
                    result.append(random.choice(choices))
            return result
        raise ValueError(f"Language key {lang_key!r} not supported")

    def _fetch_tabular_and_decode_videos(
        self,
        tabular_abs_rows: list[int],
        video_abs_rows: list[int],
    ) -> tuple[dict[str, np.ndarray], dict[int, int], dict[int, dict[str, np.ndarray]]]:
        """Single deduplicated tabular read + sparse batched video decode via ``LanceDBDataset``.

        Returns:
            - ``columns``: dict[orig_key, ndarray] indexed by ``row_pos[abs_row]``
            - ``row_pos``: dict[abs_row, int]
            - ``decoded_videos``: dict[abs_row, dict[gr00t_video_key, HWC uint8 ndarray]]
        """
        all_rows = sorted(set(tabular_abs_rows) | set(video_abs_rows))
        row_pos = {row: pos for pos, row in enumerate(all_rows)}

        decoded_videos: dict[int, dict[str, np.ndarray]] = {}
        decoded_list: list[dict[str, Any]] = []
        if all_rows:
            from .gop_cache import IO_STATS

            IO_STATS.add(frames_take_calls=1, frames_take_rows=len(all_rows))
        with self._io_lock:
            reader = self._ensure_reader_locked()
            if video_abs_rows and self._video_orig_keys:
                video_plans = reader._plan_batch(video_abs_rows)
                windows = reader._plan_file_windows(video_plans)
                prefetch_fut = reader._prefetch_pool.submit(
                    reader._prepare_files, sorted(windows), windows
                )
                columns = reader._fetch_rows(all_rows)
                prepared = prefetch_fut.result()
                decoded_list = reader._decode_videos(video_plans, columns, row_pos, prepared)
            else:
                columns = reader._fetch_rows(all_rows) if all_rows else {}

        if decoded_list:
            for abs_row, raw_frames in zip(video_abs_rows, decoded_list, strict=True):
                decoded_videos[abs_row] = {
                    img_key: _tensor_chw_to_hwc_uint8(raw_frames[orig_key])
                    for img_key, orig_key in self._video_orig_keys.items()
                }

        return columns, row_pos, decoded_videos

    def _resolve_annotation_value(self, raw_val: Any) -> str:
        if isinstance(raw_val, (int, np.integer)):
            return self.tasks_map[int(raw_val)]
        return str(raw_val)

    def load_episode(
        self,
        idx: int,
        select_frames: Callable[[int], np.ndarray] | None = None,
        *,
        rng: np.random.Generator | None = None,
    ) -> pd.DataFrame:
        """Drop-in equivalent of ``LeRobotEpisodeLoader.load_episode`` returning a DataFrame."""
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Episode index {idx} out of bounds")

        episode_meta = self.episodes_metadata[idx]
        episode_id = int(episode_meta["episode_index"])
        nominal_length = int(episode_meta["length"])
        ep_start, ep_end = self._episode_bounds(episode_id)
        actual_length = min(ep_end - ep_start, nominal_length)
        tabular_abs_rows = list(range(ep_start, ep_start + actual_length))

        frame_indices: np.ndarray | None = None
        if "video" not in self.modality_configs or not self._video_orig_keys:
            video_abs_rows: list[int] = []
        elif select_frames is None:
            video_abs_rows = tabular_abs_rows
        else:
            frame_indices = np.unique(np.asarray(select_frames(actual_length), dtype=np.int64))
            if len(frame_indices) and (frame_indices[0] < 0 or frame_indices[-1] >= actual_length):
                raise IndexError(
                    f"frame_indices out of range [0, {actual_length}) for episode {episode_id}"
                )
            video_abs_rows = [ep_start + int(r) for r in frame_indices]

        columns, row_pos, decoded_videos = self._fetch_tabular_and_decode_videos(
            tabular_abs_rows, video_abs_rows
        )

        df_data: dict[str, list[Any]] = {}

        # Language annotations from tabular columns
        if "language" in self.modality_configs:
            for key in self.modality_configs["language"].modality_keys:
                if key in LANG_KEYS:
                    df_data[f"language.{key}"] = self.create_language_from_meta(
                        episode_meta, actual_length, key, rng=rng
                    )
                else:
                    subkey = key.replace("annotation.", "", 1)
                    orig_key = self.modality_meta["annotation"][subkey].get("original_key", key)
                    col_vals = columns[orig_key]
                    df_data[f"language.{key}"] = [
                        self._resolve_annotation_value(col_vals[row_pos[r]]) for r in tabular_abs_rows
                    ]

        # State and action joint-group slicing
        for modality_type in ("state", "action"):
            if modality_type not in self.modality_configs:
                continue
            mod_meta = self.modality_meta.get(modality_type, {})
            for group_name in self.modality_configs[modality_type].modality_keys:
                if group_name not in mod_meta:
                    logger.warning(
                        "Joint group %r not found in %s modality (available: %s)",
                        group_name,
                        modality_type,
                        list(mod_meta.keys()),
                    )
                    continue
                group_info = mod_meta[group_name]
                start_idx = int(group_info["start"])
                end_idx = int(group_info["end"])
                orig_key = group_info.get("original_key", DEFAULT_COLUMN_NAMES[modality_type])
                raw_col = columns[orig_key]
                pos_indices = [row_pos[r] for r in tabular_abs_rows]
                if isinstance(raw_col, np.ndarray) and raw_col.ndim >= 2:
                    sliced = raw_col[pos_indices, start_idx:end_idx]
                    df_data[f"{modality_type}.{group_name}"] = [sliced[i] for i in range(actual_length)]
                else:
                    df_data[f"{modality_type}.{group_name}"] = [raw_col[p] for p in pos_indices]

        # Video columns
        if "video" in self.modality_configs and self._video_orig_keys:
            for img_key in self.modality_configs["video"].modality_keys:
                if frame_indices is None:
                    df_data[f"video.{img_key}"] = [
                        decoded_videos[ep_start + r][img_key] for r in range(actual_length)
                    ]
                else:
                    col: list[Any] = [None] * actual_length
                    for r in frame_indices:
                        col[int(r)] = decoded_videos[ep_start + int(r)][img_key]
                    df_data[f"video.{img_key}"] = col

        return pd.DataFrame(df_data)

    def __getitem__(self, idx: int) -> pd.DataFrame:
        return self.load_episode(idx)

    def load_steps(
        self,
        ep_idx: int,
        step_indices: Sequence[int] | np.ndarray,
        embodiment_tag: EmbodimentTag,
        allow_padding: bool = False,
        *,
        rng: np.random.Generator | None = None,
    ) -> list[VLAStepData]:
        """Fast path: fetch only the exact tabular & video rows needed for ``step_indices``.

        Skips intermediate pandas DataFrame creation while producing outputs identical to
        ``extract_step_data(load_episode(...), step, ...)``.
        """
        if ep_idx < 0 or ep_idx >= len(self):
            raise IndexError(f"Episode index {ep_idx} out of bounds")

        steps_arr = np.asarray(step_indices, dtype=np.int64)
        if len(steps_arr) == 0:
            return []

        episode_meta = self.episodes_metadata[ep_idx]
        episode_id = int(episode_meta["episode_index"])
        nominal_length = int(episode_meta["length"])
        ep_start, ep_end = self._episode_bounds(episode_id)
        actual_length = min(ep_end - ep_start, nominal_length)

        # Compute exact row indices per modality
        resolved_by_mod: dict[str, np.ndarray] = {}
        tabular_rel_set: set[int] = set()
        video_rel_set: set[int] = set()

        for mod, cfg in self.modality_configs.items():
            mat = _resolve_step_indices(steps_arr, cfg.delta_indices, actual_length, allow_padding)
            resolved_by_mod[mod] = mat
            if mod == "video":
                video_rel_set.update(int(x) for x in np.unique(mat))
            elif mod in ("state", "action"):
                tabular_rel_set.update(int(x) for x in np.unique(mat))
            elif mod == "language":
                for lk in cfg.modality_keys:
                    if lk not in LANG_KEYS:
                        tabular_rel_set.update(int(x) for x in np.unique(mat))

        tabular_abs_rows = [ep_start + r for r in sorted(tabular_rel_set)]
        video_abs_rows = [ep_start + r for r in sorted(video_rel_set)]
        columns, row_pos, decoded_videos = self._fetch_tabular_and_decode_videos(
            tabular_abs_rows, video_abs_rows
        )

        # Resolve language per frame if meta-driven
        meta_lang_table: list[str] | None = None
        lang_key: str | None = None
        if "language" in self.modality_configs:
            lang_key = self.modality_configs["language"].modality_keys[0]
            if lang_key in LANG_KEYS:
                meta_lang_table = self.create_language_from_meta(
                    episode_meta, actual_length, lang_key, rng=rng
                )

        # Precompute slice specs for state & action
        slice_specs: dict[str, list[tuple[str, str, int, int]]] = {}
        for mod in ("state", "action"):
            if mod not in self.modality_configs:
                continue
            mod_meta = self.modality_meta.get(mod, {})
            specs = []
            for group_name in self.modality_configs[mod].modality_keys:
                info = mod_meta[group_name]
                specs.append(
                    (
                        group_name,
                        info.get("original_key", DEFAULT_COLUMN_NAMES[mod]),
                        int(info["start"]),
                        int(info["end"]),
                    )
                )
            slice_specs[mod] = specs

        results: list[VLAStepData] = []
        for s_pos in range(len(steps_arr)):
            images: dict[str, list[np.ndarray]] = {}
            if "video" in self.modality_configs and self._video_orig_keys:
                vid_rows = resolved_by_mod["video"][s_pos]
                for img_key in self.modality_configs["video"].modality_keys:
                    images[img_key] = [decoded_videos[ep_start + int(r)][img_key] for r in vid_rows]

            states: dict[str, np.ndarray] = {}
            if "state" in slice_specs:
                st_rows = resolved_by_mod["state"][s_pos]
                pos_list = [row_pos[ep_start + int(r)] for r in st_rows]
                for group_name, orig_key, start_idx, end_idx in slice_specs["state"]:
                    raw_col = columns[orig_key]
                    if isinstance(raw_col, np.ndarray) and raw_col.ndim >= 2:
                        states[group_name] = np.asarray(
                            raw_col[pos_list, start_idx:end_idx], dtype=np.float32
                        )
                    else:
                        states[group_name] = np.vstack(
                            [np.asarray(raw_col[p], dtype=np.float32) for p in pos_list]
                        )

            actions: dict[str, np.ndarray] = {}
            if "action" in slice_specs:
                act_rows = resolved_by_mod["action"][s_pos]
                pos_list = [row_pos[ep_start + int(r)] for r in act_rows]
                for group_name, orig_key, start_idx, end_idx in slice_specs["action"]:
                    raw_col = columns[orig_key]
                    if isinstance(raw_col, np.ndarray) and raw_col.ndim >= 2:
                        actions[group_name] = np.asarray(
                            raw_col[pos_list, start_idx:end_idx], dtype=np.float32
                        )
                    else:
                        actions[group_name] = np.vstack(
                            [np.asarray(raw_col[p], dtype=np.float32) for p in pos_list]
                        )

            text: str | None = None
            if lang_key is not None:
                lang_row = int(resolved_by_mod["language"][s_pos][0])
                if meta_lang_table is not None:
                    text = meta_lang_table[lang_row]
                else:
                    subkey = lang_key.replace("annotation.", "", 1)
                    orig_key = self.modality_meta["annotation"][subkey].get("original_key", lang_key)
                    text = self._resolve_annotation_value(columns[orig_key][row_pos[ep_start + lang_row]])

            results.append(
                VLAStepData(
                    images=images,
                    states=states,
                    actions=actions,
                    masks=None,
                    text=text,
                    embodiment=embodiment_tag,
                )
            )
        return results

    def load_cross_episode_batch(
        self,
        ep_step_pairs: Sequence[tuple[int, int]],
        embodiment_tag: EmbodimentTag,
        allow_padding: bool = False,
        *,
        rng: np.random.Generator | None = None,
    ) -> list[VLAStepData]:
        """Batch-fetch arbitrary ``(ep_idx, step_index)`` pairs across episodes in ONE Lance read.

        Used by ``ResumableLanceMixtureDataset`` to coalesce S3 tabular & video range-GETs
        across shuffled anchors from multiple episodes.
        """
        if not ep_step_pairs:
            return []

        per_sample_resolved: list[tuple[int, int, int, dict[str, np.ndarray]]] = []
        tabular_abs_set: set[int] = set()
        video_abs_set: set[int] = set()

        for ep_idx, step_idx in ep_step_pairs:
            ep_meta = self.episodes_metadata[ep_idx]
            ep_id = int(ep_meta["episode_index"])
            nominal_length = int(ep_meta["length"])
            ep_start, ep_end = self._episode_bounds(ep_id)
            actual_length = min(ep_end - ep_start, nominal_length)

            resolved: dict[str, np.ndarray] = {}
            step_arr = np.array([step_idx], dtype=np.int64)
            for mod, cfg in self.modality_configs.items():
                rel_rows = _resolve_step_indices(step_arr, cfg.delta_indices, actual_length, allow_padding)[0]
                resolved[mod] = rel_rows
                if mod == "video":
                    video_abs_set.update(ep_start + int(r) for r in rel_rows)
                elif mod in ("state", "action"):
                    tabular_abs_set.update(ep_start + int(r) for r in rel_rows)
                elif mod == "language":
                    for lk in cfg.modality_keys:
                        if lk not in LANG_KEYS:
                            tabular_abs_set.update(ep_start + int(r) for r in rel_rows)

            per_sample_resolved.append((ep_idx, ep_start, actual_length, resolved))

        columns, row_pos, decoded_videos = self._fetch_tabular_and_decode_videos(
            sorted(tabular_abs_set), sorted(video_abs_set)
        )

        # Cache per-episode meta language tables when needed
        lang_key: str | None = None
        meta_lang_cache: dict[int, list[str]] = {}
        if "language" in self.modality_configs:
            lang_key = self.modality_configs["language"].modality_keys[0]
            if lang_key in LANG_KEYS:
                for ep_idx, _, actual_length, _ in per_sample_resolved:
                    if ep_idx not in meta_lang_cache:
                        meta_lang_cache[ep_idx] = self.create_language_from_meta(
                            self.episodes_metadata[ep_idx], actual_length, lang_key, rng=rng
                        )

        slice_specs: dict[str, list[tuple[str, str, int, int]]] = {}
        for mod in ("state", "action"):
            if mod not in self.modality_configs:
                continue
            mod_meta = self.modality_meta.get(mod, {})
            slice_specs[mod] = [
                (
                    g,
                    mod_meta[g].get("original_key", DEFAULT_COLUMN_NAMES[mod]),
                    int(mod_meta[g]["start"]),
                    int(mod_meta[g]["end"]),
                )
                for g in self.modality_configs[mod].modality_keys
            ]

        outputs: list[VLAStepData] = []
        for ep_idx, ep_start, _, resolved in per_sample_resolved:
            images: dict[str, list[np.ndarray]] = {}
            if "video" in self.modality_configs and self._video_orig_keys:
                vid_rows = resolved["video"]
                for img_key in self.modality_configs["video"].modality_keys:
                    images[img_key] = [decoded_videos[ep_start + int(r)][img_key] for r in vid_rows]

            states: dict[str, np.ndarray] = {}
            if "state" in slice_specs:
                pos_list = [row_pos[ep_start + int(r)] for r in resolved["state"]]
                for group_name, orig_key, start_idx, end_idx in slice_specs["state"]:
                    raw_col = columns[orig_key]
                    if isinstance(raw_col, np.ndarray) and raw_col.ndim >= 2:
                        states[group_name] = np.asarray(
                            raw_col[pos_list, start_idx:end_idx], dtype=np.float32
                        )
                    else:
                        states[group_name] = np.vstack(
                            [np.asarray(raw_col[p], dtype=np.float32) for p in pos_list]
                        )

            actions: dict[str, np.ndarray] = {}
            if "action" in slice_specs:
                pos_list = [row_pos[ep_start + int(r)] for r in resolved["action"]]
                for group_name, orig_key, start_idx, end_idx in slice_specs["action"]:
                    raw_col = columns[orig_key]
                    if isinstance(raw_col, np.ndarray) and raw_col.ndim >= 2:
                        actions[group_name] = np.asarray(
                            raw_col[pos_list, start_idx:end_idx], dtype=np.float32
                        )
                    else:
                        actions[group_name] = np.vstack(
                            [np.asarray(raw_col[p], dtype=np.float32) for p in pos_list]
                        )

            text: str | None = None
            if lang_key is not None:
                lang_row = int(resolved["language"][0])
                if lang_key in LANG_KEYS:
                    text = meta_lang_cache[ep_idx][lang_row]
                else:
                    subkey = lang_key.replace("annotation.", "", 1)
                    orig_key = self.modality_meta["annotation"][subkey].get("original_key", lang_key)
                    text = self._resolve_annotation_value(columns[orig_key][row_pos[ep_start + lang_row]])

            outputs.append(
                VLAStepData(
                    images=images,
                    states=states,
                    actions=actions,
                    masks=None,
                    text=text,
                    embodiment=embodiment_tag,
                )
            )
        return outputs

    def close(self) -> None:
        with self._io_lock:
            reader = self._reader0
            if reader is not None:
                for pool_name in ("_prefetch_pool", "_decode_pool"):
                    pool = getattr(reader, pool_name, None)
                    if pool is not None:
                        try:
                            pool.shutdown(wait=True, cancel_futures=True)
                        except Exception:
                            # During Python interpreter teardown, concurrent.futures internals
                            # may already be cleared.
                            pass
                        setattr(reader, pool_name, None)
                for attr in ("_frames_perm", "_videos_table", "_video_row_ids"):
                    if hasattr(reader, attr):
                        setattr(reader, attr, None)
                self._reader0 = None
                self._opened_pid = None


__all__ = ["LanceEpisodeLoader"]
