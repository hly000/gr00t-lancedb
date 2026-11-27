# SPDX-License-Identifier: Apache-2.0
"""LanceDB-backed drop-in replacement for GR00T's ``LeRobotEpisodeLoader``.

Streams tabular and video frames directly from local or object-store (``s3://``,
``gs://``, ``hf://``) Lance tables via ``lerobot_lancedb.LanceDBDataset`` while
preserving 100% of GR00T's ``modality.json`` slicing, camera key mapping,
language annotation resolution, and ``VLAStepData`` output contract.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import logging
from pathlib import Path
import random
import threading
from typing import Any

import lancedb
import numpy as np
import pandas as pd
import torch

from lerobot_lancedb import FRAMES_TABLE, LanceDBDataset

from ._gr00t_compat import EmbodimentTag, ModalityConfig, VLAStepData
# lancedb registers its blob Arrow extension type lazily on first table open; two threads
# opening at once race on that registration ("lance.blob.v2 is already registered").
_OPEN_LOCK = threading.Lock()


def _fresh_reader(template: LanceDBDataset) -> LanceDBDataset:
    """Unopened copy of ``template`` with its own table handles, pools and decoder caches."""
    clone = object.__new__(type(template))
    state = template.__getstate__()
    if hasattr(clone, "__setstate__"):
        clone.__setstate__(state)
    else:
        clone.__dict__.update(state)
    return clone

from .metadata import (  # noqa: E402
    DEFAULT_COLUMN_NAMES,
    LANG_KEYS,
    extract_dataset_statistics,
    load_gr00t_meta_bundle,
    parse_and_validate_modality_configs,
    resolve_video_original_key,
)

logger = logging.getLogger(__name__)


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
        video_decoder_cache_size: int = 16,
        decoder_kwargs: dict[str, Any] | None = None,
        where: str | None = None,
        lance_version: int | None = None,
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

        # Open a lightweight metadata-resolving LanceDBDataset first (no video/frame I/O yet)
        # so remote URIs automatically materialize meta/ from meta.lance once.
        probe_reader = LanceDBDataset(
            repo_id=repo_id,
            root=self.dataset_path,
            return_uint8=True,
            storage_options=self.storage_options,
            revision=self.revision,
            token=self.token,
            use_mini_mp4=self.use_mini_mp4,
            video_decoder_cache_size=self.video_decoder_cache_size,
        )
        self.local_meta_root: Path = Path(probe_reader.root)

        bundle = load_gr00t_meta_bundle(
            self.local_meta_root / "meta",
            modality_json_path=self.modality_json_path,
            stats_json_path=self.stats_json_path,
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
        available_features = set(probe_reader.meta.features.keys())

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
        # One reader per thread: GR00T's fast loader decodes splits on a thread pool, and
        # LanceDBDataset's decoder caches / prefetch state are not shared-thread safe.
        self._reader0 = LanceDBDataset(**self._reader_kwargs)
        self._init_thread_state()

        # Map GR00T episode positional idx -> Lance episode_index & bounds.
        self.episode_lengths = [int(ep["length"]) for ep in self.episodes_metadata]
        self._ep_id_to_pos = {
            int(ep["episode_index"]): pos for pos, ep in enumerate(self.episodes_metadata)
        }

        # Optional frame-level SQL predicate pushdown against frames.lance (zero video reads).
        self.valid_steps_by_episode: dict[int, np.ndarray] | None = None
        if self.where is not None:
            self.valid_steps_by_episode = self._resolve_valid_steps_from_where(
                where=self.where,
                lance_version=self.lance_version,
            )

    def _open_frames_table(self, lance_version: int | None = None) -> Any:
        opts = self._reader0._storage_options or self.storage_options
        db_uri = getattr(self._reader0, "_db_uri", self.dataset_path)
        db = lancedb.connect(db_uri, storage_options=opts) if opts else lancedb.connect(db_uri)
        table = db.open_table(FRAMES_TABLE, **({"storage_options": opts} if opts else {}))
        if lance_version is not None:
            table.checkout(lance_version)
        return table

    def _resolve_valid_steps_from_where(
        self,
        *,
        where: str,
        lance_version: int | None = None,
    ) -> dict[int, np.ndarray]:
        """Query ``frames.lance`` projecting ONLY ``["episode_index", "frame_index"]`` with ``where``.

        Builds a per-episode sorted array of matching anchor ``frame_index`` values without
        touching ``videos.lance`` or breaking full-episode horizon continuity.
        """
        with _OPEN_LOCK:
            table = self._open_frames_table(lance_version=lance_version)
            arrow_tbl = (
                table.search()
                .where(where)
                .select(["episode_index", "frame_index"])
                .limit(None)
                .to_arrow()
            )

        valid_by_pos: dict[int, np.ndarray] = {
            pos: np.empty(0, dtype=np.int64) for pos in range(len(self.episodes_metadata))
        }
        if arrow_tbl.num_rows == 0:
            return valid_by_pos

        ep_arr = arrow_tbl.column("episode_index").combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        frame_arr = arrow_tbl.column("frame_index").combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64, copy=False)

        order = np.lexsort((frame_arr, ep_arr))
        ep_sorted = ep_arr[order]
        frame_sorted = frame_arr[order]
        unique_eps, split_starts = np.unique(ep_sorted, return_index=True)
        split_ends = np.append(split_starts[1:], len(ep_sorted))

        for ep_id_np, s, e in zip(unique_eps, split_starts, split_ends, strict=True):
            pos = self._ep_id_to_pos.get(int(ep_id_np))
            if pos is None:
                continue
            ep_frames = np.unique(frame_sorted[s:e])
            ep_len = self.episode_lengths[pos]
            if len(ep_frames):
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

    # -- per-thread readers ---------------------------------------------------------
    def _init_thread_state(self) -> None:
        self._tls = threading.local()
        self._readers_lock = threading.Lock()
        self._thread_readers: list[LanceDBDataset] = []

    @property
    def reader(self) -> LanceDBDataset:
        r = getattr(self._tls, "reader", None)
        if r is None:
            with self._readers_lock:
                first = not self._thread_readers
            # The first thread reuses the reader built in __init__; others get a clone made
            # from its pickle state (fresh handles/caches, shared read-only metadata) —
            # re-running LanceDBDataset.__init__ per thread would reload metadata.
            r = self._reader0 if first else _fresh_reader(self._reader0)
            with _OPEN_LOCK:
                r._ensure_open()
                if self.lance_version is not None:
                    try:
                        from lancedb.permutation import Permutation

                        pinned_table = self._open_frames_table(lance_version=self.lance_version)
                        r._frames_perm = (
                            Permutation.identity(pinned_table)
                            .select_columns(r._fetch_columns)
                            .with_format("arrow")
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "Could not pin reader frames_perm to lance_version=%s: %s",
                            self.lance_version,
                            exc,
                        )
            with self._readers_lock:
                self._thread_readers.append(r)
            self._tls.reader = r
        return r

    @reader.setter
    def reader(self, value: LanceDBDataset) -> None:
        """Replace the base reader (e.g. a per-worker copy); thread readers start over."""
        self._reader0 = value
        self._init_thread_state()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        for k in ("_tls", "_readers_lock", "_thread_readers"):
            state.pop(k, None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._init_thread_state()

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
        self.reader._ensure_open()

        all_rows = sorted(set(tabular_abs_rows) | set(video_abs_rows))
        row_pos = {row: pos for pos, row in enumerate(all_rows)}

        decoded_videos: dict[int, dict[str, np.ndarray]] = {}
        if video_abs_rows and self._video_orig_keys:
            video_plans = self.reader._plan_batch(video_abs_rows)
            windows = self.reader._plan_file_windows(video_plans)
            prefetch_fut = self.reader._prefetch_pool.submit(
                self.reader._prepare_files, sorted(windows), windows
            )
            columns = self.reader._fetch_rows(all_rows)
            prepared = prefetch_fut.result()
            decoded_list = self.reader._decode_videos(video_plans, columns, row_pos, prepared)
            for abs_row, raw_frames in zip(video_abs_rows, decoded_list, strict=True):
                decoded_videos[abs_row] = {
                    img_key: _tensor_chw_to_hwc_uint8(raw_frames[orig_key])
                    for img_key, orig_key in self._video_orig_keys.items()
                }
        else:
            columns = self.reader._fetch_rows(all_rows) if all_rows else {}

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
        ep_start, ep_end = self.reader._episode_bounds(episode_id)
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
        ep_start, ep_end = self.reader._episode_bounds(episode_id)
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
            ep_start, ep_end = self.reader._episode_bounds(ep_id)
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
        with self._readers_lock:
            readers = list(self._thread_readers) or [self._reader0]
        for reader in readers:
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


__all__ = ["LanceEpisodeLoader"]
