# SPDX-License-Identifier: Apache-2.0
"""Tabular-only curation, anomaly/idle tagging, and stats recomputation on LanceDB.

Appends per-frame quality columns (``action_delta_norm``, ``state_delta_norm``,
``is_idle``, ``has_anomaly``, ``is_bad_episode``, ``is_valid``) directly onto
``frames.lance`` via LanceDB's native zero-copy column evolution (``add_columns``),
builds BITMAP scalar indexes for millisecond-level SQL predicate pushdown, and
recomputes exact GR00T ``gr00t_stats.json`` (including ``q01``/``q99``) strictly over
valid rows — with **zero reads or writes to ``videos.lance``**.

Unmodified ``lerobot_lancedb.LanceDBDataset`` continues to open curated datasets
natively because row count and video blobs remain untouched.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np
import pyarrow as pa

from lerobot_lancedb import FRAMES_TABLE, META_TABLE, VIDEOS_TABLE, to_lance_column
from lerobot_lancedb._source import is_remote_uri
from lerobot_lancedb.convert import _connect_output

from .metadata import (
    CURATION_REPORT_FILENAME,
    DEFAULT_COLUMN_NAMES,
    GROOT_STATS_FILENAME,
    LEROBOT_MODALITY_FILENAME,
    LEROBOT_STATS_FILENAME,
    compute_gr00t_stats_from_arrays,
)

logger = logging.getLogger(__name__)

CURATION_COLUMNS: tuple[str, ...] = (
    "action_delta_norm",
    "state_delta_norm",
    "is_idle",
    "has_anomaly",
    "is_bad_episode",
    "is_valid",
)


@dataclass
class CurationReport:
    """Summary metrics produced by :func:`curate_lance_dataset`."""

    lance_root: str
    frames_table_version_before: int
    frames_table_version: int
    total_episodes: int
    bad_episodes: int
    valid_episodes: int
    bad_episode_indices: list[int]
    total_frames: int
    idle_frames: int
    anomaly_frames: int
    bad_episode_frames: int
    valid_frames: int
    idle_pct: float
    anomaly_pct: float
    valid_pct: float
    idle_eps: float
    spike_threshold: float | None
    drop_random_frac: float
    drop_seed: int
    keep_idle: bool
    recompute_stats: bool
    wall_time_s: float
    bytes_written_frames: int
    bytes_written_meta: int
    bytes_written_videos: int
    bytes_written_total: int
    files_written_frames: int
    files_written_videos: int
    scalar_indexes_built: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _snapshot_dir_files(path: Path) -> dict[str, tuple[int, int]]:
    """Return ``{rel_posix_path: (size_bytes, mtime_ns)}`` for all files under ``path``."""
    if not path.exists():
        return {}
    snap: dict[str, tuple[int, int]] = {}
    for f in path.rglob("*"):
        if f.is_file():
            st = f.stat()
            snap[f.relative_to(path).as_posix()] = (st.st_size, st.st_mtime_ns)
    return snap


def _diff_snapshots(
    before: dict[str, tuple[int, int]],
    after: dict[str, tuple[int, int]],
) -> tuple[int, int]:
    """Return ``(bytes_written, files_written)`` for new or modified files."""
    bytes_written = 0
    files_written = 0
    for rel, (sz, mtime_ns) in after.items():
        prev = before.get(rel)
        if prev is None or prev != (sz, mtime_ns):
            bytes_written += sz
            files_written += 1
    return bytes_written, files_written


def _arrow_col_to_2d_float64(col: pa.ChunkedArray) -> np.ndarray:
    """Convert an Arrow FixedSizeList/List/numeric column into a contiguous 2D float64 array."""
    arr = col.combine_chunks()
    n = len(arr)
    if n == 0:
        return np.empty((0, 0), dtype=np.float64)
    if pa.types.is_fixed_size_list(arr.type):
        width = arr.type.list_size
        vals = arr.values.to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
        return vals.reshape(n, width)
    if pa.types.is_list(arr.type) or pa.types.is_large_list(arr.type):
        flat = arr.flatten().to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
        if n > 0 and len(flat) % n == 0:
            return flat.reshape(n, len(flat) // n)
        return np.vstack([np.asarray(x, dtype=np.float64).reshape(-1) for x in arr.to_pylist()])
    vals = arr.to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
    return vals.reshape(n, 1)


def _extract_modality_keys(modality_meta: dict[str, Any]) -> tuple[list[str], list[str]]:
    state_keys: list[str] = []
    for group_meta in modality_meta.get("state", {}).values():
        k = group_meta.get("original_key") or DEFAULT_COLUMN_NAMES["state"]
        if k not in state_keys:
            state_keys.append(k)
    action_keys: list[str] = []
    for group_meta in modality_meta.get("action", {}).values():
        k = group_meta.get("original_key") or DEFAULT_COLUMN_NAMES["action"]
        if k not in action_keys:
            action_keys.append(k)
    return state_keys, action_keys


def _build_bitmap_indexes(frames_table: Any, columns: Sequence[str]) -> list[str]:
    """Build BITMAP scalar indexes on boolean curation columns (best-effort across LanceDB versions)."""
    built: list[str] = []
    bitmap_cfg = None
    try:
        from lancedb.index import Bitmap

        bitmap_cfg = Bitmap()
    except Exception:
        bitmap_cfg = None

    for col in columns:
        try:
            if bitmap_cfg is not None:
                frames_table.create_index(col, config=bitmap_cfg, replace=True)
            else:
                frames_table.create_scalar_index(col, index_type="BITMAP", replace=True)
            built.append(col)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not build BITMAP scalar index on %s: %s", col, exc)
    return built


def curate_lance_dataset(
    lance_root: str | Path,
    *,
    idle_eps: float = 1e-4,
    spike_threshold: float | None = None,
    drop_episodes: Sequence[int] | None = None,
    drop_random_frac: float = 0.0,
    drop_seed: int = 42,
    keep_idle: bool = False,
    recompute_stats: bool = True,
    storage_options: dict[str, str] | None = None,
) -> CurationReport:
    """Tag per-frame quality metrics on ``frames.lance`` without touching ``videos.lance``.

    Reads ONLY tabular columns (``index``, ``episode_index``, ``frame_index``, and state/action
    columns from ``modality.json``), appends/updates the 6 curation columns via LanceDB zero-copy
    column evolution, builds BITMAP scalar indexes on ``is_valid`` and ``is_bad_episode``, and
    recomputes GR00T ``gr00t_stats.json`` strictly over valid rows.
    """
    if not (0.0 <= drop_random_frac <= 1.0):
        raise ValueError(f"drop_random_frac must be in [0.0, 1.0], got {drop_random_frac}")
    if idle_eps < 0.0:
        raise ValueError(f"idle_eps must be >= 0, got {idle_eps}")

    t0 = time.perf_counter()
    remote = is_remote_uri(lance_root)
    local_root = None if remote else Path(lance_root).expanduser().resolve()

    frames_snap_before = _snapshot_dir_files(local_root / f"{FRAMES_TABLE}.lance") if local_root else {}
    meta_snap_before = _snapshot_dir_files(local_root / f"{META_TABLE}.lance") if local_root else {}
    local_meta_snap_before = _snapshot_dir_files(local_root / "meta") if local_root else {}
    videos_snap_before = _snapshot_dir_files(local_root / f"{VIDEOS_TABLE}.lance") if local_root else {}

    db = _connect_output(lance_root, storage_options)
    open_kw = {"storage_options": storage_options} if storage_options else {}
    frames_table = db.open_table(FRAMES_TABLE, **open_kw)
    meta_table = db.open_table(META_TABLE, **open_kw)
    version_before = int(frames_table.version)

    # 1. Read existing metadata entries from meta.lance (zero video reads)
    meta_arrow = meta_table.search().select(["path", "data"]).limit(None).to_arrow()
    entries: dict[str, bytes] = {
        p: d
        for p, d in zip(
            meta_arrow.column("path").to_pylist(),
            meta_arrow.column("data").to_pylist(),
            strict=True,
        )
    }

    if LEROBOT_MODALITY_FILENAME in entries:
        modality_meta = json.loads(entries[LEROBOT_MODALITY_FILENAME].decode("utf-8"))
    elif local_root is not None and (local_root / "meta" / LEROBOT_MODALITY_FILENAME).exists():
        mod_bytes = (local_root / "meta" / LEROBOT_MODALITY_FILENAME).read_bytes()
        entries[LEROBOT_MODALITY_FILENAME] = mod_bytes
        modality_meta = json.loads(mod_bytes.decode("utf-8"))
    else:
        raise FileNotFoundError(
            f"No {LEROBOT_MODALITY_FILENAME} found in {META_TABLE}.lance at {lance_root}."
        )

    state_orig_keys, action_orig_keys = _extract_modality_keys(modality_meta)
    available_cols = set(frames_table.schema.names)

    orig_to_lance: dict[str, str] = {}
    select_cols: list[str] = ["index", "episode_index", "frame_index"]
    for orig_key in [*state_orig_keys, *action_orig_keys]:
        lance_col = to_lance_column(orig_key)
        if lance_col in available_cols:
            orig_to_lance[orig_key] = lance_col
            if lance_col not in select_cols:
                select_cols.append(lance_col)

    # 2. Read ONLY tabular columns from frames.lance (preserves physical row order for add_columns)
    frames_arrow = frames_table.search().select(select_cols).limit(None).to_arrow()
    n_rows = frames_arrow.num_rows

    ep_arr = frames_arrow.column("episode_index").combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    frame_arr = frames_arrow.column("frame_index").combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64, copy=False)

    per_key_arrays: dict[str, np.ndarray] = {}
    for orig_key, lance_col in orig_to_lance.items():
        per_key_arrays[orig_key] = _arrow_col_to_2d_float64(frames_arrow.column(lance_col))

    state_blocks = [per_key_arrays[k] for k in state_orig_keys if k in per_key_arrays and per_key_arrays[k].shape[1] > 0]
    action_blocks = [per_key_arrays[k] for k in action_orig_keys if k in per_key_arrays and per_key_arrays[k].shape[1] > 0]
    state_mat = np.concatenate(state_blocks, axis=1) if state_blocks else np.empty((n_rows, 0), dtype=np.float64)
    action_mat = np.concatenate(action_blocks, axis=1) if action_blocks else np.empty((n_rows, 0), dtype=np.float64)

    # 3. Compute per-frame quality metrics within each episode (sorted by frame_index)
    action_delta_norm = np.zeros(n_rows, dtype=np.float32)
    state_delta_norm = np.zeros(n_rows, dtype=np.float32)
    frame_nonfinite = np.zeros(n_rows, dtype=bool)
    nan_episodes: set[int] = set()

    if n_rows > 0:
        order = np.lexsort((frame_arr, ep_arr))
        ep_sorted = ep_arr[order]
        unique_eps, split_starts = np.unique(ep_sorted, return_index=True)
        split_ends = np.append(split_starts[1:], n_rows)

        has_action = action_mat.shape[1] > 0
        has_state = state_mat.shape[1] > 0

        for ep_id_np, s, e in zip(unique_eps, split_starts, split_ends, strict=True):
            ep_id = int(ep_id_np)
            pos = order[s:e]
            length = len(pos)

            ep_nonfinite = np.zeros(length, dtype=bool)
            if has_action:
                ep_act = action_mat[pos]
                ep_nonfinite |= ~np.all(np.isfinite(ep_act), axis=1)
                if length >= 2:
                    diffs = ep_act[1:] - ep_act[:-1]
                    norms = np.linalg.norm(diffs, axis=1).astype(np.float32)
                    ep_act_delta = np.empty(length, dtype=np.float32)
                    ep_act_delta[0] = norms[0]
                    ep_act_delta[1:] = norms
                    action_delta_norm[pos] = ep_act_delta
            if has_state:
                ep_st = state_mat[pos]
                ep_nonfinite |= ~np.all(np.isfinite(ep_st), axis=1)
                if length >= 2:
                    diffs = ep_st[1:] - ep_st[:-1]
                    norms = np.linalg.norm(diffs, axis=1).astype(np.float32)
                    ep_st_delta = np.empty(length, dtype=np.float32)
                    ep_st_delta[0] = norms[0]
                    ep_st_delta[1:] = norms
                    state_delta_norm[pos] = ep_st_delta

            if np.any(ep_nonfinite):
                frame_nonfinite[pos] = ep_nonfinite
                nan_episodes.add(ep_id)
    else:
        unique_eps = np.empty(0, dtype=np.int64)

    # 4. Determine bad episodes & boolean masks
    dropped_ep_set: set[int] = {int(e) for e in (drop_episodes or [])}
    if drop_random_frac > 0.0 and len(unique_eps) > 0:
        n_drop = min(len(unique_eps), max(1, int(round(len(unique_eps) * float(drop_random_frac)))))
        rng = np.random.default_rng(drop_seed)
        chosen = rng.choice(unique_eps, size=n_drop, replace=False)
        dropped_ep_set.update(int(e) for e in chosen)

    # Keep only episode indices that actually exist in the dataset for accurate reporting
    existing_ep_set = {int(e) for e in unique_eps}
    bad_ep_set = (dropped_ep_set & existing_ep_set) | nan_episodes

    is_idle = action_delta_norm.astype(np.float64) < float(idle_eps)
    has_anomaly = frame_nonfinite.copy()
    if spike_threshold is not None:
        has_anomaly |= action_delta_norm.astype(np.float64) > float(spike_threshold)

    if bad_ep_set:
        bad_ep_arr = np.array(sorted(bad_ep_set), dtype=np.int64)
        is_bad_episode = np.isin(ep_arr, bad_ep_arr)
    else:
        is_bad_episode = np.zeros(n_rows, dtype=bool)

    if keep_idle:
        is_valid = (~is_bad_episode) & (~has_anomaly)
    else:
        is_valid = (~is_bad_episode) & (~has_anomaly) & (~is_idle)

    # 5. Zero-copy column evolution on frames.lance (existing columns & videos.lance untouched!)
    curation_arrow = pa.table(
        {
            "action_delta_norm": pa.array(action_delta_norm, type=pa.float32()),
            "state_delta_norm": pa.array(state_delta_norm, type=pa.float32()),
            "is_idle": pa.array(is_idle, type=pa.bool_()),
            "has_anomaly": pa.array(has_anomaly, type=pa.bool_()),
            "is_bad_episode": pa.array(is_bad_episode, type=pa.bool_()),
            "is_valid": pa.array(is_valid, type=pa.bool_()),
        }
    )

    existing_curation = [c for c in CURATION_COLUMNS if c in frames_table.schema.names]
    if existing_curation:
        frames_table.drop_columns(existing_curation)
    frames_table.add_columns(curation_arrow)

    scalar_indexes_built = _build_bitmap_indexes(frames_table, ("is_valid", "is_bad_episode"))
    version_after = int(frames_table.version)

    # Measure frames.lance writes before writing meta
    if local_root is not None:
        frames_snap_after = _snapshot_dir_files(local_root / f"{FRAMES_TABLE}.lance")
        bytes_written_frames, files_written_frames = _diff_snapshots(frames_snap_before, frames_snap_after)
    else:
        bytes_written_frames = int(curation_arrow.nbytes)
        files_written_frames = 1

    # 6. Recompute GR00T stats strictly over is_valid rows (fallback to non-bad finite rows if 0 valid)
    if recompute_stats and per_key_arrays:
        stats_mask = is_valid if np.any(is_valid) else ((~is_bad_episode) & (~frame_nonfinite))
        if np.any(stats_mask):
            masked_arrays = {k: arr[stats_mask] for k, arr in per_key_arrays.items()}
            recomputed = compute_gr00t_stats_from_arrays(masked_arrays)
            entries[GROOT_STATS_FILENAME] = (json.dumps(recomputed, indent=2) + "\n").encode("utf-8")

            # Also update stats.json (v3 compatible) with updated stats + valid frame count
            existing_v3_stats: dict[str, Any] = {}
            if LEROBOT_STATS_FILENAME in entries:
                try:
                    existing_v3_stats = json.loads(entries[LEROBOT_STATS_FILENAME].decode("utf-8"))
                except Exception:
                    existing_v3_stats = {}
            valid_count = int(np.sum(stats_mask))
            for feat_key, st in recomputed.items():
                entry = dict(st)
                entry["count"] = [valid_count]
                existing_v3_stats[feat_key] = entry
            entries[LEROBOT_STATS_FILENAME] = (
                json.dumps(existing_v3_stats, indent=2) + "\n"
            ).encode("utf-8")

    # 7. Assemble report and persist to meta.lance (+ local meta/ if local)
    total_episodes = int(len(unique_eps))
    bad_episodes = int(len(bad_ep_set))
    valid_episodes = total_episodes - bad_episodes
    idle_frames = int(np.sum(is_idle))
    anomaly_frames = int(np.sum(has_anomaly))
    bad_episode_frames = int(np.sum(is_bad_episode))
    valid_frames = int(np.sum(is_valid))
    denom = float(n_rows) if n_rows > 0 else 1.0

    est_meta_bytes = sum(len(v) for v in entries.values())
    report = CurationReport(
        lance_root=str(lance_root),
        frames_table_version_before=version_before,
        frames_table_version=version_after,
        total_episodes=total_episodes,
        bad_episodes=bad_episodes,
        valid_episodes=valid_episodes,
        bad_episode_indices=sorted(bad_ep_set),
        total_frames=int(n_rows),
        idle_frames=idle_frames,
        anomaly_frames=anomaly_frames,
        bad_episode_frames=bad_episode_frames,
        valid_frames=valid_frames,
        idle_pct=round(100.0 * idle_frames / denom, 4) if n_rows > 0 else 0.0,
        anomaly_pct=round(100.0 * anomaly_frames / denom, 4) if n_rows > 0 else 0.0,
        valid_pct=round(100.0 * valid_frames / denom, 4) if n_rows > 0 else 0.0,
        idle_eps=float(idle_eps),
        spike_threshold=float(spike_threshold) if spike_threshold is not None else None,
        drop_random_frac=float(drop_random_frac),
        drop_seed=int(drop_seed),
        keep_idle=bool(keep_idle),
        recompute_stats=bool(recompute_stats),
        wall_time_s=round(time.perf_counter() - t0, 4),
        bytes_written_frames=bytes_written_frames,
        bytes_written_meta=est_meta_bytes,
        bytes_written_videos=0,
        bytes_written_total=bytes_written_frames + est_meta_bytes,
        files_written_frames=files_written_frames,
        files_written_videos=0,
        scalar_indexes_built=scalar_indexes_built,
    )

    entries[CURATION_REPORT_FILENAME] = (json.dumps(report.to_dict(), indent=2) + "\n").encode("utf-8")

    meta_schema = pa.schema([pa.field("path", pa.string()), pa.field("data", pa.large_binary())])
    db.create_table(
        META_TABLE,
        pa.Table.from_pylist(
            [{"path": p, "data": entries[p]} for p in sorted(entries)],
            schema=meta_schema,
        ),
        mode="overwrite",
    )

    if local_root is not None:
        local_meta = local_root / "meta"
        local_meta.mkdir(parents=True, exist_ok=True)
        for fname in (GROOT_STATS_FILENAME, LEROBOT_STATS_FILENAME, CURATION_REPORT_FILENAME):
            if fname in entries:
                (local_meta / fname).write_bytes(entries[fname])

        meta_snap_after = _snapshot_dir_files(local_root / f"{META_TABLE}.lance")
        local_meta_snap_after = _snapshot_dir_files(local_root / "meta")
        videos_snap_after = _snapshot_dir_files(local_root / f"{VIDEOS_TABLE}.lance")

        meta_lance_bytes, _ = _diff_snapshots(meta_snap_before, meta_snap_after)
        local_meta_bytes, _ = _diff_snapshots(local_meta_snap_before, local_meta_snap_after)
        bytes_written_videos, files_written_videos = _diff_snapshots(videos_snap_before, videos_snap_after)

        report.bytes_written_meta = meta_lance_bytes + local_meta_bytes
        report.bytes_written_videos = bytes_written_videos
        report.files_written_videos = files_written_videos
        report.bytes_written_total = report.bytes_written_frames + report.bytes_written_meta + report.bytes_written_videos
        report.wall_time_s = round(time.perf_counter() - t0, 4)

        # Keep local curation_report.json synced with final measured byte/time totals
        (local_meta / CURATION_REPORT_FILENAME).write_text(
            json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
    else:
        report.bytes_written_meta = sum(len(v) for v in entries.values())
        report.bytes_written_total = report.bytes_written_frames + report.bytes_written_meta
        report.wall_time_s = round(time.perf_counter() - t0, 4)

    return report


def _parse_drop_episodes(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    out: list[int] = []
    for part in raw.split(","):
        s = part.strip()
        if s:
            out.append(int(s))
    return out


def _parse_kv_options(items: list[str] | None) -> dict[str, str] | None:
    if not items:
        return None
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE storage option, got {item!r}")
        k, v = item.split("=", 1)
        out[k] = v
    return out


def main_curate(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="gr00t-lance-curate",
        description=(
            "Curate a GR00T Lance dataset in place: tag idle/anomaly/bad-episode frames on "
            "frames.lance via zero-copy column evolution (zero video reads/writes) and recompute "
            "GR00T q01/q99 statistics."
        ),
    )
    parser.add_argument("--root", required=True, help="Lance dataset directory or s3:// URI.")
    parser.add_argument(
        "--idle-eps",
        type=float,
        default=1e-4,
        help="L2 action-delta threshold below which a frame is tagged is_idle=True (default: 1e-4).",
    )
    parser.add_argument(
        "--spike-threshold",
        type=float,
        default=None,
        help="Optional L2 action-delta upper bound above which a frame is tagged has_anomaly=True.",
    )
    parser.add_argument(
        "--drop-episodes",
        default=None,
        help="Comma-separated episode indices to mark is_bad_episode=True (e.g. '0,5,12').",
    )
    parser.add_argument(
        "--drop-random-frac",
        type=float,
        default=0.0,
        help="Deterministic fraction of episodes to mark as bad (default: 0.0).",
    )
    parser.add_argument(
        "--drop-seed",
        type=int,
        default=42,
        help="Random seed for --drop-random-frac (default: 42).",
    )
    parser.add_argument(
        "--keep-idle",
        action="store_true",
        help="Keep idle frames valid (is_valid ignores is_idle).",
    )
    parser.add_argument(
        "--no-recompute-stats",
        action="store_true",
        help="Skip recomputing gr00t_stats.json over valid frames.",
    )
    parser.add_argument(
        "--storage-option",
        action="append",
        default=None,
        help="Storage KEY=VALUE option (repeatable).",
    )
    parser.add_argument(
        "--out-json",
        default=None,
        help="Optional path to write the curation report JSON.",
    )
    args = parser.parse_args(argv)

    report = curate_lance_dataset(
        lance_root=args.root,
        idle_eps=args.idle_eps,
        spike_threshold=args.spike_threshold,
        drop_episodes=_parse_drop_episodes(args.drop_episodes),
        drop_random_frac=args.drop_random_frac,
        drop_seed=args.drop_seed,
        keep_idle=args.keep_idle,
        recompute_stats=not args.no_recompute_stats,
        storage_options=_parse_kv_options(args.storage_option),
    )
    payload = json.dumps(report.to_dict(), indent=2) + "\n"
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")
    print(payload, end="")


__all__ = [
    "CURATION_COLUMNS",
    "CurationReport",
    "curate_lance_dataset",
    "main_curate",
]
