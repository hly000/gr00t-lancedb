# SPDX-License-Identifier: Apache-2.0
"""Dual-path converter for NVIDIA Isaac-GR00T datasets on LanceDB.

Supports two workflows producing a 100% ``lerobot-lancedb``-compatible three-table layout
(``frames.lance``, ``videos.lance``, ``meta.lance``):

1. ``convert_v2_to_lance`` (``gr00t-lance-convert``):
   Converts a GR00T LeRobot v2.1 dataset (local dir or ``s3://`` prefix) into the
   standard three-table Lance layout, preserving ``modality.json``, ``stats.json`` /
   ``gr00t_stats.json``, ``relative_stats.json``, ``episodes.jsonl`` (including
   ``sub_tasks``), ``tasks.jsonl``, and ``initial_actions.npz`` inside ``meta.lance``
   alongside standard LeRobot v3.0 metadata so unmodified ``lerobot-lancedb`` opens it
   natively. In ``external`` mode, MP4 files are indexed in place via range reads with
   zero payload copying.

2. ``attach_gr00t_meta`` (``gr00t-lance-attach``):
   Attaches ``modality.json`` (and optionally computes GR00T ``q01``/``q99`` stats
   directly from ``frames.lance`` tabular columns) onto an existing LeRobot v3.0 Lance
   dataset (local or ``s3://``), enabling GR00T training with zero v3->v2 downgrade.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Any, Literal

import lancedb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot_lancedb import (
    FRAMES_TABLE,
    META_TABLE,
    VIDEO_BLOB_COLUMN,
    VIDEOS_TABLE,
    build_video_byte_index,
    to_lance_column,
)
from lerobot_lancedb._source import (
    DEFAULT_IO_WORKERS,
    is_remote_uri,
    normalize_base_uri,
    open_source,
)
from lerobot_lancedb.convert import (
    EXTERNAL_VIDEO_BATCH_BYTES,
    VIDEO_BATCH_BYTES,
    _connect_output,
    _create_scalar_indexes,
    _ensure_remote_output_empty,
    _external_table_options,
    _external_video_uri,
    _list_int64_array,
    _local_build_dir,
    _commit_local_output,
    _storage_type,
    _videos_schema,
    _write_meta_table,
)
from lerobot_lancedb.mp4_index import (
    DEFAULT_HEADER_PROBE_BYTES,
    ProbedMp4,
    probe_mp4_byte_index,
    sample_index_from_mp4,
)

from .metadata import (
    DEFAULT_COLUMN_NAMES,
    GROOT_STATS_FILENAME,
    INITIAL_ACTIONS_FILENAME,
    LEROBOT_EPISODES_FILENAME,
    LEROBOT_INFO_FILENAME,
    LEROBOT_MODALITY_FILENAME,
    LEROBOT_RELATIVE_STATS_FILENAME,
    LEROBOT_STATS_FILENAME,
    LEROBOT_TASKS_FILENAME,
    compute_gr00t_stats_from_arrays,
    compute_gr00t_stats_from_lance_table,
)

VideoMode = Literal["auto", "external", "copy"]

V3_DATA_PATH_TEMPLATE = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
V3_VIDEO_PATH_TEMPLATE = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _v2_frames_reader(
    episode_parquets: list[tuple[int, int, Path]],
    fps: float,
) -> pa.RecordBatchReader:
    """Stream GR00T v2.1 per-episode parquets into a monotonically index-sorted Arrow stream.

    Ensures ``index`` is globally contiguous ``0..total_frames-1``, ``episode_index`` and
    ``frame_index`` and ``timestamp`` exist, dotted feature names become underscores, and
    list vectors become fixed-size lists (matching ``lerobot-lancedb``'s exact schema).
    """
    first_path = episode_parquets[0][2]
    schema0 = pq.read_schema(first_path)
    existing_names = set(schema0.names)

    out_fields: list[pa.Field] = []
    plan: list[tuple[str, str, Any]] = []

    for field in schema0:
        if field.name == "index":
            plan.append(("index", "global_index", None))
            out_fields.append(pa.field("index", pa.int64()))
        elif field.name == "episode_index":
            plan.append(("episode_index", "ep_index", None))
            out_fields.append(pa.field("episode_index", pa.int64()))
        elif field.name == "frame_index":
            plan.append(("frame_index", "frame_index", None))
            out_fields.append(pa.field("frame_index", pa.int64()))
        elif pa.types.is_fixed_size_list(field.type):
            plan.append((field.name, "asis", None))
            out_fields.append(pa.field(to_lance_column(field.name), field.type))
        elif pa.types.is_list(field.type) or pa.types.is_large_list(field.type):
            sample_col = pq.read_table(first_path, columns=[field.name]).column(field.name)
            first_val = sample_col[0].as_py()
            if isinstance(first_val, list) and (
                not first_val or isinstance(first_val[0], (int, float, np.number))
            ):
                width = len(first_val)
                val_type = field.type.value_type
                out_type = pa.list_(val_type, width)
                plan.append((field.name, "fsl", width))
                out_fields.append(pa.field(to_lance_column(field.name), out_type))
            else:
                target = _storage_type(field.type)
                plan.append((field.name, "cast", target))
                out_fields.append(pa.field(to_lance_column(field.name), target))
        else:
            target = _storage_type(field.type)
            plan.append((field.name, "cast", target))
            out_fields.append(pa.field(to_lance_column(field.name), target))

    # Synthesize any standard LeRobot v3 control columns absent in minimal v2.1 files.
    if "index" not in existing_names:
        plan.append(("index", "global_index", None))
        out_fields.append(pa.field("index", pa.int64()))
    if "episode_index" not in existing_names:
        plan.append(("episode_index", "ep_index", None))
        out_fields.append(pa.field("episode_index", pa.int64()))
    if "frame_index" not in existing_names:
        plan.append(("frame_index", "frame_index", None))
        out_fields.append(pa.field("frame_index", pa.int64()))
    if "timestamp" not in existing_names:
        plan.append(("timestamp", "synth_timestamp", None))
        out_fields.append(pa.field("timestamp", pa.float32()))
    if "task_index" not in existing_names:
        plan.append(("task_index", "synth_task_index", None))
        out_fields.append(pa.field("task_index", pa.int64()))

    out_schema = pa.schema(out_fields)

    def batches():
        global_cursor = 0
        for ep_idx, expected_len, pq_path in episode_parquets:
            table = pq.read_table(pq_path)
            n = min(table.num_rows, expected_len)
            if n < table.num_rows:
                table = table.slice(0, n)
            if n == 0:
                continue

            idx_arr = pa.array(np.arange(global_cursor, global_cursor + n, dtype=np.int64), type=pa.int64())
            ep_arr = pa.array(np.full(n, ep_idx, dtype=np.int64), type=pa.int64())
            frame_arr = pa.array(np.arange(n, dtype=np.int64), type=pa.int64())
            ts_arr = pa.array((np.arange(n, dtype=np.float64) / fps).astype(np.float32), type=pa.float32())
            task_arr = pa.array(np.zeros(n, dtype=np.int64), type=pa.int64())

            # Emit per-episode combined batch (bounded memory, guarantees contiguous index).
            combined = table.combine_chunks()
            cols = []
            for name, kind, arg in plan:
                if kind == "global_index":
                    cols.append(idx_arr)
                elif kind == "ep_index":
                    cols.append(ep_arr)
                elif kind == "frame_index":
                    cols.append(frame_arr)
                elif kind == "synth_timestamp":
                    cols.append(ts_arr)
                elif kind == "synth_task_index":
                    cols.append(task_arr)
                elif kind == "fsl":
                    col = combined.column(name).combine_chunks()
                    cols.append(pa.FixedSizeListArray.from_arrays(col.flatten(), arg))
                elif kind == "cast":
                    col = combined.column(name).combine_chunks()
                    cols.append(col.cast(arg) if col.type != arg else col)
                else:
                    cols.append(combined.column(name).combine_chunks())

            yield pa.RecordBatch.from_arrays(cols, schema=out_schema)
            global_cursor += n

    return pa.RecordBatchReader.from_batches(out_schema, batches())


def _build_v3_compatible_meta_dir(
    src_meta_dir: Path,
    dst_meta_dir: Path,
    *,
    info_v2: dict[str, Any],
    episodes_v2: list[dict[str, Any]],
    tasks_v2: list[dict[str, Any]],
    episode_lengths: list[int],
    video_keys: list[str],
    chunks_size: int,
    fps: float,
    gr00t_stats: dict[str, Any] | None,
    external_bases: dict[str, Any] | None = None,
) -> None:
    """Populate ``meta/`` with BOTH LeRobot v3.0 metadata AND GR00T metadata."""
    if dst_meta_dir.exists():
        shutil.rmtree(dst_meta_dir)
    dst_meta_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy GR00T-specific metadata artifacts verbatim when present.
    for fname in (
        LEROBOT_MODALITY_FILENAME,
        LEROBOT_EPISODES_FILENAME,
        LEROBOT_TASKS_FILENAME,
        LEROBOT_RELATIVE_STATS_FILENAME,
        INITIAL_ACTIONS_FILENAME,
    ):
        src = src_meta_dir / fname
        if src.exists():
            shutil.copy2(src, dst_meta_dir / fname)

    # 2. Write meta/tasks.parquet (LeRobot v3.0 schema: DataFrame indexed by task string).
    if tasks_v2:
        tasks_sorted = sorted(tasks_v2, key=lambda r: int(r["task_index"]))
        task_df = pd.DataFrame(
            {"task_index": [int(t["task_index"]) for t in tasks_sorted]},
            index=pd.Index([str(t["task"]) for t in tasks_sorted], name="task"),
        )
    else:
        task_df = pd.DataFrame({"task_index": [0]}, index=pd.Index([""], name="task"))
    task_df.to_parquet(dst_meta_dir / "tasks.parquet")

    # 3. Write meta/episodes/chunk-000/file-000.parquet (LeRobot v3.0 schema).
    ep_rows: list[dict[str, Any]] = []
    cursor = 0
    for ep_meta, length in zip(episodes_v2, episode_lengths, strict=True):
        ep_idx = int(ep_meta["episode_index"])
        ep_chunk = ep_idx // chunks_size
        tasks_list = ep_meta.get("tasks") or [""]
        row: dict[str, Any] = {
            "episode_index": ep_idx,
            "tasks": [str(t) for t in tasks_list],
            "length": int(length),
            "data/chunk_index": ep_chunk,
            "data/file_index": ep_idx,
            "dataset_from_index": cursor,
            "dataset_to_index": cursor + int(length),
        }
        for vk in video_keys:
            row[f"videos/{vk}/chunk_index"] = ep_chunk
            row[f"videos/{vk}/file_index"] = ep_idx
            row[f"videos/{vk}/from_timestamp"] = 0.0
            row[f"videos/{vk}/to_timestamp"] = float(length) / float(fps)
        ep_rows.append(row)
        cursor += int(length)

    ep_dir = dst_meta_dir / "episodes" / "chunk-000"
    ep_dir.mkdir(parents=True, exist_ok=True)
    ep_table = pa.Table.from_pylist(ep_rows)
    pq.write_table(ep_table, ep_dir / "file-000.parquet")

    # 4. Write meta/gr00t_stats.json (full GR00T stats with q01/q99) + meta/stats.json (v3 compatible).
    if gr00t_stats:
        (dst_meta_dir / GROOT_STATS_FILENAME).write_text(
            json.dumps(gr00t_stats, indent=2) + "\n", encoding="utf-8"
        )
        # Keep standard stats keys (and q01/q99) in stats.json too so both readers work.
        v3_stats: dict[str, Any] = {}
        for feat_key, st in gr00t_stats.items():
            if not isinstance(st, dict) or "mean" not in st:
                continue
            entry = {k: v for k, v in st.items() if k in ("mean", "std", "min", "max", "q01", "q99")}
            entry["count"] = [cursor]
            v3_stats[feat_key] = entry
        (dst_meta_dir / LEROBOT_STATS_FILENAME).write_text(
            json.dumps(v3_stats, indent=2) + "\n", encoding="utf-8"
        )

    # 5. Write meta/info.json (v3.0 compatible + storage_format="lance").
    total_frames = cursor
    total_episodes = len(ep_rows)
    features_v3: dict[str, Any] = {}
    for feat_name, feat_meta in info_v2.get("features", {}).items():
        fm = dict(feat_meta)
        if fm.get("dtype") == "video":
            info_block = dict(fm.get("info") or {})
            info_block.setdefault("video.fps", float(fps))
            fm["info"] = info_block
        features_v3[feat_name] = fm
    # Ensure standard LeRobot control columns are present in features.
    for ctrl_col, dtype, shape in (
        ("timestamp", "float32", [1]),
        ("frame_index", "int64", [1]),
        ("episode_index", "int64", [1]),
        ("index", "int64", [1]),
        ("task_index", "int64", [1]),
    ):
        features_v3.setdefault(ctrl_col, {"dtype": dtype, "shape": shape, "names": None})

    info_v3: dict[str, Any] = {
        "codebase_version": "v3.0",
        "storage_format": "lance",
        "robot_type": info_v2.get("robot_type", "unknown"),
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": max(1, len(tasks_v2)),
        "chunks_size": chunks_size,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 500,
        "fps": int(fps) if float(fps).is_integer() else float(fps),
        "splits": info_v2.get("splits") or {"train": f"0:{total_episodes}"},
        "data_path": V3_DATA_PATH_TEMPLATE,
        "video_path": V3_VIDEO_PATH_TEMPLATE if video_keys else None,
        "features": features_v3,
    }
    if external_bases:
        info_v3["lance_external_bases"] = external_bases

    (dst_meta_dir / LEROBOT_INFO_FILENAME).write_text(
        json.dumps(info_v3, indent=2) + "\n", encoding="utf-8"
    )


def convert_v2_to_lance(
    src_root: str | Path,
    out_target: str | Path,
    *,
    video_mode: VideoMode = "auto",
    video_base_uri: str | None = None,
    modality_json_path: str | Path | None = None,
    storage_options: dict[str, str] | None = None,
    source_storage_options: dict[str, str] | None = None,
    video_workers: int = DEFAULT_IO_WORKERS,
    header_probe_bytes: int = DEFAULT_HEADER_PROBE_BYTES,
    write_sample_index: bool = True,
    scalar_indexes: bool = True,
    compute_stats_if_missing: bool = True,
) -> Path | str:
    """Convert a GR00T LeRobot v2.1 dataset (local or ``s3://``) to ``lerobot-lancedb`` Lance layout.

    Args:
        src_root: Local directory or object-store URI (``s3://bucket/prefix``) of a GR00T v2.1 dataset.
        out_target: Output local directory or object-store URI (``s3://bucket/out-lance``).
        video_mode: ``"auto"`` (external for remote sources, copy for local), ``"external"``
            (zero-copy External Blob v2 descriptors referencing source MP4s in place), or
            ``"copy"`` (managed blobs inside ``videos.lance``).
        video_base_uri: Override external base URI (e.g. ``s3://bucket/dataset`` when indexing
            from a local mirror of an S3 bucket).
        modality_json_path: Optional path to a ``modality.json`` to embed if the source
            v2.1 dataset doesn't have one in ``meta/``.
        storage_options: LanceDB / object-store options for output.
        source_storage_options: Object-store options for remote source reads (falls back to
            ``storage_options``).
    """
    src_is_remote = is_remote_uri(src_root)
    resolved_mode: Literal["external", "copy"]
    if video_mode == "auto":
        resolved_mode = "external" if (src_is_remote or video_base_uri is not None) else "copy"
    else:
        resolved_mode = video_mode

    if src_is_remote and resolved_mode == "copy":
        raise ValueError(
            "copy mode is only supported for local v2.1 sources; use video_mode='external' for S3/remote sources."
        )

    if is_remote_uri(out_target):
        _ensure_remote_output_empty(_connect_output(out_target, storage_options), out_target)

    src_opts = source_storage_options if source_storage_options is not None else storage_options

    with (
        open_source(
            root=src_root,
            storage_options=src_opts,
            max_workers=video_workers,
        ) as source,
        tempfile.TemporaryDirectory(prefix="gr00t-lance-convert-") as tmp,
    ):
        staging = Path(tmp)
        source_files = source.list_files()
        meta_root = source.materialize_meta(staging)
        src_meta_dir = meta_root / "meta"

        info_v2 = json.loads((src_meta_dir / LEROBOT_INFO_FILENAME).read_text(encoding="utf-8"))
        episodes_v2 = sorted(
            _load_jsonl(src_meta_dir / LEROBOT_EPISODES_FILENAME),
            key=lambda r: int(r["episode_index"]),
        )
        tasks_path = src_meta_dir / LEROBOT_TASKS_FILENAME
        tasks_v2 = _load_jsonl(tasks_path) if tasks_path.exists() else []

        if modality_json_path is not None:
            shutil.copy2(Path(modality_json_path), src_meta_dir / LEROBOT_MODALITY_FILENAME)
        if not (src_meta_dir / LEROBOT_MODALITY_FILENAME).exists():
            raise FileNotFoundError(
                f"No {LEROBOT_MODALITY_FILENAME} found in {src_meta_dir}; pass modality_json_path=..."
            )
        modality_meta = json.loads(
            (src_meta_dir / LEROBOT_MODALITY_FILENAME).read_text(encoding="utf-8")
        )

        chunks_size = int(info_v2.get("chunks_size", 1000))
        fps = float(info_v2.get("fps", 30))
        data_pattern = info_v2["data_path"]
        video_pattern = info_v2.get("video_path")
        features = info_v2.get("features", {})
        video_keys = [k for k, v in features.items() if v.get("dtype") == "video"]

        # Resolve per-episode parquet & video relative paths
        data_rels: list[str] = []
        ep_data_rel_map: list[tuple[int, int, str]] = []
        video_entries: list[tuple[str, int, int, str]] = []  # (video_key, chunk_idx, ep_idx, rel_path)

        for ep in episodes_v2:
            ep_idx = int(ep["episode_index"])
            chunk_idx = ep_idx // chunks_size
            data_rel = data_pattern.format(episode_chunk=chunk_idx, episode_index=ep_idx)
            if data_rel not in source_files:
                raise FileNotFoundError(f"Missing referenced v2.1 parquet: {data_rel}")
            data_rels.append(data_rel)
            ep_data_rel_map.append((ep_idx, int(ep["length"]), data_rel))

            if video_pattern and video_keys:
                for vk in video_keys:
                    vid_rel = video_pattern.format(
                        episode_chunk=chunk_idx, video_key=vk, episode_index=ep_idx
                    )
                    if vid_rel not in source_files:
                        raise FileNotFoundError(f"Missing referenced v2.1 video: {vid_rel}")
                    video_entries.append((vk, chunk_idx, ep_idx, vid_rel))

        # Stage referenced parquets (small tabular files)
        materialized_data = source.materialize_data(data_rels, staging)
        rel_to_local_pq = dict(zip(data_rels, materialized_data, strict=True))

        # Verify actual lengths against metadata lengths
        ep_parquets: list[tuple[int, int, Path]] = []
        actual_lengths: list[int] = []
        for ep_idx, nom_len, rel in ep_data_rel_map:
            pq_path = rel_to_local_pq[rel]
            actual_len = min(pq.read_metadata(pq_path).num_rows, nom_len)
            ep_parquets.append((ep_idx, actual_len, pq_path))
            actual_lengths.append(actual_len)

        # Load existing GR00T stats or compute them after writing frames.lance
        existing_stats: dict[str, Any] | None = None
        for cand in (src_meta_dir / GROOT_STATS_FILENAME, src_meta_dir / LEROBOT_STATS_FILENAME):
            if cand.exists():
                loaded = json.loads(cand.read_text(encoding="utf-8"))
                loaded.pop("__fingerprints__", None)
                existing_stats = loaded
                break

        # Probe videos for external mode or read bytes for copy mode
        videos_schema = _videos_schema()
        probed: dict[str, ProbedMp4] = {}
        resolved_base_uri = ""
        table_options: dict[str, Any] = {}

        if resolved_mode == "external" and video_entries:
            vid_rels = [rel for _, _, _, rel in video_entries]
            resolved_base_uri = (
                normalize_base_uri(video_base_uri)
                if video_base_uri is not None
                else normalize_base_uri(source.default_video_base_uri(vid_rels))
            )
            table_options = _external_table_options(
                resolved_base_uri, storage_options, source_storage_options
            )

            def _probe_one(rel: str) -> tuple[str, ProbedMp4]:
                res = probe_mp4_byte_index(
                    rel,
                    source.read_range,
                    file_size=source_files[rel].size,
                    write_sample_index=write_sample_index,
                    header_probe_bytes=header_probe_bytes,
                )
                return rel, res

            with ThreadPoolExecutor(max_workers=max(1, video_workers)) as pool:
                futs = {pool.submit(_probe_one, rel): rel for rel in set(vid_rels)}
                for fut in as_completed(futs):
                    rel, res = fut.result()
                    probed[rel] = res

        def _build_videos_table(db: Any):
            if not video_entries:
                return db.create_table(VIDEOS_TABLE, schema=videos_schema, mode="create")

            if resolved_mode == "external":
                def ext_batches():
                    vkeys, cidxs, fidxs, fsizes, moffs, msizes = [], [], [], [], [], []
                    kf_i, kf_p, sidx, uris, pos, sizes = [], [], [], [], [], []
                    pending_bytes = 0

                    def flush() -> pa.RecordBatch:
                        return pa.RecordBatch.from_arrays(
                            [
                                pa.array(vkeys, type=pa.string()),
                                pa.array(cidxs, type=pa.int64()),
                                pa.array(fidxs, type=pa.int64()),
                                pa.array(fsizes, type=pa.int64()),
                                pa.array(moffs, type=pa.int64()),
                                pa.array(msizes, type=pa.int64()),
                                _list_int64_array(kf_i),
                                _list_int64_array(kf_p),
                                lancedb.blob_uri_array(uris, positions=pos, sizes=sizes),
                                pa.array(sidx, type=pa.binary()),
                            ],
                            schema=videos_schema,
                        )

                    for vk, chunk_idx, ep_idx, rel in video_entries:
                        p = probed[rel]
                        vkeys.append(vk)
                        cidxs.append(chunk_idx)
                        fidxs.append(ep_idx)
                        fsizes.append(p.file_size)
                        moffs.append(p.moov_offset)
                        msizes.append(p.moov_size)
                        kf_i.append(p.kf_indices)
                        kf_p.append(p.kf_positions)
                        uris.append(_external_video_uri(resolved_base_uri, rel))
                        pos.append(0)
                        sizes.append(p.file_size)
                        sidx.append(p.sample_index)
                        pending_bytes += len(p.kf_indices) * 16 + len(p.sample_index or b"") + 256
                        if pending_bytes >= EXTERNAL_VIDEO_BATCH_BYTES:
                            yield flush()
                            for lst in (vkeys, cidxs, fidxs, fsizes, moffs, msizes, kf_i, kf_p, sidx, uris, pos, sizes):
                                lst.clear()
                            pending_bytes = 0
                    if vkeys:
                        yield flush()

                return db.create_table(
                    VIDEOS_TABLE,
                    pa.RecordBatchReader.from_batches(videos_schema, ext_batches()),
                    schema=videos_schema,
                    mode="create",
                    **table_options,
                )

            # Managed copy mode (local sources)
            storage_schema = pa.schema(
                [
                    pa.field(VIDEO_BLOB_COLUMN, pa.large_binary()) if f.name == VIDEO_BLOB_COLUMN else f
                    for f in videos_schema
                ]
            )
            local_src = Path(src_root)

            def copy_batches():
                pending, pending_bytes = [], 0
                for vk, chunk_idx, ep_idx, rel in video_entries:
                    mp4 = local_src / rel
                    data = mp4.read_bytes()
                    pending.append(
                        {
                            "video_key": vk,
                            "chunk_index": chunk_idx,
                            "file_index": ep_idx,
                            **build_video_byte_index(mp4),
                            VIDEO_BLOB_COLUMN: data,
                            "sample_index": sample_index_from_mp4(data) if write_sample_index else None,
                        }
                    )
                    pending_bytes += len(data)
                    if pending_bytes >= VIDEO_BATCH_BYTES:
                        yield pa.RecordBatch.from_pylist(pending, schema=storage_schema)
                        pending, pending_bytes = [], 0
                if pending:
                    yield pa.RecordBatch.from_pylist(pending, schema=storage_schema)

            return db.create_table(
                VIDEOS_TABLE,
                pa.RecordBatchReader.from_batches(storage_schema, copy_batches()),
                schema=videos_schema,
                mode="create",
            )

        def _needed_stat_keys() -> list[str]:
            keys: list[str] = []
            for mod in ("state", "action"):
                for g in modality_meta.get(mod, {}).values():
                    k = g.get("original_key") or DEFAULT_COLUMN_NAMES[mod]
                    if k not in keys:
                        keys.append(k)
            return keys

        def _build_into(db: Any, meta_dst_dir: Path) -> None:
            frames_table = db.create_table(
                FRAMES_TABLE,
                _v2_frames_reader(ep_parquets, fps),
                mode="create",
            )
            videos_table = _build_videos_table(db)

            # Compute GR00T q01/q99 stats from frames.lance if missing or incomplete
            stats_to_write = existing_stats
            needed = _needed_stat_keys()
            missing_q = (
                stats_to_write is None
                or any(
                    k not in stats_to_write
                    or "q01" not in stats_to_write[k]
                    or "q99" not in stats_to_write[k]
                    for k in needed
                )
            )
            if missing_q and compute_stats_if_missing and needed:
                computed = compute_gr00t_stats_from_lance_table(frames_table, needed)
                stats_to_write = {**(stats_to_write or {}), **computed}

            _build_v3_compatible_meta_dir(
                src_meta_dir=src_meta_dir,
                dst_meta_dir=meta_dst_dir,
                info_v2=info_v2,
                episodes_v2=episodes_v2,
                tasks_v2=tasks_v2,
                episode_lengths=actual_lengths,
                video_keys=video_keys,
                chunks_size=chunks_size,
                fps=fps,
                gr00t_stats=stats_to_write,
            )
            _write_meta_table(db, meta_dst_dir, mode="create")
            if scalar_indexes:
                _create_scalar_indexes(frames_table, videos_table if video_entries else None)

        if is_remote_uri(out_target):
            db = _connect_output(out_target, storage_options)
            _ensure_remote_output_empty(db, out_target)
            _build_into(db, staging / "_meta_out")
            return str(out_target)

        out_path = Path(out_target).expanduser().resolve()
        with _local_build_dir(out_path) as build_dir:
            db = _connect_output(build_dir, storage_options)
            _build_into(db, build_dir / "meta")
            _commit_local_output(build_dir, out_path)
        return out_path


def attach_gr00t_meta(
    lance_root: str | Path,
    modality_json_path: str | Path,
    *,
    relative_stats_path: str | Path | None = None,
    initial_actions_path: str | Path | None = None,
    compute_stats: bool = True,
    storage_options: dict[str, str] | None = None,
) -> None:
    """Attach GR00T ``modality.json`` (+ computed q01/q99 stats) to an existing v3 Lance dataset.

    Works on BOTH local Lance directories and remote ``s3://`` / ``gs://`` Lance prefixes
    without downloading or modifying any video blobs (zero v3->v2 downgrade).
    """
    modality_bytes = Path(modality_json_path).read_bytes()
    modality_meta = json.loads(modality_bytes.decode("utf-8"))

    db = _connect_output(lance_root, storage_options)
    meta_table = db.open_table(META_TABLE, **({"storage_options": storage_options} if storage_options else {}))
    frames_table = db.open_table(FRAMES_TABLE, **({"storage_options": storage_options} if storage_options else {}))

    # Read existing meta.lance entries
    existing_arrow = meta_table.search().select(["path", "data"]).to_arrow()
    entries: dict[str, bytes] = {
        p: d
        for p, d in zip(
            existing_arrow.column("path").to_pylist(),
            existing_arrow.column("data").to_pylist(),
            strict=True,
        )
    }

    entries[LEROBOT_MODALITY_FILENAME] = modality_bytes

    if compute_stats:
        needed: list[str] = []
        for mod in ("state", "action"):
            for g in modality_meta.get(mod, {}).values():
                k = g.get("original_key") or DEFAULT_COLUMN_NAMES[mod]
                if k not in needed:
                    needed.append(k)
        if needed:
            computed = compute_gr00t_stats_from_lance_table(frames_table, needed)
            entries[GROOT_STATS_FILENAME] = (json.dumps(computed, indent=2) + "\n").encode("utf-8")

    if relative_stats_path is not None:
        entries[LEROBOT_RELATIVE_STATS_FILENAME] = Path(relative_stats_path).read_bytes()
    if initial_actions_path is not None:
        entries[INITIAL_ACTIONS_FILENAME] = Path(initial_actions_path).read_bytes()

    # Overwrite meta.lance cleanly (tiny table, atomic Lance commit)
    meta_schema = pa.schema([pa.field("path", pa.string()), pa.field("data", pa.large_binary())])
    db.create_table(
        META_TABLE,
        pa.Table.from_pylist(
            [{"path": p, "data": entries[p]} for p in sorted(entries)],
            schema=meta_schema,
        ),
        mode="overwrite",
    )

    # Also update local meta/ directory if local root
    if not is_remote_uri(lance_root):
        local_meta = Path(lance_root) / "meta"
        local_meta.mkdir(parents=True, exist_ok=True)
        for fname in (
            LEROBOT_MODALITY_FILENAME,
            GROOT_STATS_FILENAME,
            LEROBOT_RELATIVE_STATS_FILENAME,
            INITIAL_ACTIONS_FILENAME,
        ):
            if fname in entries:
                (local_meta / fname).write_bytes(entries[fname])


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


def main_convert(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="gr00t-lance-convert",
        description="Convert a GR00T LeRobot v2.1 dataset (local or s3://) to lerobot-lancedb Lance layout.",
    )
    parser.add_argument("--src", required=True, help="Source v2.1 dataset directory or s3:// URI.")
    parser.add_argument("--out", required=True, help="Output Lance directory or s3:// URI.")
    parser.add_argument(
        "--video-mode",
        choices=("auto", "external", "copy"),
        default="auto",
        help="How to store videos (default: auto -> external for remote, copy for local).",
    )
    parser.add_argument(
        "--video-base-uri",
        default=None,
        help="Optional external base URI (e.g. s3://bucket/dataset) for local-mirror indexing.",
    )
    parser.add_argument("--modality-json", default=None, help="Optional modality.json path to embed.")
    parser.add_argument("--storage-option", action="append", default=None, help="Output KEY=VALUE option.")
    parser.add_argument(
        "--source-storage-option", action="append", default=None, help="Source KEY=VALUE option."
    )
    parser.add_argument("--video-workers", type=int, default=DEFAULT_IO_WORKERS)
    parser.add_argument("--no-sample-index", action="store_true")
    parser.add_argument("--no-scalar-indexes", action="store_true")
    args = parser.parse_args(argv)

    out = convert_v2_to_lance(
        src_root=args.src,
        out_target=args.out,
        video_mode=args.video_mode,
        video_base_uri=args.video_base_uri,
        modality_json_path=args.modality_json,
        storage_options=_parse_kv_options(args.storage_option),
        source_storage_options=_parse_kv_options(args.source_storage_option),
        video_workers=args.video_workers,
        write_sample_index=not args.no_sample_index,
        scalar_indexes=not args.no_scalar_indexes,
    )
    print(f"Converted GR00T v2.1 dataset -> {out}")


def main_attach(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="gr00t-lance-attach",
        description="Attach GR00T modality.json + computed q01/q99 stats to an existing v3 Lance dataset.",
    )
    parser.add_argument("--root", required=True, help="Existing Lance dataset dir or s3:// URI.")
    parser.add_argument("--modality-json", required=True, help="Path to GR00T modality.json.")
    parser.add_argument("--relative-stats", default=None, help="Optional relative_stats.json path.")
    parser.add_argument("--initial-actions", default=None, help="Optional initial_actions.npz path.")
    parser.add_argument("--no-compute-stats", action="store_true")
    parser.add_argument("--storage-option", action="append", default=None, help="KEY=VALUE option.")
    args = parser.parse_args(argv)

    attach_gr00t_meta(
        lance_root=args.root,
        modality_json_path=args.modality_json,
        relative_stats_path=args.relative_stats,
        initial_actions_path=args.initial_actions,
        compute_stats=not args.no_compute_stats,
        storage_options=_parse_kv_options(args.storage_option),
    )
    print(f"Attached GR00T modality metadata to {args.root}")


__all__ = [
    "attach_gr00t_meta",
    "convert_v2_to_lance",
    "main_attach",
    "main_convert",
]
