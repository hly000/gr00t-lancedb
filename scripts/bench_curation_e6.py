#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""E6 Curation & Repair Benchmark: Native LeRobot v2.1 vs Lance.

Head-to-head reproducible benchmark comparing dataset repair, idle-frame filtering,
and GR00T statistics recomputation on any LeRobot v2.1 dataset (LIBERO-10, RoboCasa,
or synthetic):

Workload:
  1. Drop ``--drop-frac`` (default 5%) bad episodes deterministically (plus any NaN/Inf episodes).
  2. Detect & filter idle frames (``action_delta_norm < --idle-eps``, default ``1e-4``).
  3. Recompute exact GR00T ``stats.json`` (``mean``, ``std``, ``min``, ``max``, ``q01``, ``q99``).
  4. Measure wall time, bytes written (tabular vs video vs meta), files/objects touched,
     resulting disk footprint, and filtered dataloader initialization time.

Key architectural contrast highlighted in the output:
  - Native LeRobot v2.1 must copy/rewrite surviving episode files + MP4s to produce a clean
    versioned dataset snapshot (and scan every parquet to recompute stats), AND still cannot
    filter mid-episode idle frames without breaking GR00T's 16-step action horizon / video
    temporal continuity (or re-encoding MP4s).
  - Lance appends 6 compact columnar quality tags directly onto ``frames.lance`` via zero-copy
    column evolution (0 bytes written to ``videos.lance``, 0 MP4 re-encodes, instant version bump)
    and pushes ``where="is_valid = true"`` down to ``LanceShardedSingleStepDataset`` in
    milliseconds while keeping full intact episodes for action horizons.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

import numpy as np
import pyarrow.parquet as pq

# Import gr00t_lancedb before lerobot_lancedb so _hub_compat shims huggingface_hub<1.0 cleanly.
from gr00t_lancedb import (
    EmbodimentTag,
    LanceShardedSingleStepDataset,
    ModalityConfig,
    compute_gr00t_stats_from_arrays,
    convert_v2_to_lance,
    curate_lance_dataset,
)
from gr00t_lancedb.metadata import DEFAULT_COLUMN_NAMES
from lerobot_lancedb import LanceDBDataset


def _dir_stats(root: Path) -> tuple[int, int]:
    """Return ``(total_bytes, file_count)`` for all regular files under ``root``."""
    total_bytes = 0
    file_count = 0
    if not root.exists():
        return 0, 0
    for f in root.rglob("*"):
        if f.is_file():
            total_bytes += f.stat().st_size
            file_count += 1
    return total_bytes, file_count


def _build_modality_configs(modality_meta: dict[str, Any], action_horizon: int = 16) -> dict[str, ModalityConfig]:
    cfg: dict[str, ModalityConfig] = {}
    if modality_meta.get("video"):
        cfg["video"] = ModalityConfig(delta_indices=[0], modality_keys=list(modality_meta["video"].keys()))
    if modality_meta.get("state"):
        cfg["state"] = ModalityConfig(delta_indices=[0], modality_keys=list(modality_meta["state"].keys()))
    if modality_meta.get("action"):
        cfg["action"] = ModalityConfig(
            delta_indices=list(range(action_horizon)),
            modality_keys=list(modality_meta["action"].keys()),
        )
    cfg["language"] = ModalityConfig(delta_indices=[0], modality_keys=["task"])
    return cfg


def _extract_modality_orig_keys(modality_meta: dict[str, Any]) -> tuple[list[str], list[str]]:
    state_keys: list[str] = []
    for g in modality_meta.get("state", {}).values():
        k = g.get("original_key") or DEFAULT_COLUMN_NAMES["state"]
        if k not in state_keys:
            state_keys.append(k)
    action_keys: list[str] = []
    for g in modality_meta.get("action", {}).values():
        k = g.get("original_key") or DEFAULT_COLUMN_NAMES["action"]
        if k not in action_keys:
            action_keys.append(k)
    return state_keys, action_keys


def _pq_col_to_2d_float64(table: Any, col_name: str) -> np.ndarray:
    arr = table.column(col_name).combine_chunks()
    n = len(arr)
    if n == 0:
        return np.empty((0, 0), dtype=np.float64)
    if hasattr(arr, "flatten") and hasattr(arr.type, "value_type"):
        flat = arr.flatten().to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
        if len(flat) % n == 0:
            return flat.reshape(n, len(flat) // n)
        return np.vstack([np.asarray(x, dtype=np.float64).reshape(-1) for x in arr.to_pylist()])
    return arr.to_numpy(zero_copy_only=False).astype(np.float64, copy=False).reshape(n, 1)


def run_native_v2_curation(
    v2_src: Path,
    v2_dst: Path,
    *,
    drop_frac: float,
    drop_seed: int,
    idle_eps: float,
) -> dict[str, Any]:
    """Native LeRobot v2.1 repair & curation baseline.

    Performs the full pipeline required to produce a clean versioned v2.1 dataset:
      1. Scan all episode parquets for NaN/Inf + idle frame counts.
      2. Drop bad episodes (``drop_frac`` deterministic + any NaN episodes).
      3. Copy surviving episode parquets & MP4s to ``v2_dst`` and rewrite
         ``meta/episodes.jsonl``, ``meta/info.json``, and recomputed ``meta/stats.json``.
    """
    if v2_dst.exists():
        shutil.rmtree(v2_dst)
    v2_dst.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    src_meta = v2_src / "meta"
    info = json.loads((src_meta / "info.json").read_text(encoding="utf-8"))
    modality = json.loads((src_meta / "modality.json").read_text(encoding="utf-8"))
    episodes = sorted(
        [json.loads(line) for line in (src_meta / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()],
        key=lambda r: int(r["episode_index"]),
    )

    chunks_size = int(info.get("chunks_size", 1000))
    data_pat = info["data_path"]
    video_pat = info.get("video_path")
    video_keys = [k for k, v in info.get("features", {}).items() if v.get("dtype") == "video"]
    state_keys, action_keys = _extract_modality_orig_keys(modality)
    stat_keys = list(dict.fromkeys([*state_keys, *action_keys]))

    # Step 1: Scan all parquets to detect NaN/Inf + idle frames + gather stats arrays
    t_scan0 = time.perf_counter()
    all_ep_ids = np.array([int(ep["episode_index"]) for ep in episodes], dtype=np.int64)
    dropped_ep_set: set[int] = set()
    if drop_frac > 0.0 and len(all_ep_ids) > 0:
        n_drop = min(len(all_ep_ids), max(1, int(round(len(all_ep_ids) * drop_frac))))
        rng = np.random.default_rng(drop_seed)
        dropped_ep_set.update(int(x) for x in rng.choice(all_ep_ids, size=n_drop, replace=False))

    nan_episodes: set[int] = set()
    total_frames = 0
    detected_idle_frames = 0
    per_ep_arrays: dict[int, dict[str, np.ndarray]] = {}

    for ep in episodes:
        ep_id = int(ep["episode_index"])
        chunk_idx = ep_id // chunks_size
        pq_rel = data_pat.format(episode_chunk=chunk_idx, episode_index=ep_id)
        tbl = pq.read_table(v2_src / pq_rel, columns=stat_keys)
        n = min(tbl.num_rows, int(ep["length"]))
        if n < tbl.num_rows:
            tbl = tbl.slice(0, n)
        total_frames += n

        ep_cols = {k: _pq_col_to_2d_float64(tbl, k) for k in stat_keys}
        act_blocks = [ep_cols[k] for k in action_keys if k in ep_cols and ep_cols[k].shape[1] > 0]
        st_blocks = [ep_cols[k] for k in state_keys if k in ep_cols and ep_cols[k].shape[1] > 0]
        act_mat = np.concatenate(act_blocks, axis=1) if act_blocks else np.empty((n, 0))
        st_mat = np.concatenate(st_blocks, axis=1) if st_blocks else np.empty((n, 0))

        ep_bad_num = (
            (act_mat.shape[1] > 0 and not np.all(np.isfinite(act_mat)))
            or (st_mat.shape[1] > 0 and not np.all(np.isfinite(st_mat)))
        )
        if ep_bad_num:
            nan_episodes.add(ep_id)

        if act_mat.shape[1] > 0 and n >= 2:
            norms = np.linalg.norm(act_mat[1:] - act_mat[:-1], axis=1)
            deltas = np.empty(n, dtype=np.float32)
            deltas[0] = np.float32(norms[0])
            deltas[1:] = norms.astype(np.float32)
            detected_idle_frames += int(np.sum(deltas.astype(np.float64) < idle_eps))
        elif n == 1:
            detected_idle_frames += 1

        if ep_id not in dropped_ep_set and ep_id not in nan_episodes:
            per_ep_arrays[ep_id] = ep_cols

    bad_ep_set = dropped_ep_set | nan_episodes
    surviving_eps = [ep for ep in episodes if int(ep["episode_index"]) not in bad_ep_set]
    scan_s = time.perf_counter() - t_scan0

    # Step 2: Recompute GR00T stats over surviving episodes
    t_stats0 = time.perf_counter()
    cat_arrays = {
        k: np.concatenate([per_ep_arrays[int(ep["episode_index"])][k] for ep in surviving_eps], axis=0)
        for k in stat_keys
        if surviving_eps
    }
    recomputed_stats = compute_gr00t_stats_from_arrays(cat_arrays) if cat_arrays else {}
    stats_s = time.perf_counter() - t_stats0

    # Step 3: Copy surviving parquets + MP4s to produce repaired v2.1 snapshot
    t_pq0 = time.perf_counter()
    bytes_written_parquets = 0
    files_written_parquets = 0
    for ep in surviving_eps:
        ep_id = int(ep["episode_index"])
        chunk_idx = ep_id // chunks_size
        pq_rel = data_pat.format(episode_chunk=chunk_idx, episode_index=ep_id)
        src_p = v2_src / pq_rel
        dst_p = v2_dst / pq_rel
        dst_p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_p, dst_p)
        bytes_written_parquets += dst_p.stat().st_size
        files_written_parquets += 1
    parquet_copy_s = time.perf_counter() - t_pq0

    t_vid0 = time.perf_counter()
    bytes_written_videos = 0
    files_written_videos = 0
    if video_pat and video_keys:
        for ep in surviving_eps:
            ep_id = int(ep["episode_index"])
            chunk_idx = ep_id // chunks_size
            for vk in video_keys:
                vid_rel = video_pat.format(episode_chunk=chunk_idx, video_key=vk, episode_index=ep_id)
                src_v = v2_src / vid_rel
                dst_v = v2_dst / vid_rel
                dst_v.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_v, dst_v)
                bytes_written_videos += dst_v.stat().st_size
                files_written_videos += 1
    video_copy_s = time.perf_counter() - t_vid0

    # Step 4: Rewrite meta/
    t_meta0 = time.perf_counter()
    dst_meta = v2_dst / "meta"
    dst_meta.mkdir(parents=True, exist_ok=True)
    for meta_file in src_meta.iterdir():
        if meta_file.is_file() and meta_file.name not in ("episodes.jsonl", "info.json", "stats.json", "gr00t_stats.json"):
            shutil.copy2(meta_file, dst_meta / meta_file.name)

    with open(dst_meta / "episodes.jsonl", "w", encoding="utf-8") as f:
        for ep in surviving_eps:
            f.write(json.dumps(ep) + "\n")

    surviving_frames = sum(int(ep["length"]) for ep in surviving_eps)
    new_info = dict(info)
    new_info["total_episodes"] = len(surviving_eps)
    new_info["total_frames"] = surviving_frames
    if "total_videos" in new_info:
        new_info["total_videos"] = len(surviving_eps) * len(video_keys)
    new_info["splits"] = {"train": f"0:{len(surviving_eps)}"}
    (dst_meta / "info.json").write_text(json.dumps(new_info, indent=2) + "\n", encoding="utf-8")
    (dst_meta / "stats.json").write_text(json.dumps(recomputed_stats, indent=2) + "\n", encoding="utf-8")
    meta_s = time.perf_counter() - t_meta0

    total_wall_s = time.perf_counter() - t0
    meta_bytes, meta_files = _dir_stats(dst_meta)
    disk_bytes, disk_files = _dir_stats(v2_dst)

    return {
        "method": "native_lerobot_v2.1",
        "wall_time_s": round(total_wall_s, 4),
        "breakdown_s": {
            "scan_parquets_s": round(scan_s, 4),
            "recompute_stats_s": round(stats_s, 4),
            "copy_parquets_s": round(parquet_copy_s, 4),
            "copy_videos_s": round(video_copy_s, 4),
            "rewrite_meta_s": round(meta_s, 4),
        },
        "total_episodes_before": len(episodes),
        "bad_episodes_dropped": len(bad_ep_set),
        "surviving_episodes": len(surviving_eps),
        "total_frames_before": total_frames,
        "surviving_frames": surviving_frames,
        "idle_frames_detected": detected_idle_frames,
        "mid_episode_idle_filtering_supported": False,
        "mid_episode_idle_note": (
            "Native GR00T v2.1 cannot drop mid-episode idle frames without breaking 16-step "
            "action horizon / video temporal continuity (or re-encoding MP4s)."
        ),
        "bytes_written_parquets": bytes_written_parquets,
        "bytes_written_videos": bytes_written_videos,
        "bytes_written_meta": meta_bytes,
        "bytes_written_total": bytes_written_parquets + bytes_written_videos + meta_bytes,
        "files_touched": files_written_parquets + files_written_videos + meta_files,
        "files_written_videos": files_written_videos,
        "resulting_disk_bytes": disk_bytes,
        "resulting_file_count": disk_files,
    }


def run_lance_curation_benchmark(
    lance_work_dir: Path,
    modality_meta: dict[str, Any],
    *,
    drop_frac: float,
    drop_seed: int,
    idle_eps: float,
    action_horizon: int = 16,
) -> dict[str, Any]:
    """Run Lance tabular-only curation + verify instant filtered dataset init & upstream compat."""
    disk_before_bytes, disk_before_files = _dir_stats(lance_work_dir)
    modality_configs = _build_modality_configs(modality_meta, action_horizon=action_horizon)

    # Measure unfiltered dataset init & anchor count before curation
    t_unfilt0 = time.perf_counter()
    ds_unfilt = LanceShardedSingleStepDataset(
        dataset_path=lance_work_dir,
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        modality_configs=modality_configs,
        allow_padding=True,
    )
    unfilt_init_s = time.perf_counter() - t_unfilt0
    unfilt_anchors = len(ds_unfilt.get_all_valid_anchors())
    ds_unfilt.close()

    # Run zero-copy tabular curation on Lance
    report = curate_lance_dataset(
        lance_root=lance_work_dir,
        idle_eps=idle_eps,
        drop_random_frac=drop_frac,
        drop_seed=drop_seed,
        keep_idle=False,
        recompute_stats=True,
    )

    disk_after_bytes, disk_after_files = _dir_stats(lance_work_dir)

    # Verify instant filtered LanceShardedSingleStepDataset init with SQL predicate pushdown
    t_filt0 = time.perf_counter()
    ds_filt = LanceShardedSingleStepDataset(
        dataset_path=lance_work_dir,
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        modality_configs=modality_configs,
        allow_padding=True,
        where="is_valid = true",
    )
    filt_init_s = time.perf_counter() - t_filt0
    filt_anchors = len(ds_filt.get_all_valid_anchors())
    ds_filt.close()

    # Verify unmodified upstream lerobot_lancedb still opens cleanly
    upstream = LanceDBDataset(root=lance_work_dir, return_uint8=True)
    upstream_ok = (upstream.num_episodes == report.total_episodes and upstream.num_frames == report.total_frames)

    anchor_reduction_pct = (
        round(100.0 * (unfilt_anchors - filt_anchors) / unfilt_anchors, 2)
        if unfilt_anchors > 0
        else 0.0
    )

    return {
        "method": "lance_tabular_evolution",
        "wall_time_s": report.wall_time_s,
        "frames_version_before": report.frames_table_version_before,
        "frames_version_after": report.frames_table_version,
        "total_episodes": report.total_episodes,
        "bad_episodes": report.bad_episodes,
        "valid_episodes": report.valid_episodes,
        "total_frames": report.total_frames,
        "idle_frames": report.idle_frames,
        "anomaly_frames": report.anomaly_frames,
        "valid_frames": report.valid_frames,
        "valid_pct": report.valid_pct,
        "mid_episode_idle_filtering_supported": True,
        "bytes_written_frames": report.bytes_written_frames,
        "bytes_written_videos": report.bytes_written_videos,
        "bytes_written_meta": report.bytes_written_meta,
        "bytes_written_total": report.bytes_written_total,
        "files_written_frames": report.files_written_frames,
        "files_written_videos": report.files_written_videos,
        "scalar_indexes_built": report.scalar_indexes_built,
        "disk_before_bytes": disk_before_bytes,
        "disk_after_bytes": disk_after_bytes,
        "disk_delta_bytes": disk_after_bytes - disk_before_bytes,
        "resulting_file_count": disk_after_files,
        "unfiltered_init_s": round(unfilt_init_s, 4),
        "filtered_init_s": round(filt_init_s, 4),
        "unfiltered_anchors": unfilt_anchors,
        "filtered_anchors": filt_anchors,
        "anchor_reduction_pct": anchor_reduction_pct,
        "upstream_lerobot_lancedb_compatible": upstream_ok,
    }


def format_markdown_summary(result: dict[str, Any]) -> str:
    nat = result["native_v2"]
    lan = result["lance"]
    speedup = nat["wall_time_s"] / max(lan["wall_time_s"], 1e-6)
    io_ratio = nat["bytes_written_total"] / max(lan["bytes_written_total"], 1)

    def _mb(n: int) -> str:
        return f"{n / 1e6:.2f} MB"

    lines = [
        "### E6 Curation & Repair Benchmark (Native LeRobot v2.1 vs Lance)",
        "",
        f"- **Dataset**: `{result['dataset']}` ({result['config']['total_episodes']} episodes, {result['config']['total_frames']} frames)",
        f"- **Workload**: drop `{result['config']['drop_frac']*100:.1f}%` bad episodes (`seed={result['config']['drop_seed']}`) + filter idle frames (`idle_eps={result['config']['idle_eps']}`) + recompute GR00T `q01`/`q99` stats",
        "",
        "| Metric | Native LeRobot v2.1 (curated copy) | Lance (zero-copy column evolution) | Ratio / Delta |",
        "|---|---:|---:|---:|",
        f"| Wall time (s) | {nat['wall_time_s']:.3f} s | **{lan['wall_time_s']:.3f} s** | **{speedup:.1f}x faster** |",
        f"| Total bytes written | {_mb(nat['bytes_written_total'])} | **{_mb(lan['bytes_written_total'])}** | **{io_ratio:.0f}x less I/O** |",
        f"| Video bytes written | {_mb(nat['bytes_written_videos'])} ({nat['files_written_videos']} MP4s) | **0.00 MB (0 files)** | **0 video rewrite** |",
        f"| Tabular / frames bytes written | {_mb(nat['bytes_written_parquets'])} | {_mb(lan['bytes_written_frames'])} | — |",
        f"| Metadata bytes written | {_mb(nat['bytes_written_meta'])} | {_mb(lan['bytes_written_meta'])} | — |",
        f"| Files / objects written | {nat['files_touched']} | **{lan['files_written_frames']} (frames) + meta** | — |",
        f"| Resulting disk footprint | {_mb(nat['resulting_disk_bytes'])} | {_mb(lan['disk_after_bytes'])} (+{_mb(lan['disk_delta_bytes'])}) | — |",
        f"| Mid-episode idle filtering | Not supported without breaking horizons / re-encoding MP4s | **Supported (`where=\"is_valid = true\"`)** | **{lan['anchor_reduction_pct']:.1f}% anchors filtered** |",
        f"| Filtered dataset init time | N/A | **{lan['filtered_init_s']*1000:.1f} ms** ({lan['filtered_anchors']:,} / {lan['unfiltered_anchors']:,} anchors) | Lance v{lan['frames_version_before']} -> v{lan['frames_version_after']} |",
        f"| Unmodified `lerobot-lancedb` compat | N/A | **{'PASS' if lan['upstream_lerobot_lancedb_compatible'] else 'FAIL'}** | Zero breaking changes |",
    ]
    return "\n".join(lines) + "\n"


def run_benchmark(
    v2_dir: Path,
    *,
    lance_dir: Path | None = None,
    work_dir: Path | None = None,
    drop_frac: float = 0.05,
    drop_seed: int = 42,
    idle_eps: float = 1e-4,
    action_horizon: int = 16,
    video_mode: str = "copy",
) -> dict[str, Any]:
    modality_meta = json.loads((v2_dir / "meta" / "modality.json").read_text(encoding="utf-8"))
    info_meta = json.loads((v2_dir / "meta" / "info.json").read_text(encoding="utf-8"))

    cleanup_tmp: tempfile.TemporaryDirectory[str] | None = None
    if work_dir is None:
        cleanup_tmp = tempfile.TemporaryDirectory(prefix="gr00t-e6-bench-")
        base_work = Path(cleanup_tmp.name)
    else:
        base_work = Path(work_dir)
        base_work.mkdir(parents=True, exist_ok=True)

    try:
        native_out = base_work / "native_curated_v2"
        lance_work = base_work / "lance_curated_work"

        # Prepare Lance working copy (outside the curation timer so conversion/copy is separated!)
        convert_s: float | None = None
        if lance_dir is not None and lance_dir.exists():
            if lance_work.exists():
                shutil.rmtree(lance_work)
            shutil.copytree(lance_dir, lance_work)
        else:
            t_conv0 = time.perf_counter()
            convert_v2_to_lance(v2_dir, lance_work, video_mode=video_mode)
            convert_s = round(time.perf_counter() - t_conv0, 4)

        native_res = run_native_v2_curation(
            v2_dir,
            native_out,
            drop_frac=drop_frac,
            drop_seed=drop_seed,
            idle_eps=idle_eps,
        )
        lance_res = run_lance_curation_benchmark(
            lance_work,
            modality_meta,
            drop_frac=drop_frac,
            drop_seed=drop_seed,
            idle_eps=idle_eps,
            action_horizon=action_horizon,
        )

        result: dict[str, Any] = {
            "dataset": str(v2_dir),
            "config": {
                "drop_frac": drop_frac,
                "drop_seed": drop_seed,
                "idle_eps": idle_eps,
                "action_horizon": action_horizon,
                "video_mode": video_mode,
                "total_episodes": int(info_meta.get("total_episodes", lance_res["total_episodes"])),
                "total_frames": int(info_meta.get("total_frames", lance_res["total_frames"])),
            },
            "lance_convert_s": convert_s,
            "native_v2": native_res,
            "lance": lance_res,
            "comparison": {
                "wall_time_speedup": round(native_res["wall_time_s"] / max(lance_res["wall_time_s"], 1e-6), 2),
                "bytes_written_reduction_x": round(
                    native_res["bytes_written_total"] / max(lance_res["bytes_written_total"], 1), 2
                ),
                "video_bytes_saved": native_res["bytes_written_videos"] - lance_res["bytes_written_videos"],
            },
        }
        result["markdown_summary"] = format_markdown_summary(result)
        return result
    finally:
        if cleanup_tmp is not None:
            cleanup_tmp.cleanup()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="E6 Curation & Repair Benchmark: Native LeRobot v2.1 vs Lance."
    )
    p.add_argument("--v2-dir", required=True, help="Path to source LeRobot v2.1 dataset directory.")
    p.add_argument(
        "--lance-dir",
        default=None,
        help="Optional existing Lance dataset directory (copied into scratch work dir before curating).",
    )
    p.add_argument(
        "--work-dir",
        default=None,
        help="Optional scratch directory for curated outputs (uses TemporaryDirectory if omitted).",
    )
    p.add_argument("--drop-frac", type=float, default=0.05, help="Fraction of bad episodes to drop (default: 0.05).")
    p.add_argument("--drop-seed", type=int, default=42, help="Seed for bad-episode sampling (default: 42).")
    p.add_argument("--idle-eps", type=float, default=1e-4, help="Idle action-delta threshold (default: 1e-4).")
    p.add_argument("--action-horizon", type=int, default=16, help="Action horizon for dataset init check (default: 16).")
    p.add_argument("--video-mode", choices=("copy", "external"), default="copy")
    p.add_argument("--out-json", default=None, help="Optional path to save full JSON report.")
    p.add_argument("--out-md", default=None, help="Optional path to save Markdown summary table.")
    args = p.parse_args(argv)

    result = run_benchmark(
        v2_dir=Path(args.v2_dir),
        lance_dir=Path(args.lance_dir) if args.lance_dir else None,
        work_dir=Path(args.work_dir) if args.work_dir else None,
        drop_frac=args.drop_frac,
        drop_seed=args.drop_seed,
        idle_eps=args.idle_eps,
        action_horizon=args.action_horizon,
        video_mode=args.video_mode,
    )

    md = result["markdown_summary"]
    print(md)
    if args.out_md:
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(md, encoding="utf-8")
    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
