# SPDX-License-Identifier: Apache-2.0
"""Comprehensive unit tests for tabular-only Lance curation, SQL predicate pushdown, and E6 benchmark."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import lancedb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# Import gr00t_lancedb before lerobot_lancedb so _hub_compat shims huggingface_hub<1.0 cleanly.
from gr00t_lancedb import (
    CURATION_COLUMNS,
    CURATION_REPORT_FILENAME,
    EmbodimentTag,
    LanceEpisodeLoader,
    LanceShardedSingleStepDataset,
    ModalityConfig,
    ResumableLanceMixtureDataset,
    convert_v2_to_lance,
    curate_lance_dataset,
)
from gr00t_lancedb.curate import main_curate
from lerobot_lancedb import FRAMES_TABLE, LanceDBDataset

from conftest import _make_distinct_rgb_frame, _write_h264_mp4


def _create_curatable_v2_dataset(root: Path, *, fps: int = 10) -> tuple[Path, dict[str, object]]:
    """Build a synthetic v2.1 dataset with known idle blocks, a NaN episode, and normal episodes.

    Episodes (each length 20):
      - ep 0: normal moving actions, EXCEPT frames 5..8 are identical to frame 4 ->
              frame 5 has delta vs 4 (active? wait: action[5]==action[4], so frames 5,6,7,8 have delta==0!)
              Specifically we set action[5] = action[4], action[6] = action[4], action[7] = action[4], action[8] = action[4].
              So idle frames in ep 0 are EXACTLY {5, 6, 7, 8}.
      - ep 1: normal moving actions, EXCEPT frames 0..2 are identical (action[0]==action[1]==action[2]),
              so frame 0 (uses action[1]-action[0]==0), frame 1, and frame 2 are idle -> EXACTLY {0, 1, 2}.
      - ep 2: contains NaN in action at frame 7 -> entire episode is_bad_episode=True, frame 7 has_anomaly=True.
      - ep 3: completely normal moving actions (0 idle, 0 anomalies) — will also be tested with drop_episodes=[3] or kept.
      - ep 4: completely normal moving actions (0 idle, 0 anomalies), with a known spike at frame 10 if spike_threshold tested.
    """
    ep_lengths = [20, 20, 20, 20, 20]
    width, height = 64, 48
    root.mkdir(parents=True, exist_ok=True)
    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        {"task_index": 0, "task": "pick up the red block"},
        {"task_index": 1, "task": "place the block in the bowl"},
    ]
    with open(meta_dir / "tasks.jsonl", "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")

    video_keys = ["observation.images.front_cam", "observation.images.wrist_cam"]
    episodes_records = []
    expected_actions: dict[int, np.ndarray] = {}
    expected_states: dict[int, np.ndarray] = {}
    global_idx = 0

    for ep_idx, length in enumerate(ep_lengths):
        task_idx = ep_idx % len(tasks)
        # Smooth, deterministic, guaranteed non-zero step-to-step deltas (~0.5 per step)
        t_axis = np.arange(length, dtype=np.float32)[:, None]
        dims = np.arange(8, dtype=np.float32)[None, :] + 1.0
        states = (ep_idx * 10.0 + t_axis * 0.25 * (dims / 4.0)).astype(np.float32)
        actions = (ep_idx * 5.0 + t_axis * 0.30 * (dims / 4.0)).astype(np.float32)

        if ep_idx == 0:
            # Mid-episode idle block: frames 5, 6, 7, 8 repeat frame 4 exactly -> delta_norm == 0
            actions[5:9] = actions[4]
        elif ep_idx == 1:
            # Start-of-episode idle block: frames 0, 1, 2 identical -> delta_norm[0..2] == 0
            actions[0:3] = actions[0]
        elif ep_idx == 2:
            # Corrupt episode: NaN at frame 7
            actions[7, 2] = np.nan
        elif ep_idx == 4:
            # Intentional large action spike at frame 10 (delta_norm > 50)
            actions[10] = actions[9] + 30.0
            actions[11:] = actions[10] + (np.arange(1, length - 10, dtype=np.float32)[:, None] * 0.30 * (dims / 4.0))

        expected_states[ep_idx] = states
        expected_actions[ep_idx] = actions

        timestamps = (np.arange(length, dtype=np.float64) / fps).astype(np.float32)
        frame_indices = np.arange(length, dtype=np.int64)
        ep_indices = np.full(length, ep_idx, dtype=np.int64)
        indices = np.arange(global_idx, global_idx + length, dtype=np.int64)
        task_indices = np.full(length, task_idx, dtype=np.int64)
        global_idx += length

        pq_path = root / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
        pq_path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.table(
            {
                "observation.state": pa.array([row.tolist() for row in states], type=pa.list_(pa.float32())),
                "action": pa.array([row.tolist() for row in actions], type=pa.list_(pa.float32())),
                "timestamp": pa.array(timestamps, type=pa.float32()),
                "frame_index": pa.array(frame_indices, type=pa.int64()),
                "episode_index": pa.array(ep_indices, type=pa.int64()),
                "index": pa.array(indices, type=pa.int64()),
                "task_index": pa.array(task_indices, type=pa.int64()),
            }
        )
        pq.write_table(table, pq_path)

        for cam_id, vk in enumerate(video_keys):
            frames = [_make_distinct_rgb_frame(cam_id, ep_idx, f_i, height, width) for f_i in range(length)]
            vid_path = root / "videos" / "chunk-000" / vk / f"episode_{ep_idx:06d}.mp4"
            _write_h264_mp4(vid_path, frames, fps=fps)

        episodes_records.append(
            {"episode_index": ep_idx, "tasks": [tasks[task_idx]["task"]], "length": length}
        )

    with open(meta_dir / "episodes.jsonl", "w", encoding="utf-8") as f:
        for rec in episodes_records:
            f.write(json.dumps(rec) + "\n")

    modality = {
        "state": {
            "arm": {"start": 0, "end": 7, "original_key": "observation.state"},
            "gripper": {"start": 7, "end": 8, "original_key": "observation.state"},
        },
        "action": {
            "arm": {"start": 0, "end": 7, "original_key": "action"},
            "gripper": {"start": 7, "end": 8, "original_key": "action"},
        },
        "video": {
            "front": {"original_key": "observation.images.front_cam"},
            "wrist": {"original_key": "observation.images.wrist_cam"},
        },
    }
    (meta_dir / "modality.json").write_text(json.dumps(modality, indent=2) + "\n", encoding="utf-8")

    info = {
        "codebase_version": "v2.1",
        "robot_type": "franka_curation_test",
        "total_episodes": len(ep_lengths),
        "total_frames": global_idx,
        "total_tasks": len(tasks),
        "total_videos": len(ep_lengths) * len(video_keys),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{len(ep_lengths)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [8], "names": None},
            "action": {"dtype": "float32", "shape": [8], "names": None},
            "observation.images.front_cam": {
                "dtype": "video",
                "shape": [height, width, 3],
                "names": ["height", "width", "channels"],
                "info": {"video.fps": fps, "video.height": height, "video.width": width, "video.codec": "h264"},
            },
            "observation.images.wrist_cam": {
                "dtype": "video",
                "shape": [height, width, 3],
                "names": ["height", "width", "channels"],
                "info": {"video.fps": fps, "video.height": height, "video.width": width, "video.codec": "h264"},
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")

    ground_truth = {
        "ep_lengths": ep_lengths,
        "states": expected_states,
        "actions": expected_actions,
        "idle_ep0": {5, 6, 7, 8},
        "idle_ep1": {0, 1, 2},
        "nan_ep": 2,
        "nan_frame": 7,
        "spike_ep": 4,
        "spike_frame": 10,
    }
    return root, ground_truth


def _snapshot_tree(root: Path) -> dict[str, tuple[int, int]]:
    return {
        f.relative_to(root).as_posix(): (f.stat().st_size, f.stat().st_mtime_ns)
        for f in sorted(root.rglob("*"))
        if f.is_file()
    }


def _identity_collate(items: list) -> list:
    return items


def _modality_configs(action_horizon: int = 4) -> dict[str, ModalityConfig]:
    return {
        "video": ModalityConfig(delta_indices=[-1, 0], modality_keys=["front", "wrist"]),
        "state": ModalityConfig(delta_indices=[0], modality_keys=["arm", "gripper"]),
        "action": ModalityConfig(delta_indices=list(range(action_horizon)), modality_keys=["arm", "gripper"]),
        "language": ModalityConfig(delta_indices=[0], modality_keys=["task"]),
    }


def test_curate_tags_zero_video_touch_stats_and_pushdown(tmp_path: Path) -> None:
    """Verify 100% tagging accuracy, zero videos.lance writes, upstream compat, pushdown, & exact stats."""
    v2_dir, gt = _create_curatable_v2_dataset(tmp_path / "v2_src")
    lance_dir = tmp_path / "lance_ds"
    convert_v2_to_lance(v2_dir, lance_dir, video_mode="copy")

    # Snapshot every file (size + mtime_ns) in videos.lance BEFORE curation
    videos_dir = lance_dir / "videos.lance"
    videos_before = _snapshot_tree(videos_dir)
    assert len(videos_before) > 0

    # Run curation: drop episode 3 explicitly, detect NaN ep 2, detect idle blocks in ep 0 & 1, spike in ep 4
    report = curate_lance_dataset(
        lance_dir,
        idle_eps=1e-4,
        spike_threshold=20.0,
        drop_episodes=[3],
        keep_idle=False,
        recompute_stats=True,
    )

    # 1. videos.lance MUST be 100% UNTOUCHED (zero files added, modified, or rewritten!)
    videos_after = _snapshot_tree(videos_dir)
    assert videos_after == videos_before
    assert report.bytes_written_videos == 0
    assert report.files_written_videos == 0
    assert report.frames_table_version > report.frames_table_version_before

    # 2. Verify exact report counts
    assert report.total_episodes == 5
    assert report.bad_episode_indices == [2, 3]
    assert report.bad_episodes == 2
    assert report.valid_episodes == 3
    assert report.total_frames == 100
    assert report.idle_frames == 4 + 3  # ep0 {5,6,7,8} + ep1 {0,1,2}
    assert report.anomaly_frames == 2  # ep2 frame 7 (NaN) + ep4 frame 10 (spike)
    assert report.bad_episode_frames == 40  # ep2 (20) + ep3 (20)
    # Valid frames: 100 - 40 (bad eps) - 7 (idle in ep0/ep1) - 1 (spike in ep4) = 52
    assert report.valid_frames == 52
    assert (lance_dir / "meta" / CURATION_REPORT_FILENAME).exists()

    # 3. Verify every single per-frame tag in frames.lance matches ground truth 100%
    db = lancedb.connect(str(lance_dir))
    ft = db.open_table(FRAMES_TABLE)
    for col in CURATION_COLUMNS:
        assert col in ft.schema.names
    tbl = ft.search().select(["episode_index", "frame_index", *CURATION_COLUMNS]).limit(None).to_arrow()
    rows = tbl.to_pylist()
    assert len(rows) == 100

    valid_state_rows = []
    valid_action_rows = []
    for row in rows:
        ep = int(row["episode_index"])
        fr = int(row["frame_index"])

        exp_idle = (ep == 0 and fr in gt["idle_ep0"]) or (ep == 1 and fr in gt["idle_ep1"])
        exp_anom = (ep == 2 and fr == 7) or (ep == 4 and fr == 10)
        exp_bad = ep in (2, 3)
        exp_valid = (not exp_bad) and (not exp_anom) and (not exp_idle)

        assert row["is_idle"] is exp_idle, f"is_idle mismatch at ep={ep} frame={fr}"
        assert row["has_anomaly"] is exp_anom, f"has_anomaly mismatch at ep={ep} frame={fr}"
        assert row["is_bad_episode"] is exp_bad, f"is_bad_episode mismatch at ep={ep} frame={fr}"
        assert row["is_valid"] is exp_valid, f"is_valid mismatch at ep={ep} frame={fr}"

        if exp_valid:
            valid_state_rows.append(gt["states"][ep][fr])
            valid_action_rows.append(gt["actions"][ep][fr])

    # 4. Recomputed gr00t_stats.json (mean/std/min/max/q01/q99) matches direct numpy over valid frames to < 1e-6!
    gr00t_stats = json.loads((lance_dir / "meta" / "gr00t_stats.json").read_text(encoding="utf-8"))
    exp_s = np.asarray(valid_state_rows, dtype=np.float64)
    exp_a = np.asarray(valid_action_rows, dtype=np.float64)
    for key, mat in (("observation.state", exp_s), ("action", exp_a)):
        np.testing.assert_allclose(gr00t_stats[key]["mean"], np.mean(mat, axis=0), rtol=0, atol=1e-6)
        np.testing.assert_allclose(gr00t_stats[key]["std"], np.std(mat, axis=0), rtol=0, atol=1e-6)
        np.testing.assert_allclose(gr00t_stats[key]["min"], np.min(mat, axis=0), rtol=0, atol=1e-6)
        np.testing.assert_allclose(gr00t_stats[key]["max"], np.max(mat, axis=0), rtol=0, atol=1e-6)
        np.testing.assert_allclose(gr00t_stats[key]["q01"], np.quantile(mat, 0.01, axis=0), rtol=0, atol=1e-6)
        np.testing.assert_allclose(gr00t_stats[key]["q99"], np.quantile(mat, 0.99, axis=0), rtol=0, atol=1e-6)

    # 5. Unmodified lerobot_lancedb.LanceDBDataset MUST open and decode frames cleanly after curation!
    upstream_ds = LanceDBDataset(root=lance_dir, return_uint8=True)
    assert upstream_ds.num_episodes == 5
    assert upstream_ds.num_frames == 100
    u_sample = upstream_ds[0]
    assert "observation.images.front_cam" in u_sample
    assert u_sample["observation.images.front_cam"].shape == (3, 48, 64)

    # 6. LanceShardedSingleStepDataset with SQL pushdown where="is_valid = true"
    action_horizon = 4
    mod_cfg = _modality_configs(action_horizon=action_horizon)
    sharded = LanceShardedSingleStepDataset(
        dataset_path=lance_dir,
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        modality_configs=mod_cfg,
        shard_size=16,
        episode_sampling_rate=0.5,
        allow_padding=True,
        where="is_valid = true",
    )

    # Collect EVERY anchor emitted across all shards
    shard_anchors: list[tuple[int, int]] = []
    for s_idx in range(len(sharded)):
        for ep_idx, step_arr in sharded.get_shard_splits(s_idx):
            for st in step_arr:
                shard_anchors.append((int(ep_idx), int(st)))

    # Expected anchors: frame_index in [0, 20 - 4 + 1 = 17) intersected with is_valid==True
    expected_anchors = set()
    for ep in (0, 1, 4):
        for fr in range(20 - action_horizon + 1):
            if ep == 0 and fr in gt["idle_ep0"]:
                continue
            if ep == 1 and fr in gt["idle_ep1"]:
                continue
            if ep == 4 and fr == 10:
                continue
            expected_anchors.add((ep, fr))

    assert len(shard_anchors) == len(expected_anchors)
    assert set(shard_anchors) == expected_anchors
    assert set(sharded.get_all_valid_anchors()) == expected_anchors
    # Zero bad-episode anchors, zero idle anchors, zero spike anchors
    assert all(ep not in (2, 3) for ep, _ in shard_anchors)
    assert all(not (ep == 0 and fr in gt["idle_ep0"]) for ep, fr in shard_anchors)
    assert all(not (ep == 1 and fr in gt["idle_ep1"]) for ep, fr in shard_anchors)
    assert (4, 10) not in expected_anchors

    # CRITICAL CONTINUITY CHECK: anchor (ep=0, step=4) is valid, and its 4-step action horizon
    # [4, 5, 6, 7] MUST cross seamlessly into the intact neighboring frames (5, 6, 7) without skipping!
    step_ep0_4 = sharded.episode_loader.load_steps(0, [4], EmbodimentTag.NEW_EMBODIMENT, allow_padding=True)[0]
    expected_arm_horizon = gt["actions"][0][4:8, 0:7]
    np.testing.assert_allclose(step_ep0_4.actions["arm"], expected_arm_horizon, rtol=0, atol=1e-6)
    # And video delta_indices=[-1, 0] for anchor (ep=1, step=3) fetches intact frame 2 & frame 3!
    step_ep1_3 = sharded.episode_loader.load_steps(1, [3], EmbodimentTag.NEW_EMBODIMENT, allow_padding=True)[0]
    assert len(step_ep1_3.images["front"]) == 2
    assert step_ep1_3.images["front"][0].shape == (48, 64, 3)

    sharded.close()

    # 7. ResumableLanceMixtureDataset also respects where="is_valid = true" cleanly (both num_workers=0 and 2)!
    for nw in (0, 2):
        ds_mix = LanceShardedSingleStepDataset(
            dataset_path=lance_dir,
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
            modality_configs=mod_cfg,
            shard_size=16,
            episode_sampling_rate=0.5,
            allow_padding=True,
            where="is_valid = true",
        )
        mix = ResumableLanceMixtureDataset(
            datasets=[ds_mix],
            weights=[1.0],
            batch_size=4,
            episode_pool_size=2,
            prefetch_anchors=8,
            infinite=False,
        )
        assert len(mix) == len(expected_anchors)
        mix_anchors = []
        for batch in mix.loader(num_workers=nw, collate_fn=_identity_collate):
            for item in batch:
                mix_anchors.append((item.metadata["episode_index"], item.metadata["step_index"]))
        assert len(mix_anchors) == len(expected_anchors)
        assert set(mix_anchors) == expected_anchors
        mix.close()


def test_curation_length_one_episode_and_deterministic_random_drop(tmp_path: Path) -> None:
    """Verify length-1 episode gets delta_norm=0.0 and drop_random_frac is deterministic."""
    from conftest import create_synthetic_gr00t_v2_dataset

    v2_dir = create_synthetic_gr00t_v2_dataset(tmp_path / "v2_len1", ep_lengths=[1, 10, 12, 14], seed=55)
    lance_dir = tmp_path / "lance_len1"
    convert_v2_to_lance(v2_dir, lance_dir, video_mode="copy")

    r1 = curate_lance_dataset(lance_dir, idle_eps=1e-4, drop_random_frac=0.25, drop_seed=99)
    r2 = curate_lance_dataset(lance_dir, idle_eps=1e-4, drop_random_frac=0.25, drop_seed=99)
    assert r1.bad_episode_indices == r2.bad_episode_indices
    assert len(r1.bad_episode_indices) == 1

    db = lancedb.connect(str(lance_dir))
    ft = db.open_table(FRAMES_TABLE)
    ep0_rows = ft.search().where("episode_index = 0").select(["action_delta_norm", "state_delta_norm", "is_idle"]).limit(None).to_arrow().to_pylist()
    assert len(ep0_rows) == 1
    assert ep0_rows[0]["action_delta_norm"] == 0.0
    assert ep0_rows[0]["state_delta_norm"] == 0.0
    assert ep0_rows[0]["is_idle"] is True


def test_curation_idempotent_re_run_keep_idle_and_cli(tmp_path: Path) -> None:
    """Re-running curation with different params (and via CLI) replaces tags cleanly and preserves time-travel."""
    v2_dir, _ = _create_curatable_v2_dataset(tmp_path / "v2_src")
    lance_dir = tmp_path / "lance_ds"
    convert_v2_to_lance(v2_dir, lance_dir, video_mode="copy")

    r1 = curate_lance_dataset(lance_dir, idle_eps=1e-4, drop_random_frac=0.2, drop_seed=7, keep_idle=False)
    v1 = r1.frames_table_version
    assert r1.bad_episodes >= 1  # NaN ep 2 + deterministic 20% drop

    # Re-run via CLI with --keep-idle and --out-json
    out_json = tmp_path / "cli_report.json"
    main_curate(
        [
            "--root",
            str(lance_dir),
            "--idle-eps",
            "1e-4",
            "--keep-idle",
            "--out-json",
            str(out_json),
        ]
    )
    r2 = json.loads(out_json.read_text(encoding="utf-8"))
    assert r2["frames_table_version"] > v1
    assert r2["keep_idle"] is True
    # Only NaN ep 2 is bad (20 frames); idle frames are kept valid -> 80 valid frames!
    assert r2["bad_episode_indices"] == [2]
    assert r2["valid_frames"] == 80
    assert r2["bytes_written_videos"] == 0

    # Time-travel check: opening pinned lance_version=v1 sees r1's filter count, latest sees r2's!
    mod_cfg = _modality_configs(action_horizon=4)
    loader_v1 = LanceEpisodeLoader(lance_dir, mod_cfg, where="is_valid = true", lance_version=v1)
    loader_v2 = LanceEpisodeLoader(lance_dir, mod_cfg, where="is_valid = true")
    total_v1 = sum(len(loader_v1.get_valid_steps_for_episode(i)) for i in range(len(loader_v1)))
    total_v2 = sum(len(loader_v2.get_valid_steps_for_episode(i)) for i in range(len(loader_v2)))
    assert total_v1 == r1.valid_frames
    assert total_v2 == 80
    loader_v1.close()
    loader_v2.close()


def test_bench_curation_e6_script_end_to_end(tmp_path: Path) -> None:
    """Verify scripts/bench_curation_e6.py runs end-to-end and outputs a valid JSON + markdown report."""
    v2_dir, _ = _create_curatable_v2_dataset(tmp_path / "v2_e6")
    out_json = tmp_path / "e6_report.json"
    out_md = tmp_path / "e6_report.md"

    script_path = Path(__file__).resolve().parent.parent / "scripts" / "bench_curation_e6.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--v2-dir",
            str(v2_dir),
            "--drop-frac",
            "0.2",
            "--drop-seed",
            "42",
            "--idle-eps",
            "1e-4",
            "--action-horizon",
            "4",
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "E6 Curation & Repair Benchmark" in proc.stdout
    assert out_json.exists() and out_md.exists()

    rep = json.loads(out_json.read_text(encoding="utf-8"))
    assert rep["lance"]["bytes_written_videos"] == 0
    assert rep["lance"]["files_written_videos"] == 0
    assert rep["lance"]["upstream_lerobot_lancedb_compatible"] is True
    assert rep["lance"]["filtered_anchors"] < rep["lance"]["unfiltered_anchors"]
    assert rep["native_v2"]["bytes_written_videos"] > 0
    assert rep["native_v2"]["mid_episode_idle_filtering_supported"] is False
    assert rep["lance"]["mid_episode_idle_filtering_supported"] is True
