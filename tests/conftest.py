# SPDX-License-Identifier: Apache-2.0
"""Shared test fixtures: synthetic GR00T v2.1 datasets with real H.264 MP4s & local moto S3."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import av
import boto3
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from torchcodec.decoders import VideoDecoder

from gr00t_lancedb import ModalityConfig, VLAStepData, EmbodimentTag


def _make_distinct_rgb_frame(cam_id: int, ep_idx: int, frame_idx: int, height: int, width: int) -> np.ndarray:
    """Create a smooth block-patterned RGB uint8 frame uniquely identifying (cam, ep, frame).

    Uses large spatial blocks so H.264 compression preserves unique mean RGB signatures cleanly.
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    r = int((40 + cam_id * 90 + ep_idx * 35) % 240)
    g = int((30 + frame_idx * 7) % 240)
    b = int((200 - frame_idx * 5 + cam_id * 40) % 240)
    frame[:, :, 0] = r
    frame[:, :, 1] = g
    frame[:, :, 2] = b
    # Put a crisp moving bright bar in the top half so each frame index is unmistakable.
    bar_x = (frame_idx * 3) % max(1, width - 8)
    frame[4:16, bar_x : bar_x + 8, :] = 240
    return frame


def _write_h264_mp4(path: Path, frames: list[np.ndarray], fps: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    container = av.open(str(path), mode="w")
    try:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {"g": "5", "bf": "0", "crf": "18"}
        for arr in frames:
            vf = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for pkt in stream.encode(vf):
                container.mux(pkt)
        for pkt in stream.encode():
            container.mux(pkt)
    finally:
        container.close()


def create_synthetic_gr00t_v2_dataset(
    root: Path,
    *,
    ep_lengths: list[int] | None = None,
    fps: int = 10,
    width: int = 64,
    height: int = 48,
    seed: int = 123,
) -> Path:
    """Create a complete, valid GR00T LeRobot v2.1 dataset with real H.264 MP4s."""
    ep_lengths = ep_lengths or [18, 22, 16, 20]
    rng = np.random.default_rng(seed)
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
    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    episodes_records: list[dict[str, Any]] = []

    global_idx = 0
    for ep_idx, length in enumerate(ep_lengths):
        task_idx = ep_idx % len(tasks)
        task_str = tasks[task_idx]["task"]
        states = rng.normal(loc=ep_idx * 0.5, scale=0.25, size=(length, 8)).astype(np.float32)
        actions = rng.normal(loc=-ep_idx * 0.25, scale=0.5, size=(length, 8)).astype(np.float32)
        all_states.append(states)
        all_actions.append(actions)

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
                "annotation.human.task_idx": pa.array(task_indices, type=pa.int64()),
            }
        )
        pq.write_table(table, pq_path)

        for cam_id, vk in enumerate(video_keys):
            frames = [_make_distinct_rgb_frame(cam_id, ep_idx, f_i, height, width) for f_i in range(length)]
            vid_path = root / "videos" / "chunk-000" / vk / f"episode_{ep_idx:06d}.mp4"
            _write_h264_mp4(vid_path, frames, fps=fps)

        episodes_records.append(
            {
                "episode_index": ep_idx,
                "tasks": [task_str],
                "length": length,
                "sub_tasks": [
                    {"start": 0, "end": length // 2, "text": f"reach ep {ep_idx}"},
                    {"start": length // 2, "end": length, "text": f"grasp ep {ep_idx}"},
                ],
            }
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
        "annotation": {
            "human.task_desc": {"original_key": "annotation.human.task_idx"},
        },
    }
    (meta_dir / "modality.json").write_text(json.dumps(modality, indent=2) + "\n", encoding="utf-8")

    cat_s = np.concatenate(all_states, axis=0).astype(np.float64)
    cat_a = np.concatenate(all_actions, axis=0).astype(np.float64)
    stats = {
        "observation.state": {
            "mean": np.mean(cat_s, axis=0).tolist(),
            "std": np.std(cat_s, axis=0).tolist(),
            "min": np.min(cat_s, axis=0).tolist(),
            "max": np.max(cat_s, axis=0).tolist(),
            "q01": np.quantile(cat_s, 0.01, axis=0).tolist(),
            "q99": np.quantile(cat_s, 0.99, axis=0).tolist(),
        },
        "action": {
            "mean": np.mean(cat_a, axis=0).tolist(),
            "std": np.std(cat_a, axis=0).tolist(),
            "min": np.min(cat_a, axis=0).tolist(),
            "max": np.max(cat_a, axis=0).tolist(),
            "q01": np.quantile(cat_a, 0.01, axis=0).tolist(),
            "q99": np.quantile(cat_a, 0.99, axis=0).tolist(),
        },
    }
    (meta_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    rel_stats = {
        "__fingerprints__": {"arm": "abc123"},
        "arm": {
            "mean": [0.0] * 7,
            "std": [1.0] * 7,
            "min": [-1.0] * 7,
            "max": [1.0] * 7,
            "q01": [-0.9] * 7,
            "q99": [0.9] * 7,
        },
    }
    (meta_dir / "relative_stats.json").write_text(json.dumps(rel_stats, indent=2) + "\n", encoding="utf-8")

    np.savez(meta_dir / "initial_actions.npz", **{"new_embodiment/arm": np.zeros((1, 7), dtype=np.float32)})

    info = {
        "codebase_version": "v2.1",
        "robot_type": "franka_test",
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
            "annotation.human.task_idx": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return root


def reference_load_v2_step(
    v2_root: Path,
    ep_idx: int,
    step_idx: int,
    modality_configs: dict[str, ModalityConfig],
    embodiment_tag: EmbodimentTag = EmbodimentTag.NEW_EMBODIMENT,
    allow_padding: bool = False,
) -> VLAStepData:
    """Exact reference implementation matching NVIDIA Isaac-GR00T ``LeRobotEpisodeLoader``."""
    meta = v2_root / "meta"
    modality_meta = json.loads((meta / "modality.json").read_text())
    tasks_map = {
        int(r["task_index"]): r["task"]
        for r in (json.loads(line) for line in (meta / "tasks.jsonl").read_text().splitlines() if line)
    }
    ep_meta = [
        json.loads(line) for line in (meta / "episodes.jsonl").read_text().splitlines() if line
    ][ep_idx]
    df = pd.read_parquet(v2_root / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet")
    length = min(len(df), int(ep_meta["length"]))

    def _idxs(deltas: list[int]) -> list[int]:
        raw = [step_idx + d for d in deltas]
        if allow_padding:
            return [max(0, min(i, length - 1)) for i in raw]
        return [i + length if i < 0 else i for i in raw]

    states: dict[str, np.ndarray] = {}
    if "state" in modality_configs:
        rows = _idxs(modality_configs["state"].delta_indices)
        for k in modality_configs["state"].modality_keys:
            gm = modality_configs["state"]
            info = modality_meta["state"][k]
            orig = info.get("original_key", "observation.state")
            states[k] = np.vstack(
                [np.asarray(df[orig].iloc[r])[info["start"] : info["end"]].astype(np.float32) for r in rows]
            )

    actions: dict[str, np.ndarray] = {}
    if "action" in modality_configs:
        rows = _idxs(modality_configs["action"].delta_indices)
        for k in modality_configs["action"].modality_keys:
            info = modality_meta["action"][k]
            orig = info.get("original_key", "action")
            actions[k] = np.vstack(
                [np.asarray(df[orig].iloc[r])[info["start"] : info["end"]].astype(np.float32) for r in rows]
            )

    images: dict[str, list[np.ndarray]] = {}
    if "video" in modality_configs:
        rows = _idxs(modality_configs["video"].delta_indices)
        cfg_keys = modality_configs["video"].modality_keys
        meta_keys = list(modality_meta["video"].keys())
        mapping = {}
        if any(k not in modality_meta["video"] for k in cfg_keys):
            mapping = dict(zip(cfg_keys, meta_keys))
        for k in cfg_keys:
            mk = mapping.get(k, k)
            orig = modality_meta["video"][mk].get("original_key", f"observation.images.{mk}")
            vid_path = v2_root / "videos" / "chunk-000" / orig / f"episode_{ep_idx:06d}.mp4"
            dec = VideoDecoder(str(vid_path), device="cpu", dimension_order="NHWC")
            frames = dec.get_frames_at(indices=rows).data.numpy()
            images[k] = [frames[i] for i in range(len(rows))]

    text: str | None = None
    if "language" in modality_configs:
        lk = modality_configs["language"].modality_keys[0]
        if lk == "task":
            text = ep_meta["tasks"][0]
        elif lk.startswith("annotation."):
            sub = lk.replace("annotation.", "", 1)
            orig = modality_meta["annotation"][sub].get("original_key", lk)
            text = tasks_map[int(df[orig].iloc[step_idx])]

    return VLAStepData(
        images=images,
        states=states,
        actions=actions,
        masks=None,
        text=text,
        embodiment=embodiment_tag,
    )


@pytest.fixture(scope="session")
def v2_dataset_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("gr00t_v2_dataset")
    return create_synthetic_gr00t_v2_dataset(root, ep_lengths=[18, 22, 16, 20], seed=101)


@pytest.fixture(scope="session")
def v2_dataset_dir_b(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("gr00t_v2_dataset_b")
    return create_synthetic_gr00t_v2_dataset(root, ep_lengths=[15, 19, 17], seed=202)


@pytest.fixture(scope="session")
def default_modality_configs() -> dict[str, ModalityConfig]:
    return {
        "video": ModalityConfig(delta_indices=[-1, 0], modality_keys=["front", "wrist"]),
        "state": ModalityConfig(delta_indices=[0], modality_keys=["arm", "gripper"]),
        "action": ModalityConfig(delta_indices=list(range(4)), modality_keys=["arm", "gripper"]),
        "language": ModalityConfig(delta_indices=[0], modality_keys=["task"]),
    }


@pytest.fixture(scope="session")
def moto_s3_server():
    """Spin up a real local S3-compatible HTTP server for zero-mock S3 streaming tests."""
    from moto.server import ThreadedMotoServer

    server = ThreadedMotoServer(ip_address="127.0.0.1", port=0)
    server.start()
    host, port = server.get_host_and_port()
    endpoint_url = f"http://{host}:{port}"

    old_env = {k: os.environ.get(k) for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION")}
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
        region_name="us-east-1",
    )
    bucket = "gr00t-test"
    client.create_bucket(Bucket=bucket)

    try:
        yield {
            "endpoint_url": endpoint_url,
            "bucket": bucket,
            "client": client,
            "storage_options": {
                "aws_endpoint": endpoint_url,
                "aws_access_key_id": "testing",
                "aws_secret_access_key": "testing",
                "aws_region": "us-east-1",
                "aws_allow_http": "true",
            },
        }
    finally:
        server.stop()
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
