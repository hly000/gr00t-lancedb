# SPDX-License-Identifier: Apache-2.0
"""Parity tests: verify gr00t-lancedb matches NVIDIA Isaac-GR00T bit-for-bit on states, actions, images, text, and stats."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from lerobot_lancedb import LanceDBDataset

from gr00t_lancedb import (
    EmbodimentTag,
    LanceEpisodeLoader,
    LanceShardedSingleStepDataset,
    ModalityConfig,
    attach_gr00t_meta,
    convert_v2_to_lance,
)
from conftest import reference_load_v2_step


def _assert_step_parity(actual, expected) -> None:
    assert actual.text == expected.text
    assert actual.embodiment == expected.embodiment
    assert set(actual.states.keys()) == set(expected.states.keys())
    for k in expected.states:
        assert actual.states[k].dtype == np.float32
        np.testing.assert_allclose(actual.states[k], expected.states[k], rtol=0, atol=1e-6)

    assert set(actual.actions.keys()) == set(expected.actions.keys())
    for k in expected.actions:
        assert actual.actions[k].dtype == np.float32
        np.testing.assert_allclose(actual.actions[k], expected.actions[k], rtol=0, atol=1e-6)

    assert set(actual.images.keys()) == set(expected.images.keys())
    for k in expected.images:
        assert len(actual.images[k]) == len(expected.images[k])
        for img_a, img_e in zip(actual.images[k], expected.images[k], strict=True):
            assert img_a.dtype == np.uint8
            assert img_a.shape == img_e.shape
            np.testing.assert_array_equal(img_a, img_e)


@pytest.mark.parametrize("video_mode", ["copy", "external"])
def test_v2_to_lance_exact_parity(
    v2_dataset_dir: Path,
    default_modality_configs: dict[str, ModalityConfig],
    tmp_path: Path,
    video_mode: str,
) -> None:
    out_dir = tmp_path / f"lance_{video_mode}"
    convert_v2_to_lance(v2_dataset_dir, out_dir, video_mode=video_mode)

    # 1. Unmodified lerobot-lancedb MUST open the converted layout cleanly!
    upstream_ds = LanceDBDataset(root=out_dir, return_uint8=True)
    assert upstream_ds.num_episodes == 4
    assert upstream_ds.num_frames == 18 + 22 + 16 + 20
    sample0 = upstream_ds[0]
    assert "observation.images.front_cam" in sample0
    assert sample0["observation.images.front_cam"].dtype.is_floating_point is False

    # 2. LanceEpisodeLoader exact frame-by-frame parity (including allow_padding & negative delta_indices)
    loader = LanceEpisodeLoader(out_dir, default_modality_configs)
    assert len(loader) == 4
    assert loader.get_episode_lengths() == [18, 22, 16, 20]

    for ep_idx in range(4):
        for step_idx in (0, 1, 5, loader.get_episode_length(ep_idx) - 4):
            actual_steps = loader.load_steps(
                ep_idx, [step_idx], EmbodimentTag.NEW_EMBODIMENT, allow_padding=True
            )
            expected_step = reference_load_v2_step(
                v2_dataset_dir, ep_idx, step_idx, default_modality_configs, allow_padding=True
            )
            _assert_step_parity(actual_steps[0], expected_step)

    # 3. Verify load_episode DataFrame interface & sparse select_frames parity
    sparse_df = loader.load_episode(1, select_frames=lambda n: np.array([0, 4, n - 1]))
    assert len(sparse_df) == 22
    assert sparse_df["video.front"].iloc[0] is not None
    assert sparse_df["video.front"].iloc[1] is None
    assert sparse_df["video.front"].iloc[4] is not None

    # 4. Verify statistics (including q01/q99 and relative_action with __fingerprints__ stripped)
    stats = loader.get_dataset_statistics()
    assert set(stats["state"].keys()) == {"arm", "gripper"}
    assert len(stats["state"]["arm"]["q01"]) == 7
    assert len(stats["state"]["gripper"]["q99"]) == 1
    assert "relative_action" in stats
    assert "__fingerprints__" not in stats["relative_action"]
    assert "arm" in stats["relative_action"]

    # 5. Verify initial_actions loaded from meta.lance
    init_actions = loader.get_initial_actions()
    assert len(init_actions) == 1

    # 6. Verify LanceShardedSingleStepDataset shard & split contract
    sharded = LanceShardedSingleStepDataset(
        dataset_path=out_dir,
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        modality_configs=default_modality_configs,
        shard_size=16,
        episode_sampling_rate=0.5,
        allow_padding=True,
    )
    assert len(sharded) >= 1
    shard0 = sharded.get_shard(0)
    assert len(shard0) == sharded.get_shard_length(0)
    loader.close()
    sharded.close()


def test_video_key_automap_and_language_modes(v2_dataset_dir: Path, tmp_path: Path) -> None:
    """Verify N1.6 vs N1.7 positional video key auto-mapping + sub_task + annotation.* modes."""
    out_dir = tmp_path / "lance_copy"
    convert_v2_to_lance(v2_dataset_dir, out_dir, video_mode="copy")

    # Positional auto-map: model asks for ["cam_head", "cam_hand"] vs dataset ["front", "wrist"]
    remapped_cfg = {
        "video": ModalityConfig(delta_indices=[0], modality_keys=["cam_head", "cam_hand"]),
        "state": ModalityConfig(delta_indices=[0], modality_keys=["arm"]),
        "action": ModalityConfig(delta_indices=[0, 1, 2, 3], modality_keys=["arm"]),
        "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.task_desc"]),
    }
    loader = LanceEpisodeLoader(out_dir, remapped_cfg)
    step = loader.load_steps(1, [3], EmbodimentTag.NEW_EMBODIMENT, allow_padding=True)[0]
    ref = reference_load_v2_step(v2_dataset_dir, 1, 3, remapped_cfg, allow_padding=True)
    _assert_step_parity(step, ref)
    assert step.text == "place the block in the bowl"
    loader.close()

    # sub_task language mode from episodes.jsonl
    subtask_cfg = {
        "video": ModalityConfig(delta_indices=[0], modality_keys=["front"]),
        "state": ModalityConfig(delta_indices=[0], modality_keys=["arm"]),
        "action": ModalityConfig(delta_indices=[0, 1, 2, 3], modality_keys=["arm"]),
        "language": ModalityConfig(delta_indices=[0], modality_keys=["sub_task"]),
    }
    st_loader = LanceEpisodeLoader(out_dir, subtask_cfg)
    steps = st_loader.load_steps(0, [0, 14], EmbodimentTag.NEW_EMBODIMENT, allow_padding=True)
    assert steps[0].text == "reach ep 0"
    assert steps[1].text == "grasp ep 0"
    st_loader.close()


def test_attach_gr00t_meta_on_pure_v3_lance_no_downgrade(
    v2_dataset_dir: Path,
    default_modality_configs: dict[str, ModalityConfig],
    tmp_path: Path,
) -> None:
    """Simulate starting from an upstream LeRobot v3 Lance dataset (no modality.json / episodes.jsonl / q01/q99)."""
    v3_lance_dir = tmp_path / "pure_v3_lance"
    convert_v2_to_lance(v2_dataset_dir, v3_lance_dir, video_mode="copy")

    # Strip GR00T-only files from local meta/ & meta.lance to simulate a raw lerobot-lance-convert output.
    for fname in ("modality.json", "episodes.jsonl", "tasks.jsonl", "gr00t_stats.json", "relative_stats.json"):
        p = v3_lance_dir / "meta" / fname
        if p.exists():
            p.unlink()

    # Now attach modality.json + compute q01/q99 directly from frames.lance!
    attach_gr00t_meta(
        lance_root=v3_lance_dir,
        modality_json_path=v2_dataset_dir / "meta" / "modality.json",
        compute_stats=True,
    )

    loader = LanceEpisodeLoader(v3_lance_dir, default_modality_configs)
    actual = loader.load_steps(2, [4], EmbodimentTag.NEW_EMBODIMENT, allow_padding=True)[0]
    expected = reference_load_v2_step(v2_dataset_dir, 2, 4, default_modality_configs, allow_padding=True)
    _assert_step_parity(actual, expected)

    # Computed q01 / q99 must match original stats.json closely!
    orig_stats = json.loads((v2_dataset_dir / "meta" / "stats.json").read_text())
    computed_stats = loader.get_dataset_statistics()
    np.testing.assert_allclose(
        computed_stats["state"]["arm"]["q01"],
        orig_stats["observation.state"]["q01"][0:7],
        rtol=0,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        computed_stats["action"]["gripper"]["q99"],
        orig_stats["action"]["q99"][7:8],
        rtol=0,
        atol=1e-5,
    )
    loader.close()
