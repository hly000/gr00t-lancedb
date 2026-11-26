# SPDX-License-Identifier: Apache-2.0
"""Tests for ResumableLanceMixtureDataset: multi-dataset mixing, exact multi-worker resume, DDP disjointness."""

from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import pytest

from gr00t_lancedb import (
    BaseProcessor,
    EmbodimentTag,
    LanceShardedSingleStepDataset,
    ModalityConfig,
    ResumableLanceMixtureDataset,
    convert_v2_to_lance,
    load_dataloader_state,
    save_dataloader_state,
)
from gr00t_lancedb.bench import audit_sample_stream, simulate_groot_reseed_coverage


class _DeterministicAugProcessor(BaseProcessor):
    """Lightweight processor that draws random numbers to verify per-anchor RNG determinism across resume."""

    def __init__(self) -> None:
        self.statistics = {}

    def set_statistics(self, statistics, override: bool = False) -> None:
        self.statistics = statistics

    def __call__(self, messages):
        step = messages[0]["content"]
        # Draw a stochastic augmentation value; per-anchor seeding must reproduce this identically!
        aug_noise = float(random.random() + np.random.random())
        return {
            "id": (
                step.metadata["dataset_index"],
                step.metadata["episode_index"],
                step.metadata["step_index"],
                step.metadata["epoch"],
                step.metadata["global_pos"],
            ),
            "aug_noise": aug_noise,
            "state_arm": step.states["arm"].copy(),
            "action_arm": step.actions["arm"].copy(),
            "img_mean": float(step.images["front"][0].mean()),
        }


def _identity_collate(items: list[dict]) -> list[dict]:
    return items


@pytest.mark.parametrize("num_workers", [0, 2])
def test_exact_multi_kill_resume_parity(
    v2_dataset_dir: Path,
    v2_dataset_dir_b: Path,
    default_modality_configs: dict[str, ModalityConfig],
    tmp_path: Path,
    num_workers: int,
) -> None:
    """Prove 0 duplicates, 0 misses, and IDENTICAL batch order + stochastic augs across two kills."""
    lance_a = tmp_path / f"lance_a_w{num_workers}"
    lance_b = tmp_path / f"lance_b_w{num_workers}"
    convert_v2_to_lance(v2_dataset_dir, lance_a, video_mode="copy")
    convert_v2_to_lance(v2_dataset_dir_b, lance_b, video_mode="copy")

    def _make_mixture() -> ResumableLanceMixtureDataset:
        ds_a = LanceShardedSingleStepDataset(
            lance_a,
            EmbodimentTag.NEW_EMBODIMENT,
            default_modality_configs,
            allow_padding=True,
        )
        ds_b = LanceShardedSingleStepDataset(
            lance_b,
            EmbodimentTag.NEW_EMBODIMENT,
            default_modality_configs,
            allow_padding=True,
        )
        return ResumableLanceMixtureDataset(
            datasets=[ds_a, ds_b],
            weights=[0.7, 0.3],
            processor=_DeterministicAugProcessor(),
            seed=42,
            batch_size=4,
            epoch_samples=48,
            episode_pool_size=2,
            prefetch_anchors=6,
            infinite=False,
        )

    # 1. Uninterrupted reference run
    ref_mix = _make_mixture()
    assert "new_embodiment" in ref_mix.get_dataset_statistics()
    assert "arm" in ref_mix.get_dataset_statistics()["new_embodiment"]["state"]
    ref_samples = [
        sample
        for batch in ref_mix.loader(num_workers=num_workers, collate_fn=_identity_collate)
        for sample in batch
    ]
    ref_mix.close()
    assert len(ref_samples) == 48

    # 2. Preempted run: stop at batch 3 (12 samples), resume, stop at batch 7 (28 samples), resume to end!
    ckpt_dir = tmp_path / f"ckpts_w{num_workers}"
    observed: list[dict] = []

    # Segment 1: batches 0..2 (3 batches = 12 samples)
    m1 = _make_mixture()
    for b_idx, batch in enumerate(m1.loader(num_workers=num_workers, collate_fn=_identity_collate)):
        observed.extend(batch)
        if b_idx == 2:
            save_dataloader_state(m1, ckpt_dir / "checkpoint-3")
            break
    m1.close()

    # Segment 2: resume from checkpoint-3, consume 4 more batches (16 samples -> 28 total)
    m2 = _make_mixture()
    load_dataloader_state(m2, ckpt_dir / "checkpoint-3")
    for b_idx, batch in enumerate(m2.loader(num_workers=num_workers, collate_fn=_identity_collate)):
        observed.extend(batch)
        if b_idx == 3:
            save_dataloader_state(m2, ckpt_dir / "checkpoint-7")
            break
    m2.close()

    # Segment 3: resume from checkpoint-7, finish the remaining 20 samples
    m3 = _make_mixture()
    load_dataloader_state(m3, ckpt_dir / "checkpoint-7")
    for batch in m3.loader(num_workers=num_workers, collate_fn=_identity_collate):
        observed.extend(batch)
    m3.close()

    ref_ids = [s["id"] for s in ref_samples]
    obs_ids = [s["id"] for s in observed]
    audit = audit_sample_stream(ref_ids, obs_ids)
    assert audit["duplicate_count"] == 0
    assert audit["missed_from_reference"] == 0
    assert audit["coverage_rate"] == 1.0
    assert audit["exact_order_match"] is True

    # Even randomized augmentation values and decoded image means match identically!
    for r, o in zip(ref_samples, observed, strict=True):
        assert o["aug_noise"] == pytest.approx(r["aug_noise"], rel=0, abs=1e-12)
        assert o["img_mean"] == pytest.approx(r["img_mean"], rel=0, abs=1e-6)
        np.testing.assert_array_equal(o["state_arm"], r["state_arm"])


def test_distributed_rank_disjointness_and_reseed_comparison(
    v2_dataset_dir: Path,
    default_modality_configs: dict[str, ModalityConfig],
    tmp_path: Path,
) -> None:
    """Verify world_size=2 ranks get disjoint exact coverage and compare vs GR00T reseed."""
    lance_dir = tmp_path / "lance_ddp"
    convert_v2_to_lance(v2_dataset_dir, lance_dir, video_mode="copy")

    rank_ids = []
    for rank in (0, 1):
        ds = LanceShardedSingleStepDataset(
            lance_dir, EmbodimentTag.NEW_EMBODIMENT, default_modality_configs, allow_padding=True
        )
        mix = ResumableLanceMixtureDataset(
            datasets=[ds],
            weights=[1.0],
            processor=_DeterministicAugProcessor(),
            seed=42,
            batch_size=4,
            episode_pool_size=2,
            infinite=False,
            rank=rank,
            world_size=2,
        )
        ids = [
            s["id"]
            for b in mix.loader(num_workers=0, collate_fn=_identity_collate)
            for s in b
        ]
        rank_ids.append(ids)
        mix.close()

    set0, set1 = set(rank_ids[0]), set(rank_ids[1])
    assert set0.isdisjoint(set1)
    # Combined coverage equals 100% of all valid anchors across all 4 episodes!
    expected_total = sum(max(0, L - 4 + 1) for L in (18, 22, 16, 20))
    assert len(set0) + len(set1) == expected_total

    # Contrast with GR00T's stock seed+global_step reseed under 3 preemptions
    reseed_audit = simulate_groot_reseed_coverage(
        total_anchors=1000, preemption_steps=[30, 70, 100], batch_size=8, base_seed=42
    )
    assert reseed_audit["duplicate_count"] > 100
    assert reseed_audit["missed_from_reference"] > 100
    assert reseed_audit["exact_order_match"] is False
