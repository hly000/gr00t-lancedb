# SPDX-License-Identifier: Apache-2.0
"""GR00T ``ShardedDataset`` implementation backed by ``LanceEpisodeLoader``.

Drop-in replacement for GR00T's ``FastShardedSingleStepDataset`` that streams
directly from local or S3/GCS/HF Lance tables without downloading parquets or MP4s.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ._gr00t_compat import (
    EmbodimentTag,
    MessageType,
    ModalityConfig,
    ShardedDataset,
    VLAStepData,
    coerce_embodiment_tag,
)
from .episode_loader import LanceEpisodeLoader


class LanceShardedSingleStepDataset(ShardedDataset):
    """Sharded single-step GR00T dataset streaming from a LanceDB table layout.

    Satisfies GR00T's ``ShardedDataset`` contract (``get_shard_splits``,
    ``load_split``, ``process_split_item``, ``get_shard``,
    ``get_dataset_statistics``, ``get_initial_actions``) so it plugs directly
    into either GR00T's ``FastShardedMixtureDataset`` or our
    ``ResumableLanceMixtureDataset``.
    """

    def __init__(
        self,
        dataset_path: str | Path,
        embodiment_tag: EmbodimentTag | str,
        modality_configs: dict[str, ModalityConfig],
        shard_size: int = 2**10,
        episode_sampling_rate: float = 0.1,
        seed: int = 42,
        allow_padding: bool = False,
        *,
        repo_id: str | None = None,
        modality_json_path: str | Path | None = None,
        stats_json_path: str | Path | None = None,
        storage_options: dict[str, str] | None = None,
        revision: str | None = None,
        token: str | bool | None = None,
        use_mini_mp4: bool = True,
        video_decoder_cache_size: int = 16,
        where: str | None = None,
        lance_version: int | None = None,
    ) -> None:
        super().__init__(str(dataset_path))
        self.embodiment_tag = coerce_embodiment_tag(embodiment_tag)
        self.modality_configs = modality_configs
        self.shard_size = int(shard_size)
        self.episode_sampling_rate = float(episode_sampling_rate)
        self.seed = int(seed)
        self.allow_padding = bool(allow_padding)
        self.where = where
        self.lance_version = lance_version
        self.processor = None
        self.rng = np.random.default_rng(self.seed)

        action_delta_indices = modality_configs["action"].delta_indices
        self.action_horizon = max(action_delta_indices) - min(action_delta_indices) + 1

        self.episode_loader = LanceEpisodeLoader(
            dataset_path=dataset_path,
            modality_configs=modality_configs,
            repo_id=repo_id,
            modality_json_path=modality_json_path,
            stats_json_path=stats_json_path,
            storage_options=storage_options,
            revision=revision,
            token=token,
            use_mini_mp4=use_mini_mp4,
            video_decoder_cache_size=video_decoder_cache_size,
            where=where,
            lance_version=lance_version,
        )
        # Keep validated modality configs from the loader (handles multi-language truncation, etc.)
        self.modality_configs = self.episode_loader.modality_configs

        self.sharded_episodes: list[list[tuple[int, np.ndarray]]] = []
        self.shard_lengths: np.ndarray = np.zeros(0, dtype=int)
        self.shard_dataset()

    def get_effective_episode_length(self, episode_index: int) -> int:
        original_length = self.episode_loader.get_episode_length(episode_index)
        return max(0, original_length - self.action_horizon + 1)

    def get_valid_steps_for_episode(self, episode_index: int) -> np.ndarray:
        """Return sorted valid anchor ``frame_index`` values for ``episode_index``.

        When ``where`` is active, intersects ``[0, get_effective_episode_length(episode_index))``
        with the frame indices matching the SQL predicate while preserving full episode temporal
        continuity for action horizons and video ``delta_indices``.
        """
        eff_len = self.get_effective_episode_length(episode_index)
        return self.episode_loader.get_valid_steps_for_episode(
            episode_index, max_step_exclusive=eff_len
        )

    def get_all_valid_anchors(self) -> list[tuple[int, int]]:
        """Return every valid ``(ep_idx, step_index)`` pair in deterministic order."""
        anchors: list[tuple[int, int]] = []
        for ep_idx in range(len(self.episode_loader)):
            for step_idx in self.get_valid_steps_for_episode(ep_idx):
                anchors.append((ep_idx, int(step_idx)))
        return anchors

    def shard_dataset(self) -> None:
        """Mirror ``ShardedSingleStepDataset.shard_dataset`` with optional SQL anchor filtering."""
        shuffled_episode_indices = self.rng.permutation(len(self.episode_loader.episode_lengths))
        num_splits = max(1, int(1 / self.episode_sampling_rate))

        if len(shuffled_episode_indices) == 0:
            raise AssertionError(f"No valid trajectories found for dataset {self.dataset_path}")

        episode_splits: list[tuple[int, np.ndarray]] = []
        total_steps = 0
        for ep_idx in shuffled_episode_indices:
            step_indices = self.get_valid_steps_for_episode(int(ep_idx)).copy()
            if len(step_indices) == 0:
                continue
            self.rng.shuffle(step_indices)
            total_steps += len(step_indices)
            for i in range(num_splits):
                split_step_indices = step_indices[i::num_splits]
                if len(split_step_indices) > 0:
                    episode_splits.append((int(ep_idx), split_step_indices))

        if total_steps == 0 or not episode_splits:
            raise AssertionError(
                f"No valid timesteps found for dataset {self.dataset_path}; "
                f"episode lengths may be shorter than action horizon {self.action_horizon}"
            )

        num_shards = min(int(np.ceil(total_steps / self.shard_size)), len(episode_splits))
        sharded_episodes: list[list[tuple[int, np.ndarray]]] = [[] for _ in range(num_shards)]
        shard_lengths = np.zeros(num_shards, dtype=int)

        for i, (ep_idx, split_step_indices) in enumerate(episode_splits):
            shard_index = i if i < num_shards else int(np.argmin(shard_lengths))
            sharded_episodes[shard_index].append((ep_idx, split_step_indices))
            shard_lengths[shard_index] += len(split_step_indices)

        self.sharded_episodes = sharded_episodes
        self.shard_lengths = shard_lengths

    def __len__(self) -> int:
        return len(self.shard_lengths)

    def get_shard_length(self, idx: int) -> int:
        return int(self.shard_lengths[idx])

    def get_shard_splits(self, idx: int) -> list[tuple[int, np.ndarray]]:
        return self.sharded_episodes[idx]

    def load_split(self, split: tuple[int, np.ndarray]) -> list[VLAStepData]:
        ep_idx, step_indices = split
        return self.episode_loader.load_steps(
            ep_idx=ep_idx,
            step_indices=step_indices,
            embodiment_tag=self.embodiment_tag,
            allow_padding=self.allow_padding,
        )

    def load_anchor_batch(
        self,
        ep_step_pairs: list[tuple[int, int]],
        *,
        rng: np.random.Generator | None = None,
    ) -> list[VLAStepData]:
        return self.episode_loader.load_cross_episode_batch(
            ep_step_pairs=ep_step_pairs,
            embodiment_tag=self.embodiment_tag,
            allow_padding=self.allow_padding,
            rng=rng,
        )

    def process_split_item(self, item: VLAStepData) -> Any:
        if self.processor is None:
            return item
        return self.processor([{"type": MessageType.EPISODE_STEP.value, "content": item}])

    def get_shard(self, idx: int) -> list[Any]:
        return [
            self.process_split_item(item)
            for split in self.get_shard_splits(idx)
            for item in self.load_split(split)
        ]

    def get_dataset_statistics(self) -> dict[str, Any]:
        return self.episode_loader.get_dataset_statistics()

    def get_initial_actions(self) -> list[dict[str, Any]]:
        return self.episode_loader.get_initial_actions()

    @property
    def is_open(self) -> bool:
        return self.episode_loader.is_open

    def assert_unopened(self) -> None:
        self.episode_loader.assert_unopened()

    def close(self) -> None:
        self.episode_loader.close()


__all__ = ["LanceShardedSingleStepDataset"]
