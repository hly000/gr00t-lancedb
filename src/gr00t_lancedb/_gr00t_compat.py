# SPDX-License-Identifier: Apache-2.0
"""Compatibility layer for NVIDIA Isaac-GR00T core data types and interfaces.

When ``gr00t`` (NVIDIA/Isaac-GR00T) is installed in the environment, this module
re-exports the real classes so every dataset class satisfies ``isinstance`` checks
against ``gr00t.data.interfaces.ShardedDataset`` and produces authentic
``gr00t.data.types.VLAStepData`` instances.

When ``gr00t`` is not installed (e.g. lightweight CPU data-prep, CI, or
conversion boxes), duck-typed equivalents with matching field layouts and
signatures are provided so converters, parity tests, and Spot resume benchmarks
run without requiring CUDA or ``flash_attn``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

try:
    from gr00t.data.embodiment_tags import EmbodimentTag  # type: ignore[import-not-found]
    from gr00t.data.interfaces import BaseProcessor, ShardedDataset  # type: ignore[import-not-found]
    from gr00t.data.types import (  # type: ignore[import-not-found]
        MessageType,
        ModalityConfig,
        VLAStepData,
    )

    HAS_GROOT = True
except ImportError:
    HAS_GROOT = False

    class EmbodimentTag(Enum):  # type: ignore[no-redef]
        GR1 = "gr1"
        OXE_DROID = "oxe_droid"
        AGIBOT_GENIE1 = "agibot_genie1"
        NEW_EMBODIMENT = "new_embodiment"
        ROBOCASA_PANDA_OMRON = "robocasa_panda_omron"
        LIBERO_PANDA = "libero_panda"
        FRANKA = "franka"
        UNITREE_G1 = "unitree_g1"

        @classmethod
        def _missing_(cls, value: object):
            # Accept arbitrary custom embodiment strings cleanly in standalone mode.
            if isinstance(value, str):
                obj = object.__new__(cls)
                obj._name_ = value.upper()
                obj._value_ = value
                return obj
            return None

    class MessageType(Enum):  # type: ignore[no-redef]
        START_OF_EPISODE = "start_of_episode"
        END_OF_EPISODE = "end_of_episode"
        EPISODE_STEP = "episode_step"
        IMAGE = "image"
        TEXT = "text"

    @dataclass
    class VLAStepData:  # type: ignore[no-redef]
        images: dict[str, list[np.ndarray]]
        states: dict[str, np.ndarray]
        actions: dict[str, np.ndarray]
        masks: dict[str, list[np.ndarray]] | None = None
        text: str | None = None
        embodiment: EmbodimentTag = EmbodimentTag.NEW_EMBODIMENT
        is_demonstration: bool = False
        metadata: dict[str, Any] = field(default_factory=dict)

    @dataclass
    class ModalityConfig:  # type: ignore[no-redef]
        delta_indices: list[int]
        modality_keys: list[str]
        sin_cos_embedding_keys: list[str] | None = None
        mean_std_embedding_keys: list[str] | None = None
        action_configs: list[Any] | None = None

        def __post_init__(self) -> None:
            if self.delta_indices is None or not isinstance(self.delta_indices, list):
                raise ValueError(f"delta_indices must be a non-None list, got {self.delta_indices!r}")
            if (
                self.modality_keys is None
                or not isinstance(self.modality_keys, list)
                or len(self.modality_keys) == 0
            ):
                raise ValueError(f"modality_keys must be a non-empty list, got {self.modality_keys!r}")

    class BaseProcessor:  # type: ignore[no-redef]
        training: bool = True

        def __call__(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
            raise NotImplementedError

        def set_statistics(self, statistics: dict[str, Any], override: bool = False) -> None:
            self.statistics = statistics

        def train(self) -> None:
            self.training = True

        def eval(self) -> None:
            self.training = False

    class ShardedDataset(ABC):  # type: ignore[no-redef]
        def __init__(self, dataset_path: Any) -> None:
            self.dataset_path = dataset_path
            self.processor: BaseProcessor | None = None

        @abstractmethod
        def __len__(self) -> int:
            pass

        @abstractmethod
        def get_shard_length(self, idx: int) -> int:
            pass

        @abstractmethod
        def get_shard(self, idx: int) -> list:
            pass

        def get_shard_splits(self, idx: int) -> list:
            return [idx]

        def load_split(self, split: Any) -> list:
            return self.get_shard(split)

        def process_split_item(self, item: Any) -> Any:
            return item

        def set_processor(self, processor: BaseProcessor) -> None:
            self.processor = processor

        def get_dataset_statistics(self) -> dict[str, Any]:
            raise NotImplementedError()


def coerce_embodiment_tag(tag: EmbodimentTag | str) -> EmbodimentTag:
    """Convert a string or EmbodimentTag into an EmbodimentTag instance."""
    if isinstance(tag, EmbodimentTag):
        return tag
    try:
        return EmbodimentTag(tag)
    except ValueError:
        return EmbodimentTag[tag.upper()]


__all__ = [
    "HAS_GROOT",
    "BaseProcessor",
    "EmbodimentTag",
    "MessageType",
    "ModalityConfig",
    "ShardedDataset",
    "VLAStepData",
    "coerce_embodiment_tag",
]
