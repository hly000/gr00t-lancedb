# SPDX-License-Identifier: Apache-2.0
"""gr00t-lancedb: Zero-download S3 streaming & Spot-resumable training for NVIDIA Isaac-GR00T."""

from __future__ import annotations

from ._gr00t_compat import (
    HAS_GROOT,
    BaseProcessor,
    EmbodimentTag,
    MessageType,
    ModalityConfig,
    ShardedDataset,
    VLAStepData,
)
from .convert import attach_gr00t_meta, convert_v2_to_lance
from .episode_loader import LanceEpisodeLoader
from .metadata import (
    Gr00tMetaBundle,
    compute_gr00t_stats_from_arrays,
    compute_gr00t_stats_from_lance_table,
    extract_dataset_statistics,
    load_gr00t_meta_bundle,
    merge_statistics,
)
from .resumable_mixture import ResumableLanceMixtureDataset
from .sharded_dataset import LanceShardedSingleStepDataset
from .trainer_integration import (
    SpotPreemptionMonitor,
    build_lance_mixture_dataset,
    create_spot_resumable_gr00t_trainer_class,
    load_dataloader_state,
    make_hf_spot_callback,
    save_dataloader_state,
)

__version__ = "0.1.0"

__all__ = [
    "HAS_GROOT",
    "BaseProcessor",
    "EmbodimentTag",
    "Gr00tMetaBundle",
    "LanceEpisodeLoader",
    "LanceShardedSingleStepDataset",
    "MessageType",
    "ModalityConfig",
    "ResumableLanceMixtureDataset",
    "ShardedDataset",
    "SpotPreemptionMonitor",
    "VLAStepData",
    "__version__",
    "attach_gr00t_meta",
    "build_lance_mixture_dataset",
    "compute_gr00t_stats_from_arrays",
    "compute_gr00t_stats_from_lance_table",
    "convert_v2_to_lance",
    "create_spot_resumable_gr00t_trainer_class",
    "extract_dataset_statistics",
    "load_dataloader_state",
    "load_gr00t_meta_bundle",
    "make_hf_spot_callback",
    "merge_statistics",
    "save_dataloader_state",
]
