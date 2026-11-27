# SPDX-License-Identifier: Apache-2.0
"""gr00t-lancedb: Zero-download S3 streaming & Spot-resumable training for NVIDIA Isaac-GR00T."""

from __future__ import annotations

# Must run before anything imports lerobot / lerobot_lancedb (see _hub_compat docstring).
from ._hub_compat import ensure_hub_compat

ensure_hub_compat()

from ._gr00t_compat import (  # noqa: E402
    HAS_GROOT,
    BaseProcessor,
    EmbodimentTag,
    MessageType,
    ModalityConfig,
    ShardedDataset,
    VLAStepData,
)
from .convert import attach_gr00t_meta, convert_v2_to_lance
from .curate import CURATION_COLUMNS, CurationReport, curate_lance_dataset
from .episode_loader import LanceEpisodeLoader
from .metadata import (
    CURATION_REPORT_FILENAME,
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
    prewarm_lance_workers,
    save_dataloader_state,
)

__version__ = "0.1.0"

__all__ = [
    "CURATION_COLUMNS",
    "CURATION_REPORT_FILENAME",
    "HAS_GROOT",
    "BaseProcessor",
    "CurationReport",
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
    "curate_lance_dataset",
    "extract_dataset_statistics",
    "load_dataloader_state",
    "load_gr00t_meta_bundle",
    "make_hf_spot_callback",
    "merge_statistics",
    "prewarm_lance_workers",
    "save_dataloader_state",
]
