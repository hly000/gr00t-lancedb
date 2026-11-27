# SPDX-License-Identifier: Apache-2.0
"""Model-in-the-loop data flywheel for GR00T on Lance (fast-validation protocol).

* :mod:`.weights`     per-anchor sampling weights from a ``frames.lance`` column (round 2)
* :mod:`.checkpoints` keep the fork-point / curve checkpoints out of HF rotation
* :mod:`.rollouts`    on-disk format of recorded simulator rollouts
* :mod:`.mining`      failed rollouts -> divergence point -> matched training frames -> weights
* :mod:`.paired_eval` seed-paired success comparison (McNemar) between two checkpoints

See ``docs/flywheel-fast-validation.md``. Light imports only: ``mining`` pulls in lancedb and
is imported lazily by the CLI so training processes stay fork-safe.
"""

from .checkpoints import KeepCheckpointsCallback, keep_checkpoint, parse_steps
from .paired_eval import compare_rollouts, mcnemar_exact_p, wilson_interval
from .rollouts import RolloutEpisode, iter_episodes, read_summary
from .weights import (
    AnchorWeightMap,
    EpisodeAnchors,
    WeightedAnchors,
    build_weighted_anchors,
    dataset_episode_anchors,
    map_frame_weights,
)

__all__ = [
    "AnchorWeightMap",
    "EpisodeAnchors",
    "KeepCheckpointsCallback",
    "RolloutEpisode",
    "WeightedAnchors",
    "build_weighted_anchors",
    "compare_rollouts",
    "dataset_episode_anchors",
    "iter_episodes",
    "keep_checkpoint",
    "map_frame_weights",
    "mcnemar_exact_p",
    "parse_steps",
    "read_summary",
    "wilson_interval",
]
