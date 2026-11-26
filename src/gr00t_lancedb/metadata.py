# SPDX-License-Identifier: Apache-2.0
"""GR00T metadata parsing, modality.json validation, and statistics utilities.

Supports both:
1. GR00T LeRobot v2.1 metadata (``episodes.jsonl``, ``tasks.jsonl``, ``modality.json``,
   ``stats.json``, ``relative_stats.json``, ``initial_actions.npz``).
2. Standard LeRobot v3.0 Lance metadata (``episodes/chunk-*/file-*.parquet``,
   ``tasks.parquet``, ``info.json``) augmented with ``modality.json`` and GR00T stats
   either embedded in ``meta.lance`` or supplied via a sidecar path.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ._gr00t_compat import ModalityConfig

logger = logging.getLogger(__name__)

LEROBOT_META_DIR_NAME = "meta"
LEROBOT_INFO_FILENAME = "info.json"
LEROBOT_EPISODES_FILENAME = "episodes.jsonl"
LEROBOT_TASKS_FILENAME = "tasks.jsonl"
LEROBOT_MODALITY_FILENAME = "modality.json"
LEROBOT_STATS_FILENAME = "stats.json"
GROOT_STATS_FILENAME = "gr00t_stats.json"
LEROBOT_RELATIVE_STATS_FILENAME = "relative_stats.json"
INITIAL_ACTIONS_FILENAME = "initial_actions.npz"
STATS_FINGERPRINTS_KEY = "__fingerprints__"

ALLOWED_MODALITIES = ("video", "state", "action", "language", "mask")
DEFAULT_COLUMN_NAMES = {
    "state": "observation.state",
    "action": "action",
}
LANG_KEYS = ("task", "sub_task")


def _rec_defaultdict() -> defaultdict:
    return defaultdict(_rec_defaultdict)


def _to_plain_dict(tree: Any) -> Any:
    if isinstance(tree, defaultdict):
        return {k: _to_plain_dict(v) for k, v in tree.items()}
    return tree


@dataclass
class Gr00tMetaBundle:
    """All GR00T-relevant metadata loaded from a materialized ``meta/`` directory."""

    info_meta: dict[str, Any]
    modality_meta: dict[str, Any]
    episodes_metadata: list[dict[str, Any]]
    tasks_map: dict[int, str]
    stats: dict[str, Any]
    initial_actions: list[dict[str, Any]] = field(default_factory=list)
    fps: float = 30.0
    chunk_size: int = 1000
    feature_config: dict[str, Any] = field(default_factory=dict)


def _load_initial_actions_npz(path: Path) -> list[dict[str, Any]]:
    """Load GR00T ``initial_actions.npz`` without requiring ``gr00t`` to be installed."""
    if not path.exists():
        return []
    try:
        from gr00t.utils.initial_actions import load_initial_actions  # type: ignore[import-not-found]

        return load_initial_actions(path)
    except ImportError:
        pass
    data = np.load(path, allow_pickle=True)
    if "arr_0" in data and len(data.files) == 1:
        item = data["arr_0"].item()
        return item if isinstance(item, list) else [item]
    # Reconstruct nested {embodiment: {key: array}} if saved flattened with '/'
    nested: dict[str, dict[str, np.ndarray]] = {}
    for flat_key in data.files:
        if "/" in flat_key:
            emb, sub = flat_key.split("/", 1)
            nested.setdefault(emb, {})[sub] = data[flat_key]
        else:
            nested.setdefault("default", {})[flat_key] = data[flat_key]
    return [nested] if nested else []


def _load_tasks_map(meta_dir: Path) -> dict[int, str]:
    """Load task_index -> task string mapping from ``tasks.jsonl`` or v3 ``tasks.parquet``."""
    jsonl_path = meta_dir / LEROBOT_TASKS_FILENAME
    if jsonl_path.exists():
        tasks_map: dict[int, str] = {}
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                tasks_map[int(rec["task_index"])] = str(rec["task"])
        return tasks_map

    parquet_path = meta_dir / "tasks.parquet"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        tasks_map = {}
        if "task" in df.columns and "task_index" in df.columns:
            for _, row in df.iterrows():
                tasks_map[int(row["task_index"])] = str(row["task"])
        elif "task_index" in df.columns:
            # LeRobot v3 stores task string in the DataFrame index.
            for task_str, row in df.iterrows():
                tasks_map[int(row["task_index"])] = str(task_str)
        return tasks_map

    return {}


def _load_episodes_metadata(meta_dir: Path, tasks_map: dict[int, str]) -> list[dict[str, Any]]:
    """Load per-episode metadata from ``episodes.jsonl`` or v3 ``episodes/*.parquet``."""
    jsonl_path = meta_dir / LEROBOT_EPISODES_FILENAME
    if jsonl_path.exists():
        records = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        records.sort(key=lambda r: int(r["episode_index"]))
        return records

    episodes_dir = meta_dir / "episodes"
    pq_files = sorted(episodes_dir.glob("chunk-*/file-*.parquet")) if episodes_dir.exists() else []
    if not pq_files:
        raise FileNotFoundError(
            f"Neither {jsonl_path} nor v3 episode parquets under {episodes_dir} were found."
        )

    records = []
    for pq_file in pq_files:
        table = pq.read_table(pq_file)
        for row in table.to_pylist():
            raw_tasks = row.get("tasks")
            if raw_tasks is None:
                tasks_list = [tasks_map[0]] if 0 in tasks_map else [""]
            elif isinstance(raw_tasks, (list, tuple)):
                tasks_list = [
                    tasks_map[int(t)] if isinstance(t, (int, np.integer)) and int(t) in tasks_map else str(t)
                    for t in raw_tasks
                ]
            else:
                tasks_list = [str(raw_tasks)]
            rec: dict[str, Any] = {
                "episode_index": int(row["episode_index"]),
                "length": int(row["length"]),
                "tasks": tasks_list,
            }
            if "sub_tasks" in row and row["sub_tasks"] is not None:
                rec["sub_tasks"] = row["sub_tasks"]
            records.append(rec)
    records.sort(key=lambda r: int(r["episode_index"]))
    return records


def load_gr00t_meta_bundle(
    meta_dir: Path,
    *,
    modality_json_path: str | Path | None = None,
    stats_json_path: str | Path | None = None,
) -> Gr00tMetaBundle:
    """Load GR00T metadata bundle from a materialized local ``meta/`` directory."""
    info_path = meta_dir / LEROBOT_INFO_FILENAME
    if not info_path.exists():
        raise FileNotFoundError(f"Missing {info_path}")
    info_meta = json.loads(info_path.read_text(encoding="utf-8"))

    mod_path = Path(modality_json_path) if modality_json_path is not None else meta_dir / LEROBOT_MODALITY_FILENAME
    if not mod_path.exists():
        raise FileNotFoundError(
            f"Missing GR00T modality.json at {mod_path}. Pass `modality_json_path=...` "
            "or attach it with `gr00t-lance-attach`."
        )
    modality_meta = json.loads(mod_path.read_text(encoding="utf-8"))

    tasks_map = _load_tasks_map(meta_dir)
    episodes_metadata = _load_episodes_metadata(meta_dir, tasks_map)

    # Prefer dedicated GR00T stats (which always include q01/q99) if present.
    stats: dict[str, Any] = {}
    candidate_stats_paths: list[Path] = []
    if stats_json_path is not None:
        candidate_stats_paths.append(Path(stats_json_path))
    candidate_stats_paths.extend([meta_dir / GROOT_STATS_FILENAME, meta_dir / LEROBOT_STATS_FILENAME])
    for sp in candidate_stats_paths:
        if sp.exists():
            stats = json.loads(sp.read_text(encoding="utf-8"))
            stats.pop(STATS_FINGERPRINTS_KEY, None)
            break

    rel_stats_path = meta_dir / LEROBOT_RELATIVE_STATS_FILENAME
    if rel_stats_path.exists():
        rel_stats = json.loads(rel_stats_path.read_text(encoding="utf-8"))
        rel_stats.pop(STATS_FINGERPRINTS_KEY, None)
        stats["relative_action"] = rel_stats

    initial_actions = _load_initial_actions_npz(meta_dir / INITIAL_ACTIONS_FILENAME)

    return Gr00tMetaBundle(
        info_meta=info_meta,
        modality_meta=modality_meta,
        episodes_metadata=episodes_metadata,
        tasks_map=tasks_map,
        stats=stats,
        initial_actions=initial_actions,
        fps=float(info_meta.get("fps", 30)),
        chunk_size=int(info_meta.get("chunks_size", 1000)),
        feature_config=info_meta.get("features", {}),
    )


def parse_and_validate_modality_configs(
    modality_configs: dict[str, ModalityConfig],
    modality_meta: dict[str, Any],
) -> tuple[dict[str, ModalityConfig], dict[str, str]]:
    """Mirror ``LeRobotEpisodeLoader._parse_and_validate_modality_configs``.

    Returns:
        ``(validated_configs, video_key_mapping)`` where ``video_key_mapping`` maps
        model config video keys to dataset ``modality.json`` video keys when naming
        differs (e.g. GR00T N1.6 vs N1.7).
    """
    unknown = [m for m in modality_configs if m not in ALLOWED_MODALITIES]
    if unknown:
        logger.debug("Skipping modalities not supported by dataset loader: %s", unknown)
    validated: dict[str, ModalityConfig] = {
        k: v for k, v in modality_configs.items() if k in ALLOWED_MODALITIES
    }

    if "language" in validated:
        lang_cfg = validated["language"]
        if len(lang_cfg.modality_keys) < 1:
            raise AssertionError("Language modality must have at least one key")
        if len(lang_cfg.modality_keys) > 1:
            logger.warning(
                "Language modality has %d keys, only the first key will be used: %s",
                len(lang_cfg.modality_keys),
                lang_cfg.modality_keys[0],
            )
            validated["language"] = ModalityConfig(
                delta_indices=lang_cfg.delta_indices,
                modality_keys=[lang_cfg.modality_keys[0]],
                sin_cos_embedding_keys=lang_cfg.sin_cos_embedding_keys,
                mean_std_embedding_keys=lang_cfg.mean_std_embedding_keys,
                action_configs=lang_cfg.action_configs[:1] if lang_cfg.action_configs is not None else None,
            )
        if validated["language"].delta_indices != [0]:
            raise AssertionError("Only single timestep is supported for language modality")

    video_key_mapping: dict[str, str] = {}
    if "video" in validated and "video" in modality_meta:
        config_keys = validated["video"].modality_keys
        meta_keys = list(modality_meta["video"].keys())
        if any(k not in modality_meta["video"] for k in config_keys):
            if len(config_keys) != len(meta_keys):
                raise AssertionError(
                    f"Cannot auto-map video keys: config has {len(config_keys)} keys "
                    f"{config_keys} but dataset modality meta has {len(meta_keys)} keys "
                    f"{meta_keys}. Counts must match for positional mapping."
                )
            for cfg_key, meta_key in zip(config_keys, meta_keys):
                video_key_mapping[cfg_key] = meta_key
            logger.warning(
                "Video key mismatch between model config and dataset. Auto-mapping by position: %s",
                video_key_mapping,
            )

    return validated, video_key_mapping


def resolve_video_original_key(
    image_key: str,
    modality_meta: dict[str, Any],
    video_key_mapping: dict[str, str],
) -> str:
    meta_key = video_key_mapping.get(image_key, image_key)
    return modality_meta["video"][meta_key].get("original_key", f"observation.images.{meta_key}")


def extract_dataset_statistics(
    modality_configs: dict[str, ModalityConfig],
    modality_meta: dict[str, Any],
    raw_stats: dict[str, Any],
) -> dict[str, Any]:
    """Mirror ``LeRobotEpisodeLoader.get_dataset_statistics`` exactly."""
    dataset_statistics = _rec_defaultdict()
    for modality in ("state", "action"):
        if modality not in modality_configs:
            continue
        for joint_key in modality_configs[modality].modality_keys:
            group_meta = modality_meta[modality][joint_key]
            stats_key = group_meta.get("original_key") or DEFAULT_COLUMN_NAMES[modality]
            if stats_key not in raw_stats:
                raise KeyError(
                    f"Statistics key {stats_key!r} not found in stats metadata. "
                    f"Available keys: {sorted(raw_stats.keys())}. Run `gr00t-lance-attach --compute-stats`."
                )
            start_idx = int(group_meta["start"])
            end_idx = int(group_meta["end"])
            for stat_type, values in raw_stats[stats_key].items():
                if stat_type == "count":
                    continue
                if isinstance(values, (list, tuple, np.ndarray)):
                    dataset_statistics[modality][joint_key][stat_type] = list(values[start_idx:end_idx])
                else:
                    dataset_statistics[modality][joint_key][stat_type] = values

    stats = _to_plain_dict(dataset_statistics)
    if "relative_action" in raw_stats:
        stats["relative_action"] = raw_stats["relative_action"]
    return stats


def compute_gr00t_stats_from_arrays(
    columns: dict[str, np.ndarray],
) -> dict[str, dict[str, list[float]]]:
    """Compute GR00T statistics (mean, std, min, max, q01, q99) from 2D float arrays."""
    result: dict[str, dict[str, list[float]]] = {}
    for key, arr in columns.items():
        mat = np.asarray(arr, dtype=np.float64)
        if mat.ndim == 1:
            mat = mat[:, None]
        result[key] = {
            "mean": np.mean(mat, axis=0).tolist(),
            "std": np.std(mat, axis=0).tolist(),
            "min": np.min(mat, axis=0).tolist(),
            "max": np.max(mat, axis=0).tolist(),
            "q01": np.quantile(mat, 0.01, axis=0).tolist(),
            "q99": np.quantile(mat, 0.99, axis=0).tolist(),
        }
    return result


def compute_gr00t_stats_from_lance_table(
    frames_table: Any,
    feature_keys: Sequence[str],
    *,
    to_lance_column_fn: Any = None,
) -> dict[str, dict[str, list[float]]]:
    """Compute GR00T stats (including q01/q99) directly from ``frames.lance`` tabular columns.

    Reads only the requested tabular columns (zero video traffic on S3).
    """
    if to_lance_column_fn is None:
        from lerobot_lancedb import to_lance_column

        to_lance_column_fn = to_lance_column

    lance_cols = [to_lance_column_fn(k) for k in feature_keys]
    arrow_tbl = frames_table.search().select(lance_cols).to_arrow()
    arrays: dict[str, np.ndarray] = {}
    for orig_key, lance_col in zip(feature_keys, lance_cols):
        col = arrow_tbl.column(lance_col).combine_chunks()
        if hasattr(col, "flatten") and hasattr(col.type, "value_type"):
            vals = col.flatten().to_numpy(zero_copy_only=False)
            arrays[orig_key] = vals.reshape(len(col), -1)
        else:
            arrays[orig_key] = col.to_numpy(zero_copy_only=False)
    return compute_gr00t_stats_from_arrays(arrays)


def merge_statistics(
    per_dataset_stats: list[dict[str, dict[str, list[float] | np.ndarray]]],
    dataset_sampling_weights: list[float] | np.ndarray,
    is_relative_stats: bool = False,
) -> dict[str, dict[str, list[float]]]:
    """Exact mirror of GR00T ``sharded_mixture_dataset.merge_statistics``."""
    weights = np.asarray(dataset_sampling_weights, dtype=np.float64)
    normalized_weights = weights / weights.sum()
    overall_stats: dict[str, dict[str, list[float]]] = {}

    for modality, modality_stats in per_dataset_stats[0].items():
        if not isinstance(modality_stats, dict) or "mean" not in modality_stats:
            continue
        dim = (
            [len(modality_stats["mean"])]
            if not is_relative_stats
            else np.asarray(modality_stats["mean"]).shape
        )
        weighted_means = np.zeros(dim, dtype=np.float64)
        weighted_squares = np.zeros(dim, dtype=np.float64)
        min_list, max_list, q01_list, q99_list = [], [], [], []

        for ds_idx, ds_stats in enumerate(per_dataset_stats):
            w_i = normalized_weights[ds_idx]
            st = ds_stats[modality]
            means = np.asarray(st["mean"], dtype=np.float64)
            stds = np.asarray(st["std"], dtype=np.float64)
            weighted_means += w_i * means
            weighted_squares += w_i * (stds**2 + means**2)
            min_list.append(st["min"])
            max_list.append(st["max"])
            q01_list.append(st["q01"])
            q99_list.append(st["q99"])

        overall_min = np.min(np.asarray(min_list, dtype=np.float64), axis=0)
        overall_max = np.max(np.asarray(max_list, dtype=np.float64), axis=0)
        overall_var = np.maximum(weighted_squares - weighted_means**2, 0.0)
        overall_var[overall_min == overall_max] = 0.0

        overall_stats[modality] = {
            "min": overall_min.tolist(),
            "max": overall_max.tolist(),
            "mean": weighted_means.tolist(),
            "std": np.sqrt(overall_var).tolist(),
            "q01": np.min(np.asarray(q01_list, dtype=np.float64), axis=0).tolist(),
            "q99": np.max(np.asarray(q99_list, dtype=np.float64), axis=0).tolist(),
        }

    return overall_stats


__all__ = [
    "ALLOWED_MODALITIES",
    "DEFAULT_COLUMN_NAMES",
    "GROOT_STATS_FILENAME",
    "INITIAL_ACTIONS_FILENAME",
    "LANG_KEYS",
    "LEROBOT_EPISODES_FILENAME",
    "LEROBOT_INFO_FILENAME",
    "LEROBOT_META_DIR_NAME",
    "LEROBOT_MODALITY_FILENAME",
    "LEROBOT_RELATIVE_STATS_FILENAME",
    "LEROBOT_STATS_FILENAME",
    "LEROBOT_TASKS_FILENAME",
    "Gr00tMetaBundle",
    "compute_gr00t_stats_from_arrays",
    "compute_gr00t_stats_from_lance_table",
    "extract_dataset_statistics",
    "load_gr00t_meta_bundle",
    "merge_statistics",
    "parse_and_validate_modality_configs",
    "resolve_video_original_key",
]
