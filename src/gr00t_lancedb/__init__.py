# SPDX-License-Identifier: Apache-2.0
"""gr00t-lancedb: Zero-download S3 streaming & Spot-resumable training for NVIDIA Isaac-GR00T.

Public names are resolved lazily (PEP 562): ``from gr00t_lancedb import
LanceShardedSingleStepDataset`` imports only the dataset modules, never ``lerobot`` /
``lancedb`` / ``av`` (~2-3 s, and a Tokio runtime that is not fork-safe). Those are
imported by DataLoader workers on first read, by converters/curation, or up front in the
background via :func:`preload_lance_stack`.
"""

from __future__ import annotations

from importlib import import_module
import threading
from typing import TYPE_CHECKING, Any

# Must run before anything imports lerobot / lerobot_lancedb (see _hub_compat docstring).
from ._hub_compat import ensure_hub_compat

ensure_hub_compat()

__version__ = "0.1.0"

_LAZY: dict[str, str] = {
    # GR00T compat types
    "HAS_GROOT": "._gr00t_compat",
    "BaseProcessor": "._gr00t_compat",
    "EmbodimentTag": "._gr00t_compat",
    "MessageType": "._gr00t_compat",
    "ModalityConfig": "._gr00t_compat",
    "ShardedDataset": "._gr00t_compat",
    "VLAStepData": "._gr00t_compat",
    # conversion / curation (import lancedb + lerobot)
    "attach_gr00t_meta": ".convert",
    "convert_v2_to_lance": ".convert",
    "CURATION_COLUMNS": ".curate",
    "CurationReport": ".curate",
    "curate_lance_dataset": ".curate",
    # data loading (lightweight in the main process)
    "LanceEpisodeLoader": ".episode_loader",
    "LanceShardedSingleStepDataset": ".sharded_dataset",
    "ResumableLanceMixtureDataset": ".resumable_mixture",
    # metadata
    "CURATION_REPORT_FILENAME": ".metadata",
    "CURATION_VERSIONS_FILENAME": ".metadata",
    "Gr00tMetaBundle": ".metadata",
    "compute_gr00t_stats_from_arrays": ".metadata",
    "compute_gr00t_stats_from_lance_table": ".metadata",
    "extract_dataset_statistics": ".metadata",
    "load_gr00t_meta_bundle": ".metadata",
    "merge_statistics": ".metadata",
    "select_curation_entry": ".metadata",
    # trainer integration
    "SpotPreemptionMonitor": ".trainer_integration",
    "build_lance_mixture_dataset": ".trainer_integration",
    "create_spot_resumable_gr00t_trainer_class": ".trainer_integration",
    "load_dataloader_state": ".trainer_integration",
    "make_hf_spot_callback": ".trainer_integration",
    "prewarm_lance_workers": ".trainer_integration",
    "save_dataloader_state": ".trainer_integration",
    # I/O stats & GOP cache
    "GopDiskCache": ".gop_cache",
    "aggregate_io_stats": ".gop_cache",
}

__all__ = sorted([*_LAZY, "__version__", "preload_lance_stack", "join_preload"])


def __getattr__(name: str) -> Any:
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(mod, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


_PRELOAD: dict[str, Any] = {"thread": None, "error": None, "wall_s": None}

# Modules a DataLoader worker needs on its first read. Importing them is fork-safe (no
# Tokio runtime starts until a table is opened); having them in the parent before fork
# means every forked worker inherits them instead of paying ~2-3 s on the first batch.
_PRELOAD_MODULES = ("lerobot_lancedb", "lancedb.permutation", "torchcodec.decoders")


def preload_lance_stack(background: bool = True) -> threading.Thread | None:
    """Import lerobot-lancedb / lancedb / torchcodec, optionally on a daemon thread.

    Meant to overlap the import with model loading. Call :func:`join_preload` before the
    first ``fork()`` (DataLoader iterator creation) so no fork happens mid-import.
    """
    import time

    def _run() -> None:
        t0 = time.perf_counter()
        try:
            for name in _PRELOAD_MODULES:
                try:
                    import_module(name)
                except ImportError:
                    if name == "lerobot_lancedb":
                        raise
        except BaseException as exc:  # noqa: BLE001 - surfaced by join_preload()
            _PRELOAD["error"] = exc
        finally:
            _PRELOAD["wall_s"] = time.perf_counter() - t0

    if not background:
        _run()
        return None
    if _PRELOAD["thread"] is None:
        t = threading.Thread(target=_run, name="gr00t-lance-preload", daemon=True)
        _PRELOAD["thread"] = t
        t.start()
    return _PRELOAD["thread"]


def join_preload(timeout: float | None = None) -> float | None:
    """Wait for :func:`preload_lance_stack`; returns its wall time (``None`` if never started)."""
    t = _PRELOAD["thread"]
    if t is not None:
        t.join(timeout)
    err = _PRELOAD["error"]
    if err is not None:
        _PRELOAD["error"] = None
        import logging

        logging.getLogger(__name__).warning("Background Lance import failed (%s); workers will import lazily.", err)
    return _PRELOAD["wall_s"]


if TYPE_CHECKING:  # pragma: no cover - static typing only
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
    from .curate import CURATION_COLUMNS, CurationReport, curate_lance_dataset
    from .episode_loader import LanceEpisodeLoader
    from .gop_cache import GopDiskCache, aggregate_io_stats
    from .metadata import (
        CURATION_REPORT_FILENAME,
        CURATION_VERSIONS_FILENAME,
        Gr00tMetaBundle,
        compute_gr00t_stats_from_arrays,
        compute_gr00t_stats_from_lance_table,
        extract_dataset_statistics,
        load_gr00t_meta_bundle,
        merge_statistics,
        select_curation_entry,
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
