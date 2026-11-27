# SPDX-License-Identifier: Apache-2.0
"""Non-invasive integration hooks for NVIDIA Isaac-GR00T's ``Gr00tTrainer`` and ``DatasetFactory``.

Provides:
1. ``build_lance_mixture_dataset``: constructs a ``ResumableLanceMixtureDataset`` directly from
   local or S3 Lance dataset URIs (skipping GR00T's local-POSIX ``generate_stats`` scan).
2. ``SpotPreemptionState`` + ``save_dataloader_state`` / ``load_dataloader_state``: atomic
   per-rank checkpoint persistence for ``ResumableLanceMixtureDataset`` (with optional S3 sync)
   and ``SIGTERM`` / ``SIGUSR1`` graceful Spot preemption capture.
3. ``SpotResumableGr00tTrainer`` / ``create_spot_resumable_trainer``: drop-in subclass/wrapper
   around GR00T's ``Gr00tTrainer`` that:
   - Forces ``lance_mp_context()`` (``forkserver``/``spawn``) instead of deadlocking ``fork``
   - Restores exact ``ResumableLanceMixtureDataset`` cursors from ``checkpoint-<step>/`` instead
     of GR00T's non-reproducible ``seed + global_step`` reseed
   - Tracks parent-process delivered batches across multi-worker DataLoaders so ``state_dict()``
     is exact at every checkpoint and Spot preemption boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import json
import logging
import os
from pathlib import Path
import signal
import tempfile
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from ._isolated import lance_mp_context

from ._gr00t_compat import BaseProcessor, EmbodimentTag, ModalityConfig
from .resumable_mixture import ResumableLanceMixtureDataset, _infer_batch_size
from .sharded_dataset import LanceShardedSingleStepDataset

logger = logging.getLogger(__name__)

DATALOADER_STATE_PREFIX = "lance_dataloader_state"


DEFAULT_WORKER_PRELOAD: tuple[str, ...] = (
    "numpy",
    "pandas",
    "pyarrow",
    "torch",
    "transformers",
    "lerobot_lancedb",
    "gr00t_lancedb",
)


def prewarm_lance_workers(extra_modules: Sequence[str] = ()) -> str:
    """Start the ``forkserver`` now, with heavy modules pre-imported, and return the context name.

    Lance's Rust/Tokio runtime is not fork-safe, so DataLoader workers must use ``forkserver``
    (or ``spawn``) instead of GR00T's default ``fork``. Without preloading, every worker
    re-imports torch / transformers / GR00T from scratch when the first batch is requested
    (~60 s for 4 workers on a slow filesystem). Calling this at program start makes the
    forkserver import those modules *once*, in parallel with model loading; workers are
    then cheap ``fork()``s of that warm, Lance-free server.

    Only modules that do not open Lance datasets at import time may be preloaded.
    """
    import multiprocessing as mp

    ctx_name = lance_mp_context()
    if ctx_name != "forkserver":
        return ctx_name
    from multiprocessing import forkserver

    modules = list(dict.fromkeys([*DEFAULT_WORKER_PRELOAD, *extra_modules]))
    mp.get_context("forkserver").set_forkserver_preload(modules)
    forkserver.ensure_running()
    logger.info("Pre-warmed forkserver for Lance dataloader workers (preload=%s)", modules)
    return ctx_name


def dataloader_state_path(checkpoint_dir: str | Path, rank: int = 0) -> Path:
    return Path(checkpoint_dir) / f"{DATALOADER_STATE_PREFIX}_rank{rank}.json"


def save_dataloader_state(
    dataset: ResumableLanceMixtureDataset,
    checkpoint_dir: str | Path,
    *,
    rank: int | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Atomically write ``dataset.state_dict()`` into ``checkpoint_dir`` (fsync + os.replace)."""
    r = dataset.rank if rank is None else int(rank)
    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)
    target = dataloader_state_path(ckpt_path, r)
    payload = dataset.state_dict()
    if extra:
        payload["extra"] = extra

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=ckpt_path,
            prefix=f".{target.name}.",
            suffix=".tmp",
            encoding="utf-8",
            delete=False,
        ) as f:
            tmp_path = Path(f.name)
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, target)
    except BaseException:
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise
    return target


def load_dataloader_state(
    dataset: ResumableLanceMixtureDataset,
    checkpoint_dir: str | Path,
    *,
    rank: int | None = None,
) -> dict[str, Any]:
    """Load and restore ``dataset`` cursor state from ``checkpoint_dir``."""
    r = dataset.rank if rank is None else int(rank)
    path = dataloader_state_path(checkpoint_dir, r)
    if not path.exists():
        raise FileNotFoundError(f"No Lance dataloader state file found at {path}")
    state = json.loads(path.read_text(encoding="utf-8"))
    dataset.load_state_dict(state)
    return state


class SpotPreemptionMonitor:
    """Signal monitor for cloud Spot preemption warnings (``SIGTERM`` / ``SIGUSR1``).

    Can be installed once per training process; sets ``preempted = True`` upon signal
    delivery so the training loop or HuggingFace ``TrainerCallback`` saves a clean
    checkpoint + exact dataloader cursor before the instance terminates.
    """

    def __init__(self, signals: Sequence[int] = (signal.SIGTERM, signal.SIGUSR1)) -> None:
        self.signals = tuple(signals)
        self.preempted: bool = False
        self.received_signal: int | None = None
        self._previous_handlers: dict[int, Any] = {}

    def _handle(self, signum: int, _frame: Any) -> None:
        self.preempted = True
        self.received_signal = signum
        logger.warning(
            "Spot preemption signal %s received — scheduling graceful checkpoint & stop.",
            signum,
        )

    def install(self) -> SpotPreemptionMonitor:
        for sig in self.signals:
            self._previous_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, self._handle)
        return self

    def restore(self) -> None:
        for sig, prev in self._previous_handlers.items():
            signal.signal(sig, prev)
        self._previous_handlers.clear()

    def broadcast_preempted(self) -> bool:
        """All-reduce preemption flag across DDP ranks so every rank stops cleanly together."""
        flag = 1 if self.preempted else 0
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            backend = str(dist.get_backend()).lower()
            device = torch.device("cuda", torch.cuda.current_device()) if backend == "nccl" else torch.device("cpu")
            t = torch.tensor([flag], dtype=torch.int32, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            flag = int(t.item())
        self.preempted = bool(flag)
        return self.preempted


class _TrackedDataLoader:
    """Thin wrapper around PyTorch DataLoader that updates ``ResumableLanceMixtureDataset`` cursor."""

    def __init__(self, dataset: ResumableLanceMixtureDataset, dataloader: DataLoader) -> None:
        self._dataset = dataset
        self._dataloader = dataloader
        self._track_in_parent = dataloader.num_workers > 0

    def __iter__(self):
        for batch in self._dataloader:
            if self._track_in_parent:
                self._dataset.advance_delivered(_infer_batch_size(batch, self._dataset.batch_size))
            yield batch

    def __len__(self) -> int:
        return max(1, len(self._dataset) // self._dataset.batch_size)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._dataloader, name)


def build_lance_mixture_dataset(
    dataset_specs: Sequence[dict[str, Any]],
    modality_configs_by_emb: dict[str, dict[str, ModalityConfig]],
    processor: BaseProcessor | None = None,
    *,
    seed: int = 42,
    batch_size: int = 1,
    shard_size: int = 2**10,
    episode_sampling_rate: float = 0.1,
    allow_padding: bool = False,
    epoch_samples: int | None = None,
    episode_pool_size: int = 16,
    prefetch_anchors: int = 16,
    ds_weights_alpha: float | None = None,
    override_pretraining_statistics: bool = True,
    storage_options: dict[str, str] | None = None,
    use_mini_mp4: bool = True,
    infinite: bool = True,
    where: str | None = None,
    lance_version: int | None = None,
) -> ResumableLanceMixtureDataset:
    """Build a ``ResumableLanceMixtureDataset`` from GR00T-style dataset specs.

    Each entry in ``dataset_specs`` has:
        - ``dataset_paths``: list of local dirs or ``s3://`` / ``gs://`` / ``hf://`` URIs
        - ``embodiment_tag``: string or ``EmbodimentTag``
        - ``mix_ratio`` (optional, default 1.0)
        - ``modality_json_path`` (optional sidecar for unmodified v3-Lance datasets)
        - ``storage_options`` (optional per-spec override)
    """
    all_datasets: list[LanceShardedSingleStepDataset] = []
    all_weights: list[float] = []

    for spec in dataset_specs:
        emb_raw = spec["embodiment_tag"]
        emb_key = emb_raw.value if isinstance(emb_raw, EmbodimentTag) else str(emb_raw)
        mix_ratio = float(spec.get("mix_ratio", 1.0))
        mod_json = spec.get("modality_json_path")
        spec_opts = spec.get("storage_options", storage_options)
        spec_where = spec.get("where", where)
        spec_version = spec.get("lance_version", lance_version)

        group_datasets: list[LanceShardedSingleStepDataset] = []
        for path in spec["dataset_paths"]:
            ds = LanceShardedSingleStepDataset(
                dataset_path=path,
                embodiment_tag=emb_raw,
                modality_configs=modality_configs_by_emb[emb_key],
                shard_size=shard_size,
                episode_sampling_rate=episode_sampling_rate,
                seed=seed,
                allow_padding=allow_padding,
                modality_json_path=mod_json,
                storage_options=spec_opts,
                use_mini_mp4=use_mini_mp4,
                where=spec_where,
                lance_version=spec_version,
            )
            group_datasets.append(ds)

        lengths = [sum(ds.get_shard_length(i) for i in range(len(ds))) for ds in group_datasets]
        total_len = sum(lengths)
        for ds, length in zip(group_datasets, lengths, strict=True):
            all_datasets.append(ds)
            all_weights.append((length / total_len) * mix_ratio)

    return ResumableLanceMixtureDataset(
        datasets=all_datasets,
        weights=all_weights,
        processor=processor,
        seed=seed,
        batch_size=batch_size,
        epoch_samples=epoch_samples,
        episode_pool_size=episode_pool_size,
        prefetch_anchors=prefetch_anchors,
        ds_weights_alpha=ds_weights_alpha,
        override_pretraining_statistics=override_pretraining_statistics,
        infinite=infinite,
    )


def make_hf_spot_callback(
    dataset: ResumableLanceMixtureDataset,
    *,
    monitor: SpotPreemptionMonitor | None = None,
    on_checkpoint_saved: Callable[[Path], None] | None = None,
    exit_on_preempt: bool = True,
):
    """Build a HuggingFace ``TrainerCallback`` for ``Gr00tTrainer`` Spot preemption & cursor saving.

    On ``SIGTERM``/``SIGUSR1`` the current step finishes, a full checkpoint plus the exact
    dataloader cursor is written, and (with ``exit_on_preempt``) the process exits with
    status 143 right after training stops, so the caller's post-training work (e.g. GR00T's
    final ``save_model()``, another multi-GB write) does not eat into the provider's
    preemption grace period.

    Lazily imports ``transformers.TrainerCallback`` so the package stays lightweight when
    ``transformers`` is not installed.
    """
    from transformers import TrainerCallback  # type: ignore[import-not-found]

    class LanceSpotPreemptionCallback(TrainerCallback):
        def __init__(self) -> None:
            self.monitor = monitor or SpotPreemptionMonitor()

        def on_train_begin(self, args, state, control, **kwargs):
            self.monitor.install()
            return control

        def on_step_end(self, args, state, control, **kwargs):
            if self.monitor.broadcast_preempted():
                control.should_save = True
                control.should_training_stop = True
            return control

        def on_save(self, args, state, control, **kwargs):
            ckpt_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            save_dataloader_state(
                dataset,
                ckpt_dir,
                extra={"global_step": int(state.global_step), "preempted": self.monitor.preempted},
            )
            if on_checkpoint_saved is not None:
                on_checkpoint_saved(ckpt_dir)
            return control

        def on_train_end(self, args, state, control, **kwargs):
            self.monitor.restore()
            if self.monitor.preempted and exit_on_preempt:
                logger.warning(
                    "Preempted at step %d: checkpoint + dataloader cursor saved; exiting (143).",
                    state.global_step,
                )
                raise SystemExit(143)
            return control

    return LanceSpotPreemptionCallback()


def create_spot_resumable_gr00t_trainer_class(base_trainer_cls: type):
    """Wrap GR00T's ``Gr00tTrainer`` (or HF ``Trainer``) with exact Lance cursor resume.

    Overrides ``get_train_dataloader`` so:
    1. When ``train_dataset`` is a ``ResumableLanceMixtureDataset`` and resuming from a
       checkpoint, it loads ``lance_dataloader_state_rank<R>.json`` instead of calling
       ``reset_seed(seed + global_step)``.
    2. Forces ``multiprocessing_context=lance_mp_context()`` when ``num_workers > 0`` to
       prevent Rust Tokio fork deadlocks.
    3. Wraps the DataLoader in ``_TrackedDataLoader`` so ``state_dict()`` stays exact across
       multi-worker prefetching.
    """

    class SpotResumableGr00tTrainer(base_trainer_cls):  # type: ignore[valid-type, misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._resume_checkpoint_dir: str | None = None
            if kwargs.get("multiprocessing_context") == "fork":
                kwargs["multiprocessing_context"] = lance_mp_context()
            super().__init__(*args, **kwargs)
            if isinstance(self.train_dataset, ResumableLanceMixtureDataset):
                self.add_callback(make_hf_spot_callback(self.train_dataset))

        def train(self, resume_from_checkpoint=None, **kwargs: Any):
            if isinstance(resume_from_checkpoint, (str, Path)):
                self._resume_checkpoint_dir = str(resume_from_checkpoint)
            elif resume_from_checkpoint is True:
                from transformers.trainer_utils import get_last_checkpoint  # type: ignore[import-not-found]

                self._resume_checkpoint_dir = get_last_checkpoint(self.args.output_dir)
            else:
                self._resume_checkpoint_dir = None
            return super().train(resume_from_checkpoint=resume_from_checkpoint, **kwargs)

        def get_train_dataloader(self):
            if not isinstance(self.train_dataset, ResumableLanceMixtureDataset):
                return super().get_train_dataloader()

            self.args.ignore_data_skip = True
            self.train_dataset.batch_size = self._train_batch_size

            if self._resume_checkpoint_dir is not None:
                ckpt_state = dataloader_state_path(self._resume_checkpoint_dir, self.train_dataset.rank)
                if ckpt_state.exists():
                    logger.info("Restoring exact Lance dataloader state from %s", ckpt_state)
                    load_dataloader_state(self.train_dataset, self._resume_checkpoint_dir)
                else:
                    logger.warning(
                        "Checkpoint %s has no %s; starting Lance dataloader from offset 0.",
                        self._resume_checkpoint_dir,
                        ckpt_state.name,
                    )

            data_collator = self._get_collator_with_removed_columns(
                self.data_collator, description="training"
            )
            num_workers = self.args.dataloader_num_workers
            dl_kwargs: dict[str, Any] = {
                "batch_size": self._train_batch_size,
                "collate_fn": data_collator,
                "num_workers": num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "persistent_workers": num_workers > 0,
            }
            if num_workers > 0:
                dl_kwargs["multiprocessing_context"] = lance_mp_context()

            dl = DataLoader(self.train_dataset, **dl_kwargs)
            return _TrackedDataLoader(self.train_dataset, dl)

    SpotResumableGr00tTrainer.__name__ = f"SpotResumable{base_trainer_cls.__name__}"
    return SpotResumableGr00tTrainer


__all__ = [
    "DATALOADER_STATE_PREFIX",
    "DEFAULT_WORKER_PRELOAD",
    "prewarm_lance_workers",
    "SpotPreemptionMonitor",
    "build_lance_mixture_dataset",
    "create_spot_resumable_gr00t_trainer_class",
    "dataloader_state_path",
    "load_dataloader_state",
    "make_hf_spot_callback",
    "save_dataloader_state",
]
