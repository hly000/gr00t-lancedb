# SPDX-License-Identifier: Apache-2.0
"""Exact, backend-agnostic resume for Isaac-GR00T's ``FastShardedMixtureDataset``.

GR00T's fast loader serves a sample order that depends only on the seed, the shard
schedule and how many samples each dataloader worker has served — not on decode timing.
On resume, however, ``Gr00tTrainer`` calls ``reset_seed(seed + global_step)``, which draws
a brand-new order: already-seen samples repeat and unseen ones are skipped.

``make_exact_resume_mixture_class`` subclasses the fast mixture so that, given the number
of batches the trainer already consumed, each worker

1. works out how many samples it served (PyTorch hands batches out round-robin across
   workers, each batch coming from one worker's stream),
2. replays the shuffle-buffer algorithm on placeholders — integer bookkeeping only, no
   decoding — to recover the exact buffer contents, split cursor and RNG state, and
3. decodes only the splits still referenced by the buffer and continues.

Optionally each sample's augmentation RNG (``random``, ``numpy``, ``torch``) is seeded from
its stream position, so augmentations after a resume match the uninterrupted run too.

Works with any ``ShardedDataset`` whose ``load_split((ep, steps))`` returns one item per
step: GR00T's own ``FastShardedSingleStepDataset`` and ``LanceShardedSingleStepDataset``.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import logging
import random
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


def served_by_worker(batches_consumed: int, num_workers: int, worker_id: int, batch_size: int) -> int:
    """Samples worker ``worker_id`` served once the trainer consumed ``batches_consumed`` batches.

    ``torch.utils.data.DataLoader`` (``in_order=True``) takes batch ``k`` from worker
    ``k % num_workers``; with ``num_workers=0`` the main process is the single worker.
    """
    w = max(1, num_workers)
    n = batches_consumed // w + (1 if worker_id < batches_consumed % w else 0)
    return n * batch_size


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


class ExactResumeMixin:
    """Exact-resume behaviour for GR00T's ``FastShardedMixtureDataset`` (use via subclassing).

    Must come first in the MRO, before ``FastShardedMixtureDataset``.
    """

    def __init__(self, *args: Any, per_sample_seed: bool = True, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.per_sample_seed = bool(per_sample_seed)
        self.resume_batches = 0
        self.loader_batch_size: int | None = None

    # -- resume API -------------------------------------------------------------
    def set_resume_point(self, batches_consumed: int, batch_size: int) -> None:
        """Call in the main process before the DataLoader starts its workers."""
        self.resume_batches = int(batches_consumed)
        self.loader_batch_size = int(batch_size)

    def resume_metadata(self) -> dict[str, Any]:
        return {
            "seed": int(self.seed),
            "world_size": int(self.world_size),
            "rank": int(self.rank),
            "shuffle_buffer": int(self.shuffle_buffer),
            "min_fill": int(self.min_fill),
            "decode_threads": int(self.decode_threads),
            "num_shards_per_epoch": int(self.num_shards_per_epoch),
            "per_sample_seed": self.per_sample_seed,
        }

    def reset_seed(self, seed: int) -> None:  # pragma: no cover - guard
        raise RuntimeError(
            "ExactResumeFastShardedMixtureDataset resumes by replay; reset_seed() would "
            "change the sample order. Use set_resume_point() instead."
        )

    # -- iteration ----------------------------------------------------------------
    def filter_shard_sample_schedule(self):
        """GR00T's per-worker shard partition, keyed by a *logical* worker id.

        A fresh DataLoader always asks physical worker 0 for the first batch, but after
        ``B`` consumed batches the next batch belongs to worker ``B % W``. Physical worker
        ``p`` therefore plays logical worker ``(p + B) % W``; every per-worker quantity
        (shard partition, skip count, per-sample seed) uses the logical id.
        """
        from torch.utils.data import get_worker_info

        info = get_worker_info()
        physical, num_workers = (info.id, info.num_workers) if info is not None else (0, 1)
        logical = (physical + int(self.resume_batches)) % num_workers
        if self.worker_id is None:
            self.worker_id, self.num_workers = logical, num_workers
        elif (self.worker_id, self.num_workers) != (logical, num_workers):
            raise AssertionError("worker id / count changed since the schedule was built")
        stride = self.world_size * num_workers
        slot = self.rank * num_workers + logical
        return [shard for i, shard in enumerate(self.shard_sampling_schedule) if i % stride == slot]

    def _sample_seed(self, served: int) -> int:
        ss = np.random.SeedSequence([int(self.seed), int(self.rank), int(self.worker_id or 0), served])
        return int(ss.generate_state(1, dtype=np.uint64)[0] % (2**63 - 1))

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        splits = self._iter_splits(self._worker_schedule())
        max_in_flight = 8 * self.decode_threads

        skip = 0
        if self.resume_batches:
            if not self.loader_batch_size:
                raise ValueError("set_resume_point() needs the DataLoader batch size")
            skip = served_by_worker(
                self.resume_batches, self.num_workers, self.worker_id, self.loader_batch_size
            )

        # ---- 1. replay on placeholders (no I/O) ---------------------------------
        buffer: list = []
        pending: deque = deque()
        split_refs: dict[int, tuple[int, Any]] = {}
        n_split = 0
        served = 0
        while served < skip:
            while len(buffer) < min(self.shuffle_buffer, self.min_fill + served):
                if not pending:
                    dataset_index, split = next(splits)
                    split_refs[n_split] = (dataset_index, split)
                    pending.extend(("ph", n_split, j) for j in range(len(split[1])))
                    n_split += 1
                buffer.append(pending.popleft())
            pick = int(rng.integers(len(buffer)))
            buffer[pick], buffer[-1] = buffer[-1], buffer[pick]
            buffer.pop()
            served += 1

        # ---- 2. materialize splits still referenced ------------------------------
        needed = sorted({ph[1] for ph in list(buffer) + list(pending)})
        loaded: dict[int, list] = {}
        if needed:
            with ThreadPoolExecutor(max_workers=self.decode_threads) as ex:
                futs = {
                    k: ex.submit(self.datasets[split_refs[k][0]].load_split, split_refs[k][1])
                    for k in needed
                }
                for k, fut in futs.items():
                    items = fut.result()
                    if len(items) != len(split_refs[k][1][1]):
                        raise RuntimeError("load_split must return one item per step index")
                    loaded[k] = items

        def real(ph):
            return split_refs[ph[1]][0], loaded[ph[1]][ph[2]]

        buffer = [real(ph) for ph in buffer]
        pending = deque(real(ph) for ph in pending)
        if skip:
            logger.info(
                "rank %d worker %d: resumed after %d served samples (%d splits replayed, %d re-decoded)",
                self.rank, self.worker_id, skip, n_split, len(needed),
            )

        # ---- 3. continue exactly like FastShardedMixtureDataset.__iter__ ----------
        with ThreadPoolExecutor(max_workers=self.decode_threads) as executor:
            in_flight: deque = deque()

            def submit():
                dataset_index, split = next(splits)
                future = executor.submit(self.datasets[dataset_index].load_split, split)
                in_flight.append((dataset_index, future))

            for _ in range(max_in_flight):
                submit()
            while True:
                while len(buffer) < min(self.shuffle_buffer, self.min_fill + served):
                    if not pending:
                        dataset_index, future = in_flight.popleft()
                        pending.extend((dataset_index, item) for item in future.result())
                        submit()
                    buffer.append(pending.popleft())
                pick = int(rng.integers(len(buffer)))
                buffer[pick], buffer[-1] = buffer[-1], buffer[pick]
                dataset_index, item = buffer.pop()
                if self.per_sample_seed:
                    _seed_everything(self._sample_seed(served))
                served += 1
                yield self.datasets[dataset_index].process_split_item(item)


try:  # Module-level concrete class so DataLoader workers (forkserver/spawn) can unpickle it.
    from gr00t.data.dataset.fast_sharded_dataset import (  # type: ignore[import-not-found]
        FastShardedMixtureDataset as _GrootFastMixture,
    )
except ImportError:  # pragma: no cover - GR00T not installed
    _GrootFastMixture = None

if _GrootFastMixture is not None:

    class ExactResumeFastShardedMixtureDataset(ExactResumeMixin, _GrootFastMixture):
        """GR00T ``FastShardedMixtureDataset`` with exact, replay-based resume."""


def make_exact_resume_mixture_class(base_cls: type) -> type:
    """Return the exact-resume subclass of GR00T's ``FastShardedMixtureDataset``.

    For GR00T's own class this is the picklable module-level
    ``ExactResumeFastShardedMixtureDataset``; other bases get a dynamic subclass that only
    works with ``num_workers=0`` or ``fork``.
    """
    if _GrootFastMixture is not None and base_cls is _GrootFastMixture:
        return ExactResumeFastShardedMixtureDataset
    return type(f"ExactResume{base_cls.__name__}", (ExactResumeMixin, base_cls), {})


def make_exact_resume_trainer_class(base_trainer_cls: type) -> type:
    """Wrap ``Gr00tTrainer`` so resume uses ``set_resume_point`` instead of ``reset_seed``.

    Also installs the SIGTERM/SIGUSR1 preemption callback (checkpoint + exit 143).
    """
    from torch.utils.data import DataLoader

    from .trainer_integration import SpotPreemptionMonitor

    class ExactResumeGr00tTrainer(base_trainer_cls):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.add_callback(_make_preempt_callback(SpotPreemptionMonitor()))

        def get_train_dataloader(self):
            ds = self.train_dataset
            if not hasattr(ds, "set_resume_point"):
                return super().get_train_dataloader()
            self.args.ignore_data_skip = True
            batches = int(self.state.global_step) * int(self.args.gradient_accumulation_steps)
            ds.set_resume_point(batches, self._train_batch_size)
            if batches:
                logger.info(
                    "Exact resume: replaying %d consumed batches (global_step=%d) instead of reseeding",
                    batches, self.state.global_step,
                )
            collator = self._get_collator_with_removed_columns(self.data_collator, description="training")
            nw = self.args.dataloader_num_workers
            params: dict[str, Any] = {
                "batch_size": self._train_batch_size,
                "collate_fn": collator,
                "num_workers": nw,
                "pin_memory": self.args.dataloader_pin_memory,
                "persistent_workers": nw > 0,
            }
            if nw > 0:
                params["multiprocessing_context"] = getattr(self, "multiprocessing_context", "fork")
            return DataLoader(ds, **params)

    ExactResumeGr00tTrainer.__name__ = f"ExactResume{base_trainer_cls.__name__}"
    return ExactResumeGr00tTrainer


def _make_preempt_callback(monitor):
    from transformers import TrainerCallback

    class PreemptCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            monitor.install()

        def on_step_end(self, args, state, control, **kwargs):
            if monitor.broadcast_preempted():
                control.should_save = True
                control.should_training_stop = True

        def on_train_end(self, args, state, control, **kwargs):
            monitor.restore()
            if monitor.preempted:
                logger.warning("Preempted at step %d: checkpoint saved; exiting (143).", state.global_step)
                raise SystemExit(143)

    return PreemptCallback()


__all__ = [
    "ExactResumeMixin",
    "make_exact_resume_mixture_class",
    "make_exact_resume_trainer_class",
    "served_by_worker",
]
