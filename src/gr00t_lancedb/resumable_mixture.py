# SPDX-License-Identifier: Apache-2.0
"""Multi-dataset weighted mixture with exact-once Spot-preemption checkpoint & resume.

Solves the key limitation of GR00T's ``Gr00tTrainer.get_train_dataloader()``, which
discards dataloader state on resume and re-seeds with ``seed + global_step`` (causing
non-reproducible training trajectories, duplicate samples, and dropped episodes under
frequent Spot GPU preemptions).

Key properties:
- **Multi-dataset weighted mixing** (``mix_ratio``, ``ds_weights_alpha``) with exact
  per-embodiment ``merge_statistics`` matching GR00T's ``ShardedMixtureDataset``.
- **Bounded episode-pool ordering** per constituent dataset for S3 MP4 cache locality
  and coalesced range reads via ``prefetch_anchors``.
- **Exact multi-worker resume** across ``spawn`` / ``forkserver`` DataLoader workers
  using a tiny ~150-byte state dict (``{version, seed, epoch, epoch_offset, ...}``)
  with **zero skipped-sample video decoding** and **0 duplicates / 0 missed samples**.
- **Per-anchor deterministic RNG** around ``processor(...)`` and language choice so
  restarting after preemption yields identical augmented tensors.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable, Iterator
from copy import copy
from itertools import islice
import os
import random
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from lerobot_lancedb import lance_mp_context

from ._gr00t_compat import BaseProcessor, MessageType
from .metadata import merge_statistics
from .sharded_dataset import LanceShardedSingleStepDataset

STATE_DICT_VERSION = 1


def _resolve_distributed(rank: int | None, world_size: int | None) -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        default_rank, default_size = dist.get_rank(), dist.get_world_size()
    else:
        default_rank = int(os.getenv("RANK", "0"))
        default_size = int(os.getenv("WORLD_SIZE", "1"))
    r = int(default_rank if rank is None else rank)
    ws = int(default_size if world_size is None else world_size)
    if not (0 <= r < ws):
        raise ValueError(f"Invalid rank {r} for world_size {ws}")
    return r, ws


def _worker_cursor(totals: list[int], offset: int, batch_size: int) -> tuple[list[int], int]:
    """Replay PyTorch round-robin multi-worker batch delivery to exact worker offsets.

    Mirrors ``lerobot_lancedb.episode_pool._worker_cursor`` so resuming with
    ``num_workers > 1`` starts each physical worker on the exact logical shard and
    sample offset where the previous run stopped.
    """
    consumed = [0] * len(totals)
    worker = 0
    remaining = offset
    while remaining > 0:
        avail = totals[worker] - consumed[worker]
        if avail <= 0:
            worker = (worker + 1) % len(totals)
            continue
        size = min(batch_size, avail)
        if size > remaining:
            raise ValueError(
                f"Multi-worker resume requires offset ({offset}) to land on a delivered batch "
                f"boundary for batch_size={batch_size} (stuck with remaining={remaining} < size={size})."
            )
        consumed[worker] += size
        remaining -= size
        worker = (worker + 1) % len(totals)
    # Advance rotation past any exhausted worker so physical worker 0 starts on a live shard.
    if any(consumed[w] < totals[w] for w in range(len(totals))):
        while consumed[worker] >= totals[worker]:
            worker = (worker + 1) % len(totals)
    return consumed, worker


def _exact_episode_pool_anchors(
    episode_lengths: list[int],
    pool_size: int,
    seed: int,
    epoch: int,
    cycle: int,
    dataset_index: int,
    strategy: str = "remaining",
) -> Iterator[tuple[int, int]]:
    """Yield every valid ``(ep_idx, step_idx)`` in one dataset exactly once with bounded pool locality."""
    counts = {ep: length for ep, length in enumerate(episode_lengths) if length > 0}
    if not counts:
        return
    rng = np.random.default_rng([seed, epoch, cycle, dataset_index])
    pending = deque(int(ep) for ep in rng.permutation(sorted(counts)))
    resident: dict[int, list[int]] = {}
    turn: list[int] = []

    while resident or pending:
        while pending and len(resident) < pool_size:
            ep = pending.popleft()
            resident[ep] = rng.permutation(counts[ep]).tolist()
        if strategy == "round_robin":
            if not turn:
                turn = list(resident)
                rng.shuffle(turn)
            ep = turn.pop()
        else:
            total_rem = sum(len(v) for v in resident.values())
            target = int(rng.integers(total_rem))
            for cand in resident:
                target -= len(resident[cand])
                if target < 0:
                    ep = cand
                    break
        step = resident[ep].pop()
        if not resident[ep]:
            del resident[ep]
        yield ep, step


def _seed_anchor_rngs(seed: int, epoch: int, global_pos: int):
    """Context manager seeding Python, NumPy, and CPU Torch RNGs per anchor deterministically."""
    seed64, seed32 = np.random.SeedSequence([seed, epoch, global_pos]).generate_state(2, dtype=np.uint64)

    class _AnchorSeedCtx:
        def __enter__(self_ctx):
            self_ctx._py_state = random.getstate()
            self_ctx._np_state = np.random.get_state()
            self_ctx._torch_fork = torch.random.fork_rng(devices=[])
            self_ctx._torch_fork.__enter__()
            torch.default_generator.manual_seed(int(seed64))
            random.seed(int(seed64))
            np.random.seed(int(seed32) % (1 << 32))
            return np.random.default_rng(int(seed64))

        def __exit__(self_ctx, exc_type, exc, tb):
            self_ctx._torch_fork.__exit__(exc_type, exc, tb)
            random.setstate(self_ctx._py_state)
            np.random.set_state(self_ctx._np_state)
            return False

    return _AnchorSeedCtx()


class ResumableLanceMixtureDataset(IterableDataset):
    """Deterministic, preemption-resumable multi-dataset mixture for GR00T on LanceDB.

    Args:
        datasets: List of ``LanceShardedSingleStepDataset`` instances.
        weights: Relative sampling weights for each dataset (normalized automatically).
        processor: Optional GR00T ``BaseProcessor`` (configured with merged per-embodiment stats).
        seed: Rank-symmetric base random seed.
        batch_size: Per-rank micro-batch size (used for exact multi-worker resume cursors).
        epoch_samples: Number of global samples per logical epoch. When ``None``, defaults to
            the total number of valid anchors summed across all datasets (exact 1x coverage for
            single-dataset training, proportional exact-cycle mixing for multi-dataset training).
        episode_pool_size: Max resident episodes per dataset for S3 MP4 byte-range locality.
        prefetch_anchors: Number of upcoming anchors coalesced per Lance read batch per worker.
        sampling_strategy: ``"remaining"`` (weighted by remaining frames in resident pool) or
            ``"round_robin"``.
        infinite: If ``True`` (matching GR00T's ``ShardedMixtureDataset`` for ``max_steps``
            training), seamlessly advances ``epoch += 1`` when an epoch finishes. If ``False``,
            stops cleanly after one epoch.
        override_pretraining_statistics: Passed to ``processor.set_statistics(...)``.
        ds_weights_alpha: Optional ``len^alpha`` weight override matching GR00T's ``DatasetFactory``.
        rank: Override distributed rank (defaults to ``torch.distributed`` / ``RANK`` env / 0).
        world_size: Override distributed world size.
    """

    def __init__(
        self,
        datasets: list[LanceShardedSingleStepDataset],
        weights: list[float],
        processor: BaseProcessor | None = None,
        *,
        seed: int = 42,
        batch_size: int = 1,
        epoch_samples: int | None = None,
        episode_pool_size: int = 16,
        prefetch_anchors: int = 16,
        sampling_strategy: str = "remaining",
        infinite: bool = True,
        override_pretraining_statistics: bool = False,
        ds_weights_alpha: float | None = None,
        rank: int | None = None,
        world_size: int | None = None,
    ) -> None:
        super().__init__()
        if not datasets:
            raise ValueError("ResumableLanceMixtureDataset requires at least one dataset.")
        if len(datasets) != len(weights):
            raise ValueError("datasets and weights must have the same length.")
        if batch_size < 1 or episode_pool_size < 1 or prefetch_anchors < 1:
            raise ValueError("batch_size, episode_pool_size, and prefetch_anchors must be >= 1.")
        if sampling_strategy not in ("remaining", "round_robin"):
            raise ValueError(f"Unknown sampling_strategy: {sampling_strategy!r}")

        self.datasets = datasets
        self.processor = processor
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.episode_pool_size = int(episode_pool_size)
        self.prefetch_anchors = int(prefetch_anchors)
        self.sampling_strategy = sampling_strategy
        self.infinite = bool(infinite)
        self.override_pretraining_statistics = bool(override_pretraining_statistics)

        # Effective lengths per episode for every dataset
        self._ds_ep_lengths: list[list[int]] = [
            [ds.get_effective_episode_length(ep) for ep in range(len(ds.episode_loader))]
            for ds in datasets
        ]
        self._ds_total_anchors: list[int] = [sum(lens) for lens in self._ds_ep_lengths]
        if any(total == 0 for total in self._ds_total_anchors):
            raise ValueError("Every constituent dataset must have at least one valid anchor step.")

        if ds_weights_alpha is not None and len(datasets) > 1:
            lengths = np.asarray(self._ds_total_anchors, dtype=np.float64)
            raw_weights = (np.power(lengths, ds_weights_alpha) / np.power(lengths[0], ds_weights_alpha)).tolist()
        else:
            raw_weights = [float(w) for w in weights]
        w_arr = np.asarray(raw_weights, dtype=np.float64)
        self.weights: list[float] = (w_arr / w_arr.sum()).tolist()

        self.epoch_samples: int = (
            int(epoch_samples) if epoch_samples is not None else int(sum(self._ds_total_anchors))
        )

        self.rank, self.world_size = _resolve_distributed(rank, world_size)
        self._assert_seed_rank_symmetric(self.seed)

        # Merge statistics across datasets by embodiment and configure processor
        self.global_stats: dict[str, Any] = {}
        self.merge_statistics()

        # Exact resumable cursor state (rank-local)
        self.epoch: int = 0
        self.epoch_offset: int = 0
        self.total_consumed: int = 0
        self._resume_epoch_offset: int = 0

    def _assert_seed_rank_symmetric(self, seed: int) -> None:
        if self.world_size <= 1 or not (dist.is_available() and dist.is_initialized()):
            return
        backend = str(dist.get_backend()).lower()
        device = torch.device("cuda", torch.cuda.current_device()) if backend == "nccl" else torch.device("cpu")
        seed_t = torch.tensor([seed], dtype=torch.long, device=device)
        gathered = [torch.zeros_like(seed_t) for _ in range(self.world_size)]
        dist.all_gather(gathered, seed_t)
        seeds = [int(s.item()) for s in gathered]
        if any(s != seed for s in seeds):
            raise ValueError(f"Seed must be identical across all ranks; got {seeds}.")

    def merge_statistics(self) -> None:
        all_stats_by_emb: dict[str, list[dict[str, Any]]] = {}
        weights_by_emb: dict[str, list[float]] = {}
        for ds, w in zip(self.datasets, self.weights, strict=True):
            emb = ds.embodiment_tag.value
            all_stats_by_emb.setdefault(emb, []).append(ds.get_dataset_statistics())
            weights_by_emb.setdefault(emb, []).append(w)

        stats_by_emb: dict[str, Any] = {}
        for emb, stats_list in all_stats_by_emb.items():
            stats_by_emb[emb] = {}
            for modality in ("state", "action", "relative_action"):
                if modality in stats_list[0]:
                    stats_by_emb[emb][modality] = merge_statistics(
                        per_dataset_stats=[s[modality] for s in stats_list],
                        dataset_sampling_weights=weights_by_emb[emb],
                        is_relative_stats=(modality == "relative_action"),
                    )

        self.global_stats = stats_by_emb
        if self.processor is not None:
            self.processor.set_statistics(
                self.global_stats, override=self.override_pretraining_statistics
            )
            for ds in self.datasets:
                ds.set_processor(self.processor)

    def get_dataset_statistics(self) -> dict[str, Any]:
        return self.global_stats

    def get_initial_actions(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for ds in self.datasets:
            out.extend(ds.get_initial_actions())
        return out

    def rank_epoch_length(self) -> int:
        """Exact number of samples owned by this rank in one logical epoch."""
        base = self.epoch_samples // self.world_size
        rem = self.epoch_samples % self.world_size
        return base + (1 if self.rank < rem else 0)

    def __len__(self) -> int:
        return self.rank_epoch_length()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.epoch_offset = 0
        self._resume_epoch_offset = 0

    def reset_seed(self, seed: int) -> None:
        """Provided for API compatibility with ``ShardedMixtureDataset`` (prefer ``load_state_dict``)."""
        self.seed = int(seed)
        self._assert_seed_rank_symmetric(self.seed)
        self.epoch = 0
        self.epoch_offset = 0
        self.total_consumed = 0
        self._resume_epoch_offset = 0

    def state_dict(self) -> dict[str, Any]:
        """Return a minimal JSON-serializable checkpoint capturing the exact sampler cursor."""
        return {
            "version": STATE_DICT_VERSION,
            "seed": self.seed,
            "epoch": self.epoch,
            "epoch_offset": self.epoch_offset,
            "total_consumed": self.total_consumed,
            "batch_size": self.batch_size,
            "world_size": self.world_size,
            "epoch_samples": self.epoch_samples,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore the exact epoch & rank sample cursor from a checkpoint."""
        version = int(state.get("version", 0))
        if version != STATE_DICT_VERSION:
            raise ValueError(f"Unsupported ResumableLanceMixtureDataset state version: {version}")
        if int(state["world_size"]) != self.world_size:
            raise ValueError(
                f"Cannot exact-resume across world_size change ({state['world_size']} -> {self.world_size})."
            )
        if int(state["epoch_samples"]) != self.epoch_samples:
            raise ValueError(
                f"epoch_samples mismatch on resume ({state['epoch_samples']} -> {self.epoch_samples})."
            )
        epoch = int(state["epoch"])
        epoch_offset = int(state["epoch_offset"])
        batch_size = int(state["batch_size"])
        rank_len = self.rank_epoch_length()
        if epoch < 0 or not (0 <= epoch_offset <= rank_len) or batch_size < 1:
            raise ValueError(f"Invalid resumable mixture checkpoint state: {state!r}")
        if epoch_offset == rank_len:
            epoch += 1
            epoch_offset = 0
        self.seed = int(state["seed"])
        self.epoch = epoch
        self.epoch_offset = epoch_offset
        self._resume_epoch_offset = epoch_offset
        self.total_consumed = int(state.get("total_consumed", epoch * rank_len + epoch_offset))
        self.batch_size = batch_size

    def _build_global_epoch_schedule(self, epoch: int) -> list[tuple[int, int, int, int]]:
        """Build deterministic ``(global_pos, ds_idx, ep_idx, step_idx)`` for one epoch.

        - Single dataset: pure exact-coverage episode pool (every anchor appears exactly once).
        - Multi-dataset: deterministic weighted dataset draw pulling from each dataset's
          exact-coverage episode pool cycle (0 duplicates within every cycle).
        """
        n_ds = len(self.datasets)
        if n_ds == 1:
            schedule: list[tuple[int, int, int, int]] = []
            cycle = 0
            while len(schedule) < self.epoch_samples:
                for ep_idx, step_idx in _exact_episode_pool_anchors(
                    self._ds_ep_lengths[0],
                    self.episode_pool_size,
                    self.seed,
                    epoch,
                    cycle,
                    0,
                    self.sampling_strategy,
                ):
                    schedule.append((len(schedule), 0, ep_idx, step_idx))
                    if len(schedule) >= self.epoch_samples:
                        break
                cycle += 1
            return schedule

        mix_rng = np.random.default_rng([self.seed, epoch, 0x4D4958])
        ds_choices = mix_rng.choice(n_ds, size=self.epoch_samples, p=self.weights)

        cycles = [0] * n_ds
        iters: list[Iterator[tuple[int, int]]] = [
            _exact_episode_pool_anchors(
                self._ds_ep_lengths[d],
                self.episode_pool_size,
                self.seed,
                epoch,
                0,
                d,
                self.sampling_strategy,
            )
            for d in range(n_ds)
        ]

        schedule = []
        for global_pos, d_raw in enumerate(ds_choices):
            d = int(d_raw)
            try:
                ep_idx, step_idx = next(iters[d])
            except StopIteration:
                cycles[d] += 1
                iters[d] = _exact_episode_pool_anchors(
                    self._ds_ep_lengths[d],
                    self.episode_pool_size,
                    self.seed,
                    epoch,
                    cycles[d],
                    d,
                    self.sampling_strategy,
                )
                ep_idx, step_idx = next(iters[d])
            schedule.append((global_pos, d, ep_idx, step_idx))
        return schedule

    def _rank_schedule(self, epoch: int) -> list[tuple[int, int, int, int]]:
        global_schedule = self._build_global_epoch_schedule(epoch)
        return global_schedule[self.rank :: self.world_size]

    def _emit_chunk(
        self,
        local_datasets: list[LanceShardedSingleStepDataset],
        chunk: list[tuple[int, int, int, int]],
        epoch: int,
    ) -> list[Any]:
        """Coalesce Lance reads per dataset across ``chunk``, then emit in exact stream order."""
        grouped: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
        for out_idx, (global_pos, ds_idx, ep_idx, step_idx) in enumerate(chunk):
            grouped[ds_idx].append((out_idx, global_pos, ep_idx, step_idx))

        results: list[Any] = [None] * len(chunk)
        for ds_idx, entries in grouped.items():
            ds = local_datasets[ds_idx]
            pairs = [(ep_idx, step_idx) for _, _, ep_idx, step_idx in entries]
            # Deterministic language choice RNG seeded per batch anchor
            batch_rng = np.random.default_rng([self.seed, epoch, entries[0][1]])
            steps = ds.load_anchor_batch(pairs, rng=batch_rng)
            for (out_idx, global_pos, ep_idx, step_idx), step_data in zip(entries, steps, strict=True):
                # Attach provenance metadata useful for debugging & exact-resume audits
                step_data.metadata.update(
                    {
                        "dataset_index": ds_idx,
                        "episode_index": ep_idx,
                        "step_index": step_idx,
                        "epoch": epoch,
                        "global_pos": global_pos,
                    }
                )
                with _seed_anchor_rngs(self.seed, epoch, global_pos):
                    results[out_idx] = ds.process_split_item(step_data)
        return results

    def __iter__(self) -> Iterator[Any]:
        worker = get_worker_info()
        num_workers, phys_worker_id = (worker.num_workers, worker.id) if worker else (1, 0)

        # Shallow-copy datasets per worker so Lance handles/pools remain worker-private.
        local_datasets: list[LanceShardedSingleStepDataset] = []
        for ds in self.datasets:
            ds_copy = copy(ds)
            ep_loader_copy = copy(ds.episode_loader)
            ep_loader_copy.reader = copy(ds.episode_loader.reader)
            ds_copy.episode_loader = ep_loader_copy
            local_datasets.append(ds_copy)

        epoch = self.epoch
        resume_offset = self._resume_epoch_offset
        self._resume_epoch_offset = 0

        try:
            while True:
                rank_items = self._rank_schedule(epoch)
                worker_shards = [rank_items[w::num_workers] for w in range(num_workers)]
                totals = [len(s) for s in worker_shards]

                if resume_offset > 0:
                    if num_workers > 1:
                        consumed, rotation = _worker_cursor(totals, resume_offset, self.batch_size)
                        logical_worker = (rotation + phys_worker_id) % num_workers
                        worker_offset = consumed[logical_worker]
                    else:
                        logical_worker = 0
                        worker_offset = resume_offset
                else:
                    logical_worker = phys_worker_id
                    worker_offset = 0

                shard = worker_shards[logical_worker]
                stream = islice(shard, worker_offset, None)
                while chunk := list(islice(stream, self.prefetch_anchors)):
                    for sample in self._emit_chunk(local_datasets, chunk, epoch):
                        if worker is None:
                            self.epoch = epoch
                            self.epoch_offset += 1
                            self.total_consumed += 1
                        yield sample

                if not self.infinite:
                    break
                epoch += 1
                resume_offset = 0
                if worker is None:
                    self.epoch = epoch
                    self.epoch_offset = 0
        finally:
            for ds_copy in local_datasets:
                ds_copy.close()

    def advance_delivered(self, n_samples: int) -> None:
        """Advance parent-process cursor after consuming a batch of ``n_samples``."""
        rank_len = self.rank_epoch_length()
        self.total_consumed += n_samples
        self.epoch_offset += n_samples
        while self.epoch_offset >= rank_len:
            self.epoch_offset -= rank_len
            self.epoch += 1

    def loader(
        self,
        *,
        num_workers: int = 0,
        collate_fn: Callable | None = None,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        **dataloader_kwargs: Any,
    ) -> Iterator[Any]:
        """Parent-tracked DataLoader iterator keeping ``state_dict()`` exact across workers.

        Automatically selects ``lance_mp_context()`` (``forkserver``/``spawn``) whenever
        ``num_workers > 0`` so LanceDB's Rust runtime never deadlocks under ``fork``.
        """
        if dataloader_kwargs.get("drop_last"):
            raise ValueError("Exact checkpointed iteration requires drop_last=False.")
        if num_workers > 0:
            dataloader_kwargs.setdefault("multiprocessing_context", lance_mp_context())
            if dataloader_kwargs["multiprocessing_context"] == "fork":
                raise ValueError(
                    "LanceDB cannot run with multiprocessing_context='fork' (Rust Tokio deadlock). "
                    f"Use '{lance_mp_context()}'."
                )

        dl = DataLoader(
            self,
            batch_size=self.batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers and num_workers > 0,
            **dataloader_kwargs,
        )

        # In single-process mode (__iter__ runs in parent), __iter__ already advances
        # self.epoch / self.epoch_offset; only track in parent when num_workers > 0!
        track_in_parent = num_workers > 0
        iterator = iter(dl)
        try:
            for batch in iterator:
                if track_in_parent:
                    batch_n = _infer_batch_size(batch, self.batch_size)
                    self.advance_delivered(batch_n)
                yield batch
        finally:
            shutdown = getattr(iterator, "_shutdown_workers", None)
            if shutdown is not None:
                shutdown()
            self._resume_epoch_offset = 0

    def close(self) -> None:
        for ds in self.datasets:
            ds.close()


def _infer_batch_size(batch: Any, fallback: int) -> int:
    if isinstance(batch, list):
        return len(batch)
    if isinstance(batch, dict):
        for v in batch.values():
            if isinstance(v, (torch.Tensor, np.ndarray, list)):
                return len(v)
    if isinstance(batch, (torch.Tensor, np.ndarray)):
        return len(batch)
    return fallback


__all__ = ["ResumableLanceMixtureDataset", "STATE_DICT_VERSION"]
