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
3. either decodes the splits still referenced by the buffer (``lazy_decode=False``) or
   keeps the shuffle buffer anchor-only and batch-decodes only chosen anchors on demand
   (``lazy_decode=True``), yielding bit-identical sample order and tensors with ~32 KB
   buffer RAM and ~0 ms resume replay I/O.

Optionally each sample's augmentation RNG (``random``, ``numpy``, ``torch``) is seeded from
its stream position, so augmentations after a resume match the uninterrupted run too.

Works with any ``ShardedDataset`` whose ``load_split((ep, steps))`` returns one item per
step: GR00T's own ``FastShardedSingleStepDataset`` and ``LanceShardedSingleStepDataset``.
"""

from __future__ import annotations

from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
from pathlib import Path
import random
import shutil
import time
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

EMERGENCY_CKPT_FILENAME = "emergency_trainable.pt"
_FULL_WEIGHT_FILES = (
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
    "adapter_model.safetensors",
    "adapter_model.bin",
)


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


def _unwrap_model(model: Any) -> Any:
    while hasattr(model, "module"):
        model = model.module
    return model


def is_emergency_trainable_checkpoint(checkpoint_dir: str | Path | None) -> bool:
    """Return True if ``checkpoint_dir`` is a lightweight trainable-only emergency checkpoint."""
    if checkpoint_dir is None:
        return False
    p = Path(checkpoint_dir)
    if not (p / EMERGENCY_CKPT_FILENAME).is_file():
        return False
    return not any((p / fname).exists() for fname in _FULL_WEIGHT_FILES)


def _capture_rng_state() -> dict[str, Any]:
    rng: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        try:
            rng["cuda"] = torch.cuda.random.get_rng_state_all()
        except Exception:  # noqa: BLE001
            pass
    return rng


def _restore_rng_state(rng: dict[str, Any] | None) -> None:
    if not rng:
        return
    if "python" in rng:
        random.setstate(rng["python"])
    if "numpy" in rng:
        np.random.set_state(rng["numpy"])
    if "cpu" in rng:
        cpu_state = rng["cpu"]
        if not isinstance(cpu_state, torch.ByteTensor):
            cpu_state = torch.as_tensor(cpu_state, dtype=torch.uint8)
        torch.random.set_rng_state(cpu_state)
    if "cuda" in rng and torch.cuda.is_available():
        try:
            torch.cuda.random.set_rng_state_all(rng["cuda"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to restore CUDA RNG state from emergency checkpoint: %s", exc)


def save_emergency_trainable_checkpoint(
    output_dir: str | Path,
    state: Any,
    model: Any,
    optimizer: Any = None,
    lr_scheduler: Any = None,
) -> Path:
    """Atomically save ONLY trainable parameters + optimizer + scheduler + RNG + trainer_state.json.

    Writes into a staging directory first and atomically commits ``checkpoint-<step>`` so a
    hard ``SIGKILL`` mid-write never leaves a half-written checkpoint visible to
    ``get_last_checkpoint()``.
    """
    step = int(state.global_step)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    target_dir = out_root / f"checkpoint-{step}"
    tmp_dir = out_root / f".tmp-checkpoint-{step}-{os.getpid()}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    raw_model = _unwrap_model(model)
    trainable_state = {
        k: v.detach().cpu().clone()
        for k, v in raw_model.named_parameters()
        if v.requires_grad
    }
    payload = {
        "format": "gr00t_emergency_trainable_v1",
        "global_step": step,
        "trainable_state_dict": trainable_state,
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "lr_scheduler_state_dict": lr_scheduler.state_dict() if lr_scheduler is not None else None,
        "rng_state": _capture_rng_state(),
    }

    try:
        torch.save(payload, tmp_dir / EMERGENCY_CKPT_FILENAME)
        state_json_path = tmp_dir / "trainer_state.json"
        if hasattr(state, "save_to_json"):
            state.save_to_json(str(state_json_path))
        else:
            state_json_path.write_text(json.dumps(vars(state), default=str, indent=2), encoding="utf-8")

        if target_dir.exists():
            # Replace files atomically inside existing checkpoint dir if same step already exists.
            os.replace(tmp_dir / EMERGENCY_CKPT_FILENAME, target_dir / EMERGENCY_CKPT_FILENAME)
            os.replace(state_json_path, target_dir / "trainer_state.json")
            shutil.rmtree(tmp_dir, ignore_errors=True)
        else:
            os.replace(tmp_dir, target_dir)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    return target_dir


def _apply_trainable_weights(model: Any, trainable_state_dict: dict[str, torch.Tensor]) -> int:
    """Copy saved trainable parameter tensors in-place onto ``model`` (preserving param identity)."""
    raw_model = _unwrap_model(model)
    named_params = dict(raw_model.named_parameters())
    missing = [k for k in trainable_state_dict if k not in named_params]
    if missing:
        raise KeyError(
            f"Emergency trainable checkpoint has {len(missing)} parameter(s) not found in model "
            f"(e.g. {missing[:5]})"
        )
    with torch.no_grad():
        for k, tensor in trainable_state_dict.items():
            param = named_params[k]
            param.data.copy_(tensor.to(device=param.device, dtype=param.dtype))
    return len(trainable_state_dict)


def load_emergency_trainable_checkpoint(
    checkpoint_dir: str | Path,
    model: Any,
    optimizer: Any = None,
    lr_scheduler: Any = None,
    *,
    load_weights: bool = True,
    load_optim: bool = True,
    load_rng: bool = True,
) -> dict[str, Any]:
    """Load an ``emergency_trainable.pt`` checkpoint into model / optimizer / scheduler / RNG."""
    ckpt_file = Path(checkpoint_dir) / EMERGENCY_CKPT_FILENAME
    payload = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    if load_weights and model is not None:
        _apply_trainable_weights(model, payload["trainable_state_dict"])
    if load_optim:
        if optimizer is not None and payload.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        if lr_scheduler is not None and payload.get("lr_scheduler_state_dict") is not None:
            lr_scheduler.load_state_dict(payload["lr_scheduler_state_dict"])
    if load_rng:
        _restore_rng_state(payload.get("rng_state"))
    return payload


class ExactResumeMixin:
    """Exact-resume behaviour for GR00T's ``FastShardedMixtureDataset`` (use via subclassing).

    Must come first in the MRO, before ``FastShardedMixtureDataset``.
    """

    def __init__(
        self,
        *args: Any,
        per_sample_seed: bool = True,
        lazy_decode: bool = False,
        decode_batch_size: int = 32,
        prefetch_batches: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if decode_batch_size < 1:
            raise ValueError(f"decode_batch_size must be >= 1, got {decode_batch_size}")
        if prefetch_batches < 0:
            raise ValueError(f"prefetch_batches must be >= 0, got {prefetch_batches}")
        self.per_sample_seed = bool(per_sample_seed)
        self.lazy_decode = bool(lazy_decode)
        self.decode_batch_size = int(decode_batch_size)
        self.prefetch_batches = int(prefetch_batches)
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
            "lazy_decode": self.lazy_decode,
            "decode_batch_size": self.decode_batch_size,
            "prefetch_batches": self.prefetch_batches,
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

    def _decode_anchor_chunk(self, chunk: list[tuple[int, int, int]]) -> list[tuple[int, Any]]:
        """Batch-decode ``(dataset_index, ep_idx, step_idx)`` anchors, preserving exact order.

        Uses ``ds.load_anchor_batch`` when available (coalescing cross-episode Lance reads and
        video decodes into one call), otherwise falls back to episode-grouped ``ds.load_split``.
        """
        grouped: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
        for pos, (ds_idx, ep_idx, step_idx) in enumerate(chunk):
            grouped[ds_idx].append((pos, ep_idx, step_idx))

        decoded: list[Any] = [None] * len(chunk)
        for ds_idx, entries in grouped.items():
            ds = self.datasets[ds_idx]
            if hasattr(ds, "load_anchor_batch"):
                items = ds.load_anchor_batch([(ep_idx, step_idx) for _, ep_idx, step_idx in entries])
                if len(items) != len(entries):
                    raise RuntimeError("load_anchor_batch must return one item per anchor")
                for (pos, _, _), item in zip(entries, items, strict=True):
                    decoded[pos] = (ds_idx, item)
            else:
                ep_groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
                for pos, ep_idx, step_idx in entries:
                    ep_groups[ep_idx].append((pos, step_idx))
                for ep_idx, ep_entries in ep_groups.items():
                    steps_arr = np.asarray([s for _, s in ep_entries], dtype=np.int64)
                    items = ds.load_split((ep_idx, steps_arr))
                    if len(items) != len(ep_entries):
                        raise RuntimeError("load_split must return one item per step index")
                    for (pos, _), item in zip(ep_entries, items, strict=True):
                        decoded[pos] = (ds_idx, item)
        return decoded

    def _iter_lazy_decode(self, rng: np.random.Generator, splits, skip: int):
        """Anchor-only shuffle buffer + chunked prefetch decode with bit-identical order."""
        buffer: list[tuple[int, int, int]] = []
        pending: deque[tuple[int, int, int]] = deque()
        served = 0

        def draw_one() -> tuple[int, int, int]:
            nonlocal served
            while len(buffer) < min(self.shuffle_buffer, self.min_fill + served):
                if not pending:
                    dataset_index, split = next(splits)
                    ep_idx, step_indices = split
                    pending.extend((dataset_index, int(ep_idx), int(s)) for s in step_indices)
                buffer.append(pending.popleft())
            pick = int(rng.integers(len(buffer)))
            buffer[pick], buffer[-1] = buffer[-1], buffer[pick]
            chosen = buffer.pop()
            served += 1
            return chosen

        # Integer-only skip on resume: zero buffer decoding!
        while served < skip:
            draw_one()
        if skip:
            logger.info(
                "rank %d worker %d: lazy-decode resumed after %d served samples (0 buffer splits decoded)",
                self.rank, self.worker_id, skip,
            )

        def next_chunk() -> list[tuple[int, int, int, int]]:
            chunk_meta: list[tuple[int, int, int, int]] = []
            for _ in range(self.decode_batch_size):
                s_idx = served
                ds_idx, ep_idx, step_idx = draw_one()
                chunk_meta.append((s_idx, ds_idx, ep_idx, step_idx))
            return chunk_meta

        if self.prefetch_batches <= 0:
            while True:
                chunk_meta = next_chunk()
                decoded = self._decode_anchor_chunk([(d, e, s) for _, d, e, s in chunk_meta])
                for (s_idx, _, _, _), (dataset_index, item) in zip(chunk_meta, decoded, strict=True):
                    if self.per_sample_seed:
                        _seed_everything(self._sample_seed(s_idx))
                    yield self.datasets[dataset_index].process_split_item(item)
            return

        with ThreadPoolExecutor(max_workers=1) as executor:
            in_flight: deque = deque()

            def enqueue() -> None:
                chunk_meta = next_chunk()
                anchors = [(d, e, s) for _, d, e, s in chunk_meta]
                in_flight.append((chunk_meta, executor.submit(self._decode_anchor_chunk, anchors)))

            for _ in range(self.prefetch_batches):
                enqueue()

            while True:
                chunk_meta, fut = in_flight.popleft()
                decoded = fut.result()
                enqueue()
                for (s_idx, _, _, _), (dataset_index, item) in zip(chunk_meta, decoded, strict=True):
                    if self.per_sample_seed:
                        _seed_everything(self._sample_seed(s_idx))
                    yield self.datasets[dataset_index].process_split_item(item)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        splits = self._iter_splits(self._worker_schedule())

        skip = 0
        if self.resume_batches:
            if not self.loader_batch_size:
                raise ValueError("set_resume_point() needs the DataLoader batch size")
            skip = served_by_worker(
                self.resume_batches, self.num_workers, self.worker_id, self.loader_batch_size
            )

        if self.lazy_decode:
            yield from self._iter_lazy_decode(rng, splits, skip)
            return

        max_in_flight = 8 * self.decode_threads

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

    Also installs the cost-gated SIGTERM/SIGUSR1 preemption callback (with lightweight
    trainable-only emergency checkpointing) and supports resuming seamlessly from
    ``emergency_trainable.pt`` checkpoints.
    """
    from torch.utils.data import DataLoader

    from .trainer_integration import SpotPreemptionMonitor

    class ExactResumeGr00tTrainer(base_trainer_cls):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._emergency_ckpt_dir: Path | None = None
            self._emergency_ckpt_payload: dict[str, Any] | None = None
            super().__init__(*args, **kwargs)
            self.add_callback(_make_preempt_callback(SpotPreemptionMonitor(), trainer_ref=self))

        def _matches_emergency_ckpt(self, checkpoint: str | Path | None) -> bool:
            if self._emergency_ckpt_payload is None or self._emergency_ckpt_dir is None or checkpoint is None:
                return False
            return Path(checkpoint).resolve() == self._emergency_ckpt_dir

        def train(self, resume_from_checkpoint=None, **kwargs: Any):
            resolved_ckpt: str | None = None
            if resume_from_checkpoint is True:
                from transformers.trainer_utils import get_last_checkpoint

                resolved_ckpt = get_last_checkpoint(self.args.output_dir)
                if resolved_ckpt is None:
                    raise ValueError(
                        f"No valid checkpoint found in output directory ({self.args.output_dir})"
                    )
            elif isinstance(resume_from_checkpoint, (str, Path)):
                resolved_ckpt = str(resume_from_checkpoint)

            if resolved_ckpt is not None:
                # Pre-load TrainerState so get_train_dataloader() sees the exact global_step
                # even when wrapping plain HuggingFace Trainer (not only Gr00tTrainer).
                state_path = Path(resolved_ckpt) / "trainer_state.json"
                if state_path.is_file():
                    from transformers import TrainerState

                    self.state = TrainerState.load_from_json(str(state_path))

                if is_emergency_trainable_checkpoint(resolved_ckpt):
                    ckpt_file = Path(resolved_ckpt) / EMERGENCY_CKPT_FILENAME
                    logger.info("Detected lightweight emergency trainable checkpoint at %s", ckpt_file)
                    self._emergency_ckpt_dir = Path(resolved_ckpt).resolve()
                    self._emergency_ckpt_payload = torch.load(
                        ckpt_file, map_location="cpu", weights_only=False
                    )

            try:
                return super().train(
                    resume_from_checkpoint=resolved_ckpt if resolved_ckpt is not None else resume_from_checkpoint,
                    **kwargs,
                )
            finally:
                self._emergency_ckpt_dir = None
                self._emergency_ckpt_payload = None

        def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
            if self._matches_emergency_ckpt(resume_from_checkpoint):
                target_model = self.model if model is None else model
                n_loaded = _apply_trainable_weights(
                    target_model, self._emergency_ckpt_payload["trainable_state_dict"]
                )
                logger.info(
                    "Restored %d trainable parameter tensors from emergency checkpoint %s",
                    n_loaded,
                    resume_from_checkpoint,
                )
                return
            return super()._load_from_checkpoint(resume_from_checkpoint, model=model)

        def _load_optimizer_and_scheduler(self, checkpoint):
            if self._matches_emergency_ckpt(checkpoint):
                opt_sd = self._emergency_ckpt_payload.get("optimizer_state_dict")
                if self.optimizer is not None and opt_sd is not None:
                    self.optimizer.load_state_dict(opt_sd)
                sched_sd = self._emergency_ckpt_payload.get("lr_scheduler_state_dict")
                if self.lr_scheduler is not None and sched_sd is not None:
                    self.lr_scheduler.load_state_dict(sched_sd)
                logger.info("Restored optimizer & scheduler state from emergency checkpoint %s", checkpoint)
                return
            return super()._load_optimizer_and_scheduler(checkpoint)

        def _load_rng_state(self, checkpoint):
            if self._matches_emergency_ckpt(checkpoint):
                _restore_rng_state(self._emergency_ckpt_payload.get("rng_state"))
                logger.info("Restored RNG states from emergency checkpoint %s", checkpoint)
                return
            return super()._load_rng_state(checkpoint)

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


def _make_preempt_callback(
    monitor,
    *,
    cost_gate: bool | None = None,
    light_ckpt: bool | None = None,
    est_save_s: float | None = None,
    trainer_ref: Any | None = None,
    exit_on_preempt: bool = True,
):
    from transformers import TrainerCallback

    use_cost_gate = (
        os.environ.get("GR00T_PREEMPT_COST_GATE", "1") == "1" if cost_gate is None else bool(cost_gate)
    )
    use_light_ckpt = (
        os.environ.get("GR00T_PREEMPT_LIGHT_CKPT", "1") == "1" if light_ckpt is None else bool(light_ckpt)
    )
    if est_save_s is None:
        env_est = os.environ.get("GR00T_PREEMPT_EST_SAVE_S")
        if env_est is not None and env_est.strip() != "":
            est_save = float(env_est)
        else:
            est_save = 10.0 if use_light_ckpt else 55.0
    else:
        est_save = float(est_save_s)

    class PreemptCallback(TrainerCallback):
        def __init__(self) -> None:
            self.cost_gate = use_cost_gate
            self.light_ckpt = use_light_ckpt
            self.est_save_s = est_save
            self.last_saved_step: int = 0
            self._step_times: deque[float] = deque(maxlen=20)
            self._last_step_t: float | None = None
            self.skipped_save: bool = False
            self.saved_light_ckpt: Path | None = None

        @property
        def avg_step_s(self) -> float:
            if not self._step_times:
                return 0.0
            return sum(self._step_times) / len(self._step_times)

        def on_train_begin(self, args, state, control, **kwargs):
            monitor.install()
            self.last_saved_step = int(state.global_step)
            self._last_step_t = time.monotonic()
            self.skipped_save = False
            self.saved_light_ckpt = None

        def on_save(self, args, state, control, **kwargs):
            self.last_saved_step = int(state.global_step)
            self._last_step_t = time.monotonic()

        def on_step_end(self, args, state, control, model=None, optimizer=None, lr_scheduler=None, **kwargs):
            now = time.monotonic()
            if self._last_step_t is not None:
                dt = now - self._last_step_t
                if dt > 0:
                    self._step_times.append(dt)
            self._last_step_t = now

            if not monitor.broadcast_preempted():
                return control

            steps_since_save = max(0, int(state.global_step) - self.last_saved_step)
            avg_s = self.avg_step_s
            wasted_s = steps_since_save * avg_s

            if self.cost_gate and self.last_saved_step > 0 and wasted_s <= self.est_save_s:
                logger.warning(
                    "Preempted at step %d: skipping checkpoint save (cost-gated: "
                    "%d steps * %.3fs = %.2fs <= est_save_s=%.2fs; last_saved_step=%d).",
                    state.global_step,
                    steps_since_save,
                    avg_s,
                    wasted_s,
                    self.est_save_s,
                    self.last_saved_step,
                )
                self.skipped_save = True
                control.should_save = False
                control.should_training_stop = True
                return control

            if self.light_ckpt:
                mdl = model if model is not None else getattr(trainer_ref, "model", None)
                opt = optimizer if optimizer is not None else getattr(trainer_ref, "optimizer", None)
                sched = lr_scheduler if lr_scheduler is not None else getattr(trainer_ref, "lr_scheduler", None)
                if mdl is not None:
                    should_write = getattr(args, "should_save", True)
                    t_save0 = time.monotonic()
                    if should_write:
                        ckpt_dir = save_emergency_trainable_checkpoint(
                            args.output_dir, state, mdl, optimizer=opt, lr_scheduler=sched
                        )
                        self.saved_light_ckpt = ckpt_dir
                        logger.warning(
                            "Preempted at step %d: wrote lightweight trainable-only checkpoint to %s in %.2fs "
                            "(wasted_s=%.2fs > est_save_s=%.2fs).",
                            state.global_step,
                            ckpt_dir,
                            time.monotonic() - t_save0,
                            wasted_s,
                            self.est_save_s,
                        )
                    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                        dist.barrier()
                    self.last_saved_step = int(state.global_step)
                    control.should_save = False
                    control.should_training_stop = True
                    return control

            control.should_save = True
            control.should_training_stop = True
            return control

        def on_train_end(self, args, state, control, **kwargs):
            monitor.restore()
            if monitor.preempted and exit_on_preempt:
                if self.skipped_save:
                    msg = f"Preempted at step {state.global_step}: save skipped (cost-gated); exiting (143)."
                elif self.saved_light_ckpt is not None:
                    msg = (
                        f"Preempted at step {state.global_step}: lightweight checkpoint saved at "
                        f"{self.saved_light_ckpt}; exiting (143)."
                    )
                else:
                    msg = f"Preempted at step {state.global_step}: checkpoint saved; exiting (143)."
                logger.warning("%s", msg)
                raise SystemExit(143)

    return PreemptCallback()


__all__ = [
    "EMERGENCY_CKPT_FILENAME",
    "ExactResumeMixin",
    "is_emergency_trainable_checkpoint",
    "load_emergency_trainable_checkpoint",
    "make_exact_resume_mixture_class",
    "make_exact_resume_trainer_class",
    "save_emergency_trainable_checkpoint",
    "served_by_worker",
]
