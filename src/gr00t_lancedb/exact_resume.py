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


def _fsync_path(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    except OSError:  # some network / 9p filesystems reject fsync on directories
        pass
    finally:
        os.close(fd)


def save_emergency_trainable_checkpoint(
    output_dir: str | Path,
    state: Any,
    model: Any,
    optimizer: Any = None,
    lr_scheduler: Any = None,
    *,
    dtype: str = "fp32",
    exact_weights: bool = False,
    fsync: bool = True,
    stats: dict[str, Any] | None = None,
) -> Path:
    """Atomically save ONLY trainable parameters + optimizer + scheduler + RNG + trainer_state.json.

    Writes into a staging directory first and atomically commits ``checkpoint-<step>`` so a
    hard ``SIGKILL`` mid-write never leaves a half-written checkpoint visible to
    ``get_last_checkpoint()``. ``dtype`` selects a :mod:`gr00t_lancedb.ckpt_codec` preset
    (``fp32`` exact, ``bf16`` / ``int8`` lossy and 2-3x smaller). With ``fsync`` the data is
    on stable storage before the directory becomes visible (a stopped spot VM may lose
    un-flushed page cache). ``stats`` (if given) receives bytes / seconds / dtype.
    """
    from .ckpt_codec import encode_optimizer_state, encode_trainable_state, is_lossless, resolve_preset

    t0 = time.monotonic()
    spec = resolve_preset(dtype, exact_weights=exact_weights)
    step = int(state.global_step)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    target_dir = out_root / f"checkpoint-{step}"
    tmp_dir = out_root / f".tmp-checkpoint-{step}-{os.getpid()}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    raw_model = _unwrap_model(model)
    trainable = {k: v for k, v in raw_model.named_parameters() if v.requires_grad}
    payload = {
        "format": "gr00t_emergency_trainable_v2",
        "global_step": step,
        "codec": spec,
        "lossless": is_lossless(spec),
        "trainable_state_dict": encode_trainable_state(trainable, spec),
        "optimizer_state_dict": (
            encode_optimizer_state(optimizer.state_dict(), spec) if optimizer is not None else None
        ),
        "lr_scheduler_state_dict": lr_scheduler.state_dict() if lr_scheduler is not None else None,
        "rng_state": _capture_rng_state(),
    }
    t_encoded = time.monotonic()

    try:
        ckpt_path = tmp_dir / EMERGENCY_CKPT_FILENAME
        torch.save(payload, ckpt_path)
        state_json_path = tmp_dir / "trainer_state.json"
        if hasattr(state, "save_to_json"):
            state.save_to_json(str(state_json_path))
        else:
            state_json_path.write_text(json.dumps(vars(state), default=str, indent=2), encoding="utf-8")
        if fsync:
            _fsync_path(ckpt_path)
            _fsync_path(state_json_path)
            _fsync_path(tmp_dir)
        nbytes = ckpt_path.stat().st_size + state_json_path.stat().st_size

        if target_dir.exists():
            # Replace files atomically inside existing checkpoint dir if same step already exists.
            os.replace(tmp_dir / EMERGENCY_CKPT_FILENAME, target_dir / EMERGENCY_CKPT_FILENAME)
            os.replace(state_json_path, target_dir / "trainer_state.json")
            shutil.rmtree(tmp_dir, ignore_errors=True)
        else:
            os.replace(tmp_dir, target_dir)
        if fsync:
            _fsync_path(out_root)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    if stats is not None:
        t1 = time.monotonic()
        stats.update(
            {
                "bytes": int(nbytes),
                "seconds": t1 - t0,
                "encode_s": t_encoded - t0,
                "write_s": t1 - t_encoded,
                "dtype": spec["name"],
                "exact_weights": spec["weights"] == "fp32",
                "lossless": is_lossless(spec),
            }
        )
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
    payload = _load_emergency_payload(Path(checkpoint_dir) / EMERGENCY_CKPT_FILENAME)
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


def _load_emergency_payload(ckpt_file: Path) -> dict[str, Any]:
    """``torch.load`` + decode any compressed tensors back to their original dtype."""
    from .ckpt_codec import decode_tree

    payload = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    if payload.get("format") == "gr00t_emergency_trainable_v2":
        payload["trainable_state_dict"] = decode_tree(payload["trainable_state_dict"])
        payload["optimizer_state_dict"] = decode_tree(payload.get("optimizer_state_dict"))
        if not payload.get("lossless", True):
            logger.warning(
                "Emergency checkpoint %s uses lossy preset %r: resume is close to, but not "
                "bit-identical with, the uninterrupted run.",
                ckpt_file,
                (payload.get("codec") or {}).get("name"),
            )
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
        first_decode_batch_size: int | None = None,
        weighted_anchors: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if decode_batch_size < 1:
            raise ValueError(f"decode_batch_size must be >= 1, got {decode_batch_size}")
        if first_decode_batch_size is not None and first_decode_batch_size < 1:
            raise ValueError(f"first_decode_batch_size must be >= 1, got {first_decode_batch_size}")
        if prefetch_batches < 0:
            raise ValueError(f"prefetch_batches must be >= 0, got {prefetch_batches}")
        self.per_sample_seed = bool(per_sample_seed)
        self.lazy_decode = bool(lazy_decode)
        # Steady-state chunk size (more anchors per Lance take / blob fetch) and a smaller
        # first chunk after every (re)start so the first batch is not delayed. Chunking only
        # groups decode calls: anchors are drawn in the same RNG order either way, so the
        # sample stream is bit-identical for any sizes.
        self.decode_batch_size = int(decode_batch_size)
        self.first_decode_batch_size = (
            int(first_decode_batch_size) if first_decode_batch_size is not None else self.decode_batch_size
        )
        self.prefetch_batches = int(prefetch_batches)
        # Flywheel round 2: draw anchors i.i.d. from a per-anchor weight column instead of the
        # shuffle buffer (``gr00t_lancedb.flywheel.weights.WeightedAnchors``). Always decoded
        # lazily; the stream depends only on (seed, rank, logical worker), so resume is exact.
        self.weighted_anchors = weighted_anchors
        if weighted_anchors is not None:
            self.lazy_decode = True
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
            "first_decode_batch_size": self.first_decode_batch_size,
            "prefetch_batches": self.prefetch_batches,
            "weighted_anchors": (
                None
                if self.weighted_anchors is None
                else {"column": self.weighted_anchors.column, "digest": self.weighted_anchors.digest}
            ),
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

    def _resume_skip(self) -> int:
        """Samples this (logical) worker already served before the resume point."""
        if not self.resume_batches:
            return 0
        if not self.loader_batch_size:
            raise ValueError("set_resume_point() needs the DataLoader batch size")
        return served_by_worker(self.resume_batches, self.num_workers, self.worker_id, self.loader_batch_size)

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

        yield from self._serve_anchor_stream(draw_one, skip, label="lazy-decode")

    def _iter_weighted(self, skip: int):
        """Weighted i.i.d. anchor draws (flywheel round 2), decoded like the lazy path."""
        stream = self.weighted_anchors.stream(self.seed, self.rank, int(self.worker_id or 0))
        yield from self._serve_anchor_stream(lambda: next(stream), skip, label="weighted")

    def _serve_anchor_stream(self, draw_one, skip: int, *, label: str):
        """Skip ``skip`` draws (integer-only), then decode anchors in prefetched chunks."""
        served = 0
        while served < skip:
            draw_one()
            served += 1
        if skip:
            logger.info(
                "rank %d worker %d: %s resumed after %d served samples (0 buffer splits decoded)",
                self.rank, self.worker_id, label, skip,
            )

        n_chunks = 0

        def next_chunk() -> list[tuple[int, int, int, int]]:
            nonlocal n_chunks, served
            size = self.first_decode_batch_size if n_chunks == 0 else self.decode_batch_size
            n_chunks += 1
            chunk_meta: list[tuple[int, int, int, int]] = []
            for _ in range(size):
                s_idx = served
                ds_idx, ep_idx, step_idx = draw_one()
                served += 1
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

        executor = ThreadPoolExecutor(max_workers=1)
        try:
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
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def __iter__(self):
        if self.weighted_anchors is not None:
            # Resolves this physical worker's *logical* id; the shard schedule itself is unused.
            self.filter_shard_sample_schedule()
            yield from self._iter_weighted(self._resume_skip())
            return

        rng = np.random.default_rng(self.seed + self.epoch)
        splits = self._iter_splits(self._worker_schedule())
        skip = self._resume_skip()

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
                    self._emergency_ckpt_payload = _load_emergency_payload(ckpt_file)

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
                trainable_sd = self._emergency_ckpt_payload.pop("trainable_state_dict", None)
                if trainable_sd is None:
                    raise RuntimeError("Emergency checkpoint missing trainable_state_dict")
                n_loaded = _apply_trainable_weights(target_model, trainable_sd)
                del trainable_sd
                logger.info(
                    "Restored %d trainable parameter tensors from emergency checkpoint %s",
                    n_loaded,
                    resume_from_checkpoint,
                )
                return
            return super()._load_from_checkpoint(resume_from_checkpoint, model=model)

        def _load_optimizer_and_scheduler(self, checkpoint):
            if self._matches_emergency_ckpt(checkpoint):
                opt_sd = self._emergency_ckpt_payload.pop("optimizer_state_dict", None)
                if self.optimizer is not None and opt_sd is not None:
                    self.optimizer.load_state_dict(opt_sd)
                del opt_sd
                sched_sd = self._emergency_ckpt_payload.pop("lr_scheduler_state_dict", None)
                if self.lr_scheduler is not None and sched_sd is not None:
                    self.lr_scheduler.load_state_dict(sched_sd)
                del sched_sd
                logger.info("Restored optimizer & scheduler state from emergency checkpoint %s", checkpoint)
                return
            return super()._load_optimizer_and_scheduler(checkpoint)

        def _load_rng_state(self, checkpoint):
            if self._matches_emergency_ckpt(checkpoint):
                rng_state = self._emergency_ckpt_payload.pop("rng_state", None)
                _restore_rng_state(rng_state)
                del rng_state
                # Release the multi-GB payload immediately so it is not kept alive
                # across forked DataLoader workers or the rest of training.
                self._emergency_ckpt_payload = None
                self._emergency_ckpt_dir = None
                import gc

                gc.collect()
                logger.info("Restored RNG states from emergency checkpoint %s (payload released)", checkpoint)
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


CALIBRATION_FILENAME = "preempt_calibration.json"
PREEMPT_EVENTS_FILENAME = "preempt_events.jsonl"


class WriteBandwidthCalibration:
    """Self-calibrating estimate of checkpoint write bandwidth (bytes/s) for the cost gate.

    Samples come from (in order of trust) real emergency saves, regular HF checkpoint saves
    (``on_save``: bytes of ``checkpoint-<step>`` / wall time) and an optional background
    fsync'ed probe write at train start. They persist in ``preempt_calibration.json`` in the
    output dir, so a run resumed after preemption starts calibrated. With no samples a
    conservative prior (``GR00T_PREEMPT_PRIOR_WRITE_MBPS``, default 400 MB/s) is used.
    """

    def __init__(self, path: Path | None, prior_bps: float, overhead_s: float, max_samples: int = 8) -> None:
        self.path = path
        self.prior_bps = float(prior_bps)
        self.overhead_s = float(overhead_s)
        self.max_samples = int(max_samples)
        self.samples: list[dict[str, Any]] = []
        if path is not None and path.is_file():
            try:
                self.samples = list(json.loads(path.read_text(encoding="utf-8")).get("samples", []))
            except (OSError, ValueError):
                self.samples = []

    def add(self, nbytes: int, seconds: float, source: str) -> None:
        if nbytes <= 0 or seconds <= 0:
            return
        self.samples.append({"bytes": int(nbytes), "seconds": float(seconds), "source": source, "t": time.time()})
        self.samples = self.samples[-self.max_samples :]
        self.save()

    def save(self) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps({"samples": self.samples, "bytes_per_s": self.bytes_per_s()}, indent=2))
            os.replace(tmp, self.path)
        except OSError:
            pass

    def bytes_per_s(self) -> float:
        """Median bandwidth of the most trusted sample class available."""
        for sources in (("emergency",), ("hf_save",), ("probe",)):
            rates = sorted(s["bytes"] / s["seconds"] for s in self.samples if s.get("source") in sources)
            if rates:
                return float(rates[len(rates) // 2])
        return self.prior_bps

    def estimate_s(self, nbytes: int) -> float:
        return self.overhead_s + float(nbytes) / max(self.bytes_per_s(), 1.0)

    @property
    def calibrated(self) -> bool:
        return bool(self.samples)


def _dir_bytes(path: Path) -> int:
    total = 0
    for f in path.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            pass
    return total


def _full_checkpoint_bytes(model: Any, optimizer: Any) -> int:
    """Rough size of HF's full ``_save_checkpoint`` (all weights + optimizer state)."""
    total = 0
    if model is not None:
        raw = _unwrap_model(model)
        total += sum(p.numel() * p.element_size() for p in raw.parameters())
    if optimizer is not None:
        for pstate in getattr(optimizer, "state", {}).values():
            for v in pstate.values():
                if isinstance(v, torch.Tensor):
                    total += v.numel() * v.element_size()
    return int(total)


def plan_preempt_save(
    *,
    steps_since_save: int,
    avg_step_s: float,
    last_saved_step: int,
    est_by_dtype: dict[str, float],
    dtype: str,
    deadline_s: float,
    cost_gate: bool,
    skip_over_deadline: bool = True,
) -> dict[str, Any]:
    """Decide what to do on a preemption notice (pure function, unit-tested).

    ``est_by_dtype`` maps candidate dtype presets to estimated save seconds. Returns
    ``{"action": "skip"|"save", "dtype", "est_save_s", "wasted_s", "reason"}``.

    * ``dtype == "auto"`` picks the most precise preset whose estimate fits ``deadline_s``
      (else the smallest).
    * Cost gate: saving only pays off if redoing ``steps_since_save`` steps would cost more
      than writing the checkpoint; with a prior checkpoint and ``wasted_s <= est_save_s``
      the save is skipped (resume from the last checkpoint is exact anyway).
    * A save that cannot finish inside the grace period is skipped when a previous
      checkpoint exists (a torn save is pure loss); with no checkpoint at all we try anyway.
    """
    wasted_s = max(0, int(steps_since_save)) * max(0.0, float(avg_step_s))
    if dtype == "auto":
        order = [d for d in ("fp32", "bf16", "int8") if d in est_by_dtype]
        chosen = next((d for d in order if est_by_dtype[d] <= deadline_s), order[-1])
    else:
        chosen = dtype
    est = float(est_by_dtype[chosen])
    base = {"dtype": chosen, "est_save_s": est, "wasted_s": wasted_s, "deadline_s": float(deadline_s)}
    if cost_gate and last_saved_step > 0 and wasted_s <= est:
        return {**base, "action": "skip", "reason": "cost_gate"}
    if skip_over_deadline and last_saved_step > 0 and est > deadline_s:
        return {**base, "action": "skip", "reason": "over_deadline"}
    return {**base, "action": "save", "reason": "worth_saving" if last_saved_step > 0 else "no_prior_checkpoint"}


def _make_preempt_callback(
    monitor,
    *,
    cost_gate: bool | None = None,
    light_ckpt: bool | None = None,
    est_save_s: float | None = None,
    trainer_ref: Any | None = None,
    exit_on_preempt: bool = True,
    ckpt_dtype: str | None = None,
    deadline_s: float | None = None,
):
    """SIGTERM/SIGUSR1 handling: cost-gated, deadline-aware light emergency checkpoint.

    Env (defaults): ``GR00T_PREEMPT_COST_GATE`` (1), ``GR00T_PREEMPT_LIGHT_CKPT`` (1),
    ``GR00T_PREEMPT_CKPT_DTYPE`` (``auto``; ``fp32|bf16|int8``), ``GR00T_PREEMPT_DEADLINE_S``
    (25: GCP/AWS grace minus margin), ``GR00T_PREEMPT_EXACT_WEIGHTS`` (0),
    ``GR00T_PREEMPT_EST_SAVE_S`` (unset = self-calibrated; a number pins the estimate),
    ``GR00T_PREEMPT_PRIOR_WRITE_MBPS`` (400), ``GR00T_PREEMPT_SAVE_OVERHEAD_S`` (1.0),
    ``GR00T_PREEMPT_PROBE_MB`` (64; background fsync'ed probe write at train start, 0=off),
    ``GR00T_PREEMPT_FSYNC`` (1), ``GR00T_PREEMPT_FAST_EXIT`` (1: ``os._exit(143)`` once the
    checkpoint decision is durable, skipping ~20 s of DataLoader / interpreter teardown).
    """
    from transformers import TrainerCallback

    env = os.environ
    use_cost_gate = env.get("GR00T_PREEMPT_COST_GATE", "1") == "1" if cost_gate is None else bool(cost_gate)
    use_light_ckpt = env.get("GR00T_PREEMPT_LIGHT_CKPT", "1") == "1" if light_ckpt is None else bool(light_ckpt)
    if est_save_s is None:
        raw = env.get("GR00T_PREEMPT_EST_SAVE_S", "").strip()
        fixed_est = float(raw) if raw else None
    else:
        fixed_est = float(est_save_s)
    dtype_setting = (ckpt_dtype or env.get("GR00T_PREEMPT_CKPT_DTYPE", "auto")).strip().lower()
    if dtype_setting not in ("auto", "fp32", "bf16", "int8"):
        raise ValueError(f"GR00T_PREEMPT_CKPT_DTYPE must be auto|fp32|bf16|int8, got {dtype_setting!r}")
    deadline = float(deadline_s if deadline_s is not None else env.get("GR00T_PREEMPT_DEADLINE_S", "25"))
    exact_weights = env.get("GR00T_PREEMPT_EXACT_WEIGHTS", "0") == "1"
    prior_bps = float(env.get("GR00T_PREEMPT_PRIOR_WRITE_MBPS", "400")) * 1e6
    overhead_s = float(env.get("GR00T_PREEMPT_SAVE_OVERHEAD_S", "1.0"))
    probe_mb = int(float(env.get("GR00T_PREEMPT_PROBE_MB", "64")))
    do_fsync = env.get("GR00T_PREEMPT_FSYNC", "1") == "1"
    # A100 run: after the decision, `raise SystemExit` still spent ~22 s tearing down
    # DataLoader workers / CUDA / the interpreter, which alone eats most of a 30 s grace
    # period. Everything durable is written by then, so exit immediately.
    fast_exit = env.get("GR00T_PREEMPT_FAST_EXIT", "1") == "1"

    class PreemptCallback(TrainerCallback):
        def __init__(self) -> None:
            self.cost_gate = use_cost_gate
            self.light_ckpt = use_light_ckpt
            self.est_save_s = fixed_est  # None => self-calibrated per decision
            self.ckpt_dtype = dtype_setting
            self.deadline_s = deadline
            self.last_saved_step: int = 0
            self._step_times: deque[float] = deque(maxlen=20)
            self._last_step_t: float | None = None
            self._save_mark_t: float | None = None
            self.skipped_save: bool = False
            self.saved_light_ckpt: Path | None = None
            self.last_decision: dict[str, Any] | None = None
            self.calibration = WriteBandwidthCalibration(None, prior_bps, overhead_s)
            self._output_dir: Path | None = None
            self.fast_exit = fast_exit
            # Callables run right before a fast exit (e.g. metrics writers' train_end).
            self.pre_exit_hooks: list[Any] = []

        @property
        def avg_step_s(self) -> float:
            if not self._step_times:
                return 0.0
            return sum(self._step_times) / len(self._step_times)

        def estimate_save_s(self, nbytes: int) -> float:
            return self.est_save_s if self.est_save_s is not None else self.calibration.estimate_s(nbytes)

        # -- calibration ------------------------------------------------------------
        def _bind_output_dir(self, args) -> None:
            out = getattr(args, "output_dir", None)
            if out is None or self._output_dir == Path(out):
                return
            self._output_dir = Path(out)
            self.calibration = WriteBandwidthCalibration(self._output_dir / CALIBRATION_FILENAME, prior_bps, overhead_s)

        def _start_probe(self) -> None:
            if probe_mb <= 0 or self._output_dir is None or self.est_save_s is not None:
                return
            if any(s.get("source") in ("emergency", "hf_save") for s in self.calibration.samples):
                return
            out_dir = self._output_dir
            calib = self.calibration

            def _probe() -> None:
                path = out_dir / f".preempt-probe-{os.getpid()}.tmp"
                try:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    block = os.urandom(1 << 20)
                    t0 = time.monotonic()
                    with open(path, "wb") as fh:
                        for _ in range(probe_mb):
                            fh.write(block)
                        fh.flush()
                        os.fsync(fh.fileno())
                    calib.add(probe_mb << 20, time.monotonic() - t0, "probe")
                except OSError:
                    pass
                finally:
                    try:
                        path.unlink()
                    except OSError:
                        pass

            import threading

            threading.Thread(target=_probe, name="gr00t-preempt-probe", daemon=True).start()

        def _log_event(self, rec: dict[str, Any]) -> None:
            if self._output_dir is None:
                return
            try:
                self._output_dir.mkdir(parents=True, exist_ok=True)
                with open(self._output_dir / PREEMPT_EVENTS_FILENAME, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"t": time.time(), **rec}) + "\n")
            except OSError:
                pass

        # -- callbacks ----------------------------------------------------------------
        def on_train_begin(self, args, state, control, **kwargs):
            monitor.install()
            self._bind_output_dir(args)
            self.last_saved_step = int(state.global_step)
            self._last_step_t = time.monotonic()
            self.skipped_save = False
            self.saved_light_ckpt = None
            self._start_probe()

        def on_save(self, args, state, control, **kwargs):
            if self._save_mark_t is not None:
                ckpt = Path(args.output_dir) / f"checkpoint-{int(state.global_step)}"
                elapsed = time.monotonic() - self._save_mark_t
                if ckpt.is_dir():
                    nbytes = _dir_bytes(ckpt)
                    if nbytes >= (64 << 20):
                        self.calibration.add(nbytes, elapsed, "hf_save")
                self._save_mark_t = None
            self.last_saved_step = int(state.global_step)
            self._last_step_t = time.monotonic()

        def on_step_end(self, args, state, control, model=None, optimizer=None, lr_scheduler=None, **kwargs):
            now = time.monotonic()
            if self._last_step_t is not None:
                dt = now - self._last_step_t
                if dt > 0:
                    self._step_times.append(dt)
            self._last_step_t = now
            self._bind_output_dir(args)

            if not monitor.broadcast_preempted():
                if getattr(control, "should_save", False):
                    self._save_mark_t = time.monotonic()  # HF saves right after this callback
                return control

            mdl = model if model is not None else getattr(trainer_ref, "model", None)
            opt = optimizer if optimizer is not None else getattr(trainer_ref, "optimizer", None)
            sched = lr_scheduler if lr_scheduler is not None else getattr(trainer_ref, "lr_scheduler", None)
            steps_since_save = max(0, int(state.global_step) - self.last_saved_step)

            if self.light_ckpt and mdl is not None:
                from .ckpt_codec import estimate_payload_bytes, resolve_preset

                cands = ("fp32", "bf16", "int8") if self.ckpt_dtype == "auto" else (self.ckpt_dtype,)
                est_by_dtype = {
                    d: self.estimate_save_s(
                        estimate_payload_bytes(mdl, opt, resolve_preset(d, exact_weights=exact_weights))
                    )
                    for d in cands
                }
                dtype_for_plan = self.ckpt_dtype
            else:
                full_bytes = _full_checkpoint_bytes(mdl, opt) if mdl is not None else 0
                est_full = self.est_save_s if self.est_save_s is not None else (
                    self.calibration.estimate_s(full_bytes) if full_bytes else 55.0
                )
                est_by_dtype = {"full": est_full}
                dtype_for_plan = "full"

            decision = plan_preempt_save(
                steps_since_save=steps_since_save,
                avg_step_s=self.avg_step_s,
                last_saved_step=self.last_saved_step,
                est_by_dtype=est_by_dtype,
                dtype=dtype_for_plan,
                deadline_s=self.deadline_s,
                cost_gate=self.cost_gate,
                # Fixed estimates (tests / manual override) keep the old cost-gate-only rule.
                skip_over_deadline=self.est_save_s is None,
            )
            decision.update(
                {
                    "event": "preempt_decision",
                    "global_step": int(state.global_step),
                    "steps_since_save": steps_since_save,
                    "avg_step_s": self.avg_step_s,
                    "last_saved_step": self.last_saved_step,
                    "calibrated_bytes_per_s": self.calibration.bytes_per_s(),
                    "calibrated": self.calibration.calibrated,
                    "candidates_est_s": est_by_dtype,
                }
            )
            self.last_decision = decision
            control.should_training_stop = True

            if decision["action"] == "skip":
                logger.warning(
                    "Preempted at step %d: skipping checkpoint (%s: wasted %.1fs vs est save %.1fs [%s], "
                    "deadline %.0fs; last_saved_step=%d).",
                    state.global_step, decision["reason"], decision["wasted_s"], decision["est_save_s"],
                    decision["dtype"], self.deadline_s, self.last_saved_step,
                )
                self.skipped_save = True
                control.should_save = False
                self._log_event(decision)
                return control

            if self.light_ckpt and mdl is not None:
                should_write = getattr(args, "should_save", True)
                stats: dict[str, Any] = {}
                if should_write:
                    ckpt_dir = save_emergency_trainable_checkpoint(
                        args.output_dir, state, mdl, optimizer=opt, lr_scheduler=sched,
                        dtype=decision["dtype"], exact_weights=exact_weights, fsync=do_fsync, stats=stats,
                    )
                    self.saved_light_ckpt = ckpt_dir
                    self.calibration.add(stats["bytes"], stats["write_s"], "emergency")
                    logger.warning(
                        "Preempted at step %d: wrote %s emergency checkpoint (%.2f GB) to %s in %.2fs "
                        "(est %.2fs, wasted_s=%.2fs).",
                        state.global_step, stats["dtype"], stats["bytes"] / 1e9, ckpt_dir,
                        stats["seconds"], decision["est_save_s"], decision["wasted_s"],
                    )
                if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                    dist.barrier()
                self.last_saved_step = int(state.global_step)
                control.should_save = False
                self._log_event({**decision, "save": stats})
                return control

            control.should_save = True
            self._log_event(decision)
            return control

        def _exit_now(self, code: int) -> None:
            import sys

            for hook in self.pre_exit_hooks:
                try:
                    hook()
                except Exception:  # noqa: BLE001 - never block the exit
                    pass
            for stream in (sys.stdout, sys.stderr):
                try:
                    stream.flush()
                except Exception:  # noqa: BLE001
                    pass
            logging.shutdown()
            os._exit(code)

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
                sig_t = getattr(monitor, "signal_time", None)
                self._log_event(
                    {
                        "event": "preempt_exit",
                        "global_step": int(state.global_step),
                        "signal_to_exit_s": (time.time() - sig_t) if sig_t else None,
                        "fast_exit": self.fast_exit,
                    }
                )
                if self.fast_exit:
                    self._exit_now(143)
                raise SystemExit(143)

    return PreemptCallback()


__all__ = [
    "CALIBRATION_FILENAME",
    "EMERGENCY_CKPT_FILENAME",
    "PREEMPT_EVENTS_FILENAME",
    "WriteBandwidthCalibration",
    "ExactResumeMixin",
    "is_emergency_trainable_checkpoint",
    "load_emergency_trainable_checkpoint",
    "make_exact_resume_mixture_class",
    "make_exact_resume_trainer_class",
    "plan_preempt_save",
    "save_emergency_trainable_checkpoint",
    "served_by_worker",
]
