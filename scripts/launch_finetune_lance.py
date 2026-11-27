#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Run NVIDIA Isaac-GR00T's *official* ``launch_finetune.py`` with a selectable data backend.

Everything (model, processor, hyper-parameters, CLI flags) comes from GR00T itself; this
wrapper only swaps seams, selected by ``GR00T_DATA_BACKEND``:

``native``        stock GR00T (fast loader; resume reseeds with ``seed + global_step``)
``native-exact``  GR00T's own fast datasets + ``ExactResumeFastShardedMixtureDataset`` and
                  an exact-resume ``Gr00tTrainer`` (no Lance involved)
``lance``         ``LanceShardedSingleStepDataset`` (local dir or ``s3://``) + the same
                  exact-resume mixture/trainer; supports safe ``fork`` when unopened

In every mode a ``BenchMetricsCallback`` appends one JSON line per optimizer step (plus
startup phase timestamps) to ``$GR00T_BENCH_METRICS`` (default ``<output_dir>/bench_metrics.jsonl``).

Key v1 performance / ablation env vars:
- ``GR00T_LANCE_LAZY_DECODE`` (default ``1`` for ``lance``, ``0`` for ``native-exact``)
- ``GR00T_LANCE_DECODE_BATCH_SIZE`` (default ``32``)
- ``GR00T_LANCE_PREFETCH_BATCHES`` (default ``2``)
- ``GR00T_LANCE_MP_CTX`` (default ``fork`` for ``lance`` if unopened; ``fork|forkserver|spawn``)
- ``GR00T_LANCE_ASYNC_PREFETCH`` (default ``0``; also accepts ``GR00T_LANCE_PREFETCH_FIRST_BATCH``)
- ``GR00T_PREEMPT_COST_GATE`` (default ``1``), ``GR00T_PREEMPT_LIGHT_CKPT`` (default ``1``),
  ``GR00T_PREEMPT_EST_SAVE_S`` (default ``10.0`` when light ckpt else ``55.0``)

Usage::

    GR00T_DATA_BACKEND=lance python scripts/launch_finetune_lance.py \
        --gr00t-root /data/Isaac-GR00T -- \
        --base-model-path nvidia/GR00T-N1.7-3B --dataset-path /data/datasets/libero_10_lance \
        --embodiment-tag LIBERO_PANDA --output-dir /data/runs/lance ...
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import json
import logging
import os
from pathlib import Path
import runpy
import subprocess
import sys
import threading
import time
from typing import Any

import numpy as np

logger = logging.getLogger("gr00t_lancedb.launch")

# Benchmark-only: skip the end-of-training checkpoint and GR00T's final save_model().
SKIP_FINAL_SAVE = os.environ.get("GR00T_BENCH_SKIP_FINAL_SAVE", "0") == "1"

_VALID_MP_CTXS = ("fork", "forkserver", "spawn")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _append_metric_line(metrics_path: Path, rec: dict[str, Any]) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
        fh.flush()


class _GpuSampler(threading.Thread):
    """Samples GPU utilisation / memory with ``nvidia-smi`` every ``interval`` seconds."""

    def __init__(self, interval: float = 1.0) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        self.samples: list[tuple[float, float, float]] = []  # (t, util%, mem_MiB)
        self._halt = threading.Event()

    def run(self) -> None:
        dev = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0"
        while not self._halt.is_set():
            try:
                out = subprocess.run(
                    [
                        "nvidia-smi",
                        f"--id={dev}",
                        "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                ).stdout.strip()
                util, mem = (float(x) for x in out.split(",")[:2])
                self.samples.append((time.time(), util, mem))
            except Exception:  # noqa: BLE001 - metrics must never kill training
                pass
            self._halt.wait(self.interval)

    def stop(self) -> None:
        self._halt.set()

    def window(self, t0: float, t1: float) -> tuple[float | None, float | None]:
        sel = [(u, m) for (t, u, m) in self.samples if t0 <= t <= t1]
        if not sel:
            return None, None
        return float(np.mean([u for u, _ in sel])), float(max(m for _, m in sel))


def _memory_mb(full: bool) -> dict[str, float]:
    """Host memory of the training process tree, in MiB.

    ``pss_tree_mb`` (proportional set size, summed over this process and all dataloader
    workers) counts shared pages once, unlike summed RSS which double-counts fork-shared
    memory. It is expensive (reads smaps), so it is only sampled when ``full`` is set.
    ``sys_used_mb`` is machine-wide used memory (meaningful on a dedicated box).
    """
    out: dict[str, float] = {}
    try:
        import psutil

        out["sys_used_mb"] = (psutil.virtual_memory().total - psutil.virtual_memory().available) / 2**20
        proc = psutil.Process()
        out["main_rss_mb"] = proc.memory_info().rss / 2**20
        if full:
            procs = [proc, *proc.children(recursive=True)]
            pss = 0
            for pr in procs:
                try:
                    pss += pr.memory_full_info().pss
                except (psutil.Error, AttributeError):
                    pass
            out["pss_tree_mb"] = pss / 2**20
            out["num_procs"] = float(len(procs))
    except Exception:  # noqa: BLE001
        pass
    return out


def _batch_fingerprint(batch) -> str:
    """Short hash of every tensor in a collated batch (strided sample, <=4096 values each).

    Unaffected by GPU nondeterminism, so two runs that train on the same data in the same
    order produce identical per-step fingerprints even when their losses drift apart.
    """
    import hashlib

    import torch

    h = hashlib.blake2b(digest_size=8)

    def walk(obj, key: str) -> None:
        if isinstance(obj, torch.Tensor):
            flat = obj.detach().reshape(-1)
            stride = max(1, flat.numel() // 4096)
            h.update(key.encode())
            h.update(flat[::stride].float().cpu().numpy().tobytes())
        elif hasattr(obj, "items"):
            for k in sorted(obj.keys(), key=str):
                walk(obj[k], f"{key}.{k}")
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                walk(v, f"{key}[{i}]")

    walk(batch, "")
    return h.hexdigest()


def make_bench_callback(
    metrics_path: Path,
    launch_t0: float,
    loss_box: dict,
    phase_box: dict[str, Any] | None = None,
    *,
    start_gpu_sampler: bool = True,
):
    from transformers import TrainerCallback

    phases = phase_box if phase_box is not None else {}

    class BenchMetricsCallback(TrainerCallback):
        def __init__(self) -> None:
            self.gpu = _GpuSampler()
            if start_gpu_sampler:
                self.gpu.start()
            self.fh = None
            self.t_prev_end: float | None = None
            self.t_step_begin: float | None = None
            self.data_wait = 0.0
            self._first_batch_recorded = False
            self._first_step_recorded = False
            self.peak = {"sys_used_mb": 0.0, "pss_tree_mb": 0.0, "main_rss_mb": 0.0}

        def _write(self, rec: dict) -> None:
            if self.fh is None:
                metrics_path.parent.mkdir(parents=True, exist_ok=True)
                self.fh = open(metrics_path, "a", encoding="utf-8")
            self.fh.write(json.dumps(rec) + "\n")
            self.fh.flush()

        def on_train_begin(self, args, state, control, **kwargs):
            now = time.time()
            rec: dict[str, Any] = {
                "event": "train_begin",
                "t": now,
                "since_launch_s": now - launch_t0,
                "global_step": int(state.global_step),
                "backend": os.environ.get("GR00T_DATA_BACKEND", "native"),
            }
            for k in ("imports_s", "dataset_build_s", "trainer_init_s", "dataloader_create_s", "mp_ctx"):
                if k in phases:
                    rec[k] = phases[k]
            self._write(rec)
            # Mark t_prev_end AFTER writing so callback I/O does not leak into step 1 data_wait_s.
            self.t_prev_end = time.time()

        def on_step_begin(self, args, state, control, **kwargs):
            now = time.time()
            # Only the first micro-step of an optimizer step waits on the dataloader after
            # the previous optimizer step finished (HF prefetches the rest via get_batch_samples).
            if self.t_step_begin is None and self.t_prev_end is not None:
                self.data_wait = now - self.t_prev_end
                self.t_step_begin = now
                if not self._first_batch_recorded:
                    self._first_batch_recorded = True
                    phases["first_batch_wait_s"] = self.data_wait
                    self._write(
                        {
                            "event": "phase",
                            "phase": "first_batch",
                            "t": now,
                            "since_launch_s": now - launch_t0,
                            "duration_s": self.data_wait,
                            "global_step": int(state.global_step),
                        }
                    )

        def on_step_end(self, args, state, control, **kwargs):
            # Capture step completion timestamp BEFORE expensive _memory_mb() / disk flush!
            now = time.time()
            t0 = self.t_prev_end or now
            step_s = now - t0
            util, gmem = self.gpu.window(t0, now)

            mem = _memory_mb(full=state.global_step % 10 == 0 or state.global_step <= 2)
            for k in self.peak:
                if k in mem:
                    self.peak[k] = max(self.peak[k], mem[k])
            rec = {
                "event": "step",
                "global_step": int(state.global_step),
                "t": now,
                "since_launch_s": now - launch_t0,
                "step_s": step_s,
                "data_wait_s": self.data_wait,
                **mem,
                "gpu_util": util,
                "gpu_mem_mb": gmem,
            }
            if "loss" in loss_box:
                # Raw per-step training loss (GR00T only logs a smoothed value every N steps).
                rec["step_loss"] = loss_box.pop("loss")
            if "fp" in loss_box:
                rec["batch_fp"] = loss_box.pop("fp")
            self._write(rec)

            if not self._first_step_recorded:
                self._first_step_recorded = True
                phases["first_step_s"] = step_s
                summary: dict[str, Any] = {
                    "event": "startup_summary",
                    "t": now,
                    "since_launch_s": now - launch_t0,
                    "global_step": int(state.global_step),
                    "first_batch_wait_s": phases.get("first_batch_wait_s", self.data_wait),
                    "first_step_s": step_s,
                }
                for k in (
                    "imports_s",
                    "dataset_build_s",
                    "trainer_init_s",
                    "dataloader_create_s",
                    "mp_ctx",
                    "lazy_decode",
                    "async_prefetch",
                ):
                    if k in phases:
                        summary[k] = phases[k]
                self._write(summary)

            self.t_step_begin = None
            if SKIP_FINAL_SAVE and state.global_step >= state.max_steps:
                # Benchmark runs: skip HF's end-of-training checkpoint (~24 GB here).
                control.should_save = False
            # Stamp t_prev_end RIGHT BEFORE returning to the dataloader next() call so
            # _memory_mb() and JSONL flush latency never pollute the next step's data_wait_s!
            self.t_prev_end = time.time()
            return control

        def on_save(self, args, state, control, **kwargs):
            now = time.time()
            self._write(
                {
                    "event": "save",
                    "global_step": int(state.global_step),
                    "t": now,
                    "since_launch_s": now - launch_t0,
                }
            )
            # Refresh t_prev_end after checkpoint save so save latency does not pollute data_wait_s.
            self.t_prev_end = time.time()

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                self._write({"event": "log", "global_step": int(state.global_step), "t": time.time(), **logs})

        def on_train_end(self, args, state, control, **kwargs):
            self._write(
                {
                    "event": "train_end",
                    "t": time.time(),
                    "since_launch_s": time.time() - launch_t0,
                    "global_step": int(state.global_step),
                    "peak": self.peak,
                }
            )
            self.gpu.stop()
            if self.fh:
                self.fh.close()
                self.fh = None

    return BenchMetricsCallback()


# ---------------------------------------------------------------------------
# Fork-safety & Async Prefetch helpers
# ---------------------------------------------------------------------------


def _check_datasets_unopened(train_dataset: Any) -> tuple[bool, str | None]:
    """Return ``(True, None)`` if every Lance dataset inside ``train_dataset`` is unopened."""
    candidates = getattr(train_dataset, "datasets", None)
    items = list(candidates) if candidates is not None else [train_dataset]
    for ds in items:
        checker = getattr(ds, "assert_unopened", None)
        if checker is None:
            loader = getattr(ds, "episode_loader", None)
            checker = getattr(loader, "assert_unopened", None)
        if checker is not None:
            try:
                checker()
            except RuntimeError as exc:
                return False, str(exc)
    return True, None


def resolve_lance_mp_context(train_dataset: Any, requested: str | None = None) -> str:
    """Choose the multiprocessing context for Lance DataLoader workers safely.

    Defaults to ``"fork"`` when ``assert_unopened()`` passes on every dataset; if Lance/Tokio
    was already touched in the main process (e.g. cold remote ``meta.lance`` read), raises if
    the caller explicitly forced ``"fork"`` or falls back to ``lance_mp_context()`` otherwise.
    """
    req = requested if requested is not None else os.environ.get("GR00T_LANCE_MP_CTX")
    if req is not None:
        req = req.strip().lower()
        if req and req not in _VALID_MP_CTXS:
            raise ValueError(f"Invalid GR00T_LANCE_MP_CTX={req!r}; expected one of {_VALID_MP_CTXS}")

    if req in ("forkserver", "spawn"):
        return req

    ok, err = _check_datasets_unopened(train_dataset)
    if ok:
        return "fork"

    if req == "fork":
        raise RuntimeError(f"GR00T_LANCE_MP_CTX=fork requested, but dataset is not unopened: {err}")

    from lerobot_lancedb import lance_mp_context

    fallback = lance_mp_context()
    logger.warning(
        "Dataset touched Lance/Tokio in main process (%s); falling back from 'fork' to %r.",
        err,
        fallback,
    )
    return fallback


def make_prewarmed_dataloader(base_dl):
    """Wrap a PyTorch ``DataLoader`` so batch 0 starts prefetching immediately on a background thread.

    The returned object is a ``torch.utils.data.DataLoader`` subclass so HuggingFace Trainer
    and ``accelerate.prepare()`` recognize it natively. The first ``iter()`` yields the
    prefetched batch 0 followed by the remaining batches from the already-running iterator.
    """
    from torch.utils.data import DataLoader

    class _PrewarmedDataLoader(DataLoader):
        def __init__(self, inner: DataLoader) -> None:
            # Copy DataLoader attributes without re-initializing samplers/workers.
            self.__dict__.update(inner.__dict__)
            self._inner_dl = inner
            self._prewarm_iter = iter(inner)
            self._prewarm_exec = ThreadPoolExecutor(max_workers=1)
            self._first_batch_fut: Future | None = self._prewarm_exec.submit(next, self._prewarm_iter)
            self._consumed = False

        def __iter__(self):
            if not self._consumed and self._first_batch_fut is not None:
                self._consumed = True
                fut = self._first_batch_fut
                it = self._prewarm_iter
                self._first_batch_fut = None
                self._prewarm_iter = None
                try:
                    first_batch = fut.result()
                except StopIteration:
                    self._shutdown_prewarm_exec()
                    return
                finally:
                    self._shutdown_prewarm_exec()
                yield first_batch
                yield from it
                return
            yield from super().__iter__()

        def _shutdown_prewarm_exec(self) -> None:
            ex = self._prewarm_exec
            if ex is not None:
                self._prewarm_exec = None
                ex.shutdown(wait=False)

        def close_prewarm(self) -> None:
            fut = self._first_batch_fut
            it = self._prewarm_iter
            self._first_batch_fut = None
            self._prewarm_iter = None
            if fut is not None:
                fut.cancel()
            if it is not None:
                shutdown = getattr(it, "_shutdown_workers", None)
                if shutdown is not None:
                    try:
                        shutdown()
                    except Exception:  # noqa: BLE001
                        pass
            self._shutdown_prewarm_exec()

    return _PrewarmedDataLoader(base_dl)


# ---------------------------------------------------------------------------
# Lance dataset factory
# ---------------------------------------------------------------------------


_URI_SCHEMES = ("s3", "gs", "gcs", "az", "abfs", "hf", "file", "http", "https")


def _rejoin_uris(paths: list) -> list:
    """Undo ``launch_finetune.py``'s ``split(os.pathsep)`` on URIs: ["s3", "//b/k"] -> ["s3://b/k"]."""
    out, i = [], 0
    while i < len(paths):
        p = str(paths[i])
        if p in _URI_SCHEMES and i + 1 < len(paths) and str(paths[i + 1]).startswith("//"):
            out.append(f"{p}:{paths[i + 1]}")
            i += 2
        else:
            out.append(p)
            i += 1
    return out


def _exact_build(self, processor, backend: str):  # bound as DatasetFactory.build
    """Mirror ``DatasetFactory.build`` (fast loader) but with the exact-resume mixture.

    ``backend == "native-exact"`` keeps GR00T's own ``FastShardedSingleStepDataset`` over the
    LeRobot v2.1 directory; ``backend == "lance"`` swaps in ``LanceShardedSingleStepDataset``.
    Everything else — sharding, weights, shard schedule, shuffle buffer — is GR00T's code.
    """
    from gr00t.data.dataset.fast_sharded_dataset import (
        FastShardedMixtureDataset,
        FastShardedSingleStepDataset,
        fast_loader_settings,
    )
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.stats import generate_rel_stats, generate_stats
    from gr00t.utils.dist_utils import run_or_wait_on_rank0

    from gr00t_lancedb import LanceShardedSingleStepDataset
    from gr00t_lancedb.exact_resume import make_exact_resume_mixture_class

    cfg = self.config
    assert cfg.training.eval_strategy == "no", "Sharded dataset does not support evaluation sets"
    fast = fast_loader_settings(cfg.data.fast_loader)

    all_datasets, all_weights = [], []
    for spec in cfg.data.datasets:
        emb = spec.embodiment_tag
        group = []
        for path in _rejoin_uris(list(spec.dataset_paths)):
            common = dict(
                dataset_path=path,
                embodiment_tag=EmbodimentTag(emb),
                modality_configs=cfg.data.modality_configs[emb],
                shard_size=cfg.data.shard_size,
                episode_sampling_rate=cfg.data.episode_sampling_rate,
                seed=cfg.data.seed,
                allow_padding=cfg.data.allow_padding,
            )
            if backend == "lance":
                # Stats were baked into meta.lance at conversion time: no rank-0 parquet scan.
                group.append(LanceShardedSingleStepDataset(**common))
            else:
                with run_or_wait_on_rank0(label=f"generate_stats({path})") as is_rank0:
                    if is_rank0:
                        generate_stats(path)
                        generate_rel_stats(path, EmbodimentTag(emb))
                group.append(FastShardedSingleStepDataset(**common, ffmpeg_threads=fast["ffmpeg_threads"]))
        lengths = np.array([len(ds) for ds in group], dtype=np.float64)
        for ds, rel in zip(group, lengths / lengths.sum()):
            all_datasets.append(ds)
            all_weights.append(float(rel) * spec.mix_ratio)

    alpha = cfg.data.ds_weights_alpha
    if alpha is not None and len(all_datasets) > 1:
        ds_lengths = np.array([len(d) for d in all_datasets], dtype=np.float64)
        all_weights = (np.power(ds_lengths, alpha) / np.power(ds_lengths[0], alpha)).tolist()

    default_lazy = "1" if backend == "lance" else "0"
    lazy_decode = os.environ.get("GR00T_LANCE_LAZY_DECODE", default_lazy) == "1"
    decode_batch_size = int(os.environ.get("GR00T_LANCE_DECODE_BATCH_SIZE", "32"))
    prefetch_batches = int(os.environ.get("GR00T_LANCE_PREFETCH_BATCHES", "2"))

    mixture_cls = make_exact_resume_mixture_class(FastShardedMixtureDataset)
    mix = mixture_cls(
        datasets=all_datasets,
        weights=all_weights,
        processor=processor,
        seed=cfg.data.seed,
        training=True,
        num_shards_per_epoch=cfg.data.num_shards_per_epoch,
        override_pretraining_statistics=cfg.data.override_pretraining_statistics,
        decode_threads=fast["decode_threads"],
        shuffle_buffer=fast["shuffle_buffer"],
        min_fill=fast["min_fill"],
        per_sample_seed=os.environ.get("GR00T_PER_SAMPLE_SEED", "1") == "1",
        lazy_decode=lazy_decode,
        decode_batch_size=decode_batch_size,
        prefetch_batches=prefetch_batches,
    )
    return mix, None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    launch_t0 = time.time()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gr00t-root", default=os.environ.get("GR00T_ROOT", "/data/Isaac-GR00T"))
    p.add_argument("gr00t_args", nargs=argparse.REMAINDER, help="Arguments after `--` go to launch_finetune.py")
    args = p.parse_args()
    gr00t_args = args.gr00t_args[1:] if args.gr00t_args[:1] == ["--"] else args.gr00t_args

    backend = os.environ.get("GR00T_DATA_BACKEND", "native").lower()
    # native       : stock GR00T (fast loader; resume = reseed with seed + global_step)
    # native-exact : GR00T's fast loader + datasets, exact-resume mixture/trainer
    # lance        : Lance datasets + the same exact-resume mixture/trainer
    assert backend in ("native", "native-exact", "lance"), backend
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    requested_mp_ctx = os.environ.get("GR00T_LANCE_MP_CTX")
    if requested_mp_ctx is not None:
        requested_mp_ctx = requested_mp_ctx.strip().lower() or None
        if requested_mp_ctx is not None and requested_mp_ctx not in _VALID_MP_CTXS:
            raise ValueError(
                f"Invalid GR00T_LANCE_MP_CTX={requested_mp_ctx!r}; expected one of {_VALID_MP_CTXS}"
            )

    # Only prewarm forkserver when forkserver is explicitly selected (default is safe 'fork').
    if (
        backend == "lance"
        and requested_mp_ctx == "forkserver"
        and os.environ.get("GR00T_LANCE_PREWARM", "1") == "1"
    ):
        from gr00t_lancedb import prewarm_lance_workers

        prewarm_lance_workers(
            extra_modules=(
                "gr00t.data.types",
                "gr00t.data.embodiment_tags",
                "gr00t.model.gr00t_n1d7.processing_gr00t_n1d7",
            )
        )

    def _output_dir_from_args() -> Path:
        out = "./outputs"
        exp = None
        for i, a in enumerate(gr00t_args):
            if a in ("--output-dir", "--output_dir") and i + 1 < len(gr00t_args):
                out = gr00t_args[i + 1]
            if a in ("--experiment-name", "--experiment_name") and i + 1 < len(gr00t_args):
                exp = gr00t_args[i + 1]
        return Path(out) / exp if exp else Path(out)

    metrics_path = Path(os.environ.get("GR00T_BENCH_METRICS", str(_output_dir_from_args() / "bench_metrics.jsonl")))
    phase_box: dict[str, Any] = {}

    t_imp0 = time.time()
    import gr00t.experiment.experiment as gexp
    from gr00t.data.dataset import factory as gfactory

    t_imp1 = time.time()
    phase_box["imports_s"] = t_imp1 - t_imp0
    _append_metric_line(
        metrics_path,
        {
            "event": "phase",
            "phase": "imports",
            "t": t_imp1,
            "since_launch_s": t_imp1 - launch_t0,
            "duration_s": t_imp1 - t_imp0,
            "backend": backend,
        },
    )

    base_trainer = gexp.Gr00tTrainer
    orig_build = gfactory.DatasetFactory.build

    if backend in ("native-exact", "lance"):
        from functools import partialmethod

        from gr00t_lancedb.exact_resume import make_exact_resume_trainer_class

        inner_build = partialmethod(_exact_build, backend=backend)
        base_trainer = make_exact_resume_trainer_class(base_trainer)
    else:
        inner_build = orig_build

    def _timed_build(self, processor):
        t0 = time.time()
        if isinstance(inner_build, type(orig_build)) or callable(inner_build):
            res = inner_build(self, processor)
        else:
            res = inner_build.__get__(self, type(self))(processor)
        t1 = time.time()
        phase_box["dataset_build_s"] = t1 - t0
        if hasattr(res[0], "lazy_decode"):
            phase_box["lazy_decode"] = bool(res[0].lazy_decode)
        _append_metric_line(
            metrics_path,
            {
                "event": "phase",
                "phase": "dataset_build",
                "t": t1,
                "since_launch_s": t1 - launch_t0,
                "duration_s": t1 - t0,
                "backend": backend,
            },
        )
        return res

    gfactory.DatasetFactory.build = _timed_build

    async_prefetch = (
        os.environ.get("GR00T_LANCE_ASYNC_PREFETCH", "0") == "1"
        or os.environ.get("GR00T_LANCE_PREFETCH_FIRST_BATCH", "0") == "1"
    )
    phase_box["async_prefetch"] = async_prefetch

    trainer_cls = base_trainer

    class BenchTrainer(trainer_cls):  # type: ignore[misc, valid-type]
        def __init__(self, *a, **kw):
            t_init0 = time.time()
            train_ds = kw.get("train_dataset")
            if backend == "lance":
                resolved_mp = resolve_lance_mp_context(train_ds, requested=requested_mp_ctx)
                kw["multiprocessing_context"] = resolved_mp
                phase_box["mp_ctx"] = resolved_mp
            else:
                phase_box["mp_ctx"] = kw.get("multiprocessing_context", "fork")
            super().__init__(*a, **kw)
            self._bench_loss: dict = {}
            self._prewarmed_dl = None
            self.add_callback(make_bench_callback(metrics_path, launch_t0, self._bench_loss, phase_box))
            t_init1 = time.time()
            phase_box["trainer_init_s"] = t_init1 - t_init0
            _append_metric_line(
                metrics_path,
                {
                    "event": "phase",
                    "phase": "trainer_init",
                    "t": t_init1,
                    "since_launch_s": t_init1 - launch_t0,
                    "duration_s": t_init1 - t_init0,
                    "mp_ctx": phase_box.get("mp_ctx"),
                },
            )

        def _build_base_dataloader(self):
            return super().get_train_dataloader()

        def train(self, resume_from_checkpoint=None, **kwargs):
            # Optional async prefetch: build & start DataLoader iterator before HF Trainer
            # loads checkpoint weights and sets up model/optimizer/scheduler.
            self._prewarmed_dl = None
            if async_prefetch and hasattr(self.train_dataset, "set_resume_point"):
                # Peek resume global_step so set_resume_point is exact before starting workers.
                ckpt = resume_from_checkpoint
                if ckpt is True:
                    from transformers.trainer_utils import get_last_checkpoint

                    ckpt = get_last_checkpoint(self.args.output_dir)
                if isinstance(ckpt, (str, Path)):
                    state_file = Path(ckpt) / "trainer_state.json"
                    if state_file.is_file():
                        from transformers import TrainerState

                        self.state = TrainerState.load_from_json(str(state_file))
                if getattr(self, "_train_batch_size", None) is None:
                    self._train_batch_size = self.args.train_batch_size
                t_dl0 = time.time()
                base_dl = self._build_base_dataloader()
                self._prewarmed_dl = make_prewarmed_dataloader(base_dl)
                t_dl1 = time.time()
                phase_box["dataloader_create_s"] = t_dl1 - t_dl0
                _append_metric_line(
                    metrics_path,
                    {
                        "event": "phase",
                        "phase": "dataloader_create",
                        "t": t_dl1,
                        "since_launch_s": t_dl1 - launch_t0,
                        "duration_s": t_dl1 - t_dl0,
                        "async_prefetch": True,
                    },
                )
            try:
                return super().train(resume_from_checkpoint=resume_from_checkpoint, **kwargs)
            finally:
                if self._prewarmed_dl is not None:
                    self._prewarmed_dl.close_prewarm()
                    self._prewarmed_dl = None

        def get_train_dataloader(self):
            if self._prewarmed_dl is not None:
                dl = self._prewarmed_dl
                self._prewarmed_dl = None
                return dl
            t0 = time.time()
            dl = super().get_train_dataloader()
            t1 = time.time()
            phase_box["dataloader_create_s"] = t1 - t0
            _append_metric_line(
                metrics_path,
                {
                    "event": "phase",
                    "phase": "dataloader_create",
                    "t": t1,
                    "since_launch_s": t1 - launch_t0,
                    "duration_s": t1 - t0,
                    "async_prefetch": False,
                },
            )
            return dl

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            out = super().compute_loss(
                model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
            )
            loss = out[0] if return_outputs else out
            if model.training:
                self._bench_loss["loss"] = float(loss.detach().float().item())
                self._bench_loss["fp"] = _batch_fingerprint(inputs)
            return out

        def save_model(self, output_dir=None, _internal_call=False):
            if SKIP_FINAL_SAVE and not _internal_call:
                logger.info("GR00T_BENCH_SKIP_FINAL_SAVE=1: skipping final save_model()")
                return
            return super().save_model(output_dir, _internal_call=_internal_call)

    BenchTrainer.__name__ = f"Bench{trainer_cls.__name__}"
    gexp.Gr00tTrainer = BenchTrainer

    script = Path(args.gr00t_root) / "gr00t" / "experiment" / "launch_finetune.py"
    sys.argv = [str(script), *gr00t_args]
    logger.info(
        "backend=%s mp_ctx=%s async_prefetch=%s metrics=%s argv=%s",
        backend,
        requested_mp_ctx or ("fork(auto)" if backend == "lance" else "default"),
        async_prefetch,
        metrics_path,
        sys.argv,
    )
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
