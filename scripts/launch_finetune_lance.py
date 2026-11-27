#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Run NVIDIA Isaac-GR00T's *official* ``launch_finetune.py`` with a selectable data backend.

Everything (model, processor, hyper-parameters, CLI flags) comes from GR00T itself; this
wrapper only swaps seams, selected by ``GR00T_DATA_BACKEND``:

``native``        stock GR00T (fast loader; resume reseeds with ``seed + global_step``)
``native-exact``  GR00T's own fast datasets + ``ExactResumeFastShardedMixtureDataset`` and
                  an exact-resume ``Gr00tTrainer`` (no Lance involved)
``lance``         ``LanceShardedSingleStepDataset`` (local dir or ``s3://``) + the same
                  exact-resume mixture/trainer; forkserver workers are pre-warmed

In every mode a ``BenchMetricsCallback`` appends one JSON line per optimizer step to
``$GR00T_BENCH_METRICS`` (default ``<output_dir>/bench_metrics.jsonl``): wall time, time
waiting for the batch, host memory, GPU util/memory and the raw per-step loss.

Usage::

    GR00T_DATA_BACKEND=lance python scripts/launch_finetune_lance.py \
        --gr00t-root /data/Isaac-GR00T -- \
        --base-model-path nvidia/GR00T-N1.7-3B --dataset-path /data/datasets/libero_10_lance \
        --embodiment-tag LIBERO_PANDA --output-dir /data/runs/lance ...
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import runpy
import subprocess
import sys
import threading
import time

import numpy as np

logger = logging.getLogger("gr00t_lancedb.launch")

# Benchmark-only: skip the end-of-training checkpoint and GR00T's final save_model().
SKIP_FINAL_SAVE = os.environ.get("GR00T_BENCH_SKIP_FINAL_SAVE", "0") == "1"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


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


def make_bench_callback(metrics_path: Path, launch_t0: float, loss_box: dict):
    from transformers import TrainerCallback

    class BenchMetricsCallback(TrainerCallback):
        def __init__(self) -> None:
            self.gpu = _GpuSampler()
            self.gpu.start()
            self.fh = None
            self.t_prev_end: float | None = None
            self.t_step_begin: float | None = None
            self.data_wait = 0.0
            self.peak = {"sys_used_mb": 0.0, "pss_tree_mb": 0.0, "main_rss_mb": 0.0}

        def _write(self, rec: dict) -> None:
            if self.fh is None:
                metrics_path.parent.mkdir(parents=True, exist_ok=True)
                self.fh = open(metrics_path, "a", encoding="utf-8")
            self.fh.write(json.dumps(rec) + "\n")
            self.fh.flush()

        def on_train_begin(self, args, state, control, **kwargs):
            self.t_prev_end = time.time()
            rec = {
                "event": "train_begin",
                "t": self.t_prev_end,
                "since_launch_s": self.t_prev_end - launch_t0,
                "global_step": int(state.global_step),
                "backend": os.environ.get("GR00T_DATA_BACKEND", "native"),
            }
            if os.environ.get("GR00T_LANCE_WHERE"):
                rec["lance_where"] = os.environ["GR00T_LANCE_WHERE"]
            if os.environ.get("GR00T_LANCE_VERSION"):
                rec["lance_version"] = int(os.environ["GR00T_LANCE_VERSION"])
            self._write(rec)

        def on_step_begin(self, args, state, control, **kwargs):
            now = time.time()
            # Only the first micro-step of an optimizer step waits on the dataloader after
            # the previous optimizer step finished (HF prefetches the rest via get_batch_samples).
            if self.t_step_begin is None and self.t_prev_end is not None:
                self.data_wait = now - self.t_prev_end
                self.t_step_begin = now

        def on_step_end(self, args, state, control, **kwargs):
            now = time.time()
            mem = _memory_mb(full=state.global_step % 10 == 0 or state.global_step <= 2)
            for k in self.peak:
                if k in mem:
                    self.peak[k] = max(self.peak[k], mem[k])
            t0 = self.t_prev_end or now
            util, gmem = self.gpu.window(t0, now)
            rec = {
                "event": "step",
                "global_step": int(state.global_step),
                "t": now,
                "since_launch_s": now - launch_t0,
                "step_s": now - t0,
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
            self.t_prev_end = now
            self.t_step_begin = None
            if SKIP_FINAL_SAVE and state.global_step >= state.max_steps:
                # Benchmark runs: skip HF's end-of-training checkpoint (~24 GB here).
                control.should_save = False
            return control

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

    lance_where = os.environ.get("GR00T_LANCE_WHERE") or None
    lance_version_raw = os.environ.get("GR00T_LANCE_VERSION")
    lance_version = int(lance_version_raw) if lance_version_raw else None

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
                lance_kw = dict(common)
                if lance_where is not None:
                    lance_kw["where"] = lance_where
                if lance_version is not None:
                    lance_kw["lance_version"] = lance_version
                group.append(LanceShardedSingleStepDataset(**lance_kw))
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
    )
    return mix, None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    launch_t0 = time.time()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gr00t-root", default=os.environ.get("GR00T_ROOT", "/data/Isaac-GR00T"))
    p.add_argument(
        "--lance-where",
        default=os.environ.get("GR00T_LANCE_WHERE"),
        help="Optional SQL WHERE predicate pushed down to frames.lance (e.g. 'is_valid = true').",
    )
    p.add_argument(
        "--lance-version",
        type=int,
        default=int(os.environ["GR00T_LANCE_VERSION"]) if os.environ.get("GR00T_LANCE_VERSION") else None,
        help="Optional Lance frames table version for reproducible time-travel training.",
    )
    p.add_argument("gr00t_args", nargs=argparse.REMAINDER, help="Arguments after `--` go to launch_finetune.py")
    args = p.parse_args()
    if args.lance_where:
        os.environ["GR00T_LANCE_WHERE"] = args.lance_where
    if args.lance_version is not None:
        os.environ["GR00T_LANCE_VERSION"] = str(args.lance_version)
    gr00t_args = args.gr00t_args[1:] if args.gr00t_args[:1] == ["--"] else args.gr00t_args

    backend = os.environ.get("GR00T_DATA_BACKEND", "native").lower()
    # native       : stock GR00T (fast loader; resume = reseed with seed + global_step)
    # native-exact : GR00T's fast loader + datasets, exact-resume mixture/trainer
    # lance        : Lance datasets + the same exact-resume mixture/trainer
    assert backend in ("native", "native-exact", "lance"), backend
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if backend == "lance" and os.environ.get("GR00T_LANCE_PREWARM", "1") == "1":
        # Start the forkserver before the heavy GR00T/model imports so worker-side imports
        # overlap with model loading instead of delaying the first batch.
        from gr00t_lancedb import prewarm_lance_workers

        prewarm_lance_workers(
            extra_modules=(
                "gr00t.data.types",
                "gr00t.data.embodiment_tags",
                "gr00t.model.gr00t_n1d7.processing_gr00t_n1d7",
            )
        )

    import gr00t.experiment.experiment as gexp
    from gr00t.data.dataset import factory as gfactory

    base_trainer = gexp.Gr00tTrainer
    if backend in ("native-exact", "lance"):
        from functools import partialmethod

        from gr00t_lancedb.exact_resume import make_exact_resume_trainer_class

        gfactory.DatasetFactory.build = partialmethod(_exact_build, backend=backend)
        base_trainer = make_exact_resume_trainer_class(base_trainer)

    # Attach the metrics callback to whichever trainer runs.
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
    trainer_cls = base_trainer

    class BenchTrainer(trainer_cls):  # type: ignore[misc, valid-type]
        def __init__(self, *a, **kw):
            if backend == "lance" and kw.get("multiprocessing_context", "fork") == "fork":
                # Lance's Rust/Tokio runtime is not fork-safe (prewarmed forkserver instead).
                from lerobot_lancedb import lance_mp_context

                kw["multiprocessing_context"] = lance_mp_context()
            super().__init__(*a, **kw)
            self._bench_loss: dict = {}
            self.add_callback(make_bench_callback(metrics_path, launch_t0, self._bench_loss))

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
    logger.info("backend=%s metrics=%s argv=%s", backend, metrics_path, sys.argv)
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
