#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Data-only (no model, no GPU) multi-epoch loader benchmark: throughput, RAM, object-store I/O.

Answers the E2 / issue #3 questions that do not need a GPU: how many Lance blob range
requests and bytes one sample costs, how much the GOP disk cache saves on epoch 2+, and
what the decoder-cache size / decode-chunk size do to worker RAM and samples/s.

With Isaac-GR00T installed it drives GR00T's own ``FastShardedMixtureDataset`` (through the
exact-resume lazy-decode mixture, i.e. the real training access pattern). Without GR00T it
falls back to a shuffled-anchor iterable (worst-case random access) so it also runs on a
CPU box.

    python scripts/bench_loader_io.py --dataset gs://bucket/libero_10_lance \\
        --storage-option aws_endpoint=https://storage.googleapis.com ... \\
        --embodiment-tag libero_sim --epochs 2 --batches-per-epoch 400 \\
        --video-cache-gb 20 --video-cache-dir /local/gop --out results/v1/loader_io/c1_cache.json

Each epoch uses a fresh DataLoader (``epoch`` advanced, so a new order) and its own I/O
stats directory; per-epoch deltas are reported.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np


def _kv(items: list[str] | None) -> dict[str, str] | None:
    if not items:
        return None
    return dict(item.split("=", 1) for item in items)


def _tree_memory_mb() -> dict[str, float]:
    try:
        import psutil

        me = psutil.Process()
        procs = [me, *me.children(recursive=True)]
        rss = sum(p.memory_info().rss for p in procs if p.is_running())
        pss = 0
        for pr in procs:
            try:
                pss += pr.memory_full_info().pss
            except (psutil.Error, AttributeError):
                pass
        return {"rss_tree_mb": rss / 2**20, "pss_tree_mb": pss / 2**20, "procs": float(len(procs))}
    except Exception:  # noqa: BLE001
        return {}


def _modality_configs(args, dataset_meta_dir: Path | None):
    from gr00t_lancedb import ModalityConfig

    try:
        from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS  # type: ignore[import-not-found]

        return MODALITY_CONFIGS[args.embodiment_tag], True
    except Exception:  # noqa: BLE001 - no GR00T: derive from modality.json
        pass
    if dataset_meta_dir is None:
        raise SystemExit("Without Isaac-GR00T pass a local dataset (to read meta/modality.json).")
    mod = json.loads((dataset_meta_dir / "modality.json").read_text())
    cfg = {
        "state": ModalityConfig(delta_indices=[0], modality_keys=list(mod.get("state", {}))),
        "action": ModalityConfig(delta_indices=list(range(args.action_horizon)), modality_keys=list(mod.get("action", {}))),
    }
    if mod.get("video"):
        cfg["video"] = ModalityConfig(delta_indices=[0], modality_keys=list(mod["video"]))
    return cfg, False


from torch.utils.data import IterableDataset  # noqa: E402


class _AnchorStream(IterableDataset):
    """Fallback: shuffled anchors, sharded across workers, decoded in chunks (no GR00T)."""

    def __init__(self, ds, seed: int, chunk: int) -> None:
        super().__init__()
        self.ds, self.seed, self.chunk, self.epoch = ds, seed, chunk, 0

    def __iter__(self):
        from torch.utils.data import get_worker_info

        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        anchors = self.ds.get_all_valid_anchors()
        order = np.random.default_rng(self.seed + self.epoch).permutation(len(anchors))[wid::nw]
        for i in range(0, len(order), self.chunk):
            for item in self.ds.load_anchor_batch([anchors[j] for j in order[i : i + self.chunk]]):
                yield item


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="Local Lance dir or s3:// / gs:// URI.")
    ap.add_argument("--storage-option", action="append", default=None, help="KEY=VALUE (repeatable).")
    ap.add_argument("--embodiment-tag", default="libero_sim")
    ap.add_argument("--action-horizon", type=int, default=16, help="Fallback configs only.")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batches-per-epoch", type=int, default=0, help="0 = one full pass over all anchors.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--first-decode-batch", type=int, default=32)
    ap.add_argument("--steady-decode-batch", type=int, default=128)
    ap.add_argument("--decoder-cache", type=int, default=64)
    ap.add_argument("--video-cache-gb", type=float, default=0.0)
    ap.add_argument("--video-cache-dir", default=None)
    ap.add_argument("--video-cache-force-local", action="store_true")
    ap.add_argument("--where", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    import torch
    from torch.utils.data import DataLoader

    from gr00t_lancedb import EmbodimentTag, LanceShardedSingleStepDataset, aggregate_io_stats
    from gr00t_lancedb.exact_resume import make_exact_resume_mixture_class

    local_meta = Path(args.dataset) / "meta" if "://" not in args.dataset else None
    cfg, have_groot = _modality_configs(args, local_meta)
    try:
        emb = EmbodimentTag(args.embodiment_tag)
    except ValueError:
        emb = EmbodimentTag.NEW_EMBODIMENT
    stats_root = Path(tempfile.mkdtemp(prefix="gr00t-loader-io-"))

    t0 = time.perf_counter()
    ds = LanceShardedSingleStepDataset(
        dataset_path=args.dataset, embodiment_tag=emb, modality_configs=cfg, seed=args.seed,
        storage_options=_kv(args.storage_option), video_decoder_cache_size=args.decoder_cache,
        video_cache_dir=args.video_cache_dir, video_cache_gb=args.video_cache_gb or None,
        video_cache_force_local=args.video_cache_force_local or None, where=args.where,
    )
    build_s = time.perf_counter() - t0
    n_anchors = sum(len(ds.get_valid_steps_for_episode(e)) for e in range(len(ds.episode_loader)))
    per_epoch = args.batches_per_epoch or int(np.ceil(n_anchors / args.batch_size))

    if have_groot:
        from gr00t.data.dataset.fast_sharded_dataset import FastShardedMixtureDataset  # type: ignore

        class _Pass:
            def set_statistics(self, *_a, **_k):
                pass

            def __call__(self, messages):
                return messages[0]["content"]

        stream = make_exact_resume_mixture_class(FastShardedMixtureDataset)(
            datasets=[ds], weights=[1.0], processor=_Pass(), seed=args.seed, decode_threads=2,
            lazy_decode=True, first_decode_batch_size=args.first_decode_batch,
            decode_batch_size=args.steady_decode_batch, prefetch_batches=2,
        )
        pattern = "gr00t_fast_sharded_mixture"
    else:
        stream = _AnchorStream(ds, args.seed, args.steady_decode_batch)
        pattern = "random_anchors_fallback"

    epochs = []
    for epoch in range(args.epochs):
        stats_dir = stats_root / f"epoch-{epoch}"
        os.environ["GR00T_LANCE_IO_STATS_DIR"] = str(stats_dir)
        if hasattr(stream, "epoch"):
            stream.epoch = epoch
        kw = {"multiprocessing_context": "fork", "persistent_workers": False} if args.num_workers else {}
        dl = DataLoader(stream, batch_size=args.batch_size, num_workers=args.num_workers,
                        collate_fn=lambda x: x, **kw)
        t_ep = time.perf_counter()
        first = None
        n_samples = 0
        peak: dict[str, float] = {}
        it = iter(dl)
        for b in range(per_epoch):
            try:
                batch = next(it)
            except StopIteration:
                break
            if first is None:
                first = time.perf_counter() - t_ep
            n_samples += len(batch)
            if b % 50 == 0:
                for k, v in _tree_memory_mb().items():
                    peak[k] = max(peak.get(k, 0.0), v)
        wall = time.perf_counter() - t_ep
        del it, dl  # joins workers; their exit hooks flush the I/O counters
        time.sleep(0.5)
        io = aggregate_io_stats(stats_dir)
        hits, misses = io.get("gop_cache_hits", 0), io.get("gop_cache_misses", 0)
        epochs.append({
            "epoch": epoch,
            "samples": n_samples,
            "wall_s": round(wall, 2),
            "first_batch_s": round(first or 0.0, 2),
            "samples_per_s": round(n_samples / wall, 1) if wall else None,
            "lance_ranges_per_sample": round(io.get("lance_blob_ranges", 0) / max(1, n_samples), 4),
            "lance_kb_per_sample": round(io.get("lance_blob_bytes", 0) / 1024 / max(1, n_samples), 2),
            "requested_kb_per_sample": round(io.get("requested_bytes", 0) / 1024 / max(1, n_samples), 2),
            "frames_takes_per_sample": round(io.get("frames_take_calls", 0) / max(1, n_samples), 4),
            "gop_cache_hit_rate": round(hits / (hits + misses), 4) if hits + misses else None,
            "peak_mem": {k: round(v, 1) for k, v in peak.items()},
            "io": io,
        })
        print(json.dumps({k: v for k, v in epochs[-1].items() if k != "io"}))

    report = {
        "dataset": args.dataset,
        "access_pattern": pattern,
        "anchors": n_anchors,
        "dataset_build_s": round(build_s, 2),
        "config": {k: getattr(args, k) for k in ("batch_size", "num_workers", "first_decode_batch",
                                                 "steady_decode_batch", "decoder_cache", "video_cache_gb", "where")},
        "torch": torch.__version__,
        "epochs": epochs,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    main()
