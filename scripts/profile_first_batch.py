#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Profile time-to-first-batch of the Lance mixture inside a real GR00T environment.

Uses a processor saved by a previous GR00T run (``<run>/processor``) so the per-sample
work matches training exactly.
"""

from __future__ import annotations

import argparse
import pickle
import time

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--processor-dir", required=True)
    p.add_argument("--lance-dir", required=True)
    p.add_argument("--embodiment-tag", default="libero_sim")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, nargs="+", default=[0, 4])
    p.add_argument("--mp", default=None, help="multiprocessing context override")
    p.add_argument("--prewarm", action="store_true")
    args = p.parse_args()

    if args.prewarm:
        from gr00t_lancedb import prewarm_lance_workers

        prewarm_lance_workers(("gr00t.data.types", "gr00t.model.gr00t_n1d7.processing_gr00t_n1d7"))

    t = time.perf_counter()
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
    from torch.utils.data import DataLoader

    from gr00t_lancedb import LanceShardedSingleStepDataset, ResumableLanceMixtureDataset
    from lerobot_lancedb import lance_mp_context

    print(f"imports {time.perf_counter() - t:.2f}s")

    t = time.perf_counter()
    processor = Gr00tN1d7Processor.from_pretrained(args.processor_dir)
    processor.train()
    print(f"processor load {time.perf_counter() - t:.2f}s; pickled {len(pickle.dumps(processor)) / 2**20:.1f} MiB")

    emb = EmbodimentTag(args.embodiment_tag)
    t = time.perf_counter()
    ds = LanceShardedSingleStepDataset(args.lance_dir, emb, MODALITY_CONFIGS[emb.value], seed=42)
    mix = ResumableLanceMixtureDataset(
        [ds], [1.0], processor=processor, seed=42, batch_size=args.batch_size, infinite=True
    )
    print(f"dataset init {time.perf_counter() - t:.2f}s; pickled {len(pickle.dumps(mix)) / 2**20:.1f} MiB")

    for nw in args.num_workers:
        kw = {}
        if nw > 0:
            kw = {"multiprocessing_context": args.mp or lance_mp_context(), "persistent_workers": True}
        dl = DataLoader(mix, batch_size=args.batch_size, collate_fn=processor.collator, num_workers=nw, **kw)
        t = time.perf_counter()
        it = iter(dl)
        t_iter = time.perf_counter() - t
        b = next(it)
        t_first = time.perf_counter() - t
        times = []
        for _ in range(10):
            s = time.perf_counter()
            next(it)
            times.append(time.perf_counter() - s)
        print(
            f"num_workers={nw} ctx={kw.get('multiprocessing_context')}: iter() {t_iter:.2f}s, "
            f"first batch {t_first:.2f}s, next 10 mean {np.mean(times):.3f}s "
            f"({args.batch_size / np.mean(times):.0f} samples/s); keys={list(b)[:3]}"
        )
        del it, dl
    mix.close()


if __name__ == "__main__":
    main()
