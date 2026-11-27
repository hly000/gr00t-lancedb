#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Where does the time go? Per-split cost of GR00T native vs Lance (single process, no workers).

Times: dataset construction, first reader open, ``load_split`` per split (same splits for
both backends), and the processor (``process_split_item``) per sample.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import time


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--v2-dir", required=True)
    p.add_argument("--lance-dir", action="append", required=True)
    p.add_argument("--processor-dir", required=True)
    p.add_argument("--embodiment-tag", default="libero_sim")
    p.add_argument("--splits", type=int, default=24)
    args = p.parse_args()

    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.dataset.fast_sharded_dataset import FastShardedSingleStepDataset
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor

    from gr00t_lancedb import LanceShardedSingleStepDataset

    emb = EmbodimentTag(args.embodiment_tag)
    cfg = MODALITY_CONFIGS[emb.value]
    proc = Gr00tN1d7Processor.from_pretrained(args.processor_dir)
    proc.train()

    out: dict = {}

    def run(name, make):
        t = time.perf_counter()
        ds = make()
        t_init = time.perf_counter() - t
        ds.set_processor(proc) if hasattr(ds, "set_processor") else setattr(ds, "processor", proc)
        splits = [sp for shard in range(len(ds)) for sp in ds.get_shard_splits(shard)][: args.splits]
        times, n_items = [], 0
        t_first = None
        for i, sp in enumerate(splits):
            t = time.perf_counter()
            items = ds.load_split(sp)
            dt = time.perf_counter() - t
            if i == 0:
                t_first = dt
            else:
                times.append(dt)
            n_items += len(items)
        t = time.perf_counter()
        for it in items:
            ds.process_split_item(it)
        t_proc = (time.perf_counter() - t) / len(items)
        rec = {
            "init_s": round(t_init, 3),
            "first_split_s": round(t_first, 3),
            "split_s_median": round(st.median(times), 3),
            "steps_per_split": round(n_items / len(splits), 1),
            "load_ms_per_sample": round(1000 * st.median(times) / (n_items / len(splits)), 1),
            "processor_ms_per_sample": round(1000 * t_proc, 1),
        }
        out[name] = rec
        print(name, json.dumps(rec), flush=True)

    common = dict(embodiment_tag=emb, modality_configs=cfg, seed=42)
    run("native", lambda: FastShardedSingleStepDataset(dataset_path=args.v2_dir, ffmpeg_threads=1, **common))
    for d in args.lance_dir:
        run(f"lance:{d}", lambda d=d: LanceShardedSingleStepDataset(dataset_path=d, **common))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
