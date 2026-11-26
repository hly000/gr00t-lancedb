#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Sample-stream equivalence + exact-resume check against real Isaac-GR00T (CPU only).

Builds the batch stream a ``DataLoader`` delivers from

  A. GR00T's stock ``FastShardedMixtureDataset`` over its own LeRobot v2.1 datasets,
  B. the exact-resume mixture over the same GR00T datasets,
  C. the exact-resume mixture over the converted Lance dataset,
  D. C again, but started as a resume after ``--resume-at`` consumed batches,

and checks A == B == C (bit-exact, same order) and D == C[resume_at:].
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np


class PassThroughProcessor:
    """Stand-in for the GR00T processor: returns the raw ``VLAStepData``."""

    def set_statistics(self, *_a, **_k):
        pass

    def __call__(self, messages):
        return messages[0]["content"]


def identity_collate(items):
    return items


def fingerprint(step) -> tuple:
    parts = [step.text]
    for group in (step.states, step.actions):
        for k in sorted(group):
            parts.append(np.asarray(group[k]).tobytes())
    for k in sorted(step.images):
        for frame in step.images[k]:
            parts.append(np.asarray(frame)[::8, ::8].tobytes())
    return tuple(hash(p) for p in parts)


def take_batches(ds, n, batch_size, num_workers, mp):
    from torch.utils.data import DataLoader

    kw = {"multiprocessing_context": mp, "persistent_workers": False} if num_workers else {}
    dl = DataLoader(ds, batch_size=batch_size, collate_fn=identity_collate, num_workers=num_workers, **kw)
    out = []
    t = time.perf_counter()
    first = None
    for i, batch in enumerate(dl):
        if first is None:
            first = time.perf_counter() - t
        out.append([fingerprint(s) for s in batch])
        if i + 1 >= n:
            break
    return out, first


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--v2-dir", required=True)
    p.add_argument("--lance-dir", required=True)
    p.add_argument("--embodiment-tag", default="libero_sim")
    p.add_argument("--batches", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--resume-at", type=int, default=9)
    p.add_argument("--out-json", default=None)
    args = p.parse_args()

    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.dataset.fast_sharded_dataset import FastShardedMixtureDataset, FastShardedSingleStepDataset
    from gr00t.data.embodiment_tags import EmbodimentTag

    from gr00t_lancedb import LanceShardedSingleStepDataset
    from gr00t_lancedb.exact_resume import make_exact_resume_mixture_class
    from lerobot_lancedb import lance_mp_context

    emb = EmbodimentTag(args.embodiment_tag)
    cfg = MODALITY_CONFIGS[emb.value]
    mp = lance_mp_context()
    Exact = make_exact_resume_mixture_class(FastShardedMixtureDataset)
    common = dict(embodiment_tag=emb, modality_configs=cfg, seed=42)
    mix_kw = dict(weights=[1.0], processor=PassThroughProcessor(), seed=42, decode_threads=2)

    def native_ds():
        return FastShardedSingleStepDataset(dataset_path=args.v2_dir, ffmpeg_threads=1, **common)

    def lance_ds():
        return LanceShardedSingleStepDataset(dataset_path=args.lance_dir, **common)

    n, bs, nw = args.batches, args.batch_size, args.num_workers
    A, tA = take_batches(FastShardedMixtureDataset(datasets=[native_ds()], **mix_kw), n, bs, nw, mp)
    B, tB = take_batches(Exact(datasets=[native_ds()], per_sample_seed=False, **mix_kw), n, bs, nw, mp)
    C, tC = take_batches(Exact(datasets=[lance_ds()], per_sample_seed=False, **mix_kw), n, bs, nw, mp)
    dres = Exact(datasets=[lance_ds()], per_sample_seed=False, **mix_kw)
    dres.set_resume_point(args.resume_at, bs)
    D, tD = take_batches(dres, n - args.resume_at, bs, nw, mp)

    report = {
        "batches": n,
        "batch_size": bs,
        "num_workers": nw,
        "resume_at": args.resume_at,
        "stock_eq_exact_native": A == B,
        "native_eq_lance": A == C,
        "resumed_eq_tail": D == C[args.resume_at :],
        "first_batch_s": {"stock": round(tA, 2), "exact_native": round(tB, 2), "lance": round(tC, 2), "lance_resumed": round(tD, 2)},
        "distinct_samples_checked": len({s for b in A for s in b}),
    }
    report["pass"] = report["stock_eq_exact_native"] and report["native_eq_lance"] and report["resumed_eq_tail"]
    print(json.dumps(report, indent=2))
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(report, f, indent=2)
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
