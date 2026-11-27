#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Sample-stream equivalence + exact-resume check against real Isaac-GR00T (CPU only).

Builds the batch stream a ``DataLoader`` delivers from

  A. GR00T's stock ``FastShardedMixtureDataset`` over its own LeRobot v2.1 datasets,
  B. the exact-resume mixture over the same GR00T datasets,
  C. the exact-resume mixture over the converted Lance dataset,
  D. C again, but started as a resume after ``--resume-at`` consumed batches,

and checks A == B == C (bit-exact, same order) and D == C[resume_at:].

E4 data-only mode (``--resume-points random:100:7`` or ``3,9,17``): for every resume point
``r`` it restarts the Lance exact-resume loader at ``r`` *and* simulates stock GR00T's
resume (``reset_seed(seed + r)`` on a fresh ``FastShardedMixtureDataset``), then reports
per-sample order match, duplicate and miss rates of the resumed tail against the
uninterrupted stream. No GPU / model needed, so E4 at 100 preemptions costs minutes.
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


def parse_resume_points(spec: str, batches: int) -> list[int]:
    if spec.startswith("random:"):
        _, count, seed = spec.split(":")
        rng = np.random.default_rng(int(seed))
        hi = max(2, batches)
        return sorted(int(x) for x in rng.integers(1, hi, size=int(count)))
    return sorted(int(x) for x in spec.split(",") if x.strip())


def tail_metrics(ref_tail: list[list], test_tail: list[list]) -> dict:
    """Per-sample order match / duplicate / miss of a resumed tail vs the reference tail."""
    from collections import Counter

    ref = [s for b in ref_tail for s in b]
    test = [s for b in test_tail for s in b]
    rc, tc = Counter(ref), Counter(test)
    pos = sum(a == b for a, b in zip(ref, test))
    return {
        "samples": len(ref),
        "identical": ref_tail == test_tail,
        "order_match": pos / max(1, len(ref)),
        "dup_rate": sum(max(0, c - rc.get(h, 0)) for h, c in tc.items()) / max(1, len(test)),
        "miss_rate": sum(max(0, c - tc.get(h, 0)) for h, c in rc.items()) / max(1, len(ref)),
    }


def summarize_tails(rows: list[dict]) -> dict:
    if not rows:
        return {}
    return {
        "runs": len(rows),
        "all_identical": all(r["identical"] for r in rows),
        "identical_runs": sum(r["identical"] for r in rows),
        "order_match_mean": float(np.mean([r["order_match"] for r in rows])),
        "dup_rate_mean": float(np.mean([r["dup_rate"] for r in rows])),
        "miss_rate_mean": float(np.mean([r["miss_rate"] for r in rows])),
        "samples_checked": int(sum(r["samples"] for r in rows)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--v2-dir", required=True)
    p.add_argument("--lance-dir", required=True)
    p.add_argument("--embodiment-tag", default="libero_sim")
    p.add_argument("--batches", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--resume-at", type=int, default=9)
    p.add_argument("--resume-points", default=None,
                   help="E4 mode: comma list (3,9,17) or random:N:SEED resume points in [1, batches).")
    p.add_argument("--lazy-decode", action="store_true", help="Use the index-only lazy-decode buffer for C/D.")
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

    if args.resume_points:
        points = parse_resume_points(args.resume_points, n)
        exact_rows, native_rows = [], []
        for r in points:
            ex = Exact(datasets=[lance_ds()], per_sample_seed=False, lazy_decode=args.lazy_decode, **mix_kw)
            ex.set_resume_point(r, bs)
            tail, _ = take_batches(ex, n - r, bs, nw, mp)
            exact_rows.append(tail_metrics(C[r:], tail))
            stock = FastShardedMixtureDataset(datasets=[native_ds()], **mix_kw)
            stock.reset_seed(42 + r)  # what Gr00tTrainer does on resume: seed + global_step
            tail_n, _ = take_batches(stock, n - r, bs, nw, mp)
            native_rows.append(tail_metrics(A[r:], tail_n))
        report["e4"] = {
            "resume_points": points,
            "lance_exact": summarize_tails(exact_rows),
            "native_reseed": summarize_tails(native_rows),
        }
        report["pass"] = report["pass"] and report["e4"]["lance_exact"]["all_identical"]
    print(json.dumps(report, indent=2))
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(report, f, indent=2)
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
