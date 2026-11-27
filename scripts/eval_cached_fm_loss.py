#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Probe loss of one or more checkpoints on the held-out part of a VL cache (~1 minute each).

Every checkpoint sees the same frames and the same ``(t, noise)``, so differences are paired:

    python scripts/eval_cached_fm_loss.py --cache-dir /data/tmp/libero10_vl_cache_full \\
        --ckpt native=/data/runs/e2e_ablation_3000s/group1_native_online/checkpoint-3000 \\
        --ckpt cache_v1=/data/runs/e2e_ablation_3000s/group2_vl_cache/merged_gr00t_ckpt \\
        --ckpt base=nvidia/GR00T-N1.7-3B --out-json probe_compare.json

What the three numbers above tell you:

* ``native`` far below ``cache_v1``: the cache-trained head really is worse on the same inputs
  (an optimisation problem), not merely unlucky in 50 rollouts.
* ``native`` close to ``base``: a natively trained head cannot read the int8 features, i.e. the
  cache itself is the problem (quantisation, padding, token order).
* ``native`` was trained on the held-out episodes too, so it has a small in-sample advantage
  over cache-trained checkpoints that used ``--holdout-frac``. Compare cache runs with each
  other for exact numbers.

Use the same ``--holdout-frac/--holdout-seed/--probe-rows`` as the training run (defaults match).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch

_REPO_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)


def find_cache(cache_dir: str, table_name: str | None) -> tuple[str, str]:
    """Return ``(table_name, scalars_path)``; with one cache in the directory no name is needed."""
    root = Path(cache_dir)
    if table_name is None:
        metas = sorted(root.glob("*_summary.json"))
        if len(metas) != 1:
            raise SystemExit(f"{len(metas)} caches in {root}; pass --table-name")
        table_name = json.loads(metas[0].read_text(encoding="utf-8"))["table_name"]
    scalars = root / f"{table_name}_scalars.npz"
    if not scalars.is_file():
        raise SystemExit(f"missing scalar side-car {scalars}")
    return table_name, str(scalars)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gr00t-root", default="/data/Isaac-GR00T")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--table-name", default=None)
    ap.add_argument("--ckpt", action="append", required=True, metavar="NAME=PATH", help="repeatable")
    ap.add_argument("--holdout-frac", type=float, default=0.03)
    ap.add_argument("--holdout-seed", type=int, default=0)
    ap.add_argument("--probe-rows", type=int, default=1024)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    if args.gr00t_root not in sys.path:
        sys.path.insert(0, args.gr00t_root)
    from gr00t_lancedb.vl_cache import load_action_head_only, probe_fm_loss
    from gr00t_lancedb.vl_cache.probe import paired_probe_diff, prepare_probe_isolated

    table_name, scalars_path = find_cache(args.cache_dir, args.table_name)
    probe = prepare_probe_isolated(
        args.cache_dir, table_name, scalars_path, args.holdout_frac, args.holdout_seed, args.probe_rows
    )
    seeds = [int(s) for s in args.seeds.split(",")]
    report: dict[str, Any] = {
        "cache_dir": args.cache_dir,
        "table_name": table_name,
        "probe_rows": int(len(probe["probe_rows"])),
        "held_out_episodes": probe["held_out_episodes"],
        "checkpoints": {},
        "paired_vs_first": {},
    }
    per_sample: dict[str, Any] = {}
    for item in args.ckpt:
        name, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--ckpt expects NAME=PATH, got {item!r}")
        ah, _, _ = load_action_head_only(path, device="cuda", dtype=torch.float32)
        rec = probe_fm_loss(
            ah, probe["probe_batches"], device="cuda", seeds=seeds, amp_dtype=torch.bfloat16, return_per_sample=True
        )
        per_sample[name] = rec.pop("per_sample")
        rec["path"] = path
        report["checkpoints"][name] = rec
        print(f"[probe] {name}: {rec['probe_loss']:.5f} +- {rec['probe_sem']:.5f}  ({path})", flush=True)
        del ah
        torch.cuda.empty_cache()
    names = list(per_sample)
    for name in names[1:]:
        d = paired_probe_diff(per_sample[names[0]], per_sample[name])
        report["paired_vs_first"][f"{name} - {names[0]}"] = d
        print(f"[probe] {name} - {names[0]}: {d['diff']:+.5f} +- {d['sem']:.5f} (z={d['z']:+.1f})")
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
