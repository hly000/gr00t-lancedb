#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Bit-exact parity check against the *real* NVIDIA Isaac-GR00T data path.

Loads the same (episode, step) anchors through

* GR00T's own ``LeRobotEpisodeLoader`` on the original LeRobot v2.1 directory, and
* ``gr00t_lancedb.LanceEpisodeLoader`` on the converted Lance dataset,

then turns both into ``VLAStepData`` with GR00T's own ``extract_step_data`` and compares
every state / action array, every decoded video frame, the language string, episode
lengths, and dataset statistics. Must be run inside an Isaac-GR00T environment.

    python scripts/parity_vs_gr00t.py --v2-dir /data/datasets/libero_10 \
        --lance-dir /data/datasets/libero_10_lance --embodiment-tag libero_sim
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np


def _deep_compare(a: Any, b: Any, path: str = "") -> list[str]:
    errs: list[str] = []
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            errs.append(f"{path}: key mismatch {sorted(set(a) ^ set(b))}")
        for k in set(a) & set(b):
            errs += _deep_compare(a[k], b[k], f"{path}.{k}")
        return errs
    try:
        aa, bb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    except (TypeError, ValueError):
        return [] if a == b else [f"{path}: {a!r} != {b!r}"]
    if aa.shape != bb.shape:
        return [f"{path}: shape {aa.shape} != {bb.shape}"]
    if not np.allclose(aa, bb, rtol=0, atol=1e-6, equal_nan=True):
        return [f"{path}: max abs diff {np.nanmax(np.abs(aa - bb)):.3g}"]
    return []


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--v2-dir", required=True)
    p.add_argument("--lance-dir", required=True)
    p.add_argument("--embodiment-tag", required=True)
    p.add_argument("--episodes", type=int, default=40, help="Random episodes to check")
    p.add_argument("--steps-per-episode", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-json", default=None)
    args = p.parse_args()

    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag

    from gr00t_lancedb import LanceEpisodeLoader

    emb = EmbodimentTag(args.embodiment_tag)
    cfg = MODALITY_CONFIGS[emb.value]

    t0 = time.perf_counter()
    native = LeRobotEpisodeLoader(args.v2_dir, cfg)
    t1 = time.perf_counter()
    lance = LanceEpisodeLoader(args.lance_dir, cfg)
    t2 = time.perf_counter()

    report: dict[str, Any] = {
        "v2_dir": args.v2_dir,
        "lance_dir": args.lance_dir,
        "embodiment": emb.value,
        "init_s": {"native": round(t1 - t0, 3), "lance": round(t2 - t1, 3)},
        "num_episodes": {"native": len(native), "lance": len(lance)},
    }
    errors: list[str] = []

    nl, ll = native.get_episode_lengths(), lance.get_episode_lengths()
    if list(nl) != list(ll):
        errors.append("episode lengths differ")
    errors += [f"stats{e}" for e in _deep_compare(native.get_dataset_statistics(), lance.get_dataset_statistics())]

    rng = np.random.default_rng(args.seed)
    episodes = rng.choice(len(native), size=min(args.episodes, len(native)), replace=False)
    n_steps = n_frames = 0
    max_state = max_action = 0.0
    max_pix = 0
    # Same anchor domain as GR00T's ShardedSingleStepDataset (no padding): every delta
    # index of every modality must stay inside the episode.
    max_delta = max(max(c.delta_indices) for c in cfg.values())
    for ep in episodes:
        length = int(nl[ep]) - max_delta
        if length <= 0:
            continue
        steps = np.sort(rng.choice(length, size=min(args.steps_per_episode, length), replace=False))
        vdeltas = cfg["video"].delta_indices if "video" in cfg else [0]

        def select(n: int, steps=steps) -> np.ndarray:
            idx = {int(s) + int(d) for s in steps for d in vdeltas}
            return np.array(sorted(i for i in idx if 0 <= i < n), dtype=np.int64)

        df_n = native.load_episode(int(ep), select_frames=select)
        df_l = lance.load_episode(int(ep), select_frames=select)
        for s in steps:
            a = extract_step_data(df_n, int(s), cfg, emb, allow_padding=False)
            b = extract_step_data(df_l, int(s), cfg, emb, allow_padding=False)
            tag = f"ep{int(ep)}/step{int(s)}"
            if a.text != b.text:
                errors.append(f"{tag}: text {a.text!r} != {b.text!r}")
            for kind, da, db in (("state", a.states, b.states), ("action", a.actions, b.actions)):
                if set(da) != set(db):
                    errors.append(f"{tag}: {kind} keys differ")
                    continue
                for k in da:
                    diff = float(np.max(np.abs(np.asarray(da[k]) - np.asarray(db[k]))))
                    if kind == "state":
                        max_state = max(max_state, diff)
                    else:
                        max_action = max(max_action, diff)
                    if diff > 1e-6:
                        errors.append(f"{tag}: {kind}.{k} max diff {diff}")
            for k in a.images:
                for fa, fb in zip(a.images[k], b.images[k], strict=True):
                    fa, fb = np.asarray(fa), np.asarray(fb)
                    if fa.shape != fb.shape or fa.dtype != fb.dtype:
                        errors.append(f"{tag}: video.{k} {fa.shape}/{fa.dtype} vs {fb.shape}/{fb.dtype}")
                        continue
                    d = int(np.max(np.abs(fa.astype(np.int16) - fb.astype(np.int16))))
                    max_pix = max(max_pix, d)
                    if d:
                        errors.append(f"{tag}: video.{k} max pixel diff {d}")
                    n_frames += 1
            n_steps += 1

    report.update(
        {
            "checked_episodes": int(len(episodes)),
            "checked_steps": n_steps,
            "checked_frames": n_frames,
            "max_state_abs_err": max_state,
            "max_action_abs_err": max_action,
            "max_pixel_diff": max_pix,
            "num_errors": len(errors),
            "errors_head": errors[:20],
            "pass": not errors,
        }
    )
    lance.close() if hasattr(lance, "close") else None
    out = json.dumps(report, indent=2)
    print(out)
    if args.out_json:
        with open(args.out_json, "w") as f:
            f.write(out)
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
