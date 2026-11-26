#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Real-dataset conversion, bit-exact parity audit, and throughput benchmark for gr00t-lancedb.

Designed to validate real NVIDIA Isaac-GR00T datasets (e.g. ``nvidia/GR00T-N1.7-AppleToPlate``,
LIBERO, RoboCasa, DROID) against GR00T's exact ``LeRobotEpisodeLoader`` reference logic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
from torchcodec.decoders import VideoDecoder

from lerobot_lancedb import LanceDBDataset

from gr00t_lancedb import (
    EmbodimentTag,
    LanceEpisodeLoader,
    LanceShardedSingleStepDataset,
    ModalityConfig,
    ResumableLanceMixtureDataset,
    convert_v2_to_lance,
    load_dataloader_state,
    save_dataloader_state,
)
from gr00t_lancedb.bench import audit_sample_stream, measure_process_rss_mb


def _identity_collate(items: list) -> list:
    return items


def _dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def build_full_modality_configs(
    modality_meta: dict[str, Any],
    *,
    action_horizon: int = 16,
    video_deltas: list[int] | None = None,
    language_key: str = "task",
) -> dict[str, ModalityConfig]:
    """Construct ModalityConfigs covering every state, action, and video key in ``modality.json``."""
    cfg: dict[str, ModalityConfig] = {}
    if modality_meta.get("video"):
        cfg["video"] = ModalityConfig(
            delta_indices=video_deltas or [0],
            modality_keys=list(modality_meta["video"].keys()),
        )
    if modality_meta.get("state"):
        cfg["state"] = ModalityConfig(
            delta_indices=[0],
            modality_keys=list(modality_meta["state"].keys()),
        )
    if modality_meta.get("action"):
        cfg["action"] = ModalityConfig(
            delta_indices=list(range(action_horizon)),
            modality_keys=list(modality_meta["action"].keys()),
        )
    cfg["language"] = ModalityConfig(delta_indices=[0], modality_keys=[language_key])
    return cfg


def reference_load_v2_step(
    v2_root: Path,
    info_meta: dict[str, Any],
    modality_meta: dict[str, Any],
    episodes_meta: list[dict[str, Any]],
    tasks_map: dict[int, str],
    ep_idx: int,
    step_idx: int,
    modality_configs: dict[str, ModalityConfig],
    allow_padding: bool = True,
) -> dict[str, Any]:
    ep_meta = episodes_meta[ep_idx]
    ep_id = int(ep_meta["episode_index"])
    chunk_size = int(info_meta.get("chunks_size", 1000))
    chunk_idx = ep_id // chunk_size
    pq_rel = info_meta["data_path"].format(episode_chunk=chunk_idx, episode_index=ep_id)
    df = pd.read_parquet(v2_root / pq_rel)
    length = min(len(df), int(ep_meta["length"]))

    def _idxs(deltas: list[int]) -> list[int]:
        raw = [step_idx + d for d in deltas]
        if allow_padding:
            return [max(0, min(i, length - 1)) for i in raw]
        return [i + length if i < 0 else i for i in raw]

    states: dict[str, np.ndarray] = {}
    for k in modality_configs["state"].modality_keys:
        rows = _idxs(modality_configs["state"].delta_indices)
        gm = modality_meta["state"][k]
        orig = gm.get("original_key", "observation.state")
        s, e = int(gm["start"]), int(gm["end"])
        states[k] = np.vstack(
            [
                np.asarray(df[orig].iloc[r], dtype=np.float32).reshape(-1)[s:e]
                for r in rows
            ]
        )

    actions: dict[str, np.ndarray] = {}
    for k in modality_configs["action"].modality_keys:
        rows = _idxs(modality_configs["action"].delta_indices)
        gm = modality_meta["action"][k]
        orig = gm.get("original_key", "action")
        s, e = int(gm["start"]), int(gm["end"])
        actions[k] = np.vstack(
            [
                np.asarray(df[orig].iloc[r], dtype=np.float32).reshape(-1)[s:e]
                for r in rows
            ]
        )

    images: dict[str, list[np.ndarray]] = {}
    if "video" in modality_configs:
        rows = _idxs(modality_configs["video"].delta_indices)
        for k in modality_configs["video"].modality_keys:
            orig = modality_meta["video"][k].get("original_key", f"observation.images.{k}")
            vid_rel = info_meta["video_path"].format(
                episode_chunk=chunk_idx, video_key=orig, episode_index=ep_id
            )
            dec = VideoDecoder(str(v2_root / vid_rel), device="cpu", dimension_order="NHWC")
            frames = dec.get_frames_at(indices=rows).data.numpy()
            images[k] = [frames[i] for i in range(len(rows))]

    lk = modality_configs["language"].modality_keys[0]
    if lk == "task":
        text = ep_meta["tasks"][0]
    elif lk.startswith("annotation."):
        sub = lk.replace("annotation.", "", 1)
        orig = modality_meta["annotation"][sub].get("original_key", lk)
        text = tasks_map[int(df[orig].iloc[step_idx])]
    else:
        text = ""

    return {"states": states, "actions": actions, "images": images, "text": text}


def run_parity_check(
    v2_root: Path,
    lance_root: str | Path,
    modality_configs: dict[str, ModalityConfig],
    *,
    num_episodes_sample: int = 8,
    steps_per_episode: int = 4,
    storage_options: dict[str, str] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    meta_dir = v2_root / "meta"
    info_meta = json.loads((meta_dir / "info.json").read_text())
    modality_meta = json.loads((meta_dir / "modality.json").read_text())
    episodes_meta = sorted(
        [json.loads(l) for l in (meta_dir / "episodes.jsonl").read_text().splitlines() if l],
        key=lambda r: int(r["episode_index"]),
    )
    tasks_map = {
        int(r["task_index"]): r["task"]
        for r in (json.loads(l) for l in (meta_dir / "tasks.jsonl").read_text().splitlines() if l)
    }

    loader = LanceEpisodeLoader(lance_root, modality_configs, storage_options=storage_options)
    rng = np.random.default_rng(seed)
    chosen_eps = sorted(rng.choice(len(loader), size=min(num_episodes_sample, len(loader)), replace=False).tolist())

    max_state_abs_err = 0.0
    max_action_abs_err = 0.0
    max_pixel_diff = 0
    checked_steps = 0

    for ep_idx in chosen_eps:
        ep_len = loader.get_episode_length(ep_idx)
        cand = sorted(set([0, ep_len // 3, (2 * ep_len) // 3, max(0, ep_len - 17)]))[:steps_per_episode]
        lance_steps = loader.load_steps(ep_idx, cand, EmbodimentTag.NEW_EMBODIMENT, allow_padding=True)
        for step_idx, l_step in zip(cand, lance_steps, strict=True):
            ref = reference_load_v2_step(
                v2_root, info_meta, modality_meta, episodes_meta, tasks_map,
                ep_idx, step_idx, modality_configs, allow_padding=True,
            )
            assert l_step.text == ref["text"], f"Text mismatch at ep={ep_idx}, step={step_idx}"
            for k in ref["states"]:
                err = float(np.max(np.abs(l_step.states[k] - ref["states"][k])))
                max_state_abs_err = max(max_state_abs_err, err)
                assert err < 1e-5, f"State mismatch {k}: {err}"
            for k in ref["actions"]:
                err = float(np.max(np.abs(l_step.actions[k] - ref["actions"][k])))
                max_action_abs_err = max(max_action_abs_err, err)
                assert err < 1e-5, f"Action mismatch {k}: {err}"
            for k in ref["images"]:
                for img_l, img_r in zip(l_step.images[k], ref["images"][k], strict=True):
                    assert img_l.shape == img_r.shape and img_l.dtype == np.uint8
                    diff = int(np.max(np.abs(img_l.astype(np.int16) - img_r.astype(np.int16))))
                    max_pixel_diff = max(max_pixel_diff, diff)
                    assert diff == 0, f"Pixel mismatch {k} at ep={ep_idx} step={step_idx}: max_diff={diff}"
            checked_steps += 1

    stats = loader.get_dataset_statistics()
    loader.close()
    return {
        "checked_episodes": len(chosen_eps),
        "checked_steps": checked_steps,
        "state_groups_verified": len(modality_configs["state"].modality_keys),
        "action_groups_verified": len(modality_configs["action"].modality_keys),
        "max_state_abs_err": max_state_abs_err,
        "max_action_abs_err": max_action_abs_err,
        "max_pixel_diff": max_pixel_diff,
        "has_q01_q99_stats": all(
            "q01" in stats["state"][k] and "q99" in stats["state"][k]
            for k in modality_configs["state"].modality_keys
        ),
    }


def bench_loader(
    lance_root: str | Path,
    modality_configs: dict[str, ModalityConfig],
    *,
    batch_size: int = 8,
    num_workers: int = 2,
    warmup_steps: int = 3,
    measure_steps: int = 30,
    episode_pool_size: int = 16,
    prefetch_anchors: int = 16,
    storage_options: dict[str, str] | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    ds = LanceShardedSingleStepDataset(
        dataset_path=lance_root,
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        modality_configs=modality_configs,
        allow_padding=True,
        storage_options=storage_options,
    )
    mix = ResumableLanceMixtureDataset(
        datasets=[ds],
        weights=[1.0],
        batch_size=batch_size,
        episode_pool_size=episode_pool_size,
        prefetch_anchors=prefetch_anchors,
        infinite=True,
    )
    init_s = time.perf_counter() - t0
    t_first: float | None = None
    t_meas: float | None = None
    seen = 0
    peak_rss = measure_process_rss_mb()

    for idx, batch in enumerate(mix.loader(num_workers=num_workers, collate_fn=_identity_collate)):
        now = time.perf_counter()
        if t_first is None:
            t_first = now - t0
        if idx == warmup_steps:
            t_meas = now
            seen = 0
        if idx >= warmup_steps:
            seen += len(batch)
        rss = measure_process_rss_mb()
        if rss > peak_rss:
            peak_rss = rss
        if idx + 1 >= warmup_steps + measure_steps:
            break

    meas_s = max(1e-6, time.perf_counter() - (t_meas or t0))
    mix.close()
    return {
        "root": str(lance_root),
        "batch_size": batch_size,
        "num_workers": num_workers,
        "init_s": round(init_s, 3),
        "ttfs_s": round(t_first or 0.0, 3),
        "measured_samples": seen,
        "measured_s": round(meas_s, 3),
        "samples_per_sec": round(seen / meas_s, 2),
        "peak_rss_mb": round(peak_rss, 1),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--v2-dir", required=True)
    p.add_argument("--lance-out", required=True)
    p.add_argument("--video-mode", choices=("external", "copy"), default="external")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--out-json", default=None)
    args = p.parse_args()

    v2_dir = Path(args.v2_dir)
    lance_out = Path(args.lance_out)
    modality_meta = json.loads((v2_dir / "meta" / "modality.json").read_text())

    # 1. Convert v2.1 -> Lance
    t0 = time.perf_counter()
    if not lance_out.exists():
        convert_v2_to_lance(v2_dir, lance_out, video_mode=args.video_mode)
    convert_s = time.perf_counter() - t0

    # Verify unmodified lerobot-lancedb compatibility
    upstream = LanceDBDataset(root=lance_out, return_uint8=True)
    upstream_ok = (upstream.num_episodes, upstream.num_frames)

    # 2. Full parity check (both 'task' and 'annotation.*' if present)
    cfg_task = build_full_modality_configs(modality_meta, language_key="task")
    parity_task = run_parity_check(v2_dir, lance_out, cfg_task)

    parity_ann = None
    if modality_meta.get("annotation"):
        first_ann = next(iter(modality_meta["annotation"].keys()))
        cfg_ann = build_full_modality_configs(modality_meta, language_key=f"annotation.{first_ann}")
        parity_ann = run_parity_check(v2_dir, lance_out, cfg_ann, num_episodes_sample=4, steps_per_episode=2)

    # 3. Exact resume check on real dataset with num_workers=2
    def _mk():
        ds = LanceShardedSingleStepDataset(
            lance_out, EmbodimentTag.NEW_EMBODIMENT, cfg_task, allow_padding=True
        )
        return ResumableLanceMixtureDataset(
            datasets=[ds], weights=[1.0], batch_size=args.batch_size,
            episode_pool_size=8, prefetch_anchors=8, infinite=True,
        )

    m_ref = _mk()
    ref_ids = []
    for i, b in enumerate(m_ref.loader(num_workers=args.num_workers, collate_fn=_identity_collate)):
        ref_ids.extend((x.metadata["epoch"], x.metadata["global_pos"]) for x in b)
        if i + 1 >= 10:
            break
    m_ref.close()

    ckpt_tmp = lance_out.parent / "_resume_verify_ckpt"
    m1 = _mk()
    obs_ids = []
    for i, b in enumerate(m1.loader(num_workers=args.num_workers, collate_fn=_identity_collate)):
        obs_ids.extend((x.metadata["epoch"], x.metadata["global_pos"]) for x in b)
        if i + 1 >= 4:
            save_dataloader_state(m1, ckpt_tmp)
            break
    m1.close()

    m2 = _mk()
    load_dataloader_state(m2, ckpt_tmp)
    for i, b in enumerate(m2.loader(num_workers=args.num_workers, collate_fn=_identity_collate)):
        obs_ids.extend((x.metadata["epoch"], x.metadata["global_pos"]) for x in b)
        if i + 1 >= 6:
            break
    m2.close()
    resume_audit = audit_sample_stream(ref_ids, obs_ids)

    # 4. Throughput benchmark
    perf = bench_loader(
        lance_out, cfg_task,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        measure_steps=args.steps,
    )

    report = {
        "dataset": str(v2_dir),
        "video_mode": args.video_mode,
        "source_size_mb": round(_dir_size_mb(v2_dir), 1),
        "lance_size_mb": round(_dir_size_mb(lance_out), 1),
        "convert_s": round(convert_s, 2),
        "upstream_lerobot_lancedb": {"num_episodes": upstream_ok[0], "num_frames": upstream_ok[1]},
        "parity_task": parity_task,
        "parity_annotation": parity_ann,
        "resume_audit_2workers": resume_audit,
        "throughput": perf,
    }
    print(json.dumps(report, indent=2))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
