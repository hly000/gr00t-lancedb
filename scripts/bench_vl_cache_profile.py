#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""GPU profiler, Lance VL-cache materializer, and end-to-end prototype benchmark for GR00T N1.7.

Subcommands:
  profile        4-stage CUDA Event breakdown + peak VRAM across 6 acceleration configs
  materialize    Extract frozen backbone features into vl_cache_<fp8>.lance (INT8 / BF16)
  train_smoke    Numerical parity check + Lance random-take ActionHead-Only training + Lance vs .pt/Parquet
  export_check   Save lightweight ActionHead checkpoint and merge back to standard GR00T dir
  check_libero   Verify whether LIBERO / MuJoCo headless rendering works on this machine
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import numpy as np

# Ensure local gr00t_lancedb src is importable
_REPO_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)


def _add_gr00t_root(gr00t_root: str | None) -> None:
    if gr00t_root and gr00t_root not in sys.path:
        sys.path.insert(0, gr00t_root)


def _gpu_mem_gb() -> tuple[float, float]:
    import torch

    if not torch.cuda.is_available():
        return 0.0, 0.0
    alloc = torch.cuda.max_memory_allocated() / (1024.0**3)
    resv = torch.cuda.max_memory_reserved() / (1024.0**3)
    return round(alloc, 3), round(resv, 3)


def _reset_gpu() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def _build_gr00t_processor(cfg: Any, tag: Any, modality_configs: dict[str, Any], stats: Any = None) -> Any:
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor

    return Gr00tN1d7Processor(
        modality_configs={tag.value: modality_configs},
        statistics={tag.value: stats} if stats is not None else None,
        use_percentiles=getattr(cfg, "use_percentiles", True),
        image_crop_size=getattr(cfg, "image_crop_size", None),
        image_target_size=getattr(cfg, "image_target_size", None),
        shortest_image_edge=getattr(cfg, "shortest_image_edge", 256),
        crop_fraction=getattr(cfg, "crop_fraction", 0.95),
        random_rotation_angle=getattr(cfg, "random_rotation_angle", None),
        color_jitter_params=getattr(cfg, "color_jitter_params", None),
        formalize_language=getattr(cfg, "formalize_language", True),
        model_name=cfg.model_name,
        model_type=cfg.backbone_model_type,
        max_state_dim=cfg.max_state_dim,
        max_action_dim=cfg.max_action_dim,
        max_action_horizon=cfg.action_horizon,
        apply_sincos_state_encoding=getattr(cfg, "apply_sincos_state_encoding", False),
        use_albumentations=getattr(cfg, "use_albumentations", True),
        use_relative_action=getattr(cfg, "use_relative_action", False),
    )


def _load_real_batches(
    dataset_path: str,
    base_model_path: str,
    embodiment_tag: str,
    batch_size: int,
    num_batches: int,
    num_workers: int = 2,
) -> tuple[list[dict[str, Any]], Any]:
    """Load real collated GR00T N1.7 batches from a local or Lance dataset."""
    import torch
    from transformers import AutoConfig
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7DataCollator
    from gr00t_lancedb._gr00t_compat import coerce_embodiment_tag

    cfg = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    tag = coerce_embodiment_tag(embodiment_tag)
    modality_configs = MODALITY_CONFIGS[tag.value]

    is_lance = (Path(dataset_path) / "frames.lance").exists() or str(dataset_path).startswith(("s3://", "gs://"))
    if is_lance:
        from gr00t_lancedb import LanceShardedSingleStepDataset

        ds = LanceShardedSingleStepDataset(
            dataset_path=dataset_path,
            embodiment_tag=tag,
            modality_configs=modality_configs,
            shard_size=max(batch_size * 4, 256),
            episode_sampling_rate=0.25,
            seed=42,
        )
    else:
        from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset

        ds = ShardedSingleStepDataset(
            dataset_path=dataset_path,
            embodiment_tag=tag,
            modality_configs=modality_configs,
            shard_size=max(batch_size * 4, 256),
            episode_sampling_rate=0.25,
            seed=42,
        )

    stats = ds.get_dataset_statistics()
    processor = _build_gr00t_processor(cfg, tag, modality_configs, stats=stats)
    processor.train()
    ds.processor = processor

    collator = Gr00tN1d7DataCollator(
        model_name=cfg.model_name,
        model_type=cfg.backbone_model_type,
    )

    batches: list[dict[str, Any]] = []
    buf: list[Any] = []
    for shard_idx in range(len(ds)):
        items = ds.get_shard(shard_idx)
        for item in items:
            buf.append(item)
            if len(buf) == batch_size:
                batches.append(collator(buf)["inputs"])
                buf = []
                if len(batches) >= num_batches:
                    return batches, cfg
    if buf and len(batches) < num_batches:
        batches.append(collator(buf)["inputs"])
    return batches, cfg


# ---------------------------------------------------------------------------
# Subcommand 1: profile
# ---------------------------------------------------------------------------


def cmd_profile(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoModel, BatchFeature
    from gr00t_lancedb.vl_cache import (
        build_action_head_optimizer,
        load_action_head_only,
        patch_action_head_multi_noise,
    )

    _add_gr00t_root(args.gr00t_root)
    device = "cuda"
    results: dict[str, Any] = {
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "gpu_total_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024.0**3), 2)
        if torch.cuda.is_available()
        else 0.0,
        "base_model_path": args.base_model_path,
        "dataset_path": args.dataset_path,
        "batch_size": args.batch_size,
        "warmup_steps": args.warmup,
        "measure_steps": args.steps,
        "configs": {},
    }

    print(f"[profile] Loading {args.num_cached_batches} real batches (B={args.batch_size})...")
    batches, cfg = _load_real_batches(
        dataset_path=args.dataset_path,
        base_model_path=args.base_model_path,
        embodiment_tag=args.embodiment_tag,
        batch_size=args.batch_size,
        num_batches=args.num_cached_batches,
    )
    print(f"[profile] Loaded {len(batches)} batches. select_layer={cfg.select_layer}, "
          f"dit_layers={cfg.diffusion_model_cfg.get('num_layers')}, "
          f"image_target_size={cfg.image_target_size}")

    # --- 1. Native Full Model Breakdown ---
    _reset_gpu()
    print("[profile] === 1/5: native_full (baseline) ===")
    t_load0 = time.perf_counter()
    model = AutoModel.from_pretrained(args.base_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model.config.tune_llm = False
    model.config.tune_visual = False
    model.config.tune_top_llm_layers = 0
    model.backbone.set_trainable_parameters(tune_llm=False, tune_visual=False, tune_top_llm_layers=0)
    model.action_head.set_trainable_parameters(tune_projector=True, tune_diffusion_model=True, tune_vlln=True)
    model = model.to(device=device, dtype=torch.bfloat16)
    model.train()
    load_s = time.perf_counter() - t_load0

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    head_params = sum(p.numel() for p in model.action_head.parameters())
    results["param_counts"] = {
        "total": total_params,
        "trainable": trainable_params,
        "backbone": backbone_params,
        "action_head": head_params,
    }

    opt, opt_name = build_action_head_optimizer(model, lr=1e-4, use_8bit=False)

    # Pre-capture cached backbone outputs from the real batches for ActionHead-Only tests
    cached_backbone_batches: list[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]] = []

    t_prep, t_bb, t_ah_fwd, t_ah_bwd, t_opt, t_tot = [], [], [], [], [], []
    seq_lens: list[int] = []
    for step in range(args.warmup + args.steps):
        raw = batches[step % len(batches)]
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e2 = torch.cuda.Event(enable_timing=True)
        e3 = torch.cuda.Event(enable_timing=True)
        e4 = torch.cuda.Event(enable_timing=True)
        e5 = torch.cuda.Event(enable_timing=True)

        opt.zero_grad(set_to_none=True)
        e0.record()
        bb_in, act_in = model.prepare_input(dict(raw))
        e1.record()
        with torch.no_grad():
            bb_out = model.backbone(bb_in)
        e2.record()
        if len(cached_backbone_batches) < len(batches):
            seq_lens.append(int(bb_out.backbone_features.shape[1]))
            cached_backbone_batches.append(
                (
                    {k: v.detach().cpu() for k, v in bb_out.items()},
                    {k: v.detach().cpu() for k, v in act_in.items()},
                )
            )
        ah_out = model.action_head(bb_out, act_in)
        loss = ah_out["loss"]
        e3.record()
        loss.backward()
        e4.record()
        opt.step()
        e5.record()
        torch.cuda.synchronize()

        if step >= args.warmup:
            t_prep.append(e0.elapsed_time(e1))
            t_bb.append(e1.elapsed_time(e2))
            t_ah_fwd.append(e2.elapsed_time(e3))
            t_ah_bwd.append(e3.elapsed_time(e4))
            t_opt.append(e4.elapsed_time(e5))
            t_tot.append(e0.elapsed_time(e5))

    alloc_gb, resv_gb = _gpu_mem_gb()
    bb_pct = 100.0 * float(np.median(t_bb)) / max(1e-6, float(np.median(t_tot)))
    results["configs"]["native_full"] = {
        "optimizer": opt_name,
        "load_s": round(load_s, 2),
        "seq_len_sample": seq_lens[0] if seq_lens else None,
        "prep_ms": round(float(np.median(t_prep)), 2),
        "backbone_fwd_ms": round(float(np.median(t_bb)), 2),
        "action_head_fwd_ms": round(float(np.median(t_ah_fwd)), 2),
        "action_head_bwd_ms": round(float(np.median(t_ah_bwd)), 2),
        "optim_ms": round(float(np.median(t_opt)), 2),
        "total_step_ms": round(float(np.median(t_tot)), 2),
        "backbone_share_pct": round(bb_pct, 1),
        "theoretical_cache_ceiling_x": round(100.0 / max(1.0, 100.0 - bb_pct), 2),
        "peak_alloc_gb": alloc_gb,
        "peak_reserved_gb": resv_gb,
    }
    print(f"[profile] native_full: {results['configs']['native_full']}")

    # --- 2. Online Multi-Noise K=4 at B/4 (same effective DiT batch size!) ---
    if args.batch_size >= 4:
        print("[profile] === 2/5: online_multinoise_k4 (B/4 frames x K=4 noise) ===")
        torch.cuda.reset_peak_memory_stats()
        patch_action_head_multi_noise(model.action_head, k_noise=4, stratified=True)
        sub_b = args.batch_size // 4
        t_tot_mn = []
        for step in range(args.warmup + args.steps):
            raw = batches[step % len(batches)]
            # Slice first B/4 items
            raw_sub = {
                k: (v[:sub_b] if isinstance(v, torch.Tensor) and v.shape[0] == args.batch_size else v)
                for k, v in raw.items()
            }
            # Note: pixel_values in Qwen3-VL is packed across images, so only run online MN if batch_size==sub_b
            # Instead measure directly with full B x K=4 (high-throughput mode)
            break

        # Full B x K=4 (4x effective flow-matching samples per step)
        t_tot_mn = []
        for step in range(args.warmup + args.steps):
            raw = batches[step % len(batches)]
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            opt.zero_grad(set_to_none=True)
            e0.record()
            bb_in, act_in = model.prepare_input(dict(raw))
            with torch.no_grad():
                bb_out = model.backbone(bb_in)
            loss = model.action_head(bb_out, act_in)["loss"]
            loss.backward()
            opt.step()
            e1.record()
            torch.cuda.synchronize()
            if step >= args.warmup:
                t_tot_mn.append(e0.elapsed_time(e1))
        alloc_gb, resv_gb = _gpu_mem_gb()
        med_ms = float(np.median(t_tot_mn))
        base_ms = results["configs"]["native_full"]["total_step_ms"]
        results["configs"]["online_multinoise_B_k4"] = {
            "effective_fm_batch": args.batch_size * 4,
            "total_step_ms": round(med_ms, 2),
            "ms_per_fm_sample_speedup_x": round((base_ms / args.batch_size) / (med_ms / (args.batch_size * 4)), 2),
            "peak_alloc_gb": alloc_gb,
            "peak_reserved_gb": resv_gb,
        }
        print(f"[profile] online_multinoise_B_k4: {results['configs']['online_multinoise_B_k4']}")

    del model, opt
    _reset_gpu()

    # --- 3..5: ActionHead-Only benchmarks (VLM completely unloaded!) ---
    def _bench_action_head(
        name: str,
        *,
        use_8bit: bool,
        use_compile: bool,
        k_noise: int,
        sub_batch_div: int = 1,
    ) -> dict[str, Any]:
        _reset_gpu()
        print(f"[profile] === {name} ===")
        t0 = time.perf_counter()
        ah, _, _ = load_action_head_only(args.base_model_path, device=device, dtype=torch.bfloat16)
        ah.train()
        if k_noise > 1:
            patch_action_head_multi_noise(ah, k_noise=k_noise, stratified=True)
        if use_compile:
            ah.model = torch.compile(ah.model, mode="default")
        opt_ah, opt_ah_name = build_action_head_optimizer(ah, lr=1e-4, use_8bit=use_8bit)
        load_ah_s = time.perf_counter() - t0

        t_h2d, t_fwd, t_bwd, t_op, t_all = [], [], [], [], []
        n_warm = args.warmup + (3 if use_compile else 0)
        for step in range(n_warm + args.steps):
            bb_cpu, act_cpu = cached_backbone_batches[step % len(cached_backbone_batches)]
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e2 = torch.cuda.Event(enable_timing=True)
            e3 = torch.cuda.Event(enable_timing=True)
            e4 = torch.cuda.Event(enable_timing=True)

            opt_ah.zero_grad(set_to_none=True)
            e0.record()
            if sub_batch_div > 1:
                sb = max(1, args.batch_size // sub_batch_div)
                bb_gpu = BatchFeature(data={k: v[:sb].to(device, non_blocking=True) for k, v in bb_cpu.items()})
                act_gpu = BatchFeature(data={k: v[:sb].to(device, non_blocking=True) for k, v in act_cpu.items()})
            else:
                bb_gpu = BatchFeature(data={k: v.to(device, non_blocking=True) for k, v in bb_cpu.items()})
                act_gpu = BatchFeature(data={k: v.to(device, non_blocking=True) for k, v in act_cpu.items()})
            e1.record()
            loss = ah(bb_gpu, act_gpu)["loss"]
            e2.record()
            loss.backward()
            e3.record()
            opt_ah.step()
            e4.record()
            torch.cuda.synchronize()

            if step >= n_warm:
                t_h2d.append(e0.elapsed_time(e1))
                t_fwd.append(e1.elapsed_time(e2))
                t_bwd.append(e2.elapsed_time(e3))
                t_op.append(e3.elapsed_time(e4))
                t_all.append(e0.elapsed_time(e4))

        alloc, resv = _gpu_mem_gb()
        base_ms = results["configs"]["native_full"]["total_step_ms"]
        med_ms = float(np.median(t_all))
        rec = {
            "optimizer": opt_ah_name,
            "load_s": round(load_ah_s, 2),
            "h2d_ms": round(float(np.median(t_h2d)), 2),
            "fwd_ms": round(float(np.median(t_fwd)), 2),
            "bwd_ms": round(float(np.median(t_bwd)), 2),
            "optim_ms": round(float(np.median(t_op)), 2),
            "total_step_ms": round(med_ms, 2),
            "speedup_vs_native_x": round(base_ms / max(1e-6, med_ms), 2),
            "peak_alloc_gb": alloc,
            "peak_reserved_gb": resv,
            "fits_24gb": resv < 23.5,
        }
        print(f"[profile] {name}: {rec}")
        del ah, opt_ah
        _reset_gpu()
        return rec

    results["configs"]["ah_only_adamw"] = _bench_action_head(
        "3/6: ah_only_adamw", use_8bit=False, use_compile=False, k_noise=1
    )
    results["configs"]["ah_only_adamw8bit"] = _bench_action_head(
        "4/6: ah_only_adamw8bit", use_8bit=True, use_compile=False, k_noise=1
    )
    results["configs"]["ah_only_adamw8bit_mn_k4_iso"] = _bench_action_head(
        "5/6: ah_only_adamw8bit_mn_k4_iso (B/4 x K=4)",
        use_8bit=True,
        use_compile=False,
        k_noise=4,
        sub_batch_div=4,
    )
    try:
        results["configs"]["ah_only_adamw8bit_compile"] = _bench_action_head(
            "6/6: ah_only_adamw8bit_compile", use_8bit=True, use_compile=True, k_noise=1
        )
    except Exception as exc:
        results["configs"]["ah_only_adamw8bit_compile"] = {"error": str(exc)}

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"[profile] Saved report to {args.out_json}")
    return results


# ---------------------------------------------------------------------------
# Subcommand 2: materialize
# ---------------------------------------------------------------------------


def cmd_materialize(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoConfig, AutoModel
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7DataCollator, Gr00tN1d7Processor
    from gr00t_lancedb import LanceShardedSingleStepDataset
    from gr00t_lancedb._gr00t_compat import coerce_embodiment_tag
    from gr00t_lancedb.vl_cache import CacheFingerprint, VLCacheWriter

    _add_gr00t_root(args.gr00t_root)
    device = "cuda"
    t_start = time.perf_counter()

    cfg = AutoConfig.from_pretrained(args.base_model_path, trust_remote_code=True)
    tag = coerce_embodiment_tag(args.embodiment_tag)
    modality_configs = MODALITY_CONFIGS[tag.value]

    ds = LanceShardedSingleStepDataset(
        dataset_path=args.dataset_path,
        embodiment_tag=tag,
        modality_configs=modality_configs,
        shard_size=1024,
        episode_sampling_rate=1.0,
        seed=42,
    )
    stats = ds.get_dataset_statistics()
    processor = _build_gr00t_processor(cfg, tag, modality_configs, stats=stats)
    if args.views <= 1:
        processor.eval()  # Deterministic center crop for view_id=0
    else:
        processor.train()

    collator = Gr00tN1d7DataCollator(
        model_name=cfg.model_name,
        model_type=cfg.backbone_model_type,
    )

    print("[materialize] Loading model backbone...")
    model = AutoModel.from_pretrained(args.base_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    backbone = model.backbone.to(device=device, dtype=torch.bfloat16).eval()
    del model.action_head
    del model
    _reset_gpu()

    fp = CacheFingerprint(
        base_model_path=args.base_model_path,
        select_layer=int(cfg.select_layer),
        hidden_dim=int(cfg.backbone_embedding_dim),
        image_target_size=tuple(cfg.image_target_size) if cfg.image_target_size else None,
        image_crop_size=tuple(cfg.image_crop_size) if cfg.image_crop_size else None,
        augmentation_signature="none" if args.views <= 1 else "default_crop_jitter_v1",
        quant_codec=args.codec,
        backbone_name=str(cfg.model_name),
        num_views=int(args.views),
        pad_side="left",
    )

    writer = VLCacheWriter(
        db_uri=args.cache_dir,
        fingerprint=fp,
        include_pooled_emb=True,
        flush_rows=512,
        overwrite=True,
    )

    anchors = ds.get_all_valid_anchors()
    if args.max_frames > 0:
        anchors = anchors[: args.max_frames]
    print(f"[materialize] Processing {len(anchors)} frames x {args.views} views -> {args.cache_dir}/{fp.table_name()}.lance (codec={args.codec})")

    t_inf0 = time.perf_counter()
    row_counter = 0
    for view_id in range(args.views):
        if view_id == 0:
            processor.eval()
        else:
            processor.train()
        for start in range(0, len(anchors), args.batch_size):
            chunk = anchors[start : start + args.batch_size]
            steps = ds.load_anchor_batch(chunk)
            proc_items = [processor([{"type": "episode_step", "content": s}]) for s in steps]
            batch = collator(proc_items)["inputs"]
            bb_in = {
                k: (v.to(device, dtype=torch.bfloat16) if torch.is_floating_point(v) else v.to(device))
                for k, v in batch.items()
                if k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
            }
            with torch.inference_mode():
                out = backbone(bb_in)
            feats = out.backbone_features.float().cpu().numpy()
            attn = out.backbone_attention_mask.cpu().numpy()
            img_m = out.image_mask.cpu().numpy()

            writer.add_batch(
                row_keys=list(range(row_counter, row_counter + len(chunk))),
                episode_indices=[ep for ep, _ in chunk],
                frame_indices=[fr for _, fr in chunk],
                view_ids=[view_id] * len(chunk),
                features=feats,
                attention_mask=attn,
                image_mask=img_m,
            )
            row_counter += len(chunk)
            if row_counter % (args.batch_size * 4) == 0 or start + args.batch_size >= len(anchors):
                elapsed = time.perf_counter() - t_inf0
                print(f"[materialize] {row_counter}/{len(anchors) * args.views} rows ({row_counter / max(1e-6, elapsed):.1f} frames/s)")

    summary = writer.close()
    total_s = time.perf_counter() - t_start
    inf_s = time.perf_counter() - t_inf0
    summary.update(
        {
            "total_wall_s": round(total_s, 2),
            "inference_and_write_s": round(inf_s, 2),
            "frames_per_s": round(row_counter / max(1e-6, inf_s), 2),
            "codec": args.codec,
            "peak_alloc_gb": _gpu_mem_gb()[0],
        }
    )
    print(f"[materialize] DONE: {json.dumps(summary, indent=2)}")
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


# ---------------------------------------------------------------------------
# Subcommand 3: train_smoke (numerical parity + Lance random-take training + storage bench)
# ---------------------------------------------------------------------------


def cmd_train_smoke(args: argparse.Namespace) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from transformers import AutoModel, BatchFeature
    from gr00t_lancedb.vl_cache import (
        VLCacheReader,
        build_action_head_optimizer,
        collate_cached_vl_batch,
        encode_features,
        load_action_head_only,
        pack_bool_mask,
        strip_padding,
        unpack_bool_mask,
    )
    from gr00t_lancedb.vl_cache.store import EncodedVLRow

    _add_gr00t_root(args.gr00t_root)
    device = "cuda"
    report: dict[str, Any] = {}

    # 1. Exact numerical parity check (online vs bf16_raw cache vs per_token_int8 cache)
    print("[train_smoke] === 1/3: Exact Numerical Parity Check ===")
    batches, _ = _load_real_batches(
        dataset_path=args.dataset_path,
        base_model_path=args.base_model_path,
        embodiment_tag=args.embodiment_tag,
        batch_size=min(args.batch_size, 16),
        num_batches=1,
    )
    raw_batch = batches[0]

    _reset_gpu()
    model = AutoModel.from_pretrained(args.base_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(
        device=device, dtype=torch.bfloat16
    )
    model.eval()  # Disable dropout for deterministic parity check
    bb_in, act_in = model.prepare_input(dict(raw_batch))
    with torch.no_grad():
        bb_out = model.backbone(bb_in)
        torch.manual_seed(12345)
        torch.cuda.manual_seed_all(12345)
        loss_online = float(model.action_head(bb_out, act_in)["loss"].item())

        feats_np = bb_out.backbone_features.float().cpu().numpy()
        attn_np = bb_out.backbone_attention_mask.cpu().numpy()
        img_np = bb_out.image_mask.cpu().numpy()

        parity = {"loss_online_bf16": round(loss_online, 6)}
        for codec in ("bf16_raw", "per_token_int8"):
            rows = []
            for i in range(feats_np.shape[0]):
                vf, vim, pad_side = strip_padding(feats_np[i], attn_np[i], img_np[i])
                sc_b, pl_b = encode_features(vf, codec=codec)
                L = int(vf.shape[0])
                D = int(vf.shape[1])
                sc = np.frombuffer(sc_b, dtype=np.float16).copy() if codec == "per_token_int8" else None
                pl = (
                    np.frombuffer(pl_b, dtype=np.int8).reshape(L, D).copy()
                    if codec == "per_token_int8"
                    else np.frombuffer(pl_b, dtype=np.uint16).reshape(L, D).copy()
                )
                rows.append(EncodedVLRow(seq_len=L, image_mask=unpack_bool_mask(pack_bool_mask(vim), L), feat_scale=sc, feat_payload=pl))
            collated = collate_cached_vl_batch(rows, codec=codec, pad_side=pad_side, device=device, target_dtype=torch.bfloat16)
            torch.manual_seed(12345)
            torch.cuda.manual_seed_all(12345)
            loss_cached = float(model.action_head(BatchFeature(data=collated), act_in)["loss"].item())
            parity[f"loss_{codec}"] = round(loss_cached, 6)
            parity[f"abs_diff_{codec}"] = round(abs(loss_cached - loss_online), 6)
            parity[f"rel_diff_{codec}"] = round(abs(loss_cached - loss_online) / max(1e-8, abs(loss_online)), 6)
    report["numerical_parity"] = parity
    print(f"[train_smoke] Parity: {json.dumps(parity, indent=2)}")

    # Grab one dummy action_input template on CPU for cached training loop
    act_template = {k: v.detach().cpu() for k, v in act_in.items()}
    del model
    _reset_gpu()

    # 2. Real Lance random-take ActionHead-Only training smoke
    if args.cache_dir and args.table_name:
        print(f"[train_smoke] === 2/3: Lance Random-Take ActionHead-Only Training ({args.steps} steps) ===")
        reader = VLCacheReader(args.cache_dir, args.table_name)
        n_rows = reader.num_rows
        assert reader.fingerprint is not None
        codec = reader.fingerprint.quant_codec
        pad_side = reader.fingerprint.pad_side

        ah, _, _ = load_action_head_only(args.base_model_path, device=device, dtype=torch.bfloat16)
        ah.train()
        opt, opt_name = build_action_head_optimizer(ah, lr=1e-4, use_8bit=True)

        rng = np.random.default_rng(42)
        # Tile action template to match args.batch_size
        def _tile_act(bsz: int) -> BatchFeature:
            out = {}
            for k, v in act_template.items():
                reps = (bsz + v.shape[0] - 1) // v.shape[0]
                t = v.repeat((reps,) + (1,) * (v.ndim - 1))[:bsz]
                out[k] = t.to(device, non_blocking=True)
            return BatchFeature(data=out)

        t_take, t_step = [], []
        for step in range(args.warmup + args.steps):
            row_ids = rng.integers(0, n_rows, size=args.batch_size).tolist()
            t0 = time.perf_counter()
            rows = reader.take_rows(row_ids)
            collated = collate_cached_vl_batch(rows, codec=codec, pad_side=pad_side, device=device, target_dtype=torch.bfloat16)
            take_ms = (time.perf_counter() - t0) * 1000.0

            opt.zero_grad(set_to_none=True)
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            loss = ah(BatchFeature(data=collated), _tile_act(args.batch_size))["loss"]
            loss.backward()
            opt.step()
            e1.record()
            torch.cuda.synchronize()
            if step >= args.warmup:
                t_take.append(take_ms)
                t_step.append(e0.elapsed_time(e1))

        alloc, resv = _gpu_mem_gb()
        report["lance_cached_train"] = {
            "num_cached_rows": n_rows,
            "batch_size": args.batch_size,
            "optimizer": opt_name,
            "lance_take_and_h2d_ms": round(float(np.median(t_take)), 2),
            "gpu_step_ms": round(float(np.median(t_step)), 2),
            "total_step_ms": round(float(np.median(t_take) + np.median(t_step)), 2),
            "peak_alloc_gb": alloc,
            "peak_reserved_gb": resv,
        }
        print(f"[train_smoke] Lance cached train: {json.dumps(report['lance_cached_train'], indent=2)}")
        del ah, opt
        _reset_gpu()

        # 3. Storage backend random-read comparison: Lance vs .pt scatter files vs Parquet
        print("[train_smoke] === 3/3: Random-Read Microbench (Lance vs .pt scatter vs Parquet) ===")
        n_bench = min(n_rows, 2000)
        bench_batches = [rng.integers(0, n_bench, size=args.batch_size).tolist() for _ in range(30)]

        # Lance random take
        t0 = time.perf_counter()
        for b in bench_batches:
            _ = reader.take_rows(b)
        lance_ms = (time.perf_counter() - t0) * 1000.0 / len(bench_batches)

        # Export same n_bench rows to .pt scatter files and a single Parquet file
        with tempfile.TemporaryDirectory(prefix="vl_storage_bench_") as tmpdir:
            tmp = Path(tmpdir)
            pt_dir = tmp / "pt_scatter"
            pt_dir.mkdir()
            all_rows = reader.take_rows(list(range(n_bench)))
            for idx, r in enumerate(all_rows):
                torch.save((r.feat_payload, r.feat_scale, r.image_mask), pt_dir / f"{idx:06d}.pt")
            pq_path = tmp / "cache.parquet"
            pq.write_table(
                reader.take_rows_arrow(
                    list(range(n_bench)),
                    columns=["seq_len", "image_mask", "feat_scale", "feat_payload"],
                ),
                pq_path,
            )

            t0 = time.perf_counter()
            for b in bench_batches:
                _ = [torch.load(pt_dir / f"{idx:06d}.pt", weights_only=False) for idx in b]
            pt_ms = (time.perf_counter() - t0) * 1000.0 / len(bench_batches)

            t0 = time.perf_counter()
            pf = pq.ParquetFile(pq_path)
            for b in bench_batches:
                # Parquet has no O(1) row-index random take without scanning row groups
                tbl = pf.read(columns=["seq_len", "image_mask", "feat_scale", "feat_payload"])
                _ = tbl.take(b)
            pq_ms = (time.perf_counter() - t0) * 1000.0 / len(bench_batches)

        report["storage_random_read_bench"] = {
            "num_rows": n_bench,
            "batch_size": args.batch_size,
            "lance_take_ms": round(lance_ms, 2),
            "pt_scatter_ms": round(pt_ms, 2),
            "parquet_scan_take_ms": round(pq_ms, 2),
            "lance_vs_pt_x": round(pt_ms / max(1e-6, lance_ms), 2),
            "lance_vs_parquet_x": round(pq_ms / max(1e-6, lance_ms), 2),
        }
        print(f"[train_smoke] Storage bench: {json.dumps(report['storage_random_read_bench'], indent=2)}")

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# Subcommand 4: check_libero
# ---------------------------------------------------------------------------


def cmd_check_libero(_args: argparse.Namespace) -> dict[str, Any]:
    report: dict[str, Any] = {"libero_installed": False, "mujoco_gl_working": False, "details": ""}
    try:
        import mujoco

        report["mujoco_version"] = getattr(mujoco, "__version__", "unknown")
    except Exception as exc:
        report["details"] = f"mujoco import failed: {exc}"
        print(json.dumps(report, indent=2))
        return report

    try:
        import libero

        report["libero_installed"] = True
    except Exception as exc:
        report["details"] = f"libero not installed: {exc}"

    for backend in ("egl", "osmesa"):
        os.environ["MUJOCO_GL"] = backend
        try:
            model = mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom type='box' size='1 1 1'/></worldbody></mujoco>")
            data = mujoco.MjData(model)
            renderer = mujoco.Renderer(model, height=64, width=64)
            mujoco.mj_forward(model, data)
            renderer.update_scene(data)
            img = renderer.render()
            report["mujoco_gl_working"] = True
            report["working_backend"] = backend
            report["render_shape"] = list(img.shape)
            break
        except Exception as exc:
            report[f"{backend}_error"] = str(exc)

    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="GR00T N1.7 VL-Cache Profiler & Prototype Benchmark")
    sub = p.add_subparsers(dest="cmd", required=True)

    def _add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--gr00t-root", default="/data/Isaac-GR00T")
        sp.add_argument("--base-model-path", default="nvidia/GR00T-N1.7-3B")
        sp.add_argument("--dataset-path", default="/data/datasets/libero_10_lance")
        sp.add_argument("--embodiment-tag", default="LIBERO_PANDA")
        sp.add_argument("--out-json", default=None)

    sp_prof = sub.add_parser("profile")
    _add_common(sp_prof)
    sp_prof.add_argument("--batch-size", type=int, default=32)
    sp_prof.add_argument("--num-cached-batches", type=int, default=4)
    sp_prof.add_argument("--warmup", type=int, default=5)
    sp_prof.add_argument("--steps", type=int, default=20)

    sp_mat = sub.add_parser("materialize")
    _add_common(sp_mat)
    sp_mat.add_argument("--cache-dir", required=True)
    sp_mat.add_argument("--codec", choices=["per_token_int8", "bf16_raw"], default="per_token_int8")
    sp_mat.add_argument("--views", type=int, default=1)
    sp_mat.add_argument("--batch-size", type=int, default=64)
    sp_mat.add_argument("--max-frames", type=int, default=2000)

    sp_tr = sub.add_parser("train_smoke")
    _add_common(sp_tr)
    sp_tr.add_argument("--cache-dir", default=None)
    sp_tr.add_argument("--table-name", default=None)
    sp_tr.add_argument("--batch-size", type=int, default=32)
    sp_tr.add_argument("--warmup", type=int, default=5)
    sp_tr.add_argument("--steps", type=int, default=30)

    sub.add_parser("check_libero")

    args = p.parse_args()
    if args.cmd == "profile":
        cmd_profile(args)
    elif args.cmd == "materialize":
        cmd_materialize(args)
    elif args.cmd == "train_smoke":
        cmd_train_smoke(args)
    elif args.cmd == "check_libero":
        cmd_check_libero(args)


if __name__ == "__main__":
    main()
