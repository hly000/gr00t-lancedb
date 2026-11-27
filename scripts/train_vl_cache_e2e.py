#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""End-to-end Lance VL-Cache materializer + ActionHead-Only trainer + lossless GR00T exporter.

Designed for 1-to-1 apples-to-apples comparison against official ``launch_finetune.py``:
- Same dataset, processor statistics, state_dropout_prob=0.2, lr=1e-4, cosine schedule,
  warmup_ratio=0.05, weight_decay=1e-5, global_batch_size, and max_steps.
- Training runs with VLM completely unloaded (ActionHead-Only), reading Per-Token INT8
  features randomly from Lance via ``take_offsets()`` in background DataLoader workers,
  plus optional ``torch.compile`` and ``FeatureSpaceAugmenter``.
- Exports a standard HuggingFace / Isaac-GR00T checkpoint directory (with merged backbone +
  trained action_head + processor_config.json + statistics.json) ready for
  ``gr00t/eval/run_gr00t_server.py``.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # collation now runs in forked workers

_REPO_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)


def _add_gr00t_root(gr00t_root: str) -> None:
    if gr00t_root and gr00t_root not in sys.path:
        sys.path.insert(0, gr00t_root)


def _build_processor_and_dataset(
    *,
    dataset_path: str,
    base_model_path: str,
    embodiment_tag: str,
    state_dropout_prob: float = 0.2,
) -> tuple[Any, Any, Any, Any, Any]:
    from transformers import AutoConfig
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7DataCollator, Gr00tN1d7Processor
    from gr00t_lancedb import LanceShardedSingleStepDataset
    from gr00t_lancedb._gr00t_compat import coerce_embodiment_tag

    cfg = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    cfg.state_dropout_prob = state_dropout_prob
    tag = coerce_embodiment_tag(embodiment_tag)
    modality_configs = MODALITY_CONFIGS[tag.value]

    ds = LanceShardedSingleStepDataset(
        dataset_path=dataset_path,
        embodiment_tag=tag,
        modality_configs=modality_configs,
        shard_size=1024,
        episode_sampling_rate=1.0,
        seed=42,
    )
    stats = ds.get_dataset_statistics()
    processor = Gr00tN1d7Processor(
        modality_configs={tag.value: modality_configs},
        statistics={tag.value: stats},
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
    collator = Gr00tN1d7DataCollator(
        model_name=cfg.model_name,
        model_type=cfg.backbone_model_type,
    )
    return cfg, tag, ds, processor, collator


# ---------------------------------------------------------------------------
# Phase 1: Materialize VLM features + compact scalar state/action cache
# ---------------------------------------------------------------------------


_BACKBONE_KEYS = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
_SCALAR_KEYS = ("state", "action", "action_mask", "embodiment_id")


class _AnchorBatchDataset(Dataset):
    """Decode, preprocess AND collate in the worker; the main thread only feeds the GPU."""

    def __init__(
        self, ds: Any, processor: Any, collator: Any, anchors: list[tuple[int, int]], chunk_size: int = 32
    ) -> None:
        self.ds = ds
        self.processor = processor
        self.collator = collator
        self.chunks = [anchors[i : i + chunk_size] for i in range(0, len(anchors), chunk_size)]

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> tuple[list[tuple[int, int]], dict[str, Any]]:
        chunk = self.chunks[idx]
        steps = self.ds.load_anchor_batch(chunk)
        proc_items = [self.processor([{"type": "episode_step", "content": s}]) for s in steps]
        inputs = self.collator(proc_items)["inputs"]
        return chunk, {k: inputs[k] for k in _BACKBONE_KEYS + _SCALAR_KEYS if k in inputs}


def materialize_full_cache(
    *,
    dataset_path: str,
    base_model_path: str,
    embodiment_tag: str,
    cache_dir: str,
    codec: str = "per_token_int8",
    batch_size: int = 64,
    num_workers: int = 4,
    max_frames: int = 0,
    reuse_existing: bool = True,
) -> dict[str, Any]:
    from transformers import AutoModel
    from gr00t_lancedb.vl_cache import (
        AsyncSink,
        CacheFingerprint,
        VLCacheWriter,
        encode_features_torch,
        pooled_visual_emb_torch,
    )

    cfg, tag, ds, processor, collator = _build_processor_and_dataset(
        dataset_path=dataset_path,
        base_model_path=base_model_path,
        embodiment_tag=embodiment_tag,
    )
    processor.eval()  # deterministic center crop for view 0

    fp = CacheFingerprint(
        base_model_path=base_model_path,
        select_layer=int(cfg.select_layer),
        hidden_dim=int(cfg.backbone_embedding_dim),
        image_target_size=tuple(cfg.image_target_size) if cfg.image_target_size else None,
        image_crop_size=tuple(cfg.image_crop_size) if cfg.image_crop_size else None,
        augmentation_signature="none",
        quant_codec=codec,
        backbone_name=str(cfg.model_name),
        num_views=1,
        pad_side="left",
    )

    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    meta_path = cache_root / f"{fp.table_name()}_summary.json"
    scalars_path = cache_root / f"{fp.table_name()}_scalars.npz"
    table_dir = cache_root / f"{fp.table_name()}.lance"

    if reuse_existing and meta_path.is_file() and scalars_path.is_file() and table_dir.exists():
        print(f"[materialize] Reusing existing cache at {table_dir}")
        return json.loads(meta_path.read_text(encoding="utf-8"))

    anchors = ds.get_all_valid_anchors()
    if max_frames > 0:
        anchors = anchors[:max_frames]
    total_frames = len(anchors)
    print(f"[materialize] Materializing {total_frames} frames -> {table_dir} (B={batch_size}, workers={num_workers})")

    t0 = time.perf_counter()
    device = "cuda"
    model = AutoModel.from_pretrained(base_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    backbone = model.backbone.to(device=device, dtype=torch.bfloat16).eval()
    del model.action_head
    del model
    gc.collect()
    torch.cuda.empty_cache()

    writer = VLCacheWriter(
        db_uri=cache_root,
        fingerprint=fp,
        include_pooled_emb=True,
        flush_rows=1024,
        overwrite=True,
    )
    sink = AsyncSink(writer.add_encoded_batch, maxsize=4)

    scalars: dict[str, list[np.ndarray]] = {k: [] for k in _SCALAR_KEYS}
    loader = DataLoader(
        _AnchorBatchDataset(ds, processor, collator, anchors, chunk_size=batch_size),
        batch_size=None,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        pin_memory=True,
        multiprocessing_context="fork" if num_workers > 0 else None,
    )

    t_inf0 = time.perf_counter()
    row_counter = 0
    wait_s = gpu_s = 0.0
    t_prev = time.perf_counter()
    for chunk, batch in loader:
        t_got = time.perf_counter()
        wait_s += t_got - t_prev
        # fp32 scalars: 4 bytes x ~100 values per frame is nothing next to the features
        scalars["state"].append(batch["state"].to(torch.float32).numpy())
        scalars["action"].append(batch["action"].to(torch.float32).numpy())
        scalars["action_mask"].append(batch["action_mask"].to(torch.bool).numpy())
        scalars["embodiment_id"].append(batch["embodiment_id"].to(torch.int32).numpy())

        bb_in = {
            k: (
                batch[k].to(device, dtype=torch.bfloat16, non_blocking=True)
                if torch.is_floating_point(batch[k])
                else batch[k].to(device, non_blocking=True)
            )
            for k in _BACKBONE_KEYS
            if k in batch
        }
        with torch.inference_mode():
            out = backbone(bb_in)
            feats = out.backbone_features
            attn = out.backbone_attention_mask.bool()
            img_m = out.image_mask.bool()
            # Quantise and pool on the GPU: int8 instead of fp32 crosses PCIe, and the
            # per-row numpy encode leaves the loop that the GPU waits on.
            payload, scale = encode_features_torch(feats, codec=codec)
            pooled = pooled_visual_emb_torch(feats, attn, img_m)
            host = {
                "payload": payload.cpu().numpy(),
                "scale": None if scale is None else scale.cpu().numpy(),
                "attention_mask": attn.cpu().numpy(),
                "image_mask": img_m.cpu().numpy(),
                "pooled": pooled.cpu().numpy(),
            }
        sink.put(
            row_keys=list(range(row_counter, row_counter + len(chunk))),
            episode_indices=[ep for ep, _ in chunk],
            frame_indices=[fr for _, fr in chunk],
            view_ids=[0] * len(chunk),
            **host,
        )
        row_counter += len(chunk)
        t_prev = time.perf_counter()
        gpu_s += t_prev - t_got
        if row_counter % (batch_size * 20) == 0 or row_counter >= total_frames:
            dt = time.perf_counter() - t_inf0
            print(
                f"[materialize] {row_counter}/{total_frames} ({row_counter / max(1e-6, dt):.1f} frames/s, "
                f"elapsed {dt:.1f}s | waiting for decode {wait_s:.0f}s, backbone+encode {gpu_s:.0f}s, "
                f"writer {sink.busy_s:.0f}s, backlog {sink.max_backlog})",
                flush=True,
            )

    sink.close()
    summary = writer.close()
    np.savez(
        scalars_path,
        episode_index=np.asarray([ep for ep, _ in anchors], dtype=np.int32),
        frame_index=np.asarray([fr for _, fr in anchors], dtype=np.int32),
        **{k: np.concatenate(v, axis=0) for k, v in scalars.items()},
    )
    inf_s = time.perf_counter() - t_inf0
    total_s = time.perf_counter() - t0
    summary.update(
        {
            "table_name": fp.table_name(),
            "scalars_path": str(scalars_path),
            "inference_and_write_s": round(inf_s, 2),
            "total_wall_s": round(total_s, 2),
            "frames_per_s": round(row_counter / max(1e-6, inf_s), 2),
            "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            # Which stage bounds throughput: the largest of these three is the one to fix next.
            "stage_s": {
                "waiting_for_decode": round(wait_s, 1),
                "backbone_and_gpu_encode": round(gpu_s, 1),
                "writer_thread_busy": round(sink.busy_s, 1),
            },
            "writer_max_backlog": sink.max_backlog,
            "mat_batch_size": batch_size,
            "mat_workers": num_workers,
        }
    )
    meta_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    del backbone
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[materialize] DONE: {json.dumps(summary, indent=2)}", flush=True)
    return summary


# ---------------------------------------------------------------------------
# Phase 2: ActionHead-Only Cached Training + Lossless Export
# ---------------------------------------------------------------------------


class _CachedStepBatchDataset(Dataset):
    """Generates pre-shuffled batches of (Lance VL features + scalar action inputs) in workers."""

    def __init__(
        self,
        *,
        cache_dir: str,
        table_name: str,
        scalars_path: str,
        batch_size: int,
        num_steps: int,
        seed: int = 42,
        row_pool: np.ndarray | None = None,
    ) -> None:
        from gr00t_lancedb.vl_cache.probe import load_scalars

        self.cache_dir = cache_dir
        self.table_name = table_name
        self.batch_size = int(batch_size)
        self.num_steps = int(num_steps)
        self.scalars = load_scalars(scalars_path)
        self.n_rows = int(self.scalars["state"].shape[0])
        pool = np.arange(self.n_rows, dtype=np.int64) if row_pool is None else np.asarray(row_pool, dtype=np.int64)

        # Deterministic epoch-shuffled indices covering num_steps * batch_size
        rng = np.random.default_rng(seed)
        needed = self.num_steps * self.batch_size
        chunks: list[np.ndarray] = []
        have = 0
        while have < needed:
            chunks.append(pool[rng.permutation(len(pool))])
            have += len(pool)
        self.schedule = np.concatenate(chunks)[:needed].reshape(self.num_steps, self.batch_size)
        self._reader: Any = None

    def _get_reader(self) -> Any:
        if self._reader is None:
            from gr00t_lancedb.vl_cache import VLCacheReader

            self._reader = VLCacheReader(self.cache_dir, self.table_name)
            _ = self._reader.num_rows
        return self._reader

    def __len__(self) -> int:
        return self.num_steps

    def __getitem__(self, step_idx: int) -> dict[str, Any]:
        from gr00t_lancedb.vl_cache.probe import load_cached_batch

        return load_cached_batch(self._get_reader(), self.scalars, self.schedule[step_idx].tolist())


def train_and_export(args: argparse.Namespace, mat_summary: dict[str, Any]) -> dict[str, Any]:
    from gr00t_lancedb.vl_cache import (
        FeatureSpaceAugmenter,
        build_action_head_optimizer,
        export_merged_gr00t_checkpoint,
        head_inputs_from_cached,
        load_action_head_only,
        patch_action_head_multi_noise,
        probe_fm_loss,
        save_action_head_checkpoint,
    )
    from gr00t_lancedb.vl_cache.probe import prepare_probe_isolated

    device = "cuda"
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()

    t_train_start = time.perf_counter()
    param_dtype = torch.float32 if args.fp32_master_weights else torch.bfloat16
    ah, cfg, _ = load_action_head_only(args.base_model_path, device=device, dtype=param_dtype)
    ah.state_dropout_prob = args.state_dropout_prob
    ah.train()
    if args.multi_noise > 1:
        patch_action_head_multi_noise(ah, k_noise=args.multi_noise, stratified=True)
    if args.compile:
        ah.model = torch.compile(ah.model, mode="default")

    opt, opt_name = build_action_head_optimizer(
        ah, lr=args.lr, weight_decay=args.weight_decay, kind=args.optimizer, seed=args.seed
    )
    lowmem = opt_name.startswith("LowMemAdamW")
    warmup_steps = max(1, int(args.max_steps * args.warmup_ratio))

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, args.max_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
    aug = FeatureSpaceAugmenter(token_dropout=args.token_dropout, gaussian_jitter=args.gaussian_jitter)

    frames_per_step = max(1, args.batch_size // max(1, args.multi_noise))
    # Whole episodes are held out for the probe; the split is rebuilt identically by
    # eval_cached_fm_loss.py from (holdout_frac, holdout_seed). Runs in a spawned child so this
    # process has not touched Lance when the DataLoader forks.
    probe = prepare_probe_isolated(
        args.cache_dir,
        mat_summary["table_name"],
        mat_summary["scalars_path"],
        args.holdout_frac,
        args.holdout_seed,
        args.probe_rows,
    )
    print(
        f"[train] rows: {len(probe['train_rows'])} train / {len(probe['probe_rows'])} probe "
        f"({probe['held_out_episodes']} held-out episodes)",
        flush=True,
    )
    ds = _CachedStepBatchDataset(
        cache_dir=args.cache_dir,
        table_name=mat_summary["table_name"],
        scalars_path=mat_summary["scalars_path"],
        batch_size=frames_per_step,
        num_steps=args.max_steps,
        seed=args.seed,
        row_pool=probe["train_rows"],
    )
    # The native trainer feeds fp32 state/action and lets autocast cast inside each layer.
    # With bf16 weights an op outside autocast's cast list could reject the dtype mix, so try it
    # once up front and fall back to parameter-dtype inputs (recorded in the summary) if it does.
    scalar_dtype = torch.float32
    try:
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            ah(*head_inputs_from_cached(probe["probe_batches"][0], device, scalar_dtype=scalar_dtype))
    except RuntimeError as exc:
        if param_dtype == torch.float32:
            raise
        print(f"[train] WARNING: fp32 inputs rejected by the bf16 head ({exc}); using bf16 inputs", flush=True)
        scalar_dtype = param_dtype

    probe_curve: list[dict[str, Any]] = []
    probe_s = 0.0

    def _run_probe(step: int) -> None:
        nonlocal probe_s
        t_p = time.perf_counter()
        rec = probe_fm_loss(
            ah, probe["probe_batches"], device=device, amp_dtype=torch.bfloat16, scalar_dtype=scalar_dtype
        )
        probe_s += time.perf_counter() - t_p
        rec["step"] = step
        probe_curve.append(rec)
        print(f"[train] probe @ {step}: {rec['probe_loss']:.5f} +- {rec['probe_sem']:.5f}", flush=True)

    loader = DataLoader(
        ds,
        batch_size=None,
        num_workers=args.num_workers,
        prefetch_factor=2 if args.num_workers > 0 else None,
        pin_memory=True,
        multiprocessing_context="fork" if args.num_workers > 0 else None,
    )

    print(
        f"[train] Starting ActionHead-Only training: steps={args.max_steps}, frames/step={frames_per_step}, "
        f"k_noise={args.multi_noise}, eff_B={frames_per_step * args.multi_noise}, compile={args.compile}, "
        f"opt={opt_name}",
        flush=True,
    )

    step_times: list[float] = []
    losses: list[float] = []
    t_loop0 = time.perf_counter()
    for step, batch in enumerate(loader):
        t_s0 = time.perf_counter()
        bb_out, act_in = head_inputs_from_cached(batch, device, scalar_dtype=scalar_dtype)
        bb_out = aug(bb_out, training=True)

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = ah(bb_out, act_in)["loss"]
        loss.backward()
        if lowmem:
            # fold the clip factor into the fused update instead of rewriting bf16 grads in place
            _, clip = opt.clip_scale(1.0)
            opt.step(grad_scale=clip)
        else:
            torch.nn.utils.clip_grad_norm_(ah.parameters(), 1.0)
            opt.step()
        sched.step()
        torch.cuda.synchronize()

        dt = time.perf_counter() - t_s0
        if step >= 10:
            step_times.append(dt)
        loss_val = float(loss.item())
        losses.append(loss_val)

        if args.probe_every > 0 and ((step + 1) % args.probe_every == 0 or step + 1 == args.max_steps):
            _run_probe(step + 1)
        if (step + 1) % 250 == 0 or step == 0 or step + 1 == args.max_steps:
            med_ms = float(np.median(step_times[-100:])) * 1000.0 if step_times else dt * 1000.0
            avg_loss = float(np.mean(losses[-50:]))
            print(
                f"[train] step {step + 1}/{args.max_steps} | loss={avg_loss:.4f} | "
                f"lr={sched.get_last_lr()[0]:.2e} | med_step={med_ms:.1f}ms",
                flush=True,
            )

    train_loop_s = time.perf_counter() - t_loop0
    peak_alloc_gb = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    peak_resv_gb = round(torch.cuda.max_memory_reserved() / 1024**3, 3)

    # Unwrap compiled model before saving state_dict so weight keys stay 100% standard
    if args.compile and hasattr(ah.model, "_orig_mod"):
        ah.model = ah.model._orig_mod

    out_root = Path(args.output_dir)
    ah_ckpt_dir = out_root / "action_head_ckpt"
    merged_dir = out_root / "merged_gr00t_ckpt"

    t_save0 = time.perf_counter()
    save_action_head_checkpoint(
        ah,
        ah_ckpt_dir,
        base_model_path=args.base_model_path,
        cache_fingerprint=mat_summary.get("fingerprint"),
        step=args.max_steps,
        save_dtype="bfloat16",
    )
    save_s = time.perf_counter() - t_save0

    t_exp0 = time.perf_counter()
    export_merged_gr00t_checkpoint(ah_ckpt_dir, merged_dir, base_model_path=args.base_model_path)
    # Save processor_config.json & statistics.json into merged_dir for run_gr00t_server.py
    _, _, _, processor, _ = _build_processor_and_dataset(
        dataset_path=args.dataset_path,
        base_model_path=args.base_model_path,
        embodiment_tag=args.embodiment_tag,
    )
    processor.save_pretrained(str(merged_dir))
    export_s = time.perf_counter() - t_exp0

    result = {
        "base_model_path": args.base_model_path,
        "dataset_path": args.dataset_path,
        "max_steps": args.max_steps,
        "frames_per_step": frames_per_step,
        "multi_noise": args.multi_noise,
        "effective_batch_size": frames_per_step * args.multi_noise,
        "compile": args.compile,
        "token_dropout": args.token_dropout,
        "gaussian_jitter": args.gaussian_jitter,
        "optimizer": opt_name,
        "param_dtype": str(param_dtype).replace("torch.", ""),
        "optimizer_state_gb": round(opt.state_bytes() / 1024**3, 3) if lowmem else None,
        "materialize_s": mat_summary.get("total_wall_s"),
        "train_loop_s": round(train_loop_s, 2),
        "train_total_wall_s": round(time.perf_counter() - t_train_start, 2),
        "median_step_ms": round(float(np.median(step_times)) * 1000.0, 2) if step_times else None,
        "peak_alloc_gb": peak_alloc_gb,
        "peak_reserved_gb": peak_resv_gb,
        "final_loss_mean50": round(float(np.mean(losses[-50:])), 5),
        "input_dtype": str(scalar_dtype).replace("torch.", ""),
        "holdout_frac": args.holdout_frac,
        "holdout_seed": args.holdout_seed,
        "train_rows": int(len(probe["train_rows"])),
        "probe_rows": int(len(probe["probe_rows"])),
        "probe_curve": probe_curve,
        "final_probe_loss": probe_curve[-1]["probe_loss"] if probe_curve else None,
        "probe_total_s": round(probe_s, 2),
        "save_action_head_s": round(save_s, 2),
        "export_merged_s": round(export_s, 2),
        "merged_checkpoint_dir": str(merged_dir),
    }
    (out_root / "train_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[train] DONE: {json.dumps(result, indent=2)}", flush=True)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gr00t-root", default="/data/Isaac-GR00T")
    p.add_argument("--base-model-path", default="nvidia/GR00T-N1.7-3B")
    p.add_argument("--dataset-path", default="/data/datasets/libero_10_lance")
    p.add_argument("--embodiment-tag", default="LIBERO_PANDA")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--codec", default="per_token_int8")
    p.add_argument("--mat-batch-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--multi-noise", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--state-dropout-prob", type=float, default=0.2)
    p.add_argument("--token-dropout", type=float, default=0.0)
    p.add_argument("--gaussian-jitter", type=float, default=0.0)
    p.add_argument("--fp32-master-weights", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--optimizer",
        choices=["adamw", "bnb8bit", "lowmem"],
        default="adamw",
        help="lowmem = 8-bit moments + Kahan-compensated bf16 weights; pair it with --no-fp32-master-weights",
    )
    p.add_argument("--compile", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--mat-workers", type=int, default=8, help="decode/preprocess workers for materialisation")
    p.add_argument("--holdout-frac", type=float, default=0.03, help="fraction of episodes kept out of training")
    p.add_argument("--holdout-seed", type=int, default=0)
    p.add_argument("--probe-rows", type=int, default=1024)
    p.add_argument("--probe-every", type=int, default=500, help="steps between probe evaluations; 0 disables")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-frames", type=int, default=0)
    args = p.parse_args()

    _add_gr00t_root(args.gr00t_root)
    mat_summary = materialize_full_cache(
        dataset_path=args.dataset_path,
        base_model_path=args.base_model_path,
        embodiment_tag=args.embodiment_tag,
        cache_dir=args.cache_dir,
        codec=args.codec,
        batch_size=args.mat_batch_size,
        num_workers=args.mat_workers,
        max_frames=args.max_frames,
    )
    train_and_export(args, mat_summary)


if __name__ == "__main__":
    main()
