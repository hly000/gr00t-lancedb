#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Example: Finetune NVIDIA Isaac-GR00T directly from S3 / Hybrid LanceDB with Spot resume.

Drop-in replacement for GR00T's ``launch_finetune.py`` that:
1. Streams data directly from local Hybrid or ``s3://`` Lance tables (no upfront MP4 download).
2. Replaces GR00T's non-reproducible ``seed + global_step`` resume with exact
   ``ResumableLanceMixtureDataset`` cursor checkpointing.
3. Captures cloud Spot ``SIGTERM`` notices to flush both model weights and dataloader state
   cleanly before termination.

Usage (on a Linux GPU node with NVIDIA/Isaac-GR00T installed):
    torchrun --nproc_per_node=1 examples/finetune_gr00t_lance.py \\
        --dataset-uri s3://my-bucket/droid-lance \\
        --embodiment-tag oxe_droid \\
        --base-model-path nvidia/GR00T-N1.6-3B \\
        --output-dir ./outputs/gr00t-lance-spot \\
        --max-steps 500 \\
        --batch-size 8 \\
        --num-workers 4 \\
        --resume-if-available
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from lerobot_lancedb import lance_mp_context

from gr00t_lancedb import (
    build_lance_mixture_dataset,
    create_spot_resumable_gr00t_trainer_class,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Finetune GR00T with LanceDB S3 streaming + Spot resume.")
    p.add_argument("--dataset-uri", action="append", required=True, help="Local Lance dir or s3:// URI.")
    p.add_argument("--modality-json", default=None, help="Optional sidecar modality.json for v3 Lance datasets.")
    p.add_argument("--embodiment-tag", default="new_embodiment")
    p.add_argument("--base-model-path", required=True, help="Pretrained GR00T checkpoint / HF repo id.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--episode-pool-size", type=int, default=16)
    p.add_argument("--prefetch-anchors", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume-if-available", action="store_true")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    # Lazy import GR00T components so this file remains import-safe on CPU dev boxes.
    from transformers import TrainingArguments, get_last_checkpoint, set_seed

    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.experiment.trainer import Gr00tTrainer
    from gr00t.model.gr00t_n1d6 import Gr00tN1d6, Gr00tN1d6Processor

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = Gr00tN1d6Processor.from_pretrained(args.base_model_path)
    model = Gr00tN1d6.from_pretrained(args.base_model_path)

    train_dataset = build_lance_mixture_dataset(
        dataset_specs=[
            {
                "dataset_paths": args.dataset_uri,
                "embodiment_tag": args.embodiment_tag,
                "mix_ratio": 1.0,
                "modality_json_path": args.modality_json,
            }
        ],
        modality_configs_by_emb=MODALITY_CONFIGS,
        processor=processor,
        seed=args.seed,
        batch_size=args.batch_size,
        episode_pool_size=args.episode_pool_size,
        prefetch_anchors=args.prefetch_anchors,
        infinite=True,
    )

    SpotTrainer = create_spot_resumable_gr00t_trainer_class(Gr00tTrainer)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_steps=args.save_steps,
        logging_steps=10,
        bf16=True,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
        ignore_data_skip=True,
        seed=args.seed,
    )

    trainer = SpotTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=processor.collator,
        multiprocessing_context=lance_mp_context(),
    )

    resume_ckpt = None
    if args.resume_if_available:
        resume_ckpt = get_last_checkpoint(str(output_dir))
        if resume_ckpt:
            logging.info("Resuming seamlessly from checkpoint %s", resume_ckpt)

    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model()
    train_dataset.close()


if __name__ == "__main__":
    main()
