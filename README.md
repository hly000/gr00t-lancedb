# gr00t-lancedb

**Zero-download S3 streaming, `modality.json` adaptation, and exact Spot-preemption resume for [NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) on [LanceDB](https://github.com/lancedb/lerobot-lancedb).**

## Why this exists

Training NVIDIA Isaac-GR00T on large robotics datasets (e.g. DROID ~370 GB, multi-dataset mixtures) in cloud Spot GPU environments hits three concrete bottlenecks today:

1. **Local POSIX lock-in & v3→v2 downgrade tax**: GR00T's `LeRobotEpisodeLoader` requires local LeRobot v2.1 directories (`episodes.jsonl` + per-episode Parquet/MP4 + `modality.json`). Training on upstream LeRobot v3.0 datasets like `lerobot/droid_1.0.1` forces a full download and `convert_v3_to_v2.py` downgrade before step 0.
2. **Repeated cold-start tax on Spot preemptions**: Ephemeral Spot GPUs lose local NVMe state upon reclamation, forcing re-downloads and risking host RAM spikes.
3. **Lossy resume in `Gr00tTrainer`**: `Gr00tTrainer.get_train_dataloader()` does not checkpoint dataloader cursors — on resume it executes `new_seed = seed + global_step; reset_seed(new_seed)`, which breaks reproducibility and silently duplicates/drops episodes across frequent Spot restarts.

**`gr00t-lancedb`** bridges `lerobot-lancedb`'s standard three-table Lance layout (`frames.lance`, `videos.lance`, `meta.lance` with External Blob v2 + mini-MP4 range reads) directly into GR00T's `ShardedDataset` and `Gr00tTrainer` interfaces:

- **One Lance dataset on S3 trains both Hugging Face LeRobot and NVIDIA GR00T** — zero format fragmentation and zero v3→v2 downgrade.
- **100% GR00T `modality.json` parity** — joint-group slicing (`start`/`end`/`original_key`), N1.6/N1.7 camera auto-mapping, `task`/`sub_task`/`annotation.*` language modes, precomputed `q01`/`q99` & `relative_stats` in `meta.lance`, and direct `NHWC uint8` `VLAStepData` output (no float32 round-trip).
- **Exact multi-worker Spot preemption resume** — `ResumableLanceMixtureDataset` persists a ~150-byte cursor per rank, handles `SIGTERM` Spot notices cleanly, forces safe `spawn`/`forkserver` multiprocessing (avoiding Rust Tokio `fork` deadlocks), and resumes with **0 duplicates, 0 missed samples, and 0 skipped-frame decoding**.

## Architecture

```text
  S3 LeRobot v3 Lance (e.g. DROID) ──► gr00t-lance-attach (embeds modality.json + q01/q99 stats) ──┐
                                                                                                    ▼
  GR00T LeRobot v2.1 (S3 or local) ──► gr00t-lance-convert (zero-copy External Blob v2 MP4s) ──► [frames.lance + videos.lance + meta.lance]
                                                                                                    │
                                                                                                    ▼
                                                                                     LanceEpisodeLoader (return_uint8=True, NHWC)
                                                                                                    │
                                                                                                    ▼
                                                                                    LanceShardedSingleStepDataset (GR00T ShardedDataset)
                                                                                                    │
                                                                                                    ▼
                                                                                ResumableLanceMixtureDataset + SpotResumableGr00tTrainer
                                                                                  • Multi-dataset mix_ratio / ds_weights_alpha + merge_statistics
                                                                                  • Exact multi-worker state_dict() / load_state_dict()
                                                                                  • SIGTERM Spot preemption callback + lance_mp_context()
```

## Quickstart

### Install

```bash
uv pip install \
  --prerelease=allow \
  --extra-index-url https://pypi.fury.io/lancedb/ \
  "git+https://github.com/lancedb/lerobot-lancedb.git#egg=lerobot-lancedb[s3]" \
  "lerobot[dataset]>=0.7.0" \
  "lancedb>=0.42.0b26" \
  -e ".[s3]"
```

### Path A — Train GR00T directly on an existing v3 Lance dataset (zero v3→v2 downgrade)

```bash
gr00t-lance-attach \
  --root s3://my-bucket/droid-lance \
  --modality-json ./configs/droid_modality.json
```

### Path B — Zero-copy convert a GR00T v2.1 dataset on S3

```bash
# Full-S3 output (MP4s remain in place via External Blob v2; only MP4 headers are range-read):
gr00t-lance-convert \
  --src s3://my-bucket/my-v2-dataset \
  --out s3://my-bucket/my-v2-lance \
  --video-mode external

# Hybrid output (tiny Lance tables on local NVMe, large MP4s stream from S3 on demand):
gr00t-lance-convert \
  --src s3://my-bucket/my-v2-dataset \
  --out ./my-v2-hybrid \
  --video-mode external
```

### Finetune GR00T with exact Spot preemption resume

```python
from gr00t_lancedb import (
    build_lance_mixture_dataset,
    create_spot_resumable_gr00t_trainer_class,
)
from gr00t.experiment.trainer import Gr00tTrainer

train_dataset = build_lance_mixture_dataset(
    dataset_specs=[{"dataset_paths": ["s3://my-bucket/droid-lance"], "embodiment_tag": "oxe_droid"}],
    modality_configs_by_emb=MODALITY_CONFIGS,
    processor=processor,
    batch_size=8,
    episode_pool_size=16,
    prefetch_anchors=16,
)

SpotTrainer = create_spot_resumable_gr00t_trainer_class(Gr00tTrainer)
trainer = SpotTrainer(model=model, args=training_args, train_dataset=train_dataset, data_collator=processor.collator)
trainer.train(resume_from_checkpoint=True)
```

See [examples/finetune_gr00t_lance.py](examples/finetune_gr00t_lance.py) and [examples/skypilot_spot_s3.yaml](examples/skypilot_spot_s3.yaml) for full end-to-end training & AWS Spot recipes.

### Benchmark & Spot Preemption Harness

```bash
# Compare sample coverage under repeated preemptions (GR00T seed+global_step vs exact Lance cursor)
gr00t-spot-bench coverage-compare --total-anchors 10000 --batch-size 32 --preempt-steps 50,120,210

# Inject periodic or Poisson SIGTERM/SIGKILL preemptions against any training command
gr00t-spot-bench inject --poisson-mean-s 600 --signal SIGTERM --gpu-hourly-usd 1.80 -- \
  torchrun --nproc_per_node=1 examples/finetune_gr00t_lance.py ...
```

## License

Apache-2.0
