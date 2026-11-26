# SPDX-License-Identifier: Apache-2.0
"""End-to-end S3 streaming tests via real local moto HTTP S3 server (Full-S3 & Hybrid modes)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gr00t_lancedb import (
    EmbodimentTag,
    LanceEpisodeLoader,
    LanceShardedSingleStepDataset,
    ModalityConfig,
    ResumableLanceMixtureDataset,
    attach_gr00t_meta,
    convert_v2_to_lance,
)
from conftest import reference_load_v2_step


def _upload_dir_to_s3(client, bucket: str, prefix: str, local_dir: Path) -> None:
    for path in sorted(local_dir.rglob("*")):
        if path.is_file():
            key = f"{prefix.rstrip('/')}/{path.relative_to(local_dir).as_posix()}"
            client.upload_file(str(path), bucket, key)


def test_s3_full_and_hybrid_external_blob_streaming(
    v2_dataset_dir: Path,
    default_modality_configs: dict[str, ModalityConfig],
    moto_s3_server: dict,
    tmp_path: Path,
) -> None:
    client = moto_s3_server["client"]
    bucket = moto_s3_server["bucket"]
    s3_opts = moto_s3_server["storage_options"]

    # Upload raw GR00T v2.1 dataset to S3
    _upload_dir_to_s3(client, bucket, "v2_src", v2_dataset_dir)
    s3_src_uri = f"s3://{bucket}/v2_src"

    # 1. Zero-copy S3 -> S3 conversion (External Blob v2 referencing original v2.1 MP4s in place!)
    s3_lance_uri = f"s3://{bucket}/lance_full_s3"
    convert_v2_to_lance(
        src_root=s3_src_uri,
        out_target=s3_lance_uri,
        video_mode="external",
        storage_options=s3_opts,
        source_storage_options=s3_opts,
    )

    # Stream directly from s3:// with zero local dataset download!
    read_opts = {
        **s3_opts,
        **{f"base_1.{k}": v for k, v in s3_opts.items()},
    }
    s3_loader = LanceEpisodeLoader(
        s3_lance_uri,
        default_modality_configs,
        storage_options=read_opts,
    )
    actual_s3 = s3_loader.load_steps(1, [2, 9], EmbodimentTag.NEW_EMBODIMENT, allow_padding=True)
    for step_i, act in zip((2, 9), actual_s3, strict=True):
        exp = reference_load_v2_step(v2_dataset_dir, 1, step_i, default_modality_configs, allow_padding=True)
        np.testing.assert_allclose(act.states["arm"], exp.states["arm"], rtol=0, atol=1e-6)
        np.testing.assert_allclose(act.actions["arm"], exp.actions["arm"], rtol=0, atol=1e-6)
        for cam in ("front", "wrist"):
            for img_a, img_e in zip(act.images[cam], exp.images[cam], strict=True):
                np.testing.assert_array_equal(img_a, img_e)
        assert act.text == exp.text
    s3_loader.close()

    # 2. Hybrid mode: tiny Lance tables local, MP4 blobs stay on S3 (range-GET on demand!)
    hybrid_dir = tmp_path / "lance_hybrid_local"
    convert_v2_to_lance(
        src_root=s3_src_uri,
        out_target=hybrid_dir,
        video_mode="external",
        source_storage_options=s3_opts,
    )
    # Confirm ZERO .mp4 files exist under hybrid_dir!
    assert list(hybrid_dir.rglob("*.mp4")) == []

    hybrid_ds = LanceShardedSingleStepDataset(
        dataset_path=hybrid_dir,
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        modality_configs=default_modality_configs,
        allow_padding=True,
        storage_options={f"base_1.{k}": v for k, v in s3_opts.items()},
    )
    mix = ResumableLanceMixtureDataset(
        datasets=[hybrid_ds],
        weights=[1.0],
        batch_size=4,
        episode_pool_size=2,
        prefetch_anchors=8,
        infinite=False,
    )
    batches = list(mix.loader(num_workers=0, collate_fn=lambda xs: xs))
    total_samples = sum(len(b) for b in batches)
    assert total_samples == len(mix)
    first_step = batches[0][0]
    assert first_step.images["front"][0].dtype == np.uint8
    assert first_step.images["front"][0].shape == (48, 64, 3)
    mix.close()

    # 3. Verify attach_gr00t_meta works directly against s3:// Lance table!
    attach_gr00t_meta(
        lance_root=s3_lance_uri,
        modality_json_path=v2_dataset_dir / "meta" / "modality.json",
        compute_stats=True,
        storage_options=s3_opts,
    )
