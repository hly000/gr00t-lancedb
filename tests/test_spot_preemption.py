# SPDX-License-Identifier: Apache-2.0
"""Real subprocess SIGTERM Spot preemption injection test verifying graceful checkpoint & exact resume."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import textwrap

from gr00t_lancedb import convert_v2_to_lance
from gr00t_lancedb.bench import audit_sample_stream, run_fault_injected_command


_WORKER_SCRIPT = textwrap.dedent(
    """\
    import json, sys, time
    from pathlib import Path
    from gr00t_lancedb import (
        EmbodimentTag,
        LanceShardedSingleStepDataset,
        ModalityConfig,
        ResumableLanceMixtureDataset,
        SpotPreemptionMonitor,
        load_dataloader_state,
        save_dataloader_state,
    )

    lance_dir = Path(sys.argv[1])
    ckpt_root = Path(sys.argv[2])
    out_log = Path(sys.argv[3])
    target_steps = int(sys.argv[4])
    step_sleep_s = float(sys.argv[5])

    modality_configs = {
        "video": ModalityConfig(delta_indices=[0], modality_keys=["front"]),
        "state": ModalityConfig(delta_indices=[0], modality_keys=["arm"]),
        "action": ModalityConfig(delta_indices=[0, 1, 2, 3], modality_keys=["arm"]),
        "language": ModalityConfig(delta_indices=[0], modality_keys=["task"]),
    }

    ds = LanceShardedSingleStepDataset(
        lance_dir, EmbodimentTag.NEW_EMBODIMENT, modality_configs, allow_padding=True
    )
    mix = ResumableLanceMixtureDataset(
        datasets=[ds],
        weights=[1.0],
        seed=42,
        batch_size=4,
        episode_pool_size=2,
        prefetch_anchors=4,
        infinite=True,
    )

    state_file = ckpt_root / "step_meta.json"
    step = 0
    seen = []
    if state_file.exists():
        meta = json.loads(state_file.read_text())
        step = int(meta["step"])
        seen = list(meta["seen"])
        load_dataloader_state(mix, ckpt_root)

    monitor = SpotPreemptionMonitor().install()
    try:
        for batch in mix.loader(num_workers=0, collate_fn=lambda xs: xs):
            for item in batch:
                seen.append([
                    item.metadata["epoch"],
                    item.metadata["global_pos"],
                    item.metadata["episode_index"],
                    item.metadata["step_index"],
                ])
            step += 1
            time.sleep(step_sleep_s)
            if monitor.preempted or step >= target_steps:
                ckpt_root.mkdir(parents=True, exist_ok=True)
                save_dataloader_state(mix, ckpt_root)
                state_file.write_text(json.dumps({"step": step, "seen": seen}))
                if monitor.preempted and step < target_steps:
                    sys.exit(143)
                break
        out_log.write_text(json.dumps({"step": step, "seen": seen}))
    finally:
        monitor.restore()
        mix.close()
    """
)


def test_real_sigterm_spot_preemption_and_exact_recovery(
    v2_dataset_dir: Path,
    tmp_path: Path,
) -> None:
    lance_dir = tmp_path / "lance_spot"
    convert_v2_to_lance(v2_dataset_dir, lance_dir, video_mode="copy")

    worker = tmp_path / "spot_worker.py"
    worker.write_text(_WORKER_SCRIPT, encoding="utf-8")

    # 1. Uninterrupted reference run (12 steps x batch 4 = 48 samples)
    ref_ckpt = tmp_path / "ckpt_ref"
    ref_out = tmp_path / "out_ref.json"
    ref_summary = run_fault_injected_command(
        [sys.executable, str(worker), str(lance_dir), str(ref_ckpt), str(ref_out), "12", "0.02"],
        preempt_intervals_s=None,
    )
    assert ref_summary.completed is True
    assert ref_summary.num_restarts == 0
    ref_seen = [tuple(x) for x in json.loads(ref_out.read_text())["seen"]]
    assert len(ref_seen) == 48

    # 2. Spot-preempted run: inject 2 real SIGTERM signals at 0.35s and 0.35s
    spot_ckpt = tmp_path / "ckpt_spot"
    spot_out = tmp_path / "out_spot.json"
    spot_summary = run_fault_injected_command(
        [sys.executable, str(worker), str(lance_dir), str(spot_ckpt), str(spot_out), "12", "0.10"],
        preempt_intervals_s=[0.35, 0.35],
        signal_kind="SIGTERM",
        grace_period_s=10.0,
        max_restarts=5,
        gpu_hourly_usd=1.20,
    )
    assert spot_summary.completed is True
    assert spot_summary.num_injected_preemptions >= 1
    assert spot_summary.num_restarts >= 1
    assert spot_summary.estimated_cost_usd() is not None

    spot_seen = [tuple(x) for x in json.loads(spot_out.read_text())["seen"]]
    audit = audit_sample_stream(ref_seen, spot_seen)
    assert audit["duplicate_count"] == 0
    assert audit["missed_from_reference"] == 0
    assert audit["exact_order_match"] is True
