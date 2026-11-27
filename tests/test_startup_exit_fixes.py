# SPDX-License-Identifier: Apache-2.0
"""Tests for the four startup / shutdown fixes found by the v2 A100 runs.

1. The Lance child process must not import ``gr00t`` / torch / pandas (it cost ~12 s).
2. Startup queries can be started early and are picked up by the loader.
3. The emergency-checkpoint time budget includes the encode step, not only the write.
4. DataLoader workers leave at once at shutdown instead of stalling 5 s each.
"""

from __future__ import annotations

import gc
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, IterableDataset

from gr00t_lancedb import LanceEpisodeLoader, ModalityConfig, convert_v2_to_lance, prefetch_lance_startup
from gr00t_lancedb import _isolated
from gr00t_lancedb._isolated import IsolatedCallError, is_started, run_isolated, start_isolated
from gr00t_lancedb._startup import MATERIALIZE_META, QUERY_WHERE, isolated_kwargs, where_kwargs
from gr00t_lancedb.exact_resume import CALIBRATION_FILENAME, WriteBandwidthCalibration, plan_preempt_save

TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"


def _child_env(**extra: str) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(SRC_DIR), str(TESTS_DIR), env.get("PYTHONPATH", "")])
    env.update(extra)
    return env


# ---------------------------------------------------------------------------
# 1. light child imports
# ---------------------------------------------------------------------------


def test_lance_query_import_stays_light() -> None:
    code = (
        "import sys, gr00t_lancedb.lance_query\n"
        "heavy = [m for m in ('torch', 'pandas', 'gr00t', 'lerobot', 'lerobot_lancedb', "
        "'huggingface_hub', 'gr00t_lancedb.metadata', 'gr00t_lancedb._gr00t_compat') if m in sys.modules]\n"
        "print('HEAVY', heavy)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=120, check=False,
        env=_child_env(**{_isolated.CHILD_ENV_FLAG: "1"}),
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "HEAVY []" in proc.stdout, proc.stdout


def test_manifest_helpers_still_exported_from_metadata() -> None:
    from gr00t_lancedb import _curation_manifest, metadata

    assert metadata.select_curation_entry is _curation_manifest.select_curation_entry
    assert metadata.CURATION_VERSIONS_FILENAME == "curation_versions.json"
    assert metadata.versioned_meta_filename("gr00t_stats", 7) == "gr00t_stats_v7.json"


# ---------------------------------------------------------------------------
# 2. early start + pick-up
# ---------------------------------------------------------------------------


def test_started_call_is_picked_up_without_waiting() -> None:
    func, kwargs = "_iso_helpers:slow_echo", {"value": 7, "delay": 0.2}
    assert start_isolated(func, kwargs) is True
    assert start_isolated(func, kwargs) is False  # same identity: already running
    assert is_started(func, kwargs)
    assert not is_started(func, {"value": 8, "delay": 0.2})

    # Stand-in for the model load the child overlaps with: wait until it has finished.
    assert _isolated._STARTED[_isolated.call_key(func, kwargs)]._done.wait(120)
    res = run_isolated(func, kwargs)
    assert res["value"] == 7 and res["pid"] != os.getpid()
    assert res["_isolated_wall_s"] >= 0.2
    assert res["_isolated_wait_s"] < 0.5  # nothing left to wait for
    assert not is_started(func, kwargs)

    again = run_isolated(func, kwargs)  # nothing started: runs its own child
    assert again["pid"] != res["pid"]
    assert again["_isolated_wait_s"] == pytest.approx(again["_isolated_wall_s"], abs=0.2)


def test_isolated_errors_and_timeout() -> None:
    with pytest.raises(IsolatedCallError, match="boom from the child"):
        run_isolated("_iso_helpers:boom")
    t0 = time.perf_counter()
    with pytest.raises(IsolatedCallError, match="timed out"):
        run_isolated("_iso_helpers:slow_echo", {"value": 1, "delay": 60.0}, timeout=1.5)
    assert time.perf_counter() - t0 < 20.0


def test_prefetch_is_matched_by_the_loader_local_where(
    v2_dataset_dir: Path, default_modality_configs: dict[str, ModalityConfig], tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("GR00T_LANCE_CACHE_DIR", str(tmp_path / "cache"))
    lance_dir = tmp_path / "lance"
    convert_v2_to_lance(v2_dataset_dir, lance_dir, video_mode="copy")
    where = "frame_index < 5"

    plain = LanceEpisodeLoader(lance_dir, default_modality_configs, where=where)
    assert plain.startup_info["isolated_calls"] == 1

    assert prefetch_lance_startup([lance_dir], where=where) == ["query_where"]
    key = where_kwargs(isolated_kwargs(lance_dir), where, None)
    assert is_started(QUERY_WHERE, key)
    early = LanceEpisodeLoader(lance_dir, default_modality_configs, where=where)
    try:
        assert not is_started(QUERY_WHERE, key), "the loader did not pick the early call up"
        assert early.startup_info["isolated_calls"] == 1
        assert early.startup_info["isolated_wait_s"] <= early.startup_info["isolated_wall_s"] + 1e-6
        assert early.frames_version == plain.frames_version
        for i in range(len(plain)):
            np.testing.assert_array_equal(
                early.get_valid_steps_for_episode(i), plain.get_valid_steps_for_episode(i)
            )
        early.assert_unopened()
    finally:
        plain.close()
        early.close()

    # A pinned version whose result is already cached needs no query, so none is started.
    pinned = LanceEpisodeLoader(lance_dir, default_modality_configs, where=where, lance_version=plain.frames_version)
    pinned.close()
    assert prefetch_lance_startup([lance_dir], where=where, lance_version=plain.frames_version) == []
    assert prefetch_lance_startup([lance_dir], where=where, lance_versions={str(lance_dir): plain.frames_version}) == []
    assert prefetch_lance_startup([lance_dir]) == []  # local, no predicate: nothing to do


def test_prefetch_is_matched_by_the_loader_remote_meta(
    v2_dataset_dir: Path,
    default_modality_configs: dict[str, ModalityConfig],
    moto_s3_server: dict,
    tmp_path: Path,
) -> None:
    import shutil

    from test_s3_streaming import _upload_dir_to_s3

    lance_dir = tmp_path / "lance"
    convert_v2_to_lance(v2_dataset_dir, lance_dir, video_mode="copy")
    prefix = f"prefetch-{uuid.uuid4().hex[:8]}"
    _upload_dir_to_s3(moto_s3_server["client"], moto_s3_server["bucket"], prefix, lance_dir)
    uri = f"s3://{moto_s3_server['bucket']}/{prefix}"
    opts = moto_s3_server["storage_options"]

    assert prefetch_lance_startup([uri], storage_options=opts) == ["materialize_meta"]
    cold = LanceEpisodeLoader(uri, default_modality_configs, storage_options=opts)
    try:
        assert not is_started(MATERIALIZE_META, isolated_kwargs(uri, storage_options=opts))
        assert cold.startup_info["isolated_calls"] == 1
        assert (cold.local_meta_root / "meta" / "info.json").is_file()
        cold.assert_unopened()
        local = LanceEpisodeLoader(lance_dir, default_modality_configs)
        assert cold.episode_lengths == local.episode_lengths
        local.close()

        # Warm cache: the early call still answers (it only resolves the cached root).
        assert prefetch_lance_startup([uri], storage_options=opts) == ["materialize_meta"]
        warm = LanceEpisodeLoader(uri, default_modality_configs, storage_options=opts)
        assert warm.local_meta_root == cold.local_meta_root
        warm.close()
    finally:
        cold.close()
        shutil.rmtree(cold.local_meta_root, ignore_errors=True)


def test_launcher_reads_dataset_paths_like_gr00t() -> None:
    spec = importlib.util.spec_from_file_location(
        "_launch_for_prefetch", TESTS_DIR.parent / "scripts" / "launch_finetune_lance.py"
    )
    launch = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(launch)
    paths = launch._dataset_paths_from_args
    assert paths(["--dataset-path", "/data/a", "--max-steps", "5"]) == ["/data/a"]
    assert paths(["--dataset_path=/data/a"]) == ["/data/a"]
    assert paths(["--dataset-path", "s3://bucket/x/lance"]) == ["s3://bucket/x/lance"]
    joined = os.pathsep.join(["s3://b/one", "/data/two"])
    assert paths(["--dataset-path", joined]) == ["s3://b/one", "/data/two"]
    assert paths(["--max-steps", "5"]) == []


# ---------------------------------------------------------------------------
# 3. emergency checkpoint budget
# ---------------------------------------------------------------------------


def test_save_estimate_includes_encode_time(tmp_path: Path) -> None:
    path = tmp_path / CALIBRATION_FILENAME
    cal = WriteBandwidthCalibration(path, prior_bps=400e6, overhead_s=1.0, prior_encode_bps=1e9)
    # write only (regular HF checkpoints): unchanged behaviour
    assert cal.estimate_s(800_000_000) == pytest.approx(3.0)
    # light emergency checkpoint: + encode at the prior rate
    assert cal.estimate_s(800_000_000, "bf16") == pytest.approx(3.0 + 0.8)

    cal.add_encode(9_723_845_281, 9.27, "bf16")  # the v2 A100 measurement
    assert cal.estimate_encode_s(9_723_845_281, "bf16") == pytest.approx(9.27)
    assert cal.encode_bytes_per_s("int8") == pytest.approx(1e9)  # other presets keep the prior
    again = WriteBandwidthCalibration(path, prior_bps=400e6, overhead_s=1.0)
    assert again.estimate_encode_s(9_723_845_281, "bf16") == pytest.approx(9.27)
    assert json.loads(path.read_text())["encode_samples"][0]["dtype"] == "bf16"
    assert again.bytes_per_s() == pytest.approx(400e6)  # encode samples never count as write speed


def test_auto_dtype_respects_deadline_once_encode_is_counted() -> None:
    """Replays the v2 A100 preemption: bf16 looked like 22 s but took 31 s (9.3 s encode)."""
    cal = WriteBandwidthCalibration(None, prior_bps=462.5e6, overhead_s=1.0, prior_encode_bps=1e9)
    sizes = {"fp32": 19_440_000_000, "bf16": 9_723_845_281, "int8": 6_490_000_000}
    write_only = {d: cal.estimate_s(n) for d, n in sizes.items()}
    with_encode = {d: cal.estimate_s(n, d) for d, n in sizes.items()}
    assert write_only["bf16"] < 25.0 < with_encode["bf16"]

    common = dict(steps_since_save=400, avg_step_s=0.7, last_saved_step=30, dtype="auto",
                  deadline_s=25.0, cost_gate=True)
    assert plan_preempt_save(est_by_dtype=write_only, **common)["dtype"] == "bf16"  # the old, late choice
    decision = plan_preempt_save(est_by_dtype=with_encode, **common)
    assert decision["action"] == "save" and decision["dtype"] == "int8"
    assert decision["est_save_s"] <= 25.0


def test_preempt_callback_budgets_encode(tmp_path: Path, monkeypatch) -> None:
    from gr00t_lancedb.exact_resume import _make_preempt_callback

    for var in ("GR00T_PREEMPT_EST_SAVE_S", "GR00T_PREEMPT_PRIOR_WRITE_MBPS", "GR00T_PREEMPT_PRIOR_ENCODE_MBPS"):
        monkeypatch.delenv(var, raising=False)

    class _Monitor:
        preempted = False

        def install(self) -> None: ...
        def restore(self) -> None: ...
        def broadcast_preempted(self) -> bool:
            return False

    cb = _make_preempt_callback(_Monitor(), exit_on_preempt=False)
    assert cb.estimate_save_s(400_000_000) == pytest.approx(1.0 + 1.0)
    assert cb.estimate_save_s(400_000_000, "bf16") == pytest.approx(1.0 + 1.0 + 0.4)
    monkeypatch.setenv("GR00T_PREEMPT_EST_SAVE_S", "55")
    pinned = _make_preempt_callback(_Monitor(), exit_on_preempt=False)
    assert pinned.estimate_save_s(400_000_000, "bf16") == 55.0  # a pinned estimate wins


# ---------------------------------------------------------------------------
# 4. worker exit
# ---------------------------------------------------------------------------


class _StuckThreadDataset(IterableDataset):
    """Each worker starts a non-daemon thread that never ends, like a hung decode pool."""

    def __iter__(self):
        from gr00t_lancedb._worker_exit import arm_worker_fast_exit

        arm_worker_fast_exit()
        threading.Thread(target=threading.Event().wait, name="stuck", daemon=False).start()
        i = 0
        while True:
            yield torch.tensor([i])
            i += 1


def _shutdown_seconds(num_workers: int) -> float:
    dl = DataLoader(
        _StuckThreadDataset(), batch_size=2, num_workers=num_workers,
        persistent_workers=True, multiprocessing_context="fork",
    )
    it = iter(dl)
    for _ in range(4):
        next(it)
    t0 = time.perf_counter()
    del it, dl
    gc.collect()
    return time.perf_counter() - t0


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="fork DataLoader workers")
def test_workers_leave_at_once_even_with_a_stuck_thread(monkeypatch) -> None:
    monkeypatch.delenv("GR00T_LANCE_WORKER_FAST_EXIT", raising=False)
    assert _shutdown_seconds(num_workers=3) < 3.0  # PyTorch would wait 5 s per worker

    # Control: without the hook the same dataset stalls for PyTorch's 5 s join timeout.
    monkeypatch.setenv("GR00T_LANCE_WORKER_FAST_EXIT", "0")
    assert _shutdown_seconds(num_workers=1) > 4.0


def test_fast_exit_is_a_no_op_outside_workers() -> None:
    from gr00t_lancedb._worker_exit import arm_worker_fast_exit

    assert arm_worker_fast_exit() is False
