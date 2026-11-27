# SPDX-License-Identifier: Apache-2.0
"""Plan items 1.4 / 2.2 / 2.3: GOP disk cache, I/O counters, startup + E4 metrics."""

from __future__ import annotations

import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import uuid

import numpy as np
import pytest
import torch

from gr00t_lancedb import EmbodimentTag, LanceEpisodeLoader, ModalityConfig, convert_v2_to_lance
from gr00t_lancedb.gop_cache import IO_STATS, GopDiskCache, _segments_for, aggregate_io_stats

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"_script_{name}", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# segments + disk cache
# ---------------------------------------------------------------------------


def test_segments_follow_gop_bounds_and_split_long_gops() -> None:
    bounds = [0, 100, 250, 2000, 2100]
    assert _segments_for(bounds, 120, 260, max_seg=10_000, grid=64) == [(100, 250), (250, 2000)]
    assert _segments_for(bounds, 0, 50, max_seg=10_000, grid=64) == [(0, 100)]
    long = _segments_for(bounds, 600, 1700, max_seg=500, grid=64)
    assert long == [(250, 750), (750, 1250), (1250, 1750)]
    assert _segments_for(bounds, 5, 5, max_seg=500, grid=64) == []


def _writer(root: str, budget: int, wid: int) -> None:
    cache = GopDiskCache(root, budget)
    rng = np.random.default_rng(wid)
    for i in range(60):
        start = int(rng.integers(0, 200)) * 4096
        size = int(rng.integers(1, 16)) * 1024
        data = bytes([(start + j) % 251 for j in range(size)])
        cache.put(f"ns{wid % 3}", start, start + size, data)
        got = cache.get(f"ns{(wid + 1) % 3}", start, start + size)
        if got is not None:
            assert got == bytes([(start + j) % 251 for j in range(size)])


def test_disk_cache_budget_under_concurrent_writers(tmp_path: Path) -> None:
    budget = 256 * 1024
    ctx = mp.get_context("fork")
    procs = [ctx.Process(target=_writer, args=(str(tmp_path), budget, w)) for w in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
        assert p.exitcode == 0
    cache = GopDiskCache(tmp_path, budget)
    # Each writer may overshoot by at most one in-flight segment (<16 KiB) before evicting.
    assert cache.total_bytes() <= budget + 4 * 16 * 1024
    for seg in tmp_path.glob("*/*.seg"):
        start, end = (int(x) for x in seg.stem.split("-"))
        assert seg.read_bytes() == bytes([(start + j) % 251 for j in range(end - start)])
    assert not list(tmp_path.glob("*/.*.tmp")), "no torn temp files left behind"


# ---------------------------------------------------------------------------
# cache on a real Lance dataset: bit-identical frames, epoch-2 hits
# ---------------------------------------------------------------------------


def _all_steps(loader: LanceEpisodeLoader) -> list:
    out = []
    for ep in range(len(loader)):
        n = loader.get_episode_length(ep)
        out.extend(loader.load_steps(ep, list(range(0, n, 3)), EmbodimentTag.NEW_EMBODIMENT, allow_padding=True))
    return out


def _assert_same(a: list, b: list) -> None:
    assert len(a) == len(b)
    for x, y in zip(a, b, strict=True):
        for cam in x.images:
            for fx, fy in zip(x.images[cam], y.images[cam], strict=True):
                np.testing.assert_array_equal(fx, fy)
        for k in x.actions:
            np.testing.assert_array_equal(x.actions[k], y.actions[k])


def test_gop_cache_bit_identical_and_second_epoch_served_locally(
    v2_dataset_dir: Path, default_modality_configs: dict[str, ModalityConfig], tmp_path: Path
) -> None:
    lance_dir = tmp_path / "lance"
    convert_v2_to_lance(v2_dataset_dir, lance_dir, video_mode="copy")
    cache_dir = tmp_path / "gop"

    plain = LanceEpisodeLoader(lance_dir, default_modality_configs)
    ref = _all_steps(plain)
    plain.close()

    s0 = IO_STATS.snapshot()
    cached = LanceEpisodeLoader(
        lance_dir, default_modality_configs, video_cache_dir=cache_dir, video_cache_gb=1.0,
        video_cache_force_local=True,
    )
    _assert_same(_all_steps(cached), ref)
    cached.close()
    s1 = IO_STATS.snapshot()
    assert s1["gop_cache_misses"] > s0["gop_cache_misses"]
    assert s1["lance_blob_calls"] > s0["lance_blob_calls"]
    assert any(cache_dir.rglob("*.seg"))

    # "Epoch 2" with a fresh reader (new process state): every byte comes from local disk.
    again = LanceEpisodeLoader(
        lance_dir, default_modality_configs, video_cache_dir=cache_dir, video_cache_gb=1.0,
        video_cache_force_local=True,
    )
    _assert_same(_all_steps(again), ref)
    again.close()
    s2 = IO_STATS.snapshot()
    assert s2["gop_cache_hits"] > s1["gop_cache_hits"]
    assert s2["gop_cache_misses"] == s1["gop_cache_misses"]
    assert s2["lance_blob_calls"] == s1["lance_blob_calls"], "no remote blob reads on epoch 2"
    assert s2["frames_take_calls"] > s1["frames_take_calls"]


def test_s3_cold_meta_isolated_and_gop_cache(
    v2_dataset_dir: Path,
    default_modality_configs: dict[str, ModalityConfig],
    moto_s3_server: dict,
    tmp_path: Path,
) -> None:
    from test_s3_streaming import _upload_dir_to_s3

    lance_dir = tmp_path / "lance"
    convert_v2_to_lance(v2_dataset_dir, lance_dir, video_mode="copy")
    prefix = f"gop-{uuid.uuid4().hex[:8]}"
    _upload_dir_to_s3(moto_s3_server["client"], moto_s3_server["bucket"], prefix, lance_dir)
    uri = f"s3://{moto_s3_server['bucket']}/{prefix}"
    opts = moto_s3_server["storage_options"]

    local = LanceEpisodeLoader(lance_dir, default_modality_configs)
    ref = local.load_steps(1, [0, 4, 9], EmbodimentTag.NEW_EMBODIMENT, allow_padding=True)
    local.close()

    cold = LanceEpisodeLoader(uri, default_modality_configs, storage_options=opts,
                              video_cache_dir=tmp_path / "gop", video_cache_gb=1.0)
    try:
        # Cold remote meta/ was materialized by an isolated child: this process never
        # touched Tokio, so fork stays allowed.
        assert cold.startup_info["isolated_calls"] == 1
        cold.assert_unopened()
        _assert_same(cold.load_steps(1, [0, 4, 9], EmbodimentTag.NEW_EMBODIMENT, allow_padding=True), ref)
        warm = LanceEpisodeLoader(uri, default_modality_configs, storage_options=opts,
                                  video_cache_dir=tmp_path / "gop", video_cache_gb=1.0)
        assert warm.startup_info["isolated_calls"] == 0  # meta cache now exists
        before = IO_STATS.snapshot()
        _assert_same(warm.load_steps(1, [0, 4, 9], EmbodimentTag.NEW_EMBODIMENT, allow_padding=True), ref)
        after = IO_STATS.snapshot()
        assert after["lance_blob_calls"] == before["lance_blob_calls"]
        assert after["gop_cache_hits"] > before["gop_cache_hits"]
        warm.close()
    finally:
        cold.close()
        shutil.rmtree(cold.local_meta_root, ignore_errors=True)


def test_io_stats_flush_and_aggregate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GR00T_LANCE_IO_STATS_DIR", str(tmp_path))
    IO_STATS.add(lance_blob_ranges=3)
    IO_STATS.maybe_flush(force=True)
    (tmp_path / "io-999999.json").write_text(json.dumps({"lance_blob_ranges": 4, "gop_cache_hits": 2}))
    agg = aggregate_io_stats(tmp_path)
    assert agg["lance_blob_ranges"] == IO_STATS.snapshot()["lance_blob_ranges"] + 4
    assert agg["gop_cache_hits"] >= 2


# ---------------------------------------------------------------------------
# launcher helpers, summarize_runs, analyze_e4_samples
# ---------------------------------------------------------------------------


def test_sample_fingerprints_track_samples_not_positions() -> None:
    launch = _load_script("launch_finetune_lance")
    g = torch.Generator().manual_seed(0)
    batch = {"inputs": {"x": torch.randn(4, 3, 8, generator=g), "a": torch.randn(4, 16, generator=g).bfloat16()},
             "scalar": torch.tensor(1.0)}
    fps = launch._sample_fingerprints(batch)
    assert len(fps) == 4 and len(set(fps)) == 4
    perm = [2, 0, 3, 1]
    shuffled = {"inputs": {k: v[perm] for k, v in batch["inputs"].items()}, "scalar": batch["scalar"]}
    assert launch._sample_fingerprints(shuffled) == [fps[i] for i in perm]


def test_lance_pins_roundtrip(tmp_path: Path) -> None:
    launch = _load_script("launch_finetune_lance")
    assert launch._load_lance_pins(tmp_path) == {}
    launch._save_lance_pins(tmp_path, {"s3://b/ds": 7})
    assert launch._load_lance_pins(tmp_path) == {"s3://b/ds": 7}


def _steps(seq: list[list[str]], start: int = 1) -> list[dict]:
    return [{"event": "step", "global_step": start + i, "sample_fps": s, "batch_fp": "|".join(s),
             "step_s": 0.5, "data_wait_s": 0.01, "since_launch_s": 10 + i} for i, s in enumerate(seq)]


def test_analyze_e4_exact_vs_reseeded_resume() -> None:
    e4 = _load_script("analyze_e4_samples")
    stream = [[f"s{4 * i + j}" for j in range(4)] for i in range(10)]
    ref = [{"event": "train_begin"}, *_steps(stream)]
    # Exact resume: preempted after step 6 with checkpoint at 4 -> steps 5,6 replayed identically.
    exact = [{"event": "train_begin"}, *_steps(stream[:6]), {"event": "train_begin"}, *_steps(stream[4:], start=5)]
    r = e4.compare(ref, exact)
    assert r["level"] == "sample" and r["order_match"] == 1.0 and r["dup_rate"] == 0 and r["miss_rate"] == 0
    assert r["test_replayed_steps"] == 2 and r["test_launches"] == 2 and r["first_divergence"] is None
    # Reseeded resume (native GR00T): a new order after step 4 -> repeats and misses.
    reseeded = stream[:4] + [[f"s{x}" for x in (0, 1, 2, 3)], *[[f"s{4 * i + j}" for j in range(4)] for i in range(6, 11)]]
    r2 = e4.compare(ref, [{"event": "train_begin"}, *_steps(reseeded)])
    assert r2["first_divergence"] == 5 and r2["dup_rate"] > 0 and r2["miss_rate"] > 0 and r2["order_match"] < 1


def test_summarize_io_and_startup_segments() -> None:
    summ = _load_script("summarize_runs")
    recs = [{"event": "train_begin", "since_launch_s": 30.0},
            {"event": "startup_summary", "since_launch_s": 40.0, "imports_s": 5.0, "model_load_s": 20.0,
             "dataset_build_s": 1.5, "first_batch_wait_s": 3.0}]
    for gs in range(1, 31):
        recs.append({"event": "step", "global_step": gs, "step_s": 0.6, "data_wait_s": 0.01, "since_launch_s": 40 + gs,
                     "io": {"requested_ranges": 10 * gs, "lance_blob_ranges": 4 * gs, "lance_blob_calls": gs,
                            "lance_blob_bytes": gs << 20, "frames_take_calls": gs,
                            "gop_cache_hits": 3 * gs, "gop_cache_misses": gs}})
    out = summ.summarize(recs, warmup=5)
    assert out["startup.model_load_s"] == 20.0
    assert out["io.lance_ranges_per_step"] == 4.0
    assert out["io.lance_mb_per_step"] == 1.0
    assert out["io.gop_cache_hit_rate"] == 0.75
    assert out["io.est_get_usd_per_1k_steps"] == pytest.approx(5 * 0.0004)


def test_bench_callback_logs_emergency_save_once(tmp_path: Path) -> None:
    from types import SimpleNamespace

    launch = _load_script("launch_finetune_lance")
    metrics = tmp_path / "bench_metrics.jsonl"
    cb = launch.make_bench_callback(metrics, 0.0, {}, {}, start_gpu_sampler=False)
    cb.preempt_cb = SimpleNamespace(last_decision=None, saved_light_ckpt=None)
    args, control = SimpleNamespace(), SimpleNamespace(should_save=False)
    state = SimpleNamespace(global_step=0)
    cb.on_train_begin(args, state, control)
    for gs in (1, 2):
        state.global_step = gs
        cb.on_step_begin(args, state, control)
        cb.on_step_end(args, state, control)
    cb.preempt_cb.last_decision = {"action": "save", "dtype": "bf16", "est_save_s": 12.0}
    cb.preempt_cb.saved_light_ckpt = tmp_path / "checkpoint-3"
    for gs in (3, 3):  # a second callback round must not duplicate the record
        state.global_step = gs
        cb.on_step_begin(args, state, control)
        cb.on_step_end(args, state, control)
    recs = [json.loads(ln) for ln in metrics.read_text().splitlines()]
    saves = [r for r in recs if r["event"] == "save"]
    assert len(saves) == 1 and saves[0]["emergency"] is True and saves[0]["global_step"] == 3
    assert saves[0]["decision"]["dtype"] == "bf16" and saves[0]["checkpoint"].endswith("checkpoint-3")


def test_bench_loader_io_epoch2_served_from_cache(v2_dataset_dir: Path, tmp_path: Path, monkeypatch) -> None:
    """Data-only E2 tool: per-epoch I/O deltas, worker counters flushed at worker exit."""
    lance_dir = tmp_path / "lance"
    convert_v2_to_lance(v2_dataset_dir, lance_dir, video_mode="copy")
    bench = _load_script("bench_loader_io")
    out = tmp_path / "io.json"
    rep = bench.main([
        "--dataset", str(lance_dir), "--epochs", "2", "--batch-size", "4", "--num-workers", "2",
        "--steady-decode-batch", "8", "--video-cache-gb", "1", "--video-cache-dir", str(tmp_path / "gop"),
        "--video-cache-force-local", "--action-horizon", "4", "--out", str(out),
    ])
    e1, e2 = rep["epochs"]
    assert rep["access_pattern"] in ("random_anchors_fallback", "gr00t_fast_sharded_mixture")
    assert e1["samples"] == e2["samples"] == rep["anchors"]
    assert e1["lance_ranges_per_sample"] > 0 and e1["gop_cache_hit_rate"] < 1.0
    assert e2["lance_ranges_per_sample"] == 0 and e2["gop_cache_hit_rate"] == 1.0
    assert e2["frames_takes_per_sample"] > 0
    assert json.loads(out.read_text())["epochs"][1]["epoch"] == 1
