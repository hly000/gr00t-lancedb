# SPDX-License-Identifier: Apache-2.0
"""Weighted anchor sampling (flywheel round 2): determinism, exact resume, distribution."""

from __future__ import annotations

from itertools import islice
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gr00t_lancedb.flywheel.checkpoints import KeepCheckpointsCallback, keep_checkpoint, parse_steps
from gr00t_lancedb.flywheel.weights import (
    AnchorWeightMap,
    EpisodeAnchors,
    build_weighted_anchors,
    map_frame_weights,
)

from test_exact_resume import _ExactFake, _datasets, _stream


def _anchor_maps(datasets, *, hot_ep: int, zero_ep: int) -> list[AnchorWeightMap]:
    """Every (ep, step) of the fake datasets; ``hot_ep`` x8 weight, ``zero_ep`` never drawn."""
    maps = []
    for ds in datasets:
        steps_by_ep: dict[int, set[int]] = {}
        for shard in ds.shards:
            for ep, steps in shard:
                steps_by_ep.setdefault(int(ep), set()).update(int(s) for s in steps)
        eps = sorted(steps_by_ep)
        ep_arr = np.concatenate([np.full(len(steps_by_ep[e]), e) for e in eps]).astype(np.int64)
        st_arr = np.concatenate([sorted(steps_by_ep[e]) for e in eps]).astype(np.int64)
        w = np.ones(len(ep_arr))
        w[ep_arr == hot_ep] = 8.0
        w[ep_arr == zero_ep] = 0.0
        maps.append(AnchorWeightMap(ep_arr, st_arr, w))
    return maps


def _weighted_ds(per_sample_seed: bool = True):
    dss = _datasets()
    wa = build_weighted_anchors(_anchor_maps(dss, hot_ep=1, zero_ep=3), [1.0, 1.0], column="w")
    return _ExactFake(dss, per_sample_seed=per_sample_seed, weighted_anchors=wa, decode_batch_size=7), wa


@pytest.mark.parametrize("nw", [0, 2, 3])
def test_weighted_stream_is_deterministic(nw):
    a = _stream(_weighted_ds()[0], 25, 6, nw)
    b = _stream(_weighted_ds()[0], 25, 6, nw)
    assert a == b


@pytest.mark.parametrize("nw", [0, 2, 3])
@pytest.mark.parametrize("resume_at", [1, 9, 20])
def test_weighted_resume_is_exact(nw, resume_at):
    full = _stream(_weighted_ds()[0], 30, 6, nw)
    ds, _ = _weighted_ds()
    ds.set_resume_point(resume_at, 6)
    assert _stream(ds, 30 - resume_at, 6, nw) == full[resume_at:]


def test_fork_from_uniform_checkpoint():
    """Round 2 resumes from a *uniform* run's step count: same cursor, weighted stream."""
    ds, _ = _weighted_ds()
    ds.set_resume_point(12, 6)
    fork = _stream(ds, 10, 6, 2)
    ref = _stream(_weighted_ds()[0], 22, 6, 2)[12:]
    assert fork == ref
    uniform = _stream(_ExactFake(_datasets()), 22, 6, 2)[12:]
    assert [[it[:2] for it in b] for b in fork] != [[it[:2] for it in b] for b in uniform]


def test_weighted_distribution_follows_weights():
    ds, wa = _weighted_ds(per_sample_seed=False)
    batches = _stream(ds, 400, 25, 0)
    items = [it for b in batches for it in b]
    # Dataset tag is not in the items; check per-episode frequencies against the CDF.
    p = wa.probabilities()
    expected_hot = p[wa.ep_pos == 1].sum()
    observed_hot = np.mean([it[0] == 1 for it in items])
    assert abs(observed_hot - expected_hot) < 0.03
    assert all(it[0] != 3 for it in items)  # zero-weight episode never drawn
    s = wa.summary()
    assert s["zero_weight_anchors"] == int((wa.ep_pos == 3).sum())
    assert s["max_over_uniform"] > 1.0


def test_stream_independent_of_block_size():
    _, wa = _weighted_ds()
    a = list(islice(wa.stream(1, 0, 2, block=7), 5000))
    b = list(islice(wa.stream(1, 0, 2, block=4096), 5000))
    assert a == b


def test_map_frame_weights_lookup():
    eps = [EpisodeAnchors(0, 10, np.array([0, 1, 2])), EpisodeAnchors(1, 11, np.array([5, 6]))]
    m = map_frame_weights(
        eps,
        episode_index=np.array([10, 10, 11, 11]),
        frame_index=np.array([2, 0, 5, 6]),
        weight=np.array([3.0, 2.0, np.nan, 0.5]),
        default=1.0,
    )
    assert m.ep_pos.tolist() == [0, 0, 0, 1, 1]
    assert m.step.tolist() == [0, 1, 2, 5, 6]
    assert m.weight.tolist() == [2.0, 1.0, 3.0, 1.0, 0.5]
    assert (m.missing, m.nonfinite) == (1, 1)
    with pytest.raises(ValueError):
        map_frame_weights(eps, np.array([10]), np.array([0]), np.array([-1.0]))


def test_build_weighted_anchors_mixture_share():
    a = AnchorWeightMap(np.zeros(4, np.int64), np.arange(4), np.array([1.0, 1.0, 1.0, 5.0]))
    b = AnchorWeightMap(np.zeros(2, np.int64), np.arange(2), np.array([1.0, 1.0]))
    wa = build_weighted_anchors([a, b], [3.0, 1.0])
    p = wa.probabilities()
    assert abs(p[wa.ds_index == 0].sum() - 0.75) < 1e-12
    assert abs(p[3] / p[0] - 5.0) < 1e-9
    with pytest.raises(ValueError):
        build_weighted_anchors([AnchorWeightMap(np.zeros(1, np.int64), np.zeros(1, np.int64), np.zeros(1))], [1.0])


def test_parse_steps():
    assert parse_steps("5000, 10000,,14000") == {5000, 10000, 14000}
    assert parse_steps(None) == set()
    with pytest.raises(ValueError):
        parse_steps("0")


def _fake_checkpoint(root: Path, step: int) -> Path:
    ck = root / f"checkpoint-{step}"
    ck.mkdir(parents=True)
    for name in ("model.safetensors", "config.json", "trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (ck / name).write_bytes(name.encode() * 10)
    return ck


def test_keep_checkpoint_weights_only_and_hardlinks(tmp_path: Path):
    src = _fake_checkpoint(tmp_path / "run", 5000)
    man = keep_checkpoint(src, tmp_path / "run" / "keep" / "checkpoint-5000", weights_only=True)
    kept = sorted(p.name for p in (tmp_path / "run" / "keep" / "checkpoint-5000").iterdir())
    assert kept == ["config.json", "keep_manifest.json", "model.safetensors", "trainer_state.json"]
    assert man["skipped_training_state_files"] == 3
    assert (src / "model.safetensors").stat().st_nlink == 2  # hard link, no extra bytes


def test_keep_callback_on_save(tmp_path: Path, monkeypatch):
    out = tmp_path / "run"
    _fake_checkpoint(out, 14000)
    _fake_checkpoint(out, 15000)
    monkeypatch.setenv("GR00T_KEEP_CKPT_STEPS", "14000")
    monkeypatch.setenv("GR00T_KEEP_CKPT_WEIGHTS_ONLY", "20000")
    cb = KeepCheckpointsCallback.from_env()
    args = SimpleNamespace(output_dir=str(out))
    for step in (14000, 15000):
        cb.on_save(args, SimpleNamespace(global_step=step, is_world_process_zero=True), SimpleNamespace())
    assert (out / "keep" / "checkpoint-14000" / "optimizer.pt").exists()  # fork point keeps full state
    assert not (out / "keep" / "checkpoint-15000").exists()
    assert json.loads((out / "keep" / "checkpoint-14000" / "keep_manifest.json").read_text())["weights_only"] is False
    monkeypatch.delenv("GR00T_KEEP_CKPT_STEPS")
    monkeypatch.delenv("GR00T_KEEP_CKPT_WEIGHTS_ONLY")
    assert KeepCheckpointsCallback.from_env() is None
