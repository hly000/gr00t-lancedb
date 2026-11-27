# SPDX-License-Identifier: Apache-2.0
"""Failure mining end to end on a real (synthetic) Lance dataset, CPU only.

A "failed rollout" replays one training episode's exact state/action trajectory for the first
``DIVERGE_AT`` steps and then drifts away. Mining must place the divergence point there, boost
the frames of that episode around it, leave ``videos.lance`` untouched, and the weight column
must turn into a sampling distribution for ``LanceShardedSingleStepDataset`` anchors.
"""

from __future__ import annotations

from pathlib import Path

import lancedb
import numpy as np
import pyarrow.parquet as pq
import pytest

from gr00t_lancedb import EmbodimentTag, LanceShardedSingleStepDataset, convert_v2_to_lance
from gr00t_lancedb.flywheel.mining import (
    MiningConfig,
    compose_weights,
    embed_rollout,
    find_divergence,
    load_frame_index,
    run_mining,
)
from gr00t_lancedb.flywheel.rollouts import RolloutEpisode, append_summary, episode_path, read_summary
from gr00t_lancedb.flywheel.weights import build_weighted_anchors, dataset_episode_anchors, map_frame_weights
from gr00t_lancedb.lance_query import query_anchor_weights
from gr00t_lancedb.vector_curate import KinematicEmbeddingProvider, _fsl_to_float32, add_embedding_column
from lerobot_lancedb import FRAMES_TABLE

from conftest import create_synthetic_gr00t_v2_dataset

EP_LEN = 60
NUM_EPS = 6
HORIZON = 4
SRC_EP = 2  # task 0 ("pick up the red block")
DIVERGE_AT = 30
TASKS = {0: "pick up the red block", 1: "place the block in the bowl"}


def _episode_arrays(v2: Path, ep: int) -> tuple[np.ndarray, np.ndarray]:
    tbl = pq.read_table(v2 / "data" / "chunk-000" / f"episode_{ep:06d}.parquet")
    return (
        np.asarray(tbl.column("observation.state").to_pylist(), dtype=np.float64),
        np.asarray(tbl.column("action").to_pylist(), dtype=np.float64),
    )


def _rollout(state: np.ndarray, action: np.ndarray, *, task_idx: int, ordinal: int, success: bool) -> RolloutEpisode:
    """One decision per env step, executing one action: rollout row k == trajectory row k."""
    k = len(state)
    return RolloutEpisode(
        task=f"sim/task_{task_idx}",
        ordinal=ordinal,
        seed=100 + ordinal,
        success=success,
        env_steps=np.arange(k, dtype=np.int64),
        n_env_steps=np.ones(k, dtype=np.int64),
        state={"arm": state[:, :7], "gripper": state[:, 7:8]},
        action={"arm": action[:, None, :7], "gripper": action[:, None, 7:8]},
        task_description=TASKS[task_idx],
    )


@pytest.fixture(scope="module")
def v2_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return create_synthetic_gr00t_v2_dataset(tmp_path_factory.mktemp("fw_v2"), ep_lengths=[EP_LEN] * NUM_EPS, seed=11)


@pytest.fixture()
def lance_dir(v2_dir: Path, tmp_path: Path) -> Path:
    out = tmp_path / "lance"
    convert_v2_to_lance(v2_dir, out, video_mode="copy")
    add_embedding_column(out, KinematicEmbeddingProvider(horizon=HORIZON, dim=16, seed=3))
    return out


@pytest.fixture()
def rollouts(v2_dir: Path, tmp_path: Path) -> Path:
    d = tmp_path / "rollouts"
    rng = np.random.default_rng(0)
    s, a = _episode_arrays(v2_dir, SRC_EP)
    # Failure: follow the demo, then drift far away (outside every episode's state range).
    fs, fa = s.copy(), a.copy()
    drift = np.cumsum(rng.normal(0.0, 1.5, size=(EP_LEN - DIVERGE_AT, 8)), axis=0) + 6.0
    fs[DIVERGE_AT:] = s[DIVERGE_AT] + drift
    fa[DIVERGE_AT:] = rng.normal(0.0, 3.0, size=(EP_LEN - DIVERGE_AT, 8))
    # A success that is ignored, and a failure entirely outside the data (coverage gap).
    os_, oa = s + 25.0, a + 25.0
    for ep in (
        _rollout(fs, fa, task_idx=0, ordinal=0, success=False),
        _rollout(s, a, task_idx=0, ordinal=1, success=True),
        _rollout(os_, oa, task_idx=0, ordinal=2, success=False),
    ):
        ep.save(episode_path(d, ep.task, ep.ordinal))
        append_summary(d, ep.summary())
    return d


def test_rollout_embedding_matches_stored_column(lance_dir: Path, v2_dir: Path) -> None:
    """A rollout that *is* a training episode embeds onto that episode's stored vectors."""
    fi = load_frame_index(lance_dir)
    s, a = _episode_arrays(v2_dir, SRC_EP)
    q = embed_rollout(fi, _rollout(s, a, task_idx=0, ordinal=0, success=False))
    rows = np.flatnonzero(fi.episode_index == SRC_EP)
    rows = rows[np.argsort(fi.frame_index[rows])]
    np.testing.assert_allclose(q, fi.embedding[rows], atol=2e-5)


def test_rollout_roundtrip(tmp_path: Path) -> None:
    ep = _rollout(np.zeros((5, 8)), np.ones((5, 8)), task_idx=1, ordinal=3, success=True)
    ep.image = np.zeros((5, 4, 4, 3), np.uint8)
    p = ep.save(episode_path(tmp_path, ep.task, ep.ordinal))
    back = RolloutEpisode.load(p)
    assert back.summary() == ep.summary()
    assert back.image.shape == (5, 4, 4, 3)
    np.testing.assert_array_equal(back.executed_actions("gripper"), np.ones((5, 1)))


def test_find_divergence() -> None:
    sims = np.r_[np.full(10, 0.95), np.linspace(0.95, 0.4, 6), np.full(5, 0.4)]
    k, method = find_divergence(sims, drop=0.15, smooth=1)
    assert method == "drop" and 10 <= k <= 13
    k, method = find_divergence(np.full(8, 0.9) - np.r_[0, 0, 0, 0.05, 0, 0, 0, 0], drop=0.15, smooth=1)
    assert method == "min" and k == 3


def test_compose_weights_bounds() -> None:
    cfg = MiningConfig(alpha=10.0, w_min=0.25, w_max=4.0)
    score = np.zeros(1000)
    score[:5] = 1.0
    keep = np.ones(1000, bool)
    keep[-10:] = False
    w = compose_weights(score, keep, cfg)
    assert np.all(w[~keep] == 0)
    assert abs(w[keep].mean() - 1.0) < 1e-9
    assert w[keep].max() <= 4.0 + 1e-6 and w[keep].min() >= 0.25 - 1e-6
    assert w[0] > w[100]


def test_mining_end_to_end(lance_dir: Path, rollouts: Path) -> None:
    assert [r["success"] for r in read_summary(rollouts)] == [False, True, False]
    db = lancedb.connect(str(lance_dir))
    v_before = int(db.open_table(FRAMES_TABLE).version)

    cfg = MiningConfig(min_match_sim=0.9, divergence_drop=0.15, smooth=3, spread_frames=4)
    rep = run_mining(lance_dir, [rollouts], cfg)

    eps = {e["ordinal"]: e for e in rep["episodes"]}
    assert set(eps) == {0, 2}  # the success is not mined
    assert eps[0]["method"] == "drop"
    # The embedding at step t covers actions t..t+HORIZON-1, so it sees the drift HORIZON-1 early.
    assert DIVERGE_AT - HORIZON + 1 <= eps[0]["divergence_env_step"] <= DIVERGE_AT + 2
    assert not eps[0]["coverage_gap"]
    assert eps[2]["coverage_gap"] and eps[2]["matched_rows"] == 0
    assert rep["coverage_gaps"] == 1

    w = rep["write"]
    assert w["frames_version"] > v_before == w["frames_version_before"]
    assert w["bytes_written_videos"] == 0 and w["files_written_videos"] == 0
    assert w["bytes_written_frames"] > 0

    tbl = db.open_table(FRAMES_TABLE).search().select(
        ["episode_index", "frame_index", cfg.score_column, cfg.weight_column]
    ).limit(None).to_arrow()
    ep = np.asarray(tbl.column("episode_index").to_pylist())
    fr = np.asarray(tbl.column("frame_index").to_pylist())
    score = np.asarray(tbl.column(cfg.score_column).to_pylist())
    weight = np.asarray(tbl.column(cfg.weight_column).to_pylist())
    boosted = score > 0
    assert boosted.any()
    # Boosted frames are the source episode around the divergence point, nothing else.
    assert set(ep[boosted]) == {SRC_EP}
    lo, hi = fr[boosted].min(), fr[boosted].max()
    assert DIVERGE_AT - 12 <= lo and hi <= DIVERGE_AT + 12
    assert abs(weight.mean() - 1.0) < 1e-5
    assert weight[boosted].min() > weight[~boosted].max()
    assert rep["boosted_sample_share"] > rep["boosted_uniform_share"]


def test_weight_column_drives_anchor_sampling(lance_dir: Path, rollouts: Path, default_modality_configs) -> None:
    cfg = MiningConfig(min_match_sim=0.9, spread_frames=4)
    rep = run_mining(lance_dir, [rollouts], cfg)
    res = query_anchor_weights(dataset_path=str(lance_dir), column=cfg.weight_column, lance_version=rep["write"]["frames_version"])
    assert res["field_metadata"]  # provenance travels with the column

    ds = LanceShardedSingleStepDataset(
        dataset_path=lance_dir,
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        modality_configs=default_modality_configs,
    )
    m = map_frame_weights(dataset_episode_anchors(ds), res["episode_index"], res["frame_index"], res["weight"])
    assert m.missing == 0 and m.nonfinite == 0
    wa = build_weighted_anchors([m], [1.0], column=cfg.weight_column)
    pos = {int(ds.episode_loader.episodes_metadata[i]["episode_index"]): i for i in range(len(ds.episode_loader))}
    p = wa.probabilities()
    centre = DIVERGE_AT - HORIZON  # just before the (action-window-shifted) divergence point
    hot = (wa.ep_pos == pos[SRC_EP]) & (np.abs(wa.step - centre) <= 2)
    assert hot.any() and p[hot].min() > np.median(p)
    draws = [next(it) for it in [wa.stream(42, 0, 0)] for _ in range(20000)]
    hot_draws = sum((e == pos[SRC_EP] and abs(s - centre) <= 2) for _, e, s in draws)
    assert hot_draws / len(draws) > hot.sum() / len(wa)  # over-sampled vs. uniform
    ds.close()


def test_stored_embedding_is_unit_norm(lance_dir: Path) -> None:
    tbl = lancedb.connect(str(lance_dir)).open_table(FRAMES_TABLE).search().select(["embedding"]).limit(None).to_arrow()
    vals, ok = _fsl_to_float32(tbl.column("embedding"))
    assert ok.all()
    np.testing.assert_allclose(np.linalg.norm(vals, axis=1), 1.0, atol=1e-5)


def test_launcher_loads_weighted_anchors(lance_dir: Path, rollouts: Path, default_modality_configs, monkeypatch) -> None:
    """``GR00T_LANCE_WEIGHT_COLUMN`` path of the launcher: isolated query -> CDF over anchors."""
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts" / "launch_finetune_lance.py"
    spec = importlib.util.spec_from_file_location("_fw_launch", script)
    launch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launch)

    cfg = MiningConfig(min_match_sim=0.9, spread_frames=4)
    rep = run_mining(lance_dir, [rollouts], cfg)
    monkeypatch.setenv("GR00T_LANCE_WEIGHT_VERSION", str(rep["write"]["frames_version"]))
    ds = LanceShardedSingleStepDataset(
        dataset_path=lance_dir,
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        modality_configs=default_modality_configs,
    )
    wa = launch._load_weighted_anchors([ds], [1.0], cfg.weight_column)
    s = wa.summary()
    assert s["anchors"] == sum(len(ds.get_valid_steps_for_episode(e)) for e in range(len(ds.episode_loader)))
    assert s["sources"][0]["weight_version"] == rep["write"]["frames_version"]
    assert s["sources"][0]["missing"] == 0
    assert s["max_over_uniform"] > 1.5
    assert not ds.is_open  # the training process itself never opened Lance
    ds.close()
