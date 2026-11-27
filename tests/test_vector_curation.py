# SPDX-License-Identifier: Apache-2.0
"""Embedding column + IVF-PQ index + semantic dedup + hybrid search on frames.lance (issue #4)."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import lancedb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# Import gr00t_lancedb before lerobot_lancedb so _hub_compat shims huggingface_hub<1.0 cleanly.
from gr00t_lancedb import (
    EmbodimentTag,
    LanceShardedSingleStepDataset,
    ModalityConfig,
    convert_v2_to_lance,
    curate_lance_dataset,
)
from gr00t_lancedb.vector_curate import (
    EMBEDDING_FIELD_METADATA_KEY,
    SEMANTIC_DUP_COLUMNS,
    FunctionEmbeddingProvider,
    KinematicEmbeddingProvider,
    VisualEmbeddingProvider,
    add_embedding_column,
    anchors_to_where,
    build_vector_index,
    episodes_to_where,
    greedy_duplicate_assignment,
    hybrid_search,
    main_vector_curate,
    mine_rare_frames,
    slice_episodes_by_similarity,
    suggest_ivf_pq_params,
    tag_semantic_duplicates,
)
from lerobot_lancedb import FRAMES_TABLE, LanceDBDataset

from conftest import create_synthetic_gr00t_v2_dataset

EP_LEN = 60
NUM_EPS = 10
DUP_SRC_EP, DUP_EP = 2, 7  # episode 7 replays episode 2's state/action trajectory (+1e-3 noise)
STATIC_EP, STATIC_START, STATIC_END = 5, 10, 30  # episode 5 holds still for frames 10..30
HORIZON = 4  # kinematic embedding action chunk; equals the training action horizon below
THRESHOLD = 0.002  # injected dups are ~1e-6 apart; the closest *distinct* pair here is ~6e-3
# Frames 11..27 of episode 5 have identical (state, 4-step action chunk) to frame 10; frame 28's
# chunk already contains the first post-static action.
EXPECTED_STATIC_DUPS = set(range(STATIC_START + 1, STATIC_END - HORIZON + 2))


def _rewrite_episode(v2: Path, ep: int, fn) -> None:
    path = v2 / "data" / "chunk-000" / f"episode_{ep:06d}.parquet"
    tbl = pq.read_table(path)
    state = np.asarray(tbl.column("observation.state").to_pylist(), dtype=np.float32)
    action = np.asarray(tbl.column("action").to_pylist(), dtype=np.float32)
    state, action = fn(state, action)
    cols = {c: tbl.column(c) for c in tbl.column_names}
    cols["observation.state"] = pa.array([r.tolist() for r in state], type=pa.list_(pa.float32()))
    cols["action"] = pa.array([r.tolist() for r in action], type=pa.list_(pa.float32()))
    pq.write_table(pa.table(cols), path)


def _episode_arrays(v2: Path, ep: int) -> tuple[np.ndarray, np.ndarray]:
    tbl = pq.read_table(v2 / "data" / "chunk-000" / f"episode_{ep:06d}.parquet")
    return (
        np.asarray(tbl.column("observation.state").to_pylist(), dtype=np.float32),
        np.asarray(tbl.column("action").to_pylist(), dtype=np.float32),
    )


@pytest.fixture(scope="module")
def dup_v2_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """10 x 60-frame episodes with an injected near-duplicate episode and a static segment."""
    v2 = create_synthetic_gr00t_v2_dataset(
        tmp_path_factory.mktemp("vec_v2"), ep_lengths=[EP_LEN] * NUM_EPS, seed=7
    )
    src_state, src_action = _episode_arrays(v2, DUP_SRC_EP)
    rng = np.random.default_rng(0)
    _rewrite_episode(
        v2,
        DUP_EP,
        lambda s, a: (
            src_state + rng.normal(0.0, 1e-3, src_state.shape).astype(np.float32),
            src_action + rng.normal(0.0, 1e-3, src_action.shape).astype(np.float32),
        ),
    )

    def _static(s: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s, a = s.copy(), a.copy()
        s[STATIC_START : STATIC_END + 1] = s[STATIC_START]
        a[STATIC_START : STATIC_END + 1] = a[STATIC_START]
        return s, a

    _rewrite_episode(v2, STATIC_EP, _static)
    return v2


@pytest.fixture()
def lance_dir(dup_v2_dir: Path, tmp_path: Path) -> Path:
    out = tmp_path / "lance_ds"
    convert_v2_to_lance(dup_v2_dir, out, video_mode="copy")
    return out


def _snapshot_tree(root: Path) -> dict[str, tuple[int, int]]:
    return {
        f.relative_to(root).as_posix(): (f.stat().st_size, f.stat().st_mtime_ns)
        for f in sorted(root.rglob("*"))
        if f.is_file()
    }


def _frames(lance: Path, cols: list[str]) -> dict[str, np.ndarray]:
    tbl = lancedb.connect(str(lance)).open_table(FRAMES_TABLE).search().select(cols).limit(None).to_arrow()
    return {c: tbl.column(c).to_pylist() for c in cols}


def _dup_set(lance: Path) -> set[tuple[int, int]]:
    rows = _frames(lance, ["episode_index", "frame_index", "is_semantic_dup"])
    return {
        (int(e), int(f))
        for e, f, d in zip(rows["episode_index"], rows["frame_index"], rows["is_semantic_dup"], strict=True)
        if d
    }


def _expected_dups() -> set[tuple[int, int]]:
    return {(DUP_EP, f) for f in range(EP_LEN)} | {(STATIC_EP, f) for f in EXPECTED_STATIC_DUPS}


def _modality_configs(action_horizon: int = HORIZON) -> dict[str, ModalityConfig]:
    return {
        "video": ModalityConfig(delta_indices=[-1, 0], modality_keys=["front", "wrist"]),
        "state": ModalityConfig(delta_indices=[0], modality_keys=["arm", "gripper"]),
        "action": ModalityConfig(delta_indices=list(range(action_horizon)), modality_keys=["arm", "gripper"]),
        "language": ModalityConfig(delta_indices=[0], modality_keys=["task"]),
    }


def _sharded(lance: Path, where: str | None) -> LanceShardedSingleStepDataset:
    return LanceShardedSingleStepDataset(
        dataset_path=lance,
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        modality_configs=_modality_configs(),
        shard_size=64,
        episode_sampling_rate=0.5,
        allow_padding=True,
        where=where,
    )


def _embed_index_dedup(lance: Path, **dedup_kw) -> tuple:
    emb = add_embedding_column(lance, KinematicEmbeddingProvider(horizon=HORIZON, dim=32, seed=0))
    idx = build_vector_index(lance)
    kw = {"threshold": THRESHOLD, "scope": "all", "query_batch_size": 64, **dedup_kw}
    dd = tag_semantic_duplicates(lance, **kw)
    return emb, idx, dd


# ---------------------------------------------------------------------------------------------
def test_greedy_assignment_keeps_one_representative_without_chaining() -> None:
    # a-b and b-c are near-duplicates but a-c is not: b is a dup of a, c stays (no chaining).
    is_dup, rep = greedy_duplicate_assignment(4, np.array([0, 1]), np.array([1, 2]), np.array([0, 1, 2, 3]))
    assert is_dup.tolist() == [False, True, False, False]
    assert rep.tolist() == [0, 0, 2, 3]
    # visiting order decides the representative deterministically
    is_dup2, rep2 = greedy_duplicate_assignment(3, np.array([0]), np.array([1]), np.array([1, 0, 2]))
    assert is_dup2.tolist() == [True, False, False] and rep2.tolist() == [1, 1, 2]
    assert suggest_ivf_pq_params(600, 32) == (2, 4)
    assert suggest_ivf_pq_params(1_000_000, 768) == (1000, 48)
    assert episodes_to_where([]) == "1 = 0"
    assert episodes_to_where([3, 1, 3]) == "episode_index IN (1, 3)"
    assert anchors_to_where({1: [4, 2]}) == "(episode_index = 1 AND frame_index IN (2, 4))"


def test_embed_index_dedup_zero_video_bytes_and_idempotent(lance_dir: Path) -> None:
    videos_dir = lance_dir / "videos.lance"
    videos_before = _snapshot_tree(videos_dir)
    assert videos_before

    emb, idx, dd = _embed_index_dedup(lance_dir)

    # -- embedding column: FixedSizeList<float32, 32>, provenance in field metadata, 0 video bytes
    assert emb.embedded_frames == NUM_EPS * EP_LEN and emb.null_frames == 0 and emb.dim == 32
    assert emb.bytes_measured and emb.bytes_written_frames > 0
    assert emb.bytes_written_videos == 0 and emb.files_written_videos == 0 and emb.bytes_written_meta == 0
    assert emb.frames_table_version > emb.frames_table_version_before
    table = lancedb.connect(str(lance_dir)).open_table(FRAMES_TABLE)
    fld = table.schema.field("embedding")
    assert fld.type == pa.list_(pa.float32(), 32)
    meta = json.loads(fld.metadata[EMBEDDING_FIELD_METADATA_KEY])
    assert meta["provider"] == "kinematic" and meta["metric"] == "cosine" and meta["config"]["horizon"] == HORIZON

    rows = _frames(lance_dir, ["episode_index", "frame_index", "index", "embedding", *SEMANTIC_DUP_COLUMNS])
    e = np.asarray(rows["embedding"], dtype=np.float32)
    np.testing.assert_allclose(np.linalg.norm(e, axis=1), 1.0, atol=1e-5)
    key = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(rows["episode_index"], rows["frame_index"], strict=True))}
    for f in (0, 30, 59):
        assert 1.0 - float(e[key[(DUP_EP, f)]] @ e[key[(DUP_SRC_EP, f)]]) < 1e-4

    # -- vector index: real IVF-PQ (cosine) on 600 vectors
    assert idx.search_mode == "ivf_pq" and idx.index_type.upper().replace("_", "") == "IVFPQ"
    assert idx.num_indexed_rows == NUM_EPS * EP_LEN and idx.fallback_reason is None
    assert (idx.num_partitions, idx.num_sub_vectors) == (2, 4)
    assert idx.bytes_written_videos == 0

    # -- dedup: exactly the injected episode + the static run, one representative per cluster
    assert dd.search_mode == "ivf_pq" and dd.lance_queries > 0
    assert _dup_set(lance_dir) == _expected_dups()
    assert dd.dup_frames == len(_expected_dups())
    assert dd.duplicate_episodes == [
        {"episode_index": DUP_EP, "duplicate_of_episode": DUP_SRC_EP, "dup_fraction": 1.0, "frames_from_main_source": EP_LEN}
    ]
    assert dd.bytes_written_videos == 0 and dd.files_written_videos == 0
    assert "is_semantic_dup" in dd.scalar_indexes_built
    for i, (ep, fr) in enumerate(zip(rows["episode_index"], rows["frame_index"], strict=True)):
        cid = rows["semantic_cluster_id"][i]
        if ep == DUP_EP:
            assert cid == rows["index"][key[(DUP_SRC_EP, fr)]]
        elif ep == STATIC_EP and fr in EXPECTED_STATIC_DUPS:
            assert cid == rows["index"][key[(STATIC_EP, STATIC_START)]]
        else:
            assert cid == rows["index"][i]  # representatives / unique frames point at themselves
        assert rows["semantic_nn_dist"][i] is not None

    # -- idempotent re-runs: same tags, new version, no duplicate columns
    v_before = int(table.version)
    dd2 = tag_semantic_duplicates(lance_dir, threshold=THRESHOLD, scope="all", query_batch_size=64)
    assert _dup_set(lance_dir) == _expected_dups() and dd2.dup_frames == dd.dup_frames
    rows2 = _frames(lance_dir, ["semantic_cluster_id"])
    assert rows2["semantic_cluster_id"] == rows["semantic_cluster_id"]
    table.checkout_latest()
    assert int(table.version) > v_before
    assert table.schema.names.count("is_semantic_dup") == 1

    # re-embedding drops the old index and the now-stale dedup columns; values are deterministic
    emb2 = add_embedding_column(lance_dir, KinematicEmbeddingProvider(horizon=HORIZON, dim=32, seed=0))
    assert set(emb2.dropped_columns) == {"embedding", *SEMANTIC_DUP_COLUMNS}
    table.checkout_latest()
    assert not any(c in table.schema.names for c in SEMANTIC_DUP_COLUMNS)
    e2 = np.asarray(_frames(lance_dir, ["embedding"])["embedding"], dtype=np.float32)
    np.testing.assert_array_equal(e2, e)
    assert hybrid_search(lance_dir, episode=DUP_SRC_EP, frame=5, k=1).search_mode == "flat"

    # videos.lance byte-for-byte untouched; upstream reader still opens and decodes
    assert _snapshot_tree(videos_dir) == videos_before
    ds = LanceDBDataset(root=lance_dir, return_uint8=True)
    assert ds.num_frames == NUM_EPS * EP_LEN
    assert ds[0]["observation.images.front_cam"].shape == (3, 48, 64)


def test_flat_fallback_and_grouping_do_not_change_tags(lance_dir: Path) -> None:
    add_embedding_column(lance_dir, KinematicEmbeddingProvider(horizon=HORIZON, dim=32, seed=0))
    flat = build_vector_index(lance_dir, min_rows=10_000)
    assert flat.search_mode == "flat" and flat.index_type == "FLAT" and flat.index_name is None
    assert "min_rows" in (flat.fallback_reason or "")
    r_flat = tag_semantic_duplicates(lance_dir, threshold=THRESHOLD, scope="all", query_batch_size=64)
    assert r_flat.search_mode == "flat"
    flat_tags = _frames(lance_dir, ["is_semantic_dup", "semantic_cluster_id"])

    ivf = build_vector_index(lance_dir)
    assert ivf.search_mode == "ivf_pq"
    r_ivf = tag_semantic_duplicates(lance_dir, threshold=THRESHOLD, scope="all", query_batch_size=64)
    assert r_ivf.search_mode == "ivf_pq"
    assert _frames(lance_dir, ["is_semantic_dup", "semantic_cluster_id"]) == flat_tags

    # one group covering every episode -> only the exact in-memory path, zero Lance queries
    r_one = tag_semantic_duplicates(lance_dir, threshold=THRESHOLD, scope="all", query_batch_size=10_000)
    assert r_one.lance_queries == 0
    assert _frames(lance_dir, ["is_semantic_dup", "semantic_cluster_id"]) == flat_tags
    assert _dup_set(lance_dir) == _expected_dups()

    # scopes separate cleanly
    tag_semantic_duplicates(lance_dir, threshold=THRESHOLD, scope="cross_episode", query_batch_size=64)
    assert _dup_set(lance_dir) == {(DUP_EP, f) for f in range(EP_LEN)}
    within = tag_semantic_duplicates(lance_dir, threshold=THRESHOLD, scope="within_episode")
    assert within.search_mode == "exact_local"
    assert _dup_set(lance_dir) == {(STATIC_EP, f) for f in EXPECTED_STATIC_DUPS}


def test_where_integration_keeps_full_action_horizon(lance_dir: Path) -> None:
    _embed_index_dedup(lance_dir)
    dups = _expected_dups()

    ds_all = _sharded(lance_dir, None)
    ds_f = _sharded(lance_dir, "NOT is_semantic_dup")
    try:
        all_anchors = set(ds_all.get_all_valid_anchors())
        f_anchors = set(ds_f.get_all_valid_anchors())
        assert f_anchors == {a for a in all_anchors if a not in dups}
        assert len(f_anchors) < len(all_anchors)
        assert not any(ep == DUP_EP for ep, _ in f_anchors)
        assert sum(len(ds_f.get_shard_splits(i)[j][1]) for i in range(len(ds_f)) for j in range(len(ds_f.get_shard_splits(i)))) == len(f_anchors)

        # (5, 10) is kept and its 4-step horizon runs straight into tagged frames 11..13;
        # (5, 28) sits right after the static run; (2, 30) is a representative of episode 7.
        probes = [(STATIC_EP, STATIC_START), (STATIC_EP, 28), (DUP_SRC_EP, 30), (0, 0), (9, EP_LEN - HORIZON)]
        for ep, st in probes:
            assert (ep, st) in f_anchors
            a = ds_f.episode_loader.load_steps(ep, [st], EmbodimentTag.NEW_EMBODIMENT, allow_padding=True)[0]
            b = ds_all.episode_loader.load_steps(ep, [st], EmbodimentTag.NEW_EMBODIMENT, allow_padding=True)[0]
            assert a.actions["arm"].shape == (HORIZON, 7) and a.actions["gripper"].shape == (HORIZON, 1)
            for k in a.actions:
                np.testing.assert_array_equal(a.actions[k], b.actions[k])
            for k in a.states:
                np.testing.assert_array_equal(a.states[k], b.states[k])
            for k in a.images:
                assert len(a.images[k]) == 2
                for x, y in zip(a.images[k], b.images[k], strict=True):
                    np.testing.assert_array_equal(x, y)
            assert a.text == b.text
        # the kept anchor's horizon covers the static frames verbatim
        s10 = ds_f.episode_loader.load_steps(STATIC_EP, [STATIC_START], EmbodimentTag.NEW_EMBODIMENT, allow_padding=True)[0]
        np.testing.assert_allclose(s10.actions["arm"], np.repeat(s10.actions["arm"][:1], HORIZON, axis=0))
    finally:
        ds_all.close()
        ds_f.close()

    # compose with tabular curation: dedup only among valid frames, then filter on both columns
    cur = curate_lance_dataset(lance_dir, idle_eps=1e-4)
    assert cur.bytes_written_videos == 0
    dd = tag_semantic_duplicates(lance_dir, threshold=THRESHOLD, scope="all", where="is_valid", query_batch_size=64)
    rows = _frames(lance_dir, ["episode_index", "frame_index", "is_valid", "is_semantic_dup", "semantic_cluster_id"])
    invalid = {
        (int(e), int(f))
        for e, f, v in zip(rows["episode_index"], rows["frame_index"], rows["is_valid"], strict=True)
        if not v
    }
    assert invalid == {(STATIC_EP, f) for f in range(STATIC_START + 1, STATIC_END + 1)}  # idle frames
    assert _dup_set(lance_dir) == {(DUP_EP, f) for f in range(EP_LEN)}
    assert dd.candidate_frames == NUM_EPS * EP_LEN - len(invalid)
    assert all(c == -1 for (e, f, c) in zip(rows["episode_index"], rows["frame_index"], rows["semantic_cluster_id"], strict=True) if (e, f) in invalid)
    ds_both = _sharded(lance_dir, "is_valid AND NOT is_semantic_dup")
    ds_valid = _sharded(lance_dir, "is_valid")
    try:
        both = set(ds_both.get_all_valid_anchors())
        assert both == {a for a in ds_valid.get_all_valid_anchors() if a[0] != DUP_EP}
    finally:
        ds_both.close()
        ds_valid.close()


def test_hybrid_search_slicing_and_training_predicates(lance_dir: Path) -> None:
    _embed_index_dedup(lance_dir)

    sl = slice_episodes_by_similarity(lance_dir, DUP_SRC_EP, k=3)
    assert sl.search_mode == "ivf_pq"
    assert sl.episodes()[0] == DUP_EP and sl.ranked[0].score < 1e-4 and sl.ranked[0].coverage == 1.0
    assert sl.ranked[1].score > 10 * THRESHOLD
    assert slice_episodes_by_similarity(lance_dir, DUP_EP, k=1).episodes() == [DUP_SRC_EP]

    hit = hybrid_search(lance_dir, episode=DUP_SRC_EP, frame=20, k=5)
    assert (hit.hits[0].episode_index, hit.hits[0].frame_index) == (DUP_EP, 20)
    assert all(h.episode_index != DUP_SRC_EP for h in hit.hits)
    # SQL prefilter is applied before the vector search: dups are gone, still k hits returned
    pre = hybrid_search(lance_dir, episode=DUP_SRC_EP, frame=20, k=5, where="NOT is_semantic_dup")
    assert len(pre.hits) == 5 and all(h.episode_index != DUP_EP for h in pre.hits)
    narrow = hybrid_search(lance_dir, episode=DUP_SRC_EP, k=4, where="episode_index IN (0, 1)")
    assert narrow.hits and {h.episode_index for h in narrow.hits} <= {0, 1}

    # query by raw vector (does not exclude any episode)
    vec = np.asarray(
        lancedb.connect(str(lance_dir)).open_table(FRAMES_TABLE).search()
        .where(f"episode_index = {DUP_SRC_EP} AND frame_index = 20").select(["embedding"]).limit(1)
        .to_arrow().column("embedding").to_pylist()[0],
        dtype=np.float32,
    )
    byvec = hybrid_search(lance_dir, query_vector=vec, k=2)
    assert {(h.episode_index, h.frame_index) for h in byvec.hits} == {(DUP_SRC_EP, 20), (DUP_EP, 20)}
    assert byvec.hits[0].distance < 1e-4 and byvec.hits[1].distance < 1e-4

    # results -> where predicate -> training anchors
    ds = _sharded(lance_dir, sl.to_where())
    try:
        anchors = ds.get_all_valid_anchors()
        assert {ep for ep, _ in anchors} == set(sl.episodes())
        assert len(anchors) == len(sl.episodes()) * (EP_LEN - HORIZON + 1)
    finally:
        ds.close()
    frame_pred = hit.to_where("frame")
    tbl = lancedb.connect(str(lance_dir)).open_table(FRAMES_TABLE)
    assert tbl.count_rows(frame_pred) == len({(h.episode_index, h.frame_index) for h in hit.hits})
    assert set(hit.anchors_by_episode()) == set(hit.episodes())

    rare = mine_rare_frames(lance_dir, top_k=5)
    d = [h.distance for h in rare.hits]
    assert len(d) == 5 and d == sorted(d, reverse=True) and d[-1] > 10 * THRESHOLD
    assert all(h.episode_index not in (DUP_SRC_EP, DUP_EP) for h in rare.hits)


def test_every_n_visual_provider_function_provider_and_propagation(lance_dir: Path) -> None:
    videos_before = _snapshot_tree(lance_dir / "videos.lance")
    seen_shapes: set[tuple[int, ...]] = set()

    def mean_color(images: list[np.ndarray]) -> np.ndarray:
        for im in images:
            seen_shapes.add((im.shape, str(im.dtype)))
        return np.stack([np.r_[im.reshape(-1, 3).mean(axis=0) / 255.0, 1.0] for im in images]).astype(np.float32)

    vis = add_embedding_column(lance_dir, VisualEmbeddingProvider("front", image_embed_fn=mean_color), every_n=5)
    assert seen_shapes == {((48, 64, 3), "uint8")}
    assert vis.keyframes == NUM_EPS * (EP_LEN // 5) and vis.embedded_frames == vis.keyframes
    assert vis.null_frames == NUM_EPS * EP_LEN - vis.keyframes and vis.dim == 4
    assert vis.bytes_written_videos == 0 and vis.files_written_videos == 0
    tbl = lancedb.connect(str(lance_dir)).open_table(FRAMES_TABLE)
    assert tbl.count_rows("embedding IS NOT NULL") == vis.keyframes
    assert tbl.count_rows("embedding IS NOT NULL AND frame_index % 5 != 0") == 0
    small = build_vector_index(lance_dir)  # 120 vectors < 256 -> exact flat search, recorded
    assert small.search_mode == "flat" and small.fallback_reason
    # visual neighbours stay inside the camera's colour family: ep 3 is closest to itself
    assert hybrid_search(lance_dir, episode=3, frame=10, k=1, exclude_query_episode=False).hits[0].episode_index == 3

    # plain 3-arg callable (episode_index, frame_index, tabular) -> embeddings
    def designed(ep: np.ndarray, fr: np.ndarray, tab: dict[str, np.ndarray]) -> np.ndarray:
        assert set(tab) == {"index", "state", "action"} and tab["state"].shape == (len(ep), 8)
        ep_code = np.where(ep == DUP_EP, DUP_SRC_EP, ep)
        out = np.zeros((len(ep), NUM_EPS + EP_LEN), dtype=np.float32)
        out[np.arange(len(ep)), ep_code] = 1.0
        out[np.arange(len(ep)), NUM_EPS + fr] = 1.0
        return out

    fn_rep = add_embedding_column(lance_dir, designed, every_n=5)
    assert fn_rep.provider == "designed" and fn_rep.embedded_frames == NUM_EPS * (EP_LEN // 5)
    assert isinstance(FunctionEmbeddingProvider(designed).name, str)
    dd = tag_semantic_duplicates(lance_dir, threshold=0.01, scope="cross_episode")
    # 12 keyframes of episode 7 are tagged and every in-between frame inherits its keyframe's tag
    assert _dup_set(lance_dir) == {(DUP_EP, f) for f in range(EP_LEN)}
    assert dd.propagated_frames == NUM_EPS * EP_LEN - fn_rep.embedded_frames
    rows = _frames(lance_dir, ["episode_index", "frame_index", "index", "semantic_cluster_id"])
    idx_of = {(e, f): i for e, f, i in zip(rows["episode_index"], rows["frame_index"], rows["index"], strict=True)}
    for e, f, c in zip(rows["episode_index"], rows["frame_index"], rows["semantic_cluster_id"], strict=True):
        if e == DUP_EP:
            assert c == idx_of[(DUP_SRC_EP, (f // 5) * 5)]
    assert _snapshot_tree(lance_dir / "videos.lance") == videos_before


def test_cli_smoke(lance_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "reports"
    root = str(lance_dir)
    main_vector_curate(["embed", "--root", root, "--horizon", str(HORIZON), "--dim", "32", "--out-json", str(out / "embed.json")])
    main_vector_curate(["index", "--root", root, "--out-json", str(out / "index.json")])
    main_vector_curate(
        ["dedup", "--root", root, "--threshold", str(THRESHOLD), "--scope", "all", "--query-batch-size", "64",
         "--out-json", str(out / "dedup.json")]
    )
    main_vector_curate(["search", "--root", root, "--episode", str(DUP_SRC_EP), "--level", "episode", "--k", "2",
                        "--out-json", str(out / "slice.json")])
    main_vector_curate(["search", "--root", root, "--episode", str(DUP_SRC_EP), "--frame", "20", "--k", "3",
                        "--where", "NOT is_semantic_dup", "--out-json", str(out / "search.json")])
    main_vector_curate(["search", "--root", root, "--level", "rare", "--k", "3", "--out-json", str(out / "rare.json")])
    capsys.readouterr()

    embed = json.loads((out / "embed.json").read_text())
    index = json.loads((out / "index.json").read_text())
    dedup = json.loads((out / "dedup.json").read_text())
    slc = json.loads((out / "slice.json").read_text())
    srch = json.loads((out / "search.json").read_text())
    rare = json.loads((out / "rare.json").read_text())
    assert embed["bytes_written_videos"] == 0 and embed["dim"] == 32
    assert index["search_mode"] == "ivf_pq"
    assert dedup["dup_frames"] == len(_expected_dups()) and dedup["bytes_written_videos"] == 0
    assert slc["episodes"][0] == DUP_EP and slc["where_predicate"].startswith("episode_index IN (")
    assert len(srch["hits"]) == 3 and all(h["episode_index"] != DUP_EP for h in srch["hits"])
    assert len(rare["hits"]) == 3


def test_bench_vector_curation_script(dup_v2_dir: Path, tmp_path: Path) -> None:
    script = Path(__file__).resolve().parent.parent / "scripts" / "bench_vector_curation.py"
    out_json = tmp_path / "bench.json"
    proc = subprocess.run(
        [
            sys.executable, str(script), "--v2-dir", str(dup_v2_dir), "--workdir", str(tmp_path / "work"),
            "--horizon", str(HORIZON), "--dim", "32", "--threshold", str(THRESHOLD), "--scope", "all",
            "--query-batch-size", "64", "--slice-episode", str(DUP_SRC_EP), "--out-json", str(out_json),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Vector curation benchmark" in proc.stdout
    rep = json.loads(out_json.read_text())
    assert rep["lance"]["bytes_written_videos"] == 0
    assert rep["lance"]["dup_frames"] == len(_expected_dups())
    assert rep["lance"]["search_mode"] == "ivf_pq"
    assert rep["lance"]["slice_top_episode"] == DUP_EP
    assert rep["external"]["dup_frames"] == len(_expected_dups())
    assert rep["agreement"]["jaccard"] == 1.0
    assert rep["external"]["sidecar_bytes"] > 0
    for k in ("embed_s", "index_s", "dedup_s", "slice_s", "total_s"):
        assert rep["lance"][k] >= 0.0
    for k in ("export_s", "knn_s", "write_sidecar_s", "total_s"):
        assert rep["external"][k] >= 0.0
