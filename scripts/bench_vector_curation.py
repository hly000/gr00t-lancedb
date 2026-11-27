#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Vector curation benchmark: in-table LanceDB workflow vs an "external vector store" workflow.

Lance workflow (everything inside ``frames.lance``):
  1. ``add_embedding_column``   - kinematic (state + next-k actions) embedding, zero-copy column add
  2. ``build_vector_index``     - IVF-PQ (cosine), or flat for tiny tables
  3. ``tag_semantic_duplicates``- kNN dedup through the index -> ``is_semantic_dup`` columns
  4. ``slice_episodes_by_similarity`` - scene slice around ``--slice-episode``
  5. the training filter is just ``where="NOT is_semantic_dup"`` (timed as an anchor query)

External workflow (what you do with a separate vector DB / FAISS / numpy):
  1. export embeddings + keys out of the dataset (``.npy`` + ``.parquet``)
  2. brute-force cosine kNN in numpy with the *same* dedup rules (top-k cross-episode within
     threshold, static runs, deterministic greedy representatives)
  3. write a sidecar parquet mask, and join it back to get training anchors

The embeddings are computed once (by step 1 of the Lance workflow) and reused by the external
baseline so only search/dedup/write/integration costs are compared. Both produce the same tags on
exact search; ``agreement`` reports the Jaccard overlap of the duplicate sets. ``videos.lance``
must receive 0 bytes.

Usage::

    python scripts/bench_vector_curation.py --v2-dir /data/libero_10_v2 --out-json bench.json
    python scripts/bench_vector_curation.py --lance-root /data/libero_10.lance --threshold 0.01
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# Import gr00t_lancedb before lerobot_lancedb so _hub_compat shims huggingface_hub<1.0 cleanly.
from gr00t_lancedb import convert_v2_to_lance
from gr00t_lancedb.vector_curate import (
    KinematicEmbeddingProvider,
    _fsl_to_float32,
    _static_run_edges,
    add_embedding_column,
    build_vector_index,
    greedy_duplicate_assignment,
    slice_episodes_by_similarity,
    tag_semantic_duplicates,
)
from lerobot_lancedb import FRAMES_TABLE


def _dir_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.exists() else 0


def _prepare(args: argparse.Namespace, workdir: Path) -> Path:
    if args.lance_root and args.in_place:
        return Path(args.lance_root)
    dst = workdir / "lance"
    if dst.exists():
        shutil.rmtree(dst)
    if args.v2_dir:
        convert_v2_to_lance(Path(args.v2_dir), dst, video_mode=args.video_mode)
    else:
        shutil.copytree(Path(args.lance_root), dst)
    return dst


def _io(report: Any) -> dict[str, int]:
    return {
        "frames": int(report.bytes_written_frames),
        "videos": int(report.bytes_written_videos),
        "meta": int(report.bytes_written_meta),
        "other": int(report.bytes_written_other),
    }


def run_lance(args: argparse.Namespace, lance: Path) -> dict[str, Any]:
    import lancedb

    t_total = time.perf_counter()
    t = time.perf_counter()
    emb = add_embedding_column(
        lance,
        KinematicEmbeddingProvider(horizon=args.horizon, dim=args.dim or None, seed=args.seed),
        every_n=args.every_n,
    )
    embed_s = time.perf_counter() - t

    t = time.perf_counter()
    idx = build_vector_index(lance, min_rows=args.min_index_rows)
    index_s = time.perf_counter() - t

    t = time.perf_counter()
    dd = tag_semantic_duplicates(
        lance,
        threshold=args.threshold,
        scope=args.scope,
        k=args.k,
        where=args.where,
        query_batch_size=args.query_batch_size,
    )
    dedup_s = time.perf_counter() - t

    slice_ep = args.slice_episode
    if slice_ep is None:  # default: the source of the first duplicated episode, else episode 0
        slice_ep = int(dd.duplicate_episodes[0]["duplicate_of_episode"]) if dd.duplicate_episodes else 0
    t = time.perf_counter()
    sl = slice_episodes_by_similarity(lance, slice_ep, k=args.slice_k)
    slice_s = time.perf_counter() - t

    # what the GR00T loader does for where="NOT is_semantic_dup": a 2-column anchor query
    t = time.perf_counter()
    table = lancedb.connect(str(lance)).open_table(FRAMES_TABLE)
    pred = "NOT is_semantic_dup" if not args.where else f"({args.where}) AND NOT is_semantic_dup"
    anchors = table.search().where(pred).select(["episode_index", "frame_index"]).limit(None).to_arrow()
    filter_query_s = time.perf_counter() - t

    by_table: dict[str, int] = {}
    for rep in (emb, idx, dd):
        for k, v in _io(rep).items():
            by_table[k] = by_table.get(k, 0) + v
    return {
        "lance_root": str(lance),
        "total_frames": int(emb.total_frames),
        "embedded_frames": int(emb.embedded_frames),
        "dim": int(emb.dim),
        "search_mode": dd.search_mode,
        "index": {
            "type": idx.index_type,
            "num_partitions": idx.num_partitions,
            "num_sub_vectors": idx.num_sub_vectors,
            "fallback_reason": idx.fallback_reason,
        },
        "dup_frames": int(dd.dup_frames),
        "dup_pct": float(dd.dup_pct),
        "duplicate_episodes": dd.duplicate_episodes,
        "nn_dist_quantiles": dd.nn_dist_quantiles,
        "lance_queries": int(dd.lance_queries),
        "slice_query_episode": int(slice_ep),
        "slice_top_episode": int(sl.episodes()[0]) if sl.ranked else None,
        "slice_ranked": [r.__dict__ for r in sl.ranked],
        "slice_where": sl.to_where(),
        "filtered_anchor_rows": int(anchors.num_rows),
        "embed_s": round(embed_s, 4),
        "index_s": round(index_s, 4),
        "dedup_s": round(dedup_s, 4),
        "slice_s": round(slice_s, 4),
        "filter_query_s": round(filter_query_s, 4),
        "total_s": round(time.perf_counter() - t_total, 4),
        "bytes_written_by_table": by_table,
        "bytes_written_frames": by_table["frames"],
        "bytes_written_videos": by_table["videos"],
        "bytes_measured": bool(emb.bytes_measured),
        "frames_table_version": int(dd.frames_table_version),
        "training_integration": "LanceShardedSingleStepDataset(where='NOT is_semantic_dup')",
    }


def _bruteforce_cross_episode(
    emb: np.ndarray, ep: np.ndarray, *, k: int, block_bytes: int = 256 << 20
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact top-k cross-episode neighbours by blocked numpy matmul (O(N^2 D))."""
    n = len(ep)
    block = max(1, min(4096, block_bytes // max(1, 4 * n)))
    out_q, out_n, out_d = [], [], []
    nn = np.full(n, np.nan)
    kk = min(k, max(1, n - 1))
    for r0 in range(0, n, block):
        r1 = min(n, r0 + block)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            d = 1.0 - emb[r0:r1] @ emb.T
        d[ep[r0:r1, None] == ep[None, :]] = np.inf
        part = np.argpartition(d, kk - 1, axis=1)[:, :kk] if kk < n else np.tile(np.arange(n), (r1 - r0, 1))
        qi = np.repeat(np.arange(r0, r1), part.shape[1])
        nj = part.reshape(-1)
        ok = np.isfinite(d[qi - r0, nj])
        qi, nj = qi[ok], nj[ok]
        dist = np.clip(1.0 - np.sum(emb[qi] * emb[nj], axis=1, dtype=np.float64), 0.0, 2.0)
        srt = np.lexsort((nj, dist, qi))
        qi, nj, dist = qi[srt], nj[srt], dist[srt]
        first = np.ones(len(qi), dtype=bool)
        first[1:] = qi[1:] != qi[:-1]
        start = np.maximum.accumulate(np.where(first, np.arange(len(qi)), 0))
        keep = (np.arange(len(qi)) - start) < k
        qi, nj, dist = qi[keep], nj[keep], dist[keep]
        nn[qi[first[keep]]] = dist[first[keep]]
        out_q.append(qi)
        out_n.append(nj)
        out_d.append(dist)
    cat = lambda xs, dt: np.concatenate(xs) if xs else np.empty(0, dtype=dt)  # noqa: E731
    return cat(out_q, np.int64), cat(out_n, np.int64), cat(out_d, np.float64), nn


def run_external(args: argparse.Namespace, lance: Path, workdir: Path) -> dict[str, Any]:
    import lancedb

    ext = workdir / "external"
    if ext.exists():
        shutil.rmtree(ext)
    ext.mkdir(parents=True)
    t_total = time.perf_counter()

    # 1. export embeddings + keys (what you'd push into an external vector store)
    t = time.perf_counter()
    table = lancedb.connect(str(lance)).open_table(FRAMES_TABLE)
    q = table.search()
    if args.where:
        q = q.where(args.where)
    tbl = q.select(["index", "episode_index", "frame_index", "embedding"]).limit(None).to_arrow()
    emb, valid = _fsl_to_float32(tbl.column("embedding"))
    keys = pa.table({c: tbl.column(c) for c in ("index", "episode_index", "frame_index")}).filter(pa.array(valid))
    emb = emb[valid]
    np.save(ext / "embeddings.npy", emb)
    pq.write_table(keys, ext / "keys.parquet")
    export_s = time.perf_counter() - t

    # 2. brute-force kNN + identical dedup rules
    t = time.perf_counter()
    ep = keys.column("episode_index").to_numpy().astype(np.int64)
    fr = keys.column("frame_index").to_numpy().astype(np.int64)
    gidx = keys.column("index").to_numpy().astype(np.int64)
    src_l, dst_l = [], []
    if args.scope in ("cross_episode", "all"):
        qi, nj, dist, _ = _bruteforce_cross_episode(emb, ep, k=args.k)
        hit = dist <= args.threshold
        src_l.append(qi[hit])
        dst_l.append(nj[hit])
    if args.scope in ("within_episode", "all"):
        s, d_, _ = _static_run_edges(emb, ep, fr, float(args.threshold))
        src_l.append(s)
        dst_l.append(d_)
    src = np.concatenate(src_l) if src_l else np.empty(0, dtype=np.int64)
    dst = np.concatenate(dst_l) if dst_l else np.empty(0, dtype=np.int64)
    is_dup, rep = greedy_duplicate_assignment(len(ep), src, dst, np.lexsort((fr, ep)))
    knn_s = time.perf_counter() - t

    # 3. sidecar mask parquet
    t = time.perf_counter()
    sidecar = ext / "semantic_dup_mask.parquet"
    pq.write_table(
        pa.table(
            {
                "episode_index": pa.array(ep),
                "frame_index": pa.array(fr),
                "is_semantic_dup": pa.array(is_dup),
                "semantic_cluster_id": pa.array(gidx[rep]),
            }
        ),
        sidecar,
    )
    write_sidecar_s = time.perf_counter() - t

    # 4. training-time join: read the sidecar and build the kept-anchor set
    t = time.perf_counter()
    mask = pq.read_table(sidecar)
    kept = mask.filter(pc.invert(mask.column("is_semantic_dup")))
    kept_anchors = set(zip(kept.column("episode_index").to_pylist(), kept.column("frame_index").to_pylist(), strict=True))
    join_s = time.perf_counter() - t

    return {
        "dup_frames": int(is_dup.sum()),
        "dup_set": sorted((int(e), int(f)) for e, f in zip(ep[is_dup], fr[is_dup], strict=True)),
        "cluster_ids": dict(zip(map(int, gidx), map(int, gidx[rep]), strict=True)),
        "kept_anchor_rows": len(kept_anchors),
        "export_s": round(export_s, 4),
        "knn_s": round(knn_s, 4),
        "write_sidecar_s": round(write_sidecar_s, 4),
        "join_s": round(join_s, 4),
        "total_s": round(time.perf_counter() - t_total, 4),
        "export_bytes": _dir_bytes(ext) - sidecar.stat().st_size,
        "sidecar_bytes": int(sidecar.stat().st_size),
        "training_integration": "custom loader code must join semantic_dup_mask.parquet on (episode_index, frame_index)",
    }


def _lance_tags(lance: Path) -> tuple[set[tuple[int, int]], dict[int, int]]:
    import lancedb

    tbl = (
        lancedb.connect(str(lance)).open_table(FRAMES_TABLE).search()
        .select(["index", "episode_index", "frame_index", "is_semantic_dup", "semantic_cluster_id"]).limit(None).to_arrow()
    )
    d = tbl.to_pydict()
    dups = {(int(e), int(f)) for e, f, x in zip(d["episode_index"], d["frame_index"], d["is_semantic_dup"], strict=True) if x}
    clusters = {int(i): int(c) for i, c in zip(d["index"], d["semantic_cluster_id"], strict=True) if c >= 0}
    return dups, clusters


def main(argv: list[str] | None = None) -> dict[str, Any]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--v2-dir", help="LeRobot v2.1 dataset to convert into a fresh Lance dataset.")
    src.add_argument("--lance-root", help="Existing Lance dataset (copied into --workdir unless --in-place).")
    p.add_argument("--in-place", action="store_true", help="Modify --lance-root directly instead of a copy.")
    p.add_argument("--workdir", default=None, help="Scratch directory (default: a temp dir, removed afterwards).")
    p.add_argument("--video-mode", default="copy", help="convert_v2_to_lance video_mode (default: copy).")
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--every-n", type=int, default=1)
    p.add_argument("--min-index-rows", type=int, default=256)
    p.add_argument("--threshold", type=float, default=0.02)
    p.add_argument("--scope", choices=("cross_episode", "within_episode", "all"), default="cross_episode")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--where", default=None, help="Candidate filter for both workflows, e.g. 'is_valid'.")
    p.add_argument("--query-batch-size", type=int, default=1024)
    p.add_argument("--slice-episode", type=int, default=None)
    p.add_argument("--slice-k", type=int, default=5)
    p.add_argument("--out-json", default=None)
    args = p.parse_args(argv)

    tmp = None
    if args.workdir:
        workdir = Path(args.workdir)
        workdir.mkdir(parents=True, exist_ok=True)
    else:
        tmp = tempfile.TemporaryDirectory(prefix="bench_vec_")
        workdir = Path(tmp.name)
    try:
        lance = _prepare(args, workdir)
        lance_rep = run_lance(args, lance)
        ext_rep = run_external(args, lance, workdir)
        l_dups, l_clusters = _lance_tags(lance)
        e_dups = set(map(tuple, ext_rep.pop("dup_set")))
        e_clusters = ext_rep.pop("cluster_ids")
        union = l_dups | e_dups
        shared = [i for i in e_clusters if i in l_clusters]
        agreement = {
            "jaccard": round(len(l_dups & e_dups) / len(union), 6) if union else 1.0,
            "lance_only": len(l_dups - e_dups),
            "external_only": len(e_dups - l_dups),
            "cluster_id_match_frac": round(sum(e_clusters[i] == l_clusters[i] for i in shared) / len(shared), 6) if shared else 1.0,
        }
        lance_core = lance_rep["index_s"] + lance_rep["dedup_s"]
        ext_core = ext_rep["export_s"] + ext_rep["knn_s"] + ext_rep["write_sidecar_s"]
        report = {
            "benchmark": "vector_curation",
            "params": {k: v for k, v in vars(args).items() if k != "out_json"},
            "lance": lance_rep,
            "external": ext_rep,
            "agreement": agreement,
            "side_by_side_s": {
                "lance_index_plus_dedup": round(lance_core, 4),
                "external_export_knn_sidecar": round(ext_core, 4),
                "lance_training_filter": lance_rep["filter_query_s"],
                "external_training_join": ext_rep["join_s"],
            },
        }
    finally:
        if tmp is not None:
            tmp.cleanup()

    lr, er = report["lance"], report["external"]
    print("# Vector curation benchmark")
    print(f"frames={lr['total_frames']} dim={lr['dim']} search={lr['search_mode']} index={lr['index']['type']}")
    print(f"{'step':<34}{'lance (s)':>12}{'external (s)':>14}")
    print(f"{'embed (shared)':<34}{lr['embed_s']:>12.4f}{'-':>14}")
    print(f"{'index build / export':<34}{lr['index_s']:>12.4f}{er['export_s']:>14.4f}")
    print(f"{'kNN dedup':<34}{lr['dedup_s']:>12.4f}{er['knn_s']:>14.4f}")
    print(f"{'write tags (in-table / sidecar)':<34}{'(in dedup)':>12}{er['write_sidecar_s']:>14.4f}")
    print(f"{'training filter / join':<34}{lr['filter_query_s']:>12.4f}{er['join_s']:>14.4f}")
    print(f"{'scene slice':<34}{lr['slice_s']:>12.4f}{'-':>14}")
    print(f"dup frames: lance={lr['dup_frames']} external={er['dup_frames']} jaccard={report['agreement']['jaccard']}")
    print(f"bytes written: {lr['bytes_written_by_table']} (videos must be 0); sidecar={er['sidecar_bytes']}")
    payload = json.dumps(report, indent=2, default=str) + "\n"
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(payload, encoding="utf-8")
    return report


if __name__ == "__main__":
    main()
