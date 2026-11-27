# SPDX-License-Identifier: Apache-2.0
"""Failure mining: from failed simulator rollouts to a per-frame sampling weight column.

For every failed rollout of a checkpoint:

1. each decision point is embedded in the *same* space as the ``frames.lance`` embedding
   column (kinematic provider: z-scored state + upcoming action chunk, same projection);
2. its similarity to the closest training frame of the same task gives a profile ``s_k``;
3. the **divergence point** is the first decision where ``s_k`` (smoothed) falls
   ``divergence_drop`` below its running maximum — where the policy left the demonstrated
   behaviour, which is usually long before the episode times out. If ``s_k`` never drops
   that much, the least-covered decision is used;
4. the decisions around the divergence point query the training frames; matches above
   ``min_match_sim`` score ``sim * exp(-|k - k*| / tau)`` and the score is spread to nearby
   frames of the matched demonstration (the critical second, not one frame);
5. matches below ``min_match_sim`` are **coverage gaps**: states the data does not contain,
   which re-weighting cannot fix. They are reported, not boosted.

Scores of all failures are summed per frame, normalized to ``[0, 1]`` and turned into
weights ``1 + alpha * score``, clipped to ``[w_min, w_max]`` and normalized to mean 1 over
the kept rows (rows outside ``valid_where`` get 0). Two float32 columns are appended to
``frames.lance`` with zero-copy column evolution; ``videos.lance`` is never written.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
import json
import logging
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import pyarrow as pa

from ..vector_curate import (
    EMBEDDING_COLUMN,
    KinematicEmbeddingProvider,
    _arrow_col_to_2d_float64,
    _col_to_numpy,
    _drop_columns_and_indices,
    _embedding_field_info,
    _fsl_to_float32,
    _local_root,
    _matmul,
    _modality_original_keys,
    _open_frames,
    _read_modality,
    _WriteMeter,
)
from .rollouts import RolloutEpisode, iter_episodes, modality_vector, read_summary

logger = logging.getLogger(__name__)

SCORE_COLUMN = "flywheel_fail_score"
WEIGHT_COLUMN = "flywheel_weight"
FLYWHEEL_FIELD_METADATA_KEY = b"gr00t_lancedb.flywheel"
FLYWHEEL_FORMAT = "gr00t_lancedb.flywheel/v1"


@dataclass
class MiningConfig:
    embedding_column: str = EMBEDDING_COLUMN
    top_k: int = 8
    min_match_sim: float = 0.5
    divergence_drop: float = 0.15
    smooth: int = 3
    window_before: int = 2
    window_after: int = 1
    decay_tau: float = 1.5
    spread_frames: int = 8
    alpha: float = 3.0
    w_min: float = 0.25
    w_max: float = 4.0
    valid_where: str | None = None
    same_task_only: bool = True
    score_column: str = SCORE_COLUMN
    weight_column: str = WEIGHT_COLUMN

    def validate(self) -> None:
        if self.top_k < 1 or self.smooth < 1 or self.spread_frames < 0:
            raise ValueError("top_k and smooth must be >= 1, spread_frames >= 0")
        if self.window_before < 0 or self.window_after < 0 or self.decay_tau <= 0:
            raise ValueError("window_before/after must be >= 0 and decay_tau > 0")
        if not (0 < self.w_min <= 1.0 <= self.w_max):
            raise ValueError("need 0 < w_min <= 1 <= w_max")
        if self.alpha < 0:
            raise ValueError("alpha must be >= 0")


# ---------------------------------------------------------------------------------------------
# Training-frame index (everything mining needs from frames.lance, in physical row order)
# ---------------------------------------------------------------------------------------------
@dataclass
class FrameIndex:
    lance_root: str
    frames_version: int
    index: np.ndarray  # global frame index column
    episode_index: np.ndarray
    frame_index: np.ndarray
    task_index: np.ndarray
    embedding: np.ndarray  # [N, D] float32, zeros where null
    has_embedding: np.ndarray  # [N] bool
    keep: np.ndarray  # [N] bool rows that may be sampled
    provider: KinematicEmbeddingProvider
    state_cols: list[str]
    action_cols: list[str]
    state_widths: list[int]
    action_widths: list[int]
    state_ref: tuple[np.ndarray, np.ndarray]
    action_ref: tuple[np.ndarray, np.ndarray]
    modality: dict[str, Any]
    tasks: dict[int, str]
    embedding_info: dict[str, Any]
    _order: np.ndarray = field(default=None, repr=False)  # rows sorted by (episode, frame)
    _rank: np.ndarray = field(default=None, repr=False)  # inverse of _order

    def __post_init__(self) -> None:
        self._order = np.lexsort((self.frame_index, self.episode_index))
        self._rank = np.empty_like(self._order)
        self._rank[self._order] = np.arange(len(self._order))

    def __len__(self) -> int:
        return int(len(self.index))

    def task_rows(self, task_index: int | None) -> np.ndarray:
        mask = self.has_embedding.copy()
        if task_index is not None:
            mask &= self.task_index == int(task_index)
        return np.flatnonzero(mask)

    def neighbours(self, row: int, radius: int) -> tuple[np.ndarray, np.ndarray]:
        """Rows within ``radius`` frames of ``row`` in the same episode, and their offsets."""
        r = int(self._rank[row])
        lo, hi = max(0, r - radius), min(len(self._order), r + radius + 1)
        cand = self._order[lo:hi]
        same = self.episode_index[cand] == self.episode_index[row]
        cand = cand[same]
        off = self.frame_index[cand] - self.frame_index[row]
        ok = np.abs(off) <= radius
        return cand[ok], off[ok]


def _normalize_text(s: str) -> str:
    return " ".join(str(s).lower().replace("_", " ").split())


def _load_tasks(db: Any, lance_root: str | Path, storage_options: Mapping[str, str] | None) -> dict[int, str]:
    from ..lance_query import _read_meta_files
    from ..metadata import _load_tasks_map

    local = _local_root(lance_root)
    if local is not None and (local / "meta").is_dir():
        tasks = _load_tasks_map(local / "meta")
        if tasks:
            return tasks
    files = _read_meta_files(db, dict(storage_options) if storage_options else None, ["tasks.jsonl", "tasks.parquet"])
    if files.get("tasks.jsonl"):
        out = {}
        for line in bytes(files["tasks.jsonl"]).decode("utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                out[int(rec["task_index"])] = str(rec["task"])
        return out
    if files.get("tasks.parquet"):
        import io

        import pandas as pd

        df = pd.read_parquet(io.BytesIO(bytes(files["tasks.parquet"])))
        if "task" in df.columns:
            return {int(r["task_index"]): str(r["task"]) for _, r in df.iterrows()}
        return {int(r["task_index"]): str(t) for t, r in df.iterrows()}
    return {}


def load_frame_index(
    lance_root: str | Path,
    *,
    embedding_column: str = EMBEDDING_COLUMN,
    valid_where: str | None = None,
    storage_options: Mapping[str, str] | None = None,
    lance_version: int | None = None,
) -> FrameIndex:
    db, table = _open_frames(lance_root, storage_options, lance_version)
    info = _embedding_field_info(table, embedding_column)
    if info.get("provider") != "kinematic":
        raise NotImplementedError(
            f"failure mining needs a kinematic embedding column; {embedding_column!r} was made by "
            f"{info.get('provider')!r} (visual rollout embeddings are not supported yet)"
        )
    cfg = dict(info.get("config") or {})
    provider = KinematicEmbeddingProvider(
        horizon=int(cfg.get("horizon", 16)),
        dim=cfg.get("dim"),
        seed=int(cfg.get("seed", 0)),
        state_weight=float(cfg.get("state_weight", 1.0)),
        action_weight=float(cfg.get("action_weight", 1.0)),
    )
    if int(info.get("every_n", 1)) != 1:
        logger.warning("embedding column %r covers every %s-th frame only", embedding_column, info.get("every_n"))

    from lerobot_lancedb import to_lance_column

    modality = _read_modality(db, lance_root, storage_options)
    names = set(table.schema.names)
    state_cols = [c for c in (to_lance_column(k) for k in _modality_original_keys(modality, "state")) if c in names]
    action_cols = [c for c in (to_lance_column(k) for k in _modality_original_keys(modality, "action")) if c in names]
    select = ["index", "episode_index", "frame_index", "task_index", embedding_column, *dict.fromkeys([*state_cols, *action_cols])]
    tab = table.search().select(select).limit(None).to_arrow()
    n = tab.num_rows

    def stack(cols: list[str]) -> tuple[np.ndarray, list[int]]:
        blocks = [_arrow_col_to_2d_float64(tab.column(c)) for c in cols]
        widths = [b.shape[1] for b in blocks]
        return (np.concatenate(blocks, axis=1) if blocks else np.empty((n, 0))), widths

    state, state_widths = stack(state_cols)
    action, action_widths = stack(action_cols)
    emb, has_emb = _fsl_to_float32(tab.column(embedding_column))

    keep = np.ones(n, dtype=bool)
    gidx = _col_to_numpy(tab, "index", np.int64)
    if valid_where:
        ok = table.search().where(valid_where).select(["index"]).limit(None).to_arrow()
        keep = np.isin(gidx, _col_to_numpy(ok, "index", np.int64))

    return FrameIndex(
        lance_root=str(lance_root),
        frames_version=int(table.version),
        index=gidx,
        episode_index=_col_to_numpy(tab, "episode_index", np.int64),
        frame_index=_col_to_numpy(tab, "frame_index", np.int64),
        task_index=_col_to_numpy(tab, "task_index", np.int64),
        embedding=emb,
        has_embedding=has_emb,
        keep=keep,
        provider=provider,
        state_cols=state_cols,
        action_cols=action_cols,
        state_widths=state_widths,
        action_widths=action_widths,
        state_ref=KinematicEmbeddingProvider.zscore_params(state),
        action_ref=KinematicEmbeddingProvider.zscore_params(action),
        modality=modality,
        tasks=_load_tasks(db, lance_root, storage_options),
        embedding_info=info,
    )


# ---------------------------------------------------------------------------------------------
# Rollout embedding and divergence
# ---------------------------------------------------------------------------------------------
def _section_by_column(fi: FrameIndex, section: str) -> list[tuple[str, int, Mapping[str, Any]]]:
    from lerobot_lancedb import to_lance_column

    from ..metadata import DEFAULT_COLUMN_NAMES

    cols = fi.state_cols if section == "state" else fi.action_cols
    widths = fi.state_widths if section == "state" else fi.action_widths
    by_orig = {to_lance_column(k): k for k in _modality_original_keys(fi.modality, section)}
    return [(by_orig[c], w, fi.modality.get(section, {})) for c, w in zip(cols, widths, strict=True)] or [
        (DEFAULT_COLUMN_NAMES[section], 0, {})
    ]


def rollout_features(fi: FrameIndex, ep: RolloutEpisode) -> tuple[np.ndarray, np.ndarray]:
    """``(state [K, Ds], action_window [K, horizon, Da])`` in the table's column layout."""
    from ..metadata import DEFAULT_COLUMN_NAMES

    k = ep.num_decisions
    h = fi.provider.horizon
    # State at each decision point.
    s_blocks, off = [], 0
    for orig, width, section in _section_by_column(fi, "state"):
        if width == 0:
            continue
        fill = fi.state_ref[0][off : off + width]
        s_blocks.append(
            modality_vector(ep.state, section, orig, width, fill, default_original_key=DEFAULT_COLUMN_NAMES["state"])
        )
        off += width
    state = np.concatenate(s_blocks, axis=1) if s_blocks else np.zeros((k, 0))

    # Executed action sequence, then the ``horizon`` actions following each decision.
    a_blocks, off = [], 0
    for orig, width, section in _section_by_column(fi, "action"):
        if width == 0:
            continue
        fill = fi.action_ref[0][off : off + width]
        executed = {key: ep.executed_actions(key) for key in ep.action}
        a_blocks.append(
            modality_vector(executed, section, orig, width, fill, default_original_key=DEFAULT_COLUMN_NAMES["action"])
        )
        off += width
    if a_blocks:
        seq = np.concatenate(a_blocks, axis=1)
        if len(seq) == 0:
            seq = np.broadcast_to(fi.action_ref[0], (1, off)).copy()
        t = np.asarray(ep.env_steps, dtype=np.int64)
        idx = np.minimum(t[:, None] + np.arange(h)[None, :], len(seq) - 1)
        window = seq[idx]
    else:
        window = np.zeros((k, h, 0))
    return state, window


def embed_rollout(fi: FrameIndex, ep: RolloutEpisode) -> np.ndarray:
    state, window = rollout_features(fi, ep)
    return fi.provider.embed_windows(state, window, state_ref=fi.state_ref, action_ref=fi.action_ref)


def _smooth(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(x) == 0:
        return x.copy()
    pad = w // 2
    xp = np.pad(x, (pad, w - 1 - pad), mode="edge")
    return np.convolve(xp, np.ones(w) / w, mode="valid")


def find_divergence(sims: np.ndarray, *, drop: float, smooth: int) -> tuple[int, str]:
    """First decision whose smoothed similarity is ``drop`` below its running max."""
    s = np.asarray(sims, dtype=np.float64)
    if len(s) == 0:
        raise ValueError("empty similarity profile")
    s = np.nan_to_num(s, nan=-1.0, posinf=1.0, neginf=-1.0)
    ss = _smooth(s, smooth)
    run_max = np.maximum.accumulate(ss)
    hit = np.flatnonzero(run_max - ss >= drop)
    if len(hit):
        return int(hit[0]), "drop"
    return int(np.argmin(ss)), "min"


@dataclass
class EpisodeMining:
    task: str
    ordinal: int
    seed: int
    task_index: int | None
    decisions: int
    divergence_k: int
    divergence_env_step: int
    method: str
    best_sim: float
    coverage_gap: bool
    matched_rows: int
    profile: list[float]


def _task_lookup(fi: FrameIndex) -> dict[str, int]:
    return {_normalize_text(t): i for i, t in fi.tasks.items()}


def mine_episode(
    fi: FrameIndex, ep: RolloutEpisode, cfg: MiningConfig, task_index: int | None
) -> tuple[np.ndarray, np.ndarray, EpisodeMining]:
    """Score training frames for one failed episode; returns ``(rows, scores, record)``."""
    rows_pool = fi.task_rows(task_index)
    if len(rows_pool) == 0:
        rows_pool = fi.task_rows(None)
    demo = fi.embedding[rows_pool]
    q = embed_rollout(fi, ep)
    ok_q = np.all(np.isfinite(q), axis=1)
    qz = np.where(ok_q[:, None], q, 0.0).astype(np.float32)
    sims_all = _matmul(qz, demo.T)  # [K, M] cosine (both sides unit norm)
    sims_all[~ok_q] = -1.0
    profile = sims_all.max(axis=1) if sims_all.shape[1] else np.full(len(q), -1.0)

    k_star, method = find_divergence(profile, drop=cfg.divergence_drop, smooth=cfg.smooth)
    lo, hi = max(0, k_star - cfg.window_before), min(len(q), k_star + cfg.window_after + 1)
    scores: dict[int, float] = {}
    best = -1.0
    top = min(cfg.top_k, sims_all.shape[1])
    for k in range(lo, hi):
        if not ok_q[k] or top == 0:
            continue
        row_sims = sims_all[k]
        cand = np.argpartition(-row_sims, top - 1)[:top]
        decay = math.exp(-abs(k - k_star) / cfg.decay_tau)
        for c in cand:
            sim = float(row_sims[c])
            best = max(best, sim)
            if sim < cfg.min_match_sim:
                continue
            r = int(rows_pool[c])
            val = sim * decay
            # Spread to the surrounding frames of the matched demonstration.
            nb, off = fi.neighbours(r, cfg.spread_frames)
            kern = 1.0 - np.abs(off) / (cfg.spread_frames + 1.0)
            for rr, kk in zip(nb.tolist(), kern.tolist(), strict=True):
                v = val * kk
                if v > scores.get(rr, 0.0):
                    scores[rr] = v
    rows = np.fromiter(scores.keys(), dtype=np.int64, count=len(scores))
    vals = np.fromiter(scores.values(), dtype=np.float64, count=len(scores))
    env_step = int(ep.env_steps[k_star]) if len(ep.env_steps) else 0
    rec = EpisodeMining(
        task=ep.task,
        ordinal=ep.ordinal,
        seed=ep.seed,
        task_index=task_index,
        decisions=len(q),
        divergence_k=k_star,
        divergence_env_step=env_step,
        method=method,
        best_sim=round(best, 4),
        coverage_gap=bool(best < cfg.min_match_sim),
        matched_rows=int(len(rows)),
        profile=[round(float(x), 4) for x in profile],
    )
    return rows, vals, rec


def compose_weights(score: np.ndarray, keep: np.ndarray, cfg: MiningConfig) -> np.ndarray:
    """``1 + alpha * score / max(score)`` on kept rows, clipped and normalized to mean 1."""
    score = np.asarray(score, dtype=np.float64)
    keep = np.asarray(keep, dtype=bool)
    w = np.zeros(len(score), dtype=np.float64)
    if not keep.any():
        raise ValueError("no rows kept (valid_where matched nothing)")
    peak = float(score[keep].max()) if keep.any() else 0.0
    norm = score / peak if peak > 0 else np.zeros_like(score)
    w[keep] = 1.0 + cfg.alpha * norm[keep]
    for _ in range(50):
        w[keep] /= w[keep].mean()
        clipped = np.clip(w[keep], cfg.w_min, cfg.w_max)
        if np.allclose(clipped, w[keep], rtol=0, atol=1e-9):
            break
        w[keep] = clipped
    w[keep] /= w[keep].mean()
    return w


@dataclass
class MiningResult:
    score: np.ndarray  # [N] normalized to [0, 1]
    weight: np.ndarray  # [N]
    episodes: list[EpisodeMining]
    rollouts: dict[str, Any]

    def summary(self, fi: FrameIndex, cfg: MiningConfig) -> dict[str, Any]:
        keep = fi.keep
        boosted = (self.score > 0) & keep
        w = self.weight
        share = float(w[boosted].sum() / w[keep].sum()) if keep.any() else 0.0
        per_task: dict[str, dict[str, int]] = defaultdict(lambda: {"failures": 0, "coverage_gaps": 0})
        for e in self.episodes:
            per_task[e.task]["failures"] += 1
            per_task[e.task]["coverage_gaps"] += int(e.coverage_gap)
        methods = defaultdict(int)
        for e in self.episodes:
            methods[e.method] += 1
        return {
            "frames": int(len(fi)),
            "kept_frames": int(keep.sum()),
            "boosted_frames": int(boosted.sum()),
            # Share of round-2 draws that land on boosted frames vs. their share under uniform.
            "boosted_sample_share": round(share, 5),
            "boosted_uniform_share": round(float(boosted.sum() / max(1, keep.sum())), 5),
            "weight_min": round(float(w[keep].min()), 4),
            "weight_max": round(float(w[keep].max()), 4),
            "weight_mean": round(float(w[keep].mean()), 6),
            "failures_mined": len(self.episodes),
            "coverage_gaps": int(sum(e.coverage_gap for e in self.episodes)),
            "divergence_methods": dict(methods),
            "per_task": dict(per_task),
            "config": asdict(cfg),
            "rollouts": self.rollouts,
        }


def mine_failures(fi: FrameIndex, episodes: Iterable[RolloutEpisode], cfg: MiningConfig) -> MiningResult:
    cfg.validate()
    lookup = _task_lookup(fi)
    total = np.zeros(len(fi), dtype=np.float64)
    records: list[EpisodeMining] = []
    unmatched_tasks: set[str] = set()
    for ep in episodes:
        if ep.success:
            continue
        task_index = None
        if cfg.same_task_only:
            task_index = lookup.get(_normalize_text(ep.task_description)) if ep.task_description else None
            if task_index is None:
                unmatched_tasks.add(ep.task_description or ep.task)
        rows, vals, rec = mine_episode(fi, ep, cfg, task_index)
        np.add.at(total, rows, vals)
        records.append(rec)
    if unmatched_tasks:
        logger.warning("No dataset task matches %s; searched all tasks for those", sorted(unmatched_tasks))
    peak = float(total.max()) if len(total) else 0.0
    score = total / peak if peak > 0 else total
    weight = compose_weights(score, fi.keep, cfg)
    return MiningResult(score=score, weight=weight, episodes=records, rollouts={"unmatched_tasks": sorted(unmatched_tasks)})


# ---------------------------------------------------------------------------------------------
# Write-back
# ---------------------------------------------------------------------------------------------
def write_flywheel_columns(
    lance_root: str | Path,
    fi: FrameIndex,
    result: MiningResult,
    cfg: MiningConfig,
    *,
    provenance: Mapping[str, Any] | None = None,
    storage_options: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Append ``score_column`` / ``weight_column`` (float32) to ``frames.lance`` (latest)."""
    t0 = time.perf_counter()
    meter = _WriteMeter(lance_root)
    _, table = _open_frames(lance_root, storage_options)
    version_before = int(table.version)
    # Rows must line up with the physical order used for mining: same rows, same order.
    cur = table.search().select(["index"]).limit(None).to_arrow()
    if not np.array_equal(_col_to_numpy(cur, "index", np.int64), fi.index):
        raise RuntimeError("frames.lance rows changed since mining (re-run mining on the current table)")
    info = {
        "format": FLYWHEEL_FORMAT,
        "mined_at_frames_version": fi.frames_version,
        "embedding_column": cfg.embedding_column,
        "config": asdict(cfg),
        **dict(provenance or {}),
    }
    md = {FLYWHEEL_FIELD_METADATA_KEY: json.dumps(info, sort_keys=True, default=str).encode("utf-8")}
    schema = pa.schema(
        [
            pa.field(cfg.score_column, pa.float32(), nullable=False, metadata=md),
            pa.field(cfg.weight_column, pa.float32(), nullable=False, metadata=md),
        ]
    )
    new_cols = pa.Table.from_arrays(
        [pa.array(result.score.astype(np.float32)), pa.array(result.weight.astype(np.float32))], schema=schema
    )
    dropped = _drop_columns_and_indices(table, [cfg.score_column, cfg.weight_column])
    table.add_columns(new_cols)
    return {
        "frames_version_before": version_before,
        "frames_version": int(table.version),
        "dropped_columns": dropped,
        "columns": [cfg.score_column, cfg.weight_column],
        "write_wall_s": round(time.perf_counter() - t0, 4),
        **meter.stop(),
    }


def run_mining(
    lance_root: str | Path,
    rollout_dirs: Sequence[str | Path],
    cfg: MiningConfig,
    *,
    write: bool = True,
    storage_options: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    fi = load_frame_index(
        lance_root,
        embedding_column=cfg.embedding_column,
        valid_where=cfg.valid_where,
        storage_options=storage_options,
    )
    t_load = time.perf_counter() - t0

    def episodes() -> Iterable[RolloutEpisode]:
        for d in rollout_dirs:
            yield from iter_episodes(d, failures_only=True)

    rollout_stats = []
    for d in rollout_dirs:
        recs = read_summary(d)
        rollout_stats.append(
            {"dir": str(d), "episodes": len(recs), "failures": sum(1 for r in recs if not r["success"])}
        )
    t1 = time.perf_counter()
    result = mine_failures(fi, episodes(), cfg)
    t_mine = time.perf_counter() - t1
    result.rollouts["sources"] = rollout_stats
    report: dict[str, Any] = {
        "lance_root": str(lance_root),
        "mined_frames_version": fi.frames_version,
        "load_s": round(t_load, 3),
        "mine_s": round(t_mine, 3),
        **result.summary(fi, cfg),
        "episodes": [asdict(e) for e in result.episodes],
    }
    if write:
        report["write"] = write_flywheel_columns(
            lance_root,
            fi,
            result,
            cfg,
            provenance={"rollouts": rollout_stats},
            storage_options=storage_options,
        )
    return report


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------
def _parse_kv(items: list[str] | None) -> dict[str, str] | None:
    if not items:
        return None
    out = {}
    for it in items:
        k, _, v = it.partition("=")
        out[k] = v
    return out


def add_mining_args(p: argparse.ArgumentParser) -> None:
    d = MiningConfig()
    p.add_argument("--lance-root", required=True)
    p.add_argument("--rollouts", nargs="+", required=True, help="rollout dirs (record_libero_rollouts.py output)")
    p.add_argument("--embedding-column", default=d.embedding_column)
    p.add_argument("--top-k", type=int, default=d.top_k)
    p.add_argument("--min-match-sim", type=float, default=d.min_match_sim)
    p.add_argument("--divergence-drop", type=float, default=d.divergence_drop)
    p.add_argument("--smooth", type=int, default=d.smooth)
    p.add_argument("--window-before", type=int, default=d.window_before)
    p.add_argument("--window-after", type=int, default=d.window_after)
    p.add_argument("--decay-tau", type=float, default=d.decay_tau)
    p.add_argument("--spread-frames", type=int, default=d.spread_frames)
    p.add_argument("--alpha", type=float, default=d.alpha)
    p.add_argument("--w-min", type=float, default=d.w_min)
    p.add_argument("--w-max", type=float, default=d.w_max)
    p.add_argument("--valid-where", default=None)
    p.add_argument("--all-tasks", action="store_true", help="match failures against every task's frames")
    p.add_argument("--score-column", default=d.score_column)
    p.add_argument("--weight-column", default=d.weight_column)
    p.add_argument("--dry-run", action="store_true", help="mine and report, do not write columns")
    p.add_argument("--out-json", default=None)
    p.add_argument("--storage-option", action="append", default=None, metavar="KEY=VALUE")


def mining_config_from_args(a: argparse.Namespace) -> MiningConfig:
    return MiningConfig(
        embedding_column=a.embedding_column,
        top_k=a.top_k,
        min_match_sim=a.min_match_sim,
        divergence_drop=a.divergence_drop,
        smooth=a.smooth,
        window_before=a.window_before,
        window_after=a.window_after,
        decay_tau=a.decay_tau,
        spread_frames=a.spread_frames,
        alpha=a.alpha,
        w_min=a.w_min,
        w_max=a.w_max,
        valid_where=a.valid_where,
        same_task_only=not a.all_tasks,
        score_column=a.score_column,
        weight_column=a.weight_column,
    )


def cli_mine(a: argparse.Namespace) -> dict[str, Any]:
    report = run_mining(
        a.lance_root,
        a.rollouts,
        mining_config_from_args(a),
        write=not a.dry_run,
        storage_options=_parse_kv(a.storage_option),
    )
    text = json.dumps(report, indent=2, default=str)
    if a.out_json:
        Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out_json).write_text(text, encoding="utf-8")
    brief = {k: v for k, v in report.items() if k not in ("episodes", "config", "per_task")}
    print(json.dumps(brief, indent=2, default=str))
    return report


__all__ = [
    "SCORE_COLUMN",
    "WEIGHT_COLUMN",
    "EpisodeMining",
    "FrameIndex",
    "MiningConfig",
    "MiningResult",
    "compose_weights",
    "embed_rollout",
    "find_divergence",
    "load_frame_index",
    "mine_episode",
    "mine_failures",
    "rollout_features",
    "run_mining",
    "write_flywheel_columns",
]
