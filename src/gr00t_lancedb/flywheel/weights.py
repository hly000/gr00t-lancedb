# SPDX-License-Identifier: Apache-2.0
"""Per-anchor sampling weights read from a ``frames.lance`` column.

The flywheel's round 2 trains on the *same* anchors as the baseline, but draws them with
probability proportional to a per-frame weight column (``flywheel_weight`` by default)
instead of the uniform shuffle-buffer order. Sampling is with replacement and fully
determined by ``(seed, rank, logical worker)``, so a run that forks from a baseline
checkpoint resumes exactly like any other exact-resume run: skipping ``n`` served samples
only replays ``n`` integer draws, no decoding.

Nothing here imports ``lancedb``: the weight column is fetched by the isolated child
(:func:`gr00t_lancedb.lance_query.query_anchor_weights`) and arrives as plain numpy.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
import hashlib
from typing import Any

import numpy as np

_FRAME_KEY_SHIFT = np.int64(1) << np.int64(32)
_STREAM_SALT = 0x46574C59  # "FWLY": keeps the weighted stream disjoint from other seeded RNGs


@dataclass(frozen=True)
class EpisodeAnchors:
    """Valid anchors of one positional episode of one dataset."""

    position: int  # GR00T positional episode index (what ``load_split`` / ``load_anchor_batch`` take)
    episode_index: int  # Lance ``episode_index``
    steps: np.ndarray  # valid anchor ``frame_index`` values, sorted


@dataclass(frozen=True)
class AnchorWeightMap:
    """One dataset's valid anchors with their raw (unnormalized) weights."""

    ep_pos: np.ndarray  # int64 [N] positional episode index
    step: np.ndarray  # int64 [N] frame_index
    weight: np.ndarray  # float64 [N] >= 0
    missing: int = 0  # anchors without a row/value in the column (got ``default``)
    nonfinite: int = 0  # anchors whose stored value was NaN/inf (got ``default``)

    def __len__(self) -> int:
        return int(len(self.ep_pos))


def dataset_episode_anchors(ds: Any) -> list[EpisodeAnchors]:
    """Every positional episode of a ``LanceShardedSingleStepDataset`` with its valid anchors."""
    loader = ds.episode_loader
    out: list[EpisodeAnchors] = []
    for pos in range(len(loader)):
        ep_id = int(loader.episodes_metadata[pos]["episode_index"])
        out.append(EpisodeAnchors(pos, ep_id, np.asarray(ds.get_valid_steps_for_episode(pos), dtype=np.int64)))
    return out


def map_frame_weights(
    episodes: Sequence[EpisodeAnchors],
    episode_index: np.ndarray,
    frame_index: np.ndarray,
    weight: np.ndarray,
    *,
    default: float = 1.0,
) -> AnchorWeightMap:
    """Look up each valid anchor's weight by ``(episode_index, frame_index)``.

    Anchors with no matching row, or with a NaN/inf value, get ``default``. Negative
    weights are rejected: a weight column is a sampling distribution, not a score.
    """
    episode_index = np.asarray(episode_index, dtype=np.int64)
    frame_index = np.asarray(frame_index, dtype=np.int64)
    weight = np.asarray(weight, dtype=np.float64)
    if not (len(episode_index) == len(frame_index) == len(weight)):
        raise ValueError("episode_index, frame_index and weight must have the same length")
    finite = np.isfinite(weight)
    if np.any(weight[finite] < 0):
        raise ValueError("sampling weights must be >= 0")
    if not np.isfinite(default) or default < 0:
        raise ValueError(f"default weight must be finite and >= 0, got {default}")

    keys = episode_index * _FRAME_KEY_SHIFT + frame_index
    order = np.argsort(keys, kind="stable")
    keys_sorted = keys[order]
    w_sorted = weight[order]
    finite_sorted = finite[order]

    ep_parts, step_parts, w_parts = [], [], []
    missing = nonfinite = 0
    for ep in episodes:
        steps = np.asarray(ep.steps, dtype=np.int64)
        if len(steps) == 0:
            continue
        q = np.int64(ep.episode_index) * _FRAME_KEY_SHIFT + steps
        j = np.searchsorted(keys_sorted, q)
        jc = np.minimum(j, max(len(keys_sorted) - 1, 0))
        found = (j < len(keys_sorted)) & (keys_sorted[jc] == q) if len(keys_sorted) else np.zeros(len(q), bool)
        good = found & finite_sorted[jc] if len(keys_sorted) else found
        w = np.full(len(steps), float(default), dtype=np.float64)
        w[good] = w_sorted[jc[good]]
        missing += int((~found).sum())
        nonfinite += int((found & ~good).sum())
        ep_parts.append(np.full(len(steps), ep.position, dtype=np.int64))
        step_parts.append(steps)
        w_parts.append(w)

    if not ep_parts:
        empty = np.empty(0, dtype=np.int64)
        return AnchorWeightMap(empty, empty, np.empty(0, dtype=np.float64), missing, nonfinite)
    return AnchorWeightMap(
        np.concatenate(ep_parts), np.concatenate(step_parts), np.concatenate(w_parts), missing, nonfinite
    )


@dataclass(frozen=True)
class WeightedAnchors:
    """All datasets' anchors with a global sampling CDF (dataset mix x per-anchor weight)."""

    ds_index: np.ndarray  # int32 [N]
    ep_pos: np.ndarray  # int64 [N]
    step: np.ndarray  # int64 [N]
    cdf: np.ndarray  # float64 [N], non-decreasing, cdf[-1] == 1
    column: str = ""
    info: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(len(self.cdf))

    def probabilities(self) -> np.ndarray:
        return np.diff(self.cdf, prepend=0.0)

    @property
    def digest(self) -> str:
        h = hashlib.sha256()
        for arr in (self.ds_index, self.ep_pos, self.step, self.cdf):
            h.update(np.ascontiguousarray(arr).tobytes())
        return h.hexdigest()[:16]

    def stream(self, seed: int, rank: int, worker: int, *, block: int = 4096) -> Iterator[tuple[int, int, int]]:
        """Infinite ``(dataset_index, ep_pos, step)`` draws for one ``(rank, logical worker)``.

        Draws are i.i.d. with replacement. ``block`` only groups RNG calls; the stream is the
        same for any block size because NumPy's ``random(n)`` fills sequentially.
        """
        if len(self) == 0:
            raise ValueError("WeightedAnchors is empty")
        rng = np.random.default_rng([int(seed), int(rank), int(worker), _STREAM_SALT])
        last = len(self) - 1
        while True:
            idx = np.searchsorted(self.cdf, rng.random(block), side="right")
            np.minimum(idx, last, out=idx)
            ds = self.ds_index[idx]
            ep = self.ep_pos[idx]
            st = self.step[idx]
            for i in range(block):
                yield int(ds[i]), int(ep[i]), int(st[i])

    def summary(self) -> dict[str, Any]:
        p = self.probabilities()
        n = len(self)
        uniform = 1.0 / n if n else 0.0
        top = np.sort(p)[::-1]
        k = max(1, n // 100)
        return {
            "column": self.column,
            "anchors": n,
            "zero_weight_anchors": int((p <= 0).sum()),
            "max_over_uniform": float(top[0] / uniform) if n else 0.0,
            "top1pct_mass": float(top[:k].sum()) if n else 0.0,
            "effective_sample_size_frac": float(1.0 / (n * np.square(p).sum())) if n else 0.0,
            "digest": self.digest,
            **self.info,
        }


def build_weighted_anchors(
    per_dataset: Sequence[AnchorWeightMap],
    mix_weights: Sequence[float],
    *,
    column: str = "",
    info: dict[str, Any] | None = None,
) -> WeightedAnchors:
    """Combine per-dataset weight maps into one CDF.

    Each dataset keeps its mixture share ``mix_weights[d] / sum(mix_weights)``; inside a
    dataset, anchors are drawn proportionally to their weight.
    """
    if len(per_dataset) != len(mix_weights):
        raise ValueError("per_dataset and mix_weights must have the same length")
    mix = np.asarray(mix_weights, dtype=np.float64)
    if np.any(~np.isfinite(mix)) or np.any(mix < 0) or mix.sum() <= 0:
        raise ValueError(f"invalid mixture weights {list(mix_weights)}")
    mix = mix / mix.sum()

    ds_parts, ep_parts, st_parts, p_parts = [], [], [], []
    for d, m in enumerate(per_dataset):
        total = float(m.weight.sum())
        if mix[d] == 0:
            continue
        if total <= 0:
            raise ValueError(f"dataset {d}: all anchor weights are zero")
        ds_parts.append(np.full(len(m), d, dtype=np.int32))
        ep_parts.append(m.ep_pos)
        st_parts.append(m.step)
        p_parts.append(m.weight / total * mix[d])
    if not p_parts:
        raise ValueError("no anchors with positive weight")
    p = np.concatenate(p_parts)
    cdf = np.cumsum(p)
    cdf /= cdf[-1]
    cdf[-1] = 1.0
    return WeightedAnchors(
        ds_index=np.concatenate(ds_parts),
        ep_pos=np.concatenate(ep_parts),
        step=np.concatenate(st_parts),
        cdf=cdf,
        column=column,
        info=dict(info or {}),
    )


__all__ = [
    "AnchorWeightMap",
    "EpisodeAnchors",
    "WeightedAnchors",
    "build_weighted_anchors",
    "dataset_episode_anchors",
    "map_frame_weights",
]
