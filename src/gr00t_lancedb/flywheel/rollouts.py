# SPDX-License-Identifier: Apache-2.0
"""On-disk format for simulator rollouts recorded for failure mining and paired evaluation.

One ``.npz`` per episode plus one ``summary.jsonl`` line per episode::

    <rollout_dir>/summary.jsonl
    <rollout_dir>/<task_slug>/ep_<ordinal>.npz

Each ``.npz`` holds, per *decision point* ``k`` (one policy call):

* ``env_steps``      int64 ``[K]``   env-step index at which the decision was taken
* ``n_env_steps``    int64 ``[K]``   env steps actually executed from that decision
* ``state/<key>``    float32 ``[K, d]``  observed ``state.<key>`` (last frame)
* ``action/<key>``   float32 ``[K, n_action_steps, d]``  action chunk returned by the policy
* ``image``          uint8 ``[K, h, w, 3]`` (optional) thumbnail of the main camera
* ``meta``           JSON (``task``, ``task_description``, ``ordinal``, ``seed``, ``success`` ...)

Keys are the environment's own ``state.*`` / ``action.*`` names (``x``, ``gripper`` ...), which
for GR00T simulator envs equal the ``modality.json`` keys of the training dataset, so the
mining step can rebuild full ``observation.state`` / ``action`` vectors without the simulator.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

ROLLOUT_FORMAT = "gr00t_lancedb.rollout/v1"
SUMMARY_FILENAME = "summary.jsonl"


def task_slug(task: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", task.split("/")[-1]).strip("_") or "task"


@dataclass
class RolloutEpisode:
    task: str
    ordinal: int
    seed: int
    success: bool
    env_steps: np.ndarray
    n_env_steps: np.ndarray
    state: dict[str, np.ndarray]
    action: dict[str, np.ndarray]
    task_description: str = ""
    image: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def num_decisions(self) -> int:
        return int(len(self.env_steps))

    @property
    def total_env_steps(self) -> int:
        return int(self.n_env_steps.sum()) if len(self.n_env_steps) else 0

    def summary(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "task_description": self.task_description,
            "ordinal": int(self.ordinal),
            "seed": int(self.seed),
            "success": bool(self.success),
            "env_steps": self.total_env_steps,
            "decisions": self.num_decisions,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"format": ROLLOUT_FORMAT, **self.meta, **self.summary()}
        arrays: dict[str, np.ndarray] = {
            "env_steps": np.asarray(self.env_steps, dtype=np.int64),
            "n_env_steps": np.asarray(self.n_env_steps, dtype=np.int64),
            "meta": np.frombuffer(json.dumps(meta, sort_keys=True).encode("utf-8"), dtype=np.uint8),
        }
        for k, v in self.state.items():
            arrays[f"state/{k}"] = np.asarray(v, dtype=np.float32)
        for k, v in self.action.items():
            arrays[f"action/{k}"] = np.asarray(v, dtype=np.float32)
        if self.image is not None:
            arrays["image"] = np.asarray(self.image, dtype=np.uint8)
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> RolloutEpisode:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(bytes(z["meta"]).decode("utf-8"))
            if meta.get("format") != ROLLOUT_FORMAT:
                raise ValueError(f"{path}: not a {ROLLOUT_FORMAT} file (format={meta.get('format')!r})")
            state = {k.split("/", 1)[1]: z[k] for k in z.files if k.startswith("state/")}
            action = {k.split("/", 1)[1]: z[k] for k in z.files if k.startswith("action/")}
            image = z["image"] if "image" in z.files else None
            return cls(
                task=meta["task"],
                ordinal=int(meta["ordinal"]),
                seed=int(meta["seed"]),
                success=bool(meta["success"]),
                env_steps=z["env_steps"],
                n_env_steps=z["n_env_steps"],
                state=state,
                action=action,
                task_description=meta.get("task_description", ""),
                image=image,
                meta=meta,
            )

    def executed_actions(self, key: str) -> np.ndarray:
        """``[total_env_steps, d]`` actions actually sent to the env for ``action.<key>``."""
        chunks = self.action[key]
        parts = [chunks[k, : int(n)] for k, n in enumerate(self.n_env_steps)]
        if not parts:
            return np.zeros((0, chunks.shape[-1]), dtype=np.float32)
        return np.concatenate(parts, axis=0)


def episode_path(rollout_dir: str | Path, task: str, ordinal: int) -> Path:
    return Path(rollout_dir) / task_slug(task) / f"ep_{int(ordinal):04d}.npz"


def append_summary(rollout_dir: str | Path, record: Mapping[str, Any]) -> None:
    path = Path(rollout_dir) / SUMMARY_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(record), sort_keys=True) + "\n")


def read_summary(rollout_dir: str | Path) -> list[dict[str, Any]]:
    """Summary records; a re-recorded ``(task, ordinal)`` keeps its last line."""
    path = Path(rollout_dir) / SUMMARY_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"no {SUMMARY_FILENAME} in {rollout_dir}")
    latest: dict[tuple[str, int], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            latest[(rec["task"], int(rec["ordinal"]))] = rec
    return [latest[k] for k in sorted(latest)]


def iter_episodes(rollout_dir: str | Path, *, failures_only: bool = False) -> Iterator[RolloutEpisode]:
    for rec in read_summary(rollout_dir):
        if failures_only and rec["success"]:
            continue
        path = episode_path(rollout_dir, rec["task"], rec["ordinal"])
        if path.exists():
            yield RolloutEpisode.load(path)


def modality_vector(
    parts: Mapping[str, np.ndarray],
    modality_section: Mapping[str, Any],
    original_key: str,
    width: int,
    fill: np.ndarray,
    *,
    default_original_key: str | None = None,
) -> np.ndarray:
    """Rebuild ``[..., width]`` values of one dataset column from per-key arrays.

    Every ``modality.json`` key mapped to ``original_key`` is written at its ``[start:end]``
    slice; dimensions no key covers keep ``fill`` (the dataset mean, i.e. z-score 0).
    """
    lead = None
    placed = []
    for key, spec in modality_section.items():
        spec_key = spec.get("original_key") or default_original_key or original_key
        if spec_key != original_key or key not in parts:
            continue
        arr = np.asarray(parts[key], dtype=np.float64)
        start, end = int(spec["start"]), int(spec["end"])
        if arr.shape[-1] != end - start:
            raise ValueError(f"{key}: expected last dim {end - start}, got {arr.shape}")
        lead = arr.shape[:-1] if lead is None else lead
        if arr.shape[:-1] != lead:
            raise ValueError(f"{key}: leading shape {arr.shape[:-1]} != {lead}")
        placed.append((start, end, arr))
    if lead is None:
        raise KeyError(f"no rollout keys map to {original_key!r}")
    out = np.broadcast_to(np.asarray(fill, dtype=np.float64), (*lead, width)).copy()
    for start, end, arr in placed:
        out[..., start:end] = arr
    return out


__all__ = [
    "ROLLOUT_FORMAT",
    "SUMMARY_FILENAME",
    "RolloutEpisode",
    "append_summary",
    "episode_path",
    "iter_episodes",
    "modality_vector",
    "read_summary",
    "task_slug",
]
