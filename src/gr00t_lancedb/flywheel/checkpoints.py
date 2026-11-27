# SPDX-License-Identifier: Apache-2.0
"""Keep selected checkpoints out of the HF ``save_total_limit`` rotation.

The fast-validation protocol needs four checkpoints of the baseline run: three for the
learning-curve evaluation (weights only) and the fork point (full state: optimizer,
scheduler, RNG) that round 2 resumes from. A 3B GR00T checkpoint with optimizer state is
~24 GB, so saving every 1,000 steps with no rotation does not fit a 200 GB disk.

:class:`KeepCheckpointsCallback` hard-links the wanted ``checkpoint-<step>`` directories
into ``<output_dir>/keep/`` right after they are written (no extra bytes until rotation
deletes the original), optionally leaving out the optimizer / scheduler / RNG files.
"""

from __future__ import annotations

from collections.abc import Iterable
import json
import logging
import os
from pathlib import Path
import shutil
from typing import Any

logger = logging.getLogger(__name__)

KEEP_DIRNAME = "keep"
TRAINING_STATE_PATTERNS = ("optimizer", "scheduler", "rng_state", "emergency_trainable")


def parse_steps(raw: str | None) -> set[int]:
    """``"5000,10000,14000"`` -> ``{5000, 10000, 14000}`` (empty / None -> empty set)."""
    if not raw:
        return set()
    out = set()
    for part in raw.replace(" ", "").split(","):
        if part:
            step = int(part)
            if step <= 0:
                raise ValueError(f"checkpoint steps must be > 0, got {step}")
            out.add(step)
    return out


def _is_training_state(name: str) -> bool:
    return any(name.startswith(p) for p in TRAINING_STATE_PATTERNS)


def keep_checkpoint(src: Path, dest: Path, *, weights_only: bool) -> dict[str, Any]:
    """Hard-link (copy across filesystems) ``src`` into ``dest``; returns a small manifest."""
    if dest.exists():
        shutil.rmtree(dest)
    files = linked = copied = skipped = 0
    for path in sorted(src.rglob("*")):
        rel = path.relative_to(src)
        if path.is_dir():
            continue
        if weights_only and _is_training_state(rel.parts[0]):
            skipped += 1
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(path, target)
            linked += 1
        except OSError:
            shutil.copy2(path, target)
            copied += 1
        files += 1
    manifest = {
        "source": str(src),
        "weights_only": bool(weights_only),
        "files": files,
        "hardlinked": linked,
        "copied": copied,
        "skipped_training_state_files": skipped,
    }
    (dest / "keep_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _make_callback_base():
    try:
        from transformers import TrainerCallback
    except ImportError:  # pragma: no cover - transformers always present next to GR00T
        return object
    return TrainerCallback


class KeepCheckpointsCallback(_make_callback_base()):  # type: ignore[misc]
    """``on_save``: link ``checkpoint-<step>`` into ``keep/`` for the configured steps."""

    def __init__(self, keep_steps: Iterable[int], weights_only_steps: Iterable[int] = ()) -> None:
        self.keep_steps = set(int(s) for s in keep_steps) | set(int(s) for s in weights_only_steps)
        self.weights_only_steps = set(int(s) for s in weights_only_steps)
        self.kept: dict[int, dict[str, Any]] = {}

    @classmethod
    def from_env(cls) -> KeepCheckpointsCallback | None:
        full = parse_steps(os.environ.get("GR00T_KEEP_CKPT_STEPS"))
        weights_only = parse_steps(os.environ.get("GR00T_KEEP_CKPT_WEIGHTS_ONLY"))
        if not (full or weights_only):
            return None
        return cls(full - weights_only, weights_only)

    def on_save(self, args, state, control, **kwargs):  # noqa: D401 - HF callback signature
        step = int(state.global_step)
        if step not in self.keep_steps or not getattr(state, "is_world_process_zero", True):
            return control
        src = Path(args.output_dir) / f"checkpoint-{step}"
        if not src.is_dir():
            logger.warning("KeepCheckpointsCallback: %s not found (save_steps must divide %d)", src, step)
            return control
        dest = Path(args.output_dir) / KEEP_DIRNAME / f"checkpoint-{step}"
        manifest = keep_checkpoint(src, dest, weights_only=step in self.weights_only_steps)
        self.kept[step] = manifest
        logger.info("Kept %s -> %s (%s)", src, dest, "weights only" if manifest["weights_only"] else "full")
        return control


__all__ = ["KEEP_DIRNAME", "KeepCheckpointsCallback", "keep_checkpoint", "parse_steps"]
