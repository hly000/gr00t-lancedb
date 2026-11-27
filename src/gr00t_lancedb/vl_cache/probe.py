# SPDX-License-Identifier: Apache-2.0
"""Held-out flow-matching probe loss: a 2-second, low-variance stand-in for closed-loop eval.

A 50-episode LIBERO run costs 16 GPU-minutes and has a 95 % interval of roughly +-11 points,
and the training loss printed by the loop is one random ``(t, noise)`` draw per sample computed
in bf16. Neither can rank two configurations that differ by a few percent.

The probe fixes both ends: a set of frames from episodes the trainer never samples, and the
same ``(t, noise)`` for every checkpoint (seeded, eval mode, loss reduced in fp32). It works
for any checkpoint whose action head can read the cache, including a natively trained one, so
"native head on cached features" vs "cache-trained head on cached features" is one command.

It is a proxy. A lower probe loss has not been shown to imply a higher success rate here;
use it to reject configurations and to find regressions, then confirm with closed-loop eval.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

SCALAR_KEYS = ("state", "action", "action_mask", "embodiment_id")


def split_holdout_rows(
    episode_index: np.ndarray,
    frac: float,
    seed: int,
    max_probe_rows: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """Split cache rows into ``(train_rows, probe_rows)`` by whole episodes.

    ``frac`` of the episodes (at least one) are held out; every row of a held-out episode is
    removed from training and up to ``max_probe_rows`` of them form the probe. ``frac <= 0``
    keeps all rows for training and draws an in-sample probe instead. Deterministic in
    ``(episode_index, frac, seed, max_probe_rows)``, so a later evaluation rebuilds the same probe.
    """
    episode_index = np.asarray(episode_index)
    n = int(episode_index.shape[0])
    rng = np.random.default_rng([int(seed), 0x50524F42])
    if frac <= 0:
        probe = np.sort(rng.choice(n, size=min(max_probe_rows, n), replace=False))
        return np.arange(n, dtype=np.int64), probe.astype(np.int64)
    episodes = np.unique(episode_index)
    n_hold = min(len(episodes) - 1, max(1, round(len(episodes) * float(frac))))
    if n_hold < 1:
        raise ValueError("need at least two episodes to hold one out")
    held = rng.choice(episodes, size=n_hold, replace=False)
    is_probe = np.isin(episode_index, held)
    probe = np.flatnonzero(is_probe)
    if len(probe) > max_probe_rows:
        probe = np.sort(rng.choice(probe, size=max_probe_rows, replace=False))
    return np.flatnonzero(~is_probe).astype(np.int64), probe.astype(np.int64)


def load_scalars(path: str) -> dict[str, np.ndarray]:
    """Load the scalar side-car; float16 state/action from older caches are widened to fp32."""
    npz = np.load(path)
    out = {k: npz[k] for k in SCALAR_KEYS}
    out["state"] = out["state"].astype(np.float32, copy=False)
    out["action"] = out["action"].astype(np.float32, copy=False)
    return out


def load_cached_batch(reader: Any, scalars: dict[str, np.ndarray], row_ids: Sequence[int]) -> dict[str, Any]:
    """One CPU batch: compact VL payload from Lance plus fp32 scalars for the same rows."""
    import torch

    from .store import collate_cached_vl_batch

    rows = reader.take_rows(list(row_ids))
    batch = collate_cached_vl_batch(
        rows,
        codec=reader.fingerprint.quant_codec,
        pad_side=reader.fingerprint.pad_side,
        dequantize_on_cpu=False,
        device=None,
    )
    idx = np.asarray(row_ids, dtype=np.int64)
    batch["state"] = torch.from_numpy(scalars["state"][idx])
    batch["action"] = torch.from_numpy(scalars["action"][idx])
    batch["action_mask"] = torch.from_numpy(scalars["action_mask"][idx])
    batch["embodiment_id"] = torch.from_numpy(scalars["embodiment_id"][idx].astype(np.int64))
    return batch


def build_probe_batches(
    reader: Any, scalars: dict[str, np.ndarray], probe_rows: np.ndarray, batch_size: int = 64
) -> list[dict[str, Any]]:
    rows = [int(r) for r in probe_rows]
    return [load_cached_batch(reader, scalars, rows[i : i + batch_size]) for i in range(0, len(rows), batch_size)]


def head_inputs_from_cached(
    batch: dict[str, Any], device: Any, feature_dtype: Any | None = None, scalar_dtype: Any | None = None
) -> tuple[Any, Any]:
    """Cached CPU batch -> ``(backbone_output, action_input)`` for ``Gr00tN1d7ActionHead``.

    State, action and mask stay fp32, as in the native trainer where ``autocast`` casts inside
    each layer: the flow-matching time, the noisy trajectory and the velocity target are then
    built in fp32 instead of bf16.
    """
    import torch
    from transformers import BatchFeature

    from .codec import dequantize_features_torch

    if feature_dtype is None:
        feature_dtype = torch.bfloat16
    if scalar_dtype is None:
        scalar_dtype = torch.float32
    payload = batch["feat_payload"].to(device, non_blocking=True)
    scale = batch["feat_scale"]
    scale = scale.to(device, non_blocking=True) if scale is not None else None
    feats = dequantize_features_torch(payload, scale, codec=batch["quant_codec"], target_dtype=feature_dtype)
    bb_out = BatchFeature(
        data={
            "backbone_features": feats,
            "backbone_attention_mask": batch["backbone_attention_mask"].to(device, non_blocking=True),
            "image_mask": batch["image_mask"].to(device, non_blocking=True),
        }
    )
    act_in = BatchFeature(
        data={
            "state": batch["state"].to(device, dtype=scalar_dtype, non_blocking=True),
            "action": batch["action"].to(device, dtype=scalar_dtype, non_blocking=True),
            "action_mask": batch["action_mask"].to(device, dtype=scalar_dtype, non_blocking=True),
            "embodiment_id": batch["embodiment_id"].to(device, non_blocking=True),
        }
    )
    return bb_out, act_in


def probe_fm_loss(
    action_head: Any,
    probe_batches: list[dict[str, Any]],
    *,
    device: Any,
    seeds: Sequence[int] = (0, 1),
    amp_dtype: Any | None = None,
    scalar_dtype: Any | None = None,
    return_per_sample: bool = False,
) -> dict[str, Any]:
    """Mean per-sample flow-matching loss on the probe with fixed ``(t, noise)``.

    Runs in eval mode (no state dropout, no DiT dropout) under ``no_grad`` and restores both
    the train/eval flag and the global RNG state, so calling it mid-training does not change
    the training stream.
    """
    import torch

    was_training = bool(action_head.training)
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    per_sample: list[Any] = []
    try:
        action_head.eval()
        with torch.no_grad():
            for seed in seeds:
                for bi, batch in enumerate(probe_batches):
                    torch.manual_seed(1_000_003 * (int(seed) + 1) + bi)
                    bb_out, act_in = head_inputs_from_cached(batch, device, scalar_dtype=scalar_dtype)
                    if amp_dtype is not None:
                        with torch.autocast(device_type=torch.device(device).type, dtype=amp_dtype):
                            out = action_head(bb_out, act_in)
                    else:
                        out = action_head(bb_out, act_in)
                    if "action_loss" in out and "action_mask" in out:
                        num = out["action_loss"].float().flatten(1).sum(1)
                        den = out["action_mask"].float().flatten(1).sum(1).clamp_min(1.0)
                        per_sample.append((num / den).cpu())
                    else:
                        per_sample.append(out["loss"].float().reshape(1).cpu())
    finally:
        action_head.train(was_training)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
    losses = torch.cat(per_sample).double()
    n = int(losses.numel())
    out_d: dict[str, Any] = {
        "probe_loss": float(losses.mean()),
        "probe_sem": float(losses.std(unbiased=True) / math.sqrt(n)) if n > 1 else float("nan"),
        "probe_samples": n,
        "probe_seeds": [int(s) for s in seeds],
    }
    if return_per_sample:
        out_d["per_sample"] = losses.numpy()
    return out_d


def paired_probe_diff(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """``b - a`` on identical samples and noise: mean, standard error and z-score of the difference."""
    d = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    sem = float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float("nan")
    mean = float(d.mean())
    return {"diff": mean, "sem": sem, "z": mean / sem if sem and sem > 0 else float("nan")}


def prepare_probe(
    cache_dir: str,
    table_name: str,
    scalars_path: str,
    holdout_frac: float,
    seed: int,
    max_probe_rows: int = 1024,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Open the cache, split rows by episode and load the probe batches (touches Lance)."""
    from .store import VLCacheReader

    reader = VLCacheReader(cache_dir, table_name)
    n = reader.num_rows
    scalars = load_scalars(scalars_path)
    if int(scalars["state"].shape[0]) != n:
        raise ValueError(f"scalar side-car has {scalars['state'].shape[0]} rows, Lance table has {n}")
    if "episode_index" in np.load(scalars_path).files:
        episode_index = np.load(scalars_path)["episode_index"]
    else:  # caches written before the side-car carried it
        tbl = reader.take_rows_arrow(range(n), columns=["episode_index"])  # physical row order
        episode_index = tbl["episode_index"].to_numpy(zero_copy_only=False)
    train_rows, probe_rows = split_holdout_rows(episode_index, holdout_frac, seed, max_probe_rows)
    return {
        "num_rows": n,
        "train_rows": train_rows,
        "probe_rows": probe_rows,
        "held_out_episodes": int(len(np.unique(episode_index))) - int(len(np.unique(episode_index[train_rows]))),
        "probe_batches": build_probe_batches(reader, scalars, probe_rows, batch_size),
    }


def prepare_probe_isolated(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """:func:`prepare_probe` in a spawned child, so the training process never opens Lance
    before it forks DataLoader workers (the readers there open lazily, post-fork)."""
    import multiprocessing as mp

    with mp.get_context("spawn").Pool(1) as pool:
        return pool.apply(prepare_probe, args, kwargs)


def first_working(attempts: Sequence[Any], run: Any, log: Any = print) -> Any:
    """Return the first item of ``attempts`` for which ``run(item)`` does not raise.

    Earlier failures are logged, the last one propagates. Used to fall back from a compiled
    to an eager model (and from fp32 to bf16 inputs) before training starts instead of
    crashing at step 1.
    """
    if not attempts:
        raise ValueError("no attempts given")
    for i, item in enumerate(attempts):
        try:
            run(item)
            return item
        except Exception as exc:  # noqa: BLE001 - compiler failures come in many types
            if i + 1 == len(attempts):
                raise
            log(f"[fallback] {item!r} failed: {type(exc).__name__}: {str(exc)[:300]} -- trying {attempts[i + 1]!r}")
    raise AssertionError("unreachable")
