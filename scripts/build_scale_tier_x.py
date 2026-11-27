#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Build the X scale tier (E1 / H1 / H4): replicate one LeRobot v2.1 dataset N times.

Only the size changes, never the content: replica ``r`` of source episode at position ``i``
becomes episode ``r * N_src + i``. Rewritten per replica:

- data parquets: ``episode_index`` and the global ``index`` column (new contiguous range),
  stored under the new chunk / file name from ``info.json``'s ``data_path``;
- videos: same bytes under the new chunk / file name (hard link by default, ``--link-mode
  symlink|copy``; ``--copy-videos`` is shorthand for ``copy``);
- ``meta/episodes.jsonl`` and ``meta/episodes_stats.jsonl`` (``episode_index`` plus the
  ``episode_index`` / ``index`` feature stats), ``meta/info.json`` (totals, chunks,
  splits), and the ``index`` / ``episode_index`` entries of ``meta/stats.json``.

State/action statistics are unchanged by exact replication (the same multiset of values,
repeated), so every other meta file is copied verbatim.

Optionally each tier is converted to Lance with ``convert_v2_to_lance`` in ``copy`` and/or
``external`` video mode. ``modality.json`` is embedded by the converter itself (taken from
``meta/`` or from ``--modality-json``), so no separate ``attach_gr00t_meta`` step is needed.

Layout::

    <out-root>/x<F>/v21/              replicated LeRobot v2.1 dataset
    <out-root>/x<F>/lance_copy/       Lance, videos stored inside videos.lance
    <out-root>/x<F>/lance_external/   Lance, videos referenced in place (v21 dir or
                                      --external-base-uri, e.g. s3://bucket/x_tier/x{factor}/v21)
    <out-root>/manifest.json          bytes / files / frames / episodes per tier and variant

``--lance-uri-template 's3://bucket/x_tier/x{factor}/lance_{mode}'`` writes the Lance
variants straight to the bucket instead (the 200x copy variant is ~128 GB).

Examples::

    python scripts/build_scale_tier_x.py --src /data/datasets/libero_10 \
        --out-root /data/datasets/x_tier --factors 1,10,50,200 --lance copy,external \
        --external-base-uri 's3://my-bucket/x_tier/x{factor}/v21' --dry-run

Hard links keep local disk use at ~1x; uploads still send every byte (``rclone copy`` /
``s5cmd cp`` read each link as a full file). With ``--link-mode symlink`` use
``rclone copy --copy-links``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import errno
import json
import math
import os
from pathlib import Path
import shutil
import sys
import threading
import time
from typing import Any

# Meta files rewritten here; everything else in meta/ is copied verbatim.
_REWRITTEN_META = {"info.json", "episodes.jsonl", "episodes_stats.jsonl", "stats.json"}
_SHIFTED_STATS = ("min", "max", "mean")  # stats that move with an index offset (std does not)


# ---------------------------------------------------------------------------
# Source description
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def describe_source(src: Path) -> dict[str, Any]:
    """Read the parts of a v2.1 dataset this script needs (no parquet/video reads)."""
    meta = src / "meta"
    info = json.loads((meta / "info.json").read_text(encoding="utf-8"))
    episodes = sorted(_load_jsonl(meta / "episodes.jsonl"), key=lambda r: int(r["episode_index"]))
    if not episodes:
        raise ValueError(f"{meta / 'episodes.jsonl'} is empty or missing")
    features = info.get("features", {})
    video_keys = [k for k, v in features.items() if isinstance(v, dict) and v.get("dtype") == "video"]
    if video_keys and not info.get("video_path"):
        raise ValueError("info.json lists video features but has no video_path")
    return {
        "info": info,
        "episodes": episodes,
        "episodes_stats": {
            int(r["episode_index"]): r for r in _load_jsonl(meta / "episodes_stats.jsonl")
        },
        "video_keys": video_keys,
        "chunks_size": int(info.get("chunks_size", 1000)),
    }


def _data_rel(info: dict[str, Any], ep: int, chunks_size: int) -> str:
    return info["data_path"].format(episode_chunk=ep // chunks_size, episode_index=ep)


def _video_rel(info: dict[str, Any], ep: int, chunks_size: int, video_key: str) -> str:
    return info["video_path"].format(episode_chunk=ep // chunks_size, video_key=video_key, episode_index=ep)


def tree_stats(root: Path) -> dict[str, int]:
    """File count, logical bytes (what an upload sends) and unique on-disk bytes (by inode)."""
    files = 0
    logical = 0
    seen: set[tuple[int, int]] = set()
    unique = 0
    for dirpath, _dirs, names in os.walk(root):
        for n in names:
            p = Path(dirpath) / n
            try:
                stt = p.stat()  # follows symlinks: an upload with --copy-links sends the target
            except OSError:
                continue
            files += 1
            logical += stt.st_size
            if not p.is_symlink():
                key = (stt.st_dev, stt.st_ino)
                if key not in seen:
                    seen.add(key)
                    unique += stt.st_size
    return {"files": files, "logical_bytes": logical, "unique_disk_bytes": unique}


# ---------------------------------------------------------------------------
# Replication
# ---------------------------------------------------------------------------


def _place_file(src: Path, dst: Path, mode: str) -> str:
    """Hard link / symlink / copy ``src`` to ``dst``; returns the mode actually used."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError as exc:  # cross-device or FS without hard links -> symlink
            if exc.errno not in (errno.EXDEV, errno.EPERM, errno.EMLINK, errno.ENOTSUP):
                raise
            mode = "symlink"
    if mode == "symlink":
        os.symlink(src.resolve(), dst)
        return "symlink"
    shutil.copy2(src, dst)
    return "copy"


def _shift_feature_stats(stats: dict[str, Any], key: str, delta: float | None, const: float | None) -> None:
    """Patch per-episode stats of ``index`` (shift by delta) / ``episode_index`` (constant)."""
    entry = stats.get(key)
    if not isinstance(entry, dict):
        return
    for name, val in list(entry.items()):
        if name == "count" or not (name in _SHIFTED_STATS or (name.startswith("q") and name[1:].isdigit())):
            continue
        if const is not None:
            entry[name] = [const] * len(val) if isinstance(val, list) else const
        elif delta is not None:
            entry[name] = [v + delta for v in val] if isinstance(val, list) else val + delta
    if const is not None and "std" in entry:
        v = entry["std"]
        entry["std"] = [0.0] * len(v) if isinstance(v, list) else 0.0


def _global_index_stats(lengths: list[int], existing: dict[str, Any]) -> dict[str, Any]:
    """Exact stats of the ``index`` column (0..T-1) and the frame-weighted ``episode_index``."""
    total = sum(lengths)
    out: dict[str, Any] = {}

    def like(entry: dict[str, Any], name: str, value: float) -> Any:
        old = entry.get(name)
        return [value] * len(old) if isinstance(old, list) else value

    if "index" in existing and total > 0:
        mean = (total - 1) / 2.0
        std = math.sqrt((total * total - 1) / 12.0)
        ent = dict(existing["index"])
        for name in ent:
            if name == "min":
                ent[name] = like(existing["index"], name, 0)
            elif name == "max":
                ent[name] = like(existing["index"], name, total - 1)
            elif name == "mean":
                ent[name] = like(existing["index"], name, mean)
            elif name == "std":
                ent[name] = like(existing["index"], name, std)
            elif name == "count":
                ent[name] = like(existing["index"], name, total)
            elif name.startswith("q") and name[1:].isdigit():
                ent[name] = like(existing["index"], name, float(name[1:]) / 100.0 * (total - 1))
        out["index"] = ent
    if "episode_index" in existing and total > 0:
        mean = sum(e * n for e, n in enumerate(lengths)) / total
        var = sum(n * (e - mean) ** 2 for e, n in enumerate(lengths)) / total
        cum: list[int] = []
        acc = 0
        for n in lengths:
            acc += n
            cum.append(acc)

        def quantile(p: float) -> int:
            target = p * (total - 1)
            for e, c in enumerate(cum):
                if c > target:
                    return e
            return len(lengths) - 1

        ent = dict(existing["episode_index"])
        for name in ent:
            if name == "min":
                ent[name] = like(existing["episode_index"], name, 0)
            elif name == "max":
                ent[name] = like(existing["episode_index"], name, len(lengths) - 1)
            elif name == "mean":
                ent[name] = like(existing["episode_index"], name, mean)
            elif name == "std":
                ent[name] = like(existing["episode_index"], name, math.sqrt(var))
            elif name == "count":
                ent[name] = like(existing["episode_index"], name, total)
            elif name.startswith("q") and name[1:].isdigit():
                ent[name] = like(existing["episode_index"], name, quantile(float(name[1:]) / 100.0))
        out["episode_index"] = ent
    return out


def _rewrite_splits(splits: dict[str, Any], n_src: int, n_total: int, warnings: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for name, rng in (splits or {}).items():
        try:
            a, b = (int(x) for x in str(rng).split(":"))
        except ValueError:
            warnings.append(f"split {name!r}={rng!r} is not 'a:b'; dropped")
            continue
        if (a, b) == (0, n_src):
            out[name] = f"0:{n_total}"
        else:
            warnings.append(f"split {name!r}={rng!r} is partial and cannot stay contiguous after replication; dropped")
    if not out:
        out["train"] = f"0:{n_total}"
    return out


def replicate_v21(
    src: Path,
    dst: Path,
    factor: int,
    *,
    link_mode: str = "hardlink",
    modality_json: Path | None = None,
    workers: int = 8,
) -> dict[str, Any]:
    """Write a ``factor``-times replicated copy of the v2.1 dataset ``src`` into ``dst``."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    if factor < 1:
        raise ValueError("factor must be >= 1")
    desc = describe_source(src)
    info, episodes = desc["info"], desc["episodes"]
    chunks_size, video_keys = desc["chunks_size"], desc["video_keys"]
    n_src = len(episodes)
    n_total = n_src * factor

    # Real row counts decide the global index; episodes.jsonl "length" is copied as-is.
    src_rows: list[int] = []
    for ep in episodes:
        p = src / _data_rel(info, int(ep["episode_index"]), chunks_size)
        if not p.is_file():
            raise FileNotFoundError(f"missing source parquet {p}")
        src_rows.append(pq.read_metadata(p).num_rows)
    new_lengths = src_rows * factor
    starts: list[int] = []
    acc = 0
    for n in new_lengths:
        starts.append(acc)
        acc += n
    total_frames = acc

    dst.mkdir(parents=True, exist_ok=True)
    (dst / "meta").mkdir(parents=True, exist_ok=True)
    used_modes: dict[str, int] = {}
    lock = threading.Lock()

    def do_episode(pos: int) -> None:
        ep_src = int(episodes[pos]["episode_index"])
        table = pq.read_table(src / _data_rel(info, ep_src, chunks_size))
        schema = table.schema
        n = table.num_rows
        for r in range(factor):
            new_ep = r * n_src + pos
            start = starts[new_ep]
            t = table
            if "episode_index" in schema.names:
                i = schema.get_field_index("episode_index")
                t = t.set_column(i, schema.field(i), pa.array(np.full(n, new_ep, dtype=np.int64)).cast(schema.field(i).type))
            if "index" in schema.names:
                i = schema.get_field_index("index")
                t = t.set_column(
                    i, schema.field(i), pa.array(np.arange(start, start + n, dtype=np.int64)).cast(schema.field(i).type)
                )
            t = t.replace_schema_metadata(schema.metadata)
            out = dst / _data_rel(info, new_ep, chunks_size)
            out.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(t, out)
            for vk in video_keys:
                used = _place_file(
                    src / _video_rel(info, ep_src, chunks_size, vk),
                    dst / _video_rel(info, new_ep, chunks_size, vk),
                    link_mode,
                )
                with lock:
                    used_modes[used] = used_modes.get(used, 0) + 1

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(do_episode, range(n_src)))

    # ---- meta ----
    warnings: list[str] = []
    for item in sorted((src / "meta").iterdir()):
        if item.name in _REWRITTEN_META:
            continue
        if item.is_dir():
            shutil.copytree(item, dst / "meta" / item.name)
        else:
            shutil.copy2(item, dst / "meta" / item.name)
    if modality_json is not None:
        shutil.copy2(modality_json, dst / "meta" / "modality.json")
    for item in sorted(src.iterdir()):  # README.md etc. at the top level
        if item.is_file():
            shutil.copy2(item, dst / item.name)

    new_episodes: list[dict[str, Any]] = []
    new_ep_stats: list[dict[str, Any]] = []
    for r in range(factor):
        for pos, ep in enumerate(episodes):
            new_ep = r * n_src + pos
            new_episodes.append({**ep, "episode_index": new_ep})
            old = desc["episodes_stats"].get(int(ep["episode_index"]))
            if old is not None:
                rec = json.loads(json.dumps(old))
                rec["episode_index"] = new_ep
                stats = rec.get("stats", {})
                old_idx_min = (stats.get("index") or {}).get("min")
                old_first = old_idx_min[0] if isinstance(old_idx_min, list) and old_idx_min else old_idx_min
                delta = (starts[new_ep] - old_first) if isinstance(old_first, (int, float)) else None
                _shift_feature_stats(stats, "index", delta, None)
                _shift_feature_stats(stats, "episode_index", None, new_ep)
                new_ep_stats.append(rec)
    _write_jsonl(dst / "meta" / "episodes.jsonl", new_episodes)
    if new_ep_stats:
        _write_jsonl(dst / "meta" / "episodes_stats.jsonl", new_ep_stats)

    stats_path = src / "meta" / "stats.json"
    if stats_path.is_file():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        stats.update(_global_index_stats(new_lengths, stats))
        (dst / "meta" / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    new_info = dict(info)
    new_info["total_episodes"] = n_total
    new_info["total_frames"] = total_frames
    if "total_videos" in info or video_keys:
        new_info["total_videos"] = n_total * len(video_keys)
    new_info["total_chunks"] = max(1, math.ceil(n_total / chunks_size))
    new_info["splits"] = _rewrite_splits(info.get("splits", {}), n_src, n_total, warnings)
    (dst / "meta" / "info.json").write_text(json.dumps(new_info, indent=2) + "\n", encoding="utf-8")

    for w in warnings:
        print(f"[x-tier] warning: {w}", file=sys.stderr)
    return {
        "factor": factor,
        "episodes": n_total,
        "frames": total_frames,
        "videos": n_total * len(video_keys),
        "video_placement": used_modes,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Tier driver
# ---------------------------------------------------------------------------


def _commit_dir(tmp: Path, final: Path, overwrite: bool) -> None:
    if final.exists():
        if not overwrite:
            raise FileExistsError(f"{final} exists (pass --overwrite)")
        shutil.rmtree(final)
    tmp.rename(final)


def build_tier(
    src: Path,
    out_root: Path,
    factor: int,
    *,
    lance_modes: list[str],
    link_mode: str = "hardlink",
    modality_json: Path | None = None,
    external_base_uri: str | None = None,
    lance_uri_template: str | None = None,
    overwrite: bool = False,
    workers: int = 8,
) -> dict[str, Any]:
    tier_dir = out_root / f"x{factor}"
    tier_dir.mkdir(parents=True, exist_ok=True)
    entry: dict[str, Any] = {"factor": factor, "variants": {}}

    v21 = tier_dir / "v21"
    if v21.exists() and not overwrite:
        raise FileExistsError(f"{v21} exists (pass --overwrite)")
    tmp = tier_dir / f".tmp-v21-{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    t0 = time.time()
    rep = replicate_v21(src, tmp, factor, link_mode=link_mode, modality_json=modality_json, workers=workers)
    _commit_dir(tmp, v21, overwrite)
    entry.update({k: rep[k] for k in ("episodes", "frames", "videos")})
    entry["warnings"] = rep["warnings"]
    entry["variants"]["v21"] = {
        "path": str(v21),
        "build_s": round(time.time() - t0, 3),
        "video_placement": rep["video_placement"],
        **tree_stats(v21),
    }

    if lance_modes:
        from gr00t_lancedb.convert import convert_v2_to_lance

        for mode in lance_modes:
            target: str | Path
            if lance_uri_template:  # e.g. s3://bucket/x_tier/x{factor}/lance_{mode} (must be empty)
                target = lance_uri_template.format(factor=factor, mode=mode)
            else:
                target = tier_dir / f"lance_{mode}"
            local = isinstance(target, Path) or "://" not in str(target)
            if local:
                target = Path(target)
                if target.exists():
                    if not overwrite:
                        raise FileExistsError(f"{target} exists (pass --overwrite)")
                    shutil.rmtree(target)
            kwargs: dict[str, Any] = {"video_mode": mode}
            if mode == "external" and external_base_uri:
                kwargs["video_base_uri"] = external_base_uri.format(factor=factor)
            t0 = time.time()
            convert_v2_to_lance(v21, target, **kwargs)
            entry["variants"][f"lance_{mode}"] = {
                "path": str(target),
                "build_s": round(time.time() - t0, 3),
                "video_base_uri": kwargs.get("video_base_uri"),
                # remote targets: measure with `rclone size` / `gsutil du` (not listed here)
                **(tree_stats(Path(target)) if local else {"files": None, "logical_bytes": None,
                                                           "unique_disk_bytes": None}),
            }
    return entry


def plan_tier(
    src: Path,
    out_root: Path,
    factor: int,
    lance_modes: list[str],
    src_stats: dict[str, int],
    lance_uri_template: str | None = None,
) -> dict[str, Any]:
    """Dry-run estimate: counts are exact, sizes are linear extrapolations of the source."""
    desc = describe_source(src)
    info = desc["info"]
    n_src = len(desc["episodes"])
    frames = int(info.get("total_frames") or sum(int(e.get("length", 0)) for e in desc["episodes"]))
    variants = {"v21": {"path": str(out_root / f"x{factor}" / "v21"),
                        "est_logical_bytes": src_stats["logical_bytes"] * factor,
                        "est_files": src_stats["files"] * factor}}
    for m in lance_modes:
        variants[f"lance_{m}"] = {"path": (lance_uri_template.format(factor=factor, mode=m) if lance_uri_template
                                           else str(out_root / f"x{factor}" / f"lance_{m}"))}
    return {
        "factor": factor,
        "episodes": n_src * factor,
        "frames": frames * factor,
        "videos": n_src * factor * len(desc["video_keys"]),
        "variants": variants,
    }


def _parse_factors(s: str) -> list[int]:
    out = [int(x) for x in s.replace(" ", "").split(",") if x]
    if not out or any(f < 1 for f in out):
        raise argparse.ArgumentTypeError("factors must be positive integers, e.g. 1,10,50,200")
    return out


def _parse_lance(s: str) -> list[str]:
    modes = [m for m in s.replace(" ", "").split(",") if m and m != "none"]
    bad = [m for m in modes if m not in ("copy", "external")]
    if bad:
        raise argparse.ArgumentTypeError(f"unknown lance mode(s) {bad}; use copy,external or none")
    return modes


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, type=Path, help="Source LeRobot v2.1 dataset directory")
    p.add_argument("--out-root", required=True, type=Path)
    p.add_argument("--factors", type=_parse_factors, default=[1, 10, 50, 200])
    p.add_argument("--lance", type=_parse_lance, default=[], help="Lance variants: copy,external (default none)")
    p.add_argument("--link-mode", choices=("hardlink", "symlink", "copy"), default="hardlink")
    p.add_argument("--copy-videos", action="store_true", help="Same as --link-mode copy")
    p.add_argument("--modality-json", type=Path, default=None, help="GR00T modality.json to put in meta/")
    p.add_argument(
        "--external-base-uri",
        default=None,
        help="Base URI the external Lance variant points at; '{factor}' is substituted "
        "(default: the local v21 directory)",
    )
    p.add_argument(
        "--lance-uri-template",
        default=None,
        help="Write Lance variants here instead of <out-root>/x<F>/lance_<mode>; '{factor}' and '{mode}' are "
        "substituted, e.g. s3://bucket/x_tier/x{factor}/lance_{mode} (credentials from the environment; "
        "the target must be empty). Saves local disk for the 50x/200x tiers.",
    )
    p.add_argument("--manifest", type=Path, default=None, help="Default: <out-root>/manifest.json")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Print the plan and estimated sizes; write nothing")
    args = p.parse_args(argv)

    src = args.src.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()
    link_mode = "copy" if args.copy_videos else args.link_mode
    if args.modality_json is None and not (src / "meta" / "modality.json").is_file():
        print("[x-tier] note: no meta/modality.json in source; GR00T needs one (--modality-json)", file=sys.stderr)
    src_stats = tree_stats(src)
    manifest: dict[str, Any] = {
        "created_unix": time.time(),
        "source": {"path": str(src), **src_stats},
        "link_mode": link_mode,
        "lance_modes": args.lance,
        "dry_run": bool(args.dry_run),
        "tiers": [],
    }
    for factor in args.factors:
        if args.dry_run:
            entry = plan_tier(src, out_root, factor, args.lance, src_stats, args.lance_uri_template)
            print(
                f"[dry-run] x{factor}: {entry['episodes']} episodes, {entry['frames']} frames, "
                f"~{entry['variants']['v21']['est_logical_bytes'] / 2**30:.2f} GiB v21 -> "
                + ", ".join(v["path"] for v in entry["variants"].values())
            )
        else:
            print(f"[x-tier] building x{factor} ...", file=sys.stderr)
            entry = build_tier(
                src,
                out_root,
                factor,
                lance_modes=args.lance,
                link_mode=link_mode,
                modality_json=args.modality_json,
                external_base_uri=args.external_base_uri,
                lance_uri_template=args.lance_uri_template,
                overwrite=args.overwrite,
                workers=args.workers,
            )
        manifest["tiers"].append(entry)

    text = json.dumps(manifest, indent=2)
    if args.dry_run:
        print(text)
    else:
        path = args.manifest or (out_root / "manifest.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
        print(f"[x-tier] manifest: {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
