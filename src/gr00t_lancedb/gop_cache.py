# SPDX-License-Identifier: Apache-2.0
"""GOP-aligned, size-bounded local-disk cache for Lance video blob ranges (issue #3).

``lerobot_lancedb.LanceDBDataset`` reads every compressed video byte through two calls
on its ``videos`` table: ``fetch_blob_ranges(column, [(row_id, offset, length), ...])``
(planned prefetch) and ``fetch_blob_files(...)[i].read_range(offset, length)`` (rare
fallback). :class:`CachedVideoTable` wraps that table:

* Every requested range is mapped onto **GOP segments** of its MP4 — the byte spans
  between consecutive keyframe positions ``kf_positions`` that ``videos.lance`` already
  stores per file (long GOPs are split into ``max_segment_bytes`` pieces). Decoding any
  frame needs its whole GOP, so a segment is the natural cache unit: epoch 2+ hits no
  matter how a batch's spans were merged, and only GOPs that training actually touched
  are ever stored (unlike ``rclone --vfs-cache-mode full``, which caches whole files).
* Missing segments of one call are coalesced into one ``fetch_blob_ranges`` call
  (adjacent segments merged), so batching towards the object store is preserved.
* Segments are files ``<root>/<namespace>/<start>-<end>.seg`` written to a temp file and
  ``os.replace``d — atomic, so concurrent DataLoader workers never see torn data. A
  ``flock``-protected byte ledger bounds the total size; when it exceeds the budget the
  least-recently-used segments (mtime, refreshed on every hit) are evicted down to
  ``evict_to`` x budget. Overshoot is bounded by one in-flight segment per writer.
* Returned bytes are exactly the remote bytes, so decoded frames are bit-identical with
  the cache on or off.

Independently of caching, the wrapper counts Lance blob calls / ranges / bytes and cache
hits per process (:data:`IO_STATS`); workers flush them to ``$GR00T_LANCE_IO_STATS_DIR``
so the training process can log object-store request counts per step (issue #3, E2).
"""

from __future__ import annotations

import bisect
from contextlib import contextmanager
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

VIDEO_BLOB_COLUMN = "video_bytes"  # == lerobot_lancedb.VIDEO_BLOB_COLUMN
_DEFAULT_MAX_SEGMENT = 1 << 20
_DEFAULT_GRID = 256 << 10

# ---------------------------------------------------------------------------
# Per-process I/O counters
# ---------------------------------------------------------------------------

_IO_FIELDS = (
    "lance_blob_calls",  # fetch_blob_ranges calls that reached Lance (after the cache)
    "lance_blob_ranges",  # (row, offset, length) ranges sent to Lance
    "lance_blob_bytes",  # bytes Lance returned for them
    "lance_handle_reads",  # BlobFile.read_range fallbacks that reached Lance
    "lance_handle_bytes",
    "requested_ranges",  # ranges the reader asked for (before the cache)
    "requested_bytes",
    "gop_cache_hits",  # segments served from local disk
    "gop_cache_misses",
    "gop_cache_hit_bytes",
    "gop_cache_miss_bytes",
    "gop_cache_evictions",
    "gop_cache_evicted_bytes",
    "frames_take_calls",  # frames.lance row takes (tabular reads)
    "frames_take_rows",
)


class IoStats:
    """Thread-safe counters, reset automatically in a forked child."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pid = os.getpid()
        self._counts = dict.fromkeys(_IO_FIELDS, 0)
        self._last_flush = 0.0

    def _check_pid(self) -> None:
        if self._pid != os.getpid():  # inherited through fork: start from zero
            self._lock = threading.Lock()
            self._pid = os.getpid()
            self._counts = dict.fromkeys(_IO_FIELDS, 0)
            self._last_flush = 0.0

    def _register_exit_flush(self) -> None:
        """Flush once more when this process exits (main: atexit; DataLoader workers:
        multiprocessing finalizers, since worker processes leave via os._exit)."""
        if getattr(self, "_exit_hook_pid", None) == os.getpid():
            return
        self._exit_hook_pid = os.getpid()
        import atexit
        from multiprocessing import util as mp_util

        atexit.register(self.maybe_flush, True)
        mp_util.Finalize(None, self.maybe_flush, args=(True,), exitpriority=10)

    def add(self, **deltas: int) -> None:
        self._check_pid()
        if os.environ.get("GR00T_LANCE_IO_STATS_DIR"):
            self._register_exit_flush()
        with self._lock:
            for k, v in deltas.items():
                self._counts[k] = self._counts.get(k, 0) + int(v)
        self.maybe_flush()

    def snapshot(self) -> dict[str, int]:
        self._check_pid()
        with self._lock:
            return dict(self._counts)

    def reset(self) -> None:
        with self._lock:
            self._counts = dict.fromkeys(_IO_FIELDS, 0)

    def maybe_flush(self, force: bool = False, interval_s: float = 2.0) -> None:
        out_dir = os.environ.get("GR00T_LANCE_IO_STATS_DIR")
        if not out_dir:
            return
        now = time.monotonic()
        if not force and now - self._last_flush < interval_s:
            return
        self._last_flush = now
        try:
            d = Path(out_dir)
            d.mkdir(parents=True, exist_ok=True)
            final = d / f"io-{os.getpid()}.json"
            tmp = d / f".io-{os.getpid()}.tmp"
            tmp.write_text(json.dumps(self.snapshot()), encoding="utf-8")
            os.replace(tmp, final)
        except OSError:  # metrics must never break data loading
            pass


IO_STATS = IoStats()


def aggregate_io_stats(stats_dir: str | Path) -> dict[str, int]:
    """Sum every worker's ``io-<pid>.json`` under ``stats_dir`` (cumulative for the run)."""
    total = dict.fromkeys(_IO_FIELDS, 0)
    d = Path(stats_dir)
    if not d.is_dir():
        return total
    for f in d.glob("io-*.json"):
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for k, v in rec.items():
            if isinstance(v, (int, float)):
                total[k] = total.get(k, 0) + int(v)
    return total


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------


class GopDiskCache:
    """Multi-process safe, size-bounded segment store (see module docstring)."""

    def __init__(self, root: str | Path, budget_bytes: int, *, evict_to: float = 0.9) -> None:
        if budget_bytes <= 0:
            raise ValueError(f"budget_bytes must be > 0, got {budget_bytes}")
        self.root = Path(root)
        self.budget = int(budget_bytes)
        self.evict_to = float(evict_to)
        self.root.mkdir(parents=True, exist_ok=True)
        self._ledger = self.root / ".ledger"
        self._lock_path = self.root / ".lock"
        self._lock_fd: int | None = None
        self._lock_pid: int | None = None
        self._tlock = threading.Lock()

    # -- locking (flock is per open file description: reopen after fork) -----------
    @contextmanager
    def _locked(self):
        with self._tlock:
            if self._lock_fd is None or self._lock_pid != os.getpid():
                self._lock_fd = os.open(self._lock_path, os.O_RDWR | os.O_CREAT, 0o644)
                self._lock_pid = os.getpid()
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)

    def _path(self, namespace: str, start: int, end: int) -> Path:
        return self.root / namespace / f"{start:013d}-{end:013d}.seg"

    def get(self, namespace: str, start: int, end: int) -> bytes | None:
        p = self._path(namespace, start, end)
        try:
            data = p.read_bytes()
        except OSError:
            return None
        if len(data) != end - start:  # should never happen with atomic writes
            try:
                p.unlink()
            except OSError:
                pass
            return None
        try:
            os.utime(p)  # LRU recency
        except OSError:
            pass
        return data

    def put(self, namespace: str, start: int, end: int, data: bytes) -> None:
        if len(data) != end - start or len(data) > self.budget:
            return
        final = self._path(namespace, start, end)
        if final.exists():
            return
        final.parent.mkdir(parents=True, exist_ok=True)
        tmp = final.parent / f".{final.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        try:
            with open(tmp, "wb") as fh:
                fh.write(data)
            os.replace(tmp, final)
        except OSError as exc:
            logger.debug("gop cache write failed for %s: %s", final, exc)
            try:
                tmp.unlink()
            except OSError:
                pass
            return
        self._account(len(data))

    def _read_ledger(self) -> int | None:
        try:
            return int(self._ledger.read_text(encoding="utf-8").strip() or "0")
        except (OSError, ValueError):
            return None

    def _write_ledger(self, total: int) -> None:
        tmp = self.root / f".ledger.{os.getpid()}.tmp"
        tmp.write_text(str(int(total)), encoding="utf-8")
        os.replace(tmp, self._ledger)

    def _account(self, delta: int) -> None:
        with self._locked():
            total = self._read_ledger()
            total = self._scan_total() if total is None else total + delta
            if total > self.budget:
                total = self._evict_locked()
            self._write_ledger(total)

    def _segments_on_disk(self) -> list[tuple[float, int, Path]]:
        out = []
        for p in self.root.glob("*/*.seg"):
            try:
                st = p.stat()
            except OSError:
                continue
            out.append((st.st_mtime, st.st_size, p))
        return out

    def _scan_total(self) -> int:
        return sum(size for _, size, _ in self._segments_on_disk())

    def _evict_locked(self) -> int:
        segs = self._segments_on_disk()
        total = sum(size for _, size, _ in segs)
        if total <= self.budget:
            return total
        target = int(self.budget * self.evict_to)
        n = freed = 0
        for _, size, p in sorted(segs, key=lambda t: t[0]):
            if total <= target:
                break
            try:
                p.unlink()
            except OSError:
                continue
            total -= size
            freed += size
            n += 1
        IO_STATS.add(gop_cache_evictions=n, gop_cache_evicted_bytes=freed)
        return total

    def total_bytes(self) -> int:
        return self._scan_total()


# ---------------------------------------------------------------------------
# videos-table proxy
# ---------------------------------------------------------------------------


def _segments_for(bounds: list[int] | None, start: int, end: int, max_seg: int, grid: int) -> list[tuple[int, int]]:
    """Cache segments covering ``[start, end)``: GOP spans (split at ``max_seg``) or a grid."""
    if end <= start:
        return []
    segs: list[tuple[int, int]] = []
    if not bounds:
        first = start - start % grid
        for s in range(first, end, grid):
            segs.append((s, s + grid))
        return segs
    i = max(0, bisect.bisect_right(bounds, start) - 1)
    while i < len(bounds) - 1 and bounds[i] < end:
        s, e = bounds[i], bounds[i + 1]
        if e - s > max_seg:
            k = max(0, (start - s) // max_seg)
            ps = s + k * max_seg
            while ps < min(e, end):
                segs.append((ps, min(e, ps + max_seg)))
                ps += max_seg
        else:
            segs.append((s, e))
        i += 1
    return segs


class _CachedBlobFile:
    """Routes ``read_range`` of a lazy Lance ``BlobFile`` through the segment cache."""

    def __init__(self, inner: Any, owner: CachedVideoTable, row_id: int) -> None:
        self._inner = inner
        self._owner = owner
        self._row = row_id

    def read_range(self, offset: int, length: int) -> bytes:
        return self._owner._read_one(self._row, int(offset), int(length), handle=self._inner)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class CachedVideoTable:
    """Drop-in proxy for ``LanceDBDataset._videos_table`` (cache optional, stats always)."""

    def __init__(
        self,
        table: Any,
        reader: Any,
        cache: GopDiskCache | None,
        *,
        dataset_id: str,
        max_segment_bytes: int = _DEFAULT_MAX_SEGMENT,
        grid_bytes: int = _DEFAULT_GRID,
    ) -> None:
        self._table = table
        self._reader = reader
        self._cache = cache
        self._dataset_ns = hashlib.sha1(dataset_id.encode()).hexdigest()[:16]
        self._max_seg = int(max_segment_bytes)
        self._grid = int(grid_bytes)
        self._rowid_to_key: dict[int, tuple] | None = None
        self._bounds: dict[tuple, list[int]] = {}

    def __getattr__(self, name: str) -> Any:  # search(), take_row_ids(), schema, ...
        return getattr(self._table, name)

    @property
    def cache(self) -> GopDiskCache | None:
        return self._cache

    # -- helpers --------------------------------------------------------------
    def _file_key(self, row_id: int) -> tuple | None:
        if self._rowid_to_key is None:
            ids = getattr(self._reader, "_video_row_ids", None) or {}
            self._rowid_to_key = {int(v): k for k, v in ids.items()}
        return self._rowid_to_key.get(int(row_id))

    def _namespace(self, key: tuple | None, row_id: int) -> tuple[str, list[int] | None]:
        if key is None:
            return f"{self._dataset_ns}/row-{int(row_id)}-grid", None
        bounds = self._bounds.get(key)
        meta = (getattr(self._reader, "_file_meta", None) or {}).get(key)
        if bounds is None and meta is not None:
            size = int(meta["file_size"])
            kf = sorted({int(x) for x in meta.get("kf_positions", ()) if 0 < int(x) < size})
            bounds = [0, *kf, size]
            self._bounds[key] = bounds
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(key[0]))
        size_tag = bounds[-1] if bounds else "unk"
        kind = "gop" if bounds else "grid"
        return f"{self._dataset_ns}/{name}-{int(key[1])}-{int(key[2])}-{size_tag}-{kind}", bounds

    # -- cached range reads ---------------------------------------------------------
    def _read_many(self, column: str, requests: list[tuple[int, int, int]], *, handle: Any = None) -> list[bytes]:
        cache = self._cache
        assert cache is not None
        plans: list[tuple[str | None, int, int, list[tuple[int, int]]]] = []
        wanted: dict[tuple[str, int, int], bytes | None] = {}
        seg_row: dict[tuple[str, int, int], int] = {}
        passthrough: list[int] = []  # request positions without GOP bounds (not cached)
        for pos, (row, start, length) in enumerate(requests):
            key = self._file_key(row)
            ns, bounds = self._namespace(key, row)
            if bounds is None:
                plans.append((None, start, length, []))
                passthrough.append(pos)
                continue
            segs = _segments_for(bounds, start, start + length, self._max_seg, self._grid)
            plans.append((ns, start, length, segs))
            for s, e in segs:
                k = (ns, s, e)
                if k not in wanted:
                    wanted[k] = cache.get(ns, s, e)
                    seg_row[k] = row
        hits = [k for k, v in wanted.items() if v is not None]
        misses = sorted((k for k, v in wanted.items() if v is None), key=lambda k: (k[0], k[1]))
        IO_STATS.add(
            gop_cache_hits=len(hits),
            gop_cache_misses=len(misses),
            gop_cache_hit_bytes=sum(k[2] - k[1] for k in hits),
            gop_cache_miss_bytes=sum(k[2] - k[1] for k in misses),
        )
        # Coalesce adjacent missing segments of the same file into one remote range.
        merged: list[tuple[int, int, int, list[tuple[str, int, int]]]] = []
        last_ns: str | None = None
        for k in misses:
            ns, s, e = k
            if merged and last_ns == ns and merged[-1][2] == s:
                row, ms, _, parts = merged[-1]
                merged[-1] = (row, ms, e, [*parts, k])
            else:
                merged.append((seg_row[k], s, e, [k]))
            last_ns = ns
        remote = [(row, ms, me - ms) for row, ms, me, _ in merged]
        remote += [requests[pos] for pos in passthrough]
        payloads: list[bytes] = []
        if remote:
            if handle is not None and len(remote) == 1:
                payloads = [handle.read_range(remote[0][1], remote[0][2])]
                IO_STATS.add(lance_handle_reads=1, lance_handle_bytes=len(payloads[0]))
            else:
                payloads = [b.as_py() or b"" for b in self._table.fetch_blob_ranges(column, remote)]
                IO_STATS.add(
                    lance_blob_calls=1,
                    lance_blob_ranges=len(remote),
                    lance_blob_bytes=sum(len(p) for p in payloads),
                )
        for (_, ms, me, parts), data in zip(merged, payloads[: len(merged)], strict=True):
            complete = len(data) == me - ms
            for k in parts:
                seg = data[k[1] - ms : k[2] - ms]
                wanted[k] = seg
                if complete:
                    cache.put(k[0], k[1], k[2], seg)
        passthrough_data = dict(zip(passthrough, payloads[len(merged) :], strict=True))
        out: list[bytes] = []
        for pos, (ns, start, length, segs) in enumerate(plans):
            if ns is None:
                out.append(passthrough_data[pos])
            elif not segs:
                out.append(b"")
            else:
                buf = b"".join(wanted[(ns, s, e)] for s, e in segs)
                off = start - segs[0][0]
                out.append(buf[off : off + length])
        return out

    def _read_one(self, row: int, offset: int, length: int, *, handle: Any) -> bytes:
        IO_STATS.add(requested_ranges=1, requested_bytes=length)
        if self._cache is None:
            data = handle.read_range(offset, length)
            IO_STATS.add(lance_handle_reads=1, lance_handle_bytes=len(data))
            return data
        return self._read_many(VIDEO_BLOB_COLUMN, [(row, offset, length)], handle=handle)[0]

    # -- table API ----------------------------------------------------------------
    def fetch_blob_ranges(self, column: str, requests):
        reqs = [(int(r), int(o), int(n)) for r, o, n in requests]
        IO_STATS.add(requested_ranges=len(reqs), requested_bytes=sum(n for _, _, n in reqs))
        if self._cache is None or column != VIDEO_BLOB_COLUMN or not reqs:
            arr = self._table.fetch_blob_ranges(column, requests)
            if reqs:
                IO_STATS.add(
                    lance_blob_calls=1,
                    lance_blob_ranges=len(reqs),
                    lance_blob_bytes=sum(len(b.as_py() or b"") for b in arr),
                )
            return arr
        import pyarrow as pa

        return pa.array(self._read_many(column, reqs), type=pa.large_binary())

    def fetch_blob_files(self, column: str, row_ids, **kwargs):
        handles = self._table.fetch_blob_files(column, row_ids, **kwargs)
        if column != VIDEO_BLOB_COLUMN:
            return handles
        ids = list(row_ids) if isinstance(row_ids, (list, tuple)) else None
        if ids is None or len(ids) != len(handles):
            return handles
        return [None if h is None else _CachedBlobFile(h, self, int(r)) for h, r in zip(handles, ids, strict=True)]


def resolve_cache_config(
    dataset_path: str,
    *,
    cache_dir: str | os.PathLike | None = None,
    cache_gb: float | None = None,
    force_local: bool | None = None,
) -> dict[str, Any] | None:
    """Cache settings from arguments / ``GR00T_LANCE_VIDEO_CACHE_{GB,DIR,FORCE}`` env.

    Enabled only for object-store datasets unless forced (a local Lance dataset already
    is a local disk; caching it would only duplicate bytes).
    """
    if cache_gb is None:
        raw = os.environ.get("GR00T_LANCE_VIDEO_CACHE_GB", "").strip()
        cache_gb = float(raw) if raw else 0.0
    if cache_gb <= 0:
        return None
    if force_local is None:
        force_local = os.environ.get("GR00T_LANCE_VIDEO_CACHE_FORCE", "0") == "1"
    if "://" not in str(dataset_path) and not force_local:
        return None
    if cache_dir is None:
        cache_dir = os.environ.get("GR00T_LANCE_VIDEO_CACHE_DIR") or str(
            Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "gr00t_lancedb" / "gop"
        )
    return {"dir": str(cache_dir), "budget_bytes": int(cache_gb * (1 << 30))}


def wrap_videos_table(reader: Any, *, dataset_id: str, cache_config: dict[str, Any] | None) -> None:
    """Install :class:`CachedVideoTable` on an opened ``LanceDBDataset`` (idempotent)."""
    table = getattr(reader, "_videos_table", None)
    if table is None or isinstance(table, CachedVideoTable):
        return
    cache = None
    if cache_config:
        cache = GopDiskCache(cache_config["dir"], cache_config["budget_bytes"])
    reader._videos_table = CachedVideoTable(table, reader, cache, dataset_id=dataset_id)


__all__ = [
    "IO_STATS",
    "CachedVideoTable",
    "GopDiskCache",
    "IoStats",
    "aggregate_io_stats",
    "resolve_cache_config",
    "wrap_videos_table",
]
