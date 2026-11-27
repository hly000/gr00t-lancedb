# SPDX-License-Identifier: Apache-2.0
"""Lance queries that run inside the isolated child interpreter (see ``_isolated.py``).

Every function here may start lancedb's Tokio runtime, so the training main process only
ever calls them through :func:`gr00t_lancedb._isolated.run_isolated`. Results are plain
numpy arrays / bytes / ints.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from .metadata import (
    CURATION_VERSIONS_FILENAME,
    select_curation_entry,
)


def _connect(
    dataset_path: str,
    *,
    repo_id: str | None,
    storage_options: dict[str, str] | None,
    revision: str | None,
    token: str | bool | None,
):
    from lerobot_lancedb.reader import _connect as ll_connect
    from lerobot_lancedb.reader import _storage_options, resolve_lance_root

    db_uri, local_root = resolve_lance_root(repo_id, dataset_path, storage_options, revision, token)
    options = _storage_options(db_uri, storage_options, revision, token)
    return ll_connect(db_uri, options), options, db_uri, local_root


def _open(db, name: str, options: dict[str, str] | None):
    return db.open_table(name, storage_options=options) if options else db.open_table(name)


def _read_meta_files(db, options, names: list[str] | None) -> dict[str, bytes]:
    from lerobot_lancedb import META_TABLE

    try:
        table = _open(db, META_TABLE, options or None)
    except Exception:  # noqa: BLE001 - datasets converted without meta.lance
        return {}
    q = table.search().select(["path", "data"]).limit(None)
    if names:
        quoted = ", ".join("'" + n.replace("'", "''") + "'" for n in names)
        q = q.where(f"path IN ({quoted})")
    tbl = q.to_arrow()
    return dict(zip(tbl.column("path").to_pylist(), tbl.column("data").to_pylist(), strict=True))


def fetch_versioned_meta(
    *,
    dataset_path: str,
    frames_version: int | None,
    repo_id: str | None = None,
    storage_options: dict[str, str] | None = None,
    revision: str | None = None,
    token: str | bool | None = None,
    _db=None,
    _options=None,
) -> dict[str, Any]:
    """Return ``curation_versions.json`` plus the stats files that match ``frames_version``.

    The manifest lives in the *latest* ``meta.lance`` (curation carries every older
    version-keyed entry forward), so a stale local ``meta/`` cache is never consulted.
    """
    if _db is None:
        _db, _options, _, _ = _connect(
            dataset_path, repo_id=repo_id, storage_options=storage_options, revision=revision, token=token
        )
    files = _read_meta_files(_db, _options, [CURATION_VERSIONS_FILENAME])
    manifest_bytes = files.get(CURATION_VERSIONS_FILENAME)
    out: dict[str, Any] = {"manifest": manifest_bytes, "files": {}, "entry": None}
    if manifest_bytes is None or frames_version is None:
        return out
    entry = select_curation_entry(json.loads(manifest_bytes), int(frames_version))
    out["entry"] = entry
    if entry is None:
        return out
    wanted = [entry[k] for k in ("gr00t_stats", "stats", "report") if entry.get(k)]
    out["files"] = _read_meta_files(_db, _options, wanted)
    return out


def query_where_anchors(
    *,
    dataset_path: str,
    where: str,
    lance_version: int | None = None,
    repo_id: str | None = None,
    storage_options: dict[str, str] | None = None,
    revision: str | None = None,
    token: str | bool | None = None,
    include_versioned_meta: bool = True,
) -> dict[str, Any]:
    """Run ``SELECT episode_index, frame_index FROM frames WHERE <where>`` at one version.

    Only the two integer columns (plus whatever the predicate touches, served by BITMAP
    scalar indexes when present) are read — never ``videos.lance``. Returns the resolved
    table version so the caller can pin readers and stats to the same MVCC snapshot.
    """
    from lerobot_lancedb import FRAMES_TABLE

    db, options, db_uri, _ = _connect(
        dataset_path, repo_id=repo_id, storage_options=storage_options, revision=revision, token=token
    )
    table = _open(db, FRAMES_TABLE, options or None)
    if lance_version is not None:
        table.checkout(int(lance_version))
    version = int(table.version)
    tbl = (
        table.search()
        .where(where)
        .select(["episode_index", "frame_index"])
        .limit(None)
        .to_arrow()
    )

    def col(name: str) -> np.ndarray:
        return (
            tbl.column(name).combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        )

    out: dict[str, Any] = {
        "frames_version": version,
        "db_uri": db_uri,
        "num_rows": int(table.count_rows()),
        "episode_index": col("episode_index") if tbl.num_rows else np.empty(0, np.int64),
        "frame_index": col("frame_index") if tbl.num_rows else np.empty(0, np.int64),
    }
    if include_versioned_meta:
        out["versioned_meta"] = fetch_versioned_meta(
            dataset_path=dataset_path, frames_version=version, _db=db, _options=options
        )
    return out


def materialize_meta(
    *,
    dataset_path: str,
    repo_id: str | None = None,
    storage_options: dict[str, str] | None = None,
    revision: str | None = None,
    token: str | bool | None = None,
) -> dict[str, Any]:
    """Materialize a remote dataset's ``meta/`` cache (cold start) and return its root."""
    from lerobot_lancedb.reader import resolve_lance_root

    db_uri, local_root = resolve_lance_root(repo_id, dataset_path, storage_options, revision, token)
    return {"db_uri": db_uri, "local_root": str(local_root)}


__all__ = ["fetch_versioned_meta", "materialize_meta", "query_where_anchors"]
