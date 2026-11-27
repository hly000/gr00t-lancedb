# SPDX-License-Identifier: Apache-2.0
"""Start a Lance dataset's startup queries early, so they overlap with model loading.

Building a :class:`~gr00t_lancedb.episode_loader.LanceEpisodeLoader` may need up to two
child-process calls (see ``_isolated``): materializing a remote dataset's ``meta/`` cache
and evaluating a ``where`` predicate. Run one after the other behind the model load, they
added 12-17 s to a cold start on the A100 box. :func:`prefetch_lance_startup` starts them
right away; the loader later picks the results up and usually does not wait at all.

Standard-library imports only: this module is imported before GR00T, torch or lerobot.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
import json
import os
from pathlib import Path
from typing import Any

MATERIALIZE_META = "gr00t_lancedb.lance_query:materialize_meta"
QUERY_WHERE = "gr00t_lancedb.lance_query:query_where_anchors"
FETCH_VERSIONED_META = "gr00t_lancedb.lance_query:fetch_versioned_meta"


def user_cache_dir() -> Path:
    base = os.environ.get("GR00T_LANCE_CACHE_DIR")
    if base:
        return Path(base)
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "gr00t_lancedb"


def isolated_kwargs(
    dataset_path: str | Path,
    repo_id: str | None = None,
    storage_options: Mapping[str, str] | None = None,
    revision: str | None = None,
    token: str | bool | None = None,
) -> dict[str, Any]:
    """Connection arguments shared by every child call for one dataset.

    The loader and :func:`prefetch_lance_startup` must build these identically, because a
    started call is matched to its consumer by function name plus arguments.
    """
    return {
        "dataset_path": str(dataset_path),
        "repo_id": repo_id,
        "storage_options": dict(storage_options) if storage_options else None,
        "revision": revision,
        "token": token,
    }


def where_kwargs(base: Mapping[str, Any], where: str, lance_version: int | None) -> dict[str, Any]:
    return {**base, "where": where, "lance_version": int(lance_version) if lance_version is not None else None}


def where_cache_path(dataset_path: str | Path, repo_id: str | None, where: str, version: int) -> Path:
    """On-disk cache of a ``where`` result at one (immutable) ``frames.lance`` version."""
    ident = json.dumps([str(dataset_path), repo_id, where, int(version)])
    return user_cache_dir() / "where" / f"{hashlib.sha256(ident.encode()).hexdigest()[:24]}.npz"


def needs_meta_child(dataset_path: str | Path, repo_id: str | None) -> bool:
    """Whether the dataset's ``meta/`` lives in a local cache filled from ``meta.lance``."""
    return "://" in str(dataset_path) and repo_id is None


def prefetch_lance_startup(
    dataset_paths: Iterable[str | Path],
    *,
    where: str | None = None,
    lance_version: int | None = None,
    lance_versions: Mapping[str, int] | None = None,
    repo_id: str | None = None,
    storage_options: Mapping[str, str] | None = None,
    revision: str | None = None,
    token: str | bool | None = None,
) -> list[str]:
    """Start, without waiting, the child calls these datasets will need at build time.

    Call it as early as possible (before importing GR00T / loading the model) with the same
    arguments the datasets will be built with. ``lance_versions`` maps a dataset path to a
    pinned ``frames.lance`` version and is used when ``lance_version`` is not given.

    Returns the names of the calls it started, e.g. ``["materialize_meta", "query_where"]``.
    Nothing here is required for correctness: a call that was not started, or was started
    with different arguments, simply runs when the loader asks for it.
    """
    from ._isolated import start_isolated

    started: list[str] = []
    for raw in dataset_paths:
        path = str(raw)
        base = isolated_kwargs(path, repo_id, storage_options, revision, token)
        if needs_meta_child(path, repo_id) and start_isolated(MATERIALIZE_META, base):
            started.append("materialize_meta")
        if where is None:
            continue
        version = lance_version if lance_version is not None else (lance_versions or {}).get(path)
        if version is not None and where_cache_path(path, repo_id, where, version).exists():
            continue  # the loader will read the cached result; no query needed
        if start_isolated(QUERY_WHERE, where_kwargs(base, where, version)):
            started.append("query_where")
    return started


__all__ = [
    "FETCH_VERSIONED_META",
    "MATERIALIZE_META",
    "QUERY_WHERE",
    "isolated_kwargs",
    "needs_meta_child",
    "prefetch_lance_startup",
    "user_cache_dir",
    "where_cache_path",
    "where_kwargs",
]
