# SPDX-License-Identifier: Apache-2.0
"""Curation manifest names and lookup, with standard-library imports only.

Kept apart from :mod:`gr00t_lancedb.metadata` on purpose: that module imports pandas,
pyarrow and (through ``_gr00t_compat``) NVIDIA's ``gr00t`` package, which pulls in torch.
The short-lived Lance child process (``_isolated`` / ``lance_query``) only needs the names
below, and importing ``gr00t`` there cost about 12 s per call on the A100 box.
"""

from __future__ import annotations

from typing import Any

LEROBOT_STATS_FILENAME = "stats.json"
GROOT_STATS_FILENAME = "gr00t_stats.json"
CURATION_REPORT_FILENAME = "curation_report.json"
# MVCC: curation keeps one stats snapshot per frames.lance version range (issue #2).
CURATION_VERSIONS_FILENAME = "curation_versions.json"
CURATION_VERSIONS_FORMAT = "gr00t_lancedb.curation_versions/v1"


def versioned_meta_filename(kind: str, frames_version: int) -> str:
    """``gr00t_stats_v7.json`` / ``stats_v7.json`` / ``curation_report_v7.json``."""
    base = {
        "gr00t_stats": GROOT_STATS_FILENAME,
        "stats": LEROBOT_STATS_FILENAME,
        "report": CURATION_REPORT_FILENAME,
    }[kind]
    stem, ext = base.rsplit(".", 1)
    return f"{stem}_v{int(frames_version)}.{ext}"


def select_curation_entry(manifest: dict[str, Any] | None, frames_version: int) -> dict[str, Any] | None:
    """Pick the curation snapshot that is in force at ``frames_version``.

    Each entry covers ``[frames_version_from, frames_version]`` (the versions created by
    one curation run: drop/add columns + index builds). A later version that no curation
    produced (e.g. an unrelated index rebuild) inherits the newest snapshot at or below it.
    """
    if not manifest:
        return None
    best: dict[str, Any] | None = None
    for entry in manifest.get("entries", []):
        start = int(entry.get("frames_version_from", entry.get("frames_version", 0)))
        if start <= int(frames_version) and (best is None or start > int(best.get("frames_version_from", 0))):
            best = entry
    return best


__all__ = [
    "CURATION_REPORT_FILENAME",
    "CURATION_VERSIONS_FILENAME",
    "CURATION_VERSIONS_FORMAT",
    "GROOT_STATS_FILENAME",
    "LEROBOT_STATS_FILENAME",
    "select_curation_entry",
    "versioned_meta_filename",
]
