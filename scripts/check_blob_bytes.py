#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""P0.1 conversion fidelity: every MP4 stored in ``videos.lance`` equals the source file byte-for-byte.

    python scripts/check_blob_bytes.py --v2-dir /data/datasets/libero_10 --lance-dir /data/datasets/libero_10_lance
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--v2-dir", required=True)
    p.add_argument("--lance-dir", required=True)
    p.add_argument("--out-json", default=None)
    a = p.parse_args()

    import lancedb

    info = json.loads((Path(a.v2_dir) / "meta" / "info.json").read_text())
    video_path = info["video_path"]
    table = lancedb.connect(a.lance_dir).open_table("videos")
    meta = (
        table.search()
        .with_row_id(True)
        .select(["video_key", "chunk_index", "file_index", "file_size"])
        .limit(None)
        .to_arrow()
        .to_pylist()
    )
    n_ok = n_bad = n_bytes = 0
    bad = []
    for row in meta:
        src = Path(a.v2_dir) / video_path.format(
            episode_chunk=row["chunk_index"], video_key=row["video_key"], episode_index=row["file_index"]
        )
        blob = table.fetch_blob_ranges("video_bytes", [(row["_rowid"], 0, row["file_size"])])[0].as_py()
        orig = src.read_bytes()
        n_bytes += len(orig)
        if hashlib.sha256(blob).digest() == hashlib.sha256(orig).digest():
            n_ok += 1
        else:
            n_bad += 1
            bad.append(str(src))
    v2_eps = sum(1 for _ in open(Path(a.v2_dir) / "meta" / "episodes.jsonl"))
    frames = lancedb.connect(a.lance_dir).open_table("frames").count_rows()
    report = {
        "videos_checked": len(meta),
        "byte_identical": n_ok,
        "mismatched": n_bad,
        "bytes_compared": n_bytes,
        "episodes_v2": v2_eps,
        "frames_v2_info": info["total_frames"],
        "frames_lance": frames,
        "bad_head": bad[:10],
        "pass": n_bad == 0 and frames == info["total_frames"],
    }
    print(json.dumps(report, indent=2))
    if a.out_json:
        Path(a.out_json).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
