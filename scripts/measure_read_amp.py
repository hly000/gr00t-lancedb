#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Where do the bytes go? Object-store bytes per training sample for the Lance backend.

Serves a converted dataset from a local moto S3 behind a byte-counting proxy, replays the
GR00T fast-loader read pattern (shard -> split -> ``load_split``) once per ``--rate``, then
probes how Lance turns ``fetch_blob_ranges`` calls into HTTP range requests. CPU only.

Keep ``--v2-dir`` to ~100 MB of video: copy mode packs every MP4 into one ``.blob`` object and
moto reads the whole object on each Range GET, so a full LIBERO-10 epoch takes hours.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading

from bench_real_dataset import build_full_modality_configs

from gr00t_lancedb import LanceShardedSingleStepDataset, convert_v2_to_lance

REQUESTS: list[dict] = []
_LOCK = threading.Lock()
_HOP = {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade", "proxy-connection"}


def _table_of(key: str) -> str:
    return next((t for t in ("frames", "videos", "meta") if f"/{t}.lance/" in key), "other")


def _start_counting_proxy(upstream_port: int) -> int:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _forward(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(n) if n else None
            conn = http.client.HTTPConnection("127.0.0.1", upstream_port, timeout=120)
            conn.request(self.command, self.path, body=body,
                         headers={k: v for k, v in self.headers.items() if k.lower() not in _HOP})
            resp = conn.getresponse()
            data = resp.read()
            conn.close()
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() not in _HOP and k.lower() != "content-length":
                    self.send_header(k, v)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(data)
            with _LOCK:
                REQUESTS.append({"method": self.command, "path": self.path.split("?")[0], "bytes": len(data)})

        do_GET = do_HEAD = do_PUT = do_POST = do_DELETE = _forward

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server.server_address[1]


def _summarize(reqs: list[dict], samples: int, stored: dict[str, int]) -> dict:
    by_table: dict[str, list[int]] = defaultdict(list)
    for r in reqs:
        by_table[_table_of(r["path"])].append(r["bytes"])
    total = sum(sum(v) for v in by_table.values())
    out = {
        t: {"requests": len(v), "MB": round(sum(v) / 1e6, 2), "KB_per_sample": round(sum(v) / samples / 1e3, 2),
            "read_over_stored": round(sum(v) / stored[t], 2) if stored.get(t) else None}
        for t, v in sorted(by_table.items())
    }
    out["total"] = {"samples": samples, "MB": round(total / 1e6, 2), "KB_per_sample": round(total / samples / 1e3, 2)}
    return out


def _probe_blob_coalescing(uri: str, opts: dict) -> None:
    import lancedb

    table = lancedb.connect(uri, storage_options=opts).open_table("videos")
    row_id = table.search().with_row_id(True).limit(1).to_arrow().column("_rowid")[0].as_py()
    table.fetch_blob_ranges("video_bytes", [(row_id, 0, 1)])
    for gap in (60_000, 70_000):
        mark = len(REQUESTS)
        table.fetch_blob_ranges("video_bytes", [(row_id, 0, 6000), (row_id, 6000 + gap, 6000)])
        got = [r["bytes"] for r in REQUESTS[mark:] if r["method"] == "GET" and r["bytes"]]
        print(f"probe 2 x 6KB, gap {gap // 1000}KB -> {len(got)} GET(s), {sum(got) / 1e3:.1f} KB", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--v2-dir", required=True, help="Local GR00T v2.1 dataset with meta/modality.json.")
    p.add_argument("--rates", default="0.1,1.0", help="episode_sampling_rate values to replay.")
    p.add_argument("--action-horizon", type=int, default=16)
    p.add_argument("--reader-merge-gap", type=int, default=None,
                   help="Override lerobot-lancedb's own span-merge gap (bytes) for the replay.")
    args = p.parse_args()

    for k, v in {"AWS_ACCESS_KEY_ID": "testing", "AWS_SECRET_ACCESS_KEY": "testing",
                 "AWS_DEFAULT_REGION": "us-east-1"}.items():
        os.environ.setdefault(k, v)
    import boto3
    from moto.server import ThreadedMotoServer

    moto = ThreadedMotoServer(ip_address="127.0.0.1", port=0)
    moto.start()
    _, moto_port = moto.get_host_and_port()
    s3 = boto3.client("s3", endpoint_url=f"http://127.0.0.1:{moto_port}")
    s3.create_bucket(Bucket="bench")
    proxy_port = _start_counting_proxy(moto_port)

    def opts(port: int) -> dict:
        return {"aws_endpoint": f"http://127.0.0.1:{port}", "aws_allow_http": "true", "aws_region": "us-east-1",
                "aws_access_key_id": "testing", "aws_secret_access_key": "testing"}

    uri = "s3://bench/lance"
    convert_v2_to_lance(args.v2_dir, uri, video_mode="copy", storage_options=opts(moto_port))
    stored: dict[str, int] = defaultdict(int)
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket="bench"):
        for obj in page.get("Contents", []):
            stored[_table_of("/" + obj["Key"])] += obj["Size"]
    print("stored", json.dumps({t: round(b / 1e6, 2) for t, b in sorted(stored.items())}), flush=True)

    if args.reader_merge_gap is not None:
        import lerobot_lancedb.reader as reader

        merge = reader._merge_spans
        reader._merge_spans = lambda spans, gap: merge(spans, args.reader_merge_gap)

    modality = json.loads((Path(args.v2_dir) / "meta" / "modality.json").read_text())
    cfg = build_full_modality_configs(modality, action_horizon=args.action_horizon,
                                      language_key=f"annotation.{next(iter(modality['annotation']))}")
    for rate in (float(r) for r in args.rates.split(",")):
        ds = LanceShardedSingleStepDataset(uri, "new_embodiment", cfg, episode_sampling_rate=rate,
                                           storage_options=opts(proxy_port))
        mark = len(REQUESTS)
        samples = sum(len(ds.load_split(sp)) for shard in range(len(ds)) for sp in ds.get_shard_splits(shard))
        ds.close()
        print(f"rate={rate}", json.dumps(_summarize(REQUESTS[mark:], samples, stored)), flush=True)

    _probe_blob_coalescing(uri, opts(proxy_port))


if __name__ == "__main__":
    main()
