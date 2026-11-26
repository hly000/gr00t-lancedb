#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Compare per-step batch fingerprints (``batch_fp``) of two runs.

    python scripts/compare_fingerprints.py REF_RUN OTHER_RUN [--after STEP]

A run that was preempted and resumed logs some steps twice (the steps redone after
restarting from an older checkpoint); the *last* record for each step is what the
final model trained on, and every logged record is counted for "redone" steps.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def load_steps(run: Path) -> tuple[dict[int, str], Counter, list[int]]:
    last: dict[int, str] = {}
    seen: Counter = Counter()
    segments: list[int] = []
    for line in (run / "bench_metrics.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["event"] == "train_begin":
            segments.append(int(r["global_step"]))
        if r["event"] == "step" and "batch_fp" in r:
            last[r["global_step"]] = r["batch_fp"]
            seen[r["global_step"]] += 1
    return last, seen, segments


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("ref")
    p.add_argument("other")
    p.add_argument("--after", type=int, default=0, help="only compare steps > AFTER")
    p.add_argument("--json", action="store_true")
    a = p.parse_args()

    ref, _, _ = load_steps(Path(a.ref))
    oth, seen, segments = load_steps(Path(a.other))
    common = sorted(s for s in set(ref) & set(oth) if s > a.after)
    match = [s for s in common if ref[s] == oth[s]]
    mismatch = [s for s in common if ref[s] != oth[s]]
    # Batches the reference trained on that never appear anywhere in the other run.
    ref_set, oth_set = set(ref[s] for s in common), set(oth[s] for s in common)
    out = {
        "ref": a.ref,
        "other": a.other,
        "segments_start_steps": segments,
        "redone_steps": sum(1 for s, c in seen.items() if c > 1),
        "compared_steps": len(common),
        "matching_steps": len(match),
        "first_mismatch": mismatch[0] if mismatch else None,
        "ref_batches_missing_in_other": len(ref_set - oth_set),
    }
    print(json.dumps(out) if a.json else json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
