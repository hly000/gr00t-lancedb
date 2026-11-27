#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""E4: sample-level correctness of a preempted-and-resumed run vs an uninterrupted one.

Both runs must be launched with ``GR00T_SAMPLE_FINGERPRINTS=1`` (``launch_finetune_lance.py``
then logs ``sample_fps``: one hash per sample per optimizer step). Without it the script
falls back to the per-step ``batch_fp`` (batch-level agreement only).

A resumed run appends every relaunch to the same ``bench_metrics.jsonl``; steps between
the last checkpoint and the kill are executed twice. The *effective* stream is the last
execution of every ``global_step`` (that is the one whose gradients survived).

Reported (over the steps both runs cover):

``order_match``      fraction of (step, slot) positions with the same sample as the reference
``batch_match``      fraction of steps whose whole batch matches
``dup_rate``         extra copies: sum(max(0, test[h] - ref[h])) / #test samples
``miss_rate``        samples the reference saw that the test run never saw / #ref samples
``first_divergence`` first global_step whose batch differs (None = identical)
``replayed_steps``   steps executed more than once (work lost to preemption)

Usage::

    python scripts/analyze_e4_samples.py --ref /data/runs/e4/ref --test /data/runs/e4/lance_pre \\
        --test /data/runs/e4/native_pre --out results/v1/e4_summary.json
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


def load_records(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if p.is_dir():
        p = p / "bench_metrics.jsonl"
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue  # torn last line of a SIGKILLed process
    return out


def effective_steps(records: list[dict[str, Any]]) -> tuple[dict[int, dict[str, Any]], dict[str, int]]:
    """Last execution of each global_step, plus launch / replay counts."""
    eff: dict[int, dict[str, Any]] = {}
    seen: Counter[int] = Counter()
    launches = 0
    for rec in records:
        ev = rec.get("event")
        if ev == "train_begin":
            launches += 1
        elif ev == "step" and "global_step" in rec:
            gs = int(rec["global_step"])
            seen[gs] += 1
            eff[gs] = rec
    replayed = sum(1 for n in seen.values() if n > 1)
    executed = sum(seen.values())
    return eff, {"launches": launches, "replayed_steps": replayed, "executed_steps": executed}


def _samples(rec: dict[str, Any]) -> list[str] | None:
    fps = rec.get("sample_fps")
    if isinstance(fps, list) and fps:
        return [str(x) for x in fps]
    return None


def compare(ref_records: list[dict[str, Any]], test_records: list[dict[str, Any]]) -> dict[str, Any]:
    ref, ref_meta = effective_steps(ref_records)
    test, test_meta = effective_steps(test_records)
    common = sorted(set(ref) & set(test))
    out: dict[str, Any] = {
        "steps_ref": len(ref),
        "steps_test": len(test),
        "steps_compared": len(common),
        **{f"test_{k}": v for k, v in test_meta.items()},
    }
    if not common:
        return out
    sample_level = all(_samples(ref[s]) is not None and _samples(test[s]) is not None for s in common)
    out["level"] = "sample" if sample_level else "batch"

    batch_equal = 0
    first_div = None
    for s in common:
        if sample_level:
            same = _samples(ref[s]) == _samples(test[s])
        else:
            same = ref[s].get("batch_fp") is not None and ref[s].get("batch_fp") == test[s].get("batch_fp")
        batch_equal += same
        if not same and first_div is None:
            first_div = s
    out["batch_match"] = batch_equal / len(common)
    out["first_divergence"] = first_div

    if sample_level:
        pos_total = pos_equal = 0
        ref_ms: Counter[str] = Counter()
        test_ms: Counter[str] = Counter()
        for s in common:
            a, b = _samples(ref[s]), _samples(test[s])
            ref_ms.update(a)
            test_ms.update(b)
            for x, y in zip(a, b):
                pos_total += 1
                pos_equal += x == y
            pos_total += abs(len(a) - len(b))
        n_ref, n_test = sum(ref_ms.values()), sum(test_ms.values())
        dup = sum(max(0, c - ref_ms.get(h, 0)) for h, c in test_ms.items())
        miss = sum(max(0, c - test_ms.get(h, 0)) for h, c in ref_ms.items())
        out.update(
            {
                "order_match": pos_equal / pos_total if pos_total else None,
                "samples_ref": n_ref,
                "samples_test": n_test,
                "dup_rate": dup / n_test if n_test else None,
                "miss_rate": miss / n_ref if n_ref else None,
                "dup_samples": dup,
                "missed_samples": miss,
            }
        )
    return out


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{100 * v:.2f}%" if 0 <= v <= 1 else f"{v:.3f}"
    return "-" if v is None else str(v)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="Uninterrupted run dir (or bench_metrics.jsonl).")
    ap.add_argument("--test", action="append", required=True, help="Resumed run dir(s); repeatable.")
    ap.add_argument("--name", action="append", default=None, help="Display names for --test (same order).")
    ap.add_argument("--out", default=None, help="Write the JSON summary here.")
    args = ap.parse_args(argv)

    ref_records = load_records(args.ref)
    names = args.name or [Path(t).name for t in args.test]
    rows = {name: compare(ref_records, load_records(t)) for name, t in zip(names, args.test, strict=True)}

    keys = ["level", "steps_compared", "test_launches", "test_replayed_steps", "order_match", "batch_match",
            "dup_rate", "miss_rate", "first_divergence"]
    width = max(len(k) for k in keys)
    print("metric".ljust(width), *[n.rjust(16) for n in rows])
    for k in keys:
        print(k.ljust(width), *[_fmt(rows[n].get(k)).rjust(16) for n in rows])
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
