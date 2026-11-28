#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare closed-loop eval JSONs against a baseline, episode by episode.

    python scripts/compare_libero_evals.py --baseline b0/eval.json mn2=mn2/eval.json mn4=mn4/eval.json

For every candidate: success rate, difference to the baseline with a 95 % interval, the
exact McNemar p-value on the episodes where the two disagree, and whether the lower end of
the interval stays above ``-margin`` (non-inferiority). Needs no GPU and no dependencies;
see ``libero_eval_stats.py`` for what the pairing means and what n=200 can resolve.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from libero_eval_stats import compare_evals, episodes_needed, format_comparison  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True, help="eval JSON of the reference checkpoint")
    ap.add_argument("candidates", nargs="+", help="name=path/to/eval.json (or just a path)")
    ap.add_argument("--margin", type=float, default=0.02, help="non-inferiority margin (fraction)")
    ap.add_argument("--out-json", default=None)
    ap.add_argument(
        "--require-non-inferior",
        action="store_true",
        help="exit 1 unless every candidate's 95%% interval stays above -margin",
    )
    args = ap.parse_args(argv)

    baseline = json.loads(Path(args.baseline).read_text())
    report = {"baseline": args.baseline, "margin": args.margin, "comparisons": {}}
    ok = True
    for spec in args.candidates:
        name, _, path = spec.rpartition("=")
        name = name or Path(path).parent.name or path
        cmp = compare_evals(baseline, json.loads(Path(path).read_text()), margin=args.margin)
        report["comparisons"][name] = cmp
        ok &= bool(cmp["non_inferior_at_margin"])
        print(format_comparison(name, cmp))
    print(
        f"(to resolve +-{100 * args.margin:.0f} pts at 95% with ~15% discordant episodes you need "
        f"~{episodes_needed(args.margin)} paired episodes)"
    )
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 1 if (args.require_non_inferior and not ok) else 0


if __name__ == "__main__":
    raise SystemExit(main())
