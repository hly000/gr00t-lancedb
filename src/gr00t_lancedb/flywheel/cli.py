# SPDX-License-Identifier: Apache-2.0
"""``gr00t-lance-flywheel {mine,compare,success}``."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json

from .rollouts import read_summary


def _success(a: argparse.Namespace) -> dict:
    recs = read_summary(a.rollouts)
    by_task: dict[str, list[bool]] = defaultdict(list)
    for r in recs:
        by_task[r["task"]].append(bool(r["success"]))
    from .paired_eval import wilson_interval

    n = len(recs)
    k = sum(r["success"] for r in recs)
    out = {
        "rollouts": a.rollouts,
        "episodes": n,
        "success_rate": round(k / n, 4) if n else 0.0,
        "ci95": [round(x, 4) for x in wilson_interval(k, n)],
        "per_task": {t: round(sum(v) / len(v), 4) for t, v in sorted(by_task.items())},
    }
    print(json.dumps(out, indent=2))
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="gr00t-lance-flywheel", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    from .mining import add_mining_args

    pm = sub.add_parser("mine", help="mine failed rollouts and write the weight column")
    add_mining_args(pm)

    pc = sub.add_parser("compare", help="seed-paired success comparison of two rollout dirs")
    pc.add_argument("--baseline", required=True)
    pc.add_argument("--candidate", required=True)
    pc.add_argument("--baseline-name", default="G0")
    pc.add_argument("--candidate-name", default="G3")
    pc.add_argument("--worst-k", type=int, default=3)
    pc.add_argument("--out-json", default=None)
    pc.add_argument("--out-md", default=None)

    ps = sub.add_parser("success", help="success rate of one rollout dir")
    ps.add_argument("--rollouts", required=True)

    a = p.parse_args(argv)
    if a.cmd == "mine":
        from .mining import cli_mine

        cli_mine(a)
    elif a.cmd == "compare":
        from .paired_eval import cli_compare

        cli_compare(a)
    else:
        _success(a)


if __name__ == "__main__":  # pragma: no cover
    main()
