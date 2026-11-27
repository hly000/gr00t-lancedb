# SPDX-License-Identifier: Apache-2.0
"""Paired comparison of two checkpoints evaluated on the same seeded episodes.

Both rollout directories must come from ``record_libero_rollouts.py`` with the same
``--base-seed``: episode ``(task, ordinal)`` then starts from the same initial state for both
checkpoints, so each pair differs only by the policy. The exact McNemar test on discordant
pairs is far more sensitive than comparing two independent success rates.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from .rollouts import read_summary

# Decision thresholds of the fast-validation protocol (percentage points, overall).
SCALE_UP_PP = 5.0
ADD_SEED_PP = 2.0


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value (binomial test of ``b`` vs ``c`` discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


@dataclass
class PairCounts:
    n: int = 0
    a: int = 0  # baseline successes
    b: int = 0  # candidate successes
    a_only: int = 0  # baseline succeeded, candidate failed
    b_only: int = 0  # candidate succeeded, baseline failed

    def add(self, sa: bool, sb: bool) -> None:
        self.n += 1
        self.a += int(sa)
        self.b += int(sb)
        self.a_only += int(sa and not sb)
        self.b_only += int(sb and not sa)

    def as_dict(self) -> dict[str, Any]:
        ra = self.a / self.n if self.n else 0.0
        rb = self.b / self.n if self.n else 0.0
        return {
            "episodes": self.n,
            "baseline_success": self.a,
            "candidate_success": self.b,
            "baseline_rate": round(ra, 4),
            "candidate_rate": round(rb, 4),
            "diff_pp": round(100 * (rb - ra), 2),
            "baseline_ci95": [round(x, 4) for x in wilson_interval(self.a, self.n)],
            "candidate_ci95": [round(x, 4) for x in wilson_interval(self.b, self.n)],
            "baseline_only": self.a_only,
            "candidate_only": self.b_only,
            "net_flips": self.b_only - self.a_only,
            "mcnemar_p": round(mcnemar_exact_p(self.a_only, self.b_only), 5),
        }


def compare_rollouts(baseline_dir: str | Path, candidate_dir: str | Path, *, worst_k: int = 3) -> dict[str, Any]:
    base = {(r["task"], int(r["ordinal"])): r for r in read_summary(baseline_dir)}
    cand = {(r["task"], int(r["ordinal"])): r for r in read_summary(candidate_dir)}
    shared = sorted(set(base) & set(cand))
    mismatched = [k for k in shared if int(base[k]["seed"]) != int(cand[k]["seed"])]
    if mismatched:
        raise ValueError(
            f"{len(mismatched)} episode(s) have different seeds, e.g. {mismatched[0]}: "
            "record both runs with the same --base-seed"
        )
    total = PairCounts()
    per_task: dict[str, PairCounts] = defaultdict(PairCounts)
    for k in shared:
        sa, sb = bool(base[k]["success"]), bool(cand[k]["success"])
        total.add(sa, sb)
        per_task[k[0]].add(sa, sb)

    tasks = {t: c.as_dict() for t, c in sorted(per_task.items())}
    worst = sorted(tasks, key=lambda t: (tasks[t]["baseline_rate"], t))[:worst_k]
    worst_counts = PairCounts()
    for t in worst:
        c = per_task[t]
        worst_counts.n += c.n
        worst_counts.a += c.a
        worst_counts.b += c.b
        worst_counts.a_only += c.a_only
        worst_counts.b_only += c.b_only

    overall = total.as_dict()
    diff = overall["diff_pp"]
    if diff >= SCALE_UP_PP:
        verdict = "scale_up"
        advice = "clear signal: add 2 training seeds, then the G1/G2 groups"
    elif diff >= ADD_SEED_PP:
        verdict = "add_seed"
        advice = "plausible gain: one seed is not enough, add one seed before deciding"
    else:
        verdict = "revisit_weights"
        advice = "no gain: tune mining (similarity threshold, top-k, w_max) before spending GPU"
    return {
        "baseline_dir": str(baseline_dir),
        "candidate_dir": str(candidate_dir),
        "paired_episodes": len(shared),
        "baseline_only_episodes": len(set(base) - set(cand)),
        "candidate_only_episodes": len(set(cand) - set(base)),
        "overall": overall,
        f"worst{worst_k}_tasks": worst,
        f"worst{worst_k}": worst_counts.as_dict(),
        "per_task": tasks,
        "verdict": verdict,
        "advice": advice,
    }


def to_markdown(rep: dict[str, Any], *, baseline: str = "G0", candidate: str = "G3") -> str:
    o = rep["overall"]
    lines = [
        f"# {candidate} vs {baseline}: paired LIBERO evaluation",
        "",
        f"Paired episodes: {rep['paired_episodes']} (same task, same seed).",
        "",
        f"| | {baseline} | {candidate} | diff (pp) | {candidate} only / {baseline} only | McNemar p |",
        "|---|---:|---:|---:|---:|---:|",
        f"| overall | {o['baseline_rate']:.1%} | {o['candidate_rate']:.1%} | {o['diff_pp']:+.1f} | "
        f"{o['candidate_only']} / {o['baseline_only']} | {o['mcnemar_p']:.4f} |",
    ]
    wk = next(k for k in rep if k.startswith("worst") and not k.endswith("_tasks"))
    w = rep[wk]
    lines.append(
        f"| {wk} (lowest {baseline}) | {w['baseline_rate']:.1%} | {w['candidate_rate']:.1%} | {w['diff_pp']:+.1f} | "
        f"{w['candidate_only']} / {w['baseline_only']} | {w['mcnemar_p']:.4f} |"
    )
    lines += ["", "| task | n | " + baseline + " | " + candidate + " | diff (pp) |", "|---|---:|---:|---:|---:|"]
    for t, d in rep["per_task"].items():
        lines.append(
            f"| {t.split('/')[-1]} | {d['episodes']} | {d['baseline_rate']:.0%} | {d['candidate_rate']:.0%} | {d['diff_pp']:+.0f} |"
        )
    lines += ["", f"Verdict: **{rep['verdict']}** — {rep['advice']}.", ""]
    return "\n".join(lines)


def cli_compare(a) -> dict[str, Any]:
    rep = compare_rollouts(a.baseline, a.candidate, worst_k=a.worst_k)
    if a.out_json:
        Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out_json).write_text(json.dumps(rep, indent=2), encoding="utf-8")
    md = to_markdown(rep, baseline=a.baseline_name, candidate=a.candidate_name)
    if a.out_md:
        Path(a.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out_md).write_text(md, encoding="utf-8")
    print(md)
    return rep


__all__ = ["compare_rollouts", "mcnemar_exact_p", "to_markdown", "wilson_interval"]
