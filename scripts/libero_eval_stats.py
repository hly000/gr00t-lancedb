#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Statistics for closed-loop evals with per-episode outcomes (standard library only).

An eval JSON written by ``eval_libero10_checkpoint.py`` lists every episode with the seed
its simulator was reset with. Two checkpoints evaluated with the same seeds therefore face
the same initial scenes, and their outcomes can be compared episode by episode:

* per-checkpoint success rate with a Wilson 95 % interval;
* paired difference (candidate - baseline) with a 95 % interval;
* exact McNemar test on the discordant episodes (one succeeded, the other failed);
* a bootstrap interval that resamples episodes within each task, as a cross-check that
  does not rely on the normal approximation.

What the pairing does and does not control: the scene is the same, the policy's own
sampling noise is not (the policy server draws fresh flow-matching noise per call). The
pairing removes scene difficulty from the comparison; it does not make an episode
deterministic.

How much 200 episodes can show: with ~15 % discordant episodes the 95 % interval on the
difference is about +-5 points. "No significant difference" at n=200 means "not worse by
more than ~5 points", not "within 2 points"; :func:`episodes_needed` gives the n for a
target margin.
"""

from __future__ import annotations

import math
import random
from typing import Any, Iterable

Z95 = 1.959963984540054


def wilson_ci(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value: ``b`` and ``c`` are the two discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2.0**n
    return min(1.0, 2.0 * tail)


def paired_diff_ci(b: int, c: int, n: int, z: float = Z95) -> tuple[float, float, float]:
    """``(diff, low, high)`` for candidate - baseline over ``n`` pairs.

    ``b``: candidate succeeded where baseline failed; ``c``: the opposite. Wald interval for
    a difference of correlated proportions.
    """
    if n <= 0:
        return (0.0, -1.0, 1.0)
    diff = (b - c) / n
    se = math.sqrt(max(0.0, (b + c) - (b - c) ** 2 / n)) / n
    return (diff, max(-1.0, diff - z * se), min(1.0, diff + z * se))


def unpaired_diff_ci(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float, float]:
    """``(diff, low, high)`` for ``k2/n2 - k1/n1`` from independent samples (Newcombe)."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson_ci(k1, n1)
    l2, u2 = wilson_ci(k2, n2)
    diff = p2 - p1
    return (diff, diff - math.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2), diff + math.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2))


def stratified_bootstrap_ci(
    diffs_by_task: dict[str, list[int]], *, iters: int = 10000, seed: int = 0
) -> tuple[float, float]:
    """95 % percentile interval of the mean paired difference, resampling within each task."""
    rng = random.Random(seed)
    tasks = [d for d in diffs_by_task.values() if d]
    total = sum(len(d) for d in tasks)
    if total == 0:
        return (-1.0, 1.0)
    means = []
    for _ in range(iters):
        acc = 0
        for d in tasks:
            acc += sum(rng.choices(d, k=len(d)))
        means.append(acc / total)
    means.sort()
    return (means[int(0.025 * (iters - 1))], means[int(math.ceil(0.975 * (iters - 1)))])


def episodes_needed(margin: float, discordant_rate: float = 0.15, z: float = Z95) -> int:
    """Pairs needed for the 95 % interval on the paired difference to be +-``margin`` when the
    two checkpoints are truly equal and disagree on ``discordant_rate`` of the episodes."""
    return int(math.ceil(z * z * discordant_rate / (margin * margin)))


# ---------------------------------------------------------------------------
# Eval JSON helpers
# ---------------------------------------------------------------------------


def episode_table(eval_json: dict[str, Any]) -> dict[tuple[str, int], bool]:
    """``{(task, seed): success}`` for every valid episode; empty for a legacy JSON that has
    only per-task rates."""
    out: dict[tuple[str, int], bool] = {}
    for task in eval_json.get("tasks", []):
        for ep in task.get("episodes") or []:
            if ep.get("success") is None or ep.get("seed") is None:
                continue
            key = (task["task"], int(ep["seed"]))
            if key in out:
                raise ValueError(f"duplicate episode {key} in eval JSON")
            out[key] = bool(ep["success"])
    return out


def summarize_episodes(successes: Iterable[bool]) -> dict[str, Any]:
    vals = [bool(s) for s in successes]
    n, k = len(vals), sum(vals)
    lo, hi = wilson_ci(k, n)
    return {
        "episodes": n,
        "successes": k,
        "success_rate": round(k / n, 4) if n else None,
        "ci95": [round(lo, 4), round(hi, 4)],
    }


def _legacy_counts(eval_json: dict[str, Any]) -> tuple[int, int]:
    """Successes / episodes reconstructed from per-task rates (legacy JSON; approximate when
    the rollout ran more episodes than requested)."""
    k = n = 0
    for task in eval_json.get("tasks", []):
        if task.get("success_rate") is None:
            continue
        n_task = int(task.get("n_episodes") or 0)
        k += round(float(task["success_rate"]) * n_task)
        n += n_task
    return k, n


def compare_evals(baseline: dict[str, Any], candidate: dict[str, Any], *, margin: float = 0.02) -> dict[str, Any]:
    """Candidate vs baseline. Paired when both JSONs carry seeded per-episode outcomes."""
    base_eps, cand_eps = episode_table(baseline), episode_table(candidate)
    seeded = bool(baseline.get("seeded")) and bool(candidate.get("seeded"))
    common = sorted(set(base_eps) & set(cand_eps))
    if not (seeded and common):
        k1, n1 = (sum(base_eps.values()), len(base_eps)) if base_eps else _legacy_counts(baseline)
        k2, n2 = (sum(cand_eps.values()), len(cand_eps)) if cand_eps else _legacy_counts(candidate)
        if not (n1 and n2):
            raise ValueError("an eval JSON has no usable episodes")
        diff, lo, hi = unpaired_diff_ci(k1, n1, k2, n2)
        return {
            "paired": False,
            "why_unpaired": "per-episode seeds missing or evals not seeded" if not seeded else "no common episodes",
            "baseline": {"episodes": n1, "successes": k1, "success_rate": round(k1 / n1, 4)},
            "candidate": {"episodes": n2, "successes": k2, "success_rate": round(k2 / n2, 4)},
            "diff": round(diff, 4),
            "diff_ci95": [round(lo, 4), round(hi, 4)],
            "mcnemar_p": None,
            "margin": margin,
            "non_inferior_at_margin": lo >= -margin,
        }

    both = base_fail_cand_ok = base_ok_cand_fail = neither = 0
    diffs_by_task: dict[str, list[int]] = {}
    for key in common:
        b_ok, c_ok = base_eps[key], cand_eps[key]
        diffs_by_task.setdefault(key[0], []).append(int(c_ok) - int(b_ok))
        if b_ok and c_ok:
            both += 1
        elif c_ok:
            base_fail_cand_ok += 1
        elif b_ok:
            base_ok_cand_fail += 1
        else:
            neither += 1
    n = len(common)
    diff, lo, hi = paired_diff_ci(base_fail_cand_ok, base_ok_cand_fail, n)
    boot_lo, boot_hi = stratified_bootstrap_ci(diffs_by_task)
    return {
        "paired": True,
        "pairs": n,
        "unmatched_baseline": len(base_eps) - n,
        "unmatched_candidate": len(cand_eps) - n,
        "baseline": summarize_episodes(base_eps[k] for k in common),
        "candidate": summarize_episodes(cand_eps[k] for k in common),
        "both_success": both,
        "candidate_only_success": base_fail_cand_ok,
        "baseline_only_success": base_ok_cand_fail,
        "both_fail": neither,
        "diff": round(diff, 4),
        "diff_ci95": [round(lo, 4), round(hi, 4)],
        "diff_ci95_task_bootstrap": [round(boot_lo, 4), round(boot_hi, 4)],
        "mcnemar_p": round(mcnemar_exact_p(base_fail_cand_ok, base_ok_cand_fail), 5),
        "margin": margin,
        "non_inferior_at_margin": lo >= -margin,
        "per_task_diff": {t: round(sum(d) / len(d), 4) for t, d in sorted(diffs_by_task.items())},
    }


def format_comparison(name: str, cmp: dict[str, Any]) -> str:
    def pct(x: float) -> str:
        return f"{100 * x:+.1f}" if x is not None else "n/a"

    b, c = cmp["baseline"], cmp["candidate"]
    head = (
        f"{name}: {100 * c['success_rate']:.1f}% ({c['successes']}/{c['episodes']}) vs baseline "
        f"{100 * b['success_rate']:.1f}% ({b['successes']}/{b['episodes']})"
    )
    lo, hi = cmp["diff_ci95"]
    line = f"  diff {pct(cmp['diff'])} pts, 95% CI [{pct(lo)}, {pct(hi)}]"
    if cmp["paired"]:
        line += (
            f" | paired n={cmp['pairs']}, discordant {cmp['candidate_only_success']} vs "
            f"{cmp['baseline_only_success']}, McNemar p={cmp['mcnemar_p']}"
        )
    else:
        line += f" | UNPAIRED ({cmp['why_unpaired']})"
    verdict = "yes" if cmp["non_inferior_at_margin"] else "not shown"
    return f"{head}\n{line}\n  non-inferior at -{100 * cmp['margin']:.0f} pts: {verdict}"
