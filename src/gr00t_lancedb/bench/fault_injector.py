# SPDX-License-Identifier: Apache-2.0
"""Spot GPU preemption fault injector and sample-coverage auditor.

Supports both:
- Deterministic periodic or Poisson-distributed ``SIGTERM`` (graceful Spot notice) and
  ``SIGKILL`` (hard preemption) injection against any training command.
- Exact sample-coverage auditing comparing ``ResumableLanceMixtureDataset`` against
  GR00T's stock ``seed + global_step`` reseed behavior.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any, Sequence

import numpy as np


def measure_process_rss_mb(pid: int | None = None) -> float:
    """Read current RSS (in MiB) from Linux ``/proc/<pid>/status`` or ``resource`` fallback."""
    target_pid = os.getpid() if pid is None else pid
    status_file = Path(f"/proc/{target_pid}/status")
    if status_file.exists():
        try:
            for line in status_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return kb / 1024.0
        except OSError:
            pass
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS returns bytes; Linux returns KiB.
    return usage / (1024.0 * 1024.0) if os.uname().sysname == "Darwin" else usage / 1024.0


@dataclass
class RestartRecord:
    attempt: int
    exit_code: int
    wall_s: float
    injected_signal: str | None = None
    grace_exit_s: float | None = None
    peak_rss_mb: float = 0.0


@dataclass
class PreemptionRunSummary:
    command: list[str]
    completed: bool
    total_wall_s: float
    num_restarts: int
    num_injected_preemptions: int
    restarts: list[RestartRecord] = field(default_factory=list)
    gpu_hourly_usd: float | None = None

    def estimated_cost_usd(self) -> float | None:
        if self.gpu_hourly_usd is None:
            return None
        return (self.total_wall_s / 3600.0) * self.gpu_hourly_usd

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["estimated_cost_usd"] = self.estimated_cost_usd()
        return d


def run_fault_injected_command(
    cmd: Sequence[str],
    *,
    preempt_intervals_s: Sequence[float] | None = None,
    poisson_mean_s: float | None = None,
    poisson_seed: int = 42,
    signal_kind: str = "SIGTERM",
    grace_period_s: float = 30.0,
    max_restarts: int = 20,
    env: dict[str, str] | None = None,
    cwd: str | Path | None = None,
    gpu_hourly_usd: float | None = None,
) -> PreemptionRunSummary:
    """Supervise ``cmd``, injecting Spot preemptions and automatically restarting until exit 0.

    Each attempt receives ``SPOT_BENCH_ATTEMPT=<k>`` in its environment.
    """
    sig_num = signal.SIGTERM if signal_kind.upper() == "SIGTERM" else signal.SIGKILL
    rng = np.random.default_rng(poisson_seed)
    intervals = list(preempt_intervals_s) if preempt_intervals_s is not None else None

    start_all = time.monotonic()
    records: list[RestartRecord] = []
    injected_count = 0
    completed = False

    for attempt in range(max_restarts + 1):
        run_env = {**os.environ, **(env or {}), "SPOT_BENCH_ATTEMPT": str(attempt)}
        if intervals is not None:
            delay_s = intervals[attempt] if attempt < len(intervals) else None
        elif poisson_mean_s is not None and poisson_mean_s > 0:
            delay_s = float(rng.exponential(poisson_mean_s))
        else:
            delay_s = None

        t0 = time.monotonic()
        proc = subprocess.Popen(list(cmd), env=run_env, cwd=str(cwd) if cwd else None)
        peak_rss = 0.0
        injected: str | None = None
        grace_s: float | None = None

        while True:
            rc = proc.poll()
            rss = measure_process_rss_mb(proc.pid)
            if rss > peak_rss:
                peak_rss = rss
            if rc is not None:
                break
            elapsed = time.monotonic() - t0
            if delay_s is not None and elapsed >= delay_s and injected is None:
                injected = signal_kind.upper()
                injected_count += 1
                t_sig = time.monotonic()
                proc.send_signal(sig_num)
                try:
                    proc.wait(timeout=grace_period_s)
                    grace_s = time.monotonic() - t_sig
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                    grace_s = time.monotonic() - t_sig
                break
            time.sleep(0.05)

        rc = proc.wait()
        wall_s = time.monotonic() - t0
        records.append(
            RestartRecord(
                attempt=attempt,
                exit_code=rc,
                wall_s=wall_s,
                injected_signal=injected,
                grace_exit_s=grace_s,
                peak_rss_mb=peak_rss,
            )
        )
        if rc == 0 and injected is None:
            completed = True
            break

    total_wall = time.monotonic() - start_all
    return PreemptionRunSummary(
        command=list(cmd),
        completed=completed,
        total_wall_s=total_wall,
        num_restarts=max(0, len(records) - 1),
        num_injected_preemptions=injected_count,
        restarts=records,
        gpu_hourly_usd=gpu_hourly_usd,
    )


def audit_sample_stream(
    reference_ids: Sequence[Any],
    observed_ids: Sequence[Any],
) -> dict[str, Any]:
    """Compare an observed (preempted & resumed) sample stream against an uninterrupted reference."""
    ref_list = list(reference_ids)
    obs_list = list(observed_ids)
    ref_set = set(ref_list)
    obs_counts = Counter(obs_list)

    unique_obs = len(obs_counts)
    duplicates = sum(c - 1 for c in obs_counts.values() if c > 1)
    missed = len(ref_set - set(obs_counts.keys()))
    exact_order_match = obs_list == ref_list

    return {
        "reference_total": len(ref_list),
        "observed_total": len(obs_list),
        "unique_observed": unique_obs,
        "duplicate_count": duplicates,
        "duplicate_rate": duplicates / max(1, len(obs_list)),
        "missed_from_reference": missed,
        "coverage_rate": len(ref_set & set(obs_counts.keys())) / max(1, len(ref_set)),
        "exact_order_match": exact_order_match,
    }


def simulate_groot_reseed_coverage(
    total_anchors: int,
    preemption_steps: Sequence[int],
    batch_size: int = 8,
    base_seed: int = 42,
) -> dict[str, Any]:
    """Simulate GR00T's stock ``new_seed = seed + global_step`` resume over one epoch.

    Returns coverage and duplicate stats for direct comparison against exact cursor resume.
    """
    all_ids = list(range(total_anchors))
    rng = np.random.default_rng(base_seed)
    order = rng.permutation(total_anchors).tolist()

    observed: list[int] = []
    preempt_set = set(preemption_steps)
    cursor = 0
    step = 0
    while len(observed) < total_anchors:
        take = min(batch_size, total_anchors - len(observed))
        for _ in range(take):
            if cursor >= len(order):
                order = rng.permutation(total_anchors).tolist()
                cursor = 0
            observed.append(order[cursor])
            cursor += 1
        step += 1
        if step in preempt_set and len(observed) < total_anchors:
            # Mirrors Gr00tTrainer.get_train_dataloader(): new_seed = seed + global_step
            rng = np.random.default_rng(base_seed + step)
            order = rng.permutation(total_anchors).tolist()
            cursor = 0

    return audit_sample_stream(all_ids, observed)


def write_summary_json(summary: dict[str, Any], out_path: str | Path) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
