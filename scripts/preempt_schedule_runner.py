#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""E3 multi-preemption supervisor (simulates a SkyPilot managed spot job on one box).

Launches the training command given after ``--`` in its own process group, interrupts it on
a schedule, runs the group's "new instance" reset hook, relaunches with resume, and repeats
until training completes. Writes:

- ``<runner-dir>/events.jsonl``: one line per launch / preempt / escalate / exit / reset /
  quarantine / done (unix ``t`` comparable with ``bench_metrics.jsonl``);
- ``<runner-dir>/preempt_summary.json``: total wall time, interruptions, steps re-executed,
  wasted GPU seconds, cost, per-interruption breakdown, per-launch startup breakdown;
- ``<runner-dir>/launch-<k>.log``: stdout+stderr of each launch;
- ``<runner-dir>/bench_metrics.jsonl``: copy of the trainer's metrics (for plot_v1_report.py).

Schedules (JSON file, or ``--gen-poisson``)::

    {"mode": "since_launch", "times_s": [412.0, 95.3, ...]}   # i-th preemption t_i s after the i-th launch
    {"mode": "wall", "times_s": [412.0, 1630.2, ...]}         # cumulative s since the first launch
    {"mode": "at_step", "times_s": [130, 260]}                # when bench_metrics shows global_step >= v
    optional keys: "kill_mode": "kill"|"term", "grace_s": 30, "generator": {...}

Poisson schedules use exponential gaps with mean ``3600 / rate`` s; in ``since_launch`` mode
the gaps are instance lifetimes (each new instance draws the next one), which is what the
same file means for every group. A ``wall`` preemption that falls while no trainer is
running (reset / relaunch) fires right after the next launch (logged ``deferred``).

Example (group B, 2,000 steps, mean gap 10 min of instance time, GCP-like 30 s notice)::

    python scripts/preempt_schedule_runner.py --gen-poisson --rate-per-hour 6 --horizon-h 4 \
        --seed 1 --write-schedule results/v1/e3/sched_r6_s1.json
    python scripts/preempt_schedule_runner.py --schedule results/v1/e3/sched_r6_s1.json \
        --kill-mode term --grace-s 30 --output-dir /data/runs/e3/B_r6_s1 \
        --runner-dir results/v1/e3/group=B/rate=6/seed=1 \
        --reset-cmd 'bash scripts/run_group_b_rclone.sh reset' --gpu-usd-per-hour 1.20 \
        --meta group=B --meta rate_per_hour=6 --meta seed=1 -- \
        env BACKEND=native DATASET=/data/groupb/current OUT=/data/runs/e3/B_r6_s1 \
            MAX_STEPS=2000 SAVE_STEPS=250 bash scripts/run_gr00t_finetune.sh

On relaunch the resume arguments (default ``--resume_from_checkpoint``) are appended only if
a complete checkpoint exists (``checkpoint-<N>/`` containing ``trainer_state.json``);
incomplete checkpoint dirs are renamed out of the way first so the trainer cannot pick one.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import random
import re
import shlex
import shutil
import signal
import statistics as st
import subprocess
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v1_common import last_step, num, startup_breakdown, steady_stats, steps_of  # noqa: E402

SCHEDULE_MODES = ("since_launch", "wall", "at_step")
_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------


@dataclass
class Schedule:
    mode: str = "since_launch"
    times_s: list[float] = field(default_factory=list)
    kill_mode: str | None = None
    grace_s: float | None = None
    generator: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.mode not in SCHEDULE_MODES:
            raise ValueError(f"schedule mode must be one of {SCHEDULE_MODES}, got {self.mode!r}")
        if self.kill_mode not in (None, "kill", "term"):
            raise ValueError(f"kill_mode must be 'kill' or 'term', got {self.kill_mode!r}")
        self.times_s = [float(t) for t in self.times_s]
        if any(t < 0 for t in self.times_s):
            raise ValueError("schedule times must be >= 0")
        if self.mode == "wall" and self.times_s != sorted(self.times_s):
            raise ValueError("wall schedules must be sorted (cumulative offsets)")

    def to_json(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


def load_schedule(path: str | Path) -> Schedule:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):  # bare list = since_launch offsets
        return Schedule(times_s=data)
    times = data.get("times_s", data.get("times", []))
    return Schedule(
        mode=data.get("mode", "since_launch"),
        times_s=times,
        kill_mode=data.get("kill_mode"),
        grace_s=data.get("grace_s"),
        generator=data.get("generator"),
    )


def gen_poisson(rate_per_hour: float, horizon_h: float, seed: int, mode: str = "since_launch") -> Schedule:
    """Poisson arrivals on ``[0, horizon)``: exponential gaps with mean ``3600 / rate`` s."""
    if rate_per_hour <= 0 or horizon_h <= 0:
        raise ValueError("rate_per_hour and horizon_h must be > 0")
    if mode not in ("since_launch", "wall"):
        raise ValueError("Poisson schedules are 'since_launch' or 'wall'")
    rng = random.Random(seed)
    horizon_s = horizon_h * 3600.0
    gaps: list[float] = []
    t = 0.0
    while True:
        gap = rng.expovariate(rate_per_hour / 3600.0)
        if t + gap >= horizon_s:
            break
        t += gap
        gaps.append(round(gap, 3))
    times = gaps
    if mode == "wall":
        acc, times = 0.0, []
        for g in gaps:
            acc += g
            times.append(round(acc, 3))
    return Schedule(
        mode=mode,
        times_s=times,
        generator={"kind": "poisson", "rate_per_hour": rate_per_hour, "horizon_h": horizon_h, "seed": seed},
    )


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def _ckpt_step(d: Path) -> int:
    state = d / "trainer_state.json"
    if state.is_file():
        try:
            v = num(json.loads(state.read_text(encoding="utf-8")).get("global_step"))
            if v is not None:
                return int(v)
        except (json.JSONDecodeError, OSError):
            pass
    m = _CKPT_RE.match(d.name)
    return int(m.group(1)) if m else -1


def list_checkpoints(output_dir: Path, marker: str = "trainer_state.json") -> tuple[list[Path], list[Path]]:
    """``(complete, incomplete)`` ``checkpoint-<N>`` dirs, each sorted by N."""
    complete: list[Path] = []
    incomplete: list[Path] = []
    if output_dir.is_dir():
        for d in output_dir.iterdir():
            if d.is_dir() and _CKPT_RE.match(d.name):
                (complete if (d / marker).is_file() else incomplete).append(d)
    key = lambda p: int(_CKPT_RE.match(p.name).group(1))  # noqa: E731
    return sorted(complete, key=key), sorted(incomplete, key=key)


def latest_complete_checkpoint(output_dir: Path, marker: str = "trainer_state.json") -> tuple[Path | None, int]:
    complete, _ = list_checkpoints(output_dir, marker)
    if not complete:
        return None, 0
    return complete[-1], _ckpt_step(complete[-1])


def quarantine_incomplete(output_dir: Path, marker: str = "trainer_state.json") -> list[tuple[str, str]]:
    """Rename incomplete checkpoints (newer than the last complete one) so resume skips them."""
    complete, incomplete = list_checkpoints(output_dir, marker)
    moved: list[tuple[str, str]] = []
    for d in incomplete:
        dst = d.with_name(f"incomplete-{d.name}-{int(time.time() * 1000)}")
        d.rename(dst)
        moved.append((str(d), str(dst)))
    return moved


# ---------------------------------------------------------------------------
# Metrics tail
# ---------------------------------------------------------------------------


class MetricsTail:
    """Incrementally reads complete JSON lines appended to a file after ``offset``."""

    def __init__(self, path: Path, offset: int) -> None:
        self.path = path
        self.pos = offset
        self.records: list[dict[str, Any]] = []
        self._buf = b""

    def poll(self) -> None:
        try:
            with open(self.path, "rb") as fh:
                fh.seek(self.pos)
                chunk = fh.read()
        except OSError:
            return
        if not chunk:
            return
        self.pos += len(chunk)
        data = self._buf + chunk
        lines = data.split(b"\n")
        self._buf = lines.pop()  # incomplete tail (no newline yet)
        for raw in lines:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(rec, dict):
                self.records.append(rec)

    def last_step(self) -> int | None:
        return last_step(self.records)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass
class RunnerConfig:
    cmd: list[str]
    output_dir: Path
    runner_dir: Path
    schedule: Schedule = field(default_factory=Schedule)
    metrics_path: Path | None = None
    kill_mode: str = "term"
    grace_s: float = 30.0
    signal_target: str = "group"
    reset_cmd: str | None = None
    reset_before_first: bool = True
    resume_args: list[str] = field(default_factory=lambda: ["--resume_from_checkpoint"])
    ckpt_marker: str = "trainer_state.json"
    quarantine: bool = True
    max_steps: int | None = None
    max_launches: int = 100
    max_crash_restarts: int = 0
    gpu_usd_per_hour: float = 0.0
    boot_overhead_s: float = 0.0
    poll_s: float = 0.2
    copy_metrics: bool = True
    env: dict[str, str] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def metrics(self) -> Path:
        return self.metrics_path or (self.output_dir / "bench_metrics.jsonl")


@dataclass
class LaunchRecord:
    index: int
    t_launch: float
    resume_from_step: int
    resumed: bool
    t_exit: float = 0.0
    returncode: int | None = None
    preempted: bool = False
    t_signal: float | None = None
    signal: str | None = None
    escalated: bool = False
    scheduled_s: float | None = None
    deferred: bool = False
    collapsed: int = 0
    ckpt_step_at_signal: int = 0
    ckpt_step_after_exit: int = 0
    reset_s: float | None = None
    records: list[dict[str, Any]] = field(default_factory=list)


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _signal(proc: subprocess.Popen, sig: int, target: str) -> None:
    try:
        if target == "group":
            os.killpg(proc.pid, sig)
        else:
            proc.send_signal(sig)
    except ProcessLookupError:
        pass


def _reap_group(proc: subprocess.Popen, wait_s: float) -> bool:
    """Wait for the leader and every group member to exit; SIGKILL leftovers. True if killed."""
    deadline = time.time() + wait_s
    while time.time() < deadline:
        if proc.poll() is not None and not _group_alive(proc.pid):
            return False
        time.sleep(0.05)
    killed = proc.poll() is None or _group_alive(proc.pid)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()
    end = time.time() + 5.0
    while _group_alive(proc.pid) and time.time() < end:
        time.sleep(0.05)
    return killed


class Runner:
    def __init__(self, cfg: RunnerConfig) -> None:
        self.cfg = cfg
        self.cfg.runner_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = cfg.runner_dir / "events.jsonl"
        self.summary_path = cfg.runner_dir / "preempt_summary.json"
        self.launches: list[LaunchRecord] = []
        self.sched_idx = 0
        self.t0 = 0.0
        self.initial_reset_s: float | None = None
        self.metrics_offset0: int | None = None  # where this run's lines start in bench_metrics.jsonl

    # -- logging --
    def event(self, name: str, **kw: Any) -> dict[str, Any]:
        rec = {"t": time.time(), "event": name, **kw}
        with open(self.events_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
        print(f"[runner] {name} " + " ".join(f"{k}={v}" for k, v in kw.items() if k != "cmd"), file=sys.stderr)
        return rec

    # -- reset hook --
    def reset(self, reason: str, next_launch: int) -> float | None:
        if not self.cfg.reset_cmd:
            return None
        env = {**os.environ, **self.cfg.env, "RUNNER_RESET_REASON": reason,
               "RUNNER_LAUNCH_INDEX": str(next_launch), "RUNNER_OUTPUT_DIR": str(self.cfg.output_dir)}
        self.event("reset_start", reason=reason, cmd=self.cfg.reset_cmd)
        t = time.time()
        with open(self.cfg.runner_dir / f"reset-{next_launch}.log", "wb") as log:
            rc = subprocess.run(["bash", "-c", self.cfg.reset_cmd], env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        dur = time.time() - t
        self.event("reset_end", reason=reason, returncode=rc, duration_s=round(dur, 3))
        if rc != 0:
            raise RuntimeError(f"reset hook failed with exit code {rc}: {self.cfg.reset_cmd}")
        return dur

    # -- schedule --
    def _deadline(self, t_launch: float) -> tuple[float | None, float | None, bool, int]:
        """(deadline unix time | None, scheduled value, deferred, collapsed) for this launch.

        ``wall`` arrivals that fell while no trainer was running collapse into a single
        preemption right after this launch (``collapsed`` counts the extra ones dropped).
        """
        sch = self.cfg.schedule
        if self.sched_idx >= len(sch.times_s) or sch.mode == "at_step":
            return None, (sch.times_s[self.sched_idx] if self.sched_idx < len(sch.times_s) else None), False, 0
        if sch.mode == "since_launch":
            v = sch.times_s[self.sched_idx]
            return t_launch + v, v, False, 0
        collapsed = 0
        while self.sched_idx + 1 < len(sch.times_s) and self.t0 + sch.times_s[self.sched_idx + 1] <= t_launch:
            self.sched_idx += 1
            collapsed += 1
        v = sch.times_s[self.sched_idx]
        due = self.t0 + v
        return max(due, t_launch), v, due < t_launch, collapsed

    def _preempt(self, proc: subprocess.Popen, lr: LaunchRecord) -> None:
        mode = self.cfg.schedule.kill_mode or self.cfg.kill_mode
        grace = self.cfg.schedule.grace_s if self.cfg.schedule.grace_s is not None else self.cfg.grace_s
        lr.preempted = True
        _, lr.ckpt_step_at_signal = latest_complete_checkpoint(self.cfg.output_dir, self.cfg.ckpt_marker)
        lr.t_signal = time.time()
        sig = signal.SIGKILL if mode == "kill" else signal.SIGTERM
        lr.signal = sig.name
        self.event("preempt", launch=lr.index, signal=sig.name, scheduled=lr.scheduled_s, deferred=lr.deferred,
                   collapsed=lr.collapsed,
                   since_launch_s=round(lr.t_signal - lr.t_launch, 3), last_step=self._tail.last_step())
        _signal(proc, sig, self.cfg.signal_target)
        if mode == "kill":
            # A real hard preemption takes the whole machine: kill every process in the group.
            _reap_group(proc, 0.0)
            return
        lr.escalated = _reap_group(proc, grace)
        if lr.escalated:
            self.event("escalate_sigkill", launch=lr.index, after_s=round(time.time() - lr.t_signal, 3))

    def _build_cmd(self, resumed: bool) -> list[str]:
        return [*self.cfg.cmd, *(self.cfg.resume_args if resumed else [])]

    def run_once(self, index: int) -> LaunchRecord:
        cfg = self.cfg
        if cfg.quarantine:
            for src, dst in quarantine_incomplete(cfg.output_dir, cfg.ckpt_marker):
                self.event("quarantine_ckpt", src=src, dst=dst)
        ckpt, ckpt_step = latest_complete_checkpoint(cfg.output_dir, cfg.ckpt_marker)
        resumed = index > 0 and ckpt is not None
        cmd = self._build_cmd(resumed)
        offset = cfg.metrics.stat().st_size if cfg.metrics.is_file() else 0
        if self.metrics_offset0 is None:
            self.metrics_offset0 = offset
        self._tail = MetricsTail(cfg.metrics, offset)
        env = {**os.environ, **cfg.env, "RUNNER_LAUNCH_INDEX": str(index), "RUNNER_RESUME": "1" if resumed else "0",
               "RUNNER_RESUME_STEP": str(ckpt_step if resumed else 0)}
        log = open(cfg.runner_dir / f"launch-{index}.log", "wb")
        lr = LaunchRecord(index=index, t_launch=time.time(), resume_from_step=ckpt_step if resumed else 0, resumed=resumed)
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        self.event("launch", launch=index, pid=proc.pid, resumed=resumed, resume_from_step=lr.resume_from_step,
                   checkpoint=str(ckpt) if resumed else None, cmd=cmd)
        deadline, lr.scheduled_s, lr.deferred, lr.collapsed = self._deadline(lr.t_launch)
        sch = cfg.schedule
        try:
            while proc.poll() is None:
                self._tail.poll()
                fire = False
                if deadline is not None and time.time() >= deadline:
                    fire = True
                elif sch.mode == "at_step" and lr.scheduled_s is not None:
                    fire = (self._tail.last_step() or 0) >= lr.scheduled_s
                if fire:
                    self.sched_idx += 1
                    self._preempt(proc, lr)
                    break
                time.sleep(cfg.poll_s)
            if not lr.preempted:
                _reap_group(proc, 10.0)  # leftover workers of a finished/crashed trainer
        finally:
            if proc.poll() is None:
                _reap_group(proc, 0.0)
            log.close()
        lr.t_exit = time.time()
        lr.returncode = proc.returncode
        self._tail.poll()
        lr.records = list(self._tail.records)
        _, lr.ckpt_step_after_exit = latest_complete_checkpoint(cfg.output_dir, cfg.ckpt_marker)
        self.event("exit", launch=index, returncode=lr.returncode, preempted=lr.preempted,
                   duration_s=round(lr.t_exit - lr.t_launch, 3), last_step=last_step(lr.records),
                   ckpt_step=lr.ckpt_step_after_exit,
                   shutdown_s=round(lr.t_exit - lr.t_signal, 3) if lr.t_signal else None)
        return lr

    def completed(self, lr: LaunchRecord) -> bool:
        ms = self.cfg.max_steps
        if ms is None:
            return lr.returncode == 0
        end_ok = any(r.get("event") == "train_end" and (num(r.get("global_step")) or 0) >= ms for r in lr.records)
        steps_ok = (last_step(lr.records) or 0) >= ms
        return end_ok or (lr.returncode == 0 and steps_ok)

    def run(self) -> dict[str, Any]:
        cfg = self.cfg
        self.t0 = time.time()
        self.event("start", schedule=cfg.schedule.to_json(), kill_mode=cfg.kill_mode, grace_s=cfg.grace_s,
                   max_steps=cfg.max_steps, output_dir=str(cfg.output_dir), meta=cfg.meta)
        status = "running"
        crashes = 0
        try:
            if cfg.reset_before_first:
                self.initial_reset_s = self.reset("initial", 0)
            index = 0
            while True:
                if index >= cfg.max_launches:
                    status = "max_launches"
                    break
                lr = self.run_once(index)
                self.launches.append(lr)
                if not lr.preempted:
                    if self.completed(lr):
                        status = "completed"
                        break
                    crashes += 1
                    self.event("crash", launch=index, returncode=lr.returncode, crashes=crashes)
                    if crashes > cfg.max_crash_restarts:
                        status = "failed"
                        break
                lr.reset_s = self.reset("preempt" if lr.preempted else "crash", index + 1)
                index += 1
        except KeyboardInterrupt:
            status = "interrupted"
        except RuntimeError as exc:
            status = "failed"
            self.event("error", message=str(exc))
        t_end = time.time()
        summary = analyze(self.launches, cfg, t0=self.t0, t_end=t_end, initial_reset_s=self.initial_reset_s,
                          status=status, crashes=crashes)
        if cfg.copy_metrics and cfg.metrics.is_file():
            dst = cfg.runner_dir / "bench_metrics.jsonl"
            if dst.resolve() != cfg.metrics.resolve():
                # Only this run's lines: an output dir reused across repeats keeps older runs too.
                with open(cfg.metrics, "rb") as src, open(dst, "wb") as out:
                    src.seek(self.metrics_offset0 or 0)
                    shutil.copyfileobj(src, out)
        self.summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
        self.event("done", status=status, total_wall_s=summary["total_wall_s"], interruptions=summary["interruptions"])
        return summary


# ---------------------------------------------------------------------------
# Analysis (pure; unit-testable)
# ---------------------------------------------------------------------------


def _first_step_end(lr: LaunchRecord) -> tuple[float | None, float | None]:
    """(unix time the first step of this launch ended, its step_s)."""
    ss = steps_of(lr.records)
    if not ss:
        return None, None
    f = ss[0]
    t = num(f.get("t"))
    if t is None and num(f.get("since_launch_s")) is not None:
        t = lr.t_launch + float(f["since_launch_s"])
    return t, num(f.get("step_s"))


def _durable_anchor(lr: LaunchRecord, retained: int) -> float | None:
    """When the work kept in the resume checkpoint was last made durable inside this launch."""
    saves = [r for r in lr.records if r.get("event") == "save" and int(num(r.get("global_step")) or -1) == retained]
    if saves and num(saves[-1].get("t")) is not None:
        return float(saves[-1]["t"])
    steps = [r for r in steps_of(lr.records) if int(num(r.get("global_step")) or -1) == retained]
    if steps and num(steps[-1].get("t")) is not None:
        return float(steps[-1]["t"])
    return None


def analyze(
    launches: list[LaunchRecord],
    cfg: RunnerConfig,
    *,
    t0: float,
    t_end: float,
    initial_reset_s: float | None = None,
    status: str = "completed",
    crashes: int = 0,
) -> dict[str, Any]:
    all_records = [r for lr in launches for r in lr.records]
    stats = steady_stats(all_records)
    steady = stats.get("steady_step_s_median")
    step_records = [int(num(r.get("global_step")) or 0) for r in steps_of(all_records)]
    final_step = max(step_records) if step_records else 0

    interruptions: list[dict[str, Any]] = []
    for i, lr in enumerate(launches[:-1] if status == "completed" else launches):
        if not lr.preempted and lr.returncode == 0:
            continue
        nxt = launches[i + 1] if i + 1 < len(launches) else None
        retained = nxt.resume_from_step if nxt is not None else lr.ckpt_step_after_exit
        m = last_step(lr.records)
        m_eff = m if m is not None else lr.resume_from_step
        steps_lost = max(0, m_eff - retained)
        t_cut = lr.t_signal if lr.t_signal is not None else lr.t_exit
        lost_work_s = 0.0
        if m is not None and steps_lost > 0:
            anchor = _durable_anchor(lr, retained) if retained > lr.resume_from_step else None
            if anchor is None:
                fe, fs = _first_step_end(lr)
                anchor = (fe - (fs or 0.0)) if fe is not None else lr.t_launch
            lost_work_s = max(0.0, t_cut - anchor)
        rec: dict[str, Any] = {
            "launch": lr.index,
            "cause": "preempt" if lr.preempted else "crash",
            "signal": lr.signal,
            "escalated_to_sigkill": lr.escalated,
            "scheduled": lr.scheduled_s,
            "deferred": lr.deferred,
            "collapsed": lr.collapsed,
            "since_launch_s": round(t_cut - lr.t_launch, 3),
            "last_step_before": m,
            "resumed_from_step": retained,
            "ckpt_written_on_signal": lr.preempted and lr.ckpt_step_after_exit > lr.ckpt_step_at_signal,
            "steps_lost": steps_lost,
            "lost_work_s": round(lost_work_s, 3),
            "shutdown_s": round(lr.t_exit - t_cut, 3),
            "reset_s": round(lr.reset_s, 3) if lr.reset_s is not None else None,
            "relaunch_to_first_step_s": None,
            "recovery_s": None,
            "recovery_complete": False,
        }
        if nxt is not None:
            fe, _ = _first_step_end(nxt)
            if fe is not None:
                rec["relaunch_to_first_step_s"] = round(fe - nxt.t_launch, 3)
                rec["recovery_s"] = round(fe - t_cut, 3)
                rec["recovery_complete"] = True
            else:
                end = nxt.t_signal if nxt.t_signal is not None else nxt.t_exit
                rec["recovery_s"] = round(end - t_cut, 3)
        interruptions.append(rec)

    preempts = [r for r in interruptions if r["cause"] == "preempt"]
    recovery = [r["recovery_s"] for r in interruptions if r["recovery_s"] is not None]
    steps_reexecuted = sum(r["steps_lost"] for r in interruptions)
    lost_work = sum(r["lost_work_s"] for r in interruptions)
    total_wall = t_end - t0
    billable = total_wall + cfg.boot_overhead_s * len(preempts)
    launch_info = []
    for lr in launches:
        sb = startup_breakdown(lr.records)
        launch_info.append({
            "launch": lr.index,
            "resumed": lr.resumed,
            "resume_from_step": lr.resume_from_step,
            "returncode": lr.returncode,
            "preempted": lr.preempted,
            "duration_s": round(lr.t_exit - lr.t_launch, 3),
            "last_step": last_step(lr.records),
            "startup": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in sb.items()},
        })
    return {
        "status": status,
        "completed": status == "completed",
        "meta": cfg.meta,
        "cmd": cfg.cmd,
        "output_dir": str(cfg.output_dir),
        "schedule": cfg.schedule.to_json(),
        "kill_mode": cfg.schedule.kill_mode or cfg.kill_mode,
        "grace_s": cfg.schedule.grace_s if cfg.schedule.grace_s is not None else cfg.grace_s,
        "max_steps": cfg.max_steps,
        "final_step": final_step,
        "total_wall_s": round(total_wall, 3),
        "initial_reset_s": round(initial_reset_s, 3) if initial_reset_s is not None else None,
        "launches": len(launches),
        "interruptions": len(preempts),
        "crashes": crashes,
        "steps_reexecuted": steps_reexecuted,
        "step_records_total": len(step_records),
        "steps_reexecuted_from_records": len(step_records) - len(set(step_records)),
        "steady_step_s": steady,
        "recovery_s_total": round(sum(recovery), 3),
        "recovery_s_median": round(st.median(recovery), 3) if recovery else None,
        "lost_work_s_total": round(lost_work, 3),
        # wasted = work lost since the last durable save + restart overhead (signal -> first step)
        "wasted_gpu_s": round(lost_work + sum(recovery), 3),
        # design-doc formula: recovery + re-executed steps x steady step time
        "wasted_gpu_s_doc_formula": round(sum(recovery) + steps_reexecuted * steady, 3) if steady else None,
        "gpu_usd_per_hour": cfg.gpu_usd_per_hour,
        "boot_overhead_s": cfg.boot_overhead_s,
        "billable_s": round(billable, 3),
        "total_cost_usd": round(billable / 3600.0 * cfg.gpu_usd_per_hour, 4),
        "wasted_cost_usd": round((lost_work + sum(recovery) + cfg.boot_overhead_s * len(preempts)) / 3600.0
                                 * cfg.gpu_usd_per_hour, 4),
        "steady": stats,
        "interruption_details": interruptions,
        "launch_details": launch_info,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _infer_max_steps(cmd: list[str]) -> int | None:
    for i, a in enumerate(cmd):
        if a in ("--max_steps", "--max-steps") and i + 1 < len(cmd):
            try:
                return int(cmd[i + 1])
            except ValueError:
                return None
        m = re.match(r"^(?:--max[_-]steps=|MAX_STEPS=)(\d+)$", a)
        if m:
            return int(m.group(1))
    return None


def _parse_meta(items: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for it in items:
        k, sep, v = it.partition("=")
        if not sep:
            raise SystemExit(f"--meta expects key=value, got {it!r}")
        out[k] = num(v) if num(v) is not None else v
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("schedule")
    g.add_argument("--schedule", type=Path, help="Schedule JSON file")
    g.add_argument("--gen-poisson", action="store_true", help="Generate a Poisson schedule")
    g.add_argument("--rate-per-hour", type=float, default=None)
    g.add_argument("--horizon-h", type=float, default=None)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--schedule-mode", choices=("since_launch", "wall"), default="since_launch")
    g.add_argument("--write-schedule", type=Path, help="Write the (generated) schedule here")
    g.add_argument("--no-preempt", action="store_true", help="Empty schedule (E1 baseline / uninterrupted ref)")
    k = p.add_argument_group("preemption")
    k.add_argument("--kill-mode", choices=("kill", "term"), default="term",
                   help="kill = SIGKILL at once; term = SIGTERM then SIGKILL after --grace-s")
    k.add_argument("--grace-s", type=float, default=30.0, help="SIGTERM -> SIGKILL grace (GCP spot: 30)")
    k.add_argument("--signal-target", choices=("group", "leader"), default="group",
                   help="group = every process of the launch (like a VM shutdown); leader = launched process only")
    r = p.add_argument_group("relaunch")
    r.add_argument("--output-dir", type=Path, required=False, help="Trainer output dir (checkpoints)")
    r.add_argument("--metrics", type=Path, default=None, help="bench_metrics.jsonl (default <output-dir>/bench_metrics.jsonl)")
    r.add_argument("--runner-dir", type=Path, default=None, help="Runner outputs (default <output-dir>/runner)")
    r.add_argument("--reset-cmd", default=None, help="Shell command run as the 'new instance' reset (bash -c)")
    r.add_argument("--no-reset-before-first", action="store_true", help="Skip the reset before the first launch")
    r.add_argument("--resume-args", default="--resume_from_checkpoint",
                   help="Appended (shlex-split) on relaunch when a complete checkpoint exists")
    r.add_argument("--ckpt-marker", default="trainer_state.json")
    r.add_argument("--no-quarantine", action="store_true", help="Leave incomplete checkpoint dirs in place")
    r.add_argument("--max-steps", type=int, default=None, help="Completion step (default: parsed from the command)")
    r.add_argument("--max-launches", type=int, default=100)
    r.add_argument("--max-crash-restarts", type=int, default=0)
    c = p.add_argument_group("cost")
    c.add_argument("--gpu-usd-per-hour", type=float, default=0.0)
    c.add_argument("--boot-overhead-s", type=float, default=0.0,
                   help="Instance boot time added per interruption in the cost model (doc: 60-180)")
    p.add_argument("--meta", action="append", default=[], help="key=value copied into the summary (group, rate, seed)")
    p.add_argument("--poll-s", type=float, default=0.2)
    p.add_argument("--dry-run", action="store_true", help="Print schedule and commands; run nothing")
    p.add_argument("cmd", nargs=argparse.REMAINDER, help="-- training command")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd

    if args.gen_poisson:
        if args.rate_per_hour is None or args.horizon_h is None:
            raise SystemExit("--gen-poisson needs --rate-per-hour and --horizon-h")
        sched = gen_poisson(args.rate_per_hour, args.horizon_h, args.seed, args.schedule_mode)
    elif args.schedule:
        sched = load_schedule(args.schedule)
    elif args.no_preempt:
        sched = Schedule()
    else:
        raise SystemExit("give --schedule FILE, --gen-poisson ... or --no-preempt")
    if args.write_schedule:
        args.write_schedule.parent.mkdir(parents=True, exist_ok=True)
        args.write_schedule.write_text(json.dumps(sched.to_json(), indent=2) + "\n", encoding="utf-8")
        print(f"[runner] schedule ({len(sched.times_s)} preemptions, {sched.mode}) -> {args.write_schedule}", file=sys.stderr)
    if not cmd:
        if args.write_schedule:
            return 0
        raise SystemExit("no training command after --")
    if args.output_dir is None:
        raise SystemExit("--output-dir is required to run")

    cfg = RunnerConfig(
        cmd=cmd,
        output_dir=args.output_dir,
        runner_dir=args.runner_dir or (args.output_dir / "runner"),
        schedule=sched,
        metrics_path=args.metrics,
        kill_mode=args.kill_mode,
        grace_s=args.grace_s,
        signal_target=args.signal_target,
        reset_cmd=args.reset_cmd,
        reset_before_first=not args.no_reset_before_first,
        resume_args=shlex.split(args.resume_args),
        ckpt_marker=args.ckpt_marker,
        quarantine=not args.no_quarantine,
        max_steps=args.max_steps if args.max_steps is not None else _infer_max_steps(cmd),
        max_launches=args.max_launches,
        max_crash_restarts=args.max_crash_restarts,
        gpu_usd_per_hour=args.gpu_usd_per_hour,
        boot_overhead_s=args.boot_overhead_s,
        poll_s=args.poll_s,
        meta=_parse_meta(args.meta),
    )
    if args.dry_run:
        print(json.dumps({
            "schedule": sched.to_json(),
            "kill_mode": sched.kill_mode or cfg.kill_mode,
            "grace_s": sched.grace_s if sched.grace_s is not None else cfg.grace_s,
            "max_steps": cfg.max_steps,
            "reset_cmd": cfg.reset_cmd,
            "first_launch": shlex.join(cfg.cmd),
            "relaunch_with_checkpoint": shlex.join([*cfg.cmd, *cfg.resume_args]),
            "runner_dir": str(cfg.runner_dir),
        }, indent=2))
        return 0
    summary = Runner(cfg).run()
    print(json.dumps({k: summary[k] for k in ("status", "total_wall_s", "interruptions", "steps_reexecuted",
                                               "wasted_gpu_s", "total_cost_usd")}))
    return 0 if summary["completed"] else 1


if __name__ == "__main__":
    sys.exit(main())
