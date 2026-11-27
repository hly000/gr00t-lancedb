# SPDX-License-Identifier: Apache-2.0
"""CLI entrypoint for ``gr00t-spot-bench``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from .._isolated import lance_mp_context

from .._gr00t_compat import ModalityConfig
from ..resumable_mixture import ResumableLanceMixtureDataset
from ..sharded_dataset import LanceShardedSingleStepDataset
from .fault_injector import (
    audit_sample_stream,
    measure_process_rss_mb,
    run_fault_injected_command,
    simulate_groot_reseed_coverage,
    write_summary_json,
)


def _identity_collate(items: list) -> list:
    return items


def _cmd_throughput(args: argparse.Namespace) -> None:
    modality_configs = {
        "video": ModalityConfig(delta_indices=[0], modality_keys=args.video_keys),
        "state": ModalityConfig(delta_indices=[0], modality_keys=args.state_keys),
        "action": ModalityConfig(
            delta_indices=list(range(args.action_horizon)), modality_keys=args.action_keys
        ),
        "language": ModalityConfig(delta_indices=[0], modality_keys=[args.language_key]),
    }

    t0 = time.perf_counter()
    ds = LanceShardedSingleStepDataset(
        dataset_path=args.root,
        embodiment_tag=args.embodiment_tag,
        modality_configs=modality_configs,
        modality_json_path=args.modality_json,
        allow_padding=True,
    )
    mix = ResumableLanceMixtureDataset(
        datasets=[ds],
        weights=[1.0],
        batch_size=args.batch_size,
        episode_pool_size=args.episode_pool_size,
        prefetch_anchors=args.prefetch_anchors,
        infinite=True,
    )
    init_s = time.perf_counter() - t0

    seen = 0
    t_first: float | None = None
    t_measure_start: float | None = None
    peak_rss = measure_process_rss_mb()

    for step_idx, batch in enumerate(
        mix.loader(num_workers=args.num_workers, collate_fn=_identity_collate)
    ):
        now = time.perf_counter()
        if t_first is None:
            t_first = now - t0
        if step_idx == args.warmup_steps:
            t_measure_start = now
            seen = 0
        if step_idx >= args.warmup_steps:
            seen += len(batch)
        rss = measure_process_rss_mb()
        if rss > peak_rss:
            peak_rss = rss
        if step_idx + 1 >= args.warmup_steps + args.steps:
            break

    measure_s = max(1e-6, time.perf_counter() - (t_measure_start or t0))
    report = {
        "root": args.root,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "mp_context": lance_mp_context() if args.num_workers > 0 else "main",
        "init_s": round(init_s, 4),
        "ttfs_s": round(t_first or 0.0, 4),
        "measured_samples": seen,
        "measured_s": round(measure_s, 4),
        "samples_per_sec": round(seen / measure_s, 2),
        "peak_rss_mb": round(peak_rss, 1),
    }
    print(json.dumps(report, indent=2))
    if args.out:
        write_summary_json(report, args.out)


def _cmd_coverage_sim(args: argparse.Namespace) -> None:
    preempt_steps = [int(x) for x in args.preempt_steps.split(",") if x.strip()]
    groot_reseed = simulate_groot_reseed_coverage(
        total_anchors=args.total_anchors,
        preemption_steps=preempt_steps,
        batch_size=args.batch_size,
        base_seed=args.seed,
    )
    exact_resume = audit_sample_stream(list(range(args.total_anchors)), list(range(args.total_anchors)))
    report = {
        "total_anchors": args.total_anchors,
        "batch_size": args.batch_size,
        "preemption_steps": preempt_steps,
        "groot_stock_reseed": groot_reseed,
        "gr00t_lancedb_exact_resume": exact_resume,
    }
    print(json.dumps(report, indent=2))
    if args.out:
        write_summary_json(report, args.out)


def _cmd_inject(args: argparse.Namespace) -> None:
    intervals = [float(x) for x in args.intervals.split(",")] if args.intervals else None
    summary = run_fault_injected_command(
        args.command,
        preempt_intervals_s=intervals,
        poisson_mean_s=args.poisson_mean_s,
        poisson_seed=args.seed,
        signal_kind=args.signal,
        grace_period_s=args.grace_period_s,
        max_restarts=args.max_restarts,
        gpu_hourly_usd=args.gpu_hourly_usd,
    )
    print(json.dumps(summary.to_dict(), indent=2))
    if args.out:
        write_summary_json(summary.to_dict(), args.out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="gr00t-spot-bench")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p_tp = sub.add_parser("throughput", help="Measure TTFS, steady-state throughput, and peak RSS.")
    p_tp.add_argument("--root", required=True)
    p_tp.add_argument("--modality-json", default=None)
    p_tp.add_argument("--embodiment-tag", default="new_embodiment")
    p_tp.add_argument("--video-keys", nargs="+", required=True)
    p_tp.add_argument("--state-keys", nargs="+", required=True)
    p_tp.add_argument("--action-keys", nargs="+", required=True)
    p_tp.add_argument("--language-key", default="task")
    p_tp.add_argument("--action-horizon", type=int, default=16)
    p_tp.add_argument("--batch-size", type=int, default=8)
    p_tp.add_argument("--num-workers", type=int, default=2)
    p_tp.add_argument("--episode-pool-size", type=int, default=16)
    p_tp.add_argument("--prefetch-anchors", type=int, default=16)
    p_tp.add_argument("--warmup-steps", type=int, default=2)
    p_tp.add_argument("--steps", type=int, default=20)
    p_tp.add_argument("--out", default=None)
    p_tp.set_defaults(func=_cmd_throughput)

    p_cov = sub.add_parser(
        "coverage-compare",
        help="Compare sample duplicate/miss rates: GR00T seed+global_step vs exact Lance resume.",
    )
    p_cov.add_argument("--total-anchors", type=int, default=10000)
    p_cov.add_argument("--batch-size", type=int, default=32)
    p_cov.add_argument("--preempt-steps", default="50,120,210")
    p_cov.add_argument("--seed", type=int, default=42)
    p_cov.add_argument("--out", default=None)
    p_cov.set_defaults(func=_cmd_coverage_sim)

    p_inj = sub.add_parser(
        "inject",
        help="Supervise a training command and inject periodic or Poisson Spot preemptions.",
    )
    p_inj.add_argument("--intervals", default=None, help="Comma-separated preemption delays in seconds.")
    p_inj.add_argument("--poisson-mean-s", type=float, default=None)
    p_inj.add_argument("--signal", choices=("SIGTERM", "SIGKILL"), default="SIGTERM")
    p_inj.add_argument("--grace-period-s", type=float, default=30.0)
    p_inj.add_argument("--max-restarts", type=int, default=20)
    p_inj.add_argument("--gpu-hourly-usd", type=float, default=None)
    p_inj.add_argument("--seed", type=int, default=42)
    p_inj.add_argument("--out", default=None)
    p_inj.add_argument("command", nargs=argparse.REMAINDER)
    p_inj.set_defaults(func=_cmd_inject)

    args = parser.parse_args(argv)
    if args.subcommand == "inject" and args.command and args.command[0] == "--":
        args.command = args.command[1:]
    args.func(args)


if __name__ == "__main__":
    main()
