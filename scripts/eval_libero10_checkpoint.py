#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Evaluate a GR00T N1.7 checkpoint on LIBERO-10 with an exact, seeded episode list.

Every episode is its own rollout process with one simulator and one explicit seed
(``seed_base + 1000 * task_index + episode_index``); ``--n-envs`` of them run side by side
against one policy server. The JSON records the outcome of every episode next to its seed,
so two checkpoints evaluated with the same ``--seed-base`` can be compared episode by
episode (``compare_libero_evals.py``: paired difference, 95 % interval, exact McNemar).

Why not one vectorised rollout per task, as before: ``rollout_policy.py`` stops as soon as
*at least* ``n_episodes`` have finished across its parallel simulators. Two consequences:

* several simulators can finish in the same step, so a task asked for 5 episodes sometimes
  reported 6 (the 66.7 % / 50 % per-task rates in the archived n=5 runs);
* a simulator that finishes early starts another episode. Successes end early
  (``terminate_on_success``) and failures run to the step limit, so the first ``n`` episodes
  to finish over-represent successes. The reported rate is biased upwards, more so for
  weaker checkpoints.

One simulator per process has neither problem, and gives every episode a known seed. The
cost is one simulator start-up per episode (~10-15 s).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

from libero_eval_stats import summarize_episodes  # noqa: E402

LIBERO_10_TASKS = [
    "libero_sim/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
    "libero_sim/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
    "libero_sim/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
    "libero_sim/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
    "libero_sim/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
    "libero_sim/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
    "libero_sim/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
    "libero_sim/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
    "libero_sim/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
    "libero_sim/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
]

SCHEMA = 2
SEED_STRIDE = 1000  # seeds of different tasks never overlap for up to 1000 episodes per task

EpisodeRunner = Callable[[str, int], dict[str, Any]]


def episode_seed(seed_base: int, task_index: int, episode_index: int) -> int:
    if not 0 <= episode_index < SEED_STRIDE:
        raise ValueError(f"episode_index must be in [0, {SEED_STRIDE}), got {episode_index}")
    return seed_base + SEED_STRIDE * task_index + episode_index


def parse_episode_outcomes(output: str) -> list[bool] | None:
    """Per-episode successes from ``rollout_policy.py`` output, or None if it printed none.

    Reads the ``results: (env_name, [successes], {...})`` line, whose list has one entry per
    finished episode. Handles ``True``, ``np.True_`` and ``np.bool_(True)`` reprs.
    """
    matches = re.findall(r"results:\s*\(\s*(['\"]).*?\1\s*,\s*\[(.*?)\]", output, flags=re.DOTALL)
    if not matches:
        return None
    return [tok == "True" for tok in re.findall(r"True|False", matches[-1][1])]


def parse_success_rate(output: str) -> float | None:
    """Fallback for rollout scripts that only print ``success rate: x``."""
    m = re.findall(r"success[_\s]*rate[:=\s]+([0-9]+(?:\.[0-9]+)?)", output, flags=re.IGNORECASE)
    return float(m[-1]) if m else None


def single_episode_outcome(output: str) -> bool | None:
    """Outcome of a rollout that was asked for exactly one episode; None if the output does
    not describe exactly one episode (never guess: such a run is retried, then reported)."""
    outcomes = parse_episode_outcomes(output)
    if outcomes is not None:
        return outcomes[0] if len(outcomes) == 1 else None
    rate = parse_success_rate(output)
    if rate in (0.0, 1.0):
        return bool(rate)
    return None


def evaluate_tasks(
    tasks: list[str],
    n_episodes: int,
    seed_base: int,
    run_episode: EpisodeRunner,
    *,
    parallel: int = 1,
    retries: int = 1,
    log: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """Run exactly ``n_episodes`` seeded episodes per task. ``run_episode(task, seed)`` returns
    a dict with at least ``success`` (bool, or None when the run produced no outcome)."""

    def _one(task: str, seed: int, index: int) -> dict[str, Any]:
        rec: dict[str, Any] = {}
        for attempt in range(1, retries + 2):
            rec = dict(run_episode(task, seed))
            rec.update(episode=index, seed=seed, attempts=attempt)
            if rec.get("success") is not None:
                rec.pop("tail", None)
                break
        return rec

    per_task = []
    with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
        for t_idx, task in enumerate(tasks):
            t0 = time.perf_counter()
            seeds = [episode_seed(seed_base, t_idx, e) for e in range(n_episodes)]
            episodes = list(pool.map(lambda a, task=task: _one(task, a[1], a[0]), enumerate(seeds)))
            valid = [e["success"] for e in episodes if e["success"] is not None]
            rate = sum(valid) / len(valid) if valid else None
            per_task.append(
                {
                    "task": task,
                    "n_episodes": n_episodes,
                    "valid_episodes": len(valid),
                    "successes": int(sum(valid)),
                    "success_rate": round(rate, 4) if rate is not None else None,
                    "wall_s": round(time.perf_counter() - t0, 2),
                    "episodes": episodes,
                }
            )
            log(
                f"[eval] ({t_idx + 1}/{len(tasks)}) {task.split('/', 1)[-1]}: "
                f"{int(sum(valid))}/{len(valid)} success"
                + (f", {n_episodes - len(valid)} episode(s) without outcome" if len(valid) != n_episodes else "")
                + f" ({per_task[-1]['wall_s']:.0f}s)"
            )
    return per_task


def build_summary(per_task: list[dict[str, Any]], **meta: Any) -> dict[str, Any]:
    outcomes = [e["success"] for t in per_task for e in t["episodes"] if e["success"] is not None]
    total = sum(t["n_episodes"] for t in per_task)
    stats = summarize_episodes(outcomes)
    return {
        "schema": SCHEMA,
        **meta,
        "total_episodes": total,
        "valid_episodes": stats["episodes"],
        "successes": stats["successes"],
        # Fraction of valid episodes. With complete tasks this equals the mean of task rates.
        "mean_success_rate": stats["success_rate"],
        "ci95": stats["ci95"],
        "complete": stats["episodes"] == total,
        "tasks": per_task,
    }


# ---------------------------------------------------------------------------
# Real runner: policy server + one rollout process per episode
# ---------------------------------------------------------------------------


def _wait_for_port(host: str, port: int, timeout_s: float = 180.0, proc: subprocess.Popen | None = None) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"Policy server exited prematurely with code {proc.returncode}")
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError:
            time.sleep(1.0)
    raise TimeoutError(f"Timed out waiting for policy server on {host}:{port}")


def build_client_cmd(libero_py: str, task: str, seed: int | None, args: argparse.Namespace) -> list[str]:
    cmd = [
        libero_py,
        "gr00t/eval/rollout_policy.py",
        "--n-episodes",
        "1",
        "--n-envs",
        "1",
        "--policy-client-host",
        "127.0.0.1",
        "--policy-client-port",
        str(args.port),
        "--max-episode-steps",
        str(args.max_episode_steps),
        "--env-name",
        task,
        "--n-action-steps",
        str(args.n_action_steps),
    ]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    return cmd


def rollout_supports_seed(libero_py: str, gr00t_root: Path) -> bool:
    res = subprocess.run(
        [libero_py, "gr00t/eval/rollout_policy.py", "--help"],
        cwd=str(gr00t_root),
        capture_output=True,
        text=True,
        timeout=300,
    )
    return "--seed" in (res.stdout + res.stderr)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gr00t-root", default="/data/Isaac-GR00T")
    p.add_argument("--model-path", required=True)
    p.add_argument("--embodiment-tag", default="LIBERO_PANDA")
    p.add_argument("--n-episodes", type=int, default=20, help="episodes per task (exact)")
    p.add_argument("--n-envs", type=int, default=5, help="rollout processes running side by side")
    p.add_argument("--seed-base", type=int, default=20260, help="same value for every checkpoint you compare")
    p.add_argument("--max-episode-steps", type=int, default=720)
    p.add_argument("--n-action-steps", type=int, default=8)
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--episode-timeout", type=float, default=900.0)
    p.add_argument("--retries", type=int, default=1, help="re-runs (same seed) of an episode without outcome")
    p.add_argument("--tasks", default=None, help="comma list of task indices (0-9); default all")
    p.add_argument(
        "--allow-unseeded",
        action="store_true",
        help="run even if rollout_policy.py has no --seed (results cannot be paired)",
    )
    p.add_argument("--allow-missing", action="store_true", help="exit 0 even if some episodes have no outcome")
    p.add_argument("--out-json", required=True)
    args = p.parse_args()

    gr00t_root = Path(args.gr00t_root)
    libero_py = gr00t_root / "gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python"
    if not libero_py.is_file():
        raise FileNotFoundError(f"LIBERO venv python not found at {libero_py}; run setup_libero.sh first")
    task_ids = [int(x) for x in args.tasks.split(",")] if args.tasks else list(range(len(LIBERO_10_TASKS)))

    seeded = rollout_supports_seed(str(libero_py), gr00t_root)
    if not seeded and not args.allow_unseeded:
        raise SystemExit(
            f"FAIL: {gr00t_root}/gr00t/eval/rollout_policy.py has no --seed option, so episodes cannot be "
            "paired across checkpoints. Update Isaac-GR00T, or pass --allow-unseeded for an unpaired eval."
        )

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    server_fh = open(out_json.with_suffix(".server.log"), "w", encoding="utf-8")
    server_cmd = [
        "uv",
        "run",
        "--no-sync",
        "python",
        "gr00t/eval/run_gr00t_server.py",
        "--model-path",
        str(args.model_path),
        "--embodiment-tag",
        args.embodiment_tag,
        "--port",
        str(args.port),
        "--use-sim-policy-wrapper",
    ]
    print(f"[eval] Launching policy server: {' '.join(server_cmd)}", flush=True)
    server_proc = subprocess.Popen(
        server_cmd, cwd=str(gr00t_root), stdout=server_fh, stderr=subprocess.STDOUT, preexec_fn=os.setsid
    )

    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")

    def run_episode(task: str, seed: int) -> dict[str, Any]:
        t0 = time.perf_counter()
        try:
            res = subprocess.run(
                build_client_cmd(str(libero_py), task, seed if seeded else None, args),
                cwd=str(gr00t_root),
                env=env,
                capture_output=True,
                text=True,
                timeout=args.episode_timeout,
            )
            combined, rc = res.stdout + "\n" + res.stderr, res.returncode
        except subprocess.TimeoutExpired as exc:
            combined, rc = f"timeout after {args.episode_timeout}s\n{exc.stdout or ''}", -9
        success = single_episode_outcome(combined) if rc == 0 else None
        rec = {"success": success, "wall_s": round(time.perf_counter() - t0, 2), "returncode": rc}
        if success is None:
            rec["tail"] = combined[-800:]
        return rec

    per_task: list[dict[str, Any]] = []
    t_eval0 = time.perf_counter()
    try:
        _wait_for_port("127.0.0.1", args.port, timeout_s=180.0, proc=server_proc)
        print(
            f"[eval] Policy server ready on port {args.port}; {args.n_episodes} ep/task, "
            f"seed_base={args.seed_base}, seeded={seeded}, {args.n_envs} rollouts in parallel",
            flush=True,
        )
        per_task = evaluate_tasks(
            [LIBERO_10_TASKS[i] for i in task_ids],
            args.n_episodes,
            args.seed_base,
            run_episode,
            parallel=args.n_envs,
            retries=args.retries,
            log=lambda s: print(s, flush=True),
        )
    finally:
        try:
            os.killpg(os.getpgid(server_proc.pid), signal.SIGTERM)
        except OSError:
            pass
        try:
            server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(server_proc.pid), signal.SIGKILL)
            except OSError:
                pass
        server_fh.close()

    summary = build_summary(
        per_task,
        model_path=str(args.model_path),
        n_episodes_per_task=args.n_episodes,
        seed_base=args.seed_base,
        seeded=seeded,
        episode_runner="one rollout process per episode (n_envs=1)",
        eval_wall_s=round(time.perf_counter() - t_eval0, 2),
    )
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    rate = summary["mean_success_rate"]
    print(
        f"[eval] SUMMARY ({args.model_path}): "
        + (f"{100 * rate:.2f}%" if rate is not None else "no valid episodes")
        + f" ({summary['successes']}/{summary['valid_episodes']}, 95% CI {summary['ci95']}, "
        f"{summary['eval_wall_s']}s)",
        flush=True,
    )
    if not summary["complete"]:
        missing = summary["total_episodes"] - summary["valid_episodes"]
        print(f"[eval] {missing} episode(s) produced no outcome (see 'tail' in {out_json})", flush=True)
        if not args.allow_missing:
            raise SystemExit(3)


if __name__ == "__main__":
    main()
