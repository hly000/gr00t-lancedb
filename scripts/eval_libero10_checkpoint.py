#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Evaluate any standard GR00T N1.7 checkpoint on all 10 LIBERO-10 tasks in closed-loop sim."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
from typing import Any

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


def _parse_success_rate(output: str) -> float | None:
    # Match common rollout_policy.py outputs: "Success rate: 0.80" or "80.0%" or "success_rate: 0.6"
    for pat in (
        r"success[_\s]*rate[:=\s]+([0-9]+(?:\.[0-9]+)?)\s*%",
        r"success[_\s]*rate[:=\s]+([0-9]+(?:\.[0-9]+)?)",
        r"Mean success[:=\s]+([0-9]+(?:\.[0-9]+)?)",
    ):
        m = re.findall(pat, output, flags=re.IGNORECASE)
        if m:
            val = float(m[-1])
            if "%" in pat or val > 1.0:
                return val / 100.0
            return val
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gr00t-root", default="/data/Isaac-GR00T")
    p.add_argument("--model-path", required=True)
    p.add_argument("--embodiment-tag", default="LIBERO_PANDA")
    p.add_argument("--n-episodes", type=int, default=5)
    p.add_argument("--n-envs", type=int, default=5)
    p.add_argument("--max-episode-steps", type=int, default=720)
    p.add_argument("--n-action-steps", type=int, default=8)
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--out-json", required=True)
    args = p.parse_args()

    gr00t_root = Path(args.gr00t_root)
    libero_py = gr00t_root / "gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python"
    if not libero_py.is_file():
        raise FileNotFoundError(f"LIBERO venv python not found at {libero_py}; run setup_libero.sh first")

    server_log = Path(args.out_json).with_suffix(".server.log")
    server_log.parent.mkdir(parents=True, exist_ok=True)
    server_fh = open(server_log, "w", encoding="utf-8")

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
        server_cmd,
        cwd=str(gr00t_root),
        stdout=server_fh,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )

    per_task: list[dict[str, Any]] = []
    t_eval0 = time.perf_counter()
    try:
        _wait_for_port("127.0.0.1", args.port, timeout_s=180.0, proc=server_proc)
        print(f"[eval] Policy server ready on port {args.port}!", flush=True)

        env = os.environ.copy()
        env.setdefault("MUJOCO_GL", "egl")
        env.setdefault("PYOPENGL_PLATFORM", "egl")

        for idx, task in enumerate(LIBERO_10_TASKS):
            t_t0 = time.perf_counter()
            client_cmd = [
                str(libero_py),
                "gr00t/eval/rollout_policy.py",
                "--n-episodes",
                str(args.n_episodes),
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
                "--n-envs",
                str(min(args.n_envs, args.n_episodes)),
            ]
            res = subprocess.run(
                client_cmd,
                cwd=str(gr00t_root),
                env=env,
                capture_output=True,
                text=True,
                timeout=600,
            )
            combined = res.stdout + "\n" + res.stderr
            sr = _parse_success_rate(combined)
            dt = time.perf_counter() - t_t0
            short_name = task.split("/", 1)[-1]
            print(
                f"[eval] ({idx + 1}/10) {short_name}: success_rate={sr} ({dt:.1f}s, rc={res.returncode})",
                flush=True,
            )
            if sr is None:
                print(combined[-1200:], flush=True)
            per_task.append(
                {
                    "task": task,
                    "n_episodes": args.n_episodes,
                    "success_rate": sr,
                    "wall_s": round(dt, 2),
                    "returncode": res.returncode,
                    "tail": combined[-600:],
                }
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

    valid_srs = [t["success_rate"] for t in per_task if t["success_rate"] is not None]
    mean_sr = float(sum(valid_srs) / len(valid_srs)) if valid_srs else None
    summary = {
        "model_path": str(args.model_path),
        "n_episodes_per_task": args.n_episodes,
        "total_episodes": args.n_episodes * len(LIBERO_10_TASKS),
        "mean_success_rate": round(mean_sr, 4) if mean_sr is not None else None,
        "eval_wall_s": round(time.perf_counter() - t_eval0, 2),
        "tasks": per_task,
    }
    Path(args.out_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[eval] SUMMARY ({args.model_path}): mean_success_rate={summary['mean_success_rate']} ({summary['eval_wall_s']}s)", flush=True)


if __name__ == "__main__":
    main()
