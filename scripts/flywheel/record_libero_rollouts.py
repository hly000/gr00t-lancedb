#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Record seeded LIBERO rollouts of a GR00T policy server for failure mining / paired eval.

Runs as the *client* in Isaac-GR00T's LIBERO venv, next to ``gr00t/eval/run_gr00t_server.py``
(started with ``--use-sim-policy-wrapper``). Compared with ``gr00t/eval/rollout_policy.py``:

* **Paired seeds.** Episodes run in waves of ``--n-envs``; every episode gets its own reset
  seed ``base_seed + 1000 * task_id + ordinal``, so two checkpoints recorded with the same
  ``--base-seed`` see the same initial states episode by episode (the stock loop only seeds
  the first reset of each sub-env). Envs that finish early idle until the wave ends.
* **Per-decision data.** For every policy call: the observed ``state.*`` (last frame), the
  returned ``action.*`` chunk, the env steps executed, and (failures by default) a small
  thumbnail of the main camera. Written in the ``gr00t_lancedb.rollout/v1`` format.

Usage (from the Isaac-GR00T root)::

    gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python \\
        /path/to/gr00t-lancedb/scripts/flywheel/record_libero_rollouts.py \\
        --out runs/g0/eval_14000 --suite libero_10 --episodes-per-task 50 \\
        --policy-client-host 127.0.0.1 --policy-client-port 5555 --n-envs 10 --base-seed 7
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
_ROLLOUTS_PY = HERE.parents[1] / "src" / "gr00t_lancedb" / "flywheel" / "rollouts.py"

LIBERO_10 = [
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
SUITES = {"libero_10": LIBERO_10}
TASK_DESCRIPTION_KEY = "annotation.human.action.task_description"


def _load_rollouts_module():
    """Load ``rollouts.py`` by path: the sim venv need not have gr00t-lancedb installed."""
    spec = importlib.util.spec_from_file_location("_gr00t_lancedb_rollouts", _ROLLOUTS_PY)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _any_true(x) -> bool:
    if x is None:
        return False
    if isinstance(x, (list, tuple, np.ndarray)):
        return bool(np.any(np.asarray(x, dtype=bool)))
    return bool(x)


def _step_success(env_infos: dict, i: int) -> bool:
    ok = False
    if "success" in env_infos and env_infos["success"] is not None:
        ok |= _any_true(env_infos["success"][i])
    final = env_infos.get("final_info")
    if final is not None and final[i] is not None and "success" in final[i]:
        ok |= _any_true(final[i]["success"])
    return ok


def _last_frame(x, frame_ndim: int) -> np.ndarray:
    """Drop MultiStepWrapper's time axis (if present) by taking the newest frame."""
    arr = np.asarray(x)
    return arr[-1] if arr.ndim > frame_ndim else arr


def _thumb(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    sy, sx = max(1, h // size), max(1, w // size)
    return np.ascontiguousarray(img[::sy, ::sx][:size, :size]).astype(np.uint8)


def _record_wave(env, policy, n_envs, seeds, max_decisions, image_key, thumb, step_fn):
    observations, _ = env.reset(seed=[int(s) for s in seeds])
    policy.reset()
    state_keys = [k for k in observations if k.startswith("state.")]
    done = np.zeros(n_envs, dtype=bool)
    success = np.zeros(n_envs, dtype=bool)
    env_step = np.zeros(n_envs, dtype=np.int64)
    rec = [
        {"env_steps": [], "n_env_steps": [], "state": {k[6:]: [] for k in state_keys}, "action": {}, "image": []}
        for _ in range(n_envs)
    ]
    desc = [""] * n_envs
    for _ in range(max_decisions):
        if done.all():
            break
        actions, _ = policy.get_action(observations)
        next_obs, _, terminations, truncations, env_infos = env.step(actions)
        for i in range(n_envs):
            if done[i]:
                continue
            r = rec[i]
            if not desc[i] and TASK_DESCRIPTION_KEY in observations:
                desc[i] = str(observations[TASK_DESCRIPTION_KEY][i])
            r["env_steps"].append(int(env_step[i]))
            for k in state_keys:
                r["state"][k[6:]].append(_last_frame(observations[k][i], 1).reshape(-1))
            for k, v in actions.items():
                if k.startswith("action."):
                    chunk = np.asarray(v[i])
                    r["action"].setdefault(k[7:], []).append(chunk.reshape(chunk.shape[0], -1) if chunk.ndim else chunk.reshape(1, 1))
            if image_key in observations:
                r["image"].append(_thumb(_last_frame(observations[image_key][i], 3), thumb))
            n = max(1, step_fn(env_infos, i))
            r["n_env_steps"].append(n)
            env_step[i] += n
            success[i] |= _step_success(env_infos, i)
            if terminations[i] or truncations[i]:
                done[i] = True
        observations = next_obs
    return rec, success, desc


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="rollout directory (summary.jsonl + per-task npz)")
    p.add_argument("--suite", default="libero_10", choices=sorted(SUITES))
    p.add_argument("--tasks", nargs="*", default=None, help="env names (default: the whole suite)")
    p.add_argument("--episodes-per-task", type=int, default=50)
    p.add_argument("--n-envs", type=int, default=10)
    p.add_argument("--base-seed", type=int, default=7)
    p.add_argument("--max-episode-steps", type=int, default=720)
    p.add_argument("--n-action-steps", type=int, default=8)
    p.add_argument("--policy-client-host", default="127.0.0.1")
    p.add_argument("--policy-client-port", type=int, default=5555)
    p.add_argument("--image-key", default="video.image")
    p.add_argument("--thumb", type=int, default=64, help="thumbnail side length in pixels")
    p.add_argument("--images", choices=["failures", "all", "none"], default="failures")
    p.add_argument("--checkpoint", default="", help="recorded in each episode's metadata")
    p.add_argument("--resume", action="store_true", help="skip episodes already in summary.jsonl")
    a = p.parse_args(argv)

    # Run from the Isaac-GR00T root: make its ``gr00t`` package importable even if the sim
    # venv does not have it installed.
    sys.path.insert(0, os.getcwd())
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    ro = _load_rollouts_module()

    from functools import partial

    from gr00t.eval.rollout_policy import (
        MultiStepConfig,
        VideoConfig,
        WrapperConfigs,
        _macro_step_env_steps,
        _RobustAsyncVectorEnv,
        create_eval_env,
        create_gr00t_sim_policy,
    )
    from gr00t.eval._horizon_contract import PolicyHorizonSpec
    from gr00t.eval.sim.env_utils import get_embodiment_tag_from_env_name
    import gymnasium as gym

    tasks = a.tasks or SUITES[a.suite]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    done_keys = set()
    if a.resume and (out / ro.SUMMARY_FILENAME).exists():
        done_keys = {(r["task"], int(r["ordinal"])) for r in ro.read_summary(out)}
    run_meta = {
        "suite": a.suite,
        "base_seed": a.base_seed,
        "episodes_per_task": a.episodes_per_task,
        "max_episode_steps": a.max_episode_steps,
        "n_action_steps": a.n_action_steps,
        "checkpoint": a.checkpoint,
    }
    (out / "run_config.json").write_text(json.dumps({**run_meta, "tasks": tasks}, indent=2), encoding="utf-8")

    policy = create_gr00t_sim_policy("", get_embodiment_tag_from_env_name(tasks[0]), a.policy_client_host, a.policy_client_port)
    contract = PolicyHorizonSpec.from_policy(policy, n_action_steps=a.n_action_steps)
    wrapper_configs = WrapperConfigs(
        multistep=MultiStepConfig(contract=contract, max_episode_steps=a.max_episode_steps, terminate_on_success=True),
        video=VideoConfig(video_dir=None, max_episode_steps=a.max_episode_steps),
    )
    max_decisions = -(-a.max_episode_steps // a.n_action_steps) + 1

    t_all = time.time()
    for task_id, task in enumerate(tasks):
        todo = [o for o in range(a.episodes_per_task) if (task, o) not in done_keys]
        if not todo:
            continue
        n_envs = min(a.n_envs, len(todo))
        fns = [
            partial(create_eval_env, env_name=task, env_idx=i, total_n_envs=n_envs, wrapper_configs=wrapper_configs)
            for i in range(n_envs)
        ]
        env = gym.vector.SyncVectorEnv(fns) if n_envs == 1 else _RobustAsyncVectorEnv(fns, shared_memory=False, context="spawn")
        t_task = time.time()
        n_ok = 0
        try:
            for w0 in range(0, len(todo), n_envs):
                wave = todo[w0 : w0 + n_envs]
                pad = wave + [wave[-1]] * (n_envs - len(wave))  # idle duplicates are not saved
                seeds = [a.base_seed + 1000 * task_id + o for o in pad]
                rec, success, desc = _record_wave(
                    env, policy, n_envs, seeds, max_decisions, a.image_key, a.thumb, _macro_step_env_steps
                )
                for i, ordinal in enumerate(wave):
                    r = rec[i]
                    keep_img = a.images == "all" or (a.images == "failures" and not success[i])
                    ep = ro.RolloutEpisode(
                        task=task,
                        ordinal=ordinal,
                        seed=seeds[i],
                        success=bool(success[i]),
                        env_steps=np.asarray(r["env_steps"], dtype=np.int64),
                        n_env_steps=np.asarray(r["n_env_steps"], dtype=np.int64),
                        state={k: np.stack(v) for k, v in r["state"].items() if v},
                        action={k: np.stack(v) for k, v in r["action"].items() if v},
                        task_description=desc[i],
                        image=np.stack(r["image"]) if (keep_img and r["image"]) else None,
                        meta=run_meta,
                    )
                    ep.save(ro.episode_path(out, task, ordinal))
                    ro.append_summary(out, ep.summary())
                    n_ok += int(success[i])
        finally:
            try:
                env.close()
            except Exception as exc:  # noqa: BLE001
                print(f"env.close() failed: {exc}")
        print(
            f"[{task_id + 1}/{len(tasks)}] {task.split('/')[-1]}: {n_ok}/{len(todo)} success "
            f"({time.time() - t_task:.0f}s)",
            flush=True,
        )
    recs = ro.read_summary(out)
    k = sum(r["success"] for r in recs)
    print(json.dumps({"episodes": len(recs), "success": k, "rate": round(k / max(1, len(recs)), 4),
                      "wall_s": round(time.time() - t_all, 1)}))


if __name__ == "__main__":
    main()
