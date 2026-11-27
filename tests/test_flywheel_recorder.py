# SPDX-License-Identifier: Apache-2.0
"""The LIBERO recorder's wave loop against a fake vector env (no simulator, no GR00T)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "flywheel" / "record_libero_rollouts.py"
spec = importlib.util.spec_from_file_location("_fw_recorder", SCRIPT)
rec_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rec_mod)

N_ACT = 4


class FakeVecEnv:
    """Sub-env i succeeds after ``3 + seed % 3`` decisions unless seed is odd (then times out at 6)."""

    def __init__(self, n):
        self.n = n

    def _obs(self):
        return {
            "state.x": np.stack([np.full((1, 1), float(s)) for s in self.pos])[:, :, :],  # [n, T=1, 1]
            "state.gripper": np.zeros((self.n, 1, 2)),
            "video.image": np.zeros((self.n, 1, 16, 16, 3), np.uint8),
            "annotation.human.action.task_description": ["do it"] * self.n,
        }

    def reset(self, seed):
        self.seeds = list(seed)
        self.t = [0] * self.n
        self.pos = [0.0] * self.n
        return self._obs(), {}

    def step(self, actions):
        term = np.zeros(self.n, bool)
        trunc = np.zeros(self.n, bool)
        succ = []
        for i in range(self.n):
            self.t[i] += 1
            self.pos[i] += 1.0
            ok = self.seeds[i] % 2 == 0 and self.t[i] >= 3 + self.seeds[i] % 3
            succ.append(ok)
            term[i] = ok
            trunc[i] = self.t[i] >= 6
        return self._obs(), np.zeros(self.n), term, trunc, {"success": succ, "n_env_steps": [N_ACT] * self.n}


class FakePolicy:
    def reset(self):
        pass

    def get_action(self, obs):
        n = len(obs["state.x"])
        return {"action.x": np.ones((n, 16, 1)), "action.gripper": np.zeros((n, 16, 1))}, {}


def _steps(infos, i):
    return int(infos["n_env_steps"][i])


def test_wave_records_until_done():
    env = FakeVecEnv(3)
    rec, success, desc = rec_mod._record_wave(env, FakePolicy(), 3, [10, 11, 12], 20, "video.image", 8, _steps)
    assert success.tolist() == [True, False, True]
    assert [len(r["env_steps"]) for r in rec] == [4, 6, 3]  # seed 10: 3+1, 11: timeout 6, 12: 3+0
    assert rec[0]["env_steps"] == [0, 4, 8, 12]
    assert rec[0]["state"]["x"][1].tolist() == [1.0]
    assert rec[0]["action"]["x"][0].shape == (16, 1)
    assert rec[0]["image"][0].shape == (8, 8, 3)
    assert desc == ["do it"] * 3
