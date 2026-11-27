# SPDX-License-Identifier: Apache-2.0
"""Seed-paired G0 vs G3 comparison."""

from __future__ import annotations

from pathlib import Path

import pytest

from gr00t_lancedb.flywheel.cli import main as flywheel_main
from gr00t_lancedb.flywheel.paired_eval import compare_rollouts, mcnemar_exact_p, to_markdown, wilson_interval
from gr00t_lancedb.flywheel.rollouts import append_summary


def _write(d: Path, outcomes: dict[str, list[bool]], seed0: int = 7) -> None:
    for t, (task, succ) in enumerate(sorted(outcomes.items())):
        for o, s in enumerate(succ):
            append_summary(d, {"task": task, "ordinal": o, "seed": seed0 + 1000 * t + o, "success": s})


def test_mcnemar_and_wilson():
    assert mcnemar_exact_p(0, 0) == 1.0
    assert mcnemar_exact_p(0, 5) == pytest.approx(2 / 32)
    assert mcnemar_exact_p(3, 3) == 1.0
    lo, hi = wilson_interval(8, 10)
    assert 0.49 < lo < 0.5 and 0.94 < hi < 0.95


def test_compare_paired(tmp_path: Path, capsys):
    base = {"sim/easy": [True] * 10, "sim/hard": [False] * 8 + [True] * 2, "sim/mid": [True] * 5 + [False] * 5}
    cand = {"sim/easy": [True] * 9 + [False], "sim/hard": [True] * 6 + [False] * 2 + [True] * 2, "sim/mid": [True] * 6 + [False] * 4}
    _write(tmp_path / "g0", base)
    _write(tmp_path / "g3", cand)
    rep = compare_rollouts(tmp_path / "g0", tmp_path / "g3", worst_k=1)
    o = rep["overall"]
    assert rep["paired_episodes"] == 30
    assert (o["baseline_success"], o["candidate_success"]) == (17, 23)
    assert (o["baseline_only"], o["candidate_only"]) == (1, 7)
    assert o["net_flips"] == 6 and o["diff_pp"] == 20.0
    assert rep["worst1_tasks"] == ["sim/hard"]
    assert rep["worst1"]["diff_pp"] == 60.0
    assert rep["verdict"] == "scale_up"
    assert "McNemar" in to_markdown(rep)

    flywheel_main(["compare", "--baseline", str(tmp_path / "g0"), "--candidate", str(tmp_path / "g3"),
                   "--out-md", str(tmp_path / "r.md")])
    assert (tmp_path / "r.md").read_text().startswith("# G3 vs G0")
    flywheel_main(["success", "--rollouts", str(tmp_path / "g0")])
    assert '"success_rate": 0.5667' in capsys.readouterr().out


def test_compare_rejects_unpaired_seeds(tmp_path: Path):
    _write(tmp_path / "a", {"sim/t": [True, False]}, seed0=7)
    _write(tmp_path / "b", {"sim/t": [True, False]}, seed0=8)
    with pytest.raises(ValueError, match="different seeds"):
        compare_rollouts(tmp_path / "a", tmp_path / "b")


def test_verdict_thresholds(tmp_path: Path):
    _write(tmp_path / "a", {"sim/t": [False] * 100})
    _write(tmp_path / "b", {"sim/t": [True] * 3 + [False] * 97})
    assert compare_rollouts(tmp_path / "a", tmp_path / "b")["verdict"] == "add_seed"
    _write(tmp_path / "c", {"sim/t": [True] + [False] * 99})
    assert compare_rollouts(tmp_path / "a", tmp_path / "c")["verdict"] == "revisit_weights"
