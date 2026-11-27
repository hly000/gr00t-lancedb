# SPDX-License-Identifier: Apache-2.0
"""Offline tests for the v1 experiment scripts (no GPU, no GR00T, no rclone, no cloud)."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"


def _load(name: str) -> ModuleType:
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(f"v1s_{name}", SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 1. build_scale_tier_x.py
# ---------------------------------------------------------------------------


def test_scale_tier_replicates_and_converts(v2_dataset_dir: Path, tmp_path: Path, default_modality_configs) -> None:
    import numpy as np
    import pandas as pd

    from conftest import reference_load_v2_step

    xt = _load("build_scale_tier_x")
    factor = 3
    out_root = tmp_path / "x_tier"
    rc = xt.main(["--src", str(v2_dataset_dir), "--out-root", str(out_root), "--factors", f"1,{factor}",
                  "--lance", "copy,external"])
    assert rc == 0
    src_info = json.loads((v2_dataset_dir / "meta" / "info.json").read_text())
    n_src, f_src = src_info["total_episodes"], src_info["total_frames"]

    v21 = out_root / f"x{factor}" / "v21"
    info = json.loads((v21 / "meta" / "info.json").read_text())
    assert info["total_episodes"] == n_src * factor
    assert info["total_frames"] == f_src * factor
    assert info["total_videos"] == n_src * factor * 2
    assert info["splits"] == {"train": f"0:{n_src * factor}"}
    eps = [json.loads(x) for x in (v21 / "meta" / "episodes.jsonl").read_text().splitlines()]
    assert [e["episode_index"] for e in eps] == list(range(n_src * factor))

    # Parquets: renumbered episode_index, globally contiguous index, identical payload columns.
    all_idx = []
    for e in range(n_src * factor):
        df = pd.read_parquet(v21 / "data" / "chunk-000" / f"episode_{e:06d}.parquet")
        src_df = pd.read_parquet(v2_dataset_dir / "data" / "chunk-000" / f"episode_{e % n_src:06d}.parquet")
        assert (df["episode_index"] == e).all()
        all_idx.extend(df["index"].tolist())
        np.testing.assert_array_equal(np.stack(df["action"].to_numpy()), np.stack(src_df["action"].to_numpy()))
        for cam in ("observation.images.front_cam", "observation.images.wrist_cam"):
            vid = v21 / "videos" / "chunk-000" / cam / f"episode_{e:06d}.mp4"
            src_vid = v2_dataset_dir / "videos" / "chunk-000" / cam / f"episode_{e % n_src:06d}.mp4"
            assert vid.read_bytes() == src_vid.read_bytes()
    assert all_idx == list(range(f_src * factor))

    # Replica 2 of source episode 1 loads exactly like the source episode (GR00T reference loader).
    ep_new = 2 * n_src + 1
    for step in (1, 5):
        a = reference_load_v2_step(v21, ep_new, step, default_modality_configs)
        b = reference_load_v2_step(v2_dataset_dir, 1, step, default_modality_configs)
        for k in b.states:
            np.testing.assert_array_equal(a.states[k], b.states[k])
        for k in b.actions:
            np.testing.assert_array_equal(a.actions[k], b.actions[k])
        for k in b.images:
            for fa, fb in zip(a.images[k], b.images[k]):
                np.testing.assert_array_equal(fa, fb)
        assert a.text == b.text

    # Lance variants: row counts match the replicated frame count.
    import lancedb

    for mode in ("copy", "external"):
        db = lancedb.connect(str(out_root / f"x{factor}" / f"lance_{mode}"))
        assert db.open_table("frames").count_rows() == f_src * factor
        assert db.open_table("videos").count_rows() == n_src * factor * 2

    manifest = json.loads((out_root / "manifest.json").read_text())
    tiers = {t["factor"]: t for t in manifest["tiers"]}
    assert tiers[factor]["frames"] == f_src * factor and tiers[1]["episodes"] == n_src
    v = tiers[factor]["variants"]
    assert v["v21"]["logical_bytes"] > v["v21"]["unique_disk_bytes"]  # hard-linked videos
    assert v["lance_copy"]["logical_bytes"] > v["lance_external"]["logical_bytes"]
    assert v["v21"]["video_placement"].get("hardlink", 0) + v["v21"]["video_placement"].get("symlink", 0) == n_src * factor * 2


def test_scale_tier_writes_lance_to_bucket(v2_dataset_dir: Path, tmp_path: Path, moto_s3_server, monkeypatch) -> None:
    import lancedb

    xt = _load("build_scale_tier_x")
    srv = moto_s3_server
    monkeypatch.setenv("AWS_ENDPOINT", srv["endpoint_url"])  # credentials from the environment only
    monkeypatch.setenv("AWS_ALLOW_HTTP", "true")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    template = f"s3://{srv['bucket']}/x_tier/x{{factor}}/lance_{{mode}}"
    entry = xt.build_tier(v2_dataset_dir, tmp_path / "x", 2, lance_modes=["copy"], lance_uri_template=template)
    var = entry["variants"]["lance_copy"]
    assert var["path"] == f"s3://{srv['bucket']}/x_tier/x2/lance_copy" and var["logical_bytes"] is None
    assert not (tmp_path / "x" / "x2" / "lance_copy").exists()
    db = lancedb.connect(var["path"], storage_options=srv["storage_options"])
    assert db.open_table("frames").count_rows() == entry["frames"]


def test_scale_tier_dry_run_writes_nothing(v2_dataset_dir: Path, tmp_path: Path, capsys) -> None:
    xt = _load("build_scale_tier_x")
    out_root = tmp_path / "plan"
    assert xt.main(["--src", str(v2_dataset_dir), "--out-root", str(out_root), "--factors", "2,200",
                    "--lance", "copy", "--dry-run"]) == 0
    assert not out_root.exists()
    out = capsys.readouterr().out
    plan = json.loads(out[out.index("{"):])
    assert plan["dry_run"] is True
    assert [t["episodes"] for t in plan["tiers"]] == [8, 800]


def test_scale_tier_patches_episode_stats(tmp_path: Path) -> None:
    xt = _load("build_scale_tier_x")
    stats = {"index": {"min": [100], "max": [119], "mean": [109.5], "std": [5.77], "count": [20]},
             "episode_index": {"min": [3], "max": [3], "mean": [3.0], "std": [0.0]}}
    xt._shift_feature_stats(stats, "index", 400, None)
    xt._shift_feature_stats(stats, "episode_index", None, 17)
    assert stats["index"]["min"] == [500] and stats["index"]["max"] == [519] and stats["index"]["std"] == [5.77]
    assert stats["index"]["count"] == [20]
    assert stats["episode_index"]["mean"] == [17] and stats["episode_index"]["std"] == [0.0]
    g = xt._global_index_stats([2, 2], {"index": {"min": [0], "max": [9], "mean": [0], "std": [0]},
                                        "episode_index": {"min": [0], "max": [0], "mean": [0], "std": [0]}})
    assert g["index"]["max"] == [3] and g["index"]["mean"] == [1.5]
    assert g["episode_index"]["max"] == [1] and g["episode_index"]["mean"] == [0.5]


# ---------------------------------------------------------------------------
# 2. run_group_b_rclone.sh
# ---------------------------------------------------------------------------


def _rclone_sh(args: list[str], base: Path, **env: str) -> subprocess.CompletedProcess:
    full_env = {**os.environ, "DRY_RUN": "1", "B_BASE_DIR": str(base), "B_REMOTE": "gcs:bkt/datasets",
                "SUDO": "", "B_UID": "1000", "B_GID": "1001", **env}
    return subprocess.run(["bash", str(SCRIPTS / "run_group_b_rclone.sh"), *args], env=full_env,
                          capture_output=True, text=True, timeout=30)


def test_group_b_rclone_dry_run(tmp_path: Path) -> None:
    base = tmp_path / "groupb"
    r = _rclone_sh(["mount"], base, B_DATASET_SUBDIR="libero_10")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == f"{base}/mnt/gen-1/libero_10"
    for flag in ("rclone mount gcs:bkt/datasets", "--allow-other", "--uid 1000", "--gid 1001",
                 "--vfs-cache-mode full", f"--cache-dir {base}/vfs-cache/gen-1"):
        assert flag in r.stderr, flag
    assert not base.exists()  # dry-run never touches disk

    (base / "state").mkdir(parents=True)
    (base / "state" / "gen").write_text("3\n")
    r = _rclone_sh(["reset"], base)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == f"{base}/mnt/gen-4"  # rotated to a fresh directory name
    assert f"umount {base}/mnt/gen-3" in r.stderr
    assert f"rm -rf {base}/vfs-cache/gen-3" in r.stderr
    assert f"--cache-dir {base}/vfs-cache/gen-4" in r.stderr
    assert (base / "state" / "gen").read_text().strip() == "3"

    r = _rclone_sh(["status"], base)
    assert "generation: 3" in r.stdout and "mounted: no" in r.stdout
    assert _rclone_sh(["bogus"], base).returncode == 2
    r = _rclone_sh(["mount"], base, B_REMOTE="")
    assert r.returncode != 0 and "B_REMOTE" in r.stderr


# ---------------------------------------------------------------------------
# 3. preempt_schedule_runner.py
# ---------------------------------------------------------------------------

FAKE_TRAINER = r'''
import argparse, json, shutil, signal, sys, time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--output-dir", required=True)
p.add_argument("--max-steps", type=int, required=True)
p.add_argument("--save-steps", type=int, default=10)
p.add_argument("--step-s", type=float, default=0.04)
p.add_argument("--startup-s", type=float, default=0.1)
p.add_argument("--resume_from_checkpoint", action="store_true")
a = p.parse_args()
t0 = time.time()
out = Path(a.output_dir)
out.mkdir(parents=True, exist_ok=True)
fh = open(out / "bench_metrics.jsonl", "a")


def w(rec):
    rec.setdefault("t", time.time())
    rec.setdefault("since_launch_s", rec["t"] - t0)
    fh.write(json.dumps(rec) + "\n")
    fh.flush()


stop = {"flag": False}
signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("flag", True))


def save(step):
    d = out / f"checkpoint-{step}"
    if d.exists():
        return
    tmp = out / f".tmp-checkpoint-{step}"
    tmp.mkdir(exist_ok=True)
    (tmp / "model.bin").write_bytes(b"x" * 16)
    (tmp / "trainer_state.json").write_text(json.dumps({"global_step": step}))
    tmp.rename(d)
    w({"event": "save", "global_step": step})


start = 0
if a.resume_from_checkpoint:  # like HF get_last_checkpoint: highest number, no completeness check
    cks = sorted((int(d.name.split("-")[1]), d) for d in out.glob("checkpoint-*"))
    start = json.loads((cks[-1][1] / "trainer_state.json").read_text())["global_step"]
w({"event": "phase", "phase": "imports", "duration_s": 0.01})
time.sleep(a.startup_s)
w({"event": "train_begin", "global_step": start})
prev = time.time()
io = 0
for step in range(start + 1, a.max_steps + 1):
    time.sleep(a.step_s)
    now = time.time()
    io += 3
    w({"event": "step", "global_step": step, "t": now, "step_s": now - prev, "data_wait_s": 0.001,
       "pss_tree_mb": 1000.0 + step, "sys_used_mb": 5000.0,
       "io": {"lance_blob_calls": io, "net_rx_bytes": io * 1000}})
    if step == start + 1:
        w({"event": "startup_summary", "global_step": step, "imports_s": 0.01, "first_step_s": now - prev,
           "first_batch_wait_s": 0.001})
    prev = now
    if step % a.save_steps == 0:
        save(step)
        prev = time.time()
    if stop["flag"]:
        save(step)
        sys.exit(143)
w({"event": "train_end", "global_step": a.max_steps})
sys.exit(0)
'''


@pytest.fixture()
def fake_trainer(tmp_path: Path) -> Path:
    p = tmp_path / "fake_trainer.py"
    p.write_text(FAKE_TRAINER)
    return p


def _events(runner_dir: Path) -> list[dict]:
    return [json.loads(x) for x in (runner_dir / "events.jsonl").read_text().splitlines()]


def test_poisson_schedule_and_loading(tmp_path: Path) -> None:
    pr = _load("preempt_schedule_runner")
    a = pr.gen_poisson(6, 50, seed=1)
    assert a.times_s == pr.gen_poisson(6, 50, seed=1).times_s != pr.gen_poisson(6, 50, seed=2).times_s
    assert 200 < len(a.times_s) < 400  # ~ rate x horizon = 300
    assert abs(sum(a.times_s) / len(a.times_s) - 600) < 120  # mean gap ~ 3600 / 6
    w = pr.gen_poisson(6, 50, seed=1, mode="wall")
    assert w.times_s[-1] < 50 * 3600 and abs(w.times_s[-1] - sum(a.times_s)) < 1.0

    f = tmp_path / "s.json"
    f.write_text(json.dumps([5, 7]))
    assert pr.load_schedule(f).mode == "since_launch"
    f.write_text(json.dumps({"mode": "wall", "times_s": [1, 3], "kill_mode": "kill", "grace_s": 5}))
    s = pr.load_schedule(f)
    assert (s.mode, s.kill_mode, s.grace_s) == ("wall", "kill", 5)
    with pytest.raises(ValueError):
        pr.Schedule(mode="hourly", times_s=[1])
    with pytest.raises(ValueError):
        pr.Schedule(mode="wall", times_s=[3, 1])

    out = tmp_path / "gen.json"
    assert pr.main(["--gen-poisson", "--rate-per-hour", "12", "--horizon-h", "1", "--seed", "3",
                    "--write-schedule", str(out)]) == 0
    assert json.loads(out.read_text())["generator"]["rate_per_hour"] == 12


def test_wall_schedule_collapses_missed_arrivals(tmp_path: Path) -> None:
    pr = _load("preempt_schedule_runner")
    cfg = pr.RunnerConfig(cmd=["true"], output_dir=tmp_path, runner_dir=tmp_path / "r",
                          schedule=pr.Schedule(mode="wall", times_s=[1.0, 2.0, 3.0, 10.0]))
    r = pr.Runner(cfg)
    r.t0 = 1000.0
    deadline, v, deferred, collapsed = r._deadline(1002.5)  # arrivals at 1 and 2 s happened during downtime
    assert (deadline, v, deferred, collapsed) == (1002.5, 2.0, True, 1)
    r.sched_idx += 1
    assert r._deadline(1002.6) == (1003.0, 3.0, False, 0)


def test_checkpoint_quarantine(tmp_path: Path) -> None:
    pr = _load("preempt_schedule_runner")
    for step, complete in ((10, True), (20, True), (30, False)):
        d = tmp_path / f"checkpoint-{step}"
        d.mkdir()
        if complete:
            (d / "trainer_state.json").write_text(json.dumps({"global_step": step}))
    assert pr.latest_complete_checkpoint(tmp_path)[1] == 20
    moved = pr.quarantine_incomplete(tmp_path)
    assert len(moved) == 1 and not (tmp_path / "checkpoint-30").exists()
    assert sorted(p.name for p in tmp_path.glob("checkpoint-*")) == ["checkpoint-10", "checkpoint-20"]


def test_runner_sigterm_resumes_without_redo(tmp_path: Path, fake_trainer: Path) -> None:
    pr = _load("preempt_schedule_runner")
    out, rdir = tmp_path / "run", tmp_path / "runner"
    reset_log = tmp_path / "resets.txt"
    rc = pr.main([
        "--output-dir", str(out), "--runner-dir", str(rdir), "--kill-mode", "term", "--grace-s", "10",
        "--reset-cmd", f"echo $RUNNER_RESET_REASON:$RUNNER_LAUNCH_INDEX >> {reset_log}",
        "--gpu-usd-per-hour", "3.6", "--boot-overhead-s", "100", "--meta", "group=C1", "--meta", "rate_per_hour=6",
        "--poll-s", "0.02", "--schedule", str(_write(tmp_path / "s.json", {"mode": "since_launch", "times_s": [0.5, 0.5]})),
        "--", sys.executable, str(fake_trainer), "--output-dir", str(out), "--max-steps", "30", "--save-steps", "10",
    ])
    assert rc == 0
    s = json.loads((rdir / "preempt_summary.json").read_text())
    assert s["status"] == "completed" and s["max_steps"] == 30 and s["final_step"] == 30
    assert s["launches"] == 3 and s["interruptions"] == 2
    assert s["steps_reexecuted"] == 0 == s["steps_reexecuted_from_records"]  # checkpoint written on SIGTERM
    assert s["step_records_total"] == 30
    for d in s["interruption_details"]:
        assert d["signal"] == "SIGTERM" and not d["escalated_to_sigkill"] and d["ckpt_written_on_signal"]
        assert d["recovery_complete"] and d["recovery_s"] > 0 and d["reset_s"] is not None
    assert [x["returncode"] for x in s["launch_details"]] == [143, 143, 0]
    assert [x["resumed"] for x in s["launch_details"]] == [False, True, True]
    assert s["meta"] == {"group": "C1", "rate_per_hour": 6.0}
    assert s["total_cost_usd"] == pytest.approx((s["total_wall_s"] + 200) / 3600 * 3.6, abs=1e-3)
    assert s["wasted_gpu_s"] == pytest.approx(s["recovery_s_total"] + s["lost_work_s_total"], abs=1e-2)
    assert s["launch_details"][1]["startup"]["imports_s"] == 0.01
    assert reset_log.read_text().split() == ["initial:0", "preempt:1", "preempt:2"]
    names = [e["event"] for e in _events(rdir)]
    assert names.count("launch") == 3 and names.count("preempt") == 2 and names[-1] == "done"
    assert (rdir / "bench_metrics.jsonl").is_file() and (rdir / "launch-2.log").is_file()


def test_runner_sigkill_counts_redone_steps(tmp_path: Path, fake_trainer: Path) -> None:
    pr = _load("preempt_schedule_runner")
    out, rdir = tmp_path / "run", tmp_path / "runner"
    out.mkdir()
    old_line = '{"event": "step", "global_step": 999, "since_launch_s": 1.0, "step_s": 1.0}\n'
    (out / "bench_metrics.jsonl").write_text(old_line)  # left over from an earlier run in the same dir
    rc = pr.main([
        "--output-dir", str(out), "--runner-dir", str(rdir), "--kill-mode", "kill", "--no-reset-before-first",
        "--poll-s", "0.02", "--schedule", str(_write(tmp_path / "s.json", {"mode": "at_step", "times_s": [15]})),
        "--", sys.executable, str(fake_trainer), "--output-dir", str(out), "--max-steps", "25", "--save-steps", "10",
    ])
    assert rc == 0
    s = json.loads((rdir / "preempt_summary.json").read_text())
    assert s["completed"] and s["interruptions"] == 1 and s["final_step"] == 25
    d = s["interruption_details"][0]
    assert d["signal"] == "SIGKILL" and d["resumed_from_step"] == 10 and not d["ckpt_written_on_signal"]
    assert d["last_step_before"] >= 15 and d["steps_lost"] == d["last_step_before"] - 10
    assert s["steps_reexecuted"] == s["steps_reexecuted_from_records"] == s["step_records_total"] - 25
    assert d["lost_work_s"] > 0.04 * (d["steps_lost"] - 1)
    assert s["launch_details"][0]["returncode"] == -9
    assert s["wasted_gpu_s_doc_formula"] is not None
    copied = (rdir / "bench_metrics.jsonl").read_text()
    assert '"global_step": 999' not in copied and copied.count('"event": "step"') == s["step_records_total"]


def test_runner_analyze_is_exact(tmp_path: Path) -> None:
    pr = _load("preempt_schedule_runner")
    cfg = pr.RunnerConfig(cmd=["x"], output_dir=tmp_path, runner_dir=tmp_path / "r", gpu_usd_per_hour=36.0,
                          boot_overhead_s=60.0, max_steps=6)

    def steps(t_launch: float, first_t: float, gsteps: list[int]) -> list[dict]:
        recs = [{"event": "train_begin", "t": first_t - 1, "since_launch_s": first_t - 1 - t_launch}]
        for i, g in enumerate(gsteps):
            t = first_t + i
            recs.append({"event": "step", "global_step": g, "t": t, "since_launch_s": t - t_launch, "step_s": 1.0})
        return recs

    l0 = pr.LaunchRecord(index=0, t_launch=0.0, resume_from_step=0, resumed=False, t_exit=16.0, returncode=-9,
                         preempted=True, t_signal=15.5, signal="SIGKILL", ckpt_step_after_exit=2,
                         records=steps(0.0, 10.0, [1, 2, 3, 4]) + [{"event": "save", "global_step": 2, "t": 11.5}],
                         reset_s=4.0)
    l1 = pr.LaunchRecord(index=1, t_launch=20.0, resume_from_step=2, resumed=True, t_exit=40.0, returncode=0,
                         records=steps(20.0, 30.0, [3, 4, 5, 6]) + [{"event": "train_end", "global_step": 6}])
    s = pr.analyze([l0, l1], cfg, t0=0.0, t_end=40.0)
    d = s["interruption_details"][0]
    assert d["steps_lost"] == 2 and d["lost_work_s"] == 4.0  # killed at 15.5, durable since save at 11.5
    assert d["recovery_s"] == 14.5 and d["relaunch_to_first_step_s"] == 10.0 and d["shutdown_s"] == 0.5
    assert s["steps_reexecuted"] == 2 == s["steps_reexecuted_from_records"]
    assert s["wasted_gpu_s"] == 18.5
    assert s["wasted_gpu_s_doc_formula"] == 14.5 + 2 * 1.0
    assert s["total_cost_usd"] == pytest.approx((40 + 60) / 3600 * 36)


def test_runner_dry_run_prints_commands(tmp_path: Path, capsys) -> None:
    pr = _load("preempt_schedule_runner")
    assert pr.main(["--no-preempt", "--output-dir", str(tmp_path), "--dry-run", "--",
                    "bash", "run.sh", "--max_steps", "2000"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["max_steps"] == 2000
    assert plan["relaunch_with_checkpoint"].endswith("--max_steps 2000 --resume_from_checkpoint")


def _write(path: Path, obj) -> Path:
    path.write_text(json.dumps(obj))
    return path


# ---------------------------------------------------------------------------
# 4. bench_lerobot_groot_baseline.py
# ---------------------------------------------------------------------------

FAKE_LEROBOT_TRAIN = r'''
import sys, time
steps = int([a for a in sys.argv if a.startswith("--steps=")][0].split("=")[1])
print("INFO loading policy ...", flush=True)
time.sleep(0.3)
for s in range(1, steps + 1):
    time.sleep(0.03)
    print(f"INFO 2026-11-26 12:00:00 ot_train.py:894 step:{s} smpl:{s * 8} ep:0 epch:0.00 "
          f"loss:{1.0 / s:.3f} grdn:1.000 lr:1.0e-04 updt_s:0.025 data_s:0.005", flush=True)
'''


def test_lerobot_baseline_dry_run_and_parse(capsys) -> None:
    bl = _load("bench_lerobot_groot_baseline")
    assert bl.main(["run", "--dry-run", "--dataset-root", "/data/ds/libero_10_lance", "--steps", "300",
                    "--batch-size", "32", "--seed", "7", "--output-dir", "/data/runs/x", "--",
                    "--policy.tune_visual=false"]) == 0
    first = capsys.readouterr().out.splitlines()[0]
    for flag in ("lerobot-train", "--policy.type=groot", "--dataset.root=/data/ds/libero_10_lance",
                 "--dataset.repo_id=local/libero_10_lance", "--steps=300", "--batch_size=32", "--seed=7",
                 "--log_freq=1", "--policy.base_model_path=nvidia/GR00T-N1.7-3B", "--policy.tune_visual=false"):
        assert flag in first.split(), flag
    rec = bl.parse_log_line("INFO 2026-11-26 ot_train.py:894 step:12 smpl:384 ep:1 epch:0.01 loss:0.512 "
                            "grdn:1.2 lr:1.0e-04 updt_s:0.311 data_s:0.020")
    assert rec == {"step": 12.0, "loss": 0.512, "updt_s": 0.311, "data_s": 0.02}
    assert bl.parse_log_line("step:1K smpl:32K ep:9 epch:0.3 loss:0.1 updt_s:0.3 data_s:0.0")["step"] == 1000
    assert bl.parse_log_line("Creating dataset step: none") is None

    lines = [(10.0 + 0.5 * i, f"step:{i + 1} smpl:8 ep:0 epch:0 loss:0.5 updt_s:0.4 data_s:0.1") for i in range(30)]
    m = bl.summarize_lerobot_run(0.0, lines, [{"rss_tree_mb": 900.0, "sys_used_mb": 4000.0},
                                              {"rss_tree_mb": 1200.0, "sys_used_mb": 3000.0}])
    assert m["launch_to_first_step_s"] == 10.0 and m["launch_to_first_batch_s"] == pytest.approx(9.6)
    assert m["steady_step_s_wall_median"] == pytest.approx(0.5) and m["steady_step_s_logged_median"] == 0.5
    assert m["peak_rss_tree_mb"] == 1200.0 and m["peak_sys_used_mb"] == 4000.0 and m["step_lines"] == 30


def test_lerobot_baseline_run_with_fake_trainer(tmp_path: Path) -> None:
    bl = _load("bench_lerobot_groot_baseline")
    fake = tmp_path / "fake_lerobot_train.py"
    fake.write_text(FAKE_LEROBOT_TRAIN)
    res_path = tmp_path / "res" / "lerobot.json"
    rc = bl.main(["run", "--dataset-root", str(tmp_path / "ds"), "--steps", "20", "--output-dir", str(tmp_path / "o"),
                  "--lerobot-train", f"{sys.executable} {fake}", "--results", str(res_path), "--mem-interval-s", "0.05"])
    assert rc == 0
    res = json.loads(res_path.read_text())
    m = res["metrics"]
    assert res["returncode"] == 0 and m["step_lines"] == 20 and m["last_step"] == 20
    assert m["launch_to_first_step_s"] > 0.3 and 0.02 < m["steady_step_s_wall_median"] < 0.5
    assert res["config"]["steps"] == 20 and "optimizer/scheduler preset" in res["config"]["not_matched"]
    assert Path(res["log"]).is_file()


def test_lerobot_gap_table(tmp_path: Path) -> None:
    bl = _load("bench_lerobot_groot_baseline")
    md = bl.gap_table_markdown(None)
    for _, label in bl.FEATURES:
        assert f"| {label} |" in md
    assert md.count("\n| ") == len(bl.FEATURES) + 1  # header + one row per feature
    assert "待验证" in md and "lerobot 0.7.0 源码" in md

    good = {"installed": True, "version": "0.7.0", "path": "/x", "n17": True, "multi_dataset_disabled": True,
            "q01q99": True, "lance_backend": True, "lance_frame_filter": False, "lance_version_pin": False,
            "sampler_resume": True, "sigterm_handler": False, "episode_aware_sampler": True}
    assert bl.column_c(good)["mixture"][0] == bl.NO and bl.column_c(good)["n17"][0] == bl.YES
    changed = bl.column_c({**good, "version": "0.9.0", "lance_version_pin": True})
    assert changed["version"][0] == bl.TBD and "0.9.0" in changed["version"][1]
    assert bl.column_c({**good, "sigterm_handler": True})["exact_resume"][0] == bl.TBD

    probe = bl.probe_lerobot()  # whatever is installed here: must not raise, must be JSON-able
    json.dumps(probe)

    metrics = tmp_path / "bench_metrics.jsonl"
    _write_bench(metrics, startup={"imports_s": 5.0, "dataset_build_s": 3.0, "first_step_s": 2.0}, total=20.0,
                 n_steps=30, step_s=0.8, wait=0.02)
    results = {"metrics": {"launch_to_first_batch_s": 40.0, "launch_to_first_step_s": 41.0,
                           "steady_step_s_wall_median": 1.1, "steady_data_s_mean": 0.3, "peak_rss_tree_mb": 9000.0}}
    rpath = _write(tmp_path / "r.json", results)
    out = tmp_path / "gap.md"
    assert bl.main(["table", "--no-probe", "--out", str(out), "--results", str(rpath),
                    "--gr00t-metrics", str(metrics)]) == 0
    text = out.read_text()
    assert "实测对比" in text and "| (B) GR00T + Lance | 18.00 | 20.00 | 0.80 |" in text
    assert "| (C) LeRobot GR00T + Lance | 40.00 | 41.00 | 1.10 | 0.30 | 9000.00 |" in text


# ---------------------------------------------------------------------------
# 5. plot_v1_report.py
# ---------------------------------------------------------------------------


def _write_bench(path: Path, *, startup: dict, total: float, n_steps: int, step_s: float, wait: float,
                 io_per_step: dict | None = None, pss: float = 2000.0, sys_mb: float = 30000.0,
                 legacy: bool = False, launches: int = 1) -> Path:
    """Synthetic bench_metrics.jsonl: ``launches`` launches, each with startup + ``n_steps`` steps."""
    path.parent.mkdir(parents=True, exist_ok=True)
    recs = []
    t = 1_700_000_000.0
    g = 0
    for _ in range(launches):
        t_launch = t
        for ph, key in (("imports", "imports_s"), ("dataset_build", "dataset_build_s")):
            if key in startup:
                recs.append({"event": "phase", "phase": ph, "duration_s": startup[key], "t": t_launch + 1,
                             "since_launch_s": 1.0})
        first_end = t_launch + total
        recs.append({"event": "train_begin", "t": first_end - 3, "since_launch_s": total - 3, "global_step": g})
        io = {k: 0.0 for k in (io_per_step or {})}
        for i in range(n_steps):
            g += 1
            te = first_end + i * step_s
            rec = {"event": "step", "global_step": g, "t": te, "since_launch_s": te - t_launch,
                   "step_s": startup.get("first_step_s", step_s) if i == 0 else step_s, "data_wait_s": wait,
                   "sys_used_mb": sys_mb, "pss_tree_mb": pss if i % 10 == 0 else None}
            if io_per_step:
                io = {k: io[k] + v for k, v in io_per_step.items()}
                rec["io"] = dict(io)
            recs.append(rec)
            if i == 0 and not legacy:
                recs.append({"event": "startup_summary", "t": te, "since_launch_s": total, "global_step": g, **startup})
        t = first_end + n_steps * step_s + 5
    recs.append({"event": "train_end", "global_step": g, "t": t, "since_launch_s": t - t_launch})
    with open(path, "w") as fh:
        for r in recs:
            fh.write(json.dumps({k: v for k, v in r.items() if v is not None}) + "\n")
        fh.write('{"event": "step", "global_')  # truncated last line (killed mid-write)
    return path


def _summary(path: Path, *, group: str, rate: float, seed: int, wall: float, kill_mode: str = "kill") -> None:
    path.mkdir(parents=True, exist_ok=True)
    _write(path / "preempt_summary.json", {
        "status": "completed", "completed": True, "meta": {"group": group, "rate_per_hour": rate, "seed": seed},
        "schedule": {"mode": "since_launch", "times_s": [1.0], "generator": {"rate_per_hour": rate}},
        "kill_mode": kill_mode, "total_wall_s": wall, "billable_s": wall, "gpu_usd_per_hour": 3.6,
        "interruptions": rate, "steps_reexecuted": 10 * rate, "wasted_gpu_s": 30.0 * rate, "recovery_s_median": 25.0})


def test_plot_report_end_to_end(tmp_path: Path) -> None:
    pv = _load("plot_v1_report")
    root = tmp_path / "results" / "v1"
    for g, base in (("A", 30.0), ("C1", 40.0)):
        for tier, factor in (("x1", 1), ("x10", 10)):
            for rep, jitter in ((1, 0.0), (2, 2.0), (3, 4.0)):
                prep = (60.0 * factor if g == "A" else 1.0) + jitter
                _write_bench(root / "e1" / f"group={g}" / f"tier={tier}" / f"rep={rep}" / "bench_metrics.jsonl",
                             startup={"imports_s": 5.0, "model_load_s": 10.0, "dataset_build_s": 3.0,
                                      "trainer_init_s": 1.0, "dataloader_create_s": 0.5,
                                      "first_batch_wait_s": 2.0, "first_step_s": 1.5},
                             total=base + jitter, n_steps=25, step_s=0.8, wait=0.01)
                _write(root / "e1" / f"group={g}" / f"tier={tier}" / f"rep={rep}" / "run_meta.json",
                       {"data_prep_s": prep})
    _write_bench(root / "e2" / "group=C1" / "tier=S" / "bench_metrics.jsonl", startup={"first_step_s": 2.0},
                 total=20.0, n_steps=40, step_s=0.9, wait=0.05, pss=12000.0, io_per_step={
                     "lance_blob_calls": 2.0, "lance_blob_ranges": 6.0, "lance_blob_bytes": 1e6,
                     "gop_cache_hits": 3.0, "gop_cache_misses": 1.0, "net_rx_bytes": 2e6}, launches=2)
    _write_bench(root / "e2" / "group=A" / "tier=S" / "bench_metrics.jsonl", startup={}, total=15.0, n_steps=40,
                 step_s=0.85, wait=0.02, legacy=True)  # old log: phase records only, no startup_summary
    for g, slope in (("A", 100.0), ("B", 60.0), ("C1", 20.0)):
        for rate in (3, 6, 12):
            for seed in (1, 2):
                _summary(root / "e3" / f"group={g}" / f"rate={rate}" / f"seed={seed}", group=g, rate=rate, seed=seed,
                         wall=3600.0 + slope * rate + seed)
    manifest = {"tiers": [{"factor": 1, "variants": {"v21": {"logical_bytes": 0.64e9}, "lance_external": {"logical_bytes": 0.1e9}}},
                          {"factor": 10, "variants": {"v21": {"logical_bytes": 6.4e9}, "lance_external": {"logical_bytes": 1e9}}}]}
    mpath = _write(tmp_path / "manifest.json", manifest)

    out = tmp_path / "report"
    assert pv.main(["--root", str(root), "--out-dir", str(out), "--manifest", str(mpath),
                    "--fig1-exp", "e1", "--fig3-exp", "e2"]) == 0
    res = pv.build_report(root, out, manifest=manifest, exp={"fig1": ["e1"], "fig3": ["e2"]}, png=False)
    cold = {(a["group"], a["tier"]): a for a in res["cold"]}
    assert set(cold) == {("A", "x1"), ("A", "x10"), ("C1", "x1"), ("C1", "x10")}
    a10 = cold[("A", "x10")]
    assert a10["n"] == 3 and a10["dataset_gb"] == pytest.approx(6.4) and cold[("C1", "x10")]["dataset_gb"] == 1.0
    assert a10["data_prep_s"] == 602.0 and a10["cold_start_s"] == pytest.approx(602.0 + 32.0)
    assert (a10["cold_start_min"], a10["cold_start_max"]) == (pytest.approx(630.0), pytest.approx(638.0))
    assert a10["proc_s"] == pytest.approx(5.0 + (32.0 - 23.0))  # imports + unexplained time

    pre = {(p["group"], p["rate_per_hour"]): p for p in res["preempt"]}
    assert len(pre) == 9 and res["fig2_metric"] == "cost_usd"
    assert pre[("B", 6)]["cost_usd"] == pytest.approx((3600 + 360 + 1.5) / 3600 * 3.6)
    assert pre[("C1", 12)]["n"] == 2 and pre[("C1", 12)]["completed"] == 2

    steady = {a["group"]: a for a in res["steady"]}
    assert steady["C1"]["step_s"] == 0.9 and steady["C1"]["pss_tree_mb"] == 12000.0
    assert steady["C1"]["requests_per_1k"] == pytest.approx(2000.0) and steady["C1"]["gop_hit_rate"] == 0.75
    assert steady["A"]["requests_per_1k"] is None and steady["A"]["step_s"] == 0.85

    text = (out / "report.md").read_text()
    assert "## 图 1" in text and "## 图 2" in text and "## 图 3" in text and "| A | x10 | 3 |" in text
    pytest.importorskip("matplotlib")
    for name in ("fig1_cold_start.png", "fig2_preempt_cost.png", "fig3_steady.png"):
        assert (out / name).stat().st_size > 10_000, name


def test_common_helpers_tolerate_old_logs(tmp_path: Path) -> None:
    common = _load("v1_common")
    p = tmp_path / "m.jsonl"
    p.write_text('{"event": "step", "global_step": 1, "since_launch_s": 5, "step_s": 1}\nnot json\n'
                 '{"event": "step", "global_step": 2, "since_launch_s": 6, "step_s": 1}\n'
                 '{"event": "step", "global_step": 2, "since_launch_s": 3, "step_s": 1}\n')
    recs = common.read_jsonl(p)
    assert len(recs) == 3 and len(common.split_segments(recs)) == 2
    sb = common.startup_breakdown(recs[:1])
    assert sb["launch_to_first_step_s"] == 5 and sb["first_step_s"] == 1 and sb["model_load_s"] is None
    assert sb["other_s"] == 4
    assert common.steady_stats([])["step_records"] == 0
    assert common.md_table(["a", "b"], [[None, 1.25]]) == "| a | b |\n|---|---|\n| – | 1.2 |"
