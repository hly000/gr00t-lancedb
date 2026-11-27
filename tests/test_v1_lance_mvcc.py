# SPDX-License-Identifier: Apache-2.0
"""Issue #1 (``where`` without breaking ``fork``) and issue #2 (MVCC-pinned stats).

* ``where`` anchors are computed in an isolated child interpreter: the training main
  process never imports ``lancedb`` / starts Tokio, so ``fork`` workers stay safe.
* ``curate_lance_dataset`` keeps one stats snapshot per ``frames.lance`` version range;
  opening ``lance_version=v`` yields v's anchors *and* v's q01/q99.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from gr00t_lancedb import (
    CURATION_VERSIONS_FILENAME,
    LanceEpisodeLoader,
    convert_v2_to_lance,
    curate_lance_dataset,
)
from gr00t_lancedb.metadata import select_curation_entry

from test_curation import _create_curatable_v2_dataset, _modality_configs

TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(SRC_DIR), str(TESTS_DIR), env.get("PYTHONPATH", "")])
    env["GR00T_LANCE_CACHE_DIR"] = env.get("GR00T_LANCE_CACHE_DIR", "")
    return env


@pytest.fixture()
def curated_twice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GR00T_LANCE_CACHE_DIR", str(tmp_path / "cache"))
    v2_dir, _ = _create_curatable_v2_dataset(tmp_path / "v2_src")
    lance_dir = tmp_path / "lance_ds"
    convert_v2_to_lance(v2_dir, lance_dir, video_mode="copy")
    raw_stats = json.loads((lance_dir / "meta" / "gr00t_stats.json").read_text())
    r1 = curate_lance_dataset(lance_dir, idle_eps=1e-4, drop_episodes=[0, 1], keep_idle=False)
    r2 = curate_lance_dataset(lance_dir, idle_eps=1e-4, keep_idle=True)
    return lance_dir, raw_stats, r1, r2


def test_curation_manifest_keeps_every_version(curated_twice) -> None:
    lance_dir, raw_stats, r1, r2 = curated_twice
    manifest = json.loads((lance_dir / "meta" / CURATION_VERSIONS_FILENAME).read_text())
    kinds = [e["kind"] for e in manifest["entries"]]
    assert kinds == ["original", "curation", "curation"]
    assert r1.frames_table_version < r2.frames_version_from <= r2.frames_table_version
    assert r2.meta_table_version > r1.meta_table_version > r1.meta_table_version_before

    # Every version maps to the right snapshot.
    assert select_curation_entry(manifest, r1.frames_table_version_before)["kind"] == "original"
    assert select_curation_entry(manifest, r1.frames_table_version)["frames_version"] == r1.frames_table_version
    assert select_curation_entry(manifest, r2.frames_table_version)["frames_version"] == r2.frames_table_version

    s1 = json.loads((lance_dir / "meta" / r1.stats_snapshot_file).read_text())
    s2 = json.loads((lance_dir / "meta" / r2.stats_snapshot_file).read_text())
    assert s1 != s2, "different curation params must yield different stats"
    # The original (pre-curation) stats are preserved too.
    original = json.loads((lance_dir / "meta" / manifest["entries"][0]["gr00t_stats"]).read_text())
    assert original == raw_stats

    # meta.lance history is intact: the older meta version still has r1's manifest only.
    import lancedb

    meta = lancedb.connect(str(lance_dir)).open_table("meta")
    meta.checkout(r1.meta_table_version)
    rows = meta.search().where(f"path = '{CURATION_VERSIONS_FILENAME}'").limit(None).to_arrow()
    old_manifest = json.loads(rows.column("data")[0].as_py())
    assert len(old_manifest["entries"]) == 2


def test_lance_version_pins_anchors_and_stats(curated_twice) -> None:
    lance_dir, raw_stats, r1, r2 = curated_twice
    cfg = _modality_configs(action_horizon=4)
    s1 = json.loads((lance_dir / "meta" / r1.stats_snapshot_file).read_text())
    s2 = json.loads((lance_dir / "meta" / r2.stats_snapshot_file).read_text())

    l1 = LanceEpisodeLoader(lance_dir, cfg, where="is_valid = true", lance_version=r1.frames_table_version)
    l2 = LanceEpisodeLoader(lance_dir, cfg, where="is_valid = true")
    l0 = LanceEpisodeLoader(lance_dir, cfg, lance_version=r1.frames_table_version_before)
    try:
        assert l1.frames_version == r1.frames_table_version
        assert l2.frames_version == r2.frames_table_version  # where pins the latest it saw
        assert sum(len(l1.get_valid_steps_for_episode(i)) for i in range(len(l1))) == r1.valid_frames
        assert sum(len(l2.get_valid_steps_for_episode(i)) for i in range(len(l2))) == r2.valid_frames
        assert l1.stats["action"]["q01"] == s1["action"]["q01"]
        assert l1.stats["action"]["q99"] == s1["action"]["q99"]
        assert l2.stats["action"]["q01"] == s2["action"]["q01"]
        assert l0.stats["action"]["q99"] == raw_stats["action"]["q99"]
        assert l1.stats_snapshot["frames_version"] == r1.frames_table_version
        # Rows served by a pinned reader come from the pinned version (same row count).
        steps = l1.load_steps(int(np.flatnonzero([len(l1.get_valid_steps_for_episode(i)) for i in range(len(l1))])[0]),
                              [0], embodiment_tag=None, allow_padding=True)
        assert len(steps) == 1
    finally:
        for loader in (l0, l1, l2):
            loader.close()


def test_where_result_cached_for_explicit_version(curated_twice) -> None:
    lance_dir, _, r1, _ = curated_twice
    cfg = _modality_configs(action_horizon=4)
    a = LanceEpisodeLoader(lance_dir, cfg, where="is_valid = true", lance_version=r1.frames_table_version)
    b = LanceEpisodeLoader(lance_dir, cfg, where="is_valid = true", lance_version=r1.frames_table_version)
    assert b.startup_info.get("where_cache_hit") is True
    assert b.startup_info["isolated_calls"] == 0
    for i in range(len(a)):
        np.testing.assert_array_equal(a.get_valid_steps_for_episode(i), b.get_valid_steps_for_episode(i))


def test_where_keeps_main_process_tokio_free_and_fork_works(curated_twice, tmp_path: Path) -> None:
    """Issue #1 acceptance: ``where`` + multi-worker ``fork`` DataLoader, no Lance in parent."""
    lance_dir, _, _, r2 = curated_twice
    script = tmp_path / "where_fork.py"
    script.write_text(
        textwrap.dedent(
            """\
            import importlib.util, sys
            from pathlib import Path
            import numpy as np
            from torch.utils.data import DataLoader
            from gr00t_lancedb import EmbodimentTag, LanceShardedSingleStepDataset, ModalityConfig
            from gr00t_lancedb.exact_resume import ExactResumeMixin
            from test_exact_resume import _FakeFastMixture, _ident

            def _modality_configs(h):
                return {
                    "video": ModalityConfig(delta_indices=[-1, 0], modality_keys=["front", "wrist"]),
                    "state": ModalityConfig(delta_indices=[0], modality_keys=["arm", "gripper"]),
                    "action": ModalityConfig(delta_indices=list(range(h)), modality_keys=["arm", "gripper"]),
                    "language": ModalityConfig(delta_indices=[0], modality_keys=["task"]),
                }

            class _Mix(ExactResumeMixin, _FakeFastMixture):
                pass

            ds = LanceShardedSingleStepDataset(
                dataset_path=sys.argv[1], embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
                modality_configs=_modality_configs(4), shard_size=8, episode_sampling_rate=0.5,
                seed=3, allow_padding=True, where="is_valid = true",
            )
            assert "lancedb" not in sys.modules, "parent imported lancedb"
            assert "lerobot_lancedb" not in sys.modules, "parent imported lerobot_lancedb"
            ds.assert_unopened()
            spec = importlib.util.spec_from_file_location("launch", sys.argv[2])
            launch = importlib.util.module_from_spec(spec); spec.loader.exec_module(launch)
            assert launch.resolve_lance_mp_context(ds, requested=None) == "fork"
            valid = {(ep, int(s)) for ep in range(len(ds.episode_loader)) for s in ds.get_valid_steps_for_episode(ep)}
            mix = _Mix(datasets=[ds], seed=1, num_shards_per_epoch=4, shuffle_buffer=8, min_fill=2,
                       decode_threads=2, lazy_decode=True, decode_batch_size=4, prefetch_batches=1)
            dl = DataLoader(mix, batch_size=4, num_workers=2, collate_fn=_ident, multiprocessing_context="fork")
            it = iter(dl)
            n = 0
            for _ in range(6):
                for item in next(it):
                    assert item.states["arm"].shape == (1, 7)
                    assert item.actions["arm"].shape == (4, 7)
                    n += 1
            ds.assert_unopened()
            print("WHERE_FORK_OK", n, ds.frames_version, len(valid))
            """
        ),
        encoding="utf-8",
    )
    launch = TESTS_DIR.parent / "scripts" / "launch_finetune_lance.py"
    proc = subprocess.run(
        [sys.executable, str(script), str(lance_dir), str(launch)],
        capture_output=True, text=True, timeout=180, env=_env(), check=False,
    )
    assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr[-4000:]}"
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("WHERE_FORK_OK")][0]
    _, n, version, n_valid = line.split()
    assert int(n) == 24
    assert int(version) == r2.frames_table_version
    assert int(n_valid) > 0
