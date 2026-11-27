# SPDX-License-Identifier: Apache-2.0
"""Comprehensive unit tests for feat/train-perf-v1 optimizations.

Covers:
1. Anchor-only lazy-decode shuffle buffer (bit-identical sample order & tensors vs eager,
   uninterrupted and multi-worker resume at odd/even batch boundaries, zero buffer pre-decode).
2. LanceEpisodeLoader lazy single-reader sharing, thread safety, is_open / assert_unopened(),
   and repeated fork() DataLoader stress test (20 cycles).
3. Cost-gated preemption saving (skip vs save thresholds, last_saved_step==0 guard, env defaults).
4. Lightweight trainable-only emergency checkpoint save & seamless HF Trainer resume.
5. BenchMetricsCallback data_wait_s isolation from slow _memory_mb() + startup phase events,
   plus make_prewarmed_dataloader integration with HF Trainer.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import textwrap
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from conftest import reference_load_v2_step
from gr00t_lancedb import (
    EmbodimentTag,
    LanceEpisodeLoader,
    LanceShardedSingleStepDataset,
    ModalityConfig,
    convert_v2_to_lance,
)
from gr00t_lancedb.exact_resume import (
    EMERGENCY_CKPT_FILENAME,
    ExactResumeMixin,
    _make_preempt_callback,
    is_emergency_trainable_checkpoint,
    load_emergency_trainable_checkpoint,
    make_exact_resume_trainer_class,
    save_emergency_trainable_checkpoint,
)
from test_exact_resume import _FakeDataset, _FakeFastMixture, _datasets, _ident, _stream


def _load_launch_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "launch_finetune_lance.py"
    spec = importlib.util.spec_from_file_location("launch_finetune_lance_mod", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _ExactFake(ExactResumeMixin, _FakeFastMixture):
    pass


class _CountingFakeDataset(_FakeDataset):
    """Tracks how many items were decoded via load_split / load_anchor_batch."""

    def __init__(self, *args, use_anchor_batch: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.decoded_items = 0
        self._use_anchor_batch = use_anchor_batch
        if not use_anchor_batch:
            # Ensure hasattr(self, "load_anchor_batch") is False for fallback test.
            self.__dict__.pop("load_anchor_batch", None)

    def load_split(self, split):
        ep, steps = split
        self.decoded_items += len(steps)
        return [(ep, int(s)) for s in steps]


class _CountingAnchorDataset(_CountingFakeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, use_anchor_batch=True, **kwargs)
        self.anchor_batch_calls = 0

    def load_anchor_batch(self, ep_step_pairs):
        self.anchor_batch_calls += 1
        self.decoded_items += len(ep_step_pairs)
        return [(int(ep), int(s)) for ep, s in ep_step_pairs]


# ---------------------------------------------------------------------------
# 1. Lazy-decode shuffle buffer parity & zero-predecode resume
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nw", [0, 2, 3])
@pytest.mark.parametrize("prefetch_batches", [0, 2])
def test_lazy_decode_matches_eager_and_stock_uninterrupted(nw: int, prefetch_batches: int) -> None:
    stock = _stream(_FakeFastMixture(_datasets()), 30, 5, nw)
    eager = _stream(_ExactFake(_datasets(), lazy_decode=False, per_sample_seed=True), 30, 5, nw)
    lazy = _stream(
        _ExactFake(
            _datasets(),
            lazy_decode=True,
            decode_batch_size=7,
            prefetch_batches=prefetch_batches,
            per_sample_seed=True,
        ),
        30,
        5,
        nw,
    )
    assert [[it[:2] for it in b] for b in lazy] == [[it[:2] for it in b] for b in stock]
    # Including exact per-sample RNG augmentation draws!
    assert lazy == eager


@pytest.mark.parametrize("nw", [0, 2, 3])
@pytest.mark.parametrize("resume_at", [1, 7, 12, 29])
def test_lazy_decode_exact_resume_odd_and_even_boundaries(nw: int, resume_at: int) -> None:
    full_eager = _stream(_ExactFake(_datasets(), lazy_decode=False), 45, 6, nw)
    full_lazy = _stream(
        _ExactFake(_datasets(), lazy_decode=True, decode_batch_size=11, prefetch_batches=2),
        45,
        6,
        nw,
    )
    assert full_lazy == full_eager

    resumed_ds = _ExactFake(_datasets(), lazy_decode=True, decode_batch_size=11, prefetch_batches=2)
    resumed_ds.set_resume_point(resume_at, 6)
    resumed = _stream(resumed_ds, 45 - resume_at, 6, nw)
    assert resumed == full_eager[resume_at:]


def test_lazy_decode_resume_skips_buffer_decoding() -> None:
    """Resuming with lazy_decode=True must NOT decode the shuffle_buffer contents."""
    ds_eager = _CountingFakeDataset(n_episodes=10, ep_len=20, n_shards=8, seed=1)
    mix_eager = _ExactFake([ds_eager], shuffle_buffer=48, min_fill=16, lazy_decode=False)
    mix_eager.set_resume_point(10, 4)
    out_eager = _stream(mix_eager, 2, 4, nw=0)

    ds_lazy = _CountingAnchorDataset(n_episodes=10, ep_len=20, n_shards=8, seed=1)
    mix_lazy = _ExactFake(
        [ds_lazy],
        shuffle_buffer=48,
        min_fill=16,
        lazy_decode=True,
        decode_batch_size=8,
        prefetch_batches=0,
    )
    mix_lazy.set_resume_point(10, 4)
    out_lazy = _stream(mix_lazy, 2, 4, nw=0)

    assert out_lazy == out_eager
    # Eager decoded the entire 48-item buffer + pending splits; lazy decoded ONLY 1 chunk of 8 items!
    assert ds_lazy.decoded_items == 8
    assert ds_lazy.anchor_batch_calls == 1
    assert ds_eager.decoded_items > 48


def test_lazy_decode_real_lance_tensor_parity_and_resume(
    v2_dataset_dir: Path,
    default_modality_configs: dict[str, ModalityConfig],
    tmp_path: Path,
) -> None:
    """Verify bit-identical state/action/RGB images/text between lazy_decode=True and False on real Lance."""
    lance_dir = tmp_path / "lance_lazy_parity"
    convert_v2_to_lance(v2_dataset_dir, lance_dir, video_mode="copy")

    def _make_mix(lazy_decode: bool) -> _ExactFake:
        ds = LanceShardedSingleStepDataset(
            dataset_path=lance_dir,
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
            modality_configs=default_modality_configs,
            shard_size=12,
            episode_sampling_rate=0.5,
            seed=42,
            allow_padding=True,
        )
        # Main process must remain completely unopened before fork!
        ds.assert_unopened()
        return _ExactFake(
            datasets=[ds],
            seed=42,
            num_shards_per_epoch=4,
            shuffle_buffer=16,
            min_fill=4,
            decode_threads=2,
            per_sample_seed=True,
            lazy_decode=lazy_decode,
            decode_batch_size=8,
            prefetch_batches=2,
        )

    def _step_sig(step) -> tuple:
        return (
            step.text,
            step.states["arm"].tobytes(),
            step.states["gripper"].tobytes(),
            step.actions["arm"].tobytes(),
            step.actions["gripper"].tobytes(),
            step.images["front"][0].tobytes(),
            step.images["front"][1].tobytes(),
            step.images["wrist"][0].tobytes(),
            step.images["wrist"][1].tobytes(),
        )

    eager_batches = _stream(_make_mix(False), 10, 4, nw=2)
    lazy_batches = _stream(_make_mix(True), 10, 4, nw=2)
    eager_sigs = [[_step_sig(s) for s in b] for b in eager_batches]
    lazy_sigs = [[_step_sig(s) for s in b] for b in lazy_batches]
    assert lazy_sigs == eager_sigs

    # Resume at odd batch boundary (3) with 2 forked workers!
    resumed_mix = _make_mix(True)
    resumed_mix.set_resume_point(3, 4)
    resumed_batches = _stream(resumed_mix, 7, 4, nw=2)
    resumed_sigs = [[_step_sig(s) for s in b] for b in resumed_batches]
    assert resumed_sigs == eager_sigs[3:]


# ---------------------------------------------------------------------------
# 2. LanceEpisodeLoader fork safety, thread safety & 20-cycle fork stress test
# ---------------------------------------------------------------------------


def test_episode_loader_unopened_thread_safety_and_20_fork_cycles(
    v2_dataset_dir: Path,
    default_modality_configs: dict[str, ModalityConfig],
    tmp_path: Path,
) -> None:
    lance_dir = tmp_path / "lance_fork_stress"
    convert_v2_to_lance(v2_dataset_dir, lance_dir, video_mode="copy")

    launch_mod = _load_launch_module()

    # 1. Verify unopened state right after construction
    ds = LanceShardedSingleStepDataset(
        dataset_path=lance_dir,
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        modality_configs=default_modality_configs,
        shard_size=12,
        episode_sampling_rate=0.5,
        seed=42,
        allow_padding=True,
    )
    assert ds.is_open is False
    ds.assert_unopened()
    assert launch_mod.resolve_lance_mp_context(ds, requested=None) == "fork"
    assert launch_mod.resolve_lance_mp_context(ds, requested="fork") == "fork"

    # 2. Stress test 20 repeated fork() DataLoader cycles in a clean subprocess where
    # convert_v2_to_lance never ran (guarantees a fresh main process with zero prior Tokio runtime).
    fork_script = tmp_path / "fork_stress_worker.py"
    fork_script.write_text(
        textwrap.dedent(
            """\
            import sys
            import numpy as np
            from torch.utils.data import DataLoader
            from gr00t_lancedb import EmbodimentTag, LanceShardedSingleStepDataset, ModalityConfig
            from gr00t_lancedb.exact_resume import ExactResumeMixin
            from test_exact_resume import _FakeFastMixture, _ident

            class _Exact(_FakeFastMixture):
                pass

            class _ExactMix(ExactResumeMixin, _FakeFastMixture):
                pass

            lance_dir = sys.argv[1]
            cfg = {
                "video": ModalityConfig(delta_indices=[-1, 0], modality_keys=["front", "wrist"]),
                "state": ModalityConfig(delta_indices=[0], modality_keys=["arm", "gripper"]),
                "action": ModalityConfig(delta_indices=[0, 1, 2, 3], modality_keys=["arm", "gripper"]),
                "language": ModalityConfig(delta_indices=[0], modality_keys=["task"]),
            }
            ds = LanceShardedSingleStepDataset(
                dataset_path=lance_dir,
                embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
                modality_configs=cfg,
                shard_size=12,
                episode_sampling_rate=0.5,
                seed=42,
                allow_padding=True,
            )
            ds.assert_unopened()
            mix = _ExactMix(
                datasets=[ds],
                seed=42,
                num_shards_per_epoch=4,
                shuffle_buffer=12,
                min_fill=4,
                decode_threads=2,
                lazy_decode=True,
                decode_batch_size=4,
                prefetch_batches=1,
            )
            ref_first_state = None
            for cycle in range(20):
                ds.assert_unopened()
                dl = DataLoader(
                    mix,
                    batch_size=4,
                    num_workers=2,
                    collate_fn=_ident,
                    multiprocessing_context="fork",
                    persistent_workers=False,
                )
                it = iter(dl)
                batch0 = next(it)
                assert len(batch0) == 4
                state0 = batch0[0].states["arm"]
                if ref_first_state is None:
                    ref_first_state = state0.copy()
                else:
                    np.testing.assert_array_equal(state0, ref_first_state)
                del it, dl
            ds.assert_unopened()
            print("FORK_STRESS_OK:20")
            """
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, str(fork_script), str(lance_dir)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, f"fork stress failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert "FORK_STRESS_OK:20" in proc.stdout

    # 3. Verify two concurrent decode threads share a single _reader0 safely with exact parity
    loader = LanceEpisodeLoader(lance_dir, default_modality_configs)
    assert loader.is_open is False
    loader.assert_unopened()

    def _worker_load(ep_idx: int):
        return loader.load_steps(ep_idx, [0, 3, 7], EmbodimentTag.NEW_EMBODIMENT, allow_padding=True)

    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(_worker_load, ep) for ep in (0, 1, 2, 3, 0, 1, 2, 3)]
        results = [f.result() for f in futs]

    assert loader.is_open is True
    with pytest.raises(RuntimeError, match="cannot safely fork"):
        loader.assert_unopened()
    with pytest.raises(RuntimeError, match="not unopened"):
        launch_mod.resolve_lance_mp_context(loader, requested="fork")
    # Auto resolution safely falls back when already opened!
    assert launch_mod.resolve_lance_mp_context(loader, requested=None) in ("forkserver", "spawn")

    for ep_idx, steps in zip((0, 1, 2, 3), results[:4], strict=True):
        for s_idx, actual in zip((0, 3, 7), steps, strict=True):
            exp = reference_load_v2_step(
                v2_dataset_dir, ep_idx, s_idx, default_modality_configs, allow_padding=True
            )
            np.testing.assert_allclose(actual.states["arm"], exp.states["arm"], rtol=0, atol=1e-6)
            np.testing.assert_array_equal(actual.images["front"][0], exp.images["front"][0])

    loader.close()
    assert loader.is_open is False


# ---------------------------------------------------------------------------
# 3. Cost-gated preemption callback tests
# ---------------------------------------------------------------------------


class _FakeMonitor:
    def __init__(self, preempted: bool = False) -> None:
        self.preempted = preempted
        self.installed = False
        self.restored = False

    def install(self):
        self.installed = True
        return self

    def restore(self):
        self.restored = True

    def broadcast_preempted(self) -> bool:
        return self.preempted


def test_preempt_cost_gate_skips_when_wasted_le_est_save(tmp_path: Path) -> None:
    mon = _FakeMonitor(preempted=False)
    cb = _make_preempt_callback(
        mon, cost_gate=True, light_ckpt=False, est_save_s=10.0, exit_on_preempt=False
    )
    args = SimpleNamespace(output_dir=str(tmp_path), should_save=True)
    state = SimpleNamespace(global_step=100)
    control = SimpleNamespace(should_save=False, should_training_stop=False)

    cb.on_train_begin(args, state, control)
    assert cb.last_saved_step == 100

    # Simulate 5 fast steps (~0.01s each) after last_saved_step=100 -> wasted_s ~ 0.05s <= 10.0s
    for step in range(101, 106):
        time.sleep(0.01)
        state.global_step = step
        cb.on_step_end(args, state, control)

    # Preempt on step 106: wasted_s << est_save_s -> MUST skip save!
    mon.preempted = True
    time.sleep(0.01)
    state.global_step = 106
    cb.on_step_end(args, state, control)
    assert cb.skipped_save is True
    assert control.should_save is False
    assert control.should_training_stop is True


def test_preempt_cost_gate_saves_when_wasted_gt_est_save_or_no_prior_save(tmp_path: Path) -> None:
    # Case A: wasted_s > est_save_s triggers save
    mon = _FakeMonitor(preempted=False)
    cb = _make_preempt_callback(
        mon, cost_gate=True, light_ckpt=False, est_save_s=0.02, exit_on_preempt=False
    )
    args = SimpleNamespace(output_dir=str(tmp_path), should_save=True)
    state = SimpleNamespace(global_step=50)
    control = SimpleNamespace(should_save=False, should_training_stop=False)

    cb.on_train_begin(args, state, control)
    for step in range(51, 56):
        time.sleep(0.015)
        state.global_step = step
        cb.on_step_end(args, state, control)

    mon.preempted = True
    time.sleep(0.015)
    state.global_step = 56
    cb.on_step_end(args, state, control)
    assert cb.skipped_save is False
    assert control.should_save is True
    assert control.should_training_stop is True

    # Case B: last_saved_step == 0 NEVER skips even if wasted_s <= est_save_s
    mon0 = _FakeMonitor(preempted=True)
    cb0 = _make_preempt_callback(
        mon0, cost_gate=True, light_ckpt=False, est_save_s=999.0, exit_on_preempt=False
    )
    state0 = SimpleNamespace(global_step=0)
    ctrl0 = SimpleNamespace(should_save=False, should_training_stop=False)
    cb0.on_train_begin(args, state0, ctrl0)
    state0.global_step = 3
    cb0.on_step_end(args, state0, ctrl0)
    assert cb0.skipped_save is False
    assert ctrl0.should_save is True
    assert ctrl0.should_training_stop is True


def test_preempt_env_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GR00T_PREEMPT_COST_GATE", raising=False)
    monkeypatch.delenv("GR00T_PREEMPT_LIGHT_CKPT", raising=False)
    monkeypatch.delenv("GR00T_PREEMPT_EST_SAVE_S", raising=False)

    cb_light = _make_preempt_callback(_FakeMonitor(), exit_on_preempt=False)
    assert cb_light.cost_gate is True
    assert cb_light.light_ckpt is True
    assert cb_light.est_save_s == 10.0

    monkeypatch.setenv("GR00T_PREEMPT_LIGHT_CKPT", "0")
    cb_full = _make_preempt_callback(_FakeMonitor(), exit_on_preempt=False)
    assert cb_full.light_ckpt is False
    assert cb_full.est_save_s == 55.0


# ---------------------------------------------------------------------------
# 4. Lightweight trainable-only emergency checkpoint + real HF Trainer resume
# ---------------------------------------------------------------------------


class _FrozenPlusTrainableModel(nn.Module):
    """Toy VLA-like model: frozen 'backbone' + trainable 'action_head'."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.backbone = nn.Linear(8, 8)
        self.action_head = nn.Linear(8, 4)
        with torch.no_grad():
            self.backbone.weight.copy_(torch.randn(8, 8, generator=g))
            self.backbone.bias.copy_(torch.randn(8, generator=g))
            self.action_head.weight.copy_(torch.randn(4, 8, generator=g))
            self.action_head.bias.copy_(torch.randn(4, generator=g))
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor, labels: torch.Tensor):
        pred = self.action_head(torch.relu(self.backbone(x)))
        loss = nn.functional.mse_loss(pred, labels)
        return {"loss": loss, "logits": pred}


class _DetTensorDataset(torch.utils.data.IterableDataset):
    """Deterministic iterable dataset supporting ExactResumeMixin's set_resume_point API."""

    def __init__(self, total: int = 200, seed: int = 7) -> None:
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(total, 8, generator=g)
        self.y = torch.randn(total, 4, generator=g)
        self.resume_batches = 0
        self.loader_batch_size = 4

    def set_resume_point(self, batches_consumed: int, batch_size: int) -> None:
        self.resume_batches = int(batches_consumed)
        self.loader_batch_size = int(batch_size)

    def __iter__(self):
        start = self.resume_batches * self.loader_batch_size
        for i in range(start, len(self.x)):
            yield {"x": self.x[i], "labels": self.y[i]}


def test_emergency_trainable_checkpoint_save_and_hf_trainer_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end HF Trainer test: freeze backbone, preempt at step 5, resume to step 10."""
    from transformers import Trainer, TrainerCallback, TrainingArguments
    from transformers.trainer_utils import get_last_checkpoint

    # Direct helper check: frozen weights are excluded from emergency_trainable.pt!
    direct_model = _FrozenPlusTrainableModel(seed=11)
    direct_opt = torch.optim.AdamW([p for p in direct_model.parameters() if p.requires_grad], lr=1e-2)
    direct_sched = torch.optim.lr_scheduler.LinearLR(direct_opt, start_factor=1.0, end_factor=0.1, total_iters=10)
    out = direct_model(torch.ones(2, 8), torch.zeros(2, 4))
    out["loss"].backward()
    direct_opt.step()
    direct_sched.step()
    ckpt_direct = save_emergency_trainable_checkpoint(
        tmp_path / "direct", SimpleNamespace(global_step=1), direct_model, direct_opt, direct_sched
    )
    assert is_emergency_trainable_checkpoint(ckpt_direct)
    saved = torch.load(ckpt_direct / EMERGENCY_CKPT_FILENAME, map_location="cpu", weights_only=False)
    assert set(saved["trainable_state_dict"].keys()) == {"action_head.weight", "action_head.bias"}
    assert "backbone.weight" not in saved["trainable_state_dict"]

    restored_model = _FrozenPlusTrainableModel(seed=11)
    with torch.no_grad():
        restored_model.action_head.weight.zero_()
        restored_model.action_head.bias.zero_()
    restored_opt = torch.optim.AdamW([p for p in restored_model.parameters() if p.requires_grad], lr=1e-2)
    restored_sched = torch.optim.lr_scheduler.LinearLR(
        restored_opt, start_factor=1.0, end_factor=0.1, total_iters=10
    )
    load_emergency_trainable_checkpoint(ckpt_direct, restored_model, restored_opt, restored_sched)
    torch.testing.assert_close(restored_model.action_head.weight, direct_model.action_head.weight)
    torch.testing.assert_close(restored_model.action_head.bias, direct_model.action_head.bias)
    assert restored_sched.last_epoch == direct_sched.last_epoch

    # Full HF Trainer integration test!
    ExactTrainer = make_exact_resume_trainer_class(Trainer)
    monkeypatch.setenv("GR00T_PREEMPT_COST_GATE", "0")
    monkeypatch.setenv("GR00T_PREEMPT_LIGHT_CKPT", "1")

    def _collate(items):
        return {
            "x": torch.stack([it["x"] for it in items]),
            "labels": torch.stack([it["labels"] for it in items]),
        }

    def _args(out_dir: Path) -> TrainingArguments:
        return TrainingArguments(
            output_dir=str(out_dir),
            max_steps=10,
            per_device_train_batch_size=4,
            learning_rate=1e-2,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            use_cpu=True,
            dataloader_num_workers=0,
            seed=42,
            data_seed=42,
        )

    # 1. Uninterrupted 10-step reference run
    ref_model = _FrozenPlusTrainableModel(seed=99)
    ref_trainer = ExactTrainer(
        model=ref_model,
        args=_args(tmp_path / "ref"),
        train_dataset=_DetTensorDataset(),
        data_collator=_collate,
    )
    ref_trainer.train()

    # 2. Preempted run: trigger preemption at step 5 -> writes emergency_trainable.pt!
    pre_dir = tmp_path / "pre"
    pre_model = _FrozenPlusTrainableModel(seed=99)
    backbone_before = pre_model.backbone.weight.detach().clone()

    pre_trainer = ExactTrainer(
        model=pre_model,
        args=_args(pre_dir),
        train_dataset=_DetTensorDataset(),
        data_collator=_collate,
    )
    # Replace the default preempt callback with a controlled one (exit_on_preempt=False for in-process test)
    pre_mon = _FakeMonitor(preempted=False)
    pre_trainer.callback_handler.callbacks = [
        cb for cb in pre_trainer.callback_handler.callbacks if cb.__class__.__name__ != "PreemptCallback"
    ]
    preempt_cb = _make_preempt_callback(
        pre_mon, cost_gate=False, light_ckpt=True, trainer_ref=pre_trainer, exit_on_preempt=False
    )

    class _TriggerAtStep5(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 5:
                pre_mon.preempted = True

    # Ensure trigger fires BEFORE preempt_cb checks monitor!
    pre_trainer.callback_handler.callbacks.insert(0, _TriggerAtStep5())
    pre_trainer.add_callback(preempt_cb)
    pre_trainer.train()

    assert pre_trainer.state.global_step == 5
    last_ckpt = get_last_checkpoint(str(pre_dir))
    assert last_ckpt is not None and Path(last_ckpt).name == "checkpoint-5"
    assert is_emergency_trainable_checkpoint(last_ckpt)
    assert not (Path(last_ckpt) / "model.safetensors").exists()
    assert not (Path(last_ckpt) / "pytorch_model.bin").exists()

    # 3. Resume from checkpoint-5 on a fresh model instance (frozen backbone matches base, trainable fresh)
    res_model = _FrozenPlusTrainableModel(seed=99)
    with torch.no_grad():
        res_model.action_head.weight.fill_(999.0)  # must be overwritten by emergency_trainable.pt!
    res_trainer = ExactTrainer(
        model=res_model,
        args=_args(pre_dir),
        train_dataset=_DetTensorDataset(),
        data_collator=_collate,
    )
    res_trainer.train(resume_from_checkpoint=True)

    assert res_trainer.state.global_step == 10
    # Frozen weights stayed untouched, and resumed trainable weights match uninterrupted 10-step ref!
    torch.testing.assert_close(res_model.backbone.weight, backbone_before)
    torch.testing.assert_close(res_model.action_head.weight, ref_model.action_head.weight, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(res_model.action_head.bias, ref_model.action_head.bias, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# 5. data_wait_s measurement accuracy + startup phase events + prewarmed DL
# ---------------------------------------------------------------------------


def test_bench_callback_data_wait_not_polluted_by_slow_memory_mb(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Slow _memory_mb() in on_step_end must NOT leak into the next step's data_wait_s."""
    launch_mod = _load_launch_module()

    def _slow_memory_mb(full: bool) -> dict[str, float]:
        if full:
            time.sleep(0.12)
        return {"sys_used_mb": 1000.0, "main_rss_mb": 200.0}

    monkeypatch.setattr(launch_mod, "_memory_mb", _slow_memory_mb)

    metrics_path = tmp_path / "bench_metrics.jsonl"
    launch_t0 = time.time() - 1.0
    loss_box: dict[str, Any] = {}
    phase_box: dict[str, Any] = {"dataset_build_s": 0.42, "dataloader_create_s": 0.15}
    cb = launch_mod.make_bench_callback(
        metrics_path, launch_t0, loss_box, phase_box, start_gpu_sampler=False
    )

    args = SimpleNamespace()
    state = SimpleNamespace(global_step=0, max_steps=4)
    control = SimpleNamespace(should_save=False)

    cb.on_train_begin(args, state, control)
    for step in range(1, 5):
        # Simulate tiny 5ms dataloader wait, then 15ms GPU compute.
        time.sleep(0.005)
        cb.on_step_begin(args, state, control)
        time.sleep(0.015)
        state.global_step = step
        loss_box["loss"] = 1.0 / step
        cb.on_step_end(args, state, control)
    cb.on_train_end(args, state, control)

    recs = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    steps = [r for r in recs if r["event"] == "step"]
    assert len(steps) == 4

    # Steps 1 and 2 triggered full _memory_mb (120ms sleep each!).
    # Neither step 2's nor step 3's data_wait_s may include that 120ms!
    for s in steps:
        assert s["data_wait_s"] < 0.05, f"step {s['global_step']} data_wait_s polluted: {s['data_wait_s']}"
        assert s["step_s"] < 0.08, f"step {s['global_step']} step_s polluted: {s['step_s']}"

    # Verify startup phase records & summary were emitted cleanly
    phases = [r for r in recs if r["event"] == "phase"]
    assert any(p["phase"] == "first_batch" for p in phases)
    summaries = [r for r in recs if r["event"] == "startup_summary"]
    assert len(summaries) == 1
    assert summaries[0]["dataset_build_s"] == pytest.approx(0.42)
    assert summaries[0]["dataloader_create_s"] == pytest.approx(0.15)
    assert summaries[0]["first_batch_wait_s"] < 0.05
    assert summaries[0]["first_step_s"] < 0.08


def test_prewarmed_dataloader_yields_exact_stream_in_hf_trainer(tmp_path: Path) -> None:
    """make_prewarmed_dataloader yields batch 0..N identically with zero dropped/duplicated batches."""
    from transformers import Trainer, TrainingArguments

    launch_mod = _load_launch_module()

    def _collate(items):
        return {
            "x": torch.stack([it["x"] for it in items]),
            "labels": torch.stack([it["labels"] for it in items]),
        }

    ds_ref = _DetTensorDataset(total=40, seed=5)
    ref_batches = list(DataLoader(ds_ref, batch_size=4, collate_fn=_collate))

    ds_pw = _DetTensorDataset(total=40, seed=5)
    pw_dl = launch_mod.make_prewarmed_dataloader(DataLoader(ds_pw, batch_size=4, collate_fn=_collate))
    pw_batches = list(pw_dl)
    assert len(pw_batches) == len(ref_batches)
    for b_pw, b_ref in zip(pw_batches, ref_batches, strict=True):
        torch.testing.assert_close(b_pw["x"], b_ref["x"])
        torch.testing.assert_close(b_pw["labels"], b_ref["labels"])

    # Also verify HF Trainer accepts make_prewarmed_dataloader cleanly through accelerator.prepare()
    class _PrewarmTrainer(Trainer):
        def get_train_dataloader(self):
            dl = DataLoader(self.train_dataset, batch_size=self._train_batch_size, collate_fn=_collate)
            return launch_mod.make_prewarmed_dataloader(dl)

    model = _FrozenPlusTrainableModel(seed=5)
    trainer = _PrewarmTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(tmp_path / "pw_run"),
            max_steps=4,
            per_device_train_batch_size=4,
            save_strategy="no",
            report_to=[],
            use_cpu=True,
            dataloader_num_workers=0,
        ),
        train_dataset=_DetTensorDataset(total=40, seed=5),
    )
    trainer.train()
    assert trainer.state.global_step == 4
