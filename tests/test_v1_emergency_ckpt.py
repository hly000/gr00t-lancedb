# SPDX-License-Identifier: Apache-2.0
"""Plan item 1.3: compact emergency checkpoints + self-calibrated, deadline-aware cost gate."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from gr00t_lancedb.ckpt_codec import (
    decode_tensor,
    decode_tree,
    encode_optimizer_state,
    encode_tensor,
    estimate_payload_bytes,
    resolve_preset,
)
from gr00t_lancedb.exact_resume import (
    CALIBRATION_FILENAME,
    EMERGENCY_CKPT_FILENAME,
    PREEMPT_EVENTS_FILENAME,
    WriteBandwidthCalibration,
    _make_preempt_callback,
    load_emergency_trainable_checkpoint,
    plan_preempt_save,
    save_emergency_trainable_checkpoint,
)


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.frozen = nn.Linear(64, 64)
        self.head = nn.Linear(64, 512)  # 33k trainable params
        for p in self.frozen.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.head(torch.relu(self.frozen(x)))


def _trained(steps: int = 3):
    model = _Model()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    g = torch.Generator().manual_seed(1)
    for _ in range(steps):
        loss = model(torch.randn(16, 64, generator=g)).square().mean()
        loss.backward()
        opt.step()
        sched.step()
        opt.zero_grad()
    return model, opt, sched


class _Monitor:
    def __init__(self, preempted: bool = False) -> None:
        self.preempted = preempted

    def install(self):
        pass

    def restore(self):
        pass

    def broadcast_preempted(self) -> bool:
        return self.preempted


def test_codecs_error_bounds_and_second_moment_never_shrinks() -> None:
    torch.manual_seed(3)
    w = torch.randn(4096, 3) * 0.05
    m = torch.randn(20000) * 1e-4
    v = (torch.randn(20000) * 1e-3) ** 2
    v[:100] = 0.0
    for codec, x, tol in (("bf16", w, 4e-3), ("int8", m, 5e-3)):
        d = decode_tensor(encode_tensor(x, codec))
        assert d.dtype == x.dtype and d.shape == x.shape
        assert ((d - x).abs().max() / x.abs().max()).item() < tol
    dv = decode_tensor(encode_tensor(v, "sqrt_u8"))
    assert torch.all(dv.sqrt() >= v.sqrt() * (1 - 1e-6)), "ceil quantization must not shrink sqrt(v)"
    assert torch.all(dv[:100] == 0)
    # tiny tensors (step counters) stay exact
    step = torch.tensor(7.0)
    assert decode_tensor(encode_tensor(step, "int8")) is not None
    assert torch.equal(decode_tensor(encode_tensor(step, "int8")), step)


@pytest.mark.parametrize("dtype", ["fp32", "bf16", "int8"])
def test_payload_estimate_matches_file_size_and_roundtrip(tmp_path: Path, dtype: str) -> None:
    model, opt, sched = _trained()
    spec = resolve_preset(dtype)
    est = estimate_payload_bytes(model, opt, spec)
    stats: dict = {}
    state = SimpleNamespace(global_step=3)
    ckpt = save_emergency_trainable_checkpoint(tmp_path, state, model, opt, sched, dtype=dtype, stats=stats)
    size = (ckpt / EMERGENCY_CKPT_FILENAME).stat().st_size
    assert stats["bytes"] >= size and stats["dtype"] == dtype
    assert est <= size <= est * 1.15 + 64_000  # pickle / zip overhead only

    fresh, fresh_opt, fresh_sched = _Model(), None, None
    fresh_opt = torch.optim.AdamW([p for p in fresh.parameters() if p.requires_grad], lr=1e-3)
    fresh_sched = torch.optim.lr_scheduler.LambdaLR(fresh_opt, lambda s: 1.0)
    payload = load_emergency_trainable_checkpoint(ckpt, fresh, fresh_opt, fresh_sched, load_rng=False)
    assert payload["lossless"] is (dtype == "fp32")
    for (_, p_ref), (_, p_new) in zip(model.head.named_parameters(), fresh.head.named_parameters(), strict=True):
        if dtype == "fp32":
            assert torch.equal(p_ref, p_new)
        else:
            torch.testing.assert_close(p_new, p_ref, rtol=1e-2, atol=1e-3)
    ref_state = opt.state_dict()["state"]
    new_state = fresh_opt.state_dict()["state"]
    for pid in ref_state:
        assert torch.equal(ref_state[pid]["step"], new_state[pid]["step"])
        if dtype == "fp32":
            assert torch.equal(ref_state[pid]["exp_avg_sq"], new_state[pid]["exp_avg_sq"])
        else:
            assert torch.all(new_state[pid]["exp_avg_sq"] >= ref_state[pid]["exp_avg_sq"] * (1 - 1e-2) - 1e-12)


def test_size_ratios_match_presets() -> None:
    model, opt, _ = _trained()
    sizes = {d: estimate_payload_bytes(model, opt, resolve_preset(d)) for d in ("fp32", "bf16", "int8")}
    assert sizes["bf16"] / sizes["fp32"] == pytest.approx(0.5, rel=0.05)
    assert sizes["int8"] / sizes["fp32"] == pytest.approx(4 / 12, rel=0.05)
    exact_w = estimate_payload_bytes(model, opt, resolve_preset("int8", exact_weights=True))
    assert exact_w / sizes["fp32"] == pytest.approx(6 / 12, rel=0.05)
    enc = encode_optimizer_state(opt.state_dict(), resolve_preset("int8"))
    dec = decode_tree(enc)
    assert set(dec["state"]) == set(opt.state_dict()["state"])


def test_plan_preempt_save_rules() -> None:
    est = {"fp32": 38.0, "bf16": 19.5, "int8": 13.2}  # ~19 / 9.7 / 6.5 GB at 0.5 GB/s
    common = dict(avg_step_s=0.65, deadline_s=25.0, cost_gate=True)
    d = plan_preempt_save(steps_since_save=120, last_saved_step=100, est_by_dtype=est, dtype="auto", **common)
    assert d["action"] == "save" and d["dtype"] == "bf16"  # most precise preset that fits 25 s
    d = plan_preempt_save(steps_since_save=120, last_saved_step=100, est_by_dtype={**est, "fp32": 20.0},
                          dtype="auto", **common)
    assert d["dtype"] == "fp32"
    d = plan_preempt_save(steps_since_save=10, last_saved_step=100, est_by_dtype=est, dtype="auto", **common)
    assert d["action"] == "skip" and d["reason"] == "cost_gate"  # 6.5 s of work < 19.5 s save
    d = plan_preempt_save(steps_since_save=500, last_saved_step=100, est_by_dtype=est, dtype="fp32", **common)
    assert d["action"] == "skip" and d["reason"] == "over_deadline"  # would be torn by SIGKILL
    d = plan_preempt_save(steps_since_save=5, last_saved_step=0, est_by_dtype=est, dtype="fp32", **common)
    assert d["action"] == "save" and d["reason"] == "no_prior_checkpoint"


def test_calibration_priority_and_persistence(tmp_path: Path) -> None:
    path = tmp_path / CALIBRATION_FILENAME
    cal = WriteBandwidthCalibration(path, prior_bps=400e6, overhead_s=1.0)
    assert cal.estimate_s(800_000_000) == pytest.approx(3.0)
    cal.add(100 << 20, 0.05, "probe")  # optimistic small probe
    cal.add(24_000_000_000, 48.0, "hf_save")  # 0.5 GB/s real checkpoint
    assert cal.bytes_per_s() == pytest.approx(0.5e9)  # real saves beat the probe
    cal.add(9_700_000_000, 25.0, "emergency")
    again = WriteBandwidthCalibration(path, prior_bps=1.0, overhead_s=1.0)
    assert again.bytes_per_s() == pytest.approx(9.7e9 / 25.0)
    assert json.loads(path.read_text())["samples"][-1]["source"] == "emergency"


def test_callback_auto_picks_smaller_preset_and_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GR00T_PREEMPT_PROBE_MB", "0")
    model, opt, sched = _trained()
    fp32_bytes = estimate_payload_bytes(model, opt, resolve_preset("fp32"))
    # Pretend the disk is so slow that fp32 misses a 1 s deadline but int8 fits.
    monkeypatch.setenv("GR00T_PREEMPT_PRIOR_WRITE_MBPS", str(fp32_bytes / 1e6 / 0.8))
    monkeypatch.setenv("GR00T_PREEMPT_SAVE_OVERHEAD_S", "0.5")
    mon = _Monitor()
    cb = _make_preempt_callback(mon, exit_on_preempt=False, deadline_s=1.0)
    args = SimpleNamespace(output_dir=str(tmp_path), should_save=True)
    state = SimpleNamespace(global_step=0, save_to_json=None)
    del state.save_to_json
    control = SimpleNamespace(should_save=False, should_training_stop=False)
    cb.on_train_begin(args, state, control)
    for step in range(1, 4):
        state.global_step = step
        cb.on_step_end(args, state, control, model=model, optimizer=opt, lr_scheduler=sched)
    mon.preempted = True
    state.global_step = 4
    cb.on_step_end(args, state, control, model=model, optimizer=opt, lr_scheduler=sched)
    assert cb.last_decision["action"] == "save"
    assert cb.last_decision["dtype"] in ("bf16", "int8")
    assert cb.last_decision["candidates_est_s"]["fp32"] > 1.0
    assert (tmp_path / "checkpoint-4" / EMERGENCY_CKPT_FILENAME).is_file()
    events = [json.loads(ln) for ln in (tmp_path / PREEMPT_EVENTS_FILENAME).read_text().splitlines()]
    assert events[-1]["event"] == "preempt_decision" and events[-1]["save"]["bytes"] > 0
    assert json.loads((tmp_path / CALIBRATION_FILENAME).read_text())["samples"][-1]["source"] == "emergency"
    assert control.should_training_stop is True and control.should_save is False
