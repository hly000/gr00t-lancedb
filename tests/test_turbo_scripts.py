# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the GPU-only scripts: everything that does not need the real model."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gr00t_lancedb.vl_cache import build_action_head_optimizer  # noqa: E402
from gr00t_lancedb.vl_cache.codec import decode_features_numpy, encode_features  # noqa: E402

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pk = _load("profile_kernels")
bench = _load("bench_lowmem_adamw")


def test_categorize():
    assert pk.categorize("aten::linear") == "gemm"
    assert pk.categorize("aten::addmm") == "gemm"
    assert pk.categorize("AddmmBackward0") == "gemm"
    assert pk.categorize("autograd::engine::evaluate_function: MmBackward0") == "gemm"
    assert pk.categorize("aten::native_layer_norm") == "norm"
    assert pk.categorize("aten::scaled_dot_product_attention") == "attention"
    assert pk.categorize("aten::_fused_adamw_") == "optimizer"
    assert pk.categorize("Optimizer.step#AdamW.step") == "optimizer"
    assert pk.categorize("aten::to") == "copy_cast"
    assert pk.categorize("aten::topk") == "other"  # no substring matching
    assert pk.categorize("aten::mul") == "elementwise"


def test_run_profile_on_cpu(tmp_path: Path):
    model = torch.nn.Sequential(torch.nn.Linear(64, 256), torch.nn.GELU(), torch.nn.Linear(256, 8))
    opt = torch.optim.AdamW(model.parameters())
    x = torch.randn(32, 64)

    def step():
        opt.zero_grad()
        model(x).pow(2).mean().backward()
        opt.step()

    rep = pk.run_profile(step, 3, tmp_path / "trace.json", on_gpu=False)
    assert rep["profiled_steps"] == 3 and rep["top_ops"]
    assert "gemm" in rep["by_category_pct"]
    assert abs(sum(rep["by_category_pct"].values()) - 100) < 1.0
    assert all(r["calls_per_step"] >= 0 for r in rep["top_ops"])


def test_int8_per_token_matches_cache_codec():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((37, 64)) * rng.lognormal(0, 1.5, (37, 1))).astype(np.float32)
    x[:, 5] *= 40  # an outlier channel
    scale, payload = encode_features(x, codec="per_token_int8")
    ref = decode_features_numpy(scale, payload, 37, 64, codec="per_token_int8")
    got = pk.int8_per_token(torch.from_numpy(x)).numpy()
    assert np.array_equal(got, ref)


class _FakeHead:
    """Loss depends on the features, the noise drawn after seeding, and nothing else."""

    def __call__(self, bb, act):
        feats = bb["backbone_features"].float()
        noise = torch.randn(act["action"].shape)
        pred = feats.mean(dim=(1, 2), keepdim=False)[:, None, None] + noise
        return {"action_loss": (pred - act["action"]) ** 2, "action_mask": torch.ones_like(act["action"])}


def test_feature_stats_and_memory_ledger():
    from contextlib import nullcontext

    from transformers import BatchFeature

    g = torch.Generator().manual_seed(0)
    bbs, acts = [], []
    for _ in range(2):
        f = torch.randn(4, 12, 32, generator=g)
        f[..., 3] *= 25
        m = torch.ones(4, 12, dtype=torch.bool)
        m[:, :3] = False
        bbs.append(BatchFeature(data={"backbone_features": f, "backbone_attention_mask": m}))
        acts.append({"action": torch.randn(4, 16, 7, generator=g)})
    model = types.SimpleNamespace(action_head=_FakeHead())
    st = pk.feature_stats(model, bbs, acts, nullcontext)
    assert st["tokens"] == 2 * 4 * 9 and st["hidden_dim"] == 32
    assert st["top_outlier_channels"][0] == 3
    assert 0 < st["int8_rel_l2_error"] < 0.05
    assert st["int8_channel_rel_error_p50_p90_p99_max"][0] > st["int8_rel_l2_error"]  # small channels pay
    assert len(st["head_loss_exact_features"]) == 2
    assert 0 <= st["head_loss_per_sample_rel_diff_mean_max"][0] < 0.05  # same noise for both variants

    backbone = torch.nn.Linear(10, 10).requires_grad_(False)
    head = torch.nn.Linear(10, 20)
    opt = torch.optim.AdamW(head.parameters())
    head(torch.randn(2, 10)).sum().backward()
    opt.step()
    led = pk.memory_ledger(types.SimpleNamespace(backbone=backbone, action_head=head), opt)
    n_head = 10 * 20 + 20
    assert led["static_total_gb"] == pytest.approx(((110 + n_head) * 4 + n_head * 4 + n_head * 8) / 1024**3, abs=1e-3)


def test_synthetic_shapes_hit_requested_size():
    shapes = bench.synthetic_shapes(int(1.62e9))
    total = sum(int(np.prod(s)) for s in shapes)
    assert 0.95 * 1.62e9 < total <= 1.62e9
    assert len(shapes) > 300  # per-tensor launch overhead is part of what the bench measures


def test_optimizer_builder_kinds():
    lin = torch.nn.Linear(8, 8)
    opt, name = build_action_head_optimizer(lin, kind="lowmem", lr=1e-3)
    assert name == "LowMemAdamW(torch)" and type(opt).__name__ == "LowMemAdamW"
    _, name = build_action_head_optimizer(lin, kind="adamw")
    assert name.startswith("torch.AdamW")
    with pytest.raises(ValueError):
        build_action_head_optimizer(lin, kind="nope")
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError):  # an explicit request must not silently fall back
            build_action_head_optimizer(lin, kind="bnb8bit")


# ---------------------------------------------------------------------------
# Phase-1 hardening: fused-launch bench gate, seeded eval, paired statistics, run-script gates
# ---------------------------------------------------------------------------

import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, str(SCRIPTS))  # eval/compare scripts import their sibling stats module
stats = _load("libero_eval_stats")
ev = _load("eval_libero10_checkpoint")
cmp_cli = _load("compare_libero_evals")
checks = _load("turbo_run_checks")
RESULTS = SCRIPTS.parent / "results"


def test_bench_fused_gate():
    ok = {"lowmem_bf16": {"step_ms": 19.0, "peak_alloc_gb": 12.2, "fused": "active"}}
    assert bench.fused_gate(ok, 25.0, 12.5) == []
    slow = {"lowmem_bf16": {"step_ms": 56.3, "peak_alloc_gb": 12.2, "fused": "fallback: RuntimeError: x"}}
    assert len(bench.fused_gate(slow, 25.0, 12.5)) == 2
    assert bench.fused_gate({"lowmem_bf16": {"error": "boom"}}, 25.0, 12.5)
    assert bench.fused_gate({}, 25.0, 12.5)


def test_rollout_output_parsing_never_guesses():
    one = "noise\nresults:  ('libero_sim/KITCHEN', [True], {'episode_lengths': [120], 'valid': [True]})\nsuccess rate:  1.0\n"
    assert ev.parse_episode_outcomes(one) == [True] and ev.single_episode_outcome(one) is True
    np2 = "results:  ('libero_sim/X', [np.False_], {})"
    assert ev.single_episode_outcome(np2) is False
    assert ev.single_episode_outcome("results: ('x', [np.bool_(True)], {})") is True
    # The old glitch: a vectorised rollout asked for 5 episodes reports 6. Not one episode -> no outcome.
    six = "results: ('x', [True, True, False, True, True, False], {})\nsuccess rate:  0.6666666666666666"
    assert ev.parse_episode_outcomes(six) == [True, True, False, True, True, False]
    assert ev.single_episode_outcome(six) is None
    # Rollout scripts that only print a rate: usable only when it is exactly 0 or 1.
    assert ev.single_episode_outcome("success rate:  0.0") is False
    assert ev.single_episode_outcome("success rate:  1.0") is True
    assert ev.single_episode_outcome("success rate:  0.5") is None
    assert ev.single_episode_outcome("Traceback (most recent call last): ...") is None


def test_eval_runs_exact_seeded_episode_list_and_retries():
    calls: list[tuple[str, int]] = []

    def runner(task, seed):
        calls.append((task, seed))
        if task == "t1" and (seed - 500) % 1000 == 3:
            # first attempt of this episode yields no outcome, the retry succeeds
            if calls.count((task, seed)) == 1:
                return {"success": None, "returncode": 1, "tail": "boom"}
        if task == "t2" and (seed - 500) % 1000 == 0:
            return {"success": None, "returncode": -9, "tail": "timeout"}  # never recovers
        return {"success": seed % 2 == 0, "returncode": 0}

    per_task = ev.evaluate_tasks(["t0", "t1", "t2"], 20, 500, runner, parallel=4, retries=1, log=lambda s: None)
    assert [len(t["episodes"]) for t in per_task] == [20, 20, 20]  # exactly n, never n+1
    for t_idx, t in enumerate(per_task):
        assert [e["seed"] for e in t["episodes"]] == [500 + 1000 * t_idx + i for i in range(20)]
        assert [e["episode"] for e in t["episodes"]] == list(range(20))
    assert per_task[0]["successes"] == 10 and per_task[0]["success_rate"] == 0.5
    retried = per_task[1]["episodes"][3]
    assert retried["attempts"] == 2 and retried["success"] is False and "tail" not in retried
    dead = per_task[2]["episodes"][0]
    assert dead["success"] is None and dead["attempts"] == 2 and dead["tail"] == "timeout"
    assert per_task[2]["valid_episodes"] == 19

    summary = ev.build_summary(per_task, seed_base=500, seeded=True)
    assert summary["total_episodes"] == 60 and summary["valid_episodes"] == 59 and not summary["complete"]
    assert summary["mean_success_rate"] == round(summary["successes"] / 59, 4)
    assert summary["ci95"][0] < summary["mean_success_rate"] < summary["ci95"][1]
    with pytest.raises(ValueError):
        ev.episode_seed(0, 0, 1000)


def test_eval_client_command_is_one_seeded_simulator():
    args = types.SimpleNamespace(port=5560, max_episode_steps=720, n_action_steps=8)
    cmd = ev.build_client_cmd("/venv/python", "libero_sim/T", 4242, args)
    assert cmd[cmd.index("--n-envs") + 1] == "1" and cmd[cmd.index("--n-episodes") + 1] == "1"
    assert cmd[cmd.index("--seed") + 1] == "4242"
    assert "--seed" not in ev.build_client_cmd("/venv/python", "libero_sim/T", None, args)


def test_interval_and_mcnemar_reference_values():
    lo, hi = stats.wilson_ci(170, 200)
    assert lo == pytest.approx(0.7939, abs=2e-4) and hi == pytest.approx(0.8929, abs=2e-4)
    assert stats.wilson_ci(0, 0) == (0.0, 1.0)
    assert stats.mcnemar_exact_p(0, 0) == 1.0
    assert stats.mcnemar_exact_p(5, 5) == 1.0
    assert stats.mcnemar_exact_p(1, 9) == pytest.approx(2 * (1 + 10) / 1024)  # exact binomial tail
    assert stats.mcnemar_exact_p(9, 1) == stats.mcnemar_exact_p(1, 9)
    diff, lo, hi = stats.paired_diff_ci(12, 18, 200)
    assert diff == pytest.approx(-0.03) and lo < diff < hi
    assert hi - lo == pytest.approx(2 * stats.Z95 * math.sqrt(30 - 36 / 200) / 200)
    # 200 pairs cannot establish a 2-point margin; the module says how many would.
    assert stats.episodes_needed(0.02) > 1400 and stats.episodes_needed(0.055) <= 200


def _fake_eval(rates_by_task, n=20, seed_base=100, seeded=True, flip=()):
    tasks = []
    for t_idx, (task, k) in enumerate(rates_by_task.items()):
        eps = [{"episode": i, "seed": seed_base + 1000 * t_idx + i, "success": i < k} for i in range(n)]
        for i in flip:
            eps[i]["success"] = not eps[i]["success"]
        tasks.append({"task": task, "n_episodes": n, "success_rate": sum(e["success"] for e in eps) / n, "episodes": eps})
    return {"schema": 2, "seeded": seeded, "tasks": tasks}


def test_paired_comparison_counts_discordant_episodes():
    base = _fake_eval({"a": 16, "b": 10})
    cand = _fake_eval({"a": 18, "b": 9})
    cmp = stats.compare_evals(base, cand, margin=0.02)
    assert cmp["paired"] and cmp["pairs"] == 40
    assert (cmp["candidate_only_success"], cmp["baseline_only_success"]) == (2, 1)
    assert cmp["both_success"] == 25 and cmp["both_fail"] == 12
    assert cmp["diff"] == pytest.approx(1 / 40) and cmp["mcnemar_p"] == 1.0
    assert cmp["per_task_diff"] == {"a": 0.1, "b": -0.05}
    assert cmp["diff_ci95"][0] < cmp["diff"] < cmp["diff_ci95"][1]
    assert cmp["diff_ci95_task_bootstrap"][0] <= cmp["diff"] <= cmp["diff_ci95_task_bootstrap"][1]
    assert not cmp["non_inferior_at_margin"]  # +2.5 pts measured, but 40 pairs cannot exclude -2
    assert "McNemar p=1.0" in stats.format_comparison("cand", cmp)

    same = stats.compare_evals(base, base)
    assert same["diff"] == 0 and same["diff_ci95"] == [0.0, 0.0] and same["non_inferior_at_margin"]

    # An episode without outcome on one side drops out of the pairing instead of counting as a failure.
    cand["tasks"][0]["episodes"][0]["success"] = None
    assert stats.compare_evals(base, cand)["pairs"] == 39
    # Different seeds, or an unseeded run: never silently "paired".
    assert not stats.compare_evals(base, _fake_eval({"a": 18, "b": 9}, seed_base=7))["paired"]
    assert not stats.compare_evals(base, _fake_eval({"a": 18, "b": 9}, seeded=False))["paired"]


def test_archived_n5_evals_compare_unpaired_and_inconclusive(tmp_path, capsys):
    """The two archived 50-episode runs (85.00 % vs 81.67 %) have no per-episode records."""
    base = RESULTS / "e2e_ablation_3000s" / "eval_group1_native.json"
    cand = RESULTS / "online_turbo_3000s" / "eval_mn2.json"
    out = tmp_path / "cmp.json"
    rc = cmp_cli.main(["--baseline", str(base), f"mn2={cand}", "--out-json", str(out)])
    assert rc == 0
    cmp = json.loads(out.read_text())["comparisons"]["mn2"]
    assert not cmp["paired"] and cmp["mcnemar_p"] is None
    lo, hi = cmp["diff_ci95"]
    assert lo < -0.10 and hi > 0.10  # 50 vs 50 episodes: the interval spans more than +-10 points
    assert "UNPAIRED" in capsys.readouterr().out
    assert cmp_cli.main(["--baseline", str(base), str(cand), "--require-non-inferior"]) == 1


def _rows(n, loss=0.5, mem=18000.0, step_s=0.33, turbo=None):
    rows = [{"event": "train_begin", "online_turbo": turbo}] if turbo is not None else []
    rows += [{"event": "step", "global_step": i + 1, "step_loss": loss, "gpu_mem_mb": mem, "step_s": step_s} for i in range(n)]
    return rows


def test_smoke_gate_catches_silent_slow_paths():
    from gr00t_lancedb.turbo import lowmem_adamw as L

    assert checks.FUSED_ACTIVE_MARKER == L.FUSED_ACTIVE_MARKER
    assert checks.FUSED_FALLBACK_MARKER == L.FUSED_FALLBACK_MARKER
    turbo = {"enabled": True, "compile_applied": True}
    log = f"x\n{L.FUSED_ACTIVE_MARKER}: 537 tensors\n"
    assert checks.smoke_problems(_rows(5, turbo=turbo), log, expect_compile=True, expect_fused=True) == []

    def has(problems, text):
        return any(text in p for p in problems)

    assert has(checks.smoke_problems(_rows(4, turbo=turbo), log), "expected 5 step rows")
    assert has(checks.smoke_problems(_rows(5, loss=float("nan"), turbo=turbo), log), "non-finite loss")
    assert has(checks.smoke_problems(_rows(5), log), "online_turbo was not active")
    skipped = {"enabled": True, "compile_applied": False, "compile_skipped_reason": "fp32 weights"}
    assert has(checks.smoke_problems(_rows(5, turbo=skipped), log, expect_compile=True), "fp32 weights")
    fb = f"WARNING {L.FUSED_FALLBACK_MARKER}: fused multi-tensor launch unavailable (CompilationError: x)"
    assert has(checks.smoke_problems(_rows(5, turbo=turbo), fb, expect_fused=True), "one launch per tensor")
    assert has(checks.smoke_problems(_rows(5, turbo=turbo), "", expect_fused=True), "never became active")
    assert has(checks.smoke_problems(_rows(5, turbo=turbo), log, orig_mod_keys=["a._orig_mod.w"]), "_orig_mod")
    # stock baseline: turbo must be OFF
    assert checks.smoke_problems(_rows(5), "", expect_turbo=False) == []
    assert has(checks.smoke_problems(_rows(5, turbo=turbo), "", expect_turbo=False), "stock run has online_turbo")


def test_bench_summary_separates_oom_from_other_failures():
    rows = _rows(10, step_s=5.0) + _rows(50, step_s=0.33, mem=18990.0)
    ok = checks.bench_summary(rows, "", mode="mn2", steps=60)
    assert ok["status"] == "ok" and ok["median_step_ms"] == 330.0 and ok["steady_steps"] == 50
    assert ok["peak_gpu_mem_gb"] == pytest.approx(18.54, abs=0.01) and ok["fits_24gb"]

    oom_log = "step 1\ntorch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.2 GiB\n"
    oom = checks.bench_summary(_rows(2, mem=24100.0), oom_log, mode="stock", steps=60)
    assert oom["status"] == "oom" and oom["steps_done"] == 2 and "median_step_ms" not in oom
    assert "CUDA out of memory" in oom["oom_line"] and oom["fits_24gb"]  # peak before the failing allocation
    assert checks.bench_summary([], "RuntimeError: CUDA error: out of memory", mode="stock", steps=60)["status"] == "oom"
    assert checks.bench_summary(_rows(3), "ValueError: bad flag", mode="x", steps=60)["status"] == "error"

    table = checks.bench_table([ok, oom, {"mode": "x", "status": "error", "steps_done": 0}], gpu="L40 24GB")
    assert "| `mn2` | runs | 18.54 GB | 330.0 | 1.00x |" in table
    assert "| `stock` | CUDA OOM at step 3 | >= 23.54 GB | - | - |" in table
    assert "ERROR (not OOM)" in table


_STUB_FINETUNE = r"""#!/usr/bin/env bash
# Stands in for run_gr00t_finetune.sh: records how it was called, then behaves like a GPU run.
set -euo pipefail
mkdir -p "$OUT"
{
  echo "ARGS=$*"
  env | grep -E '^(GR00T_TURBO_|GR00T_BENCH_|GLOBAL_BATCH_SIZE=|MAX_STEPS=|SAVE_STEPS=)' | sort
} > "$OUT.call"
if [ "$GLOBAL_BATCH_SIZE" = "32" ] || [ "${GR00T_TURBO_OPTIMIZER}" = "native" ]; then
  echo '{"event": "step", "global_step": 1, "step_s": 1.0, "step_loss": 0.9, "gpu_mem_mb": 24050}' > "$OUT/bench_metrics.jsonl"
  echo "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total capacity of 23.99 GiB"
  exit 1
fi
python3 - "$OUT" "$MAX_STEPS" <<'PY'
import json, sys
with open(sys.argv[1] + "/bench_metrics.jsonl", "w") as fh:
    for i in range(int(sys.argv[2])):
        fh.write(json.dumps({"event": "step", "global_step": i + 1, "step_s": 0.3, "step_loss": 0.5, "gpu_mem_mb": 18900}) + "\n")
PY
echo "LOWMEM_FUSED_ACTIVE: 537 tensors"
"""


def test_proof_script_end_to_end_with_stubbed_training(tmp_path):
    """Runs the real bench_24gb_proof.sh -> run_online_turbo.sh -> turbo_run_checks.py chain;
    only the training launch and nvidia-smi are stubs. Catches shell bugs before a GPU does."""
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    for name in ("run_online_turbo.sh", "bench_24gb_proof.sh", "turbo_run_checks.py"):
        (repo / "scripts" / name).write_text((SCRIPTS / name).read_text())
    stub = repo / "scripts" / "run_gr00t_finetune.sh"
    stub.write_text(_STUB_FINETUNE)
    stub.chmod(0o755)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    smi = bin_dir / "nvidia-smi"
    smi.write_text(
        '#!/usr/bin/env bash\ncase "$*" in\n'
        '  *name,memory.total*) echo "NVIDIA L40, 24576 MiB";;\n'
        '  *memory.total*nounits*) echo 24576;;\n'
        '  *) echo "smi $*";;\nesac\n'
    )
    smi.chmod(0o755)
    out_root = tmp_path / "runs"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "REPO_ROOT": str(repo),
        "GR00T_ROOT": str(tmp_path),
        "GR00T_ENV_FILE": str(tmp_path / "none.sh"),
        "OUT_ROOT": str(out_root),
        "SKIP_TRITON_TEST": "1",
        "STEPS": "30",
    }
    res = subprocess.run(
        ["bash", str(repo / "scripts" / "bench_24gb_proof.sh")], env=env, capture_output=True, text=True, timeout=120
    )
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    table = (out_root / "proof_24gb.md").read_text()
    assert "| `mn2` | runs | 18.46 GB | 300.0 | 1.00x |" in table
    assert "| `mn4` | runs |" in table
    assert "| `stock` | CUDA OOM at step 2 | >= 23.49 GB | - | - |" in table
    assert "| `stock_ga2` | CUDA OOM at step 2 |" in table
    runs = {r["mode"]: r for r in json.loads((out_root / "proof_24gb.json").read_text())["runs"]}
    assert [runs[m]["status"] for m in ("mn2", "mn4", "stock", "stock_ga2")] == ["ok", "ok", "oom", "oom"]

    def call(mode):
        return (out_root / f"bench_{mode}.call").read_text()

    mn2, mn4, stock, ga2 = call("mn2"), call("mn4"), call("stock"), call("stock_ga2")
    assert "GLOBAL_BATCH_SIZE=16" in mn2 and "GR00T_TURBO_MULTI_NOISE=2" in mn2 and "GR00T_TURBO_OPTIMIZER=lowmem" in mn2
    assert "GR00T_TURBO_COMPILE_HEAD=1" in mn2 and "GR00T_TURBO_LOWMEM_FUSED=auto" in mn2
    assert "GR00T_BENCH_SKIP_FINAL_SAVE=1" in mn2 and "MAX_STEPS=30" in mn2 and "ARGS=\n" in mn2
    assert "GLOBAL_BATCH_SIZE=8" in mn4 and "GR00T_TURBO_MULTI_NOISE=4" in mn4
    # B0: nothing of the turbo path may be on
    assert "GLOBAL_BATCH_SIZE=32" in stock and "GR00T_TURBO_OPTIMIZER=native" in stock
    for off in ("GR00T_TURBO_BF16_WEIGHTS=0", "GR00T_TURBO_BF16_BACKBONE=0", "GR00T_TURBO_COMPILE_HEAD=0", "GR00T_TURBO_MULTI_NOISE=1"):
        assert off in stock
    # B1: batch 16 x grad-accum 2, frozen backbone in bf16, fp32 head, stock optimizer
    assert "ARGS=--gradient_accumulation_steps 2" in ga2 and "GLOBAL_BATCH_SIZE=16" in ga2
    assert "GR00T_TURBO_BF16_BACKBONE=1" in ga2 and "GR00T_TURBO_BF16_WEIGHTS=0" in ga2
    assert "GR00T_TURBO_OPTIMIZER=native" in ga2 and "GR00T_TURBO_COMPILE_HEAD=0" in ga2

    # A failure that is not an out-of-memory error must fail the proof run.
    stub.write_text("#!/usr/bin/env bash\necho 'ValueError: unrecognized arguments'\nexit 2\n")
    res = subprocess.run(
        ["bash", str(repo / "scripts" / "bench_24gb_proof.sh")],
        env={**env, "MODES": "mn2"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert res.returncode != 0 and "ERROR (not OOM)" in res.stdout


def test_bnb8bit_mode_exports():
    """MODE=bnb8bit (B2) through the real run script, bench mode, stubbed training."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        repo = Path(td) / "repo"
        (repo / "scripts").mkdir(parents=True)
        for name in ("run_online_turbo.sh", "turbo_run_checks.py"):
            (repo / "scripts" / name).write_text((SCRIPTS / name).read_text())
        stub = repo / "scripts" / "run_gr00t_finetune.sh"
        stub.write_text(_STUB_FINETUNE.replace('|| [ "${GR00T_TURBO_OPTIMIZER}" = "native" ]', ""))
        stub.chmod(0o755)
        env = {
            **os.environ,
            "REPO_ROOT": str(repo),
            "GR00T_ROOT": td,
            "GR00T_ENV_FILE": f"{td}/none.sh",
            "OUT_ROOT": f"{td}/runs",
            "MODE": "bnb8bit",
            "BENCH_STEPS": "30",
        }
        res = subprocess.run(["bash", str(repo / "scripts" / "run_online_turbo.sh")], env=env, capture_output=True, text=True)
        assert res.returncode == 0, res.stdout[-2000:] + res.stderr[-2000:]
        call = Path(td, "runs", "bench_bnb8bit.call").read_text()
        assert "GR00T_TURBO_OPTIMIZER=bnb8bit" in call and "GR00T_TURBO_MULTI_NOISE=2" in call
        assert "GR00T_TURBO_BF16_WEIGHTS=0" in call and "GR00T_TURBO_ALLOW_PURE_BF16=0" in call
        res = subprocess.run(
            ["bash", str(repo / "scripts" / "run_online_turbo.sh")], env={**env, "MODE": "nope"}, capture_output=True, text=True
        )
        assert res.returncode == 2 and "Unknown MODE" in res.stderr


_STUB_FULL = r"""#!/usr/bin/env bash
set -euo pipefail
mkdir -p "$OUT/checkpoint-$MAX_STEPS"
"$STUB_PY" - "$OUT" "$MAX_STEPS" <<'PY'
import json, os, sys
import torch
from safetensors.torch import save_file
out, n = sys.argv[1], int(sys.argv[2])
turbo = {"enabled": True, "compile_applied": os.environ["STUB_COMPILE_APPLIED"] == "1", "compile_skipped_reason": "stub says no"}
with open(out + "/bench_metrics.jsonl", "w") as fh:
    fh.write(json.dumps({"event": "train_begin", "online_turbo": turbo}) + "\n")
    for i in range(n):
        fh.write(json.dumps({"event": "step", "global_step": i + 1, "step_s": 0.3, "step_loss": 0.5, "gpu_mem_mb": 18900}) + "\n")
save_file({os.environ.get("STUB_KEY", "action_head.model.w"): torch.zeros(2)}, f"{out}/checkpoint-{n}/model.safetensors")
PY
echo "$STUB_FUSED_LINE"
"""


def test_smoke_gate_in_the_real_run_script(tmp_path):
    """Gate 1 -> main run through run_online_turbo.sh with a stubbed trainer and `uv`."""
    pytest.importorskip("safetensors")
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    for name in ("run_online_turbo.sh", "turbo_run_checks.py"):
        (repo / "scripts" / name).write_text((SCRIPTS / name).read_text())
    stub = repo / "scripts" / "run_gr00t_finetune.sh"
    stub.write_text(_STUB_FULL)
    stub.chmod(0o755)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"  # `uv run --no-sync python X ...` -> this interpreter
    uv.write_text(f'#!/usr/bin/env bash\nshift 3\nexec "{sys.executable}" "$@"\n')
    uv.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "REPO_ROOT": str(repo),
        "GR00T_ROOT": str(tmp_path),
        "GR00T_ENV_FILE": str(tmp_path / "none.sh"),
        "OUT_ROOT": str(tmp_path / "runs"),
        "SKIP_TRITON_TEST": "1",
        "EVAL": "0",
        "MAX_STEPS": "7",
        "MODE": "mn2",
        "COMPILE_HEAD": "1",
        "STUB_PY": sys.executable,
        "STUB_COMPILE_APPLIED": "1",
        "STUB_FUSED_LINE": "LOWMEM_FUSED_ACTIVE: 537 tensors (1581870 programs) per launch, backend=triton",
    }

    def run(**over):
        return subprocess.run(
            ["bash", str(repo / "scripts" / "run_online_turbo.sh")], env={**env, **over}, capture_output=True, text=True, timeout=120
        )

    ok = run()
    assert ok.returncode == 0, ok.stdout[-3000:] + ok.stderr[-3000:]
    assert "[Gate 1 PASS]" in ok.stdout and "compile=applied fused=active" in ok.stdout
    meta = json.loads((tmp_path / "runs" / "mn2" / "online_turbo_run.json").read_text())
    assert meta["mode"] == "mn2" and meta["compile_head"] == 1 and meta["multi_noise"] == 2
    assert not (tmp_path / "runs" / "_smoke_mn2").exists()

    for over, why in (
        ({"STUB_COMPILE_APPLIED": "0"}, "torch.compile was not applied (stub says no)"),
        ({"STUB_FUSED_LINE": "LOWMEM_FUSED_FALLBACK: fused multi-tensor launch unavailable (X)"}, "one launch per tensor"),
        ({"STUB_FUSED_LINE": "nothing"}, "never became active"),
        ({"STUB_KEY": "action_head.model._orig_mod.w"}, "_orig_mod keys leaked"),
    ):
        bad = run(RERUN="1", **over)
        assert bad.returncode != 0 and "[Gate 1 FAIL]" in bad.stdout and why in bad.stdout, (why, bad.stdout[-1500:])
        assert "[Main]" not in bad.stdout  # stopped before the training run
    # LOWMEM_FUSED=off is an explicit choice: the gate must not demand the fused marker.
    assert run(RERUN="1", STUB_FUSED_LINE="nothing", LOWMEM_FUSED="off").returncode == 0


# ---------------------------------------------------------------------------
# Exp-Falsify (docs/plans/07): modes, 24 GiB cap, verdict rules, orchestration
# ---------------------------------------------------------------------------

verdict = _load("falsify_verdict")
launcher = _load("launch_finetune_lance")


def _paired(n_pairs: int, both: int, cand_only: int, base_only: int, seed_base: int = 20260):
    """Two seeded eval JSONs (one task) with the given 2x2 counts of paired outcomes."""
    outcomes = [(True, True)] * both + [(False, True)] * cand_only + [(True, False)] * base_only
    outcomes += [(False, False)] * (n_pairs - len(outcomes))

    def as_eval(side: int):
        eps = [{"episode": i, "seed": seed_base + i, "success": o[side]} for i, o in enumerate(outcomes)]
        k = sum(e["success"] for e in eps)
        return {
            "schema": 2, "seeded": True, "complete": True, "mean_success_rate": k / n_pairs, "successes": k,
            "valid_episodes": n_pairs, "total_episodes": n_pairs, "ci95": list(stats.wilson_ci(k, n_pairs)),
            "tasks": [{"task": "t", "n_episodes": n_pairs, "episodes": eps}],
        }

    return as_eval(0), as_eval(1)  # (baseline, candidate)


def test_falsify_verdict_a_lines():
    # PASS: -3 pts, p well above 0.05
    base, cand = _paired(100, 70, 5, 8)
    a = verdict.verdict_a_accuracy(stats.compare_evals(base, cand))
    assert a["result"] == "PASS" and a["diff"] == pytest.approx(-0.03) and a["mcnemar_p"] >= 0.05
    # ALARM: -6 pts, even though p >= 0.05 at n=100
    base, cand = _paired(100, 70, 4, 10)
    a = verdict.verdict_a_accuracy(stats.compare_evals(base, cand))
    assert a["result"] == "ALARM" and a["diff"] == pytest.approx(-0.06)
    # ALARM: only -4.5 pts, but significant (0 vs 9 discordant episodes, p < 0.01)
    base, cand = _paired(200, 140, 0, 9)
    a = verdict.verdict_a_accuracy(stats.compare_evals(base, cand))
    assert a["result"] == "ALARM" and a["diff"] == pytest.approx(-0.045) and a["mcnemar_p"] < 0.05
    # GRAY: -4.5 pts, not significant
    base, cand = _paired(200, 140, 6, 15)
    a = verdict.verdict_a_accuracy(stats.compare_evals(base, cand))
    assert a["result"] == "GRAY" and a["mcnemar_p"] >= 0.05
    # significantly BETTER: PASS, flagged
    base, cand = _paired(100, 60, 12, 1)
    a = verdict.verdict_a_accuracy(stats.compare_evals(base, cand))
    assert a["result"] == "PASS" and a["mcnemar_p"] < 0.05 and "significantly better" in a["note"]


def test_falsify_verdict_b_burden_is_on_multi_noise():
    base, cand = _paired(100, 70, 8, 5)  # mn2 +3 pts, not significant -> CUT
    assert verdict.verdict_b(stats.compare_evals(base, cand))["result"] == "CUT"
    base, cand = _paired(100, 70, 5, 5)  # tie -> CUT
    assert verdict.verdict_b(stats.compare_evals(base, cand))["result"] == "CUT"
    base, cand = _paired(100, 60, 12, 1)  # mn2 +11 pts, p < 0.05 -> KEEP (preliminary)
    assert verdict.verdict_b(stats.compare_evals(base, cand))["result"].startswith("KEEP")
    base, cand = _paired(100, 60, 1, 12)  # mn2 significantly worse -> CUT
    assert verdict.verdict_b(stats.compare_evals(base, cand))["result"] == "CUT"


def test_falsify_verdict_memory_needs_the_capped_run():
    assert verdict.verdict_a_memory(None)["result"] == "NOT RUN"
    ok = {"runs": [{"mode": "b16_k1_cap24", "status": "ok", "peak_gpu_mem_gb": 17.9, "cuda_mem_limit_gib": 23.6}]}
    assert verdict.verdict_a_memory(ok)["result"] == "PASS"
    big = {"runs": [{"mode": "b16_k1_cap24", "status": "ok", "peak_gpu_mem_gb": 21.0, "cuda_mem_limit_gib": 23.6}]}
    assert verdict.verdict_a_memory(big)["result"] == "FAIL"
    oom = {"runs": [{"mode": "b16_k1_cap24", "status": "oom", "peak_gpu_mem_gb": 23.5, "cuda_mem_limit_gib": 23.6}]}
    assert verdict.verdict_a_memory(oom)["result"] == "FAIL"
    uncapped = {"runs": [{"mode": "b16_k1", "status": "ok", "peak_gpu_mem_gb": 17.9}]}
    assert verdict.verdict_a_memory(uncapped)["result"] == "FAIL"


def test_falsify_verdict_cli_end_to_end(tmp_path, capsys):
    b16_fp32, b16_lowmem = _paired(100, 70, 5, 8)
    # mn2 must be paired with b16_lowmem on the same seeds: reuse the b16_lowmem outcomes, flip 3 to success
    mn2 = json.loads(json.dumps(b16_lowmem))
    flipped = 0
    for e in mn2["tasks"][0]["episodes"]:
        if not e["success"] and flipped < 3:
            e["success"], flipped = True, flipped + 1
    paths = {}
    for name, obj in (("b0", b16_fp32), ("b16_fp32", b16_fp32), ("b16_lowmem", b16_lowmem), ("mn2", mn2)):
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(obj))
    proof = tmp_path / "proof.json"
    proof.write_text(json.dumps({"runs": [{"mode": "b16_k1_cap24", "status": "ok", "peak_gpu_mem_gb": 17.2, "cuda_mem_limit_gib": 23.6}]}))
    out = tmp_path / "v.json"
    rc = verdict.main([
        "--b0", str(paths["b0"]), "--b16-fp32", str(paths["b16_fp32"]), "--b16-lowmem", str(paths["b16_lowmem"]),
        "--mn2", str(paths["mn2"]), "--proof-json", str(proof), "--out-json", str(out), "--out-md", str(tmp_path / "v.md"),
    ])
    assert rc == 0
    v = json.loads(out.read_text())
    assert v["verdict_a_accuracy"]["result"] == "PASS" and v["verdict_a_memory"]["result"] == "PASS"
    assert v["verdict_b_multi_noise"]["result"] == "CUT"
    assert v["route"].startswith("Optimizer holds, multi-noise cut")
    assert len(v["pairs"]) == 6 and all(c["paired"] for c in v["pairs"].values())
    md = (tmp_path / "v.md").read_text()
    assert "**Verdict B, multi-noise** (G_b16_mn2 vs G_b16_lowmem): **CUT**" in md
    # Unpaired evals cannot be judged by these rules.
    unseeded = dict(b16_lowmem, seeded=False)
    paths["b16_lowmem"].write_text(json.dumps(unseeded))
    with pytest.raises(SystemExit, match="not paired"):
        verdict.main(["--b0", str(paths["b0"]), "--b16-fp32", str(paths["b16_fp32"]),
                      "--b16-lowmem", str(paths["b16_lowmem"]), "--mn2", str(paths["mn2"])])


def test_bench_summary_records_allocator_cap_and_torch_peaks():
    rows = [{"event": "cuda_mem_limit", "fraction": 0.6, "limit_gib": 23.636, "device_total_gib": 39.394}]
    rows += [dict(r, torch_max_alloc_mb=15000.0 + r["global_step"], torch_max_reserved_mb=16500.0) for r in _rows(20, mem=17900.0)]
    s = checks.bench_summary(rows, "", mode="b16_k1_cap24", steps=20)
    assert s["status"] == "ok" and s["cuda_mem_fraction"] == 0.6 and s["cuda_mem_limit_gib"] == 23.636
    assert s["peak_torch_alloc_gb"] == pytest.approx(15020 / 1024, abs=0.01)
    assert s["peak_torch_reserved_gb"] == pytest.approx(16.11, abs=0.01)
    table = checks.bench_table([s])
    assert "| `b16_k1_cap24` | runs | 17.48 GB |" in table and "| 14.67 GB | 23.6 GiB |" in table
    assert "| none |" in checks.bench_table([checks.bench_summary(_rows(20), "", mode="x", steps=20)])


def test_launcher_mem_fraction_is_strict():
    assert launcher._apply_cuda_mem_fraction(None) is None and launcher._apply_cuda_mem_fraction("  ") is None
    for bad in ("0", "1.5", "-0.1"):
        with pytest.raises(ValueError):
            launcher._apply_cuda_mem_fraction(bad)
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            launcher._apply_cuda_mem_fraction("0.6")  # never a silently uncapped run


def test_optim_bench_table_keeps_the_single_tensor_floor_apart():
    res = {
        "lowmem_bf16_pt": {
            "tensors": 980, "step_ms": 56.3, "weights_plus_state_bytes_per_param": 8.06, "peak_alloc_gb": 12.2, "fused": "off"
        },
        "torchao_8bit_fp32": {"skipped": "torchao not installed"},
        "lowmem_bf16_1t": {"tensors": 1, "step_ms": 16.9, "weights_plus_state_bytes_per_param": 8.06, "peak_alloc_gb": 12.2},
    }
    table = bench.results_table(res)
    main, floor = table.split("Single-tensor floor")
    assert "`lowmem_bf16_pt` | 980 | 56.3" in main and "torchao not installed" in main and "_1t" not in main
    assert "`lowmem_bf16_1t` | 1 | 16.9" in floor


_STUB_RECORD = r"""#!/usr/bin/env bash
set -euo pipefail
mkdir -p "$OUT"
env | grep -E '^(GR00T_TURBO_|GR00T_CUDA_MEM_FRACTION=|GLOBAL_BATCH_SIZE=|MAX_STEPS=)' | sort > "$OUT.call"
python3 - "$OUT" "$MAX_STEPS" <<'PY'
import json, sys
with open(sys.argv[1] + "/bench_metrics.jsonl", "w") as fh:
    for i in range(int(sys.argv[2])):
        fh.write(json.dumps({"event": "step", "global_step": i + 1, "step_s": 0.3, "step_loss": 0.5, "gpu_mem_mb": 17000}) + "\n")
PY
"""


def test_falsify_modes_are_single_variable_steps(tmp_path):
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    for name in ("run_online_turbo.sh", "turbo_run_checks.py"):
        (repo / "scripts" / name).write_text((SCRIPTS / name).read_text())
    stub = repo / "scripts" / "run_gr00t_finetune.sh"
    stub.write_text(_STUB_RECORD)
    stub.chmod(0o755)
    env = {**os.environ, "REPO_ROOT": str(repo), "GR00T_ROOT": str(tmp_path), "GR00T_ENV_FILE": str(tmp_path / "none.sh"),
           "OUT_ROOT": str(tmp_path / "runs"), "BENCH_STEPS": "20", "SKIP_TRITON_TEST": "1"}
    calls = {}
    for mode, tag, frac in (("b16_k1", "_cap24", "0.6"), ("b16_k1_fp32", "", ""), ("mn2", "", "")):
        over = {"MODE": mode, "BENCH_TAG": tag, **({"GR00T_CUDA_MEM_FRACTION": frac} if frac else {})}
        res = subprocess.run(["bash", str(repo / "scripts" / "run_online_turbo.sh")], env={**env, **over}, capture_output=True, text=True)
        assert res.returncode == 0, res.stdout[-2000:] + res.stderr[-2000:]
        run = tmp_path / "runs" / f"bench_{mode}{tag}"
        assert json.loads((run / "bench_summary.json").read_text())["mode"] == f"{mode}{tag}"
        calls[mode] = dict(line.split("=", 1) for line in Path(f"{run}.call").read_text().splitlines())
    k1, fp32, mn2 = calls["b16_k1"], calls["b16_k1_fp32"], calls["mn2"]
    assert k1["GR00T_CUDA_MEM_FRACTION"] == "0.6" and "GR00T_CUDA_MEM_FRACTION" not in fp32
    # b16_k1 -> mn2: only the number of noise draws differs
    diff = {k for k in set(k1) | set(mn2) if k1.get(k) != mn2.get(k)} - {"GR00T_CUDA_MEM_FRACTION"}
    assert diff == {"GR00T_TURBO_MULTI_NOISE"} and (k1["GR00T_TURBO_MULTI_NOISE"], mn2["GR00T_TURBO_MULTI_NOISE"]) == ("1", "2")
    # b16_k1_fp32: B0 at batch 16, nothing of the turbo path
    assert fp32["GLOBAL_BATCH_SIZE"] == "16" and fp32["GR00T_TURBO_OPTIMIZER"] == "native"
    for key in ("GR00T_TURBO_BF16_WEIGHTS", "GR00T_TURBO_BF16_BACKBONE", "GR00T_TURBO_COMPILE_HEAD"):
        assert fp32[key] == "0"
    assert fp32["GR00T_TURBO_MULTI_NOISE"] == "1"
    assert k1["GLOBAL_BATCH_SIZE"] == "16" and k1["GR00T_TURBO_OPTIMIZER"] == "lowmem" and k1["GR00T_TURBO_BF16_WEIGHTS"] == "1"


_STUB_EVAL = r"""import argparse, json, os, sys, time
p = argparse.ArgumentParser()
for a in ("--gr00t-root", "--model-path", "--n-episodes", "--n-envs", "--seed-base", "--port", "--out-json"):
    p.add_argument(a)
a = p.parse_args()
lock = os.path.join(os.environ["STUB_LOCKS"], a.port)
if os.path.exists(lock):
    sys.exit(f"port {a.port} already in use")
open(lock, "w").close()
with open(os.environ["STUB_LOG"], "a") as fh:
    fh.write(f"{a.model_path} {a.port} {a.n_episodes} {a.seed_base}\n")
time.sleep(float(os.environ.get("STUB_SLEEP_" + os.path.basename(os.path.dirname(a.model_path)), "0.2")))
os.remove(lock)
if "fail" in a.model_path:
    sys.exit(3)
json.dump({"ok": True}, open(a.out_json, "w"))
"""


def test_run_falsify_orchestration(tmp_path):
    def tree(name, src_text):
        root = tmp_path / name
        (root / "scripts").mkdir(parents=True)
        (root / "src/gr00t_lancedb/turbo").mkdir(parents=True)
        (root / "src/gr00t_lancedb/turbo/online_turbo.py").write_text(src_text)
        (root / "scripts" / "run_falsify.sh").write_text((SCRIPTS / "run_falsify.sh").read_text())
        (root / "scripts" / "eval_libero10_checkpoint.py").write_text(_STUB_EVAL)
        return root

    frozen = tree("frozen", 'optimizer: str = "native"\n')
    wip = tree("wip", 'lowmem_fused: str = "auto"\n')
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "uv").write_text(f'#!/usr/bin/env bash\nshift 3\nexec "{sys.executable}" "$@"\n')
    (bin_dir / "uv").chmod(0o755)
    train = tmp_path / "train"
    ckpts = {}
    for name in ("b0", "mn2", "b16_k1", "b16_k1_fp32"):
        ckpts[name] = train / name / "checkpoint-3000"
        ckpts[name].mkdir(parents=True)
    (tmp_path / "locks").mkdir()
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "GR00T_ENV_FILE": str(tmp_path / "none.sh"),
           "GR00T_ROOT": str(tmp_path), "FALSIFY_ROOT": str(tmp_path / "falsify"), "TRAIN_ROOT": str(train),
           "B0_CKPT": str(ckpts["b0"]), "STUB_LOCKS": str(tmp_path / "locks"), "STUB_LOG": str(tmp_path / "evals.log"),
           "STUB_SLEEP_b0": "1.0"}

    def run(root, **over):
        return subprocess.run(["bash", str(root / "scripts" / "run_falsify.sh")], env={**env, **over},
                              capture_output=True, text=True, timeout=120)

    # Training from a tree with the post-93abd3a optimizer code is refused.
    for step in ("1", "2", "3"):
        res = run(wip, STEP=step)
        assert res.returncode == 2 and "not the frozen 93abd3a" in res.stderr
    # Step 4: four evals, two at a time, same seeds and episode count, no port shared concurrently.
    res = run(frozen, STEP="4")
    assert res.returncode == 0, res.stdout[-2000:] + res.stderr[-2000:]
    lines = (tmp_path / "evals.log").read_text().splitlines()
    assert len(lines) == 4 and {ln.split()[2] for ln in lines} == {"10"} and {ln.split()[3] for ln in lines} == {"20260"}
    assert {Path(ln.split()[0]).parent.name for ln in lines} == {"b0", "mn2", "b16_k1", "b16_k1_fp32"}
    assert {ln.split()[1] for ln in lines} == {"5561", "5562"}
    for name in ("b0", "mn2", "b16_lowmem", "b16_fp32"):
        assert (tmp_path / "falsify" / "eval" / f"{name}.json").is_file()
    # Re-running skips finished evals; a failing eval is reported and fails the step.
    assert "skipping" in run(frozen, STEP="4").stdout
    ckpts["fail"] = train / "fail" / "checkpoint-3000"
    ckpts["fail"].mkdir(parents=True)
    res = run(frozen, STEP="4", MN2_CKPT=str(ckpts["fail"]), RERUN_EVAL="1", EVAL_ONLY="mn2")
    assert res.returncode == 3 and "did not complete: mn2" in res.stdout
    # A missing checkpoint stops Step 4 before any eval starts.
    res = run(frozen, STEP="4", B0_CKPT=str(tmp_path / "nope"))
    assert res.returncode == 2 and "not found" in res.stderr
