# SPDX-License-Identifier: Apache-2.0
"""Fast materialisation path, held-out probe loss and the ablation tooling (all CPU)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lancedb")

from gr00t_lancedb.vl_cache import (  # noqa: E402
    AsyncSink,
    CacheFingerprint,
    VLCacheReader,
    VLCacheWriter,
    encode_features_torch,
    pooled_visual_emb_torch,
    probe_fm_loss,
    split_holdout_rows,
)
from gr00t_lancedb.vl_cache.codec import compute_pooled_visual_emb, encode_features  # noqa: E402
from gr00t_lancedb.vl_cache.probe import (  # noqa: E402
    load_cached_batch,
    load_scalars,
    paired_probe_diff,
    prepare_probe,
    prepare_probe_isolated,
)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
D = 32


def _load_script(name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_backbone_batch(rng: np.random.Generator, bsz: int = 6, t: int = 16):
    feats = (rng.standard_normal((bsz, t, D)) * rng.lognormal(0, 1, (bsz, t, 1))).astype(np.float32)
    feats[..., 4] *= 30  # outlier channel
    feats[0, 5] = 0.0  # an all-zero token exercises the 1e-6 scale floor
    attn = np.zeros((bsz, t), dtype=bool)
    img = np.zeros((bsz, t), dtype=bool)
    for i in range(bsz):
        length = 9 + i
        attn[i, t - length :] = True  # left padded
        img[i, t - length : t - length + 5] = True
    img[1] = False  # a sample without image tokens falls back to all valid tokens
    return feats, attn, img


@pytest.mark.parametrize("codec", ["per_token_int8", "bf16_raw"])
def test_torch_encoder_is_byte_identical_to_numpy(codec):
    feats, _, _ = _fake_backbone_batch(np.random.default_rng(0))
    payload, scale = encode_features_torch(torch.from_numpy(feats), codec=codec)
    for i in range(feats.shape[0]):
        scale_b, payload_b = encode_features(feats[i], codec=codec)
        got = payload[i].numpy()
        assert (got.view(np.uint16) if codec == "bf16_raw" else got).tobytes() == payload_b
        assert (b"" if scale is None else scale[i].numpy().tobytes()) == scale_b


def _write(tmp: Path, codec: str, encoded: bool, n_batches: int = 3):
    fp = CacheFingerprint(base_model_path="test/model", select_layer=16, hidden_dim=D, quant_codec=codec)
    writer = VLCacheWriter(tmp, fp, flush_rows=5)
    rng = np.random.default_rng(1)
    row = 0
    for _ in range(n_batches):
        feats, attn, img = _fake_backbone_batch(rng)
        keys = dict(
            row_keys=list(range(row, row + len(feats))),
            episode_indices=[(row + i) // 4 for i in range(len(feats))],
            frame_indices=[(row + i) % 4 for i in range(len(feats))],
            view_ids=[0] * len(feats),
        )
        if encoded:
            ft = torch.from_numpy(feats)
            payload, scale = encode_features_torch(ft, codec=codec)
            writer.add_encoded_batch(
                **keys,
                payload=payload.numpy(),
                scale=None if scale is None else scale.numpy(),
                attention_mask=attn,
                image_mask=img,
                pooled=pooled_visual_emb_torch(ft, torch.from_numpy(attn), torch.from_numpy(img)).numpy(),
            )
        else:
            writer.add_batch(**keys, features=feats, attention_mask=attn, image_mask=img)
        row += len(feats)
    return fp, writer.close()


@pytest.mark.parametrize("codec", ["per_token_int8", "bf16_raw"])
def test_encoded_batch_path_writes_the_same_table(tmp_path: Path, codec):
    fp, s_ref = _write(tmp_path / "ref", codec, encoded=False)
    _, s_new = _write(tmp_path / "new", codec, encoded=True)
    assert s_new["rows_written"] == s_ref["rows_written"] == 18
    assert s_new["payload_bytes_written"] == s_ref["payload_bytes_written"]
    cols = ["row_key", "episode_index", "frame_index", "seq_len", "image_mask", "feat_scale", "feat_payload"]
    a = VLCacheReader(tmp_path / "ref", fp.table_name()).take_rows_arrow(range(18), columns=cols)
    b = VLCacheReader(tmp_path / "new", fp.table_name()).take_rows_arrow(range(18), columns=cols)
    assert a.equals(b)
    pa_ = VLCacheReader(tmp_path / "ref", fp.table_name()).take_rows_arrow(range(18), columns=["pooled_visual_emb"])
    pb_ = VLCacheReader(tmp_path / "new", fp.table_name()).take_rows_arrow(range(18), columns=["pooled_visual_emb"])
    ea = np.stack(pa_["pooled_visual_emb"].to_numpy(zero_copy_only=False))
    eb = np.stack(pb_["pooled_visual_emb"].to_numpy(zero_copy_only=False))
    assert np.allclose(ea, eb, atol=1e-5)


def test_pooled_embedding_matches_numpy():
    feats, attn, img = _fake_backbone_batch(np.random.default_rng(3))
    got = pooled_visual_emb_torch(torch.from_numpy(feats), torch.from_numpy(attn), torch.from_numpy(img)).numpy()
    for i in range(len(feats)):
        ref = compute_pooled_visual_emb(feats[i][attn[i]], img[i][attn[i]])
        assert np.allclose(got[i], ref, atol=1e-5)


def test_async_sink_keeps_order_and_reraises():
    seen: list[int] = []
    sink = AsyncSink(lambda x: seen.append(x), maxsize=2)
    for i in range(50):
        sink.put(x=i)
    sink.close()
    assert seen == list(range(50))

    def boom(x):
        if x == 3:
            raise ValueError("disk full")

    sink = AsyncSink(boom, maxsize=2)
    with pytest.raises(RuntimeError, match="background writer failed") as err:
        for i in range(100):  # must not dead-lock on the bounded queue after the failure
            sink.put(x=i)
        sink.close()
    assert isinstance(err.value.__cause__, ValueError)


def test_split_holdout_rows_by_episode():
    episode_index = np.repeat(np.arange(100), 30)
    train, probe = split_holdout_rows(episode_index, 0.03, seed=0, max_probe_rows=64)
    held = set(np.unique(episode_index)) - set(np.unique(episode_index[train]))
    assert len(held) == 3 and len(train) == 97 * 30 and len(probe) == 64
    assert set(episode_index[probe]) <= held  # no probe frame shares an episode with training
    t2, p2 = split_holdout_rows(episode_index, 0.03, seed=0, max_probe_rows=64)
    assert np.array_equal(train, t2) and np.array_equal(probe, p2)
    assert not np.array_equal(probe, split_holdout_rows(episode_index, 0.03, seed=1, max_probe_rows=64)[1])
    train0, probe0 = split_holdout_rows(episode_index, 0.0, seed=0, max_probe_rows=64)
    assert len(train0) == 3000 and len(probe0) == 64  # in-sample probe when nothing is held out
    with pytest.raises(ValueError):
        split_holdout_rows(np.zeros(10, dtype=int), 0.5, seed=0)


def _make_cache(tmp_path: Path, with_episode_index: bool):
    fp, summary = _write(tmp_path, "per_token_int8", encoded=True, n_batches=5)  # 30 rows, 4 per episode
    n = summary["rows_written"]
    rng = np.random.default_rng(9)
    arrays = dict(
        state=rng.standard_normal((n, 1, 8)).astype(np.float16),  # an old float16 side-car
        action=rng.standard_normal((n, 16, 7)).astype(np.float16),
        action_mask=np.ones((n, 16, 7), dtype=bool),
        embodiment_id=np.full(n, 2, dtype=np.int32),
    )
    if with_episode_index:
        arrays["episode_index"] = (np.arange(n) // 4).astype(np.int32)
    scalars_path = tmp_path / f"{fp.table_name()}_scalars.npz"
    np.savez(scalars_path, **arrays)
    (tmp_path / f"{fp.table_name()}_summary.json").write_text(json.dumps({"table_name": fp.table_name()}))
    return fp, str(scalars_path)


class _FakeHead(torch.nn.Module):
    """Loss = (mean feature + dropout-in-train-mode + seeded noise - action)^2."""

    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(()))

    def forward(self, bb, act):
        feats = bb["backbone_features"].float()
        m = bb["backbone_attention_mask"].unsqueeze(-1).float()
        ctx = 0.01 * (feats * m).sum((1, 2)) / m.sum((1, 2)) / feats.shape[-1]
        if self.training:
            ctx = ctx + 100.0  # the probe must run in eval mode
        noise = torch.randn(act["action"].shape)
        loss = (ctx[:, None, None] + self.bias + noise - act["action"]) ** 2
        return {"action_loss": loss, "action_mask": act["action_mask"], "loss": loss.mean()}


@pytest.mark.parametrize("with_episode_index", [True, False])
def test_probe_setup_and_loss(tmp_path: Path, with_episode_index):
    fp, scalars_path = _make_cache(tmp_path, with_episode_index)
    probe = prepare_probe(str(tmp_path), fp.table_name(), scalars_path, 0.25, 0, max_probe_rows=6, batch_size=4)
    assert probe["num_rows"] == 30 and probe["held_out_episodes"] == 2
    # 30 rows = 7 episodes of 4 + one of 2; the seed holds out one of each
    assert len(probe["probe_rows"]) == 6 and len(probe["train_rows"]) == 24
    assert not set(probe["probe_rows"]) & set(probe["train_rows"])
    assert [len(b["state"]) for b in probe["probe_batches"]] == [4, 2]
    assert probe["probe_batches"][0]["state"].dtype == torch.float32  # float16 side-car widened

    head = _FakeHead().train()
    torch.manual_seed(5)
    before = torch.rand(3)
    torch.manual_seed(5)
    a = probe_fm_loss(head, probe["probe_batches"], device="cpu", return_per_sample=True)
    assert torch.equal(torch.rand(3), before)  # global RNG untouched
    assert head.training  # mode restored
    b = probe_fm_loss(head, probe["probe_batches"], device="cpu", return_per_sample=True)
    assert a["probe_loss"] == b["probe_loss"] and a["probe_samples"] == 12
    assert a["probe_loss"] < 50  # eval mode: the +100 train-mode offset is absent

    with torch.no_grad():
        head.bias.fill_(0.5)
    c = probe_fm_loss(head, probe["probe_batches"], device="cpu", return_per_sample=True)
    d = paired_probe_diff(a["per_sample"], c["per_sample"])
    # paired on identical noise, a 0.5 shift is unmistakable even with 12 samples
    assert d["diff"] == pytest.approx(c["probe_loss"] - a["probe_loss"], abs=1e-9)
    assert abs(d["z"]) > 3 and d["sem"] < a["probe_sem"]


def test_probe_setup_in_spawned_child_matches(tmp_path: Path):
    fp, scalars_path = _make_cache(tmp_path, True)
    args = (str(tmp_path), fp.table_name(), scalars_path, 0.25, 0, 6, 4)
    here, child = prepare_probe(*args), prepare_probe_isolated(*args)
    assert np.array_equal(here["train_rows"], child["train_rows"])
    assert torch.equal(here["probe_batches"][0]["feat_payload"], child["probe_batches"][0]["feat_payload"])


def test_cached_dataset_respects_row_pool(tmp_path: Path):
    fp, scalars_path = _make_cache(tmp_path, True)
    train_mod = _load_script("train_vl_cache_e2e")
    pool = np.array([0, 1, 2, 3, 8, 9, 10, 11])
    ds = train_mod._CachedStepBatchDataset(
        cache_dir=str(tmp_path), table_name=fp.table_name(), scalars_path=scalars_path,
        batch_size=4, num_steps=7, seed=1, row_pool=pool,
    )
    assert set(ds.schedule.reshape(-1)) == set(pool)  # 28 draws from 8 rows: every row, nothing else
    batch = ds[0]
    assert batch["feat_payload"].shape[0] == 4 and batch["action"].dtype == torch.float32
    ref = load_cached_batch(VLCacheReader(str(tmp_path), fp.table_name()), load_scalars(scalars_path), ds.schedule[0])
    assert torch.equal(batch["feat_payload"], ref["feat_payload"])


def test_ablation_summary_and_cache_lookup(tmp_path: Path):
    summ = _load_script("summarize_cache_ablation")
    lo, hi = summ.wilson(41, 50)
    assert 0.69 < lo < 0.70 and 0.90 < hi < 0.91  # 82 % of 50 episodes: roughly +-11 points
    (tmp_path / "bf16_lowmem").mkdir()
    (tmp_path / "bf16_lowmem" / "train_summary.json").write_text(json.dumps({
        "optimizer": "LowMemAdamW(triton)", "param_dtype": "bfloat16", "median_step_ms": 150.0,
        "peak_reserved_gb": 16.2, "optimizer_state_gb": 6.6,
    }))
    (tmp_path / "bf16_lowmem" / "eval.json").write_text(json.dumps({"total_episodes": 100, "mean_success_rate": 0.8}))
    (tmp_path / "probe_compare.json").write_text(json.dumps({
        "checkpoints": {"native": {"probe_loss": 0.11, "probe_sem": 0.002},
                        "bf16_lowmem": {"probe_loss": 0.115, "probe_sem": 0.002}},
        "paired_vs_first": {"bf16_lowmem - native": {"diff": 0.005, "sem": 0.001, "z": 5.0}},
    }))
    table = summ.build_table(tmp_path, ["bf16_lowmem", "fp32_adamw"])
    assert "| bf16_lowmem | LowMemAdamW(triton) | bfloat16 | 150.0 | 16.2 | 6.6 | 0.1150 ± 0.0020 | 80.0% (n≈100" in table
    assert "| fp32_adamw | not run |" in table and "z = +5.0" in table

    ev = _load_script("eval_cached_fm_loss")
    fp, scalars_path = _make_cache(tmp_path / "cache", True)
    assert ev.find_cache(str(tmp_path / "cache"), None) == (fp.table_name(), scalars_path)
