# SPDX-License-Identifier: Apache-2.0
"""Exact-resume replay of GR00T's fast mixture, tested without GR00T installed.

``_FakeFastMixture`` reproduces the parts of ``gr00t.data.dataset.fast_sharded_dataset.
FastShardedMixtureDataset`` (Isaac-GR00T, Apache-2.0) that the mixin relies on: the shard
schedule, the per-worker partition, the split iterator with epoch roll-over, and the
seed-determined shuffle buffer. Real-GR00T equivalence is checked separately by
``scripts/check_exact_resume.py`` on a GPU box.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from gr00t_lancedb.exact_resume import ExactResumeMixin, served_by_worker


class _FakeDataset:
    """Shards of (episode, step-indices) splits; items are (episode, step) tuples."""

    def __init__(self, n_episodes: int = 7, ep_len: int = 11, n_shards: int = 6, seed: int = 3):
        rng = np.random.default_rng(seed)
        splits = []
        for ep in rng.permutation(n_episodes):
            steps = rng.permutation(ep_len)
            splits += [(int(ep), steps[i::5]) for i in range(5)]
        self.shards = [splits[i::n_shards] for i in range(n_shards)]

    def __len__(self):
        return len(self.shards)

    def get_shard_length(self, i):
        return sum(len(s[1]) for s in self.shards[i])

    def get_shard_splits(self, i):
        return self.shards[i]

    def load_split(self, split):
        ep, steps = split
        return [(ep, int(s)) for s in steps]

    def process_split_item(self, item):
        # Include an RNG draw so per-sample seeding is observable.
        return (*item, int(np.random.randint(1_000_000)))


class _FakeFastMixture(IterableDataset):
    def __init__(self, datasets, seed=42, num_shards_per_epoch=6, shuffle_buffer=16, min_fill=0, decode_threads=2):
        self.datasets, self.seed, self.epoch = datasets, seed, 0
        self.num_shards_per_epoch = num_shards_per_epoch
        self.rank, self.world_size = 0, 1
        self.worker_id = self.num_workers = None
        self.decode_threads = decode_threads
        self.shuffle_buffer = shuffle_buffer
        self.min_fill = min(shuffle_buffer, max(1, min_fill or shuffle_buffer // 4))
        self.shard_sampling_schedule = self.generate_shard_sampling_schedule()

    def generate_shard_sampling_schedule(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        ds = rng.choice(len(self.datasets), size=self.num_shards_per_epoch)
        pools = [list(rng.permutation(len(d))) for d in self.datasets]
        out = []
        for i in ds:
            if not pools[i]:
                pools[i] = list(rng.permutation(len(self.datasets[i])))
            out.append((int(i), int(pools[i].pop(0))))
        return out

    def filter_shard_sample_schedule(self):
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        if self.worker_id is None:
            self.worker_id, self.num_workers = wid, nw
        stride = self.world_size * nw
        return [s for i, s in enumerate(self.shard_sampling_schedule) if i % stride == self.rank * nw + wid]

    def _worker_schedule(self):
        return self.filter_shard_sample_schedule()

    def _iter_splits(self, schedule):
        while True:
            for di, si in schedule:
                for split in self.datasets[di].get_shard_splits(si):
                    yield di, split
            self.epoch += 1
            self.shard_sampling_schedule = self.generate_shard_sampling_schedule()
            schedule = self._worker_schedule()

    def __iter__(self):  # GR00T's FastShardedMixtureDataset.__iter__ algorithm
        rng = np.random.default_rng(self.seed + self.epoch)
        splits = self._iter_splits(self._worker_schedule())
        pending, buffer, served = deque(), [], 0
        with ThreadPoolExecutor(max_workers=self.decode_threads) as ex:
            in_flight = deque()

            def submit():
                di, split = next(splits)
                in_flight.append((di, ex.submit(self.datasets[di].load_split, split)))

            for _ in range(8 * self.decode_threads):
                submit()
            while True:
                while len(buffer) < min(self.shuffle_buffer, self.min_fill + served):
                    if not pending:
                        di, fut = in_flight.popleft()
                        pending.extend((di, it) for it in fut.result())
                        submit()
                    buffer.append(pending.popleft())
                pick = int(rng.integers(len(buffer)))
                buffer[pick], buffer[-1] = buffer[-1], buffer[pick]
                di, item = buffer.pop()
                served += 1
                yield self.datasets[di].process_split_item(item)


class _ExactFake(ExactResumeMixin, _FakeFastMixture):
    pass


def _ident(x):
    return x


def _stream(ds, n_batches, bs, nw):
    kw = {"multiprocessing_context": "fork"} if nw else {}
    dl = DataLoader(ds, batch_size=bs, num_workers=nw, collate_fn=_ident, **kw)
    out = []
    for b in dl:
        out.append(b)
        if len(out) == n_batches:
            return out
    return out


def _datasets():
    # Small on purpose: every worker stream rolls over several epochs within the test.
    return [_FakeDataset(seed=1), _FakeDataset(n_episodes=5, ep_len=17, seed=2)]


@pytest.mark.parametrize("nw", [0, 2, 3])
def test_matches_stock_algorithm_without_per_sample_seed(nw):
    stock = _stream(_FakeFastMixture(_datasets()), 30, 5, nw)
    exact = _stream(_ExactFake(_datasets(), per_sample_seed=False), 30, 5, nw)
    assert [[it[:2] for it in b] for b in stock] == [[it[:2] for it in b] for b in exact]


@pytest.mark.parametrize("nw", [0, 2, 3])
@pytest.mark.parametrize("resume_at", [1, 7, 12, 29])
def test_resume_continues_exact_stream(nw, resume_at):
    """Items *and* per-sample augmentation draws match the uninterrupted run, across epochs."""
    full = _stream(_ExactFake(_datasets()), 45, 6, nw)
    resumed_ds = _ExactFake(_datasets())
    resumed_ds.set_resume_point(resume_at, 6)
    resumed = _stream(resumed_ds, 45 - resume_at, 6, nw)
    assert resumed == full[resume_at:]


def test_served_by_worker_round_robin():
    assert [served_by_worker(7, 3, w, 4) for w in range(3)] == [12, 8, 8]
    assert served_by_worker(7, 0, 0, 4) == 28
