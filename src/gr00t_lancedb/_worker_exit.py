# SPDX-License-Identifier: Apache-2.0
"""Let DataLoader workers leave promptly when training ends.

On the A100 box every Lance run spent 22.3-22.6 s between the last training step and
process exit, against 2.3 s for stock GR00T. That is PyTorch waiting 5 s
(``MP_STATUS_CHECK_INTERVAL``) for each of the 4 workers before killing it: a worker that
has left its loop still runs Python's shutdown sequence, which joins every
``ThreadPoolExecutor`` thread (decode prefetch, the Lance reader's pools) and can block on
one that never finishes.

A worker has nothing left to save at that point. Its batches were handed to the parent, and
its I/O counters are flushed by a higher-priority finalizer. So once PyTorch's worker loop
returns we leave with ``os._exit`` instead of joining threads.
"""

from __future__ import annotations

import os
import sys

_ARMED = {"pid": None}
# multiprocessing runs exit finalizers highest priority first; anything that must still run
# in the worker (the I/O statistics flush uses 10) goes before this one.
_EXIT_PRIORITY = -1000


def _hard_exit() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:  # noqa: BLE001 - never block the exit
            pass
    os._exit(0)


def arm_worker_fast_exit() -> bool:
    """Call from a dataset's ``__iter__``. Does nothing outside a DataLoader worker.

    Returns ``True`` if this process is now armed. ``GR00T_LANCE_WORKER_FAST_EXIT=0``
    restores the normal interpreter shutdown.
    """
    if os.environ.get("GR00T_LANCE_WORKER_FAST_EXIT", "1") != "1":
        return False
    pid = os.getpid()
    if _ARMED["pid"] == pid:
        return True
    try:
        from torch.utils.data import get_worker_info
    except ImportError:  # pragma: no cover - torch is a hard dependency
        return False
    if get_worker_info() is None:
        return False
    from multiprocessing import util as mp_util

    mp_util.Finalize(None, _hard_exit, exitpriority=_EXIT_PRIORITY)
    _ARMED["pid"] = pid
    return True


__all__ = ["arm_worker_fast_exit"]
