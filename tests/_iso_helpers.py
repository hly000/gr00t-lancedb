# SPDX-License-Identifier: Apache-2.0
"""Functions the isolated child process calls in tests (must be importable by name)."""

from __future__ import annotations

import os
import time


def slow_echo(value: int, delay: float = 0.0) -> dict:
    time.sleep(delay)
    return {"value": value, "pid": os.getpid()}


def boom() -> dict:
    raise ValueError("boom from the child")
