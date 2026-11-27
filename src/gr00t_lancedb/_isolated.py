# SPDX-License-Identifier: Apache-2.0
"""Run one Lance/Tokio-touching function in a short-lived, clean child interpreter.

``lancedb.connect()`` starts a Rust Tokio runtime that is not fork-safe. Anything the
*main* training process needs from Lance before DataLoader workers are forked (frame
level ``where`` anchors, version-pinned stats, cold remote ``meta.lance``) is therefore
computed here, in ``python -m gr00t_lancedb._isolated``, and only plain numpy / JSON
results come back. The parent never imports ``lancedb`` and stays ``fork``-safe.

A plain ``subprocess`` (not ``multiprocessing.spawn``) is used on purpose: spawn would
re-import the parent's ``__main__`` (GR00T's ``launch_finetune.py`` run via runpy),
which is slow and may have side effects.

A call can also be started ahead of time with :func:`start_isolated`; a later
:func:`run_isolated` with the same function and arguments then only waits for whatever is
left of it. The launcher uses this to run the startup queries while the model loads.
"""

from __future__ import annotations

import atexit
import importlib
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from typing import Any

_PKG_PARENT = str(Path(__file__).resolve().parents[1])
CHILD_ENV_FLAG = "GR00T_LANCE_ISOLATED_CHILD"


def lance_mp_context() -> str:
    """Same as ``lerobot_lancedb.lance_mp_context`` without importing lerobot/lancedb."""
    import multiprocessing

    return "forkserver" if "forkserver" in multiprocessing.get_all_start_methods() else "spawn"


class IsolatedCallError(RuntimeError):
    """Raised when the isolated child failed; carries the child's traceback text."""


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    # The child must import the same gr00t_lancedb (it may come from a PYTHONPATH /
    # editable checkout rather than site-packages).
    parts = [_PKG_PARENT, *[p for p in sys.path if p and os.path.isdir(p)]]
    seen: set[str] = set()
    ordered = [p for p in parts if not (p in seen or seen.add(p))]
    env["PYTHONPATH"] = os.pathsep.join(ordered)
    env[CHILD_ENV_FLAG] = "1"  # gr00t_lancedb/__init__ skips the huggingface_hub shim
    return env


def call_key(func: str, kwargs: dict[str, Any] | None) -> str:
    """Identity of one call: the same function with equal arguments gives the same key."""
    return func + "\x00" + json.dumps(kwargs or {}, sort_keys=True, default=repr)


class _Call:
    """One running child. A daemon thread drains its pipes and records when it ended."""

    def __init__(self, func: str, kwargs: dict[str, Any] | None) -> None:
        self.func = func
        self._dir = Path(tempfile.mkdtemp(prefix="gr00t-lance-iso-"))
        self._out = self._dir / "out.pkl"
        inp = self._dir / "in.pkl"
        with open(inp, "wb") as fh:
            pickle.dump({"func": func, "kwargs": kwargs or {}}, fh, protocol=pickle.HIGHEST_PROTOCOL)
        self.t0 = time.perf_counter()
        self.wall_s: float | None = None
        self._stderr = ""
        self._done = threading.Event()
        try:
            self.proc = subprocess.Popen(
                [sys.executable, "-m", "gr00t_lancedb._isolated", str(inp), str(self._out)],
                env=_child_env(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
        except BaseException:
            shutil.rmtree(self._dir, ignore_errors=True)
            raise
        threading.Thread(target=self._wait, name="gr00t-lance-iso-wait", daemon=True).start()

    def _wait(self) -> None:
        try:
            _, self._stderr = self.proc.communicate()
        except Exception as exc:  # noqa: BLE001 - reported through collect()
            self._stderr = f"{self._stderr}\n{exc!r}"
        finally:
            self.wall_s = time.perf_counter() - self.t0
            self._done.set()

    def kill(self) -> None:
        if self.proc.poll() is None:
            try:
                self.proc.kill()
            except OSError:
                pass
        shutil.rmtree(self._dir, ignore_errors=True)

    def collect(self, timeout: float | None) -> Any:
        """Wait for the child (``timeout`` counts from its start) and return its value."""
        t_wait0 = time.perf_counter()
        remaining = None if timeout is None else max(0.0, timeout - (t_wait0 - self.t0))
        try:
            if not self._done.wait(remaining):
                self.kill()
                raise IsolatedCallError(f"isolated call {self.func} timed out after {timeout:.0f} s")
            if not self._out.exists():
                raise IsolatedCallError(
                    f"isolated call {self.func} exited with code {self.proc.returncode} and no result.\n"
                    f"stderr (tail):\n{self._stderr[-4000:]}"
                )
            with open(self._out, "rb") as fh:
                res = pickle.load(fh)
        finally:
            shutil.rmtree(self._dir, ignore_errors=True)
        if not res.get("ok"):
            raise IsolatedCallError(f"isolated call {self.func} failed:\n{res.get('error', '')}")
        value = res["value"]
        if isinstance(value, dict):
            value.setdefault("_isolated_wall_s", self.wall_s)
            value.setdefault("_isolated_wait_s", time.perf_counter() - t_wait0)
        return value


_STARTED: dict[str, _Call] = {}
_STARTED_LOCK = threading.Lock()


def _kill_unclaimed() -> None:
    with _STARTED_LOCK:
        calls = list(_STARTED.values())
        _STARTED.clear()
    for call in calls:
        call.kill()


atexit.register(_kill_unclaimed)


def start_isolated(func: str, kwargs: dict[str, Any] | None = None) -> bool:
    """Start ``module:function(**kwargs)`` in a child now, without waiting for it.

    The next :func:`run_isolated` with the same ``func`` and equal ``kwargs`` picks the
    result up. Returns ``False`` if that call is already running. Children nobody picks up
    are killed when the interpreter exits.
    """
    key = call_key(func, kwargs)
    with _STARTED_LOCK:
        if key in _STARTED:
            return False
        _STARTED[key] = _Call(func, kwargs)
    return True


def is_started(func: str, kwargs: dict[str, Any] | None = None) -> bool:
    """Whether :func:`start_isolated` has a call with this identity waiting to be picked up."""
    with _STARTED_LOCK:
        return call_key(func, kwargs) in _STARTED


def run_isolated(func: str, kwargs: dict[str, Any] | None = None, *, timeout: float | None = 900.0) -> Any:
    """Call ``module:function(**kwargs)`` in a fresh interpreter and return its result.

    ``kwargs`` and the return value must be picklable (numpy arrays, dicts, bytes ...).
    If :func:`start_isolated` already started the same call, its result is used instead of
    starting a second child. When the value is a dict, two timings are attached so callers
    can report startup segments: ``"_isolated_wall_s"`` (how long the child ran) and
    ``"_isolated_wait_s"`` (how long this call actually blocked, which is close to zero when
    the child was started early enough).
    """
    with _STARTED_LOCK:
        call = _STARTED.pop(call_key(func, kwargs), None)
    if call is None:
        call = _Call(func, kwargs)
    return call.collect(timeout)


def _main(argv: list[str]) -> int:
    inp, out = Path(argv[0]), Path(argv[1])
    with open(inp, "rb") as fh:
        req = pickle.load(fh)
    try:
        mod_name, _, fn_name = req["func"].partition(":")
        fn = getattr(importlib.import_module(mod_name), fn_name)
        res = {"ok": True, "value": fn(**req["kwargs"])}
    except BaseException:  # noqa: BLE001 - report every failure to the parent
        res = {"ok": False, "error": traceback.format_exc()}
    tmp = out.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(res, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, out)
    return 0 if res["ok"] else 1


if __name__ == "__main__":  # pragma: no cover - exercised through run_isolated
    raise SystemExit(_main(sys.argv[1:]))
