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
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any

_PKG_PARENT = str(Path(__file__).resolve().parents[1])


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
    return env


def run_isolated(func: str, kwargs: dict[str, Any] | None = None, *, timeout: float | None = 900.0) -> Any:
    """Call ``module:function(**kwargs)`` in a fresh interpreter and return its result.

    ``kwargs`` and the return value must be picklable (numpy arrays, dicts, bytes ...).
    The child's wall time is attached to the returned value when it is a dict
    (``"_isolated_wall_s"``) so callers can report it as a startup segment.
    """
    with tempfile.TemporaryDirectory(prefix="gr00t-lance-iso-") as td:
        inp = Path(td) / "in.pkl"
        out = Path(td) / "out.pkl"
        with open(inp, "wb") as fh:
            pickle.dump({"func": func, "kwargs": kwargs or {}}, fh, protocol=pickle.HIGHEST_PROTOCOL)
        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, "-m", "gr00t_lancedb._isolated", str(inp), str(out)],
            env=_child_env(),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        wall = time.perf_counter() - t0
        if not out.exists():
            raise IsolatedCallError(
                f"isolated call {func} exited with code {proc.returncode} and no result.\n"
                f"stderr (tail):\n{proc.stderr[-4000:]}"
            )
        with open(out, "rb") as fh:
            res = pickle.load(fh)
    if not res.get("ok"):
        raise IsolatedCallError(f"isolated call {func} failed:\n{res.get('error', '')}")
    value = res["value"]
    if isinstance(value, dict):
        value.setdefault("_isolated_wall_s", wall)
    return value


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
