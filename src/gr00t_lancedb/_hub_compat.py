# SPDX-License-Identifier: Apache-2.0
"""Let ``lerobot`` / ``lerobot-lancedb`` import inside NVIDIA Isaac-GR00T's environment.

Isaac-GR00T pins ``transformers==4.57.x``, which requires ``huggingface_hub<1.0``.
``lerobot>=0.7`` imports three names that only exist in ``huggingface_hub>=1.x``:

* ``download_bucket_files`` / ``sync_bucket`` — Hugging Face Storage Bucket helpers
* ``is_offline_mode`` — a trivial wrapper around ``HF_HUB_OFFLINE``

None of them are on the GR00T training path (we read local / ``s3://`` / ``gs://`` /
``hf://`` Lance roots). We add stand-ins only when the installed hub lacks them, so a
modern hub is never touched. The bucket stand-ins raise a clear error if ever called.
"""

from __future__ import annotations

_BUCKET_MSG = (
    "huggingface_hub.{name} needs huggingface_hub>=1.x, but this environment has an older "
    "version (NVIDIA Isaac-GR00T pins transformers 4.57 -> huggingface_hub<1.0). Hugging Face "
    "Storage Buckets are therefore unavailable here; use a local, s3://, gs:// or hf:// "
    "Lance dataset root instead."
)


def ensure_hub_compat() -> list[str]:
    """Install missing ``huggingface_hub`` attributes; returns the names that were patched."""
    try:
        import huggingface_hub as hh
    except ImportError:  # pragma: no cover - lerobot will raise its own error
        return []

    patched: list[str] = []

    def _unsupported(name: str):
        def _fn(*_args, **_kwargs):
            raise RuntimeError(_BUCKET_MSG.format(name=name))

        _fn.__name__ = name
        return _fn

    for name in ("download_bucket_files", "sync_bucket"):
        if not hasattr(hh, name):
            setattr(hh, name, _unsupported(name))
            patched.append(name)

    if not hasattr(hh, "is_offline_mode"):
        from huggingface_hub import constants

        def is_offline_mode() -> bool:
            return bool(getattr(constants, "HF_HUB_OFFLINE", False))

        hh.is_offline_mode = is_offline_mode
        patched.append("is_offline_mode")

    # hub>=1.x re-exports its HTTP client as ``huggingface_hub.utils.httpx``; lerobot's
    # hf:// range-fetch module imports it at module load. Point it at the real ``httpx``
    # package (hf:// streaming itself still expects hub>=1.x and is not used here).
    import huggingface_hub.utils as hh_utils

    if not hasattr(hh_utils, "httpx"):
        try:
            import httpx
        except ImportError:  # pragma: no cover - only hf:// streaming needs it
            pass
        else:
            hh_utils.httpx = httpx
            patched.append("utils.httpx")

    return patched


__all__ = ["ensure_hub_compat"]
