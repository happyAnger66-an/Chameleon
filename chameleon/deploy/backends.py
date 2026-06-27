"""deploy backend 判定。"""

from __future__ import annotations

_PI05_BACKENDS = frozenset({"pi05", "pi05_openpi"})
_COSMOS3_BACKENDS = frozenset({"cosmos3", "cosmos3_diffusers"})


def is_pi05_deploy_backend(backend: str | None) -> bool:
    return (backend or "").strip().lower() in _PI05_BACKENDS


def is_cosmos3_deploy_backend(backend: str | None) -> bool:
    return (backend or "").strip().lower() in _COSMOS3_BACKENDS
