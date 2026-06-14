"""Generic registry used across Chameleon subsystems.

All Chameleon plugins (platforms, architectures, models, quantization methods,
compiler backends, runtime backends, kernels) are discovered through a small
set of typed registries. Registration happens as an import-time side effect so
that ``import chameleon`` wires everything up (see :mod:`chameleon.__init__`).
"""

from __future__ import annotations

from typing import Generic, Iterator, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class Registry(Generic[K, V]):
    """A minimal, typed key -> value registry.

    Keys can be any hashable (commonly ``str`` or tuples such as
    ``(architecture, stage)``). The registry is intentionally tiny; richer
    behaviour (capability matching, fallbacks) lives in the subsystem that owns
    the registry.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._items: dict[K, V] = {}

    @property
    def name(self) -> str:
        return self._name

    def register(self, key: K, value: V, *, override: bool = False) -> V:
        if key in self._items and not override:
            raise KeyError(
                f"{self._name!r} registry already has an entry for {key!r}; "
                f"pass override=True to replace it."
            )
        self._items[key] = value
        return value

    def get(self, key: K) -> V:
        try:
            return self._items[key]
        except KeyError as exc:
            available = ", ".join(repr(k) for k in self._items)
            raise KeyError(
                f"No {self._name!r} registered for {key!r}. Available: [{available}]"
            ) from exc

    def get_or_none(self, key: K) -> V | None:
        return self._items.get(key)

    def __contains__(self, key: object) -> bool:
        return key in self._items

    def keys(self) -> list[K]:
        return list(self._items.keys())

    def items(self) -> Iterator[tuple[K, V]]:
        return iter(self._items.items())

    def values(self) -> Iterator[V]:
        return iter(self._items.values())

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"Registry({self._name!r}, {len(self._items)} entries)"
