"""泛型注册表 — 全框架插件发现机制的基础设施。

作用：
    提供 typed Registry[K, V]，支持 register / get / keys 等操作。
    所有子系统的插件（平台、架构、模型、量化方法、编译/运行时后端、
    自定义算子）均通过 import-time 副作用注册到此机制。

架构位置：
    基础设施层 — 被 core/platform、architectures、models、quantization、
    compile、runtime、kernels 等各包的 registry 模块复用。
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
