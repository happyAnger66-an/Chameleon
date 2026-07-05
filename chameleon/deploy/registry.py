"""deploy backend 注册表 — export / build 分发的插件化机制。

作用：
    以 :class:`~chameleon.core.registry.Registry` 承载各架构的部署后端
    （pi05 / cosmos3 / reference / 未来的 X），把「某 backend 如何 export、如何
    build engine」封装成一个 :class:`DeployBackend` 策略对象。api 层只按
    ``deploy.backend`` 查表委派，新增架构无需修改编排代码（开闭原则）。

架构位置：
    部署层基础设施 — 与 compile / runtime / quantization / policy_runner 的
    Registry 分发保持同一范式。内置后端在 :mod:`chameleon.deploy.backends`
    以 import-time 副作用注册。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from chameleon.core.registry import Registry

if TYPE_CHECKING:
    from chameleon.config.schema import TaskConfig
    from chameleon.core.artifact import Artifact, Manifest


@runtime_checkable
class DeployBackend(Protocol):
    """一个部署后端的策略接口（export + build engine）。

    实现类应保持**轻量**（模块导入期不 import torch/diffusers 等重依赖），
    真正的重依赖在 ``export`` / ``build`` 方法体内 lazy import，以维持现有的
    「用到才加载」行为。
    """

    name: str
    """后端族名（如 ``pi05`` / ``cosmos3`` / ``reference``），也是主注册键。"""

    aliases: tuple[str, ...]
    """额外注册别名（如 ``pi05_openpi`` / ``cosmos3_diffusers``）。"""

    uses_dedicated_build: bool
    """True 表示 compile 走专用 ``build``（ONNX→engine）；False 表示走通用 run_compile。"""

    default_export_stages: tuple[str, ...]
    """未显式配置 ``task.export`` 时的默认 stage 列表（用于计划展示）。"""

    def export(self, task: "TaskConfig", manifest: "Manifest") -> dict[str, "Artifact"]:
        """导出各 stage 的 ONNX。"""

    def build(self, task: "TaskConfig", manifest: "Manifest") -> dict[str, "Artifact"]:
        """从已导出的 ONNX 构建 TRT engine。"""


DEPLOY_BACKEND_REGISTRY: Registry[str, DeployBackend] = Registry("deploy_backend")


def _norm(backend: str | None) -> str:
    return (backend or "reference").strip().lower()


def register_deploy_backend(backend: DeployBackend, *, override: bool = False) -> DeployBackend:
    """把一个后端按 ``name`` + 所有 ``aliases`` 注册到全局表。"""
    for key in (backend.name, *getattr(backend, "aliases", ())):
        DEPLOY_BACKEND_REGISTRY.register(key.strip().lower(), backend, override=override)
    return backend


def resolve_deploy_backend(backend: str | None) -> DeployBackend:
    """按 ``deploy.backend`` 取后端；未知则报错并列出可用项。"""
    _ensure_builtins()
    return DEPLOY_BACKEND_REGISTRY.get(_norm(backend))


def deploy_backend_or_none(backend: str | None) -> DeployBackend | None:
    """取后端，未注册返回 ``None``（供判定类调用，不抛异常）。"""
    _ensure_builtins()
    return DEPLOY_BACKEND_REGISTRY.get_or_none(_norm(backend))


def _ensure_builtins() -> None:
    """确保内置后端已注册（import 副作用），避免调用方漏 import。"""
    if len(DEPLOY_BACKEND_REGISTRY) == 0:
        from chameleon.deploy import backends  # noqa: F401
