"""内置 deploy 后端定义与注册（pi05 / cosmos3 / reference）。

每个后端是一个轻量策略对象：模块导入期不触碰 torch/diffusers，重依赖在
``export`` / ``build`` 方法内 lazy import。import 本模块即完成注册（供
:mod:`chameleon.deploy.registry` 的 ``_ensure_builtins`` 触发）。

历史兼容：保留 ``is_pi05_deploy_backend`` / ``is_cosmos3_deploy_backend`` 判定函数，
现改为以注册表为唯一事实来源（按后端族名匹配），旧调用方无需改动。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from chameleon.deploy.registry import (
    DEPLOY_BACKEND_REGISTRY,
    deploy_backend_or_none,
    register_deploy_backend,
)

if TYPE_CHECKING:
    from chameleon.config.schema import TaskConfig
    from chameleon.core.artifact import Artifact, Manifest


class _Pi05DeployBackend:
    name = "pi05"
    aliases = ("pi05_openpi",)
    uses_dedicated_build = True
    default_export_stages = ("vit", "llm", "expert", "denoise")

    def export(self, task: "TaskConfig", manifest: "Manifest") -> dict[str, "Artifact"]:
        from chameleon.deploy.pi05_openpi import run_pi05_export

        return run_pi05_export(task, manifest)

    def build(self, task: "TaskConfig", manifest: "Manifest") -> dict[str, "Artifact"]:
        from chameleon.deploy.pi05_openpi import run_pi05_build

        return run_pi05_build(task, manifest)


class _Cosmos3DeployBackend:
    name = "cosmos3"
    aliases = ("cosmos3_diffusers",)
    uses_dedicated_build = True
    default_export_stages = ("vae_encode", "text_embed", "dit", "vae_decode")

    def export(self, task: "TaskConfig", manifest: "Manifest") -> dict[str, "Artifact"]:
        from chameleon.deploy.cosmos3_diffusers import run_cosmos3_export

        return run_cosmos3_export(task, manifest)

    def build(self, task: "TaskConfig", manifest: "Manifest") -> dict[str, "Artifact"]:
        from chameleon.deploy.cosmos3_diffusers import run_cosmos3_build

        return run_cosmos3_build(task, manifest)


class _ReferenceDeployBackend:
    """参考适配器路径：export 由 run_compile（capture→ONNX）承担，无专用 build。"""

    name = "reference"
    aliases = ()
    uses_dedicated_build = False
    default_export_stages = ("vit", "llm", "action_expert")

    def export(self, task: "TaskConfig", manifest: "Manifest") -> dict[str, "Artifact"]:
        raise NotImplementedError(
            "reference export is handled inside run_compile (capture -> ONNX). "
            "Use actions: [compile] with deploy.backend=reference, or set "
            "deploy.backend=pi05|cosmos3 for real ONNX export."
        )

    def build(self, task: "TaskConfig", manifest: "Manifest") -> dict[str, "Artifact"]:
        raise ValueError(
            "run_deploy_build requires a dedicated deploy backend (e.g. pi05|cosmos3); "
            "the reference adapter path uses run_compile."
        )


register_deploy_backend(_Pi05DeployBackend())
register_deploy_backend(_Cosmos3DeployBackend())
register_deploy_backend(_ReferenceDeployBackend())


def _family(backend: str | None) -> str | None:
    be = deploy_backend_or_none(backend)
    return be.name if be is not None else None


def is_pi05_deploy_backend(backend: str | None) -> bool:
    return _family(backend) == "pi05"


def is_cosmos3_deploy_backend(backend: str | None) -> bool:
    return _family(backend) == "cosmos3"


def is_dedicated_deploy_backend(backend: str | None) -> bool:
    """是否为带专用 export/build 的后端（compile 走 run_deploy_build）。"""
    be = deploy_backend_or_none(backend)
    return bool(be is not None and be.uses_dedicated_build)


__all__ = [
    "DEPLOY_BACKEND_REGISTRY",
    "is_pi05_deploy_backend",
    "is_cosmos3_deploy_backend",
    "is_dedicated_deploy_backend",
]
