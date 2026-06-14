"""PyTorch 参考运行时 — 直接在 PyTorch 中执行 stage 模块。

作用：
    实现 PyTorchEngine（将 nn.Module 包装为统一 run 接口，按 dict 值顺序
    positional 传参）和 PyTorchRuntime（从 reference Artifact.payload
    加载模块）。无需编译 engine 即可验证全链路。

架构位置：
    运行时层 — MVP 功能后端，被 InferenceSession 在 reference 路径或
    stage_runtimes["*"]="pytorch" 时使用。验证编排、KV handoff、去噪环
    与 stage 级后端混用。
"""

from __future__ import annotations

from typing import Any

import torch

from chameleon.core.artifact import Artifact
from chameleon.core.context import RunContext
from chameleon.runtime.base import Engine, RuntimeBackend, register_runtime


class PyTorchEngine(Engine):
    """Wraps an ``nn.Module`` so it satisfies the unified ``run`` contract.

    Inputs are passed positionally in dict-iteration order, so the orchestrator
    must build the ``inputs`` dict in the module's argument order.
    """

    def __init__(self, module: Any, stage: str | None, device: str) -> None:
        self.module = module
        self.stage = stage
        self.device = device

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        args = [
            v.to(self.device) if isinstance(v, torch.Tensor) else v
            for v in inputs.values()
        ]
        with torch.no_grad():
            out = self.module(*args)
        return {"output": out}


class PyTorchRuntime(RuntimeBackend):
    name = "pytorch"

    def load(self, artifact: Artifact, ctx: RunContext) -> Engine:
        if artifact.payload is None:
            raise ValueError(
                "PyTorchRuntime expects a reference artifact carrying an nn.Module "
                "in artifact.payload."
            )
        module = artifact.payload
        device = ctx.torch_device
        if hasattr(module, "to"):
            module = module.to(device)
        if hasattr(module, "eval"):
            module.eval()
        return PyTorchEngine(module, artifact.stage, device)


register_runtime(PyTorchRuntime(), override=True)
