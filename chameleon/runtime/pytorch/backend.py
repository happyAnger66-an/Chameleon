"""PyTorch reference runtime.

This is the functional MVP backend: it runs the (reference or real) pi05 stage
modules directly in PyTorch, validating the entire pipeline -- orchestration, KV
handoff, the flow-matching denoise loop and stage-level backend mixing -- without
any compiled engine. A reference :class:`Artifact` carries the ``nn.Module`` in
its ``payload``.
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
