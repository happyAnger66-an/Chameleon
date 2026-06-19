"""deploy.pi05.memory 单元测试。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from chameleon.deploy.pi05.memory import release_export_cuda_memory


def test_release_export_cuda_memory_calls_cpu_and_empty_cache() -> None:
    model = MagicMock()
    with patch("chameleon.deploy.pi05.memory.torch.cuda.is_available", return_value=True):
        with patch("chameleon.deploy.pi05.memory.torch.cuda.empty_cache") as empty:
            with patch("chameleon.deploy.pi05.memory.torch.cuda.synchronize") as sync:
                release_export_cuda_memory(model)
    model.cpu.assert_called_once()
    empty.assert_called_once()
    sync.assert_called_once()
