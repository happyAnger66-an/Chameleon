"""TensorRT compat 单元测试。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from chameleon.compile.tensorrt.compat import (
    apply_precision_constraints_policy,
    create_onnx_network,
    describe_network_flags,
    network_creation_flags,
    set_builder_flag_if_present,
)


class _Flags:
    EXPLICIT_BATCH = 0
    STRONGLY_TYPED = 1


def test_network_creation_flags_trt10_no_explicit_batch() -> None:
    fake_trt = MagicMock()
    fake_trt.NetworkDefinitionCreationFlag = _Flags
    del fake_trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH  # type: ignore[attr-defined]

    with patch.dict("sys.modules", {"tensorrt": fake_trt}):
        assert network_creation_flags(strongly_typed=False) is None
        assert network_creation_flags(strongly_typed=True) == (1 << 1)


def test_create_onnx_network_trt10_calls_no_arg_create() -> None:
    fake_trt = MagicMock()
    fake_trt.NetworkDefinitionCreationFlag = type("_F", (), {"STRONGLY_TYPED": 1})()
    builder = MagicMock()
    with patch.dict("sys.modules", {"tensorrt": fake_trt}):
        create_onnx_network(builder, strongly_typed=False)
    builder.create_network.assert_called_once_with()


def test_apply_precision_constraints_missing_flag() -> None:
    fake_trt = MagicMock()
    fake_trt.BuilderFlag = type("_BF", (), {"FP16": 1})()
    config = MagicMock()
    with patch.dict("sys.modules", {"tensorrt": fake_trt}):
        apply_precision_constraints_policy(config, "prefer")
    config.set_flag.assert_not_called()


def test_set_builder_flag_if_present() -> None:
    fake_trt = MagicMock()
    fake_trt.BuilderFlag = type("_BF", (), {"BF16": 2})()
    config = MagicMock()
    with patch.dict("sys.modules", {"tensorrt": fake_trt}):
        assert set_builder_flag_if_present(config, "BF16") is True
        assert set_builder_flag_if_present(config, "PREFER_PRECISION_CONSTRAINTS") is False
    config.set_flag.assert_called_once()
