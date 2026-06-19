"""加载 TensorRT build_cfg ``.py`` 文件。"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


def load_build_cfg(path: str | Path) -> dict[str, Any]:
    """Load ``build_cfg`` dict from a Python settings file."""
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"build_cfg file not found: {file_path}")

    spec = importlib.util.spec_from_file_location(f"chameleon_build_cfg_{file_path.stem}", file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load build_cfg module from {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cfg = getattr(module, "build_cfg", None)
    if not isinstance(cfg, dict):
        raise ValueError(f"{file_path} must define a dict named build_cfg.")
    return cfg
