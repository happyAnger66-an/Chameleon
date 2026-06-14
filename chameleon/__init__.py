"""Chameleon: a cross-platform edge VLA quantization / compile / inference toolkit.

Importing :mod:`chameleon` triggers import-time registration of all built-in
plugins (platforms, architectures, models, quantization methods, graph capture,
compiler backends, kernels, runtimes, orchestrators) so the registries are
populated and ready to use.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Order matters: low-level subsystems first, then those that depend on them.
from chameleon import core  # noqa: F401
from chameleon import architectures  # noqa: F401
from chameleon import frontend  # noqa: F401
from chameleon import quantization  # noqa: F401
from chameleon import kernels  # noqa: F401
from chameleon import compile  # noqa: F401
from chameleon import models  # noqa: F401
from chameleon import runtime  # noqa: F401

from chameleon.config.schema import TaskConfig  # noqa: E402

__all__ = ["TaskConfig", "__version__"]
