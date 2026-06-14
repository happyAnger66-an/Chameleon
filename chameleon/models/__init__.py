"""Model adapters."""

from chameleon.models.base import (
    MODEL_REGISTRY,
    ModelAdapter,
    get_model_adapter,
    list_models,
    register_model,
)

# Import-time registration of built-in model adapters.
from chameleon.models import pi05  # noqa: F401,E402

__all__ = [
    "MODEL_REGISTRY",
    "ModelAdapter",
    "get_model_adapter",
    "list_models",
    "register_model",
]
