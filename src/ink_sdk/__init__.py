"""Standalone Vesuvius ink inference SDK.

The package is inference-only. It does not train, fine-tune, or score a model.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .api import InkSDK  # noqa: E402
from .errors import (  # noqa: E402
    DegenerateMapError,
    IncompatibleScaleError,
    InkSDKError,
    MissingDependencyError,
    ModelIntegrityError,
    RenderError,
    ValidationError,
)

__all__ = [
    "InkSDK",
    "InkSDKError",
    "ValidationError",
    "IncompatibleScaleError",
    "ModelIntegrityError",
    "MissingDependencyError",
    "RenderError",
    "DegenerateMapError",
    "__version__",
]
