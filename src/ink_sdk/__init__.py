"""Standalone Vesuvius ink inference SDK.

The package is inference-only. It does not train, fine-tune, or score a model.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .api import InkSDK
from .errors import (
    DegenerateMapError,
    IncompatibleScaleError,
    InkSDKError,
    MissingDependencyError,
    ModelIntegrityError,
    RenderError,
    ValidationError,
)

__all__ = [
    "DegenerateMapError",
    "IncompatibleScaleError",
    "InkSDK",
    "InkSDKError",
    "MissingDependencyError",
    "ModelIntegrityError",
    "RenderError",
    "ValidationError",
    "__version__",
]
