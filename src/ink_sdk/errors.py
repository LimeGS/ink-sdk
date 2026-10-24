"""Errors the SDK deliberately raises instead of guessing."""

from __future__ import annotations


class InkSDKError(RuntimeError):
    """Base class for actionable SDK failures."""


class ValidationError(InkSDKError):
    """The caller's input or options do not satisfy the contract."""


class IncompatibleScaleError(ValidationError):
    """The source scale cannot reach the model scale without inventing data."""


class ModelIntegrityError(InkSDKError):
    """A checkpoint is missing or its SHA-256 is not the pinned one."""


class MissingDependencyError(InkSDKError):
    """An optional runtime dependency is not installed."""


class RenderError(InkSDKError):
    """The surface render failed or produced an invalid volume."""


class DegenerateMapError(InkSDKError):
    """The liveness gate found a map that carries no decision."""

    def __init__(self, report: dict):
        self.report = report
        super().__init__(
            f"map is {report.get('verdict', 'DEGENERATE')}: "
            f"{report.get('reason', 'no decision')}"
        )
