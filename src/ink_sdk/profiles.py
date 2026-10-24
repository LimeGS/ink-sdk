"""Validated model profiles.

A profile is a frozen declaration. It never contains a default checkpoint: a
missing digest is a configuration error, not an experimental mode.
"""

from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .errors import ModelIntegrityError, ValidationError


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Source(_Strict):
    repo: str
    revision: str
    upstream: str
    upstream_revision: str
    url: str
    license: str


class Checkpoint(_Strict):
    path: str
    upstream_path: str | None = None
    sha256: str
    bytes: int

    @field_validator("path")
    @classmethod
    def _relative_path(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("checkpoint.path must stay under the models directory")
        return value

    @field_validator("sha256")
    @classmethod
    def _sha(cls, value: str) -> str:
        value = value.lower()
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("checkpoint.sha256 must be 64 lowercase hex characters")
        return value

    @field_validator("bytes")
    @classmethod
    def _bytes(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("checkpoint.bytes must be positive")
        return value


class Input(_Strict):
    kind: str = "surface-volume"
    family: str
    scale_um: float
    scale_tolerance_um: float
    depth: int
    normalization: str
    tile: int | None = None
    overlap: float = 0.5
    stride: int | None = None
    min_valid_ratio: float = 0.6
    divisor: float | None = None
    clip: float | None = None

    @field_validator("scale_um", "scale_tolerance_um")
    @classmethod
    def _positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("scale values must be positive")
        return value

    @field_validator("depth")
    @classmethod
    def _depth(cls, value: int) -> int:
        if value < 1:
            raise ValueError("input.depth must be at least 1")
        return value

    @field_validator("overlap")
    @classmethod
    def _overlap(cls, value: float) -> float:
        if not 0 <= value < 1:
            raise ValueError("input.overlap must be in [0, 1)")
        return value


class Execution(_Strict):
    direction: str = "forward"
    blend: str = "hann"
    batch: int = 1
    amp: str = "bf16"
    tta: bool = False

    @field_validator("direction")
    @classmethod
    def _direction(cls, value: str) -> str:
        if value not in {"forward", "reverse", "both"}:
            raise ValueError("execution.direction must be forward, reverse or both")
        return value

    @field_validator("batch")
    @classmethod
    def _batch(cls, value: int) -> int:
        if value < 1:
            raise ValueError("execution.batch must be at least 1")
        return value


class Output(_Strict):
    primary: str = "forward"
    no_ink_floor: float = 0.0
    floor_provenance: str = "not-declared"

    @field_validator("primary")
    @classmethod
    def _primary(cls, value: str) -> str:
        if value not in {"forward", "reverse", "mean"}:
            raise ValueError("output.primary must be forward, reverse or mean")
        return value

    @field_validator("no_ink_floor")
    @classmethod
    def _floor(cls, value: float) -> float:
        if not 0 <= value < 1:
            raise ValueError("output.no_ink_floor must be in [0, 1)")
        return value


class ModelProfile(_Strict):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: str = Field(default="ink-sdk.profile.v1", alias="schema")
    id: str
    adapter: str
    family: str
    source: Source
    checkpoint: Checkpoint
    input: Input
    execution: Execution = Field(default_factory=Execution)
    output: Output = Field(default_factory=Output)

    @model_validator(mode="after")
    def _adapter_matches_family(self) -> "ModelProfile":
        allowed = {
            "ink9um": "surface-volume-2d",
            "canonical2um": "surface-volume-2d",
            "hecate": "surface-volume-3d2d",
        }
        if self.adapter not in allowed:
            raise ValueError(f"unknown adapter {self.adapter!r}")
        if self.family != allowed[self.adapter]:
            raise ValueError(
                f"{self.id}: adapter {self.adapter} declares family {allowed[self.adapter]}, "
                f"not {self.family!r}"
            )
        return self

    def checkpoint_path(self, models_dir: Path) -> Path:
        return Path(models_dir) / self.checkpoint.path

    @property
    def upstream_checkpoint_path(self) -> str:
        return self.checkpoint.upstream_path or self.checkpoint.path

    def verify_checkpoint(self, models_dir: Path) -> Path:
        path = self.checkpoint_path(models_dir)
        if not path.is_file():
            raise ModelIntegrityError(
                f"checkpoint not found: {path}. Download {self.source.repo}/"
                f"{self.checkpoint.path} or set INK_SDK_MODELS."
            )
        size = path.stat().st_size
        if size != self.checkpoint.bytes:
            raise ModelIntegrityError(
                f"checkpoint size {size} is not the pinned {self.checkpoint.bytes}: {path}"
            )
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        actual = digest.hexdigest()
        if actual != self.checkpoint.sha256:
            raise ModelIntegrityError(
                f"checkpoint digest {actual} is not the pinned {self.checkpoint.sha256}: {path}"
            )
        return path


def load_profile(path: Path) -> ModelProfile:
    path = Path(path)
    try:
        raw: dict[str, Any] = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValidationError(f"cannot read profile {path}: {exc}") from exc
    try:
        return ModelProfile.model_validate(raw)
    except Exception as exc:
        raise ValidationError(f"invalid profile {path}: {exc}") from exc
