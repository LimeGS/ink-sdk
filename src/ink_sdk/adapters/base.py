"""Adapter protocol, shared map conversion, and a deterministic dummy."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ..contracts import (
    InkMap,
    Issue,
    PreparedInput,
    RawOutput,
    RunContext,
    Segment,
)
from ..errors import MissingDependencyError, ValidationError
from ..prepare import prepare_input
from ..profiles import ModelProfile


class Adapter(Protocol):
    name: str

    def check(
        self, segment: Segment, profile: ModelProfile, ctx: RunContext | None = None
    ) -> list[Issue]: ...

    def prepare(
        self,
        segment: Segment,
        profile: ModelProfile,
        ctx: RunContext,
    ) -> PreparedInput: ...

    def run(
        self, prepared: PreparedInput, profile: ModelProfile, ctx: RunContext
    ) -> RawOutput: ...

    def to_maps(
        self, raw: RawOutput, prepared: PreparedInput, profile: ModelProfile
    ) -> dict[str, InkMap]: ...


class BaseAdapter:
    name = "base"

    def check(
        self, segment: Segment, profile: ModelProfile, ctx: RunContext | None = None
    ) -> list[Issue]:
        return []

    def prepare(
        self,
        segment: Segment,
        profile: ModelProfile,
        ctx: RunContext,
    ) -> PreparedInput:
        if segment.surface_volume is None:
            raise ValidationError("adapter got no prepared surface volume")
        scale = segment.pixel_um[0] if segment.pixel_um else None
        if scale is None:
            raise ValidationError("surface volume has no scale; pass --voxel-um")
        return prepare_input(
            segment.surface_volume,
            profile,
            source_um=scale,
            cache_dir=ctx.cache_dir,
            resample_from_um=ctx.resample_from_um,
        )

    def run(
        self, prepared: PreparedInput, profile: ModelProfile, ctx: RunContext
    ) -> RawOutput:
        raise NotImplementedError

    def to_maps(
        self, raw: RawOutput, prepared: PreparedInput, profile: ModelProfile
    ) -> dict[str, InkMap]:
        maps: dict[str, InkMap] = {}
        for direction, values in raw.maps.items():
            array = np.asarray(values, dtype=np.float32)
            if array.ndim != 2:
                raise ValidationError(f"{direction} map must be 2D, got {array.shape}")
            array = np.clip(array, 0.0, 1.0)
            maps[direction] = InkMap(
                array=array,
                valid=raw.valid,
                space="surface-plane",
                pixel_um=(float(prepared.pixel_um[1]), float(prepared.pixel_um[2])),
                origin_px=prepared.origin_px,
                direction=direction,
                model_id=profile.id,
                checkpoint_sha256=profile.checkpoint.sha256,
                no_ink_floor=float(profile.output.no_ink_floor),
                quantisation=dict(raw.quantisation),
                receipt={**raw.receipt, "direction": direction},
            )
        return maps


class DummyAdapter(BaseAdapter):
    name = "dummy"

    def run(
        self, prepared: PreparedInput, profile: ModelProfile, ctx: RunContext
    ) -> RawOutput:
        height, width = prepared.array.shape[1:]
        values = np.linspace(0.0, 1.0, height * width, dtype=np.float32).reshape(height, width)
        return RawOutput(
            maps={"forward": values},
            valid=np.ones((height, width), dtype=bool),
            quantisation={"source_dtype": "float32", "levels": None, "scale": 1.0},
            receipt={"adapter": "dummy", "shape_yx": [height, width]},
        )


def require_torch() -> None:
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise MissingDependencyError(
            "this adapter needs torch; install ink-sdk[canonical] or ink-sdk[hecate]"
        ) from exc


def resolve_device(requested: str | None, torch_module=None) -> str:
    if torch_module is None:
        import torch as torch_module  # type: ignore[no-redef]
    value = (requested or "auto").lower()
    if value == "auto":
        if torch_module.cuda.is_available():
            return "cuda"
        return "cpu"
    if value.startswith("cuda") and not torch_module.cuda.is_available():
        raise ValidationError("CUDA was requested but is not available")
    if value == "mps" and not getattr(torch_module.backends, "mps", None):
        raise ValidationError("MPS was requested but is not available")
    return value


def source_scale_issue(segment: Segment, profile: ModelProfile, ctx: RunContext | None = None) -> Issue | None:
    if segment.pixel_um is None:
        return Issue("error", "missing-scale", "surface volume has no scale; pass --voxel-um")
    from ..prepare import plan_scale

    try:
        plan_scale(
            segment.pixel_um[0],
            profile,
            resample_from_um=ctx.resample_from_um if ctx else None,
        )
    except Exception as exc:
        return Issue("error", "incompatible-scale", str(exc))
    return None


def stats(array: np.ndarray) -> dict:
    finite = np.asarray(array)[np.isfinite(array)]
    if not finite.size:
        return {"p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0, "valid_pixels": 0}
    return {
        "p50": float(np.percentile(finite, 50)),
        "p90": float(np.percentile(finite, 90)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(finite.max()),
        "valid_pixels": int(finite.size),
    }
