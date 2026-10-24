"""Renderer protocol and the dependency-free renderer used by tests."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

import numpy as np

from ..contracts import Issue, RenderedVolume, RenderSpec, RunContext, Segment
from ..io import write_zarr
from ..profiles import ModelProfile


class Renderer(Protocol):
    name: str

    def check(self, segment: Segment, profile: ModelProfile) -> list[Issue]: ...

    def render(
        self, segment: Segment, spec: RenderSpec, profile: ModelProfile, ctx: RunContext
    ) -> RenderedVolume: ...


class DummyRenderer:
    """Deterministic synthetic renderer. No external binary, no CT read."""

    name = "dummy"

    def check(self, segment: Segment, profile: ModelProfile) -> list[Issue]:
        return []

    def render(
        self, segment: Segment, spec: RenderSpec, profile: ModelProfile, ctx: RunContext
    ) -> RenderedVolume:
        depth = spec.num_slices or profile.input.depth
        height = width = 16
        if segment.surface_mesh is not None:
            try:
                import tifffile

                x = tifffile.imread(Path(segment.surface_mesh) / "x.tif")
                height, width = int(x.shape[0]), int(x.shape[1])
            except Exception:
                pass
        z, y, x = np.meshgrid(
            np.arange(depth, dtype=np.float32),
            np.arange(height, dtype=np.float32),
            np.arange(width, dtype=np.float32),
            indexing="ij",
        )
        array = np.round((z + y + x) * (255.0 / max(depth + height + width - 3, 1))).astype(np.uint8)
        scale = float(profile.input.scale_um)
        path = Path(ctx.workdir or ".") / "render"
        write_zarr(path, array, attrs={"pixel_um": [scale, scale, scale]}, overwrite=True)
        return RenderedVolume(
            path=path,
            shape_zyx=array.shape,
            pixel_um=(scale, scale, scale),
            orientation={"source": "dummy", "traspuesta": False, "filas_invertidas": False, "columnas_invertidas": False},
            receipt={"renderer": "dummy", "shape_zyx": list(array.shape), "cache_hit": False},
            cache_hit=False,
        )


def render_key(*parts: object) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:24]
