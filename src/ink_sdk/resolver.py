"""Turn CLI/API inputs into a Segment and, when needed, a render."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import numpy as np
import tifffile
import zarr

from .contracts import RenderSpec, RunContext, Segment
from .errors import IncompatibleScaleError, ValidationError
from .io import (
    ct_preflight,
    open_volume_array,
    ordered_tiff_stack,
    pixel_um_from_sidecar,
    pixel_um_from_zarr,
)
from .prepare import plan_scale
from .profiles import ModelProfile


def _surface_metadata(path: Path, explicit_um: tuple[float, float, float] | None) -> tuple[str, tuple[int, ...], str, tuple[float, float, float] | None]:
    path = Path(path)
    if path.suffix == ".npy":
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        return "npy", tuple(array.shape), str(array.dtype), explicit_um or pixel_um_from_sidecar(path.parent)
    if path.suffix.lower() in {".tif", ".tiff"}:
        array = tifffile.imread(path)
        return "tiff-stack", tuple(array.shape), str(array.dtype), explicit_um or pixel_um_from_sidecar(path.parent)
    if path.is_dir():
        if (path / "0" / ".zarray").is_file() or (path / "0" / "zarr.json").is_file() or (path / "zarr.json").is_file():
            group = zarr.open_group(str(path), mode="r")
            array = group["0"] if not hasattr(group, "shape") else group
            scale = explicit_um or pixel_um_from_zarr(path) or pixel_um_from_sidecar(path)
            return "zarr", tuple(array.shape), str(array.dtype), scale
        tiffs = ordered_tiff_stack(path) if list(path.glob("*.tif")) or list(path.glob("*.tiff")) else []
        if tiffs:
            first = tifffile.imread(tiffs[0])
            return "tiff-stack", (len(tiffs), *first.shape), str(first.dtype), explicit_um or pixel_um_from_sidecar(path)
        if (path / "surface-volumes").is_dir():
            raise ValidationError(f"{path} is a segment container; pass a surface volume or let the SDK choose one")
    raise ValidationError(f"unsupported surface volume: {path}")


def candidate_volumes(path: Path) -> list[Path]:
    path = Path(path)
    if path.is_dir() and (path / "surface-volumes").is_dir():
        path = path / "surface-volumes"
    if not path.is_dir() or path.name != "surface-volumes":
        return [path]
    candidates = []
    for child in sorted(path.iterdir()):
        if child.suffix in {".zarr", ".npy", ".tif", ".tiff"}:
            candidates.append(child)
        elif child.is_dir() and (any(child.glob("*.tif")) or (child / "0" / ".zarray").is_file()):
            candidates.append(child)
    if not candidates:
        raise ValidationError(f"no surface volumes found in {path}")
    return candidates


def choose_volume(path: Path, profile: ModelProfile, explicit_um, ctx: RunContext) -> Path:
    volumes = candidate_volumes(path)
    if len(volumes) == 1:
        return volumes[0]
    compatible: list[tuple[Path, tuple]] = []
    rejected: list[str] = []
    for candidate in volumes:
        try:
            _, shape, dtype, scale = _surface_metadata(candidate, explicit_um)
            if dtype != "uint8":
                raise IncompatibleScaleError(f"dtype {dtype}, expected uint8")
            if scale is None:
                raise IncompatibleScaleError("no pixel_um in metadata; pass --voxel-um")
            plan_scale(scale[0], profile, resample_from_um=ctx.resample_from_um)
            compatible.append((candidate, shape))
        except Exception as exc:
            rejected.append(f"  {candidate.name} ({exc})")
    if len(compatible) == 1:
        return compatible[0][0]
    if not compatible:
        raise IncompatibleScaleError(
            f"no surface volume compatible with {profile.id} ({profile.input.scale_um:g} ± "
            f"{profile.input.scale_tolerance_um:g} um):\n" + "\n".join(rejected)
        )
    raise ValidationError(
        "several surface volumes are compatible; pass --surface-volume explicitly:\n"
        + "\n".join(f"  {p} {s}" for p, s in compatible)
    )


def build_segment(
    *,
    profile: ModelProfile,
    ctx: RunContext,
    surface_volume: Path | None = None,
    tifxyz: Path | None = None,
    ct: Path | None = None,
    voxel_um: float | None = None,
    pixel_um: tuple[float, float, float] | None = None,
    force_render: bool = False,
    segment_id: str | None = None,
) -> Segment:
    explicit = pixel_um
    if voxel_um is not None:
        explicit = (float(voxel_um),) * 3
    chosen = None
    if surface_volume is not None and not force_render:
        chosen = choose_volume(Path(surface_volume), profile, explicit, ctx)
    if chosen is not None:
        fmt, shape, dtype, scale = _surface_metadata(chosen, explicit)
        if dtype != "uint8":
            raise ValidationError(f"surface volume must be uint8, got {dtype}: {chosen}")
        if scale is None:
            raise ValidationError(
                f"surface volume has no pixel_um in metadata: {chosen}. Pass --voxel-um."
            )
        return Segment(
            id=segment_id or chosen.stem,
            root=chosen,
            surface_volume=chosen,
            format=fmt,
            surface_mesh=Path(tifxyz) if tifxyz else None,
            ct=Path(ct) if ct else None,
            pixel_um=scale,
            depth=int(shape[0]) if shape else None,
            dtype=dtype,
            direction="unknown",
        )

    if tifxyz is None or ct is None:
        raise ValidationError(
            "no surface volume to use. Pass --surface-volume, or --tifxyz and --ct "
            "(local OME-Zarr uint8) so the SDK can render."
        )
    mesh = Path(tifxyz)
    ct_path = Path(ct)
    pre = ct_preflight(ct_path)
    scale = explicit or pre["pixel_um"] or pixel_um_from_zarr(ct_path) or pixel_um_from_sidecar(ct_path)
    if scale is None:
        raise ValidationError(
            f"CT {ct_path} has no pixel_um/voxel_um in metadata. Pass --voxel-um; "
            "the SDK does not guess a scale."
        )
    render = RenderSpec(
        num_slices=None,
        orientation=ctx.normal,
        umbilicus=ctx.umbilicus,
        roi=ctx.roi,
        renderer=ctx.renderer,
    )
    return Segment(
        id=segment_id or mesh.name or "segment",
        root=mesh,
        surface_volume=None,
        format="tifxyz",
        surface_mesh=mesh,
        ct=ct_path,
        pixel_um=scale,
        depth=pre["shape_zyx"][0],
        dtype=pre["dtype"],
        direction="unknown",
        render=render,
    )


def _derived_slices(segment: Segment, profile: ModelProfile) -> int:
    if segment.depth == profile.input.depth:
        return profile.input.depth
    scale = segment.pixel_um[0] if segment.pixel_um else None
    if scale and profile.input.scale_um > 5 and abs(scale - 2.4) <= 0.15:
        return profile.input.depth * 4
    return profile.input.depth


def with_render_spec(segment: Segment, profile: ModelProfile, ctx: RunContext) -> Segment:
    if segment.surface_volume is not None:
        return segment
    spec = segment.render or RenderSpec()
    spec = dataclasses.replace(
        spec,
        num_slices=spec.num_slices or _derived_slices(segment, profile),
        orientation=ctx.normal if ctx.normal != "auto" else spec.orientation,
        umbilicus=ctx.umbilicus or spec.umbilicus,
        roi=ctx.roi if ctx.roi is not None else spec.roi,
        renderer=ctx.renderer or spec.renderer,
    )
    return dataclasses.replace(segment, render=spec)


def inspect_plan(
    segment: Segment,
    profile: ModelProfile,
    ctx: RunContext,
) -> dict:
    segment = with_render_spec(segment, profile, ctx)
    scale = segment.pixel_um[0] if segment.pixel_um else None
    plan = None
    if scale is not None:
        plan = plan_scale(scale, profile, resample_from_um=ctx.resample_from_um)
    render = None
    if segment.surface_volume is None and segment.render is not None:
        render = {
            "renderer": str(ctx.renderer or segment.render.renderer or os.environ.get("INK_SDK_RENDERER") or "vc_render_tifxyz"),
            "num_slices": segment.render.num_slices,
            "slice_step": segment.render.slice_step,
            "scale": segment.render.scale,
            "orientation": segment.render.orientation,
            "roi": segment.render.roi,
            "application": "native render; ROI is applied after scaling in prepare",
        }
    return {
        "profile": profile.id,
        "adapter": profile.adapter,
        "segment": {
            "id": segment.id,
            "surface_volume": str(segment.surface_volume) if segment.surface_volume else None,
            "surface_mesh": str(segment.surface_mesh) if segment.surface_mesh else None,
            "ct": str(segment.ct) if segment.ct else None,
            "dtype": segment.dtype,
            "depth": segment.depth,
            "pixel_um": segment.pixel_um,
        },
        "scale_plan": plan,
        "render": render,
        "checkpoint": {
            "path": str(profile.checkpoint_path(ctx.models_dir or Path.home() / ".cache" / "ink-sdk" / "models")),
            "sha256": profile.checkpoint.sha256,
            "bytes": profile.checkpoint.bytes,
        },
    }
