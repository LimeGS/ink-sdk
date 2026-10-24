"""Scale planning and the two transformations v0 permits.

The model families do not share a preprocessing recipe. This module keeps the
scale decision explicit and refuses the stretches the profiles do not support.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .contracts import PreparedInput, ScalePlan
from .errors import IncompatibleScaleError
from .io import open_volume_array, validate_uint8_zyx, write_zarr
from .profiles import ModelProfile

# 8.64 -> 9.362 is the only resample Campaign X measured as an opt-in path.
RESAMPLE_OUTPUT_UM = 9.362


def plan_scale(
    source_um: float,
    profile: ModelProfile,
    *,
    resample_from_um: float | None = None,
) -> ScalePlan:
    target = float(profile.input.scale_um)
    tolerance = float(profile.input.scale_tolerance_um)
    source = float(source_um)
    if source <= 0:
        raise IncompatibleScaleError(f"source scale must be positive, got {source}")

    if abs(source - target) <= tolerance:
        return ScalePlan(source, target, "native")

    adapter = profile.adapter
    coarse = target > 5.0
    near_2_4 = abs(source - 2.4) <= 0.15
    if coarse and adapter in {"ink9um", "hecate"} and near_2_4:
        factor = int(round(target / source))
        output = source * factor
        if factor >= 2 and abs(output - target) <= tolerance:
            return ScalePlan(
                source,
                target,
                "pool",
                (factor, factor, factor),
                note=f"{source:g} um pooled {factor}x -> {output:.3f} um",
            )

    if adapter == "ink9um" and resample_from_um is not None:
        if abs(source - float(resample_from_um)) <= 0.2:
            if abs(RESAMPLE_OUTPUT_UM - target) <= tolerance:
                zoom = source / RESAMPLE_OUTPUT_UM
                return ScalePlan(
                    source,
                    RESAMPLE_OUTPUT_UM,
                    "resample",
                    resample_from_um=float(resample_from_um),
                    note=(
                        f"caller-declared XY linear resample {source:g} -> "
                        f"{RESAMPLE_OUTPUT_UM:g} um (no prefilter); Z untouched"
                    ),
                )
        raise IncompatibleScaleError(
            f"--resample-from-um {resample_from_um:g} does not describe the source "
            f"scale {source:g} um"
        )

    ratio = target / source
    if source > target:
        raise IncompatibleScaleError(
            f"{profile.id} needs {target:g} ± {tolerance:g} um, but the source is "
            f"{source:g} um ({ratio:.2f}x); this model has no upsample path"
        )
    if adapter == "canonical2um":
        raise IncompatibleScaleError(
            f"{profile.id} only accepts 2.399 ± 0.05 um; the source is {source:g} um "
            "and the canonical runner has no upsample or pooling path"
        )
    if adapter == "hecate":
        pct = abs(source / target - 1) * 100
        raise IncompatibleScaleError(
            f"{profile.id} requires {target:g} um within {tolerance:g} um; "
            f"{source:g} um differs by {pct:.2f}% and Hecate does not resample"
        )
    raise IncompatibleScaleError(
        f"{profile.id} needs {target:g} ± {tolerance:g} um; {source:g} um is "
        "outside the permitted pooling/resample recipes. Pass "
        "--resample-from-um only where the profile documents it."
    )


def pool_volume(volume: np.ndarray, factors_zyx: tuple[int, int, int]) -> np.ndarray:
    fz, fy, fx = (int(v) for v in factors_zyx)
    if fz == fy == fx == 1:
        return volume
    depth, height, width = volume.shape
    depth -= depth % fz
    height -= height % fy
    width -= width % fx
    trimmed = volume[:depth, :height, :width]
    reshaped = trimmed.reshape(
        depth // fz, fz, height // fy, fy, width // fx, fx
    )
    return reshaped.mean(axis=(1, 3, 5), dtype=np.float32).round().astype(volume.dtype)


def resample_xy(volume: np.ndarray, zoom: float) -> np.ndarray:
    """XY-only linear interpolation, no Gaussian prefilter.

    This mirrors the method whose ~4% correlation cost Campaign X measured.
    """
    from scipy import ndimage

    resized = ndimage.zoom(volume.astype(np.float32), (1.0, zoom, zoom), order=1, prefilter=False)
    return resized.round().astype(volume.dtype)


def crop_roi(volume: np.ndarray, roi: tuple[int, int, int, int] | None) -> np.ndarray:
    if roi is None:
        return volume
    y0, y1, x0, x1 = (int(v) for v in roi)
    if not (0 <= y0 < y1 <= volume.shape[1] and 0 <= x0 < x1 <= volume.shape[2]):
        raise IncompatibleScaleError(
            f"ROI {roi} is outside the prepared map {volume.shape[1:]} (half-open y,x)"
        )
    return volume[:, y0:y1, x0:x1]


def _cache_key(
    source: Path,
    profile: ModelProfile,
    plan: ScalePlan,
    roi: tuple[int, int, int, int] | None,
) -> str:
    try:
        stat = source.stat()
        identity = [str(source.resolve()), stat.st_size, stat.st_mtime_ns]
    except OSError:
        identity = [str(source)]
    payload = {
        "source": identity,
        "profile": profile.id,
        "scale": [plan.source_um, plan.target_um, plan.mode, plan.factors_zyx, plan.resample_from_um],
        "roi": roi,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]


def prepare_input(
    source: Path,
    profile: ModelProfile,
    *,
    source_um: float,
    cache_dir: Path | None = None,
    resample_from_um: float | None = None,
    roi: tuple[int, int, int, int] | None = None,
) -> PreparedInput:
    """Load, scale and optionally crop an input at model resolution."""
    source = Path(source)
    plan = plan_scale(source_um, profile, resample_from_um=resample_from_um)
    output_um = plan.output_pixel_um
    cache_dir = Path(cache_dir) if cache_dir is not None else None

    # Native, uncropped input is already exactly what the adapter reads.
    if plan.mode == "native" and roi is None:
        volume = open_volume_array(source)
        validate_uint8_zyx(volume)
        return PreparedInput(
            path=source,
            array=volume,
            pixel_um=(output_um, output_um, output_um),
            source=source,
            scale_plan=plan,
            valid=volume.any(axis=0),
            receipt={
                "source": str(source),
                "scale_plan": plan,
                "scaled": False,
                "roi": None,
            },
        )

    key = _cache_key(source, profile, plan, roi)
    prepared_dir = cache_dir / "prepared" / key if cache_dir is not None else None
    if prepared_dir is not None and (prepared_dir / "0" / ".zarray").is_file():
        volume = open_volume_array(prepared_dir)
        validate_uint8_zyx(volume)
        return PreparedInput(
            path=prepared_dir,
            array=volume,
            pixel_um=(output_um, output_um, output_um),
            source=source,
            scale_plan=plan,
            roi=roi,
            origin_px=(roi[2], roi[0]) if roi else (0, 0),
            valid=volume.any(axis=0),
            receipt={
                "source": str(source),
                "scale_plan": plan,
                "scaled": True,
                "roi": roi,
                "cache_hit": True,
                "cache_key": key,
            },
        )

    volume = open_volume_array(source)
    validate_uint8_zyx(volume)
    if plan.mode == "pool":
        volume = pool_volume(volume, plan.factors_zyx)
    elif plan.mode == "resample":
        volume = resample_xy(volume, plan.source_um / plan.target_um)
    volume = crop_roi(volume, roi)

    receipt = {
        "source": str(source),
        "scale_plan": plan,
        "scaled": True,
        "roi": roi,
        "origin_px": [roi[2], roi[0]] if roi else [0, 0],
        "shape_zyx": list(volume.shape),
        "cache_hit": False,
        "cache_key": key,
    }
    if prepared_dir is not None:
        attrs = {
            "axes": ["z", "y", "x"],
            "pixel_um": [output_um, output_um, output_um],
            "ink_sdk_scale_plan": receipt["scale_plan"].__dict__,
            "ink_sdk_source": str(source),
        }
        write_zarr(prepared_dir, volume, attrs=attrs)
    return PreparedInput(
        path=prepared_dir or source,
        array=volume,
        pixel_um=(output_um, output_um, output_um),
        source=source,
        scale_plan=plan,
        roi=roi,
        origin_px=(roi[2], roi[0]) if roi else (0, 0),
        valid=volume.any(axis=0),
        receipt=receipt,
    )


def roi_shape(roi: tuple[int, int, int, int] | None, full_hw: tuple[int, int]) -> tuple[int, int]:
    if roi is None:
        return full_hw
    y0, y1, x0, x1 = roi
    if not (0 <= y0 < y1 <= full_hw[0] and 0 <= x0 < x1 <= full_hw[1]):
        raise IncompatibleScaleError(f"ROI {roi} is outside map {full_hw}")
    return (y1 - y0, x1 - x0)


def pooled_pixel_um(source_um: float, plan: ScalePlan, axis: int = 0) -> float:
    if plan.mode == "pool":
        return source_um * plan.factors_zyx[axis]
    if plan.mode == "resample":
        return plan.target_um
    return source_um


def scale_note(plan: ScalePlan) -> str:
    if plan.mode == "native":
        return f"native {plan.source_um:g} um"
    if plan.mode == "pool":
        return f"pool {plan.factors_zyx} -> {plan.output_pixel_um:g} um"
    return plan.note or f"resample -> {plan.target_um:g} um"
