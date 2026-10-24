"""Orchestration: resolve, render, prepare, infer, gate, publish."""

from __future__ import annotations

import dataclasses
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from . import __version__
from .contracts import InkResult, InkVolume, RunContext, Segment
from .errors import DegenerateMapError, InkSDKError, ValidationError
from .io import write_json, write_zarr
from .liveness import assess_liveness, forward_reverse_asymmetry
from .prepare import prepare_input
from .profiles import ModelProfile
from .registry import get_adapter
from .renderers.base import Renderer
from .renderers.vc_render_tifxyz import VcRenderTifxyzRenderer
from .resolver import inspect_plan, with_render_spec


def _utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _models_dir(ctx: RunContext) -> Path:
    return Path(ctx.models_dir or os.environ.get("INK_SDK_MODELS", Path.home() / ".cache" / "ink-sdk" / "models"))


def _workdir(ctx: RunContext) -> Path:
    return Path(ctx.workdir) if ctx.workdir is not None else Path(tempfile.mkdtemp(prefix="ink-sdk-"))


def _prepare_out(out_dir: Path, overwrite: bool) -> Path:
    out_dir = Path(out_dir)
    if out_dir.exists() and not out_dir.is_dir():
        raise ValidationError(f"output path is not a directory: {out_dir}")
    populated = out_dir.is_dir() and any(out_dir.iterdir())
    if populated and not overwrite:
        raise ValidationError(
            f"output directory is not empty: {out_dir}. Pass --overwrite to replace files."
        )
    resolved = out_dir.resolve()
    protected = (
        resolved == Path.cwd().resolve()
        or resolved == Path.home().resolve()
        or resolved.parent == resolved
    )
    if populated and protected:
        raise ValidationError(f"refusing to delete a protected output directory: {out_dir}")
    if populated:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _publish_render(rendered, out_dir: Path, overwrite: bool) -> None:
    target = Path(out_dir) / "render"
    if target.exists():
        if not overwrite:
            raise ValidationError(f"render output already exists: {target}")
        shutil.rmtree(target)
    shutil.copytree(rendered.path, target)
    receipt_path = target / "RENDER_RECEIPT.json"
    if not receipt_path.is_file():
        write_json(receipt_path, rendered.receipt)


def _write_map(path: Path, array: np.ndarray) -> None:
    np.save(path, np.asarray(array, dtype=np.float32))


def _preview(path: Path, array: np.ndarray) -> None:
    preview = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    tifffile.imwrite(path, preview)


def _stats(array: np.ndarray) -> dict:
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


def _environment(ctx: RunContext) -> dict:
    env = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "device": ctx.device,
        "villa": str(ctx.villa or os.environ.get("VILLA") or ""),
        "deterministic": ctx.deterministic,
    }
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
    except ImportError:
        env["torch"] = None
    return env


def infer(
    segment: Segment,
    profile: ModelProfile,
    ctx: RunContext,
    *,
    out_dir: Path | None = None,
    adapter=None,
    renderer: Renderer | None = None,
) -> InkResult:
    """Run one inference and publish the maps and receipt."""
    started = time.time()
    work = _workdir(ctx)
    ctx = dataclasses.replace(ctx, workdir=work)
    if ctx.dry_run:
        plan = inspect_plan(segment, profile, ctx)
        return InkResult(maps={}, primary=profile.output.primary, volume=None, out_dir=None, receipt_path=None, receipt=plan)

    output = _prepare_out(Path(out_dir), ctx.overwrite) if out_dir is not None else None
    current = segment
    rendered = None
    if current.surface_volume is None:
        current = with_render_spec(current, profile, ctx)
        renderer_impl = renderer or VcRenderTifxyzRenderer()
        issues = renderer_impl.check(current, profile)
        errors = [issue for issue in issues if issue.is_error]
        if errors:
            raise ValidationError("; ".join(issue.message for issue in errors))
        assert current.render is not None
        rendered = renderer_impl.render(current, current.render, profile, ctx)
        current = dataclasses.replace(
            current,
            surface_volume=rendered.path,
            root=rendered.path,
            pixel_um=rendered.pixel_um,
            depth=rendered.shape_zyx[0],
            dtype="uint8",
        )
        if output is not None:
            _publish_render(rendered, output, ctx.overwrite)

    if current.pixel_um is None:
        raise ValidationError("surface volume has no scale; pass --voxel-um")
    adapter_impl = adapter or get_adapter(profile)
    issues = adapter_impl.check(current, profile, ctx)
    errors = [issue for issue in issues if issue.is_error]
    if errors:
        raise ValidationError("; ".join(issue.message for issue in errors))

    prepared = prepare_input(
        current.surface_volume,
        profile,
        source_um=current.pixel_um[0],
        cache_dir=ctx.cache_dir,
        resample_from_um=ctx.resample_from_um,
        roi=ctx.roi,
    )
    raw = adapter_impl.run(prepared, profile, ctx)
    maps = adapter_impl.to_maps(raw, prepared, profile)
    orientation_check = _orientation_check(ctx, profile)
    primary = ctx.primary or orientation_check.get("applied_primary") or profile.output.primary
    if primary not in maps:
        raise ValidationError(
            f"primary {primary!r} is not among the produced maps {sorted(maps)}"
        )
    liveness = {
        key: assess_liveness(value.array, valid=value.valid) for key, value in maps.items()
    }

    if output is not None:
        for key, value in maps.items():
            suffix = "" if key == "forward" else f"_{key}"
            _write_map(output / f"probability{suffix}.npy", value.array)
        if primary != "forward" and "forward" in maps:
            _write_map(output / "probability_primary.npy", maps[primary].array)
        _preview(output / "preview.tif", maps[primary].array)
        volume = None
        if raw.volume is not None:
            volume_path = output / "ink_3d.zarr"
            write_zarr(volume_path, raw.volume, attrs=raw.volume_attrs, overwrite=ctx.overwrite)
            interval = raw.volume_attrs.get("evaluated_z_interval")
            volume = InkVolume(
                path=volume_path,
                dims=("z", "y", "x"),
                pixel_um=prepared.pixel_um,
                evaluated_z_interval=tuple(interval) if interval else None,
                projection=None,
                attrs=raw.volume_attrs,
            )
        receipt = {
            "schema": "ink-sdk.receipt.v1",
            "generated_at_utc": _utc(),
            "sdk_version": __version__,
            "profile_id": profile.id,
            "adapter": profile.adapter,
            "checkpoint": {
                "path": str(profile.checkpoint_path(_models_dir(ctx))),
                "sha256": profile.checkpoint.sha256,
                "bytes": profile.checkpoint.bytes,
                "verified": getattr(adapter_impl, "name", None) == profile.adapter,
            },
            "segment": {
                "id": current.id,
                "surface_volume": str(current.surface_volume),
                "surface_mesh": str(current.surface_mesh) if current.surface_mesh else None,
                "ct": str(current.ct) if current.ct else None,
                "render": rendered.receipt if rendered else None,
            },
            "input": {
                "scale_plan": prepared.scale_plan,
                "pixel_um": prepared.pixel_um,
                "shape_zyx": list(prepared.array.shape),
                "roi": list(ctx.roi) if ctx.roi is not None else None,
                "origin_px": list(prepared.origin_px),
                "prepared_receipt": prepared.receipt,
            },
            "execution": {
                **raw.receipt,
                "direction": profile.execution.direction,
                "device_requested": ctx.device,
            },
            "maps": {
                key: {
                    "shape_yx": list(value.shape),
                    "direction": value.direction,
                    "pixel_um": value.pixel_um,
                    "origin_px": value.origin_px,
                    "no_ink_floor": value.no_ink_floor,
                    "quantisation": value.quantisation,
                    "statistics": _stats(value.array),
                    "liveness": liveness[key],
                }
                for key, value in maps.items()
            },
            "primary": primary,
            "primary_source": "user" if ctx.primary else ("orientation-check" if orientation_check.get("applied_primary") else "profile"),
            "orientation_check": orientation_check or None,
            "liveness": liveness,
            "environment": _environment(ctx),
            "runtime_seconds": round(time.time() - started, 2),
            "non_claims": [
                "A probability map is not OCR and does not accept ink, text, letters or First Letters.",
                "No scientific evaluation was performed by this SDK.",
            ],
        }
        if "forward" in maps and "reverse" in maps:
            receipt["reverse_summary"] = forward_reverse_asymmetry(
                maps["forward"].array, maps["reverse"].array, valid=maps["forward"].valid
            )
        receipt_path = output / "INK_SDK_RECEIPT.json"
        write_json(receipt_path, receipt)
    else:
        receipt = {}
        receipt_path = None
        volume = None

    if liveness[primary]["verdict"] != "ALIVE" and ctx.on_degenerate == "fail":
        if output is not None:
            (output / "LANE_NOT_USABLE").write_text(
                f"{liveness[primary]['verdict']}: {liveness[primary].get('reason', '')}\n",
                encoding="utf-8",
            )
        raise DegenerateMapError(liveness[primary])
    return InkResult(
        maps=maps,
        primary=primary,
        volume=volume,
        out_dir=output,
        receipt_path=receipt_path,
        receipt=receipt,
    )


def _orientation_check(ctx: RunContext, profile) -> dict:
    """Apply an `ink-sdk orient` report: only when the user did not pick a primary, only for the
    adapter the report was made for, and only when the report recommends a map."""
    report = ctx.orient_report
    if not report:
        return {}
    summary = {
        "schema": report.get("schema"),
        "layer0_face": report.get("layer0_face"),
        "recommended_primary": report.get("recommended_primary"),
        "report_adapter": report.get("adapter"),
        "sheet_side": report.get("sheet_side", {}).get("normal_points"),
        "stack": report.get("stack", {}).get("layers_grow_along"),
        "applied_primary": None,
    }
    rec = report.get("recommended_primary")
    if ctx.primary:
        summary["note"] = "user --primary wins over the report"
    elif report.get("adapter") != profile.adapter:
        summary["note"] = f"report was made for adapter {report.get('adapter')!r}, not {profile.adapter!r}; not applied"
    elif rec is None:
        summary["note"] = report.get("reason") or "no recommendation"
    else:
        summary["applied_primary"] = rec
    return summary


def render_segment(
    segment: Segment,
    profile: ModelProfile,
    ctx: RunContext,
    *,
    out_dir: Path | None = None,
    renderer: Renderer | None = None,
) -> Any:
    current = with_render_spec(segment, profile, ctx)
    renderer_impl = renderer or VcRenderTifxyzRenderer()
    issues = renderer_impl.check(current, profile)
    errors = [issue for issue in issues if issue.is_error]
    if errors:
        raise ValidationError("; ".join(issue.message for issue in errors))
    assert current.render is not None
    if ctx.dry_run:
        return {
            "segment": str(current.surface_mesh),
            "render": current.render,
            "num_slices": current.render.num_slices,
            "pixel_um": current.pixel_um,
            "issues": issues,
        }
    output = _prepare_out(Path(out_dir), ctx.overwrite) if out_dir is not None else None
    rendered = renderer_impl.render(current, current.render, profile, ctx)
    if output is not None:
        _publish_render(rendered, output, ctx.overwrite)
    return rendered


def inspect(segment: Segment, profile: ModelProfile, ctx: RunContext, *, adapter=None, renderer: Renderer | None = None) -> dict:
    current = with_render_spec(segment, profile, ctx)
    result = inspect_plan(current, profile, ctx)
    adapter_impl = adapter or get_adapter(profile)
    result["issues"] = [
        issue.__dict__ for issue in adapter_impl.check(current, profile, ctx)
    ]
    if current.surface_volume is None:
        renderer_impl = renderer or VcRenderTifxyzRenderer()
        result["issues"].extend(issue.__dict__ for issue in renderer_impl.check(current, profile))
    return result


def doctor(
    *,
    renderer: Path | None = None,
    models_dir: Path | None = None,
    villa: Path | None = None,
) -> dict:
    checks: list[dict] = []

    def add(name: str, ok: bool, detail: Any, *, required: bool = True) -> None:
        checks.append({"name": name, "ok": bool(ok), "required": required, "detail": detail})

    add("python", sys.version_info >= (3, 11), sys.version.split()[0])
    for module in ("numpy", "pydantic", "typer", "zarr", "tifffile"):
        try:
            imported = __import__(module)
            add(module, True, getattr(imported, "__version__", "installed"))
        except ImportError as exc:
            add(module, False, str(exc))
    for module, required in (("scipy", False), ("torch", False), ("imagecodecs", False), ("huggingface_hub", False)):
        try:
            imported = __import__(module)
            add(module, True, getattr(imported, "__version__", "installed"), required=required)
        except ImportError as exc:
            add(module, False, str(exc), required=required)
    try:
        import torch

        add("cuda", torch.cuda.is_available(), torch.version.cuda, required=False)
    except ImportError:
        pass

    renderer_value = renderer or os.environ.get("INK_SDK_RENDERER") or os.environ.get("VC3D_APPRUN") or shutil.which("vc_render_tifxyz")
    if renderer_value:
        path = Path(renderer_value)
        add("renderer", path.is_file() and os.access(path, os.X_OK), str(path), required=False)
        if path.is_file():
            prefix = [str(path), "vc_render_tifxyz"] if path.name == "AppRun" else [str(path)]
            try:
                completed = subprocess.run(prefix + ["--help"], capture_output=True, text=True, timeout=20)
                help_text = (completed.stdout or "") + (completed.stderr or "")
                flags = ["--num-slices", "--scale", "--auto-crop", "--flip-normals", "--slice-step", "--zarr-output"]
                add(
                    "renderer-help",
                    completed.returncode == 0,
                    {flag: flag in help_text for flag in flags},
                    required=False,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                add("renderer-help", False, str(exc), required=False)
    else:
        add("renderer", False, "not configured (INK_SDK_RENDERER / VC3D_APPRUN)", required=False)

    villa_value = villa or os.environ.get("VILLA")
    if villa_value:
        villa_path = Path(villa_value)
        candidates = [villa_path / "ink-detection", villa_path]
        found = next((c for c in candidates if (c / "koine_machines" / "inference" / "infer.py").is_file()), None)
        add("villa", found is not None, str(found or villa_path), required=False)
        if found is not None:
            env = dict(os.environ)
            env["PYTHONPATH"] = str(found) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            try:
                imported = subprocess.run(
                    [sys.executable, "-c", "import koine_machines.inference.infer, vesuvius"],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    env=env,
                )
                add(
                    "villa-import",
                    imported.returncode == 0,
                    (imported.stderr or "ok").strip()[-500:],
                    required=False,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                add("villa-import", False, str(exc), required=False)
    else:
        add("villa", False, "not configured (VILLA)", required=False)

    models = Path(models_dir or os.environ.get("INK_SDK_MODELS", Path.home() / ".cache" / "ink-sdk" / "models"))
    add("models-dir", models.is_dir(), str(models), required=False)
    from .registry import list_profiles

    profile_status = []
    for profile in list_profiles():
        path = profile.checkpoint_path(models)
        present = path.is_file()
        size = path.stat().st_size if present else None
        verified = False
        error = None
        if present:
            try:
                profile.verify_checkpoint(models)
                verified = True
            except (InkSDKError, OSError) as exc:
                error = str(exc)
        profile_status.append(
            {
                "id": profile.id,
                "path": str(path),
                "upstream_path": profile.upstream_checkpoint_path,
                "present": present,
                "size_matches": size == profile.checkpoint.bytes if size is not None else False,
                "digest_matches": verified,
                "error": error,
            }
        )
    return {
        "sdk_version": __version__,
        "checks": checks,
        "profiles": profile_status,
        "ok": all(c["ok"] or not c["required"] for c in checks),
    }
