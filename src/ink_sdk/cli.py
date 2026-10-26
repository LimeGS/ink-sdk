"""``ink-sdk`` command line."""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
from pathlib import Path
from typing import Optional

import typer

from .api import InkSDK
from .contracts import RenderSpec, RunContext, jsonable
from .errors import DegenerateMapError, InkSDKError
from .registry import get_profile, list_profiles
from .resolver import build_segment

app = typer.Typer(
    add_completion=False,
    help="Standalone inference for Vesuvius ink models. Inference only.",
)


def _roi(value: Optional[str]) -> tuple[int, int, int, int] | None:
    if not value:
        return None
    text = value.replace(":", ",").replace(" ", ",")
    parts = [p for p in text.split(",") if p]
    if len(parts) != 4:
        raise typer.BadParameter("ROI must be y0,y1,x0,x1 (half-open)")
    try:
        y0, y1, x0, x1 = (int(p) for p in parts)
    except ValueError as exc:
        raise typer.BadParameter(f"ROI values must be integers: {value}") from exc
    if not (0 <= y0 < y1 and 0 <= x0 < x1):
        raise typer.BadParameter(f"ROI must satisfy 0 <= y0 < y1 and 0 <= x0 < x1: {value}")
    return y0, y1, x0, x1


def _ctx(
    *,
    device: str,
    cache_dir: Optional[Path],
    models_dir: Optional[Path],
    renderer: Optional[Path],
    villa: Optional[Path],
    python: Optional[str],
    overwrite: bool,
    dry_run: bool,
    deterministic: bool,
    render_threads: int,
    max_render_gb: float,
    on_degenerate: str,
    resample_from_um: Optional[float],
    output_3d: bool,
    amp: Optional[str],
    roi: tuple[int, int, int, int] | None,
    normal: str,
    umbilicus: Optional[Path],
    primary: Optional[str],
    no_render_cache: bool,
    orient_report: Optional[dict] = None,
) -> RunContext:
    return RunContext(
        device=device,
        cache_dir=cache_dir,
        models_dir=models_dir,
        renderer=renderer,
        villa=villa,
        python=python,
        overwrite=overwrite,
        dry_run=dry_run,
        deterministic=deterministic,
        render_threads=render_threads,
        max_render_gb=max_render_gb,
        on_degenerate=on_degenerate,
        resample_from_um=resample_from_um,
        output_3d=output_3d,
        amp=amp,
        roi=roi,
        normal=normal,
        umbilicus=umbilicus,
        primary=primary,
        orient_report=orient_report,
        render_cache=Path(tempfile.mkdtemp(prefix="ink-sdk-no-render-cache-")) if no_render_cache else None,
    )


def _guard(callable_):
    try:
        return callable_()
    except DegenerateMapError as exc:
        typer.echo(f"REFUSED: {exc}", err=True)
        raise typer.Exit(3)
    except InkSDKError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2)


def _build_sdk(renderer, models_dir, cache_dir, device, villa) -> InkSDK:
    return InkSDK(
        models_dir=models_dir,
        cache_dir=cache_dir,
        renderer=renderer,
        device=device,
        villa=villa,
    )


def _build_segment(
    sdk: InkSDK,
    profile: str,
    *,
    surface_volume,
    tifxyz,
    ct,
    voxel_um,
    pixel_um,
    force_render,
    ctx,
):
    profile_obj = get_profile(profile)
    explicit = (float(pixel_um),) * 3 if pixel_um is not None else None
    return build_segment(
        profile=profile_obj,
        ctx=ctx,
        surface_volume=surface_volume,
        tifxyz=tifxyz,
        ct=ct,
        voxel_um=voxel_um,
        pixel_um=explicit,
        force_render=force_render,
    )


@app.command("models")
def models(json_output: bool = typer.Option(False, "--json", help="Emit JSON.")) -> None:
    """List the built-in, pinned profiles."""
    rows = []
    for profile in list_profiles():
        rows.append(
            {
                "id": profile.id,
                "adapter": profile.adapter,
                "family": profile.family,
                "scale_um": profile.input.scale_um,
                "depth": profile.input.depth,
                "checkpoint": profile.checkpoint.path,
                "sha256": profile.checkpoint.sha256,
                "direction": profile.execution.direction,
                "primary": profile.output.primary,
            }
        )
    if json_output:
        typer.echo(json.dumps(rows, indent=2, sort_keys=True))
    else:
        for row in rows:
            typer.echo(
                f"{row['id']:<42} {row['adapter']:<12} {row['scale_um']:>5.3f} um "
                f"{row['depth']:>3} layers  {row['direction']:<7} {row['primary']}"
            )


@app.command("fetch")
def fetch(
    profile: Optional[list[str]] = typer.Option(None, "--profile", "-p", help="Repeatable; default all."),
    models_dir: Optional[Path] = typer.Option(None, "--models-dir"),
    overwrite: bool = typer.Option(False, "--overwrite"),
) -> None:
    """Download pinned checkpoints from Hugging Face, then verify size and sha256."""
    from .fetch import fetch_profiles

    target = models_dir or Path(
        os.environ.get("INK_SDK_MODELS", Path.home() / ".cache" / "ink-sdk" / "models")
    )
    selected = [get_profile(item) for item in profile] if profile else list_profiles()
    paths = _guard(
        lambda: fetch_profiles(
            selected, models_dir=target, overwrite=overwrite, progress=typer.echo
        )
    )
    for path in paths:
        typer.echo(path)


@app.command("doctor")
def doctor(
    json_output: bool = typer.Option(False, "--json"),
    renderer: Optional[Path] = typer.Option(None, "--renderer"),
    models_dir: Optional[Path] = typer.Option(None, "--models-dir"),
    villa: Optional[Path] = typer.Option(None, "--villa"),
) -> None:
    """Check dependencies, renderer flags, VILLA and checkpoint presence."""
    result = InkSDK().doctor(renderer=renderer, models_dir=models_dir, villa=villa)
    if json_output:
        typer.echo(json.dumps(jsonable(result), indent=2, sort_keys=True))
        return
    for check in result["checks"]:
        mark = "ok" if check["ok"] else ("missing" if check["required"] else "optional")
        typer.echo(f"[{mark}] {check['name']}: {check['detail']}")
    for profile in result["profiles"]:
        if profile.get("digest_matches"):
            state = "verified"
        elif profile["present"]:
            state = "present-but-unverified"
        else:
            state = "missing"
        typer.echo(f"[{state}] {profile['id']}: {profile['path']}")


def _input_options(
    surface_volume: Optional[Path],
    tifxyz: Optional[Path],
    ct: Optional[Path],
    voxel_um: Optional[float],
    pixel_um: Optional[float],
) -> dict:
    return {
        "surface_volume": surface_volume,
        "tifxyz": tifxyz,
        "ct": ct,
        "voxel_um": voxel_um,
        "pixel_um": pixel_um,
    }


@app.command("inspect")
def inspect(
    profile: str = typer.Option(..., "--profile", "-p"),
    surface_volume: Optional[Path] = typer.Option(None, "--surface-volume"),
    tifxyz: Optional[Path] = typer.Option(None, "--tifxyz"),
    ct: Optional[Path] = typer.Option(None, "--ct"),
    voxel_um: Optional[float] = typer.Option(None, "--voxel-um"),
    pixel_um: Optional[float] = typer.Option(None, "--pixel-um", help="Declared isotropic scale."),
    force_render: bool = typer.Option(False, "--render", help="Ignore a supplied surface volume and render."),
    roi: Optional[str] = typer.Option(None, "--roi", help="y0,y1,x0,x1 half-open at output scale."),
    normal: str = typer.Option("auto", "--normal", help="auto | inward | outward"),
    umbilicus: Optional[Path] = typer.Option(None, "--umbilicus"),
    renderer: Optional[Path] = typer.Option(None, "--renderer"),
    villa: Optional[Path] = typer.Option(None, "--villa"),
    models_dir: Optional[Path] = typer.Option(None, "--models-dir"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Show the scale/render/inference plan without loading weights."""
    parsed_roi = _roi(roi)
    ctx = _ctx(
        device="auto", cache_dir=None, models_dir=models_dir, renderer=renderer, villa=villa,
        python=None, overwrite=False, dry_run=True, deterministic=False, render_threads=1,
        max_render_gb=64.0, on_degenerate="warn", resample_from_um=None, output_3d=False,
        amp=None, roi=parsed_roi, normal=normal, umbilicus=umbilicus, primary=None,
        no_render_cache=False,
    )
    sdk = _build_sdk(renderer, models_dir, None, "auto", villa)
    segment = _build_segment(
        sdk, profile, surface_volume=surface_volume, tifxyz=tifxyz, ct=ct,
        voxel_um=voxel_um, pixel_um=pixel_um, force_render=force_render, ctx=ctx,
    )
    result = _guard(lambda: sdk.inspect(segment, profile, ctx=ctx))
    typer.echo(json.dumps(jsonable(result), indent=2, sort_keys=True))


@app.command("render")
def render(
    profile: str = typer.Option("ink_9um@scrollprize-seed43-step75000", "--profile", "-p"),
    tifxyz: Path = typer.Option(..., "--tifxyz"),
    ct: Path = typer.Option(..., "--ct"),
    out: Path = typer.Option(Path("out"), "--out", "-o"),
    voxel_um: Optional[float] = typer.Option(None, "--voxel-um"),
    num_slices: Optional[int] = typer.Option(None, "--num-slices"),
    scale: float = typer.Option(1.0, "--scale"),
    slice_step: int = typer.Option(1, "--slice-step"),
    normal: str = typer.Option("auto", "--normal"),
    umbilicus: Optional[Path] = typer.Option(None, "--umbilicus"),
    renderer: Optional[Path] = typer.Option(None, "--renderer"),
    threads: int = typer.Option(1, "--threads"),
    max_render_gb: float = typer.Option(64.0, "--max-render-gb"),
    overwrite: bool = typer.Option(False, "--overwrite"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Render a tifxyz mesh from a local uint8 CT."""
    ctx = _ctx(
        device="cpu", cache_dir=Path.home() / ".cache" / "ink-sdk", models_dir=None,
        renderer=renderer, villa=None, python=None, overwrite=overwrite, dry_run=dry_run,
        deterministic=False, render_threads=threads, max_render_gb=max_render_gb,
        on_degenerate="warn", resample_from_um=None, output_3d=False, amp=None, roi=None,
        normal=normal, umbilicus=umbilicus, primary=None, no_render_cache=False,
    )
    sdk = _build_sdk(renderer, None, ctx.cache_dir, "cpu", None)
    profile_obj = get_profile(profile)
    segment = build_segment(profile=profile_obj, ctx=ctx, tifxyz=tifxyz, ct=ct, voxel_um=voxel_um)
    segment = dataclasses.replace(
        segment,
        render=RenderSpec(
            scale=scale,
            num_slices=num_slices,
            slice_step=slice_step,
            orientation=normal,
            umbilicus=umbilicus,
            renderer=renderer,
        ),
    )
    result = _guard(lambda: sdk.render(segment, profile_obj, out_dir=out, ctx=ctx))
    if dry_run:
        typer.echo(json.dumps(jsonable(result), indent=2, sort_keys=True))
    elif json_output:
        typer.echo(json.dumps(jsonable(result.receipt), indent=2, sort_keys=True))
    else:
        typer.echo(str(out / "render"))


@app.command("infer")
def infer(
    profile: str = typer.Option("ink_9um@scrollprize-seed43-step75000", "--profile", "-p"),
    surface_volume: Optional[Path] = typer.Option(None, "--surface-volume"),
    tifxyz: Optional[Path] = typer.Option(None, "--tifxyz"),
    ct: Optional[Path] = typer.Option(None, "--ct"),
    out: Path = typer.Option(Path("out"), "--out", "-o"),
    voxel_um: Optional[float] = typer.Option(None, "--voxel-um"),
    pixel_um: Optional[float] = typer.Option(None, "--pixel-um"),
    force_render: bool = typer.Option(False, "--render"),
    roi: Optional[str] = typer.Option(None, "--roi", help="y0,y1,x0,x1 half-open at output scale."),
    resample_from_um: Optional[float] = typer.Option(None, "--resample-from-um"),
    amp: Optional[str] = typer.Option(None, "--amp", help="Override bf16/fp16/default/fp32."),
    device: str = typer.Option("auto", "--device"),
    primary: Optional[str] = typer.Option(None, "--primary"),
    orient_report: Optional[Path] = typer.Option(
        None, "--orient", help="JSON from `ink-sdk orient`; picks the primary map when --primary is not given."
    ),
    on_degenerate: str = typer.Option("fail", "--on-degenerate", help="fail | warn"),
    output_3d: bool = typer.Option(False, "--output-3d"),
    normal: str = typer.Option("auto", "--normal"),
    umbilicus: Optional[Path] = typer.Option(None, "--umbilicus"),
    renderer: Optional[Path] = typer.Option(None, "--renderer"),
    villa: Optional[Path] = typer.Option(None, "--villa"),
    python: Optional[str] = typer.Option(None, "--python"),
    models_dir: Optional[Path] = typer.Option(None, "--models-dir"),
    cache_dir: Optional[Path] = typer.Option(None, "--cache-dir"),
    no_render_cache: bool = typer.Option(False, "--no-render-cache"),
    threads: int = typer.Option(1, "--threads"),
    max_render_gb: float = typer.Option(64.0, "--max-render-gb"),
    overwrite: bool = typer.Option(False, "--overwrite"),
    deterministic: bool = typer.Option(False, "--deterministic"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Run one profile on a segment and publish maps plus receipts."""
    parsed_roi = _roi(roi)
    cache_dir = cache_dir or Path(os.environ.get("INK_SDK_CACHE", Path.home() / ".cache" / "ink-sdk"))
    ctx = _ctx(
        device=device, cache_dir=cache_dir, models_dir=models_dir, renderer=renderer,
        villa=villa, python=python, overwrite=overwrite, dry_run=dry_run,
        deterministic=deterministic, render_threads=threads, max_render_gb=max_render_gb,
        on_degenerate=on_degenerate, resample_from_um=resample_from_um, output_3d=output_3d,
        amp=amp, roi=parsed_roi, normal=normal, umbilicus=umbilicus, primary=primary,
        no_render_cache=no_render_cache,
        orient_report=json.loads(orient_report.read_text(encoding="utf-8")) if orient_report else None,
    )
    sdk = _build_sdk(renderer, models_dir, cache_dir, device, villa)
    segment = _build_segment(
        sdk, profile, surface_volume=surface_volume, tifxyz=tifxyz, ct=ct,
        voxel_um=voxel_um, pixel_um=pixel_um, force_render=force_render, ctx=ctx,
    )
    result = _guard(lambda: sdk.infer(segment, profile, out_dir=out, ctx=ctx))
    if dry_run:
        typer.echo(json.dumps(jsonable(result.receipt), indent=2, sort_keys=True))
        return
    if json_output:
        typer.echo(json.dumps(jsonable(result.receipt), indent=2, sort_keys=True))
    else:
        liveness = result.receipt.get("liveness", {}).get(result.primary, {})
        stats = result.receipt.get("maps", {}).get(result.primary, {}).get("statistics", {})
        typer.echo(f"{profile}: {result.primary} {liveness.get('verdict', 'unknown')}")
        typer.echo(f"out: {out}")
        typer.echo(f"p50={stats.get('p50')} p99={stats.get('p99')} max={stats.get('max')}")


@app.command("orient")
def orient_cmd(
    tifxyz: Path = typer.Option(..., "--tifxyz", help="Mesh the surface volume was rendered from."),
    surface_volume: Path = typer.Option(..., "--surface-volume"),
    ct: str = typer.Option(..., "--ct", help="CT OME-Zarr (local path or URL) in the mesh's voxel frame."),
    profile: Optional[str] = typer.Option(None, "--profile", "-p", help="Recommend a map for this profile's adapter."),
    ct_level: str = typer.Option("0", "--ct-level", help="Pyramid level to read; pair with --mesh-divisor."),
    mesh_divisor: float = typer.Option(1.0, "--mesh-divisor", help="Divide mesh coords, e.g. 4 for level 2."),
    distance_vx: float = typer.Option(1000.0, "--distance-vx", help="Sagitta test distance in CT voxels."),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Write the report JSON here."),
) -> None:
    """Label-free check of the depth direction of a surface volume (sheet side + CT match)."""
    from .orient import orient

    adapter = get_profile(profile).adapter if profile else None
    report = _guard(lambda: orient(tifxyz, surface_volume, ct, adapter=adapter, ct_level=ct_level,
                                   mesh_divisor=mesh_divisor, distance_vx=distance_vx))
    text = json.dumps(jsonable(report), indent=2, sort_keys=True)
    if out:
        out.write_text(text + "\n", encoding="utf-8")
    typer.echo(text)


def _int_list(text: str) -> list[int]:
    return [int(v) for v in text.replace(" ", "").split(",") if v]


def _float_list(text: str) -> list[float]:
    return [float(v) for v in text.replace(" ", "").split(",") if v]


@app.command("battery")
def battery_cmd(
    surface_volume: Path = typer.Option(..., "--surface-volume", help="Thick uint8 ZYX surface volume, rendered once."),
    voxel_um: float = typer.Option(..., "--voxel-um", help="Pixel size of the surface volume."),
    out: Path = typer.Option(Path("battery"), "--out", "-o"),
    profile: Optional[list[str]] = typer.Option(None, "--profile", "-p", help="Repeatable; default: every profile."),
    centres: str = typer.Option("-8,-4,0,4,8", "--centres", help="Window centres, in layers from the middle."),
    steps: str = typer.Option("1,2", "--steps", help="Layers between consecutive model inputs (depth width)."),
    thresholds: str = typer.Option("0.5,0.7", "--thresholds"),
    resample_from_um: Optional[float] = typer.Option(
        None, "--resample-from-um", help="Opt-in 9 um resample (e.g. 8.64), applied to ink9um profiles only."),
    device: str = typer.Option("auto", "--device"),
    villa: Optional[Path] = typer.Option(None, "--villa"),
    python: Optional[str] = typer.Option(None, "--python"),
    models_dir: Optional[Path] = typer.Option(None, "--models-dir"),
    cache_dir: Optional[Path] = typer.Option(None, "--cache-dir"),
    thumb_px: int = typer.Option(360, "--thumb-px", help="Width of each mosaic tile."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the cells and the skipped profiles, run nothing."),
    ensemble: Optional[str] = typer.Option(
        None, "--ensemble",
        help="Also write a z-score ensemble of the cells; the value is the reading side (forward or reverse)."),
) -> None:
    """Every profile x depth centre x depth width on one segment: maps, overlays, a mosaic, a stability map."""
    from .battery import run_battery

    cache_dir = cache_dir or Path(os.environ.get("INK_SDK_CACHE", Path.home() / ".cache" / "ink-sdk"))
    ctx = _ctx(
        device=device, cache_dir=cache_dir, models_dir=models_dir, renderer=None, villa=villa, python=python,
        overwrite=True, dry_run=dry_run, deterministic=False, render_threads=1, max_render_gb=64.0,
        on_degenerate="warn", resample_from_um=resample_from_um, output_3d=False, amp=None, roi=None,
        normal="auto", umbilicus=None, primary=None, no_render_cache=False,
    )
    sdk = _build_sdk(None, models_dir, cache_dir, device, villa)
    selected = [get_profile(item) for item in profile] if profile else list_profiles()
    report = _guard(lambda: run_battery(
        sdk, surface_volume, voxel_um, selected, out, ctx, centres=_int_list(centres), steps=_int_list(steps),
        thresholds=_float_list(thresholds), resample_from_um=resample_from_um, thumb_px=thumb_px,
        ensemble=ensemble, say=typer.echo))
    if dry_run:
        typer.echo(json.dumps(jsonable(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
