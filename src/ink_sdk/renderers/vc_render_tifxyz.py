"""Official VC3D surface renderer: ``vc_render_tifxyz``.

The renderer is an external Linux binary (usually an AppImage's ``AppRun``).
This module owns the argv, the preflight, the orientation and the receipt; it
does not reimplement the C++ renderer.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import zarr

from ..contracts import Issue, RenderedVolume, RenderSpec, RunContext, Segment
from ..errors import MissingDependencyError, RenderError, ValidationError
from ..io import ct_preflight, sha256_file, write_json, write_zarr
from ..profiles import ModelProfile
from .base import render_key

_HELP_CACHE: dict[tuple[str, int, int], tuple[bool, str]] = {}


def _utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _binary_parts(path: Path) -> tuple[list[str], Path]:
    path = Path(path)
    if path.name == "AppRun":
        return [str(path), "vc_render_tifxyz"], path
    return [str(path)], path


def _help(binary: Path) -> tuple[bool, str]:
    stat = binary.stat()
    key = (str(binary), stat.st_size, stat.st_mtime_ns)
    if key in _HELP_CACHE:
        return _HELP_CACHE[key]
    prefix, _ = _binary_parts(binary)
    try:
        completed = subprocess.run(
            [*prefix, "--help"], capture_output=True, text=True, timeout=30
        )
        text = (completed.stdout or "") + (completed.stderr or "")
        result = (completed.returncode == 0, text)
    except (OSError, subprocess.SubprocessError) as exc:
        result = (False, str(exc))
    _HELP_CACHE[key] = result
    return result


def _supports(binary: Path, flag: str) -> bool:
    ok, text = _help(binary)
    # A renderer that refuses --help is treated as an older binary whose flags
    # we cannot inspect; the render itself will produce the actionable error.
    return (not ok) or flag in text


def _resolve_binary(spec: RenderSpec, ctx: RunContext) -> Path:
    value = spec.renderer or ctx.renderer or os.environ.get("INK_SDK_RENDERER") or os.environ.get("VC3D_APPRUN")
    if value:
        path = Path(value).expanduser()
    else:
        found = shutil.which("vc_render_tifxyz")
        if not found:
            raise RenderError(
                "no vc_render_tifxyz renderer. Set --renderer, INK_SDK_RENDERER or "
                "VC3D_APPRUN, or put vc_render_tifxyz on PATH."
            )
        path = Path(found)
    if not path.is_file():
        raise RenderError(f"renderer not found: {path}")
    if not os.access(path, os.X_OK):
        raise RenderError(f"renderer is not executable: {path}")
    return path


def _mesh_arrays(mesh: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = []
    for axis in "xyz":
        path = Path(mesh) / f"{axis}.tif"
        if not path.is_file():
            raise ValidationError(f"tifxyz mesh is missing {path.name}: {mesh}")
        arrays.append(tifffile.imread(path))
    x, y, z = arrays
    if x.ndim != 2 or x.shape != y.shape or x.shape != z.shape or min(x.shape) < 3:
        raise ValidationError(f"mesh x/y/z must be three compatible 2D grids: {mesh}")
    return x, y, z


def _orientation(mesh: Path, spec: RenderSpec) -> dict[str, Any]:
    x, y, z = _mesh_arrays(mesh)
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    z = z.astype(np.float64)
    ok = (x > 0) & (y > 0) & (z > 0) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if not ok.any():
        raise RenderError(f"mesh has no valid coordinates: {mesh}")
    P = np.stack([x, y, z], axis=-1)
    dr = np.zeros_like(P)
    dc = np.zeros_like(P)
    dr[1:-1] = P[2:] - P[:-2]
    dc[:, 1:-1] = P[:, 2:] - P[:, :-2]
    valid = ok.copy()
    valid[1:-1] &= ok[2:] & ok[:-2]
    valid[:, 1:-1] &= ok[:, 2:] & ok[:, :-2]
    valid[[0, -1], :] = False
    valid[:, [0, -1]] = False
    normals = np.cross(dr, dc)[valid]
    points = P[valid]
    if not len(points):
        raise RenderError(f"mesh has no valid stencil for orientation: {mesh}")

    adentro = afuera = 0.0
    source = "unverified-no-umbilicus"
    if spec.umbilicus is not None:
        try:
            from scipy.interpolate import interp1d
        except ImportError as exc:
            raise MissingDependencyError(
                "orientation with --umbilicus needs scipy; install ink-sdk[render]"
            ) from exc
        payload = json.loads(Path(spec.umbilicus).read_text(encoding="utf-8"))
        raw = payload["control_points"] if isinstance(payload, dict) else payload
        controls = raw.values() if isinstance(raw, dict) else raw
        cz = np.array(
            [
                [c["z"] if isinstance(c, dict) else c[2],
                 c["y"] if isinstance(c, dict) else c[1],
                 c["x"] if isinstance(c, dict) else c[0]]
                for c in controls
            ],
            dtype=np.float64,
        )
        cz = cz[np.argsort(cz[:, 0])]
        fy = interp1d(cz[:, 0], cz[:, 1], fill_value="extrapolate")
        fx = interp1d(cz[:, 0], cz[:, 2], fill_value="extrapolate")
        toward = np.stack(
            [fx(points[:, 2]) - points[:, 0], fy(points[:, 2]) - points[:, 1], np.zeros(len(points))],
            axis=-1,
        )
        cosang = (normals * toward).sum(1) / (
            np.linalg.norm(normals, axis=1) * np.linalg.norm(toward, axis=1) + 1e-9
        )
        adentro = float((cosang > 0.4).mean())
        afuera = float((cosang < -0.4).mean())
        if max(adentro, afuera) < 0.9:
            raise RenderError(
                f"normal orientation is ambiguous (inward {adentro:.3f}, outward {afuera:.3f}); "
                "check the mesh and umbilicus"
            )
        source = "umbilicus"

    # The reference wrapper orders the stack so VC3D's normals point outward.
    flip = adentro > afuera
    if spec.orientation == "inward":
        flip = not flip
    elif spec.orientation not in {"auto", "inward", "outward"}:
        raise ValidationError(f"unknown orientation {spec.orientation!r}")

    zr = float(np.median(dr[valid][:, 2]))
    zc = float(np.median(dc[valid][:, 2]))
    if source == "umbilicus":
        # Port of render_vc3d.orientacion(); it returns (transpose, flip_rows, flip_cols).
        sn = 1 if adentro > afuera else -1
        transpose = abs(zc) > abs(zr)
        if transpose:
            sn, sz = -sn, (1 if zc > 0 else -1)
        else:
            sz = 1 if zr > 0 else -1
        rows, cols = {
            (1, 1): (False, False),
            (-1, -1): (True, False),
            (-1, 1): (False, True),
            (1, -1): (True, True),
        }[(sn, sz)]
        reading = "verified-umbilicus"
    else:
        transpose, rows, cols = False, False, False
        reading = "unverified"
        if spec.orientation in {"inward", "outward"}:
            source = "user-declared"
    return {
        "source": source,
        "orientation_source": source,
        "inward_fraction": adentro,
        "outward_fraction": afuera,
        "flip_normals": bool(flip),
        "traspuesta": bool(transpose),
        "filas_invertidas": bool(rows),
        "columnas_invertidas": bool(cols),
        "z_por_fila": zr,
        "z_por_col": zc,
        "reading_orientation": reading,
        "requested": spec.orientation,
    }


def _derive_num_slices(segment: Segment, profile: ModelProfile) -> int:
    if segment.depth is not None and segment.depth == profile.input.depth:
        return profile.input.depth
    scale = segment.pixel_um[0] if segment.pixel_um else None
    if scale and profile.input.scale_um > 5 and abs(scale - 2.4) <= 0.15:
        return profile.input.depth * 4
    return profile.input.depth


def _read_render(stage: Path) -> np.ndarray:
    try:
        group = zarr.open_group(str(stage), mode="r")
        array = np.asarray(group["0"])
    except Exception as exc:
        raise RenderError(f"renderer produced no readable zarr array '0': {stage}: {exc}") from exc
    if array.dtype != np.uint8 or array.ndim != 3:
        raise RenderError(f"renderer produced {array.dtype} {array.shape}; expected uint8 ZYX")
    if not array.any():
        raise RenderError("render is entirely zero; refusing to publish an empty surface volume")
    return array


def _publish(path: Path, array: np.ndarray, attrs: dict[str, Any], overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not overwrite:
            raise RenderError(f"refusing to overwrite existing render: {path}")
        shutil.rmtree(path)
    tmp = path.parent / f".{path.name}.tmp-{os.getpid()}-{int(time.time() * 1000)}"
    write_zarr(tmp, array, attrs=attrs, overwrite=True)
    os.replace(tmp, path)


class VcRenderTifxyzRenderer:
    name = "vc_render_tifxyz"

    def check(self, segment: Segment, profile: ModelProfile) -> list[Issue]:
        issues: list[Issue] = []
        if segment.surface_mesh is None:
            issues.append(Issue("error", "missing-mesh", "render needs a tifxyz directory"))
            return issues
        mesh = Path(segment.surface_mesh)
        for axis in "xyz":
            if not (mesh / f"{axis}.tif").is_file():
                issues.append(Issue("error", "missing-mesh-file", f"missing {axis}.tif in {mesh}"))
        if segment.ct is None:
            issues.append(Issue("error", "missing-ct", "render needs a local OME-Zarr CT"))
        else:
            try:
                pre = ct_preflight(Path(segment.ct))
                if pre["pixel_um"] is None and segment.pixel_um is None:
                    issues.append(
                        Issue(
                            "error",
                            "missing-ct-scale",
                            "CT has no pixel_um/voxel_um in .zattrs; pass --voxel-um",
                        )
                    )
            except ValidationError as exc:
                issues.append(Issue("error", "bad-ct", str(exc)))
        spec = segment.render or RenderSpec()
        if spec.scale <= 0:
            issues.append(Issue("error", "scale", "render --scale must be positive"))
        if spec.slice_step != 1:
            issues.append(Issue("error", "slice-step", "v0 only supports --slice-step 1"))
        if spec.num_slices is not None and spec.num_slices < 1:
            issues.append(Issue("error", "num-slices", "num_slices must be positive"))
        if spec.roi is not None:
            y0, y1, x0, x1 = spec.roi
            if not (0 <= y0 < y1 and 0 <= x0 < x1):
                issues.append(Issue("error", "bad-roi", f"invalid ROI {spec.roi}"))
        try:
            _resolve_binary(spec, RunContext())
        except RenderError as exc:
            issues.append(Issue("error", "missing-renderer", str(exc)))
        if spec.umbilicus is None and spec.orientation == "auto":
            if profile.execution.direction == "both":
                issues.append(
                    Issue(
                        "warning",
                        "unverified-orientation",
                        "no umbilicus: normal direction and reading orientation are "
                        "unverified; direction=both keeps both stacks",
                    )
                )
            else:
                issues.append(
                    Issue(
                        "error",
                        "unverified-orientation",
                        f"{profile.execution.direction}-only profile and no umbilicus: "
                        "pass --umbilicus or --normal inward|outward. The SDK will not "
                        "guess the depth direction for a single-direction map.",
                    )
                )
        return issues

    def render(
        self, segment: Segment, spec: RenderSpec, profile: ModelProfile, ctx: RunContext
    ) -> RenderedVolume:
        errors = [i for i in self.check(segment, profile) if i.is_error]
        if errors:
            raise RenderError("; ".join(i.message for i in errors))
        assert segment.surface_mesh is not None and segment.ct is not None
        mesh = Path(segment.surface_mesh)
        ct = Path(segment.ct)
        binary = _resolve_binary(spec, ctx)
        prefix, apprun = _binary_parts(binary)
        num_slices = spec.num_slices or _derive_num_slices(segment, profile)
        orientation = _orientation(mesh, spec)
        ct = ct_preflight(ct)
        pixel_um = ct["pixel_um"] or segment.pixel_um
        if pixel_um is None:
            raise RenderError("CT has no scale; pass --voxel-um")
        mx, my, mz = _mesh_arrays(mesh)
        valid = (mx > 0) & (my > 0) & (mz > 0) & np.isfinite(mx) & np.isfinite(my) & np.isfinite(mz)
        if not valid.any():
            raise RenderError(f"mesh has no valid coordinates: {mesh}")
        shape_xyz = np.array(ct["shape_zyx"][::-1], dtype=np.float64)
        points = np.stack([mx[valid], my[valid], mz[valid]], axis=-1)
        if np.any(points < 0) or np.any(points > shape_xyz - 1):
            raise RenderError("mesh coordinates fall outside the CT volume")

        mesh_hashes = {axis: sha256_file(mesh / f"{axis}.tif") for axis in "xyz"}
        meta_path = mesh / "meta.json"
        mesh_hashes["meta.json"] = sha256_file(meta_path) if meta_path.is_file() else ""
        umbilicus_hash = sha256_file(spec.umbilicus) if spec.umbilicus else ""
        binary_hash = sha256_file(binary)
        cache_parts = {
            "renderer": "vc_render_tifxyz",
            "binary": str(binary),
            "binary_sha256": binary_hash,
            "mesh": mesh_hashes,
            "ct_zarray": ct["zarray_sha256"],
            "scale": spec.scale,
            "num_slices": num_slices,
            "slice_step": spec.slice_step,
            "orientation": orientation,
            "umbilicus": umbilicus_hash,
        }
        key = render_key(json.dumps(cache_parts, sort_keys=True))
        cache_root = Path(ctx.render_cache) if ctx.render_cache is not None else Path(
            ctx.cache_dir or Path.home() / ".cache" / "ink-sdk"
        ) / "render"
        cache_dir = cache_root / key
        receipt_path = cache_dir / "RENDER_RECEIPT.json"
        if cache_dir.is_dir() and (cache_dir / "0").exists() and receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["cache_hit"] = True
            receipt["roi"] = list(spec.roi) if spec.roi else None
            receipt["roi_application"] = "post-render-in-prepare" if spec.roi else None
            array = _read_render(cache_dir)
            return RenderedVolume(
                path=cache_dir,
                shape_zyx=array.shape,
                pixel_um=tuple(receipt["pixel_um"]),
                orientation=receipt["orientation"],
                receipt=receipt,
                cache_hit=True,
            )

        if not _supports(binary, "--scale"):
            raise RenderError(
                f"{binary} does not advertise --scale; the pinned renderer must "
                "support native-scale output"
            )
        supports_auto_crop = _supports(binary, "--auto-crop")
        supports_flip = _supports(binary, "--flip-normals")
        supports_slice_step = _supports(binary, "--slice-step")

        # Estimate the canvas from the mesh when auto-crop is available. If it
        # is not, the renderer writes the full CT canvas, so use that instead.
        if supports_auto_crop:
            bbox_y = int(np.ceil((float(points[:, 1].max()) - float(points[:, 1].min())) / float(spec.scale))) + 3
            bbox_x = int(np.ceil((float(points[:, 0].max()) - float(points[:, 0].min())) / float(spec.scale))) + 3
        else:
            bbox_y = int(np.ceil(float(shape_xyz[1]) / float(spec.scale)))
            bbox_x = int(np.ceil(float(shape_xyz[2]) / float(spec.scale)))
        estimated = int(num_slices) * max(bbox_y, 1) * max(bbox_x, 1)
        limit = int(float(ctx.max_render_gb) * (1024**3))
        if estimated > limit:
            raise RenderError(
                f"render would allocate about {estimated / 1024**3:.2f} GiB "
                f"(limit {ctx.max_render_gb:g} GiB); pass a larger --max-render-gb "
                "or a smaller mesh"
            )
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            free = shutil.disk_usage(cache_dir.parent).free
        except OSError:
            free = None
        # The stage zarr and the published zarr coexist briefly during publish.
        if free is not None and estimated * 2 > free:
            raise RenderError(
                f"render needs about {estimated * 2 / 1024**3:.2f} GiB free for the "
                f"stage and published zarr, but {cache_dir.parent} has "
                f"{free / 1024**3:.2f} GiB"
            )
        python_flip = bool(orientation["flip_normals"]) and not supports_flip
        cmd = [
            *prefix,
            "-v", str(ct["path"]),
            "-g", "0",
            "--scale", str(float(spec.scale)),
            "-s", str(mesh),
            "-n", str(int(num_slices)),
            "--zarr-output", "{stage}",
            "--zarr-compressor", "none",
        ]
        if supports_slice_step:
            cmd.extend(["--slice-step", str(int(spec.slice_step))])
        if supports_auto_crop:
            cmd.append("--auto-crop")
        if orientation["flip_normals"] and supports_flip:
            cmd.append("--flip-normals")

        with tempfile.TemporaryDirectory(prefix=f".{key}.render-", dir=cache_dir.parent) as tmp:
            stage = Path(tmp) / "render.zarr"
            actual = [str(stage) if part == "{stage}" else part for part in cmd]
            env = dict(os.environ)
            env["OMP_NUM_THREADS"] = str(int(ctx.render_threads))
            ctx.say(f"render: {' '.join(actual)}")
            completed = subprocess.run(
                actual, capture_output=True, text=True, timeout=4 * 3600, env=env
            )
            if completed.returncode != 0:
                raise RenderError(
                    f"vc_render_tifxyz failed with exit {completed.returncode}: "
                    f"{(completed.stderr or completed.stdout)[-2000:]}"
                )
            array = _read_render(stage)
            if array.shape[0] != num_slices:
                raise RenderError(
                    f"renderer wrote {array.shape[0]} slices, expected {num_slices}"
                )
            if python_flip:
                array = array[::-1]
            if orientation["traspuesta"]:
                array = array.transpose(0, 2, 1)
            if orientation["filas_invertidas"]:
                array = array[:, ::-1]
            if orientation["columnas_invertidas"]:
                array = array[:, :, ::-1]
            array = np.ascontiguousarray(array)
            out_px = [float(pixel_um[0] * spec.slice_step), float(pixel_um[1] / spec.scale), float(pixel_um[2] / spec.scale)]
            if orientation["traspuesta"]:
                out_px[1], out_px[2] = out_px[2], out_px[1]
            receipt = {
                "schema": "ink-sdk.render-receipt.v1",
                "generated_at_utc": _utc(),
                "renderer": "vc_render_tifxyz",
                "binary": str(binary),
                "apprun": str(apprun),
                "binary_sha256": binary_hash,
                "command": actual,
                "environment": {"OMP_NUM_THREADS": env["OMP_NUM_THREADS"]},
                "mesh": {"path": str(mesh), "sha256": mesh_hashes, "shape_hw": [int(array.shape[1]), int(array.shape[2])]},
                "ct": {"path": str(ct["path"]), "shape_zyx": ct["shape_zyx"], "dtype": ct["dtype"], "zarray_sha256": ct["zarray_sha256"], "pixel_um": pixel_um},
                "scale": float(spec.scale),
                "num_slices": int(num_slices),
                "slice_step": int(spec.slice_step),
                "orientation": orientation,
                "auto_crop": bool(supports_auto_crop),
                "slice_step_passed": bool(supports_slice_step),
                "flip_normals_python": python_flip,
                "roi": list(spec.roi) if spec.roi else None,
                "roi_application": "post-render-in-prepare" if spec.roi else None,
                "shape_zyx": [int(v) for v in array.shape],
                "pixel_um": out_px,
                "layer_step_um": float(pixel_um[0] * spec.slice_step),
                "cache_key": key,
                "cache_hit": False,
                "stderr_tail": (completed.stderr or "")[-2000:],
            }
            attrs = {
                "axes": ["z", "y", "x"],
                "pixel_um": out_px,
                "ink_sdk_render": receipt,
            }
            _publish(cache_dir, array, attrs, overwrite=True)
            write_json(receipt_path, receipt)
        return RenderedVolume(
            path=cache_dir,
            shape_zyx=tuple(int(v) for v in array.shape),
            pixel_um=tuple(out_px),
            orientation=orientation,
            receipt=receipt,
            cache_hit=False,
        )
