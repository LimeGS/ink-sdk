"""Copy only the CT chunks a tifxyz mesh needs into a local OME-Zarr.

Rendering a mesh needs local CT, but a whole scan level can be terabytes. A mesh
touches only the chunks near its surface, so this copies those chunks byte for byte
from a remote (http, https, public s3://) or local Zarr v2 array into a local group
with the same shape, chunking, compressor and indices. Mesh coordinates therefore
stay valid; chunks that are not copied read as the array's fill value.
"""

from __future__ import annotations

import concurrent.futures
import json
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .errors import ValidationError
from .renderers.vc_render_tifxyz import _mesh_arrays

DEFAULT_MARGIN = 32


def _https(source: str) -> str:
    if source.startswith("s3://"):
        bucket, _, key = source[len("s3://") :].partition("/")
        return f"https://{bucket}.s3.amazonaws.com/{key}"
    return source


class _Source:
    """Read files of a Zarr v2 array from a URL or a local directory."""

    def __init__(self, location: str):
        self.location = _https(str(location)).rstrip("/")
        self.remote = self.location.startswith(("http://", "https://"))
        if not self.remote and "://" in self.location:
            raise ValidationError(f"unsupported CT source {location}: use http(s)://, s3:// or a local path")

    def read(self, name: str) -> bytes | None:
        if self.remote:
            try:
                with urllib.request.urlopen(f"{self.location}/{name}", timeout=120) as response:
                    return response.read()
            except urllib.error.HTTPError as error:
                if error.code in (403, 404):  # never written upstream: reads as the fill value
                    return None
                raise
        path = Path(self.location) / name
        return path.read_bytes() if path.is_file() else None


def source_from_mesh(mesh: Path) -> tuple[str | None, float | None]:
    """The CT array and voxel size a winding-field-cli export records in meta.json, if any."""
    meta_path = Path(mesh) / "meta.json"
    if not meta_path.is_file():
        return None, None
    meta = json.loads(meta_path.read_text())
    ct = (meta.get("source_binding") or {}).get("ct") or {}
    origin = ct.get("origin_xyz")
    if origin is not None and any(float(value) != 0 for value in origin):
        raise ValidationError(
            f"{meta_path} records a CT origin of {origin}; mesh coordinates are not array indices"
        )
    spacing = ct.get("spacing_xyz")
    return ct.get("source"), (float(spacing[0]) if spacing else None)


def needed_chunks(
    mesh: Path, shape: list[int], chunks: list[int], margin: int
) -> list[tuple[int, int, int]]:
    """ZYX chunk keys within ``margin`` voxels of the mesh's valid points."""
    x, y, z = (array.astype(np.float64) for array in _mesh_arrays(mesh))
    valid = (x >= 0) & (y >= 0) & (z >= 0) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if not valid.any():
        raise ValidationError(f"mesh has no valid coordinates: {mesh}")
    points = np.stack([z[valid], y[valid], x[valid]], axis=1)
    keys: set[tuple[int, int, int]] = set()
    steps = (-margin, 0, margin) if margin else (0,)
    for dz in steps:
        for dy in steps:
            for dx in steps:
                index = np.floor((points + [dz, dy, dx]) / chunks).astype(np.int64)
                keys.update(map(tuple, np.unique(index, axis=0).tolist()))
    limit = [-(-size // chunk) for size, chunk in zip(shape, chunks)]
    return sorted(k for k in keys if all(0 <= k[i] < limit[i] for i in range(3)))


def copy_ct_for_mesh(
    mesh: Path,
    out: Path,
    *,
    source: str | None = None,
    voxel_um: float | None = None,
    margin: int = DEFAULT_MARGIN,
    workers: int = 16,
    say: Callable[[str], None] = lambda message: None,
) -> dict[str, Any]:
    """Copy the chunks of ``source`` that ``mesh`` needs into the OME-Zarr group ``out``.

    ``source`` and ``voxel_um`` default to what a winding-field-cli export records in the
    mesh's meta.json. Mesh coordinates must be voxel indices into ``source``. Existing
    chunks in ``out`` are kept, so an interrupted copy can be resumed.
    """
    mesh, out = Path(mesh), Path(out)
    recorded_source, recorded_um = source_from_mesh(mesh)
    source = source or recorded_source
    voxel_um = voxel_um or recorded_um
    if not source:
        raise ValidationError(f"{mesh} records no CT source; pass --source")
    if margin < 0:
        raise ValidationError("margin must be zero or positive")
    reader = _Source(source)
    header = reader.read(".zarray")
    if header is None:
        if reader.read("zarr.json") is not None:
            raise ValidationError(f"{source} is Zarr v3; ink-sdk CT input must be Zarr v2")
        raise ValidationError(f"no Zarr v2 array at {source} (expected .zarray)")
    zarray = json.loads(header)
    if zarray.get("dtype") != "|u1" or len(zarray.get("shape", [])) != 3:
        raise ValidationError(f"{source} must be a 3D uint8 array, got {zarray.get('dtype')} {zarray.get('shape')}")
    shape, chunks = zarray["shape"], zarray["chunks"]
    separator = zarray.get("dimension_separator") or "."

    array_dir = out / "0"
    existing = array_dir / ".zarray"
    if existing.is_file() and json.loads(existing.read_text()) != zarray:
        raise ValidationError(f"{out} already holds a different array; choose another --out")
    array_dir.mkdir(parents=True, exist_ok=True)
    existing.write_text(json.dumps(zarray, indent=2))
    (out / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    attrs: dict[str, Any] = {"ink_sdk_ct_copy": {"source": source, "margin_voxels": margin}}
    if voxel_um:
        attrs["pixel_um"] = [voxel_um] * 3
        attrs["multiscales"] = [
            {
                "version": "0.4",
                "axes": [{"name": axis, "type": "space", "unit": "micrometer"} for axis in "zyx"],
                "datasets": [
                    {"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [voxel_um] * 3}]}
                ],
            }
        ]
    (out / ".zattrs").write_text(json.dumps(attrs, indent=2))

    keys = needed_chunks(mesh, shape, chunks, margin)
    say(f"{len(keys)} chunks near the mesh (about {len(keys) * int(np.prod(chunks)) / 1e9:.2f} GB uncompressed)")

    def fetch(key: tuple[int, int, int]) -> int:
        name = separator.join(map(str, key))
        target = array_dir / name
        if target.is_file():
            return 0
        data = reader.read(name)
        if data is None:
            return 0
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        partial.write_bytes(data)
        partial.replace(target)
        return len(data)

    copied = 0
    with concurrent.futures.ThreadPoolExecutor(max(1, workers)) as pool:
        for done, size in enumerate(pool.map(fetch, keys), 1):
            copied += size
            if done % 200 == 0 or done == len(keys):
                say(f"{done}/{len(keys)} chunks, {copied / 1e9:.2f} GB copied")
    return {
        "out": str(out),
        "source": source,
        "shape_zyx": shape,
        "chunks_zyx": chunks,
        "pixel_um": voxel_um,
        "margin_voxels": margin,
        "chunks": len(keys),
        "bytes_copied": copied,
    }
