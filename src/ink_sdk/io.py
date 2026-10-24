"""Small, explicit readers and writers for the formats v0 accepts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import zarr

from .contracts import jsonable
from .errors import ValidationError


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _zarr_group(path: Path):
    return zarr.open_group(str(path), mode="r")


def _array_from_group(group):
    if hasattr(group, "shape"):
        return group
    try:
        return group["0"]
    except KeyError as exc:
        raise ValidationError(f"zarr group has no array '0': {group}") from exc


def zarr_attrs(path: Path) -> dict[str, Any]:
    try:
        group = _zarr_group(path)
        return dict(group.attrs)
    except Exception:
        return {}


def _ome_scale(attrs: dict[str, Any]) -> tuple[float, float, float] | None:
    for key in ("pixel_um", "voxel_um", "spacing_um"):
        value = attrs.get(key)
        if value is None:
            continue
        values = value if isinstance(value, (list, tuple)) else [value] * 3
        if len(values) == 1:
            values = [values[0]] * 3
        if len(values) >= 3 and all(float(v) > 0 for v in values[:3]):
            return (float(values[0]), float(values[1]), float(values[2]))
    multiscales = attrs.get("multiscales")
    if isinstance(multiscales, list) and multiscales:
        datasets = multiscales[0].get("datasets") or []
        if datasets:
            transforms = datasets[0].get("coordinateTransformations") or []
            for transform in transforms:
                if transform.get("type") == "scale":
                    values = transform.get("scale") or []
                    if len(values) >= 3 and all(float(v) > 0 for v in values[:3]):
                        return (float(values[0]), float(values[1]), float(values[2]))
    return None


def pixel_um_from_zarr(path: Path) -> tuple[float, float, float] | None:
    attrs = zarr_attrs(path)
    scale = _ome_scale(attrs)
    if scale is not None:
        return scale
    try:
        array = _zarr_group(path)["0"]
        return _ome_scale(dict(array.attrs))
    except Exception:
        return None


def pixel_um_from_sidecar(path: Path) -> tuple[float, float, float] | None:
    for name in ("meta.json", "metadata.json", "surface-volume.json"):
        candidate = Path(path) / name
        if candidate.is_file():
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for key in ("pixel_um", "voxel_um", "spacing_um"):
                if key in data:
                    values = data[key] if isinstance(data[key], (list, tuple)) else [data[key]] * 3
                    if len(values) >= 3:
                        return (float(values[0]), float(values[1]), float(values[2]))
    return None


def open_volume_array(path: Path) -> np.ndarray:
    """Return a ZYX array, without normalising or converting its dtype."""
    path = Path(path)
    if path.suffix == ".npy":
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        return np.asarray(array)
    if path.suffix.lower() in {".tif", ".tiff"}:
        array = tifffile.imread(path)
        return np.asarray(array)
    if path.is_dir():
        if (path / "0" / ".zarray").is_file() or (path / "zarr.json").is_file() or (path / "0" / "zarr.json").is_file():
            try:
                group = _zarr_group(path)
                array = _array_from_group(group)
                return np.asarray(array)
            except Exception as exc:
                raise ValidationError(f"cannot read zarr volume {path}: {exc}") from exc
        tiffs = sorted(
            (p for p in path.iterdir() if p.suffix.lower() in {".tif", ".tiff"} and p.stem.isdigit()),
            key=lambda p: int(p.stem),
        )
        if tiffs:
            return np.stack([np.asarray(tifffile.imread(p)) for p in tiffs])
        # A .zarr directory may have been passed without a top-level group.
        if (path / "0").is_dir():
            try:
                array = zarr.open_array(str(path / "0"), mode="r")
                return np.asarray(array)
            except Exception:
                pass
    raise ValidationError(
        f"unsupported surface volume {path}: expected a .npy, a 3D TIFF, a "
        "numbered TIFF stack, or a zarr group with array '0'"
    )


def validate_uint8_zyx(array: np.ndarray, *, what: str = "surface volume") -> None:
    if array.ndim != 3:
        raise ValidationError(f"{what} must be ZYX (3 dimensions), got {array.shape}")
    if array.dtype != np.uint8:
        raise ValidationError(f"{what} must be uint8 without normalisation, got {array.dtype}")


def write_zarr(
    path: Path,
    array: np.ndarray,
    *,
    attrs: dict[str, Any] | None = None,
    chunks: tuple[int, ...] | None = None,
    overwrite: bool = False,
) -> Path:
    path = Path(path)
    if path.exists() and not overwrite:
        raise ValidationError(f"refusing to overwrite {path}")
    if path.exists():
        import shutil

        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(str(path), mode="w", zarr_format=2)
    if chunks is None:
        chunks = tuple(min(128, size) for size in array.shape)
    group.create_array("0", shape=array.shape, dtype=array.dtype, chunks=chunks)
    group["0"][:] = array
    for key, value in (attrs or {}).items():
        group.attrs[key] = value
    group["0"].attrs.update(attrs or {})
    return path


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def ct_preflight(path: Path) -> dict[str, Any]:
    """Validate the exact level-0 OME-Zarr contract used by the renderer."""
    path = Path(path)
    zarray = path / "0" / ".zarray"
    if not zarray.is_file():
        raise ValidationError(
            f"CT {path} is not an OME-Zarr with a readable 0/.zarray"
        )
    try:
        meta = json.loads(zarray.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read {zarray}: {exc}") from exc
    if meta.get("dtype") != "|u1":
        raise ValidationError(f"CT must be uint8 (dtype |u1), got {meta.get('dtype')!r}")
    shape = meta.get("shape")
    if not isinstance(shape, list) or len(shape) != 3 or any(
        not isinstance(v, int) or v <= 0 for v in shape
    ):
        raise ValidationError(f"CT 0/.zarray has an invalid shape: {shape!r}")
    chunks = meta.get("chunks")
    if (
        not isinstance(chunks, list)
        or len(chunks) != 3
        or any(not isinstance(v, int) or v <= 0 for v in chunks)
    ):
        raise ValidationError(f"CT 0/.zarray has invalid chunks: {chunks!r}")
    attrs = zarr_attrs(path)
    attrs_path = path / "0" / ".zattrs"
    if attrs_path.is_file():
        try:
            attrs = {**attrs, **json.loads(attrs_path.read_text(encoding="utf-8"))}
        except (OSError, json.JSONDecodeError):
            pass
    scale = _ome_scale(attrs)
    return {
        "path": str(path),
        "shape_zyx": shape,
        "chunks_zyx": chunks,
        "dtype": meta.get("dtype"),
        "zarray_sha256": sha256_file(zarray),
        "pixel_um": scale,
        "attrs": attrs,
    }


def ordered_tiff_stack(directory: Path) -> list[Path]:
    files = list(Path(directory).glob("*.tif")) + list(Path(directory).glob("*.tiff"))
    if not files:
        raise ValidationError(f"no TIFF slices in {directory}")
    stems = [p.stem for p in files]
    if not all(s.isdigit() for s in stems) or len({int(s) for s in stems}) != len(stems):
        raise ValidationError("TIFF stems must be unique integers; refusing lexical order")
    return sorted(files, key=lambda p: int(p.stem))
