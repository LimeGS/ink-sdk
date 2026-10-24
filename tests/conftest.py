from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
import zarr


def write_zarr_volume(path: Path, shape: tuple[int, int, int], scale: float, *, fill: int = 7) -> Path:
    group = zarr.open_group(str(path), mode="w", zarr_format=2)
    group.create_array("0", shape=shape, dtype="u1", chunks=(min(8, shape[0]), min(16, shape[1]), min(16, shape[2])))
    group["0"][:] = np.full(shape, fill, dtype=np.uint8)
    group.attrs["pixel_um"] = [scale, scale, scale]
    return path


def write_ct(path: Path, shape: tuple[int, int, int], scale: float = 2.4) -> Path:
    group = zarr.open_group(str(path), mode="w", zarr_format=2)
    group.create_array("0", shape=shape, dtype="u1", chunks=(8, 8, 8))
    rng = np.random.default_rng(4)
    group["0"][:] = rng.integers(0, 255, shape, dtype=np.uint8)
    group.attrs["pixel_um"] = [scale, scale, scale]
    return path


def write_mesh(path: Path, height: int = 8, width: int = 8) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    x = np.tile(np.arange(width, dtype=np.float32), (height, 1))
    y = np.tile(np.arange(height, dtype=np.float32)[:, None], (1, width))
    z = np.full((height, width), 4.0, dtype=np.float32)
    tifffile.imwrite(path / "x.tif", x)
    tifffile.imwrite(path / "y.tif", y)
    tifffile.imwrite(path / "z.tif", z)
    (path / "meta.json").write_text(json.dumps({"format": "tifxyz"}), encoding="utf-8")
    return path


@pytest.fixture
def ct(tmp_path: Path) -> Path:
    return write_ct(tmp_path / "ct", (16, 8, 8))


@pytest.fixture
def mesh(tmp_path: Path) -> Path:
    return write_mesh(tmp_path / "mesh")


@pytest.fixture
def surface_volume(tmp_path: Path) -> Path:
    return write_zarr_volume(tmp_path / "surface.zarr", (8, 8, 8), 2.4)
