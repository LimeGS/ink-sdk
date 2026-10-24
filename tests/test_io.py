from __future__ import annotations

import json

import numpy as np
import pytest
import tifffile
import zarr

from ink_sdk.errors import ValidationError
from ink_sdk.io import ct_preflight, open_volume_array, pixel_um_from_zarr, write_zarr


def test_npy_and_tiff_are_read_as_zyx(tmp_path):
    array = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    npy = tmp_path / "a.npy"
    np.save(npy, array)
    assert np.array_equal(open_volume_array(npy), array)
    tif = tmp_path / "a.tif"
    tifffile.imwrite(tif, array)
    assert np.array_equal(open_volume_array(tif), array)


def test_zarr_scale_ome_multiscales(tmp_path):
    path = tmp_path / "vol.zarr"
    group = zarr.open_group(str(path), mode="w", zarr_format=2)
    group.create_array("0", shape=(2, 3, 4), dtype="u1")
    group.attrs["multiscales"] = [
        {"datasets": [{"coordinateTransformations": [{"type": "scale", "scale": [2.4, 2.4, 2.4]}]}]}
    ]
    assert pixel_um_from_zarr(path) == (2.4, 2.4, 2.4)


def test_write_zarr_roundtrip_uint8(tmp_path):
    path = tmp_path / "out.zarr"
    array = np.arange(2 * 4 * 4, dtype=np.uint8).reshape(2, 4, 4)
    write_zarr(path, array, attrs={"pixel_um": [1, 1, 1]})
    assert np.array_equal(open_volume_array(path), array)
    assert pixel_um_from_zarr(path) == (1.0, 1.0, 1.0)


def test_ct_preflight_requires_level_zero_uint8(tmp_path):
    path = tmp_path / "ct.zarr"
    group = zarr.open_group(str(path), mode="w", zarr_format=2)
    group.create_array("0", shape=(4, 4, 4), dtype="u1", chunks=(2, 2, 2))
    group.attrs["pixel_um"] = [2.4, 2.4, 2.4]
    report = ct_preflight(path)
    assert report["shape_zyx"] == [4, 4, 4]
    assert report["pixel_um"] == (2.4, 2.4, 2.4)

    meta_path = path / "0" / ".zarray"
    meta = json.loads(meta_path.read_text())
    meta["dtype"] = "<f4"
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValidationError):
        ct_preflight(path)
