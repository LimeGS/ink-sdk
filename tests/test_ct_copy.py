from __future__ import annotations

import functools
import http.server
import json
import threading
from pathlib import Path

import click
import numpy as np
import pytest
import tifffile
import zarr
from typer.testing import CliRunner

from ink_sdk.cli import app
from ink_sdk.contracts import RunContext
from ink_sdk.ct_copy import copy_ct_for_mesh, needed_chunks
from ink_sdk.errors import ValidationError
from ink_sdk.registry import get_profile
from ink_sdk.resolver import build_segment

SHAPE = (48, 40, 40)
CHUNKS = (8, 8, 8)


def _source(path: Path, separator: str = "/") -> np.ndarray:
    """A remote-style Zarr v2 CT array (default compressor) with distinct values per voxel."""
    data = (np.arange(np.prod(SHAPE)) % 251).astype(np.uint8).reshape(SHAPE)
    array = zarr.open_array(
        str(path), mode="w", zarr_format=2, shape=SHAPE, chunks=CHUNKS, dtype="u1",
        dimension_separator=separator,
    )
    array[:] = data
    return data


def _mesh(path: Path, *, z: float = 20.0, meta: dict | None = None) -> Path:
    """A flat 6 x 6 patch at x, y = 12..17 and the given z; one invalid (-1) corner."""
    path.mkdir(parents=True)
    x = np.tile(np.arange(12, 18, dtype=np.float32), (6, 1))
    y = np.tile(np.arange(12, 18, dtype=np.float32)[:, None], (1, 6))
    zz = np.full((6, 6), z, dtype=np.float32)
    x[0, 0] = y[0, 0] = zz[0, 0] = -1
    for name, array in zip("xyz", (x, y, zz)):
        tifffile.imwrite(path / f"{name}.tif", array)
    (path / "meta.json").write_text(json.dumps(meta or {"format": "tifxyz"}))
    return path


def _keys(out: Path) -> set[tuple[int, ...]]:
    array = out / "0"
    return {
        tuple(int(p) for p in f.relative_to(array).as_posix().replace(".", "/").split("/"))
        for f in array.rglob("*")
        if f.is_file() and not f.name.startswith(".")
    }


@pytest.mark.parametrize("separator", ["/", "."])
def test_copies_only_the_chunks_near_the_mesh_byte_for_byte(tmp_path, separator):
    data = _source(tmp_path / "source.zarr", separator)
    mesh = _mesh(tmp_path / "mesh")
    result = copy_ct_for_mesh(mesh, tmp_path / "ct.zarr", source=str(tmp_path / "source.zarr"), voxel_um=9.6, margin=4)
    # z 16..24, y and x 8..21 -> chunk rows 2..3, 1..2, 1..2
    expected = {(z, y, x) for z in (2, 3) for y in (1, 2) for x in (1, 2)}
    assert _keys(tmp_path / "ct.zarr") == expected
    assert result["chunks"] == len(expected) and result["bytes_copied"] > 0
    copy = zarr.open_group(str(tmp_path / "ct.zarr"), mode="r")["0"]
    assert copy.shape == SHAPE and copy.chunks == CHUNKS
    assert np.array_equal(copy[16:32, 8:24, 8:24], data[16:32, 8:24, 8:24])
    assert not copy[0:8, 0:8, 0:8].any()  # chunks far from the surface read as the fill value


def test_copy_is_accepted_as_ink_sdk_ct(tmp_path):
    _source(tmp_path / "source.zarr")
    mesh = _mesh(tmp_path / "mesh")
    out = tmp_path / "ct.zarr"
    copy_ct_for_mesh(mesh, out, source=str(tmp_path / "source.zarr"), voxel_um=9.6, margin=4)
    segment = build_segment(
        profile=get_profile("ink_9um@scrollprize-seed43-step75000"), ctx=RunContext(), tifxyz=mesh, ct=out
    )
    assert segment.pixel_um == (9.6, 9.6, 9.6)  # scale comes from the copy's metadata


def test_source_and_scale_default_to_the_cli_export_metadata(tmp_path):
    _source(tmp_path / "source.zarr")
    meta = {"source_binding": {"ct": {"source": str(tmp_path / "source.zarr"), "spacing_xyz": [9.6, 9.6, 9.6], "origin_xyz": [0, 0, 0]}}}
    mesh = _mesh(tmp_path / "mesh", meta=meta)
    result = copy_ct_for_mesh(mesh, tmp_path / "ct.zarr", margin=0)
    assert result["source"] == str(tmp_path / "source.zarr") and result["pixel_um"] == 9.6
    assert json.loads((tmp_path / "ct.zarr" / ".zattrs").read_text())["pixel_um"] == [9.6, 9.6, 9.6]


def test_refuses_a_mesh_whose_ct_origin_is_not_zero(tmp_path):
    _source(tmp_path / "source.zarr")
    meta = {"source_binding": {"ct": {"source": str(tmp_path / "source.zarr"), "spacing_xyz": [9.6] * 3, "origin_xyz": [5, 0, 0]}}}
    with pytest.raises(ValidationError, match="origin"):
        copy_ct_for_mesh(_mesh(tmp_path / "mesh", meta=meta), tmp_path / "ct.zarr")


def test_requires_a_source_when_the_mesh_records_none(tmp_path):
    with pytest.raises(ValidationError, match="--source"):
        copy_ct_for_mesh(_mesh(tmp_path / "mesh"), tmp_path / "ct.zarr")


def test_refuses_zarr_v3_and_non_uint8_sources(tmp_path):
    zarr.open_array(str(tmp_path / "v3.zarr"), mode="w", zarr_format=3, shape=SHAPE, chunks=CHUNKS, dtype="u1")
    zarr.open_array(str(tmp_path / "f32.zarr"), mode="w", zarr_format=2, shape=SHAPE, chunks=CHUNKS, dtype="f4")
    mesh = _mesh(tmp_path / "mesh")
    with pytest.raises(ValidationError, match="Zarr v3"):
        copy_ct_for_mesh(mesh, tmp_path / "a.zarr", source=str(tmp_path / "v3.zarr"))
    with pytest.raises(ValidationError, match="uint8"):
        copy_ct_for_mesh(mesh, tmp_path / "b.zarr", source=str(tmp_path / "f32.zarr"))


def test_resumes_and_refuses_to_mix_arrays(tmp_path):
    _source(tmp_path / "source.zarr")
    mesh = _mesh(tmp_path / "mesh")
    first = copy_ct_for_mesh(mesh, tmp_path / "ct.zarr", source=str(tmp_path / "source.zarr"), margin=4)
    again = copy_ct_for_mesh(mesh, tmp_path / "ct.zarr", source=str(tmp_path / "source.zarr"), margin=4)
    assert first["bytes_copied"] > 0 and again["bytes_copied"] == 0
    zarr.open_array(str(tmp_path / "other.zarr"), mode="w", zarr_format=2, shape=(8, 8, 8), chunks=CHUNKS, dtype="u1")
    with pytest.raises(ValidationError, match="different array"):
        copy_ct_for_mesh(mesh, tmp_path / "ct.zarr", source=str(tmp_path / "other.zarr"))


def test_needed_chunks_stay_inside_the_array(tmp_path):
    # A mesh at the array's corner with a wide margin must not ask for keys outside it.
    keys = needed_chunks(_mesh(tmp_path / "mesh", z=1.0), list(SHAPE), list(CHUNKS), margin=40)
    assert keys and all(0 <= k[i] < -(-SHAPE[i] // CHUNKS[i]) for k in keys for i in range(3))


def test_copies_over_http_and_treats_missing_chunks_as_fill(tmp_path):
    data = _source(tmp_path / "source.zarr")
    removed = tmp_path / "source.zarr" / "2" / "1" / "1"
    removed.unlink()  # a chunk the source never wrote: the server answers 404
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path))
    handler.log_message = lambda *args: None
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/source.zarr"
        result = copy_ct_for_mesh(_mesh(tmp_path / "mesh"), tmp_path / "ct.zarr", source=url, voxel_um=9.6, margin=4)
    finally:
        server.shutdown()
    assert result["chunks"] == 8 and (2, 1, 1) not in _keys(tmp_path / "ct.zarr")
    copy = zarr.open_group(str(tmp_path / "ct.zarr"), mode="r")["0"]
    assert np.array_equal(copy[24:32, 16:24, 16:24], data[24:32, 16:24, 16:24])
    assert not copy[16:24, 8:16, 8:16].any()


def test_cli_ct_copy_reports_json_and_errors(tmp_path):
    _source(tmp_path / "source.zarr")
    mesh = _mesh(tmp_path / "mesh")
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["ct-copy", "--tifxyz", str(mesh), "--out", str(tmp_path / "ct.zarr"),
         "--source", str(tmp_path / "source.zarr"), "--voxel-um", "9.6", "--margin", "4", "--json"],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["chunks"] == 8
    missing = runner.invoke(app, ["ct-copy", "--tifxyz", str(mesh), "--out", str(tmp_path / "x.zarr")])
    assert missing.exit_code == 2 and "--source" in click.unstyle(missing.output)
