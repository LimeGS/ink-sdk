from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
import zarr

pytest.importorskip("scipy")
from scipy.ndimage import gaussian_filter, map_coordinates  # noqa: E402

from ink_sdk.adapters.base import DummyAdapter  # noqa: E402
from ink_sdk.contracts import RunContext  # noqa: E402
from ink_sdk.orient import load_mesh, mesh_normals, orient, sheet_side  # noqa: E402
from ink_sdk.registry import get_profile  # noqa: E402
from ink_sdk.renderers.base import DummyRenderer  # noqa: E402
from ink_sdk.resolver import build_segment  # noqa: E402
from ink_sdk.runner import infer  # noqa: E402

CENTRE, RADIUS = 200.0, 150.0


def _cylinder_mesh(path: Path, *, flip_cols: bool = False, h: int = 60, w: int = 60, step: float = 2.0) -> Path:
    """A quarter of a cylinder around a vertical axis at (CENTRE, CENTRE): a rolled sheet."""
    path.mkdir(parents=True, exist_ok=True)
    th = np.arange(w) * step / RADIUS + 0.3
    z = 40.0 + np.arange(h) * step
    T, Z = np.meshgrid(th, z)
    X, Y = CENTRE + RADIUS * np.cos(T), CENTRE + RADIUS * np.sin(T)
    if flip_cols:
        X, Y, Z = X[:, ::-1], Y[:, ::-1], Z[:, ::-1]
    for c, a in zip("xyz", (X, Y, Z)):
        tifffile.imwrite(path / f"{c}.tif", a.astype(np.float32))
    (path / "meta.json").write_text(json.dumps({"format": "tifxyz", "scale": [1 / step, 1 / step]}), encoding="utf-8")
    return path


def _ct(path: Path) -> Path:
    rng = np.random.default_rng(1)
    vol = gaussian_filter(rng.normal(size=(200, 400, 400)).astype(np.float32), 2)
    vol = np.clip(128 + 60 * vol / vol.std(), 0, 255).astype(np.uint8)
    g = zarr.open_group(str(path), mode="w", zarr_format=2)
    g.create_array("0", shape=vol.shape, dtype="u1", chunks=(64, 64, 64))
    g["0"][:] = vol
    return path


def _render(path: Path, mesh: Path, ct: Path, *, along: int, layers: int = 21, px: int = 2) -> Path:
    """Surface volume whose layer k sits at p + along*(k - c)*n, n = d/drow x d/dcol."""
    P, valid, _ = load_mesh(mesh)
    n, _ = mesh_normals(P, valid)
    n[0], n[-1], n[:, 0], n[:, -1] = n[1], n[-2], n[:, 1], n[:, -2]
    up = lambda a: np.repeat(np.repeat(a, px, 0), px, 1)  # noqa: E731
    Pu, nu = up(P), up(n)
    vol = zarr.open_group(str(ct), mode="r")["0"][:].astype(np.float32)
    out = np.empty((layers,) + Pu.shape[:2], np.uint8)
    for k in range(layers):
        q = Pu + along * (k - layers // 2) * nu
        out[k] = np.clip(map_coordinates(vol, [q[..., 2], q[..., 1], q[..., 0]], order=1), 0, 255)
    g = zarr.open_group(str(path), mode="w", zarr_format=2)
    g.create_array("0", shape=out.shape, dtype="u1", chunks=(layers, 32, 32))
    g["0"][:] = out
    g.attrs["pixel_um"] = [9.6, 9.6, 9.6]
    return path


# Like VC3D meshes, this grid has d/drow x d/dcol pointing at the scroll centre.
@pytest.mark.parametrize("flip_cols,expected", [(False, "inward"), (True, "outward")])
def test_sheet_side_follows_the_concave_face(tmp_path, flip_cols, expected):
    P, valid, scale = load_mesh(_cylinder_mesh(tmp_path / "m", flip_cols=flip_cols))
    n, ok = mesh_normals(P, valid)
    radial = P[..., :2] - CENTRE
    truth = "outward" if (n[..., :2] * radial).sum(-1)[ok].mean() > 0 else "inward"
    side = sheet_side(P, valid, scale, distance_vx=60)
    assert side["decided"]
    assert side["normal_points"] == truth == expected


def test_flat_sheet_is_undetermined(tmp_path):
    m = tmp_path / "flat"
    m.mkdir()
    yy, xx = np.mgrid[0:40, 0:40].astype(np.float32) * 2 + 10
    for c, a in zip("xyz", (xx, yy, np.full_like(xx, 50))):
        tifffile.imwrite(m / f"{c}.tif", a)
    (m / "meta.json").write_text(json.dumps({"scale": [0.5, 0.5]}), encoding="utf-8")
    P, valid, scale = load_mesh(m)
    assert sheet_side(P, valid, scale, distance_vx=40)["normal_points"] == "undetermined"


@pytest.mark.parametrize("along,face,primary", [(+1, "outer", "forward"), (-1, "inner", "reverse")])
def test_orient_names_layer0_face_and_the_9um_map(tmp_path, along, face, primary):
    mesh = _cylinder_mesh(tmp_path / "m")
    ct = _ct(tmp_path / "ct.zarr")
    sv = _render(tmp_path / "sv.zarr", mesh, ct, along=along)
    report = orient(mesh, sv, ct, adapter="ink9um", distance_vx=60, win=24, T=14)
    assert report["sheet_side"]["normal_points"] == "inward"
    assert report["stack"]["decided"], report["stack"]
    assert report["stack"]["layers_grow_along"] == ("+n" if along > 0 else "-n")
    assert report["layer0_face"] == face
    assert report["recommended_primary"] == primary


def test_orient_recommends_nothing_for_uncalibrated_adapters(tmp_path):
    mesh = _cylinder_mesh(tmp_path / "m")
    ct = _ct(tmp_path / "ct.zarr")
    sv = _render(tmp_path / "sv.zarr", mesh, ct, along=1)
    report = orient(mesh, sv, ct, adapter="hecate", distance_vx=60, win=24, T=14)
    assert report["layer0_face"] == "outer"
    assert report["recommended_primary"] is None
    assert "no calibrated rule" in report["reason"]


def test_orient_fails_closed_when_ct_does_not_match(tmp_path):
    mesh = _cylinder_mesh(tmp_path / "m")
    ct = _ct(tmp_path / "ct.zarr")
    sv = tmp_path / "noise.zarr"
    g = zarr.open_group(str(sv), mode="w", zarr_format=2)
    g.create_array("0", shape=(21, 120, 120), dtype="u1", chunks=(21, 32, 32))
    g["0"][:] = np.random.default_rng(3).integers(0, 255, (21, 120, 120), dtype=np.uint8)
    report = orient(mesh, sv, ct, adapter="ink9um", distance_vx=60, win=24, T=14)
    assert report["layer0_face"] == "undetermined"
    assert report["recommended_primary"] is None


def _dummy_run(tmp_path, mesh_fixture, ct_fixture, report, primary=None):
    profile = get_profile("ink_9um@scrollprize-seed43-step75000")

    class BothAdapter(DummyAdapter):
        name = "ink9um"

        def run(self, prepared, profile, ctx):
            raw = super().run(prepared, profile, ctx)
            fwd = raw.maps["forward"]
            raw.maps["reverse"] = fwd[::-1].copy()
            return raw


    ctx = RunContext(cache_dir=tmp_path / "cache", models_dir=tmp_path / "models", workdir=tmp_path / "work",
                     overwrite=True, on_degenerate="warn", orient_report=report, primary=primary)
    segment = build_segment(profile=profile, ctx=ctx, tifxyz=mesh_fixture, ct=ct_fixture, force_render=True)
    return infer(segment, profile, ctx, out_dir=tmp_path / "out", adapter=BothAdapter(), renderer=DummyRenderer())


def test_infer_applies_the_report_and_records_it(tmp_path, mesh, ct):
    report = {"schema": "ink-sdk.orient.v1", "adapter": "ink9um", "layer0_face": "inner",
              "recommended_primary": "reverse", "sheet_side": {"normal_points": "outward"},
              "stack": {"layers_grow_along": "+n"}}
    result = _dummy_run(tmp_path, mesh, ct, report)
    assert result.primary == "reverse"
    assert result.receipt["primary_source"] == "orientation-check"
    assert result.receipt["orientation_check"]["applied_primary"] == "reverse"


def test_user_primary_wins_over_the_report(tmp_path, mesh, ct):
    report = {"schema": "ink-sdk.orient.v1", "adapter": "ink9um", "recommended_primary": "reverse"}
    result = _dummy_run(tmp_path, mesh, ct, report, primary="forward")
    assert result.primary == "forward"
    assert result.receipt["primary_source"] == "user"
    assert result.receipt["orientation_check"]["applied_primary"] is None
