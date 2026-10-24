from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from ink_sdk.contracts import RenderSpec, RunContext, Segment
from ink_sdk.errors import RenderError
from ink_sdk.registry import get_profile
from ink_sdk.renderers.vc_render_tifxyz import VcRenderTifxyzRenderer
from ink_sdk.resolver import build_segment, with_render_spec

FAKE = r'''#!/usr/bin/env python3
import argparse, json, os, sys
import numpy as np, tifffile, zarr
p = argparse.ArgumentParser()
p.add_argument("-v")
p.add_argument("-g")
p.add_argument("--scale")
p.add_argument("-s")
p.add_argument("-n", type=int)
p.add_argument("--slice-step")
p.add_argument("--zarr-output")
p.add_argument("--zarr-compressor")
p.add_argument("--auto-crop", action="store_true")
p.add_argument("--flip-normals", action="store_true")
a = p.parse_args()
x = tifffile.imread(os.path.join(a.s, "x.tif"))
h, w = x.shape
arr = np.empty((a.n, h, w), dtype=np.uint8)
for z in range(a.n):
    arr[z] = (z + np.arange(h)[:, None] + np.arange(w)[None, :]) % 256
g = zarr.open_group(a.zarr_output, mode="w", zarr_format=2)
g.create_array("0", shape=arr.shape, dtype="u1", chunks=(1, h, w))
g["0"][:] = arr
log = os.environ.get("FAKE_RENDER_LOG")
if log:
    with open(log, "a") as fh:
        fh.write(json.dumps(sys.argv) + "\n")
'''


def _fake_renderer(tmp_path: Path) -> Path:
    path = tmp_path / "vc_render_tifxyz"
    path.write_text(FAKE)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def test_render_command_and_cache(tmp_path, mesh, ct, monkeypatch):
    binary = _fake_renderer(tmp_path)
    log = tmp_path / "argv.log"
    monkeypatch.setenv("FAKE_RENDER_LOG", str(log))
    profile = get_profile("ink_9um@scrollprize-seed43-step75000")
    ctx = RunContext(
        renderer=binary,
        cache_dir=tmp_path / "cache",
        workdir=tmp_path / "work",
        render_cache=tmp_path / "render-cache",
        overwrite=True,
    )
    segment = build_segment(profile=profile, ctx=ctx, tifxyz=mesh, ct=ct, force_render=True)
    renderer = VcRenderTifxyzRenderer()
    current = with_render_spec(segment, profile, ctx)
    assert current.render is not None
    assert current.render.num_slices == 68
    rendered = renderer.render(current, current.render, profile, ctx)
    assert rendered.shape_zyx == (68, 8, 8)
    assert rendered.cache_hit is False
    assert (rendered.path / "RENDER_RECEIPT.json").is_file()
    assert (rendered.path / "0" / ".zarray").is_file()
    argv = json.loads(log.read_text().splitlines()[0])
    assert "--scale" in argv and argv[argv.index("--scale") + 1] == "1.0"
    assert "--auto-crop" in argv
    assert "--slice-step" in argv and argv[argv.index("--slice-step") + 1] == "1"

    again = renderer.render(current, current.render, profile, ctx)
    assert again.cache_hit is True
    assert again.shape_zyx == (68, 8, 8)
    assert len(log.read_text().splitlines()) == 1


def test_single_direction_profile_requires_orientation(tmp_path, mesh, ct):
    profile = get_profile("canonical_2um@r152-3ddec-v2-l5-epoch13")
    segment = Segment(
        id="s",
        root=mesh,
        surface_mesh=mesh,
        ct=ct,
        pixel_um=(2.4, 2.4, 2.4),
        render=RenderSpec(num_slices=62, orientation="auto"),
    )
    issues = VcRenderTifxyzRenderer().check(segment, profile)
    orientation = [i for i in issues if i.code == "unverified-orientation"]
    assert orientation and orientation[0].is_error


def test_render_memory_guard_stops_before_the_binary(tmp_path, mesh, ct):
    binary = _fake_renderer(tmp_path)
    ctx = RunContext(
        renderer=binary,
        cache_dir=tmp_path / "cache",
        render_cache=tmp_path / "render-cache",
        max_render_gb=1e-9,
    )
    profile = get_profile("ink_9um@scrollprize-seed43-step75000")
    segment = with_render_spec(
        build_segment(profile=profile, ctx=ctx, tifxyz=mesh, ct=ct, force_render=True),
        profile,
        ctx,
    )
    assert segment.render is not None
    with pytest.raises(RenderError, match="max-render-gb"):
        VcRenderTifxyzRenderer().render(segment, segment.render, profile, ctx)


def test_missing_renderer_is_an_issue(tmp_path, mesh, ct, monkeypatch):
    monkeypatch.delenv("INK_SDK_RENDERER", raising=False)
    monkeypatch.delenv("VC3D_APPRUN", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    profile = get_profile("canonical_2um@r152-3ddec-v2-l5-epoch13")
    segment = Segment(id="s", root=mesh, surface_mesh=mesh, ct=ct, pixel_um=(2.4, 2.4, 2.4))
    issues = VcRenderTifxyzRenderer().check(segment, profile)
    assert any(i.code == "missing-renderer" for i in issues)
