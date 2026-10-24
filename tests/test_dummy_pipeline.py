from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ink_sdk.adapters.base import DummyAdapter
from ink_sdk.contracts import RawOutput, RunContext
from ink_sdk.errors import DegenerateMapError, ValidationError
from ink_sdk.registry import get_profile
from ink_sdk.renderers.base import DummyRenderer
from ink_sdk.resolver import build_segment
from ink_sdk.runner import infer


def _segment(tmp_path: Path, mesh: Path, ct: Path, profile):
    ctx = RunContext(
        cache_dir=tmp_path / "cache",
        models_dir=tmp_path / "models",
        workdir=tmp_path / "work",
        overwrite=True,
        on_degenerate="warn",
    )
    segment = build_segment(profile=profile, ctx=ctx, tifxyz=mesh, ct=ct, force_render=True)
    return segment, ctx


def test_dummy_end_to_end_publishes_maps_and_receipt(tmp_path, mesh, ct):
    profile = get_profile("canonical_2um@r152-3ddec-v2-l5-epoch13")
    segment, ctx = _segment(tmp_path, mesh, ct, profile)
    result = infer(
        segment, profile, ctx, out_dir=tmp_path / "out", adapter=DummyAdapter(), renderer=DummyRenderer()
    )
    out = tmp_path / "out"
    assert result.primary == "forward"
    assert result.maps["forward"].shape == (8, 8)
    for name in ("probability.npy", "preview.tif", "INK_SDK_RECEIPT.json"):
        assert (out / name).is_file()
    assert (out / "render" / "0" / ".zarray").is_file()
    assert (out / "render" / "RENDER_RECEIPT.json").is_file()
    receipt = json.loads((out / "INK_SDK_RECEIPT.json").read_text())
    assert receipt["schema"] == "ink-sdk.receipt.v1"
    assert receipt["liveness"]["forward"]["verdict"] == "ALIVE"
    assert receipt["segment"]["render"]["renderer"] == "dummy"
    assert receipt["input"]["scale_plan"]["mode"] == "native"


def test_roi_is_applied_at_output_scale(tmp_path, mesh, ct):
    profile = get_profile("canonical_2um@r152-3ddec-v2-l5-epoch13")
    segment, ctx = _segment(tmp_path, mesh, ct, profile)
    ctx = RunContext(**{**ctx.__dict__, "roi": (2, 6, 3, 7)})
    result = infer(
        segment, profile, ctx, out_dir=tmp_path / "out", adapter=DummyAdapter(), renderer=DummyRenderer()
    )
    assert result.maps["forward"].shape == (4, 4)
    assert result.maps["forward"].origin_px == (3, 2)
    assert np.load(tmp_path / "out" / "probability.npy").shape == (4, 4)
    receipt = result.receipt
    assert receipt["input"]["roi"] == [2, 6, 3, 7]
    assert receipt["input"]["origin_px"] == [3, 2]


def test_degenerate_fails_and_keeps_diagnosis(tmp_path, mesh, ct):
    class ConstantAdapter(DummyAdapter):
        def run(self, prepared, profile, ctx):
            height, width = prepared.array.shape[1:]
            return RawOutput(maps={"forward": np.full((height, width), 0.5, dtype=np.float32)})

    profile = get_profile("canonical_2um@r152-3ddec-v2-l5-epoch13")
    segment, ctx = _segment(tmp_path, mesh, ct, profile)
    ctx = RunContext(**{**ctx.__dict__, "on_degenerate": "fail"})
    with pytest.raises(DegenerateMapError):
        infer(
            segment,
            profile,
            ctx,
            out_dir=tmp_path / "out",
            adapter=ConstantAdapter(),
            renderer=DummyRenderer(),
        )
    assert (tmp_path / "out" / "LANE_NOT_USABLE").is_file()
    assert (tmp_path / "out" / "probability.npy").is_file()


def test_existing_output_needs_overwrite(tmp_path, mesh, ct):
    profile = get_profile("canonical_2um@r152-3ddec-v2-l5-epoch13")
    segment, ctx = _segment(tmp_path, mesh, ct, profile)
    infer(segment, profile, ctx, out_dir=tmp_path / "out", adapter=DummyAdapter(), renderer=DummyRenderer())
    strict = RunContext(**{**ctx.__dict__, "overwrite": False})
    with pytest.raises(ValidationError):
        infer(segment, profile, strict, out_dir=tmp_path / "out", adapter=DummyAdapter(), renderer=DummyRenderer())
