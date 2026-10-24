from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ink_sdk.adapters.canonical2um import Canonical2umAdapter
from ink_sdk.adapters.hecate import HecateAdapter
from ink_sdk.contracts import PreparedInput, RunContext
from ink_sdk.profiles import ModelProfile
from ink_sdk.registry import get_profile


def _prepared(tmp_path: Path, shape, scale=2.4) -> PreparedInput:
    from ink_sdk.contracts import ScalePlan

    array = np.full(shape, 100, dtype=np.uint8)
    path = tmp_path / "prepared.zarr"
    return PreparedInput(
        path=path,
        array=array,
        pixel_um=(scale, scale, scale),
        source=path,
        scale_plan=ScalePlan(scale, scale, "native"),
        valid=np.ones(shape[1:], dtype=bool),
    )


def _allow_checkpoint(monkeypatch, tmp_path: Path):
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"fake")
    monkeypatch.setattr(ModelProfile, "verify_checkpoint", lambda self, models: checkpoint)
    return checkpoint


def test_canonical_adapter_with_fake_model(tmp_path, monkeypatch):
    _allow_checkpoint(monkeypatch, tmp_path)
    profile = get_profile("canonical_2um@r152-3ddec-v2-l5-epoch13")

    class FakeModel:
        def eval(self):
            pass

        def get_output_scale_factor(self):
            return 4

        def forward(self, batch):
            return torch.zeros((batch.shape[0], 1, 64, 64), device=batch.device)

    adapter = Canonical2umAdapter(model_factory=lambda ckpt, device, frames: FakeModel())
    prepared = _prepared(tmp_path, (62, 200, 200), 2.399)
    ctx = RunContext(device="cpu", models_dir=tmp_path, workdir=tmp_path, cache_dir=tmp_path)
    raw = adapter.run(prepared, profile, ctx)
    assert raw.maps["forward"].shape == (200, 200)
    assert np.allclose(raw.maps["forward"], 0.5)
    maps = adapter.to_maps(raw, prepared, profile)
    assert maps["forward"].checkpoint_sha256 == profile.checkpoint.sha256
    assert raw.receipt["normalization"] == "divide-255-no-clip"


def test_hecate_adapter_with_fake_predict(tmp_path, monkeypatch):
    _allow_checkpoint(monkeypatch, tmp_path)
    profile = get_profile("hecate@2.4um")

    class FakeModel:
        patch_size = (64, 256, 256)
        sampling_um = 2.4
        divisor = 200.0
        margin = 1

        def eval(self):
            pass

    import ink_sdk._vendor.hecate as vendor_hecate

    calls = []

    def fake_predict(model, volume, output, reverse=False, stride=None, batch_size=1, precision="fp32", three_d=False):
        calls.append((volume.shape, output.shape, reverse, three_d))
        output[...] = np.linspace(0, 255, output.size, dtype=np.float32).reshape(output.shape).astype(np.uint8)

    monkeypatch.setattr(vendor_hecate, "predict", fake_predict)
    adapter = HecateAdapter(model_factory=lambda ckpt, device: FakeModel())
    prepared = _prepared(tmp_path, (64, 32, 32), 2.4)
    ctx = RunContext(
        device="cpu", models_dir=tmp_path, workdir=tmp_path / "work", output_3d=True
    )
    raw = adapter.run(prepared, profile, ctx)
    assert raw.maps["forward"].shape == (32, 32)
    assert (tmp_path / "work" / "ink_3d.zarr" / "0" / ".zarray").is_file()
    assert raw.volume.shape == (64, 32, 32)
    assert len(calls) == 2
    assert calls[0][3] is False and calls[1][3] is True
    assert raw.volume_attrs["spacing_um"] == [2.4, 2.4, 2.4]
