from __future__ import annotations

import hashlib

import numpy as np
import pytest

from ink_sdk.adapters.studio import VENDOR, StudioAdapter, planes
from ink_sdk.contracts import PreparedInput, RunContext, ScalePlan
from ink_sdk.errors import InkSDKError
from ink_sdk.profiles import ModelProfile
from ink_sdk.registry import get_adapter, get_profile

pytest.importorskip("torch")

# byte-identical copies of priyankaranke/vesuvius-studio-multiscroll-cpt@235e3be8, studio-multiscroll-cpt-20261023/hf/
UPSTREAM_SHA256 = {
    "model.py": "efbb390b6208c480fa8e8681137690b7df897c3e406956b9e804fa66645316d7",
    "inference.py": "09752cf7aa0b682c6ca9a46e41bf1f9f279e30d74d35eaab8fc0311504223ec0",
}


def test_vendored_files_are_upstream():
    for name, digest in UPSTREAM_SHA256.items():
        assert hashlib.sha256((VENDOR / name).read_bytes()).hexdigest() == digest


def test_planes_are_upstream_recipe():
    assert planes(31) == list(range(8, 24))       # upstream: render 00-30, stack 23 down to 8
    assert planes(65) == list(range(25, 41))
    assert planes(16) == list(range(16)) and planes(17) == list(range(1, 17))
    with pytest.raises(InkSDKError):
        planes(15)


def test_profile_selects_the_studio_adapter():
    assert isinstance(get_adapter(get_profile("studio@softloss-20261023")), StudioAdapter)


def test_studio_adapter_orders_and_tiling(tmp_path, monkeypatch):
    checkpoint = tmp_path / "w.safetensors"
    checkpoint.write_bytes(b"fake")
    monkeypatch.setattr(ModelProfile, "verify_checkpoint", lambda self, models: checkpoint)
    seen = []

    def fake_predict(model, volume, output, device, batch, amp, stride, blend):
        seen.append((volume[:, 0, 0].tolist(), device, batch, amp, stride, blend))
        return np.full(volume.shape[1:], volume[0, 0, 0] / 255.0, np.float32)

    z = np.arange(31, dtype=np.uint8)[:, None, None] * np.ones((1, 20, 30), np.uint8)
    path = tmp_path / "p.zarr"
    prepared = PreparedInput(path=path, array=z, pixel_um=(9.362,) * 3, source=path,
                             scale_plan=ScalePlan(9.362, 9.362, "native"), valid=np.ones((20, 30), bool))
    profile = get_profile("studio@softloss-20261023")
    adapter = StudioAdapter(model_factory=lambda ckpt, device: object(), predict=fake_predict)
    raw = adapter.run(prepared, profile, RunContext(device="cpu", workdir=tmp_path, models_dir=tmp_path))
    (fwd, *rest_f), (rev, *rest_r) = seen
    assert fwd == list(range(8, 24)) and rev == list(range(23, 7, -1))
    assert rest_f == ["cpu", 8, "fp32", 64, "hann"]          # bf16 falls back to fp32 off CUDA
    assert raw.maps["forward"].shape == (20, 30) and set(raw.maps) == {"forward", "reverse", "mean"}
    assert np.allclose(raw.maps["reverse"], 23 / 255) and raw.receipt["planes"] == list(range(8, 24))
    maps = adapter.to_maps(raw, prepared, profile)
    assert maps["reverse"].checkpoint_sha256 == profile.checkpoint.sha256
