from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from ink_sdk.contracts import PreparedInput, RunContext, ScalePlan, Segment
from ink_sdk.errors import InkSDKError, ModelIntegrityError
from ink_sdk.registry import get_adapter, get_profile

IDS = ("timesformer@gp-scroll1", "timesformer@large-scroll1", "timesformer@scroll5")


def profile(identifier=IDS[0], *, direction="reverse", batch=3):
    result = get_profile(identifier).model_copy(deep=True)
    result.execution.direction = direction
    result.execution.batch = batch
    result.output.primary = "forward" if direction == "both" else direction
    return result


def prepared(tmp_path, array):
    return PreparedInput(tmp_path / "input.npy", array, (7.91,) * 3, tmp_path / "input.npy",
                         ScalePlan(7.91, 7.91, "native"))


class SpatialInk(nn.Module):
    def forward(self, image):
        assert image.ndim == 5 and image.shape[1:] == (1, 26, 64, 64)
        return F.avg_pool2d(image[:, :, 0] * 3 - image[:, :, -1] - 1, 16)


class ConstantInk(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.))

    def forward(self, image):
        return self.bias.expand(image.shape[0], 1, 4, 4)


def pin_fixture(p, path):
    p.checkpoint.path = path.name
    p.checkpoint.bytes = path.stat().st_size
    p.checkpoint.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("identifier", IDS)
def test_profile_is_usable_by_registry_scale_and_pinned_fetch(tmp_path, monkeypatch, identifier):
    import huggingface_hub
    from ink_sdk.fetch import fetch_profile
    from ink_sdk.prepare import plan_scale
    p = profile(identifier)
    assert get_adapter(p).name == "timesformer"
    assert (p.input.depth, p.input.tile, p.input.stride, p.input.clip, p.input.divisor) == (26, 64, 16, 200, 200)
    assert plan_scale(7.91, p).mode == "native"
    with pytest.raises(InkSDKError): plan_scale(8.64, p)
    payload = b"fixture weights"
    p.checkpoint.bytes = len(payload)
    p.checkpoint.sha256 = hashlib.sha256(payload).hexdigest()
    def download(*, repo_id, filename, revision, local_dir):
        assert repo_id == p.source.repo and revision == p.source.revision
        assert filename == ("timesformer_scroll5_27112024_20241108111522_epoch=19.ckpt"
                            if identifier == IDS[2] else "model.safetensors")
        path = Path(local_dir) / filename
        path.write_bytes(payload)
        return str(path)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    destination = fetch_profile(p, models_dir=tmp_path)
    assert destination.read_bytes() == payload and p.verify_checkpoint(tmp_path) == destination


@pytest.mark.parametrize("direction", ["forward", "reverse", "both"])
def test_streamed_adapter_matches_spatial_reference_tiles_and_partial_batches(tmp_path, direction):
    from ink_sdk.adapters.timesformer import TimeSformerAdapter
    p = profile(direction=direction)
    checkpoint = tmp_path / "weights.safetensors"
    checkpoint.write_bytes(b"fixture")
    pin_fixture(p, checkpoint)
    z, y, x = np.indices((31, 87, 101))
    data = ((17*z + 3*y + x) % 256).astype(np.uint8)
    data[:, 20:30, 40:50] = 0
    original = data.copy()
    class StripeOnly:
        shape, dtype, ndim = data.shape, data.dtype, 3
        def __array__(self, *args): raise AssertionError("must not materialise full stack")
        def __getitem__(self, key):
            assert isinstance(key, tuple) and key[1].stop - key[1].start <= 64
            return data[key]
    window = np.outer(np.hanning(64), np.hanning(64))
    expected = {}
    for order in ("forward", "reverse"):
        total, norm = np.zeros((87, 101)), np.zeros((87, 101))
        with torch.inference_mode():
            for oy in (0, 16, 23):
                for ox in (0, 16, 32, 37):
                    tile = data[2:28, oy:oy+64, ox:ox+64]
                    if order == "reverse": tile = tile[::-1]
                    image = np.clip(tile, 0, 200).astype(np.float32) / 200
                    logits = SpatialInk()(torch.from_numpy(image[None, None]))
                    probability = F.interpolate(logits.sigmoid(), (64, 64), mode="bilinear",
                                                align_corners=False)[0, 0].numpy()
                    total[oy:oy+64, ox:ox+64] += probability * window
                    norm[oy:oy+64, ox:ox+64] += window
        valid = (data[2:28] > 0).any(0) & (norm > 0)
        value = np.divide(total, norm, out=np.zeros_like(total), where=norm > 0)
        value[~valid] = 0
        expected[order] = value.astype(np.float32)
    expected["mean"] = (expected["forward"] + expected["reverse"]) / 2
    for batch in (1, 3, 16):
        p.execution.batch = batch
        adapter = TimeSformerAdapter(model_factory=lambda *args: SpatialInk())
        output = adapter.run(prepared(tmp_path, StripeOnly()), p, RunContext(device="cpu", models_dir=tmp_path))
        assert set(output.maps) == ({"forward", "reverse", "mean"} if direction == "both" else {direction})
        for name, values in output.maps.items():
            np.testing.assert_allclose(values, expected[name], rtol=0, atol=1e-7)
        np.testing.assert_array_equal(output.valid, valid)
        assert output.receipt["depth_window"] == [2, 28]
        assert output.receipt["adapter_status"] == "EXPERIMENTAL_PHYSICAL_TRANSFER_UNVALIDATED"
    np.testing.assert_array_equal(data, original)


def test_small_xy_is_padded_then_cropped_without_scoring_zero_ct(tmp_path):
    from ink_sdk.adapters.timesformer import TimeSformerAdapter
    p = profile()
    checkpoint = tmp_path / "weights"
    checkpoint.write_bytes(b"fixture")
    pin_fixture(p, checkpoint)
    array = np.full((26, 32, 40), 100, np.uint8)
    array[:, 10:15, 20:25] = 0
    raw = TimeSformerAdapter(model_factory=lambda *args: ConstantInk()).run(
        prepared(tmp_path, array), p, RunContext(device="cpu", models_dir=tmp_path))
    assert raw.maps["reverse"].shape == (32, 40)
    np.testing.assert_allclose(raw.maps["reverse"][raw.valid], .5)
    assert not raw.valid[10:15, 20:25].any() and not raw.maps["reverse"][~raw.valid].any()


@pytest.mark.parametrize("damage", ["depth", "dtype", "empty", "tile", "divisor", "blend", "amp", "tta", "3d", "scale", "stride", "variant"])
def test_unsupported_inputs_and_settings_fail_closed(tmp_path, damage):
    from ink_sdk.adapters.timesformer import TimeSformerAdapter
    p = profile()
    checkpoint = tmp_path / "weights"
    checkpoint.write_bytes(b"fixture")
    pin_fixture(p, checkpoint)
    array = np.ones((26, 64, 64), np.uint8)
    ctx = RunContext(device="cpu", models_dir=tmp_path)
    if damage == "depth": array = array[:18]
    elif damage == "dtype": array = array.astype(np.float32)
    elif damage == "empty": array[:] = 0
    elif damage == "tile": p.input.tile = 32
    elif damage == "divisor": p.input.divisor = 255
    elif damage == "blend": p.execution.blend = "gaussian"
    elif damage == "amp": ctx = dataclasses.replace(ctx, amp="bf16")
    elif damage == "tta": p.execution.tta = True
    elif damage == "3d": ctx = dataclasses.replace(ctx, output_3d=True)
    elif damage == "stride": p.input.stride = 0
    elif damage == "variant": p.source.repo = "unrecognised/model"
    value = prepared(tmp_path, array)
    if damage == "scale": value = dataclasses.replace(value, pixel_um=(8.64,)*3)
    with pytest.raises(InkSDKError):
        TimeSformerAdapter(model_factory=lambda *args: ConstantInk()).run(value, p, ctx)


@pytest.mark.parametrize("bad_shape", [False, True])
def test_invalid_logits_are_not_published(tmp_path, bad_shape):
    from ink_sdk.adapters.timesformer import TimeSformerAdapter
    p = profile()
    checkpoint = tmp_path / "weights"
    checkpoint.write_bytes(b"fixture")
    pin_fixture(p, checkpoint)
    class Broken(nn.Module):
        def forward(self, image):
            return image.new_full((image.shape[0], 1, 8, 8) if bad_shape else
                                  (image.shape[0], 1, 4, 4), 0 if bad_shape else float("nan"))
    with pytest.raises(InkSDKError):
        TimeSformerAdapter(model_factory=lambda *args: Broken()).run(
            prepared(tmp_path, np.ones((26, 64, 64), np.uint8)), p, RunContext(device="cpu", models_dir=tmp_path))


def test_checkpoint_hash_is_checked_before_model_factory(tmp_path):
    from ink_sdk.adapters.timesformer import TimeSformerAdapter
    def forbidden(*args): raise AssertionError("unverified checkpoint reached loader")
    with pytest.raises(ModelIntegrityError):
        TimeSformerAdapter(model_factory=forbidden).run(prepared(tmp_path, np.ones((26,64,64), np.uint8)),
                                                      profile(), RunContext(device="cpu", models_dir=tmp_path))


def test_archive_training_objects_are_inert_and_unknown_globals_are_rejected(tmp_path, monkeypatch):
    from ink_sdk.adapters.timesformer import _checkpoint_state
    parameter = nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)
    path = tmp_path / "archive.ckpt"
    torch.save({"state_dict": {"bias": torch.tensor(2.)}, "lr_schedulers": [scheduler]}, path)
    def forbidden(*args): raise AssertionError("archived training object methods executed")
    monkeypatch.setattr(torch.optim.AdamW, "__setstate__", forbidden)
    monkeypatch.setattr(torch.optim.lr_scheduler.CosineAnnealingLR, "__setstate__", forbidden, raising=False)
    assert _checkpoint_state(path)["bias"].item() == 2
    torch.save({"state_dict": {"bias": torch.tensor(2.)}, "unsafe": range(5)}, path)
    with pytest.raises(ModelIntegrityError, match="global"): _checkpoint_state(path)


@pytest.mark.parametrize("state", [{}, {"unexpected": torch.tensor(0.)},
                                   {"bias": torch.tensor(0.), "extra": torch.tensor(1.)}])
def test_loader_requires_exact_model_state(tmp_path, monkeypatch, state):
    from safetensors.torch import save_file
    import ink_sdk.adapters.timesformer as adapter
    path = tmp_path / "model.safetensors"
    save_file(state, str(path))
    monkeypatch.setattr(adapter, "_build_model", lambda p: ConstantInk())
    with pytest.raises(ModelIntegrityError): adapter._default_model_factory(path, "cpu", profile())


def test_optional_model_dependencies_are_not_imported_by_registry():
    code = """import sys
class BlockModels:
 def find_spec(self,fullname,*args):
  if fullname.split('.')[0] in ('torch','safetensors','timesformer_pytorch'):raise ImportError('optional model dependency blocked')
sys.meta_path.insert(0,BlockModels())
from ink_sdk.registry import list_profiles,get_adapter
profiles=[p for p in list_profiles() if p.adapter=='timesformer']
assert len(profiles)==3
assert all(get_adapter(p).name=='timesformer' for p in profiles)
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_check_reports_bad_scale_missing_checkpoint_and_experimental_transfer(tmp_path):
    p = profile()
    segment = Segment("example", tmp_path, surface_volume=tmp_path/"volume", pixel_um=(8.64,)*3, depth=18, dtype="uint8")
    issues = get_adapter(p).check(segment, p, RunContext(models_dir=tmp_path))
    assert {i.code for i in issues if i.is_error} >= {"incompatible-scale", "missing-checkpoint", "insufficient-depth"}
    assert any(i.code == "experimental-transfer" and not i.is_error for i in issues)


def test_full_sdk_pipeline_writes_reverse_map_and_receipt(tmp_path, monkeypatch):
    from safetensors.torch import save_file
    import ink_sdk.adapters.timesformer as adapter
    from ink_sdk.runner import infer
    p = profile()
    checkpoint = tmp_path / "model.safetensors"
    save_file(ConstantInk().state_dict(), str(checkpoint))
    pin_fixture(p, checkpoint)
    monkeypatch.setattr(adapter, "_build_model", lambda p: SpatialInk())
    # Keep a real learned state in the model used by the default loader.
    class LearnedSpatial(SpatialInk):
        def __init__(self): super().__init__(); self.bias = nn.Parameter(torch.tensor(0.))
    monkeypatch.setattr(adapter, "_build_model", lambda p: LearnedSpatial())
    z, y, x = np.indices((26, 80, 96))
    array = ((17*z + 3*y + x) % 256).astype(np.uint8)
    array[:,20:30,40:50] = 0
    volume = tmp_path / "input.npy"
    np.save(volume, array)
    segment = Segment("test", tmp_path, surface_volume=volume, pixel_um=(7.91,)*3, depth=26, dtype="uint8")
    output = tmp_path / "result"
    result = infer(segment, p, RunContext(device="cpu", models_dir=tmp_path, on_degenerate="warn"), out_dir=output)
    assert result.primary == "reverse" and result.primary_map.shape == (80,96)
    np.testing.assert_array_equal(np.load(output/"probability_reverse.npy"), result.primary_map.array)
    receipt = json.loads((output/"INK_SDK_RECEIPT.json").read_text())
    assert receipt["checkpoint"]["verified"] is True
    assert receipt["execution"]["adapter_status"] == "EXPERIMENTAL_PHYSICAL_TRANSFER_UNVALIDATED"
    assert not result.primary_map.valid[20:30,40:50].any()
