from __future__ import annotations

import numpy as np

from ink_sdk.prepare import crop_roi, pool_volume, prepare_input, resample_xy
from ink_sdk.registry import get_profile


def test_pool_uses_mean_and_trims_remainder():
    volume = np.arange(2 * 8 * 8, dtype=np.uint8).reshape(2, 8, 8)
    pooled = pool_volume(volume, (2, 4, 4))
    assert pooled.shape == (1, 2, 2)
    assert pooled[0, 0, 0] == round(volume[:2, :4, :4].mean())


def test_crop_roi_is_half_open():
    volume = np.arange(3 * 5 * 6).reshape(3, 5, 6)
    cropped = crop_roi(volume, (1, 4, 2, 5))
    assert cropped.shape == (3, 3, 3)
    assert cropped[0, 0, 0] == volume[0, 1, 2]


def test_prepare_native_returns_source_and_crop(tmp_path):
    import zarr

    path = tmp_path / "vol.zarr"
    group = zarr.open_group(str(path), mode="w", zarr_format=2)
    group.create_array("0", shape=(4, 8, 8), dtype="u1", chunks=(2, 4, 4))
    group["0"][:] = 3
    group.attrs["pixel_um"] = [2.399, 2.399, 2.399]
    profile = get_profile("canonical_2um@r152-3ddec-v2-l5-epoch13")
    prepared = prepare_input(path, profile, source_um=2.399, cache_dir=tmp_path / "cache")
    assert prepared.array.shape == (4, 8, 8)
    assert prepared.scale_plan.mode == "native"
    cropped = prepare_input(
        path, profile, source_um=2.399, cache_dir=tmp_path / "cache", roi=(2, 6, 1, 7)
    )
    assert cropped.array.shape == (4, 4, 6)
    assert cropped.origin_px == (1, 2)


def test_prepare_pool_writes_cache_and_reuses_it(tmp_path):
    import zarr

    path = tmp_path / "vol.zarr"
    group = zarr.open_group(str(path), mode="w", zarr_format=2)
    group.create_array("0", shape=(8, 8, 8), dtype="u1", chunks=(4, 4, 4))
    group["0"][:] = 9
    group.attrs["pixel_um"] = [2.4, 2.4, 2.4]
    profile = get_profile("hecate@9.6um")
    prepared = prepare_input(path, profile, source_um=2.4, cache_dir=tmp_path / "cache")
    assert prepared.array.shape == (2, 2, 2)
    assert prepared.pixel_um[0] == 9.6
    again = prepare_input(path, profile, source_um=2.4, cache_dir=tmp_path / "cache")
    assert again.receipt["cache_hit"] is True


def test_resample_xy_changes_only_xy():
    volume = np.zeros((2, 10, 10), dtype=np.uint8)
    out = resample_xy(volume, 0.5)
    assert out.shape[0] == 2
    assert out.shape[1] == 5
    assert out.shape[2] == 5
