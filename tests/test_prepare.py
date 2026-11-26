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


def test_transformed_input_without_cache_gets_a_materialised_zarr(tmp_path):
    import zarr

    path = tmp_path / "vol.zarr"
    group = zarr.open_group(str(path), mode="w", zarr_format=2)
    group.create_array("0", shape=(4, 8, 8), dtype="u1", chunks=(2, 4, 4))
    group["0"][:] = 3
    group.attrs["pixel_um"] = [2.4, 2.4, 2.4]
    prepared = prepare_input(
        path,
        get_profile("canonical_2um@r152-3ddec-v2-l5-epoch13"),
        source_um=2.4,
        cache_dir=None,
        roi=(1, 3, 2, 6),
    )
    assert prepared.array.shape == (4, 2, 4)
    assert prepared.path != path
    assert (prepared.path / "0" / ".zarray").is_file()


def test_prepare_cache_misses_after_the_source_is_rewritten_in_place(tmp_path):
    import zarr

    from ink_sdk.io import write_zarr

    source = tmp_path / "sv.zarr"
    write_zarr(source, np.full((16, 64, 64), 10, np.uint8))
    profile = get_profile("studio@softloss-20261023")
    first = prepare_input(source, profile, source_um=9.362, cache_dir=tmp_path / "cache", roi=(0, 32, 0, 32))
    assert int(first.array[0, 0, 0]) == 10
    zarr.open(str(source / "0"), mode="r+")[:] = 200  # a renderer rewriting the chunks of the same segment
    second = prepare_input(source, profile, source_um=9.362, cache_dir=tmp_path / "cache", roi=(0, 32, 0, 32))
    assert second.receipt["cache_hit"] is False
    assert int(np.asarray(second.array)[0, 0, 0]) == 200
    third = prepare_input(source, profile, source_um=9.362, cache_dir=tmp_path / "cache", roi=(0, 32, 0, 32))
    assert third.receipt["cache_hit"] is True


def test_source_identity_ignores_a_file_it_cannot_stat_but_still_sees_rewrites(tmp_path):
    from ink_sdk.prepare import _source_identity

    source = tmp_path / "sv.zarr"
    (source / "0").mkdir(parents=True)
    (source / "0" / "0.0.0").write_bytes(b"a" * 8)
    (source / "dangling").symlink_to(tmp_path / "missing")  # os.stat fails on it
    before = _source_identity(source)
    assert len(before) == 3  # not the bare-path fallback
    (source / "0" / "0.0.0").write_bytes(b"b" * 9)
    assert _source_identity(source) != before


def test_an_interrupted_prepare_is_redone_not_read_back_as_zeros(tmp_path, monkeypatch):
    import pytest
    import zarr

    import ink_sdk.prepare as prepare_module

    path = tmp_path / "vol.zarr"
    group = zarr.open_group(str(path), mode="w", zarr_format=2)
    group.create_array("0", shape=(8, 8, 8), dtype="u1", chunks=(4, 4, 4))
    group["0"][:] = 9
    group.attrs["pixel_um"] = [2.4, 2.4, 2.4]
    profile = get_profile("hecate@9.6um")
    real = prepare_module.write_zarr

    class CutOff:
        def __init__(self, volume):
            self.shape, self.dtype, self.ndim = volume.shape, volume.dtype, volume.ndim

        def __getitem__(self, key):
            raise RuntimeError("interrupted")

        def __array__(self, *args, **kwargs):
            raise RuntimeError("interrupted")

    monkeypatch.setattr(prepare_module, "write_zarr", lambda p, volume, **kw: real(p, CutOff(volume), **kw))
    with pytest.raises(RuntimeError, match="interrupted"):
        prepare_input(path, profile, source_um=2.4, cache_dir=tmp_path / "cache")
    monkeypatch.setattr(prepare_module, "write_zarr", real)
    again = prepare_input(path, profile, source_um=2.4, cache_dir=tmp_path / "cache")
    assert again.receipt["cache_hit"] is False
    assert int(np.asarray(again.array).min()) == 9
