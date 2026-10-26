import zlib

import numpy as np

from ink_sdk.battery import plan_battery, png_bytes, window
from ink_sdk.registry import list_profiles


def test_window_is_centred_and_spaced():
    assert window(65, 17, 0, 1) == list(range(24, 41))
    assert window(65, 17, -8, 1)[0] == 16
    w = window(65, 17, 0, 2)
    assert w[8] == 32 and w[1] - w[0] == 2
    assert window(21, 17, 8, 2) is None          # needs layers outside the volume


def test_plan_skips_profiles_the_scale_does_not_allow():
    cells, skipped = plan_battery(65, 8.64, list_profiles(), (-4, 0, 4), (1, 2), resample_from_um=8.64)
    ran = {c.profile_id for c in cells}
    assert all(p.startswith("ink_9um") for p in ran)
    assert any("hecate@9.6um" in s["profile"] for s in skipped)
    assert len(cells) == len(ran) * 3 * 2


def test_plan_without_resample_runs_only_native_8_64_profiles():
    cells, skipped = plan_battery(65, 8.64, list_profiles(), (0,), (1,), resample_from_um=None)
    # Reader V3 and dense_native are pinned at 9.362 +- 0.75 um, so 8.64 um is native to both;
    # every other profile needs a resample or is skipped.
    assert {c.profile_id for c in cells} == {"ink_9um@reader-v3", "ink_9um@dense-native-step016000"}
    assert len(skipped) == len(list_profiles()) - 2


def test_reader_v3_profile_matches_its_checkpoint_config():
    from ink_sdk.registry import get_profile
    p = get_profile("ink_9um@reader-v3")
    assert p.adapter == "ink9um" and p.input.depth == 17 and p.input.tile == 384
    assert p.checkpoint.sha256.startswith("ea405417") and p.checkpoint.bytes == 138379659


def test_png_is_valid():
    data = png_bytes(np.zeros((3, 4, 3), np.uint8))
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    ihdr = data[8 + 8: 8 + 8 + 13]
    assert int.from_bytes(ihdr[:4], "big") == 4 and int.from_bytes(ihdr[4:8], "big") == 3
    assert zlib.crc32(data[12:12 + 17]) & 0xFFFFFFFF == int.from_bytes(data[29:33], "big")


def test_plan_pools_depth_at_source_scale():
    # 2.4 um source pooled 4x for the 9 um models: the window must be depth x 4 source layers.
    cells, skipped = plan_battery(65, 2.399, list_profiles(), (0,), (1,), resample_from_um=None)
    by = {c.profile_id: c for c in cells}
    assert len(by["hecate@9.6um"].layers) == 16 * 4          # 64 of 65 layers
    assert len(by["hecate@2.4um"].layers) == 64
    assert not any(p.startswith("ink_9um") for p in by)        # 17 x 4 = 68 > 65
    assert any("17 x pool 4" in s["reason"] for s in skipped)
