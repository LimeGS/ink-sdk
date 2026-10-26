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
    # Reader V3's card declares 8.64 um scans native; every other profile needs a resample or is skipped.
    assert {c.profile_id for c in cells} == {"ink_9um@reader-v3"}
    assert len(skipped) == len(list_profiles()) - 1


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


def test_battery_page_renders_summary_failures_and_escapes(tmp_path):
    from ink_sdk.battery import to_html

    report = {
        "surface_volume": "/data/seg<1>/sv.zarr", "pixel_um": 9.366, "layers": 28, "centres": [0], "steps": [1],
        "runtime_seconds": 12.0, "images": {}, "non_claims": ["not OCR"],
        "cells": [
            {"profile": "a@x", "centre": 0, "step": 1, "ok": True, "liveness": "ALIVE", "p99": 0.7},
            {"profile": "b@y", "centre": 0, "step": 1, "ok": False, "error": "Boom <bad>"},
        ],
        "skipped": [{"profile": "c@z", "reason": "scale"}],
        "ensemble": {"reading_side": "reverse", "blind_side": "forward", "steps": {"1": {
            "profiles": ["a@x", "b@y"], "centres": [0],
            "images": {"c+0": "ensemble/step1/r.png", "c+0_blind": "ensemble/step1/b.png"}}}},
    }
    page = to_html(report, tmp_path)
    assert "seg&lt;1&gt;" in page and "Boom &lt;bad&gt;" in page and "<bad>" not in page
    assert "FAILED" in page and "ALIVE" in page and "not OCR" in page
    assert 'href="#ensemble"' in page and 'src="ensemble/step1/b.png"' in page
    assert "reading side · reverse" in page and "blind side · forward" in page
