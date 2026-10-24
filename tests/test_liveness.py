from __future__ import annotations

import numpy as np

from ink_sdk.liveness import assess_liveness, forward_reverse_asymmetry


def test_alive_map():
    values = np.linspace(0, 1, 10000, dtype=np.float32).reshape(100, 100)
    report = assess_liveness(values, valid=np.ones_like(values, dtype=bool))
    assert report["verdict"] == "ALIVE"
    assert report["metrics"]["valid_pixels"] == 10000


def test_empty_map():
    report = assess_liveness(np.zeros((10, 10), dtype=np.float32))
    assert report["verdict"] == "EMPTY"


def test_degenerate_map():
    values = np.full((100, 100), 0.5, dtype=np.float32)
    report = assess_liveness(values)
    assert report["verdict"] == "DEGENERATE"
    assert "p99-p50" in report["reason"]


def test_valid_mask_excludes_outside_pixels():
    values = np.zeros((20, 20), dtype=np.float32)
    values[:10] = np.linspace(0, 1, 200).reshape(10, 20)
    valid = np.zeros((20, 20), dtype=bool)
    valid[:10] = True
    report = assess_liveness(values, valid=valid)
    assert report["verdict"] == "ALIVE"
    assert report["metrics"]["valid_pixels"] == 200


def test_forward_reverse_asymmetry():
    forward = np.zeros((40, 40), dtype=np.float32)
    reverse = np.zeros((40, 40), dtype=np.float32)
    forward[:20] = 0.8
    reverse[:10] = 0.8
    report = forward_reverse_asymmetry(forward, reverse)
    assert report is not None
    assert report["sustained_above_1_5"] is True
