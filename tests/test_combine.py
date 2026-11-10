from __future__ import annotations

import json

import numpy as np
import pytest

from ink_sdk.combine import combine


def _run(tmp_path, name, forward, reverse=None):
    d = tmp_path / name
    d.mkdir()
    np.save(d / "probability.npy", np.asarray(forward, np.float32))
    if reverse is not None:
        np.save(d / "probability_reverse.npy", np.asarray(reverse, np.float32))
    return d


def test_product_is_pixelwise_and_writes_a_receipt(tmp_path):
    a = _run(tmp_path, "a", [[0.5, 1.0], [0.2, 0.0]])
    b = _run(tmp_path, "b", [[0.5, 0.5], [1.0, 0.9]])
    receipt = combine([a, b], "product", "forward", tmp_path / "out")
    out = np.load(tmp_path / "out" / "probability.npy")
    assert np.allclose(out, [[0.25, 0.5], [0.2, 0.0]])
    assert receipt["mode"] == "product"
    assert json.loads((tmp_path / "out" / "combine_forward.json").read_text())["shape"] == [2, 2]


def test_mean_of_three_seeds_and_resize(tmp_path):
    seeds = [_run(tmp_path, f"s{i}", np.full((4, 4), v), np.full((4, 4), 1 - v)) for i, v in enumerate((0.1, 0.2, 0.6))]
    small = _run(tmp_path, "small", np.full((2, 2), 0.3), np.full((2, 2), 0.3))
    combine(seeds, "mean", "reverse", tmp_path / "mean")
    assert np.allclose(np.load(tmp_path / "mean" / "probability_reverse.npy"), 1 - 0.3)
    combine([seeds[0], small], "product", "forward", tmp_path / "mixed")
    assert np.load(tmp_path / "mixed" / "probability.npy").shape == (4, 4)


def test_combine_rejects_bad_input(tmp_path):
    a = _run(tmp_path, "a", [[0.5]])
    with pytest.raises(ValueError):
        combine([a], "product", "forward", tmp_path / "o")
    with pytest.raises(ValueError):
        combine([a, a], "max", "forward", tmp_path / "o")
    with pytest.raises(FileNotFoundError):
        combine([a, a], "product", "reverse", tmp_path / "o")
