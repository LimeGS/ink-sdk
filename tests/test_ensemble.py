from __future__ import annotations

import numpy as np
import pytest

from ink_sdk.ensemble import ensemble_battery, map_values, resize_bilinear


def test_villa_values_map_back_to_erwins_scale():
    v = np.array([0, 64, 128, 192, 255], np.float32)
    assert np.allclose(map_values(v / 255, "ink9um"), np.clip((v - 64) / 128, 0, 1))
    assert np.allclose(map_values(v / 255, "hecate"), v / 255)


def test_resize_bilinear():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert resize_bilinear(a, (3, 4)) is not None and np.array_equal(resize_bilinear(a, (3, 4)), a)
    assert np.allclose(resize_bilinear(np.full((5, 7), 2.5), (11, 3)), 2.5)
    up = resize_bilinear(a, (6, 8))
    assert up.shape == (6, 8) and up.min() >= a.min() and up.max() <= a.max()


def _cell(tmp, name, centre, fwd, rev):
    d = tmp / "cells" / name / f"c{centre:+d}_s1"
    d.mkdir(parents=True)
    np.save(d / "probability.npy", fwd.astype(np.float32))
    np.save(d / "probability_reverse.npy", rev.astype(np.float32))
    return {"profile": name, "centre": centre, "step": 1, "ok": True, "dir": str(d)}


def test_ensemble_matches_the_recipe(tmp_path):
    rng = np.random.default_rng(0)
    h, w = 40, 48
    thick = np.full((9, h, w), 100, np.uint8)
    thick[3, :5, :5] = 0                       # invalid corner: CT 0 in one layer
    valid = (thick > 0).all(0)
    maps = {}
    results = []
    for name, shape in (("a", (h, w)), ("b", (h, w)), ("h", (h // 2, w // 2))):
        for c in (-1, 0, 1):
            f, r = rng.random(shape), rng.random(shape)
            maps[(name, c)] = (f, r)
            results.append(_cell(tmp_path, name, c, f, r))
    results.append({"profile": "a", "centre": 2, "step": 1, "ok": False})
    adapters = {"a": "ink9um", "b": "ink9um", "h": "hecate"}
    rep = ensemble_battery(thick, results, adapters, tmp_path, "reverse")

    e = rep["steps"]["1"]
    assert e["profiles"] == ["a", "b", "h"] and e["centres"] == [-1, 0, 1]
    # independent recomputation: pooled reverse statistics, z-score mean
    st = {}
    for name in "abh":
        vals = np.concatenate([resize_bilinear(map_values(maps[(name, c)][1], adapters[name]), (h, w))[valid]
                               for c in (-1, 0, 1)]).astype(np.float64)
        st[name] = (vals.mean(), vals.std() + 1e-6)
        assert np.isclose(e["per_profile"][name]["mean"], st[name][0])
        assert np.isclose(e["per_profile"][name]["sd"], st[name][1])
    for side, suffix, k in (("reverse", "", 1), ("forward", "_blind", 0)):
        z = np.mean([(resize_bilinear(map_values(maps[(n, 0)][k], adapters[n]), (h, w)) - st[n][0]) / st[n][1]
                     for n in "abh"], 0)
        got = np.load(tmp_path / "ensemble" / "step1" / f"ensemble_reverse_c+0{suffix}.npy")
        assert np.allclose(got, z, atol=1e-5)
        assert (tmp_path / e["images"][f"c+0{suffix}"]).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert e["p1"] < e["p99_5"]


def test_ensemble_needs_two_profiles(tmp_path):
    thick = np.full((5, 8, 8), 9, np.uint8)
    one = [_cell(tmp_path, "a", 0, np.zeros((8, 8)), np.ones((8, 8)))]
    rep = ensemble_battery(thick, one, {"a": "ink9um"}, tmp_path, "forward")
    assert "no ensemble" in rep["steps"]["1"]["note"]
    with pytest.raises(ValueError):
        ensemble_battery(thick, one, {"a": "ink9um"}, tmp_path, "both")
