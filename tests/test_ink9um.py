from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
import tifffile

from ink_sdk.adapters.ink9um import Ink9umAdapter, _read_map, _villa_python_path
from ink_sdk.contracts import PreparedInput, RunContext, ScalePlan
from ink_sdk.errors import MissingDependencyError
from ink_sdk.profiles import ModelProfile
from ink_sdk.registry import get_profile


def _villa(tmp_path: Path) -> Path:
    root = tmp_path / "villa"
    module = root / "ink-detection" / "koine_machines" / "inference"
    module.mkdir(parents=True)
    (module / "infer.py").write_text("")
    return root


def test_villa_path_requires_the_upstream_module(tmp_path):
    with pytest.raises(MissingDependencyError):
        _villa_python_path(RunContext())
    root = _villa(tmp_path)
    _, python_path = _villa_python_path(RunContext(villa=root))
    assert python_path == root / "ink-detection"


def test_read_map_is_uint8_over_255(tmp_path):
    path = tmp_path / "ink.tif"
    tifffile.imwrite(path, np.array([[0, 128, 255]], dtype=np.uint8))
    values = _read_map(path)
    assert values.shape == (1, 3)
    assert values[0, 1] == pytest.approx(128 / 255)


def test_ink9um_runs_upstream_and_reads_both_directions(tmp_path, monkeypatch):
    root = _villa(tmp_path)
    profile = get_profile("ink_9um@scrollprize-seed43-step75000")
    checkpoint = tmp_path / "step.pth"
    checkpoint.write_bytes(b"fake")
    monkeypatch.setattr(ModelProfile, "verify_checkpoint", lambda self, models: checkpoint)
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[0] == "git":
            return subprocess.CompletedProcess(command, 0, "abc123\n", "")
        output = Path(command[5])
        reverse = output.with_name("ink_reverse.tif")
        tifffile.imwrite(output, np.full((8, 8), 128, dtype=np.uint8))
        tifffile.imwrite(reverse, np.full((8, 8), 64, dtype=np.uint8))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    adapter = Ink9umAdapter()
    prepared = PreparedInput(
        path=tmp_path / "prepared.zarr",
        array=np.full((17, 8, 8), 100, dtype=np.uint8),
        pixel_um=(9.6, 9.6, 9.6),
        source=tmp_path / "prepared.zarr",
        scale_plan=ScalePlan(9.6, 9.6, "native"),
        valid=np.ones((8, 8), dtype=bool),
    )
    ctx = RunContext(villa=root, workdir=tmp_path / "work", models_dir=tmp_path, device="cpu")
    raw = adapter.run(prepared, profile, ctx)
    assert set(raw.maps) == {"forward", "reverse", "mean"}
    assert raw.maps["forward"][0, 0] == pytest.approx(128 / 255)
    assert raw.maps["reverse"][0, 0] == pytest.approx(64 / 255)
    assert raw.receipt["direction"] == "both"
    infer_calls = [c for c in calls if len(c) > 2 and c[2] == "koine_machines.inference.infer"]
    assert len(infer_calls) == 1
    assert "--amp-dtype" in infer_calls[0]
    assert "bf16" in infer_calls[0]
