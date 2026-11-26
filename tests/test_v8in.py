from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from ink_sdk.errors import InkSDKError

from ink_sdk.adapters import v8in
from ink_sdk.contracts import PreparedInput, RunContext, ScalePlan
from ink_sdk.profiles import ModelProfile
from ink_sdk.registry import get_profile


def _commands(tmp_path, monkeypatch, device, resolved=None, platform="linux", fits=None):
    monkeypatch.setattr(v8in.sys, "platform", platform)
    code = tmp_path / "ink-8um-v8in"
    (code / "ink8um").mkdir(parents=True)
    (code / "predict.py").write_text("")
    checkpoint = code / "model.safetensors"
    checkpoint.write_bytes(b"fake")
    monkeypatch.setattr(ModelProfile, "verify_checkpoint", lambda self, models: checkpoint)
    if resolved is not None:
        monkeypatch.setattr(v8in, "resolve_device", lambda requested: resolved)
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if fits is not None and int(command[command.index("--batch-size") + 1]) > fits:
            return subprocess.CompletedProcess(command, 1, "", "RuntimeError: MPS backend out of memory (MPS allocated: 4.09 GiB)")
        np.save(command[command.index("--output") + 1], np.full((8, 8), 0.5, dtype=np.float32))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    prepared = PreparedInput(path=tmp_path / "p.zarr", array=np.full((28, 8, 8), 100, dtype=np.uint8),
                             pixel_um=(9.362,) * 3, source=tmp_path / "p.zarr",
                             scale_plan=ScalePlan(9.362, 9.362, "native"))
    ctx = RunContext(workdir=tmp_path / "work", models_dir=tmp_path, device=device, python="py")
    raw = v8in.V8inAdapter().run(prepared, get_profile("v8in@ink-8um-v8in"), ctx)
    assert set(raw.maps) == {"forward", "reverse", "mean"}
    _commands.receipt = raw.receipt
    return calls, code / "predict.py"


def test_auto_on_a_mac_runs_predict_on_mps_through_the_runner(tmp_path, monkeypatch):
    calls, script = _commands(tmp_path, monkeypatch, "auto", resolved="mps")
    for command in calls:
        assert command[:3] == ["py", str(v8in.MPS_RUNNER), str(script)]
        assert command[command.index("--device") + 1] == "mps"
    assert v8in.MPS_RUNNER.is_file()


def test_explicit_devices_are_passed_and_run_predict_directly(tmp_path, monkeypatch):
    calls, script = _commands(tmp_path, monkeypatch, "cpu")
    for command in calls:
        assert command[:2] == ["py", str(script)]
        assert command[command.index("--device") + 1] == "cpu"


@pytest.mark.parametrize("resolved", ["cuda", "cpu"])
def test_auto_off_a_mac_is_left_to_predict_py(tmp_path, monkeypatch, resolved):
    calls, script = _commands(tmp_path, monkeypatch, "auto", resolved)
    for command in calls:
        assert command[:2] == ["py", str(script)]
        assert "--device" not in command  # as before: predict.py picks CUDA or the CPU itself


def test_macos_loads_tiles_in_the_main_process(tmp_path, monkeypatch):
    calls, _ = _commands(tmp_path, monkeypatch, "mps", platform="darwin")
    assert all(command[command.index("--workers") + 1] == "0" for command in calls)
    calls, _ = _commands(tmp_path / "linux", monkeypatch, "cuda")
    assert all("--workers" not in command for command in calls)


def _batches(calls):
    return [int(c[c.index("--batch-size") + 1]) for c in calls]


def test_mps_out_of_memory_halves_the_batch_and_keeps_it(tmp_path, monkeypatch):
    calls, _ = _commands(tmp_path, monkeypatch, "mps", fits=2)
    assert _batches(calls) == [8, 4, 2, 2]  # forward 8 -> 4 -> 2, then reverse starts at 2
    assert _commands.receipt["batch"] == 2


def test_out_of_memory_off_mps_is_an_error(tmp_path, monkeypatch):
    with pytest.raises(InkSDKError, match="out of memory"):
        _commands(tmp_path, monkeypatch, "cuda", fits=2)


def test_v8in_reports_the_scan_coverage_as_valid(tmp_path, monkeypatch):
    code = tmp_path / "ink-8um-v8in"
    (code / "ink8um").mkdir(parents=True)
    (code / "predict.py").write_text("")
    checkpoint = code / "model.safetensors"
    checkpoint.write_bytes(b"fake")
    monkeypatch.setattr(ModelProfile, "verify_checkpoint", lambda self, models: checkpoint)

    def fake_run(command, **kwargs):
        np.save(command[command.index("--output") + 1], np.full((8, 8), 0.5, dtype=np.float32))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    scanned = np.zeros((8, 8), dtype=bool)
    scanned[:3] = True
    prepared = PreparedInput(path=tmp_path / "p.zarr", array=np.full((28, 8, 8), 100, dtype=np.uint8),
                             pixel_um=(9.362,) * 3, source=tmp_path / "p.zarr",
                             scale_plan=ScalePlan(9.362, 9.362, "native"), valid=scanned)
    ctx = RunContext(workdir=tmp_path / "work", models_dir=tmp_path, device="cpu", python="py")
    raw = v8in.V8inAdapter().run(prepared, get_profile("v8in@ink-8um-v8in"), ctx)
    assert np.array_equal(raw.valid, scanned)
