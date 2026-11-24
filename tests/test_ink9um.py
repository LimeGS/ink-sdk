from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
import tifffile

from ink_sdk.adapters.ink9um import (
    LEGACY_MODULE,
    MAIN_MODULE,
    Ink9umAdapter,
    _device_args,
    _loader_args,
    _read_map,
    _villa_python_path,
    villa_entrypoint,
)
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


def _villa_main(tmp_path: Path) -> Path:
    root = tmp_path / "villa"
    module = root / "vesuvius" / "src" / "vesuvius" / "ink_detection" / "inference"
    module.mkdir(parents=True)
    (module / "infer.py").write_text("")
    return root


def test_villa_main_layout_is_found(tmp_path):
    root = _villa_main(tmp_path)
    assert villa_entrypoint(root) == (root / "vesuvius" / "src", MAIN_MODULE)
    assert villa_entrypoint(_villa(tmp_path / "legacy")) == (
        tmp_path / "legacy" / "villa" / "ink-detection",
        LEGACY_MODULE,
    )
    assert villa_entrypoint(tmp_path / "empty") is None


def test_read_map_is_uint8_over_255(tmp_path):
    path = tmp_path / "ink.tif"
    tifffile.imwrite(path, np.array([[0, 128, 255]], dtype=np.uint8))
    values = _read_map(path)
    assert values.shape == (1, 3)
    assert values[0, 1] == pytest.approx(128 / 255)


def test_ink9um_runs_upstream_and_reads_both_directions(tmp_path, monkeypatch):
    _run_both_directions(tmp_path, monkeypatch, _villa(tmp_path), LEGACY_MODULE)


def test_ink9um_runs_villa_main_module(tmp_path, monkeypatch):
    _run_both_directions(tmp_path, monkeypatch, _villa_main(tmp_path), MAIN_MODULE)


def _run_both_directions(tmp_path, monkeypatch, root, module):
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
    infer_calls = [c for c in calls if len(c) > 2 and c[2] == module]
    assert len(infer_calls) == 1
    assert raw.receipt["villa_module"] == module
    assert "--amp-dtype" in infer_calls[0]
    assert "bf16" in infer_calls[0]


def test_device_args_follow_what_upstream_infer_supports(tmp_path):
    legacy = _villa(tmp_path / "legacy") / "ink-detection"
    main = _villa_main(tmp_path / "main") / "vesuvius" / "src"
    (main / "vesuvius" / "ink_detection" / "inference" / "infer.py").write_text(
        'parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")\n'
    )
    assert _device_args(main, MAIN_MODULE, "auto") == ([], {})
    assert _device_args(main, MAIN_MODULE, None) == ([], {})
    assert _device_args(main, MAIN_MODULE, "mps") == (["--device", "mps"], {})
    assert _device_args(main, MAIN_MODULE, "cpu") == (["--device", "cpu"], {})
    assert _device_args(main, MAIN_MODULE, "cuda:1") == (["--gpus", "1"], {})
    # The pinned revision has --gpus but no --device.
    assert _device_args(legacy, LEGACY_MODULE, "cuda:1") == (["--gpus", "1"], {})
    assert _device_args(legacy, LEGACY_MODULE, "cpu") == ([], {"CUDA_VISIBLE_DEVICES": ""})
    assert _device_args(legacy, LEGACY_MODULE, "cuda") == ([], {})
    # ...and no MPS path: warn and run as before rather than break existing commands.
    said = []
    assert _device_args(legacy, LEGACY_MODULE, "mps", said.append) == ([], {})
    assert len(said) == 1 and "villa checkout from main" in said[0]


def test_ink9um_passes_device_to_villa_main(tmp_path, monkeypatch):
    root = _villa_main(tmp_path)
    (root / "vesuvius" / "src" / "vesuvius" / "ink_detection" / "inference" / "infer.py").write_text(
        'parser.add_argument("--device")\n'
    )
    profile = get_profile("ink_9um@scrollprize-seed43-step75000")
    checkpoint = tmp_path / "step.pth"
    checkpoint.write_bytes(b"fake")
    monkeypatch.setattr(ModelProfile, "verify_checkpoint", lambda self, models: checkpoint)
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[0] == "git":
            return subprocess.CompletedProcess(command, 0, "abc123\n", "")
        for name in ("ink.tif", "ink_reverse.tif"):
            tifffile.imwrite(Path(command[5]).with_name(name), np.full((8, 8), 128, dtype=np.uint8))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    prepared = PreparedInput(
        path=tmp_path / "prepared.zarr",
        array=np.full((17, 8, 8), 100, dtype=np.uint8),
        pixel_um=(9.6, 9.6, 9.6),
        source=tmp_path / "prepared.zarr",
        scale_plan=ScalePlan(9.6, 9.6, "native"),
        valid=np.ones((8, 8), dtype=bool),
    )
    ctx = RunContext(villa=root, workdir=tmp_path / "work", models_dir=tmp_path, device="mps")
    raw = Ink9umAdapter().run(prepared, profile, ctx)
    infer_call = next(c for c in calls if len(c) > 2 and c[2] == MAIN_MODULE)
    assert infer_call[-2:] == ["--device", "mps"]
    assert raw.receipt["command"][-2:] == ["--device", "mps"]


def test_loader_args_use_threads_on_macos_when_upstream_has_them(tmp_path):
    main = _villa_main(tmp_path / "main") / "vesuvius" / "src"
    infer = main / "vesuvius" / "ink_detection" / "inference" / "infer.py"
    infer.write_text('parser.add_argument("--loader-threads", type=int, default=0)\n')
    assert _loader_args(main, MAIN_MODULE, "darwin") == ["--num-workers", "0", "--loader-threads", "4"]
    assert _loader_args(main, MAIN_MODULE, "linux", shared_memory=8 << 30) == []
    infer.write_text("parser.add_argument('--gpus')\n")
    assert _loader_args(main, MAIN_MODULE, "darwin") == []


def test_loader_args_avoid_worker_processes_when_shared_memory_is_small(tmp_path):
    # Docker gives containers 64 MiB of /dev/shm; DataLoader workers cannot hand tiles over in that.
    main = _villa_main(tmp_path / "main") / "vesuvius" / "src"
    infer = main / "vesuvius" / "ink_detection" / "inference" / "infer.py"
    small, large = 64 << 20, 8 << 30
    infer.write_text('parser.add_argument("--num-workers", type=int)\nparser.add_argument("--loader-threads", type=int)\n')
    assert _loader_args(main, MAIN_MODULE, "linux", shared_memory=small) == ["--num-workers", "0", "--loader-threads", "4"]
    assert _loader_args(main, MAIN_MODULE, "linux", shared_memory=large) == []
    # The revision the 9 um profiles pin has --num-workers but no --loader-threads.
    infer.write_text('parser.add_argument("--num-workers", "--workers", dest="num_workers", type=int, default=4)\n')
    assert _loader_args(main, MAIN_MODULE, "linux", shared_memory=small) == ["--num-workers", "0"]
    infer.write_text("parser.add_argument('--gpus')\n")
    assert _loader_args(main, MAIN_MODULE, "linux", shared_memory=small) == []


def test_model_python_prefers_flag_then_environment_then_this_interpreter(monkeypatch):
    import sys

    from ink_sdk.adapters.base import model_python

    monkeypatch.delenv("INK_SDK_PYTHON", raising=False)
    assert model_python(RunContext()) == sys.executable
    assert model_python(None) == sys.executable
    monkeypatch.setenv("INK_SDK_PYTHON", "/opt/villa/env/bin/python")
    assert model_python(RunContext()) == "/opt/villa/env/bin/python"
    assert model_python(RunContext(python="/flag/python")) == "/flag/python"


def test_ink9um_runs_upstream_with_ink_sdk_python_from_environment(tmp_path, monkeypatch):
    # VILLA needs zarr 2 and ink-sdk zarr 3, so VILLA's model code runs in its own environment.
    monkeypatch.setenv("INK_SDK_PYTHON", "/opt/villa/env/bin/python")
    profile = get_profile("ink_9um@scrollprize-seed43-step75000")
    checkpoint = tmp_path / "step.pth"
    checkpoint.write_bytes(b"fake")
    monkeypatch.setattr(ModelProfile, "verify_checkpoint", lambda self, models: checkpoint)
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[0] == "git":
            return subprocess.CompletedProcess(command, 0, "abc123\n", "")
        for name in ("ink.tif", "ink_reverse.tif"):
            tifffile.imwrite(Path(command[5]).with_name(name), np.full((8, 8), 128, dtype=np.uint8))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    prepared = PreparedInput(
        path=tmp_path / "prepared.zarr",
        array=np.full((17, 8, 8), 100, dtype=np.uint8),
        pixel_um=(9.6, 9.6, 9.6),
        source=tmp_path / "prepared.zarr",
        scale_plan=ScalePlan(9.6, 9.6, "native"),
        valid=np.ones((8, 8), dtype=bool),
    )
    ctx = RunContext(villa=_villa(tmp_path), workdir=tmp_path / "work", models_dir=tmp_path, device="cpu")
    Ink9umAdapter().run(prepared, profile, ctx)
    infer_call = next(c for c in calls if len(c) > 2 and c[2] == LEGACY_MODULE)
    assert infer_call[0] == "/opt/villa/env/bin/python"


def test_doctor_checks_villa_imports_with_the_model_interpreter(tmp_path, monkeypatch):
    from ink_sdk import runner

    root = _villa(tmp_path)
    monkeypatch.setenv("INK_SDK_PYTHON", "/opt/villa/env/bin/python")
    seen = []

    def fake_run(command, **kwargs):
        seen.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    result = runner.doctor(villa=root, models_dir=tmp_path)
    checks = {check["name"]: check for check in result["checks"]}
    assert checks["villa-import"]["ok"]
    assert checks["model-python"]["detail"] == "/opt/villa/env/bin/python"
    assert any(command[0] == "/opt/villa/env/bin/python" for command in seen)
    seen.clear()
    runner.doctor(villa=root, models_dir=tmp_path, python="/flag/python")
    assert any(command[0] == "/flag/python" for command in seen)
