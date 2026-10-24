"""The 9 um hybrid 3D/2D lane, run by upstream's own inference module.

The checkpoint carries its model and normalisation. Reimplementing the sliding
window here would be a second copy of facts the checkpoint already carries.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import tifffile

from ..contracts import Issue, PreparedInput, RawOutput, RunContext, Segment
from ..errors import InkSDKError, MissingDependencyError
from ..io import write_zarr
from ..profiles import ModelProfile
from .base import BaseAdapter, source_scale_issue, stats


def _villa_python_path(ctx: RunContext) -> tuple[Path, Path]:
    root = ctx.villa or os.environ.get("VILLA")
    if not root:
        raise MissingDependencyError(
            "9 um inference needs a ScrollPrize/villa checkout. Set VILLA or pass "
            "--villa; the pinned revision is in the profile."
        )
    root = Path(root).expanduser()
    candidates = [root / "ink-detection", root]
    for candidate in candidates:
        if (candidate / "koine_machines" / "inference" / "infer.py").is_file():
            return root, candidate
    raise MissingDependencyError(
        f"VILLA={root} has no ink-detection/koine_machines/inference/infer.py. "
        "Point it at the merge-ink-pipelines checkout named by the profile."
    )


def _read_map(path: Path) -> np.ndarray:
    if not path.is_file():
        raise InkSDKError(f"inference reported success but wrote no map: {path}")
    array = tifffile.imread(path)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise InkSDKError(f"upstream map is not 2D: {path} {array.shape}")
    if array.dtype == np.uint8:
        return array.astype(np.float32) / 255.0
    if np.issubdtype(array.dtype, np.floating):
        return array.astype(np.float32)
    raise InkSDKError(f"upstream map has unsupported dtype {array.dtype}: {path}")


class Ink9umAdapter(BaseAdapter):
    name = "ink9um"

    def check(
        self, segment: Segment, profile: ModelProfile, ctx: RunContext | None = None
    ) -> list[Issue]:
        issues: list[Issue] = []
        try:
            _villa_python_path(ctx or RunContext())
        except MissingDependencyError as exc:
            issues.append(Issue("error", "missing-villa", str(exc)))
        checkpoint = profile.checkpoint_path(
            (ctx.models_dir if ctx and ctx.models_dir else Path(os.environ.get("INK_SDK_MODELS", Path.home() / ".cache" / "ink-sdk" / "models")))
        )
        if not checkpoint.is_file():
            issues.append(Issue("error", "missing-checkpoint", f"checkpoint not found: {checkpoint}"))
        scale_issue = source_scale_issue(segment, profile, ctx)
        if scale_issue:
            issues.append(scale_issue)
        if profile.execution.direction not in {"forward", "reverse", "both"}:
            issues.append(Issue("error", "direction", f"unknown direction {profile.execution.direction}"))
        return issues

    def run(
        self, prepared: PreparedInput, profile: ModelProfile, ctx: RunContext
    ) -> RawOutput:
        started = time.time()
        models_dir = ctx.models_dir or Path(
            os.environ.get("INK_SDK_MODELS", Path.home() / ".cache" / "ink-sdk" / "models")
        )
        checkpoint = profile.verify_checkpoint(models_dir)
        villa_root, python_path = _villa_python_path(ctx)
        work = Path(ctx.workdir or ".") / "ink9um"
        work.mkdir(parents=True, exist_ok=True)
        input_path = Path(prepared.path)
        is_zarr = input_path.is_dir() and (
            (input_path / "0" / ".zarray").is_file()
            or (input_path / "zarr.json").is_file()
            or (input_path / "0" / "zarr.json").is_file()
        )
        if not is_zarr:
            # Upstream reads OME-Zarr only; npy/TIFF inputs must be materialised.
            input_path = work / "prepared.zarr"
            write_zarr(
                input_path,
                prepared.array,
                attrs={"pixel_um": list(prepared.pixel_um)},
                overwrite=True,
            )
        output = work / "ink.tif"
        if output.exists():
            output.unlink()
        reverse_output = work / "ink_reverse.tif"
        if reverse_output.exists():
            reverse_output.unlink()

        interpreter = ctx.python or sys.executable
        amp = str(ctx.amp or profile.execution.amp)
        # Upstream names the fp32 path "default"; it has no "fp32" choice.
        upstream_amp = "default" if amp == "fp32" else amp
        command = [
            interpreter,
            "-m",
            "koine_machines.inference.infer",
            str(input_path),
            str(checkpoint),
            str(output),
            "--overlap",
            str(profile.input.overlap),
            "--blend-mode",
            str(profile.execution.blend),
            "--direction",
            str(profile.execution.direction),
            "--batch-size",
            str(profile.execution.batch),
            "--amp-dtype",
            upstream_amp,
            "--no-compile",
        ]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(python_path) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        ctx.say(f"infer: {' '.join(command)}")
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=6 * 3600, env=env
        )
        if completed.returncode != 0:
            tail = (completed.stderr or completed.stdout)[-3000:]
            hint = ""
            if "No module named 'vesuvius'" in tail:
                hint = (
                    " The VILLA inference environment needs the `vesuvius` package "
                    "importable (the 9 um model builder imports it); activate the "
                    "VILLA environment or add it to PYTHONPATH."
                )
            raise InkSDKError(
                f"koine_machines.inference.infer failed with exit {completed.returncode}: "
                f"{tail}{hint}"
            )
        maps: dict[str, np.ndarray] = {}
        expected_shape = prepared.array.shape[1:]
        if profile.execution.direction in {"forward", "both"}:
            forward = _read_map(output)
            if forward.shape != expected_shape:
                raise InkSDKError(f"forward map {forward.shape} != prepared {expected_shape}")
            maps["forward"] = forward
        if profile.execution.direction == "both":
            if not reverse_output.is_file():
                raise InkSDKError("direction=both produced no ink_reverse.tif")
            reverse = _read_map(reverse_output)
            if reverse.shape != expected_shape:
                raise InkSDKError(f"reverse map {reverse.shape} != prepared {expected_shape}")
            maps["reverse"] = reverse
            maps["mean"] = (maps["forward"] + maps["reverse"]) / 2.0
        elif profile.execution.direction == "reverse":
            reverse = _read_map(reverse_output if reverse_output.is_file() else output)
            if reverse.shape != expected_shape:
                raise InkSDKError(f"reverse map {reverse.shape} != prepared {expected_shape}")
            maps["reverse"] = reverse

        villa_revision = ""
        try:
            completed_git = subprocess.run(
                ["git", "-C", str(villa_root), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if completed_git.returncode == 0:
                villa_revision = completed_git.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
        return RawOutput(
            maps=maps,
            # The 9 um lane has a legitimate 0.25 no-ink floor; do not exclude
            # zeros from liveness as if they were missing coverage.
            valid=np.ones_like(next(iter(maps.values())), dtype=bool),
            quantisation={
                "source_dtype": "uint8",
                "levels": 255,
                "scale": 255.0,
                "display_rescale": False,
                "note": "upstream writes uint8 = round(255 * probability); no (p-0.25)/0.5 display rescale",
            },
            receipt={
                "adapter": "ink9um",
                "command": command,
                "input": str(input_path),
                "villa": str(villa_root),
                "villa_revision": villa_revision,
                "checkpoint": str(checkpoint),
                "direction": profile.execution.direction,
                "blend": profile.execution.blend,
                "batch": profile.execution.batch,
                "amp": upstream_amp,
                "amp_requested": amp,
                "runtime_seconds": round(time.time() - started, 2),
                "statistics": {key: stats(value) for key, value in maps.items()},
                "non_claims": [
                    "A probability map is not OCR and does not prove ink or letters.",
                    "This lane is screening-only and target-uncalibrated.",
                ],
            },
        )
