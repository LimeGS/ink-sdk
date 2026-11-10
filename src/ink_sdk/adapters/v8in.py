"""The v8-in lane (Youssef Nader, YoussefMoNader/ink-8um-v8in) and its fine-tunes, e.g. A-hn (Prodigle/pherc0306b-ink-ahn).

Runs the model repository's own predict.py, as the 9 um lane runs upstream's infer: the checkpoint folder
(config.json + model.safetensors) carries the model, and predict.py carries the tiling (64 px tiles, stride 21,
Gaussian stitch) and the choice of the central 24 layers. Reimplementing that here would be a second copy.

predict.py lives in the v8-in model repository. A fine-tune such as A-hn ships only weights + config, so the
code comes from V8IN_CODE (or <models>/ink-8um-v8in/predict.py).
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
from ..profiles import ModelProfile
from .base import BaseAdapter, source_scale_issue, stats


def _models_dir(ctx: RunContext | None) -> Path:
    if ctx is not None and ctx.models_dir:
        return Path(ctx.models_dir)
    return Path(os.environ.get("INK_SDK_MODELS", Path.home() / ".cache" / "ink-sdk" / "models"))


def _predict_py(ctx: RunContext | None) -> Path:
    env = os.environ.get("V8IN_CODE")
    candidates = [Path(env).expanduser()] if env else []
    candidates.append(_models_dir(ctx) / "ink-8um-v8in")
    for c in candidates:
        script = c / "predict.py" if c.is_dir() else c
        if script.is_file() and (script.parent / "ink8um").is_dir():
            return script
    raise MissingDependencyError(
        "the v8-in lane needs predict.py and the ink8um package from YoussefMoNader/ink-8um-v8in. "
        "Run `ink-sdk fetch v8in@ink-8um-v8in` or set V8IN_CODE to that checkout."
    )


class V8inAdapter(BaseAdapter):
    name = "v8in"

    def check(self, segment: Segment, profile: ModelProfile, ctx: RunContext | None = None) -> list[Issue]:
        issues: list[Issue] = []
        try:
            _predict_py(ctx)
        except MissingDependencyError as exc:
            issues.append(Issue("error", "missing-v8in-code", str(exc)))
        checkpoint = profile.checkpoint_path(_models_dir(ctx))
        if not checkpoint.is_file():
            issues.append(Issue("error", "missing-checkpoint", f"checkpoint not found: {checkpoint}"))
        elif not (checkpoint.parent / "config.json").is_file():
            issues.append(Issue("error", "missing-config", f"no config.json next to {checkpoint}"))
        scale_issue = source_scale_issue(segment, profile, ctx)
        if scale_issue:
            issues.append(scale_issue)
        return issues

    def run(self, prepared: PreparedInput, profile: ModelProfile, ctx: RunContext) -> RawOutput:
        started = time.time()
        layers = int(prepared.array.shape[0])
        if layers < int(profile.input.depth):
            raise InkSDKError(f"{profile.id} needs {profile.input.depth} layers, the prepared volume has {layers}")
        checkpoint = profile.verify_checkpoint(_models_dir(ctx))
        model_dir = checkpoint.parent                      # config.json + model.safetensors
        script = _predict_py(ctx)
        work = Path(ctx.workdir or ".") / "v8in"
        layer_dir = work / "layers"
        layer_dir.mkdir(parents=True, exist_ok=True)
        for old in layer_dir.glob("*.tif"):
            old.unlink()
        # predict.py takes 00.tif, 01.tif, ... and uses the central 24 (same window as A-hn's layers 8..31 of 41).
        for i in range(layers):
            tifffile.imwrite(layer_dir / f"{i:02d}.tif", np.asarray(prepared.array[i]))
        interpreter = ctx.python or sys.executable
        device = getattr(ctx, "device", None) or "auto"
        runs = {"forward": "--no-reverse", "reverse": "--reverse"}
        wanted = ["forward", "reverse"] if profile.execution.direction == "both" else [profile.execution.direction]
        maps: dict[str, np.ndarray] = {}
        commands = []
        for direction in wanted:
            out = work / f"{direction}.npy"
            if out.exists():
                out.unlink()
            command = [interpreter, str(script), "--model", str(model_dir), "--layers", str(layer_dir),
                       "--output", str(out), "--batch-size", str(profile.execution.batch), runs[direction]]
            if profile.input.stride:
                command += ["--stride", str(profile.input.stride)]
            if device != "auto":
                command += ["--device", device]
            commands.append(command)
            ctx.say(f"v8in: {' '.join(command)}")
            done = subprocess.run(command, capture_output=True, text=True, timeout=6 * 3600, cwd=str(script.parent))
            if done.returncode != 0:
                raise InkSDKError(f"v8-in predict.py failed with exit {done.returncode}: "
                                  f"{(done.stderr or done.stdout)[-3000:]}")
            if not out.is_file():
                raise InkSDKError(f"predict.py reported success but wrote no map: {out}")
            arr = np.load(out).astype(np.float32)
            if arr.shape != tuple(prepared.array.shape[1:]):
                raise InkSDKError(f"{direction} map {arr.shape} != prepared {prepared.array.shape[1:]}")
            maps[direction] = arr
        if len(maps) == 2:
            maps["mean"] = (maps["forward"] + maps["reverse"]) / 2.0
        return RawOutput(
            maps=maps,
            valid=np.ones_like(next(iter(maps.values())), dtype=bool),
            quantisation={"source_dtype": "float32", "levels": None, "scale": 1.0},
            receipt={
                "adapter": "v8in",
                "commands": commands,
                "predict_py": str(script),
                "model_dir": str(model_dir),
                "checkpoint": str(checkpoint),
                "direction": profile.execution.direction,
                "batch": profile.execution.batch,
                "runtime_seconds": round(time.time() - started, 2),
                "statistics": {key: stats(value) for key, value in maps.items()},
                "non_claims": [
                    "A probability map is not OCR and does not prove ink or letters.",
                    "This lane is screening-only and target-uncalibrated.",
                ],
            },
        )
