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
from .base import (
    BaseAdapter,
    batch_size,
    model_python,
    resolve_device,
    source_scale_issue,
    stats,
)


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
            return script.resolve()  # predict.py runs with cwd=its folder, so relative paths would break
    raise MissingDependencyError(
        "the v8-in lane needs predict.py and the ink8um package from YoussefMoNader/ink-8um-v8in. "
        "Run `ink-sdk fetch v8in@ink-8um-v8in` or set V8IN_CODE to that checkout."
    )


MPS_RUNNER = Path(__file__).resolve().parents[1] / "v8in_mps_runner.py"


def _device(requested: str | None) -> str:
    """The device for predict.py. Left alone, predict.py picks CUDA or else the CPU, so `auto` becomes `mps` when this
    machine has an Apple GPU, as for the in-process adapters. Otherwise `auto` is left to predict.py: it may run under
    another interpreter (--python) whose torch sees a different set of devices."""
    value = (requested or "auto").lower()
    if value != "auto":
        return value
    try:
        return "mps" if resolve_device("auto") == "mps" else "auto"
    except ImportError:
        return "auto"


def _mps_out_of_memory(done: subprocess.CompletedProcess) -> bool:
    return "MPS backend out of memory" in (done.stderr or "") + (done.stdout or "")


class V8inAdapter(BaseAdapter):
    name = "v8in"

    @staticmethod
    def _predict(launcher, model_dir, layer_dir, out, batch, run, profile, device, script, ctx, commands):
        command = [*launcher, "--model", str(model_dir), "--layers", str(layer_dir),
                   "--output", str(out), "--batch-size", str(batch), run]
        if profile.input.stride:
            command += ["--stride", str(profile.input.stride)]
        if device != "auto":
            command += ["--device", device]
        if sys.platform == "darwin":
            # macOS spawns DataLoader workers: each re-imports torch and is sent the whole layer stack. Slicing
            # tiles in the main process is cheap and gives the same map (512² w00: 365 s -> 331 s on MPS).
            command += ["--workers", "0"]
        commands.append(command)
        ctx.say(f"v8in: {' '.join(command)}")
        done = subprocess.run(command, capture_output=True, text=True, timeout=6 * 3600, cwd=str(script.parent))
        return done, command

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
        model_dir = checkpoint.parent.resolve()            # config.json + model.safetensors
        script = _predict_py(ctx)
        work = (Path(ctx.workdir or ".") / "v8in").resolve()
        layer_dir = work / "layers"
        layer_dir.mkdir(parents=True, exist_ok=True)
        for old in layer_dir.glob("*.tif"):
            old.unlink()
        # predict.py takes 00.tif, 01.tif, ... and uses the central 24 (same window as A-hn's layers 8..31 of 41).
        for i in range(layers):
            tifffile.imwrite(layer_dir / f"{i:02d}.tif", np.asarray(prepared.array[i]))
        interpreter = model_python(ctx)
        if os.sep in interpreter or (os.altsep and os.altsep in interpreter):
            interpreter = str(Path(interpreter).absolute())  # a path, so not looked up on PATH: make it absolute too
        device = _device(getattr(ctx, "device", None))
        # On MPS, predict.py runs through a wrapper that sends the ResNet3D's 3-D convolutions through 2-D kernels.
        launcher = [interpreter, str(MPS_RUNNER), str(script)] if device == "mps" else [interpreter, str(script)]
        runs = {"forward": "--no-reverse", "reverse": "--reverse"}
        wanted = ["forward", "reverse"] if profile.execution.direction == "both" else [profile.execution.direction]
        maps: dict[str, np.ndarray] = {}
        commands = []
        batch = batch_size(profile, ctx)
        explicit = ctx.batch is not None  # --batch is taken as given
        for direction in wanted:
            out = work / f"{direction}.npy"
            if out.exists():
                out.unlink()
            done, command = self._predict(launcher, model_dir, layer_dir, out, batch, runs[direction], profile, device,
                                          script, ctx, commands)
            # v8-in needs about 2.1 GiB of GPU memory per tile, so the profile's batch of 8 does not fit most Macs.
            # On MPS, halve it until it fits; the map does not depend on the batch.
            while done.returncode != 0 and device == "mps" and not explicit and batch > 1 and _mps_out_of_memory(done):
                batch //= 2
                ctx.say(f"v8in: MPS out of memory, retrying with batch {batch}")
                done, command = self._predict(launcher, model_dir, layer_dir, out, batch, runs[direction], profile,
                                              device, script, ctx, commands)
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
            valid=prepared.valid if prepared.valid is not None else np.ones_like(next(iter(maps.values())), dtype=bool),
            quantisation={"source_dtype": "float32", "levels": None, "scale": 1.0},
            receipt={
                "adapter": "v8in",
                "commands": commands,
                "predict_py": str(script),
                "model_dir": str(model_dir),
                "checkpoint": str(checkpoint),
                "direction": profile.execution.direction,
                "batch": batch,
                "runtime_seconds": round(time.time() - started, 2),
                "statistics": {key: stats(value) for key, value in maps.items()},
                "non_claims": [
                    "A probability map is not OCR and does not prove ink or letters.",
                    "This lane is screening-only and target-uncalibrated.",
                ],
            },
        )
