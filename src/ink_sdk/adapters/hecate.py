"""Hecate 2.4 and 9.6 um, using the released inference architecture."""

from __future__ import annotations

import importlib
import os
import time
from pathlib import Path
from typing import Callable

import numpy as np

from ..contracts import Issue, PreparedInput, RawOutput, RunContext, Segment
from ..errors import InkSDKError, ValidationError
from ..io import write_zarr
from ..profiles import ModelProfile
from .base import BaseAdapter, resolve_device, source_scale_issue, stats


def _default_model_factory(checkpoint: Path, device):
    from .._vendor import hecate

    return hecate.load_model(str(checkpoint), device)


class HecateAdapter(BaseAdapter):
    name = "hecate"

    def __init__(self, model_factory: Callable | None = None):
        self.model_factory = model_factory or _default_model_factory

    def check(
        self, segment: Segment, profile: ModelProfile, ctx: RunContext | None = None
    ) -> list[Issue]:
        issues: list[Issue] = []
        try:
            import torch  # noqa: F401
        except ImportError:
            issues.append(Issue("error", "missing-torch", "Hecate needs torch; install ink-sdk[hecate]"))
        models_dir = (
            ctx.models_dir
            if ctx and ctx.models_dir
            else Path(os.environ.get("INK_SDK_MODELS", Path.home() / ".cache" / "ink-sdk" / "models"))
        )
        if not profile.checkpoint_path(models_dir).is_file():
            issues.append(Issue("error", "missing-checkpoint", f"checkpoint not found: {profile.checkpoint_path(models_dir)}"))
        issue = source_scale_issue(segment, profile, ctx)
        if issue:
            issues.append(issue)
        if profile.execution.direction not in {"forward", "reverse", "both"}:
            issues.append(Issue("error", "direction", f"unknown direction {profile.execution.direction}"))
        return issues

    def run(
        self, prepared: PreparedInput, profile: ModelProfile, ctx: RunContext
    ) -> RawOutput:
        torch = importlib.import_module("torch")
        from .._vendor import hecate

        models_dir = ctx.models_dir or Path(
            os.environ.get("INK_SDK_MODELS", Path.home() / ".cache" / "ink-sdk" / "models")
        )
        checkpoint = profile.verify_checkpoint(models_dir)
        device_name = resolve_device(ctx.device, torch)
        device = torch.device(device_name)
        model = self.model_factory(checkpoint, device)
        if hasattr(model, "eval"):
            model.eval()
        sampling = float(getattr(model, "sampling_um", profile.input.scale_um))
        if abs(float(prepared.pixel_um[0]) / sampling - 1) > 0.02:
            raise InkSDKError(
                f"Hecate {sampling:g} um needs a {sampling:g} um render within 2%; "
                f"prepared input is {prepared.pixel_um[0]:g} um"
            )
        volume = np.asarray(prepared.array)
        if volume.ndim != 3 or volume.dtype != np.uint8:
            raise ValidationError("Hecate input must be uint8 ZYX")
        patch_depth = int(model.patch_size[0])
        if volume.shape[0] < patch_depth:
            raise InkSDKError(
                f"Hecate needs {patch_depth} layers, the prepared volume has {volume.shape[0]}"
            )
        height, width = volume.shape[1:]
        precision = str(ctx.amp or profile.execution.amp).lower()
        if precision in {"default", "fp32", "fp16"}:
            precision = "fp32"
        elif precision == "bf16":
            if device_name != "cuda":
                raise InkSDKError("Hecate bf16 requires CUDA; use --device cpu with fp32")
        else:
            raise ValidationError(f"unsupported Hecate precision {profile.execution.amp!r}")

        started = time.time()
        direction = profile.execution.direction
        maps: dict[str, np.ndarray] = {}
        reverse = direction in {"reverse", "both"}

        def one(reverse_z: bool) -> np.ndarray:
            out = np.zeros((height, width), dtype=np.uint8)
            hecate.predict(
                model,
                volume,
                out,
                reverse=reverse_z,
                stride=profile.input.stride,
                batch_size=int(profile.execution.batch),
                precision=precision,
                three_d=False,
            )
            return out.astype(np.float32) / 255.0

        if direction in {"forward", "both"}:
            maps["forward"] = one(False)
        if reverse:
            maps["reverse"] = one(True)
        if direction == "both":
            maps["mean"] = (maps["forward"] + maps["reverse"]) / 2.0

        volume_out = None
        volume_attrs: dict = {}
        if ctx.output_3d:
            out3d = np.zeros(volume.shape, dtype=np.uint8)
            hecate.predict(
                model,
                volume,
                out3d,
                reverse=(direction == "reverse"),
                stride=profile.input.stride,
                batch_size=int(profile.execution.batch),
                precision=precision,
                three_d=True,
            )
            depth = volume.shape[0]
            start = depth // 2 - patch_depth // 2
            if direction == "reverse":
                start = depth - start - patch_depth
            margin = int(getattr(model, "margin", 0))
            volume_out = out3d
            volume_attrs = {
                "axes": ["z", "y", "x"],
                "spacing_um": [sampling, sampling, sampling],
                "z_order": "same as input",
                "evaluated_z_interval": [start + margin, start + patch_depth - margin],
                "outside_evaluated_z": "zero; not evaluated",
                "background": "empty CT columns zeroed; no intensity threshold",
                "probabilities": "uint8 = round(255 * probability); no min-max scaling",
            }
            target = Path(ctx.workdir or ".") / "ink_3d.zarr"
            write_zarr(target, out3d, attrs=volume_attrs, overwrite=True)

        valid = volume.any(axis=0)
        return RawOutput(
            maps=maps,
            valid=valid,
            volume=volume_out,
            volume_attrs=volume_attrs,
            quantisation={
                "source_dtype": "uint8",
                "levels": 255,
                "scale": 255.0,
                "display_rescale": False,
            },
            receipt={
                "adapter": "hecate",
                "checkpoint": str(checkpoint),
                "sampling_um": sampling,
                "patch_size_zyx": list(model.patch_size),
                "divisor": float(model.divisor),
                "stride": profile.input.stride or int(model.patch_size[1] // 2),
                "precision": precision,
                "device": device_name,
                "direction": direction,
                "runtime_seconds": round(time.time() - started, 2),
                "statistics": {key: stats(value) for key, value in maps.items()},
                "non_claims": [
                    "A probability map is not OCR and does not prove ink or letters.",
                    "Hecate is experimental and its 3D output is not a surface segmentation.",
                ],
            },
        )
