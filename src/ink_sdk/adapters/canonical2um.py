"""Canonical 2.399 um ResNet-152 + 3D decoder.

The upstream architecture is vendored unchanged. Only the sliding window and
the receipt are local, exactly as the reference runner does.
"""

from __future__ import annotations

import importlib
import os
import time
from pathlib import Path
from typing import Callable

import numpy as np

from ..contracts import Issue, PreparedInput, RawOutput, RunContext, Segment
from ..errors import InkSDKError, ValidationError
from ..profiles import ModelProfile
from .base import BaseAdapter, source_scale_issue, stats


def _default_model_factory(checkpoint: Path, device, num_frames: int):
    from .._vendor.canonical.model_resnet3d_3d_decoder import load_model

    return load_model(str(checkpoint), device, num_frames=num_frames)


def hann2d(n: int) -> np.ndarray:
    window = np.hanning(n + 2).astype(np.float32)[1:-1]
    kernel = np.outer(window, window)
    return kernel / float(kernel.max())


def grid(length: int, tile: int, stride: int) -> list[int]:
    if length < tile:
        raise InkSDKError(f"axis {length} is smaller than tile {tile}")
    positions = set(range(0, length - tile + 1, stride))
    positions.add(length - tile)
    return sorted(positions)


class Canonical2umAdapter(BaseAdapter):
    name = "canonical2um"

    def __init__(self, model_factory: Callable | None = None):
        self.model_factory = model_factory or _default_model_factory

    def check(
        self, segment: Segment, profile: ModelProfile, ctx: RunContext | None = None
    ) -> list[Issue]:
        issues: list[Issue] = []
        try:
            import torch  # noqa: F401
        except ImportError:
            issues.append(Issue("error", "missing-torch", "canonical needs torch; install ink-sdk[canonical]"))
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
        if profile.execution.direction != "forward":
            issues.append(Issue("error", "direction", "canonical v0 only provides forward"))
        return issues

    def run(
        self, prepared: PreparedInput, profile: ModelProfile, ctx: RunContext
    ) -> RawOutput:
        torch = importlib.import_module("torch")
        torch_f = importlib.import_module("torch.nn.functional")
        from .base import resolve_device

        models_dir = ctx.models_dir or Path(
            os.environ.get("INK_SDK_MODELS", Path.home() / ".cache" / "ink-sdk" / "models")
        )
        checkpoint = profile.verify_checkpoint(models_dir)
        device_name = resolve_device(ctx.device, torch)
        device = torch.device(device_name)
        if ctx.deterministic:
            torch.use_deterministic_algorithms(True)

        frames = int(profile.input.depth)
        volume = np.asarray(prepared.array)
        if volume.ndim != 3 or volume.dtype != np.uint8:
            raise ValidationError("canonical input must be uint8 ZYX")
        if volume.shape[0] < frames:
            raise InkSDKError(
                f"canonical needs {frames} layers, the prepared volume has {volume.shape[0]}"
            )
        centre = volume.shape[0] // 2
        lo = centre - frames // 2
        stack = volume[lo : lo + frames]
        if stack.shape[0] != frames:
            raise InkSDKError(f"could not select {frames} central layers")
        height, width = stack.shape[1:]
        tile = int(profile.input.tile or 256)
        stride = int(profile.input.stride or tile // 2)
        if stride < 1:
            raise ValidationError("canonical stride must be positive")
        pad_h, pad_w = max(0, tile - height), max(0, tile - width)
        if pad_h or pad_w:
            stack = np.pad(stack, ((0, 0), (0, pad_h), (0, pad_w)))
        valid_np = (stack > 0).any(axis=0)
        valid_plane = torch.from_numpy(valid_np)

        started = time.time()
        factory = self.model_factory
        model = factory(checkpoint, device, frames)
        if hasattr(model, "eval"):
            model.eval()
        output_scale = int(model.get_output_scale_factor()) if hasattr(model, "get_output_scale_factor") else 1

        volume_t = torch.from_numpy(stack)
        positions = [(y, x) for y in grid(stack.shape[1], tile, stride) for x in grid(stack.shape[2], tile, stride)]
        kept = [
            (y, x)
            for (y, x) in positions
            if float(valid_plane[y : y + tile, x : x + tile].float().mean())
            >= float(profile.input.min_valid_ratio)
        ]
        if not kept:
            raise InkSDKError("canonical found no tile with enough valid CT pixels")

        weight = torch.from_numpy(hann2d(tile)).to(device)
        total = torch.zeros((stack.shape[1], stack.shape[2]), dtype=torch.float32, device=device)
        norm = torch.zeros_like(total)
        divisor = float(profile.input.divisor or 255.0)
        with torch.inference_mode():
            for start in range(0, len(kept), int(profile.execution.batch)):
                chunk = kept[start : start + int(profile.execution.batch)]
                batch = (
                    torch.stack([volume_t[:, y : y + tile, x : x + tile] for y, x in chunk])
                    .to(device=device, dtype=torch.float32)
                    .div_(divisor)
                )
                predicted = model.forward(batch)
                if isinstance(predicted, (tuple, list)):
                    predicted = predicted[0]
                if predicted.ndim == 4:
                    predicted = predicted[:, 0]
                predicted = torch.sigmoid(predicted.float())
                if predicted.shape[-1] != tile:
                    predicted = torch_f.interpolate(
                        predicted[:, None],
                        size=(tile, tile),
                        mode="bilinear",
                        align_corners=False,
                    )[:, 0]
                for (y, x), value in zip(chunk, predicted):
                    total[y : y + tile, x : x + tile] += value * weight
                    norm[y : y + tile, x : x + tile] += weight

        # The Hann window is ~2e-8 at the border; a 1e-6 clamp would darken
        # every edge. Clamp only to keep the division finite where norm is 0.
        probability = torch.where(norm > 0, total / norm.clamp(min=1e-20), torch.zeros_like(total))
        result = np.where(valid_np, probability.detach().cpu().numpy(), 0.0).astype(np.float32)
        result = result[:height, :width]
        valid_out = valid_np[:height, :width]
        return RawOutput(
            maps={"forward": result},
            valid=valid_out,
            quantisation={
                "source_dtype": "float32",
                "levels": None,
                "scale": 1.0,
                "display_rescale": False,
            },
            receipt={
                "adapter": "canonical2um",
                "checkpoint": str(checkpoint),
                "model_loader": "ScrollPrize/villa ink-detection/optimized_inference/model_resnet3d_3d_decoder.py",
                "normalization": "divide-255-no-clip",
                "divisor": divisor,
                "device": device_name,
                "batch": profile.execution.batch,
                "sliding_window": {
                    "tile_size": tile,
                    "stride": stride,
                    "candidate_tiles": len(positions),
                    "kept_tiles": len(kept),
                    "discarded_fraction": 1 - len(kept) / max(len(positions), 1),
                    "min_valid_ratio": float(profile.input.min_valid_ratio),
                    "output_scale_factor": output_scale,
                },
                "runtime_seconds": round(time.time() - started, 2),
                "statistics": stats(result),
                "non_claims": [
                    "A probability map is not OCR and does not prove ink or letters.",
                    "This lane is scale-restricted: it was validated at about 2 um.",
                ],
            },
        )
