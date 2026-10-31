"""Pinned GP/Large/Scroll5 TimeSformers; physical transfer remains experimental."""
from __future__ import annotations

import importlib
import os
import pickle
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ..contracts import Issue, PreparedInput, RawOutput, RunContext, Segment
from ..errors import (
    InkSDKError,
    MissingDependencyError,
    ModelIntegrityError,
    ValidationError,
)
from ..profiles import ModelProfile
from .base import BaseAdapter, resolve_device, source_scale_issue, stats
from .canonical2um import grid

VARIANTS = {
    "scrollprize/timesformer_GP_scroll1": (8, 6),
    "scrollprize/timesformer_large_scroll1_01122024": (16, 8),
    "scrollprize/timesformer_scroll5_27112024": (8, 6),
}
EXPERIMENTAL = "EXPERIMENTAL_PHYSICAL_TRANSFER_UNVALIDATED"


def _model_config(profile: ModelProfile) -> dict:
    if profile.source.repo not in VARIANTS:
        raise ValidationError(f"unsupported TimeSformer source: {profile.source.repo}")
    depth, heads = VARIANTS[profile.source.repo]
    return {"dim": 512, "image_size": 64, "patch_size": 16, "num_frames": 26,
            "num_classes": 16, "channels": 1, "depth": depth, "heads": heads,
            "dim_head": 64, "attn_dropout": .1, "ff_dropout": .1}


def _validate_contract(profile: ModelProfile, ctx: RunContext) -> None:
    _model_config(profile)
    p = profile.input
    if (p.depth != 26 or p.tile != 64 or p.divisor != 200 or p.clip != 200
            or p.normalization != "clip-200-div-200" or p.scale_um != 7.91
            or not 0 < p.scale_tolerance_um <= .1 or not 0 <= p.min_valid_ratio <= 1
            or p.stride is None or not 1 <= p.stride <= 64):
        raise ValidationError("TimeSformer requires 26 planes/64px, 7.91um, clip-200-div-200 and stride 1..64")
    if (profile.execution.blend != "hann" or (ctx.amp or profile.execution.amp) != "fp32"
            or profile.execution.tta or ctx.output_3d):
        raise ValidationError("TimeSformer supports Hann/FP32 only, without TTA or 3D output")


class _ArchivedTrainingState:
    """Store discarded archive metadata without running optimizer/scheduler code."""


def _checkpoint_state(path: Path):
    try:
        import torch
        from safetensors.torch import load_file
    except ImportError as exc:
        raise MissingDependencyError("install ink-sdk[timesformer] for TimeSformer inference") from exc
    try:
        if path.suffix == ".safetensors":
            state = load_file(str(path), device="cpu")
        elif path.suffix == ".ckpt":
            if not hasattr(torch.serialization, "get_unsafe_globals_in_checkpoint"):
                raise MissingDependencyError("TimeSformer archives require torch>=2.6; install ink-sdk[timesformer]")
            inert = [(_ArchivedTrainingState, name) for name in
                     ("torch.optim.adamw.AdamW", "torch.optim.lr_scheduler.CosineAnnealingLR")]
            allowed = {name for _, name in inert} | {"builtins.dict", "collections.defaultdict"}
            unknown = set(torch.serialization.get_unsafe_globals_in_checkpoint(path)) - allowed
            if unknown:
                raise ModelIntegrityError(f"unapproved checkpoint globals: {sorted(unknown)}")
            with torch.serialization.safe_globals([*inert, dict, defaultdict]):
                payload = torch.load(path, map_location="cpu", weights_only=True)
            state = payload.get("state_dict") if isinstance(payload, dict) else None
        else:
            raise ModelIntegrityError("TimeSformer checkpoint must be .safetensors or .ckpt")
    except (OSError, ValueError, RuntimeError, TypeError, EOFError, pickle.UnpicklingError) as exc:
        raise ModelIntegrityError(f"cannot safely read TimeSformer checkpoint: {exc}") from exc
    if not isinstance(state, dict) or not state or any(not isinstance(v, torch.Tensor) for v in state.values()):
        raise ModelIntegrityError("TimeSformer requires a nonempty tensor state_dict")
    return state


def _build_model(profile: ModelProfile):
    try:
        from timesformer_pytorch import TimeSformer
        from torch import nn
    except ImportError as exc:
        raise MissingDependencyError("install ink-sdk[timesformer] for TimeSformer inference") from exc

    class InkTimeSformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = TimeSformer(**_model_config(profile))

        def forward(self, image):
            return self.backbone(image.permute(0, 2, 1, 3, 4)).reshape(-1, 1, 4, 4)

    return InkTimeSformer()


def _default_model_factory(checkpoint: Path, device, profile: ModelProfile):
    model = _build_model(profile)
    try:
        model.load_state_dict(_checkpoint_state(checkpoint), strict=True)
    except RuntimeError as exc:
        raise ModelIntegrityError(f"TimeSformer model state mismatch: {exc}") from exc
    return model.to(device).eval()


class TimeSformerAdapter(BaseAdapter):
    name = "timesformer"

    def __init__(self, model_factory: Callable | None = None):
        self.model_factory = model_factory or _default_model_factory

    def check(self, segment: Segment, profile: ModelProfile, ctx: RunContext | None = None) -> list[Issue]:
        ctx = ctx or RunContext()
        issues = [Issue("warning", "experimental-transfer",
                        "7.91um TimeSformer transfer and physical ink face are unvalidated; no ink/letter claims")]
        try:
            for module in ("torch", "safetensors", "timesformer_pytorch"):
                importlib.import_module(module)
        except ImportError:
            issues.append(Issue("error", "missing-timesformer-dependency", "install ink-sdk[timesformer]"))
        models = ctx.models_dir or Path(os.environ.get("INK_SDK_MODELS", Path.home() / ".cache/ink-sdk/models"))
        if not profile.checkpoint_path(models).is_file():
            issues.append(Issue("error", "missing-checkpoint", f"checkpoint not found: {profile.checkpoint_path(models)}"))
        issue = source_scale_issue(segment, profile, ctx)
        if issue:
            issues.append(issue)
        elif segment.pixel_um is not None and any(not np.isfinite(v) or abs(v - 7.91) > profile.input.scale_tolerance_um
                                                  for v in segment.pixel_um):
            issues.append(Issue("error", "incompatible-scale", "TimeSformer needs 7.91um on every axis"))
        if segment.depth is not None and segment.depth < 26:
            issues.append(Issue("error", "insufficient-depth", "TimeSformer needs at least 26 planes, not BASE's 18"))
        try:
            _validate_contract(profile, ctx)
        except ValidationError as exc:
            issues.append(Issue("error", "unsupported-timesformer-contract", str(exc)))
        return issues

    def run(self, prepared: PreparedInput, profile: ModelProfile, ctx: RunContext) -> RawOutput:
        _validate_contract(profile, ctx)
        try:
            import torch
            from torch.nn import functional as F
        except ImportError as exc:
            raise MissingDependencyError("install ink-sdk[timesformer] for TimeSformer inference") from exc
        models = ctx.models_dir or Path(os.environ.get("INK_SDK_MODELS", Path.home() / ".cache/ink-sdk/models"))
        checkpoint = profile.verify_checkpoint(models)
        volume = prepared.array
        if volume.ndim != 3 or volume.dtype != np.uint8 or volume.shape[0] < 26 or min(volume.shape[1:]) < 1:
            raise ValidationError("TimeSformer requires uint8 ZYX with at least 26 planes")
        if any(not np.isfinite(v) or abs(v - 7.91) > profile.input.scale_tolerance_um for v in prepared.pixel_um):
            raise ValidationError("TimeSformer prepared input must have 7.91um pixels on every axis")
        lo = volume.shape[0] // 2 - 13
        height, width = volume.shape[1:]
        h, w = max(height, 64), max(width, 64)
        valid = np.zeros((h, w), bool)
        for y in range(0, height, 64):
            valid[y:min(y+64, height), :width] = (np.asarray(volume[lo:lo+26, y:y+64, :]) > 0).any(0)
        if prepared.valid is not None:
            if prepared.valid.shape != (height, width) or prepared.valid.dtype != bool:
                raise ValidationError("prepared validity mask must be boolean and match XY")
            valid[:height, :width] &= prepared.valid
        if not valid.any():
            raise InkSDKError("TimeSformer input has no observed CT support")
        device_name = resolve_device(ctx.device, torch)
        if ctx.deterministic:
            torch.use_deterministic_algorithms(True)
        model = self.model_factory(checkpoint, torch.device(device_name), profile).eval()
        orders = ("forward", "reverse") if profile.execution.direction == "both" else (profile.execution.direction,)
        xs = grid(w, 64, profile.input.stride)
        ys = grid(h, 64, profile.input.stride)
        window = np.outer(np.hanning(64), np.hanning(64))
        maps, coverage, kept = {}, None, 0
        started = time.time()
        with torch.inference_mode():
            for order in orders:
                # Full rasters stay on CPU; GPU memory is bounded by the tile batch.
                total, norm = np.zeros((h, w), np.float64), np.zeros((h, w), np.float64)
                kept = 0
                for y in ys:
                    row = [x for x in xs if valid[y:y+64, x:x+64].any()
                           and valid[y:y+64, x:x+64].mean() >= profile.input.min_valid_ratio]
                    if not row:
                        continue
                    stripe = np.asarray(volume[lo:lo+26, y:y+64, :])
                    if stripe.shape[1:] != (64, w):
                        stripe = np.pad(stripe, ((0, 0), (0, 64-stripe.shape[1]), (0, w-width)))
                    for start in range(0, len(row), profile.execution.batch):
                        positions = row[start:start+profile.execution.batch]
                        tiles = np.stack([stripe[:, :, x:x+64] for x in positions])
                        if order == "reverse":
                            tiles = tiles[:, ::-1]
                        images = np.clip(tiles, 0, 200).astype(np.float32)[:, None] / 200
                        logits = model(torch.from_numpy(images).to(device_name))
                        if (not isinstance(logits, torch.Tensor) or logits.shape != (len(positions), 1, 4, 4)
                                or not torch.isfinite(logits).all()):
                            raise InkSDKError("TimeSformer returned nonfinite or malformed 4x4 logits")
                        probability = F.interpolate(logits.float().sigmoid(), (64, 64), mode="bilinear",
                                                    align_corners=False)[:, 0].cpu().numpy()
                        for x, value in zip(positions, probability):
                            total[y:y+64, x:x+64] += value * window
                            norm[y:y+64, x:x+64] += window
                        kept += len(positions)
                    ctx.say(f"TimeSformer {order}: row{y}/{h}, tiles{kept}")
                if not kept:
                    raise InkSDKError("TimeSformer found no tile with enough valid CT pixels")
                observed = (norm > 0) & valid
                np.divide(total, norm, out=total, where=norm > 0)
                total[~observed] = 0
                maps[order] = total[:height, :width].astype(np.float32)
                if coverage is None:
                    coverage = observed[:height, :width]
                elif not np.array_equal(coverage, observed[:height, :width]):
                    raise InkSDKError("TimeSformer directions have different support")
        if len(orders) == 2:
            maps["mean"] = (maps["forward"] + maps["reverse"]) / 2
        return RawOutput(maps=maps, valid=coverage,
            quantisation={"source_dtype": "float32", "levels": None, "scale": 1., "display_rescale": False},
            receipt={"adapter": self.name, "checkpoint": str(checkpoint), "architecture": _model_config(profile),
                "adapter_status": EXPERIMENTAL, "normalization": "clip-200-div-200", "clip": 200, "divisor": 200,
                "device": device_name, "batch": profile.execution.batch, "precision": "fp32", "tta": False,
                "depth_window": [lo, lo+26], "state_dict_strict": True,
                "sliding_window": {"tile_size": 64, "stride": profile.input.stride, "blend": "hann",
                    "candidate_tiles": len(xs)*len(ys), "kept_tiles": kept, "output_scale_factor": 16,
                    "min_valid_ratio": profile.input.min_valid_ratio},
                "runtime_seconds": round(time.time()-started, 2), "statistics": {k: stats(v) for k,v in maps.items()},
                "non_claims": ["Probability maps are not OCR and do not prove ink or letters.",
                    "7.91um physical transfer and ink-bearing face remain unvalidated; no accuracy claim."]})
