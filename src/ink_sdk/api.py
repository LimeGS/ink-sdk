"""The small Python surface over the resolver, renderer and adapters."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from .contracts import InkResult, RenderSpec, RunContext, Segment
from .profiles import ModelProfile
from .registry import get_profile, list_profiles
from .resolver import build_segment, inspect_plan
from .runner import doctor as _doctor
from .runner import infer as _infer
from .runner import inspect as _inspect
from .runner import render_segment as _render


class InkSDK:
    def __init__(
        self,
        *,
        models_dir: Path | None = None,
        cache_dir: Path | None = None,
        renderer: Path | None = None,
        device: str = "auto",
        villa: Path | None = None,
    ):
        self.models_dir = models_dir
        self.cache_dir = cache_dir
        self.renderer = renderer
        self.device = device
        self.villa = villa

    def models(self) -> list[ModelProfile]:
        return list_profiles()

    def profile(self, identifier: str) -> ModelProfile:
        return get_profile(identifier)

    def context(self, **overrides) -> RunContext:
        base = RunContext(
            device=self.device,
            models_dir=self.models_dir,
            cache_dir=self.cache_dir,
            renderer=self.renderer,
            villa=self.villa,
        )
        return dataclasses.replace(base, **overrides)

    def segment(
        self,
        *,
        profile: str | ModelProfile,
        surface_volume: Path | None = None,
        tifxyz: Path | None = None,
        ct: Path | None = None,
        voxel_um: float | None = None,
        render: bool = False,
        ctx: RunContext | None = None,
        **ctx_overrides,
    ) -> Segment:
        profile_obj = profile if isinstance(profile, ModelProfile) else get_profile(profile)
        context = ctx or self.context(**ctx_overrides)
        return build_segment(
            profile=profile_obj,
            ctx=context,
            surface_volume=surface_volume,
            tifxyz=tifxyz,
            ct=ct,
            voxel_um=voxel_um,
            force_render=render,
        )

    def inspect(
        self,
        segment: Segment,
        profile: str | ModelProfile,
        *,
        ctx: RunContext | None = None,
        **ctx_overrides,
    ) -> dict:
        profile_obj = profile if isinstance(profile, ModelProfile) else get_profile(profile)
        context = ctx or self.context(**ctx_overrides)
        return _inspect(segment, profile_obj, context)

    def infer(
        self,
        segment: Segment,
        profile: str | ModelProfile,
        *,
        out_dir: Path | None = None,
        ctx: RunContext | None = None,
        **ctx_overrides,
    ) -> InkResult:
        profile_obj = profile if isinstance(profile, ModelProfile) else get_profile(profile)
        context = ctx or self.context(**ctx_overrides)
        return _infer(segment, profile_obj, context, out_dir=out_dir)

    def render(
        self,
        segment: Segment,
        profile: str | ModelProfile,
        *,
        out_dir: Path | None = None,
        ctx: RunContext | None = None,
        **ctx_overrides,
    ):
        profile_obj = profile if isinstance(profile, ModelProfile) else get_profile(profile)
        context = ctx or self.context(**ctx_overrides)
        return _render(segment, profile_obj, context, out_dir=out_dir)

    def doctor(self, **kwargs) -> dict:
        return _doctor(**kwargs)
