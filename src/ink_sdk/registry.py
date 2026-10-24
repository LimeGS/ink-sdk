"""Explicit adapter registry and profile discovery."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from .errors import ValidationError
from .profiles import ModelProfile, load_profile


def profile_dirs() -> list[Path]:
    dirs: list[Path] = []
    env = os.environ.get("INK_SDK_PROFILES")
    if env:
        dirs.extend(Path(p) for p in env.split(os.pathsep) if p)
    dirs.append(Path(__file__).resolve().parent / "profiles")
    return dirs


@lru_cache(maxsize=1)
def builtin_profiles() -> dict[str, ModelProfile]:
    found: dict[str, ModelProfile] = {}
    for directory in profile_dirs():
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.toml")):
            profile = load_profile(path)
            if profile.id in found:
                raise ValidationError(f"duplicate profile id {profile.id!r} in {path}")
            found[profile.id] = profile
    return found


def list_profiles() -> list[ModelProfile]:
    return sorted(builtin_profiles().values(), key=lambda p: p.id)


def get_profile(identifier: str) -> ModelProfile:
    """Accept a profile id, a profile file, or a unique suffix."""
    candidate = Path(identifier).expanduser()
    if candidate.is_file():
        return load_profile(candidate)
    profiles = builtin_profiles()
    if identifier in profiles:
        return profiles[identifier]
    matches = [p for key, p in profiles.items() if key.endswith(identifier)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValidationError(
            f"unknown profile {identifier!r}. Known profiles: {', '.join(sorted(profiles))}"
        )
    raise ValidationError(f"profile {identifier!r} is ambiguous: {', '.join(p.id for p in matches)}")


def get_adapter(profile: ModelProfile):
    # Lazy imports keep the core usable without torch.
    if profile.adapter == "ink9um":
        from .adapters.ink9um import Ink9umAdapter

        return Ink9umAdapter()
    if profile.adapter == "canonical2um":
        from .adapters.canonical2um import Canonical2umAdapter

        return Canonical2umAdapter()
    if profile.adapter == "hecate":
        from .adapters.hecate import HecateAdapter

        return HecateAdapter()
    raise ValidationError(f"unknown adapter {profile.adapter!r}")
