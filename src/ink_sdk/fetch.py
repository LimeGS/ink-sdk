"""Opt-in checkpoint fetch. Nothing downloads during inference."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path
from typing import Callable

from .errors import MissingDependencyError, ModelIntegrityError
from .profiles import ModelProfile


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_profile(
    profile: ModelProfile,
    *,
    models_dir: Path,
    overwrite: bool = False,
    progress: Callable[[str], None] | None = None,
) -> Path:
    destination = profile.checkpoint_path(models_dir)
    if destination.is_file():
        checksum = _hash(destination)
        if destination.stat().st_size == profile.checkpoint.bytes and checksum == profile.checkpoint.sha256:
            if progress:
                progress(f"already verified: {destination}")
            return destination
        if not overwrite:
            raise ModelIntegrityError(
                f"existing checkpoint does not match the pin: {destination}. Pass --overwrite."
            )
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise MissingDependencyError("fetch needs huggingface_hub; install ink-sdk[fetch]") from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ink-sdk-fetch-", dir=destination.parent) as tmp:
        if progress:
            progress(
                f"downloading {profile.source.repo}/{profile.upstream_checkpoint_path} "
                f"@{profile.source.revision}"
            )
        try:
            downloaded = Path(
                hf_hub_download(
                    repo_id=profile.source.repo,
                    filename=profile.upstream_checkpoint_path,
                    revision=profile.source.revision,
                    local_dir=tmp,
                )
            )
        except Exception as exc:
            raise ModelIntegrityError(
                f"download failed for {profile.source.repo}/"
                f"{profile.upstream_checkpoint_path}: {exc}"
            ) from exc
        if downloaded.stat().st_size != profile.checkpoint.bytes:
            raise ModelIntegrityError(
                f"downloaded size {downloaded.stat().st_size} != pinned "
                f"{profile.checkpoint.bytes}: {profile.source.repo}"
            )
        digest = _hash(downloaded)
        if digest != profile.checkpoint.sha256:
            raise ModelIntegrityError(
                f"downloaded digest {digest} != pinned {profile.checkpoint.sha256}: "
                f"{profile.source.repo}"
            )
        try:
            shutil.move(str(downloaded), destination)
        except OSError as exc:
            raise ModelIntegrityError(f"could not place checkpoint at {destination}: {exc}") from exc
    if progress:
        progress(f"placed {destination}")
    return destination


def fetch_profiles(
    profiles: list[ModelProfile],
    *,
    models_dir: Path,
    overwrite: bool = False,
    progress: Callable[[str], None] | None = None,
) -> list[Path]:
    return [
        fetch_profile(profile, models_dir=models_dir, overwrite=overwrite, progress=progress)
        for profile in profiles
    ]
