"""Opt-in checkpoint fetch. Nothing downloads during inference."""

from __future__ import annotations

import hashlib
import json
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
    destination.parent.mkdir(parents=True, exist_ok=True)
    if profile.checkpoint.url is not None:
        return _fetch_url(profile, destination, progress)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise MissingDependencyError("fetch needs huggingface_hub; install ink-sdk[fetch]") from exc

    if profile.adapter == "v8in":
        return _fetch_v8in(profile, destination, progress)
    if profile.checkpoint.rebuild is not None:
        return _fetch_rebuild(profile, destination, hf_hub_download, progress)
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


def _download_pinned(profile: ModelProfile, item, tmp: str, hf_hub_download) -> Path:
    try:
        path = Path(
            hf_hub_download(
                repo_id=profile.source.repo,
                filename=item.path,
                revision=profile.source.revision,
                local_dir=tmp,
            )
        )
    except Exception as exc:
        raise ModelIntegrityError(f"download failed for {profile.source.repo}/{item.path}: {exc}") from exc
    if path.stat().st_size != item.bytes or _hash(path) != item.sha256:
        raise ModelIntegrityError(
            f"{profile.source.repo}/{item.path} does not match its pinned size and sha256"
        )
    return path


def _rebuild_pth(weights: Path, config: Path, step: int, out: Path) -> None:
    try:
        import torch
        from safetensors.torch import load_file
    except ImportError as exc:
        raise MissingDependencyError(
            "rebuilding this checkpoint needs torch and safetensors; pip install safetensors"
        ) from exc
    with config.open(encoding="utf-8") as handle:
        cfg = json.load(handle)
    # torch.save records the file stem inside the archive, so `out` must carry the final name.
    torch.save({"model": load_file(str(weights)), "config": cfg, "step": step}, str(out))


def _fetch_rebuild(profile: ModelProfile, destination: Path, hf_hub_download, progress) -> Path:
    rebuild = profile.checkpoint.rebuild
    with tempfile.TemporaryDirectory(prefix="ink-sdk-fetch-", dir=destination.parent) as tmp:
        if progress:
            progress(
                f"downloading {profile.source.repo}/{rebuild.weights.path} and "
                f"{rebuild.config.path} @{profile.source.revision}"
            )
        weights = _download_pinned(profile, rebuild.weights, tmp, hf_hub_download)
        config = _download_pinned(profile, rebuild.config, tmp, hf_hub_download)
        built_dir = Path(tmp) / "rebuilt"
        built_dir.mkdir()
        built = built_dir / destination.name
        if progress:
            progress(f"rebuilding {destination.name} ({rebuild.recipe})")
        _rebuild_pth(weights, config, rebuild.step, built)
        size, digest = built.stat().st_size, _hash(built)
        if size != profile.checkpoint.bytes or digest != profile.checkpoint.sha256:
            raise ModelIntegrityError(
                f"rebuilt {destination.name} is {size} bytes, sha256 {digest}; the pin is "
                f"{profile.checkpoint.bytes} bytes, {profile.checkpoint.sha256}. The upstream files "
                "matched their pins, so the difference comes from the local torch version."
            )
        try:
            shutil.move(str(built), destination)
        except OSError as exc:
            raise ModelIntegrityError(f"could not place checkpoint at {destination}: {exc}") from exc
    if progress:
        progress(f"placed {destination}")
    return destination


def _check_pin(path: Path, profile: ModelProfile, label: str) -> None:
    if path.stat().st_size != profile.checkpoint.bytes:
        raise ModelIntegrityError(
            f"downloaded size {path.stat().st_size} != pinned {profile.checkpoint.bytes}: {label}"
        )
    digest = _hash(path)
    if digest != profile.checkpoint.sha256:
        raise ModelIntegrityError(f"downloaded digest {digest} != pinned {profile.checkpoint.sha256}: {label}")


def _fetch_url(profile: ModelProfile, destination: Path, progress) -> Path:
    """A checkpoint published as a plain https file (e.g. a GitHub release asset), pinned by size and sha256."""
    import urllib.request

    url = profile.checkpoint.url
    with tempfile.TemporaryDirectory(prefix="ink-sdk-fetch-", dir=destination.parent) as tmp:
        if progress:
            progress(f"downloading {url}")
        part = Path(tmp) / destination.name
        try:
            with urllib.request.urlopen(url, timeout=600) as response, part.open("wb") as handle:
                shutil.copyfileobj(response, handle, 1 << 20)
        except Exception as exc:
            raise ModelIntegrityError(f"download failed for {url}: {exc}") from exc
        _check_pin(part, profile, url)
        try:
            shutil.move(str(part), destination)
        except OSError as exc:
            raise ModelIntegrityError(f"could not place checkpoint at {destination}: {exc}") from exc
    if progress:
        progress(f"placed {destination}")
    return destination


def _fetch_v8in(profile: ModelProfile, destination: Path, progress) -> Path:
    """v8-in style checkpoints run with the model repository's predict.py, which needs the config.json next to
    the weights and, for the base repository, predict.py and the ink8um package. Download that folder at the
    pinned revision, then check the weights against the pin."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise MissingDependencyError("fetch needs huggingface_hub; install ink-sdk[fetch]") from exc
    upstream = Path(profile.upstream_checkpoint_path)
    root = profile.checkpoint_path(Path("/")).relative_to("/")
    local_root = destination
    for _ in upstream.parts:
        local_root = local_root.parent
    folder = upstream.parent.as_posix()
    patterns = [f"{folder}/*"] if folder not in ("", ".") else ["*.safetensors", "*.json", "*.py", "ink8um/**"]
    if progress:
        progress(f"downloading {profile.source.repo} {patterns} @{profile.source.revision} -> {local_root}")
    try:
        snapshot_download(
            repo_id=profile.source.repo,
            revision=profile.source.revision,
            allow_patterns=patterns,
            local_dir=str(local_root),
        )
    except Exception as exc:
        raise ModelIntegrityError(f"download failed for {profile.source.repo} ({root}): {exc}") from exc
    if not destination.is_file():
        raise ModelIntegrityError(f"{profile.source.repo} has no {upstream} at {profile.source.revision}")
    _check_pin(destination, profile, f"{profile.source.repo}/{upstream}")
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
