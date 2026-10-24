from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ink_sdk.errors import ModelIntegrityError
from ink_sdk.fetch import fetch_profile
from ink_sdk.registry import get_profile


def test_fetch_verifies_and_places_checkpoint(tmp_path, monkeypatch):
    import huggingface_hub

    payload = b"checkpoint-bytes"
    profile = get_profile("hecate@2.4um")
    profile.checkpoint.bytes = len(payload)
    profile.checkpoint.sha256 = hashlib.sha256(payload).hexdigest()
    seen = {}

    def fake_download(*, repo_id, filename, revision, local_dir):
        seen.update(repo_id=repo_id, filename=filename, revision=revision)
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    destination = fetch_profile(profile, models_dir=tmp_path / "models")
    assert destination.read_bytes() == payload
    assert seen == {
        "repo_id": profile.source.repo,
        "filename": profile.upstream_checkpoint_path,
        "revision": profile.source.revision,
    }


def test_fetch_rejects_wrong_digest(tmp_path, monkeypatch):
    import huggingface_hub

    profile = get_profile("hecate@2.4um")
    profile.checkpoint.bytes = 4
    profile.checkpoint.sha256 = "0" * 64

    def fake_download(*, repo_id, filename, revision, local_dir):
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"nope")
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    with pytest.raises(ModelIntegrityError):
        fetch_profile(profile, models_dir=tmp_path / "models")
    assert not (tmp_path / "models" / profile.checkpoint.path).exists()


def test_existing_verified_checkpoint_is_not_downloaded(tmp_path, monkeypatch):
    import huggingface_hub

    payload = b"already-here"
    profile = get_profile("hecate@9.6um")
    profile.checkpoint.bytes = len(payload)
    profile.checkpoint.sha256 = hashlib.sha256(payload).hexdigest()
    path = profile.checkpoint_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)

    def fail(**kwargs):
        raise AssertionError("must not download")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fail)
    assert fetch_profile(profile, models_dir=tmp_path) == path
