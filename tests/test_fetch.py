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


def _rebuild_profile(weights: bytes, config: bytes):
    profile = get_profile("ink_9um@dense-native-step016000").model_copy(deep=True)
    rb = profile.checkpoint.rebuild
    rb.weights.bytes, rb.weights.sha256 = len(weights), hashlib.sha256(weights).hexdigest()
    rb.config.bytes, rb.config.sha256 = len(config), hashlib.sha256(config).hexdigest()
    return profile


def _serve(files):
    def fake_download(*, repo_id, filename, revision, local_dir):
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(files[filename])
        return str(path)
    return fake_download


def test_fetch_rebuilds_verifies_and_places(tmp_path, monkeypatch):
    import huggingface_hub
    from ink_sdk import fetch as fetch_mod

    weights, config, built = b"w", b'{"a": 1}', b"rebuilt-pth"
    profile = _rebuild_profile(weights, config)
    profile.checkpoint.bytes, profile.checkpoint.sha256 = len(built), hashlib.sha256(built).hexdigest()
    rb = profile.checkpoint.rebuild
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _serve({rb.weights.path: weights, rb.config.path: config}))
    names = []

    def fake_rebuild(w, c, step, out):
        assert w.read_bytes() == weights and c.read_bytes() == config and step == 16000
        names.append(out.name)
        out.write_bytes(built)

    monkeypatch.setattr(fetch_mod, "_rebuild_pth", fake_rebuild)
    destination = fetch_profile(profile, models_dir=tmp_path)
    assert destination.read_bytes() == built
    assert names == [destination.name]  # torch.save stores the stem, so the final name is used


def test_fetch_rebuild_rejects_wrong_source(tmp_path, monkeypatch):
    import huggingface_hub
    from ink_sdk import fetch as fetch_mod

    profile = _rebuild_profile(b"w", b"{}")
    rb = profile.checkpoint.rebuild
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _serve({rb.weights.path: b"X", rb.config.path: b"{}"}))
    monkeypatch.setattr(fetch_mod, "_rebuild_pth", lambda *a: (_ for _ in ()).throw(AssertionError("must not rebuild")))
    with pytest.raises(ModelIntegrityError):
        fetch_profile(profile, models_dir=tmp_path)
    assert not profile.checkpoint_path(tmp_path).exists()


def test_fetch_rebuild_rejects_wrong_result(tmp_path, monkeypatch):
    import huggingface_hub
    from ink_sdk import fetch as fetch_mod

    profile = _rebuild_profile(b"w", b"{}")
    profile.checkpoint.bytes, profile.checkpoint.sha256 = 3, "0" * 64
    rb = profile.checkpoint.rebuild
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _serve({rb.weights.path: b"w", rb.config.path: b"{}"}))
    monkeypatch.setattr(fetch_mod, "_rebuild_pth", lambda w, c, s, out: out.write_bytes(b"abc"))
    with pytest.raises(ModelIntegrityError, match="torch version"):
        fetch_profile(profile, models_dir=tmp_path)
    assert not profile.checkpoint_path(tmp_path).exists()


def test_rebuild_pth_is_deterministic(tmp_path):
    torch = pytest.importorskip("torch")
    st = pytest.importorskip("safetensors.torch")
    from ink_sdk.fetch import _hash, _rebuild_pth

    weights, config = tmp_path / "w.safetensors", tmp_path / "c.json"
    st.save_file({"x": torch.arange(6, dtype=torch.float32)}, str(weights))
    config.write_text('{"in_channels": 1}', encoding="utf-8")
    digests = []
    for i in (1, 2):
        out = tmp_path / str(i) / "model.pth"
        out.parent.mkdir()
        _rebuild_pth(weights, config, 7, out)
        digests.append(_hash(out))
        loaded = torch.load(str(out), weights_only=True)
        assert loaded["step"] == 7 and loaded["config"] == {"in_channels": 1}
        assert torch.equal(loaded["model"]["x"], torch.arange(6, dtype=torch.float32))
    assert digests[0] == digests[1]
