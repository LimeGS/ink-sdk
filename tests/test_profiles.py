from __future__ import annotations

import pytest

from ink_sdk.errors import ModelIntegrityError, ValidationError
from ink_sdk.profiles import load_profile
from ink_sdk.registry import get_profile, list_profiles


def test_five_builtin_profiles_are_exact():
    ids = {p.id for p in list_profiles()}
    assert ids == {
        "ink_9um@scrollprize-seed43-step75000",
        "ink_9um@reader-v2-step040000",
        "canonical_2um@r152-3ddec-v2-l5-epoch13",
        "hecate@2.4um",
        "hecate@9.6um",
    }
    assert get_profile("hecate@9.6um").input.depth == 16
    assert get_profile("canonical_2um@r152-3ddec-v2-l5-epoch13").input.depth == 62
    assert get_profile("ink_9um@scrollprize-seed43-step75000").checkpoint.sha256.startswith("2aeaa85a")


def test_profile_can_be_loaded_from_a_path(tmp_path):
    profile = get_profile("ink_9um@scrollprize-seed43-step75000")
    path = tmp_path / "copy.toml"
    path.write_text(
        f'schema = "ink-sdk.profile.v1"\nid = "copy"\nadapter = "{profile.adapter}"\n'
        f'family = "{profile.family}"\n'
        '[source]\nrepo = "r"\nrevision = "a"\nupstream = "u"\nupstream_revision = "b"\n'
        'url = "https://example.invalid"\nlicense = "MIT"\n'
        '[checkpoint]\npath = "x"\nsha256 = "' + "a" * 64 + '"\nbytes = 1\n'
        '[input]\nfamily = "surface-volume-2d"\nscale_um = 9.6\n'
        'scale_tolerance_um = 0.5\ndepth = 17\nnormalization = "from-checkpoint"\n',
        encoding="utf-8",
    )
    assert load_profile(path).id == "copy"


def test_unknown_adapter_is_rejected(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text(
        'schema = "ink-sdk.profile.v1"\nid = "bad"\nadapter = "nope"\n'
        'family = "surface-volume-2d"\n'
        '[source]\nrepo = "r"\nrevision = "a"\nupstream = "u"\nupstream_revision = "b"\n'
        'url = "https://example.invalid"\nlicense = "MIT"\n'
        '[checkpoint]\npath = "x"\nsha256 = "' + "a" * 64 + '"\nbytes = 1\n'
        '[input]\nfamily = "surface-volume-2d"\nscale_um = 2.0\n'
        'scale_tolerance_um = 0.1\ndepth = 1\nnormalization = "divide-255-no-clip"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_profile(path)


def test_checkpoint_digest_validation(tmp_path, monkeypatch):
    profile = get_profile("canonical_2um@r152-3ddec-v2-l5-epoch13")
    path = tmp_path / profile.checkpoint.path
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not the pinned checkpoint")
    with pytest.raises(ModelIntegrityError):
        profile.verify_checkpoint(tmp_path)
