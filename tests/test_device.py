from __future__ import annotations

from types import SimpleNamespace

import pytest

from ink_sdk.adapters.base import resolve_device
from ink_sdk.errors import ValidationError


def _torch(cuda: bool, mps: bool | None):
    backends = SimpleNamespace()
    if mps is not None:
        backends.mps = SimpleNamespace(is_available=lambda: mps)
    return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda), backends=backends)


@pytest.mark.parametrize(
    "cuda, mps, expected",
    [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu"), (False, None, "cpu")],
)
def test_auto_prefers_cuda_then_mps_then_cpu(cuda, mps, expected):
    assert resolve_device("auto", _torch(cuda, mps)) == expected
    assert resolve_device(None, _torch(cuda, mps)) == expected


def test_mps_request_checks_availability_not_just_the_backend():
    # torch builds for Intel macs and Linux still have torch.backends.mps.
    with pytest.raises(ValidationError, match="MPS"):
        resolve_device("mps", _torch(False, False))
    assert resolve_device("mps", _torch(False, True)) == "mps"


def test_cuda_index_is_kept():
    assert resolve_device("cuda:1", _torch(True, False)) == "cuda:1"
    with pytest.raises(ValidationError, match="CUDA"):
        resolve_device("cuda:0", _torch(False, True))
