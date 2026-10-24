from __future__ import annotations

import pytest

from ink_sdk.errors import IncompatibleScaleError
from ink_sdk.prepare import plan_scale
from ink_sdk.registry import get_profile

INK = get_profile("ink_9um@scrollprize-seed43-step75000")
CANON = get_profile("canonical_2um@r152-3ddec-v2-l5-epoch13")
H24 = get_profile("hecate@2.4um")
H96 = get_profile("hecate@9.6um")


def test_2_399_matrix():
    assert plan_scale(2.399, INK).mode == "pool"
    assert plan_scale(2.399, INK).factors_zyx == (4, 4, 4)
    assert plan_scale(2.399, CANON).mode == "native"
    assert plan_scale(2.399, H24).mode == "native"
    plan = plan_scale(2.399, H96)
    assert plan.mode == "pool"
    assert plan.output_pixel_um == pytest.approx(9.596)


def test_9_362_matrix():
    assert plan_scale(9.362, INK).mode == "native"
    for profile in (CANON, H24, H96):
        with pytest.raises(IncompatibleScaleError):
            plan_scale(9.362, profile)


def test_9_596_matrix():
    assert plan_scale(9.596, INK).mode == "native"
    assert plan_scale(9.596, H96).mode == "native"


def test_8_64_requires_opt_in_and_uses_9_362():
    with pytest.raises(IncompatibleScaleError):
        plan_scale(8.64, INK)
    plan = plan_scale(8.64, INK, resample_from_um=8.64)
    assert plan.mode == "resample"
    assert plan.target_um == 9.362
    for profile in (CANON, H24, H96):
        with pytest.raises(IncompatibleScaleError):
            plan_scale(8.64, profile, resample_from_um=8.64)


def test_resample_flag_must_describe_source():
    with pytest.raises(IncompatibleScaleError):
        plan_scale(9.0, INK, resample_from_um=8.64)
