"""CPU-only tests for the training-overlap flag. No weights, GPU or scroll imagery."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from conftest import write_zarr_volume

from ink_sdk.adapters.base import DummyAdapter
from ink_sdk.battery import run_battery, to_html
from ink_sdk.contracts import RunContext
from ink_sdk.registry import get_profile
from ink_sdk.renderers.base import DummyRenderer
from ink_sdk.resolver import build_segment
from ink_sdk.runner import infer, inspect
from ink_sdk.runner import infer as runner_infer
from ink_sdk.training_overlap import (
    ids_from_paths,
    is_in_sample,
    load_registry,
    lookup_profile,
    lookup_segment,
    norm_scroll,
    resolve_input,
    segment_ids,
    short_flag,
)

SEED42 = "ink_9um@scrollprize-seed42-step75000"
SEED43 = "ink_9um@scrollprize-seed43-step75000"
DENSE = "ink_9um@dense-native-step016000"
V8IN = "v8in@ink-8um-v8in"
AHN0 = "v8in@pherc0306b-ahn-s0"
READER = "ink_9um@reader-v3"
CANON = "canonical_2um@r152-3ddec-v2-l5-epoch13"
TRACE = "ink_9um@trace-v3b-step5000"

W030_FOLDER = "20250108000005-w030_2025010818"
W030_PATH = f"/data/segments/PHerc0139/{W030_FOLDER}/surface-volumes/9.362um.zarr"
W045 = "20260126000000-w045_x"
P0139 = {
    "w029": ("20250108000004", "pherc0139-w016"),
    "w030": ("20250108000005", "pherc0139-w017"),
    "w033": ("20260219000000", None),
    "w035": ("20260317000000", "pherc0139-w035"),
    "w039": ("20260302000000", "pherc0139-w039"),
    "w040": ("20250831000000", "pherc0139-w040"),
    "w041": ("20260108000000", "pherc0139-w041"),
    "w043": ("20260112000000", "pherc0139-w043"),
    "w044": ("20260115000000", "pherc0139-w028"),
    "w045": ("20260126000000", "pherc0139-w029"),
}


def test_seed42_and_dense_in_sample_on_w030_from_path():
    for profile in (SEED42, SEED43, DENSE):
        result = lookup_profile(profile, paths=[W030_PATH])
        assert result["status"] is True
        assert result["flag"].startswith("in-sample")
        assert result["matched"] == "20250108000005"
        assert "villa/pull/2349" in result["source"]


def test_v8in_held_out_on_w030_ahn_stays_unknown():
    held = lookup_profile(V8IN, paths=[W030_PATH])
    assert held["status"] is False
    assert held["flag"] == "held out (complete list)" and held["basis"] == "complete list"
    # A-hn starts from v8-in, but only part of its own training list is recorded here:
    # v8-in's held out does not carry over to the fine-tune.
    ahn = lookup_profile(AHN0, paths=[W030_PATH])
    assert ahn["status"] is None
    assert ahn["lineage"] == ["A-hn s0-s2", "YoussefMoNader/ink-8um-v8in"]
    assert "part of the lineage" in ahn["note"]
    assert is_in_sample(AHN0, paths=[W030_PATH]) is None


def test_unlisted_profile_is_unknown():
    assert lookup_profile(READER, paths=[W030_PATH])["status"] is None
    assert lookup_profile(CANON, paths=[W030_PATH])["status"] is None


def test_0343p_held_out_only_where_the_evidence_establishes_it():
    path = "/data/segments/PHerc0343P/20250511003658/surface-volumes/8.64um.zarr"
    for profile in (SEED42, SEED43, V8IN):
        assert lookup_profile(profile, paths=[path])["status"] is False
    # dense-native starts from unidentified weights (init.pth): a gap in a partial list is unknown
    dense = lookup_profile(DENSE, paths=[path])
    assert dense["status"] is None
    # v8-in's record names the ir-truth-9um segment only
    other = "/data/segments/PHerc0343P/20250904233748/surface-volumes/8.64um.zarr"
    assert lookup_profile(V8IN, paths=[other])["status"] is None
    assert lookup_profile(SEED42, paths=[other])["status"] is False  # ink_9um never trained on 0343P
    # rolled 343 is not the fragment 343P
    rolled = "/data/segments/PHerc0343/20250509220833/surface-volumes/8.64um.zarr"
    assert lookup_profile(SEED42, paths=[rolled])["status"] is None
    # Rectina is known to have trained on 0343P, but it is not in this table.
    assert lookup_profile("ink_9um@rectina-m4", paths=[path])["status"] is None


def test_label_cases_are_not_confused_with_public_w_numbers():
    # pherc0139-w029 is public w045 (held out for v8-in), not public w029.
    assert is_in_sample(V8IN, "PHerc0139", "pherc0139-w029") is False
    assert is_in_sample(V8IN, "PHerc0139", "pherc0139-w028") is True  # public w044
    assert is_in_sample(V8IN, "PHerc0139", "pherc0139-w044") is None
    assert is_in_sample(SEED42, "PHerc0139", _folder("w033")) is None  # "?" in jo-tud's table


def _folder(w: str) -> str:
    return f"{P0139[w][0]}-{w}_x"


def test_table_by_public_id_folder_and_label_case():
    want_ink9 = {
        "w029": True, "w030": True, "w033": None, "w035": True, "w039": True,
        "w040": True, "w041": True, "w043": True, "w044": True, "w045": True,
    }
    want_v8 = {
        "w029": False, "w030": False, "w033": True, "w035": True, "w039": False,
        "w040": False, "w041": True, "w043": False, "w044": True, "w045": False,
    }
    for w, sid_case in P0139.items():
        sid, case = sid_case
        folder = f"/data/PHerc0139/{_folder(w)}/sv.zarr"
        assert is_in_sample(SEED42, paths=[folder]) is want_ink9[w]
        assert is_in_sample(V8IN, "PHerc0139", sid) is want_v8[w]
        if case:
            assert is_in_sample(SEED42, "PHerc0139", case) is want_ink9[w]


def test_path_without_scroll_is_unknown_until_overrides():
    bare = "/tmp/runs/2.4um.zarr"
    assert lookup_profile(SEED42, paths=[bare])["status"] is None
    flagged = lookup_profile(SEED42, scroll="PHerc0139", segment=W030_FOLDER, paths=[bare])
    assert flagged["status"] is True
    ids = ids_from_paths(bare, scroll="PHerc0139", segment=W030_FOLDER)
    assert ids["scroll"] == "139" and ids["segment_id"] == "20250108000005" and ids["public_w"] == "w030"


def test_battery_window_path_needs_original_volume():
    window = "/tmp/battery/_windows/d17_c+0_s1.zarr"
    assert lookup_profile(SEED42, paths=[window])["status"] is None
    assert lookup_profile(SEED42, paths=[window, W030_PATH])["status"] is True


def test_helpers():
    assert norm_scroll("PHerc0139") == norm_scroll("pherc0139") == norm_scroll("PHerc. 139") == "139"
    assert norm_scroll("PHerc0343P") == "343p" != norm_scroll("PHerc0343")
    ids = segment_ids(W030_FOLDER, "PHerc0139")
    assert ids["segment_id"] == "20250108000005" and ids["public_w"] == "w030"
    assert short_flag({"status": True}) == "in-sample"
    assert short_flag({"status": False}) == "held out"
    assert short_flag({}) == "unknown"
    assert short_flag({"status": True, "inherited": True}) == "in-sample (inherited)"


def test_dense_native_config_only_entries():
    for w in ("w032", "w034", "w046", "w049", "w050"):
        result = lookup_profile(DENSE, "PHerc0139", f"20990101000000-{w}_x")
        assert result["status"] is True and "mapping not checked" in result["confidence"]
        assert is_in_sample(SEED42, "PHerc0139", f"20990101000000-{w}_x") is None


def _infer_on_mesh(tmp_path, mesh, ct, profile_id, **ctx_kw):
    profile = get_profile(profile_id)
    ctx = RunContext(
        cache_dir=tmp_path / "cache",
        models_dir=tmp_path / "models",
        workdir=tmp_path / "work",
        overwrite=True,
        on_degenerate="warn",
        **ctx_kw,
    )
    segment = build_segment(profile=profile, ctx=ctx, tifxyz=mesh, ct=ct, force_render=True)
    result = infer(
        segment, profile, ctx, out_dir=tmp_path / "out", adapter=DummyAdapter(), renderer=DummyRenderer()
    )
    return result, json.loads((tmp_path / "out" / "INK_SDK_RECEIPT.json").read_text())


def test_receipt_flag_does_not_change_maps(tmp_path, mesh, ct):
    # A generic fixture path is unknown; the same maps must be produced with an in-sample override.
    plain, plain_receipt = _infer_on_mesh(tmp_path, mesh, ct, SEED42)
    assert plain_receipt["training_overlap"]["status"] is None
    flagged_dir = tmp_path / "flagged"
    profile = get_profile(SEED42)
    ctx = RunContext(
        cache_dir=tmp_path / "cache2",
        models_dir=tmp_path / "models2",
        workdir=tmp_path / "work2",
        overwrite=True,
        on_degenerate="warn",
        scroll="PHerc0139",
        segment=W030_FOLDER,
    )
    segment = build_segment(profile=profile, ctx=ctx, tifxyz=mesh, ct=ct, force_render=True)
    flagged = infer(
        segment, profile, ctx, out_dir=flagged_dir, adapter=DummyAdapter(), renderer=DummyRenderer()
    )
    receipt = json.loads((flagged_dir / "INK_SDK_RECEIPT.json").read_text())
    assert receipt["training_overlap"]["status"] is True
    assert receipt["liveness"]["forward"]["verdict"] == plain_receipt["liveness"]["forward"]["verdict"]
    assert flagged.maps["forward"].array.shape == plain.maps["forward"].array.shape
    assert np.allclose(flagged.maps["forward"].array, plain.maps["forward"].array)
    assert flagged.primary == plain.primary


def test_inspect_and_lookup_segment_respect_context_overrides(tmp_path, surface_volume):
    profile = get_profile(SEED42)
    ctx = RunContext(scroll="PHerc0139", segment=W045, dry_run=True)
    segment = build_segment(profile=profile, ctx=ctx, surface_volume=surface_volume)
    plan = inspect(segment, profile, ctx, adapter=DummyAdapter())
    assert plan["training_overlap"]["status"] is True
    # without overrides, a bare surface.zarr stem is unknown
    bare = lookup_segment(SEED42, segment, RunContext())
    assert bare["status"] is None


def test_battery_html_includes_overlap_column():
    report = {
        "surface_volume": "/data/PHerc0139/20250108000005-w030_x/sv.zarr",
        "pixel_um": 9.366,
        "layers": 28,
        "centres": [0],
        "steps": [1],
        "runtime_seconds": 1.0,
        "images": {},
        "non_claims": ["not OCR"],
        "cells": [
            {
                "profile": SEED42,
                "centre": 0,
                "step": 1,
                "ok": True,
                "liveness": "ALIVE",
                "p99": 0.7,
                "training_overlap": lookup_profile(SEED42, paths=["/data/PHerc0139/20250108000005-w030_x/sv.zarr"]),
            },
            {"profile": "b@y", "centre": 0, "step": 1, "ok": False, "error": "Boom <bad>"},
        ],
        "skipped": [{"profile": "c@z", "reason": "scale"}],
    }
    page = to_html(report, Path("/tmp"))
    assert "training overlap" in page
    assert "in-sample" in page
    assert "unknown" in page  # the failed cell has no overlap dict
    assert "Boom &lt;bad&gt;" in page


# ---------- inherited overlap (fine-tunes) ----------
def test_trace_v3b_w043_surfaces_inherited_overlap():
    # LimeGS's example: TRACE v3B used w043 for checkpoint selection, and its seed-42 base trained on it.
    result = lookup_profile(TRACE, "PHerc0139", "20260112000000-w043_2026011217")
    assert result["status"] is True
    assert result["seen_by"] == "base model" and result["inherited"] is True
    assert result["flag"] == "in-sample (inherited from base model)"
    assert result["lineage"] == ["TRACE v3B", "scrollprize ink_9um seed 42"]
    uses = {(e["relation"], e["use"]) for e in result["evidence"]}
    assert ("checkpoint", "checkpoint selection") in uses
    assert ("base model (fine-tuned from)", "training") in uses
    assert "pherc0139-w043" in result["matched"] or result["matched"] == "20260112000000"
    assert "checkpoint selection" in result["note"]
    # the same through the label case name
    assert is_in_sample(TRACE, "PHerc0139", "pherc0139-w043") is True


def test_inherited_overlap_without_own_record():
    assert lookup_profile(TRACE, paths=[W030_PATH])["seen_by"] == "base model"
    ahn = lookup_profile(AHN0, "PHerc0139", _folder("w033"))
    assert ahn["status"] is True and ahn["inherited"] and ahn["evidence"][0]["model"] == "YoussefMoNader/ink-8um-v8in"
    own = lookup_profile(AHN0, "PHerc1447", "w058")
    assert own["status"] is True and own["seen_by"] == "checkpoint" and not own["inherited"]


def test_checkpoint_selection_alone_is_not_held_out():
    result = lookup_profile(TRACE, "PHerc0841", "20260220213127")
    assert result["status"] is None
    assert "checkpoint selection" in result["note"]


def _user_registry(tmp_path, entries):
    path = tmp_path / "overlap.json"
    path.write_text(json.dumps({"entries": entries}), encoding="utf-8")
    return load_registry(path, include_builtin=False)


def test_held_out_basis_rules(tmp_path):
    seg = {"scroll": "PHerc0841", "segment_id": "20260220213127"}
    reg = _user_registry(tmp_path, [
        {"key": "a", "model": "A", "profiles": ["x@after"], "held_out": [{**seg, "confidence": "after upload"}]},
        {"key": "b", "model": "B", "profiles": ["x@nobasis"], "held_out": [dict(seg)]},
        {"key": "c", "model": "C", "profiles": ["x@stated"], "held_out": [{**seg, "basis": "stated"}]},
        {"key": "d", "model": "D", "profiles": ["x@ft-stated"], "fine_tuned_from": "e",
         "held_out": [{**seg, "basis": "stated"}]},
        {"key": "e", "model": "E", "profiles": ["x@base"], "in_sample": []},
        {"key": "f", "model": "F", "profiles": ["x@ft-complete"], "fine_tuned_from": "e",
         "held_out": [{**seg, "basis": "complete list"}]},
        {"key": "g", "model": "G", "profiles": ["x@dangling"], "fine_tuned_from": "missing",
         "held_out": [{**seg, "basis": "stated"}]},
    ])
    look = lambda pid: lookup_profile(pid, "PHerc0841", "20260220213127", registry=reg)  # noqa: E731
    assert look("x@after")["status"] is None and "after upload" in look("x@after")["note"]
    assert look("x@nobasis")["status"] is None
    assert look("x@stated")["status"] is False and look("x@stated")["basis"] == "stated"
    # a stated held out on a fine-tune does not cover a base with no held-out record
    assert look("x@ft-stated")["status"] is None and "E" in look("x@ft-stated")["note"]
    # a complete list covers the whole lineage by definition
    assert look("x@ft-complete")["status"] is False
    # an unidentified base never gives held out from a stated record
    assert look("x@dangling")["status"] is None


def test_in_sample_in_the_base_wins_over_a_held_out_claim(tmp_path):
    seg = {"scroll": "PHerc0139", "segment_id": "20260112000000"}
    reg = _user_registry(tmp_path, [
        {"key": "ft", "model": "FT", "profiles": ["x@ft"], "fine_tuned_from": "base",
         "held_out": [{**seg, "basis": "stated"}]},
        {"key": "base", "model": "Base", "profiles": ["x@base"], "in_sample": [dict(seg)]},
    ])
    result = lookup_profile("x@ft", "PHerc0139", "20260112000000", registry=reg)
    assert result["status"] is True and result["inherited"]
    assert "overridden" in result["note"]


# ---------- ambiguous input ----------
def test_ambiguous_input_is_unknown():
    two_scrolls = "/data/PHerc0139/PHerc0343P/20250108000005-w030_x/sv.zarr"
    result = lookup_profile(SEED42, paths=[two_scrolls])
    assert result["status"] is None and "ambiguous" in result["note"]
    assert result["input"]["ambiguous"] == ["scroll"]
    # an explicit --scroll settles it
    assert lookup_profile(SEED42, scroll="PHerc0139", paths=[two_scrolls])["status"] is True
    # two different segment ids in the original input
    two_ids = "/data/PHerc0139/20250108000005-w030_x/20260126000000-w045_y/sv.zarr"
    assert lookup_profile(V8IN, paths=[two_ids])["status"] is None
    # a known segment id with another w-number
    wrong_w = lookup_profile(V8IN, "PHerc0139", "20250108000005-w045_x")
    assert wrong_w["status"] is None and "ambiguous" in wrong_w["note"]
    # a w-number with no scroll name anywhere
    assert lookup_profile(V8IN, paths=["/tmp/w030/sv.zarr"])["status"] is None


def test_volume_file_id_is_not_read_as_a_segment_id():
    path = ("/data/PHerc0139/segments/20260112000000-w043_2026011217/surface-volumes/"
            "9.362um-1.2m-113keV-volume-20250728140407.zarr")
    ids = resolve_input(path)
    assert ids["segment_id"] == "20260112000000" and ids["ambiguous"] == []
    assert lookup_profile(V8IN, paths=[path])["status"] is False


# ---------- infer and battery agree ----------
class _SDK:
    def infer(self, segment, profile, *, out_dir=None, ctx=None):
        return runner_infer(segment, profile, ctx, out_dir=out_dir, adapter=DummyAdapter())


@pytest.mark.parametrize(
    "folder, scroll, segment, want",
    [
        ("PHerc0139/20250108000005-w030_x", None, None, True),     # in-sample
        ("PHerc0343P/20250511003658", None, None, False),          # held out
        ("PHerc0139/PHerc0343P/20250108000005-w030_x", None, None, None),  # ambiguous
        ("runs/plain", None, None, None),                          # no identifiers
        ("runs/plain", "PHerc0139", "pherc0139-w043", True),       # overrides
    ],
)
def test_single_infer_and_battery_cell_report_the_same_status(tmp_path, folder, scroll, segment, want):
    original = write_zarr_volume(tmp_path / folder / "sv.zarr", (21, 16, 16), 9.6)
    profile = get_profile(SEED42)
    ctx = RunContext(cache_dir=tmp_path / "cache", models_dir=tmp_path / "models", workdir=tmp_path / "work",
                     overwrite=True, on_degenerate="warn", scroll=scroll, segment=segment)
    single_seg = build_segment(profile=profile, ctx=ctx, surface_volume=original, voxel_um=9.6)
    single = runner_infer(single_seg, profile, ctx, out_dir=tmp_path / "single", adapter=DummyAdapter())
    single_flag = single.receipt["training_overlap"]
    assert single_flag["status"] is want

    report = run_battery(_SDK(), original, 9.6, [profile], tmp_path / "battery", ctx,
                         centres=(0,), steps=(1,), thresholds=(0.5,), thumb_px=32, say=lambda _m: None)
    cell = report["cells"][0]
    assert cell.get("ok"), cell.get("error")
    cell_receipt = json.loads((Path(cell["dir"]) / "INK_SDK_RECEIPT.json").read_text())
    for flag in (cell["training_overlap"], cell_receipt["training_overlap"]):
        assert flag["status"] is want
        assert {k: v for k, v in flag.items() if k != "input"} == {
            k: v for k, v in single_flag.items() if k != "input"}
        assert flag["input"]["segment_id"] == single_flag["input"]["segment_id"]
        assert flag["input"]["scroll"] == single_flag["input"]["scroll"]
    # resolved from the original volume, never from the temporary window
    assert "_windows" not in json.dumps(report["training_overlap_input"])


def test_battery_dry_run_and_inspect_agree_for_every_listed_profile(tmp_path):
    original = write_zarr_volume(tmp_path / "PHerc0139" / "20260112000000-w043_x" / "sv.zarr", (24, 16, 16), 9.362)
    ctx = RunContext(dry_run=True)
    profiles = [get_profile(p) for p in (SEED42, SEED43, DENSE, V8IN, TRACE, AHN0, READER)]
    report = run_battery(_SDK(), original, 9.362, profiles, tmp_path / "b", ctx, centres=(0,), steps=(1,),
                         say=lambda _m: None)
    for profile in profiles:
        seg = build_segment(profile=profile, ctx=ctx, surface_volume=original, voxel_um=9.362)
        plan = inspect(seg, profile, ctx, adapter=DummyAdapter())
        assert plan["training_overlap"]["status"] is report["training_overlap"][profile.id]["status"]
    assert report["training_overlap"][TRACE]["inherited"] is True
    assert report["training_overlap"][V8IN]["status"] is False
    assert report["training_overlap"][READER]["status"] is None
