from __future__ import annotations

import json

from typer.testing import CliRunner

from ink_sdk.cli import app

runner = CliRunner()


def test_models_json_lists_all_profiles():
    result = runner.invoke(app, ["models", "--json"])
    assert result.exit_code == 0, result.output
    rows = json.loads(result.output)
    assert len(rows) == 5
    assert {row["id"] for row in rows} == {
        "ink_9um@scrollprize-seed43-step75000",
        "ink_9um@reader-v2-step040000",
        "canonical_2um@r152-3ddec-v2-l5-epoch13",
        "hecate@2.4um",
        "hecate@9.6um",
    }


def test_doctor_json_returns_checks():
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["sdk_version"]
    assert any(check["name"] == "python" for check in data["checks"])


def test_inspect_surface_volume(tmp_path, surface_volume):
    result = runner.invoke(
        app,
        [
            "inspect",
            "--profile",
            "canonical_2um@r152-3ddec-v2-l5-epoch13",
            "--surface-volume",
            str(surface_volume),
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["profile"] == "canonical_2um@r152-3ddec-v2-l5-epoch13"
    assert data["scale_plan"]["mode"] == "native"


def test_inspect_rejects_incompatible_scale(tmp_path):
    import numpy as np
    import zarr

    volume = tmp_path / "coarse.zarr"
    group = zarr.open_group(str(volume), mode="w", zarr_format=2)
    group.create_array("0", shape=(4, 4, 4), dtype="u1")
    group["0"][:] = np.zeros((4, 4, 4), dtype=np.uint8)
    group.attrs["pixel_um"] = [9.362, 9.362, 9.362]
    result = runner.invoke(
        app,
        [
            "inspect",
            "--profile",
            "canonical_2um@r152-3ddec-v2-l5-epoch13",
            "--surface-volume",
            str(volume),
        ],
    )
    assert result.exit_code == 2
    assert "2.399" in result.output or "incompatible" in result.output.lower()
