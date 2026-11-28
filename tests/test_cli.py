from __future__ import annotations

import json

import click

from typer.testing import CliRunner

from ink_sdk.cli import app

runner = CliRunner()


def test_models_json_lists_all_profiles():
    result = runner.invoke(app, ["models", "--json"])
    assert result.exit_code == 0, result.output
    rows = json.loads(result.output)
    assert len(rows) == 24
    assert {row["id"] for row in rows} == {
        "ink_9um@scrollprize-seed42-step75000",
        "ink_9um@scrollprize-seed43-step75000",
        "ink_9um@dense-native-step016000",
        "ink_9um@reader-v2-step040000",
        "ink_9um@reader-v3",
        "ink_9um@reader-v4",
        "canonical_2um@r152-3ddec-v2-l5-epoch13",
        "hecate@2.4um",
        "hecate@9.6um",
        "timesformer@gp-scroll1",
        "timesformer@large-scroll1",
        "timesformer@scroll5",
        "ink_9um@trace-v3b-step5000",
        "v8in@ink-8um-v8in",
        "v8in@pherc0306b-ahn-s0",
        "v8in@pherc0306b-ahn-s1",
        "v8in@pherc0306b-ahn-s2",
        "ink_9um@rectina-m4",
        "studio@softloss-20261023",
        "ink_9um@jeff-0846a-sub8",
        "studio@jeff-0846a-sub8",
        "v8in@jeff-0846a-sub8",
        "ink_9um@argus-v2",
        "ink_9um@argus-v3",
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


def test_infer_passes_batch_to_the_run(monkeypatch, tmp_path, surface_volume):
    import ink_sdk.cli as cli

    seen = {}
    real = cli._ctx

    def spy(**kwargs):
        ctx = real(**kwargs)
        seen["batch"] = ctx.batch
        return ctx

    monkeypatch.setattr(cli, "_ctx", spy)
    runner.invoke(app, ["infer", "-p", "studio@softloss-20261023", "--surface-volume", str(surface_volume),
                        "--voxel-um", "9.362", "--out", str(tmp_path / "o"), "--batch", "3", "--dry-run"])
    assert seen["batch"] == 3
    args = ["infer", "-p", "studio@softloss-20261023", "--surface-volume", str(surface_volume),
            "--voxel-um", "9.362", "--out", str(tmp_path / "o0"), "--dry-run"]
    assert runner.invoke(app, args + ["--batch", "1"]).exit_code == 0
    result = runner.invoke(app, args + ["--batch", "0"])
    assert result.exit_code == 2  # usage error from the option's min=1, not a later failure
    # Rich colours the usage error on CI (GITHUB_ACTIONS), splitting the option name with escape codes.
    assert "--batch" in click.unstyle(result.output)
