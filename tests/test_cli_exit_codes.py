from typer.testing import CliRunner

from ink_sdk.cli import app

runner = CliRunner()
PROFILE = "ink_9um@scrollprize-seed43-step75000"


def _check(args):
    result = runner.invoke(app, args)
    assert result.exit_code == 2, result.output
    assert isinstance(result.exception, SystemExit)  # no raw traceback


def test_infer_without_inputs_is_a_clean_error():
    _check(["infer", "--dry-run"])


def test_inspect_bad_surface_volume_is_a_clean_error(tmp_path):
    _check(["inspect", "-p", PROFILE, "--surface-volume", str(tmp_path / "nope")])


def test_unknown_profile_is_a_clean_error(tmp_path):
    _check(["inspect", "-p", "nope", "--surface-volume", str(tmp_path)])


def test_render_missing_ct_is_a_clean_error(tmp_path):
    _check(["render", "--tifxyz", str(tmp_path), "--ct", str(tmp_path)])
