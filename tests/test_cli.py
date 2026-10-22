from typer.testing import CliRunner

from omni import __version__
from omni.cli import app

runner = CliRunner()


def test_version_flag_prints_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"omni {__version__}"


def test_no_args_shows_help():
    result = runner.invoke(app, [])
    assert "Swap-Safety Engine" in result.output
