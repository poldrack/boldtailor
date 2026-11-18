"""The command-line entry points run end to end on synthetic inputs."""

from experiments.hrf_simulations import cli


def test_smoke_command_fits_and_reports_the_synthetic_case(tmp_path, capsys):
    """The smoke command builds its small library, fits one case and writes a report."""
    output = tmp_path / "smoke"
    cli.main(["smoke", "--output", str(output)])
    assert "Synthetic smoke complete" in capsys.readouterr().out
    assert (output / "case-synthetic-smoke").is_dir()
    assert any((output / "report").iterdir())
