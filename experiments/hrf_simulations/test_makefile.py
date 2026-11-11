"""The workflow must be inspectable and ordered without launching NSD jobs."""

from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
MAKEFILE = ROOT / "experiments/hrf_simulations/Makefile"


def dry_run(*args):
    assert MAKEFILE.exists(), "simulation Makefile is missing"
    return subprocess.run(
        ["make", "-n", "-f", str(MAKEFILE), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def test_workflow_runs_components_in_order_with_fifteen_workers():
    commands = dry_run("workflow")
    assert commands.index("cli calibrate") < commands.index("cli run")
    assert commands.index("cli run") < commands.index("cli report")
    assert commands.index("cli report") < commands.index("hrf_simulations.render")
    assert '--n-jobs "15"' in commands
    assert "uv run" in commands
    assert "HRF_SIM_REPORT=" in commands
    assert "python -m experiments.hrf_simulations.render" in commands


def test_workflow_overrides_and_all_component_targets():
    commands = dry_run(
        "run", "N_JOBS=2", "STAGE=sensitivity", "LIMIT=3", "REPLICATES=1"
    )
    for expected in (
        '--n-jobs "2"',
        '--stage "sensitivity"',
        "--limit 3",
        "--replicates 1",
    ):
        assert expected in commands
    for target in (
        "help",
        "plan",
        "calibrate",
        "run",
        "report",
        "notebook",
        "smoke",
        "test",
        "all",
    ):
        assert dry_run(target).strip()


def test_completed_calibration_is_verified_without_recomputing(tmp_path):
    (tmp_path / "COMPLETE.json").write_text("{}")
    commands = dry_run("calibrate", f"CALIBRATION={tmp_path}")
    assert "cli calibrate" not in commands
    assert "verify" in commands
