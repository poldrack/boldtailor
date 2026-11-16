"""The documented integration command must select real cross-module checks."""

from pathlib import Path
import subprocess
import sys


def test_integration_selection_covers_public_boundaries_without_unit_cases():
    """Verify integration selection covers public boundaries without unit cases."""
    root = Path(__file__).resolve().parents[1]
    collected = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-m",
            "integration",
            "tests/test_fit.py",
            "tests/test_reml_selection.py",
            "tests/workflow/test_run.py",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert collected.returncode == 0, collected.stdout + collected.stderr
    selected = {line for line in collected.stdout.splitlines() if "::" in line}
    assert "tests/test_fit.py::test_fit_matches_nilearn_ols_contrast" in selected
    assert "tests/test_reml_selection.py::test_planted_design_wins" in selected
    assert (
        "tests/workflow/test_run.py::test_full_run_writes_every_stage_and_the_report"
        in selected
    )
    assert "tests/test_fit.py::test_fit_returns_readonly_arrays" not in selected
    assert "tests/test_reml_selection.py::test_results_are_read_only" not in selected
