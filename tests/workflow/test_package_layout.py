"""The workflow subpackage is importable, empty at the package level, and ships its runtime deps."""

import importlib
import tomllib
from pathlib import Path


def test_workflow_package_is_importable():
    """Verify workflow package is importable."""
    package = importlib.import_module("boldtailor.workflow")
    assert package is not None


def test_matplotlib_is_a_runtime_dependency_and_the_script_is_declared():
    """Verify matplotlib is a runtime dependency and the script is declared."""
    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
    assert any(dep.startswith("matplotlib") for dep in project["dependencies"])
    assert project["scripts"] == {"boldtailor": "boldtailor.cli:main"}
