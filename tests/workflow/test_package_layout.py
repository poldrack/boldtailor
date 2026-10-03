"""The workflow subpackage is importable, empty at the package level, and ships its runtime deps."""

import importlib
import tomllib
from pathlib import Path


def test_workflow_package_is_empty_and_importable():
    package = importlib.import_module("boldtailor.workflow")
    assert Path(package.__file__).read_text() == ""


def test_matplotlib_is_a_runtime_dependency_and_the_script_is_declared():
    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
    assert any(dep.startswith("matplotlib") for dep in project["dependencies"])
    assert project["scripts"] == {"boldtailor": "boldtailor.cli:main"}
