"""Example presentation preserves plotted values without loading or fitting data."""

import importlib

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def helper(module):
    try:
        return importlib.import_module(f"examples.NSD.{module}")
    except ModuleNotFoundError as error:
        pytest.fail(f"Missing example-local helper: {error}")


@pytest.fixture(autouse=True)
def clean_figures():
    yield
    plt.close("all")


@pytest.fixture
def no_path_environment(monkeypatch):
    for name in ("NSD_BIDS_ROOT", "NSD_FMRIPREP_ROOT", "NSD_OUTPUT_ROOT"):
        monkeypatch.delenv(name, raising=False)


def test_paths_require_explicit_data_location(no_path_environment, tmp_path):
    with pytest.raises(ValueError, match="NSD_BIDS_ROOT"):
        helper("notebook_paths").notebook_paths()
    assert not list(tmp_path.iterdir())


def test_paths_derive_from_root_without_creating_directories(
    no_path_environment, monkeypatch, tmp_path
):
    root = tmp_path / "bids"
    monkeypatch.setenv("NSD_BIDS_ROOT", str(root))
    result = helper("notebook_paths").notebook_paths()
    assert result == {
        "bids_root": str(root),
        "fmriprep_root": str(root / "derivatives/fmriprep-25.2.5"),
        "output_root": str(root / "derivatives/boldtailor"),
    }
    assert not root.exists()


def test_explicit_paths_override_environment(
    no_path_environment, monkeypatch, tmp_path
):
    for name in ("NSD_BIDS_ROOT", "NSD_FMRIPREP_ROOT", "NSD_OUTPUT_ROOT"):
        monkeypatch.setenv(name, str(tmp_path / "environment"))
    overrides = {
        "bids_root": tmp_path / "data",
        "fmriprep_root": tmp_path / "prep",
        "output_root": tmp_path / "results",
        "n_jobs": 1,
    }
    result = helper("notebook_paths").notebook_paths(overrides)
    assert result == {k: str(v) for k, v in overrides.items() if k != "n_jobs"}
    assert overrides["n_jobs"] == 1


def test_empty_root_gets_actionable_error(no_path_environment):
    with pytest.raises(ValueError, match="NSD_BIDS_ROOT"):
        helper("notebook_paths").notebook_paths({"bids_root": "  "})
