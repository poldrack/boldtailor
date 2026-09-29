"""Example presentation preserves plotted values without loading or fitting data."""

import importlib

import matplotlib
import numpy as np
import pandas as pd
import pytest

from boldtailor.hrf_library import HrfLibrary
from examples.NSD.session_hrf_reliability import compare_hrfs

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
def library():
    return HrfLibrary.from_parameters([[3, 10, 0.5, 0.5, 2, 0, 36]])


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


def test_explicit_paths_override_environment(no_path_environment, monkeypatch, tmp_path):
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


def test_library_plot_preserves_curves_and_peak_colors(library):
    fig = helper("workflow_plots").library_figure(library)
    axis = fig.axes[0]
    np.testing.assert_array_equal(axis.lines[0].get_xdata(), library.times)
    np.testing.assert_array_equal(axis.lines[0].get_ydata(), library.curves[0])
    np.testing.assert_array_equal(
        axis.collections[0].get_segments()[0][:, 1], library.curves[1]
    )
    np.testing.assert_array_equal(
        axis.collections[0].get_array(), library.parameter_table.peak_time.to_numpy()[1:]
    )


def test_design_plot_retains_original_acquisition_times():
    times = np.array([4.2, 5.8, 7.4])
    design = pd.DataFrame({"task": [1, 2, 3], "response_time": [4, 5, 6]})
    fig = helper("workflow_plots").design_figure(times, design, list(design))
    for line, name in zip(fig.axes[0].lines, design, strict=True):
        np.testing.assert_array_equal(line.get_xdata(), times)
        np.testing.assert_array_equal(line.get_ydata(), design[name])


def test_glm_comparison_uses_paired_values_and_signed_difference():
    canonical = {"r2": np.array([[0.2, np.nan, 0.8], [0.1, 0.1, 0.1], [0.1, np.nan, 0.7]])}
    optimized = {"r2": np.array([[0.4, 0.5, 0.6], [0.1, 0.1, 0.1], [0.3, 0.4, 0.5]])}
    table, fig = helper("workflow_plots").glm_comparison(
        {"CanonicalGLM": canonical, "OptimizedGLM": optimized}
    )
    row = table[(table.model == "CanonicalGLM") & (table.statistic == "full R²")].iloc[0]
    assert row.n == 2
    assert row["median"] == pytest.approx(0.5)
    np.testing.assert_allclose(fig.axes[0].collections[0].get_offsets(), [[0.2, 0.4], [0.8, 0.6]])
    assert sum(p.get_height() for p in fig.axes[1].patches) == 2
    assert fig.axes[1].get_xlim()[0] < -0.2


@pytest.mark.parametrize("missing", [False, True])
def test_parameter_agreement_excludes_unpaired_and_constant_values(library, missing):
    odd = np.array([0, 1, np.nan, 0.])
    even = np.array([0, 1, 1, np.nan])
    if missing:
        odd[:] = np.nan
    table, fig = helper("workflow_plots").parameter_agreement(library, odd, even)
    assert (table.grayordinates == (0 if missing else 2)).all()
    for name in ("response_delay", "peak_time"):
        value = table.loc[table.parameter == name, "pearson_r"].iloc[0]
        assert np.isnan(value) if missing else value == pytest.approx(1)
    if not missing:
        assert np.isnan(table.loc[table.parameter == "duration", "pearson_r"].iloc[0])
    assert len(fig.axes) == len(table)


@pytest.mark.parametrize("missing", [False, True])
def test_curve_agreement_uses_same_grayordinates_for_every_comparison(missing):
    values = np.array([[0.2, np.nan, 0.4], [0.6, 0.7, 0.8], [0.1, 0.3, 0.5]])
    if missing:
        values[:] = np.nan
    original = values.copy()
    table, fig = helper("workflow_plots").curve_agreement(values)
    assert (table.grayordinates == (0 if missing else 2)).all()
    if missing:
        assert table["median"].isna().all()
        assert fig.axes[1].texts[0].get_text() == "No paired HRFs"
    else:
        np.testing.assert_allclose(table["median"], [0.3, 0.7, 0.3])
        np.testing.assert_allclose(table.q25, [0.25, 0.65, 0.2])
    np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize("missing", [False, True])
def test_session_figures_preserve_pair_means_and_undefined_sessions(library, missing):
    ids = np.array([[0, 1, np.nan], [0, 1, np.nan], [np.nan, np.nan, np.nan]])
    if missing:
        ids[:] = np.nan
    comparison = compare_hrfs(library, ids, ["a", "b", "c"])
    plots = helper("session_hrf_plots")
    fig = plots.agreement_figure(comparison)
    matrix = np.asarray(fig.axes[0].images[0].get_array())
    assert matrix.shape == (4, 4)
    assert np.isnan(matrix[2, :3]).all()
    assert matrix[-1, -1] == 1
    if missing:
        assert np.isnan(matrix[:3]).all()
    else:
        np.testing.assert_allclose(matrix[:2, :2], 1)
    variability = plots.parameter_variability_figure(comparison)
    assert len(variability.axes) == 2
    for ax in variability.axes:
        assert sum(p.get_height() for p in ax.patches) == (0 if missing else 2)
