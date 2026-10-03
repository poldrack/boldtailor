"""Workflow plots return figures that preserve the plotted values."""

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from boldtailor.hrf_library import HrfLibrary
from boldtailor.workflow import plots
from boldtailor.workflow.beta_series import BetaModel


@pytest.fixture(autouse=True)
def clean_figures():
    yield
    plt.close("all")


@pytest.fixture
def library():
    return HrfLibrary.from_parameters([[3, 10, 0.5, 0.5, 2, 0, 36]])


def make_model(name, rt, betas):
    return BetaModel(
        name=name, hrf="canonical", estimator="OLS", fit={"rt": rt, "betas": betas}
    )


@pytest.fixture
def two_models_without_rt():
    return {
        name: make_model(name, None, [])
        for name in ("CanonicalTrialOLS", "OptimizedTrialOLS")
    }


@pytest.fixture
def rt_runs():
    """Runs 1 and 2 with four trials each; only run 2 is an even run."""
    rt = [0.5, 0.7, 0.9, 1.1]
    events = pd.DataFrame({"response_time": rt})
    return [SimpleNamespace(number=n, events=events) for n in (1, 2)]


@pytest.fixture
def two_models_with_rt():
    odd_r = np.array([np.nan, 0.2, -0.9])
    rt = {"all": odd_r, "odd": odd_r, "even": odd_r}
    betas = [np.arange(12.0).reshape(4, 3), np.arange(12.0).reshape(4, 3)[::-1]]
    return {
        name: make_model(name, rt, betas)
        for name in ("CanonicalTrialOLS", "OptimizedTrialOLS")
    }


def test_design_figure_plots_each_regressor_against_time():
    frame_times = np.arange(5.0)
    design = pd.DataFrame({"a": np.arange(5.0), "b": np.ones(5)})
    fig = plots.design_figure(frame_times, design, ["a", "b"])
    assert [line.get_label() for line in fig.axes[0].lines] == ["a", "b"]
    np.testing.assert_allclose(fig.axes[0].lines[0].get_ydata(), np.arange(5.0))


def test_library_figure_has_one_black_canonical_curve_and_colorbar(library):
    fig = plots.library_figure(library)
    assert len(fig.axes) == 1  # a one-candidate library has no colored curves
    assert fig.axes[0].lines[0].get_color() == "black"


def test_glm_comparison_uses_paired_values_and_signed_difference():
    canonical = {
        "r2": np.array([[0.2, np.nan, 0.8], [0.1, 0.1, 0.1], [0.1, np.nan, 0.7]])
    }
    optimized = {"r2": np.array([[0.4, 0.5, 0.6], [0.1, 0.1, 0.1], [0.3, 0.4, 0.5]])}
    table, fig = plots.glm_comparison(
        {"CanonicalGLM": canonical, "OptimizedGLM": optimized}
    )
    row = table[(table.model == "CanonicalGLM") & (table.statistic == "full R²")].iloc[
        0
    ]
    assert row.n == 2
    assert row["median"] == pytest.approx(0.5)
    np.testing.assert_allclose(
        fig.axes[0].collections[0].get_offsets(), [[0.2, 0.4], [0.8, 0.6]]
    )
    assert sum(p.get_height() for p in fig.axes[1].patches) == 2
    assert fig.axes[1].get_xlim()[0] < -0.2


@pytest.mark.parametrize("missing", [False, True])
def test_parameter_agreement_excludes_unpaired_and_constant_values(library, missing):
    odd = np.array([0, 1, np.nan, 0.0])
    even = np.array([0, 1, 1, np.nan])
    if missing:
        odd[:] = np.nan
    table, fig = plots.parameter_agreement(library, odd, even)
    assert (table.grayordinates == (0 if missing else 2)).all()
    for name in ("response_delay", "peak_time"):
        value = table.loc[table.parameter == name, "pearson_r"].iloc[0]
        assert np.isnan(value) if missing else value == pytest.approx(1)
    if not missing:
        # Canonical duration is 32 s and the custom HRF is 36 s; onset is
        # the constant parameter in this fixture (zero for both curves).
        assert np.isnan(
            table.loc[table.parameter == "onset_delay", "pearson_r"].iloc[0]
        )
    assert len(fig.axes) == len(table)


@pytest.mark.parametrize("missing", [False, True])
def test_curve_agreement_uses_same_grayordinates_for_every_comparison(missing):
    values = np.array([[0.2, np.nan, 0.4], [0.6, 0.7, 0.8], [0.1, 0.3, 0.5]])
    if missing:
        values[:] = np.nan
    original = values.copy()
    table, fig = plots.curve_agreement(values)
    assert (table.grayordinates == (0 if missing else 2)).all()
    if missing:
        assert table["median"].isna().all()
        assert fig.axes[1].texts[0].get_text() == "No paired HRFs"
    else:
        np.testing.assert_allclose(table["median"], [0.3, 0.7, 0.3])
        np.testing.assert_allclose(table.q25, [0.25, 0.65, 0.2])
    np.testing.assert_array_equal(values, original)


def test_activation_histogram_draws_one_finite_histogram_per_model():
    activation = {
        "A": {"t": np.array([1.0, 2.0, np.nan, -1.0])},
        "B": {"t": np.array([0.5, np.nan, np.nan, 3.0])},
    }
    fig = plots.activation_histogram(activation)
    ax = fig.axes[0]
    assert len(ax.patches) == 2  # one step polygon per model
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["A", "B"]
    assert len(ax.lines) == 1  # the zero reference line
    assert ax.get_xlabel().startswith("Mean trial beta versus zero")


def test_rt_check_figure_is_none_without_reaction_times(two_models_without_rt):
    assert plots.rt_check_figure(two_models_without_rt, runs=[], brain=None) is None


def test_rt_check_figure_plots_even_run_points_at_best_odd_vertex(
    two_models_with_rt, rt_runs
):
    brain = SimpleNamespace(vertex=np.array([0, 1, 2]))
    fig = plots.rt_check_figure(two_models_with_rt, rt_runs, brain)
    assert len(fig.axes) == 2
    for ax in fig.axes:
        assert "grayordinate 2" in ax.get_title()  # largest finite |odd r|
        assert len(ax.collections[0].get_offsets()) == 4  # run 2 only


def test_rt_check_figure_reports_no_eligible_vertices(two_models_with_rt, rt_runs):
    brain = SimpleNamespace(vertex=np.array([-1, -1, -1]))
    fig = plots.rt_check_figure(two_models_with_rt, rt_runs, brain)
    assert [ax.texts[0].get_text() for ax in fig.axes] == [
        "No eligible cortical vertices"
    ] * 2


def test_fraction_selection_figure_has_one_bar_panel_per_model():
    rows = [
        dict(mode=mode, scope=scope, fraction=f, selected_grayordinates=n)
        for mode in ("Canonical", "Optimized")
        for scope in ("odd", "even")
        for f, n in ((0.1, 3), (0.9, 5))
    ]
    fig = plots.fraction_selection_figure(pd.DataFrame(rows))
    assert [ax.get_title() for ax in fig.axes] == ["Canonical", "Optimized"]
    for ax in fig.axes:
        assert len(ax.patches) == 4  # two fractions x two scopes
        assert ax.get_xlabel() == "Selected fraction"
