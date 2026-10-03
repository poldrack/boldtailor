"""Nested tuning, common recovery metrics, and seed-level experiment exports."""

import importlib

import numpy as np
import pandas as pd
import pytest

from boldtailor.single_trial import project_trial_design


def simulation():
    try:
        return importlib.import_module("fractional_ridge_ablation_simulation")
    except ImportError as error:
        pytest.fail(f"Ablation runner missing: {error}")


@pytest.fixture
def problem():
    return simulation().make_problem(
        0,
        isi=4.0,
        durations="variable",
        noise="ar_hetero",
        hrf_mismatch=True,
        offset=1.0,
        n_trials=12,
    )


def test_generator_is_paired_and_stresses_the_design(problem):
    other = simulation().make_problem(
        0,
        isi=4.0,
        durations="variable",
        noise="ar_hetero",
        hrf_mismatch=True,
        offset=1.0,
        n_trials=12,
    )
    for key in ["signals", "truth"]:
        for a, b in zip(problem[key], other[key]):
            np.testing.assert_array_equal(a, b)
    assert len(problem["signals"]) == 6
    assert problem["truth"][0].shape == (12, 3)
    assert np.ptp(problem["events"][0].duration) > 0
    for (x, n), y, beta, truth_x in zip(
        problem["designs"],
        problem["signals"],
        problem["truth"],
        problem["truth_designs"],
    ):
        assert not np.allclose(x, truth_x)
        assert np.isfinite(y).all()
        assert not np.allclose(y, 25 + x @ beta)
        # Peak normalization equalizes the raw columns by design; the design
        # still stresses the solver through unequal nuisance-residualized scales.
        assert np.max(x, axis=0).min() > 0.9
        assert np.ptp(project_trial_design(x, n).column_scale) > 0.1


def test_noiseless_ols_recovers_truth():
    sim = simulation()
    p = sim.make_problem(
        3,
        isi=12.0,
        durations="uniform",
        noise="none",
        hrf_mismatch=False,
        offset=0.0,
        n_trials=12,
    )
    rows, candidates = sim.evaluate_problem(p, fractions=(1.0,))
    assert len(rows) == 51 and len(candidates) == 24
    for row in rows:
        assert row["beta_rmse"] < 1e-9
        assert row["within_run_beta_rmse"] < 1e-9
        assert row["amplitude_ratio"] == pytest.approx(1)
        assert row["within_run_amplitude_ratio"] == pytest.approx(1)
        assert row["selected_fraction"] == 1


def test_outer_outcomes_and_latent_truth_do_not_select_candidates(problem):
    sim = simulation()
    first, curve = sim.evaluate_problem(problem, fractions=(1.0, 0.5))
    altered = dict(problem)
    altered["signals"] = [
        *problem["signals"][:4],
        problem["signals"][4] * -3 + 12,
        problem["signals"][5] * 4,
    ]
    altered["truth"] = [b * 100 + 50 for b in problem["truth"]]
    second, changed_curve = sim.evaluate_problem(altered, fractions=(1.0, 0.5))
    pd.testing.assert_frame_equal(pd.DataFrame(curve), pd.DataFrame(changed_curve))
    keys = [
        "config",
        "calibration",
        "feature",
        "selected_fraction",
        "inner_r2",
        "scale",
        "offset",
        "training_alpha_mean",
    ]
    pd.testing.assert_frame_equal(pd.DataFrame(first)[keys], pd.DataFrame(second)[keys])
    assert not np.allclose(
        [r["beta_rmse"] for r in first], [r["beta_rmse"] for r in second]
    )


def test_metrics_keep_offsets_and_shape_errors_distinct():
    sim = simulation()
    truth = [np.array([1.0, 3.0, 5.0])[:, None], np.array([11.0, 13.0, 15.0])[:, None]]
    estimated = [2 * x + 10 for x in truth]
    values = sim.recovery_metrics(estimated, truth, truth, truth)
    assert values["within_run_beta_rmse"][0] == pytest.approx(np.sqrt(8 / 3))
    assert values["within_run_amplitude_ratio"][0] == pytest.approx(2)
    assert values["within_run_correlation"][0] == pytest.approx(1)
    assert values["common_ols_r2"][0] == pytest.approx(1)
    assert values["beta_rmse"][0] > 10
