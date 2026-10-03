"""Descriptive beta-versus-zero and RT diagnostics on trial-by-feature arrays."""

import numpy as np
import pytest
from scipy.stats import ttest_1samp

from boldtailor import diagnostics


def test_trial_weighted_t_test_matches_scipy_and_preserves_inputs():
    runs = [
        np.array([[1, -2, 0], [2, -5, 2]], dtype=np.float32),
        np.array([[4, -6, -1], [7, -1, 3], [8, -3, 1]], dtype=np.float32),
    ]
    before = [run.copy() for run in runs]
    result = diagnostics.one_sample_t(runs)
    pooled = np.concatenate(runs).astype(float)
    expected = ttest_1samp(pooled, 0, axis=0)
    np.testing.assert_allclose(result["mean_beta"], [4.4, -3.4, 1])
    np.testing.assert_allclose(result["t"], expected.statistic)
    np.testing.assert_allclose(result["p_uncorrected"], expected.pvalue)
    np.testing.assert_array_equal(result["n_trials"], [5, 5, 5])
    np.testing.assert_array_equal(result["df"], [4, 4, 4])
    for original, run in zip(before, runs, strict=True):
        np.testing.assert_array_equal(run, original)


def test_missing_values_and_degenerate_columns_are_explicit():
    runs = [
        np.array(
            [
                [1, np.nan, 4, 0, 3],
                [3, np.inf, np.nan, 0, 3],
                [np.nan, np.nan, np.nan, 0, 3],
            ]
        )
    ]
    result = diagnostics.one_sample_t(runs)
    np.testing.assert_allclose(result["mean_beta"], [2, np.nan, 4, 0, 3])
    np.testing.assert_allclose(result["n_trials"], [2, np.nan, 1, 3, 3])
    np.testing.assert_allclose(result["df"], [1, np.nan, 0, 2, 2])
    assert result["t"][0] == pytest.approx(2)
    assert result["p_uncorrected"][0] == pytest.approx(ttest_1samp([1, 3], 0).pvalue)
    assert np.isnan(result["t"][1:]).all()
    assert np.isnan(result["p_uncorrected"][1:]).all()


def test_large_offsets_use_centered_sample_variance():
    values = 1e10 + np.array([[1, -2], [2, 0], [3, 2]], dtype=float)
    result = diagnostics.one_sample_t([values[:1], values[1:]])
    expected = ttest_1samp(values, 0, axis=0)
    np.testing.assert_allclose(result["t"], expected.statistic)


@pytest.mark.parametrize("runs", [[], [np.ones(3)], [np.ones((2, 3)), np.ones((2, 4))]])
def test_invalid_beta_dimensions_are_rejected(runs):
    with pytest.raises(ValueError, match="trial.*grayordinate|at least one"):
        diagnostics.one_sample_t(runs)


def test_correlation_removes_run_mean_confounding():
    beta = [np.array([[3.0], [2.0], [1.0]]), np.array([[13.0], [12.0], [11.0]])]
    rt = [np.array([1.0, 2.0, 3.0]), np.array([11.0, 12.0, 13.0])]
    result = diagnostics.correlate_rt(beta, rt, run_numbers=[1, 2])
    for split in ("all", "odd", "even"):
        np.testing.assert_allclose(result[split], [-1.0])
    np.testing.assert_array_equal(result["counts"]["all"], [6])
    np.testing.assert_allclose(result["per_run"], [[-1.0], [-1.0]])


def test_matched_feature_masks_exclude_only_invalid_rt_or_beta():
    beta = np.array(
        [
            [1.0, 1.0],
            [2.0, np.nan],
            [4.0, 4.0],
            [999.0, 5.0],
            [999.0, 6.0],
            [999.0, 7.0],
        ]
    )
    rt = np.array([1.0, 2.0, 4.0, np.nan, 0.0, -1.0])
    result = diagnostics.correlate_rt([beta], [rt], run_numbers=[11])
    np.testing.assert_allclose(result["all"], [1.0, np.nan], equal_nan=True)
    np.testing.assert_array_equal(result["counts"]["all"], [3, 2])
    assert np.isnan(result["even"]).all()
    np.testing.assert_array_equal(result["counts"]["even"], [0, 0])


def test_constant_rt_and_constant_beta_are_undefined():
    beta = np.column_stack([np.ones(4), np.arange(4)])
    result = diagnostics.correlate_rt([beta], [np.ones(4)], run_numbers=[2])
    assert np.isnan(result["all"]).all()
    result = diagnostics.correlate_rt([beta], [np.arange(1, 5)], run_numbers=[2])
    np.testing.assert_allclose(result["all"], [np.nan, 1.0], equal_nan=True)


def test_even_run_points_are_even_only_and_centered_with_matching_mask():
    beta = [
        np.array([[100.0], [200.0], [300.0]]),
        np.array([[20.0], [np.nan], [40.0], [50.0]]),
    ]
    rt = [np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 4.0, 5.0])]
    x, y = diagnostics.even_run_points(beta, rt, [1, 2], 0)
    np.testing.assert_allclose(x, np.array([2.0, 4.0, 5.0]) - 11 / 3)
    np.testing.assert_allclose(y, np.array([20.0, 40.0, 50.0]) - 110 / 3)


def test_mismatched_trials_and_ambiguous_run_numbers_fail():
    for beta, rt, numbers in [
        ([np.ones((4, 2))], [np.ones(3)], [1]),
        ([np.ones((4, 2))] * 2, [np.ones(4)] * 2, [1, 1]),
        ([np.ones((4, 2))], [np.ones(4)], ["01"]),
    ]:
        with pytest.raises(ValueError):
            diagnostics.correlate_rt(beta, rt, run_numbers=numbers)


def test_map_names_follow_the_returned_statistics():
    result = diagnostics.one_sample_t([np.ones((3, 2))])
    assert tuple(result) == diagnostics.ONE_SAMPLE_T_NAMES


def test_pearson_correlation_needs_three_observations_and_variance():
    r = diagnostics.pearson_correlation(
        np.array([2.0, 2.0, 0.0, 5.0]),
        np.array([4.0, 4.0, 0.0, 1.0]),
        np.array([1.0, 1.0, 1.0, 1.0]),
        np.array([3, 2, 5, 3]),
    )
    np.testing.assert_allclose(r, [1.0, np.nan, np.nan, 1.0], equal_nan=True)
