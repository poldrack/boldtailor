import importlib

import numpy as np
import pytest


def diagnostics():
    try:
        return importlib.import_module("examples.NSD.rt_diagnostics")
    except ModuleNotFoundError:
        pytest.fail("RT diagnostics are not implemented")


def test_correlation_removes_run_mean_confounding():
    beta = [np.array([[3.], [2.], [1.]]), np.array([[13.], [12.], [11.]])]
    rt = [np.array([1., 2., 3.]), np.array([11., 12., 13.])]
    result = diagnostics().correlate_rt(beta, rt, run_numbers=[1, 2])
    for split in ("all", "odd", "even"):
        np.testing.assert_allclose(result[split], [-1.])
    np.testing.assert_array_equal(result["counts"]["all"], [6])
    np.testing.assert_allclose(result["per_run"], [[-1.], [-1.]])


def test_matched_feature_masks_exclude_only_invalid_rt_or_beta():
    beta = np.array([[1., 1.], [2., np.nan], [4., 4.], [999., 5.], [999., 6.], [999., 7.]])
    rt = np.array([1., 2., 4., np.nan, 0., -1.])
    result = diagnostics().correlate_rt([beta], [rt], run_numbers=[11])
    np.testing.assert_allclose(result["all"], [1., np.nan], equal_nan=True)
    np.testing.assert_array_equal(result["counts"]["all"], [3, 2])
    assert np.isnan(result["even"]).all()
    np.testing.assert_array_equal(result["counts"]["even"], [0, 0])


def test_constant_rt_and_constant_beta_are_undefined():
    beta = np.column_stack([np.ones(4), np.arange(4)])
    result = diagnostics().correlate_rt([beta], [np.ones(4)], run_numbers=[2])
    assert np.isnan(result["all"]).all()
    result = diagnostics().correlate_rt([beta], [np.arange(1, 5)], run_numbers=[2])
    np.testing.assert_allclose(result["all"], [np.nan, 1.], equal_nan=True)


def test_select_cortex_stable_ties_and_no_even_run_leakage():
    rt = [np.arange(1, 5.), np.arange(1, 5.)]
    beta = [np.column_stack([rt[0], -rt[0], rt[0], np.ones(4)]), np.zeros((4, 4))]
    api = diagnostics()
    first = api.correlate_rt(beta, rt, run_numbers=[1, 2])
    mask = np.array([True, True, False, True])
    chosen = api.select_vertices(first["odd"], mask)
    np.testing.assert_array_equal(chosen, [0, 1])
    beta[1] = np.arange(16).reshape(4, 4)
    rt[1] = rt[1][::-1]
    changed = api.correlate_rt(beta, rt, run_numbers=[1, 2])
    np.testing.assert_array_equal(api.select_vertices(changed["odd"], mask), chosen)
    assert len(api.select_vertices(np.full(4, np.nan), mask)) == 0


def test_scatter_points_are_even_only_and_centered_with_matching_mask():
    api = diagnostics()
    beta = [np.array([[100.], [200.], [300.]]), np.array([[20.], [np.nan], [40.], [50.]])]
    rt = [np.array([1., 2., 3.]), np.array([2., 3., 4., 5.])]
    x, y = api.even_run_points(beta, rt, [1, 2], 0)
    np.testing.assert_allclose(x, np.array([2., 4., 5.]) - 11 / 3)
    np.testing.assert_allclose(y, np.array([20., 40., 50.]) - 110 / 3)
    artifact = api.scatter_artifact({"OLS": beta}, rt, [1, 2], [0], "scatter.png")
    assert artifact.path == "scatter.png"
    assert artifact.payload.startswith(b"\x89PNG\r\n\x1a\n")
    empty = api.scatter_artifact({"OLS": beta}, rt, [1, 2], [], "empty.png")
    assert empty.payload.startswith(b"\x89PNG")


def test_mismatched_trials_and_ambiguous_run_numbers_fail():
    for beta, rt, numbers in [([np.ones((4, 2))], [np.ones(3)], [1]),
                               ([np.ones((4, 2))] * 2, [np.ones(4)] * 2, [1, 1]),
                               ([np.ones((4, 2))], [np.ones(4)], ["01"])]:
        with pytest.raises(ValueError):
            diagnostics().correlate_rt(beta, rt, run_numbers=numbers)
