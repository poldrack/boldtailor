import importlib

import numpy as np
import pytest


def diagnostics():
    try:
        return importlib.import_module("examples.NSD.rt_diagnostics")
    except ModuleNotFoundError:
        pytest.fail("RT diagnostics are not implemented")


def test_select_cortex_stable_ties_and_no_even_run_leakage():
    rt = [np.arange(1, 5.0), np.arange(1, 5.0)]
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
    beta = [
        np.array([[100.0], [200.0], [300.0]]),
        np.array([[20.0], [np.nan], [40.0], [50.0]]),
    ]
    rt = [np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 4.0, 5.0])]
    x, y = api.even_run_points(beta, rt, [1, 2], 0)
    np.testing.assert_allclose(x, np.array([2.0, 4.0, 5.0]) - 11 / 3)
    np.testing.assert_allclose(y, np.array([20.0, 40.0, 50.0]) - 110 / 3)
    artifact = api.scatter_artifact({"OLS": beta}, rt, [1, 2], [0], "scatter.png")
    assert artifact.path == "scatter.png"
    assert artifact.payload.startswith(b"\x89PNG\r\n\x1a\n")
    empty = api.scatter_artifact({"OLS": beta}, rt, [1, 2], [], "empty.png")
    assert empty.payload.startswith(b"\x89PNG")
