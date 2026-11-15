import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

from boldtailor._ar1_moments import (
    correct_lag_ratio,
    expected_lag_ratio,
    pool_rho,
)
from boldtailor.ar1 import estimate_rho, whiten
from tests.reml_oracles import ar1_correlation


@pytest.fixture
def design():
    return np.random.default_rng(3).normal(size=(40, 6))


def dense_expected(design, rho):
    n = len(design)
    m = np.eye(n) - design @ np.linalg.pinv(design)
    lag = 0.5 * (np.eye(n, k=1) + np.eye(n, k=-1))
    previous = np.eye(n)
    previous[-1, -1] = 0.0
    r = ar1_correlation(n, rho)
    return np.trace(m @ lag @ m @ r) / np.trace(m @ previous @ m @ r)


@settings(derandomize=True, deadline=None)
@given(
    n=st.integers(3, 40),
    rho=st.floats(-0.95, 0.95, exclude_min=True, exclude_max=True),
)
def test_whitened_ar1_covariance_is_identity(n, rho):
    w = whiten(np.eye(n), rho)
    assert_allclose(w @ ar1_correlation(n, rho) @ w.T, np.eye(n), atol=1e-10)


def test_zero_rho_is_identity():
    values = np.random.default_rng(0).normal(size=(10, 3))
    assert_allclose(whiten(values, 0.0), values)


def test_whiten_does_not_modify_input():
    values = np.random.default_rng(0).normal(size=(10, 3))
    before = values.copy()
    whiten(values, 0.4)
    np.testing.assert_array_equal(values, before)


@pytest.mark.parametrize("rho", [1.0, -1.0, np.nan, np.inf])
def test_invalid_rho_raises(rho):
    with pytest.raises(ValueError):
        whiten(np.ones(5), rho)


def test_whiten_accepts_one_dimensional_input():
    out = whiten(np.array([1.0, 2.0, 3.0]), 0.5)
    assert_allclose(out, [1.0, 1.5 / np.sqrt(0.75), 2.0 / np.sqrt(0.75)])


def test_expected_lag_ratio_matches_dense_traces(design):
    rhos = np.array([-0.5, 0.0, 0.3, 0.7])
    expected = np.array([dense_expected(design, r) for r in rhos])
    assert_allclose(expected_lag_ratio(design, rhos), expected, rtol=1e-10)


def test_correction_inverts_expectation(design):
    rhos = np.linspace(-0.8, 0.8, 9)
    recovered = correct_lag_ratio(expected_lag_ratio(design, rhos), design)
    assert_allclose(recovered, rhos, atol=1e-3)


def test_degenerate_feature_is_nan(design):
    rng = np.random.default_rng(5)
    signals = np.column_stack(
        [rng.normal(size=40), design @ rng.normal(size=6), rng.normal(size=40)]
    )
    result = estimate_rho([signals], [design])
    assert np.isnan(result.raw[0, 1]) and np.isnan(result.values[0, 1])
    assert np.isfinite(result.raw[0, [0, 2]]).all()
    finite = result.values[np.isfinite(result.values)]
    assert result.pooled == pytest.approx(np.clip(np.median(finite), -0.8, 0.8))


def test_pooled_is_clipped_median():
    assert pool_rho(np.array([[0.1, 0.2, np.nan, 0.9]])) == pytest.approx(0.2)
    assert pool_rho(np.array([0.95, 0.99, 0.97])) == pytest.approx(0.8)
    assert pool_rho(np.array([-0.99, -0.95])) == pytest.approx(-0.8)


def test_no_valid_values_pools_to_zero():
    assert pool_rho(np.array([np.nan, np.nan])) == 0.0
