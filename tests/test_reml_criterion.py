import numpy as np
import pytest
from numpy.testing import assert_allclose

from boldtailor.reml import (
    criterion,
    criterion_slope,
    profiled_variance,
    project_fixed,
    reml_terms,
)
from tests.reml_oracles import dense_criterion, dense_profiled_variance, simulate

RATIOS = [0.0, 0.01, 1.0, 10.0, 1e3]
N_FEATURES = 3


def _build(rho):
    signals, fixed, random = simulate(
        np.random.default_rng(5),
        scans=(40, 55, 48),
        n_features=N_FEATURES,
        rho=rho,
    )
    terms = [
        reml_terms(project_fixed(y, c, rho), z)
        for y, c, z in zip(signals, fixed, random, strict=True)
    ]
    return signals, fixed, random, terms


def _at(ratio):
    return np.full(N_FEATURES, ratio)


@pytest.fixture
def iid():
    return _build(0.0)


@pytest.fixture
def ar1():
    return _build(0.5)


def _assert_differences_match(data, rho):
    signals, fixed, random, terms = data
    base = criterion(_at(0.0), terms)
    dense_base = dense_criterion(signals, fixed, random, 0.0, rho)
    for r in RATIOS:
        expected = dense_criterion(signals, fixed, random, r, rho) - dense_base
        assert_allclose(criterion(_at(r), terms) - base, expected, rtol=1e-9, atol=1e-9)


def test_criterion_differences_match_dense_iid(iid):
    """Verify criterion differences match dense independent-noise."""
    _assert_differences_match(iid, 0.0)


def test_criterion_differences_match_dense_ar1(ar1):
    """Verify criterion differences match dense AR(1)."""
    _assert_differences_match(ar1, 0.5)


@pytest.mark.parametrize("rho", [0.0, 0.5])
def test_profiled_variance_matches_dense_quadratic_form(rho):
    """Verify profiled variance matches dense quadratic form."""
    signals, fixed, random, terms = _build(rho)
    for r in RATIOS:
        expected = dense_profiled_variance(signals, fixed, random, r, rho)
        assert_allclose(profiled_variance(_at(r), terms), expected, rtol=1e-9)


def test_criterion_broadcasts_over_leading_axes(iid):
    """Verify criterion broadcasts over leading axes."""
    terms = iid[3]
    ratios = np.random.default_rng(1).uniform(0, 5, size=(7, N_FEATURES))
    stacked = criterion(ratios, terms)
    assert stacked.shape == ratios.shape
    separate = np.stack([criterion(row, terms) for row in ratios])
    assert_allclose(stacked, separate, rtol=1e-12)


def test_no_random_directions_is_constant_in_ratio():
    """Verify no random directions is constant in ratio."""
    signals, fixed, _ = simulate(
        np.random.default_rng(2), scans=(40, 30), n_features=N_FEATURES
    )
    terms = [
        reml_terms(project_fixed(y, c), None)
        for y, c in zip(signals, fixed, strict=True)
    ]
    values = [criterion(_at(r), terms) for r in RATIOS]
    for value in values:
        assert_allclose(value, values[0], rtol=1e-12)


SLOPE_RATIOS = [1e-3, 0.1, 1.0, 10.0, 1e3]


def _central_difference(terms, ratio, step):
    upper = criterion(_at(ratio * np.exp(step)), terms)
    lower = criterion(_at(ratio * np.exp(-step)), terms)
    return (upper - lower) / (2 * step)


def _richardson_difference(terms, ratio, step=1e-3):
    """Central difference in log ratio with the O(step**2) error removed."""
    coarse = _central_difference(terms, ratio, step)
    fine = _central_difference(terms, ratio, step / 2)
    return (4 * fine - coarse) / 3


@pytest.mark.parametrize("rho", [0.0, 0.5])
def test_slope_matches_finite_differences(rho):
    """Verify slope matches finite differences."""
    terms = _build(rho)[3]
    for r in SLOPE_RATIOS:
        expected = _richardson_difference(terms, r)
        assert_allclose(criterion_slope(_at(r), terms), expected, rtol=1e-6)


def test_joint_value_and_slope_match_separate_evaluations(ar1):
    """Verify joint value and slope match separate evaluations."""
    from boldtailor.reml import pool_terms, pooled_value_and_slope

    terms = ar1[3]
    ratios = np.array([_at(r) for r in RATIOS])
    value, slope = pooled_value_and_slope(ratios, pool_terms(terms))
    assert_allclose(value, criterion(ratios, terms), rtol=1e-13)
    assert_allclose(slope, criterion_slope(ratios, terms), rtol=1e-13, atol=1e-13)
