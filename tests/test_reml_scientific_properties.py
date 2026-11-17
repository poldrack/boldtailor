"""Generated independent scientific oracles for REML and AR(1).

Tests target lost residual energy, incorrect likelihood/score algebra,
confounding classification, and temporal covariance invariances.
"""

import numpy as np
from hypothesis import given, settings, strategies as st
from numpy.testing import assert_allclose, assert_array_equal

from boldtailor._ar1_moments import basis_expected_lag_ratio, invert_lag_ratio
from boldtailor.reml import (
    PooledTerms,
    RemlGroup,
    _confounded,
    fit_reml,
    minimize_ratio,
    pooled_criterion,
    pooled_quadratic,
    pooled_slope,
    project_fixed,
    reml_terms,
)

PROPERTIES = settings(max_examples=50, deadline=None, derandomize=True)
SEEDS = st.integers(0, 2**32 - 1)


@PROPERTIES
@given(SEEDS, st.integers(0, 10), st.floats(-10, 0))
def test_known_orthogonal_energy_survives_random_projection(seed, rank, log_noise):
    """Verify known orthogonal energy survives random projection."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(12, 12)))
    fixed, random = q[:, :2], q[:, 2 : 2 + rank]
    coefficients = rng.normal(size=(rank, 2)) * 1e4
    residual = q[:, 2 + rank :] @ np.full((10 - rank, 2), 10.0**log_noise)
    signal = random @ coefficients + residual
    terms = reml_terms(project_fixed(signal, fixed), random)
    expected = np.linalg.norm(residual, axis=0)
    # Error is bounded in amplitude units, not relative to the tiny variance.
    tolerance = 50 * 12 * np.finfo(float).eps * np.linalg.norm(signal, axis=0)
    assert np.all(np.abs(np.sqrt(terms.residual_ss) - expected) <= tolerance)
    if rank == 10:
        assert_array_equal(terms.residual_ss, 0.0)


def _diagonal_problem(seed, rank, dof=12):
    rng = np.random.default_rng(seed)
    spectrum = 10.0 ** rng.uniform(-4, 4, size=rank)
    signal = rng.normal(size=dof)
    terms = PooledTerms(
        spectrum,
        signal[:rank, None] ** 2,
        np.array([signal[rank:] @ signal[rank:]]),
        dof,
    )
    return terms, signal, np.r_[spectrum, np.zeros(dof - rank)]


def _dense_value(signal, spectrum, ratio):
    covariance = np.diag(1 + ratio * spectrum)
    quadratic = signal @ np.linalg.solve(covariance, signal)
    return (
        len(signal) * np.log(quadratic / len(signal)) + np.linalg.slogdet(covariance)[1]
    )


@PROPERTIES
@given(SEEDS, st.integers(0, 12), st.floats(-5, 5))
def test_pooled_likelihood_and_score_match_independent_covariance(
    seed, rank, log_ratio
):
    """Verify pooled likelihood and score match independent covariance."""
    terms, signal, spectrum = _diagonal_problem(seed, rank)
    ratio = 10.0**log_ratio
    actual = pooled_criterion(np.array([ratio]), terms)[0]
    assert_allclose(
        actual, _dense_value(signal, spectrum, ratio), rtol=1e-10, atol=1e-10
    )
    expected_q = signal @ np.linalg.solve(np.diag(1 + ratio * spectrum), signal)
    assert_allclose(pooled_quadratic(np.array([ratio]), terms), expected_q, rtol=1e-12)
    assert pooled_quadratic(np.array([2 * ratio]), terms)[0] <= expected_q * (1 + 1e-14)
    step = 1e-4
    difference = (
        _dense_value(signal, spectrum, ratio * np.exp(step))
        - _dense_value(signal, spectrum, ratio * np.exp(-step))
    ) / (2 * step)
    assert_allclose(
        pooled_slope(np.array([ratio]), terms), difference, atol=2e-7, rtol=2e-6
    )


@PROPERTIES
@given(SEEDS, st.integers(1, 12))
def test_searched_likelihood_beats_independent_feasible_points(seed, rank):
    """Verify searched likelihood beats independent feasible points."""
    terms, signal, spectrum = _diagonal_problem(seed, rank)
    optimum = minimize_ratio(terms)
    feasible = [
        _dense_value(signal, spectrum, r) for r in np.r_[0, np.logspace(-8, 8, 101)]
    ]
    assert optimum.criterion[0] <= min(feasible) + 1e-8


@PROPERTIES
@given(
    st.integers(2, 30),
    st.integers(2, 30),
    st.floats(-6, 6),
    st.sampled_from(["common", "different", "missing"]),
)
def test_confounding_requires_full_rank_and_a_common_group_scale(
    n1, n2, exponent, kind
):
    """Verify confounding requires full rank and a common group scale."""
    spectrum = np.full(n1 + n2, 10.0**exponent)
    if kind == "different":
        spectrum[n1:] *= 2
    if kind == "missing":
        spectrum = spectrum[:-1]
    terms = PooledTerms(spectrum, np.ones((len(spectrum), 1)), np.ones(1), n1 + n2)
    assert _confounded(terms) == (kind == "common")


@PROPERTIES
@given(SEEDS, st.integers(4, 24), st.floats(-0.9, 0.9))
def test_ar1_moment_polynomials_match_dense_restricted_traces(seed, n, rho):
    """Verify AR(1) moment polynomials match dense restricted traces."""
    rng = np.random.default_rng(seed)
    rank = int(rng.integers(0, n))
    basis = np.linalg.qr(rng.normal(size=(n, n)))[0][:, :rank]
    m = np.eye(n) - basis @ basis.T
    covariance = rho ** np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    lag = (np.eye(n, k=1) + np.eye(n, k=-1)) / 2
    previous = np.diag(np.r_[np.ones(n - 1), 0])
    expected = np.trace(m @ lag @ m @ covariance) / np.trace(
        m @ previous @ m @ covariance
    )
    assert_allclose(
        basis_expected_lag_ratio(basis, [rho]), expected, rtol=1e-8, atol=1e-9
    )


@PROPERTIES
@given(st.integers(4, 40), st.floats(-0.9, 0.9))
def test_unprojected_ar1_curve_recovers_known_rho(n, rho):
    """Verify unprojected AR(1) curve recovers known rho."""
    grid = np.linspace(-0.95, 0.95, 301)
    expected = basis_expected_lag_ratio(np.empty((n, 0)), grid)
    assert_allclose(invert_lag_ratio([rho], expected, grid), rho, atol=1e-12)


@PROPERTIES
@given(SEEDS, st.integers(1, 14), st.floats(-0.7, 0.7))
def test_reml_is_invariant_to_ar1_time_reversal_at_high_rank(seed, rank, rho):
    """Verify REML is invariant to AR(1) time reversal at high rank."""
    rng = np.random.default_rng(seed)
    fixed = rng.normal(size=(16, 2))
    random = rng.normal(size=(16, rank))
    signal = random @ rng.normal(size=(rank, 2)) + rng.normal(size=(16, 2))
    original = fit_reml([RemlGroup(signal, fixed)], [random], rho=rho)
    reversed_fit = fit_reml(
        [RemlGroup(signal[::-1], fixed[::-1])], [random[::-1]], rho=rho
    )
    assert original.status == reversed_fit.status
    assert_allclose(original.criterion, reversed_fit.criterion, atol=1e-8, rtol=0)
    # At a boundary plateau, the likelihood is identified more accurately
    # than an arbitrary enormous ratio; compare the statistical objective.
    interior = ~(original.unrefined | reversed_fit.unrefined)
    assert_allclose(
        original.noise_variance[interior],
        reversed_fit.noise_variance[interior],
        atol=1e-8,
        rtol=1e-5,
    )
