import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

from boldtailor._reml_criterion import criterion, criterion_slope
from boldtailor._reml_search import minimize_ratio
from boldtailor._reml_terms import project_fixed, reml_terms
from tests.reml_oracles import dense_fit, simulate

N_FEATURES = 4


def _terms(signals, fixed, random, rho=0.0):
    return [
        reml_terms(project_fixed(y, c, rho), z)
        for y, c, z in zip(signals, fixed, random, strict=True)
    ]


def _simulated_terms(seed, ratio, rho=0.0, n_features=N_FEATURES):
    data = simulate(
        np.random.default_rng(seed),
        scans=(40, 55),
        ratio=ratio,
        rho=rho,
        n_features=n_features,
    )
    return data, _terms(*data, rho)


def _best_of_grid_and_zero(terms):
    grid = np.exp(np.linspace(np.log(1e-8), np.log(1e8), 400))
    values = [criterion(np.full(N_FEATURES, r), terms) for r in (0.0, *grid)]
    return np.min(values, axis=0)


@settings(
    max_examples=15,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(seed=st.integers(0, 10_000), ratio=st.sampled_from([0.0, 0.1, 10.0]))
def test_optimum_not_worse_than_any_grid_point_or_zero(seed, ratio):
    _, terms = _simulated_terms(seed, ratio)
    optimum = minimize_ratio(terms)
    assert np.all(optimum.criterion <= _best_of_grid_and_zero(terms) + 1e-9)


@pytest.mark.parametrize("rho", [0.0, 0.5])
@pytest.mark.parametrize("true_ratio", [0.05, 1.0, 20.0])
def test_matches_dense_fit(true_ratio, rho):
    (signals, fixed, random), terms = _simulated_terms(3, true_ratio, rho)
    expected, _ = dense_fit(signals, fixed, random, rho)
    assert_allclose(minimize_ratio(terms).ratio, expected, rtol=1e-5)


def _orthogonal_signal():
    signals, fixed, random = simulate(np.random.default_rng(8), scans=(60,))
    both = np.column_stack([fixed[0], random[0]])
    residual = signals[0] - both @ np.linalg.lstsq(both, signals[0], rcond=None)[0]
    return [residual + fixed[0] @ np.ones((4, 1))], fixed, random


def test_signal_orthogonal_to_random_design_is_at_zero():
    optimum = minimize_ratio(_terms(*_orthogonal_signal()))
    assert optimum.at_zero.all()
    assert_allclose(optimum.ratio, 0.0, atol=0.0)
    assert not optimum.at_upper.any()


def test_noise_free_signal_hits_upper_edge():
    rng = np.random.default_rng(4)
    _, fixed, random = simulate(rng, scans=(60,), n_features=2)
    signals = [fixed[0] @ rng.normal(size=(4, 2)) + random[0] @ rng.normal(size=(3, 2))]
    optimum = minimize_ratio(_terms(signals, fixed, random))
    assert optimum.at_upper.all()
    assert not optimum.at_zero.any()


def test_no_directions_returns_zero():
    signals, fixed, _ = simulate(np.random.default_rng(2), scans=(50,), n_features=3)
    empty = [np.empty((50, 0))]
    optimum = minimize_ratio(_terms(signals, fixed, empty))
    assert optimum.at_zero.all()
    assert not optimum.at_upper.any()
    assert_allclose(optimum.ratio, 0.0, atol=0.0)


@pytest.mark.parametrize("true_ratio", [0.05, 1.0, 20.0])
def test_optimum_is_a_score_root(true_ratio):
    _, terms = _simulated_terms(3, true_ratio)
    optimum = minimize_ratio(terms)
    interior = ~optimum.at_zero & ~optimum.at_upper
    assert interior.any()
    slope = criterion_slope(optimum.ratio, terms)[interior]
    assert np.all(np.abs(slope) <= 1e-8 * sum(t.dof for t in terms))


def _shift_example():
    """Flat-criterion case where golden section alone loses ratio precision."""
    rng = np.random.default_rng(12)
    signals, fixed, random = simulate(
        rng, scans=(34,), n_fixed=2, n_random=3, ratio=0.0, n_features=3
    )
    shift = fixed[0] @ rng.normal(scale=10.0, size=(2, 3))
    return _terms(signals, fixed, random), _terms([signals[0] + shift], fixed, random)


def test_optimum_is_stable_under_fixed_span_shift():
    base_terms, shifted_terms = _shift_example()
    base = minimize_ratio(base_terms)
    shifted = minimize_ratio(shifted_terms)
    np.testing.assert_array_equal(shifted.at_zero, base.at_zero)
    assert (~base.at_zero).any()
    assert_allclose(shifted.ratio, base.ratio, rtol=1e-9, atol=0.0)
