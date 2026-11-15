import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

from boldtailor._reml_criterion import criterion, criterion_slope
from boldtailor._reml_search import _log_grid, minimize_ratio
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
    derandomize=True,
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
    assert np.all(np.abs(slope) <= 1e-10 * sum(t.dof for t in terms))


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


def _best_search_grid_value(terms, n_features):
    """Lowest criterion over zero and the search's own starting grid."""
    ratios = np.concatenate([[0.0], np.exp(_log_grid(terms, 121))])
    stacked = np.broadcast_to(ratios[:, None], (ratios.size, n_features))
    return criterion(stacked, terms).min(axis=0)


@pytest.mark.parametrize("true_ratio", [0.0, 0.1, 1.0, 10.0])
@pytest.mark.parametrize("seed", range(5))
def test_polish_never_worsens_criterion_beyond_rounding(seed, true_ratio):
    _, terms = _simulated_terms(seed, true_ratio)
    grid_value = _best_search_grid_value(terms, N_FEATURES)
    n = sum(t.dof for t in terms)
    rounding = 64 * np.finfo(float).eps * (np.abs(grid_value) + n)
    assert np.all(minimize_ratio(terms).criterion <= grid_value + rounding)


def _property_style_cases(count=300, seed=2026):
    """Seeded draws over the property tests' simulation settings, some AR(1)."""
    rng = np.random.default_rng(seed)
    for _ in range(count):
        n_groups = int(rng.integers(1, 5))
        rho = float(rng.choice([0.0, 0.0, 0.5]))
        data = simulate(
            rng,
            scans=tuple(int(n) for n in rng.integers(12, 41, size=n_groups)),
            n_fixed=int(rng.integers(1, 5)),
            n_random=int(rng.integers(1, 5)),
            ratio=float(rng.choice([0.0, 0.1, 1.0, 10.0])),
            rho=rho,
            n_features=int(rng.integers(1, 4)),
        )
        yield _terms(*data, rho)


def _oracle_cases():
    from tests.test_reml_oracle import CASES, _case_data

    for case in CASES:
        rho = case.get("rho", 0.0)
        yield _terms(*_case_data(case), rho)


def _bracket(log_grid, best_index):
    """The best grid point's neighbours, clipped to the grid."""
    last = log_grid.size - 1
    return (
        log_grid[np.maximum(best_index - 1, 0)],
        log_grid[np.minimum(best_index + 1, last)],
    )


def _unbracketed_interior(terms):
    """Count features whose interior grid minimum lacks a slope sign change."""
    log_grid = _log_grid(terms, 121)
    n_features = terms[0].residual_ss.shape[0]
    ratios = np.broadcast_to(np.exp(log_grid)[:, None], (log_grid.size, n_features))
    index = np.argmin(criterion(ratios, terms), axis=0)
    low, high = _bracket(log_grid, index)
    interior = (index > 0) & (index < log_grid.size - 1)
    falling = criterion_slope(np.exp(low), terms) < 0
    rising = criterion_slope(np.exp(high), terms) > 0
    return int(np.sum(interior & ~(falling & rising))), int(np.sum(interior))


@pytest.mark.parametrize("cases", [_oracle_cases, _property_style_cases])
def test_neighbour_bracket_shows_sign_change_for_interior_optima(cases):
    counts = np.array([_unbracketed_interior(terms) for terms in cases()])
    assert counts[:, 1].sum() > 0
    assert counts[:, 0].sum() == 0


def test_grid_chunks_bound_temporaries_without_changing_the_optimum(monkeypatch):
    import boldtailor._reml_search as search
    from boldtailor._reml_criterion import pool_terms

    _, terms = _simulated_terms(5, 1.0, n_features=1750)
    pooled = pool_terms(terms)
    row_bytes = 8 * pooled.singular2.size * pooled.residual_ss.size
    assert search._chunk_rows(pooled) * row_bytes <= 32 * 2**20
    whole = minimize_ratio(terms)
    monkeypatch.setattr(search, "_GRID_CHUNK_BYTES", 3 * row_bytes)
    assert search._chunk_rows(pooled) == 3
    chunked = minimize_ratio(terms)
    for name in ("ratio", "criterion", "at_zero", "at_upper"):
        np.testing.assert_array_equal(getattr(chunked, name), getattr(whole, name))


def _two_basin_terms():
    """The review's narrow minimum: bracketed by a slope crossing, yet not the
    lowest grid point, so refining only around the grid argmin loses it."""
    from boldtailor._reml_terms import RemlTerms

    y = [5.659277805656175, 1.3575913902516368, 12.166250878076637]
    y += [440.03410697431354, 0.024484911495792103, 0.2989731950657336]
    y += [754.0944608984961, 0.023686778553470386]
    y2 = np.array(y) ** 2
    rest = 23.645757013142806**2 + 0.005169198169685577**2
    return [
        RemlTerms(
            singular2=np.logspace(-10, 10, 8),
            coordinates2=y2[:, None],
            residual_ss=np.array([rest]),
            dof=10,
            energy=np.array([y2.sum() + rest]),
        )
    ]


def test_every_slope_crossing_competes():
    terms = _two_basin_terms()
    fine = np.exp(np.linspace(-45.0, 45.0, 90001))
    best = criterion(fine[:, None], terms).min()
    assert minimize_ratio(terms).criterion[0] <= best + 1e-9


def test_grid_spacing_does_not_grow_with_spectral_spread():
    from boldtailor._reml_search import _search_grid

    base_step = 2 * np.log(1e8) / 120
    for singular2 in (np.array([3.0]), np.array([1.0, 2.0]), np.logspace(-10, 10, 8)):
        grid = _search_grid(singular2, 121)
        assert np.diff(grid).max() <= base_step * (1 + 1e-12)
        assert grid[0] == pytest.approx(np.log(1e-8 / singular2.max()))
        assert grid[-1] == pytest.approx(np.log(1e8 / singular2.min()))
    assert _search_grid(np.array([3.0]), 121).size == 121
