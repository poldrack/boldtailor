import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

from boldtailor.reml import (
    _log_grid,
    criterion,
    criterion_slope,
    minimize_ratio,
    project_fixed,
    reml_terms,
)
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
    """Verify optimum not worse than any grid point or zero."""
    _, terms = _simulated_terms(seed, ratio)
    optimum = minimize_ratio(terms)
    assert np.all(optimum.criterion <= _best_of_grid_and_zero(terms) + 1e-9)


@pytest.mark.parametrize("rho", [0.0, 0.5])
@pytest.mark.parametrize("true_ratio", [0.05, 1.0, 20.0])
def test_matches_dense_fit(true_ratio, rho):
    """Compare REML variance estimates and likelihood with the dense covariance oracle."""
    (signals, fixed, random), terms = _simulated_terms(3, true_ratio, rho)
    expected, _ = dense_fit(signals, fixed, random, rho)
    assert_allclose(minimize_ratio(terms).ratio, expected, rtol=1e-5)


def _orthogonal_signal():
    signals, fixed, random = simulate(np.random.default_rng(8), scans=(60,))
    both = np.column_stack([fixed[0], random[0]])
    residual = signals[0] - both @ np.linalg.lstsq(both, signals[0], rcond=None)[0]
    return [residual + fixed[0] @ np.ones((4, 1))], fixed, random


def test_signal_orthogonal_to_random_design_is_at_zero():
    """Verify signal orthogonal to random design is at zero."""
    optimum = minimize_ratio(_terms(*_orthogonal_signal()))
    assert optimum.at_zero.all()
    assert_allclose(optimum.ratio, 0.0, atol=0.0)
    assert not optimum.at_upper.any()


def test_noise_free_signal_hits_upper_edge():
    """Verify noise free signal hits upper edge."""
    rng = np.random.default_rng(4)
    _, fixed, random = simulate(rng, scans=(60,), n_features=2)
    signals = [fixed[0] @ rng.normal(size=(4, 2)) + random[0] @ rng.normal(size=(3, 2))]
    optimum = minimize_ratio(_terms(signals, fixed, random))
    assert optimum.at_upper.all()
    assert not optimum.at_zero.any()


def test_no_directions_returns_zero():
    """Verify no directions returns zero."""
    signals, fixed, _ = simulate(np.random.default_rng(2), scans=(50,), n_features=3)
    empty = [np.empty((50, 0))]
    optimum = minimize_ratio(_terms(signals, fixed, empty))
    assert optimum.at_zero.all()
    assert not optimum.at_upper.any()
    assert_allclose(optimum.ratio, 0.0, atol=0.0)


@pytest.mark.parametrize("true_ratio", [0.05, 1.0, 20.0])
def test_optimum_is_a_score_root(true_ratio):
    """Verify optimum is a score root."""
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
    """Verify optimum is stable under fixed span shift."""
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
    """Verify polish never worsens criterion beyond rounding."""
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
    """Verify neighbour bracket shows sign change for interior optima."""
    counts = np.array([_unbracketed_interior(terms) for terms in cases()])
    assert counts[:, 1].sum() > 0
    assert counts[:, 0].sum() == 0


def _two_basin_terms():
    """The review's narrow minimum: bracketed by a slope crossing, yet not the
    lowest grid point, so refining only around the grid argmin loses it."""
    from boldtailor.reml import RemlTerms

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


def _recorded_bytes(monkeypatch):
    """Record the bytes of the (k, ...) temporary behind every evaluation."""
    import boldtailor.reml as search

    sizes = []

    def recording(function):
        def wrapped(ratios, pooled):
            sizes.append(8 * pooled.singular2.size * np.size(ratios))
            return function(ratios, pooled)

        return wrapped

    for name in ("pooled_criterion", "pooled_slope", "pooled_value_and_slope"):
        monkeypatch.setattr(search, name, recording(getattr(search, name)))
    return sizes


OPTIMUM_FIELDS = ("ratio", "criterion", "at_zero", "at_upper", "unidentifiable")


def _assert_same_optimum(result, expected):
    for name in OPTIMUM_FIELDS:
        np.testing.assert_array_equal(getattr(result, name), getattr(expected, name))


def test_search_temporaries_stay_within_the_budget(monkeypatch):
    """One grid row over all features exceeds the budget; batches must not."""
    import boldtailor.reml as search

    _, terms = _simulated_terms(5, 1.0, n_features=1750)
    directions = sum(t.singular2.size for t in terms)
    whole = minimize_ratio(terms)
    budget = 8 * directions * 100
    monkeypatch.setattr(search, "_WORKING_BYTES", budget)
    sizes = _recorded_bytes(monkeypatch)
    _assert_same_optimum(minimize_ratio(terms), whole)
    assert sizes and max(sizes) <= budget


def test_default_budget_bounds_a_wide_problem(monkeypatch):
    """The review's shape: 100 directions by 30000 features is 22.9 MiB a row."""
    import boldtailor.reml as search
    from boldtailor.reml import RemlTerms

    rng = np.random.default_rng(0)
    terms = [
        RemlTerms(
            singular2=rng.uniform(0.5, 2.0, size=100),
            coordinates2=rng.normal(size=(100, 30000)) ** 2,
            residual_ss=rng.chisquare(50, size=30000),
            dof=150,
            energy=np.zeros(30000),
        )
    ]
    sizes = _recorded_bytes(monkeypatch)
    minimize_ratio(terms, grid_size=9, iterations=2)
    assert max(sizes) <= search._WORKING_BYTES == 8 * 2**20


def test_budget_below_one_feature_falls_back_to_single_features(monkeypatch):
    """Verify budget below one feature falls back to single features."""
    import boldtailor.reml as search

    _, terms = _simulated_terms(5, 1.0, n_features=7)
    directions = sum(t.singular2.size for t in terms)
    whole = minimize_ratio(terms)
    monkeypatch.setattr(search, "_WORKING_BYTES", 1)
    sizes = _recorded_bytes(monkeypatch)
    _assert_same_optimum(minimize_ratio(terms), whole)
    assert max(sizes) == 8 * directions


def test_batches_do_not_change_a_multi_basin_optimum(monkeypatch):
    """Verify batches do not change a multi basin optimum."""
    import boldtailor.reml as search

    terms = _two_basin_terms()
    tiled = [
        type(terms[0])(
            singular2=terms[0].singular2,
            coordinates2=np.tile(terms[0].coordinates2, (1, 5)),
            residual_ss=np.tile(terms[0].residual_ss, 5),
            dof=10,
            energy=np.tile(terms[0].energy, 5),
        )
    ]
    whole = minimize_ratio(tiled)
    monkeypatch.setattr(search, "_WORKING_BYTES", 8 * 8 * 2)
    _assert_same_optimum(minimize_ratio(tiled), whole)
    assert_allclose(whole.ratio, minimize_ratio(terms).ratio[0], rtol=1e-12)


def test_every_slope_crossing_competes():
    """Verify every slope crossing competes."""
    terms = _two_basin_terms()
    fine = np.exp(np.linspace(-45.0, 45.0, 90001))
    best = criterion(fine[:, None], terms).min()
    assert minimize_ratio(terms).criterion[0] <= best + 1e-9


def test_grid_spacing_does_not_grow_with_spectral_spread():
    """Verify grid spacing does not grow with spectral spread."""
    from boldtailor.reml import _search_grid

    base_step = 2 * np.log(1e8) / 120
    for singular2 in (np.array([3.0]), np.array([1.0, 2.0]), np.logspace(-10, 10, 8)):
        grid = _search_grid(singular2, 121)
        assert np.diff(grid).max() <= base_step * (1 + 1e-12)
        assert grid[0] == pytest.approx(np.log(1e-8 / singular2.max()))
        assert grid[-1] == pytest.approx(np.log(1e8 / singular2.min()))
    assert _search_grid(np.array([3.0]), 121).size == 121


def _nearly_flat_terms(seed):
    """Saturated design whose spectrum differs only at rounding level: the
    profile varies by less than the criterion's own rounding noise."""
    from boldtailor.reml import RemlTerms

    y2 = np.random.default_rng(seed).normal(size=(6, 4)) ** 2
    return [
        RemlTerms(
            singular2=1.0 + 4e-15 * np.arange(6),
            coordinates2=y2,
            residual_ss=np.zeros(4),
            dof=6,
            energy=y2.sum(axis=0),
        )
    ]


@pytest.mark.parametrize("seed", range(10))
def test_improvement_within_rounding_noise_does_not_beat_zero(seed):
    """Verify improvement within rounding noise does not beat zero."""
    optimum = minimize_ratio(_nearly_flat_terms(seed))
    assert not optimum.unidentifiable.any()
    assert optimum.at_zero.all()
    assert_allclose(optimum.ratio, 0.0, atol=0.0)


def test_saturated_equal_spectrum_is_unidentifiable():
    """Verify saturated equal spectrum is unidentifiable."""
    terms = _nearly_flat_terms(0)
    flat = [type(terms[0])(**{**vars(terms[0]), "singular2": np.full(6, 2.5)})]
    optimum = minimize_ratio(flat)
    assert optimum.unidentifiable.all()
    assert np.isnan(optimum.ratio).all()
    assert_allclose(optimum.criterion, criterion(np.zeros(4), flat), rtol=1e-13)
    assert not optimum.at_zero.any() and not optimum.at_upper.any()


def test_identifiable_optimum_is_not_flagged():
    """Verify identifiable optimum is not flagged."""
    _, terms = _simulated_terms(3, 1.0)
    assert not minimize_ratio(terms).unidentifiable.any()


# ------------------------------------- follow-up F2: unrefined grid fallbacks


def _fallback_terms():
    """A spread spectrum whose optimum a very coarse grid cannot bracket."""
    from tests.test_reml_adversarial import _diagonal_problem

    y, z, _ = _diagonal_problem(54, "wide")
    return [reml_terms(project_fixed(y, np.empty((len(y), 0))), z)]


def test_grid_fallback_is_flagged_unrefined():
    """With no accepted slope root the reported ratio is a grid point, which
    need not be stationary; the optimum says so."""
    terms = _fallback_terms()
    optimum = minimize_ratio(terms, grid_size=3)
    assert optimum.unrefined.all()
    assert not optimum.at_zero.any() and not optimum.at_upper.any()
    assert np.all(np.abs(criterion_slope(optimum.ratio, terms)) > 1.0)


def test_default_grid_refines_the_same_problem():
    """Verify default grid refines the same problem."""
    terms = _fallback_terms()
    optimum = minimize_ratio(terms)
    assert not optimum.unrefined.any() and not optimum.at_zero.any()
    assert np.all(np.abs(criterion_slope(optimum.ratio, terms)) <= 1e-9)


def test_upper_edge_is_a_case_of_unrefined():
    """Verify upper edge is a case of unrefined."""
    rng = np.random.default_rng(4)
    _, fixed, random = simulate(rng, scans=(60,), n_features=2)
    signals = [fixed[0] @ rng.normal(size=(4, 2)) + random[0] @ rng.normal(size=(3, 2))]
    optimum = minimize_ratio(_terms(signals, fixed, random))
    assert optimum.at_upper.all() and optimum.unrefined.all()


def test_boundary_optima_are_not_unrefined():
    """Verify boundary optima are not unrefined."""
    orthogonal = minimize_ratio(_terms(*_orthogonal_signal()))
    assert orthogonal.at_zero.all() and not orthogonal.unrefined.any()
    flat = _nearly_flat_terms(0)
    confounded = [type(flat[0])(**{**vars(flat[0]), "singular2": np.full(6, 2.5)})]
    assert not minimize_ratio(confounded).unrefined.any()


@pytest.mark.parametrize("cases", [_oracle_cases, _property_style_cases])
def test_every_refined_positive_optimum_is_stationary(cases):
    """The stationarity guarantee, stated on exactly the optima it covers."""
    checked = 0
    for terms in cases():
        optimum = minimize_ratio(terms)
        refined = ~optimum.at_zero & ~optimum.unrefined & ~optimum.unidentifiable
        slope = criterion_slope(np.where(refined, optimum.ratio, 1.0), terms)
        dof = sum(t.dof for t in terms)
        assert np.all(np.abs(slope[refined]) <= 1e-9 * dof)
        checked += int(refined.sum())
    assert checked > 0


def test_exact_zero_slope_grid_point_is_not_unrefined():
    """A grid point whose analytic slope is 0.0 is an exact stationary point."""
    from boldtailor.reml import RemlTerms

    terms = [
        RemlTerms(
            singular2=np.array([1.0]),
            coordinates2=np.array([[2.0]]),
            residual_ss=np.array([1.0]),
            dof=2,
            energy=np.array([3.0]),
        )
    ]
    optimum = minimize_ratio(terms)
    assert_allclose(optimum.ratio, [1.0], rtol=1e-12)
    assert not optimum.unrefined.any()
    assert not optimum.at_zero.any()
    assert not optimum.at_upper.any()
