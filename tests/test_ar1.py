import dataclasses

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

from boldtailor._ar1_moments import (
    correct_lag_ratio,
    expected_lag_ratio,
    invert_lag_ratio,
    lag_moments,
    moment_polynomials,
    orthonormal_basis,
    pool_rho,
)
from boldtailor.ar1 import RhoReference, estimate_rho, prepare_reference, whiten
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
    assert pool_rho(np.array([-0.99, -0.95])) == pytest.approx(-0.2)
    assert pool_rho(np.array([-0.99, -0.95]), (-0.8, 0.8)) == pytest.approx(-0.8)


def test_no_valid_values_pools_to_zero():
    assert pool_rho(np.array([np.nan, np.nan])) == 0.0


def _ar1_problem(seed=7, scans=(40, 52), n_features=5):
    rng = np.random.default_rng(seed)
    designs = [rng.normal(size=(n, 6)) for n in scans]
    signals = [
        d @ rng.normal(size=(6, n_features))
        + np.linalg.cholesky(ar1_correlation(len(d), 0.4))
        @ rng.normal(size=(len(d), n_features))
        for d in designs
    ]
    return signals, designs


def _assert_same_estimate(actual, expected):
    np.testing.assert_array_equal(actual.values, expected.values)
    np.testing.assert_array_equal(actual.raw, expected.raw)
    assert actual.pooled == expected.pooled


def test_prepared_reference_matches_design():
    signals, designs = _ar1_problem()
    expected = estimate_rho(signals, designs)
    prepared = [prepare_reference(d) for d in designs]
    assert all(isinstance(r, RhoReference) for r in prepared)
    _assert_same_estimate(estimate_rho(signals, prepared), expected)
    mixed = [prepared[0], designs[1]]
    _assert_same_estimate(estimate_rho(signals, mixed), expected)


def test_prepared_reference_is_reusable_across_signals():
    signals, designs = _ar1_problem()
    other, _ = _ar1_problem(seed=8)
    prepared = [prepare_reference(d) for d in designs]
    estimate_rho(signals, prepared)
    _assert_same_estimate(estimate_rho(other, prepared), estimate_rho(other, designs))


def test_prepared_reference_is_frozen_and_read_only(design):
    reference = prepare_reference(design)
    with pytest.raises(dataclasses.FrozenInstanceError):
        reference.basis = np.zeros((40, 6))
    for name in ("basis", "expected"):
        with pytest.raises(ValueError):
            getattr(reference, name)[0] = 0.0
    assert reference.basis.shape == (40, 6)


def test_prepared_reference_owns_its_design(design):
    reference = prepare_reference(design)
    before = reference.basis.copy()
    design[:] = 0.0
    np.testing.assert_array_equal(reference.basis, before)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_prepare_reference_rejects_invalid_design(design, value):
    with pytest.raises(ValueError, match="design"):
        prepare_reference(design[:, 0])
    corrupt = design.copy()
    corrupt[3, 2] = value
    with pytest.raises(ValueError, match="design"):
        prepare_reference(corrupt)


def _with(parts, index, value):
    out = list(parts)
    out[index] = value
    return out


def test_estimate_rho_validates_groups():
    signals, designs = _ar1_problem()
    bad_design = designs[1].copy()
    bad_design[0, 0] = np.nan
    cases = [
        (_with(signals, 1, signals[1][:, 0]), designs, "signals.*group 1"),
        (_with(signals, 1, signals[1][:-1]), designs, "scan.*group 1"),
        (_with(signals, 1, signals[1][:, :3]), designs, "feature"),
        (signals, _with(designs, 1, designs[1][:, 0]), "design.*group 1"),
        (signals, _with(designs, 1, bad_design), "design.*group 1"),
        (
            signals,
            _with(designs, 1, prepare_reference(designs[0])),
            "scan.*group 1",
        ),
        (signals, designs[:1], "one .* per group"),
        ([], [], "at least one group"),
    ]
    for group_signals, references, message in cases:
        with pytest.raises(ValueError, match=message):
            estimate_rho(group_signals, references)


def test_inversion_interpolates_a_monotonic_expectation():
    grid = np.linspace(-0.9, 0.9, 7)
    expected = 0.5 * grid + 0.1
    values = np.array([[0.1, np.nan], [0.5 * 0.45 + 0.1, 0.5 * -0.9 + 0.1]])
    corrected = invert_lag_ratio(values, expected, grid)
    assert_allclose(corrected[0, 0], 0.0, atol=1e-15)
    assert np.isnan(corrected[0, 1])
    assert_allclose(corrected[1], [0.45, -0.9], atol=1e-15)


def test_inversion_is_undefined_when_the_curve_turns_inside_the_pooling_range():
    """Several inverse branches where estimates are pooled identify nothing."""
    grid = np.array([-0.6, -0.3, 0.0, 0.3, 0.6])
    expected = np.array([0.0, 0.5, 0.4, 0.8, 1.0])
    values = np.array([0.41, 0.52, 0.95, np.nan, -0.2])
    assert np.isnan(invert_lag_ratio(values, expected, grid)).all()


def _bent_curve(turn=0.85):
    """Rising with slope 0.6 up to ``turn``, falling beyond it."""
    grid = np.linspace(-0.95, 0.95, 39)
    peak = 0.6 * turn
    expected = np.where(grid <= turn, 0.6 * grid, peak - 0.4 * (grid - turn))
    return grid, expected


def test_a_bend_beyond_the_pooling_range_leaves_the_curve_usable():
    """The branch spanning the pooling range is inverted for every value;
    levels the bend repeats are read on that branch, not dropped."""
    grid, expected = _bent_curve()
    values = np.array([-0.5, -0.1, 0.0, 0.2, 0.41, 0.5, 0.505])
    assert_allclose(invert_lag_ratio(values, expected, grid), values / 0.6, atol=1e-12)


def test_values_beyond_the_usable_branch_are_clipped_to_its_ends():
    grid, expected = _bent_curve()
    corrected = invert_lag_ratio(np.array([0.7, 5.0, -0.9]), expected, grid)
    assert_allclose(corrected, [0.85, 0.85, -0.95], atol=1e-12)


@pytest.mark.parametrize("turn", [0.0, 0.5, 0.75])
def test_a_bend_inside_the_pooling_range_identifies_nothing(turn):
    grid, expected = _bent_curve(turn)
    values = np.array([-0.5, 0.0, 0.2])
    assert np.isnan(invert_lag_ratio(values, expected, grid)).all()


def test_a_flat_stretch_inside_the_pooling_range_identifies_nothing():
    grid = np.linspace(-0.9, 0.9, 7)
    expected = np.array([-0.5, -0.3, 0.0, 0.0, 0.0, 0.3, 0.5])
    corrected = invert_lag_ratio(np.array([-0.4, 0.0, 0.4]), expected, grid)
    assert np.isnan(corrected).all()


def test_default_rho_range_excludes_strong_negative_autocorrelation():
    from boldtailor._ar1_moments import RHO_RANGE

    assert RHO_RANGE == (-0.2, 0.8)
    assert pool_rho(np.array([0.9, 0.95, 0.99])) == 0.8
    assert pool_rho(np.array([-0.9, -0.95])) == -0.2


def _curve_turning_at(turn):
    """Rising with slope 0.6 above ``turn``, falling toward it from below."""
    grid = np.linspace(-0.95, 0.95, 39)
    return grid, 0.6 * np.abs(grid - turn)


def test_a_bend_below_the_rho_range_does_not_block_inversion():
    """A curve that turns at rho = -0.5 is unusable when strong negative
    autocorrelation is admitted and usable when it is not."""
    grid, expected = _curve_turning_at(-0.5)
    values = np.array([0.36, 0.6, 0.01])
    assert np.isnan(invert_lag_ratio(values, expected, grid, (-0.8, 0.8))).all()
    corrected = invert_lag_ratio(values, expected, grid)
    assert_allclose(corrected, values / 0.6 - 0.5, atol=1e-12)


def test_a_bend_inside_the_default_rho_range_still_identifies_nothing():
    grid, expected = _curve_turning_at(-0.1)
    assert np.isnan(invert_lag_ratio(np.array([0.3]), expected, grid)).all()


def test_values_below_the_range_are_clipped_to_the_branch_not_to_the_range():
    """Per-feature estimates keep what the usable branch supports; only the
    pooled value is confined to the range."""
    grid = np.linspace(-0.95, 0.95, 39)
    corrected = invert_lag_ratio(np.array([-0.3, -3.0]), 0.6 * grid, grid)
    assert_allclose(corrected, [-0.5, -0.95], atol=1e-12)


@pytest.mark.parametrize("true_rho", [-0.5, 0.4])
def test_estimate_reports_whether_the_pooled_value_was_clipped(true_rho):
    signals = [_ar1_noise(3, 150, 600, true_rho)]
    reference = [np.ones((150, 1))]
    default = estimate_rho(signals, reference)
    wide = estimate_rho(signals, reference, rho_range=(-0.8, 0.8))
    assert default.rho_range == (-0.2, 0.8) and wide.rho_range == (-0.8, 0.8)
    assert wide.pooled == pytest.approx(true_rho, abs=0.03)
    assert not wide.pooled_clipped
    assert default.pooled_clipped == (true_rho < -0.2)
    assert default.pooled == (wide.pooled if true_rho > 0 else -0.2)
    assert_allclose(default.values, wide.values, atol=0.0)


@pytest.mark.parametrize(
    "rho_range",
    [(0.8, -0.2), (0.3, 0.3), (-0.99, 0.8), (-0.2, 0.96), (np.nan, 0.8), (-0.2,)],
)
def test_invalid_rho_range_raises(rho_range):
    signals, reference = [_ar1_noise(3, 60, 4, 0.2)], [np.ones((60, 1))]
    with pytest.raises(ValueError, match="rho_range"):
        estimate_rho(signals, reference, rho_range=rho_range)


def test_rho_range_reaches_both_the_inversion_and_the_pooling():
    """With a design whose curve turns below -0.83, the wide range rejects
    the curve while the default range inverts it."""
    design = _saturated_design(5)
    signals = [_ar1_noise(2, 100, 500, 0.4)]
    assert np.isfinite(estimate_rho(signals, [design]).values).all()
    wide = estimate_rho(signals, [design], rho_range=(-0.9, 0.8))
    assert np.isnan(wide.values).all() and wide.pooled == 0.0
    assert not wide.pooled_clipped


def _censored_design(seed, n=200, drift=10, random=29, censored=100):
    """Drifts, nuisance regressors and one regressor per censored volume."""
    rng = np.random.default_rng(seed)
    t = (np.arange(n) + 0.5) / n
    columns = [np.ones(n)] + [np.cos(np.pi * j * t) for j in range(1, drift + 1)]
    columns.append(rng.normal(size=(n, random)))
    columns.append(np.eye(n)[:, rng.choice(n, censored, replace=False)])
    return np.column_stack(columns)


def _ar1_noise(seed, n, n_features, rho, burn=50):
    noise = np.random.default_rng(seed).normal(size=(n + burn, n_features))
    for t in range(1, n + burn):
        noise[t] += rho * noise[t - 1]
    return noise[burn:]


def test_curve_bent_only_beyond_the_pooling_range_still_estimates_rho():
    """With half the volumes censored this design's correction curve turns
    for rho below -0.928 only. Estimates away from that end must survive: all
    NaN would make the pooled value 0.0, silently treating the noise as white."""
    design = _censored_design(18)
    curve = expected_lag_ratio(design, np.linspace(-0.95, 0.95, 2001))
    turning = np.flatnonzero(np.diff(curve) <= 0)
    assert turning.size > 0 and turning.max() < 30
    estimate = estimate_rho([_ar1_noise(1, 200, 2000, 0.4)], [design])
    assert np.isfinite(estimate.values).all()
    assert estimate.pooled == pytest.approx(0.4, abs=0.04)


def _saturated_design(seed, n=100, residual_dof=10):
    rng = np.random.default_rng(seed)
    t = (np.arange(n) + 0.5) / n
    k = n - residual_dof - 1
    drifts = [np.cos(np.pi * j * t) for j in range(1, k // 2 + 1)]
    return np.column_stack([np.ones(n), *drifts, rng.normal(size=(n, k - k // 2))])


def test_pooled_rho_is_not_pulled_to_the_far_end_by_a_bend_at_one_end():
    """Ten residual degrees of freedom and a curve that turns below rho =
    -0.83: raw ratios are noisy and many fall at the levels the bend repeats.
    Dropping those values alone leaves an upward-biased median near the 0.8
    limit; keeping every value keeps the pooled estimate near the truth."""
    design = _saturated_design(5)
    curve = expected_lag_ratio(design, np.linspace(-0.95, 0.95, 2001))
    turning = np.flatnonzero(np.diff(curve) <= 0)
    assert turning.size > 0 and turning.max() < 130
    estimate = estimate_rho([_ar1_noise(2, 100, 4000, 0.4)], [design])
    assert np.isfinite(estimate.values).all()
    assert abs(estimate.pooled - 0.4) < 0.25


TAILS = st.lists(st.integers(-6, 6), min_size=0, max_size=4)


def _tailed_curve(left, core, right):
    """Rising across [-0.8, 0.8], with arbitrary steps on either side of it."""
    core_grid = np.linspace(-0.8, 0.8, len(core) + 1)
    step = core_grid[1] - core_grid[0]
    below = core_grid[0] - step * np.arange(len(left), 0, -1)
    above = core_grid[-1] + step * np.arange(1, len(right) + 1)
    steps = np.array([*left, *core, *right], dtype=float)
    expected = np.concatenate([[0.0], np.cumsum(steps)]) / 8.0
    return np.concatenate([below, core_grid, above]), expected


@settings(derandomize=True, deadline=None, max_examples=200)
@given(
    TAILS,
    st.lists(st.integers(1, 6), min_size=2, max_size=8),
    TAILS,
    st.lists(st.floats(-20.0, 20.0), min_size=2, max_size=8),
    st.booleans(),
)
def test_no_value_is_dropped_and_order_is_kept_whatever_the_tails(
    left, core, right, values, falling
):
    """Whatever the curve does beyond the pooling range, every finite value
    gets an estimate and estimates keep the order of the values. Dropping or
    reordering by value would bias the pooled median."""
    grid, expected = _tailed_curve(left, core, right)
    sign = -1.0 if falling else 1.0
    corrected = invert_lag_ratio(np.sort(values), sign * expected, grid)
    assert np.all(np.isfinite(corrected))
    assert np.all(np.diff(corrected) * sign >= 0)
    assert corrected.min() >= grid[0] and corrected.max() <= grid[-1]


@settings(derandomize=True, deadline=None, max_examples=200)
@given(TAILS, st.lists(st.integers(1, 6), min_size=2, max_size=8), TAILS)
def test_values_on_the_pooling_branch_are_inverted_exactly(left, core, right):
    grid, expected = _tailed_curve(left, core, right)
    inside = np.abs(grid) <= 0.8 + 1e-12
    rhos = np.linspace(-0.8, 0.8, 23)
    levels = np.interp(rhos, grid[inside], expected[inside])
    assert_allclose(invert_lag_ratio(levels, expected, grid), rhos, atol=1e-12)


@settings(derandomize=True, deadline=None, max_examples=100)
@given(
    st.lists(st.integers(1, 6), min_size=2, max_size=12),
    st.lists(st.floats(-3.0, 12.0), min_size=1, max_size=6),
    st.booleans(),
)
def test_inversion_of_a_monotone_curve_is_plain_interpolation(steps, values, falling):
    expected = np.cumsum(np.array([0, *steps], dtype=float)) / 8.0
    grid = np.linspace(-0.9, 0.9, expected.size)
    curve = -expected if falling else expected
    order = slice(None, None, -1) if falling else slice(None)
    reference = np.interp(values, curve[order], grid[order])
    assert_allclose(invert_lag_ratio(values, curve, grid), reference, atol=1e-15)


@pytest.mark.parametrize(
    "curve",
    [np.zeros(7), np.linspace(0, 1e-15, 7), np.full(7, np.nan), np.full(7, np.inf)],
)
def test_uninformative_correction_curves_do_not_estimate_rho(curve):
    corrected = invert_lag_ratio([0.0, 0.5], curve, np.linspace(-0.9, 0.9, 7))
    assert np.isnan(corrected).all()


def test_decreasing_identifiable_curve_is_inverted():
    grid = np.linspace(-0.9, 0.9, 7)
    assert_allclose(invert_lag_ratio([-0.2, 0.4], -grid, grid), [0.2, -0.4])


def test_nonadjacent_residual_coordinates_do_not_identify_rho():
    eye = np.eye(12)
    estimate = estimate_rho([eye[:, [0]]], [eye[:, 1::2]])
    assert np.isnan(estimate.values).all()
    assert estimate.pooled == 0.0


@pytest.mark.parametrize("rho", [0.0, 0.4, -0.3])
def test_estimator_moments_match_moment_polynomials_exactly(design, rho):
    """With the columns of chol(R) as features, sum_f e'Le = tr(MLMR)."""
    basis = orthonormal_basis(design)
    signals = np.linalg.cholesky(ar1_correlation(len(design), rho))
    residuals = signals - basis @ (basis.T @ signals)
    numerator, denominator = lag_moments(residuals)
    numerator_poly, denominator_poly = moment_polynomials(basis)
    polyval = np.polynomial.polynomial.polyval
    assert_allclose(numerator.sum(), polyval(rho, numerator_poly), rtol=1e-10)
    assert_allclose(denominator.sum(), polyval(rho, denominator_poly), rtol=1e-10)


def test_non_finite_signal_feature_gives_nan_not_error():
    signals, designs = _ar1_problem()
    signals = [s.copy() for s in signals]
    signals[1][4, 2] = np.nan
    result = estimate_rho(signals, designs)
    assert np.isnan(result.values[1, 2]) and np.isnan(result.raw[1, 2])
    assert np.isfinite(np.delete(result.values[1], 2)).all()
    assert np.isfinite(result.values[0]).all()


def _lag_problem(n_fixed, n=60):
    rng = np.random.default_rng(11)
    design = np.column_stack([np.ones(n), rng.normal(size=(n, n_fixed - 1))])
    noise = rng.normal(size=(n, 5))
    noise[1:] += 0.5 * noise[:-1]
    return noise, design


@pytest.mark.parametrize("scale", [1e4, 1e6, 1e8, 1e10])
@pytest.mark.parametrize("n_fixed", [1, 3])
def test_lag_ratio_is_invariant_to_resolvable_design_span_shifts(n_fixed, scale):
    from boldtailor._ar1_moments import residual_lag_ratio

    signals, design = _lag_problem(n_fixed)
    shift = scale * (design @ np.random.default_rng(0).normal(size=(n_fixed, 1)))
    base = residual_lag_ratio(signals, design)
    shifted = residual_lag_ratio(signals + shift, design)
    tolerance = 100 * len(signals) * np.finfo(float).eps * scale
    assert np.isfinite(base).all()
    np.testing.assert_allclose(shifted, base, rtol=0.0, atol=tolerance)


@pytest.mark.parametrize("scale", [1.0, 1e6])
@pytest.mark.parametrize("n_fixed", [1, 3, 8])
def test_lag_ratio_of_signals_inside_the_design_span_is_nan(n_fixed, scale):
    from boldtailor._ar1_moments import residual_lag_ratio

    _, design = _lag_problem(n_fixed, n=200)
    signals = scale * (design @ np.random.default_rng(4).normal(size=(n_fixed, 6)))
    assert np.isnan(residual_lag_ratio(signals, design)).all()


def test_lag_ratio_of_zero_signal_is_nan():
    from boldtailor._ar1_moments import residual_lag_ratio

    _, design = _lag_problem(2)
    assert np.isnan(residual_lag_ratio(np.zeros((60, 2)), design)).all()


def test_reference_carries_the_conditioning_of_its_basis(design):
    reference = prepare_reference(design * np.logspace(0, 4, design.shape[1]))
    unit = design / np.linalg.norm(design, axis=0)
    singular = np.linalg.svd(unit, compute_uv=False)
    assert_allclose(reference.amplification, singular[0] / singular, rtol=1e-10)
    assert not reference.amplification.flags.writeable


def test_exactly_removable_shift_preserves_the_lag_ratio():
    """Follow-up review probe: orthogonal reference columns in unequal units."""
    eye = np.eye(30)
    reference = [prepare_reference(eye[:, :2] * np.array([1e-12, 1.0]))]
    signal = (2 * eye[:, 2] + eye[:, 3])[:, None]
    base = estimate_rho([signal], reference)
    moved = estimate_rho([signal + 100 * eye[:, [0]]], reference)
    assert_allclose(base.raw, [[0.4]])
    assert_allclose(moved.raw, base.raw, rtol=1e-12)


@pytest.mark.parametrize("scales", [(1e-20, 1.0, 1e15), (1e6, 1e-9, 1.0)])
def test_estimate_does_not_depend_on_reference_column_units(scales):
    signals, reference = _lag_problem(3)
    base = estimate_rho([signals], [reference])
    rescaled = estimate_rho([signals], [reference * np.array(scales)])
    assert_allclose(rescaled.raw, base.raw, rtol=1e-9)
    assert_allclose(rescaled.values, base.values, rtol=1e-9)


@pytest.mark.parametrize("scale", [1.0, 1e4, 1e8])
def test_lag_ratio_is_invariant_to_shifts_in_a_span_of_unequal_units(scale):
    n = 300
    t = np.arange(n, dtype=float)
    rng = np.random.default_rng(2)
    reference = np.column_stack(
        [np.ones(n), t, t**2, t**3, 0.05 * rng.normal(size=(n, 6))]
    )
    signals = rng.normal(size=(n, 4))
    unit = reference / np.linalg.norm(reference, axis=0)
    moved = signals + scale * (unit @ rng.normal(size=(10, 1)))
    base = estimate_rho([signals], [reference]).raw
    tolerance = max(100 * n * np.finfo(float).eps * scale, 1e-9)
    assert np.isfinite(base).all()
    assert_allclose(
        estimate_rho([moved], [reference]).raw, base, rtol=0.0, atol=tolerance
    )


def _weak_column_problem(seed=0):
    """Signals in the span of an ill-conditioned saturated design, dominated
    by its weakest column, where the computed span is least accurate."""
    rng = np.random.default_rng(seed)
    design = rng.normal(size=(10, 9)) * np.logspace(0, 9, 9)
    weights = rng.normal(size=(9, 40)) * np.logspace(0, -12, 9)[:, None]
    return design @ weights, design


def test_ill_conditioned_saturated_span_signal_is_degenerate():
    signals, reference = _weak_column_problem()
    assert np.isnan(estimate_rho([signals], [reference]).raw).all()
