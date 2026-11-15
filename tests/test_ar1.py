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
    assert pool_rho(np.array([-0.99, -0.95])) == pytest.approx(-0.8)


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


def test_inversion_is_undefined_when_the_curve_is_not_identifying():
    grid = np.array([-0.6, -0.3, 0.0, 0.3, 0.6])
    expected = np.array([0.0, 0.5, 0.4, 0.8, 1.0])
    values = np.array([0.41, 0.52, 0.95, np.nan, -0.2])
    corrected = invert_lag_ratio(values, expected, grid)
    # Multiple inverse branches cannot identify a unique rho. This replaces
    # the old nearest-grid policy, which invented an estimate on flat curves.
    assert np.isnan(corrected).all()


@pytest.mark.parametrize("curve", [np.zeros(7), np.linspace(0, 1e-15, 7),
                                  np.full(7, np.nan), np.full(7, np.inf)])
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
