"""Property-based tests for single-trial normalized ridge and fractional ridge."""

from hypothesis import given, settings, strategies as st
import numpy as np

from boldtailor._fractional_ridge import prepare_fraction_betas
from boldtailor._single_trial_fit import prepare_trial_betas


def _well_posed_trial_design(
    rng, n_scans=40, n_trials=8, n_nuisance=2, ill_conditioned=False
):
    x = rng.normal(size=(n_scans, n_trials))
    if ill_conditioned:
        # Introduce near-collinearity without collapsing rank below numerical tolerance.
        x[:, 1] = x[:, 0] + 1e-3 * rng.normal(size=n_scans)
    nuisance = np.column_stack(
        [np.ones(n_scans), rng.normal(size=(n_scans, n_nuisance))]
    )
    return x, nuisance


@settings(max_examples=25, deadline=None)
@given(
    seed=st.integers(0, 10_000),
    alpha_low=st.floats(
        min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False
    ),
    alpha_delta=st.floats(
        min_value=0.01, max_value=20.0, allow_nan=False, allow_infinity=False
    ),
    y_scale=st.floats(
        min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False
    ),
)
def test_normalized_ridge_monotonicity_scale_equivariance_and_nuisance_invariance(
    seed, alpha_low, alpha_delta, y_scale
):
    """Verify normalized ridge norm shrinks monotonically with alpha, scales with signal, and ignores nuisance transforms."""
    rng = np.random.default_rng(seed)
    x, nuisance = _well_posed_trial_design(rng)
    y = rng.normal(size=(len(x), 3))

    prep = prepare_trial_betas(x, nuisance, y)
    b_low = prep.betas_at(alpha_low)
    b_high = prep.betas_at(alpha_low + alpha_delta)
    scale = prep.design.column_scale[:, None]
    assert np.all(
        np.linalg.norm(scale * b_high, axis=0)
        <= np.linalg.norm(scale * b_low, axis=0) + 1e-12
    )

    prep_scaled = prepare_trial_betas(x, nuisance, y * y_scale)
    np.testing.assert_allclose(
        prep_scaled.betas_at(alpha_low), b_low * y_scale, rtol=1e-10, atol=1e-12
    )

    mix = rng.normal(size=(nuisance.shape[1], nuisance.shape[1]))
    while np.linalg.cond(mix) > 50:
        mix = rng.normal(size=(nuisance.shape[1], nuisance.shape[1]))
    prep_nuis = prepare_trial_betas(x, nuisance @ mix, y)
    np.testing.assert_allclose(
        prep_nuis.betas_at(alpha_low), b_low, rtol=1e-10, atol=1e-12
    )


@settings(max_examples=25, deadline=None)
@given(
    seed=st.integers(0, 10_000),
    f_low=st.floats(
        min_value=0.05, max_value=0.5, allow_nan=False, allow_infinity=False
    ),
    f_high=st.floats(
        min_value=0.55, max_value=0.99, allow_nan=False, allow_infinity=False
    ),
    y_scale=st.floats(
        min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False
    ),
    ill_conditioned=st.booleans(),
)
def test_fractional_ridge_exact_norm_ratio_and_equivariances(
    seed, f_low, f_high, y_scale, ill_conditioned
):
    """Verify fractional ridge hits the exact OLS norm ratio, increases with fraction, and is scale-equivariant."""
    rng = np.random.default_rng(seed)
    x, nuisance = _well_posed_trial_design(rng, ill_conditioned=ill_conditioned)
    y = rng.normal(size=(len(x), 3))

    prep = prepare_fraction_betas(x, nuisance, y)
    b_ols = prep.betas_at(1.0)
    b_low = prep.betas_at(f_low)
    b_high = prep.betas_at(f_high)

    ols_norm = np.linalg.norm(b_ols, axis=0)
    np.testing.assert_allclose(
        np.linalg.norm(b_low, axis=0) / ols_norm, f_low, rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        np.linalg.norm(b_high, axis=0) / ols_norm, f_high, rtol=1e-10, atol=1e-12
    )
    assert np.all(
        np.linalg.norm(b_low, axis=0) <= np.linalg.norm(b_high, axis=0) + 1e-12
    )

    prep_scaled = prepare_fraction_betas(x, nuisance, y * y_scale)
    np.testing.assert_allclose(
        prep_scaled.betas_at(f_low), b_low * y_scale, rtol=1e-10, atol=1e-12
    )
