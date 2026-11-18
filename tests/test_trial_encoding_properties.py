"""Property-based tests for trial encoding models."""

from hypothesis import given, settings, strategies as st
import numpy as np
import pandas as pd

from boldtailor.trial_encoding import evaluate_trial_encoding


def _encoding_problem(seed, n_runs=4):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(6, 15, size=n_runs)
    predictors, betas = [], []
    for size in sizes:
        x = rng.normal(size=(size, 2))
        b = x @ np.array([[1.2, -0.5], [0.4, 0.8]]) + rng.normal(
            scale=0.3, size=(size, 2)
        )
        if size > 7:
            x[0, 0] = np.nan  # exercise missing-trial masking
        predictors.append(pd.DataFrame(x, columns=["p0", "p1"]))
        betas.append(b)
    return betas, predictors, sizes


@settings(max_examples=25, deadline=None)
@given(
    seed=st.integers(0, 10_000),
    p_scale=st.floats(
        min_value=0.1, max_value=50.0, allow_nan=False, allow_infinity=False
    ),
)
def test_within_run_encoding_is_invariant_to_run_shifts_and_predictor_scaling(
    seed, p_scale
):
    """Verify within-run encoding slopes and R2 are invariant to per-run shifts and predictor scaling."""
    betas, predictors, sizes = _encoding_problem(seed)
    rng = np.random.default_rng(seed + 1)
    train_runs, test_runs = (0, 1, 2), (3,)

    base = evaluate_trial_encoding(
        betas,
        predictors,
        train_runs=train_runs,
        test_runs=test_runs,
        encoding_mode="within_run",
    )
    shifted_betas = [b + rng.uniform(-20.0, 20.0, size=(1, 2)) for b in betas]
    shifted_preds = [p + rng.uniform(-10.0, 10.0, size=(1, 2)) for p in predictors]
    shifted = evaluate_trial_encoding(
        shifted_betas,
        shifted_preds,
        train_runs=train_runs,
        test_runs=test_runs,
        encoding_mode="within_run",
    )
    np.testing.assert_allclose(
        shifted.coefficients[1:], base.coefficients[1:], rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(shifted.r2, base.r2, rtol=1e-10, atol=1e-12)

    scaled_preds = [p * p_scale for p in predictors]
    scaled = evaluate_trial_encoding(
        betas,
        scaled_preds,
        train_runs=train_runs,
        test_runs=test_runs,
        encoding_mode="within_run",
    )
    np.testing.assert_allclose(scaled.r2, base.r2, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(scaled.run_sse, base.run_sse, rtol=1e-10, atol=1e-12)


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(0, 10_000))
def test_within_run_encoding_matches_dummy_variable_ols_with_unbalanced_missing_trials(
    seed,
):
    """Verify within-run encoding slopes match explicit run-dummy OLS across unbalanced runs and NaNs."""
    betas, predictors, _ = _encoding_problem(seed)
    train_runs, test_runs = (0, 1, 2), (3,)
    result = evaluate_trial_encoding(
        betas,
        predictors,
        train_runs=train_runs,
        test_runs=test_runs,
        encoding_mode="within_run",
    )

    rows_x, rows_y, run_ids = [], [], []
    for idx, r in enumerate(train_runs):
        x = predictors[r].to_numpy()
        mask = np.isfinite(x).all(axis=1)
        rows_x.append(x[mask])
        rows_y.append(betas[r][mask])
        run_ids.append(np.full(mask.sum(), idx))
    x_cat = np.vstack(rows_x)
    y_cat = np.vstack(rows_y)
    ids = np.concatenate(run_ids)
    dummies = np.column_stack(
        [(ids == idx).astype(float) for idx in range(len(train_runs))]
    )
    theta = np.linalg.lstsq(np.hstack([dummies, x_cat]), y_cat, rcond=None)[0]
    np.testing.assert_allclose(
        result.coefficients[1:], theta[len(train_runs) :], rtol=1e-10, atol=1e-12
    )
