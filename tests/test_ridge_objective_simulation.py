"""Independent arithmetic for experimental objectives, not a default change."""

import numpy as np
import pandas as pd
import pytest


def test_centering_removes_only_held_out_offset():
    from examples.validation.ridge_objective_simulation import evaluate_candidate

    x = np.array([-1.0, 0.0, 1.0, 2.0])
    predictors = [pd.DataFrame(dict(value=x))] * 3
    betas = [(2 + 3 * x)[:, None], (2 + 3 * x)[:, None], (12 + 3 * x)[:, None]]
    raw = evaluate_candidate(betas, predictors, [0, 1], [2], "current", betas)
    centered = evaluate_candidate(betas, predictors, [0, 1], [2], "centered", betas)
    assert raw["r2"][0] == pytest.approx(1 - 400 / 45)
    assert centered["r2"][0] == pytest.approx(1.0)
    np.testing.assert_allclose(centered["offsets"], [[10.0]])
    np.testing.assert_array_equal(raw["coefficients"], centered["coefficients"])
    changed = [*betas[:2], (12 + 4 * x)[:, None]]
    slope = evaluate_candidate(changed, predictors, [0, 1], [2], "centered", changed)
    assert slope["r2"][0] < 1


def test_pool_sums_before_division_and_exclude_missing_predictions():
    from examples.validation.ridge_objective_simulation import score_prediction_runs

    y = [np.array([0.0, 2.0, 100.0])[:, None], np.array([0.0, 10.0])[:, None]]
    pred = [np.array([1.0, 1.0, np.nan])[:, None], np.array([1.0, 9.0])[:, None]]
    result = score_prediction_runs(y, pred, center=False)
    assert result["r2"][0] == pytest.approx(1 - 4 / 52)
    np.testing.assert_allclose(result["sst"], [52.0])


def test_fixed_ols_targets_are_candidate_independent():
    from examples.validation.ridge_objective_simulation import evaluate_candidate

    x = np.arange(5.0)
    predictors = [pd.DataFrame(dict(value=x + r)) for r in range(3)]
    ols = [(3 + 2 * p.value.to_numpy())[:, None] for p in predictors]
    first = evaluate_candidate(ols, predictors, [0, 1], [2], "fixed_ols", ols)
    shrunk = [b * 0.1 for b in ols]
    second = evaluate_candidate(shrunk, predictors, [0, 1], [2], "fixed_ols", ols)
    np.testing.assert_array_equal(first["targets"][0], second["targets"][0])
    assert second["r2"][0] < first["r2"][0]


@pytest.mark.parametrize("encoding_mode", ["absolute", "within_run"])
def test_current_alpha_score_matches_matched_filter_limit(encoding_mode):
    from boldtailor._single_trial_fit import trial_beta_path
    from boldtailor.trial_encoding import evaluate_trial_encoding

    rng = np.random.default_rng(58)
    xs = [rng.normal(size=(50, 5)) for _ in range(3)]
    ys = [rng.normal(size=(50, 1)) for _ in xs]
    betas, matched = [], []
    for x, y in zip(xs, ys):
        betas.append(next(trial_beta_path(x, np.ones((50, 1)), y, alphas=[1e8]))[1])
        xr, yr = x - x.mean(0), y - y.mean(0)
        matched.append(xr.T @ yr / np.sum(xr * xr, axis=0)[:, None])
    predictors = [pd.DataFrame(dict(value=np.arange(5.0)))] * 3
    actual = evaluate_trial_encoding(
        betas, predictors, train_runs=[0, 1], test_runs=[2], encoding_mode=encoding_mode
    )
    expected = evaluate_trial_encoding(
        matched,
        predictors,
        train_runs=[0, 1],
        test_runs=[2],
        encoding_mode=encoding_mode,
    )
    np.testing.assert_allclose(actual.r2, expected.r2, atol=1e-7)


def test_historical_objectives_keep_shared_intercept_fit():
    from examples.validation.ridge_objective_simulation import evaluate_candidate

    x = [np.arange(5.0) + 10 * r for r in range(3)]
    predictors = [pd.DataFrame({"x": v}) for v in x]
    betas = [(3 * v + 20 * r)[:, None] for r, v in enumerate(x)]
    train_x = np.concatenate(x[:2])
    train_y = np.concatenate(betas[:2])
    oracle = np.linalg.lstsq(
        np.column_stack([np.ones(10), train_x]), train_y, rcond=None
    )[0]
    for objective in ["current", "centered", "fixed_ols"]:
        actual = evaluate_candidate(betas, predictors, [0, 1], [2], objective, betas)
        np.testing.assert_allclose(actual["coefficients"][1:], oracle[1:])
        np.testing.assert_allclose(
            actual["predictions"][0], np.column_stack([np.ones(5), x[2]]) @ oracle
        )
    within = evaluate_candidate(betas, predictors, [0, 1], [2], "within_run", betas)
    np.testing.assert_allclose(within["coefficients"][1:], [[3]])
    np.testing.assert_allclose(within["r2"], [1])
    assert abs(oracle[1, 0] - 3) > 1


def test_recovery_metrics_center_each_run_before_pooling():
    from examples.validation.ridge_objective_simulation import _recovery_metrics

    true = [np.array([r + 1.0, r + 3.0, r + 5.0])[:, None] for r in range(6)]
    estimated = [2 * y + 100 * r for r, y in enumerate(true)]
    metrics = _recovery_metrics(estimated, true)
    assert metrics["within_run_beta_rmse"] == pytest.approx(np.sqrt(8 / 3))
    assert metrics["within_run_amplitude_ratio"] == pytest.approx(2)
    flat = [np.ones((3, 1)) for _ in range(6)]
    undefined = _recovery_metrics(flat, flat)
    assert np.isnan(undefined["within_run_amplitude_ratio"])
    assert np.isnan(undefined["within_run_correlation"])


def test_confounding_stress_experiment_recovers_known_slopes():
    from examples.validation.ridge_objective_simulation import confounding_stress

    rows = confounding_stress()
    assert {r["objective"] for r in rows} == {"current", "centered", "within_run"}
    assert len(rows) == 6
    for row in rows:
        if row["objective"] == "within_run":
            assert row["slope"] == pytest.approx(3)
            assert row["outer_r2"] == pytest.approx(1)
        elif row["offset_increment"]:
            assert abs(row["slope"] - 3) > 1
