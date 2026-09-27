"""Held-out predictions use training coefficients and preserve event alignment."""

import importlib

import numpy as np
import pandas as pd
import pytest


def evaluate(*args, **kwargs):
    try:
        function = importlib.import_module(
            "boldtailor.trial_encoding"
        ).evaluate_trial_encoding
    except (ImportError, AttributeError) as error:
        pytest.fail(f"Missing trial encoding: {error}")
    return function(*args, **kwargs)


@pytest.fixture
def encoding_runs():
    rng = np.random.default_rng(32)
    predictors, betas = [], []
    for r, n in enumerate((9, 7, 6, 11)):
        x = pd.DataFrame(
            {"response_time": rng.uniform(0.3, 2, n), "trial_type": np.arange(n) % 2}
        )
        design = np.column_stack([np.ones(n), x])
        y = design @ np.array([[3.0, -1.0], [0.8, 2.0], [-0.5, 1.0]])
        y += rng.normal(0, 0.2, y.shape) + (2 if r == 3 else 0)
        predictors.append(x)
        betas.append(y)
    return betas, predictors


def test_no_validation_intercept_refit():
    x = np.array([-1.0, 0.0, 1.0, 2.0])
    predictors = [pd.DataFrame({"response_time": x}) for _ in range(3)]
    betas = [(2 + 3 * x)[:, None], (2 + 3 * x)[:, None], (12 + 3 * x)[:, None]]
    result = evaluate(betas, predictors, train_runs=[0, 1], test_runs=[2])
    np.testing.assert_allclose(result.predictions[0][:, 0], 2 + 3 * x)
    np.testing.assert_allclose(result.r2, 1 - 400 / 45)


def test_predictions_and_pooled_loss_match_independent_ols(encoding_runs):
    betas, predictors = encoding_runs
    predictors[0].loc[2, "response_time"] = np.nan
    predictors[2].loc[1, "response_time"] = np.inf
    original = [x.copy(deep=True) for x in predictors]
    result = evaluate(betas, predictors, train_runs=[0, 1], test_runs=[2, 3])
    masks = [np.isfinite(x).all(axis=1).to_numpy() for x in predictors]
    xtrain = np.concatenate([predictors[r].to_numpy()[masks[r]] for r in (0, 1)])
    ytrain = np.concatenate([betas[r][masks[r]] for r in (0, 1)])
    coefficients = np.linalg.lstsq(
        np.column_stack([np.ones(len(xtrain)), xtrain]), ytrain, rcond=None
    )[0]
    losses, totals = [], []
    for j, r in enumerate((2, 3)):
        x = predictors[r].to_numpy()[masks[r]]
        predicted = np.column_stack([np.ones(len(x)), x]) @ coefficients
        np.testing.assert_allclose(result.predictions[j][masks[r]], predicted)
        assert np.isnan(result.predictions[j][~masks[r]]).all()
        y = betas[r][masks[r]]
        losses.append(np.sum((y - predicted) ** 2, axis=0))
        totals.append(np.sum((y - y.mean(axis=0)) ** 2, axis=0))
    np.testing.assert_allclose(result.run_sse, losses)
    np.testing.assert_allclose(result.run_sst, totals)
    np.testing.assert_allclose(
        result.r2, 1 - np.sum(losses, axis=0) / np.sum(totals, axis=0)
    )
    assert not np.allclose(result.r2, np.mean(1 - np.array(losses) / totals, axis=0))
    np.testing.assert_allclose(result.predictor_means, xtrain.mean(axis=0))
    np.testing.assert_allclose(result.coefficients[1:], coefficients[1:])
    assert result.predictor_names == ("task", "response_time", "trial_type")
    for actual, expected in zip(predictors, original):
        pd.testing.assert_frame_equal(actual, expected)
    for actual, expected in zip(result.trial_masks, masks):
        np.testing.assert_array_equal(actual, expected)
    for array in [
        result.coefficients,
        result.predictor_means,
        result.r2,
        result.run_sse,
        result.run_sst,
        *result.predictions,
        *result.trial_masks,
    ]:
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_validation_covariates_do_not_change_training_transform(encoding_runs):
    betas, predictors = encoding_runs
    first = evaluate(betas, predictors, train_runs=[0, 1], test_runs=[2, 3])
    predictors[2] = predictors[2].assign(response_time=99.0, trial_type=0)
    second = evaluate(betas, predictors, train_runs=[0, 1], test_runs=[2, 3])
    np.testing.assert_array_equal(first.coefficients, second.coefficients)
    np.testing.assert_array_equal(first.predictor_means, second.predictor_means)
    assert not np.allclose(first.predictions[0], second.predictions[0])


def test_constant_and_incomplete_features_are_undefined(encoding_runs):
    betas, predictors = encoding_runs
    betas = [np.column_stack([y, np.ones(len(y))]) for y in betas]
    betas[0][1, 1] = np.nan
    result = evaluate(betas, predictors, train_runs=[0, 1], test_runs=[2, 3])
    assert np.isfinite(result.r2[0])
    assert np.isnan(result.r2[1:]).all()
    assert np.isnan(result.predictions[0][:, 1]).all()


@pytest.mark.parametrize(
    "train,test",
    [
        ([], [2]),
        ([0, 0], [2]),
        ([0, 1], [1]),
        ([0], [4]),
        ([True], [2]),
        ([0, 1], []),
        ([0, 1], [2, 2]),
        ([0.5], [2]),
    ],
)
def test_bad_splits_rejected(encoding_runs, train, test):
    with pytest.raises(ValueError, match="run|split"):
        evaluate(*encoding_runs, train_runs=train, test_runs=test)


@pytest.mark.parametrize(
    "case",
    [
        "rows",
        "columns",
        "duplicate",
        "nonnumeric",
        "rank",
        "fewtrain",
        "fewtest",
        "run_count",
    ],
)
def test_invalid_predictors_rejected(encoding_runs, case):
    betas, predictors = encoding_runs
    if case == "rows":
        predictors[0] = predictors[0].iloc[:-1]
    elif case == "columns":
        predictors[2] = predictors[2][["trial_type", "response_time"]]
    elif case == "duplicate":
        predictors = [x.set_axis(["same", "same"], axis=1) for x in predictors]
    elif case == "nonnumeric":
        predictors[0] = predictors[0].astype(str)
    elif case == "rank":
        predictors = [x.assign(trial_type=1) for x in predictors]
    elif case == "fewtrain":
        for r in (0, 1):
            predictors[r].loc[1:, "response_time"] = np.nan
    elif case == "fewtest":
        predictors[2].loc[1:, "response_time"] = np.nan
    else:
        predictors.pop()
    with pytest.raises(ValueError):
        evaluate(betas, predictors, train_runs=[0, 1], test_runs=[2, 3])
