"""Predict trial betas in independent runs with a shared OLS encoding model."""

from numbers import Integral

import numpy as np
import pandas as pd

from boldtailor._single_trial_fit import r_squared
from boldtailor.ridge_results import TrialEncodingResult


def _run_indices(values, n_runs, name):
    indices = tuple(values)
    if (
        not indices
        or len(set(indices)) != len(indices)
        or any(
            isinstance(i, (bool, np.bool_))
            or not isinstance(i, Integral)
            or not 0 <= i < n_runs
            for i in indices
        )
    ):
        raise ValueError(f"{name} requires unique valid run indices")
    return tuple(int(i) for i in indices)


def _predictor_arrays(predictors, n_trials):
    if len(predictors) != len(n_trials) or not predictors:
        raise ValueError("predictor and beta run counts must agree")
    columns = (
        tuple(predictors[0].columns) if isinstance(predictors[0], pd.DataFrame) else ()
    )
    if (
        not columns
        or len(set(columns)) != len(columns)
        or "task" in columns
        or any(not isinstance(c, str) or not c for c in columns)
    ):
        raise ValueError(
            "predictors need unique named columns; task is the added intercept"
        )
    arrays = []
    for frame, count in zip(predictors, n_trials, strict=True):
        if (
            not isinstance(frame, pd.DataFrame)
            or tuple(frame.columns) != columns
            or len(frame) != count
        ):
            raise ValueError("predictor rows and ordered columns must match all runs")
        if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes):
            raise ValueError("predictor columns must be numeric")
        arrays.append(frame.to_numpy(dtype=float, na_value=np.nan))
    return arrays, columns


def _training_design(arrays, masks, train):
    values = np.concatenate([arrays[r][masks[r]] for r in train])
    if len(values) <= values.shape[1] + 1:
        raise ValueError(
            "encoding training runs need more complete trials than columns"
        )
    means = values.mean(axis=0)
    design = np.column_stack([np.ones(len(values)), values - means])
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("encoding training predictor design is rank deficient")
    return design, means


def _fit_coefficients(design, betas, masks, train):
    target = np.concatenate([betas[r][masks[r]] for r in train])
    valid = np.isfinite(target).all(axis=0)
    coefficients = np.full((design.shape[1], target.shape[1]), np.nan)
    coefficients[:, valid] = np.linalg.lstsq(design, target[:, valid], rcond=None)[0]
    return coefficients


def _predict_run(beta, values, mask, means, coefficients):
    if mask.sum() < 2:
        raise ValueError("encoding test runs need at least two complete trials")
    design = np.column_stack([np.ones(mask.sum()), values[mask] - means])
    target = beta[mask]
    valid = np.isfinite(target).all(axis=0) & np.isfinite(coefficients).all(axis=0)
    prediction = np.full(beta.shape, np.nan)
    prediction[np.ix_(mask, valid)] = design @ coefficients[:, valid]
    sse = np.full(beta.shape[1], np.nan)
    sst = sse.copy()
    observed = target[:, valid]
    sse[valid] = np.sum((observed - prediction[np.ix_(mask, valid)]) ** 2, axis=0)
    sst[valid] = np.sum((observed - observed.mean(axis=0)) ** 2, axis=0)
    return prediction, sse, sst


def evaluate_trial_encoding(beta_runs, predictors, *, train_runs, test_runs):
    """Fit on training betas, predict test betas without refitting their mean.

    Predictor rows correspond positionally to beta/event rows. Nonfinite
    predictors exclude a trial from encoding only; its output row remains NaN.
    Scores pool SSE and within-run SST, retaining negative and undefined R².
    """
    betas = tuple(np.asarray(b, dtype=float) for b in beta_runs)
    if (
        not betas
        or any(b.ndim != 2 or not all(b.shape) for b in betas)
        or len({b.shape[1] for b in betas}) != 1
    ):
        raise ValueError("beta runs must have trial rows and matching feature columns")
    train = _run_indices(train_runs, len(betas), "train_runs")
    test = _run_indices(test_runs, len(betas), "test_runs")
    if set(train) & set(test):
        raise ValueError("training and test runs must be disjoint")
    arrays, columns = _predictor_arrays(predictors, [len(b) for b in betas])
    masks = tuple(np.isfinite(x).all(axis=1) for x in arrays)
    design, means = _training_design(arrays, masks, train)
    coefficients = _fit_coefficients(design, betas, masks, train)
    fits = [
        _predict_run(betas[r], arrays[r], masks[r], means, coefficients) for r in test
    ]
    sse, sst = np.stack([f[1] for f in fits]), np.stack([f[2] for f in fits])
    return TrialEncodingResult(
        coefficients,
        means,
        ("task", *columns),
        train,
        test,
        masks,
        tuple(f[0] for f in fits),
        sse,
        sst,
        r_squared(sse.sum(axis=0), sst.sum(axis=0)),
    )
