"""Predict trial betas with shared slopes and explicit run-baseline semantics."""

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


def validate_encoding_mode(encoding_mode):
    """Reject ambiguous objectives before doing any model fitting."""
    if not isinstance(encoding_mode, str) or encoding_mode not in (
        "within_run",
        "absolute",
    ):
        raise ValueError("encoding_mode must be 'within_run' or 'absolute'")
    return encoding_mode


def _center(values):
    # Subtract a reference first so exactly constant columns remain exactly zero.
    shifted = values - values[0]
    return shifted - shifted.mean(axis=0)


def _within_design(arrays, masks, train):
    runs = [arrays[r][masks[r]] for r in train]
    if any(len(x) == 0 for x in runs):
        raise ValueError("each encoding training run needs a complete predictor row")
    values = np.concatenate(runs)
    if len(values) <= len(train) + values.shape[1]:
        raise ValueError(
            "encoding training design needs positive residual degrees of freedom"
        )
    design = np.concatenate([_center(x) for x in runs])
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("within-run encoding predictor design is rank deficient")
    return design, values.mean(axis=0)


def _training_design(arrays, masks, train, *, encoding_mode="absolute"):
    validate_encoding_mode(encoding_mode)
    if encoding_mode == "within_run":
        return _within_design(arrays, masks, train)
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


def _fit_within_coefficients(design, betas, masks, train, run_means):
    runs = [betas[r][masks[r]] for r in train]
    target = np.concatenate(runs)
    valid = np.isfinite(target).all(axis=0)
    coefficients = np.full((design.shape[1] + 1, target.shape[1]), np.nan)
    coefficients[0, valid] = target[:, valid].mean(axis=0)
    centered = np.concatenate([_center(y[:, valid]) for y in runs])
    coefficients[1:, valid] = np.linalg.lstsq(design, centered, rcond=None)[0]
    intercepts = np.full((len(train), target.shape[1]), np.nan)
    intercepts[:, valid] = (
        np.stack([y[:, valid].mean(0) for y in runs])
        - run_means @ coefficients[1:, valid]
    )
    return coefficients, intercepts


def _predict_run(beta, values, mask, means, coefficients, *, encoding_mode):
    if mask.sum() < 2:
        raise ValueError("encoding test runs need at least two complete trials")
    design = np.column_stack([np.ones(mask.sum()), values[mask] - means])
    target = beta[mask]
    fitted = np.isfinite(coefficients).all(axis=0)
    valid = np.isfinite(target).all(axis=0) & fitted
    prediction = np.full(beta.shape, np.nan)
    predictable = fitted if encoding_mode == "within_run" else valid
    prediction[np.ix_(mask, predictable)] = design @ coefficients[:, predictable]
    sse, sst, offsets = np.full((3, beta.shape[1]), np.nan)
    observed = target[:, valid]
    residual = observed - prediction[np.ix_(mask, valid)]
    offsets[valid] = residual.mean(axis=0) if encoding_mode == "within_run" else 0.0
    sse[valid] = np.sum((residual - offsets[valid]) ** 2, axis=0)
    sst[valid] = np.sum((observed - observed.mean(axis=0)) ** 2, axis=0)
    return prediction, sse, sst, offsets


def evaluate_trial_encoding(
    beta_runs, predictors, *, train_runs, test_runs, encoding_mode="within_run"
):
    """Fit training slopes, then score within-run deviations or absolute levels.

    Predictor rows correspond positionally to beta/event rows. Nonfinite
    predictors exclude a trial from encoding only; its output row remains NaN.
    The default fits separate training-run intercepts and centers test residuals
    for scoring only. Returned predictions use a training-derived reference;
    scoring offsets never alter predictions. Absolute mode preserves the shared
    intercept and uncentered loss. Scores pool SSE/SST, retaining undefined R².
    """
    validate_encoding_mode(encoding_mode)
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
    design, means = _training_design(arrays, masks, train, encoding_mode=encoding_mode)
    # Absolute mode historically permits an empty individual training run.
    run_means = np.stack(
        [
            (
                arrays[r][masks[r]].mean(0)
                if masks[r].any()
                else np.full(len(columns), np.nan)
            )
            for r in train
        ]
    )
    if encoding_mode == "within_run":
        coefficients, intercepts = _fit_within_coefficients(
            design, betas, masks, train, run_means
        )
    else:
        coefficients = _fit_coefficients(design, betas, masks, train)
        intercepts = np.repeat(
            (coefficients[0] - means @ coefficients[1:])[None], len(train), axis=0
        )
    fits = [
        _predict_run(
            betas[r],
            arrays[r],
            masks[r],
            means,
            coefficients,
            encoding_mode=encoding_mode,
        )
        for r in test
    ]
    sse, sst = np.stack([f[1] for f in fits]), np.stack([f[2] for f in fits])
    return TrialEncodingResult(
        coefficients=coefficients,
        predictor_means=means,
        predictor_names=("task", *columns),
        train_runs=train,
        test_runs=test,
        trial_masks=masks,
        predictions=tuple(f[0] for f in fits),
        run_sse=sse,
        run_sst=sst,
        r2=r_squared(sse.sum(axis=0), sst.sum(axis=0)),
        encoding_mode=encoding_mode,
        train_run_predictor_means=run_means,
        train_run_intercepts=intercepts,
        scoring_offsets=np.stack([f[3] for f in fits]),
    )
