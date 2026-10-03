"""Experimental fractional fits with explicit training-only parameter estimation."""

from dataclasses import dataclass

import numpy as np

from boldtailor._fractional_ridge import _alphas
from boldtailor.fractional_ridge import fraction_grid
from boldtailor._single_trial_fit import _project_design
from boldtailor.trial_encoding import evaluate_trial_encoding
from examples.validation.ridge_objective_simulation import score_prediction_runs


@dataclass(frozen=True)
class PreparedRun:
    scale: np.ndarray
    singular_values: np.ndarray
    rotation: np.ndarray
    ols_coordinates: np.ndarray


def prepare_runs(designs, signals, *, basis):
    """Project nuisances independently per run, preserving original beta units."""
    if basis not in ("raw", "normalized"):
        raise ValueError("basis must be raw or normalized")
    prepared = []
    for (x, n), y in zip(designs, signals, strict=True):
        design = _project_design(x, n)
        q, scale = design.nuisance_basis, design.column_scale
        u, s, vt = design.left_vectors, design.singular_values, design.right_vectors
        if basis == "raw":
            scale = np.ones(x.shape[1])
            u, s, vt = np.linalg.svd(x - q @ (q.T @ x), full_matrices=False)
        yr = y - q @ (q.T @ y)
        coordinates = (u.T @ yr) / s[:, None]
        if not np.isfinite(coordinates).all() or np.any(
            np.linalg.norm(coordinates, axis=0) == 0
        ):
            raise ValueError("experiment requires finite, nonzero projected responses")
        prepared.append(PreparedRun(scale, s, vt, coordinates))
    return tuple(prepared)


def solve_run(run, alpha):
    squared = run.singular_values[:, None] ** 2
    coordinates = run.ols_coordinates * squared / (squared + alpha)
    return (run.rotation.T @ coordinates) / run.scale[:, None]


def _fraction_alpha(runs, fraction):
    singular = np.concatenate([r.singular_values for r in runs])
    coordinates = np.concatenate([r.ols_coordinates for r in runs])
    return _alphas(singular, coordinates, np.full(coordinates.shape[1], fraction))


def _split_betas(prepared, train, test, fraction, scope):
    if scope not in ("per_run", "pooled_train"):
        raise ValueError("scope must be per_run or pooled_train")
    shared = (
        _fraction_alpha([prepared[r] for r in train], fraction)
        if scope == "pooled_train"
        else None
    )
    alphas = {}
    betas = [
        np.full((len(r.scale), r.ols_coordinates.shape[1]), np.nan) for r in prepared
    ]
    for r in (*train, *test):
        alpha = _fraction_alpha([prepared[r]], fraction) if shared is None else shared
        alphas[r] = alpha
        betas[r] = solve_run(prepared[r], alpha)
    return betas, alphas


def fit_calibration(beta_runs, ols_runs):
    """Learn an affine map using training pairs only; unstable slopes use identity."""
    raw, target = np.concatenate(beta_runs), np.concatenate(ols_runs)
    mapping = np.vstack([np.ones(raw.shape[1]), np.zeros(raw.shape[1])])
    for v in range(raw.shape[1]):
        if np.ptp(raw[:, v]) == 0:
            continue
        design = np.column_stack([raw[:, v], np.ones(len(raw))])
        coef, _, rank, _ = np.linalg.lstsq(design, target[:, v], rcond=None)
        if rank == 2 and np.isfinite(coef).all() and coef[0] > 0:
            mapping[:, v] = coef
    return mapping


def apply_calibration(beta_runs, mapping):
    return tuple(b * mapping[0] + mapping[1] for b in beta_runs)


def _achieved_fractions(prepared, test, betas):
    return np.stack(
        [
            np.linalg.norm(betas[r] * prepared[r].scale[:, None], axis=0)
            / np.linalg.norm(prepared[r].ols_coordinates, axis=0)
            for r in test
        ]
    )


def fit_split(prepared, predictors, *, train, test, fraction, target, scope):
    """Fit a candidate without sharing held-out outcomes with training parameters."""
    fraction_grid([fraction])
    if target not in ("fixed_ols", "candidate"):
        raise ValueError("target must be fixed_ols or candidate")
    train, test = tuple(train), tuple(test)
    if not train or not test or set(train) & set(test):
        raise ValueError("training and test runs must be nonempty and disjoint")
    betas, alphas = _split_betas(prepared, train, test, fraction, scope)
    encoded = evaluate_trial_encoding(
        betas, predictors, train_runs=train, test_runs=test
    )
    ols = {r: solve_run(prepared[r], 0.0) for r in (*train, *test)}
    targets = tuple(ols[r] if target == "fixed_ols" else betas[r] for r in test)
    train_betas, test_betas = tuple(betas[r] for r in train), tuple(
        betas[r] for r in test
    )
    mapping = fit_calibration(train_betas, [ols[r] for r in train])
    scores = score_prediction_runs(targets, encoded.predictions, center=True)
    return dict(
        scores,
        train_betas=train_betas,
        test_betas=test_betas,
        targets=targets,
        ols_targets=tuple(ols[r] for r in test),
        training_alphas=np.stack([alphas[r] for r in train]),
        test_alphas=np.stack([alphas[r] for r in test]),
        test_fractions=_achieved_fractions(prepared, test, betas),
        coefficients=encoded.coefficients,
        train_intercepts=encoded.train_run_intercepts,
        predictions=encoded.predictions,
        calibration=mapping,
        calibrated_betas=apply_calibration(test_betas, mapping),
        calibrated_predictions=apply_calibration(encoded.predictions, mapping),
    )
