"""Run-wise OLS and normalized ridge with an unpenalized nuisance span."""

from dataclasses import dataclass
from numbers import Real

import numpy as np
from nilearn.glm.first_level import run_glm


@dataclass(frozen=True)
class TrialRunFit:
    betas: np.ndarray
    nuisance_betas: np.ndarray
    full_sse: np.ndarray
    nuisance_sse: np.ndarray
    total_ss: np.ndarray
    diagnostics: dict


def validate_alpha(alpha):
    if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, Real):
        raise ValueError("ridge_alpha must be a finite nonnegative number")
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("ridge_alpha must be a finite nonnegative number")
    return float(alpha)


def _project_design(x, nuisance):
    u, s, _ = np.linalg.svd(nuisance, full_matrices=False)
    tolerance = s[0] * max(nuisance.shape) * np.finfo(float).eps
    nuisance_rank = int(np.sum(s > tolerance))
    q = u[:, :nuisance_rank]
    xr = x - q @ (q.T @ x)
    scale = np.linalg.norm(xr, axis=0)
    tolerance = np.linalg.norm(x, ord=2) * max(x.shape) * np.finfo(float).eps
    if np.any(scale <= tolerance):
        raise ValueError("trial columns have no support outside nuisance span")
    u, s, vt = np.linalg.svd(xr / scale, full_matrices=False)
    task_rank = int(np.sum(s > s[0] * max(x.shape) * np.finfo(float).eps))
    if task_rank != x.shape[1]:
        raise ValueError("residualized trial design is rank deficient")
    dof = len(x) - nuisance_rank - task_rank
    if dof <= 0:
        raise ValueError(
            "single-trial fit requires positive residual degrees of freedom"
        )
    diagnostics = dict(
        rank=nuisance_rank + task_rank,
        nuisance_rank=nuisance_rank,
        task_rank=task_rank,
        residual_dof=dof,
        condition_number=float(s[0] / s[-1]),
    )
    return q, scale, u, s, vt, diagnostics


def fit_trial_run(x, nuisance, signals, *, alpha):
    """Fit all trials together; alpha acts on unit-norm projected columns."""
    alpha = validate_alpha(alpha)
    x, nuisance, y = (np.asarray(a, dtype=float) for a in (x, nuisance, signals))
    q, scale, u, s, vt, diagnostics = _project_design(x, nuisance)
    varying = np.ptp(y, axis=0) > 0
    values = y[:, varying]
    betas = np.full((x.shape[1], y.shape[1]), np.nan)
    gamma = np.full((nuisance.shape[1], y.shape[1]), np.nan)
    full_sse = np.zeros(y.shape[1])
    nuisance_sse = np.zeros(y.shape[1])
    total_ss = np.zeros(y.shape[1])
    if values.shape[1]:
        yr = values - q @ (q.T @ values)
        if alpha == 0:
            # Match the normalized span used for rank validation and ridge.
            # Raw nuisance units must not determine which trials are retained.
            xs = (x - q @ (q.T @ x)) / scale
            _, fits = run_glm(yr, xs, noise_model="ols")
            beta = fits[0.0].theta / scale[:, None]
        else:
            weights = (vt.T * (s / (s * s + alpha))) @ (u.T @ yr)
            beta = weights / scale[:, None]
        coefficients = np.linalg.lstsq(nuisance, values - x @ beta, rcond=None)[0]
        betas[:, varying], gamma[:, varying] = beta, coefficients
        residual = values - x @ beta - nuisance @ coefficients
        full_sse[varying] = np.sum(residual**2, axis=0)
        nuisance_sse[varying] = np.sum(yr**2, axis=0)
        total_ss[varying] = np.sum((values - values.mean(axis=0)) ** 2, axis=0)
    return TrialRunFit(betas, gamma, full_sse, nuisance_sse, total_ss, diagnostics)


def r_squared(sse, total_ss):
    result = np.full_like(total_ss, np.nan, dtype=float)
    np.divide(sse, total_ss, out=result, where=total_ss > 0)
    return 1 - result
