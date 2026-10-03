"""Run-wise OLS and normalized ridge with an unpenalized nuisance span."""

from dataclasses import dataclass

import numpy as np

from boldtailor._scalars import is_real


@dataclass(frozen=True)
class TrialRunFit:
    betas: np.ndarray
    nuisance_betas: np.ndarray
    full_sse: np.ndarray
    nuisance_sse: np.ndarray
    total_ss: np.ndarray
    diagnostics: dict


@dataclass(frozen=True)
class ProjectedTrialDesign:
    """SVD of unit-norm trial columns after removing the nuisance span."""

    nuisance_basis: np.ndarray
    column_scale: np.ndarray
    left_vectors: np.ndarray
    singular_values: np.ndarray
    right_vectors: np.ndarray
    diagnostics: dict


@dataclass(frozen=True)
class PreparedTrialBetas:
    """Signal coordinates shared by all normalized ridge candidates."""

    design: ProjectedTrialDesign
    coordinates: np.ndarray
    varying: np.ndarray
    shape: tuple[int, int]

    def betas_at(self, alpha):
        """Return trial-by-feature coefficients without retaining the output."""
        alpha = validate_alpha(alpha)
        d = self.design
        s = d.singular_values
        attenuation = 1 / s if alpha == 0 else s / (s * s + alpha)
        coefficients = d.right_vectors.T @ (attenuation[:, None] * self.coordinates)
        betas = np.full(self.shape, np.nan)
        betas[:, self.varying] = coefficients / d.column_scale[:, None]
        return betas


def prepare_trial_betas(x, nuisance, signals):
    """Project signals and factor the normalized trial design once."""
    x, nuisance, y = (np.asarray(a, dtype=float) for a in (x, nuisance, signals))
    design = project_trial_design(x, nuisance)
    varying = np.ptp(y, axis=0) > 0
    values = y[:, varying]
    q = design.nuisance_basis
    coordinates = design.left_vectors.T @ (values - q @ (q.T @ values))
    return PreparedTrialBetas(design, coordinates, varying, (x.shape[1], y.shape[1]))


def validate_alpha(alpha):
    if not is_real(alpha):
        raise ValueError("ridge_alpha must be a finite nonnegative number")
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("ridge_alpha must be a finite nonnegative number")
    return float(alpha)


def nuisance_span(nuisance):
    """Orthonormal basis of the nuisance column span and its rank."""
    u, s, _ = np.linalg.svd(nuisance, full_matrices=False)
    tolerance = s[0] * max(nuisance.shape) * np.finfo(float).eps
    rank = int(np.sum(s > tolerance))
    return u[:, :rank], rank


def residual_column_scale(x, q):
    """Column norms of the nuisance-residualized design; reject unsupported ones."""
    scale = np.linalg.norm(x - q @ (q.T @ x), axis=0)
    tolerance = np.linalg.norm(x, ord=2) * max(x.shape) * np.finfo(float).eps
    if np.any(scale <= tolerance):
        raise ValueError("trial columns have no support outside nuisance span")
    return scale


def design_diagnostics(x, normalized_s, nuisance_rank):
    """Rank and degrees-of-freedom checks from normalized singular values."""
    s = normalized_s
    task_rank = int(np.sum(s > s[0] * max(x.shape) * np.finfo(float).eps))
    if task_rank != x.shape[1]:
        raise ValueError("residualized trial design is rank deficient")
    dof = len(x) - nuisance_rank - task_rank
    if dof <= 0:
        raise ValueError(
            "single-trial fit requires positive residual degrees of freedom"
        )
    return dict(
        rank=nuisance_rank + task_rank,
        nuisance_rank=nuisance_rank,
        task_rank=task_rank,
        residual_dof=dof,
        condition_number=float(s[0] / s[-1]),
    )


def project_trial_design(x, nuisance):
    q, nuisance_rank = nuisance_span(nuisance)
    scale = residual_column_scale(x, q)
    u, s, vt = np.linalg.svd((x - q @ (q.T @ x)) / scale, full_matrices=False)
    diagnostics = design_diagnostics(x, s, nuisance_rank)
    return ProjectedTrialDesign(q, scale, u, s, vt, diagnostics)


def fit_trial_run(x, nuisance, signals, *, alpha):
    """Fit all trials together; alpha acts on unit-norm projected columns."""
    alpha = validate_alpha(alpha)
    x, nuisance, y = (np.asarray(a, dtype=float) for a in (x, nuisance, signals))
    prepared = prepare_trial_betas(x, nuisance, y)
    q = prepared.design.nuisance_basis
    varying = prepared.varying
    values = y[:, varying]
    betas = prepared.betas_at(alpha)
    gamma = np.full((nuisance.shape[1], y.shape[1]), np.nan)
    full_sse = np.zeros(y.shape[1])
    nuisance_sse = np.zeros(y.shape[1])
    total_ss = np.zeros(y.shape[1])
    if values.shape[1]:
        yr = values - q @ (q.T @ values)
        beta = betas[:, varying]
        coefficients = np.linalg.lstsq(nuisance, values - x @ beta, rcond=None)[0]
        betas[:, varying], gamma[:, varying] = beta, coefficients
        residual = values - x @ beta - nuisance @ coefficients
        full_sse[varying] = np.sum(residual**2, axis=0)
        nuisance_sse[varying] = np.sum(yr**2, axis=0)
        total_ss[varying] = np.sum((values - values.mean(axis=0)) ** 2, axis=0)
    return TrialRunFit(
        betas, gamma, full_sse, nuisance_sse, total_ss, prepared.design.diagnostics
    )


def r_squared(sse, total_ss):
    result = np.full_like(total_ss, np.nan, dtype=float)
    np.divide(sse, total_ss, out=result, where=total_ss > 0)
    return 1 - result
