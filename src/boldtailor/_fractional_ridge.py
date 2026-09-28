"""Fractional ridge in the nuisance-projected, raw trial-coefficient basis."""

from dataclasses import dataclass
from hashlib import sha256
from numbers import Real

import numpy as np

from boldtailor._arrays import readonly_array
from boldtailor._single_trial_fit import TrialRunFit, _project_design, validate_alpha

NORM_BASIS = "raw_trial_coefficients_after_nuisance_projection"


def fraction_grid(values):
    grid = tuple(values)
    if (
        not grid
        or any(
            isinstance(f, (bool, np.bool_))
            or not isinstance(f, Real)
            or not np.isfinite(f)
            or not 0 < f <= 1
            for f in grid
        )
        or len(set(grid)) != len(grid)
    ):
        raise ValueError("fractions must be distinct finite numbers in (0, 1]")
    return tuple(float(f) for f in grid)


def fraction_map(values, n_features):
    raw = np.asarray(values, dtype=object)
    if raw.shape not in ((), (n_features,)) or any(
        isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) for v in raw.flat
    ):
        raise ValueError("ridge_fraction must be numeric, scalar or one per feature")
    array = np.full(n_features, float(raw)) if raw.ndim == 0 else np.asarray(raw, float)
    if np.any(np.isinf(array) | (array <= 0) | (array > 1)) or (
        raw.ndim == 0 and np.isnan(array).any()
    ):
        raise ValueError("fractions must be in (0, 1]; map NaNs exclude features")
    return array.copy()


def regularization(ridge_alpha, ridge_fraction, n_features):
    alpha = validate_alpha(ridge_alpha)
    if ridge_fraction is None:
        return alpha, None
    if alpha != 0:
        raise ValueError("choose ridge_alpha or ridge_fraction, not both")
    return None, fraction_map(ridge_fraction, n_features)


def fraction_metadata(fractions):
    return dict(
        regularization="fractional_ridge",
        normalization="none_after_nuisance_projection",
        fraction_norm_basis=NORM_BASIS,
        fraction_fingerprint=sha256(
            np.asarray(fractions, dtype="<f8").tobytes()
        ).hexdigest(),
        ridge_fraction=[float(f) if np.isfinite(f) else None for f in fractions],
        implied_alpha="computed_separately_for_each_run_and_feature",
    )


def freeze_fraction_result(result):
    if result.ridge_fraction is not None:
        object.__setattr__(
            result, "ridge_fraction", readonly_array(result.ridge_fraction)
        )
        object.__setattr__(
            result,
            "run_ridge_alphas",
            tuple(readonly_array(a) for a in result.run_ridge_alphas),
        )


@dataclass(frozen=True)
class PreparedFractionBetas:
    """Raw trial-coefficient basis and signal coordinates for fractional ridge."""

    trial_design: np.ndarray
    nuisance: np.ndarray
    signals: np.ndarray
    projected_signals: np.ndarray
    singular_values: np.ndarray
    right_vectors: np.ndarray
    ols_coordinates: np.ndarray
    valid: np.ndarray
    diagnostics: dict

    def solve(self, fractions):
        """Return trial betas and implied alpha for each feature's fraction."""
        fractions = fraction_map(fractions, self.signals.shape[1])
        valid = self.valid & np.isfinite(fractions)
        betas = np.full((self.trial_design.shape[1], self.signals.shape[1]), np.nan)
        alphas = np.full(self.signals.shape[1], np.nan)
        if valid.any():
            s, ols = self.singular_values, self.ols_coordinates[:, valid]
            alpha = _alphas(s, ols, fractions[valid])
            attenuation = (s * s)[:, None] / ((s * s)[:, None] + alpha)
            betas[:, valid] = self.right_vectors.T @ (ols * attenuation)
            alphas[valid] = alpha
        return betas, alphas

    def betas_at(self, fraction):
        """Evaluate a fraction without retaining candidate outputs."""
        return self.solve(fraction)[0]


def prepare_fraction_betas(x, nuisance, signals):
    """Validate in normalized coordinates, then prepare the raw coefficient SVD."""
    x, n, y = (np.asarray(a, dtype=float) for a in (x, nuisance, signals))
    design = _project_design(x, n)
    q = design.nuisance_basis
    # Rank validation must be scale invariant. Fractional shrinkage instead uses
    # raw trial amplitudes: column normalization would change the requested norm.
    u, s, vt = np.linalg.svd(x - q @ (q.T @ x), full_matrices=False)
    diagnostics = dict(design.diagnostics, condition_number=float(s[0] / s[-1]))
    yr = y - q @ (q.T @ y)
    coordinates = u.T @ yr
    tolerance = max(x.shape) * np.finfo(float).eps * np.linalg.norm(y, axis=0)
    valid = (np.ptp(y, axis=0) > 0) & (np.linalg.norm(coordinates, axis=0) > tolerance)
    return PreparedFractionBetas(
        x, n, y, yr, s, vt, coordinates / s[:, None], valid, diagnostics
    )


def _alphas(s, ols, fractions):
    """Bracket using singular-value extrema, then bisect in log-alpha space."""
    alphas = np.zeros(len(fractions))
    shrink = fractions < 1
    if not shrink.any():
        return alphas
    f = fractions[shrink]
    weights = ols[:, shrink] / np.linalg.norm(ols[:, shrink], axis=0)
    squared = s * s
    factor = np.log1p(-f) - np.log(f)
    lo, hi = np.log(squared.min()) + factor, np.log(squared.max()) + factor
    for _ in range(60):
        mid = (lo + hi) / 2
        attenuation = squared[:, None] / (squared[:, None] + np.exp(mid))
        above = np.linalg.norm(weights * attenuation, axis=0) > f
        lo, hi = np.where(above, mid, lo), np.where(above, hi, mid)
    alphas[shrink] = np.exp((lo + hi) / 2)
    return alphas


def fraction_beta_path(x, nuisance, signals, *, fractions):
    """Stream candidate betas/alpha maps with one design decomposition."""
    grid = fraction_grid(fractions)
    prepared = prepare_fraction_betas(x, nuisance, signals)
    for fraction in grid:
        betas, alphas = prepared.solve(fraction)
        yield fraction, betas, alphas


def fit_fraction_run(x, nuisance, signals, *, fractions):
    prepared = prepare_fraction_betas(x, nuisance, signals)
    x, n, y = prepared.trial_design, prepared.nuisance, prepared.signals
    betas, alphas = prepared.solve(fractions)
    valid = np.isfinite(alphas)
    gamma = np.full((n.shape[1], y.shape[1]), np.nan)
    full, null, total = (np.full(y.shape[1], np.nan) for _ in range(3))
    if valid.any():
        remainder = y[:, valid] - x @ betas[:, valid]
        gamma[:, valid] = np.linalg.lstsq(n, remainder, rcond=None)[0]
        full[valid] = np.sum((remainder - n @ gamma[:, valid]) ** 2, axis=0)
        null[valid] = np.sum(prepared.projected_signals[:, valid] ** 2, axis=0)
        total[valid] = np.sum((y[:, valid] - y[:, valid].mean(0)) ** 2, axis=0)
    return TrialRunFit(
        betas,
        gamma,
        full,
        null,
        total,
        dict(prepared.diagnostics, ridge_alphas=alphas, fraction_norm_basis=NORM_BASIS),
    )
