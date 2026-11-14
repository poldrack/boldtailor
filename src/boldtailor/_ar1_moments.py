"""Lag-one residual autocorrelation and its projection-induced bias correction."""

import numpy as np

_GRID = np.linspace(-0.95, 0.95, 2001)
_DEGENERACY_FACTOR = 64


def _projector(design):
    q, singular, _ = np.linalg.svd(design, full_matrices=False)
    rank = np.sum(singular > singular[0] * max(design.shape) * np.finfo(float).eps)
    return np.eye(len(design)) - q[:, :rank] @ q[:, :rank].T


def residual_lag_ratio(signals, design) -> np.ndarray:
    """Lag-one ratio of OLS residuals per feature; NaN where degenerate."""
    signals = np.asarray(signals, dtype=float)
    residuals = _projector(np.asarray(design, dtype=float)) @ signals
    denominator = np.sum(residuals[:-1] ** 2, axis=0)
    numerator = np.sum(residuals[1:] * residuals[:-1], axis=0)
    floor = _DEGENERACY_FACTOR * np.finfo(float).eps * np.sum(signals**2, axis=0)
    valid = denominator > floor
    return np.where(valid, numerator / np.where(valid, denominator, 1.0), np.nan)


def _lag_sums(matrix):
    """Coefficients c_k with tr(matrix R(rho)) = sum_k c_k rho**k."""
    k = np.arange(len(matrix))
    sums = np.array([np.trace(matrix, offset=int(i)) for i in k])
    return np.where(k == 0, sums, 2 * sums)


def _moment_polynomials(design):
    m = _projector(np.asarray(design, dtype=float))
    n = len(m)
    lag = 0.5 * (np.eye(n, k=1) + np.eye(n, k=-1))
    previous = np.eye(n)
    previous[-1, -1] = 0.0
    return _lag_sums(m @ lag @ m), _lag_sums(m @ previous @ m)


def _evaluate(coefficients, rhos):
    return np.polynomial.polynomial.polyval(rhos, coefficients)


def expected_lag_ratio(design, rhos) -> np.ndarray:
    """tr(M L M R(rho)) / tr(M D M R(rho)) for each rho."""
    numerator, denominator = _moment_polynomials(design)
    rhos = np.asarray(rhos, dtype=float)
    return _evaluate(numerator, rhos) / _evaluate(denominator, rhos)


def correct_lag_ratio(values, design) -> np.ndarray:
    """Invert the expected lag ratio on a fixed rho grid; NaN stays NaN."""
    values = np.asarray(values, dtype=float)
    expected = expected_lag_ratio(design, _GRID)
    if np.all(np.diff(expected) > 0):
        corrected = np.interp(values, expected, _GRID)
    else:
        nearest = np.abs(expected[:, None] - values.ravel()[None, :]).argmin(axis=0)
        corrected = _GRID[nearest].reshape(values.shape)
    return np.where(np.isfinite(values), corrected, np.nan)


def pool_rho(values) -> float:
    """Median of finite values clipped to [-0.8, 0.8]; 0.0 if none are finite."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    return float(np.clip(np.median(finite), -0.8, 0.8))
