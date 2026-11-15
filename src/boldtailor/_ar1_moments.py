"""Lag-one residual autocorrelation and its projection-induced bias correction.

With ``M = I - QQ'`` the residual-forming matrix of a design with orthonormal
basis ``Q``, the estimator is ``e'Le / e'De`` for residuals ``e = My``, where
``L`` has 0.5 on the first off-diagonals and ``D`` is the identity without its
last diagonal entry. Its expectation under AR(1) noise is approximated by
``tr(MLM R(rho)) / tr(MDM R(rho))``, inverted on a fixed rho grid.
"""

import numpy as np

from boldtailor._numerics import conditioned_basis, orthonormal_basis, project_out

GRID = np.linspace(-0.95, 0.95, 2001)


def lag_moments(residuals):
    """Per-feature numerator ``e'Le`` and denominator ``e'De``."""
    numerator = np.sum(residuals[1:] * residuals[:-1], axis=0)
    return numerator, np.sum(residuals[:-1] ** 2, axis=0)


def basis_lag_ratio(signals, basis, amplification) -> np.ndarray:
    """Lag-one ratio of residuals after projecting out ``basis``; NaN if degenerate.

    A feature is degenerate when its denominator does not exceed the rounding
    floor of the projection (see ``boldtailor._numerics.project_out``).
    """
    signals = np.asarray(signals, dtype=float)
    residuals, floor = project_out(signals, basis, amplification)
    numerator, denominator = lag_moments(residuals)
    valid = denominator > floor
    return np.where(valid, numerator / np.where(valid, denominator, 1.0), np.nan)


def residual_lag_ratio(signals, design) -> np.ndarray:
    """Lag-one ratio of OLS residuals per feature; NaN where degenerate."""
    return basis_lag_ratio(signals, *conditioned_basis(design))


def _lag_sums(matrix):
    """Coefficients c_k with tr(matrix R(rho)) = sum_k c_k rho**k."""
    k = np.arange(len(matrix))
    sums = np.array([np.trace(matrix, offset=int(i)) for i in k])
    return np.where(k == 0, sums, 2 * sums)


def _sandwich(basis, matrix):
    """M A M for M = I - QQ', using only (scans x rank) products."""
    left = matrix - basis @ (basis.T @ matrix)
    return left - (left @ basis) @ basis.T


def _moment_matrices(n):
    lag = 0.5 * (np.eye(n, k=1) + np.eye(n, k=-1))
    previous = np.eye(n)
    previous[-1, -1] = 0.0
    return lag, previous


def moment_polynomials(basis):
    """Polynomial coefficients in rho of tr(MLM R) and tr(MDM R)."""
    lag, previous = _moment_matrices(len(basis))
    return _lag_sums(_sandwich(basis, lag)), _lag_sums(_sandwich(basis, previous))


def _evaluate(coefficients, rhos):
    return np.polynomial.polynomial.polyval(rhos, coefficients)


def basis_expected_lag_ratio(basis, rhos) -> np.ndarray:
    """tr(M L M R(rho)) / tr(M D M R(rho)) for each rho."""
    numerator, denominator = moment_polynomials(basis)
    rhos = np.asarray(rhos, dtype=float)
    return _evaluate(numerator, rhos) / _evaluate(denominator, rhos)


def expected_lag_ratio(design, rhos) -> np.ndarray:
    """tr(M L M R(rho)) / tr(M D M R(rho)) for each rho."""
    return basis_expected_lag_ratio(orthonormal_basis(design), rhos)


def invert_lag_ratio(values, expected, grid) -> np.ndarray:
    """Map observed ratios to rho through ``expected`` on ``grid``.

    Linear interpolation where ``expected`` increases strictly, else the
    nearest grid value; NaN stays NaN.
    """
    values = np.asarray(values, dtype=float)
    if np.all(np.diff(expected) > 0):
        corrected = np.interp(values, expected, grid)
    else:
        nearest = np.abs(expected[:, None] - values.ravel()[None, :]).argmin(axis=0)
        corrected = grid[nearest].reshape(values.shape)
    return np.where(np.isfinite(values), corrected, np.nan)


def correct_lag_ratio(values, design) -> np.ndarray:
    """Invert the expected lag ratio on the fixed rho grid; NaN stays NaN."""
    return invert_lag_ratio(values, expected_lag_ratio(design, GRID), GRID)


def pool_rho(values) -> float:
    """Median of finite values clipped to [-0.8, 0.8]; 0.0 if none are finite."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    return float(np.clip(np.median(finite), -0.8, 0.8))
