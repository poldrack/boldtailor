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
RHO_RANGE = (-0.2, 0.95)
"""Default admissible range of the pooled rho. Strong negative
autocorrelation is not plausible for fMRI noise, so the lower limit sits just
below zero, leaving room for sampling error around a true value near zero.
The upper limit is the end of the correction grid."""


def lag_moments(residuals):
    """Per-feature numerator ``e'Le`` and denominator ``e'De``."""
    numerator = np.sum(residuals[1:] * residuals[:-1], axis=0)
    return numerator, np.sum(residuals[:-1] ** 2, axis=0)


def basis_lag_ratio(signals, basis, amplification) -> np.ndarray:
    """Lag-one ratio of residuals after projecting out ``basis``; NaN if degenerate.

    A feature is degenerate when its denominator does not exceed the
    uncertainty floor of the projection (``boldtailor._numerics.project_out``).
    """
    signals = np.asarray(signals, dtype=float)
    residuals, _, floor = project_out(signals, basis, amplification)
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


def _resolution(expected) -> float:
    """Differences of the dimensionless lag ratio at or below this are rounding."""
    return 64 * np.finfo(float).eps * max(1.0, np.max(np.abs(expected)))


def _branches(expected, slack):
    """Maximal runs of grid cells with one direction, as (first, last, direction).

    ``first`` and ``last`` index the run's end points. Direction is +1 or -1
    for a resolved rise or fall and 0 where the differences are rounding.
    """
    difference = np.diff(expected)
    direction = np.sign(difference) * (np.abs(difference) > slack)
    breaks = np.flatnonzero(np.diff(direction)) + 1
    firsts = np.concatenate([[0], breaks])
    lasts = np.concatenate([breaks, [direction.size]])
    return [(a, b, int(direction[a])) for a, b in zip(firsts, lasts, strict=True)]


def _pooling_branch(expected, grid, slack, rho_range):
    """The resolved monotone branch spanning ``rho_range``, or ``None``.

    The range is cut to the grid, which must be increasing.
    """
    low, high = max(rho_range[0], grid[0]), min(rho_range[1], grid[-1])
    for first, last, direction in _branches(expected, slack):
        if direction and grid[first] <= low and grid[last] >= high:
            return first, last, direction
    return None


def _branch_inverse(values, levels, rhos, direction):
    """Rho interpolated on one branch, read in its rising direction; values
    beyond the branch take the rho of its nearer end."""
    order = slice(None, None, direction)
    return np.interp(values, levels[order], rhos[order])


def invert_lag_ratio(values, expected, grid, rho_range=RHO_RANGE) -> np.ndarray:
    """Map observed ratios to rho through ``expected`` on an increasing ``grid``.

    The curve must rise or fall, by more than rounding at every step, across
    the whole of ``rho_range``, in which estimates are pooled. That branch, extended as
    far as it stays monotone, is inverted by interpolation for every value,
    and values beyond it are clipped to its ends, exactly as a fully monotone
    curve clips at the ends of the grid; individual estimates are not
    confined to ``rho_range``. A bend outside the range therefore only
    shortens the usable branch. No finite value is dropped and
    the order of the values is kept, so the pooled median is not biased by
    which values happen to sit at levels the bend repeats.

    A curve that is nonfinite, flat, or turns inside ``rho_range``
    identifies nothing there and gives NaN for every value. NaN observations
    stay NaN.
    """
    values = np.asarray(values, dtype=float)
    expected, grid = np.asarray(expected, dtype=float), np.asarray(grid, dtype=float)
    if expected.size < 2 or not np.all(np.isfinite(expected)):
        return np.full(values.shape, np.nan)
    branch = _pooling_branch(expected, grid, _resolution(expected), rho_range)
    if branch is None:
        return np.full(values.shape, np.nan)
    points = slice(branch[0], branch[1] + 1)
    corrected = _branch_inverse(values, expected[points], grid[points], branch[2])
    return np.where(np.isfinite(values), corrected, np.nan)


def correct_lag_ratio(values, design, rho_range=RHO_RANGE) -> np.ndarray:
    """Invert the expected lag ratio on the fixed rho grid; NaN stays NaN."""
    expected = expected_lag_ratio(design, GRID)
    return invert_lag_ratio(values, expected, GRID, rho_range)


def finite_median(values) -> float:
    """Median of the finite values; NaN if there are none."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else np.nan


def pool_rho(values, rho_range=RHO_RANGE) -> float:
    """Median of finite values clipped to ``rho_range``; 0.0 if none are finite."""
    median = finite_median(values)
    return float(np.clip(median, *rho_range)) if np.isfinite(median) else 0.0
