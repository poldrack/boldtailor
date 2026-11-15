"""Shared numerical policy: rank tolerance, column bases and projection error.

Callers keep their own statistical requirements (full column rank, redundant
columns allowed, ...); this module only fixes how rank and degeneracy are
judged numerically, so that related kernels agree.
"""

import numpy as np

PROJECTION_ERROR_FACTOR = 8
"""Multiple of a projection's rounding scale below which a residual is noise.

For ``r = y - Q(Q'y)`` with ``n`` scans there are two scales, both in norm
units and both calibrated on signals lying exactly in the span of the design
(``tests/test_numerics.py``):

* rounding, ``n eps ||y||``: the arithmetic of the projection. With columns
  normalised, exact-span signals leave at most about 1.2 times this scale
  when ill-conditioning comes only from column units (condition numbers to
  1e12, ranks from 1 to n - 1, n = 10..1000).
* uncertainty, ``n eps`` times the norm of ``y`` with its coordinate on each
  basis vector amplified by ``s_max / s_i`` of the normalised design. A basis
  computed from ``C`` spans the range of ``C + E`` with ``||E|| ~ n eps``, so a
  component of ``y`` along a weak direction, one that near-collinear columns
  barely determine, leaves proportionally more. Exact-span signals leave at
  most 1.0 times this scale, and up to 1e5 times the rounding scale.

A residual above the uncertainty floor is resolved. One below the rounding
floor is rounding. In between it may be real but cannot be told apart from
the error of the computed span. These are floating-point floors in float64,
not statistical tests, and cannot detect residual information already lost
when the signal was stored.
"""


def rank_tolerance(shape, spectral_norm) -> float:
    """Singular values at or below this are treated as zero."""
    return max(shape) * np.finfo(float).eps * spectral_norm


def _unit_columns(matrix) -> np.ndarray:
    """Nonzero columns scaled to unit norm, which leaves their span unchanged.

    Each column is divided by its largest magnitude first, so that columns in
    extreme units neither overflow nor underflow when their norm is taken.
    """
    peak = np.abs(matrix).max(axis=0)
    scaled = matrix[:, peak > 0] / peak[peak > 0]
    return scaled / np.linalg.norm(scaled, axis=0)


def conditioned_basis(matrix) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal basis of the column span, and ``s_max / s_i`` of each vector.

    Columns are normalised first, so the basis, the rank decision and the
    amplification depend on the span and the collinearity of the columns, not
    on their units. Only exactly zero columns are dropped for their size.
    The basis is row-major, like the copies that projections store, so that
    products with either round identically.
    """
    scaled = _unit_columns(np.asarray(matrix, dtype=float))
    if scaled.shape[1] == 0:
        return np.empty((scaled.shape[0], 0)), np.empty(0)
    u, s, _ = np.linalg.svd(scaled, full_matrices=False)
    keep = s > rank_tolerance(scaled.shape, s[0])
    return np.ascontiguousarray(u[:, keep]), s[0] / s[keep]


def orthonormal_basis(matrix) -> np.ndarray:
    """Orthonormal basis of the column span of ``matrix``."""
    return conditioned_basis(matrix)[0]


def residualise(signals, basis) -> np.ndarray:
    """Signals minus their projection on an orthonormal ``basis``."""
    return signals - basis @ (basis.T @ signals)


def _amplified_energy(energy, coordinates, amplification):
    """Squared norm of the signals with basis coordinates amplified."""
    excess = (amplification**2 - 1.0)[:, None] * coordinates**2
    return energy + excess.sum(axis=0)


def project_out(signals, basis, amplification):
    """Residual off ``basis`` with its rounding and uncertainty energy floors.

    Returns ``(residual, rounding_floor, uncertainty_floor)``; the floors are
    squared norms, per feature, with ``rounding_floor <= uncertainty_floor``.
    See ``PROJECTION_ERROR_FACTOR``.
    """
    coordinates = basis.T @ signals
    relative = PROJECTION_ERROR_FACTOR * len(signals) * np.finfo(float).eps
    peak = np.abs(signals).max(axis=0)
    safe_peak = np.where(peak > 0, peak, 1.0)
    unit_energy = np.sum((signals / safe_peak) ** 2, axis=0)
    unit_amplified = _amplified_energy(
        unit_energy, coordinates / safe_peak, amplification
    )
    scale2 = (relative * peak) ** 2
    return (
        signals - basis @ coordinates,
        scale2 * unit_energy,
        scale2 * unit_amplified,
    )
