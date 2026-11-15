"""Shared numerical policy: rank tolerance, column bases and projection error.

Callers keep their own statistical requirements (full column rank, redundant
columns allowed, ...); this module only fixes how rank and degeneracy are
judged numerically, so that related kernels agree.
"""

import numpy as np

PROJECTION_ERROR_FACTOR = 8
"""Multiple of the projection's rounding scale below which a residual is noise.

A basis ``Q`` computed from a design ``C`` spans the range of ``C + E`` with
``||E|| ~ n eps ||C||``, so for ``r = y - Q(Q'y)`` a component of ``y`` along
the basis vector with singular value ``s_i`` leaves rounding of relative size
``n eps s_max / s_i``. The error scale is therefore ``n eps`` times the norm
of ``y`` with its coordinate on each basis vector amplified by
``s_max / s_i``. Signals lying exactly in the span leave at most 1.0 times
that scale over n = 10..1000, ranks from 1 to n - 1, condition numbers up to
1e12, amplitudes 1e-3..1e6 and AR(1) whitening (observed while calibrating;
pinned in ``tests/test_numerics.py``), whereas against ``||y||`` alone the
residue reaches 1e5 times ``n eps``. The factor leaves most of an order of
magnitude above the observed maximum. It is a floating-point floor, not a
statistical test, and it cannot detect residual information already lost
when the signal was stored.
"""


def rank_tolerance(shape, spectral_norm) -> float:
    """Singular values at or below this are treated as zero."""
    return max(shape) * np.finfo(float).eps * spectral_norm


def conditioned_basis(matrix) -> tuple[np.ndarray, np.ndarray]:
    """Left singular vectors above the rank tolerance, and ``s_max / s_i`` each."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape[1] == 0:
        return np.empty((matrix.shape[0], 0)), np.empty(0)
    u, s, _ = np.linalg.svd(matrix, full_matrices=False)
    keep = s > rank_tolerance(matrix.shape, s[0])
    return u[:, keep], s[0] / s[keep]


def orthonormal_basis(matrix) -> np.ndarray:
    """Left singular vectors of ``matrix`` above the rank tolerance."""
    return conditioned_basis(matrix)[0]


def residualise(signals, basis) -> np.ndarray:
    """Signals minus their projection on an orthonormal ``basis``."""
    return signals - basis @ (basis.T @ signals)


def _error_scale2(signals, coordinates, amplification):
    """Squared norm of the signals with basis coordinates amplified."""
    excess = (amplification**2 - 1.0)[:, None] * coordinates**2
    return np.sum(signals**2, axis=0) + excess.sum(axis=0)


def project_out(signals, basis, amplification) -> tuple[np.ndarray, np.ndarray]:
    """Residual off ``basis`` and the squared-norm floor of its rounding error.

    Residual energy at or below the floor is indistinguishable from the
    rounding of the projection; see ``PROJECTION_ERROR_FACTOR``.
    """
    coordinates = basis.T @ signals
    relative = PROJECTION_ERROR_FACTOR * len(signals) * np.finfo(float).eps
    scale2 = _error_scale2(signals, coordinates, amplification)
    return signals - basis @ coordinates, relative**2 * scale2
