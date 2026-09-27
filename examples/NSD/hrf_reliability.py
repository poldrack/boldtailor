"""Compare selected HRF shapes across time, separately at each grayordinate."""

import numpy as np

CORRELATION_NAMES = ("odd_even_r", "odd_canonical_r", "even_canonical_r")


def _indices(values, size):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("HRF indices must be one-dimensional")
    missing = np.isnan(values) | (values == -1)
    selected = values[~missing]
    if (
        not np.isfinite(selected).all()
        or np.any(selected < 0)
        or np.any(selected >= size)
        or np.any(selected != np.floor(selected))
    ):
        raise ValueError("HRF indices must identify a library entry, or be NaN/-1")
    return np.where(missing, -1, values).astype(int)


def hrf_curve_correlations(library, odd_ids, even_ids):
    """Pearson r on the common stored time grid, including padded tails.

    Rows are odd/even, odd/canonical, and even/canonical. Each pair retains
    NaN if either selection is missing or either curve is constant. Curves
    are not shifted in time. Correlation ignores amplitude scale and offset.
    """
    odd = _indices(odd_ids, len(library.curves))
    even = _indices(even_ids, len(library.curves))
    if odd.shape != even.shape:
        raise ValueError("Odd/even HRF indices must have the same shape")
    centered = library.curves - library.curves.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    unit = np.divide(
        centered, norms, out=np.full_like(centered, np.nan), where=norms > 0
    )
    lookup = np.clip(unit @ unit.T, -1.0, 1.0)
    canonical = np.zeros_like(odd)
    result = np.full((3, len(odd)), np.nan)
    for row, (a, b) in enumerate(((odd, even), (odd, canonical), (even, canonical))):
        valid = (a >= 0) & (b >= 0)
        result[row, valid] = lookup[a[valid], b[valid]]
    return result
