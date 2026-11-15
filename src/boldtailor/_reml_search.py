"""Vectorised search for the REML variance ratio.

A log grid locates the best grid point; its neighbours bracket the root of the
analytic slope in log(lambda), which bisection refines. The root replaces the
grid point unless it is worse beyond floating-point rounding, and the result
is compared explicitly with lambda = 0 (ties to zero).
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from boldtailor._arrays import own_fields
from boldtailor._reml_criterion import (
    PooledTerms,
    pool_terms,
    pooled_criterion,
    pooled_slope,
)
from boldtailor._reml_terms import RemlTerms

_RELATIVE_SPAN = 1e8
_ROUNDING_ULPS = 64
_GRID_CHUNK_BYTES = 8 * 2**20


@dataclass(frozen=True)
class RatioOptimum:
    """Per-feature variance-ratio optimum and where it sits."""

    ratio: np.ndarray
    criterion: np.ndarray
    at_zero: np.ndarray
    at_upper: np.ndarray

    def __post_init__(self):
        own_fields(self, ("ratio", "criterion"))
        own_fields(self, ("at_zero", "at_upper"), dtype=bool)


def _singular2_grid(singular2, size):
    return np.linspace(
        np.log(1.0 / _RELATIVE_SPAN / singular2.max()),
        np.log(_RELATIVE_SPAN / singular2.min()),
        size,
    )


def _log_grid(terms: Sequence[RemlTerms], size: int) -> np.ndarray:
    """Log-ratio grid spanning 1e-8 / max(s2) to 1e8 / min(s2)."""
    return _singular2_grid(np.concatenate([t.singular2 for t in terms]), size)


def _chunk_rows(pooled: PooledTerms) -> int:
    """Grid rows per chunk so one (rows, k, features) temporary stays bounded."""
    row_bytes = 8 * pooled.singular2.size * pooled.residual_ss.size
    return max(1, _GRID_CHUNK_BYTES // max(row_bytes, 1))


def _grid_values(pooled: PooledTerms, log_grid) -> np.ndarray:
    """Criterion on every grid point for every feature, shape (grid, features)."""
    n_features, rows = pooled.residual_ss.size, _chunk_rows(pooled)
    ratios = np.exp(log_grid)
    chunks = [
        pooled_criterion(
            np.broadcast_to(part[:, None], (part.size, n_features)), pooled
        )
        for part in (ratios[i : i + rows] for i in range(0, ratios.size, rows))
    ]
    return np.concatenate(chunks, axis=0)


def _bracket(log_grid, best_index):
    """The best grid point's neighbours, clipped to the grid."""
    last = log_grid.size - 1
    return (
        log_grid[np.maximum(best_index - 1, 0)],
        log_grid[np.minimum(best_index + 1, last)],
    )


def _sign_change(pooled, low, high):
    """True where the slope rises from negative at ``low`` to positive at ``high``."""
    falling = pooled_slope(np.exp(low), pooled) < 0
    return falling & (pooled_slope(np.exp(high), pooled) > 0)


def _bisect_slope(pooled, low, high, iterations):
    """Vectorised bisection for the zero of the slope in log ratio."""
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        rising = pooled_slope(np.exp(middle), pooled) > 0
        low, high = np.where(rising, low, middle), np.where(rising, middle, high)
    return 0.5 * (low + high)


def _rounding_slack(value, dof):
    """Criterion rounding noise. On a flat criterion a grid point can sit a few
    ulps below the better-determined slope root, which must still win; at an
    exact rounding tie with lambda = 0 the result may therefore flip to zero."""
    return _ROUNDING_ULPS * np.finfo(float).eps * (np.abs(value) + dof)


def _refine(pooled, log_grid, index, start_value, iterations):
    """The slope root inside the neighbour bracket, where it is no worse."""
    low, high = _bracket(log_grid, index)
    root = _bisect_slope(pooled, low, high, iterations)
    root_value = pooled_criterion(np.exp(root), pooled)
    slack = _rounding_slack(start_value, pooled.dof)
    accept = _sign_change(pooled, low, high) & (root_value <= start_value + slack)
    start = log_grid[index]
    return np.where(accept, root, start), np.where(accept, root_value, start_value)


def _best_positive(pooled, grid_size, iterations):
    log_grid = _singular2_grid(pooled.singular2, grid_size)
    values = _grid_values(pooled, log_grid)
    index = np.argmin(values, axis=0)
    start_value = np.take_along_axis(values, index[None], axis=0)[0]
    best, value = _refine(pooled, log_grid, index, start_value, iterations)
    # The grid's upper edge is a truncation, not an optimum: flag it.
    return np.exp(best), value, index == log_grid.size - 1


def _zero_optimum(pooled, n_features) -> RatioOptimum:
    zero = np.zeros(n_features)
    return RatioOptimum(
        ratio=zero,
        criterion=pooled_criterion(zero, pooled),
        at_zero=np.ones(n_features, dtype=bool),
        at_upper=np.zeros(n_features, dtype=bool),
    )


def minimize_ratio(
    terms: Sequence[RemlTerms], *, grid_size: int = 121, iterations: int = 48
) -> RatioOptimum:
    """Minimise the profiled criterion over lambda >= 0 for every feature.

    ``iterations`` is the number of slope bisections inside the bracket.
    """
    pooled = pool_terms(terms)
    n_features = pooled.residual_ss.size
    if pooled.singular2.size == 0:
        return _zero_optimum(pooled, n_features)
    ratio, value, on_edge = _best_positive(pooled, grid_size, iterations)
    at_zero_value = pooled_criterion(np.zeros(n_features), pooled)
    at_zero = at_zero_value <= value
    return RatioOptimum(
        ratio=np.where(at_zero, 0.0, ratio),
        criterion=np.where(at_zero, at_zero_value, value),
        at_zero=at_zero,
        at_upper=on_edge & ~at_zero,
    )
