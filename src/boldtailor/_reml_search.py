"""Vectorised search for the REML variance ratio.

A log grid evaluates the criterion and its analytic slope in log(lambda).
Bisection refines the slope root inside every grid cell where the slope turns
from negative to positive, so each local minimum the grid resolves competes,
not only the one nearest the best grid point. The lowest root replaces the
best grid point unless it is worse beyond floating-point rounding, and the
result is compared explicitly with lambda = 0 (ties to zero).

The search is not a certified global minimisation: a minimum and a maximum
that both fall strictly inside one grid cell leave no sign change and are not
refined. The grid spacing never exceeds log(1e16) / (grid_size - 1), about
0.31 in log(lambda) by default, whatever the spread of the singular values.
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
    pooled_value_and_slope,
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


def _search_grid(singular2, size):
    """Log-ratio grid spanning 1e-8 / max(s2) to 1e8 / min(s2).

    ``size`` points cover a single singular value; a wider spectrum adds
    points so that the spacing never exceeds that base step.
    """
    low = np.log(1.0 / _RELATIVE_SPAN / singular2.max())
    high = np.log(_RELATIVE_SPAN / singular2.min())
    step = 2.0 * np.log(_RELATIVE_SPAN) / (size - 1)
    return np.linspace(low, high, max(size, int(np.ceil((high - low) / step)) + 1))


def _log_grid(terms: Sequence[RemlTerms], size: int) -> np.ndarray:
    """The search grid of the pooled singular values."""
    return _search_grid(np.concatenate([t.singular2 for t in terms]), size)


def _chunk_rows(pooled: PooledTerms) -> int:
    """Grid rows per chunk so one (rows, k, features) temporary stays bounded."""
    row_bytes = 8 * pooled.singular2.size * pooled.residual_ss.size
    return max(1, _GRID_CHUNK_BYTES // max(row_bytes, 1))


def _grid_chunks(pooled: PooledTerms, log_grid):
    """Criterion and slope of each chunk of grid rows, for every feature."""
    n_features, rows = pooled.residual_ss.size, _chunk_rows(pooled)
    ratios = np.exp(log_grid)
    for start in range(0, ratios.size, rows):
        part = ratios[start : start + rows]
        yield pooled_value_and_slope(
            np.broadcast_to(part[:, None], (part.size, n_features)), pooled
        )


def _on_grid(pooled: PooledTerms, log_grid):
    """Criterion and slope on the grid, each of shape (grid, features)."""
    values, slopes = zip(*_grid_chunks(pooled, log_grid), strict=True)
    return np.concatenate(values, axis=0), np.concatenate(slopes, axis=0)


def _take(pooled: PooledTerms, features) -> PooledTerms:
    """The pooled statistics of the given feature columns."""
    return PooledTerms(
        singular2=pooled.singular2,
        coordinates2=pooled.coordinates2[:, features],
        residual_ss=pooled.residual_ss[features],
        dof=pooled.dof,
    )


def _bisect_slope(pooled, low, high, iterations):
    """Vectorised bisection for the zero of the slope in log ratio."""
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        rising = pooled_slope(np.exp(middle), pooled) > 0
        low, high = np.where(rising, low, middle), np.where(rising, middle, high)
    return 0.5 * (low + high)


def _crossing_roots(pooled, log_grid, slopes, iterations):
    """Feature, slope root and criterion of every cell where the slope turns up."""
    cell, feature = np.nonzero((slopes[:-1] < 0) & (slopes[1:] > 0))
    crossed = _take(pooled, feature)
    root = _bisect_slope(crossed, log_grid[cell], log_grid[cell + 1], iterations)
    return feature, root, pooled_criterion(np.exp(root), crossed)


def _lowest_root(n_features, feature, root, value):
    """Per feature, its lowest-criterion root; value inf where it has none."""
    lowest = np.full(n_features, np.inf)
    np.minimum.at(lowest, feature, value)
    position = np.full(n_features, np.nan)
    is_lowest = value == lowest[feature]
    position[feature[is_lowest]] = root[is_lowest]
    return position, lowest


def _rounding_slack(value, dof):
    """Criterion rounding noise. On a flat criterion a grid point can sit a few
    ulps below the better-determined slope root, which must still win; at an
    exact rounding tie with lambda = 0 the result may therefore flip to zero."""
    return _ROUNDING_ULPS * np.finfo(float).eps * (np.abs(value) + dof)


def _best_grid_point(values):
    index = np.argmin(values, axis=0)
    return index, np.take_along_axis(values, index[None], axis=0)[0]


def _best_positive(pooled, grid_size, iterations):
    log_grid = _search_grid(pooled.singular2, grid_size)
    values, slopes = _on_grid(pooled, log_grid)
    index, grid_value = _best_grid_point(values)
    roots = _crossing_roots(pooled, log_grid, slopes, iterations)
    root, root_value = _lowest_root(index.size, *roots)
    accept = root_value <= grid_value + _rounding_slack(grid_value, pooled.dof)
    best = np.where(accept, root, log_grid[index])
    # The grid's upper edge is a truncation, not an optimum: flag it.
    on_edge = (index == log_grid.size - 1) & ~accept
    return np.exp(best), np.where(accept, root_value, grid_value), on_edge


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
