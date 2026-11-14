"""Vectorised search for the REML variance ratio."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from boldtailor._arrays import own_fields
from boldtailor._reml_criterion import criterion, criterion_slope
from boldtailor._reml_terms import RemlTerms

_INVERSE_GOLDEN = (np.sqrt(5.0) - 1.0) / 2.0
_RELATIVE_SPAN = 1e8
_BISECTIONS = 64
_ROUNDING_ULPS = 64


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


def _log_grid(terms: Sequence[RemlTerms], size: int) -> np.ndarray:
    s2 = np.concatenate([t.singular2 for t in terms])
    return np.linspace(
        np.log(1.0 / _RELATIVE_SPAN / s2.max()),
        np.log(_RELATIVE_SPAN / s2.min()),
        size,
    )


def _has_directions(terms: Sequence[RemlTerms]) -> bool:
    return any(t.singular2.size for t in terms)


def _grid_values(terms, log_grid, n_features):
    ratios = np.broadcast_to(np.exp(log_grid)[:, None], (log_grid.size, n_features))
    return criterion(ratios, terms)


def _bracket(log_grid, best_index):
    last = log_grid.size - 1
    return log_grid[np.maximum(best_index - 1, 0)], log_grid[
        np.minimum(best_index + 1, last)
    ]


class _Tracker:
    """Evaluates the criterion in log-ratio and remembers the best point."""

    def __init__(self, terms, start, start_value):
        self.terms, self.best, self.value = terms, start.copy(), start_value.copy()

    def __call__(self, log_ratio):
        value = criterion(np.exp(log_ratio), self.terms)
        better = value < self.value
        self.best = np.where(better, log_ratio, self.best)
        self.value = np.where(better, value, self.value)
        return value


def _inner_points(low, high):
    width = high - low
    return high - _INVERSE_GOLDEN * width, low + _INVERSE_GOLDEN * width


def _golden_section(terms, low, high, start, start_value, iterations):
    """Minimise the criterion in log-ratio on [low, high], all features at once."""
    track = _Tracker(terms, start, start_value)
    c, d = _inner_points(low, high)
    fc, fd = track(c), track(d)
    for _ in range(iterations):
        left = fc < fd
        high, low = np.where(left, d, high), np.where(left, low, c)
        c_new, d_new = _inner_points(low, high)
        fresh = track(np.where(left, c_new, d_new))
        c, d = np.where(left, c_new, d), np.where(left, c, d_new)
        fc, fd = np.where(left, fresh, fd), np.where(left, fc, fresh)
    return track.best, track.value, low, high


def _slope_brackets(terms, low, high):
    """True where the slope rises from negative at ``low`` to positive at ``high``."""
    falling = criterion_slope(np.exp(low), terms) < 0
    return falling & (criterion_slope(np.exp(high), terms) > 0)


def _root_bracket(terms, golden, grid):
    """The golden bracket where it shows a sign change, else the grid bracket."""
    use_golden = _slope_brackets(terms, *golden)
    low = np.where(use_golden, golden[0], grid[0])
    high = np.where(use_golden, golden[1], grid[1])
    return low, high, use_golden | _slope_brackets(terms, low, high)


def _bisect_slope(terms, low, high):
    """Vectorised bisection for the zero of the slope in log ratio."""
    for _ in range(_BISECTIONS):
        middle = 0.5 * (low + high)
        rising = criterion_slope(np.exp(middle), terms) > 0
        low, high = np.where(rising, low, middle), np.where(rising, middle, high)
    return 0.5 * (low + high)


def _rounding_slack(value, terms):
    """Criterion rounding noise: on a flat criterion the golden point can sit a
    few ulps below the better-determined score root, which must still win."""
    return _ROUNDING_ULPS * np.finfo(float).eps * (np.abs(value) + _dof(terms))


def _dof(terms):
    return sum(t.dof for t in terms)


def _polish(terms, best, value, golden, grid):
    """Replace the incumbent by the slope root where no worse beyond rounding."""
    low, high, bracketed = _root_bracket(terms, golden, grid)
    root = _bisect_slope(terms, low, high)
    root_value = criterion(np.exp(root), terms)
    accept = bracketed & (root_value <= value + _rounding_slack(value, terms))
    return np.where(accept, root, best), np.where(accept, root_value, value)


def _zero_optimum(terms, n_features) -> RatioOptimum:
    zero = np.zeros(n_features)
    return RatioOptimum(
        ratio=zero,
        criterion=criterion(zero, terms),
        at_zero=np.ones(n_features, dtype=bool),
        at_upper=np.zeros(n_features, dtype=bool),
    )


def _best_positive(terms, grid_size, iterations, n_features):
    log_grid = _log_grid(terms, grid_size)
    values = _grid_values(terms, log_grid, n_features)
    index = np.argmin(values, axis=0)
    start = log_grid[index]
    start_value = np.take_along_axis(values, index[None], axis=0)[0]
    grid = _bracket(log_grid, index)
    best, value, *golden = _golden_section(terms, *grid, start, start_value, iterations)
    best, value = _polish(terms, best, value, golden, grid)
    return np.exp(best), value, index == log_grid.size - 1


def minimize_ratio(
    terms: Sequence[RemlTerms], *, grid_size: int = 121, iterations: int = 60
) -> RatioOptimum:
    """Minimise the profiled criterion over lambda >= 0 for every feature."""
    n_features = terms[0].residual_ss.shape[0]
    if not _has_directions(terms):
        return _zero_optimum(terms, n_features)
    ratio, value, on_edge = _best_positive(terms, grid_size, iterations, n_features)
    at_zero_value = criterion(np.zeros(n_features), terms)
    at_zero = at_zero_value <= value
    return RatioOptimum(
        ratio=np.where(at_zero, 0.0, ratio),
        criterion=np.where(at_zero, at_zero_value, value),
        at_zero=at_zero,
        at_upper=on_edge & ~at_zero,
    )
