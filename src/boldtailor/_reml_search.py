"""Vectorised search for the REML variance ratio."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from boldtailor._arrays import own_fields
from boldtailor._reml_criterion import criterion
from boldtailor._reml_terms import RemlTerms

_INVERSE_GOLDEN = (np.sqrt(5.0) - 1.0) / 2.0
_RELATIVE_SPAN = 1e8


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
    return track.best, track.value


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
    low, high = _bracket(log_grid, index)
    best, best_value = _golden_section(terms, low, high, start, start_value, iterations)
    return np.exp(best), best_value, index == log_grid.size - 1


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
