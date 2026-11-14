"""Profiled restricted likelihood for scalar REML."""

from collections.abc import Sequence

import numpy as np

from boldtailor._reml_terms import RemlTerms


def _scaled(ratios, singular2):
    """ratio * s2 with shape (..., k, features)."""
    return ratios[..., None, :] * singular2[:, None]


def _shrunk(ratios, singular2):
    """1 + ratio * s2 with shape (..., k, features)."""
    return 1.0 + _scaled(ratios, singular2)


def _group_quadratic(ratios, terms: RemlTerms):
    shrunk = terms.coordinates2 / _shrunk(ratios, terms.singular2)
    return terms.residual_ss + shrunk.sum(axis=-2)


def _group_logdet(ratios, terms: RemlTerms):
    return np.log1p(_scaled(ratios, terms.singular2)).sum(axis=-2)


def _total_dof(terms: Sequence[RemlTerms]) -> int:
    return sum(t.dof for t in terms)


def profiled_variance(ratios, terms: Sequence[RemlTerms]) -> np.ndarray:
    """Profiled noise variance Q(lambda) / N, shaped like ``ratios``."""
    ratios = np.asarray(ratios, dtype=float)
    q = sum(_group_quadratic(ratios, t) for t in terms)
    return q / _total_dof(terms)


def criterion(ratios, terms: Sequence[RemlTerms]) -> np.ndarray:
    """Profiled REML criterion N log(Q/N) + sum log(1 + lambda s2)."""
    ratios = np.asarray(ratios, dtype=float)
    logdet = sum(_group_logdet(ratios, t) for t in terms)
    return _total_dof(terms) * np.log(profiled_variance(ratios, terms)) + logdet


def _group_quadratic_slope(ratios, terms: RemlTerms):
    """lambda * dQ/dlambda for one group."""
    scaled = _scaled(ratios, terms.singular2)
    return -(terms.coordinates2 * scaled / (1.0 + scaled) ** 2).sum(axis=-2)


def _group_logdet_slope(ratios, terms: RemlTerms):
    scaled = _scaled(ratios, terms.singular2)
    return (scaled / (1.0 + scaled)).sum(axis=-2)


def criterion_slope(ratios, terms: Sequence[RemlTerms]) -> np.ndarray:
    """Derivative of ``criterion`` with respect to log(lambda)."""
    ratios = np.asarray(ratios, dtype=float)
    n = _total_dof(terms)
    q = sum(_group_quadratic(ratios, t) for t in terms)
    q_slope = sum(_group_quadratic_slope(ratios, t) for t in terms)
    logdet_slope = sum(_group_logdet_slope(ratios, t) for t in terms)
    return n * q_slope / q + logdet_slope
