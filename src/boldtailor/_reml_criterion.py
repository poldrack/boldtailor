"""Profiled restricted likelihood for scalar REML.

The criterion depends on the groups only through their pooled statistics:
all singular values, the stacked squared coordinates, the summed residual sum
of squares and the summed dof. Pooling once makes every evaluation a single
vectorised pass, whatever the number of groups.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from boldtailor._reml_terms import RemlTerms


@dataclass(frozen=True)
class PooledTerms:
    """Sufficient statistics of all groups together."""

    singular2: np.ndarray
    coordinates2: np.ndarray
    residual_ss: np.ndarray
    dof: int


def pool_terms(terms: Sequence[RemlTerms]) -> PooledTerms:
    """Concatenate singular values and coordinates; sum residuals and dof."""
    return PooledTerms(
        singular2=np.concatenate([t.singular2 for t in terms]),
        coordinates2=np.concatenate([t.coordinates2 for t in terms], axis=0),
        residual_ss=np.sum([t.residual_ss for t in terms], axis=0),
        dof=sum(t.dof for t in terms),
    )


def _scaled(ratios, singular2):
    """ratio * s2 with shape (..., k, features)."""
    return ratios[..., None, :] * singular2[:, None]


def pooled_quadratic(ratios, pooled: PooledTerms) -> np.ndarray:
    """Q(lambda), the whitened residual quadratic form, shaped like ``ratios``."""
    shrunk = 1.0 + _scaled(ratios, pooled.singular2)
    return pooled.residual_ss + (pooled.coordinates2 / shrunk).sum(axis=-2)


def pooled_criterion(ratios, pooled: PooledTerms) -> np.ndarray:
    """N log(Q/N) + sum log(1 + lambda s2) from pooled statistics."""
    scaled = _scaled(ratios, pooled.singular2)
    logdet = np.log1p(scaled).sum(axis=-2)
    scaled += 1.0
    q = pooled.residual_ss + (pooled.coordinates2 / scaled).sum(axis=-2)
    return pooled.dof * np.log(q / pooled.dof) + logdet


def pooled_slope(ratios, pooled: PooledTerms) -> np.ndarray:
    """Derivative of the criterion in log(lambda), from pooled statistics."""
    scaled = _scaled(ratios, pooled.singular2)
    shrunk = 1.0 + scaled
    fraction = scaled / shrunk
    weighted = pooled.coordinates2 / shrunk
    q = pooled.residual_ss + weighted.sum(axis=-2)
    q_slope = -(weighted * fraction).sum(axis=-2)
    return pooled.dof * q_slope / q + fraction.sum(axis=-2)


def profiled_variance(ratios, terms: Sequence[RemlTerms]) -> np.ndarray:
    """Profiled noise variance Q(lambda) / N, shaped like ``ratios``."""
    pooled = pool_terms(terms)
    return pooled_quadratic(np.asarray(ratios, dtype=float), pooled) / pooled.dof


def criterion(ratios, terms: Sequence[RemlTerms]) -> np.ndarray:
    """Profiled REML criterion N log(Q/N) + sum log(1 + lambda s2)."""
    return pooled_criterion(np.asarray(ratios, dtype=float), pool_terms(terms))


def criterion_slope(ratios, terms: Sequence[RemlTerms]) -> np.ndarray:
    """Derivative of ``criterion`` with respect to log(lambda)."""
    return pooled_slope(np.asarray(ratios, dtype=float), pool_terms(terms))
