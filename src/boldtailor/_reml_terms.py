"""Per-group sufficient statistics for scalar REML."""

from dataclasses import dataclass

import numpy as np

from boldtailor._arrays import own_fields
from boldtailor.ar1 import whiten


def _rank_cutoff(matrix, spectral_norm):
    return max(matrix.shape) * np.finfo(float).eps * spectral_norm


def _orthonormal_basis(matrix) -> np.ndarray:
    """Left singular vectors above the rank tolerance."""
    if matrix.shape[1] == 0:
        return np.empty((matrix.shape[0], 0))
    u, s, _ = np.linalg.svd(matrix, full_matrices=False)
    return u[:, s > _rank_cutoff(matrix, s[0])]


@dataclass(frozen=True)
class FixedProjection:
    """Whitened signals with the whitened fixed design projected out."""

    basis: np.ndarray
    signals: np.ndarray
    energy: np.ndarray
    raw_energy: np.ndarray
    dof: int
    rho: float

    def __post_init__(self):
        own_fields(self, ("basis", "signals", "energy", "raw_energy"))


def project_fixed(signals, fixed_design, rho: float = 0.0) -> FixedProjection:
    """Whiten by AR(1) and remove the span of the fixed design."""
    y = whiten(signals, rho)
    basis = _orthonormal_basis(whiten(fixed_design, rho))
    dof = y.shape[0] - basis.shape[1]
    if dof <= 0:
        raise ValueError("positive restricted degrees of freedom required")
    residual = y - basis @ (basis.T @ y)
    return FixedProjection(
        basis=basis,
        signals=residual,
        energy=np.sum(residual**2, axis=0),
        raw_energy=np.sum(y**2, axis=0),
        dof=dof,
        rho=float(rho),
    )


@dataclass(frozen=True)
class RemlTerms:
    """Sufficient statistics of one group for the REML criterion."""

    singular2: np.ndarray
    coordinates2: np.ndarray
    residual_ss: np.ndarray
    dof: int
    energy: np.ndarray

    def __post_init__(self):
        own_fields(self, ("singular2", "coordinates2", "residual_ss", "energy"))


def _spectral_norm(matrix):
    """Largest singular value from the small Gram matrix, without a second SVD."""
    return np.sqrt(max(np.linalg.eigvalsh(matrix.T @ matrix)[-1], 0.0))


def _projected_random(projection, random_design):
    z = whiten(random_design, projection.rho)
    projected = z - projection.basis @ (projection.basis.T @ z)
    return projected, _spectral_norm(z)


def _random_directions(projection, random_design):
    n = projection.signals.shape[0]
    if random_design is None or np.shape(random_design)[1] == 0:
        return np.empty((n, 0)), np.empty(0)
    projected, norm = _projected_random(projection, np.asarray(random_design, float))
    u, s, _ = np.linalg.svd(projected, full_matrices=False)
    keep = s > _rank_cutoff(projected, norm)
    return u[:, keep], s[keep]


def reml_terms(
    projection: FixedProjection, random_design: np.ndarray | None
) -> RemlTerms:
    """Reduce a projected group to singular values and squared coordinates."""
    u, s = _random_directions(projection, random_design)
    coordinates2 = (u.T @ projection.signals) ** 2
    energy = projection.energy
    return RemlTerms(
        singular2=s**2,
        coordinates2=coordinates2,
        residual_ss=np.maximum(energy - coordinates2.sum(axis=0), 0.0),
        dof=projection.dof,
        energy=energy,
    )
