"""Per-group sufficient statistics for scalar REML."""

from dataclasses import dataclass

import numpy as np

from boldtailor._arrays import copy_array_fields_in_place
from boldtailor._numerics import (
    conditioned_basis,
    project_out,
    rank_tolerance,
    residualise,
)
from boldtailor.ar1 import whiten


@dataclass(frozen=True)
class FixedProjection:
    """Whitened signals with the whitened fixed design projected out.

    ``energy`` at or below ``rounding_floor`` is projection rounding; at or
    below ``uncertainty_floor`` it cannot be told apart from the error of the
    computed fixed span.
    """

    basis: np.ndarray
    signals: np.ndarray
    energy: np.ndarray
    rounding_floor: np.ndarray
    uncertainty_floor: np.ndarray
    dof: int
    rho: float

    def __post_init__(self):
        copy_array_fields_in_place(self, ("basis", "signals", "energy"))
        copy_array_fields_in_place(self, ("rounding_floor", "uncertainty_floor"))


def project_fixed(signals, fixed_design, rho: float = 0.0) -> FixedProjection:
    """Whiten by AR(1) and remove the span of the fixed design."""
    y = whiten(signals, rho)
    basis, amplification = conditioned_basis(whiten(fixed_design, rho))
    dof = y.shape[0] - basis.shape[1]
    if dof <= 0:
        raise ValueError("positive restricted degrees of freedom required")
    residual, rounding, uncertainty = project_out(y, basis, amplification)
    return FixedProjection(
        basis=basis,
        signals=residual,
        energy=np.sum(residual**2, axis=0),
        rounding_floor=rounding,
        uncertainty_floor=uncertainty,
        dof=dof,
        rho=float(rho),
    )


def reproject_signals(projection: FixedProjection, signals) -> FixedProjection:
    """Whiten and project ``signals`` onto an existing fixed-design basis."""
    residual = residualise(whiten(signals, projection.rho), projection.basis)
    zeros = np.zeros(residual.shape[1])
    return FixedProjection(
        basis=projection.basis,
        signals=residual,
        energy=np.sum(residual**2, axis=0),
        rounding_floor=zeros,
        uncertainty_floor=zeros,
        dof=projection.dof,
        rho=projection.rho,
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
        copy_array_fields_in_place(
            self, ("singular2", "coordinates2", "residual_ss", "energy")
        )


def _smaller_gram(matrix):
    """Z'Z or ZZ', whichever is smaller; both share the nonzero eigenvalues."""
    rows, columns = matrix.shape
    return matrix.T @ matrix if columns <= rows else matrix @ matrix.T


def _spectral_norm(matrix):
    """Largest singular value from the smaller Gram matrix, without a second SVD."""
    peak = np.abs(matrix).max()
    if peak == 0.0:
        return 0.0
    return peak * np.sqrt(
        max(np.linalg.eigvalsh(_smaller_gram(matrix / peak))[-1], 0.0)
    )


def _projected_random(projection, random_design):
    z = whiten(random_design, projection.rho)
    return residualise(z, projection.basis), _spectral_norm(z)


def _random_directions(projection, random_design):
    n = projection.signals.shape[0]
    if random_design is None or np.shape(random_design)[1] == 0:
        return np.empty((n, 0)), np.empty(0)
    projected, norm = _projected_random(projection, np.asarray(random_design, float))
    u, s, _ = np.linalg.svd(projected, full_matrices=False)
    keep = s > rank_tolerance(projected.shape, norm)
    return u[:, keep], s[keep]


def _orthogonal_energy(projection, directions, coordinates):
    """Energy outside the retained random span, without subtracting energies.

    At full restricted rank the complement has dimension zero. Otherwise
    forming the residual preserves small orthogonal signals that disappear
    in ``||y||² - ||U'y||²``. Its rounding error is at the amplitude scale.
    """
    if directions.shape[1] == projection.dof:
        return np.zeros(projection.signals.shape[1])
    residual = projection.signals - directions @ coordinates
    return np.sum(residual**2, axis=0)


def reml_terms(
    projection: FixedProjection, random_design: np.ndarray | None
) -> RemlTerms:
    """Reduce a projected group to singular values and squared coordinates."""
    u, s = _random_directions(projection, random_design)
    coordinates = u.T @ projection.signals
    energy = projection.energy
    return RemlTerms(
        singular2=s**2,
        coordinates2=coordinates**2,
        residual_ss=_orthogonal_energy(projection, u, coordinates),
        dof=projection.dof,
        energy=energy,
    )
