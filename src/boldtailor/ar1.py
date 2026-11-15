"""AR(1) whitening and bias-corrected noise autocorrelation estimates."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from boldtailor._arrays import own_fields, readonly_array
from boldtailor._ar1_moments import (
    GRID,
    basis_expected_lag_ratio,
    basis_lag_ratio,
    invert_lag_ratio,
    orthonormal_basis,
    pool_rho,
)


def _check_rho(rho):
    if not np.isfinite(rho) or abs(rho) >= 1:
        raise ValueError(f"rho must be finite with |rho| < 1, got {rho!r}")


def whiten(values: np.ndarray, rho: float) -> np.ndarray:
    """Whiten rows by AR(1) noise so that W R(rho) W' is the identity."""
    _check_rho(rho)
    values = np.asarray(values, dtype=float)
    out = values.copy()
    out[1:] = (values[1:] - rho * values[:-1]) / np.sqrt(1.0 - rho**2)
    return out


@dataclass(frozen=True)
class RhoEstimate:
    """Per-group, per-feature AR(1) estimates and their pooled value."""

    values: np.ndarray
    raw: np.ndarray
    pooled: float

    def __post_init__(self):
        own_fields(self, ("values", "raw"))


@dataclass(frozen=True)
class RhoReference:
    """Signal-independent work of one reference design, reusable across signals.

    ``basis`` is an orthonormal basis ``(scans, rank)`` of the design and
    ``expected`` the expected residual lag-one ratio on the fixed rho grid.
    """

    basis: np.ndarray
    expected: np.ndarray

    def __post_init__(self):
        own_fields(self, ("basis", "expected"))


def _check_2d(values, name, where):
    if np.ndim(values) != 2:
        raise ValueError(f"{name}{where} must be 2-D")


def _check_matrix(values, name, where=""):
    _check_2d(values, name, where)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name}{where} contains non-finite values")


def prepare_reference(design: np.ndarray) -> RhoReference:
    """Precompute what ``estimate_rho`` needs from one reference design."""
    design = np.asarray(design, dtype=float)
    _check_matrix(design, "design")
    basis = orthonormal_basis(design)
    return RhoReference(basis=basis, expected=basis_expected_lag_ratio(basis, GRID))


def _group_reference(signals, reference, index):
    where = f" of group {index}"
    _check_2d(signals, "signals", where)  # non-finite signals give NaN estimates
    if not isinstance(reference, RhoReference):
        _check_matrix(reference, "reference design", where)
        reference = prepare_reference(reference)
    if reference.basis.shape[0] != np.shape(signals)[0]:
        raise ValueError(f"scan counts differ between signals and reference{where}")
    return reference


def _validated(signals, reference):
    if len(signals) != len(reference):
        raise ValueError("signals and reference must have one array per group")
    if len(signals) == 0:
        raise ValueError("at least one group is required")
    pairs = list(zip(signals, reference, strict=True))
    prepared = [_group_reference(s, r, i) for i, (s, r) in enumerate(pairs)]
    if len({np.shape(s)[1] for s in signals}) != 1:
        raise ValueError("feature counts differ across groups")
    return prepared


def _group_estimates(signals, reference):
    raw = basis_lag_ratio(signals, reference.basis)
    return raw, invert_lag_ratio(raw, reference.expected, GRID)


def estimate_rho(
    signals: Sequence[np.ndarray],
    reference: Sequence[np.ndarray | RhoReference],
) -> RhoEstimate:
    """Estimate AR(1) rho per group after residualising on its reference.

    Each reference is a design array ``(scans, columns)`` or a
    ``RhoReference`` from ``prepare_reference``; both give identical results.
    """
    prepared = _validated(signals, reference)
    pairs = [_group_estimates(s, r) for s, r in zip(signals, prepared, strict=True)]
    raw = readonly_array([p[0] for p in pairs])
    values = readonly_array([p[1] for p in pairs])
    return RhoEstimate(values=values, raw=raw, pooled=pool_rho(values))
