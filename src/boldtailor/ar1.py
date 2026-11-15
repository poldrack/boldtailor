"""AR(1) whitening and bias-corrected noise autocorrelation estimates."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from boldtailor._arrays import own_fields, readonly_array
from boldtailor._ar1_moments import (
    GRID,
    RHO_RANGE,
    basis_expected_lag_ratio,
    basis_lag_ratio,
    finite_median,
    invert_lag_ratio,
    pool_rho,
)
from boldtailor._numerics import conditioned_basis


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
    """Per-group, per-feature AR(1) estimates and their pooled value.

    ``pooled`` is the median of the finite ``values`` clipped to
    ``rho_range``. ``pooled_at_bound`` says that it sits on a limit of that
    range, where it is a bound on rho and not a measurement of it.
    """

    values: np.ndarray
    raw: np.ndarray
    pooled: float
    rho_range: tuple[float, float]
    pooled_at_bound: bool

    def __post_init__(self):
        own_fields(self, ("values", "raw"))


@dataclass(frozen=True)
class RhoReference:
    """Signal-independent work of one reference design, reusable across signals.

    ``basis`` is an orthonormal basis ``(scans, rank)`` of the design,
    ``amplification`` the ratio of the design's largest singular value to
    that of each basis vector (it scales the degeneracy floor), and
    ``expected`` the expected residual lag-one ratio on the fixed rho grid.
    """

    basis: np.ndarray
    amplification: np.ndarray
    expected: np.ndarray

    def __post_init__(self):
        own_fields(self, ("basis", "amplification", "expected"))


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
    basis, amplification = conditioned_basis(design)
    expected = basis_expected_lag_ratio(basis, GRID)
    return RhoReference(basis=basis, amplification=amplification, expected=expected)


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


def _checked_range(rho_range) -> tuple[float, float]:
    """``(low, high)`` as floats, inside the correction grid and increasing."""
    if np.shape(rho_range) == (2,):
        low, high = (float(limit) for limit in rho_range)
        if GRID[0] <= low < high <= GRID[-1]:
            return low, high
    limits = f"{GRID[0]} <= low < high <= {GRID[-1]}"
    raise ValueError(f"rho_range must be (low, high) with {limits}, got {rho_range!r}")


def _group_estimates(signals, reference, rho_range):
    raw = basis_lag_ratio(signals, reference.basis, reference.amplification)
    return raw, invert_lag_ratio(raw, reference.expected, GRID, rho_range)


def _pooled(values, rho_range):
    """The pooled value and whether it sits on a limit of ``rho_range``."""
    pooled = pool_rho(values, rho_range)
    estimated = np.isfinite(finite_median(values))
    return pooled, bool(estimated and pooled in rho_range)


def estimate_rho(
    signals: Sequence[np.ndarray],
    reference: Sequence[np.ndarray | RhoReference],
    *,
    rho_range: tuple[float, float] = RHO_RANGE,
) -> RhoEstimate:
    """Estimate AR(1) rho per group after residualising on its reference.

    Each reference is a design array ``(scans, columns)`` or a
    ``RhoReference`` from ``prepare_reference``; both give identical results.

    ``rho_range`` is the admissible range of the pooled value. The pooled
    median is clipped to it, and the bias-correction curve only has to be
    monotone across it. The default lower limit of -0.2 excludes strong
    negative autocorrelation, which is not plausible for fMRI noise; widen
    the range for other data. The default upper limit, 0.95, is the end of
    the correction grid. Per-feature ``values`` are not confined to the range.
    """
    rho_range = _checked_range(rho_range)
    prepared = _validated(signals, reference)
    pairs = [
        _group_estimates(s, r, rho_range)
        for s, r in zip(signals, prepared, strict=True)
    ]
    values = readonly_array([p[1] for p in pairs])
    pooled, at_bound = _pooled(values, rho_range)
    return RhoEstimate(
        values=values,
        raw=readonly_array([p[0] for p in pairs]),
        pooled=pooled,
        rho_range=rho_range,
        pooled_at_bound=at_bound,
    )
