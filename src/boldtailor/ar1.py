"""AR(1) whitening and bias-corrected noise autocorrelation estimates."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from boldtailor._arrays import own_fields, readonly_array
from boldtailor._ar1_moments import (
    correct_lag_ratio,
    pool_rho,
    residual_lag_ratio,
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


def _group_estimates(signals, design):
    raw = residual_lag_ratio(signals, design)
    return raw, correct_lag_ratio(raw, design)


def estimate_rho(
    signals: Sequence[np.ndarray], reference: Sequence[np.ndarray]
) -> RhoEstimate:
    """Estimate AR(1) rho per group after residualising on its reference design."""
    if len(signals) != len(reference):
        raise ValueError("signals and reference must have one array per group")
    pairs = [_group_estimates(s, d) for s, d in zip(signals, reference, strict=True)]
    raw = readonly_array([p[0] for p in pairs])
    values = readonly_array([p[1] for p in pairs])
    return RhoEstimate(values=values, raw=raw, pooled=pool_rho(values))
