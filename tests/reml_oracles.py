"""Dense reference constructions for REML tests; imports nothing from boldtailor."""

import numpy as np


def ar1_correlation(n: int, rho: float) -> np.ndarray:
    """Dense AR(1) correlation matrix with entries rho ** |i - j|."""
    lags = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    return float(rho) ** lags


def _fixed_design(rng, n, n_fixed):
    extra = rng.normal(size=(n, max(n_fixed - 1, 0)))
    return np.column_stack([np.ones(n), extra])[:, :n_fixed]


def _noise(rng, n, rho, sigma2, n_features):
    factor = np.linalg.cholesky(sigma2 * ar1_correlation(n, rho))
    return factor @ rng.normal(size=(n, n_features))


def _simulate_group(rng, n, n_fixed, n_random, ratio, sigma2, rho, n_features):
    fixed = _fixed_design(rng, n, n_fixed)
    random = rng.normal(size=(n, n_random))
    beta = rng.normal(size=(fixed.shape[1], n_features))
    u = rng.normal(scale=np.sqrt(ratio * sigma2), size=(n_random, n_features))
    noise = _noise(rng, n, rho, sigma2, n_features)
    return fixed @ beta + random @ u + noise, fixed, random


def simulate(
    rng,
    *,
    scans=(60, 60),
    n_fixed=4,
    n_random=3,
    ratio=1.0,
    sigma2=1.0,
    rho=0.0,
    n_features=1,
):
    """Simulate groups sharing designs across features; returns (signals, fixed, random)."""
    groups = [
        _simulate_group(rng, n, n_fixed, n_random, ratio, sigma2, rho, n_features)
        for n in scans
    ]
    return tuple(list(part) for part in zip(*groups, strict=True))
