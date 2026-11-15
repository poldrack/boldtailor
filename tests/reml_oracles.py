"""Dense reference constructions for REML tests; imports nothing from boldtailor."""

import warnings

import numpy as np
import scipy.linalg
from scipy.linalg import null_space
from scipy.optimize import minimize_scalar
from statsmodels.regression.mixed_linear_model import MixedLM, VCSpec


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


def _group_quadratic_and_logdet(y, c, z, ratio, rho):
    k = null_space(c.T)
    cov = ar1_correlation(len(y), rho) + ratio * z @ z.T
    v = k.T @ cov @ k
    projected = k.T @ y
    quad = np.sum(projected * np.linalg.solve(v, projected), axis=0)
    return quad, np.linalg.slogdet(v)[1], k.shape[1]


def _dense_parts(signals, fixed, random, ratio, rho):
    q, logdet, n = 0.0, 0.0, 0
    for y, c, z in zip(signals, fixed, random, strict=True):
        group_q, group_logdet, group_n = _group_quadratic_and_logdet(
            y, c, z, ratio, rho
        )
        q, logdet, n = q + group_q, logdet + group_logdet, n + group_n
    return q, logdet, n


def dense_criterion(signals, fixed, random, ratio: float, rho: float = 0.0):
    """Profiled restricted criterion from dense covariances on unwhitened data."""
    q, logdet, n = _dense_parts(signals, fixed, random, ratio, rho)
    return n * np.log(q / n) + logdet


def dense_profiled_variance(signals, fixed, random, ratio: float, rho: float = 0.0):
    """Q(ratio) / N from dense covariances on unwhitened data."""
    q, _, n = _dense_parts(signals, fixed, random, ratio, rho)
    return q / n


def _feature(signals, index):
    return [y[:, [index]] for y in signals]


_FIT_GRID = np.linspace(-30.0, 30.0, 241)


def _local_minima(values):
    """Indices of grid points no higher than both neighbours."""
    padded = np.concatenate([[np.inf], values, [np.inf]])
    return np.flatnonzero((values <= padded[:-2]) & (values <= padded[2:]))


def _polish(objective, index):
    """Bounded local search between the neighbours of one grid minimum."""
    low = _FIT_GRID[max(index - 1, 0)]
    high = _FIT_GRID[min(index + 1, _FIT_GRID.size - 1)]
    return minimize_scalar(
        objective, bounds=(low, high), method="bounded", options={"xatol": 1e-10}
    )


def _dense_feature_fit(signals, fixed, random, rho):
    """Scan log ratio, polish every local minimum of the scan, keep the best.

    A single bounded search is only local; the scan makes the reference
    independent of which basin such a search happens to enter.
    """

    def objective(log_ratio):
        return dense_criterion(signals, fixed, random, np.exp(log_ratio), rho)[0]

    scan = np.array([objective(x) for x in _FIT_GRID])
    found = min(
        (_polish(objective, i) for i in _local_minima(scan)), key=lambda f: f.fun
    )
    ratio = float(np.exp(found.x))
    if dense_criterion(signals, fixed, random, 0.0, rho)[0] <= found.fun:
        ratio = 0.0
    q, _, n = _dense_parts(signals, fixed, random, ratio, rho)
    return ratio, float(q[0] / n)


def dense_fit(signals, fixed, random, rho: float = 0.0):
    """Per-feature (ratio, sigma2) by scan and polish on log ratio, compared with 0."""
    fits = [
        _dense_feature_fit(_feature(signals, f), fixed, random, rho)
        for f in range(signals[0].shape[1])
    ]
    ratio, sigma2 = (np.array(column) for column in zip(*fits, strict=True))
    return ratio, sigma2


def mixedlm_fit(signals, fixed, random):
    """Single-feature (ratio, sigma2) from statsmodels MixedLM with REML."""
    groups = np.repeat(np.arange(len(signals)), [len(y) for y in signals])
    names = [[f"z{g}_{j}" for j in range(z.shape[1])] for g, z in enumerate(random)]
    vc = VCSpec(["z"], [names], [list(random)])
    model = MixedLM(
        np.concatenate(signals), scipy.linalg.block_diag(*fixed), groups, exog_vc=vc
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(reml=True, method="bfgs", gtol=1e-10)
    return result.vcomp[0] / result.scale, result.scale


def _projected_eigenvalues(c, z, tol):
    k = null_space(c.T)
    projected = k.T @ z
    values = np.linalg.eigvalsh(projected @ projected.T)
    return values[values > tol], k.shape[1]


def _pooled_eigenvalues(fixed, random, tol):
    parts = [
        _projected_eigenvalues(c, z, tol) for c, z in zip(fixed, random, strict=True)
    ]
    return np.concatenate([p[0] for p in parts]), sum(p[1] for p in parts)


_NULL_GRID = np.linspace(-30.0, 30.0, 241)
_GOLDEN_ITERATIONS = 60
_INVERSE_GOLDEN = (np.sqrt(5.0) - 1.0) / 2.0


def _null_objective(log_lam, mu, n, w2, r):
    """Restricted criterion of every null draw at its own log ratio."""
    scaled = np.exp(log_lam)[:, None] * mu
    q = r + np.sum(w2 / (1.0 + scaled), axis=1)
    return n * np.log(q / n) + np.sum(np.log1p(scaled), axis=1)


def _null_grid_minimum(mu, n, w2, r):
    """Index of the best grid point per draw."""
    values = np.stack(
        [_null_objective(np.full(r.size, g), mu, n, w2, r) for g in _NULL_GRID]
    )
    return np.argmin(values, axis=0), values.min(axis=0)


def _null_golden(mu, n, w2, r, low, high):
    """Golden-section minimum per draw on [low, high], vectorised over draws."""

    def f(x):
        return _null_objective(x, mu, n, w2, r)

    c = high - _INVERSE_GOLDEN * (high - low)
    d = low + _INVERSE_GOLDEN * (high - low)
    fc, fd = f(c), f(d)
    for _ in range(_GOLDEN_ITERATIONS):
        left = fc < fd
        high, low = np.where(left, d, high), np.where(left, low, c)
        c, d = high - _INVERSE_GOLDEN * (high - low), low + _INVERSE_GOLDEN * (
            high - low
        )
        fc, fd = f(c), f(d)
    return np.minimum(fc, fd)


def _null_statistics(mu, n, w2, r):
    """Restricted LRT per draw: criterion at zero minus the minimum over lambda."""
    f0 = n * np.log((r + w2.sum(axis=1)) / n)
    index, grid_best = _null_grid_minimum(mu, n, w2, r)
    last = _NULL_GRID.size - 1
    low = _NULL_GRID[np.maximum(index - 1, 0)]
    high = _NULL_GRID[np.minimum(index + 1, last)]
    best = np.minimum(grid_best, _null_golden(mu, n, w2, r, low, high))
    return f0 - np.minimum(f0, best)


def null_lrt_reference(fixed, random, n_draws, seed, tol=1e-10) -> np.ndarray:
    """Finite-sample null of the restricted LRT (Crainiceanu & Ruppert 2004).

    Grid over log ratio, then golden section per draw, vectorised over draws.
    """
    mu, n = _pooled_eigenvalues(fixed, random, tol)
    rng = np.random.default_rng(seed)
    w2 = rng.normal(size=(n_draws, mu.size)) ** 2
    r = rng.chisquare(n - mu.size, size=n_draws)
    return _null_statistics(mu, n, w2, r)
