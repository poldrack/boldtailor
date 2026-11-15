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


def _group_quadratic_and_logdet(y, c, z, ratio, rho):
    from scipy.linalg import null_space

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


def _feature(signals, index):
    return [y[:, [index]] for y in signals]


def _dense_feature_fit(signals, fixed, random, rho):
    from scipy.optimize import minimize_scalar

    def objective(log_ratio):
        return dense_criterion(signals, fixed, random, np.exp(log_ratio), rho)[0]

    found = minimize_scalar(
        objective, bounds=(-30.0, 30.0), method="bounded", options={"xatol": 1e-10}
    )
    ratio = float(np.exp(found.x))
    if dense_criterion(signals, fixed, random, 0.0, rho)[0] <= found.fun:
        ratio = 0.0
    q, _, n = _dense_parts(signals, fixed, random, ratio, rho)
    return ratio, float(q[0] / n)


def dense_fit(signals, fixed, random, rho: float = 0.0):
    """Per-feature (ratio, sigma2) by bounded search on log ratio, compared with 0."""
    fits = [
        _dense_feature_fit(_feature(signals, f), fixed, random, rho)
        for f in range(signals[0].shape[1])
    ]
    ratio, sigma2 = (np.array(column) for column in zip(*fits, strict=True))
    return ratio, sigma2


def mixedlm_fit(signals, fixed, random):
    """Single-feature (ratio, sigma2) from statsmodels MixedLM with REML."""
    import warnings

    import scipy.linalg
    from statsmodels.regression.mixed_linear_model import MixedLM, VCSpec

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
    from scipy.linalg import null_space

    k = null_space(c.T)
    projected = k.T @ z
    values = np.linalg.eigvalsh(projected @ projected.T)
    return values[values > tol], k.shape[1]


def _pooled_eigenvalues(fixed, random, tol):
    parts = [
        _projected_eigenvalues(c, z, tol) for c, z in zip(fixed, random, strict=True)
    ]
    return np.concatenate([p[0] for p in parts]), sum(p[1] for p in parts)


def _restricted_null_objective(mu, n, w2, r):
    """f(log lam) for one null draw; plain floats keep the per-draw search fast."""
    import math

    def f(log_lam):
        lam, q, logdet = math.exp(log_lam), r, 0.0
        for m, w in zip(mu, w2, strict=True):
            q += w / (1.0 + lam * m)
            logdet += math.log1p(lam * m)
        return n * math.log(q / n) + logdet

    return f


def _null_draw_statistic(mu, n, w2, r):
    import math

    from scipy.optimize import minimize_scalar

    f0 = n * math.log((r + sum(w2)) / n)
    objective = _restricted_null_objective(mu, n, w2, r)
    found = minimize_scalar(objective, bounds=(-30.0, 30.0), method="bounded")
    return f0 - min(f0, found.fun)


def null_lrt_reference(fixed, random, n_draws, seed, tol=1e-10) -> np.ndarray:
    """Finite-sample null of the restricted LRT (Crainiceanu & Ruppert 2004)."""
    mu, n = _pooled_eigenvalues(fixed, random, tol)
    rng = np.random.default_rng(seed)
    w2 = (rng.normal(size=(n_draws, mu.size)) ** 2).tolist()
    r = rng.chisquare(n - mu.size, size=n_draws).tolist()
    mu = mu.tolist()
    return np.array(
        [_null_draw_statistic(mu, n, w, q) for w, q in zip(w2, r, strict=True)]
    )
