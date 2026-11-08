"""Vectorized bounded scalar REML on a once-diagonalized task covariance."""

import numpy as np
from experiments.hrf_estimation.reml import _refine


def prepare_scalar(terms):
    if len(terms["names"]) != 1 or np.any(terms["groups"] != 0):
        raise ValueError("scalar REML requires exactly one variance component")
    values, vectors = np.linalg.eigh(terms["gram"])
    return dict(values=np.maximum(values, 0.),
                coordinates2=(vectors.T @ terms["coordinates"])**2,
                energy=terms["energy"], n=terms["n"])


def scalar_objective(ratios, spectral):
    ratios = np.asarray(ratios)
    values = spectral["values"][:, None]
    denominator = 1 + values * ratios
    q = spectral["energy"] - np.sum(
        (ratios / denominator) * spectral["coordinates2"], axis=0)
    q = np.maximum(q, 64*np.finfo(float).eps*spectral["energy"])
    with np.errstate(divide="ignore", invalid="ignore"):
        criterion = spectral["n"] * np.log(q/spectral["n"]) + np.log(denominator).sum(axis=0)
    return criterion, q/spectral["n"]


def _search(spectral, scale, upper, grid_size):
    grid = np.linspace(0., upper, grid_size)
    def objective(unit):
        return scalar_objective(np.expm1(unit)/scale, spectral)[0]
    scores = np.array([objective(point) for point in grid])
    scores = np.where(np.isfinite(scores), scores, np.inf)
    indices = np.argmin(scores, axis=0)
    left = grid[np.maximum(indices-1, 0)]
    right = grid[np.minimum(indices+1, len(grid)-1)]
    refined = _refine(objective, left, right, 60)
    # Keep the original grid winner and both exact boundaries as candidates.
    choices = np.stack([grid[indices], refined, np.zeros_like(refined),
                        np.full_like(refined, upper)])
    values = np.array([objective(choice) for choice in choices])
    values = np.where(np.isfinite(values), values, np.inf)
    return np.take_along_axis(choices, values.argmin(axis=0)[None, :], axis=0)[0]


def fit_scalar_components(terms, settings):
    bound = settings.get("variance_bound", 1e6)
    grid_size = settings.get("scalar_grid_size", 121)
    if not np.isfinite(bound) or bound <= 0:
        raise ValueError("variance_bound must be positive and finite")
    if not isinstance(grid_size, int) or grid_size < 3:
        raise ValueError("scalar_grid_size must be an integer of at least three")
    spectral = prepare_scalar(terms)
    scale = np.trace(terms["gram"])/terms["n"]
    inactive = scale == 0
    upper = np.log1p(bound)
    units = (np.zeros(len(terms["energy"])) if inactive else
             _search(spectral, scale, upper, grid_size))
    ratios = np.expm1(units)/(1. if inactive else scale)
    values, sigmas = scalar_objective(ratios, spectral)
    valid = terms["energy"] > 64*np.finfo(float).eps*terms["raw_energy"]
    return [dict(valid=bool(ok), criterion=float(value) if ok else np.inf,
                 ratios=[float(ratio) if ok else 0.],
                 sigma2=float(sigma) if ok else np.nan, converged=bool(ok),
                 at_zero=bool(unit == 0 or not ok),
                 at_upper=bool(unit >= upper-1e-6 and ok), inactive=[bool(inactive)])
            for ok, value, ratio, sigma, unit in zip(valid, values, ratios, sigmas, units)]
