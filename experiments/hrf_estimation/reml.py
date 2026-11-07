"""Direct library REML with common fixed nuisance and random task effects."""

from dataclasses import dataclass

import numpy as np

from experiments.hrf_estimation.design import build_design
from experiments.hrf_estimation.estimators import HrfEstimate, validate_features
from experiments.hrf_estimation.scoring import residuals


@dataclass(frozen=True)
class RunTerms:
    singular2: np.ndarray
    coordinates2: np.ndarray
    residual_ss: np.ndarray
    restricted_dof: int
    energy: np.ndarray
    raw_energy: np.ndarray


def prepare_terms(runs, candidate, rt_mode="amplitude"):
    return tuple(_run_terms(run, candidate, rt_mode) for run in runs)


def _run_terms(run, candidate, mode):
    parts = build_design(run, candidate, mode)
    y, rank = residuals(parts.nuisance, run.signals)
    z, _ = residuals(parts.nuisance, np.column_stack([parts.task, parts.junk]))
    u, s, _ = np.linalg.svd(z, full_matrices=False)
    keep = s > (s[0] * max(z.shape) * np.finfo(float).eps if len(s) else 0)
    coordinates2 = (u[:, keep].T @ y) ** 2
    energy = np.sum(y**2, axis=0)
    if len(y) - rank <= 0:
        raise ValueError(
            "REML requires positive nuisance-restricted degrees of freedom"
        )
    return RunTerms(
        s[keep] ** 2,
        coordinates2,
        np.maximum(0, energy - coordinates2.sum(axis=0)),
        len(y) - rank,
        energy,
        np.sum(run.signals**2, axis=0),
    )


def criterion_grid(ratios, terms):
    ratios = np.asarray(ratios, dtype=float)
    q = sum(t.residual_ss for t in terms)[None, :] + np.zeros((len(ratios), 1))
    determinant = np.zeros(len(ratios))
    for t in terms:
        denominator = 1 + ratios[:, None] * t.singular2[None, :]
        q += (1 / denominator) @ t.coordinates2
        determinant += np.log(denominator).sum(axis=1)
    n = sum(t.restricted_dof for t in terms)
    with np.errstate(divide="ignore", invalid="ignore"):
        return n * np.log(q / n) + determinant[:, None]


def _each(ratios, terms):
    q = sum(t.residual_ss for t in terms).copy()
    determinant = np.zeros_like(q)
    for t in terms:
        denominator = 1 + t.singular2[:, None] * ratios[None, :]
        q += (t.coordinates2 / denominator).sum(axis=0)
        determinant += np.log(denominator).sum(axis=0)
    n = sum(t.restricted_dof for t in terms)
    with np.errstate(divide="ignore", invalid="ignore"):
        return n * np.log(q / n) + determinant, q / n


def _refine(function, left, right, iterations):
    fraction = (np.sqrt(5) - 1) / 2
    a, b = left.copy(), right.copy()
    c, d = b - fraction * (b - a), a + fraction * (b - a)
    fc, fd = function(c), function(d)
    for _ in range(iterations):
        lower = fc < fd
        a, b = np.where(lower, a, c), np.where(lower, d, b)
        point = np.where(lower, b - fraction * (b - a), a + fraction * (b - a))
        value = function(point)
        c, fc, d, fd = (
            np.where(lower, point, d),
            np.where(lower, value, fd),
            np.where(lower, c, point),
            np.where(lower, fc, value),
        )
    return (a + b) / 2


def estimate_ratio(terms, grid_size=121, iterations=50):
    s2 = np.concatenate([t.singular2 for t in terms])
    count = len(terms[0].energy)
    if not len(s2):
        ratios = np.zeros(count)
        upper = np.zeros(count, dtype=bool)
    else:
        grid = np.linspace(np.log(1e-8 / s2.max()), np.log(1e8 / s2.min()), grid_size)
        values = criterion_grid(np.exp(grid), terms)
        indices = np.argmin(values, axis=0)
        refined = _refine(
            lambda x: _each(np.exp(x), terms)[0],
            grid[np.maximum(0, indices - 1)],
            grid[np.minimum(grid_size - 1, indices + 1)],
            iterations,
        )
        ratios = np.exp(refined)
        grid_ratio = np.exp(grid[indices])
        grid_better = _each(grid_ratio, terms)[0] < _each(ratios, terms)[0]
        ratios = np.where(grid_better, grid_ratio, ratios)
        upper = indices == grid_size - 1
    at_zero = _each(np.zeros(count), terms)[0] <= _each(ratios, terms)[0]
    ratios = np.where(at_zero, 0.0, ratios)
    criterion, sigma2 = _each(ratios, terms)
    valid = sum(t.energy for t in terms) > 64 * np.finfo(float).eps * sum(
        t.raw_energy for t in terms
    )
    return dict(
        ratio=ratios,
        criterion=criterion,
        sigma2=sigma2,
        at_zero=at_zero,
        at_upper=upper & ~at_zero,
        valid=valid,
    )


def fit_library_reml(runs, library, settings):
    if not runs:
        raise ValueError("REML needs training runs")
    validate_features(runs, runs[0].feature_ids, runs[0].brain_axis)
    fits = [
        estimate_ratio(
            prepare_terms(runs, k, settings.get("rt_mode", "amplitude")),
            settings.get("grid_size", 121),
            settings.get("iterations", 50),
        )
        for k in library.candidates
    ]
    values = np.stack([np.where(f["valid"], f["criterion"], np.inf) for f in fits])
    best = np.argmin(values, axis=0)
    kernels, status = [], []
    details = {
        key: []
        for key in ("variance_ratio", "sigma2", "criterion", "at_upper", "at_zero")
    }
    for feature, index in enumerate(best):
        fit = fits[index]
        state = "ok" if fit["valid"][feature] else "no_signal"
        if state == "ok" and fit["at_zero"][feature]:
            state = "variance_zero"
        kernels.append(library.candidates[index] if state == "ok" else None)
        status.append(state)
        for name, key in [
            ("variance_ratio", "ratio"),
            ("sigma2", "sigma2"),
            ("criterion", "criterion"),
            ("at_upper", "at_upper"),
            ("at_zero", "at_zero"),
        ]:
            details[name].append(fit[key][feature].item())
    details["library_fingerprint"] = library.fingerprint
    details["random_effects"] = (
        "five task/junk coefficients; shared scalar variance; independent runs"
    )
    return HrfEstimate(
        runs[0].feature_ids.copy(),
        tuple(kernels),
        tuple(status),
        tuple(r.run_id for r in runs),
        details,
        runs[0].brain_axis,
    )
