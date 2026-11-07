"""Training-only noise and random-component extensions to direct HRF REML."""

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve

from boldtailor.hrf_library import HrfCandidate
from experiments.hrf_estimation.design import build_design
from experiments.hrf_estimation.scoring import residuals


def whiten(values, rho):
    if not np.isfinite(rho) or abs(rho) >= 1:
        raise ValueError("rho must be finite and strictly within (-1,1)")
    values = np.asarray(values, dtype=float)
    result = values.copy()
    result[0] *= np.sqrt(1 - rho**2)
    result[1:] = values[1:] - rho * values[:-1]
    return result


def estimate_noise(runs, mode):
    correlations, uncorrected = [], []
    for run in runs:
        designs = []
        for parameters in ((3.5, 12, .8, 1.5, 6, 0, 36),
                           (6, 16, 1, 1, 6, 0, 36), (8, 20, 1.2, 1.5, 6, 0, 36)):
            parts = build_design(run, HrfCandidate(0, "double_gamma", parameters), mode)
            designs.append(np.column_stack([parts.task, parts.junk]))
        matrix = np.column_stack([parts.nuisance, *designs])
        error, _ = residuals(matrix, run.signals)
        denominator = np.sum(error[:-1]**2, axis=0)
        valid = denominator > 64*np.finfo(float).eps*np.sum(run.signals**2, axis=0)
        values = np.sum(error[1:]*error[:-1], axis=0)[valid] / denominator[valid]
        values = values[np.isfinite(values)]
        uncorrected.extend(values.tolist())
        correlations.extend(_correct_noise_moments(values, matrix).tolist())
    raw = float(np.median(correlations)) if correlations else 0.
    return dict(rho=float(np.clip(raw, -.8, .8)), corrected_median=raw,
                raw_median=float(np.median(uncorrected)) if uncorrected else 0.,
                n_valid=len(correlations), clipped=bool(abs(raw) > .8),
                training_run_ids=[r.run_id for r in runs],
                reference="fixed fast/canonical-like/slow task+junk basis")


def prepare_common(runs, rho=0.):
    items = []
    for run in runs:
        c = whiten(run.confounds.to_numpy(dtype=float), rho)
        signals = whiten(run.signals, rho)
        y, rank = residuals(c, signals)
        u, singular, _ = np.linalg.svd(c, full_matrices=False)
        if len(y) <= rank:
            raise ValueError("positive restricted degrees of freedom are required")
        items.append(dict(q=u[:, :rank], y=y, rho=rho, dof=len(y)-rank,
                          raw_energy=np.sum(signals**2, axis=0), c_columns=c.shape[1],
                          c_norm=singular[0] if len(singular) else 0.))
    return tuple(items)


def component_terms(runs, kernel, common, separate=False, shared=False, mode="amplitude"):
    designs = []
    for run, item in zip(runs, common):
        parts = build_design(run, kernel, mode)
        z = whiten(np.column_stack([parts.task, parts.junk]), item["rho"])
        designs.append(_project_random(z, item))
    b, groups, names = _components(designs, separate, shared)
    y = np.vstack([item["y"] for item in common])
    return dict(gram=b.T@b, coordinates=b.T@y, groups=groups, names=names,
                energy=np.sum(y**2, axis=0), n=sum(item["dof"] for item in common),
                raw_energy=sum(item["raw_energy"] for item in common))


def _components(designs, separate, shared):
    independent = block_diag(*designs)
    if separate:
        groups = np.tile([0, 0, 1, 2, 2], len(designs))
        names = ["conditions", "rt", "junk"]
    else:
        groups = np.zeros(independent.shape[1], dtype=int)
        names = ["independent_task_junk"]
    if not shared:
        return independent, groups, names
    joined = np.vstack(designs)[:, :3]
    offset = len(names)
    extra = [offset, offset, offset+1] if separate else [offset]*3
    names.extend(["shared_conditions", "shared_rt"] if separate else ["shared_task"])
    return np.column_stack([independent, joined]), np.r_[groups, extra], names


def component_objective(ratios, terms, feature):
    ratios = np.asarray(ratios, dtype=float)
    if len(ratios) != len(terms["names"]) or (ratios < 0).any() or not np.isfinite(ratios).all():
        raise ValueError("one finite nonnegative variance ratio per component is required")
    g, h = terms["gram"], terms["coordinates"][:, feature]
    root = np.sqrt(ratios[terms["groups"]])
    factor = cho_factor(np.eye(len(root)) + root[:, None]*g*root[None, :], lower=True)
    solved = cho_solve(factor, root*h)
    q = terms["energy"][feature] - (root*h)@solved
    q = max(float(q), 64*np.finfo(float).eps*terms["energy"][feature])
    if q <= 0:
        return np.inf, np.zeros(len(ratios)), 0.
    remaining_h = h - (g*root[None, :])@solved
    remaining_g = g - (g*root[None, :])@cho_solve(factor, root[:, None]*g)
    gradient = []
    for group in range(len(ratios)):
        mask = terms["groups"] == group
        gradient.append(np.diag(remaining_g)[mask].sum()
                        - terms["n"]*np.sum(remaining_h[mask]**2)/q)
    determinant = 2*np.log(np.diag(factor[0])).sum()
    value = terms["n"]*np.log(q/terms["n"]) + determinant
    return float(value), np.asarray(gradient), q/terms["n"]


def fit_components(terms, feature, settings):
    from experiments.hrf_estimation.reml_extended_fit import fit_components as fit
    return fit(terms, feature, settings)


def fit_extended_reml(runs, library, settings):
    from experiments.hrf_estimation.reml_extended_fit import fit_extended_reml as fit
    return fit(runs, library, settings)


def noise_moments(matrix):
    """Lag-covariance moments after the exact fixed reference-design projection."""
    u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    rank = np.sum(singular > singular[0]*max(matrix.shape)*np.finfo(float).eps)
    m = np.eye(len(matrix))-u[:, :rank]@u[:, :rank].T
    lag = (np.eye(len(m), k=1)+np.eye(len(m), k=-1))*.5
    previous = np.eye(len(m))
    previous[-1,-1] = 0
    def lag_sums(a):
        return np.array([np.trace(a)]+[2*np.diag(a,k).sum() for k in range(1,len(a))])
    return lag_sums(m@lag@m), lag_sums(m@previous@m)


def _correct_noise_moments(values, matrix):
    numerator, denominator = noise_moments(matrix)
    grid = np.linspace(-.95,.95,2001)
    expected = (np.polynomial.polynomial.polyval(grid,numerator)
                /np.polynomial.polynomial.polyval(grid,denominator))
    if np.all(np.diff(expected)>0):
        return np.interp(values,expected,grid)
    return grid[np.abs(expected[:,None]-values[None,:]).argmin(axis=0)]


def _project_random(z, item):
    projected = z-item["q"]@(item["q"].T@z)
    tolerance = (max(len(z),item["c_columns"]+z.shape[1])*np.finfo(float).eps
                 *max(item["c_norm"],np.linalg.norm(z,2)))
    u, singular, v = np.linalg.svd(projected,full_matrices=False)
    keep = singular>tolerance
    if not keep.all():
        projected = (u[:,keep]*singular[keep])@v[keep]
    projected[:,np.linalg.norm(projected,axis=0)<=tolerance] = 0.
    return projected
