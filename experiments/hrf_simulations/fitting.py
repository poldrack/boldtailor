"""Matrix adapters to the existing iid and AR(1) scalar REML solvers."""

from collections import OrderedDict
from hashlib import sha256

import numpy as np
from scipy.linalg import block_diag

from experiments.hrf_estimation.estimators import HrfEstimate
from experiments.hrf_estimation.reml import RunTerms, estimate_ratio
from experiments.hrf_estimation.reml_scalar import fit_scalar_components
from experiments.hrf_estimation.reml_extensions import whiten, noise_moments
from experiments.hrf_estimation.scoring import residuals
from experiments.hrf_simulations.design import task_design, reference_design


class DesignCache:
    """Bounded, signal-independent cache; keys include actual timing/nuisance."""

    def __init__(self, max_entries=65536):
        self.entries = OrderedDict()
        self.max_entries = max_entries

    def get(self, run, key, builder):
        digest = sha256(run.events.to_csv(index=False).encode())
        digest.update(run.frame_times.tobytes())
        digest.update(run.confounds.to_numpy(float).tobytes())
        identity = (digest.hexdigest(), key)
        if identity not in self.entries:
            self.entries[identity] = builder()
            if len(self.entries) > self.max_entries:
                self.entries.popitem(last=False)
        self.entries.move_to_end(identity)
        return self.entries[identity]


def _reference(run, cache):
    def build():
        matrix = reference_design(run)
        u, s, _ = np.linalg.svd(matrix, full_matrices=False)
        rank = np.sum(s > s[0] * max(matrix.shape) * np.finfo(float).eps)
        numerator, denominator = noise_moments(matrix)
        grid = np.linspace(-0.95, 0.95, 2001)
        expected = np.polynomial.polynomial.polyval(
            grid, numerator
        ) / np.polynomial.polynomial.polyval(grid, denominator)
        return u[:, :rank], grid, expected

    return cache.get(run, "reference", build)


def estimate_rho(runs, cache=None):
    cache = cache or DesignCache()
    corrected = []
    for run in runs:
        q, grid, expected = _reference(run, cache)
        error = run.signals - q @ (q.T @ run.signals)
        denominator = np.sum(error[:-1] ** 2, axis=0)
        valid = denominator > 64 * np.finfo(float).eps * np.sum(run.signals**2, axis=0)
        values = np.sum(error[1:] * error[:-1], axis=0)[valid] / denominator[valid]
        if np.all(np.diff(expected) > 0):
            corrected.extend(np.interp(values, expected, grid))
        else:
            corrected.extend(
                grid[np.abs(expected[:, None] - values[None, :]).argmin(axis=0)]
            )
    return float(np.clip(np.median(corrected), -0.8, 0.8)) if corrected else 0.0


def _terms(y, z, c):
    y, rank = residuals(c, y)
    z, _ = residuals(c, z)
    u, s, _ = np.linalg.svd(z, full_matrices=False)
    keep = s > (s[0] * max(z.shape) * np.finfo(float).eps if len(s) else 0.0)
    coordinates = u[:, keep].T @ y
    energy = np.sum(y * y, axis=0)
    if len(y) <= rank:
        raise ValueError("positive nuisance-restricted degrees of freedom required")
    return y, z, s[keep] ** 2, coordinates**2, energy, len(y) - rank


def random_scales(parts, mode):
    if mode not in ("native", "unit_rms"):
        raise ValueError("unknown regressor scaling")
    count = parts[0].task.shape[1] + parts[0].junk.shape[1]
    if mode == "native":
        return np.ones(count)
    projected = [
        residuals(p.nuisance, np.column_stack([p.task, p.junk]))[0] for p in parts
    ]
    scale = np.sqrt(
        sum(np.sum(z * z, axis=0) for z in projected) / sum(len(z) for z in projected)
    )
    return np.where(scale < 1e-12, 1.0, scale)


def _candidate_fit(runs, parts, rho, noise_mode, settings):
    scales = random_scales(parts, settings.get("regressor_scaling", "native"))
    terms, projected = [], []
    for run, part in zip(runs, parts, strict=True):
        raw = whiten(run.signals, rho)
        y, z, s2, coord2, energy, dof = _terms(
            raw,
            whiten(np.column_stack([part.task, part.junk]) / scales, rho),
            whiten(part.nuisance, rho),
        )
        terms.append(
            RunTerms(
                s2,
                coord2,
                np.maximum(0.0, energy - coord2.sum(axis=0)),
                dof,
                energy,
                np.sum(raw**2, axis=0),
            )
        )
        projected.append((y, z))
    if noise_mode == "iid":
        result = estimate_ratio(
            terms, settings.get("grid_size", 121), settings.get("iterations", 50)
        )
        return dict(
            criterion=float(result["criterion"][0]),
            variance_ratio=float(result["ratio"][0]),
            sigma2=float(result["sigma2"][0]),
            at_zero=bool(result["at_zero"][0]),
            at_upper=bool(result["at_upper"][0]),
            valid=bool(result["valid"][0]),
            converged=True,
        )
    z = block_diag(*[p[1] for p in projected])
    y = np.vstack([p[0] for p in projected])
    scalar = dict(
        gram=z.T @ z,
        coordinates=z.T @ y,
        energy=sum(t.energy for t in terms),
        raw_energy=sum(t.raw_energy for t in terms),
        n=sum(t.restricted_dof for t in terms),
        names=["independent_task_junk"],
        groups=np.zeros(z.shape[1], int),
    )
    result = fit_scalar_components(scalar, settings)[0]
    result["variance_ratio"] = result["ratios"][0]
    return result


def fit(
    runs,
    library,
    model,
    convention,
    noise_mode,
    oracle_rho=None,
    cache=None,
    settings=None,
    design_builder=task_design,
):
    if not runs or any(r.signals.shape[1] != 1 for r in runs):
        raise ValueError("fit requires one simulated vertex and at least one run")
    if noise_mode not in ("iid", "ar1"):
        raise ValueError("noise_mode must be iid or ar1")
    if oracle_rho is not None and noise_mode != "ar1":
        raise ValueError("oracle rho is only meaningful for AR1")
    cache, settings = cache or DesignCache(), settings or {}
    rho = (
        0.0
        if noise_mode == "iid"
        else estimate_rho(runs, cache)
        if oracle_rho is None
        else float(oracle_rho)
    )
    best, winner = None, None
    for candidate in library.candidates:
        key = (
            candidate.kind,
            candidate.parameters,
            model,
            convention,
            id(design_builder),
        )
        parts = [
            cache.get(
                r, key, lambda r=r: design_builder(r, candidate, model, convention)
            )
            for r in runs
        ]
        result = _candidate_fit(runs, parts, rho, noise_mode, settings)
        if result["valid"] and np.isfinite(result["criterion"]):
            if best is None or result["criterion"] < best["criterion"]:
                best, winner = result, candidate
                best["random_scales"] = random_scales(
                    parts, settings.get("regressor_scaling", "native")
                ).tolist()
    state = (
        "no_signal" if best is None else "variance_zero" if best["at_zero"] else "ok"
    )
    diagnostics = dict(
        rho=rho,
        oracle=oracle_rho is not None,
        model=model,
        convention=convention,
        noise_mode=noise_mode,
        training_run_ids=[r.run_id for r in runs],
        library_fingerprint=library.fingerprint,
    )
    if best is not None:
        diagnostics.update({k: [v] for k, v in best.items()})
    return HrfEstimate(
        runs[0].feature_ids.copy(),
        (winner if state == "ok" else None,),
        (state,),
        tuple(r.run_id for r in runs),
        diagnostics,
        runs[0].brain_axis,
    )
