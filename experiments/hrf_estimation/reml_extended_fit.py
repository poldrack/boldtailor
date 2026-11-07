"""Variance optimization and streaming library selection for extended REML."""

import numpy as np
from scipy.optimize import minimize
from joblib import Parallel, delayed, parallel_config

from experiments.hrf_estimation.estimators import HrfEstimate, validate_features
from experiments.hrf_estimation.reml_extensions import (
    component_objective, component_terms, estimate_noise, prepare_common,
)


def fit_components(terms, feature, settings):
    count = len(terms["names"])
    valid = terms["energy"][feature] > 64*np.finfo(float).eps*terms["raw_energy"][feature]
    if not valid:
        return dict(valid=False, criterion=np.inf, ratios=[0.]*count,
                    sigma2=np.nan, converged=False, at_zero=True, at_upper=False, inactive=[False]*count)
    scale = np.array([np.diag(terms["gram"])[terms["groups"] == k].sum()/terms["n"]
                      for k in range(count)])
    inactive = scale == 0
    scale = np.where(scale > 0, scale, 1.)
    upper = np.log1p(settings.get("variance_bound", 1e6))
    def objective(unit):
        ratios = np.expm1(unit)/scale
        value, gradient, _ = component_objective(ratios, terms, feature)
        return value, gradient*np.exp(unit)/scale
    best = np.zeros(count)
    value = objective(best)[0]
    converged = bool(np.all(objective(best)[1] >= 0))
    for start in (np.zeros(count), np.where(inactive,0.,np.log(2.))):
        result = minimize(objective, start, jac=True, method="L-BFGS-B",
                          bounds=[(0.,0. if absent else upper) for absent in inactive],
                          options=dict(maxiter=settings.get("reml_maxiter", 200),
                                       ftol=1e-12, gtol=1e-7))
        if np.isfinite(result.fun) and result.fun <= value:
            best, value, converged = result.x, float(result.fun), bool(result.success)
    ratios = np.expm1(best)/scale
    _, _, sigma = component_objective(ratios, terms, feature)
    return dict(valid=True, criterion=value, ratios=ratios.tolist(), sigma2=float(sigma),
                converged=converged, at_zero=bool(np.all(best == 0)),
                at_upper=bool(np.any(best >= upper-1e-6)), inactive=inactive.tolist())


def _configuration(runs, settings):
    variant = settings.get("variant", "combined")
    if variant not in ("ar1", "components", "shared", "combined"):
        raise ValueError("unknown extended REML variant")
    mode = settings.get("rt_mode", "amplitude")
    noise = (estimate_noise(runs, mode) if variant in ("ar1", "combined") else
             dict(rho=0., training_run_ids=[r.run_id for r in runs], reference="iid"))
    return variant, mode, noise


def _select(runs, library, settings, common, variant, mode):
    count = len(runs[0].feature_ids)
    best = np.full(count, np.inf)
    second = np.full(count, np.inf)
    winners, fits = [None]*count, [None]*count
    with parallel_config(backend="loky", inner_max_num_threads=1):
        parallel = Parallel(n_jobs=settings.get("n_jobs", 1))
        for candidate in library.candidates:
            terms = component_terms(runs, candidate, common,
                                    separate=variant in ("components", "combined"),
                                    shared=variant in ("shared", "combined"), mode=mode)
            results = parallel(delayed(fit_components)(terms, f, settings) for f in range(count))
            for f, result in enumerate(results):
                value = result["criterion"]
                if value < best[f]:
                    second[f], best[f] = best[f], value
                    winners[f], fits[f] = candidate, result
                elif value < second[f]:
                    second[f] = value
            if settings.get("verbose", 0) and candidate.id % 25 == 0:
                print(f"REML {variant}: candidate {candidate.id+1}/{len(library.candidates)}", flush=True)
    return winners, fits, best, second, terms["names"]


def fit_extended_reml(runs, library, settings):
    if not runs:
        raise ValueError("extended REML needs training runs")
    validate_features(runs, runs[0].feature_ids, runs[0].brain_axis)
    variant, mode, noise = _configuration(runs, settings)
    common = prepare_common(runs, noise["rho"])
    winners, fits, best, second, names = _select(runs, library, settings, common, variant, mode)
    status, kernels = [], []
    for winner, fit in zip(winners, fits):
        state = ("no_signal" if fit is None else "variance_zero" if fit["at_zero"]
                 else "ok" if fit["converged"] else "not_converged")
        status.append(state)
        kernels.append(winner if state in ("ok", "not_converged") else None)
    gap = np.full(len(best), np.nan)
    finite = np.isfinite(best) & np.isfinite(second)
    gap[finite] = second[finite]-best[finite]
    details = dict(variant=variant, noise=noise, component_names=names,
                   library_fingerprint=library.fingerprint, candidate_gap=gap.tolist())
    for key in ("ratios", "sigma2", "criterion", "converged", "at_zero", "at_upper", "inactive"):
        details[key] = [None if fit is None else fit[key] for fit in fits]
    return HrfEstimate(runs[0].feature_ids.copy(), tuple(kernels), tuple(status),
                       tuple(r.run_id for r in runs), details, runs[0].brain_axis)
