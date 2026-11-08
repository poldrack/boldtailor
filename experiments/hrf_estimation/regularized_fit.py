"""Grouped scalar REML reuses each candidate design across noise coefficients."""
import numpy as np
from experiments.hrf_estimation.design import build_design
from experiments.hrf_estimation.estimators import HrfEstimate, validate_features
from experiments.hrf_estimation.reml_extensions import prepare_common, whiten, _project_random, _components
from experiments.hrf_estimation.reml_scalar import fit_scalar_components
from experiments.hrf_estimation.regularized_noise import subset_run


def _terms(designs, common):
    projected = [_project_random(whiten(z,item["rho"]), item) for z,item in zip(designs,common)]
    b,groups,names = _components(projected,False,False)
    y = np.vstack([item["y"] for item in common])
    return dict(gram=b.T@b, coordinates=b.T@y, groups=groups, names=names,
                energy=np.sum(y**2,axis=0), n=sum(item["dof"] for item in common),
                raw_energy=sum(item["raw_energy"] for item in common))


def _groups(runs, rho):
    result = []
    for value in np.unique(rho):
        indices = np.flatnonzero(rho == value)
        subset = tuple(subset_run(r, indices) for r in runs)
        result.append((indices,prepare_common(subset,float(value))))
    return result


def fit_grouped(runs, library, rho, verbose=False):
    validate_features(runs,runs[0].feature_ids,runs[0].brain_axis)
    rho = np.asarray(rho)
    if rho.shape != (len(runs[0].feature_ids),) or not np.isfinite(rho).all() or (abs(rho)>=1).any():
        raise ValueError("one finite stationary rho per vertex required")
    groups = _groups(runs,rho)
    n = len(rho)
    best = np.full(n,np.inf)
    second = best.copy()
    winners,results = [None]*n,[None]*n
    for candidate in library.candidates:
        designs = []
        for run in runs:
            parts = build_design(run,candidate,"amplitude")
            designs.append(np.column_stack([parts.task,parts.junk]))
        for indices,common in groups:
            fits = fit_scalar_components(_terms(designs,common),{})
            for index,fit in zip(indices,fits):
                value = fit["criterion"]
                if value < best[index]:
                    second[index],best[index] = best[index],value
                    winners[index],results[index] = candidate,fit
                elif value < second[index]:
                    second[index] = value
        if verbose and candidate.id % 100 == 0:
            print(f"candidate {candidate.id+1}/{len(library.candidates)}, {len(groups)} rho groups",flush=True)
    status = tuple("no_signal" if r is None else "variance_zero" if r["at_zero"] else "ok" for r in results)
    kernels = tuple(k if s=="ok" else None for k,s in zip(winners,status))
    details = dict(rho=rho.tolist(), criterion=best.tolist(), candidate_gap=(second-best).tolist(),
                   n_rho_groups=len(groups), library_fingerprint=library.fingerprint)
    return HrfEstimate(runs[0].feature_ids.copy(),kernels,status,tuple(r.run_id for r in runs),details,runs[0].brain_axis)
