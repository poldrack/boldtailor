"""Experimental training-only shrinkage and surface regularization of AR(1)."""
from dataclasses import replace
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from boldtailor.hrf_library import HrfCandidate
from experiments.hrf_estimation.design import build_design
from experiments.hrf_estimation.scoring import residuals
from experiments.hrf_estimation.reml_extensions import _correct_noise_moments, whiten


def reference_design(run):
    designs = []
    for parameters in ((3.5,12,.8,1.5,6,0,36), (6,16,1,1,6,0,36), (8,20,1.2,1.5,6,0,36)):
        parts = build_design(run, HrfCandidate(0, "double_gamma", parameters), "amplitude")
        designs.append(np.column_stack([parts.task, parts.junk]))
    return np.column_stack([parts.nuisance, *designs])


def run_rhos(runs):
    """Bias-corrected run-by-vertex estimates, retaining missing estimates as NaN."""
    rows = []
    for run in runs:
        matrix = reference_design(run)
        error, _ = residuals(matrix, run.signals)
        denominator = np.sum(error[:-1]**2, axis=0)
        valid = denominator > 64*np.finfo(float).eps*np.sum(run.signals**2, axis=0)
        values = np.full(len(run.feature_ids), np.nan)
        raw = np.sum(error[1:]*error[:-1], axis=0)[valid] / denominator[valid]
        values[valid] = _correct_noise_moments(raw, matrix)
        rows.append(values)
    return np.asarray(rows)


def shrink_rho(samples, pooled):
    """Moment empirical Bayes on atanh(rho); moderate run variances with four df."""
    z = np.arctanh(np.clip(samples, -.95, .95))
    counts = np.isfinite(z).sum(axis=0)
    mean = np.nanmean(z, axis=0)
    variance = np.nanvar(z, axis=0, ddof=1)
    prior = float(np.nanmedian(variance[counts >= 2]))
    variance = np.where(np.isfinite(variance), variance, prior)
    moderated = ((counts-1).clip(0)*variance+4*prior)/((counts-1).clip(0)+4)
    uncertainty = moderated / np.maximum(counts, 1)
    center = np.arctanh(pooled)
    valid = counts >= 2
    tau2 = max(float(np.mean((mean[valid]-center)**2-uncertainty[valid])), 0.)
    weight = np.divide(tau2, tau2+uncertainty, out=np.zeros_like(mean), where=(tau2+uncertainty)>0)
    weight[~valid] = 0
    mean = np.where(valid, mean, center)
    rho = np.clip(np.tanh(center+weight*(mean-center)), -.8, .8)
    return dict(rho=rho, weight=weight, tau2=tau2, uncertainty=uncertainty,
                raw_rho=np.tanh(mean), prior_variance=prior)


def surface_weights(xyz, faces, vertices, fwhm):
    """Area-weighted Gaussian of mesh-edge geodesics, truncated at three sigma.

    Distances use the full mesh; contributions are restricted to the supplied ROI.
    """
    edges = np.unique(np.sort(np.concatenate([faces[:,[0,1]], faces[:,[1,2]], faces[:,[2,0]]]), axis=1), axis=0)
    lengths = np.linalg.norm(xyz[edges[:,0]]-xyz[edges[:,1]], axis=1)
    graph = csr_matrix((np.r_[lengths,lengths], (np.r_[edges[:,0],edges[:,1]], np.r_[edges[:,1],edges[:,0]])), shape=(len(xyz),len(xyz)))
    areas = np.linalg.norm(np.cross(xyz[faces[:,1]]-xyz[faces[:,0]], xyz[faces[:,2]]-xyz[faces[:,0]]), axis=1)/6
    mass = np.bincount(faces.ravel(), weights=np.repeat(areas,3), minlength=len(xyz))
    sigma = fwhm/np.sqrt(8*np.log(2))
    blocks = []
    for start in range(0,len(vertices),128):
        distances = dijkstra(graph, indices=vertices[start:start+128], limit=3*sigma)[:,vertices]
        blocks.append(np.exp(-.5*(distances/sigma)**2)*mass[vertices])
    weights = np.vstack(blocks)
    return weights/weights.sum(axis=1,keepdims=True)


def subset_run(run, indices):
    indices = np.asarray(indices)
    return replace(run, signals=run.signals[:,indices], feature_ids=run.feature_ids[indices],
                   brain_axis=None if run.brain_axis is None else run.brain_axis[indices])


def fit_grouped(runs, library, rho, verbose=False):
    from experiments.hrf_estimation.regularized_fit import fit_grouped as fit
    return fit(runs, library, rho, verbose)


def _acf(error):
    centered = error-error.mean(axis=0)
    energy = np.sum(centered**2,axis=0)
    return np.array([np.sum(centered[k:]*centered[:-k],axis=0)/energy for k in range(1,11)])


def whitening_diagnostics(runs, rho):
    """Held-out reference-design GLS residual ACF; descriptive, not a formal test."""
    rows = []
    for run in runs:
        matrix = reference_design(run)
        before, _ = residuals(matrix, run.signals)
        after = np.empty_like(before)
        for value in np.unique(rho):
            indices = np.flatnonzero(rho == value)
            after[:,indices], _ = residuals(whiten(matrix,value), whiten(run.signals[:,indices],value))
        raw, white = _acf(before), _acf(after)
        for i,vertex in enumerate(run.feature_ids):
            rows.append(dict(run_id=run.run_id, feature_id=int(vertex),
                             before_abs_lag1=abs(raw[0,i]), after_abs_lag1=abs(white[0,i]),
                             before_lag1=raw[0,i], after_lag1=white[0,i],
                             before_acf_energy=np.sum(raw[:,i]**2), after_acf_energy=np.sum(white[:,i]**2)))
    return pd.DataFrame(rows)
