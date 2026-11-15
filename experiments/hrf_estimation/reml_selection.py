"""Experiment adapters that select library HRFs with boldtailor's scalar REML."""

import numpy as np

from boldtailor.ar1 import estimate_rho
from boldtailor.hrf_library import HrfCandidate
from boldtailor.reml import RemlGroup, select_random_design
from experiments.hrf_estimation.design import build_design
from experiments.hrf_estimation.estimators import HrfEstimate, validate_features

REFERENCE_PARAMETERS = (
    (3.5, 12, 0.8, 1.5, 6, 0, 36),
    (6, 16, 1, 1, 6, 0, 36),
    (8, 20, 1.2, 1.5, 6, 0, 36),
)
REFERENCE_LABEL = "fixed fast/canonical-like/slow task+junk basis"


def random_design(run, candidate, rt_mode="amplitude"):
    """Task and junk columns of one candidate HRF: the random-effects design."""
    parts = build_design(run, candidate, rt_mode)
    return np.column_stack([parts.task, parts.junk])


def reference_design(run, rt_mode="amplitude"):
    """Nuisance plus fast, canonical-like and slow task+junk columns."""
    designs = [
        random_design(run, HrfCandidate(0, "double_gamma", p), rt_mode)
        for p in REFERENCE_PARAMETERS
    ]
    return np.column_stack([run.confounds.to_numpy(dtype=float), *designs])


def _median(values):
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else 0.0, int(finite.size)


def estimate_noise(runs, rt_mode="amplitude"):
    """Pooled training-only AR(1) coefficient after reference-design projection."""
    estimate = estimate_rho(
        [r.signals for r in runs], [reference_design(r, rt_mode) for r in runs]
    )
    corrected, n_valid = _median(estimate.values)
    raw, _ = _median(estimate.raw)
    return dict(
        rho=estimate.pooled,
        corrected_median=corrected,
        raw_median=raw,
        n_valid=n_valid,
        clipped=bool(abs(corrected) > 0.8),
        training_run_ids=[r.run_id for r in runs],
        reference=REFERENCE_LABEL,
    )


def _groups(runs):
    if not runs:
        raise ValueError("REML needs training runs")
    validate_features(runs, runs[0].feature_ids, runs[0].brain_axis)
    return [RemlGroup(r.signals, r.confounds.to_numpy(dtype=float)) for r in runs]


def _diagnostics(selection, library):
    fit = selection.fit
    return dict(
        variance_ratio=fit.ratio.tolist(),
        sigma2=fit.noise_variance.tolist(),
        criterion=fit.criterion.tolist(),
        at_zero=[s == "variance_zero" for s in fit.status],
        at_upper=fit.at_upper.tolist(),
        unrefined=fit.unrefined.tolist(),
        any_at_upper=selection.any_at_upper.tolist(),
        any_unrefined=selection.any_unrefined.tolist(),
        provisional=selection.any_unrefined.tolist(),
        candidate_gap=selection.gap.tolist(),
        rho=fit.rho.tolist(),
        library_fingerprint=library.fingerprint,
    )


def select_hrf(runs, library, design, rho=0.0):
    """Per-feature library HRF whose random design has the best REML criterion."""
    groups = _groups(runs)
    candidates = ([design(run, k) for run in runs] for k in library.candidates)
    selection = select_random_design(groups, candidates, rho=rho)
    kernels = tuple(None if i < 0 else library.candidates[i] for i in selection.winner)
    return HrfEstimate(
        runs[0].feature_ids.copy(),
        kernels,
        selection.fit.status,
        tuple(r.run_id for r in runs),
        _diagnostics(selection, library),
        runs[0].brain_axis,
    )


def _mode_design(mode):
    def design(run, candidate):
        return random_design(run, candidate, mode)

    return design


def fit_library_reml(runs, library, settings):
    """Independent-noise REML selection with run-specific task/junk effects."""
    mode = settings.get("rt_mode", "amplitude")
    estimate = select_hrf(runs, library, _mode_design(mode))
    count = random_design(runs[0], library.candidates[0], mode).shape[1]
    estimate.diagnostics.update(
        rt_mode=mode,
        n_random_effects_per_run=count,
        random_effects=(
            f"{count} task/junk coefficients; shared scalar variance; independent runs"
        ),
    )
    return estimate


def fit_ar1_reml(runs, library, settings):
    """REML selection after whitening by one pooled training-only AR(1)."""
    mode = settings.get("rt_mode", "amplitude")
    noise = estimate_noise(runs, mode)
    estimate = select_hrf(runs, library, _mode_design(mode), noise["rho"])
    estimate.diagnostics["noise"] = noise
    return estimate
