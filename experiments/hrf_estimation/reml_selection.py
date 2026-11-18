"""Select library HRFs with the package's REML selection.

The statistical model is the package's: a fixed task mean under the reference
(canonical) HRF, and independent valid-trial deviations as random effects.
Only the training-only noise estimate and its reference basis are local.
"""

import numpy as np

from boldtailor.ar1 import estimate_rho
from boldtailor.hrf_library import HrfCandidate
from experiments.hrf_estimation.design import build_design, partition_events
from experiments.hrf_estimation.package_selection import estimate, select

REFERENCE_PARAMETERS = (
    (3.5, 12, 0.8, 1.5, 6, 0, 36),
    (6, 16, 1, 1, 6, 0, 36),
    (8, 20, 1.2, 1.5, 6, 0, 36),
)
REFERENCE_LABEL = "fixed fast/canonical-like/slow task+junk basis"
RANDOM_EFFECTS = (
    "independent valid-trial deviations; shared scalar variance; "
    "fixed reference-HRF task mean"
)


def random_design(run, candidate, rt_mode="amplitude"):
    """Task and junk columns of one HRF, used only in the noise reference basis."""
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
    noise = estimate_rho(
        [r.signals for r in runs], [reference_design(r, rt_mode) for r in runs]
    )
    corrected, n_valid = _median(noise.values)
    raw, _ = _median(noise.raw)
    return dict(
        rho=noise.pooled,
        corrected_median=corrected,
        raw_median=raw,
        n_valid=n_valid,
        clipped=noise.pooled_at_bound,
        training_run_ids=[r.run_id for r in runs],
        reference=REFERENCE_LABEL,
    )


def _optional(values):
    return None if values is None else values.tolist()


def _diagnostics(reml, fallback):
    selection, fit, canonical = reml.selection, reml.selection.fit, reml.canonical_fit
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
        canonical_criterion=_optional(
            None if canonical is None else canonical.criterion
        ),
        delta_criterion=reml.delta_criterion.tolist(),
        reference_hrf_id=int(reml.reference_hrf_id),
        canonical_fallback=fallback.tolist(),
        random_effects=RANDOM_EFFECTS,
    )


def select_library_reml(runs, library, rt_mode="amplitude", rho=0.0):
    """Per-feature library HRF by the package's restricted likelihood.

    A feature whose random-effect variance is zero has no likelihood winner:
    the reference HRF already explains its task response, so it keeps that HRF.
    Features without resolved residual signal get no HRF.
    """
    result = select(runs, library, rt_mode, "reml", rho)
    reml = result.reml
    status = np.array(reml.selection.fit.status)
    fallback = (result.hrf_indices < 0) & (status == "variance_zero")
    ids = np.where(fallback, reml.reference_hrf_id, result.hrf_indices)
    diagnostics = dict(
        _diagnostics(reml, fallback),
        rt_mode=rt_mode,
        n_random_effects_per_run=[
            int((partition_events(r.events).category == "valid").sum()) for r in runs
        ],
    )
    return estimate(runs, library, result, status.tolist(), diagnostics, ids)


def fit_library_reml(runs, library, settings):
    """Independent-noise REML selection."""
    return select_library_reml(runs, library, settings.get("rt_mode", "amplitude"))


def fit_ar1_reml(runs, library, settings):
    """REML selection after whitening by one pooled training-only AR(1)."""
    mode = settings.get("rt_mode", "amplitude")
    noise = estimate_noise(runs, mode)
    fitted = select_library_reml(runs, library, mode, noise["rho"])
    fitted.diagnostics["noise"] = noise
    return fitted
