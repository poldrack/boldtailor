"""Complete single-trial fitting operation with one shared HRF."""

from collections.abc import Sequence
import hashlib
import json

import numpy as np
import pandas as pd

from boldtailor._single_trial_design import (
    compile_trial_run,
    trial_table,
)
from boldtailor._hrf_design import HRF_NORMALIZATION, hrf_metadata
from boldtailor._constants import OVERSAMPLING
from boldtailor._single_trial_fit import (
    fit_trial_run,
)
from boldtailor._fractional_ridge import (
    regularization,
    fit_fraction_run,
    fraction_metadata,
)
from boldtailor.data import AnalysisData, run_labels_for
from boldtailor.hrf_library import HrfCandidate
from boldtailor._fit_lifecycle import finish_fit, fit_operation
from boldtailor.single_trial_results import (
    SharedTrialDesign,
    SingleTrialResult,
    assemble_trial_result,
)


def fit_shared_hrf_trials(
    data: AnalysisData,
    *,
    ridge_alpha: float = 0.0,
    run_labels: Sequence[str] | None = None,
    hrf_model: str | HrfCandidate = "spm",
    ridge_fraction: float | np.ndarray | None = None,
) -> SingleTrialResult:
    """Validate, fit and record trial betas using one HRF for every feature."""
    with fit_operation("single_trial", data.provenance) as operation:
        alpha, fractions = regularization(ridge_alpha, ridge_fraction, data.n_features)
        labels = run_labels_for(data, run_labels)
        compiled = _compile_runs(data, labels, hrf_model)
        fits = _fit_runs(compiled, data.signals, alpha, fractions)
        activity = _model_metadata(compiled, data.frame_times, labels, alpha, hrf_model)
        if fractions is not None:
            activity.update(fraction_metadata(fractions))
        provenance = finish_fit(operation, activity)
        trials = trial_table(data.events, labels)
        designs = tuple(pd.concat([x, n], axis=1) for x, n, _ in compiled)
        return assemble_trial_result(
            fits,
            trial_table=trials,
            design=SharedTrialDesign(_matrices=designs),
            ridge_alpha=alpha,
            ridge_fraction=fractions,
            diagnostics=[f.diagnostics for f in fits],
            provenance=provenance,
        )


def _compile_runs(data, labels, hrf_model):
    return tuple(
        compile_trial_run(*args, hrf=hrf_model)
        for args in zip(
            data.events, data.frame_times, data.confounds, labels, strict=True
        )
    )


def _fit_runs(compiled, signals, alpha, fractions):
    return tuple(
        (
            fit_trial_run(x, n, y, alpha=alpha)
            if fractions is None
            else fit_fraction_run(x, n, y, fractions=fractions)
        )
        for (x, n, _), y in zip(compiled, signals, strict=True)
    )


def _model_metadata(compiled, times, labels, alpha, hrf="spm"):
    digest = hashlib.sha256()
    for (x, n, _), t in zip(compiled, times, strict=True):
        digest.update(json.dumps([list(x), list(n)], separators=(",", ":")).encode())
        for values in (t, x, n):
            array = np.asarray(values, dtype="<f8")
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
    return dict(
        name="single_trial",
        hrf=hrf_metadata(hrf),
        hrf_normalization=HRF_NORMALIZATION,
        oversampling=OVERSAMPLING,
        run_labels=list(labels),
        ridge_alpha=alpha,
        beta_units="native_signal",
        normalization="unit_l2_after_nuisance_projection",
        noise_model="ols",
        nuisance_columns=[list(n) for _, n, _ in compiled],
        design_fingerprint=digest.hexdigest(),
    )
