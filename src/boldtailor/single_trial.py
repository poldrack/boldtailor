"""Single-trial shared or selected HRF models without repeated-stimulus tuning."""

from collections.abc import Sequence
import hashlib
import json

import numpy as np
import pandas as pd

from boldtailor._single_trial_design import (  # compile_trial_run: public
    compile_trial_run,
    trial_table,
)
from boldtailor._hrf_design import HRF_NORMALIZATION, OVERSAMPLING, hrf_metadata
from boldtailor._single_trial_fit import (  # project_trial_design, r_squared, validate_alpha: public
    fit_trial_run,
    project_trial_design,
    r_squared,
    validate_alpha,
)
from boldtailor._fractional_ridge import (
    regularization,
    fit_fraction_run,
    fraction_metadata,
)
from boldtailor._deprecation import UNSET, renamed_keyword
from boldtailor.data import AnalysisData, run_labels_for
from boldtailor.hrf_library import HrfCandidate
from boldtailor.hrf_results import HrfSelectionResult
from boldtailor._fit_lifecycle import fit_operation
from boldtailor.provenance import analysis_fingerprint
from boldtailor.single_trial_results import SingleTrialResult, SharedTrialDesign


def fit_single_trials(
    data: AnalysisData,
    *,
    ridge_alpha: float = 0.0,
    run_labels: Sequence[str] | None = None,
    hrf_model: str | HrfCandidate = "spm",
    ridge_fraction: float | np.ndarray | None = None,
    hrf: object = UNSET,
) -> SingleTrialResult:
    """Estimate one native-unit beta per event row, independently in each run.

    Nuisance regressors and an intercept are unpenalized. Positive alpha penalizes
    trial coefficients after nuisance projection and column normalization. RT and
    condition labels are retained as metadata and never enter the design.

    Alternatively, ridge_fraction specifies the coefficient-norm ratio to OLS
    in the raw trial-coefficient basis after nuisance projection, as a scalar or
    one value per feature. Values must
    be in (0, 1]; NaNs in a feature map exclude features. A positive ridge_alpha
    cannot be combined with ridge_fraction. Fractional results include the
    selected fractions and implied alpha for each run and feature.

    ``hrf_model`` is 'spm' or one identified ``HrfCandidate`` used everywhere;
    the old keyword ``hrf`` is a deprecated alias.
    """
    hrf_model = renamed_keyword(
        "fit_single_trials", "hrf", "hrf_model", hrf, hrf_model, default="spm"
    )
    with fit_operation("single_trial", data.provenance) as operation:
        alpha, fractions = regularization(ridge_alpha, ridge_fraction, data.n_features)
        labels = run_labels_for(data, run_labels)
        compiled = _compile_runs(data, labels, hrf_model)
        fits = _fit_runs(compiled, data.signals, alpha, fractions)
        activity = _model_metadata(compiled, data.frame_times, labels, alpha, hrf_model)
        if fractions is not None:
            activity.update(fraction_metadata(fractions))
        provenance = operation.provenance(
            activity,
            analysis_id=analysis_fingerprint(
                data.provenance.metadata_fingerprint, activity
            ),
        )
        trials = trial_table(data.events, labels)
        return _assemble_result(compiled, trials, fits, alpha, provenance, fractions)


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


def _assemble_result(compiled, trials, fits, alpha, provenance, fractions=None):
    designs = tuple(pd.concat([x, n], axis=1) for x, n, _ in compiled)
    total = np.sum([f.total_ss for f in fits], axis=0)
    full = r_squared(np.sum([f.full_sse for f in fits], axis=0), total)
    nuisance = r_squared(np.sum([f.nuisance_sse for f in fits], axis=0), total)
    return SingleTrialResult(
        run_betas=tuple(f.betas for f in fits),
        _trial_table=trials,
        design=SharedTrialDesign(_matrices=designs),
        run_full_r2=tuple(r_squared(f.full_sse, f.total_ss) for f in fits),
        run_nuisance_r2=tuple(r_squared(f.nuisance_sse, f.total_ss) for f in fits),
        full_r2=full,
        nuisance_r2=nuisance,
        delta_r2=full - nuisance,
        _diagnostics=tuple(f.diagnostics for f in fits),
        ridge_alpha=alpha,
        provenance=provenance,
        ridge_fraction=fractions,
        run_ridge_alphas=(
            None
            if fractions is None
            else tuple(f.diagnostics["ridge_alphas"] for f in fits)
        ),
    )


def fit_selected_hrfs(
    data: AnalysisData,
    *,
    hrf_selection: HrfSelectionResult = UNSET,
    ridge_alpha: float = 0.0,
    run_labels: Sequence[str] | None = None,
    feature_signature: str | None = None,
    ridge_fraction: float | np.ndarray | None = None,
    selection: HrfSelectionResult = UNSET,
) -> SingleTrialResult:
    """Fit unrestricted trial betas using each feature's previously selected HRF.

    The selection may come from different runs. A supplied spatial signature
    must match; for anonymous arrays, the caller must preserve feature order.
    OLS, fixed-alpha ridge, and fractional ridge use the identical selection
    and peak-normalized HRFs. ridge_fraction follows fit_single_trials semantics;
    its implied alpha is computed separately for each run and feature.
    The old keyword ``selection`` is a deprecated alias of ``hrf_selection``.
    """
    from boldtailor._selected_hrf_fit import fit_groups

    hrf_selection = renamed_keyword(
        "fit_selected_hrfs", "selection", "hrf_selection", selection, hrf_selection
    )
    return fit_groups(
        data,
        selection=hrf_selection,
        ridge_alpha=ridge_alpha,
        run_labels=run_labels,
        feature_signature=feature_signature,
        ridge_fraction=ridge_fraction,
    )
