"""Single-trial shared or selected HRF models without repeated-stimulus tuning."""

from collections.abc import Sequence
import hashlib
import json
import logging
from uuid import uuid4

import nilearn
import numpy as np
import pandas as pd

from boldtailor._single_trial_design import compile_trial_run
from boldtailor._hrf_design import hrf_metadata
from boldtailor._single_trial_fit import fit_trial_run, r_squared
from boldtailor._fractional_ridge import (
    regularization,
    fit_fraction_run,
    fraction_metadata,
)
from boldtailor.data import AnalysisData
from boldtailor.logging import append_event_history, bind_context, emit_event
from boldtailor.provenance import analysis_fingerprint, extend_provenance
from boldtailor.single_trial_results import SingleTrialResult, SharedTrialDesign


def fit_single_trials(
    data: AnalysisData,
    *,
    ridge_alpha: float = 0.0,
    run_labels: Sequence[str] | None = None,
    hrf="spm",
    ridge_fraction=None,
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
    """
    alpha, fractions = regularization(ridge_alpha, ridge_fraction, data.n_features)
    labels = (
        tuple(run_labels)
        if run_labels is not None
        else tuple(f"run-{i + 1:02d}" for i in range(data.n_runs))
    )
    if (
        len(labels) != data.n_runs
        or not all(isinstance(x, str) for x in labels)
        or len(set(labels)) != len(labels)
    ):
        raise ValueError("run labels must be unique and match the number of runs")
    execution_id = str(uuid4())
    history = data.provenance.events
    with bind_context(
        execution_id=execution_id, data_id=data.provenance.metadata_fingerprint
    ):
        history = append_event_history(
            history, emit_event("single_trial_started", stage="fit")
        )
        try:
            compiled = tuple(
                compile_trial_run(*args, hrf=hrf)
                for args in zip(
                    data.events, data.frame_times, data.confounds, labels, strict=True
                )
            )
            fits = tuple(
                (
                    fit_trial_run(x, n, y, alpha=alpha)
                    if fractions is None
                    else fit_fraction_run(x, n, y, fractions=fractions)
                )
                for (x, n, _), y in zip(compiled, data.signals, strict=True)
            )
        except Exception:
            emit_event("single_trial_failed", stage="fit", level=logging.ERROR)
            raise
        history = append_event_history(
            history, emit_event("single_trial_completed", stage="fit")
        )
    activity = _model_metadata(compiled, data.frame_times, labels, alpha, hrf)
    if fractions is not None:
        activity.update(fraction_metadata(fractions))
    provenance = extend_provenance(
        data.provenance,
        execution_id=execution_id,
        activity=activity,
        events=history,
        warnings=(),
        analysis_id=analysis_fingerprint(
            data.provenance.metadata_fingerprint, activity
        ),
    )
    return _assemble_result(compiled, fits, alpha, provenance, fractions)


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
        oversampling=50,
        run_labels=list(labels),
        ridge_alpha=alpha,
        beta_units="native_signal",
        normalization="unit_l2_after_nuisance_projection",
        noise_model="ols",
        nuisance_columns=[list(n) for _, n, _ in compiled],
        design_fingerprint=digest.hexdigest(),
        numpy_version=np.__version__,
        nilearn_version=nilearn.__version__,
    )


def _assemble_result(compiled, fits, alpha, provenance, fractions=None):
    tables, designs = [], []
    for run, (x, n, table) in enumerate(compiled):
        table = table.assign(run_index=run)
        tables.append(table)
        designs.append(pd.concat([x, n], axis=1))
    trials = pd.concat(tables, ignore_index=True)
    trials.insert(0, "trial_index", np.arange(len(trials)))
    total = np.sum([f.total_ss for f in fits], axis=0)
    full = r_squared(np.sum([f.full_sse for f in fits], axis=0), total)
    nuisance = r_squared(np.sum([f.nuisance_sse for f in fits], axis=0), total)
    return SingleTrialResult(
        run_betas=tuple(f.betas for f in fits),
        _trial_table=trials,
        design=SharedTrialDesign(tuple(designs)),
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
    data,
    *,
    selection,
    ridge_alpha=0.0,
    run_labels=None,
    feature_signature=None,
    ridge_fraction=None,
):
    """Fit unrestricted trial betas using each feature's previously selected HRF.

    The selection may come from different runs. A supplied spatial signature
    must match; for anonymous arrays, the caller must preserve feature order.
    OLS, fixed-alpha ridge, and fractional ridge use the identical selection
    and sum-normalized HRFs. ridge_fraction follows fit_single_trials semantics;
    its implied alpha is computed separately for each run and feature.
    """
    from boldtailor._selected_hrf_fit import fit_groups

    return fit_groups(
        data,
        selection,
        ridge_alpha,
        run_labels,
        feature_signature,
        ridge_fraction=ridge_fraction,
    )
