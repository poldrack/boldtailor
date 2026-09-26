"""Single-trial canonical-HRF models without repeated-stimulus tuning."""

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
from boldtailor._single_trial_fit import fit_trial_run, r_squared, validate_alpha
from boldtailor.data import AnalysisData
from boldtailor.logging import append_event_history, bind_context, emit_event
from boldtailor.provenance import analysis_fingerprint, extend_provenance
from boldtailor.single_trial_results import SingleTrialResult


def fit_single_trials(
    data: AnalysisData,
    *,
    ridge_alpha: float = 0.0,
    run_labels: Sequence[str] | None = None,
    hrf="spm",
) -> SingleTrialResult:
    """Estimate one native-unit beta per event row, independently in each run.

    Nuisance regressors and an intercept are unpenalized. Positive alpha penalizes
    trial coefficients after nuisance projection and column normalization. RT and
    condition labels are retained as metadata and never enter the design.
    """
    alpha = validate_alpha(ridge_alpha)
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
                fit_trial_run(x, n, y, alpha=alpha)
                for (x, n, _), y in zip(compiled, data.signals, strict=True)
            )
        except Exception:
            emit_event("single_trial_failed", stage="fit", level=logging.ERROR)
            raise
        history = append_event_history(
            history, emit_event("single_trial_completed", stage="fit")
        )
    activity = _model_metadata(compiled, data.frame_times, labels, alpha, hrf)
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
    return _assemble_result(compiled, fits, alpha, provenance)


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


def _assemble_result(compiled, fits, alpha, provenance):
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
        tuple(f.betas for f in fits),
        trials,
        tuple(designs),
        tuple(r_squared(f.full_sse, f.total_ss) for f in fits),
        tuple(r_squared(f.nuisance_sse, f.total_ss) for f in fits),
        full,
        nuisance,
        full - nuisance,
        tuple(f.diagnostics for f in fits),
        alpha,
        provenance,
    )
