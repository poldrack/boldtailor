"""Apply an identified HRF assignment, fitting only its feature groups."""

from hashlib import sha256
import json
from uuid import uuid4

import numpy as np
import pandas as pd

from boldtailor._hrf_cv import prepare_runs
from boldtailor._hrf_assignment import validate_selection
from boldtailor._single_trial_design import _validate_events
from boldtailor._single_trial_fit import fit_trial_run, r_squared
from boldtailor._fractional_ridge import (
    regularization,
    fit_fraction_run,
    fraction_metadata,
)
from boldtailor.single_trial_results import SingleTrialResult, SelectedTrialDesign
from boldtailor.hrf_selection import run_labels_for
from boldtailor.provenance import analysis_fingerprint, extend_provenance


def _tables(data, labels):
    tables = []
    for r, (events, times, confounds, label) in enumerate(
        zip(data.events, data.frame_times, data.confounds, labels, strict=True)
    ):
        _validate_events(events, times, label)
        ids = [f"{label}_trial-{i+1:04d}" for i in range(len(events))]
        if set(ids) & set(confounds):
            raise ValueError("nuisance columns collide with reserved trial IDs")
        table = events.reset_index(drop=True).assign(
            trial_id=ids,
            event_index=np.arange(len(events)),
            run_index=r,
            run_label=label,
        )
        tables.append(table)
    trials = pd.concat(tables, ignore_index=True)
    trials.insert(0, "trial_index", np.arange(len(trials)))
    return trials


def _fit_run(run, y, ids, alpha, run_index, fractions=None):
    betas = np.full((len(run.events), y.shape[1]), np.nan)
    full = np.zeros(y.shape[1])
    null = full.copy()
    total = full.copy()
    designs = {}
    diagnostics = []
    alphas = np.full(y.shape[1], np.nan)
    for cid in np.unique(ids[ids >= 0]):
        features = np.flatnonzero(ids == cid)
        x = run.trial_matrix(int(cid))
        fit = (
            fit_trial_run(x, run.nuisance, y[:, features], alpha=alpha)
            if fractions is None
            else fit_fraction_run(
                x, run.nuisance, y[:, features], fractions=fractions[features]
            )
        )
        if fractions is not None:
            alphas[features] = fit.diagnostics["ridge_alphas"]
        betas[:, features] = fit.betas
        full[features] = fit.full_sse
        null[features] = fit.nuisance_sse
        total[features] = fit.total_ss
        designs[run_index, int(cid)] = np.column_stack([x, run.nuisance])
        diagnostics.append(
            dict(
                run_index=run_index,
                hrf_id=int(cid),
                n_features=len(features),
                **fit.diagnostics,
            )
        )
    return betas, full, null, total, designs, diagnostics, alphas


def _provenance(data, selection, labels, alpha, assignment, designs, fractions=None):
    digest = sha256()
    for key, values in sorted(designs.items()):
        digest.update(json.dumps(key).encode())
        digest.update(str(values.shape).encode())
        digest.update(np.asarray(values, dtype="<f8").tobytes())
    for times, confounds, label in zip(
        data.frame_times, data.confounds, labels, strict=True
    ):
        digest.update(np.asarray(times, dtype="<f8").tobytes())
        digest.update(json.dumps([label, list(confounds)]).encode())
    activity = dict(
        name="selected_hrf_single_trial",
        library_fingerprint=selection.library.fingerprint,
        design_fingerprint=digest.hexdigest(),
        hrf_assignment_fingerprint=assignment,
        feature_signature=selection.feature_signature,
        run_labels=list(labels),
        ridge_alpha=alpha,
        beta_units="native_signal",
        normalization="unit_l2_after_nuisance_projection",
        hrf_normalization="discrete_sum_one",
        noise_model="ols",
        oversampling=50,
        selection=selection.provenance.to_dict()["activities"][-1],
    )
    if fractions is not None:
        activity.update(fraction_metadata(fractions))
    return extend_provenance(
        data.provenance,
        execution_id=str(uuid4()),
        activity=activity,
        events=data.provenance.events,
        warnings=(),
        analysis_id=analysis_fingerprint(
            data.provenance.metadata_fingerprint, activity
        ),
    )


def fit_groups(
    data, selection, ridge_alpha, run_labels, feature_signature, *, ridge_fraction=None
):
    alpha, fractions = regularization(ridge_alpha, ridge_fraction, data.n_features)
    assignment = validate_selection(data, selection, feature_signature)
    labels = run_labels_for(data, run_labels)
    trials = _tables(data, labels)
    runs = prepare_runs(data, selection.library)
    fits = [
        _fit_run(run, y, selection.hrf_indices, alpha, r, fractions)
        for r, (run, y) in enumerate(zip(runs, data.signals, strict=True))
    ]
    designs = {k: v for f in fits for k, v in f[4].items()}
    total = sum(f[3] for f in fits)
    full = r_squared(sum(f[1] for f in fits), total)
    null = r_squared(sum(f[2] for f in fits), total)
    provenance = _provenance(
        data, selection, labels, alpha, assignment, designs, fractions
    )
    return SingleTrialResult(
        run_betas=tuple(f[0] for f in fits),
        _trial_table=trials,
        design=SelectedTrialDesign(
            selection.hrf_indices, designs, selection.provenance
        ),
        run_full_r2=tuple(r_squared(f[1], f[3]) for f in fits),
        run_nuisance_r2=tuple(r_squared(f[2], f[3]) for f in fits),
        full_r2=full,
        nuisance_r2=null,
        delta_r2=full - null,
        _diagnostics=tuple(d for f in fits for d in f[5]),
        ridge_alpha=alpha,
        provenance=provenance,
        ridge_fraction=fractions,
        run_ridge_alphas=None if fractions is None else tuple(f[6] for f in fits),
    )
