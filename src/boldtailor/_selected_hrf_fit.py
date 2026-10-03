"""Apply an identified HRF assignment, fitting only its feature groups."""

from dataclasses import dataclass
from hashlib import sha256
import json

import numpy as np
import pandas as pd

from boldtailor._hrf_cv import prepare_runs
from boldtailor._hrf_assignment import validate_selection
from boldtailor._hrf_design import HRF_NORMALIZATION, OVERSAMPLING
from boldtailor._single_trial_design import (
    _validate_events,
    check_trial_ids,
    trial_table,
)
from boldtailor._single_trial_fit import fit_trial_run, r_squared
from boldtailor._fractional_ridge import (
    regularization,
    fit_fraction_run,
    fraction_metadata,
)
from boldtailor.single_trial_results import SingleTrialResult, SelectedTrialDesign
from boldtailor.data import run_labels_for
from boldtailor.provenance import analysis_fingerprint, identity_activity
from boldtailor._fit_lifecycle import fit_operation


def _tables(data, labels):
    for events, times, label in zip(data.events, data.frame_times, labels, strict=True):
        _validate_events(events, times, label)
    trials = trial_table(data.events, labels)
    for run, confounds in enumerate(data.confounds):
        check_trial_ids(trials.trial_id[trials.run_index == run], confounds)
    return trials


def _hash_design(digest, run_index, cid, x, nuisance):
    """Feed one (run, HRF) design into the digest without retaining it."""
    values = np.column_stack([x, nuisance])
    digest.update(json.dumps([run_index, cid]).encode())
    digest.update(str(values.shape).encode())
    digest.update(np.asarray(values, dtype="<f8").tobytes())


def _fit_group(run, x, y, alpha, fractions):
    if fractions is None:
        return fit_trial_run(x, run.nuisance, y, alpha=alpha)
    return fit_fraction_run(x, run.nuisance, y, fractions=fractions)


@dataclass(frozen=True, kw_only=True)
class GroupRunFit:
    """One run's fits over every HRF group, assembled in feature order."""

    betas: np.ndarray
    full_sse: np.ndarray
    nuisance_sse: np.ndarray
    total_ss: np.ndarray
    diagnostics: list[dict]
    alphas: np.ndarray


def _fit_run(run, y, ids, alpha, run_index, digest, fractions=None) -> GroupRunFit:
    n_features = y.shape[1]
    betas = np.full((len(run.events), n_features), np.nan)
    full, null, total = (np.zeros(n_features) for _ in range(3))
    diagnostics = []
    alphas = np.full(n_features, np.nan)
    for cid in np.unique(ids[ids >= 0]):
        features = np.flatnonzero(ids == cid)
        x = run.trial_matrix(int(cid))
        part = None if fractions is None else fractions[features]
        fit = _fit_group(run, x, y[:, features], alpha, part)
        if fractions is not None:
            alphas[features] = fit.diagnostics["ridge_alphas"]
        betas[:, features] = fit.betas
        full[features] = fit.full_sse
        null[features] = fit.nuisance_sse
        total[features] = fit.total_ss
        _hash_design(digest, run_index, int(cid), x, run.nuisance)
        diagnostics.append(
            dict(
                run_index=run_index,
                hrf_id=int(cid),
                n_features=len(features),
                **fit.diagnostics,
            )
        )
    return GroupRunFit(
        betas=betas,
        full_sse=full,
        nuisance_sse=null,
        total_ss=total,
        diagnostics=diagnostics,
        alphas=alphas,
    )


def _fit_activity(data, selection, labels, alpha, assignment, digest, fractions=None):
    """Finish the design digest (fed run-major, HRF-ascending) and describe the fit."""
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
        hrf_normalization=HRF_NORMALIZATION,
        noise_model="ols",
        oversampling=OVERSAMPLING,
        selection=identity_activity(selection.provenance),
    )
    if fractions is not None:
        activity.update(fraction_metadata(fractions))
    return activity


def _rebuilder(runs, ids):
    """Rebuild fitted designs from the prepared runs' cached trial matrices."""
    fitted = frozenset(int(cid) for cid in np.unique(ids[ids >= 0]))

    def rebuild(run, hrf_id):
        valid_run = isinstance(run, (int, np.integer)) and 0 <= run < len(runs)
        if not valid_run or hrf_id not in fitted:
            raise KeyError(f"no fitted design for run {run!r}, HRF {hrf_id!r}")
        design = runs[run]
        return np.column_stack([design.trial_matrix(int(hrf_id)), design.nuisance])

    return rebuild


def fit_groups(
    data,
    *,
    selection,
    ridge_alpha,
    run_labels,
    feature_signature,
    ridge_fraction=None,
):
    with fit_operation("selected_hrf_single_trial", data.provenance) as operation:
        alpha, fractions = regularization(ridge_alpha, ridge_fraction, data.n_features)
        assignment = validate_selection(data, selection, feature_signature)
        labels = run_labels_for(data, run_labels)
        trials = _tables(data, labels)
        runs = prepare_runs(data, selection.library)
        digest = sha256()
        fits = [
            _fit_run(run, y, selection.hrf_indices, alpha, r, digest, fractions)
            for r, (run, y) in enumerate(zip(runs, data.signals, strict=True))
        ]
        total = sum(f.total_ss for f in fits)
        full = r_squared(sum(f.full_sse for f in fits), total)
        null = r_squared(sum(f.nuisance_sse for f in fits), total)
        activity = _fit_activity(
            data, selection, labels, alpha, assignment, digest, fractions
        )
        provenance = operation.provenance(
            activity,
            analysis_id=analysis_fingerprint(
                data.provenance.metadata_fingerprint, activity
            ),
        )
        return SingleTrialResult(
            run_betas=tuple(f.betas for f in fits),
            _trial_table=trials,
            design=SelectedTrialDesign(
                selection.hrf_indices,
                activity["design_fingerprint"],
                selection.provenance,
                _rebuilder(runs, selection.hrf_indices),
            ),
            run_full_r2=tuple(r_squared(f.full_sse, f.total_ss) for f in fits),
            run_nuisance_r2=tuple(r_squared(f.nuisance_sse, f.total_ss) for f in fits),
            full_r2=full,
            nuisance_r2=null,
            delta_r2=full - null,
            _diagnostics=tuple(d for f in fits for d in f.diagnostics),
            ridge_alpha=alpha,
            provenance=provenance,
            ridge_fraction=fractions,
            run_ridge_alphas=(
                None if fractions is None else tuple(f.alphas for f in fits)
            ),
        )
