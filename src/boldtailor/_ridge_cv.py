"""Training-only HRF selection and streamed candidate beta-series prediction."""

from hashlib import sha256
import json
from uuid import uuid4

import numpy as np

from boldtailor._hrf_cv import prepare_runs
from boldtailor._single_trial_design import compile_trial_run
from boldtailor._single_trial_fit import r_squared, trial_beta_path
from boldtailor.data import from_arrays
from boldtailor.hrf_library import HrfLibrary
from boldtailor.hrf_selection import run_labels_for, select_hrf
from boldtailor.provenance import analysis_fingerprint, extend_provenance
from boldtailor.ridge_results import RidgeCandidateScores
from boldtailor.trial_encoding import (
    _predictor_arrays,
    _training_design,
    evaluate_trial_encoding,
)


def subset_runs(data, indices):
    """Subset run data and source records together without relabeling features."""
    return from_arrays(
        [data.signals[i] for i in indices],
        [data.events[i] for i in indices],
        frame_times=[data.frame_times[i] for i in indices],
        confounds=[data.confounds[i] for i in indices],
        sources=[data.provenance.sources[i] for i in indices],
    )


def _validate(data, predictors, library, signature):
    if library is not None and not isinstance(library, HrfLibrary):
        raise ValueError("library must be an HrfLibrary or None for canonical SPM")
    minimum = 2 if library is None else 3
    if data.n_runs < minimum:
        raise ValueError(f"Ridge CV requires at least {minimum} runs")
    if signature is not None and (not isinstance(signature, str) or not signature):
        raise ValueError("feature_signature must be a nonempty string or None")
    arrays, columns = _predictor_arrays(predictors, [len(e) for e in data.events])
    masks = tuple(np.isfinite(x).all(axis=1) for x in arrays)
    for test in range(data.n_runs):
        if masks[test].sum() < 2:
            raise ValueError(f"validation run {test} needs two complete predictor rows")
        _training_design(arrays, masks, [r for r in range(data.n_runs) if r != test])
    return arrays, columns, masks


def _run_beta_path(data, r, prepared, ids, alphas, label):
    groups = []
    if prepared is None:
        x, n, _ = compile_trial_run(
            data.events[r], data.frame_times[r], data.confounds[r], label
        )
    for cid in np.unique(ids[ids >= 0]):
        features = np.flatnonzero(ids == cid)
        if prepared is not None:
            x, n = prepared[r].trial_matrix(int(cid)), prepared[r].nuisance
        path = trial_beta_path(x, n, data.signals[r][:, features], alphas=alphas)
        groups.append((features, path))
    for alpha in alphas:
        betas = np.full((len(data.events[r]), data.n_features), np.nan)
        for features, path in groups:
            _, values = next(path)
            betas[:, features] = values
        yield betas


def _fold_selection(data, library, labels, signature, train):
    if library is None:
        return np.zeros(data.n_features, dtype=int), None
    selected = select_hrf(
        subset_runs(data, train),
        library=library,
        run_labels=[labels[r] for r in train],
        feature_signature=signature,
    )
    return selected.hrf_indices, selected.provenance.to_dict()["activities"][-1]


def _score_fold(data, predictors, prepared, library, labels, signature, alphas, test):
    train = [r for r in range(data.n_runs) if r != test]
    ids, selection_record = _fold_selection(data, library, labels, signature, train)
    paths = [
        _run_beta_path(data, r, prepared, ids, alphas, labels[r])
        for r in range(data.n_runs)
    ]
    losses, totals = [], []
    for alpha in alphas:
        betas = []
        for r, path in enumerate(paths):
            try:
                betas.append(next(path))
            except ValueError as error:
                raise ValueError(
                    f"validation {labels[test]}, beta run {labels[r]}: {error}"
                ) from error
        fit = evaluate_trial_encoding(
            betas, predictors, train_runs=train, test_runs=[test]
        )
        losses.append(fit.run_sse[0])
        totals.append(fit.run_sst[0])
    record = dict(
        train=[labels[r] for r in train],
        validation=labels[test],
        predictor_means=fit.predictor_means.tolist(),
        hrf_selection=selection_record,
    )
    return losses, totals, ids, record


def _fingerprint(arrays, names):
    digest = sha256(json.dumps(list(names)).encode())
    for array in arrays:
        value = np.asarray(array, dtype="<f8")
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _provenance(data, arrays, columns, labels, alphas, library, signature, folds, ids):
    designs = []
    for e, t, n in zip(data.events, data.frame_times, data.confounds, strict=True):
        designs.extend([e[["onset", "duration"]], t, n])
    activity = dict(
        name="encoding_guided_ridge_cv",
        score="pooled_within_run_trial_encoding_r2",
        validation_target="candidate_regularized_betas",
        selection_statistic=True,
        alphas=list(alphas),
        run_labels=list(labels),
        folds=folds,
        predictor_names=["task", *columns],
        predictor_fingerprint=_fingerprint(arrays, columns),
        design_fingerprint=_fingerprint(designs, [list(n) for n in data.confounds]),
        hrf_assignment_fingerprint=sha256(
            np.asarray(ids, dtype="<i8").tobytes()
        ).hexdigest(),
        feature_signature=signature,
        library_fingerprint=None if library is None else library.fingerprint,
        hrf_model="spm" if library is None else "inner_training_selected",
        normalization="unit_l2_after_nuisance_projection",
        beta_units="native_signal",
        predictor_transform="center_on_pooled_complete_training_trials",
        nuisance="unpenalized_run_specific",
        encoding_model="ols_with_shared_intercept",
        trial_masks=[np.isfinite(x).all(axis=1).tolist() for x in arrays],
    )
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


def score_candidates(data, predictors, alphas, library, run_labels, feature_signature):
    arrays, columns, masks = _validate(data, predictors, library, feature_signature)
    labels = run_labels_for(data, run_labels)
    prepared = None if library is None else prepare_runs(data, library)
    folds = [
        _score_fold(
            data, predictors, prepared, library, labels, feature_signature, alphas, test
        )
        for test in range(data.n_runs)
    ]
    losses, totals = np.array([f[0] for f in folds]), np.array([f[1] for f in folds])
    ids = np.array([f[2] for f in folds])
    provenance = _provenance(
        data,
        arrays,
        columns,
        labels,
        alphas,
        library,
        feature_signature,
        [f[3] for f in folds],
        ids,
    )
    return RidgeCandidateScores(
        alphas,
        r_squared(losses.sum(axis=0), totals.sum(axis=0)),
        losses,
        totals,
        ids,
        masks,
        labels,
        provenance,
    )
