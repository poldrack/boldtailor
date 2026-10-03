"""Select HRFs by held-out task-model prediction across runs, without repeated images."""

from hashlib import sha256
from numbers import Integral

import numpy as np

from boldtailor._fit_lifecycle import fit_operation
from boldtailor._hrf_cv import (
    MIN_ONSET,
    OVERSAMPLING,
    prepare_runs,
    signal_statistics,
    loro_scores,
    choose_eligible,
    pooled_amplitude,
    prediction_loss,
)
from boldtailor._hrf_design import HRF_NORMALIZATION
from boldtailor.data import run_labels_for
from boldtailor.hrf_library import HrfLibrary
from boldtailor.hrf_results import HrfSelectionResult, HrfEvaluationResult
from boldtailor.model import TaskModel
from boldtailor.provenance import analysis_fingerprint, identity_activity


def _validate(library, signature, batch, task_model):
    if not isinstance(library, HrfLibrary):
        raise ValueError("library must be an HrfLibrary")
    if not isinstance(task_model, TaskModel):
        raise ValueError("task_model must be a TaskModel")
    if signature is not None and (not isinstance(signature, str) or not signature):
        raise ValueError("feature_signature must be a nonempty string or None")
    if (
        isinstance(batch, (bool, np.bool_))
        or not isinstance(batch, Integral)
        or batch < 1
    ):
        raise ValueError("candidate_batch_size must be a positive integer")


def _activity(runs, library, labels, signature, task_model, name, **extra):
    activity = dict(
        name=name,
        library=dict(library.origin),
        library_fingerprint=library.fingerprint,
        design_fingerprint=sha256(
            "|".join(r.fingerprint for r in runs).encode()
        ).hexdigest(),
        run_labels=list(labels),
        feature_signature=signature,
        task_model=task_model.to_dict(),
        task_model_fingerprint=task_model.fingerprint,
        task_regressors=list(task_model.regressor_names),
        profiled_regressors=list(task_model.profiled_names),
        profiled_columns="fit in-sample per run with the candidate kernel; see user guide",
        score="nuisance_adjusted_task_model_prediction_r2",
        beta_units="native_signal",
        hrf_normalization=HRF_NORMALIZATION,
        oversampling=OVERSAMPLING,
        min_onset=MIN_ONSET,
        nuisance="conditional_projection_of_confounds_and_profiled_task_columns_per_run",
        invalid_features="zero signal outside nuisance span at numerical precision",
        sse_roundoff_tolerance=(
            "64 * eps * (C + sum_k abs(2*beta_k*B_k) + abs(beta' A beta))"
        ),
        **extra,
    )
    return activity


def _provenance(operation, data, activity):
    analysis_id = analysis_fingerprint(data.provenance.metadata_fingerprint, activity)
    return operation.provenance(activity, analysis_id=analysis_id)


def _at_parameter_bound(library, indices, margin=0.02):
    """Flag custom picks within ``margin`` of the box width of any edge.

    Canonical (0) and ineligible (-1) features are False. A zero-width
    parameter range counts as at the bound.
    """
    bounds = library.parameter_bounds
    low, high = bounds["low"].to_numpy(), bounds["high"].to_numpy()
    tol = margin * (high - low)
    params = np.array([c.parameters[:6] for c in library.candidates], dtype=float)
    near = (params - low <= tol) | (high - params <= tol)
    flags = near.any(axis=1) & np.array([c.kind != "spm" for c in library.candidates])
    return flags[np.maximum(indices, 0)] & (indices > 0)


def _select(
    operation,
    data,
    runs,
    signals,
    library,
    labels,
    signature,
    batch,
    eligibility_runs,
    task_model,
):
    a, b, c, energy = signal_statistics(runs, signals, batch)
    indices, scores, eligibility = choose_eligible(
        loro_scores(a, b, c, energy), eligibility_runs
    )
    selected = np.full(data.n_features, np.nan)
    valid = indices >= 0
    selected[valid] = scores[indices[valid], np.flatnonzero(valid)]
    canonical = scores[0].copy()
    canonical[~np.isfinite(canonical)] = np.nan
    canonical[~valid] = np.nan
    activity = _activity(
        runs,
        library,
        labels,
        signature,
        task_model,
        "hrf_selection",
        folds=[
            dict(
                train=[label for j, label in enumerate(labels) if j != i], test=[label]
            )
            for i, label in enumerate(labels)
        ],
        selection_statistic=True,
        hrf_assignment_fingerprint=sha256(indices.astype("<i8").tobytes()).hexdigest(),
    )
    provenance = _provenance(operation, data, activity)
    return HrfSelectionResult(
        indices,
        selected,
        canonical,
        selected - canonical,
        library,
        eligibility,
        labels,
        signature,
        provenance,
        task_model=task_model,
        at_parameter_bound=_at_parameter_bound(library, indices),
    )


def select_hrf(
    data,
    *,
    library,
    run_labels=None,
    feature_signature=None,
    candidate_batch_size=32,
    task_model=TaskModel(),
):
    """Choose each feature's HRF by leave-one-run-out task-model prediction.

    This is a selection statistic. Confounds and missing-value indicators are
    profiled in each run; task-model amplitudes are learned only from other
    runs. Anonymous arrays without feature_signature require the caller to
    preserve feature order.
    """
    with fit_operation("hrf_selection", data.provenance) as operation:
        return _select_hrf(
            operation,
            data,
            library,
            run_labels,
            feature_signature,
            candidate_batch_size,
            task_model,
        )


def _select_hrf(
    operation, data, library, run_labels, feature_signature, batch, task_model
):
    _validate(library, feature_signature, batch, task_model)
    if data.n_runs < 2:
        raise ValueError("HRF selection requires at least two runs")
    labels = run_labels_for(data, run_labels)
    runs = prepare_runs(data, library, task_model)
    return _select(
        operation,
        data,
        runs,
        data.signals,
        library,
        labels,
        feature_signature,
        batch,
        runs,
        task_model,
    )


def _fold_indices(values, n_runs, minimum, name):
    values = tuple(values)
    if (
        len(values) < minimum
        or len(set(values)) != len(values)
        or any(
            isinstance(i, (bool, np.bool_))
            or not isinstance(i, Integral)
            or not 0 <= i < n_runs
            for i in values
        )
    ):
        raise ValueError(f"{name} requires at least {minimum} unique valid run indices")
    return tuple(int(i) for i in values)


def _predict(a, b, c, energy, train, test):
    amplitude, ok = pooled_amplitude(
        a[list(train)].sum(axis=0), b[list(train)].sum(axis=0)
    )
    loss = sum(prediction_loss(a[r], b[r], c[r], amplitude) for r in test)
    total = energy[list(test)].sum(axis=0)
    score = np.full_like(loss, np.nan)
    np.divide(loss, total[None, :], out=score, where=total[None, :] > 0)
    score = 1 - score
    score[~ok] = np.nan
    return amplitude, score


def evaluate_hrf_split(
    data,
    *,
    library,
    train_runs,
    test_runs,
    run_labels=None,
    feature_signature=None,
    task_model=TaskModel(),
):
    """Select within training runs, then freeze HRF and amplitudes for test runs."""
    with fit_operation("hrf_independent_evaluation", data.provenance) as operation:
        return _evaluate_split(
            operation,
            data,
            library,
            (train_runs, test_runs),
            run_labels,
            feature_signature,
            task_model,
        )


def _train_selection(data, runs, library, labels, signature, task_model, split):
    train, test = split
    with fit_operation("hrf_selection", data.provenance) as operation:
        return _select(
            operation,
            data,
            tuple(runs[i] for i in train),
            tuple(data.signals[i] for i in train),
            library,
            tuple(labels[i] for i in train),
            signature,
            32,
            tuple(runs[i] for i in (*train, *test)),
            task_model,
        )


def _evaluate_split(
    operation, data, library, split, run_labels, feature_signature, task_model
):
    _validate(library, feature_signature, 32, task_model)
    train = _fold_indices(split[0], data.n_runs, 2, "train_runs")
    test = _fold_indices(split[1], data.n_runs, 1, "test_runs")
    if set(train) & set(test):
        raise ValueError("training and test runs must be disjoint")
    labels = run_labels_for(data, run_labels)
    runs = prepare_runs(data, library, task_model)
    selection = _train_selection(
        data, runs, library, labels, feature_signature, task_model, (train, test)
    )
    scores = _held_out_scores(data, runs, selection, task_model, train, test)
    activity = _activity(
        runs,
        library,
        labels,
        feature_signature,
        task_model,
        "hrf_independent_evaluation",
        train_runs=list(train),
        test_runs=list(test),
        training_selection=identity_activity(selection.provenance),
        frozen_task_amplitudes=True,
    )
    provenance = _provenance(operation, data, activity)
    return HrfEvaluationResult(
        selection,
        *scores,
        train,
        test,
        provenance,
        amplitude_names=task_model.regressor_names,
    )


def _held_out_scores(data, runs, selection, task_model, train, test):
    a, b, c, energy = signal_statistics(runs, data.signals, 32)
    amplitudes, scores = _predict(a, b, c, energy, train, test)
    ids = selection.hrf_indices
    valid = np.flatnonzero(ids >= 0)
    chosen = np.full(data.n_features, np.nan)
    coefficient = np.full((len(task_model.regressor_names), data.n_features), np.nan)
    chosen[valid] = scores[ids[valid], valid]
    coefficient[:, valid] = amplitudes[ids[valid], :, valid].T
    canonical = scores[0].copy()
    if not selection.eligibility.loc[0, "eligible"]:
        canonical[:] = np.nan
    canonical[ids < 0] = np.nan
    return coefficient, chosen, canonical, chosen - canonical
