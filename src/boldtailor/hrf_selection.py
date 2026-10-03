"""Select HRFs by held-out task-model prediction across runs, without repeated images."""

from collections.abc import Sequence
from hashlib import sha256

import numpy as np

from boldtailor._deprecation import deprecated
from boldtailor._fit_lifecycle import fit_operation
from boldtailor._hrf_cv import (
    prepare_runs,  # public re-export
    subset_runs,  # public re-export
    signal_statistics,
    loro_scores,
    choose_eligible,
    pooled_amplitude,
    prediction_loss,
)
from boldtailor._hrf_design import HRF_NORMALIZATION, MIN_ONSET, OVERSAMPLING
from boldtailor._scalars import is_integer
from boldtailor.data import AnalysisData, run_labels_for
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
    if not is_integer(batch) or batch < 1:
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
        hrf_indices=indices,
        cv_r2=selected,
        canonical_cv_r2=canonical,
        delta_cv_r2=selected - canonical,
        library=library,
        _eligibility=eligibility,
        run_labels=labels,
        feature_signature=signature,
        provenance=provenance,
        task_model=task_model,
    )


def select_hrfs(
    data: AnalysisData,
    *,
    library: HrfLibrary,
    run_labels: Sequence[str] | None = None,
    feature_signature: str | None = None,
    candidate_batch_size: int = 32,
    task_model: TaskModel = TaskModel(),
) -> HrfSelectionResult:
    """Choose each feature's HRF by leave-one-run-out task-model prediction.

    This is a selection statistic. Confounds and missing-value indicators are
    profiled in each run; task-model amplitudes are learned only from other
    runs. Anonymous arrays without feature_signature require the caller to
    preserve feature order.
    """
    with fit_operation("hrf_selection", data.provenance) as operation:
        return _select_hrfs(
            operation,
            data,
            library,
            run_labels,
            feature_signature,
            candidate_batch_size,
            task_model,
        )


def select_hrf(data: AnalysisData, **options) -> HrfSelectionResult:
    """Deprecated alias of :func:`select_hrfs`, kept for one release."""
    deprecated("select_hrf is deprecated; use select_hrfs", stacklevel=3)
    return select_hrfs(data, **options)


def _select_hrfs(
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
        or any(not is_integer(i) or not 0 <= i < n_runs for i in values)
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
    data: AnalysisData,
    *,
    library: HrfLibrary,
    train_runs: Sequence[int],
    test_runs: Sequence[int],
    run_labels: Sequence[str] | None = None,
    feature_signature: str | None = None,
    task_model: TaskModel = TaskModel(),
    candidate_batch_size: int = 32,
) -> HrfEvaluationResult:
    """Select within training runs, then freeze HRF and amplitudes for test runs.

    ``candidate_batch_size`` bounds memory exactly as in :func:`select_hrfs`;
    it never changes the result.
    """
    with fit_operation("hrf_independent_evaluation", data.provenance) as operation:
        return _evaluate_split(
            operation,
            data,
            library,
            (train_runs, test_runs),
            run_labels,
            feature_signature,
            task_model,
            candidate_batch_size,
        )


def _train_selection(data, runs, library, labels, signature, task_model, split, batch):
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
            batch,
            tuple(runs[i] for i in (*train, *test)),
            task_model,
        )


def _evaluate_split(
    operation, data, library, split, run_labels, feature_signature, task_model, batch
):
    _validate(library, feature_signature, batch, task_model)
    train = _fold_indices(split[0], data.n_runs, 2, "train_runs")
    test = _fold_indices(split[1], data.n_runs, 1, "test_runs")
    if set(train) & set(test):
        raise ValueError("training and test runs must be disjoint")
    labels = run_labels_for(data, run_labels)
    runs = prepare_runs(data, library, task_model)
    selection = _train_selection(
        data, runs, library, labels, feature_signature, task_model, (train, test), batch
    )
    scores = _held_out_scores(data, runs, selection, task_model, (train, test), batch)
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
    amplitudes, chosen, canonical, delta = scores
    return HrfEvaluationResult(
        training_selection=selection,
        training_amplitudes=amplitudes,
        test_r2=chosen,
        canonical_test_r2=canonical,
        delta_test_r2=delta,
        train_runs=train,
        test_runs=test,
        provenance=provenance,
        amplitude_names=task_model.regressor_names,
    )


def _held_out_scores(data, runs, selection, task_model, split, batch):
    train, test = split
    a, b, c, energy = signal_statistics(runs, data.signals, batch)
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
