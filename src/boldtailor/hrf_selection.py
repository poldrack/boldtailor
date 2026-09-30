"""Select HRFs by mean-stimulus prediction across runs, without repeated images."""

from hashlib import sha256
from numbers import Integral
import re
from uuid import uuid4

import numpy as np

from boldtailor._hrf_cv import (
    prepare_runs,
    signal_statistics,
    loro_scores,
    choose_eligible,
    pooled_amplitude,
    prediction_loss,
)
from boldtailor.hrf_library import HrfLibrary
from boldtailor.hrf_results import HrfSelectionResult, HrfEvaluationResult
from boldtailor.provenance import analysis_fingerprint, extend_provenance


def run_labels_for(data, run_labels):
    labels = (
        tuple(run_labels)
        if run_labels is not None
        else tuple(f"run-{i+1:02d}" for i in range(data.n_runs))
    )
    if (
        len(labels) != data.n_runs
        or len(set(labels)) != len(labels)
        or not all(
            isinstance(x, str) and re.fullmatch(r"[A-Za-z0-9_-]+", x) for x in labels
        )
    ):
        raise ValueError("run labels must be unique valid labels matching the runs")
    return labels


def _validate(library, signature, batch):
    if not isinstance(library, HrfLibrary):
        raise ValueError("library must be an HrfLibrary")
    if signature is not None and (not isinstance(signature, str) or not signature):
        raise ValueError("feature_signature must be a nonempty string or None")
    if (
        isinstance(batch, (bool, np.bool_))
        or not isinstance(batch, Integral)
        or batch < 1
    ):
        raise ValueError("candidate_batch_size must be a positive integer")


def _provenance(data, runs, library, labels, signature, name, **extra):
    activity = dict(
        name=name,
        library_fingerprint=library.fingerprint,
        design_fingerprint=sha256(
            "|".join(r.fingerprint for r in runs).encode()
        ).hexdigest(),
        run_labels=list(labels),
        feature_signature=signature,
        score="nuisance_adjusted_mean_stimulus_prediction_r2",
        beta_units="native_signal",
        oversampling=50,
        nuisance="conditional_projection_of_each_run",
        invalid_features="zero signal outside nuisance span at numerical precision",
        sse_roundoff_tolerance="64 * eps * (C + abs(2*b*B) + abs(b*b*A))",
        **extra,
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


def _select(data, runs, signals, library, labels, signature, batch, eligibility_runs):
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
    provenance = _provenance(
        data,
        runs,
        library,
        labels,
        signature,
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
    )


def select_hrf(
    data, *, library, run_labels=None, feature_signature=None, candidate_batch_size=32
):
    """Choose each feature's HRF with leave-one-run-out mean-stimulus prediction.

    This is a selection statistic. Nuisance coefficients are profiled in each
    run; task amplitudes are learned only from other runs. Anonymous arrays
    without feature_signature require the caller to preserve feature order.
    """
    _validate(library, feature_signature, candidate_batch_size)
    if data.n_runs < 2:
        raise ValueError("HRF selection requires at least two runs")
    labels = run_labels_for(data, run_labels)
    runs = prepare_runs(data, library)
    return _select(
        data,
        runs,
        data.signals,
        library,
        labels,
        feature_signature,
        candidate_batch_size,
        runs,
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
    amplitude, ok = pooled_amplitude(a[list(train)].sum(axis=0), b[list(train)].sum(axis=0))
    loss = sum(prediction_loss(a[r], b[r], c[r], amplitude) for r in test)
    total = energy[list(test)].sum(axis=0)
    score = np.full_like(loss, np.nan)
    np.divide(loss, total[None, :], out=score, where=total[None, :] > 0)
    score = 1 - score
    score[~ok] = np.nan
    return amplitude[:, 0, :], score


def evaluate_hrf_split(
    data, *, library, train_runs, test_runs, run_labels=None, feature_signature=None
):
    """Select within training runs, then freeze HRF and amplitude for test runs."""
    _validate(library, feature_signature, 32)
    train = _fold_indices(train_runs, data.n_runs, 2, "train_runs")
    test = _fold_indices(test_runs, data.n_runs, 1, "test_runs")
    if set(train) & set(test):
        raise ValueError("training and test runs must be disjoint")
    labels = run_labels_for(data, run_labels)
    runs = prepare_runs(data, library)
    selection = _select(
        data,
        tuple(runs[i] for i in train),
        tuple(data.signals[i] for i in train),
        library,
        tuple(labels[i] for i in train),
        feature_signature,
        32,
        tuple(runs[i] for i in (*train, *test)),
    )
    a, b, c, energy = signal_statistics(runs, data.signals, 32)
    amplitudes, scores = _predict(a, b, c, energy, train, test)
    ids = selection.hrf_indices
    valid = ids >= 0
    chosen = np.full(data.n_features, np.nan)
    coefficient = chosen.copy()
    chosen[valid] = scores[ids[valid], np.flatnonzero(valid)]
    coefficient[valid] = amplitudes[ids[valid], np.flatnonzero(valid)]
    canonical = scores[0].copy()
    if not selection.eligibility.loc[0, "eligible"]:
        canonical[:] = np.nan
    canonical[~valid] = np.nan
    provenance = _provenance(
        data,
        runs,
        library,
        labels,
        feature_signature,
        "hrf_independent_evaluation",
        train_runs=list(train),
        test_runs=list(test),
        training_selection=selection.provenance.to_dict()["activities"][-1],
        frozen_task_amplitudes=True,
    )
    return HrfEvaluationResult(
        selection,
        coefficient,
        chosen,
        canonical,
        chosen - canonical,
        train,
        test,
        provenance,
    )
