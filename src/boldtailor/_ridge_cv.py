"""Training-only HRF selection and streamed candidate beta-series prediction."""

from dataclasses import dataclass
from hashlib import sha256
import json
from uuid import uuid4

import numpy as np

from boldtailor._hrf_cv import prepare_runs
from boldtailor._single_trial_design import compile_trial_run
from boldtailor._single_trial_fit import r_squared, prepare_trial_betas
from boldtailor._fractional_ridge import prepare_fraction_betas, NORM_BASIS
from boldtailor.data import from_arrays
from boldtailor.hrf_library import HrfLibrary
from boldtailor.hrf_selection import run_labels_for, select_hrf
from boldtailor.provenance import analysis_fingerprint, extend_provenance
from boldtailor.ridge_results import CandidateScores
from boldtailor.trial_encoding import (
    _predictor_arrays,
    _training_design,
    evaluate_trial_encoding,
    encoding_metadata,
    validate_encoding_mode,
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


def _validate(data, predictors, library, signature, *, encoding_mode):
    validate_encoding_mode(encoding_mode)
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
        _training_design(
            arrays,
            masks,
            [r for r in range(data.n_runs) if r != test],
            encoding_mode=encoding_mode,
        )
    return arrays, columns, masks


@dataclass(frozen=True)
class RunBetaPath:
    """Prepared HRF groups, placed back into their original feature columns."""

    shape: tuple[int, int]
    groups: tuple

    def betas_at(self, value):
        betas = np.full(self.shape, np.nan)
        for features, solver in self.groups:
            betas[:, features] = solver.betas_at(value)
        return betas


def prepare_run_beta_path(data, run_index, prepared, ids, label, *, fractional=False):
    """Prepare each selected HRF group once for this run and fold."""
    r = run_index
    groups = []
    prepare = prepare_fraction_betas if fractional else prepare_trial_betas
    if prepared is None:
        x, n, _ = compile_trial_run(
            data.events[r], data.frame_times[r], data.confounds[r], label
        )
    for cid in np.unique(ids[ids >= 0]):
        features = np.flatnonzero(ids == cid)
        if prepared is not None:
            x, n = prepared[r].trial_matrix(int(cid)), prepared[r].nuisance
        groups.append((features, prepare(x, n, data.signals[r][:, features])))
    return RunBetaPath((len(data.events[r]), data.n_features), tuple(groups))


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


def _score_fold(
    data,
    predictors,
    prepared,
    library,
    labels,
    signature,
    alphas,
    test,
    *,
    fractional=False,
    encoding_mode="within_run",
):
    train = [r for r in range(data.n_runs) if r != test]
    ids, selection_record = _fold_selection(data, library, labels, signature, train)
    paths = []
    for r in range(data.n_runs):
        try:
            paths.append(
                prepare_run_beta_path(
                    data, r, prepared, ids, labels[r], fractional=fractional
                )
            )
        except ValueError as error:
            raise ValueError(
                f"validation {labels[test]}, beta run {labels[r]}: {error}"
            ) from error
    fixed_target = None
    if fractional:
        try:
            fixed_target = paths[test].betas_at(1.0)
        except ValueError as error:
            raise ValueError(
                f"validation {labels[test]}, beta run {labels[test]}: {error}"
            ) from error
    losses, totals = [], []
    for alpha in alphas:
        betas = []
        for r, path in enumerate(paths):
            try:
                betas.append(
                    fixed_target if fractional and r == test else path.betas_at(alpha)
                )
            except ValueError as error:
                raise ValueError(
                    f"validation {labels[test]}, beta run {labels[r]}: {error}"
                ) from error
        fit = evaluate_trial_encoding(
            betas,
            predictors,
            train_runs=train,
            test_runs=[test],
            encoding_mode=encoding_mode,
        )
        losses.append(fit.run_sse[0])
        totals.append(fit.run_sst[0])
    record = dict(
        train=[labels[r] for r in train],
        validation=labels[test],
        predictor_means=fit.predictor_means.tolist(),
        train_run_predictor_means=fit.train_run_predictor_means.tolist(),
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


def _provenance(
    data,
    arrays,
    columns,
    labels,
    alphas,
    library,
    signature,
    folds,
    ids,
    *,
    fractional=False,
    encoding_mode="within_run",
):
    designs = []
    for e, t, n in zip(data.events, data.frame_times, data.confounds, strict=True):
        designs.extend([e[["onset", "duration"]], t, n])
    activity = dict(
        name="encoding_guided_ridge_cv",
        **encoding_metadata(encoding_mode),
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
        nuisance="unpenalized_run_specific",
        trial_masks=[np.isfinite(x).all(axis=1).tolist() for x in arrays],
    )
    if fractional:
        activity.update(
            name="encoding_guided_fractional_ridge_cv",
            validation_target="fixed_ols_betas",
            normalization="none_after_nuisance_projection",
            fraction_norm_basis=NORM_BASIS,
            fractions=activity.pop("alphas"),
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


def score_candidates(
    data,
    predictors,
    alphas,
    library,
    run_labels,
    feature_signature,
    *,
    fractional=False,
    encoding_mode="within_run",
):
    arrays, columns, masks = _validate(
        data, predictors, library, feature_signature, encoding_mode=encoding_mode
    )
    labels = run_labels_for(data, run_labels)
    prepared = None if library is None else prepare_runs(data, library)
    folds = [
        _score_fold(
            data,
            predictors,
            prepared,
            library,
            labels,
            feature_signature,
            alphas,
            test,
            fractional=fractional,
            encoding_mode=encoding_mode,
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
        fractional=fractional,
        encoding_mode=encoding_mode,
    )
    return CandidateScores(
        regularization="fractional_ridge" if fractional else "normalized_ridge",
        grid=alphas,
        cv_r2=r_squared(losses.sum(axis=0), totals.sum(axis=0)),
        fold_sse=losses,
        fold_sst=totals,
        fold_hrf_indices=ids,
        trial_masks=masks,
        run_labels=labels,
        provenance=provenance,
    )
