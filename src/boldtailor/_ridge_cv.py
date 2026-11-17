"""Training-only HRF selection and streamed candidate beta-series prediction."""

from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import json

import numpy as np

from boldtailor._fit_lifecycle import build_fit_provenance, fit_operation
from boldtailor._run_design import prepare_runs
from boldtailor.data import subset_runs
from boldtailor._single_trial_design import compile_trial_run
from boldtailor._regularizers import NORMALIZED_RIDGE, Regularizer
from boldtailor._single_trial_fit import r_squared
from boldtailor.hrf_library import HrfLibrary
from boldtailor.hrf_options import resolve_reml_options
from boldtailor.model import TaskModel
from boldtailor.data import run_labels_for
from boldtailor.hrf_selection import select_hrfs
from boldtailor.provenance import (
    analysis_fingerprint,
    identity_activity,
)
from boldtailor.ridge_results import CandidateScores
from boldtailor.trial_encoding import (
    EncodingDesign,
    encoding_metadata,
    predictor_arrays,
    validate_encoding_mode,
)


def _check_inputs(n_runs, library, signature, hrf_method):
    if library is not None and not isinstance(library, HrfLibrary):
        raise ValueError("library must be an HrfLibrary or None for canonical SPM")
    minimum = 2 if library is None or hrf_method == "reml" else 3
    if n_runs < minimum:
        raise ValueError(f"Ridge CV requires at least {minimum} runs")
    if signature is not None and (not isinstance(signature, str) or not signature):
        raise ValueError("feature_signature must be a nonempty string or None")


def _fold_designs(predictors, n_trials, encoding_mode):
    """One encoding design per held-out run, validated before any fitting."""
    arrays, _ = predictor_arrays(predictors, n_trials)
    designs = []
    for test, values in enumerate(arrays):
        if np.isfinite(values).all(axis=1).sum() < 2:
            raise ValueError(f"validation run {test} needs two complete predictor rows")
        designs.append(
            EncodingDesign.prepare(
                predictors,
                train_runs=[r for r in range(len(arrays)) if r != test],
                test_runs=[test],
                encoding_mode=encoding_mode,
                n_trials=n_trials,
            )
        )
    return designs


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


def run_tables(data):
    """Copy every run's events and confounds once for a per-run loop."""
    return tuple(
        zip(data.events, data.frame_times, data.confounds, data.signals, strict=True)
    )


def prepare_run_beta_path(
    data, run_index, prepared, ids, label, *, regularizer=NORMALIZED_RIDGE
):
    """Prepare each selected HRF group once for this run and fold."""
    tables = run_tables(data)[run_index]
    return _beta_path(tables, run_index, prepared, ids, label, regularizer)


def _beta_path(tables, r, prepared, ids, label, regularizer):
    events, times, confounds, y = tables
    groups = []
    if prepared is None:
        x, n, _ = compile_trial_run(events, times, confounds, label)
    for cid in np.unique(ids[ids >= 0]):
        features = np.flatnonzero(ids == cid)
        if prepared is not None:
            x, n = prepared[r].trial_matrix(int(cid)), prepared[r].nuisance
        groups.append((features, regularizer.prepare(x, n, y[:, features])))
    return RunBetaPath((len(events), y.shape[1]), tuple(groups))


def _fold_selection(data, library, labels, signature, train, selection_options):
    if library is None:
        return np.zeros(data.n_features, dtype=int), None
    selected = select_hrfs(
        subset_runs(data, train),
        library=library,
        run_labels=[labels[r] for r in train],
        feature_signature=signature,
        **selection_options,
    )
    return selected.hrf_indices, identity_activity(selected.provenance)


@dataclass(frozen=True, eq=False)
class _Scoring:
    """What every fold of one scoring call shares."""

    tables: tuple
    prepared: tuple | None
    library: HrfLibrary | None
    labels: tuple[str, ...]
    signature: str | None
    grid: tuple[float, ...]
    regularizer: Regularizer
    selection_options: dict


@contextmanager
def _named(validation, run):
    """Lead a beta-path error with the fold and the run it came from."""
    try:
        yield
    except ValueError as error:
        raise ValueError(f"validation {validation}, beta run {run}: {error}") from error


def _fold_paths(scoring, ids, validation):
    paths = []
    for r, (tables, label) in enumerate(zip(scoring.tables, scoring.labels)):
        with _named(validation, label):
            paths.append(
                _beta_path(tables, r, scoring.prepared, ids, label, scoring.regularizer)
            )
    return paths


def _held_out_target(scoring, paths, test):
    """The fixed held-out betas of a family that validates against them."""
    fixed = scoring.regularizer.fixed_target
    if fixed is None:
        return None
    with _named(scoring.labels[test], scoring.labels[test]):
        return paths[test].betas_at(fixed)


def _candidate_betas(scoring, paths, penalty, test, target):
    betas = []
    for r, path in enumerate(paths):
        with _named(scoring.labels[test], scoring.labels[r]):
            held_out = target is not None and r == test
            betas.append(target if held_out else path.betas_at(penalty))
    return betas


def _score_fold(data, scoring, design):
    """Losses of every candidate on one held-out run, with the fold's record."""
    (test,), train, labels = design.test_runs, list(design.train_runs), scoring.labels
    ids, selection_record = _fold_selection(
        data,
        scoring.library,
        labels,
        scoring.signature,
        train,
        scoring.selection_options,
    )
    paths = _fold_paths(scoring, ids, labels[test])
    target = _held_out_target(scoring, paths, test)
    losses, totals = [], []
    for penalty in scoring.grid:
        fit = design.score(_candidate_betas(scoring, paths, penalty, test, target))
        losses.append(fit.run_sse[0])
        totals.append(fit.run_sst[0])
    record = dict(
        train=[labels[r] for r in train],
        validation=labels[test],
        predictor_means=design.predictor_means.tolist(),
        train_run_predictor_means=design.train_run_predictor_means.tolist(),
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


def _design_fingerprint(tables):
    designs, confound_names = [], []
    for events, times, confounds, _ in tables:
        designs.extend([events[["onset", "duration"]], times, confounds])
        confound_names.append(list(confounds))
    return _fingerprint(designs, confound_names)


def _activity(scoring, design, folds, ids, encoding_mode):
    """The scoring activity; the family supplies its own descriptors."""
    family, arrays = scoring.regularizer, design.predictor_values
    library = scoring.library
    activity = dict(
        name=family.cv_activity["name"],
        **encoding_metadata(encoding_mode),
        validation_target=family.validation_target,
        selection_statistic=True,
        alphas=list(scoring.grid),
        run_labels=list(scoring.labels),
        folds=folds,
        predictor_names=list(design.predictor_names),
        predictor_fingerprint=_fingerprint(arrays, design.predictor_names[1:]),
        design_fingerprint=_design_fingerprint(scoring.tables),
        hrf_assignment_fingerprint=sha256(
            np.asarray(ids, dtype="<i8").tobytes()
        ).hexdigest(),
        feature_signature=scoring.signature,
        library_fingerprint=None if library is None else library.fingerprint,
        hrf_model="spm" if library is None else "inner_training_selected",
        normalization=family.cv_activity["normalization"],
        beta_units="native_signal",
        nuisance="unpenalized_run_specific",
        trial_masks=[mask.tolist() for mask in design.trial_masks],
    )
    extras = {k: v for k, v in family.cv_activity.items() if k not in activity}
    activity.update(extras)
    if family.grid_name != "alphas":
        activity[family.grid_name] = activity.pop("alphas")
    return activity


def _selection_options(library, hrf_method, reml_options, task_model):
    options = resolve_reml_options(hrf_method, reml_options)
    if library is None and hrf_method != "cv":
        raise ValueError("HRF method requires a library")
    if not isinstance(task_model, TaskModel):
        raise ValueError("task_model must be a TaskModel")
    return dict(method=hrf_method, reml_options=options, task_model=task_model)


def score_candidates(
    data,
    predictors,
    grid,
    library,
    run_labels,
    feature_signature,
    *,
    regularizer=NORMALIZED_RIDGE,
    encoding_mode="within_run",
    hrf_method="cv",
    reml_options=None,
    task_model=TaskModel(),
):
    """Leave-one-run-out encoding scores of every candidate in ``grid``.

    Runs as one logged operation, so invalid arguments and late failures are
    reported like those of every other fit entry point.
    """
    with fit_operation(regularizer.cv_activity["name"], data.provenance) as operation:
        grid = regularizer.grid(grid)
        options = _selection_options(library, hrf_method, reml_options, task_model)
        validate_encoding_mode(encoding_mode)
        _check_inputs(data.n_runs, library, feature_signature, hrf_method)
        tables = run_tables(data)
        designs = _fold_designs(predictors, [len(t[0]) for t in tables], encoding_mode)
        scoring = _Scoring(
            tables=tables,
            prepared=None if library is None else prepare_runs(data, library),
            library=library,
            labels=run_labels_for(data, run_labels),
            signature=feature_signature,
            grid=grid,
            regularizer=regularizer,
            selection_options=options,
        )
        return _scores(operation, data, scoring, designs, encoding_mode)


def _scores(operation, data, scoring, designs, encoding_mode):
    """Score every fold, then record the activity under the operation's identity."""
    folds = [_score_fold(data, scoring, design) for design in designs]
    losses, totals = np.array([f[0] for f in folds]), np.array([f[1] for f in folds])
    ids = np.array([f[2] for f in folds])
    activity = _activity(scoring, designs[0], [f[3] for f in folds], ids, encoding_mode)
    operation.analysis_id = analysis_fingerprint(
        data.provenance.metadata_fingerprint, activity
    )
    return CandidateScores(
        regularization=scoring.regularizer.name,
        grid=scoring.grid,
        cv_r2=r_squared(losses.sum(axis=0), totals.sum(axis=0)),
        fold_sse=losses,
        fold_sst=totals,
        fold_hrf_indices=ids,
        trial_masks=designs[0].trial_masks,
        run_labels=scoring.labels,
        provenance=build_fit_provenance(operation, activity),
    )
