"""Opt-in task-guided denoising: choose a temporal-PC count, then append PCs.

``select_denoising`` chooses one PC count by leave-one-run-out held-out task
prediction (selection statistic, not a performance estimate). It then builds
the noise pool from an initial HRF selection on all runs, scored by the
indicator-consistent leave-one-run-out task-model R², and returns that
pool's leading run-specific PCs. By default (``pool_r2_threshold="auto"``)
each pool threshold is GLMsingle's two-component Gaussian-mixture tail
threshold, fitted separately in every fold and for the final pool. The pool is not iterated after adding PCs.
``with_denoising`` appends those PCs to the baseline confounds of the same
analysis for ordinary HRF selection and fitting.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from hashlib import sha256

import numpy as np
import pandas as pd

from boldtailor._denoising_cv import (
    select_component_count,
    validate_counts,
    validate_tolerance,
)
from boldtailor._denoising_identity import check_identity, run_identities
from boldtailor._denoising_pool import (
    analysis_components,
    pool_masks,
    pool_statistic,
    validate_feature_mask,
    validate_threshold,
)
from boldtailor._fit_lifecycle import fit_operation
from boldtailor.data import AnalysisData, run_labels_for
from boldtailor.denoising_results import (
    DenoisingFold,
    DenoisingResult,
    PcaDiagnostics,
)
from boldtailor.hrf_library import HrfLibrary, default_hrf_library
from boldtailor.hrf_selection import select_hrfs
from boldtailor.model import TaskModel
from boldtailor.provenance import (
    RunSources,
    SourceRef,
    analysis_fingerprint,
    identity_activity,
)

_AUGMENTATION = "denoising_augmentation"


# ---- validation and labels --------------------------------------------------------


def _check_data(data):
    if not isinstance(data, AnalysisData):
        raise ValueError("data must be an AnalysisData")


@dataclass(frozen=True)
class _Settings:
    """Validated, owned selection settings."""

    brain_mask: np.ndarray
    task_model: TaskModel
    library: HrfLibrary
    counts: tuple[int, ...]
    threshold: float | str
    tolerance: float
    feature_signature: str | None


def _check_models(task_model, library, signature):
    if not isinstance(task_model, TaskModel):
        raise ValueError("task_model must be a TaskModel")
    if not isinstance(library, HrfLibrary):
        raise ValueError("library must be an HrfLibrary")
    if signature is not None and (not isinstance(signature, str) or not signature):
        raise ValueError("feature_signature must be a nonempty string or None")


def _settings(data, brain_mask, task_model, library, grid, signature):
    """Validate everything before any fitting; ``grid`` is (counts, r2, tol)."""
    _check_data(data)
    _check_models(task_model, library, signature)
    counts, threshold, tolerance = grid
    return _Settings(
        brain_mask=validate_feature_mask(brain_mask, data.n_features),
        task_model=task_model,
        library=library,
        counts=validate_counts(counts),
        threshold=validate_threshold(threshold),
        tolerance=validate_tolerance(tolerance),
        feature_signature=signature,
    )


def _digest(values, dtype) -> str:
    return sha256(np.ascontiguousarray(values, dtype=dtype).tobytes()).hexdigest()


# ---- count selection ----------------------------------------------------------------


def _count_selection(data, labels, settings):
    return select_component_count(
        data,
        brain_mask=settings.brain_mask,
        task_model=settings.task_model,
        library=settings.library,
        counts=settings.counts,
        threshold=settings.threshold,
        tolerance=settings.tolerance,
        run_labels=labels,
    )


def _public_tables(selection):
    """Aggregate scores and per-fold scores keyed by the held-out run label."""
    folds = selection.fold_scores.copy()
    folds["validation_run"] = folds.pop("validation_label")
    return selection.scores, folds


def _diagnostics(components, labels, pool_size):
    return PcaDiagnostics(
        run_labels=labels,
        pool_size=pool_size,
        ranks=[c.rank for c in components],
        singular_values=[c.singular_values for c in components],
        rank_tolerances=[c.rank_tolerance for c in components],
        retained_columns=[c.retained_columns for c in components],
    )


def _fold_result(fold, labels):
    training = tuple(labels[r] for r in fold.training_runs)
    return DenoisingFold(
        validation_run=labels[fold.validation_run],
        training_runs=training,
        pool=fold.masks.pool,
        scoring=fold.masks.scoring,
        scored=fold.scored,
        zero_target=fold.zero_target,
        pool_r2=fold.pool_statistic,
        hrf_indices=fold.selection.hrf_indices,
        pool_threshold=fold.masks.threshold,
        pool_mixture=fold.masks.mixture,
        components=_diagnostics(fold.components, training, fold.masks.pool_size),
    )


# ---- final full-data pool -------------------------------------------------------------


def _final_pool(data, labels, settings):
    """Initial selection, pool statistic, masks, and PCA from all runs."""
    selection = select_hrfs(
        data,
        library=settings.library,
        task_model=settings.task_model,
        run_labels=labels,
        feature_signature=settings.feature_signature,
    )
    statistic = pool_statistic(data, selection)
    runs = ", ".join(f"'{label}'" for label in labels)
    masks = pool_masks(
        selection,
        settings.brain_mask,
        settings.threshold,
        statistic=statistic,
        context=f"the final full-data noise pool (runs {runs})",
    )
    return selection, statistic, masks, analysis_components(data, masks.pool)


def _final_prefixes(components, masks, count, labels):
    """Each run's leading ``count`` PCs; raises instead of changing the count."""
    for label, comps in zip(labels, components, strict=True):
        reason = comps.unavailable_reason(count)
        if reason:
            ranks = ", ".join(
                f"run '{name}' rank {c.rank}" for name, c in zip(labels, components)
            )
            raise ValueError(
                f"the final full-data noise pool ({masks.pool_size} feature(s); "
                f"PCA {ranks}) cannot support the selected count {count}: "
                f"run '{label}': {reason}"
            )
    return tuple(c.prefix(count) for c in components)


# ---- provenance ---------------------------------------------------------------------------


def _baseline_confounds(data, labels):
    return [
        dict(
            run=label,
            columns=list(identity.confound_columns),
            fingerprint=identity.confounds,
        )
        for label, identity in zip(labels, run_identities(data))
    ]


def _fold_identities(folds):
    return dict(
        folds=[
            dict(train=list(f.training_runs), test=[f.validation_run]) for f in folds
        ],
        fold_hrf_assignment_fingerprints=[_digest(f.hrf_indices, "<i8") for f in folds],
        fold_noise_pool_fingerprints=[_digest(f.pool, "|b1") for f in folds],
        fold_scoring_mask_fingerprints=[_digest(f.scoring, "|b1") for f in folds],
    )


def _mixture_record(mixture):
    return None if mixture is None else mixture.to_dict()


def _threshold_records(settings, masks, folds):
    rule = "fixed" if masks.mixture is None else masks.mixture.to_dict()["method"]
    return dict(
        pool_r2_threshold=settings.threshold,
        pool_threshold_rule=rule,
        noise_pool_threshold=masks.threshold,
        noise_pool_mixture=_mixture_record(masks.mixture),
        fold_pool_thresholds=[f.pool_threshold for f in folds],
        fold_pool_mixtures=[_mixture_record(f.pool_mixture) for f in folds],
        fold_pool_sizes=[int(f.pool.sum()) for f in folds],
    )


def _excluded_counts(scores):
    failed = scores[~scores["eligible"]]
    return [
        dict(count=int(c), reason=r) for c, r in zip(failed["count"], failed["reason"])
    ]


def _selection_activity(context, scores, folds, final):
    data, labels, settings = context
    selection, masks, prefixes = final
    task_model, library = settings.task_model, settings.library
    return dict(
        name="denoising_selection",
        task_model=task_model.to_dict(),
        task_model_fingerprint=task_model.fingerprint,
        task_regressors=list(task_model.regressor_names),
        profiled_regressors=list(task_model.profiled_names),
        library=dict(library.origin),
        library_fingerprint=library.fingerprint,
        run_labels=list(labels),
        feature_signature=settings.feature_signature,
        counts=list(settings.counts),
        n_components=prefixes[0].shape[1],
        **_threshold_records(settings, masks, folds),
        score_tolerance=settings.tolerance,
        excluded_counts=_excluded_counts(scores),
        pool_statistic="indicator_consistent_loro_task_model_r2",
        score="heldout_task_prediction_r2_fixed_baseline_target",
        count_rule="smallest count within score_tolerance of the best fold-mean R2",
        conditional_prediction="missing-value indicators profiled on held-out BOLD",
        selection_statistic=True,
        brain_mask_fingerprint=_digest(settings.brain_mask, "|b1"),
        noise_pool_fingerprint=_digest(masks.pool, "|b1"),
        noise_pool_size=masks.pool_size,
        scoring_mask_size=masks.scoring_size,
        initial_hrf_assignment_fingerprint=_digest(selection.hrf_indices, "<i8"),
        initial_selection=identity_activity(selection.provenance),
        component_fingerprints=[_digest(p, "<f8") for p in prefixes],
        baseline_confounds=_baseline_confounds(data, labels),
        data_identity=[identity.to_dict() for identity in run_identities(data)],
        **_fold_identities(folds),
    )


def _provenance(operation, record, activity):
    analysis_id = analysis_fingerprint(record.metadata_fingerprint, activity)
    return operation.provenance(activity, analysis_id=analysis_id)


# ---- public selection ------------------------------------------------------------------------


def _select(operation, data, labels, settings):
    count = _count_selection(data, labels, settings)
    return _assemble(operation, (data, labels, settings), count)


def _assemble(operation, context, count):
    data, labels, settings = context
    selection, statistic, masks, components = _final_pool(data, labels, settings)
    prefixes = _final_prefixes(components, masks, count.n_components, labels)
    scores, fold_scores = _public_tables(count)
    folds = tuple(_fold_result(fold, labels) for fold in count.folds)
    activity = _selection_activity(context, scores, folds, (selection, masks, prefixes))
    return DenoisingResult(
        n_components=count.n_components,
        counts=count.counts,
        pool_r2_threshold=settings.threshold,
        score_tolerance=settings.tolerance,
        noise_pool_threshold=masks.threshold,
        noise_pool_mixture=masks.mixture,
        noise_pool=masks.pool,
        scoring_mask=masks.scoring,
        pool_r2=statistic,
        initial_selection=selection,
        run_components=prefixes,
        components=_diagnostics(components, labels, masks.pool_size),
        folds=folds,
        _candidate_scores=scores,
        _fold_scores=fold_scores,
        run_labels=labels,
        feature_signature=settings.feature_signature,
        source_identity=run_identities(data),
        provenance=_provenance(operation, data.provenance, activity),
    )


def select_denoising(
    data: AnalysisData,
    *,
    brain_mask: np.ndarray,
    task_model: TaskModel = TaskModel(),
    library: HrfLibrary | None = None,
    counts: tuple[int, ...] = (0, 1, 2, 4, 6, 8, 10),
    pool_r2_threshold: float | str = "auto",
    score_tolerance: float = 0.001,
    feature_signature: str | None = None,
    run_labels: Sequence[str] | None = None,
) -> DenoisingResult:
    """Choose a temporal-PC count by held-out task prediction; build final PCs.

    Requires at least three runs and a caller-supplied Boolean ``brain_mask``
    in feature order. Initial HRFs use baseline confounds only. Count scores
    are selection statistics, not independent performance estimates.
    ``pool_r2_threshold="auto"`` fits GLMsingle's two-component
    Gaussian-mixture tail threshold to the pool statistic in every fold and
    for the final pool; a degenerate fit raises (no fallback). A finite float
    applies that fixed threshold everywhere. Positive
    counts that any fold cannot support are unavailable; zero always remains.
    Raises when the final full-data pool cannot support the chosen count.
    """
    library = default_hrf_library() if library is None else library
    grid = (counts, pool_r2_threshold, score_tolerance)
    settings = _settings(data, brain_mask, task_model, library, grid, feature_signature)
    labels = run_labels_for(data, run_labels)
    with fit_operation("denoising_selection", data.provenance) as operation:
        return _select(operation, data, labels, settings)


# ---- augmentation -------------------------------------------------------------------------------


def _already_applied(data, result):
    token = result.provenance.execution_id
    return any(
        activity.get("name") == _AUGMENTATION
        and activity.get("denoising_execution_id") == token
        for activity in data.provenance.activities
    )


def _check_application(data, result, signature):
    _check_data(data)
    if not isinstance(result, DenoisingResult):
        raise ValueError("result must be a DenoisingResult")
    if _already_applied(data, result):
        raise ValueError("this denoising result was already applied to the data")
    if signature != result.feature_signature:
        raise ValueError("feature_signature must match the denoising result")
    check_identity(
        result.source_identity, len(result.noise_pool), data, result.run_labels
    )
    names = set(result.component_names)
    for label, frame in zip(result.run_labels, data.confounds):
        clash = sorted(names.intersection(map(str, frame.columns)))
        if clash:
            raise ValueError(
                f"run '{label}' confounds already contain {', '.join(clash)}"
            )


def _augmented_confounds(data, result):
    names = list(result.component_names)
    return tuple(
        pd.concat(
            [frame, pd.DataFrame(components, columns=names, index=frame.index)], axis=1
        )
        for frame, components in zip(data.confounds, result.run_components, strict=True)
    )


def _augmentation_activity(result):
    return dict(
        name=_AUGMENTATION,
        denoising_execution_id=result.provenance.execution_id,
        denoising_analysis_id=result.provenance.analysis_fingerprint,
        n_components=result.n_components,
        added_columns=list(result.component_names),
        component_fingerprints=[_digest(c, "<f8") for c in result.run_components],
        run_labels=list(result.run_labels),
        feature_signature=result.feature_signature,
    )


def _annotated(ref, mark):
    values = ref.to_dict()
    values["annotations"] = {**values.get("annotations", {}), _AUGMENTATION: mark}
    return SourceRef.from_dict(values)


def _annotated_run(sources, mark):
    """Mark the confounds source (else the signal); file identity is kept."""
    if sources.confounds is not None:
        confounds = _annotated(sources.confounds, mark)
        return RunSources(sources.signal, sources.events, confounds)
    return RunSources(_annotated(sources.signal, mark), sources.events, None)


def _annotated_record(data, result):
    """Parent provenance whose sources identify the augmented confounds.

    Downstream analysis ids derive from source metadata, so distinct
    augmentations of the same files must not share it.
    """
    mark = dict(
        selection_execution_id=result.provenance.execution_id,
        component_fingerprints=[_digest(c, "<f8") for c in result.run_components],
        n_components=result.n_components,
    )
    sources = tuple(_annotated_run(s, mark) for s in data.provenance.sources)
    return replace(data.provenance, sources=sources)


def with_denoising(
    data: AnalysisData,
    result: DenoisingResult,
    *,
    feature_signature: str | None = None,
) -> AnalysisData:
    """Append ``denoise_pc_000``, ... to the baseline confounds of each run.

    ``data`` must be the analysis ``result`` was selected on: same runs in the
    same order, time grids, signals (feature order), baseline confounds, and
    events, and a matching ``feature_signature``. Applying a result twice or
    colliding with existing confound names raises. Signals, events, timing,
    and source file identities are preserved. Each run's confounds source
    (the signal source when there are no confounds) gains a
    ``denoising_augmentation`` annotation, so downstream analysis ids differ
    between augmentations, and provenance gains one activity. A zero-count
    result adds no columns but is still recorded.
    """
    _check_application(data, result, feature_signature)
    parent = _annotated_record(data, result)
    with fit_operation(_AUGMENTATION, parent) as operation:
        confounds = _augmented_confounds(data, result)
        provenance = _provenance(operation, parent, _augmentation_activity(result))
        return AnalysisData(
            _signals=data.signals,
            _events=data.events,
            _confounds=confounds,
            _frame_times=data.frame_times,
            _timing_source=data.timing_source,
            _provenance=provenance,
        )
