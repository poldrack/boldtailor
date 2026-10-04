"""Opt-in task-guided denoising following GLMsingle's GLMdenoise stage.

``select_denoising`` follows GLMsingle with stated deviations. It selects
per-feature HRFs once on all runs and freezes them; computes GLMsingle's
ON-OFF R² (canonical HRF, one task regressor, shared coefficient, each run's
confounds plus intercept as nuisance; GLMsingle uses polynomials); pools
features below GLMsingle's Gaussian-mixture tail threshold (by default) and
scores features above it (the 100 best when none passes); extracts run-wise
PCs of that single pool; and chooses one PC count with GLMsingle's pcstop
rule over the median of fold-pooled held-out R², then (a Boldtailor
addition, not part of GLMsingle) keeps that count only if the PCs pass a
per-feature OLS F-test gate with a binomial test across features. Counts are
scored by leave-one-run-out time-series prediction against a fixed target,
because repeated conditions are not assumed. Every input feature is a
candidate: the core is anatomy-agnostic. ``with_denoising`` appends the chosen PCs to the
baseline confounds of the same analysis for ordinary HRF selection and
fitting.

This module reimplements procedures from GLMsingle
(https://github.com/cvnlab/GLMsingle), Copyright (c) 2021, Kendrick Kay,
distributed under the BSD 3-Clause License; see
LICENSES/GLMsingle-BSD-3-Clause.txt for the copyright notice, conditions,
and disclaimer. It is an independent reimplementation, not a copy of
GLMsingle code. Follows the GLMdenoise stage of GLMsingle (noise pool,
run-wise PCs, cross-validated PC count, select_noise_regressors).
Reference: Prince, J.S., Charest, I., Kurzawski, J.W., Pyles, J.A., Tarr,
M.J., Kay, K.N. (2022). Improving the accuracy of single-trial fMRI response
estimates using GLMsingle. eLife, 11, e77599.
https://doi.org/10.7554/eLife.77599
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from hashlib import sha256

import numpy as np
import pandas as pd

from boldtailor._denoising_cv import (
    NO_TASK_SIGNAL,
    select_component_count,
    validate_counts,
    validate_pcstop,
)
from boldtailor._denoising_gate import apply_gate, validate_gate
from boldtailor._denoising_identity import check_identity, run_identities
from boldtailor._denoising_pool import (
    analysis_components,
    onoff_r2,
    pool_masks,
    validate_threshold,
)
from boldtailor._fit_lifecycle import fit_operation
from boldtailor._hrf_cv import prepare_runs
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

    task_model: TaskModel
    library: HrfLibrary
    counts: tuple[int, ...]
    threshold: float | str
    pcstop: float
    gate: tuple[bool, float, float]
    feature_signature: str | None


def _check_models(task_model, library, signature):
    if not isinstance(task_model, TaskModel):
        raise ValueError("task_model must be a TaskModel")
    if not isinstance(library, HrfLibrary):
        raise ValueError("library must be an HrfLibrary")
    if signature is not None and (not isinstance(signature, str) or not signature):
        raise ValueError("feature_signature must be a nonempty string or None")


def _settings(data, task_model, library, grid, signature):
    """Validate everything before any fitting.

    ``grid`` is (counts, r2 threshold, pcstop, (gate, alpha, binomial alpha)).
    """
    _check_data(data)
    _check_models(task_model, library, signature)
    counts, threshold, pcstop, gate = grid
    return _Settings(
        task_model=task_model,
        library=library,
        counts=validate_counts(counts),
        threshold=validate_threshold(threshold),
        pcstop=validate_pcstop(pcstop),
        gate=validate_gate(*gate),
        feature_signature=signature,
    )


def _check_runs(data, labels, settings):
    """At least three runs; every run design valid, errors named by label."""
    if data.n_runs < 3:
        raise ValueError("denoising selection requires at least three runs")
    prepare_runs(data, settings.library, settings.task_model, labels=labels)
    prepare_runs(data, settings.library, TaskModel(), labels=labels)


def _digest(values, dtype) -> str:
    return sha256(np.ascontiguousarray(values, dtype=dtype).tobytes()).hexdigest()


# ---- full-data HRFs, pool, and PCs ---------------------------------------------------


def _pool(data, labels, settings):
    """Frozen full-data HRFs, ON-OFF R², masks, and run-wise pool PCs."""
    selection = select_hrfs(
        data,
        library=settings.library,
        task_model=settings.task_model,
        run_labels=labels,
        feature_signature=settings.feature_signature,
    )
    statistic = onoff_r2(data, settings.library, run_labels=labels)
    runs = ", ".join(f"'{label}'" for label in labels)
    masks = pool_masks(
        statistic,
        settings.threshold,
        hrf_indices=selection.hrf_indices,
        context=f"the noise pool (runs {runs})",
    )
    if masks.scoring_size == 0:
        raise ValueError(
            f"{NO_TASK_SIGNAL}: no feature has a finite ON-OFF R² and a defined HRF"
        )
    return selection, statistic, masks, analysis_components(data, masks.pool)


def _count_selection(data, labels, settings, pool):
    """pcstop's count, then Boldtailor's significance gate on it."""
    selection, _, masks, components = pool
    count = select_component_count(
        data,
        hrf_indices=selection.hrf_indices,
        components=components,
        scoring=masks.scoring,
        task_model=settings.task_model,
        library=settings.library,
        counts=settings.counts,
        pcstop=settings.pcstop,
        run_labels=labels,
    )
    enabled, alpha, binomial_alpha = settings.gate
    gate = apply_gate(
        count.setup,
        count.n_components,
        enabled=enabled,
        alpha=alpha,
        binomial_alpha=binomial_alpha,
    )
    return count, gate


# ---- result parts ----------------------------------------------------------------------


def _public_tables(count):
    """Aggregate scores and per-fold scores keyed by the held-out run label."""
    folds = count.fold_scores.copy()
    folds["validation_run"] = folds.pop("validation_label")
    return count.scores, folds


def _diagnostics(components, labels, pool_size):
    return PcaDiagnostics(
        run_labels=labels,
        pool_size=pool_size,
        ranks=[c.rank for c in components],
        singular_values=[c.singular_values for c in components],
        rank_tolerances=[c.rank_tolerance for c in components],
        retained_columns=[c.retained_columns for c in components],
    )


def _folds(setup):
    labels = setup.run_labels
    return tuple(
        DenoisingFold(
            validation_run=label,
            training_runs=tuple(x for x in labels if x != label),
            zero_target=setup.zero_target[v],
            target_energy=setup.target_energy[v],
        )
        for v, label in enumerate(labels)
    )


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


def _excluded_counts(scores):
    failed = scores[~scores["eligible"]]
    return [
        dict(count=int(c), reason=r) for c, r in zip(failed["count"], failed["reason"])
    ]


def _floats(values):
    return [None if not np.isfinite(v) else float(v) for v in values]


def _gate_record(gate):
    return dict(
        enabled=gate.enabled,
        test="ols_nested_f_in_sample_all_runs",
        aggregate="one_sided_binomial_over_tested_features",
        gate_alpha=gate.alpha,
        gate_binomial_alpha=gate.binomial_alpha,
        pcstop_count=gate.pcstop_count,
        n_components=gate.n_components,
        decision=gate.decision,
        m=gate.m,
        n=gate.n,
        n_excluded=gate.n_excluded,
        binomial_p=_floats([gate.binomial_p])[0],
        f_statistic_fingerprint=_digest(gate.f_statistic, "<f8"),
        p_value_fingerprint=_digest(gate.p_value, "<f8"),
        exclusions=[
            dict(hrf_index=h, n_features=n, reason=r) for h, n, r in gate.exclusions
        ],
    )


def _pool_records(settings, statistic, masks, scored):
    rule = "fixed" if masks.mixture is None else masks.mixture.to_dict()["method"]
    return dict(
        pool_statistic="glmsingle_onoff_r2",
        pool_r2_threshold=settings.threshold,
        pool_threshold_rule=rule,
        noise_pool_threshold=masks.threshold,
        noise_pool_mixture=None if masks.mixture is None else masks.mixture.to_dict(),
        scoring_fallback=masks.fallback,
        onoff_r2_fingerprint=_digest(statistic, "<f8"),
        noise_pool_fingerprint=_digest(masks.pool, "|b1"),
        noise_pool_size=masks.pool_size,
        scoring_mask_fingerprint=_digest(masks.scoring, "|b1"),
        scoring_mask_size=masks.scoring_size,
        scored_fingerprint=_digest(scored, "|b1"),
    )


def _count_records(settings, count):
    scores, labels = count.scores, count.setup.run_labels
    return dict(
        counts=list(settings.counts),
        count_rule="glmsingle_pcstop",
        pcstop=settings.pcstop,
        performance="median_over_scored_features_of_fold_pooled_r2",
        perf=_floats(scores["perf"]),
        curve=_floats(scores["curve"]),
        excluded_counts=_excluded_counts(scores),
        score="heldout_task_prediction_r2_fixed_baseline_target",
        conditional_prediction="missing-value indicators profiled on held-out BOLD",
        selection_statistic=True,
        folds=[
            dict(train=[x for x in labels if x != label], test=[label])
            for label in labels
        ],
    )


def _selection_activity(context, pool, selected, prefixes):
    data, labels, settings = context
    selection, statistic, masks, _ = pool
    count, gate = selected
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
        n_components=prefixes[0].shape[1],
        **_pool_records(settings, statistic, masks, count.setup.scored),
        **_count_records(settings, count),
        significance_gate=_gate_record(gate),
        initial_hrf_assignment_fingerprint=_digest(selection.hrf_indices, "<i8"),
        initial_selection=identity_activity(selection.provenance),
        component_fingerprints=[_digest(p, "<f8") for p in prefixes],
        baseline_confounds=_baseline_confounds(data, labels),
        data_identity=[identity.to_dict() for identity in run_identities(data)],
    )


def _provenance(operation, record, activity):
    analysis_id = analysis_fingerprint(record.metadata_fingerprint, activity)
    return operation.provenance(activity, analysis_id=analysis_id)


# ---- public selection ------------------------------------------------------------------------


def _select(operation, data, labels, settings):
    pool = _pool(data, labels, settings)
    selected = _count_selection(data, labels, settings, pool)
    return _assemble(operation, (data, labels, settings), pool, selected)


def _assemble(operation, context, pool, selected):
    data, labels, settings = context
    selection, statistic, masks, components = pool
    count, gate = selected
    # An eligible count is supported by every run (each run trains some fold).
    prefixes = tuple(c.prefix(gate.n_components) for c in components)
    scores, fold_scores = _public_tables(count)
    activity = _selection_activity(context, pool, selected, prefixes)
    return DenoisingResult(
        n_components=gate.n_components,
        pcstop_count=count.n_components,
        significance_gate=gate,
        counts=count.counts,
        pool_r2_threshold=settings.threshold,
        pcstop=settings.pcstop,
        noise_pool_threshold=masks.threshold,
        noise_pool_mixture=masks.mixture,
        noise_pool=masks.pool,
        scoring_mask=masks.scoring,
        scoring_fallback=masks.fallback,
        scored=count.setup.scored,
        onoff_r2=statistic,
        initial_selection=selection,
        run_components=prefixes,
        components=_diagnostics(components, labels, masks.pool_size),
        folds=_folds(count.setup),
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
    task_model: TaskModel = TaskModel(),
    library: HrfLibrary | None = None,
    counts: tuple[int, ...] = tuple(range(11)),
    pool_r2_threshold: float | str = "auto",
    pcstop: float = 1.05,
    significance_gate: bool = True,
    gate_alpha: float = 0.05,
    gate_binomial_alpha: float = 0.05,
    feature_signature: str | None = None,
    run_labels: Sequence[str] | None = None,
) -> DenoisingResult:
    """Choose a temporal-PC count as GLMsingle does; return full-data PCs.

    Requires at least three runs. Every feature is a candidate (no mask).
    HRFs are selected once on all runs with baseline confounds and frozen.
    The pool is the features whose ON-OFF R² falls below
    ``pool_r2_threshold`` (``"auto"``: GLMsingle's Gaussian-mixture tail
    threshold; a degenerate fit raises suggesting a fixed value); features
    above it are scored (the 100 best when none passes). Counts are scored by
    leave-one-run-out held-out task prediction, pooled per feature across
    folds, with the median over features, and chosen by GLMsingle's
    ``pcstop`` rule (``>= 1``). Positive counts that any fold cannot support
    are unavailable; zero always remains. Scores are selection statistics,
    not independent performance estimates.

    ``significance_gate`` (a Boldtailor addition, not part of GLMsingle)
    then keeps the pcstop count only if adding those PCs passes per-feature
    in-sample OLS F-tests (``p < gate_alpha``) in more scoring features than
    chance, by a one-sided binomial test at ``gate_binomial_alpha``;
    otherwise zero PCs are chosen. ``significance_gate=False`` reproduces
    GLMsingle's pcstop-only choice.
    """
    library = default_hrf_library() if library is None else library
    gate = (significance_gate, gate_alpha, gate_binomial_alpha)
    grid = (counts, pool_r2_threshold, pcstop, gate)
    settings = _settings(data, task_model, library, grid, feature_signature)
    labels = run_labels_for(data, run_labels)
    _check_runs(data, labels, settings)
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
