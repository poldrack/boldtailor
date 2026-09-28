"""Fit conventional models in HRF groups and restore input feature order."""

from dataclasses import dataclass, fields, replace
import logging
from uuid import uuid4

import numpy as np

from boldtailor._conventional import fit_designs, fit_r2_designs
from boldtailor._hrf_assignment import validate_selection
from boldtailor._hrf_glm_design import (
    compile_group_designs,
    design_identity,
    group_diagnostics,
)
from boldtailor.hrf_glm_results import HrfAnalysisResult, _masked_delta_result
from boldtailor.logging import append_event_history, bind_context, emit_event
from boldtailor.provenance import analysis_fingerprint, extend_provenance
from boldtailor.results import _ContrastResult


@dataclass(frozen=True)
class _FitContext:
    groups: dict
    nuisance: tuple
    activity: dict
    analysis_id: str | None


def _prepare(data, model, selection, signature, model_settings):
    assignment = validate_selection(data, selection, signature)
    groups, nuisance = compile_group_designs(data, model, selection)
    settings = {**model_settings, "hrf_model": {"kind": "selected"}}
    activity = dict(
        name="selected_hrf_glm",
        stage="fit",
        model=settings,
        library_fingerprint=selection.library.fingerprint,
        hrf_assignment_fingerprint=assignment,
        design_fingerprint=design_identity(groups, nuisance, data.frame_times),
        feature_signature=signature,
        n_features=data.n_features,
        n_runs=data.n_runs,
        hrf_normalization="discrete_sum_one",
        selection=selection.provenance.to_dict()["activities"][-1],
        selection_analysis_id=selection.provenance.analysis_fingerprint,
        inference="conditional_on_selected_hrfs",
    )
    return _FitContext(
        groups,
        nuisance,
        activity,
        analysis_fingerprint(data.provenance.metadata_fingerprint, activity),
    )


def _feature_groups(selection):
    ids = selection.hrf_indices
    return [(int(cid), np.flatnonzero(ids == cid)) for cid in np.unique(ids[ids >= 0])]


def _group_fits(data, model, selection, context):
    return {
        cid: fit_designs(
            tuple(y[:, indices] for y in data.signals),
            tuple(context.groups[run, cid].matrix for run in range(data.n_runs)),
            model.contrasts,
            model.noise_model,
        )
        for cid, indices in _feature_groups(selection)
    }


def _assemble(data, model, selection, context, fits, provenance):
    contrasts = {}
    for name in model.contrasts:
        values = {
            field.name: np.full(data.n_features, np.nan)
            for field in fields(_ContrastResult)
        }
        for cid, indices in _feature_groups(selection):
            for field, array in values.items():
                array[indices] = getattr(fits[cid].contrasts[name], field)
        contrasts[name] = _ContrastResult(**values)
    run_r2 = tuple(np.full(data.n_features, np.nan) for _ in range(data.n_runs))
    r2 = np.full(data.n_features, np.nan)
    for cid, indices in _feature_groups(selection):
        r2[indices] = fits[cid].aggregate_r2
        for run, fitted in enumerate(fits[cid].run_fits):
            run_r2[run][indices] = fitted.r2
    return HrfAnalysisResult(
        contrasts,
        run_r2,
        r2,
        provenance,
        selection,
        {key: compiled.matrix for key, compiled in context.groups.items()},
        group_diagnostics(context.groups),
    )


def fit_selected_glm(data, model, selection, signature, model_settings):
    execution_id = str(uuid4())
    with bind_context(
        execution_id=execution_id, data_id=data.provenance.metadata_fingerprint
    ):
        history = append_event_history(
            data.provenance.events, emit_event("fit_started", stage="fit")
        )
        try:
            context = _prepare(data, model, selection, signature, model_settings)
            fits = _group_fits(data, model, selection, context)
        except Exception as error:
            emit_event("fit_failed", stage="fit", level=logging.ERROR, error=error)
            raise
        history = append_event_history(
            history,
            emit_event("fit_completed", stage="fit", analysis_id=context.analysis_id),
        )
    provenance = extend_provenance(
        data.provenance,
        execution_id=execution_id,
        activity=context.activity,
        events=history,
        warnings=(),
        analysis_id=context.analysis_id,
    )
    return _assemble(data, model, selection, context, fits, provenance)


def _validate_parent(data, result, context):
    if context.analysis_id is None:
        raise ValueError(
            "task delta r-squared requires fingerprintable source provenance"
        )
    if context.analysis_id != result.provenance.analysis_fingerprint:
        raise ValueError("full result does not match data, model, or HRF identity")
    if len(result.run_r2) != data.n_runs or result.r2.shape != (data.n_features,):
        raise ValueError("full result dimensions do not match input")


def _ols_comparison(data, selection, context):
    full, nuisance = (np.full(data.n_features, np.nan) for _ in range(2))
    nuisance_designs = tuple(c.matrix for c in context.nuisance)
    for cid, indices in _feature_groups(selection):
        signals = tuple(y[:, indices] for y in data.signals)
        designs = tuple(context.groups[run, cid].matrix for run in range(data.n_runs))
        full[indices] = fit_r2_designs(signals, designs, "ols")
        nuisance[indices] = fit_r2_designs(signals, nuisance_designs, "ols")
    return full, nuisance, nuisance_designs


def selected_task_delta_r2(data, model, result, model_settings):
    execution_id = str(uuid4())
    selection = result.hrf_selection
    with bind_context(
        execution_id=execution_id, data_id=data.provenance.metadata_fingerprint
    ):
        history = append_event_history(
            result.provenance.events, emit_event("task_delta_r2_started", stage="fit")
        )
        try:
            context = _prepare(
                data, model, selection, selection.feature_signature, model_settings
            )
            _validate_parent(data, result, context)
            full, nuisance, designs = _ols_comparison(data, selection, context)
            comparison = _masked_delta_result(
                full, nuisance, designs, result.provenance
            )
        except Exception as error:
            emit_event(
                "task_delta_r2_failed",
                stage="fit",
                level=logging.ERROR,
                error=error,
            )
            raise
    provenance = _comparison_provenance(
        result, model, comparison, context, execution_id, history
    )
    completed = emit_event(
        "task_delta_r2_completed",
        stage="fit",
        execution_id=execution_id,
        data_id=data.provenance.metadata_fingerprint,
        analysis_id=provenance.analysis_fingerprint,
    )
    provenance = replace(provenance, events=append_event_history(history, completed))
    return replace(comparison, _provenance=provenance)


def _comparison_provenance(result, model, comparison, context, execution_id, history):
    activity = dict(
        name="task_delta_r2",
        stage="fit",
        parent_analysis_id=context.analysis_id,
        definition="full_r2 - nuisance_r2",
        diagnostic_noise_model="ols",
        inferential_noise_model=model.noise_model,
        clip_below_zero=True,
        clip_policy="numerical_roundoff_guard",
        undefined_features=int(np.count_nonzero(~np.isfinite(comparison.delta_r2))),
    )
    return extend_provenance(
        result.provenance,
        execution_id=execution_id,
        activity=activity,
        events=history,
        warnings=(),
        analysis_id=analysis_fingerprint(context.analysis_id, activity),
    )
