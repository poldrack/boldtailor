"""Fit conventional models in HRF groups and restore input feature order."""

from dataclasses import dataclass, fields, replace

import numpy as np

from boldtailor._conventional import fit_designs
from boldtailor._fit_diagnostics import (
    delta_r2_activity,
    delta_r2_identity,
    nested_ols_delta,
)
from boldtailor._hrf_assignment import validate_selection
from boldtailor._hrf_design import HRF_NORMALIZATION
from boldtailor._hrf_glm_design import (
    compile_group_designs,
    design_identity,
    group_design_builder,
    group_diagnostics,
)
from boldtailor.hrf_glm_results import HrfAnalysisResult
from boldtailor._fit_lifecycle import fit_operation
from boldtailor.model import (
    TaskModel,
    model_identity,
    nuisance_model_settings,
)
from boldtailor.provenance import analysis_fingerprint, identity_activity
from boldtailor.results import _ContrastResult, make_task_delta_r2_result

SELECTED_HRF = {"kind": "selected"}


@dataclass(frozen=True)
class _FitContext:
    groups: dict
    nuisance: tuple
    activity: dict
    analysis_id: str | None


def _check_task_model(model, selection):
    if model.task_model is None:
        if selection.task_model != TaskModel():
            raise ValueError(
                "selection used a task_model; set ModelSpec.task_model to the same model"
            )
        return
    if not selection.task_model.is_subset_of(model.task_model):
        raise ValueError(
            "the selection's task_model must equal ModelSpec.task_model or be a "
            "subset of it with identical modulator settings"
        )
    activity = selection.provenance.to_dict()["activities"][-1]
    recorded = (activity.get("oversampling"), activity.get("min_onset"))
    if (model.oversampling, model.min_onset) != recorded:
        raise ValueError(
            "ModelSpec.oversampling and min_onset must match the selection's settings"
        )


def _prepare(data, model, selection, signature, model_settings):
    assignment = validate_selection(data, selection, signature)
    _check_task_model(model, selection)
    groups, nuisance = compile_group_designs(data, model, selection)
    activity = dict(
        name="selected_hrf_glm",
        stage="fit",
        model=model_settings,
        library_fingerprint=selection.library.fingerprint,
        hrf_assignment_fingerprint=assignment,
        design_fingerprint=design_identity(groups, nuisance, data.frame_times),
        feature_signature=signature,
        n_features=data.n_features,
        n_runs=data.n_runs,
        hrf_normalization=HRF_NORMALIZATION,
        selection=identity_activity(selection.provenance),
        selection_analysis_id=selection.provenance.analysis_fingerprint,
        inference="conditional_on_selected_hrfs",
    )
    if model.task_model is not None:
        activity["task_model"] = model.task_model.to_dict()
        activity["task_model_fingerprint"] = model.task_model.fingerprint
        activity["selection_task_model_fingerprint"] = selection.task_model.fingerprint
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
        _contrasts=contrasts,
        _run_r2=run_r2,
        _r2=r2,
        _provenance=provenance,
        _selection=selection,
        _group_design_provenance=group_diagnostics(context.groups),
        _group_builder=group_design_builder(
            data, model, selection, context.nuisance, context.groups.keys()
        ),
    )


def fit_selected_glm(data, model, selection, signature):
    with fit_operation("fit", data.provenance) as operation:
        settings = model_identity(model, hrf_model=SELECTED_HRF).activity
        context = _prepare(data, model, selection, signature, settings)
        operation.analysis_id = context.analysis_id
        fits = _group_fits(data, model, selection, context)
        provenance = operation.provenance(
            context.activity, analysis_id=context.analysis_id
        )
        return _assemble(data, model, selection, context, fits, provenance)


def _validate_parent(data, result, context):
    if context.analysis_id is None:
        raise ValueError(
            "task delta r-squared requires fingerprintable source provenance"
        )
    if context.analysis_id != result.provenance.analysis_fingerprint:
        raise ValueError(
            "full result provenance identity does not match this data and model (identity uses source metadata and sha256 when supplied)"
        )
    if len(result.run_r2) != data.n_runs or result.r2.shape != (data.n_features,):
        raise ValueError("full result dimensions do not match input")


def _ols_comparison(data, selection, context):
    full, nuisance = (np.full(data.n_features, np.nan) for _ in range(2))
    nuisance_designs = tuple(c.matrix for c in context.nuisance)
    for cid, indices in _feature_groups(selection):
        signals = tuple(y[:, indices] for y in data.signals)
        designs = tuple(context.groups[run, cid].matrix for run in range(data.n_runs))
        full[indices], nuisance[indices] = nested_ols_delta(
            signals, designs, nuisance_designs, allow_undefined=True
        )
    return full, nuisance, nuisance_designs


def selected_task_delta_r2(data, model, result):
    with fit_operation("task_delta_r2", result.provenance) as operation:
        selection = result.hrf_selection
        settings = model_identity(model, hrf_model=SELECTED_HRF).activity
        context = _prepare(
            data, model, selection, selection.feature_signature, settings
        )
        _validate_parent(data, result, context)
        full, nuisance, designs = _ols_comparison(data, selection, context)
        comparison = make_task_delta_r2_result(
            full_r2=full,
            nuisance_r2=nuisance,
            nuisance_designs=designs,
            provenance=result.provenance,
            allow_undefined=True,
        )
        activity = _comparison_activity(model, comparison, context)
        provenance = operation.provenance(
            activity,
            analysis_id=analysis_fingerprint(
                context.analysis_id, _comparison_identity(model)
            ),
        )
        return replace(comparison, _provenance=provenance)


def _comparison_activity(model, comparison, context):
    return delta_r2_activity(
        name="task_delta_r2",
        parent_id=context.analysis_id,
        inferential_noise_model=model.noise_model,
        nuisance_model=nuisance_model_settings(model),
        undefined_features=int(np.count_nonzero(~np.isfinite(comparison.delta_r2))),
    )


def _comparison_identity(model):
    return delta_r2_identity(
        name="task_delta_r2",
        inferential_noise_model=model.noise_model,
        nuisance_model=nuisance_model_settings(model),
    )
