"""Fit conventional models in HRF groups and restore input feature order."""

from dataclasses import dataclass, fields, replace

import numpy as np

from boldtailor._conventional import fit_designs, fit_r2_designs
from boldtailor._hrf_assignment import validate_selection
from boldtailor._hrf_glm_design import (
    compile_group_designs,
    design_identity,
    group_diagnostics,
)
from boldtailor.hrf_glm_results import HrfAnalysisResult, _masked_delta_result
from boldtailor._fit_lifecycle import fit_operation
from boldtailor.provenance import analysis_fingerprint
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


def fit_selected_glm(data, model, selection, signature):
    from boldtailor.fit import _model_provenance

    with fit_operation("fit", data.provenance) as operation:
        settings = _model_provenance(replace(model, hrf_model=None)).activity
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


def selected_task_delta_r2(data, model, result):
    from boldtailor.fit import _model_provenance

    with fit_operation("task_delta_r2", result.provenance) as operation:
        selection = result.hrf_selection
        settings = _model_provenance(replace(model, hrf_model=None)).activity
        context = _prepare(
            data, model, selection, selection.feature_signature, settings
        )
        _validate_parent(data, result, context)
        full, nuisance, designs = _ols_comparison(data, selection, context)
        comparison = _masked_delta_result(full, nuisance, designs, result.provenance)
        activity = _comparison_activity(model, comparison, context)
        provenance = operation.provenance(
            activity, analysis_id=analysis_fingerprint(context.analysis_id, activity)
        )
        return replace(comparison, _provenance=provenance)


def _comparison_activity(model, comparison, context):
    return dict(
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
