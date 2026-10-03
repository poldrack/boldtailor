from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import numpy as np

from boldtailor._conventional import ConventionalFit, fit_designs
from boldtailor._fit_diagnostics import (
    rank_warnings,
    delta_r2_activity,
    delta_r2_identity,
    nested_ols_delta,
    validate_result_dimensions,
)
from boldtailor.data import AnalysisData
from boldtailor.design import CompiledDesign, compile_designs, compile_nuisance_designs
from boldtailor._fit_lifecycle import fit_operation
from boldtailor._hrf_glm import fit_selected_glm, selected_task_delta_r2
from boldtailor.model import ModelSpec, model_identity, nuisance_model_settings
from boldtailor.hrf_results import HrfSelectionResult
from boldtailor.hrf_glm_results import HrfAnalysisResult
from boldtailor.provenance import analysis_fingerprint
from boldtailor.results import (
    AnalysisResult,
    TaskDeltaR2Result,
    make_result,
    make_task_delta_r2_result,
)


def fit(
    data: AnalysisData,
    model: ModelSpec,
    *,
    hrf_selection: HrfSelectionResult | None = None,
    feature_signature: str | None = None,
) -> AnalysisResult | HrfAnalysisResult:
    """Fit conventional contrasts, optionally using an HRF per feature.

    A supplied selection replaces model.hrf_model. Its spatial signature must
    match feature_signature; all other model settings retain their meaning.
    Selected-HRF results expose group_design(run, hrf_id), recompiled on
    request, instead of design_matrices.
    """
    if hrf_selection is not None:
        return fit_selected_glm(data, model, hrf_selection, feature_signature)
    with fit_operation("fit", data.provenance) as operation:
        if feature_signature is not None:
            raise ValueError("feature_signature requires hrf_selection")
        model_provenance = model_identity(model)
        operation.analysis_id = _analysis_id(
            data.provenance.metadata_fingerprint, model_provenance.fingerprint
        )
        compiled, numerical = _fit_analysis(data, model)
        provenance = operation.provenance(
            _fit_activity(data, model_provenance.activity, compiled),
            warnings=model_provenance.warnings,
            analysis_id=operation.analysis_id,
        )
        return make_result(
            numerical.contrasts,
            tuple(design.matrix for design in compiled),
            tuple(_design_provenance(design) for design in compiled),
            tuple(run.r2 for run in numerical.run_fits),
            numerical.aggregate_r2,
            provenance,
        )


def task_delta_r2(
    data: AnalysisData,
    model: ModelSpec,
    full_result: AnalysisResult | HrfAnalysisResult,
) -> TaskDeltaR2Result:
    if isinstance(full_result, HrfAnalysisResult):
        return selected_task_delta_r2(data, model, full_result)
    with fit_operation("task_delta_r2", full_result.provenance) as operation:
        model_provenance = model_identity(model)
        data_id = data.provenance.metadata_fingerprint
        expected_parent_id = _analysis_id(data_id, model_provenance.fingerprint)
        operation.analysis_id = _comparison_id(expected_parent_id, model)
        _validate_parent_analysis(expected_parent_id, full_result)
        validate_result_dimensions(data, full_result, input_label="data")
        full_designs = compile_designs(data, model)
        nuisance_designs = compile_nuisance_designs(data, model)
        full_r2, nuisance_r2 = nested_ols_delta(
            data.signals,
            tuple(d.matrix for d in full_designs),
            tuple(d.matrix for d in nuisance_designs),
            allow_undefined=False,
        )
        comparison = make_task_delta_r2_result(
            full_r2=full_r2,
            nuisance_r2=nuisance_r2,
            nuisance_designs=tuple(design.matrix for design in nuisance_designs),
            provenance=full_result.provenance,
        )
        activity = _task_delta_r2_activity(
            data,
            model,
            nuisance_designs,
            comparison,
            expected_parent_id,
        )
        provenance = operation.provenance(activity, analysis_id=operation.analysis_id)
        return replace(comparison, _provenance=provenance)


def _fit_analysis(
    data: AnalysisData,
    model: ModelSpec,
) -> tuple[tuple[CompiledDesign, ...], ConventionalFit]:
    compiled = compile_designs(data, model)
    numerical = fit_designs(
        data.signals,
        tuple(item.matrix for item in compiled),
        model.contrasts,
        model.noise_model,
    )
    return compiled, numerical


def _validate_parent_analysis(
    expected_parent_id: str | None,
    full_result: AnalysisResult,
) -> None:
    if expected_parent_id is None:
        raise ValueError(
            "task delta r-squared requires fingerprintable data and model provenance"
        )
    if full_result.provenance.analysis_fingerprint != expected_parent_id:
        raise ValueError(
            "full result provenance identity does not match this data and model (identity uses source metadata and sha256 when supplied)"
        )


def _comparison_id(parent_id: str | None, model: ModelSpec) -> str | None:
    identity = delta_r2_identity(
        name="task_delta_r2",
        inferential_noise_model=model.noise_model,
        nuisance_model=nuisance_model_settings(model),
    )
    return _analysis_id(parent_id, identity)


def _analysis_id(
    data_id: str | None,
    model: Mapping[str, object] | None,
) -> str | None:
    if model is None:
        return None
    return analysis_fingerprint(data_id, model)


def _fit_activity(
    data: AnalysisData,
    model: Mapping[str, object],
    compiled: tuple[CompiledDesign, ...],
) -> dict[str, object]:
    runs = tuple(
        _run_diagnostic(data, design, run) for run, design in enumerate(compiled)
    )
    return {"name": "fit", "stage": "fit", "model": model, "runs": runs}


def _task_delta_r2_activity(
    data: AnalysisData,
    model: ModelSpec,
    compiled: tuple[CompiledDesign, ...],
    comparison: TaskDeltaR2Result,
    parent_id: str | None,
) -> dict[str, object]:
    runs = tuple(
        _run_diagnostic(data, design, run) for run, design in enumerate(compiled)
    )
    return {
        **delta_r2_activity(
            name="task_delta_r2",
            parent_id=parent_id,
            inferential_noise_model=model.noise_model,
            nuisance_model=nuisance_model_settings(model),
            undefined_features=0,
        ),
        "runs": runs,
        "diagnostics": {
            "raw_min": comparison.raw_min,
            "negative_voxel_count": comparison.negative_voxel_count,
            "mean_delta_r2": float(comparison.delta_r2.mean()),
            "max_delta_r2": float(comparison.delta_r2.max()),
        },
    }


def _run_diagnostic(
    data: AnalysisData,
    compiled: CompiledDesign,
    run: int,
) -> dict[str, object]:
    matrix = compiled.matrix.to_numpy()
    rank = int(np.linalg.matrix_rank(matrix))
    warnings = rank_warnings(rank, matrix.shape[1], run)
    return {
        "timing_source": data.timing_source,
        "n_scans": matrix.shape[0],
        "n_features": data.signals[run].shape[1],
        "design_columns": list(compiled.matrix.columns),
        "design_rank": rank,
        "residual_dof": matrix.shape[0] - rank,
        "excluded_event_count": compiled.excluded_event_count,
        "min_onset_cutoff": compiled.min_onset_cutoff,
        "warnings": warnings,
    }


def _design_provenance(compiled: CompiledDesign) -> dict[str, int | float]:
    return {
        "excluded_event_count": compiled.excluded_event_count,
        "min_onset_cutoff": compiled.min_onset_cutoff,
    }
