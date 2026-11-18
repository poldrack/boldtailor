"""Complete shared-HRF GLM fitting and task-variance comparison operations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import numpy as np

from boldtailor._conventional import ConventionalFit, design_rank, fit_designs
from boldtailor._fit_diagnostics import (
    rank_warnings,
    delta_r2_activity,
    delta_r2_identity,
    validate_result_dimensions,
)
from boldtailor._conventional import nested_ols_delta
from boldtailor.data import AnalysisData
from boldtailor.design import CompiledDesign, compile_designs, compile_nuisance_designs
from boldtailor._fit_lifecycle import build_fit_provenance, fit_operation
from boldtailor.model import ModelSpec, model_identity, nuisance_model_settings
from boldtailor.provenance import analysis_fingerprint
from boldtailor.results import (
    AnalysisResult,
    TaskDeltaR2Result,
    make_result,
    make_task_delta_r2_result,
)


def fit_standard_glm(
    data: AnalysisData,
    model: ModelSpec,
    *,
    feature_signature: str | None = None,
) -> AnalysisResult:
    """Validate, fit and record a GLM using the model's shared HRF."""
    with fit_operation("fit", data.provenance) as operation:
        if feature_signature is not None:
            raise ValueError("feature_signature requires hrf_selection")
        model_provenance = model_identity(model)
        operation.analysis_fingerprint = _analysis_fingerprint(
            data.provenance.data_fingerprint, model_provenance.fingerprint
        )
        compiled, numerical = _fit_analysis(data, model)
        provenance = build_fit_provenance(
            operation,
            _fit_activity(data, model_provenance.activity, compiled),
            warnings=model_provenance.warnings,
        )
        return make_result(
            numerical.contrasts,
            tuple(design.matrix for design in compiled),
            tuple(_design_provenance(design) for design in compiled),
            tuple(run.r2 for run in numerical.run_fits),
            numerical.aggregate_r2,
            provenance,
        )


def standard_task_delta_r2(
    data: AnalysisData,
    model: ModelSpec,
    full_result: AnalysisResult,
) -> TaskDeltaR2Result:
    with fit_operation("task_delta_r2", full_result.provenance) as operation:
        model_provenance = model_identity(model)
        data_fingerprint = data.provenance.data_fingerprint
        expected_parent_fingerprint = _analysis_fingerprint(
            data_fingerprint, model_provenance.fingerprint
        )
        operation.analysis_fingerprint = _comparison_fingerprint(
            expected_parent_fingerprint, model
        )
        _validate_parent_analysis(expected_parent_fingerprint, full_result)
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
            expected_parent_fingerprint,
        )
        provenance = build_fit_provenance(operation, activity)
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
    expected_parent_fingerprint: str | None,
    full_result: AnalysisResult,
) -> None:
    if (
        expected_parent_fingerprint is not None
        and full_result.provenance.analysis_fingerprint is not None
        and full_result.provenance.analysis_fingerprint != expected_parent_fingerprint
    ):
        raise ValueError(
            "full result provenance identity does not match this data and model (identity uses source metadata and sha256 when supplied)"
        )


def _comparison_fingerprint(
    parent_fingerprint: str | None, model: ModelSpec
) -> str | None:
    identity = delta_r2_identity(
        name="task_delta_r2",
        inferential_noise_model=model.noise_model,
        nuisance_model=nuisance_model_settings(model),
    )
    return _analysis_fingerprint(parent_fingerprint, identity)


def _analysis_fingerprint(
    data_fingerprint: str | None,
    model: Mapping[str, object] | None,
) -> str | None:
    if model is None:
        return None
    return analysis_fingerprint(data_fingerprint, model)


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
    parent_fingerprint: str | None,
) -> dict[str, object]:
    runs = tuple(
        _run_diagnostic(data, design, run) for run, design in enumerate(compiled)
    )
    return {
        **delta_r2_activity(
            name="task_delta_r2",
            parent_fingerprint=parent_fingerprint,
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
    rank = design_rank(matrix)
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
