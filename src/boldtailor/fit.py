from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import sys

import numpy as np

from boldtailor._conventional import ConventionalFit, fit_designs, fit_r2_designs
from boldtailor._fit_diagnostics import (
    TASK_DELTA_R2_DEFINITION,
    DIAGNOSTIC_NOISE_MODEL,
    rank_warnings,
    validate_nested_ols_delta,
    validate_result_dimensions,
)
from boldtailor.data import AnalysisData
from boldtailor.design import CompiledDesign, compile_designs, compile_nuisance_designs
from boldtailor._fit_lifecycle import fit_operation
from boldtailor.model import ModelSpec, contrast_metadata
from boldtailor.hrf_results import HrfSelectionResult
from boldtailor.hrf_glm_results import HrfAnalysisResult
from boldtailor.provenance import analysis_fingerprint
from boldtailor.results import (
    AnalysisResult,
    TaskDeltaR2Result,
    make_result,
    make_task_delta_r2_result,
)


@dataclass(frozen=True)
class _ModelProvenance:
    activity: dict[str, object]
    fingerprint: dict[str, object] | None
    warnings: tuple[Mapping[str, object], ...]


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
    Selected-HRF results expose group_designs instead of design_matrices.
    """
    if hrf_selection is not None:
        from boldtailor._hrf_glm import fit_selected_glm

        return fit_selected_glm(data, model, hrf_selection, feature_signature)
    with fit_operation("fit", data.provenance) as operation:
        if feature_signature is not None:
            raise ValueError("feature_signature requires hrf_selection")
        model_provenance = _model_provenance(model)
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
        from boldtailor._hrf_glm import selected_task_delta_r2

        return selected_task_delta_r2(data, model, full_result)
    with fit_operation("task_delta_r2", full_result.provenance) as operation:
        model_provenance = _model_provenance(model)
        data_id = data.provenance.metadata_fingerprint
        expected_parent_id = _analysis_id(data_id, model_provenance.fingerprint)
        operation.analysis_id = _comparison_id(expected_parent_id, model)
        _validate_parent_analysis(expected_parent_id, full_result)
        validate_result_dimensions(data, full_result, input_label="data")
        full_designs = compile_designs(data, model)
        nuisance_designs = compile_nuisance_designs(data, model)
        full_r2 = _fit_r2_analysis(data, full_designs, DIAGNOSTIC_NOISE_MODEL)
        nuisance_r2 = _fit_r2_analysis(
            data,
            nuisance_designs,
            DIAGNOSTIC_NOISE_MODEL,
        )
        validate_nested_ols_delta(full_r2 - nuisance_r2)
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


def _fit_r2_analysis(
    data: AnalysisData,
    compiled: tuple[CompiledDesign, ...],
    noise_model: str,
) -> np.ndarray:
    r2 = fit_r2_designs(
        data.signals,
        tuple(item.matrix for item in compiled),
        noise_model,
    )
    if not np.isfinite(r2).all():
        raise ValueError("diagnostic fit produced nonfinite r-squared values")
    return r2


def _validate_parent_analysis(
    expected_parent_id: str | None,
    full_result: AnalysisResult,
) -> None:
    if expected_parent_id is None:
        raise ValueError(
            "task delta r-squared requires fingerprintable data and model provenance"
        )
    if full_result.provenance.analysis_fingerprint != expected_parent_id:
        raise ValueError("full result does not match data and model")


def _comparison_id(parent_id: str | None, model: ModelSpec) -> str | None:
    return _analysis_id(parent_id, _task_delta_r2_settings(model))


def _task_delta_r2_settings(model: ModelSpec) -> dict[str, object]:
    return {
        "name": "task_delta_r2",
        "definition": TASK_DELTA_R2_DEFINITION,
        "clip_below_zero": True,
        "diagnostic_noise_model": DIAGNOSTIC_NOISE_MODEL,
        "inferential_noise_model": model.noise_model,
        "clip_policy": "numerical_roundoff_guard",
        "nuisance_model": _nuisance_model_settings(model),
    }


def _nuisance_model_settings(model: ModelSpec) -> dict[str, object]:
    return {
        "events": False,
        "confounds": list(model.confounds),
        "drift_model": model.drift_model,
        "high_pass": model.high_pass,
        "drift_order": model.drift_order,
        "noise_model": DIAGNOSTIC_NOISE_MODEL,
    }


def _model_provenance(model: ModelSpec) -> _ModelProvenance:
    hrf_model, callable_warnings = _serialize_hrf(model.hrf_model)
    activity = {
        "contrasts": contrast_metadata(model.contrasts),
        "confounds": list(model.confounds),
        "hrf_model": hrf_model,
        "drift_model": model.drift_model,
        "high_pass": model.high_pass,
        "drift_order": model.drift_order,
        "oversampling": model.oversampling,
        "min_onset": model.min_onset,
        "noise_model": model.noise_model,
    }
    if model.task_model is not None:
        activity["task_model"] = model.task_model.to_dict()
        activity["task_model_fingerprint"] = model.task_model.fingerprint
    fingerprint = None
    if not callable_warnings:
        fingerprint = activity.copy()
    return _ModelProvenance(activity, fingerprint, callable_warnings)


def _analysis_id(
    data_id: str | None,
    model: Mapping[str, object] | None,
) -> str | None:
    if model is None:
        return None
    return analysis_fingerprint(data_id, model)


def _serialize_hrf(
    value: object,
) -> tuple[object, tuple[Mapping[str, object], ...]]:
    if not callable(value):
        return value, ()
    identity = _callable_identity(value)
    if identity is not None:
        return {"kind": "callable", **identity}, ()
    warning = {
        "code": "reproducibility",
        "message": "HRF callable is not importable; reproducibility is partial",
    }
    return {"kind": "callable", "reproducibility": "partial"}, (warning,)


def _callable_identity(value: object) -> dict[str, str] | None:
    module_name = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if not isinstance(module_name, str) or not isinstance(qualname, str):
        return None
    if "<" in qualname or module_name not in sys.modules:
        return None
    resolved = sys.modules[module_name]
    try:
        for part in qualname.split("."):
            resolved = getattr(resolved, part)
    except AttributeError:
        return None
    if resolved is not value:
        return None
    return {"module": module_name, "qualname": qualname}


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
        "name": "task_delta_r2",
        "stage": "fit",
        "parent_analysis_id": parent_id,
        "definition": TASK_DELTA_R2_DEFINITION,
        "clip_below_zero": True,
        "diagnostic_noise_model": DIAGNOSTIC_NOISE_MODEL,
        "inferential_noise_model": model.noise_model,
        "clip_policy": "numerical_roundoff_guard",
        "nuisance_model": _nuisance_model_settings(model),
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
