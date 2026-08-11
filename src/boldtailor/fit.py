from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import logging
import sys
import warnings
from uuid import uuid4

import numpy as np
import pandas as pd
from nilearn.glm import compute_contrast
from nilearn.glm.contrasts import expression_to_contrast_vector
from nilearn.glm.first_level import run_glm

from boldtailor.data import AnalysisData
from boldtailor.design import CompiledDesign, compile_designs, compile_nuisance_designs
from boldtailor.logging import append_event_history, bind_context, emit_event
from boldtailor.model import ContrastValue, ModelSpec
from boldtailor.provenance import analysis_fingerprint, extend_provenance
from boldtailor.results import (
    AnalysisResult,
    TaskDeltaR2Result,
    contrast_result,
    make_result,
    make_task_delta_r2_result,
)

_TASK_DELTA_R2_DEFINITION = "full_r2 - nuisance_r2"
_DIAGNOSTIC_NOISE_MODEL = "ols"
_NESTED_OLS_TOLERANCE = 1e-12


@dataclass(frozen=True)
class _RunFit:
    contrasts: dict[str, object]
    r2: np.ndarray
    residual_sum: np.ndarray
    total_sum: np.ndarray


@dataclass(frozen=True)
class _GLMFit:
    labels: np.ndarray
    regression_results: dict
    residual_sum: np.ndarray
    total_sum: np.ndarray


@dataclass(frozen=True)
class _R2Fit:
    residual_sum: np.ndarray
    total_sum: np.ndarray


@dataclass(frozen=True)
class _ModelProvenance:
    activity: dict[str, object]
    fingerprint: dict[str, object] | None
    warnings: tuple[Mapping[str, object], ...]


def fit(data: AnalysisData, model: ModelSpec) -> AnalysisResult:
    execution_id = str(uuid4())
    model_provenance = _model_provenance(model)
    data_id = data.provenance.metadata_fingerprint
    analysis_id = _analysis_id(data_id, model_provenance.fingerprint)
    history = data.provenance.events
    with bind_context(
        execution_id=execution_id,
        data_id=data_id,
        analysis_id=analysis_id,
    ):
        history = append_event_history(
            history,
            emit_event("fit_started", stage="fit"),
        )
        try:
            compiled, run_fits, combined, aggregate_r2 = _fit_analysis(data, model)
        except Exception as error:
            emit_event(
                "fit_failed",
                stage="fit",
                level=logging.ERROR,
                error=str(error),
            )
            raise
        history = append_event_history(
            history,
            emit_event("fit_completed", stage="fit"),
        )
    provenance = extend_provenance(
        data.provenance,
        execution_id=execution_id,
        activity=_fit_activity(data, model_provenance.activity, compiled),
        events=history,
        warnings=model_provenance.warnings,
        analysis_id=analysis_id,
    )
    return make_result(
        combined,
        tuple(design.matrix for design in compiled),
        tuple(_design_provenance(design) for design in compiled),
        tuple(run.r2 for run in run_fits),
        aggregate_r2,
        provenance,
    )


def task_delta_r2(
    data: AnalysisData,
    model: ModelSpec,
    full_result: AnalysisResult,
) -> TaskDeltaR2Result:
    execution_id = str(uuid4())
    model_provenance = _model_provenance(model)
    data_id = data.provenance.metadata_fingerprint
    expected_parent_id = _analysis_id(data_id, model_provenance.fingerprint)
    comparison_id = _comparison_id(expected_parent_id, model)
    history = full_result.provenance.events
    with bind_context(
        execution_id=execution_id,
        data_id=data_id,
        analysis_id=comparison_id,
    ):
        history = append_event_history(
            history,
            emit_event("task_delta_r2_started", stage="fit"),
        )
        try:
            _validate_parent_analysis(expected_parent_id, full_result)
            _validate_full_result_dimensions(data, full_result)
            full_designs = compile_designs(data, model)
            nuisance_designs = compile_nuisance_designs(data, model)
            full_r2 = _fit_r2_analysis(data, full_designs, _DIAGNOSTIC_NOISE_MODEL)
            nuisance_r2 = _fit_r2_analysis(
                data,
                nuisance_designs,
                _DIAGNOSTIC_NOISE_MODEL,
            )
            _validate_nested_ols_delta(full_r2 - nuisance_r2)
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
            provenance = extend_provenance(
                full_result.provenance,
                execution_id=execution_id,
                activity=activity,
                events=history,
                warnings=(),
                analysis_id=comparison_id,
            )
        except Exception as error:
            emit_event(
                "task_delta_r2_failed",
                stage="fit",
                level=logging.ERROR,
                error=str(error),
            )
            raise
        history = append_event_history(
            history,
            emit_event("task_delta_r2_completed", stage="fit"),
        )
        provenance = replace(provenance, events=history)
        comparison = replace(comparison, _provenance=provenance)
    return comparison


def _fit_analysis(
    data: AnalysisData,
    model: ModelSpec,
) -> tuple[
    tuple[CompiledDesign, ...],
    tuple[_RunFit, ...],
    dict[str, object],
    np.ndarray,
]:
    compiled = compile_designs(data, model)
    run_fits = tuple(
        _fit_run(signals, design, model, run)
        for run, (signals, design) in enumerate(
            zip(data.signals, compiled, strict=True)
        )
    )
    combined = _combine_contrasts(run_fits, model.contrast_names)
    aggregate_r2 = _r2_from_sums(
        np.sum([run.residual_sum for run in run_fits], axis=0),
        np.sum([run.total_sum for run in run_fits], axis=0),
    )
    return compiled, run_fits, combined, aggregate_r2


def _fit_r2_analysis(
    data: AnalysisData,
    compiled: tuple[CompiledDesign, ...],
    noise_model: str,
) -> np.ndarray:
    run_fits = tuple(
        _fit_r2_run(signals, design, noise_model, run)
        for run, (signals, design) in enumerate(
            zip(data.signals, compiled, strict=True)
        )
    )
    r2 = _r2_from_sums(
        np.sum([run.residual_sum for run in run_fits], axis=0),
        np.sum([run.total_sum for run in run_fits], axis=0),
    )
    if not np.isfinite(r2).all():
        raise ValueError("diagnostic fit produced nonfinite r-squared values")
    return r2


def _validate_nested_ols_delta(raw_delta_r2: np.ndarray) -> None:
    if np.any(raw_delta_r2 < -_NESTED_OLS_TOLERANCE):
        raise ValueError("nested OLS monotonicity violated")


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


def _validate_full_result_dimensions(
    data: AnalysisData,
    full_result: AnalysisResult,
) -> None:
    designs = full_result.design_matrices
    run_r2 = full_result.run_r2
    if len(designs) != data.n_runs or len(run_r2) != data.n_runs:
        raise ValueError("full result run dimensions do not match data")
    for run, (signals, design, values) in enumerate(
        zip(data.signals, designs, run_r2, strict=True)
    ):
        if design.shape[0] != signals.shape[0]:
            raise ValueError(f"full result run {run} dimensions do not match data")
        if values.shape != (data.n_features,):
            raise ValueError(
                f"full result run {run} feature dimensions do not match data"
            )
    if full_result.r2.shape != (data.n_features,):
        raise ValueError("full result feature dimensions do not match data")
    if not np.isfinite(full_result.r2).all():
        raise ValueError("full result r-squared values must be finite")


def _comparison_id(parent_id: str | None, model: ModelSpec) -> str | None:
    return _analysis_id(parent_id, _task_delta_r2_settings(model))


def _task_delta_r2_settings(model: ModelSpec) -> dict[str, object]:
    return {
        "name": "task_delta_r2",
        "definition": _TASK_DELTA_R2_DEFINITION,
        "clip_below_zero": True,
        "diagnostic_noise_model": _DIAGNOSTIC_NOISE_MODEL,
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
        "noise_model": _DIAGNOSTIC_NOISE_MODEL,
    }


def _model_provenance(model: ModelSpec) -> _ModelProvenance:
    hrf_model, callable_warnings = _serialize_hrf(model.hrf_model)
    activity = {
        "contrasts": _serialize_contrasts(model.contrasts),
        "confounds": list(model.confounds),
        "hrf_model": hrf_model,
        "drift_model": model.drift_model,
        "high_pass": model.high_pass,
        "drift_order": model.drift_order,
        "oversampling": model.oversampling,
        "min_onset": model.min_onset,
        "noise_model": model.noise_model,
    }
    fingerprint = None
    if not callable_warnings:
        fingerprint = {**activity, "drift_order": model.drift_order}
    return _ModelProvenance(activity, fingerprint, callable_warnings)


def _analysis_id(
    data_id: str | None,
    model: Mapping[str, object] | None,
) -> str | None:
    if model is None:
        return None
    return analysis_fingerprint(data_id, model)


def _serialize_contrasts(
    contrasts: Mapping[str, ContrastValue],
) -> dict[str, object]:
    return {name: _serialize_contrast(value) for name, value in contrasts.items()}


def _serialize_contrast(value: ContrastValue) -> dict[str, object]:
    if isinstance(value, str):
        return {"kind": "expression", "value": value}
    return {"kind": "weights", "weights": dict(value)}


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
        "definition": _TASK_DELTA_R2_DEFINITION,
        "clip_below_zero": True,
        "diagnostic_noise_model": _DIAGNOSTIC_NOISE_MODEL,
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
    warnings = _rank_warnings(rank, matrix.shape[1], run)
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


def _rank_warnings(rank: int, columns: int, run: int) -> list[str]:
    if rank == columns:
        return []
    return [f"run {run} design rank is {rank} for {columns} columns"]


def _fit_run(
    signals: np.ndarray,
    compiled: CompiledDesign,
    model: ModelSpec,
    run: int,
) -> _RunFit:
    design = compiled.matrix
    matrix = design.to_numpy()
    glm_fit = _fit_glm(signals, matrix, model.noise_model, run)
    contrasts = {
        name: _compute_contrast(
            glm_fit.labels,
            glm_fit.regression_results,
            value,
            design.columns,
            matrix,
            name,
            run,
        )
        for name, value in model.contrasts.items()
    }
    return _RunFit(
        contrasts=contrasts,
        r2=_r2_from_sums(glm_fit.residual_sum, glm_fit.total_sum),
        residual_sum=glm_fit.residual_sum,
        total_sum=glm_fit.total_sum,
    )


def _fit_r2_run(
    signals: np.ndarray,
    compiled: CompiledDesign,
    noise_model: str,
    run: int,
) -> _R2Fit:
    glm_fit = _fit_glm(
        signals,
        compiled.matrix.to_numpy(),
        noise_model,
        run,
    )
    return _R2Fit(glm_fit.residual_sum, glm_fit.total_sum)


def _fit_glm(
    signals: np.ndarray,
    design: np.ndarray,
    noise_model: str,
    run: int,
) -> _GLMFit:
    _warn_if_rank_deficient(design, run)
    _validate_residual_dof(design, run)
    labels, regression_results = run_glm(
        signals,
        design,
        noise_model=noise_model,
    )
    prediction = _prediction(labels, regression_results, design, signals.shape)
    residual_sum, total_sum = _sums_of_squares(signals, prediction)
    return _GLMFit(labels, regression_results, residual_sum, total_sum)


def _combine_contrasts(
    run_fits: tuple[_RunFit, ...],
    names: tuple[str, ...],
) -> dict[str, object]:
    return {
        name: contrast_result(_fixed_effects([run.contrasts[name] for run in run_fits]))
        for name in names
    }


def _fixed_effects(contrasts: list[object]) -> object:
    combined = contrasts[0]
    for contrast in contrasts[1:]:
        combined = combined + contrast
    return (1.0 / len(contrasts)) * combined


def _design_provenance(compiled: CompiledDesign) -> dict[str, int | float]:
    return {
        "excluded_event_count": compiled.excluded_event_count,
        "min_onset_cutoff": compiled.min_onset_cutoff,
    }


def _compute_contrast(
    labels: np.ndarray,
    regression_results: dict,
    value: ContrastValue,
    columns: pd.Index,
    design: np.ndarray,
    name: str,
    run: int,
) -> object:
    vector = _contrast_vector(value, columns, name, run)
    if not np.any(vector):
        raise ValueError(f"run {run} contrast {name!r} resolves to all zeros")
    _validate_estimable(vector, design, name, run)
    return _nilearn_t_contrast(labels, regression_results, vector)


def _nilearn_t_contrast(
    labels: np.ndarray,
    regression_results: dict,
    vector: np.ndarray,
) -> object:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"^divide by zero encountered in divide$",
            category=RuntimeWarning,
            module=r"^nilearn\.glm\._utils$",
        )
        return compute_contrast(
            labels,
            regression_results,
            vector,
            stat_type="t",
        )


def _contrast_vector(
    value: ContrastValue,
    columns: pd.Index,
    name: str,
    run: int,
) -> np.ndarray:
    if isinstance(value, str):
        try:
            return expression_to_contrast_vector(value, columns)
        except (KeyError, NameError, SyntaxError, TypeError, ValueError) as error:
            raise ValueError(
                f"run {run} contrast {name!r} is invalid: {error}"
            ) from error
    return _weight_vector(value, columns, name, run)


def _weight_vector(
    weights: Mapping[str, float],
    columns: pd.Index,
    name: str,
    run: int,
) -> np.ndarray:
    vector = np.zeros(len(columns), dtype=float)
    positions = {column: index for index, column in enumerate(columns)}
    for regressor, weight in weights.items():
        if weight == 0.0:
            continue
        if regressor not in positions:
            raise ValueError(
                f"run {run} contrast {name!r} references missing "
                f"regressor {regressor!r}"
            )
        vector[positions[regressor]] = weight
    return vector


def _validate_estimable(
    vector: np.ndarray,
    design: np.ndarray,
    name: str,
    run: int,
) -> None:
    projection = vector @ np.linalg.pinv(design) @ design
    if not np.allclose(vector, projection, rtol=1e-7, atol=1e-9):
        raise ValueError(f"run {run} contrast {name!r} is not estimable")


def _warn_if_rank_deficient(design: np.ndarray, run: int) -> None:
    rank = np.linalg.matrix_rank(design)
    if rank < design.shape[1]:
        warnings.warn(
            f"run {run} design rank is {rank} for {design.shape[1]} columns",
            UserWarning,
            stacklevel=2,
        )


def _validate_residual_dof(design: np.ndarray, run: int) -> None:
    residual_dof = design.shape[0] - np.linalg.matrix_rank(design)
    if residual_dof <= 0:
        raise ValueError(
            f"run {run} has residual degrees of freedom {residual_dof}; "
            "contrast inference requires a positive value"
        )


def _prediction(
    labels: np.ndarray,
    regression_results: dict,
    design: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    prediction = np.empty(shape, dtype=float)
    for label, result in regression_results.items():
        prediction[:, labels == label] = design @ result.theta
    return prediction


def _sums_of_squares(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    residual_sum = np.sum((observed - predicted) ** 2, axis=0)
    total_sum = np.sum((observed - observed.mean(axis=0)) ** 2, axis=0)
    return residual_sum, total_sum


def _r2_from_sums(
    residual_sum: np.ndarray,
    total_sum: np.ndarray,
) -> np.ndarray:
    ratio = np.full(residual_sum.shape, np.nan, dtype=float)
    np.divide(residual_sum, total_sum, out=ratio, where=total_sum > 0)
    return 1.0 - ratio
