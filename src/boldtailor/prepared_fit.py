from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import logging
from types import MappingProxyType
from uuid import uuid4

import numpy as np
import pandas as pd

from boldtailor._conventional import fit_designs, fit_r2_designs
from boldtailor._fit_diagnostics import (
    TASK_DELTA_R2_DEFINITION,
    DIAGNOSTIC_NOISE_MODEL,
    NESTED_OLS_TOLERANCE,
    rank_warnings,
    validate_nested_ols_delta,
    validate_result_dimensions,
)
from boldtailor._software import package_version
from boldtailor.logging import append_event_history, bind_context, emit_event
from boldtailor.model import (
    ContrastValue,
    _prepare_contrasts,
    _validate_noise_model,
    contrast_metadata,
)
from boldtailor.prepared import PreparedDesignAnalysis
from boldtailor.provenance import (
    ProvenanceRecord,
    _freeze_mapping,
    analysis_fingerprint,
    extend_provenance,
)
from boldtailor.results import (
    AnalysisResult,
    TaskDeltaR2Result,
    make_result,
    make_task_delta_r2_result,
)


@dataclass(frozen=True)
class _PreparedFitSpec:
    contrasts: Mapping[str, ContrastValue]
    noise_model: str
    model_metadata: Mapping[str, object]


def fit_prepared(
    prepared: PreparedDesignAnalysis,
    *,
    contrasts: Mapping[str, ContrastValue],
    noise_model: str = "ar1",
    model_metadata: Mapping[str, object] | None = None,
) -> AnalysisResult:
    execution_id = str(uuid4())
    data_id = prepared.provenance.metadata_fingerprint
    history = prepared.provenance.events
    with bind_context(execution_id=execution_id, data_id=data_id):
        try:
            fit_spec = _prepare_fit_spec(contrasts, noise_model, model_metadata)
            model = _model_identity(prepared, fit_spec)
            analysis_id = analysis_fingerprint(data_id, model)
        except Exception as error:
            _log_validation_failure(history, error)
            raise
        with bind_context(analysis_id=analysis_id):
            return _fit_with_lifecycle(
                prepared,
                fit_spec,
                model,
                execution_id,
                analysis_id,
                history,
            )


def task_delta_r2_prepared(
    prepared: PreparedDesignAnalysis,
    full_result: AnalysisResult,
    *,
    contrasts: Mapping[str, ContrastValue],
    noise_model: str = "ar1",
    model_metadata: Mapping[str, object] | None = None,
) -> TaskDeltaR2Result:
    return _compare_prepared_models(
        prepared,
        full_result,
        contrasts=contrasts,
        noise_model=noise_model,
        model_metadata=model_metadata,
    )


def _compare_prepared_models(
    prepared: PreparedDesignAnalysis,
    full_result: AnalysisResult,
    *,
    contrasts: Mapping[str, ContrastValue],
    noise_model: str,
    model_metadata: Mapping[str, object] | None,
) -> TaskDeltaR2Result:
    execution_id = str(uuid4())
    data_id = prepared.provenance.metadata_fingerprint
    history = full_result.provenance.events
    with bind_context(execution_id=execution_id, data_id=data_id):
        try:
            fit_spec, parent_id, comparison_id = _prepare_comparison_identity(
                prepared,
                contrasts,
                noise_model,
                model_metadata,
                data_id,
            )
        except Exception as error:
            _log_comparison_validation_failure(history, error)
            raise
        with bind_context(analysis_id=comparison_id):
            return _comparison_with_lifecycle(
                prepared,
                full_result,
                fit_spec,
                parent_id,
                comparison_id,
                execution_id,
                history,
            )


def _prepare_comparison_identity(
    prepared: PreparedDesignAnalysis,
    contrasts: Mapping[str, ContrastValue],
    noise_model: str,
    model_metadata: Mapping[str, object] | None,
    data_id: str | None,
) -> tuple[_PreparedFitSpec, str | None, str | None]:
    fit_spec = _prepare_fit_spec(contrasts, noise_model, model_metadata)
    parent_id = analysis_fingerprint(data_id, _model_identity(prepared, fit_spec))
    comparison_id = _prepared_comparison_id(data_id, parent_id, fit_spec)
    return fit_spec, parent_id, comparison_id


def _comparison_with_lifecycle(
    prepared: PreparedDesignAnalysis,
    full_result: AnalysisResult,
    fit_spec: _PreparedFitSpec,
    parent_id: str | None,
    comparison_id: str | None,
    execution_id: str,
    history: tuple[Mapping[str, object], ...],
) -> TaskDeltaR2Result:
    history = append_event_history(
        history,
        emit_event("task_delta_r2_prepared_started", stage="fit"),
    )
    try:
        comparison, provenance = _run_prepared_comparison(
            prepared,
            full_result,
            fit_spec,
            parent_id,
            comparison_id,
            execution_id,
            history,
        )
    except Exception as error:
        emit_event(
            "task_delta_r2_prepared_failed",
            stage="fit",
            level=logging.ERROR,
            error=error,
        )
        raise
    history = append_event_history(
        history,
        emit_event("task_delta_r2_prepared_completed", stage="fit"),
    )
    return replace(comparison, _provenance=_with_events(provenance, history))


def _log_comparison_validation_failure(
    history: tuple[Mapping[str, object], ...],
    error: Exception,
) -> None:
    append_event_history(
        history,
        emit_event("task_delta_r2_prepared_started", stage="fit"),
    )
    emit_event(
        "task_delta_r2_prepared_failed",
        stage="fit",
        level=logging.ERROR,
        error=error,
    )


def _run_prepared_comparison(
    prepared: PreparedDesignAnalysis,
    full_result: AnalysisResult,
    fit_spec: _PreparedFitSpec,
    parent_id: str | None,
    comparison_id: str | None,
    execution_id: str,
    history: tuple[Mapping[str, object], ...],
) -> tuple[TaskDeltaR2Result, ProvenanceRecord]:
    _validate_prepared_parent(prepared, full_result, parent_id)
    nuisance_designs = _nuisance_designs(prepared)
    full_r2 = _fit_prepared_r2(prepared, prepared._design_matrices)
    nuisance_r2 = _fit_prepared_r2(prepared, nuisance_designs)
    validate_nested_ols_delta(full_r2 - nuisance_r2)
    comparison = make_task_delta_r2_result(
        full_r2=full_r2,
        nuisance_r2=nuisance_r2,
        nuisance_designs=nuisance_designs,
        provenance=full_result.provenance,
    )
    activity = _prepared_delta_activity(
        prepared,
        fit_spec,
        nuisance_designs,
        comparison,
        parent_id,
    )
    provenance = extend_provenance(
        full_result.provenance,
        execution_id=execution_id,
        activity=activity,
        events=history,
        warnings=(),
        analysis_id=comparison_id,
    )
    return comparison, provenance


def _prepared_comparison_id(
    data_id: str | None,
    parent_id: str | None,
    fit_spec: _PreparedFitSpec,
) -> str | None:
    return analysis_fingerprint(
        data_id,
        {
            "name": "task_delta_r2_prepared",
            "parent_analysis_id": parent_id,
            "definition": TASK_DELTA_R2_DEFINITION,
            "diagnostic_noise_model": DIAGNOSTIC_NOISE_MODEL,
            "inferential_noise_model": fit_spec.noise_model,
        },
    )


def _validate_prepared_parent(
    prepared: PreparedDesignAnalysis,
    full_result: AnalysisResult,
    parent_id: str | None,
) -> None:
    if parent_id is None:
        raise ValueError(
            "task delta r-squared requires fingerprintable prepared sources"
        )
    if full_result.provenance.analysis_fingerprint != parent_id:
        raise ValueError("full result does not match prepared input and model")
    validate_result_dimensions(prepared, full_result, input_label="prepared input")


def _nuisance_designs(
    prepared: PreparedDesignAnalysis,
) -> tuple[pd.DataFrame, ...]:
    designs = []
    for run, (matrix, roles) in enumerate(
        zip(prepared._design_matrices, prepared._column_roles, strict=True)
    ):
        if "other" in roles.values():
            raise ValueError(
                f"run {run} column roles are incomplete for task delta r-squared"
            )
        task = [name for name in matrix if roles[name] == "task"]
        nuisance = [name for name in matrix if roles[name] in {"nuisance", "intercept"}]
        if not task:
            raise ValueError(f"run {run} requires at least one task column")
        if not nuisance:
            raise ValueError(f"run {run} requires a nuisance or intercept column")
        designs.append(matrix.loc[:, nuisance])
    return tuple(designs)


def _fit_prepared_r2(
    prepared: PreparedDesignAnalysis,
    designs: tuple[pd.DataFrame, ...],
) -> np.ndarray:
    r2 = fit_r2_designs(prepared.signals, designs, DIAGNOSTIC_NOISE_MODEL)
    if not np.isfinite(r2).all():
        raise ValueError("diagnostic fit produced nonfinite r-squared values")
    return r2


def _prepared_delta_activity(
    prepared: PreparedDesignAnalysis,
    fit_spec: _PreparedFitSpec,
    nuisance_designs: tuple[pd.DataFrame, ...],
    comparison: TaskDeltaR2Result,
    parent_id: str,
) -> dict[str, object]:
    return {
        "name": "task_delta_r2_prepared",
        "stage": "fit",
        "parent_analysis_id": parent_id,
        "definition": TASK_DELTA_R2_DEFINITION,
        "diagnostic_noise_model": DIAGNOSTIC_NOISE_MODEL,
        "inferential_noise_model": fit_spec.noise_model,
        "clip_policy": "numerical_roundoff_guard",
        "roundoff_tolerance": NESTED_OLS_TOLERANCE,
        "nuisance_rule": "column roles nuisance or intercept",
        "runs": tuple(
            _prepared_delta_run_diagnostic(prepared, design, run)
            for run, design in enumerate(nuisance_designs)
        ),
        "diagnostics": {
            "raw_min": comparison.raw_min,
            "negative_voxel_count": comparison.negative_voxel_count,
            "mean_delta_r2": float(comparison.delta_r2.mean()),
            "max_delta_r2": float(comparison.delta_r2.max()),
        },
    }


def _prepared_delta_run_diagnostic(
    prepared: PreparedDesignAnalysis,
    nuisance_design: pd.DataFrame,
    run: int,
) -> dict[str, object]:
    diagnostic = _run_diagnostic(prepared, run)
    diagnostic["nuisance_columns"] = list(nuisance_design.columns)
    return diagnostic


def _fit_with_lifecycle(
    prepared: PreparedDesignAnalysis,
    fit_spec: _PreparedFitSpec,
    model: Mapping[str, object],
    execution_id: str,
    analysis_id: str | None,
    history: tuple[Mapping[str, object], ...],
) -> AnalysisResult:
    designs = prepared._design_matrices
    history = append_event_history(history, emit_event("fit_started", stage="fit"))
    try:
        numerical = fit_designs(
            prepared.signals,
            designs,
            fit_spec.contrasts,
            fit_spec.noise_model,
        )
        provenance = _fit_provenance(
            prepared,
            execution_id=execution_id,
            activity=_fit_activity(prepared, model),
            events=history,
            analysis_id=analysis_id,
        )
    except Exception as error:
        emit_event(
            "fit_failed",
            stage="fit",
            level=logging.ERROR,
            error=error,
        )
        raise
    history = append_event_history(history, emit_event("fit_completed", stage="fit"))
    provenance = _with_events(provenance, history)
    return make_result(
        numerical.contrasts,
        designs,
        _prepared_design_provenance(prepared),
        tuple(run.r2 for run in numerical.run_fits),
        numerical.aggregate_r2,
        provenance,
    )


def _log_validation_failure(
    history: tuple[Mapping[str, object], ...],
    error: Exception,
) -> None:
    append_event_history(history, emit_event("fit_started", stage="fit"))
    emit_event(
        "fit_failed",
        stage="fit",
        level=logging.ERROR,
        error=error,
    )


def _prepare_fit_spec(
    contrasts: Mapping[str, ContrastValue],
    noise_model: str,
    model_metadata: Mapping[str, object] | None,
) -> _PreparedFitSpec:
    _validate_path_safe_model_keys(contrasts, model_metadata)
    if not isinstance(contrasts, Mapping):
        raise ValueError("contrasts must be a mapping")
    normalized = _prepare_contrasts(contrasts)
    _validate_noise_model(noise_model)
    metadata = _freeze_mapping(model_metadata, path_safe=True)
    return _PreparedFitSpec(
        contrasts=MappingProxyType(normalized),
        noise_model=noise_model,
        model_metadata=metadata,
    )


def _validate_path_safe_model_keys(
    contrasts: object,
    model_metadata: object,
) -> None:
    _validate_path_safe_mapping_keys(contrasts)
    _validate_path_safe_mapping_keys(model_metadata)


def _validate_path_safe_mapping_keys(values: object) -> None:
    if isinstance(values, Mapping):
        _validate_mapping_keys(values)
        return
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
        for value in values:
            _validate_path_safe_mapping_keys(value)


def _validate_mapping_keys(values: Mapping[object, object]) -> None:
    for key, value in values.items():
        if _is_path_like_key(key):
            raise ValueError("model identity keys must not be path-like")
        _validate_path_safe_mapping_keys(value)


def _is_path_like_key(value: object) -> bool:
    if not isinstance(value, str):
        return False
    return value.startswith(("~", "/", "../", "./", "bids:")) or "\\" in value


def _prepared_design_provenance(
    prepared: PreparedDesignAnalysis,
) -> tuple[dict[str, str], ...]:
    return tuple(
        {"source": "prepared", "design_fingerprint": fingerprint}
        for fingerprint in prepared.run_design_fingerprints
    )


def _model_identity(
    prepared: PreparedDesignAnalysis,
    fit_spec: _PreparedFitSpec,
) -> dict[str, object]:
    return {
        "kind": "prepared_design",
        "contrasts": contrast_metadata(fit_spec.contrasts),
        "noise_model": fit_spec.noise_model,
        "metadata": dict(fit_spec.model_metadata),
        "design_fingerprint": prepared.design_fingerprint,
    }


def _fit_provenance(
    prepared: PreparedDesignAnalysis,
    *,
    execution_id: str,
    activity: Mapping[str, object],
    events: tuple[Mapping[str, object], ...],
    analysis_id: str | None,
) -> ProvenanceRecord:
    return extend_provenance(
        prepared.provenance,
        execution_id=execution_id,
        activity=activity,
        events=events,
        warnings=(),
        analysis_id=analysis_id,
    )


def _fit_activity(
    prepared: PreparedDesignAnalysis,
    model: Mapping[str, object],
) -> dict[str, object]:
    return {
        "name": "fit_prepared",
        "stage": "fit",
        "numerical_backend": {"name": "nilearn", "version": package_version("nilearn")},
        "model": model,
        "runs": tuple(_run_diagnostic(prepared, run) for run in range(prepared.n_runs)),
    }


def _with_events(
    provenance: ProvenanceRecord,
    events: tuple[Mapping[str, object], ...],
) -> ProvenanceRecord:
    payload = provenance.to_dict()
    payload["events"] = list(events)
    return ProvenanceRecord.from_dict(payload)


def _run_diagnostic(
    prepared: PreparedDesignAnalysis,
    run: int,
) -> dict[str, object]:
    design = prepared._design_matrices[run]
    matrix = design.to_numpy()
    rank = int(np.linalg.matrix_rank(matrix))
    return {
        "n_scans": int(matrix.shape[0]),
        "n_features": int(prepared.signals[run].shape[1]),
        "design_columns": list(design.columns),
        "role_counts": _role_counts(prepared._column_roles[run]),
        "design_rank": rank,
        "residual_dof": int(matrix.shape[0] - rank),
        "run_design_fingerprint": prepared.run_design_fingerprints[run],
        "warnings": rank_warnings(rank, matrix.shape[1], run),
    }


def _role_counts(roles: Mapping[str, str]) -> dict[str, int]:
    return {
        role: sum(value == role for value in roles.values())
        for role in ("task", "nuisance", "intercept", "other")
    }
