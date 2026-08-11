from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import logging
import os
import re
from types import MappingProxyType
from uuid import uuid4

import numpy as np

from boldtailor._conventional import fit_designs
from boldtailor.logging import append_event_history, bind_context, emit_event
from boldtailor.model import ContrastValue, _prepare_contrasts, _validate_noise_model
from boldtailor.prepared import PreparedDesignAnalysis
from boldtailor.provenance import (
    ProvenanceRecord,
    _freeze_mapping,
    analysis_fingerprint,
    extend_provenance,
)
from boldtailor.results import AnalysisResult, make_result

_PATH_PATTERN = re.compile(r"(?<![\w.-])/(?:[^\s'\"<>]+)")
_ADDRESS_PATTERN = re.compile(r"0x[0-9a-fA-F]+")
_OBJECT_REPR_PATTERN = re.compile(r"<[^>\n]*\bobject\b[^>\n]*>")
_TRACEBACK_PATTERN = re.compile(r"Traceback \(most recent call last\):.*", re.DOTALL)


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


def _fit_with_lifecycle(
    prepared: PreparedDesignAnalysis,
    fit_spec: _PreparedFitSpec,
    model: Mapping[str, object],
    execution_id: str,
    analysis_id: str | None,
    history: tuple[Mapping[str, object], ...],
) -> AnalysisResult:
    history = append_event_history(history, emit_event("fit_started", stage="fit"))
    try:
        numerical = fit_designs(
            prepared.signals,
            prepared.design_matrices,
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
            error=_sanitize_error(error),
        )
        raise
    history = append_event_history(history, emit_event("fit_completed", stage="fit"))
    provenance = _with_events(provenance, history)
    return make_result(
        numerical.contrasts,
        prepared.design_matrices,
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
        error=_sanitize_error(error),
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
    if not isinstance(values, Mapping):
        return
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
        "contrasts": _provenance_contrasts(fit_spec.contrasts),
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
    design = prepared.design_matrices[run]
    matrix = design.to_numpy()
    rank = int(np.linalg.matrix_rank(matrix))
    return {
        "n_scans": int(matrix.shape[0]),
        "n_features": int(prepared.signals[run].shape[1]),
        "design_columns": list(design.columns),
        "role_counts": _role_counts(prepared.column_roles[run]),
        "design_rank": rank,
        "residual_dof": int(matrix.shape[0] - rank),
        "run_design_fingerprint": prepared.run_design_fingerprints[run],
        "warnings": _rank_warnings(rank, matrix.shape[1], run),
    }


def _role_counts(roles: Mapping[str, str]) -> dict[str, int]:
    return {
        role: sum(value == role for value in roles.values())
        for role in ("task", "nuisance", "intercept", "other")
    }


def _rank_warnings(rank: int, columns: int, run: int) -> list[str]:
    if rank == columns:
        return []
    return [f"run {run} design rank is {rank} for {columns} columns"]


def _sanitize_error(error: Exception) -> str:
    message = _TRACEBACK_PATTERN.sub("<redacted-traceback>", str(error))
    message = _PATH_PATTERN.sub("<redacted>", message)
    message = _OBJECT_REPR_PATTERN.sub("<redacted-object>", message)
    message = _redact_environment_values(message)
    return _ADDRESS_PATTERN.sub("<redacted-address>", message)


def _redact_environment_values(message: str) -> str:
    for name, value in os.environ.items():
        if value and (len(value) >= 8 or _is_sensitive_environment_name(name)):
            message = message.replace(value, "<redacted>")
    return message


def _is_sensitive_environment_name(name: str) -> bool:
    return any(term in name.upper() for term in ("KEY", "PASSWORD", "SECRET", "TOKEN"))


def _provenance_contrasts(
    contrasts: Mapping[str, ContrastValue],
) -> dict[str, object]:
    return {
        name: (
            {"kind": "expression", "value": value}
            if isinstance(value, str)
            else {"kind": "weights", "weights": dict(value)}
        )
        for name, value in contrasts.items()
    }
