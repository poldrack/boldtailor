from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType

import numpy as np
import pandas as pd

from boldtailor._conventional import fit_designs
from boldtailor._fit_diagnostics import (
    DIAGNOSTIC_NOISE_MODEL,
    NESTED_OLS_TOLERANCE,
    rank_warnings,
    delta_r2_activity,
    delta_r2_identity,
    nested_ols_delta,
    validate_result_dimensions,
)
from boldtailor._fit_lifecycle import fit_operation
from boldtailor.model import (
    ContrastValue,
    _prepare_contrasts,
    _validate_noise_model,
    contrast_metadata,
)
from boldtailor.prepared import PreparedDesignAnalysis
from boldtailor.provenance import (
    _freeze_mapping,
    analysis_fingerprint,
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
    with fit_operation("fit", prepared.provenance) as operation:
        fit_spec = _prepare_fit_spec(contrasts, noise_model, model_metadata)
        model = _model_identity(prepared, fit_spec)
        operation.analysis_id = analysis_fingerprint(
            prepared.provenance.metadata_fingerprint, model
        )
        designs = prepared._design_matrices
        numerical = fit_designs(
            prepared.signals, designs, fit_spec.contrasts, fit_spec.noise_model
        )
        provenance = operation.provenance(
            _fit_activity(prepared, model), analysis_id=operation.analysis_id
        )
        return make_result(
            numerical.contrasts,
            designs,
            _prepared_design_provenance(prepared),
            tuple(run.r2 for run in numerical.run_fits),
            numerical.aggregate_r2,
            provenance,
        )


def task_delta_r2_prepared(
    prepared: PreparedDesignAnalysis,
    full_result: AnalysisResult,
    *,
    contrasts: Mapping[str, ContrastValue],
    noise_model: str = "ar1",
    model_metadata: Mapping[str, object] | None = None,
) -> TaskDeltaR2Result:
    with fit_operation("task_delta_r2_prepared", full_result.provenance) as operation:
        fit_spec, parent_id, comparison_id = _prepare_comparison_identity(
            prepared,
            contrasts,
            noise_model,
            model_metadata,
            prepared.provenance.metadata_fingerprint,
        )
        operation.analysis_id = comparison_id
        _validate_prepared_parent(prepared, full_result, parent_id)
        nuisance_designs = _nuisance_designs(prepared)
        full_r2, nuisance_r2 = nested_ols_delta(
            prepared.signals,
            prepared._design_matrices,
            nuisance_designs,
            allow_undefined=False,
        )
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
        provenance = operation.provenance(activity, analysis_id=comparison_id)
        return replace(comparison, _provenance=provenance)


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


def _prepared_comparison_id(
    data_id: str | None,
    parent_id: str | None,
    fit_spec: _PreparedFitSpec,
) -> str | None:
    identity = delta_r2_identity(
        name="task_delta_r2_prepared",
        inferential_noise_model=fit_spec.noise_model,
        nuisance_model=_PREPARED_NUISANCE_MODEL,
    )
    return analysis_fingerprint(parent_id, identity)


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
        raise ValueError(
            "full result provenance identity does not match this data and model (identity uses source metadata and sha256 when supplied)"
        )
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


_PREPARED_NUISANCE_MODEL = {
    "events": False,
    "column_roles": ["nuisance", "intercept"],
    "noise_model": DIAGNOSTIC_NOISE_MODEL,
}


def _prepared_delta_activity(
    prepared: PreparedDesignAnalysis,
    fit_spec: _PreparedFitSpec,
    nuisance_designs: tuple[pd.DataFrame, ...],
    comparison: TaskDeltaR2Result,
    parent_id: str,
) -> dict[str, object]:
    return {
        **delta_r2_activity(
            name="task_delta_r2_prepared",
            parent_id=parent_id,
            inferential_noise_model=fit_spec.noise_model,
            nuisance_model=dict(_PREPARED_NUISANCE_MODEL),
            undefined_features=0,
        ),
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


def _prepare_fit_spec(
    contrasts: Mapping[str, ContrastValue],
    noise_model: str,
    model_metadata: Mapping[str, object] | None,
) -> _PreparedFitSpec:
    if not isinstance(contrasts, Mapping):
        raise ValueError("contrasts must be a mapping")
    normalized = _prepare_contrasts(contrasts)
    _validate_noise_model(noise_model)
    metadata = _freeze_mapping(model_metadata)
    return _PreparedFitSpec(
        contrasts=MappingProxyType(normalized),
        noise_model=noise_model,
        model_metadata=metadata,
    )


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
