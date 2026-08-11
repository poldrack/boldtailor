from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from uuid import uuid4

from boldtailor._conventional import fit_designs
from boldtailor.model import ContrastValue, _prepare_contrasts, _validate_noise_model
from boldtailor.prepared import PreparedDesignAnalysis
from boldtailor.provenance import _freeze_mapping, extend_provenance
from boldtailor.results import AnalysisResult, make_result


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
    fit_spec = _prepare_fit_spec(contrasts, noise_model, model_metadata)
    numerical = fit_designs(
        prepared.signals,
        prepared.design_matrices,
        fit_spec.contrasts,
        fit_spec.noise_model,
    )
    provenance = _fit_provenance(prepared, fit_spec)
    return make_result(
        numerical.contrasts,
        prepared.design_matrices,
        _prepared_design_provenance(prepared),
        tuple(run.r2 for run in numerical.run_fits),
        numerical.aggregate_r2,
        provenance,
    )


def _prepare_fit_spec(
    contrasts: Mapping[str, ContrastValue],
    noise_model: str,
    model_metadata: Mapping[str, object] | None,
) -> _PreparedFitSpec:
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


def _prepared_design_provenance(
    prepared: PreparedDesignAnalysis,
) -> tuple[dict[str, str], ...]:
    return tuple(
        {"source": "prepared", "design_fingerprint": fingerprint}
        for fingerprint in prepared.run_design_fingerprints
    )


def _fit_provenance(
    prepared: PreparedDesignAnalysis,
    fit_spec: _PreparedFitSpec,
):
    return extend_provenance(
        prepared.provenance,
        execution_id=str(uuid4()),
        activity={
            "name": "fit_prepared",
            "stage": "fit",
            "model": {
                "contrasts": _provenance_contrasts(fit_spec.contrasts),
                "noise_model": fit_spec.noise_model,
                "metadata": fit_spec.model_metadata,
            },
        },
        events=prepared.provenance.events,
        warnings=(),
        analysis_id=None,
    )


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
