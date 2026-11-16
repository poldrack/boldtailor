"""Dispatch conventional and selected-HRF GLM operations."""

from __future__ import annotations

from boldtailor._hrf_glm import fit_selected_glm, selected_task_delta_r2
from boldtailor._standard_glm import fit_standard_glm, standard_task_delta_r2
from boldtailor.data import AnalysisData
from boldtailor.model import ModelSpec
from boldtailor.hrf_results import HrfSelectionResult
from boldtailor.hrf_glm_results import HrfAnalysisResult
from boldtailor.results import AnalysisResult, TaskDeltaR2Result


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
    Selected-HRF results expose rebuild_group_design(run, hrf_id), recompiled on
    request, instead of design_matrices.
    """
    if hrf_selection is not None:
        return fit_selected_glm(data, model, hrf_selection, feature_signature)
    return fit_standard_glm(data, model, feature_signature=feature_signature)


def task_delta_r2(
    data: AnalysisData,
    model: ModelSpec,
    full_result: AnalysisResult | HrfAnalysisResult,
) -> TaskDeltaR2Result:
    """Compare the fitted task model with its nuisance-only model."""
    if isinstance(full_result, HrfAnalysisResult):
        return selected_task_delta_r2(data, model, full_result)
    return standard_task_delta_r2(data, model, full_result)
