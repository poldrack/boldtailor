"""Internal workflow mappings; annotations do not change serialized outputs."""

from typing import NotRequired, TypedDict

import numpy as np
import pandas as pd
from boldtailor.hrf_library import HrfLibrary
from boldtailor.hrf_results import HrfSelectionResult, HrfEvaluationResult
from boldtailor.hrf_options import RemlHrfOptions
from boldtailor.model import TaskModel
from boldtailor.ridge_results import CandidateScores, FractionSelection, RidgeSelection
from boldtailor.provenance import ProvenanceRecord


class SelectionBundle(TypedDict):
    all: HrfSelectionResult
    odd: NotRequired[HrfEvaluationResult]
    even: NotRequired[HrfEvaluationResult]


class HrfOptions(TypedDict, total=False):
    method: str
    reml_options: RemlHrfOptions | None
    task_model: TaskModel


class SelectionOptions(HrfOptions, total=False):
    library: HrfLibrary | None
    run_labels: list[str]
    feature_signature: str


class TrialOptions(TypedDict):
    ridge_alpha: float
    run_labels: list[str]
    ridge_fraction: NotRequired[np.ndarray]


class GlmBlock(TypedDict):
    effects: np.ndarray
    variances: np.ndarray
    t: np.ndarray
    z: np.ndarray
    r2: np.ndarray
    designs: dict[tuple[int, int], pd.DataFrame]
    provenance: dict[str, object]


class BetaBlock(TypedDict):
    betas: tuple[np.ndarray, ...]
    trial_table: pd.DataFrame
    r2: np.ndarray
    designs: dict[tuple[int, int], pd.DataFrame]
    provenance: dict[str, object]
    run_ridge_alphas: tuple[np.ndarray, ...] | None


class FitMetadata(TypedDict):
    designs: dict[tuple[int, int], pd.DataFrame]
    provenance: list[dict[str, object]]


class GlmFit(FitMetadata):
    effects: np.ndarray
    variances: np.ndarray
    t: np.ndarray
    z: np.ndarray
    r2: np.ndarray


class BetaFit(FitMetadata):
    betas: list[np.ndarray]
    r2: np.ndarray
    ridge_alpha: float | None
    ridge_fraction: NotRequired[np.ndarray]
    run_ridge_alphas: NotRequired[np.ndarray]
    trial_table: NotRequired[pd.DataFrame]
    rt: NotRequired[object]
    hrf_selections: NotRequired[dict[tuple[int, ...], SelectionBundle] | None]


class OuterEvaluation(TypedDict):
    encoding_mode: str
    selection_task_model: dict[str, object]
    coefficient_names: list[str]
    train_run_intercepts: np.ndarray
    scoring_offsets: np.ndarray
    encoding_r2: np.ndarray
    coefficients: np.ndarray
    run_sse: np.ndarray
    run_sst: np.ndarray
    hrf_indices: np.ndarray
    ridge_alpha: float | None
    provenance: list[dict[str, object]]
    train_run_labels: list[str]
    test_run_labels: list[str]
    ridge_fraction: NotRequired[np.ndarray]
    run_ridge_alphas: NotRequired[np.ndarray]
    betas: list[np.ndarray]
    predictions: list[np.ndarray]
    targets: list[np.ndarray]
    train_run_predictor_means: NotRequired[np.ndarray]
    predictor_means: NotRequired[np.ndarray]
    trial_masks: NotRequired[tuple[np.ndarray, ...]]
    tuning_analysis_fingerprint: NotRequired[str | None]


class OuterBlock(TypedDict):
    encoding_mode: str
    train_run_intercepts: np.ndarray
    train_run_predictor_means: np.ndarray
    scoring_offsets: np.ndarray
    encoding_r2: np.ndarray
    run_sse: np.ndarray
    run_sst: np.ndarray
    coefficients: np.ndarray
    predictor_means: np.ndarray
    predictions: tuple[np.ndarray, ...]
    betas: list[np.ndarray]
    targets: list[np.ndarray]
    hrf_indices: np.ndarray
    trial_masks: tuple[np.ndarray, ...]
    run_ridge_alphas: tuple[np.ndarray, ...] | None
    provenance: dict[str, object]
    selection_provenance: dict[str, object] | None


class TuningResult(TypedDict):
    scores: CandidateScores
    selection: FractionSelection | RidgeSelection
    run_labels: list[str]
    provenance: ProvenanceRecord
    summary_percentile: float


class CvBetaResult(TypedDict):
    tuning: dict[str, TuningResult]
    evaluation: dict[str, OuterEvaluation]
    predictors: list[pd.DataFrame]
    selection_task_model: dict[str, object]
    final: NotRequired[BetaFit]
    provenance: NotRequired[dict[str, object]]
