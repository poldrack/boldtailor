"""Workflow records: frozen assembled results and the per-block worker payloads."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, NotRequired, TypedDict

import numpy as np
import pandas as pd
from boldtailor.hrf_library import HrfLibrary
from boldtailor.hrf_results import HrfSelectionResult, HrfEvaluationResult
from boldtailor.hrf_options import RemlHrfOptions
from boldtailor.model import TaskModel
from boldtailor.ridge_results import CandidateScores, FractionSelection, RidgeSelection
from boldtailor.provenance import ProvenanceRecord

Regularization = Literal["normalized_ridge", "fractional_ridge"]
BlockRecord = Mapping[str, object]


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


@dataclass(frozen=True, kw_only=True, eq=False)
class GlmFit:
    """Session-wide GLM maps (rows are task regressors) assembled from blocks."""

    effects: np.ndarray
    variances: np.ndarray
    t: np.ndarray
    z: np.ndarray
    r2: np.ndarray
    designs: Mapping[tuple[int, int], pd.DataFrame]
    provenance: tuple[BlockRecord, ...]


@dataclass(frozen=True, kw_only=True, eq=False)
class BetaFit:
    """Session-wide trial betas at one penalty, assembled from blocks."""

    betas: tuple[np.ndarray, ...]
    r2: np.ndarray
    designs: Mapping[tuple[int, int], pd.DataFrame]
    provenance: tuple[BlockRecord, ...]
    trial_table: pd.DataFrame | None
    regularization: Regularization
    ridge_alpha: float | None
    ridge_fraction: np.ndarray | None = None
    run_ridge_alphas: np.ndarray | None = None
    rt: Mapping[str, np.ndarray] | None = None
    hrf_selections: Mapping[tuple[int, ...], SelectionBundle] | None = None

    @property
    def fractional(self) -> bool:
        return self.regularization == "fractional_ridge"


@dataclass(frozen=True, kw_only=True, eq=False)
class OuterEvaluation:
    """Held-out encoding of one odd/even split; trial arrays hold test runs only."""

    regularization: Regularization
    encoding_mode: str
    selection_task_model: Mapping[str, object]
    coefficient_names: tuple[str, ...]
    train_run_labels: tuple[str, ...]
    test_run_labels: tuple[str, ...]
    train_run_intercepts: np.ndarray
    train_run_predictor_means: np.ndarray
    scoring_offsets: np.ndarray
    encoding_r2: np.ndarray
    coefficients: np.ndarray
    predictor_means: np.ndarray
    run_sse: np.ndarray
    run_sst: np.ndarray
    hrf_indices: np.ndarray
    trial_masks: tuple[np.ndarray, ...]
    betas: tuple[np.ndarray, ...]
    predictions: tuple[np.ndarray, ...]
    targets: tuple[np.ndarray, ...]
    provenance: tuple[BlockRecord, ...]
    tuning_analysis_fingerprint: str | None
    ridge_alpha: float | None = None
    ridge_fraction: np.ndarray | None = None
    run_ridge_alphas: np.ndarray | None = None

    @property
    def fractional(self) -> bool:
        return self.regularization == "fractional_ridge"


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


@dataclass(frozen=True, kw_only=True, eq=False)
class TuningResult:
    """Candidate scores for one training scope and the penalty chosen from them."""

    scores: CandidateScores
    selection: FractionSelection | RidgeSelection
    run_labels: tuple[str, ...]
    provenance: ProvenanceRecord
    summary_percentile: float

    @property
    def regularization(self) -> Regularization:
        return self.scores.regularization

    @property
    def fractional(self) -> bool:
        return self.regularization == "fractional_ridge"


@dataclass(frozen=True, kw_only=True, eq=False)
class CvBetaResult:
    """Tuning per scope, both outer evaluations, and the all-run refit."""

    tuning: Mapping[str, TuningResult]
    evaluation: Mapping[str, OuterEvaluation]
    predictors: tuple[pd.DataFrame, ...]
    selection_task_model: Mapping[str, object]
    final: BetaFit
    provenance: Mapping[str, object]
