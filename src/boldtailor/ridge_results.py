"""Owned numerical results for trial encoding and ridge selection."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from boldtailor._arrays import readonly_array
from boldtailor.provenance import ProvenanceRecord


@dataclass(frozen=True)
class TrialEncodingResult:
    coefficients: np.ndarray
    predictor_means: np.ndarray
    predictor_names: tuple[str, ...]
    train_runs: tuple[int, ...]
    test_runs: tuple[int, ...]
    trial_masks: tuple[np.ndarray, ...]
    predictions: tuple[np.ndarray, ...]
    run_sse: np.ndarray
    run_sst: np.ndarray
    r2: np.ndarray
    encoding_mode: str
    train_run_predictor_means: np.ndarray
    train_run_intercepts: np.ndarray
    scoring_offsets: np.ndarray

    def __post_init__(self):
        for name in (
            "coefficients",
            "predictor_means",
            "run_sse",
            "run_sst",
            "r2",
            "train_run_predictor_means",
            "train_run_intercepts",
            "scoring_offsets",
        ):
            object.__setattr__(self, name, readonly_array(getattr(self, name)))
        object.__setattr__(
            self,
            "predictions",
            tuple(readonly_array(a) for a in self.predictions),
        )
        object.__setattr__(
            self,
            "trial_masks",
            tuple(readonly_array(a, dtype=bool) for a in self.trial_masks),
        )
        for name in ("predictor_names", "train_runs", "test_runs"):
            object.__setattr__(self, name, tuple(getattr(self, name)))


@dataclass(frozen=True)
class RidgeSelection:
    ridge_alpha: float
    alphas: tuple[float, ...]
    objective_scores: np.ndarray
    percentile: float
    scoring_mask: np.ndarray
    at_boundary: bool = False

    def __post_init__(self):
        object.__setattr__(self, "at_boundary", bool(self.at_boundary))
        object.__setattr__(self, "alphas", tuple(self.alphas))
        object.__setattr__(
            self, "objective_scores", readonly_array(self.objective_scores)
        )
        object.__setattr__(
            self, "scoring_mask", readonly_array(self.scoring_mask, dtype=bool)
        )


@dataclass(frozen=True)
class CandidateScores:
    regularization: Literal["normalized_ridge", "fractional_ridge"]
    grid: tuple[float, ...]
    cv_r2: np.ndarray
    fold_sse: np.ndarray
    fold_sst: np.ndarray
    fold_hrf_indices: np.ndarray
    trial_masks: tuple[np.ndarray, ...]
    run_labels: tuple[str, ...]
    provenance: ProvenanceRecord

    def __post_init__(self):
        if self.regularization not in ("normalized_ridge", "fractional_ridge"):
            raise ValueError("unknown regularization kind")
        for name in ("grid", "run_labels"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for name in ("cv_r2", "fold_sse", "fold_sst"):
            object.__setattr__(self, name, readonly_array(getattr(self, name)))
        object.__setattr__(
            self,
            "fold_hrf_indices",
            readonly_array(self.fold_hrf_indices, dtype=np.int64),
        )
        object.__setattr__(
            self,
            "trial_masks",
            tuple(readonly_array(m, dtype=bool) for m in self.trial_masks),
        )


@dataclass(frozen=True)
class FractionSelection:
    fractions: tuple[float, ...]
    ridge_fraction: np.ndarray
    selected_r2: np.ndarray
    fraction_indices: np.ndarray
    scoring_mask: np.ndarray
    at_boundary: np.ndarray | None = None

    def __post_init__(self):
        object.__setattr__(self, "fractions", tuple(self.fractions))
        if self.at_boundary is None:
            object.__setattr__(self, "at_boundary", np.zeros_like(self.scoring_mask))
        object.__setattr__(
            self, "at_boundary", readonly_array(self.at_boundary, dtype=bool)
        )
        for name in ("ridge_fraction", "selected_r2"):
            object.__setattr__(self, name, readonly_array(getattr(self, name)))
        object.__setattr__(
            self,
            "fraction_indices",
            readonly_array(self.fraction_indices, dtype=np.int64),
        )
        object.__setattr__(
            self, "scoring_mask", readonly_array(self.scoring_mask, dtype=bool)
        )
