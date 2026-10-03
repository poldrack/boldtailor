"""Owned numerical results for trial encoding and ridge selection."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from boldtailor._arrays import own_array_tuples, own_fields, own_tuples, rebind
from boldtailor.provenance import ProvenanceRecord


@dataclass(frozen=True, kw_only=True)
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
        own_fields(self, _ENCODING_ARRAYS)
        own_array_tuples(self, ("predictions",))
        own_array_tuples(self, ("trial_masks",), dtype=bool)
        own_tuples(self, ("predictor_names", "train_runs", "test_runs"))


_ENCODING_ARRAYS = (
    "coefficients",
    "predictor_means",
    "run_sse",
    "run_sst",
    "r2",
    "train_run_predictor_means",
    "train_run_intercepts",
    "scoring_offsets",
)


@dataclass(frozen=True, kw_only=True)
class RidgeSelection:
    ridge_alpha: float
    alphas: tuple[float, ...]
    objective_scores: np.ndarray
    percentile: float
    scoring_mask: np.ndarray
    at_boundary: bool = False

    def __post_init__(self):
        rebind(self, at_boundary=bool(self.at_boundary))
        own_tuples(self, ("alphas",))
        own_fields(self, ("objective_scores",))
        own_fields(self, ("scoring_mask",), dtype=bool)


@dataclass(frozen=True, kw_only=True)
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
        own_tuples(self, ("grid", "run_labels"))
        own_fields(self, ("cv_r2", "fold_sse", "fold_sst"))
        own_fields(self, ("fold_hrf_indices",), dtype=np.int64)
        own_array_tuples(self, ("trial_masks",), dtype=bool)


@dataclass(frozen=True, kw_only=True)
class FractionSelection:
    fractions: tuple[float, ...]
    ridge_fraction: np.ndarray
    selected_r2: np.ndarray
    fraction_indices: np.ndarray
    scoring_mask: np.ndarray
    at_boundary: np.ndarray | None = None

    def __post_init__(self):
        own_tuples(self, ("fractions",))
        if self.at_boundary is None:
            rebind(self, at_boundary=np.zeros_like(self.scoring_mask))
        own_fields(self, ("at_boundary", "scoring_mask"), dtype=bool)
        own_fields(self, ("ridge_fraction", "selected_r2"))
        own_fields(self, ("fraction_indices",), dtype=np.int64)


def paired_scores(scores, grid, *, kind, grid_name):
    """Score matrix and grid from CandidateScores, or from an array plus grid."""
    if isinstance(scores, CandidateScores):
        if grid is not None:
            raise TypeError(
                f"the grid comes from the CandidateScores; do not pass {grid_name}"
            )
        if scores.regularization != kind:
            raise ValueError(
                f"scores are {scores.regularization}; this selector needs {kind}"
            )
        return scores.cv_r2, scores.grid
    if grid is None:
        raise TypeError(f"{grid_name} is required with an array of candidate scores")
    return scores, grid


def scoring_mask(scores, feature_mask):
    """Features finite at every candidate, restricted by an optional mask."""
    mask = np.isfinite(scores).all(axis=0)
    if feature_mask is not None:
        requested = np.asarray(feature_mask)
        if requested.shape != mask.shape or requested.dtype.kind != "b":
            raise ValueError("feature_mask must be a matching boolean feature array")
        mask &= requested
    return mask
