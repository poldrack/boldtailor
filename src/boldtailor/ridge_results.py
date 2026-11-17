"""Owned numerical results for trial encoding and ridge selection."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from boldtailor._arrays import (
    copy_array_tuple_fields_in_place,
    copy_array_fields_in_place,
    freeze_sequence_fields_in_place,
    set_fields_in_place,
)
from boldtailor.provenance import ProvenanceRecord


@dataclass(frozen=True, kw_only=True, eq=False)
class RidgeSelection:
    ridge_alpha: float
    alphas: tuple[float, ...]
    objective_scores: np.ndarray
    percentile: float
    scoring_mask: np.ndarray
    at_boundary: bool = False

    def __post_init__(self):
        set_fields_in_place(self, at_boundary=bool(self.at_boundary))
        freeze_sequence_fields_in_place(self, ("alphas",))
        copy_array_fields_in_place(self, ("objective_scores",))
        copy_array_fields_in_place(self, ("scoring_mask",), dtype=bool)


@dataclass(frozen=True, kw_only=True, eq=False)
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
        freeze_sequence_fields_in_place(self, ("grid", "run_labels"))
        copy_array_fields_in_place(self, ("cv_r2", "fold_sse", "fold_sst"))
        copy_array_fields_in_place(self, ("fold_hrf_indices",), dtype=np.int64)
        copy_array_tuple_fields_in_place(self, ("trial_masks",), dtype=bool)


@dataclass(frozen=True, kw_only=True, eq=False)
class FractionSelection:
    fractions: tuple[float, ...]
    ridge_fraction: np.ndarray
    selected_r2: np.ndarray
    fraction_indices: np.ndarray
    scoring_mask: np.ndarray
    at_boundary: np.ndarray | None = None

    def __post_init__(self):
        freeze_sequence_fields_in_place(self, ("fractions",))
        if self.at_boundary is None:
            set_fields_in_place(self, at_boundary=np.zeros_like(self.scoring_mask))
        copy_array_fields_in_place(self, ("at_boundary", "scoring_mask"), dtype=bool)
        copy_array_fields_in_place(self, ("ridge_fraction", "selected_r2"))
        copy_array_fields_in_place(self, ("fraction_indices",), dtype=np.int64)


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
