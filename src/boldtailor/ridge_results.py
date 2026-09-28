"""Owned numerical results for trial encoding and ridge selection."""

from dataclasses import dataclass

import numpy as np

from boldtailor._arrays import immutable_float_array
from boldtailor.hrf_results import immutable_indices
from boldtailor.provenance import ProvenanceRecord


def immutable_bool_array(values):
    array = np.asarray(values, dtype=bool)
    return np.frombuffer(array.tobytes(), dtype=bool).reshape(array.shape)


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
            object.__setattr__(self, name, immutable_float_array(getattr(self, name)))
        object.__setattr__(
            self,
            "predictions",
            tuple(immutable_float_array(a) for a in self.predictions),
        )
        object.__setattr__(
            self,
            "trial_masks",
            tuple(immutable_bool_array(a) for a in self.trial_masks),
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

    def __post_init__(self):
        object.__setattr__(self, "alphas", tuple(self.alphas))
        object.__setattr__(
            self, "objective_scores", immutable_float_array(self.objective_scores)
        )
        object.__setattr__(
            self, "scoring_mask", immutable_bool_array(self.scoring_mask)
        )


@dataclass(frozen=True)
class RidgeCandidateScores:
    alphas: tuple[float, ...]
    cv_r2: np.ndarray
    fold_sse: np.ndarray
    fold_sst: np.ndarray
    fold_hrf_indices: np.ndarray
    trial_masks: tuple[np.ndarray, ...]
    run_labels: tuple[str, ...]
    provenance: ProvenanceRecord

    def __post_init__(self):
        for name in ("alphas", "run_labels"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for name in ("cv_r2", "fold_sse", "fold_sst"):
            object.__setattr__(self, name, immutable_float_array(getattr(self, name)))
        object.__setattr__(
            self, "fold_hrf_indices", immutable_indices(self.fold_hrf_indices)
        )
        object.__setattr__(
            self,
            "trial_masks",
            tuple(immutable_bool_array(m) for m in self.trial_masks),
        )


@dataclass(frozen=True)
class FractionCandidateScores:
    fractions: tuple[float, ...]
    cv_r2: np.ndarray
    fold_sse: np.ndarray
    fold_sst: np.ndarray
    fold_hrf_indices: np.ndarray
    trial_masks: tuple[np.ndarray, ...]
    run_labels: tuple[str, ...]
    provenance: ProvenanceRecord

    def __post_init__(self):
        object.__setattr__(self, "fractions", tuple(self.fractions))
        object.__setattr__(self, "run_labels", tuple(self.run_labels))
        for name in ("cv_r2", "fold_sse", "fold_sst"):
            object.__setattr__(self, name, immutable_float_array(getattr(self, name)))
        object.__setattr__(
            self, "fold_hrf_indices", immutable_indices(self.fold_hrf_indices)
        )
        object.__setattr__(
            self,
            "trial_masks",
            tuple(immutable_bool_array(m) for m in self.trial_masks),
        )


@dataclass(frozen=True)
class FractionSelection:
    fractions: tuple[float, ...]
    ridge_fraction: np.ndarray
    selected_r2: np.ndarray
    fraction_indices: np.ndarray
    scoring_mask: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "fractions", tuple(self.fractions))
        for name in ("ridge_fraction", "selected_r2"):
            object.__setattr__(self, name, immutable_float_array(getattr(self, name)))
        object.__setattr__(
            self, "fraction_indices", immutable_indices(self.fraction_indices)
        )
        object.__setattr__(
            self, "scoring_mask", immutable_bool_array(self.scoring_mask)
        )
