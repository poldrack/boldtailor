"""Owned numerical results for trial encoding and ridge selection."""

from dataclasses import dataclass

import numpy as np

from boldtailor._arrays import immutable_float_array


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

    def __post_init__(self):
        for name in ("coefficients", "predictor_means", "run_sse", "run_sst", "r2"):
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
