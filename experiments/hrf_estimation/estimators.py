"""Common result contract for experimental HRF estimators."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HrfEstimate:
    feature_ids: np.ndarray
    kernels: tuple
    status: tuple[str, ...]
    training_run_ids: tuple[str, ...]
    diagnostics: dict
    brain_axis: object = None


def validate_features(runs, feature_ids, brain_axis):
    for run in runs:
        if not np.array_equal(run.feature_ids, feature_ids):
            raise ValueError("feature identities/order differ")
        if (run.brain_axis is None) != (brain_axis is None):
            raise ValueError("anatomical identities are missing from one input")
        if brain_axis is not None and not np.array_equal(run.brain_axis, brain_axis):
            raise ValueError("anatomical identities/order differ")
