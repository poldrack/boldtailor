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
