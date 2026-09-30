"""Owned results for HRF selection and independent evaluation."""

from dataclasses import dataclass
import numpy as np
import pandas as pd

from boldtailor._arrays import readonly_array
from boldtailor.data import _owned_table
from boldtailor.hrf_library import HrfLibrary
from boldtailor.model import TaskModel
from boldtailor.provenance import ProvenanceRecord


@dataclass(frozen=True)
class HrfSelectionResult:
    hrf_indices: np.ndarray
    cv_r2: np.ndarray
    canonical_cv_r2: np.ndarray
    delta_cv_r2: np.ndarray
    library: HrfLibrary
    _eligibility: pd.DataFrame
    run_labels: tuple[str, ...]
    feature_signature: str | None
    provenance: ProvenanceRecord
    task_model: TaskModel = TaskModel()

    def __post_init__(self):
        if not isinstance(self.task_model, TaskModel):
            raise ValueError("task_model must be a TaskModel")
        object.__setattr__(
            self, "hrf_indices", readonly_array(self.hrf_indices, dtype=np.int64)
        )
        for name in ("cv_r2", "canonical_cv_r2", "delta_cv_r2"):
            object.__setattr__(self, name, readonly_array(getattr(self, name)))
        object.__setattr__(self, "_eligibility", _owned_table(self._eligibility))
        object.__setattr__(self, "run_labels", tuple(self.run_labels))

    @property
    def eligibility(self):
        return _owned_table(self._eligibility)


@dataclass(frozen=True)
class HrfEvaluationResult:
    training_selection: HrfSelectionResult
    training_amplitudes: np.ndarray
    test_r2: np.ndarray
    canonical_test_r2: np.ndarray
    delta_test_r2: np.ndarray
    train_runs: tuple[int, ...]
    test_runs: tuple[int, ...]
    provenance: ProvenanceRecord
    amplitude_names: tuple[str, ...] = ("task",)

    def __post_init__(self):
        for name in (
            "training_amplitudes",
            "test_r2",
            "canonical_test_r2",
            "delta_test_r2",
        ):
            object.__setattr__(self, name, readonly_array(getattr(self, name)))
        for name in ("train_runs", "test_runs"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "amplitude_names", tuple(self.amplitude_names))
        amplitudes = self.training_amplitudes
        if amplitudes.ndim != 2 or amplitudes.shape[0] != len(self.amplitude_names):
            raise ValueError("training_amplitudes needs one row per amplitude name")
