"""Owned results for HRF selection and independent evaluation."""

from dataclasses import dataclass
import numpy as np
import pandas as pd

from boldtailor._arrays import own_fields, own_tuples, rebind
from boldtailor.data import _owned_table
from boldtailor.hrf_library import HrfLibrary
from boldtailor.model import TaskModel
from boldtailor.provenance import ProvenanceRecord


@dataclass(frozen=True, kw_only=True)
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
    at_parameter_bound: np.ndarray | None = None

    def __post_init__(self):
        if not isinstance(self.task_model, TaskModel):
            raise ValueError("task_model must be a TaskModel")
        own_fields(self, ("hrf_indices",), dtype=np.int64)
        own_fields(self, ("cv_r2", "canonical_cv_r2", "delta_cv_r2"))
        own_tuples(self, ("run_labels",))
        rebind(self, _eligibility=_owned_table(self._eligibility))
        if self.at_parameter_bound is None:
            rebind(self, at_parameter_bound=np.zeros(self.hrf_indices.shape, bool))
        own_fields(self, ("at_parameter_bound",), dtype=bool)

    @property
    def eligibility(self):
        return _owned_table(self._eligibility)


@dataclass(frozen=True, kw_only=True)
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
        own_fields(
            self,
            ("training_amplitudes", "test_r2", "canonical_test_r2", "delta_test_r2"),
        )
        own_tuples(self, ("train_runs", "test_runs", "amplitude_names"))
        amplitudes = self.training_amplitudes
        if amplitudes.ndim != 2 or amplitudes.shape[0] != len(self.amplitude_names):
            raise ValueError("training_amplitudes needs one row per amplitude name")
