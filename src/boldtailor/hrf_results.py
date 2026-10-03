"""Owned results for HRF selection and independent evaluation."""

from dataclasses import dataclass
import numpy as np
import pandas as pd

from boldtailor._arrays import own_fields, own_tuples, readonly_array, rebind
from boldtailor.data import _owned_table
from boldtailor.hrf_library import PARAMETER_NAMES, HrfLibrary
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

    def __post_init__(self):
        if not isinstance(self.task_model, TaskModel):
            raise ValueError("task_model must be a TaskModel")
        own_fields(self, ("hrf_indices",), dtype=np.int64)
        own_fields(self, ("cv_r2", "canonical_cv_r2", "delta_cv_r2"))
        own_tuples(self, ("run_labels",))
        rebind(self, _eligibility=_owned_table(self._eligibility))

    @property
    def eligibility(self):
        return _owned_table(self._eligibility)

    @property
    def parameter_bound_flags(self) -> np.ndarray:
        """``(n_features, 6, 2)`` edge flags of each selected custom kernel.

        Parameters follow ``PARAMETER_NAMES[:6]``; the last axis is
        ``[low, high]``. True within 2 % of the library box width of an edge;
        canonical (0) and ineligible (-1) features are all False.
        """
        flags = self.library.candidate_bound_flags()[np.maximum(self.hrf_indices, 0)]
        flags &= (self.hrf_indices > 0)[:, None, None]
        return readonly_array(flags, dtype=bool)

    @property
    def at_parameter_bound(self) -> np.ndarray:
        """Features flagged at an edge of any informative library parameter."""
        informative = [
            PARAMETER_NAMES.index(name) for name in self.library.informative_parameters
        ]
        flags = self.parameter_bound_flags[:, informative]
        return readonly_array(flags.any(axis=(1, 2)), dtype=bool)

    def parameter_bound_table(self) -> pd.DataFrame:
        """Fraction of custom picks (ID > 0) flagged at each parameter edge."""
        custom = self.parameter_bound_flags[self.hrf_indices > 0]
        fractions = custom.mean(axis=0) if len(custom) else np.full((6, 2), np.nan)
        return pd.DataFrame(
            [
                dict(parameter=name, edge=edge, fraction_flagged=float(fractions[p, e]))
                for p, name in enumerate(PARAMETER_NAMES[:6])
                for e, edge in enumerate(("low", "high"))
            ]
        )


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
