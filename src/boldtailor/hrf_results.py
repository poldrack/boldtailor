"""Owned results for HRF selection and independent evaluation."""

from dataclasses import dataclass
import numpy as np
import pandas as pd

from boldtailor._arrays import (
    copy_array_fields_in_place,
    freeze_sequence_fields_in_place,
    owned_table,
    readonly_array,
    set_fields_in_place,
)
from boldtailor.hrf_library import PARAMETER_NAMES, HrfLibrary
from boldtailor.model import TaskModel
from boldtailor.provenance import ProvenanceRecord
from boldtailor.reml import RemlFit, RemlSelection


@dataclass(frozen=True, kw_only=True, eq=False)
class RemlHrfDiagnostics:
    """Likelihood comparisons and search qualifications, never prediction scores."""

    selection: RemlSelection
    candidate_ids: np.ndarray
    canonical_fit: RemlFit | None
    delta_criterion: np.ndarray
    reference_hrf_id: int

    def __post_init__(self):
        copy_array_fields_in_place(self, ("candidate_ids",), dtype=np.int64)
        copy_array_fields_in_place(self, ("delta_criterion",))
        if self.candidate_ids.shape != (self.selection.n_candidates,):
            raise ValueError("candidate_ids must map every REML candidate")
        if self.delta_criterion.shape != self.selection.winner.shape:
            raise ValueError("delta_criterion must have one value per feature")


@dataclass(frozen=True, kw_only=True, eq=False)
class HrfSelectionResult:
    hrf_indices: np.ndarray
    cv_r2: np.ndarray | None
    canonical_cv_r2: np.ndarray | None
    delta_cv_r2: np.ndarray | None
    library: HrfLibrary
    _eligibility: pd.DataFrame
    run_labels: tuple[str, ...]
    feature_signature: str | None
    provenance: ProvenanceRecord
    task_model: TaskModel = TaskModel()
    method: str = "cv"
    reml: RemlHrfDiagnostics | None = None

    def __post_init__(self):
        if not isinstance(self.task_model, TaskModel):
            raise ValueError("task_model must be a TaskModel")
        copy_array_fields_in_place(self, ("hrf_indices",), dtype=np.int64)
        names = ("cv_r2", "canonical_cv_r2", "delta_cv_r2")
        if self.method not in ("cv", "reml"):
            raise ValueError("unknown HRF selection method")
        if self.method == "cv":
            if self.reml is not None or any(getattr(self, n) is None for n in names):
                raise ValueError("CV selection requires only CV diagnostics")
            copy_array_fields_in_place(self, names)
        elif not isinstance(self.reml, RemlHrfDiagnostics) or any(
            getattr(self, n) is not None for n in names
        ):
            raise ValueError(
                "REML selection requires REML diagnostics and no CV scores"
            )
        freeze_sequence_fields_in_place(self, ("run_labels",))
        set_fields_in_place(self, _eligibility=owned_table(self._eligibility))

    @property
    def eligibility(self):
        return owned_table(self._eligibility)

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


@dataclass(frozen=True, kw_only=True, eq=False)
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
        copy_array_fields_in_place(
            self,
            ("training_amplitudes", "test_r2", "canonical_test_r2", "delta_test_r2"),
        )
        freeze_sequence_fields_in_place(
            self, ("train_runs", "test_runs", "amplitude_names")
        )
        amplitudes = self.training_amplitudes
        if amplitudes.ndim != 2 or amplitudes.shape[0] != len(self.amplitude_names):
            raise ValueError("training_amplitudes needs one row per amplitude name")


def validate_selection(data, selection, signature):
    """Validate an HrfSelectionResult against data, library, and signature."""
    from hashlib import sha256

    if not isinstance(selection, HrfSelectionResult):
        raise ValueError("selection must be an HrfSelectionResult")
    if signature != selection.feature_signature:
        raise ValueError("feature_signature must match selection")
    ids = selection.hrf_indices
    if ids.shape != (data.n_features,):
        raise ValueError("selection feature count must match data")
    activity = selection.provenance.activity_records()[-1]
    assignment = sha256(ids.astype("<i8").tobytes()).hexdigest()
    if activity.get("library_fingerprint") != selection.library.fingerprint:
        raise ValueError("selection library fingerprint does not match identity")
    if np.any(ids < -1) or np.any(ids >= len(selection.library.candidates)):
        raise ValueError("selection contains invalid HRF IDs")
    if activity.get("hrf_assignment_fingerprint") != assignment:
        raise ValueError("selection HRF assignment identity does not match provenance")
    return assignment
