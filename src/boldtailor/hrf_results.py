"""Owned results for HRF selection, independent evaluation and grouped fits."""

from dataclasses import dataclass
from copy import deepcopy
import numpy as np
import pandas as pd

from boldtailor._arrays import readonly_array
from boldtailor.data import _owned_table
from boldtailor.hrf_library import HrfLibrary
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

    def __post_init__(self):
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


@dataclass(frozen=True)
class HrfSingleTrialResult:
    run_betas: tuple[np.ndarray, ...]
    _trial_table: pd.DataFrame
    hrf_indices: np.ndarray
    _group_designs: dict
    run_full_r2: tuple[np.ndarray, ...]
    run_nuisance_r2: tuple[np.ndarray, ...]
    full_r2: np.ndarray
    nuisance_r2: np.ndarray
    delta_r2: np.ndarray
    _diagnostics: tuple[dict, ...]
    ridge_alpha: float | None
    selection_provenance: ProvenanceRecord
    provenance: ProvenanceRecord
    ridge_fraction: np.ndarray | None = None
    run_ridge_alphas: tuple[np.ndarray, ...] | None = None

    def __post_init__(self):
        if self.ridge_fraction is not None:
            object.__setattr__(
                self, "ridge_fraction", readonly_array(self.ridge_fraction)
            )
            object.__setattr__(
                self,
                "run_ridge_alphas",
                tuple(readonly_array(a) for a in self.run_ridge_alphas),
            )
        for name in ("run_betas", "run_full_r2", "run_nuisance_r2"):
            object.__setattr__(
                self, name, tuple(readonly_array(a) for a in getattr(self, name))
            )
        for name in ("full_r2", "nuisance_r2", "delta_r2"):
            object.__setattr__(self, name, readonly_array(getattr(self, name)))
        object.__setattr__(
            self, "hrf_indices", readonly_array(self.hrf_indices, dtype=np.int64)
        )
        object.__setattr__(self, "_trial_table", _owned_table(self._trial_table))
        object.__setattr__(
            self,
            "_group_designs",
            {k: readonly_array(v) for k, v in self._group_designs.items()},
        )
        object.__setattr__(self, "_diagnostics", deepcopy(self._diagnostics))

    @property
    def trial_table(self):
        return _owned_table(self._trial_table)

    @property
    def group_designs(self):
        return dict(self._group_designs)

    @property
    def diagnostics(self):
        return deepcopy(self._diagnostics)
