"""Owned, read-only single-trial numerical results."""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from boldtailor._arrays import readonly_array
from boldtailor.data import _owned_table
from boldtailor.provenance import ProvenanceRecord


@dataclass(frozen=True)
class SharedTrialDesign:
    _matrices: tuple[pd.DataFrame, ...]

    def __post_init__(self):
        object.__setattr__(
            self, "_matrices", tuple(d.copy(deep=True) for d in self._matrices)
        )

    @property
    def matrices(self):
        return tuple(d.copy(deep=True) for d in self._matrices)


@dataclass(frozen=True)
class SelectedTrialDesign:
    """Selected HRF IDs; each fitted (run, HRF) design is rebuilt on request."""

    hrf_indices: np.ndarray
    design_fingerprint: str
    selection_provenance: ProvenanceRecord
    _rebuild: Callable[[int, int], np.ndarray] = field(compare=False, repr=False)

    def __post_init__(self):
        object.__setattr__(
            self, "hrf_indices", readonly_array(self.hrf_indices, dtype=np.int64)
        )

    def matrix(self, run: int, hrf_id: int) -> np.ndarray:
        """Trial columns then nuisance columns; KeyError for unfitted pairs."""
        return readonly_array(self._rebuild(run, hrf_id))


@dataclass(frozen=True)
class SingleTrialResult:
    run_betas: tuple[np.ndarray, ...]
    _trial_table: pd.DataFrame
    design: SharedTrialDesign | SelectedTrialDesign
    run_full_r2: tuple[np.ndarray, ...]
    run_nuisance_r2: tuple[np.ndarray, ...]
    full_r2: np.ndarray
    nuisance_r2: np.ndarray
    delta_r2: np.ndarray
    _diagnostics: tuple[dict, ...]
    ridge_alpha: float | None
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
        object.__setattr__(self, "_trial_table", _owned_table(self._trial_table))
        object.__setattr__(self, "_diagnostics", deepcopy(self._diagnostics))

    @property
    def trial_table(self):
        return _owned_table(self._trial_table)

    @property
    def diagnostics(self):
        return deepcopy(self._diagnostics)
