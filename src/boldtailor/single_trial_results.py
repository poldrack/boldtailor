"""Owned, read-only single-trial numerical results."""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from boldtailor._arrays import readonly_array
from boldtailor.data import _owned_table
from boldtailor.provenance import ProvenanceRecord


@dataclass(frozen=True)
class SingleTrialResult:
    run_betas: tuple[np.ndarray, ...]
    _trial_table: pd.DataFrame
    _design_matrices: tuple[pd.DataFrame, ...]
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
        object.__setattr__(
            self,
            "_design_matrices",
            tuple(d.copy(deep=True) for d in self._design_matrices),
        )
        object.__setattr__(self, "_diagnostics", deepcopy(self._diagnostics))

    @property
    def trial_table(self):
        return _owned_table(self._trial_table)

    @property
    def design_matrices(self):
        return tuple(d.copy(deep=True) for d in self._design_matrices)

    @property
    def diagnostics(self):
        return deepcopy(self._diagnostics)
