"""Owned, immutable single-trial numerical results."""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from boldtailor._arrays import immutable_float_array
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
    ridge_alpha: float
    provenance: ProvenanceRecord

    def __post_init__(self):
        for name in ("run_betas", "run_full_r2", "run_nuisance_r2"):
            object.__setattr__(
                self, name, tuple(immutable_float_array(a) for a in getattr(self, name))
            )
        for name in ("full_r2", "nuisance_r2", "delta_r2"):
            object.__setattr__(self, name, immutable_float_array(getattr(self, name)))
        object.__setattr__(self, "_trial_table", self._trial_table.copy(deep=True))
        object.__setattr__(
            self,
            "_design_matrices",
            tuple(d.copy(deep=True) for d in self._design_matrices),
        )
        object.__setattr__(self, "_diagnostics", deepcopy(self._diagnostics))

    @property
    def trial_table(self):
        return self._trial_table.copy(deep=True)

    @property
    def design_matrices(self):
        return tuple(d.copy(deep=True) for d in self._design_matrices)

    @property
    def diagnostics(self):
        return deepcopy(self._diagnostics)
