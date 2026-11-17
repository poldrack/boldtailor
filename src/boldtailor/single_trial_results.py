"""Owned, read-only single-trial numerical results."""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from boldtailor._arrays import (
    copy_array_fields_in_place,
    copy_array_tuple_fields_in_place,
    owned_table,
    readonly_array,
    set_fields_in_place,
)
from boldtailor.provenance import ProvenanceRecord


@dataclass(frozen=True, kw_only=True, eq=False)
class SharedTrialDesign:
    _matrices: tuple[pd.DataFrame, ...]

    def __post_init__(self):
        set_fields_in_place(
            self, _matrices=tuple(d.copy(deep=True) for d in self._matrices)
        )

    @property
    def matrices(self):
        return tuple(d.copy(deep=True) for d in self._matrices)


@dataclass(frozen=True, kw_only=True, eq=False)
class SelectedTrialDesign:
    """Selected HRF IDs; each fitted (run, HRF) design is rebuilt on request."""

    hrf_indices: np.ndarray
    design_fingerprint: str
    selection_provenance: ProvenanceRecord
    _rebuild: Callable[[int, int], np.ndarray] = field(compare=False, repr=False)

    def __post_init__(self):
        copy_array_fields_in_place(self, ("hrf_indices",), dtype=np.int64)

    def rebuild_matrix(self, run: int, hrf_id: int) -> np.ndarray:
        """Trial columns then nuisance columns; KeyError for unfitted pairs."""
        return readonly_array(self._rebuild(run, hrf_id))

    def matrix(self, run: int, hrf_id: int) -> np.ndarray:
        """Compatibility alias for rebuild_matrix; performs reconstruction."""
        return self.rebuild_matrix(run, hrf_id)


@dataclass(frozen=True, kw_only=True, eq=False)
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

    def materialize_designs(self) -> dict[tuple[int, int], np.ndarray]:
        """Return owned read-only design arrays for all fitted (run, HRF) pairs.

        Shared-HRF designs use HRF ID 0. Selected-HRF designs are reconstructed
        now; this can allocate substantial memory for many fitted HRFs.
        """
        if isinstance(self.design, SharedTrialDesign):
            return {
                (run, 0): readonly_array(frame.to_numpy())
                for run, frame in enumerate(self.design.matrices)
            }
        ids = np.unique(self.design.hrf_indices[self.design.hrf_indices >= 0])
        return {
            (run, int(cid)): self.design.rebuild_matrix(run, int(cid))
            for run in range(len(self.run_betas))
            for cid in ids
        }

    def __post_init__(self):
        if self.ridge_fraction is not None:
            copy_array_fields_in_place(self, ("ridge_fraction",))
            copy_array_tuple_fields_in_place(self, ("run_ridge_alphas",))
        copy_array_tuple_fields_in_place(
            self, ("run_betas", "run_full_r2", "run_nuisance_r2")
        )
        copy_array_fields_in_place(self, ("full_r2", "nuisance_r2", "delta_r2"))
        set_fields_in_place(
            self,
            _trial_table=owned_table(self._trial_table),
            _diagnostics=deepcopy(self._diagnostics),
        )

    @property
    def trial_table(self):
        return owned_table(self._trial_table)

    @property
    def diagnostics(self):
        return deepcopy(self._diagnostics)
