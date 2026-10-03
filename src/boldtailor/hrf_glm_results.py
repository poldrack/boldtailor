"""Conventional contrast results with spatially varying HRF designs."""

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from boldtailor._arrays import own_array_tuples, own_fields, rebind
from boldtailor.hrf_results import HrfSelectionResult
from boldtailor.provenance import ProvenanceRecord
from boldtailor.results import _AnalysisAccessors, _ContrastResult, owned_contrasts


@dataclass(frozen=True, kw_only=True)
class HrfAnalysisResult(_AnalysisAccessors):
    """Contrast maps in input feature order; designs rebuilt per (run, HRF ID)."""

    _contrasts: Mapping[str, _ContrastResult]
    _run_r2: tuple[np.ndarray, ...]
    _r2: np.ndarray
    _provenance: ProvenanceRecord
    _selection: HrfSelectionResult
    _group_design_provenance: Mapping[tuple[int, int], dict]
    _group_builder: Callable[[int, int], pd.DataFrame] = field(
        compare=False, repr=False
    )

    def __post_init__(self):
        rebind(
            self,
            _contrasts=owned_contrasts(self._contrasts),
            _group_design_provenance=self.group_design_provenance,
        )
        own_fields(self, ("_r2",))
        own_array_tuples(self, ("_run_r2",))

    @property
    def hrf_selection(self) -> HrfSelectionResult:
        return self._selection

    @property
    def hrf_indices(self) -> np.ndarray:
        return self._selection.hrf_indices

    @property
    def selection_provenance(self) -> ProvenanceRecord:
        return self._selection.provenance

    def group_design(self, run: int, hrf_id: int) -> pd.DataFrame:
        """Recompiled design for a fitted (run, HRF ID); KeyError otherwise."""
        return self._group_builder(run, hrf_id).copy(deep=True)

    @property
    def group_design_provenance(self) -> dict:
        return deepcopy(dict(self._group_design_provenance))
