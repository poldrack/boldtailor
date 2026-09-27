"""Conventional contrast results with spatially varying HRF designs."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, fields
from types import MappingProxyType

import numpy as np
import pandas as pd

from boldtailor._arrays import immutable_float_array
from boldtailor.hrf_results import HrfSelectionResult
from boldtailor.provenance import ProvenanceRecord
from boldtailor.results import _AnalysisAccessors, _ContrastResult, TaskDeltaR2Result


@dataclass(frozen=True)
class HrfAnalysisResult(_AnalysisAccessors):
    """Contrast maps in input feature order; designs keyed by (run, HRF ID)."""

    _contrasts: Mapping[str, _ContrastResult]
    _run_r2: tuple[np.ndarray, ...]
    _r2: np.ndarray
    _provenance: ProvenanceRecord
    _selection: HrfSelectionResult
    _group_designs: Mapping[tuple[int, int], pd.DataFrame]
    _group_design_provenance: Mapping[tuple[int, int], dict]

    def __post_init__(self):
        contrasts = {
            name: _ContrastResult(
                **{
                    field.name: immutable_float_array(getattr(values, field.name))
                    for field in fields(_ContrastResult)
                }
            )
            for name, values in self._contrasts.items()
        }
        object.__setattr__(self, "_contrasts", MappingProxyType(contrasts))
        object.__setattr__(self, "_r2", immutable_float_array(self._r2))
        object.__setattr__(
            self, "_run_r2", tuple(immutable_float_array(a) for a in self._run_r2)
        )
        object.__setattr__(self, "_group_designs", self.group_designs)
        object.__setattr__(
            self, "_group_design_provenance", self.group_design_provenance
        )

    @property
    def hrf_selection(self) -> HrfSelectionResult:
        return self._selection

    @property
    def hrf_indices(self) -> np.ndarray:
        return self._selection.hrf_indices

    @property
    def selection_provenance(self) -> ProvenanceRecord:
        return self._selection.provenance

    @property
    def group_designs(self) -> dict[tuple[int, int], pd.DataFrame]:
        return {
            key: frame.copy(deep=True) for key, frame in self._group_designs.items()
        }

    @property
    def group_design_provenance(self) -> dict:
        return deepcopy(dict(self._group_design_provenance))


def _masked_delta_result(full, nuisance, designs, provenance):
    """Retain undefined features while applying the usual nested-OLS guard."""
    raw = full - nuisance
    defined = np.isfinite(raw)
    if np.any(raw[defined] < -1e-12):
        raise ValueError("nested OLS monotonicity violated")
    return TaskDeltaR2Result(
        immutable_float_array(full),
        immutable_float_array(nuisance),
        immutable_float_array(raw),
        immutable_float_array(np.maximum(raw, 0)),
        int(np.count_nonzero(raw[defined] < 0)),
        float(raw[defined].min()) if defined.any() else float("nan"),
        tuple(design.copy(deep=True) for design in designs),
        provenance,
    )
