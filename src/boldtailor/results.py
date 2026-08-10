from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

import numpy as np
import pandas as pd

from boldtailor._arrays import immutable_float_array
from boldtailor.provenance import ProvenanceRecord

DesignProvenance = Mapping[str, int | float]


class _NilearnContrast(Protocol):
    def effect_size(self) -> np.ndarray: ...

    def effect_variance(self) -> np.ndarray: ...

    def stat(self) -> np.ndarray: ...

    def z_score(self) -> np.ndarray: ...

    def p_value(self) -> np.ndarray: ...


@dataclass(frozen=True)
class _ContrastResult:
    effect: np.ndarray
    variance: np.ndarray
    stat: np.ndarray
    z_score: np.ndarray
    one_sided_p_value: np.ndarray


@dataclass(frozen=True)
class AnalysisResult:
    _contrasts: Mapping[str, _ContrastResult]
    _design_matrices: tuple[pd.DataFrame, ...]
    _design_provenance: tuple[DesignProvenance, ...]
    _run_r2: tuple[np.ndarray, ...]
    _r2: np.ndarray
    _provenance: ProvenanceRecord

    @property
    def contrast_names(self) -> tuple[str, ...]:
        return tuple(self._contrasts)

    @property
    def design_matrices(self) -> tuple[pd.DataFrame, ...]:
        return tuple(frame.copy(deep=True) for frame in self._design_matrices)

    @property
    def design_provenance(self) -> tuple[DesignProvenance, ...]:
        return self._design_provenance

    @property
    def run_r2(self) -> tuple[np.ndarray, ...]:
        return self._run_r2

    @property
    def r2(self) -> np.ndarray:
        return self._r2

    @property
    def provenance(self) -> ProvenanceRecord:
        return self._provenance

    def effect(self, name: str) -> np.ndarray:
        return self._contrast(name).effect

    def variance(self, name: str) -> np.ndarray:
        return self._contrast(name).variance

    def stat(self, name: str) -> np.ndarray:
        return self._contrast(name).stat

    def z_score(self, name: str) -> np.ndarray:
        return self._contrast(name).z_score

    def one_sided_p_value(self, name: str) -> np.ndarray:
        return self._contrast(name).one_sided_p_value

    def _contrast(self, name: str) -> _ContrastResult:
        try:
            return self._contrasts[name]
        except KeyError as error:
            raise KeyError(f"unknown contrast {name!r}") from error


@dataclass(frozen=True)
class TaskDeltaR2Result:
    _full_r2: np.ndarray
    _nuisance_r2: np.ndarray
    _raw_delta_r2: np.ndarray
    _delta_r2: np.ndarray
    _negative_voxel_count: int
    _raw_min: float
    _nuisance_design_matrices: tuple[pd.DataFrame, ...]
    _provenance: ProvenanceRecord

    @property
    def full_r2(self) -> np.ndarray:
        return self._full_r2

    @property
    def nuisance_r2(self) -> np.ndarray:
        return self._nuisance_r2

    @property
    def raw_delta_r2(self) -> np.ndarray:
        return self._raw_delta_r2

    @property
    def delta_r2(self) -> np.ndarray:
        return self._delta_r2

    @property
    def negative_voxel_count(self) -> int:
        return self._negative_voxel_count

    @property
    def raw_min(self) -> float:
        return self._raw_min

    @property
    def nuisance_design_matrices(self) -> tuple[pd.DataFrame, ...]:
        return tuple(frame.copy(deep=True) for frame in self._nuisance_design_matrices)

    @property
    def provenance(self) -> ProvenanceRecord:
        return self._provenance


def make_result(
    contrasts: Mapping[str, _ContrastResult],
    designs: tuple[pd.DataFrame, ...],
    design_provenance: tuple[DesignProvenance, ...],
    run_r2: tuple[np.ndarray, ...],
    r2: np.ndarray,
    provenance: ProvenanceRecord,
) -> AnalysisResult:
    return AnalysisResult(
        _contrasts=MappingProxyType(dict(contrasts)),
        _design_matrices=tuple(frame.copy(deep=True) for frame in designs),
        _design_provenance=tuple(
            MappingProxyType(dict(values)) for values in design_provenance
        ),
        _run_r2=tuple(immutable_float_array(values) for values in run_r2),
        _r2=immutable_float_array(r2),
        _provenance=provenance,
    )


def make_task_delta_r2_result(
    *,
    full_r2: np.ndarray,
    nuisance_r2: np.ndarray,
    nuisance_designs: tuple[pd.DataFrame, ...],
    provenance: ProvenanceRecord,
) -> TaskDeltaR2Result:
    full, nuisance = _validate_r2_pair(full_r2, nuisance_r2)
    raw = immutable_float_array(full - nuisance)
    clipped = immutable_float_array(np.maximum(raw, 0.0))
    return TaskDeltaR2Result(
        _full_r2=immutable_float_array(full),
        _nuisance_r2=immutable_float_array(nuisance),
        _raw_delta_r2=raw,
        _delta_r2=clipped,
        _negative_voxel_count=int(np.count_nonzero(raw < 0.0)),
        _raw_min=float(raw.min()),
        _nuisance_design_matrices=_copy_nuisance_designs(nuisance_designs),
        _provenance=provenance,
    )


def _validate_r2_pair(
    full_r2: object,
    nuisance_r2: object,
) -> tuple[np.ndarray, np.ndarray]:
    full = _validate_r2_array(full_r2, "full_r2")
    nuisance = _validate_r2_array(nuisance_r2, "nuisance_r2")
    if full.shape != nuisance.shape:
        raise ValueError("full_r2 and nuisance_r2 must have equal shapes")
    return full, nuisance


def _validate_r2_array(values: object, name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite nonempty 1-D array") from error
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite nonempty 1-D array")
    return array


def _copy_nuisance_designs(
    designs: tuple[pd.DataFrame, ...],
) -> tuple[pd.DataFrame, ...]:
    copied = tuple(designs)
    if not copied or any(not isinstance(frame, pd.DataFrame) for frame in copied):
        raise ValueError("nuisance_designs must contain pandas DataFrames")
    return tuple(frame.copy(deep=True) for frame in copied)


def contrast_result(contrast: _NilearnContrast) -> _ContrastResult:
    return _ContrastResult(
        effect=immutable_float_array(contrast.effect_size()),
        variance=immutable_float_array(contrast.effect_variance()),
        stat=immutable_float_array(contrast.stat()),
        z_score=immutable_float_array(contrast.z_score()),
        one_sided_p_value=immutable_float_array(contrast.p_value()),
    )
