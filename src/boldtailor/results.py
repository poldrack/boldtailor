from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Protocol

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

from boldtailor._arrays import readonly_array
from boldtailor._fit_diagnostics import validate_nested_ols_delta
from boldtailor.provenance import ProvenanceRecord

DesignProvenance = Mapping[str, int | float | str]


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


class _AnalysisAccessors:
    _contrasts: Mapping[str, _ContrastResult]
    _run_r2: tuple[np.ndarray, ...]
    _r2: np.ndarray
    _provenance: ProvenanceRecord

    @property
    def contrast_names(self) -> tuple[str, ...]:
        return tuple(self._contrasts)

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
class AnalysisResult(_AnalysisAccessors):
    _contrasts: Mapping[str, _ContrastResult]
    _design_matrices: tuple[pd.DataFrame, ...]
    _design_provenance: tuple[DesignProvenance, ...]
    _run_r2: tuple[np.ndarray, ...]
    _r2: np.ndarray
    _provenance: ProvenanceRecord

    @property
    def design_matrices(self) -> tuple[pd.DataFrame, ...]:
        return tuple(frame.copy(deep=True) for frame in self._design_matrices)

    @property
    def design_provenance(self) -> tuple[DesignProvenance, ...]:
        return self._design_provenance


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
        _run_r2=tuple(readonly_array(values) for values in run_r2),
        _r2=readonly_array(r2),
        _provenance=provenance,
    )


def make_task_delta_r2_result(
    *,
    full_r2: np.ndarray,
    nuisance_r2: np.ndarray,
    nuisance_designs: tuple[pd.DataFrame, ...],
    provenance: ProvenanceRecord,
    allow_undefined: bool = False,
) -> TaskDeltaR2Result:
    full, nuisance = _validate_r2_pair(full_r2, nuisance_r2, allow_undefined)
    raw, clipped = _delta_r2_arrays(full, nuisance, allow_undefined)
    defined = raw[np.isfinite(raw)]
    return TaskDeltaR2Result(
        _full_r2=readonly_array(full),
        _nuisance_r2=readonly_array(nuisance),
        _raw_delta_r2=raw,
        _delta_r2=clipped,
        _negative_voxel_count=int(np.count_nonzero(defined < 0.0)),
        _raw_min=float(defined.min()) if defined.size else float("nan"),
        _nuisance_design_matrices=_copy_nuisance_designs(nuisance_designs),
        _provenance=provenance,
    )


def _validate_r2_pair(
    full_r2: object,
    nuisance_r2: object,
    allow_undefined: bool,
) -> tuple[np.ndarray, np.ndarray]:
    full = _validate_r2_array(full_r2, "full_r2", allow_undefined)
    nuisance = _validate_r2_array(nuisance_r2, "nuisance_r2", allow_undefined)
    if full.shape != nuisance.shape:
        raise ValueError("full_r2 and nuisance_r2 must have equal shapes")
    return full, nuisance


def _validate_r2_array(
    values: object, name: str, allow_undefined: bool = False
) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite nonempty 1-D array") from error
    unusable = np.isinf(array).any() or (not allow_undefined and np.isnan(array).any())
    if array.ndim != 1 or array.size == 0 or unusable:
        raise ValueError(f"{name} must be a finite nonempty 1-D array")
    return array


def _delta_r2_arrays(
    full: np.ndarray,
    nuisance: np.ndarray,
    allow_undefined: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    with np.errstate(over="ignore", invalid="ignore"):
        raw_values = full - nuisance
        clipped_values = np.maximum(raw_values, 0.0)
    if np.isinf(raw_values).any() or (
        not allow_undefined and np.isnan(raw_values).any()
    ):
        raise ValueError("derived delta r-squared values must be finite")
    validate_nested_ols_delta(raw_values)
    return readonly_array(raw_values), readonly_array(clipped_values)


def _copy_nuisance_designs(
    designs: tuple[pd.DataFrame, ...],
) -> tuple[pd.DataFrame, ...]:
    copied = tuple(designs)
    if not copied or any(not isinstance(frame, pd.DataFrame) for frame in copied):
        raise ValueError("nuisance_designs must contain pandas DataFrames")
    if any(not _is_finite_numeric_design(frame) for frame in copied):
        raise ValueError("nuisance designs must be finite numeric matrices")
    return tuple(frame.copy(deep=True) for frame in copied)


def _is_finite_numeric_design(frame: pd.DataFrame) -> bool:
    if 0 in frame.shape:
        return False
    if any(
        not is_numeric_dtype(dtype) or is_bool_dtype(dtype) or is_complex_dtype(dtype)
        for dtype in frame.dtypes
    ):
        return False
    try:
        return bool(np.isfinite(frame.to_numpy(dtype=float)).all())
    except (TypeError, ValueError):
        return False


def contrast_result(contrast: _NilearnContrast) -> _ContrastResult:
    return _ContrastResult(
        effect=readonly_array(contrast.effect_size()),
        variance=readonly_array(contrast.effect_variance()),
        stat=readonly_array(contrast.stat()),
        z_score=readonly_array(contrast.z_score()),
        one_sided_p_value=readonly_array(contrast.p_value()),
    )


def mask_contrast(result: _ContrastResult, undefined: np.ndarray) -> _ContrastResult:
    """Return a copy with every statistic set to NaN where ``undefined`` is True."""

    def masked(values: np.ndarray) -> np.ndarray:
        out = np.array(values, dtype=float, copy=True)
        out[undefined] = np.nan
        return readonly_array(out)

    return _ContrastResult(
        **{f.name: masked(getattr(result, f.name)) for f in fields(_ContrastResult)}
    )
