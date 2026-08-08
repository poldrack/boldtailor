from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

import numpy as np
import pandas as pd

from boldtailor._arrays import immutable_float_array

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


def make_result(
    contrasts: Mapping[str, _ContrastResult],
    designs: tuple[pd.DataFrame, ...],
    design_provenance: tuple[DesignProvenance, ...],
    run_r2: tuple[np.ndarray, ...],
    r2: np.ndarray,
) -> AnalysisResult:
    return AnalysisResult(
        _contrasts=MappingProxyType(dict(contrasts)),
        _design_matrices=tuple(frame.copy(deep=True) for frame in designs),
        _design_provenance=tuple(
            MappingProxyType(dict(values)) for values in design_provenance
        ),
        _run_r2=tuple(immutable_float_array(values) for values in run_r2),
        _r2=immutable_float_array(r2),
    )


def contrast_result(contrast: _NilearnContrast) -> _ContrastResult:
    return _ContrastResult(
        effect=immutable_float_array(contrast.effect_size()),
        variance=immutable_float_array(contrast.effect_variance()),
        stat=immutable_float_array(contrast.stat()),
        z_score=immutable_float_array(contrast.z_score()),
        one_sided_p_value=immutable_float_array(contrast.p_value()),
    )
