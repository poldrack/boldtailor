from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal
from uuid import uuid4

import numpy as np
import pandas as pd

from boldtailor.data import (
    _as_signal_runs,
    _prepare_signal,
    _prepare_sources,
    _prepare_timing,
    _validate_feature_counts,
)
from boldtailor.provenance import (
    ProvenanceRecord,
    RunSources,
    _freeze_mapping,
    _thaw,
)

ColumnRole = Literal["task", "nuisance", "intercept", "other"]
_COLUMN_ROLES = frozenset({"task", "nuisance", "intercept", "other"})


@dataclass(frozen=True)
class PreparedDesignAnalysis:
    _signals: tuple[np.ndarray, ...]
    _design_matrices: tuple[pd.DataFrame, ...]
    _frame_times: tuple[np.ndarray, ...]
    _timing_source: str
    _column_roles: tuple[Mapping[str, ColumnRole], ...]
    _run_metadata: tuple[Mapping[str, object], ...]
    _provenance: ProvenanceRecord

    @classmethod
    def from_arrays(
        cls,
        signals: np.ndarray | Sequence[np.ndarray],
        design_matrices: pd.DataFrame | Sequence[pd.DataFrame],
        *,
        tr: float | None = None,
        frame_times: np.ndarray | Sequence[np.ndarray] | None = None,
        column_roles: Mapping[str, str] | Sequence[Mapping[str, str]],
        sources: Sequence[RunSources] | None = None,
        run_metadata: (
            Mapping[str, object] | Sequence[Mapping[str, object]] | None
        ) = None,
        provenance_metadata: Mapping[str, object] | None = None,
    ) -> PreparedDesignAnalysis:
        return _prepare_analysis(
            cls=cls,
            signals=signals,
            design_matrices=design_matrices,
            tr=tr,
            frame_times=frame_times,
            column_roles=column_roles,
            sources=sources,
            run_metadata=run_metadata,
            provenance_metadata=provenance_metadata,
        )

    @property
    def n_runs(self) -> int:
        return len(self._signals)

    @property
    def n_features(self) -> int:
        return self._signals[0].shape[1]

    @property
    def signals(self) -> tuple[np.ndarray, ...]:
        return self._signals

    @property
    def design_matrices(self) -> tuple[pd.DataFrame, ...]:
        return tuple(matrix.copy(deep=True) for matrix in self._design_matrices)

    @property
    def frame_times(self) -> tuple[np.ndarray, ...]:
        return self._frame_times

    @property
    def timing_source(self) -> str:
        return self._timing_source

    @property
    def column_roles(self) -> tuple[Mapping[str, ColumnRole], ...]:
        return tuple(dict(roles) for roles in self._column_roles)

    @property
    def run_metadata(self) -> tuple[Mapping[str, object], ...]:
        return tuple(_thaw(metadata) for metadata in self._run_metadata)

    @property
    def provenance(self) -> ProvenanceRecord:
        return self._provenance


def _prepare_analysis(
    *,
    cls: type[PreparedDesignAnalysis],
    signals: np.ndarray | Sequence[np.ndarray],
    design_matrices: pd.DataFrame | Sequence[pd.DataFrame],
    tr: float | None,
    frame_times: np.ndarray | Sequence[np.ndarray] | None,
    column_roles: Mapping[str, str] | Sequence[Mapping[str, str]],
    sources: Sequence[RunSources] | None,
    run_metadata: Mapping[str, object] | Sequence[Mapping[str, object]] | None,
    provenance_metadata: Mapping[str, object] | None,
) -> PreparedDesignAnalysis:
    del provenance_metadata
    signal_runs = _as_signal_runs(signals)
    prepared_signals = tuple(
        _prepare_signal(values, run) for run, values in enumerate(signal_runs)
    )
    _validate_feature_counts(prepared_signals)
    designs = _prepare_designs(design_matrices, prepared_signals)
    roles = _prepare_roles(column_roles, designs)
    metadata = _prepare_metadata(run_metadata, len(prepared_signals))
    times, timing_source = _prepare_timing(tr, frame_times, prepared_signals)
    prepared_sources = _prepare_sources(
        run_count=len(prepared_signals),
        sources=sources,
        include_confounds=False,
    )
    return cls(
        _signals=prepared_signals,
        _design_matrices=designs,
        _frame_times=times,
        _timing_source=timing_source,
        _column_roles=roles,
        _run_metadata=metadata,
        _provenance=ProvenanceRecord(
            execution_id=str(uuid4()),
            sources=prepared_sources,
        ),
    )


def _prepare_designs(
    values: pd.DataFrame | Sequence[pd.DataFrame],
    signals: Sequence[np.ndarray],
) -> tuple[pd.DataFrame, ...]:
    designs = _as_design_runs(values)
    _validate_run_count(signals, designs, "design_matrices")
    return tuple(
        _prepare_design(design, signal.shape[0], run)
        for run, (design, signal) in enumerate(zip(designs, signals))
    )


def _as_design_runs(
    values: pd.DataFrame | Sequence[pd.DataFrame],
) -> tuple[pd.DataFrame, ...]:
    if isinstance(values, pd.DataFrame):
        return (values,)
    try:
        return tuple(values)
    except TypeError as error:
        raise ValueError("design_matrices must contain pandas DataFrames") from error


def _prepare_design(design: pd.DataFrame, n_rows: int, run: int) -> pd.DataFrame:
    if not isinstance(design, pd.DataFrame):
        raise ValueError(f"run {run} design matrix must be a pandas DataFrame")
    if 0 in design.shape:
        raise ValueError(f"run {run} design matrix must have nonzero dimensions")
    _validate_columns(design.columns, run)
    if len(design) != n_rows:
        raise ValueError(f"run {run} design matrix must contain {n_rows} rows")
    _validate_design_values(design, run)
    values = np.array(design.to_numpy(dtype=np.float64), dtype=np.float64, order="C")
    return pd.DataFrame(values, columns=design.columns)


def _validate_columns(columns: pd.Index, run: int) -> None:
    if any(not isinstance(name, str) or not name.strip() for name in columns):
        raise ValueError(f"run {run} design columns must be nonempty strings")
    if columns.has_duplicates:
        raise ValueError(f"run {run} design columns must not contain duplicate names")


def _validate_design_values(design: pd.DataFrame, run: int) -> None:
    if any(not pd.api.types.is_numeric_dtype(dtype) for dtype in design.dtypes):
        raise ValueError(f"run {run} design values must be finite numeric values")
    if any(pd.api.types.is_bool_dtype(dtype) for dtype in design.dtypes):
        raise ValueError(f"run {run} design values must be finite numeric values")
    values = design.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"run {run} design values must be finite numeric values")


def _prepare_roles(
    values: Mapping[str, str] | Sequence[Mapping[str, str]],
    designs: Sequence[pd.DataFrame],
) -> tuple[Mapping[str, ColumnRole], ...]:
    roles = _as_role_runs(values)
    _validate_run_count(designs, roles, "column_roles")
    return tuple(
        _prepare_run_roles(role_map, design.columns, run)
        for run, (role_map, design) in enumerate(zip(roles, designs))
    )


def _as_role_runs(
    values: Mapping[str, str] | Sequence[Mapping[str, str]],
) -> tuple[Mapping[str, str], ...]:
    if isinstance(values, Mapping):
        return (values,)
    try:
        return tuple(values)
    except TypeError as error:
        raise ValueError(
            "column_roles must contain one role per design column"
        ) from error


def _prepare_run_roles(
    roles: Mapping[str, str], columns: pd.Index, run: int
) -> Mapping[str, ColumnRole]:
    if not isinstance(roles, Mapping) or set(roles) != set(columns):
        raise ValueError(
            f"run {run} column_roles must contain one role per design column"
        )
    if any(role not in _COLUMN_ROLES for role in roles.values()):
        raise ValueError(f"run {run} has an invalid column role")
    return MappingProxyType({name: roles[name] for name in columns})


def _prepare_metadata(
    values: Mapping[str, object] | Sequence[Mapping[str, object]] | None,
    run_count: int,
) -> tuple[Mapping[str, object], ...]:
    if values is None:
        return tuple(MappingProxyType({}) for _ in range(run_count))
    metadata = _as_metadata_runs(values)
    _validate_run_count(range(run_count), metadata, "run_metadata")
    return tuple(_freeze_metadata(item) for item in metadata)


def _as_metadata_runs(
    values: Mapping[str, object] | Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    if isinstance(values, Mapping):
        return (values,)
    try:
        return tuple(values)
    except TypeError as error:
        raise ValueError("run_metadata must contain mappings") from error


def _freeze_metadata(values: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(values, Mapping):
        raise ValueError("run_metadata must contain mappings")
    return _freeze_mapping(values, path_safe=True)


def _validate_run_count(
    signals: Sequence[object], values: Sequence[object], name: str
) -> None:
    if len(signals) != len(values):
        raise ValueError(f"{name} must contain one value per signal run")
