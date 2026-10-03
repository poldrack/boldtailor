from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import logging
from types import MappingProxyType
from typing import Any, Literal
from uuid import uuid4

import numpy as np
import pandas as pd

from boldtailor.data import (
    _as_signal_runs,
    _prepare_signal,
    _prepare_sources,
    _prepare_timing,
    _validate_run_count,
    _validate_feature_counts,
)
from boldtailor._software import software_environment
from boldtailor.logging import (
    _emit_record,
    _make_event,
    append_event_history,
    bind_context,
    emit_event,
)
from boldtailor.provenance import (
    ProvenanceRecord,
    RunSources,
    _freeze_mapping,
    _metadata_fingerprint,
    _thaw,
)

ColumnRole = Literal["task", "nuisance", "intercept", "other"]
_COLUMN_ROLES = frozenset({"task", "nuisance", "intercept", "other"})
_DESIGN_ID_SCHEMA = "boldtailor.prepared-design/1"


@dataclass(frozen=True)
class PreparedDesignAnalysis:
    """Owned inputs with defensive public table accessors.

    Package fitting code reads `_design_matrices` and `_column_roles` directly
    to avoid copying all runs per diagnostic. It must never mutate those values.
    """

    _signals: tuple[np.ndarray, ...]
    _design_matrices: tuple[pd.DataFrame, ...]
    _frame_times: tuple[np.ndarray, ...]
    _timing_source: str
    _column_roles: tuple[Mapping[str, ColumnRole], ...]
    _run_metadata: tuple[Mapping[str, object], ...]
    _run_design_fingerprints: tuple[str, ...]
    _design_fingerprint: str
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
    def run_design_fingerprints(self) -> tuple[str, ...]:
        return self._run_design_fingerprints

    @property
    def design_fingerprint(self) -> str:
        return self._design_fingerprint

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
    execution_id = str(uuid4())
    data_id = None
    with bind_context(execution_id=execution_id, inherit=False):
        history = append_event_history(
            (),
            emit_event("normalization_started", stage="prepared_design"),
        )
        try:
            signal_runs = _as_signal_runs(signals)
            prepared_sources = _prepare_sources(
                run_count=len(signal_runs),
                sources=sources,
                include_confounds=False,
            )
            data_id = _metadata_fingerprint(prepared_sources)
            prepared_signals = tuple(
                _prepare_signal(values, run) for run, values in enumerate(signal_runs)
            )
            _validate_feature_counts(prepared_signals)
            designs = _prepare_designs(design_matrices, prepared_signals)
            roles = _prepare_roles(column_roles, designs)
            metadata = _prepare_metadata(run_metadata, len(prepared_signals))
            prepared_provenance_metadata = _prepare_provenance_metadata(
                provenance_metadata
            )
            times, timing_source = _prepare_timing(
                tr,
                frame_times,
                prepared_signals,
            )
            run_fingerprints = tuple(
                _run_design_fingerprint(design, frame_time, run_roles)
                for design, frame_time, run_roles in zip(designs, times, roles)
            )
            design_fingerprint = _aggregate_design_fingerprint(run_fingerprints)
            activity = _normalization_activity(
                timing_source=timing_source,
                designs=designs,
                roles=roles,
                n_features=prepared_signals[0].shape[1],
                run_fingerprints=run_fingerprints,
                design_fingerprint=design_fingerprint,
                run_metadata=metadata,
                provenance_metadata=prepared_provenance_metadata,
            )
            completed = _make_event(
                "normalization_completed", stage="prepared_design", data_id=data_id
            )
            provenance = ProvenanceRecord(
                execution_id=execution_id,
                sources=prepared_sources,
                activities=(activity,),
                events=append_event_history(history, completed),
            )
            result = cls(
                _signals=prepared_signals,
                _design_matrices=designs,
                _frame_times=times,
                _timing_source=timing_source,
                _column_roles=roles,
                _run_metadata=metadata,
                _run_design_fingerprints=run_fingerprints,
                _design_fingerprint=design_fingerprint,
                _provenance=provenance,
            )
        except Exception as error:
            emit_event(
                "normalization_failed",
                stage="prepared_design",
                level=logging.ERROR,
                error=error,
                data_id=data_id,
            )
            raise
        _emit_record(completed)
        return result


def _normalization_activity(
    *,
    timing_source: str,
    designs: Sequence[pd.DataFrame],
    roles: Sequence[Mapping[str, ColumnRole]],
    n_features: int,
    run_fingerprints: Sequence[str],
    design_fingerprint: str,
    run_metadata: Sequence[Mapping[str, object]],
    provenance_metadata: Mapping[str, object] | None,
) -> Mapping[str, object]:
    runs = [
        {
            "columns": list(design.columns),
            "roles": [run_roles[name] for name in design.columns],
        }
        for design, run_roles in zip(designs, roles)
    ]
    activity = {
        "name": "normalize_prepared_design",
        "stage": "prepared_design",
        "timing_source": timing_source,
        "run_count": len(designs),
        "feature_count": n_features,
        "runs": runs,
        "run_design_fingerprints": list(run_fingerprints),
        "design_fingerprint": design_fingerprint,
        "run_metadata": [_thaw(metadata) for metadata in run_metadata],
        "software": software_environment(),
    }
    if provenance_metadata is not None:
        activity["metadata"] = _thaw(provenance_metadata)
    return activity


def _run_design_fingerprint(
    design: pd.DataFrame,
    frame_times: np.ndarray,
    roles: Mapping[str, ColumnRole],
) -> str:
    hasher = hashlib.sha256()
    _hash_chunk(hasher, _DESIGN_ID_SCHEMA.encode("utf-8"))
    header = {
        "shape": list(design.shape),
        "columns": list(design.columns),
        "roles": [roles[name] for name in design.columns],
        "dtype": "<f8",
    }
    _hash_chunk(hasher, _canonical_json(header).encode("utf-8"))
    _hash_chunk(
        hasher,
        np.asarray(frame_times, dtype="<f8", order="C").tobytes(),
    )
    _hash_chunk(
        hasher,
        np.asarray(design, dtype="<f8", order="C").tobytes(),
    )
    return hasher.hexdigest()


def _aggregate_design_fingerprint(run_fingerprints: Sequence[str]) -> str:
    hasher = hashlib.sha256()
    _hash_chunk(hasher, _DESIGN_ID_SCHEMA.encode("utf-8"))
    payload = {"run_design_fingerprints": list(run_fingerprints)}
    _hash_chunk(hasher, _canonical_json(payload).encode("utf-8"))
    return hasher.hexdigest()


def _hash_chunk(hasher: Any, payload: bytes) -> None:
    hasher.update(len(payload).to_bytes(8, "big"))
    hasher.update(payload)


def _canonical_json(payload: Mapping[str, object]) -> str:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
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


def _prepare_provenance_metadata(
    values: Mapping[str, object] | None,
) -> Mapping[str, object] | None:
    if values is None:
        return None
    return _freeze_mapping(values, path_safe=True)
