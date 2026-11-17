from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import logging
import re
from uuid import uuid4

import numpy as np
import pandas as pd

from boldtailor._arrays import owned_table
from boldtailor._run_inputs import (
    as_signal_runs,
    prepare_signal,
    prepare_sources,
    prepare_timing,
    validate_feature_counts,
    validate_run_count,
)
from boldtailor.provenance import software_environment
from boldtailor.logging import (
    append_event_history,
    bind_context,
    emit_event,
    emit_record,
    make_event,
)
from boldtailor.provenance import ProvenanceRecord, RunSources, sources_fingerprint

_EVENT_COLUMNS = ("onset", "duration")


@dataclass(frozen=True)
class AnalysisData:
    _signals: tuple[np.ndarray, ...]
    _events: tuple[pd.DataFrame, ...]
    _confounds: tuple[pd.DataFrame, ...]
    _frame_times: tuple[np.ndarray, ...]
    _timing_source: str
    _provenance: ProvenanceRecord

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
    def events(self) -> tuple[pd.DataFrame, ...]:
        return tuple(owned_table(frame) for frame in self._events)

    @property
    def confounds(self) -> tuple[pd.DataFrame, ...]:
        return tuple(owned_table(frame) for frame in self._confounds)

    @property
    def frame_times(self) -> tuple[np.ndarray, ...]:
        return self._frame_times

    @property
    def timing_source(self) -> str:
        return self._timing_source

    @property
    def provenance(self) -> ProvenanceRecord:
        return self._provenance


def from_arrays(
    signals: np.ndarray | Sequence[np.ndarray],
    events: pd.DataFrame | Sequence[pd.DataFrame],
    *,
    tr: float | None = None,
    frame_times: np.ndarray | Sequence[np.ndarray] | None = None,
    confounds: pd.DataFrame | Sequence[pd.DataFrame] | None = None,
    sources: Sequence[RunSources] | None = None,
    provenance_metadata: Mapping[str, object] | None = None,
) -> AnalysisData:
    execution_id = str(uuid4())
    data_id = None
    with bind_context(execution_id=execution_id, inherit=False):
        history = append_event_history(
            (),
            emit_event("normalization_started", stage="data"),
        )
        try:
            run_signals = as_signal_runs(signals)
            run_events = _as_table_runs(events)
            validate_run_count(run_signals, run_events, "events")
            prepared_sources = prepare_sources(
                run_count=len(run_signals),
                sources=sources,
                include_confounds=confounds is not None,
            )
            data_id = sources_fingerprint(prepared_sources)
            prepared_signals = tuple(
                prepare_signal(values, run) for run, values in enumerate(run_signals)
            )
            validate_feature_counts(prepared_signals)
            prepared_events = tuple(
                _prepare_events(frame, run) for run, frame in enumerate(run_events)
            )
            prepared_confounds = _prepare_confounds(confounds, prepared_signals)
            prepared_times, timing_source = prepare_timing(
                tr,
                frame_times,
                prepared_signals,
            )
            activity = _normalization_activity(
                timing_source=timing_source,
                n_runs=len(prepared_signals),
                n_features=prepared_signals[0].shape[1],
                provenance_metadata=provenance_metadata,
            )
            completed = make_event(
                "normalization_completed", stage="data", data_id=data_id
            )
            provenance = ProvenanceRecord(
                execution_id=execution_id,
                sources=prepared_sources,
                activities=(activity,),
                events=append_event_history(history, completed),
            )
            result = AnalysisData(
                _signals=prepared_signals,
                _events=prepared_events,
                _confounds=prepared_confounds,
                _frame_times=prepared_times,
                _timing_source=timing_source,
                _provenance=provenance,
            )
        except Exception as error:
            emit_event(
                "normalization_failed",
                stage="data",
                level=logging.ERROR,
                error=error,
                data_id=data_id,
            )
            raise
        emit_record(completed)
        return result


def _as_table_runs(
    values: pd.DataFrame | Sequence[pd.DataFrame],
) -> tuple[pd.DataFrame, ...]:
    if isinstance(values, pd.DataFrame):
        return (values,)
    return tuple(values)


def run_labels_for(
    data: AnalysisData, run_labels: Sequence[str] | None
) -> tuple[str, ...]:
    """Return validated run labels, defaulting to ``run-01``, ``run-02``, ..."""
    labels = (
        tuple(run_labels)
        if run_labels is not None
        else tuple(f"run-{i + 1:02d}" for i in range(data.n_runs))
    )
    if (
        len(labels) != data.n_runs
        or len(set(labels)) != len(labels)
        or not all(
            isinstance(x, str) and re.fullmatch(r"[A-Za-z0-9_-]+", x) for x in labels
        )
    ):
        raise ValueError(
            "run labels must be unique [A-Za-z0-9_-]+ strings, one per run"
        )
    return labels


def _prepare_events(frame: pd.DataFrame, run: int) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"run {run} events must be a pandas DataFrame")
    missing = [name for name in _EVENT_COLUMNS if name not in frame]
    if missing:
        raise ValueError(f"run {run} events missing columns: {', '.join(missing)}")
    copied = owned_table(frame).reset_index(drop=True)
    try:
        onset = copied["onset"].to_numpy(dtype=float)
        duration = copied["duration"].to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"run {run} event timing must be numeric") from error
    if not np.isfinite(onset).all() or not np.isfinite(duration).all():
        raise ValueError(f"run {run} event timing must be finite")
    if np.any(duration < 0):
        raise ValueError(f"run {run} event durations must be non-negative")
    return copied


def _prepare_confounds(
    confounds: pd.DataFrame | Sequence[pd.DataFrame] | None,
    signals: tuple[np.ndarray, ...],
) -> tuple[pd.DataFrame, ...]:
    if confounds is None:
        return tuple(pd.DataFrame(index=range(run.shape[0])) for run in signals)
    run_confounds = _as_table_runs(confounds)
    validate_run_count(signals, run_confounds, "confounds")
    prepared = []
    for index, (frame, run) in enumerate(zip(run_confounds, signals)):
        if not isinstance(frame, pd.DataFrame):
            raise ValueError(f"run {index} confounds must be a pandas DataFrame")
        if len(frame) != run.shape[0]:
            raise ValueError(f"run {index} confounds must contain {run.shape[0]} rows")
        prepared.append(owned_table(frame).reset_index(drop=True))
    return tuple(prepared)


def _normalization_activity(
    *,
    timing_source: str,
    n_runs: int,
    n_features: int,
    provenance_metadata: Mapping[str, object] | None,
) -> Mapping[str, object]:
    activity: dict[str, object] = {
        "name": "normalize",
        "stage": "data",
        "timing_source": timing_source,
        "run_count": n_runs,
        "feature_count": n_features,
        "software": software_environment(),
    }
    if provenance_metadata is not None:
        activity["metadata"] = provenance_metadata
    return activity


def subset_runs(data: AnalysisData, indices: Sequence[int]) -> AnalysisData:
    """Subset run data and source records together without relabeling features."""
    signals, events, times = data.signals, data.events, data.frame_times
    confounds, sources = data.confounds, data.provenance.sources
    return from_arrays(
        [signals[i] for i in indices],
        [events[i] for i in indices],
        frame_times=[times[i] for i in indices],
        confounds=[confounds[i] for i in indices],
        sources=[sources[i] for i in indices],
    )
