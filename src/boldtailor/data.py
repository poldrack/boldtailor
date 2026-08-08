from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd

from boldtailor._arrays import immutable_float_array

_EVENT_COLUMNS = ("onset", "duration")


@dataclass(frozen=True)
class AnalysisData:
    _signals: tuple[np.ndarray, ...]
    _events: tuple[pd.DataFrame, ...]
    _confounds: tuple[pd.DataFrame, ...]
    _frame_times: tuple[np.ndarray, ...]
    _timing_source: str

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
        return tuple(_owned_table(frame) for frame in self._events)

    @property
    def confounds(self) -> tuple[pd.DataFrame, ...]:
        return tuple(_owned_table(frame) for frame in self._confounds)

    @property
    def frame_times(self) -> tuple[np.ndarray, ...]:
        return self._frame_times

    @property
    def timing_source(self) -> str:
        return self._timing_source


def from_arrays(
    signals: np.ndarray | Sequence[np.ndarray],
    events: pd.DataFrame | Sequence[pd.DataFrame],
    *,
    tr: float | None = None,
    frame_times: np.ndarray | Sequence[np.ndarray] | None = None,
    confounds: pd.DataFrame | Sequence[pd.DataFrame] | None = None,
) -> AnalysisData:
    run_signals = _as_signal_runs(signals)
    run_events = _as_table_runs(events)
    _validate_run_count(run_signals, run_events, "events")
    prepared_signals = tuple(
        _prepare_signal(values, run) for run, values in enumerate(run_signals)
    )
    _validate_feature_counts(prepared_signals)
    prepared_events = tuple(
        _prepare_events(frame, run) for run, frame in enumerate(run_events)
    )
    prepared_confounds = _prepare_confounds(confounds, prepared_signals)
    prepared_times, timing_source = _prepare_timing(tr, frame_times, prepared_signals)
    return AnalysisData(
        _signals=prepared_signals,
        _events=prepared_events,
        _confounds=prepared_confounds,
        _frame_times=prepared_times,
        _timing_source=timing_source,
    )


def _as_signal_runs(
    values: np.ndarray | Sequence[np.ndarray],
) -> tuple[np.ndarray, ...]:
    if isinstance(values, np.ndarray):
        return (values,)
    runs = tuple(values)
    if not runs:
        raise ValueError("signals must contain at least one run")
    return runs


def _as_table_runs(
    values: pd.DataFrame | Sequence[pd.DataFrame],
) -> tuple[pd.DataFrame, ...]:
    if isinstance(values, pd.DataFrame):
        return (values,)
    return tuple(values)


def _validate_run_count(
    signals: Sequence[object], values: Sequence[object], name: str
) -> None:
    if len(signals) != len(values):
        raise ValueError(f"{name} must contain one value per signal run")


def _prepare_signal(values: np.ndarray, run: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError(f"run {run} signals must have shape time x features")
    if 0 in array.shape:
        raise ValueError(f"run {run} signals must have nonzero dimensions")
    try:
        finite = np.isfinite(array).all()
    except TypeError as error:
        raise ValueError(f"run {run} signals must be numeric and finite") from error
    if not finite:
        raise ValueError(f"run {run} signals must be finite")
    return immutable_float_array(array)


def _validate_feature_counts(signals: tuple[np.ndarray, ...]) -> None:
    if len({run.shape[1] for run in signals}) != 1:
        raise ValueError("all runs must have the same number of features")


def _prepare_events(frame: pd.DataFrame, run: int) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"run {run} events must be a pandas DataFrame")
    missing = [name for name in _EVENT_COLUMNS if name not in frame]
    if missing:
        raise ValueError(f"run {run} events missing columns: {', '.join(missing)}")
    copied = _owned_table(frame).reset_index(drop=True)
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
    _validate_run_count(signals, run_confounds, "confounds")
    prepared = []
    for index, (frame, run) in enumerate(zip(run_confounds, signals)):
        if not isinstance(frame, pd.DataFrame):
            raise ValueError(f"run {index} confounds must be a pandas DataFrame")
        if len(frame) != run.shape[0]:
            raise ValueError(f"run {index} confounds must contain {run.shape[0]} rows")
        prepared.append(_owned_table(frame).reset_index(drop=True))
    return tuple(prepared)


def _owned_table(frame: pd.DataFrame) -> pd.DataFrame:
    copied = frame.copy(deep=True)
    for column, dtype in copied.dtypes.items():
        if dtype != object:
            continue
        payloads = [deepcopy(value) for value in copied[column]]
        copied[column] = pd.Series(payloads, index=copied.index, dtype=object)
    return copied


def _prepare_timing(
    tr: float | None,
    frame_times: np.ndarray | Sequence[np.ndarray] | None,
    signals: tuple[np.ndarray, ...],
) -> tuple[tuple[np.ndarray, ...], str]:
    if (tr is None) == (frame_times is None):
        raise ValueError("provide exactly one of tr or frame_times")
    if tr is not None:
        tr_value = _prepare_tr(tr)
        times = tuple(
            immutable_float_array(np.arange(run.shape[0]) * tr_value) for run in signals
        )
        return times, "tr"
    run_times = _as_frame_time_runs(frame_times)
    _validate_run_count(signals, run_times, "frame_times")
    times = tuple(
        _prepare_frame_times(values, signal.shape[0], run)
        for run, (values, signal) in enumerate(zip(run_times, signals))
    )
    return times, "frame_times"


def _prepare_tr(tr: object) -> float:
    if not isinstance(tr, Real) or isinstance(tr, (bool, np.bool_)):
        raise ValueError("TR must be positive and finite")
    value = float(tr)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("TR must be positive and finite")
    return value


def _as_frame_time_runs(
    values: np.ndarray | Sequence[np.ndarray] | None,
) -> tuple[np.ndarray, ...]:
    if isinstance(values, np.ndarray):
        return (values,)
    run_values = tuple(values or ())
    if run_values and all(np.isscalar(value) for value in run_values):
        return (np.asarray(run_values, dtype=float),)
    return run_values


def _prepare_frame_times(values: np.ndarray, n_timepoints: int, run: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"run {run} frame_times must contain {n_timepoints} entries")
    if not np.isfinite(array).all():
        raise ValueError(f"run {run} frame_times must be finite")
    if np.any(np.diff(array) <= 0):
        raise ValueError(f"run {run} frame_times must be strictly increasing")
    if array.size != n_timepoints:
        raise ValueError(f"run {run} frame_times must contain {n_timepoints} entries")
    return immutable_float_array(array)
