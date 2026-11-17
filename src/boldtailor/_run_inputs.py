"""Normalize per-run signals, timing, and sources for raw and prepared analyses."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from boldtailor._arrays import readonly_array
from boldtailor._scalars import is_real
from boldtailor.provenance import RunSources, SourceRef


def as_signal_runs(
    values: np.ndarray | Sequence[np.ndarray],
) -> tuple[np.ndarray, ...]:
    if isinstance(values, np.ndarray):
        return (values,)
    runs = tuple(values)
    if not runs:
        raise ValueError("signals must contain at least one run")
    return runs


def validate_run_count(
    signals: Sequence[object], values: Sequence[object], name: str
) -> None:
    if len(signals) != len(values):
        raise ValueError(f"{name} must contain one value per signal run")


def prepare_signal(values: np.ndarray, run: int) -> np.ndarray:
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
    return readonly_array(array)


def validate_feature_counts(signals: tuple[np.ndarray, ...]) -> None:
    if len({run.shape[1] for run in signals}) != 1:
        raise ValueError("all runs must have the same number of features")


def prepare_sources(
    *,
    run_count: int,
    sources: Sequence[RunSources] | None,
    include_confounds: bool,
) -> tuple[RunSources, ...]:
    if sources is None:
        return tuple(_anonymous_sources(include_confounds) for _ in range(run_count))
    run_sources = tuple(sources)
    if len(run_sources) != run_count:
        raise ValueError("sources must contain one value per signal run")
    return tuple(RunSources.from_dict(source.to_dict()) for source in run_sources)


def _anonymous_sources(include_confounds: bool) -> RunSources:
    return RunSources(
        signal=SourceRef(role="signal"),
        events=SourceRef(role="events"),
        confounds=SourceRef(role="confounds") if include_confounds else None,
    )


def prepare_timing(
    tr: float | None,
    frame_times: np.ndarray | Sequence[np.ndarray] | None,
    signals: tuple[np.ndarray, ...],
) -> tuple[tuple[np.ndarray, ...], str]:
    if (tr is None) == (frame_times is None):
        raise ValueError("provide exactly one of tr or frame_times")
    if tr is not None:
        tr_value = _prepare_tr(tr)
        times = tuple(
            readonly_array(np.arange(run.shape[0]) * tr_value) for run in signals
        )
        return times, "tr"
    run_times = _as_frame_time_runs(frame_times)
    validate_run_count(signals, run_times, "frame_times")
    times = tuple(
        _prepare_frame_times(values, signal.shape[0], run)
        for run, (values, signal) in enumerate(zip(run_times, signals))
    )
    return times, "frame_times"


def _prepare_tr(tr: object) -> float:
    if not is_real(tr):
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
    return readonly_array(array)
