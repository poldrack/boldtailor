"""Expand raw trials into Nilearn conditions and let Nilearn build the columns."""

from contextlib import redirect_stdout
import io

import numpy as np
import pandas as pd
from nilearn.glm.first_level import make_first_level_design_matrix

from boldtailor._hrf_design import (
    MIN_ONSET,
    OVERSAMPLING,
    resolve_hrf,
    scale_event_amplitudes,
)
from boldtailor.model import TaskModel


def _numeric(events, column, run):
    if column not in events:
        raise ValueError(f"run {run}: events lack modulator column {column!r}")
    try:
        return pd.to_numeric(events[column], errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"run {run}: modulator {column!r} must be numeric") from error


def _modulator_amplitudes(events, modulator, run):
    values = _numeric(events, modulator.column, run)
    observed = np.isfinite(values)
    if not observed.any():
        raise ValueError(
            f"run {run}: modulator {modulator.column!r} has no observed values"
        )
    if modulator.missing == "error" and not observed.all():
        raise ValueError(f"run {run}: modulator {modulator.column!r} must be finite")
    amplitude = np.zeros(len(values))
    amplitude[observed] = values[observed]
    columns = {modulator.column: amplitude}
    if modulator.missing == "indicator" and not observed.all():
        columns[modulator.indicator_name] = (~observed).astype(float)
    return columns


def expand_events(events, task_model, run=0):
    """One Nilearn condition per task-model regressor; amplitudes become modulation."""
    if not isinstance(task_model, TaskModel):
        raise ValueError("task_model must be a TaskModel")
    if not {"onset", "duration"}.issubset(events.columns):
        raise ValueError(f"run {run}: events require onset and duration columns")
    all_amplitudes = {}
    all_amplitudes["task"] = np.ones(len(events))
    for modulator in task_model.modulators:
        mods = _modulator_amplitudes(events, modulator, run)
        all_amplitudes.update(mods)
    for name, values in all_amplitudes.items():
        if not np.any(values != 0):
            raise ValueError(f"run {run}: regressor {name!r} has no nonzero amplitude")
    timing = events[["onset", "duration"]].reset_index(drop=True)
    regressor_order = list(task_model.regressor_names) + list(task_model.profiled_names)
    ordered_amplitudes = {
        name: all_amplitudes[name] for name in regressor_order if name in all_amplitudes
    }
    return pd.concat(
        [
            timing.assign(trial_type=name, modulation=values)
            for name, values in ordered_amplitudes.items()
        ],
        ignore_index=True,
    )


def task_columns(
    expanded, frame_times, hrf, *, min_onset=MIN_ONSET, oversampling=OVERSAMPLING
):
    """Nilearn task columns for one HRF: no drift, no constant, semantic names.

    Plain 'spm' and 'glover' become their peak-one kernels before Nilearn, and
    kernel amplitudes are scaled so each event's response peaks at one.
    """
    times = np.asarray(frame_times, dtype=float)
    hrf = resolve_hrf(hrf)
    if callable(hrf):
        expanded = scale_event_amplitudes(expanded, hrf, times, oversampling, min_onset)
    with redirect_stdout(io.StringIO()):
        matrix = make_first_level_design_matrix(
            times,
            events=expanded,
            hrf_model=hrf,
            drift_model=None,
            min_onset=min_onset,
            oversampling=oversampling,
        )
    matrix = matrix.drop(columns="constant")
    if callable(hrf):
        matrix.columns = [name.removesuffix("_kernel") for name in matrix.columns]
    return matrix.loc[:, list(dict.fromkeys(expanded.trial_type))]


def run_task_columns(
    events, task_model, frame_times, hrf, *, run, min_onset, oversampling
):
    """Task columns for one run; data errors name the run exactly once."""
    try:
        expanded = expand_events(events, task_model, run)
        return task_columns(
            expanded, frame_times, hrf, min_onset=min_onset, oversampling=oversampling
        )
    except ValueError as error:
        detail = str(error).removeprefix(f"run {run}: ")
        raise ValueError(f"run {run} design compilation failed: {detail}") from error
