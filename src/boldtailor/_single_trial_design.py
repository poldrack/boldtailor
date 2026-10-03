"""Compile event-wise task columns without behavioral or condition pooling."""

from __future__ import annotations


import numpy as np
import pandas as pd
from boldtailor._hrf_design import (
    convolve_events,
    identified_candidate,
    trial_regressors,
)

RESERVED_COLUMNS = frozenset(
    {"trial_id", "trial_index", "run_index", "run_label", "event_index"}
)


def compile_trial_run(events, frame_times, confounds, run_label, *, hrf="spm"):
    """Return task and nuisance matrices plus a row-aligned trial table."""
    times = np.asarray(frame_times, dtype=float)
    identified_candidate(hrf)
    _validate_events(events, times, run_label)
    nuisance = _nuisance_matrix(confounds, len(times))
    table = run_trial_table(events, run_label)
    check_trial_ids(table.trial_id, nuisance.columns)
    columns = trial_regressors(events.reset_index(drop=True), times, hrf)
    return pd.DataFrame(columns, columns=list(table.trial_id)), nuisance, table


def run_trial_table(events, run_label):
    """One run's events led by trial_id, run_label, and event_index."""
    table = events.copy(deep=True).reset_index(drop=True)
    ids = [f"{run_label}_trial-{i + 1:04d}" for i in range(len(table))]
    table.insert(0, "event_index", np.arange(len(table)))
    table.insert(0, "run_label", run_label)
    table.insert(0, "trial_id", ids)
    return table


def trial_table(events_per_run, labels):
    """All runs' trials in run order, with global trial_index and run_index."""
    tables = [
        run_trial_table(events, label).assign(run_index=run)
        for run, (events, label) in enumerate(zip(events_per_run, labels, strict=True))
    ]
    trials = pd.concat(tables, ignore_index=True)
    trials.insert(0, "trial_index", np.arange(len(trials)))
    return trials


def check_trial_ids(trial_ids, nuisance_columns):
    if set(trial_ids).intersection(nuisance_columns):
        raise ValueError("nuisance columns collide with reserved trial IDs")


def _validate_events(events, times, run_label):
    if not isinstance(run_label, str):
        raise ValueError("run labels must be strings")
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all():
        raise ValueError("frame times require at least two finite samples")
    if np.any(np.diff(times) <= 0):
        raise ValueError("frame times must be strictly increasing")
    if not isinstance(events, pd.DataFrame) or events.empty:
        raise ValueError("events must be a nonempty table")
    if not events.columns.is_unique or not all(isinstance(c, str) for c in events):
        raise ValueError("event columns must have unique string names")
    if RESERVED_COLUMNS.intersection(events.columns):
        raise ValueError("events contain reserved trial-table names")
    if not {"onset", "duration"}.issubset(events):
        raise ValueError("events require onset and duration timing")
    values = events[["onset", "duration"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values[:, 1] < 0):
        raise ValueError("event timing must be finite with nonnegative duration")


def _nuisance_matrix(confounds, n_scans):
    if not isinstance(confounds, pd.DataFrame) or len(confounds) != n_scans:
        raise ValueError("confound rows must match frame times")
    if not confounds.columns.is_unique or not all(
        isinstance(c, str) for c in confounds
    ):
        raise ValueError("confounds require unique string column names")
    if "constant" in confounds:
        raise ValueError("constant is reserved for the run intercept")
    nuisance = confounds.astype(float).reset_index(drop=True)
    if not np.isfinite(nuisance.to_numpy()).all():
        raise ValueError("confounds must be finite")
    return nuisance.assign(constant=1.0)


def _trial_column(onset, duration, times, trial_id, hrf="spm"):
    return convolve_events([onset], [duration], times, hrf, trial_id)
