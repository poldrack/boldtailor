"""Compile event-wise task columns without behavioral or condition pooling."""

from __future__ import annotations


import numpy as np
import pandas as pd
from boldtailor._design_inputs import nuisance_matrix, validate_events
from boldtailor._hrf_design import (
    identified_candidate,
    trial_regressors,
)


def compile_trial_run(events, frame_times, confounds, run_label, *, hrf="spm"):
    """Return task and nuisance matrices plus a row-aligned trial table."""
    times = np.asarray(frame_times, dtype=float)
    identified_candidate(hrf)
    validate_events(events, times, run_label)
    nuisance = nuisance_matrix(confounds, len(times))
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
