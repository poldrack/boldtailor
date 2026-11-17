"""Validation of one run's events, frame times, and confounds.

A leaf module: every design builder checks its inputs here, so none of them
has to import another builder to do it.
"""

import numpy as np
import pandas as pd

from boldtailor._constants import MIN_ONSET

RESERVED_COLUMNS = frozenset(
    {"trial_id", "trial_index", "run_index", "run_label", "event_index"}
)


def validate_onsets(onsets, times, subject="onset"):
    """Reject onsets whose response would not be sampled by the run."""
    onsets = np.asarray(onsets, dtype=float)
    if np.any(onsets < times[0] + MIN_ONSET) or np.any(onsets >= times[-1]):
        raise ValueError(f"{subject} has no supported sampled response")


def validate_events(events, times, run_label):
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


def nuisance_matrix(confounds, n_scans):
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
