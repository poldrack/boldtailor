"""A shared HRF for valid conditions, common RT, and two junk categories."""

from dataclasses import dataclass

import numpy as np
from nilearn.glm.first_level import compute_regressor

from boldtailor._hrf_design import trial_regressors, OVERSAMPLING


@dataclass(frozen=True)
class DesignParts:
    task: np.ndarray
    junk: np.ndarray
    nuisance: np.ndarray
    task_names: tuple = ("trial_type_0", "trial_type_1", "response_time")
    junk_names: tuple = ("missing", "incorrect")


def partition_events(events):
    columns = [
        "onset",
        "duration",
        "trial_type",
        "response_time",
        "is_correct",
        "missing_data",
    ]
    if not set(columns).issubset(events.columns):
        raise ValueError(f"events require {columns}")
    table = events.copy()
    for name in ("trial_type", "missing_data"):
        if not table[name].isin([0, 1]).all():
            raise ValueError(f"{name} must be binary")
    timing = table[["onset", "duration"]].to_numpy(dtype=float)
    if not np.isfinite(timing).all() or (table.duration < 0).any():
        raise ValueError("event timing must be finite with nonnegative durations")
    rt = table.response_time.to_numpy(dtype=float)
    missing = (table.missing_data.to_numpy() == 1) | ~np.isfinite(rt) | (rt <= 0)
    if not table.loc[~missing, "is_correct"].isin([0, 1]).all():
        raise ValueError("responded trials require binary is_correct")
    table["category"] = np.where(
        missing, "missing", np.where(table.is_correct == 0, "incorrect", "valid")
    )
    return table


def build_design(run, kernel, rt_mode="amplitude"):
    if rt_mode not in ("amplitude", "duration", "none"):
        raise ValueError("rt_mode must be amplitude, duration, or none")
    events = partition_events(run.events)
    trial_matrix = trial_regressors(events, run.frame_times, kernel)
    valid = events.category.to_numpy() == "valid"
    task = np.column_stack(
        [
            trial_matrix[:, valid & (events.trial_type.to_numpy() == level)].sum(axis=1)
            for level in (0, 1)
        ]
    )
    names = ("trial_type_0", "trial_type_1")
    if rt_mode != "none":
        rt = _rt_column(run.frame_times, events, trial_matrix, valid, kernel, rt_mode)
        task = np.column_stack([task, rt])
        names += ("response_time",)
    junk = np.column_stack(
        [
            trial_matrix[:, events.category.to_numpy() == category].sum(axis=1)
            for category in ("missing", "incorrect")
        ]
    )
    return DesignParts(
        task, junk, run.confounds.to_numpy(dtype=float), task_names=names
    )


def _rt_column(times, events, trial_matrix, valid, kernel, mode):
    if not valid.any():
        return np.zeros(len(times))
    if mode == "amplitude":
        return trial_matrix[:, valid] @ events.response_time.to_numpy()[valid]
    condition = np.array(
        [events.onset[valid], events.response_time[valid], np.ones(valid.sum())]
    )
    values, _ = compute_regressor(
        condition, kernel.kernel, times, oversampling=OVERSAMPLING
    )
    return values[:, 0]
