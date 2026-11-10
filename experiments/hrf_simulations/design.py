"""Mumford task models with physical unit-height neural boxcars."""

from functools import lru_cache

import numpy as np
from scipy.integrate import cumulative_trapezoid

from experiments.hrf_estimation.design import DesignParts, partition_events
from experiments.hrf_estimation.timing import sampling_interval

MODELS = ("ConsDurNoRT", "RTDur", "ConsDurRTMod", "ConsDurRTDur")
CONVENTIONS = ("brief", "stimulus")


def durations(events, convention):
    if convention not in CONVENTIONS:
        raise ValueError(f"unknown duration convention: {convention}")
    return (
        np.full(len(events), 0.1)
        if convention == "brief"
        else events.duration.to_numpy(float)
    )


def convolve(onsets, lengths, heights, times, candidate):
    """Integrate a unit-peak HRF against boxcars, without event normalization."""
    sampling_interval(times)
    if not len(onsets):
        return np.zeros(len(times))
    condition = np.array([onsets, lengths, heights], dtype=float)
    if not np.isfinite(condition).all() or (condition[1] < 0).any():
        raise ValueError("boxcars require finite values and nonnegative durations")
    if (condition[0] < times[0] - 24).any() or (condition[0] >= times[-1]).any():
        raise ValueError("onsets outside supported acquisition window")

    grid, integral = integrated_hrf(candidate)
    lag = np.asarray(times)[:, None] - condition[0]
    first = np.interp(lag, grid, integral, left=0.0, right=integral[-1])
    last = np.interp(lag - condition[1], grid, integral, left=0.0, right=integral[-1])
    return (first - last) @ condition[2]


@lru_cache(maxsize=1024)
def integrated_hrf(candidate):
    """Integrate HRF once; event boundaries are never rounded to grid bins."""
    dt = 0.005
    values = candidate.kernel(dt, 1)
    grid = np.arange(len(values)) * dt
    return grid, cumulative_trapezoid(values, dx=dt, initial=0.0)


def task_design(run, candidate, model, convention="stimulus"):
    if model not in MODELS:
        raise ValueError(f"unknown model: {model}")
    events = partition_events(run.events)
    fixed = durations(events, convention)
    valid = events.category.to_numpy() == "valid"
    rt = events.response_time.to_numpy(float)
    lengths = rt if model == "RTDur" else fixed

    def column(mask, duration, height):
        return convolve(
            events.onset.to_numpy()[mask],
            duration[mask],
            np.broadcast_to(height, len(events))[mask],
            run.frame_times,
            candidate,
        )

    task = [
        column(valid & (events.trial_type.to_numpy() == c), lengths, 1.0)
        for c in (0, 1)
    ]
    names = ("trial_type_0", "trial_type_1")
    if model == "ConsDurRTMod":
        task.append(column(valid, fixed, rt))
    elif model == "ConsDurRTDur":
        task.append(column(valid, rt, 1.0))
    if len(task) == 3:
        names += ("response_time",)
    junk = [
        column(events.category.to_numpy() == c, events.duration.to_numpy(), 1.0)
        for c in ("missing", "incorrect")
    ]
    return DesignParts(
        np.column_stack(task),
        np.column_stack(junk),
        run.confounds.to_numpy(float),
        task_names=names,
    )


def reference_design(run, include_junk=True):
    """Fixed reference spanning both duration conventions and both RT mechanisms."""
    from boldtailor.hrf_library import HrfCandidate

    blocks = [run.confounds.to_numpy(float)]
    for parameters in (
        (3.5, 12, 0.8, 1.5, 6, 0, 36),
        (6, 16, 1, 1, 6, 0, 36),
        (8, 20, 1.2, 1.5, 6, 0, 36),
    ):
        candidate = HrfCandidate(0, "double_gamma", parameters)
        for convention in CONVENTIONS:
            for model in ("ConsDurRTMod", "RTDur", "ConsDurRTDur"):
                parts = task_design(run, candidate, model, convention)
                blocks.append(parts.task)
        blocks.append(parts.junk if include_junk else np.zeros_like(parts.junk))
    return np.column_stack(blocks)
