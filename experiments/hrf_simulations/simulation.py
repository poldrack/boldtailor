"""Independent high-resolution synthesis of known neural activity and HRFs."""

import numpy as np
from scipy.signal import fftconvolve

from experiments.hrf_estimation.design import partition_events
from experiments.hrf_simulations.design import durations

MECHANISMS = ("none", "amplitude", "duration", "additive_duration")


def synthesize(onsets, lengths, heights, times, truth, dt=0.005):
    """Fractional-bin boxcars and direct convolution, independent of fit designs."""
    start = min(float(times[0]) - 24.0, float(min(onsets, default=times[0])) - 1.0)
    grid = start + np.arange(int(np.ceil((times[-1] - start) / dt)) + 2) * dt
    neural = np.zeros(len(grid))
    for onset, length, height in zip(onsets, lengths, heights, strict=True):
        # Fraction of each bin occupied; handles non-grid-aligned actual RTs.
        overlap = np.maximum(
            0.0, np.minimum(grid + dt, onset + length) - np.maximum(grid, onset)
        )
        neural += height * overlap / dt
    h = truth.kernel(dt, 1)
    response = fftconvolve(neural, h)[: len(grid)] * dt
    return np.interp(times, grid, response)


def signal_components(run, truth, mechanism, convention, gamma=0.5, betas=(1.0, 1.0)):
    if mechanism not in MECHANISMS:
        raise ValueError(f"unknown generating mechanism: {mechanism}")
    if len(betas) != 2 or not np.isfinite([*betas, gamma]).all():
        raise ValueError("two finite condition heights and finite gamma required")
    events = partition_events(run.events)
    fixed = durations(events, convention)
    valid = events.category.to_numpy() == "valid"
    rt = events.response_time.to_numpy(float)
    onsets = events.onset.to_numpy(float)
    heights = np.asarray(betas)[events.trial_type.to_numpy(int)]
    lengths = rt if mechanism == "duration" else fixed
    if mechanism == "amplitude":
        heights = heights + gamma * rt
    task = synthesize(
        onsets[valid], lengths[valid], heights[valid], run.frame_times, truth
    )
    if mechanism == "additive_duration":
        task += synthesize(
            onsets[valid],
            rt[valid],
            np.full(valid.sum(), gamma),
            run.frame_times,
            truth,
        )
    junk = synthesize(
        onsets[~valid],
        events.duration.to_numpy()[~valid],
        np.ones((~valid).sum()),
        run.frame_times,
        truth,
    )
    return task, junk
