"""One identified HRF per convolution; candidate bases are never orthogonalized."""

import hashlib
from functools import lru_cache
import numpy as np
from nilearn.glm.first_level import compute_regressor
from nilearn.glm.first_level.hemodynamic_models import _sample_condition

from boldtailor.hrf_library import HrfCandidate


def hrf_model(candidate):
    if isinstance(candidate, str) and candidate == "spm":
        return "spm"
    if not isinstance(candidate, HrfCandidate):
        raise ValueError("hrf must be 'spm' or an identified HrfCandidate")
    return "spm" if candidate.kind == "spm" else candidate.kernel


def hrf_metadata(candidate):
    hrf_model(candidate)
    if isinstance(candidate, str) or candidate.kind == "spm":
        return "spm"
    return dict(
        id=candidate.id,
        kind=candidate.kind,
        parameters=list(candidate.parameters),
        kernel_fingerprint=hashlib.sha256(
            candidate.kernel(0.1, 1).tobytes()
        ).hexdigest(),
    )


def convolve_events(onsets, durations, times, candidate, name="stimulus"):
    onsets = np.asarray(onsets, dtype=float)
    if np.any(onsets < times[0] - 24) or np.any(onsets >= times[-1]):
        raise ValueError(f"{name}: onset has no supported sampled response")
    column, _ = compute_regressor(
        np.array([onsets, durations, np.ones(len(onsets))]),
        hrf_model(candidate),
        times,
        con_id=name,
        oversampling=50,
    )
    if not np.isfinite(column).all() or not np.any(column):
        raise ValueError(f"{name}: no supported sampled response")
    return column[:, 0]


def stimulus_regressor(events, frame_times, candidate):
    """Convolve unit-amplitude presentations on the actual run sampling grid."""
    from boldtailor._single_trial_design import _validate_events

    times = np.asarray(frame_times, dtype=float)
    _validate_events(events, times, "stimulus")
    return convolve_events(events.onset, events.duration, times, candidate)


@lru_cache(maxsize=64)
def _boxcar_sampling(timing_bytes, times_bytes):
    timing = np.frombuffer(timing_bytes, dtype="<f8").reshape(-1, 2)
    times = np.frombuffer(times_bytes, dtype="<f8")
    condition = np.vstack([timing.T, np.ones(len(timing))])
    # Pinned Nilearn 0.14: retain its precise event grid and impulse convention.
    _, grid = _sample_condition(condition, times, oversampling=50)
    starts = np.minimum(np.searchsorted(grid, timing[:, 0]), len(grid) - 1)
    stops = np.minimum(np.searchsorted(grid, timing.sum(axis=1)), len(grid) - 1)
    stops = np.where((stops == starts) & (stops < len(grid) - 1), stops + 1, stops)
    upper = np.searchsorted(grid, times)
    lower = upper - 1
    fraction = (times - grid[lower]) / (grid[upper] - grid[lower])
    return starts, stops, lower, upper, fraction


def trial_regressors(events, frame_times, candidate):
    """Convolve unit boxcars using kernel prefix sums on Nilearn's sample grid.

    A sampled boxcar [a,b) convolved with h equals H[j-a]-H[j-b], where H
    is the cumulative kernel. Evaluate only the two samples needed for each
    acquisition's linear interpolation. Candidate 0 retains direct Nilearn.
    """
    from boldtailor._single_trial_design import _validate_events

    times = np.asarray(frame_times, dtype=float)
    _validate_events(events, times, "trials")
    if hrf_model(candidate) == "spm":
        return np.column_stack(
            [
                convolve_events([o], [d], times, candidate)
                for o, d in zip(events.onset, events.duration, strict=True)
            ]
        )
    if np.any(events.onset < times[0] - 24) or np.any(events.onset >= times[-1]):
        raise ValueError("trial onset has no supported sampled response")
    timing = np.asarray(events[["onset", "duration"]], dtype="<f8").tobytes()
    starts, stops, lower, upper, fraction = _boxcar_sampling(
        timing, np.asarray(times, dtype="<f8").tobytes()
    )
    kernel = candidate.kernel(float(np.min(np.diff(times))), 50)
    prefix = np.r_[0.0, np.cumsum(kernel)]

    def sampled(indices):
        a = np.clip(indices[:, None] - starts + 1, 0, len(kernel))
        b = np.clip(indices[:, None] - stops + 1, 0, len(kernel))
        return prefix[a] - prefix[b]

    values = (1 - fraction[:, None]) * sampled(lower) + fraction[:, None] * sampled(
        upper
    )
    if not np.isfinite(values).all() or not np.all(np.any(values != 0, axis=0)):
        raise ValueError("trial has no supported sampled response")
    return values
