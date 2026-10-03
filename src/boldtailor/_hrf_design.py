"""One identified HRF per convolution; bases are never orthogonalized.

Kernels are stored at unit peak, and every event's amplitude is scaled so its
predicted response peaks at one. Nilearn's convolution does not multiply by
the sampling step, so without the event scale a beta would depend on TR,
oversampling, and duration. A beta is the peak BOLD response to that
presentation in signal units.
"""

import hashlib
from functools import lru_cache
import numpy as np
from nilearn.glm.first_level import compute_regressor
from nilearn.glm.first_level.hemodynamic_models import _sample_condition, glover_hrf

from boldtailor._arrays import readonly_array
from boldtailor.hrf_library import CANONICAL_PARAMETERS, HrfCandidate, _peak_normalized

HRF_NORMALIZATION = "peak_one_event_response"
_CANONICAL = HrfCandidate(0, "spm", CANONICAL_PARAMETERS)
_UNIDENTIFIED = "hrf must be 'spm', 'glover', or an identified HrfCandidate"


def _glover_kernel(tr, oversampling=50):
    return _peak_normalized(glover_hrf(tr, oversampling))


# Nilearn names columns after the callable; every package kernel is "kernel".
_glover_kernel.__name__ = "kernel"


def hrf_model(candidate):
    """Peak-one kernel callable for 'spm', 'glover', or an HrfCandidate."""
    if isinstance(candidate, str):
        named = {"spm": _CANONICAL.kernel, "glover": _glover_kernel}
        if candidate not in named:
            raise ValueError(_UNIDENTIFIED)
        return named[candidate]
    if not isinstance(candidate, HrfCandidate):
        raise ValueError(_UNIDENTIFIED)
    return candidate.kernel


def hrf_kernel(model, tr, oversampling=50):
    """Read-only peak-one kernel for a basis name or candidate."""
    return hrf_model(model)(tr, oversampling)


def resolve_hrf(model):
    """Peak-one callable for plain 'spm'/'glover'; other bases pass to Nilearn."""
    return hrf_model(model) if model in ("spm", "glover") else model


def identified_candidate(candidate):
    """The identified candidate for 'spm' or an HrfCandidate; others are rejected."""
    if isinstance(candidate, str) and candidate == "spm":
        return _CANONICAL
    if not isinstance(candidate, HrfCandidate):
        raise ValueError("hrf must be 'spm' or an identified HrfCandidate")
    return candidate


def hrf_metadata(candidate):
    candidate = identified_candidate(candidate)
    return dict(
        id=candidate.id,
        kind=candidate.kind,
        parameters=list(candidate.parameters),
        kernel_fingerprint=hashlib.sha256(
            candidate.kernel(0.1, 1).tobytes()
        ).hexdigest(),
        normalization=HRF_NORMALIZATION,
    )


def event_response_peak(kernel, duration, dt):
    """Peak of a boxcar of round(duration/dt) samples (at least one) * kernel."""
    kernel = np.asarray(kernel, dtype=float)
    n = max(1, int(round(float(duration) / dt)))
    prefix = np.r_[0.0, np.cumsum(kernel)]
    j = np.arange(len(kernel) + n - 1)
    upper = np.minimum(j + 1, len(kernel))
    lower = np.maximum(0, j - n + 1)
    peak = float(np.max(prefix[upper] - prefix[lower]))
    if not np.isfinite(peak) or peak <= 0:
        raise ValueError("event response must have a finite positive peak")
    return peak


def _kernel_scales(kernel, durations, dt):
    unique, inverse = np.unique(np.asarray(durations, dtype=float), return_inverse=True)
    peaks = np.array([event_response_peak(kernel, d, dt) for d in unique])
    return 1 / peaks[inverse.reshape(-1)]


def event_response_scales(kernel_fn, durations, tr, oversampling=50):
    """Per-event amplitude that gives each predicted response a peak of one."""
    kernel = kernel_fn(tr, oversampling)
    return _kernel_scales(kernel, durations, tr / oversampling)


def frame_tr(frame_times):
    """Nilearn's kernel TR: the minimal frame spacing."""
    return float(np.min(np.diff(np.asarray(frame_times, dtype=float))))


def scale_event_amplitudes(events, kernel_fn, frame_times, oversampling):
    """Events with modulation scaled so each response peaks at one."""
    scales = event_response_scales(
        kernel_fn, events.duration, frame_tr(frame_times), oversampling
    )
    modulation = events["modulation"] if "modulation" in events else 1.0
    return events.assign(modulation=modulation * scales)


def convolve_events(onsets, durations, times, candidate, name="stimulus"):
    onsets = np.asarray(onsets, dtype=float)
    if np.any(onsets < times[0] - 24) or np.any(onsets >= times[-1]):
        raise ValueError(f"{name}: onset has no supported sampled response")
    kernel_fn = hrf_model(candidate)
    scales = event_response_scales(kernel_fn, durations, frame_tr(times), 50)
    column, _ = compute_regressor(
        np.array([onsets, durations, scales]),
        kernel_fn,
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
    """Convolve unit-peak boxcars using kernel prefix sums on Nilearn's grid.

    A sampled boxcar [a,b) convolved with h equals H[j-a]-H[j-b], where H
    is the cumulative kernel. Evaluate only the two samples needed for each
    acquisition's linear interpolation. Every candidate, including canonical
    SPM, takes this path; it is pinned to Nilearn's compute_regressor.
    """
    from boldtailor._single_trial_design import _validate_events

    times = np.asarray(frame_times, dtype=float)
    _validate_events(events, times, "trials")
    if np.any(events.onset < times[0] - 24) or np.any(events.onset >= times[-1]):
        raise ValueError("trial onset has no supported sampled response")
    timing = np.asarray(events[["onset", "duration"]], dtype="<f8").tobytes()
    starts, stops, lower, upper, fraction = _boxcar_sampling(
        timing, np.asarray(times, dtype="<f8").tobytes()
    )
    tr = frame_tr(times)
    kernel = hrf_kernel(candidate, tr, 50)
    scales = _kernel_scales(kernel, events.duration, tr / 50)
    prefix = np.r_[0.0, np.cumsum(kernel)]

    def sampled(indices):
        a = np.clip(indices[:, None] - starts + 1, 0, len(kernel))
        b = np.clip(indices[:, None] - stops + 1, 0, len(kernel))
        return prefix[a] - prefix[b]

    values = (1 - fraction[:, None]) * sampled(lower) + fraction[:, None] * sampled(
        upper
    )
    values = values * scales
    if not np.isfinite(values).all() or not np.all(np.any(values != 0, axis=0)):
        raise ValueError("trial has no supported sampled response")
    return values
