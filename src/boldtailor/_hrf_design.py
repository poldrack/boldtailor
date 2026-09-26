"""One identified HRF per convolution; candidate bases are never orthogonalized."""

import hashlib
import numpy as np
from nilearn.glm.first_level import compute_regressor

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
