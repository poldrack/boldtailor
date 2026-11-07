"""Prepared event sampling and nuisance projections for repeated HRF fits."""

import numpy as np

from boldtailor._hrf_design import _boxcar_sampling, _count_scales, frame_tr, OVERSAMPLING
from boldtailor.hrf_library import HrfCandidate
from experiments.hrf_estimation.design import partition_events
from experiments.hrf_estimation.scoring import residuals


def _sampling(events, times):
    timing = np.asarray(events[["onset", "duration"]], dtype="<f8")
    starts, stops, lower, upper, fraction = _boxcar_sampling(
        timing.tobytes(), np.asarray(times, dtype="<f8").tobytes())
    return dict(starts=starts, stops=stops, lower=lower, upper=upper, fraction=fraction)


def prepare_profile(runs, mode):
    if mode not in ("amplitude", "duration"):
        raise ValueError("rt_mode must be amplitude or duration")
    return tuple(_prepare_run(run, mode) for run in runs)


def _prepare_run(run, mode):
    events = partition_events(run.events)
    valid = events.category.to_numpy() == "valid"
    weights = np.column_stack([
        events.category.to_numpy() == "missing",
        events.category.to_numpy() == "incorrect",
        valid & (events.trial_type.to_numpy() == 0),
        valid & (events.trial_type.to_numpy() == 1),
        np.where(valid, events.response_time, 0),
    ]).astype(float)
    c = run.confounds.to_numpy(dtype=float)
    u, s, _ = np.linalg.svd(c, full_matrices=False)
    rank = np.sum(s > (s[0] * max(c.shape) * np.finfo(float).eps)) if len(s) else 0
    q = u[:, :rank]
    y, _ = residuals(c, run.signals)
    duration = events.loc[valid].assign(duration=events.loc[valid, "response_time"])
    return dict(sampling=_sampling(events, run.frame_times), weights=weights,
                duration=_sampling(duration, run.frame_times) if valid.any() else None,
                tr=frame_tr(run.frame_times), mode=mode, q=q, y=y,
                energy=np.sum(y**2, axis=0), raw=np.sum(run.signals**2, axis=0))


def _sampled_response(sampling, kernel, normalize):
    prefix = np.r_[0.0, np.cumsum(kernel)]
    def at(indices):
        a = np.clip(indices[:, None] - sampling["starts"] + 1, 0, len(kernel))
        b = np.clip(indices[:, None] - sampling["stops"] + 1, 0, len(kernel))
        return prefix[a] - prefix[b]
    fraction = sampling["fraction"][:, None]
    values = (1 - fraction) * at(sampling["lower"]) + fraction * at(sampling["upper"])
    if normalize:
        values *= _count_scales(kernel, sampling["stops"] - sampling["starts"])
    return values


def event_design(item, candidate):
    kernel = candidate.kernel(item["tr"], OVERSAMPLING)
    z = _sampled_response(item["sampling"], kernel, True) @ item["weights"]
    if item["mode"] == "duration":
        z[:, 4] = (0 if item["duration"] is None else
                   _sampled_response(item["duration"], kernel, False).sum(axis=1))
    return z


def valid_shape(parameters):
    a, b, c, d = parameters[:4]
    return a > c and b > d and b - d > a - c


def prepared_loss(prepared, feature, parameters):
    if not valid_shape(parameters):
        return np.inf
    energy = sum(item["energy"][feature] for item in prepared)
    raw = sum(item["raw"][feature] for item in prepared)
    if energy <= 64 * np.finfo(float).eps * raw:
        return np.nan
    candidate = HrfCandidate(0, "double_gamma", tuple(parameters))
    sse = 0.0
    for item in prepared:
        z = event_design(item, candidate)
        q = item["q"]
        z = z - q @ (q.T @ z)
        error, _ = residuals(z, item["y"][:, feature])
        sse += np.sum(error**2)
    return float(sse / energy)


def library_losses(prepared, parameters):
    """All vertex losses per candidate, with a single design decomposition."""
    energy = sum(item["energy"] for item in prepared)
    raw = sum(item["raw"] for item in prepared)
    valid = energy > 64 * np.finfo(float).eps * raw
    result = np.full((len(parameters), len(energy)), np.inf)
    for i, values in enumerate(parameters):
        if not valid_shape(values):
            continue
        candidate = HrfCandidate(0, "double_gamma", tuple(values))
        sse = np.zeros(len(energy))
        for item in prepared:
            z = event_design(item, candidate)
            q = item["q"]
            z -= q @ (q.T @ z)
            u, s, _ = np.linalg.svd(z, full_matrices=False)
            rank = np.sum(s > s[0] * max(z.shape) * np.finfo(float).eps) if s[0] else 0
            sse += np.maximum(0, item["energy"] - np.sum((u[:, :rank].T @ item["y"])**2, axis=0))
        result[i, valid] = sse[valid] / energy[valid]
    return result
