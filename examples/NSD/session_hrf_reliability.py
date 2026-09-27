"""Descriptive HRF shape agreement across independently estimated sessions."""

from itertools import combinations

import numpy as np

from boldtailor.hrf_library import PARAMETER_NAMES
from .hrf_reliability import _indices

SUMMARY_NAMES = (
    "mean_between_session_r",
    "mean_matched_canonical_r",
    "mean_between_minus_canonical_r",
    "valid_sessions",
    "valid_pairs",
)


def finite_mean(values, axis=0):
    count = np.isfinite(values).sum(axis=axis)
    return np.divide(
        np.nansum(values, axis=axis),
        count,
        out=np.full(count.shape, np.nan),
        where=count > 0,
    )


def _lookup(library):
    curves = library.curves - library.curves.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(curves, axis=1, keepdims=True)
    unit = np.divide(curves, norms, out=np.full_like(curves, np.nan), where=norms > 0)
    return np.clip(unit @ unit.T, -1, 1)


def _parameter_sd(library, ids):
    names = [*PARAMETER_NAMES[:6], "peak_time"]
    values = library.parameter_table[names].to_numpy()[np.maximum(ids, 0)]
    values[ids < 0] = np.nan
    count = np.isfinite(values).sum(axis=0)
    centered = values - finite_mean(values)[None]
    variance = np.divide(
        np.nansum(centered**2, axis=0),
        count - 1,
        out=np.full(count.shape, np.nan),
        where=count > 1,
    )
    return names, np.sqrt(variance).T


def compare_hrfs(library, hrf_indices, sessions):
    """Pearson r over the full stored time grid, with no shifting of curves.

    Every unordered session pair receives equal weight. Its canonical baseline
    is the mean of the two session-to-SPM correlations at that grayordinate.
    Summary correlations require at least one valid pair. All scores are
    descriptive; shared sessions make pairs dependent observations.
    """
    raw = np.asarray(hrf_indices, dtype=float)
    sessions = list(sessions)
    if (
        raw.ndim != 2
        or raw.shape[0] < 2
        or len(sessions) != raw.shape[0]
        or len(set(sessions)) != len(sessions)
    ):
        raise ValueError("At least two unique sessions must match the HRF index rows")
    ids = np.stack([_indices(row, len(library.curves)) for row in raw])
    lookup = _lookup(library)
    canonical = np.where(ids >= 0, lookup[np.maximum(ids, 0), 0], np.nan)
    pairs = list(combinations(range(len(sessions)), 2))
    pairwise, baseline = np.full((2, len(pairs), ids.shape[1]), np.nan)
    for row, (a, b) in enumerate(pairs):
        valid = (ids[a] >= 0) & (ids[b] >= 0)
        pairwise[row, valid] = lookup[ids[a, valid], ids[b, valid]]
        baseline[row, valid] = (canonical[a, valid] + canonical[b, valid]) / 2
    baseline[~np.isfinite(pairwise)] = np.nan
    delta = pairwise - baseline
    names, sd = _parameter_sd(library, ids)
    return dict(
        sessions=sessions,
        pair_names=[f"{sessions[a]}_vs_{sessions[b]}" for a, b in pairs],
        pairs=pairs,
        pairwise=pairwise,
        canonical=canonical,
        pair_baseline=baseline,
        pair_delta=delta,
        parameter_names=names,
        parameter_sd=sd,
        summary=np.vstack(
            [
                finite_mean(pairwise),
                finite_mean(baseline),
                finite_mean(delta),
                np.isfinite(canonical).sum(axis=0),
                np.isfinite(pairwise).sum(axis=0),
            ]
        ),
    )
