"""Descriptive agreement between selected HRF shapes, per grayordinate.

Correlations use the full stored library time grid, without shifting curves,
so they ignore amplitude scale and offset. None of these scores is an ICC.
"""

from itertools import combinations

import numpy as np

from boldtailor.hrf_library import PARAMETER_NAMES

CORRELATION_NAMES = ("odd_even_r", "odd_canonical_r", "even_canonical_r")
SUMMARY_NAMES = (
    "mean_between_session_r",
    "mean_matched_canonical_r",
    "mean_between_minus_canonical_r",
    "valid_sessions",
    "valid_pairs",
)


def library_indices(values, size):
    """Integer library indices, with NaN or -1 (no selection) mapped to -1."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("HRF indices must be one-dimensional")
    missing = np.isnan(values) | (values == -1)
    selected = values[~missing]
    if (
        not np.isfinite(selected).all()
        or np.any(selected < 0)
        or np.any(selected >= size)
        or np.any(selected != np.floor(selected))
    ):
        raise ValueError("HRF indices must identify a library entry, or be NaN/-1")
    return np.where(missing, -1, values).astype(int)


def finite_mean(values, axis=0):
    """Mean over finite entries; NaN where none are finite."""
    count = np.isfinite(values).sum(axis=axis)
    return np.divide(
        np.nansum(values, axis=axis),
        count,
        out=np.full(count.shape, np.nan),
        where=count > 0,
    )


def _curve_correlation_matrix(curves):
    """Pearson r between every pair of curves; NaN for constant curves."""
    centered = curves - curves.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    unit = np.divide(
        centered, norms, out=np.full_like(centered, np.nan), where=norms > 0
    )
    return np.clip(unit @ unit.T, -1.0, 1.0)


def curve_correlations(library, ids_a, ids_b):
    """Pearson r on the common stored time grid, including padded tails.

    Rows are a/b, a/canonical, and b/canonical (``CORRELATION_NAMES`` for an
    odd/even split). Each pair retains NaN if either selection is missing or
    either curve is constant. Curves are not shifted in time.
    """
    a = library_indices(ids_a, len(library.curves))
    b = library_indices(ids_b, len(library.curves))
    if a.shape != b.shape:
        raise ValueError("Odd/even HRF indices must have the same shape")
    lookup = _curve_correlation_matrix(library.curves)
    canonical = np.zeros_like(a)
    result = np.full((3, len(a)), np.nan)
    for row, (x, y) in enumerate(((a, b), (a, canonical), (b, canonical))):
        valid = (x >= 0) & (y >= 0)
        result[row, valid] = lookup[x[valid], y[valid]]
    return result


def _session_indices(library, hrf_indices, sessions):
    raw = np.asarray(hrf_indices, dtype=float)
    if (
        raw.ndim != 2
        or raw.shape[0] < 2
        or len(sessions) != raw.shape[0]
        or len(set(sessions)) != len(sessions)
    ):
        raise ValueError("At least two unique sessions must match the HRF index rows")
    return np.stack([library_indices(row, len(library.curves)) for row in raw])


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


def _pair_scores(lookup, ids, canonical, pairs):
    pairwise, baseline = np.full((2, len(pairs), ids.shape[1]), np.nan)
    for row, (a, b) in enumerate(pairs):
        valid = (ids[a] >= 0) & (ids[b] >= 0)
        pairwise[row, valid] = lookup[ids[a, valid], ids[b, valid]]
        baseline[row, valid] = (canonical[a, valid] + canonical[b, valid]) / 2
    baseline[~np.isfinite(pairwise)] = np.nan
    return pairwise, baseline


def _peak_time_mean(library, ids):
    peak_times = library.parameter_table.peak_time.to_numpy()[np.maximum(ids, 0)]
    peak_times[ids < 0] = np.nan
    return finite_mean(peak_times)


def compare_hrfs(library, hrf_indices, sessions):
    """Pearson r over the full stored time grid, with no shifting of curves.

    Every unordered session pair receives equal weight. Its canonical baseline
    is the mean of the two session-to-SPM correlations at that grayordinate.
    Summary correlations (``SUMMARY_NAMES``) require at least one valid pair.
    All scores are descriptive; shared sessions make pairs dependent.
    """
    sessions = list(sessions)
    ids = _session_indices(library, hrf_indices, sessions)
    lookup = _curve_correlation_matrix(library.curves)
    canonical = np.where(ids >= 0, lookup[np.maximum(ids, 0), 0], np.nan)
    pairs = list(combinations(range(len(sessions)), 2))
    pairwise, baseline = _pair_scores(lookup, ids, canonical, pairs)
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
        peak_time_mean=_peak_time_mean(library, ids),
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
