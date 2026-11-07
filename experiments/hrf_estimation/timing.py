"""Nuisance regressors on the NSD-preprocessed sampling grid."""

import numpy as np
import pandas as pd

MOTION_COLUMNS = ('trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z')


def sampling_interval(frame_times):
    times = np.asarray(frame_times, dtype=float)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all():
        raise ValueError('frame times must be a finite vector with at least two scans')
    intervals = np.diff(times)
    if intervals[0] <= 0 or not np.allclose(intervals, intervals[0], rtol=1e-6):
        raise ValueError('frame times must be increasing and regularly sampled')
    return float(intervals[0])


def build_nuisance(motion, frame_times, cutoff=80.0):
    """Expand six aligned motion columns and add cosine drifts and one intercept."""
    tr = sampling_interval(frame_times)
    n = len(frame_times)
    if len(motion) != n:
        raise ValueError('motion row count differs from CIFTI scans')
    if not np.isfinite(cutoff) or cutoff <= 2 * tr:
        raise ValueError('cutoff must be finite and longer than two sampling intervals')
    missing = set(MOTION_COLUMNS) - set(motion.columns)
    if missing:
        raise ValueError(f'missing motion columns: {sorted(missing)}')
    values = motion.loc[:, MOTION_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('motion parameters must be finite; no imputation or censoring')
    result = _expand_motion(values)
    order = int(np.floor(2 * n * tr / cutoff))
    grid = (np.arange(n) + 0.5) * np.pi / n
    for k in range(1, order + 1):
        result[f'cosine{k:02}'] = np.cos(grid * k)
    result['intercept'] = np.ones(n)
    return pd.DataFrame(result)


def _expand_motion(values):
    difference = np.vstack([np.zeros(6), np.diff(values, axis=0)])
    result = {}
    for i, name in enumerate(MOTION_COLUMNS):
        result[name] = values[:, i]
        result[name + '_derivative1'] = difference[:, i]
        result[name + '_power2'] = values[:, i] ** 2
        result[name + '_derivative1_power2'] = difference[:, i] ** 2
    return result
