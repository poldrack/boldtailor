"""Variability of the median ROI reliability across session subsets."""

import numpy as np
import pandas as pd

from experiments.nsd_replication.metrics import voxel_reliability
from experiments.nsd_replication.run import _level_betas, _subject_roi
from experiments.nsd_replication.trials import images_with, repetition_array


def _median_reliability(pairs, roi):
    betas = np.vstack([b for b, _ in pairs])
    trials = pd.concat([t for _, t in pairs], ignore_index=True)
    images = images_with(trials, 3)
    values = voxel_reliability(repetition_array(betas, trials, images))
    finite = values[roi & np.isfinite(values)]
    return float(np.median(finite))


def _leave_one_out(pairs, roi):
    subsets = [pairs[:i] + pairs[i + 1 :] for i in range(len(pairs))]
    return [_median_reliability(s, roi) for s in subsets]


def _split_sd(pairs, roi):
    odd, even = _median_reliability(pairs[::2], roi), _median_reliability(
        pairs[1::2], roi
    )
    return abs(odd - even) / np.sqrt(2)


def _version_row(config, source, subject, level, roi):
    pairs = _level_betas(config, source, subject, level)
    return {
        "version": level,
        "loso_sd": float(np.std(_leave_one_out(pairs, roi), ddof=1)),
        "split_sd": float(_split_sd(pairs, roi)),
        "median": _median_reliability(pairs, roi),
    }


def variability(config, source, levels, subject=None):
    subject = subject or config.subjects[0]
    roi = _subject_roi(config, subject)
    rows = [_version_row(config, source, subject, lv, roi) for lv in levels]
    return pd.DataFrame(rows, columns=["version", "loso_sd", "split_sd", "median"])
