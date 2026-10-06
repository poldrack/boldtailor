"""Parity (TOST), replication criteria, and descriptive intervals."""

import numpy as np
from scipy import stats


def relative_difference(a, b):
    return (np.asarray(a, float) - np.asarray(b, float)) / np.asarray(b, float)


def tost_paired(diff, margin, alpha=0.05):
    if not margin > 0:
        raise ValueError("margin must be positive")
    diff = np.asarray(diff, float)
    lower = float(stats.ttest_1samp(diff, -margin, alternative="greater").pvalue)
    upper = float(stats.ttest_1samp(diff, margin, alternative="less").pvalue)
    mean, sem = float(diff.mean()), float(stats.sem(diff))
    t = float(stats.t.ppf(1 - alpha, len(diff) - 1))
    p = max(lower, upper)
    ci90_low = float(mean - t * sem)
    ci90_high = float(mean + t * sem)
    return dict(
        mean=mean,
        ci90=(ci90_low, ci90_high),
        p_lower=lower,
        p_upper=upper,
        p=p,
        equivalent=bool(p < alpha),
    )


_COMPARE = {"greater": np.greater, "less": np.less}


def criterion_met(values_by_subject, minimum, direction="greater"):
    """Count subjects whose ``(b4, b1)`` pair has b4 ``direction`` than b1."""
    if direction not in _COMPARE:
        raise ValueError(f"direction must be 'greater' or 'less', not {direction!r}")
    compare = _COMPARE[direction]
    per = {s: bool(compare(b4, b1)) for s, (b4, b1) in values_by_subject.items()}
    return dict(
        per_subject=per,
        count=sum(per.values()),
        replicated=sum(per.values()) >= minimum,
    )


def mean_ci(values, level=0.95):
    values = np.asarray(values, float)
    mean = values.mean()
    half = stats.t.ppf((1 + level) / 2, len(values) - 1) * stats.sem(values)
    return mean, mean - half, mean + half
