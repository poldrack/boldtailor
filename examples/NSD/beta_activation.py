"""Descriptive one-sample t tests of trial betas against zero."""

import numpy as np
from scipy.stats import t as student_t

MAP_NAMES = ("mean_beta", "t", "p_uncorrected", "n_trials", "df")


def beta_activation(run_betas):
    """Pool finite trial betas across runs, assuming independent observations.

    Each input has shape (trials, grayordinates). Trials have equal weight;
    no run-wise centering is applied. Return mean, signed t, two-sided p,
    observation count and degrees of freedom. P-values are uncorrected and
    ignore trial covariance and HRF/ridge selection. Fewer than two trials
    or zero sample variance gives NaN t/p; no finite trials gives NaN in all
    maps. These are descriptive maps relative to the fitted model's zero.
    """
    arrays = [np.asarray(values) for values in run_betas]
    if not arrays:
        raise ValueError("Provide at least one trial-by-grayordinate beta array")
    if any(a.ndim != 2 or a.shape[1] != arrays[0].shape[1] for a in arrays):
        raise ValueError("Use trial-by-grayordinate arrays with matching grayordinates")
    count = np.zeros(arrays[0].shape[1], dtype=float)
    total = np.zeros_like(count)
    for values in arrays:
        finite = np.isfinite(values)
        count += finite.sum(axis=0)
        total += np.sum(values, axis=0, where=finite, dtype=np.float64)
    mean = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)

    # Center before squaring, and process one run at a time: no pooled copy
    # of the full session and no cancellation from subtracting large moments.
    squared_deviations = np.zeros_like(total)
    for values in arrays:
        centered = np.subtract(
            values,
            mean,
            out=np.zeros(values.shape, dtype=float),
            where=np.isfinite(values),
        )
        squared_deviations += np.sum(centered**2, axis=0)
    variance = np.divide(
        squared_deviations,
        count - 1,
        out=np.full_like(total, np.nan),
        where=count > 1,
    )
    standard_error = np.sqrt(
        np.divide(variance, count, out=np.full_like(total, np.nan), where=count > 0)
    )
    statistic = np.divide(
        mean, standard_error, out=np.full_like(total, np.nan), where=standard_error > 0
    )
    df = np.where(count > 0, count - 1, np.nan)
    return dict(
        mean_beta=mean,
        t=statistic,
        p_uncorrected=2 * student_t.sf(np.abs(statistic), df),
        n_trials=np.where(count > 0, count, np.nan),
        df=df,
    )
