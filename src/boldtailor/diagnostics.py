"""Descriptive diagnostics of fitted trial betas: one-sample t and RT correlation.

Inputs are lists with one ``(trials, features)`` beta array per run. Nothing
here fits a model, and no p-value is corrected for multiple comparisons.
"""

from numbers import Integral

import numpy as np
from scipy.stats import t as student_t

ONE_SAMPLE_T_NAMES = ("mean_beta", "t", "p_uncorrected", "n_trials", "df")


def _beta_arrays(run_betas):
    arrays = [np.asarray(values) for values in run_betas]
    if not arrays:
        raise ValueError("Provide at least one trial-by-grayordinate beta array")
    if any(a.ndim != 2 or a.shape[1] != arrays[0].shape[1] for a in arrays):
        raise ValueError("Use trial-by-grayordinate arrays with matching grayordinates")
    return arrays


def _finite_mean(arrays):
    count = np.zeros(arrays[0].shape[1], dtype=float)
    total = np.zeros_like(count)
    for values in arrays:
        finite = np.isfinite(values)
        count += finite.sum(axis=0)
        total += np.sum(values, axis=0, where=finite, dtype=np.float64)
    mean = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
    return mean, count


def _sample_variance(arrays, mean, count):
    # Center before squaring, and process one run at a time: no pooled copy
    # of the full session and no cancellation from subtracting large moments.
    squared_deviations = np.zeros_like(mean)
    for values in arrays:
        centered = np.subtract(
            values,
            mean,
            out=np.zeros(values.shape, dtype=float),
            where=np.isfinite(values),
        )
        squared_deviations += np.sum(centered**2, axis=0)
    return np.divide(
        squared_deviations,
        count - 1,
        out=np.full_like(mean, np.nan),
        where=count > 1,
    )


def one_sample_t(run_betas):
    """Pool finite trial betas across runs, assuming independent observations.

    Each input has shape (trials, grayordinates). Trials have equal weight;
    no run-wise centering is applied. Return mean, signed t, two-sided p,
    observation count and degrees of freedom, keyed by ``ONE_SAMPLE_T_NAMES``.
    P-values are uncorrected and ignore trial covariance and HRF/ridge
    selection. Fewer than two trials or zero sample variance gives NaN t/p; no
    finite trials gives NaN in all maps. These are descriptive maps relative to
    the fitted model's zero.
    """
    arrays = _beta_arrays(run_betas)
    mean, count = _finite_mean(arrays)
    variance = _sample_variance(arrays, mean, count)
    standard_error = np.sqrt(
        np.divide(variance, count, out=np.full_like(mean, np.nan), where=count > 0)
    )
    statistic = np.divide(
        mean, standard_error, out=np.full_like(mean, np.nan), where=standard_error > 0
    )
    df = np.where(count > 0, count - 1, np.nan)
    return dict(
        mean_beta=mean,
        t=statistic,
        p_uncorrected=2 * student_t.sf(np.abs(statistic), df),
        n_trials=np.where(count > 0, count, np.nan),
        df=df,
    )


def _validate_runs(beta_runs, rt_runs, run_numbers):
    if (
        not beta_runs
        or len(beta_runs) != len(rt_runs)
        or len(beta_runs) != len(run_numbers)
    ):
        raise ValueError("beta, RT, and run lists must have equal nonzero lengths")
    if not all(
        isinstance(n, Integral) and not isinstance(n, bool) and n > 0
        for n in run_numbers
    ):
        raise ValueError("run numbers must be positive integers")
    if len(set(run_numbers)) != len(run_numbers):
        raise ValueError("run numbers must be unique")
    features = np.asarray(beta_runs[0]).shape[-1]
    for beta, rt in zip(beta_runs, rt_runs, strict=True):
        if (
            np.ndim(beta) != 2
            or np.shape(beta)[1] != features
            or np.shape(rt) != (len(beta),)
        ):
            raise ValueError("RT and beta dimensions must agree for every run")


def _centered(beta, rt):
    beta, rt = np.asarray(beta, dtype=float), np.asarray(rt, dtype=float)[:, None]
    valid = np.isfinite(beta) & np.isfinite(rt) & (rt > 0)
    counts = valid.sum(axis=0)
    divisor = np.maximum(counts, 1)
    b = np.where(valid, beta, 0.0)
    r = np.where(valid, rt, 0.0)
    b = np.where(valid, b - b.sum(axis=0) / divisor, 0.0)
    r = np.where(valid, r - r.sum(axis=0) / divisor, 0.0)
    return b, r, counts, valid


def pearson_correlation(cross, beta_ss, rt_ss, counts):
    """Pearson r from centered cross/sum-of-squares; NaN below 3 trials or no variance."""
    r = np.full(np.shape(cross), np.nan)
    denominator = np.sqrt(beta_ss * rt_ss)
    np.divide(cross, denominator, out=r, where=(counts >= 3) & (denominator > 0))
    return np.clip(r, -1.0, 1.0)


def _run_moments(beta_runs, rt_runs):
    moments, counts = [], []
    for beta, rt in zip(beta_runs, rt_runs, strict=True):
        b, r, count, _ = _centered(beta, rt)
        moments.append(
            np.stack([(b * r).sum(axis=0), (b * b).sum(axis=0), (r * r).sum(axis=0)])
        )
        counts.append(count)
    return np.stack(moments), np.stack(counts)


def correlate_rt(beta_runs, rt_runs, *, run_numbers):
    """Pearson r after matched-mask within-run centering; no p-values.

    Valid trials have a finite beta and a finite positive RT. Return per-run,
    all-run, odd-run, and even-run correlations (by BIDS run number) and the
    matching trial counts. Fewer than three trials or zero variance gives NaN.
    """
    _validate_runs(beta_runs, rt_runs, run_numbers)
    moments, counts = _run_moments(beta_runs, rt_runs)
    result = {
        "per_run": pearson_correlation(*np.moveaxis(moments, 1, 0), counts),
        "counts": {"per_run": counts},
    }
    numbers = np.asarray(run_numbers)
    for name, mask in (
        ("all", numbers > 0),
        ("odd", numbers % 2 == 1),
        ("even", numbers % 2 == 0),
    ):
        count = counts[mask].sum(axis=0)
        result[name] = pearson_correlation(*moments[mask].sum(axis=0), count)
        result["counts"][name] = count
    return result


def even_run_points(beta_runs, rt_runs, run_numbers, vertex):
    """Return matched-mask, within-run centered (RT, beta) points from even runs."""
    points = []
    for beta, rt, number in zip(beta_runs, rt_runs, run_numbers, strict=True):
        if number % 2 == 0:
            b, r, _, valid = _centered(np.asarray(beta)[:, [vertex]], rt)
            points.append((r[valid], b[valid]))
    if not points:
        return np.array([]), np.array([])
    return tuple(np.concatenate([p[i] for p in points]) for i in (0, 1))
