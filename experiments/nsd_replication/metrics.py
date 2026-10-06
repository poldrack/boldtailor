"""Prince et al. (2022) NSD metrics on z-scored single-trial betas."""

import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import LinearSVC

THRESHOLDS = np.round(np.arange(-0.2, 0.6001, 0.05), 2)
MIN_DECODING_FEATURES = 10


def columnwise_corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = np.full(a.shape[1], np.nan)
    ok = np.isfinite(a).all(0) & np.isfinite(b).all(0)
    da, db = a[:, ok] - a[:, ok].mean(0), b[:, ok] - b[:, ok].mean(0)
    na, nb = np.sqrt((da**2).sum(0)), np.sqrt((db**2).sum(0))
    good = (na > 0) & (nb > 0)
    idx = np.flatnonzero(ok)[good]
    out[idx] = (da[:, good] * db[:, good]).sum(0) / (na[good] * nb[good])
    return out


def voxel_reliability(reps):
    scores = [
        columnwise_corr(reps[k], np.delete(reps, k, axis=0).mean(axis=0))
        for k in range(reps.shape[0])
    ]
    return np.mean(scores, axis=0)


def _threshold_rows(reliabilities, composite, roi, t):
    mask = roi & np.isfinite(composite) & (composite >= t)
    rows = []
    for version, values in reliabilities.items():
        diff = values[mask] - composite[mask]
        mean = float(np.mean(diff)) if mask.any() else np.nan
        rows.append((version, float(t), int(mask.sum()), mean))
    return rows


def threshold_curves(reliabilities, roi, thresholds=THRESHOLDS):
    composite = np.vstack(list(reliabilities.values())).mean(axis=0)
    rows = [
        row
        for t in thresholds
        for row in _threshold_rows(reliabilities, composite, roi, t)
    ]
    columns = ["version", "threshold", "n_features", "mean_difference"]
    return pd.DataFrame(rows, columns=columns)


def _pattern_corr(patterns):
    centered = patterns - patterns.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        unit = centered / norms[:, None]
    return unit @ unit.T


def _run_orders(trials):
    for _, rows in trials.groupby(["session", "run"], sort=False):
        yield rows.sort_values("onset").index.to_numpy()


def lagged_correlation(betas, trials, mask, max_lag=15):
    if len(trials) != len(betas):
        raise ValueError(
            f"trials has {len(trials)} rows but betas has {len(betas)}; "
            "they must be row-aligned"
        )
    trials = trials.reset_index(drop=True)
    sums, counts = np.zeros(max_lag + 1), np.zeros(max_lag + 1, int)
    columns = np.flatnonzero(mask)
    for order in _run_orders(trials):
        corr = _pattern_corr(betas[np.ix_(order, columns)])
        for lag in range(1, min(max_lag, len(order) - 1) + 1):
            values = np.diagonal(corr, offset=lag)
            values = values[np.isfinite(values)]
            sums[lag] += values.sum()
            counts[lag] += len(values)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums[1:] / counts[1:]
    lags = np.arange(1, max_lag + 1)
    return pd.DataFrame({"lag": lags, "mean_r": means, "n_pairs": counts[1:]})


def rdm(patterns):
    return 1 - _pattern_corr(np.asarray(patterns, float))


def rdm_agreement(rdms):
    rows = []
    for a, b in combinations(sorted(rdms), 2):
        iu = np.triu_indices_from(rdms[a], k=1)
        rows.append((a, b, float(np.corrcoef(rdms[a][iu], rdms[b][iu])[0, 1])))
    return pd.DataFrame(rows, columns=["subject_a", "subject_b", "r"])


def _fit_fold(data, k, labels):
    n_reps = data.shape[0]
    train = np.delete(data, k, axis=0).reshape(-1, data.shape[2])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = LinearSVC(random_state=0).fit(train, np.tile(labels, n_reps - 1))
    return model.predict(data[k]) == labels, int(
        np.max(model.n_iter_) >= model.max_iter
    )


def _decoding_summary(reps, mask, folds):
    n_images = reps.shape[1]
    result = dict(chance=1 / n_images, n_classes=n_images, n_features=int(mask.sum()))
    if not folds:
        return {**result, "accuracy": np.nan, "n_unconverged": 0}
    correct = np.concatenate([c for c, _ in folds])
    return {
        **result,
        "accuracy": float(correct.mean()),
        "n_unconverged": sum(u for _, u in folds),
    }


def decoding_accuracies(reps, masks, n_jobs=1):
    """``decoding_accuracy`` for each mask; folds x masks run on ``n_jobs``.

    The SVM is seeded, so results do not depend on ``n_jobs``.
    """
    labels = np.arange(reps.shape[1])
    data = {
        i: np.nan_to_num(reps[:, :, m])
        for i, m in enumerate(masks)
        if m.sum() >= MIN_DECODING_FEATURES
    }
    jobs = [(i, k) for i in data for k in range(reps.shape[0])]
    fits = Parallel(n_jobs=n_jobs)(
        delayed(_fit_fold)(data[i], k, labels) for i, k in jobs
    )
    folds = {i: [] for i in range(len(masks))}
    for (i, _), fold in zip(jobs, fits):
        folds[i].append(fold)
    return [_decoding_summary(reps, m, folds[i]) for i, m in enumerate(masks)]


def decoding_accuracy(reps, mask):
    return decoding_accuracies(reps, [mask])[0]
