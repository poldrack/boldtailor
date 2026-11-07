"""Choose regularization using only runs in the outer training partition."""

import numpy as np

from experiments.hrf_estimation.continuous import fit_continuous
from experiments.hrf_estimation.scoring import score_hrfs


def choose_regularization(runs, settings):
    if len(runs) < 2:
        raise ValueError("regularization tuning requires at least two training runs")
    grid = np.asarray(
        settings.get("lambda_grid", [0.0, 0.001, 0.01, 0.1, 1.0]), dtype=float
    )
    if (
        grid.ndim != 1
        or not len(grid)
        or not np.isfinite(grid).all()
        or (grid < 0).any()
    ):
        raise ValueError("lambda grid must contain finite nonnegative values")
    grid = np.unique(grid)
    scores = []
    for strength in grid:
        folds = []
        for held_out in range(len(runs)):
            train = tuple(run for i, run in enumerate(runs) if i != held_out)
            # fit_continuous rebuilds its library reference using this inner train only.
            estimate = fit_continuous(train, {**settings, "lambda": float(strength)})
            result = score_hrfs(
                (runs[held_out],), estimate, settings.get("rt_mode", "amplitude")
            )
            folds.append(result.summary.r2_task_partial.to_numpy())
        scores.append(folds)
    scores = np.asarray(scores)
    common = np.isfinite(scores).all(axis=(0, 1))
    if not common.any():
        raise ValueError("no common valid features for regularization tuning")
    means = scores[:, :, common].mean(axis=(1, 2))
    return float(grid[np.argmax(means)])
