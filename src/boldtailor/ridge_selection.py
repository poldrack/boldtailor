"""Select one ridge penalty using held-out trial encoding across features."""

from numbers import Real

import numpy as np

from boldtailor._single_trial_fit import validate_alpha
from boldtailor.ridge_results import RidgeSelection


def _alpha_grid(alphas):
    grid = tuple(validate_alpha(a) for a in alphas)
    if not grid or len(set(grid)) != len(grid):
        raise ValueError("alphas must be a nonempty grid of distinct penalties")
    return grid


def select_ridge_penalty(candidate_r2, alphas, *, percentile=90.0, feature_mask=None):
    """Reduce once across the complete feature population, never per block.

    Only features finite for every candidate enter the common mask. Negative
    scores are valid. Objective ties within 1e-12 prefer the smaller penalty.
    """
    grid = _alpha_grid(alphas)
    scores = np.asarray(candidate_r2, dtype=float)
    if scores.ndim != 2 or scores.shape[0] != len(grid):
        raise ValueError("candidate_r2 must have one row per alpha")
    if (
        isinstance(percentile, (bool, np.bool_))
        or not isinstance(percentile, Real)
        or not np.isfinite(percentile)
        or not 0 <= percentile <= 100
    ):
        raise ValueError("percentile must be a finite number from 0 to 100")
    mask = np.isfinite(scores).all(axis=0)
    if feature_mask is not None:
        requested = np.asarray(feature_mask)
        if requested.shape != mask.shape or requested.dtype.kind != "b":
            raise ValueError("feature_mask must be a matching boolean feature array")
        mask &= requested
    if not mask.any():
        raise ValueError("No features are eligible for every ridge candidate")
    order = np.argsort(grid)
    alphas = tuple(grid[i] for i in order)
    objectives = np.percentile(
        scores[order][:, mask], percentile, axis=1, method="linear"
    )
    winner = np.flatnonzero(objectives >= objectives.max() - 1e-12)[0]
    return RidgeSelection(alphas[winner], alphas, objectives, float(percentile), mask)


def score_ridge_candidates(
    data, predictors, *, alphas, library=None, run_labels=None, feature_signature=None
):
    """Score candidate-regularized trial betas with leave-one-run-out encoding.

    Supply only outer-training runs when nesting this selection. A library
    selects HRFs separately on each inner-training set; None uses canonical
    SPM. This returns feature-level scores, without choosing a block-local
    penalty. Combine blocks before calling select_ridge_penalty.
    """
    from boldtailor._ridge_cv import score_candidates

    grid = tuple(sorted(_alpha_grid(alphas)))
    return score_candidates(
        data, predictors, grid, library, run_labels, feature_signature
    )
