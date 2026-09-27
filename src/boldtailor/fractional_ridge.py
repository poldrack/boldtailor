"""Select relative shrinkage independently for each feature using encoding CV."""

import numpy as np

from boldtailor._fractional_ridge import fraction_grid
from boldtailor.ridge_results import FractionSelection


def select_ridge_fractions(candidate_r2, fractions, *, feature_mask=None):
    """Maximize each feature's score; ties favor less shrinkage (larger f).

    A feature must have finite scores at every candidate. Ineligible features
    return NaN fractions/scores and index -1, including entirely invalid blocks.
    """
    grid = fraction_grid(fractions)
    scores = np.asarray(candidate_r2, dtype=float)
    if scores.ndim != 2 or scores.shape[0] != len(grid):
        raise ValueError("candidate_r2 must have one row per fraction")
    mask = np.isfinite(scores).all(axis=0)
    if feature_mask is not None:
        requested = np.asarray(feature_mask)
        if requested.shape != mask.shape or requested.dtype.kind != "b":
            raise ValueError("feature_mask must be a matching boolean feature array")
        mask &= requested
    order = np.argsort(grid)[::-1]
    grid = tuple(grid[i] for i in order)
    values = scores[order][:, mask]
    selected = np.full(scores.shape[1], np.nan)
    best = selected.copy()
    indices = np.full(scores.shape[1], -1, dtype=int)
    if mask.any():
        winner = np.argmax(values >= values.max(axis=0) - 1e-12, axis=0)
        selected[mask] = np.asarray(grid)[winner]
        best[mask] = values[winner, np.arange(mask.sum())]
        indices[mask] = winner
    return FractionSelection(grid, selected, best, indices, mask)


def score_fraction_candidates(
    data,
    predictors,
    *,
    fractions,
    library=None,
    run_labels=None,
    feature_signature=None,
):
    """Score same-fraction beta targets in held-out runs; no image repeats needed.

    A supplied library selects HRFs only on each fold's training runs. Pass
    outer-training runs only when nesting this scorer. Fractions act on the
    normalized trial coefficients; nuisance parameters remain unpenalized.
    """
    from boldtailor._ridge_cv import score_candidates

    grid = tuple(sorted(fraction_grid(fractions), reverse=True))
    return score_candidates(
        data, predictors, grid, library, run_labels, feature_signature, fractional=True
    )
