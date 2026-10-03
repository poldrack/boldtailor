"""Select relative shrinkage independently for each feature using encoding CV."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from boldtailor._hrf_design import TIE_TOLERANCE
from boldtailor._fractional_ridge import (  # public re-exports
    NORM_BASIS,
    fraction_grid,
    regularization,
)
from boldtailor.data import AnalysisData
from boldtailor.hrf_library import HrfLibrary
from boldtailor.ridge_results import (
    CandidateScores,
    FractionSelection,
    paired_scores,
    scoring_mask,
)


def select_ridge_fractions(
    scores: CandidateScores | np.ndarray,
    fractions: Sequence[float] | None = None,
    *,
    feature_mask: np.ndarray | None = None,
) -> FractionSelection:
    """Maximize each feature's score; ties favor less shrinkage (larger f).

    Pass the ``CandidateScores`` from :func:`score_fraction_candidates`, or a
    candidate-by-feature score array together with its ``fractions``.

    A feature must have finite scores at every candidate. Ineligible features
    return NaN fractions/scores and index -1, including entirely invalid blocks.
    ``at_boundary`` is True for scored features whose winner is the largest or
    smallest fraction; with a one-point grid every scored feature is flagged.
    """
    candidate_r2, fractions = paired_scores(
        scores, fractions, kind="fractional_ridge", grid_name="fractions"
    )
    grid = fraction_grid(fractions)
    scores = np.asarray(candidate_r2, dtype=float)
    if scores.ndim != 2 or scores.shape[0] != len(grid):
        raise ValueError("candidate_r2 must have one row per fraction")
    mask = scoring_mask(scores, feature_mask)
    order = np.argsort(grid)[::-1]
    grid = tuple(grid[i] for i in order)
    return _choose_fractions(grid, scores[order], mask)


def _choose_fractions(grid, scores, mask):
    """Per-feature winners over a descending grid; ties keep the larger f."""
    values = scores[:, mask]
    selected = np.full(scores.shape[1], np.nan)
    best = selected.copy()
    indices = np.full(scores.shape[1], -1, dtype=int)
    at_boundary = np.zeros(scores.shape[1], dtype=bool)
    if mask.any():
        winner = np.argmax(values >= values.max(axis=0) - TIE_TOLERANCE, axis=0)
        selected[mask] = np.asarray(grid)[winner]
        best[mask] = values[winner, np.arange(mask.sum())]
        indices[mask] = winner
        at_boundary[mask] = np.isin(winner, [0, len(grid) - 1])
    return FractionSelection(
        fractions=grid,
        ridge_fraction=selected,
        selected_r2=best,
        fraction_indices=indices,
        scoring_mask=mask,
        at_boundary=at_boundary,
    )


def score_fraction_candidates(
    data: AnalysisData,
    predictors: Sequence[pd.DataFrame],
    *,
    fractions: Sequence[float],
    library: HrfLibrary | None = None,
    run_labels: Sequence[str] | None = None,
    feature_signature: str | None = None,
    encoding_mode: str = "within_run",
) -> CandidateScores:
    """Score fixed OLS beta targets in held-out runs; no image repeats needed.

    A supplied library selects HRFs only on each fold's training runs. Pass
    outer-training runs only when nesting this scorer. Fractions act on the
    raw trial coefficients after nuisance projection; nuisance parameters remain
    unpenalized. Only training betas use each candidate fraction; held-out OLS
    targets use the same training-selected HRF for all candidates.
    """
    from boldtailor._ridge_cv import score_candidates

    grid = tuple(sorted(fraction_grid(fractions), reverse=True))
    return score_candidates(
        data,
        predictors,
        grid,
        library,
        run_labels,
        feature_signature,
        fractional=True,
        encoding_mode=encoding_mode,
    )
