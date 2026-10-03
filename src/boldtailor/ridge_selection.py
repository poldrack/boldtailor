"""Select one ridge penalty using held-out trial encoding across features."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from boldtailor._hrf_design import TIE_TOLERANCE
from boldtailor._scalars import is_real
from boldtailor._single_trial_fit import validate_alpha
from boldtailor.data import AnalysisData
from boldtailor.hrf_library import HrfLibrary
from boldtailor.ridge_results import CandidateScores, RidgeSelection, paired_scores


def _alpha_grid(alphas):
    grid = tuple(validate_alpha(a) for a in alphas)
    if not grid or len(set(grid)) != len(grid):
        raise ValueError("alphas must be a nonempty grid of distinct penalties")
    return grid


def select_ridge_penalty(
    scores: CandidateScores | np.ndarray,
    alphas: Sequence[float] | None = None,
    *,
    percentile: float = 90.0,
    feature_mask: np.ndarray | None = None,
) -> RidgeSelection:
    """Reduce once across the complete feature population, never per block.

    Pass the ``CandidateScores`` from :func:`score_ridge_candidates`, or a
    candidate-by-feature score array together with its ``alphas``.

    Only features finite for every candidate enter the common mask. Negative
    scores are valid. Objective ties within ``TIE_TOLERANCE`` prefer the smaller
    penalty.
    ``at_boundary`` is True when the winner is the smallest or largest alpha
    (always True for a one-point grid).
    """
    candidate_r2, alphas = paired_scores(
        scores, alphas, kind="normalized_ridge", grid_name="alphas"
    )
    grid = _alpha_grid(alphas)
    scores = np.asarray(candidate_r2, dtype=float)
    if scores.ndim != 2 or scores.shape[0] != len(grid):
        raise ValueError("candidate_r2 must have one row per alpha")
    if (
        not is_real(percentile)
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
    winner = np.flatnonzero(objectives >= objectives.max() - TIE_TOLERANCE)[0]
    at_boundary = winner in (0, len(alphas) - 1)
    return RidgeSelection(
        ridge_alpha=alphas[winner],
        alphas=alphas,
        objective_scores=objectives,
        percentile=float(percentile),
        scoring_mask=mask,
        at_boundary=at_boundary,
    )


def score_ridge_candidates(
    data: AnalysisData,
    predictors: Sequence[pd.DataFrame],
    *,
    alphas: Sequence[float],
    library: HrfLibrary | None = None,
    run_labels: Sequence[str] | None = None,
    feature_signature: str | None = None,
    encoding_mode: str = "within_run",
) -> CandidateScores:
    """Score candidate-regularized trial betas with leave-one-run-out encoding.

    Supply only outer-training runs when nesting this selection. A library
    selects HRFs separately on each inner-training set; None uses canonical
    SPM. This returns feature-level scores, without choosing a block-local
    penalty. Combine blocks before calling select_ridge_penalty.
    """
    from boldtailor._ridge_cv import score_candidates

    grid = tuple(sorted(_alpha_grid(alphas)))
    return score_candidates(
        data,
        predictors,
        grid,
        library,
        run_labels,
        feature_signature,
        encoding_mode=encoding_mode,
    )
