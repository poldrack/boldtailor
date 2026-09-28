from dataclasses import replace

import numpy as np
import pytest

from boldtailor.fractional_ridge import score_fraction_candidates
from boldtailor.ridge_selection import score_ridge_candidates


@pytest.mark.parametrize("fractional", [False, True])
def test_candidate_scores_name_the_grid_and_scientific_basis(ridge_problem, fractional):
    from boldtailor import ridge_results

    data, predictors, _ = ridge_problem
    function = score_fraction_candidates if fractional else score_ridge_candidates
    options = {"fractions": [0.3, 1, 0.7]} if fractional else {"alphas": [1, 0, 0.1]}
    result = function(data, predictors, **options)
    assert type(result) is ridge_results.CandidateScores
    assert result.regularization == (
        "fractional_ridge" if fractional else "normalized_ridge"
    )
    assert result.grid == ((1.0, 0.7, 0.3) if fractional else (0.0, 0.1, 1.0))
    assert result.cv_r2.shape == (3, data.n_features)
    np.testing.assert_allclose(
        result.cv_r2,
        1 - result.fold_sse.sum(axis=0) / result.fold_sst.sum(axis=0),
    )
    assert result.fold_hrf_indices.dtype == np.int64
    assert all(mask.dtype == bool for mask in result.trial_masks)
    with pytest.raises(ValueError, match="regularization"):
        replace(result, regularization="unknown")
