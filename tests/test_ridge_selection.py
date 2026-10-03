"""One common spatial population and a reproducible percentile decide alpha."""

import importlib

import numpy as np
import pytest


def select(*args, **kwargs):
    try:
        function = importlib.import_module(
            "boldtailor.ridge_selection"
        ).select_ridge_penalty
    except (ImportError, AttributeError) as error:
        pytest.fail(f"Missing ridge selection: {error}")
    return function(*args, **kwargs)


def test_percentile_uses_all_grayordinates():
    scores = [[0.0, 0.0, 0.0, 0.0, 1.0, 1.0], [0.6, 0.6, 0.6, 0.6, 0.6, 0.6]]
    result = select(scores, [0.0, 0.1])
    assert result.ridge_alpha == 0.0
    np.testing.assert_allclose(result.objective_scores, [1.0, 0.6])


def test_common_finite_mask_and_negative_scores():
    result = select([[-1.0, -2.0, np.nan], [-3.0, -1.0, 20.0]], [0.0, 1.0])
    np.testing.assert_array_equal(result.scoring_mask, [True, True, False])
    np.testing.assert_allclose(result.objective_scores, [-1.1, -1.2])
    assert result.ridge_alpha == 0.0


def test_anatomical_mask_and_sort_preserve_candidate_identity():
    result = select(
        [[0.8, 0.8, 0.8], [0.9, 0, 0]], [10.0, 0.1], feature_mask=[True, False, False]
    )
    assert result.ridge_alpha == 0.1
    assert tuple(result.alphas) == (0.1, 10.0)
    np.testing.assert_allclose(result.objective_scores, [0.9, 0.8])
    for array in (result.scoring_mask, result.objective_scores):
        assert type(array) is np.ndarray
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flat[0] = 0


def test_near_ties_prefer_smaller_penalty():
    result = select([[0.2 + 1e-14], [0.2]], [1.0, 0.0])
    assert result.ridge_alpha == 0.0


@pytest.mark.parametrize(
    "alphas",
    [
        [],
        [True, 0.1],
        [0.0, np.inf],
        [0.0, np.nan],
        [0.0, -1.0],
        [0.0, 0.0],
        [0.0, ".1"],
    ],
)
def test_invalid_grid_rejected(alphas):
    with pytest.raises(ValueError, match="alpha"):
        select(np.ones((len(alphas), 3)), alphas)


@pytest.mark.parametrize("percentile", [-1, 101, np.nan, True, "90"])
def test_invalid_percentile_rejected(percentile):
    with pytest.raises(ValueError, match="percentile"):
        select([[1, 2]], [0.0], percentile=percentile)


@pytest.mark.parametrize(
    "scores,mask",
    [
        ([[np.nan], [1.0]], None),
        ([[1.0], [2.0]], [False]),
        ([[1.0], [2.0]], [True, False]),
        ([1.0, 2.0], None),
    ],
)
def test_empty_or_malformed_population_rejected(scores, mask):
    with pytest.raises(ValueError):
        select(scores, [0.0, 1.0], feature_mask=mask)


def test_ridge_penalty_selection_flags_grid_endpoint():
    scores = np.array([[0.9, 0.9], [0.1, 0.1]])
    assert select(scores, (0.0, 1.0)).at_boundary is True
    interior = np.array([[0.1, 0.1], [0.9, 0.9], [0.1, 0.1]])
    assert select(interior, (0.0, 1.0, 2.0)).at_boundary is False


def test_fraction_selection_flags_grid_endpoints():
    from boldtailor.fractional_ridge import select_ridge_fractions

    scores = np.array([[0.1, 0.5, 0.3], [0.2, 0.4, 0.9], [0.9, 0.1, 0.1]])
    choice = select_ridge_fractions(scores, (1.0, 0.5, 0.1))
    np.testing.assert_array_equal(choice.ridge_fraction, [0.1, 1.0, 0.5])
    np.testing.assert_array_equal(choice.at_boundary, [True, True, False])
    assert choice.at_boundary.dtype == bool
    assert not choice.at_boundary.flags.writeable


@pytest.fixture
def candidate_scores(ridge_problem):
    from boldtailor.ridge_selection import score_ridge_candidates

    data, predictors, _ = ridge_problem
    return score_ridge_candidates(data, predictors, alphas=[0.0, 0.1, 10.0])


def test_select_ridge_penalty_accepts_candidate_scores(candidate_scores):
    scores = candidate_scores
    paired = select(scores)
    unpaired = select(scores.cv_r2, scores.grid)
    assert paired == unpaired
    mask = np.array([True, False, True, True, True])
    assert select(scores, percentile=50.0, feature_mask=mask) == select(
        scores.cv_r2, scores.grid, percentile=50.0, feature_mask=mask
    )
    with pytest.raises(TypeError, match="grid"):
        select(scores, scores.grid)
    with pytest.raises(TypeError, match="alphas"):
        select(scores.cv_r2)


def test_select_ridge_fractions_accepts_candidate_scores(ridge_problem):
    from boldtailor.fractional_ridge import (
        score_fraction_candidates,
        select_ridge_fractions,
    )

    data, predictors, _ = ridge_problem
    scores = score_fraction_candidates(data, predictors, fractions=[1.0, 0.5])
    paired = select_ridge_fractions(scores)
    unpaired = select_ridge_fractions(scores.cv_r2, scores.grid)
    np.testing.assert_array_equal(paired.ridge_fraction, unpaired.ridge_fraction)
    np.testing.assert_array_equal(paired.fraction_indices, unpaired.fraction_indices)
    assert paired.fractions == unpaired.fractions


def test_selectors_reject_the_other_regularization_kind(candidate_scores):
    from boldtailor.fractional_ridge import select_ridge_fractions

    with pytest.raises(ValueError, match="fractional_ridge"):
        select_ridge_fractions(candidate_scores)
