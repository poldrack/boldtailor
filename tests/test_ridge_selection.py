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
    scores = [[0., 0., 0., 0., 1., 1.], [.6, .6, .6, .6, .6, .6]]
    result = select(scores, [0., .1])
    assert result.ridge_alpha == 0.
    np.testing.assert_allclose(result.objective_scores, [1., .6])


def test_common_finite_mask_and_negative_scores():
    result = select([[-1., -2., np.nan], [-3., -1., 20.]], [0., 1.])
    np.testing.assert_array_equal(result.scoring_mask, [True, True, False])
    np.testing.assert_allclose(result.objective_scores, [-1.1, -1.2])
    assert result.ridge_alpha == 0.


def test_anatomical_mask_and_sort_preserve_candidate_identity():
    result = select([[.8, .8, .8], [.9, 0, 0]], [10., .1],
                    feature_mask=[True, False, False])
    assert result.ridge_alpha == .1
    assert tuple(result.alphas) == (.1, 10.)
    np.testing.assert_allclose(result.objective_scores, [.9, .8])
    for array in (result.scoring_mask, result.objective_scores):
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_near_ties_prefer_smaller_penalty():
    result = select([[.2 + 1e-14], [.2]], [1., 0.])
    assert result.ridge_alpha == 0.


@pytest.mark.parametrize("alphas", [[], [True, .1], [0., np.inf], [0., np.nan],
    [0., -1.], [0., 0.], [0., ".1"]])
def test_invalid_grid_rejected(alphas):
    with pytest.raises(ValueError, match="alpha"):
        select(np.ones((len(alphas), 3)), alphas)


@pytest.mark.parametrize("percentile", [-1, 101, np.nan, True, "90"])
def test_invalid_percentile_rejected(percentile):
    with pytest.raises(ValueError, match="percentile"):
        select([[1, 2]], [0.], percentile=percentile)


@pytest.mark.parametrize("scores,mask", [([[np.nan], [1.]], None),
    ([[1.], [2.]], [False]), ([[1.], [2.]], [True, False]), ([1., 2.], None)])
def test_empty_or_malformed_population_rejected(scores, mask):
    with pytest.raises(ValueError):
        select(scores, [0., 1.], feature_mask=mask)
