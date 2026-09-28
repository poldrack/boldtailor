"""Holdout-specific rank and outcome isolation for both regularizers."""

import numpy as np
import pytest

import boldtailor._ridge_cv as cv
from boldtailor.fractional_ridge import score_fraction_candidates
from boldtailor.ridge_selection import score_ridge_candidates
from tests.oracles import subset_runs as subset


@pytest.mark.parametrize("fractional", [False, True])
def test_only_informative_run_cannot_supply_training_rank(
    ridge_problem, monkeypatch, fractional
):
    data, predictors, library = ridge_problem
    predictors = [
        p if r == 0 else p.assign(response_time=float(r))
        for r, p in enumerate(predictors)
    ]
    complete = [p[np.isfinite(p).all(axis=1)].to_numpy() for p in predictors]
    assert np.linalg.matrix_rank(np.vstack([x - x.mean(0) for x in complete])) == 2

    def forbidden(*args, **kwargs):
        pytest.fail("rank-deficient fold reached HRF fitting")

    monkeypatch.setattr(cv, "_fold_selection", forbidden)
    scorer = score_fraction_candidates if fractional else score_ridge_candidates
    grid = dict(fractions=[1]) if fractional else dict(alphas=[0])
    with pytest.raises(ValueError, match="rank"):
        scorer(data, predictors, library=library, **grid)


@pytest.mark.parametrize("fractional", [False, True])
def test_candidate_predictions_ignore_held_out_outcomes(
    ridge_problem, monkeypatch, fractional
):
    data, predictors, library = ridge_problem
    fits = []
    evaluate = cv.evaluate_trial_encoding

    def observe(*args, **kwargs):
        result = evaluate(*args, **kwargs)
        fits.append(result)
        return result

    monkeypatch.setattr(cv, "evaluate_trial_encoding", observe)
    scorer = score_fraction_candidates if fractional else score_ridge_candidates
    grid = dict(fractions=[1, 0.5]) if fractional else dict(alphas=[0, 0.1])
    before = scorer(data, predictors, library=library, **grid)
    first = fits[:2]
    fits.clear()
    signals = list(data.signals)
    signals[0] = np.random.default_rng(806).normal(size=signals[0].shape)
    after = scorer(
        subset(data, range(data.n_runs), signals=signals),
        predictors,
        library=library,
        **grid,
    )
    np.testing.assert_array_equal(before.fold_hrf_indices[0], after.fold_hrf_indices[0])
    for a, b in zip(first, fits[:2], strict=True):
        for key in (
            "coefficients",
            "predictor_means",
            "train_run_predictor_means",
            "train_run_intercepts",
        ):
            np.testing.assert_array_equal(getattr(a, key), getattr(b, key))
        np.testing.assert_array_equal(a.predictions[0], b.predictions[0])
    assert not np.allclose(before.fold_sse[0], after.fold_sse[0], equal_nan=True)
